import numpy as np
from scipy.ndimage import convolve

def transform(start_map, end_map, params=None, numsteps=10):
    """
    Abrikosov Vortex Flux Pinning and Depinning image transformation.
    
    Models the penetration of an external magnetic field (end_map) into a 
    Type-II superconductor (start_map). A pinning landscape is derived from 
    start_map depth (darkness) and roughness (local 3x3 std). Vortices enter 
    preferentially at pinning sites, accumulate, and trigger local collapse 
    cascades when the critical density is exceeded. Periodic boundaries are 
    used throughout.
    
    Returns a list of length numsteps+1, where index 0 is start_map and 
    index numsteps is end_map.
    """
    if params is None:
        params = {}
    
    # Physics parameters
    entry_scale        = float(params.get('entry_scale',        0.25))
    collapse_threshold = float(params.get('collapse_threshold', 0.55))
    cascade_threshold  = float(params.get('cascade_threshold',  0.32))
    vortex_rate        = float(params.get('vortex_rate',        0.35))
    applied_exponent   = float(params.get('applied_exponent',   2.0))
    roughness_weight   = float(params.get('roughness_weight',   1.0))
    noise_level        = float(params.get('noise_level',        0.03))
    cascade_iterations = int(  params.get('cascade_iterations', 2))
    global_critical    = float(params.get('global_critical',    0.88))
    
    A = start_map.astype(np.float32) / 255.0
    B = end_map.astype(np.float32)   / 255.0
    
    if A.shape != B.shape:
        raise ValueError("start_map and end_map must have identical shapes")
    
    # Ensure channel dimension exists
    if A.ndim == 2:
        A = A[..., np.newaxis]
        B = B[..., np.newaxis]
    
    h, w, ch = A.shape
    
    # --------------------------
    # Build pinning landscape U
    # --------------------------
    if ch > 1:
        Ag = A.mean(axis=2)
    else:
        Ag = A[..., 0]
    
    # Depth: dark regions interpreted as deeper crevices -> stronger pinning
    D = 1.0 - Ag
    
    # Roughness: local std in 3x3 neighborhood under periodic boundaries
    shifts = [(-1,-1), (-1,0), (-1,1),
              (0,-1),  (0,0),  (0,1),
              (1,-1),  (1,0),  (1,1)]
    sum_g  = np.zeros_like(Ag)
    sum_g2 = np.zeros_like(Ag)
    for dy, dx in shifts:
        rolled = np.roll(np.roll(Ag, dy, axis=0), dx, axis=1)
        sum_g  += rolled
        sum_g2 += rolled * rolled
    mean_g = sum_g / 9.0
    var_g  = np.maximum(sum_g2 / 9.0 - mean_g**2, 0.0)
    R = np.sqrt(var_g)
    
    # Combine and normalize to [0,1]
    U_raw = D + roughness_weight * R
    U = (U_raw - U_raw.min()) / (U_raw.max() - U_raw.min() + 1e-8)
    
    # State fields
    f = np.zeros((h, w), dtype=np.float32)   # vortex density / normal fraction
    c = np.zeros((h, w), dtype=bool)         # collapsed to normal state
    kernel_3x3 = np.ones((3,3), dtype=np.float32) / 9.0
    
    out_maps = []
    
    for step in range(numsteps + 1):
        t = step / numsteps if numsteps > 0 else 1.0
        h_field = np.power(t, applied_exponent)
        
        # Local entry barrier: strong pinning (high U) -> lower barrier
        h_th = entry_scale * (1.0 - U)
        
        # Normalized excess field [0,1]
        drive = np.maximum(0.0, h_field - h_th) / np.maximum(1.0 - h_th, 1e-8)
        drive = np.clip(drive, 0.0, 1.0)
        
        # Target vortex density: capacity higher at pinning sites, but all sites 
        # saturate toward normal state as drive -> 1
        capacity = 0.2 + 0.8 * U
        f_eq = (1.0 - np.exp(-5.0 * drive)) * capacity
        
        # Above global critical field Hc2-like, even smooth regions surrender
        if h_field > global_critical:
            global_fill = (h_field - global_critical) / max(1.0 - global_critical, 1e-8)
            f_eq = np.maximum(f_eq, global_fill)
        
        f_eq = np.clip(f_eq, 0.0, 1.0)
        
        if noise_level > 0:
            noise = np.random.normal(0.0, noise_level, size=(h, w))
            f_eq = np.clip(f_eq + noise, 0.0, 1.0)
        
        # Vortex density evolves toward equilibrium in still-superconducting regions
        active = ~c
        f[active] = f[active] + vortex_rate * (f_eq[active] - f[active])
        f[c] = 1.0
        f = np.clip(f, 0.0, 1.0)
        
        # Depinning collapse: local density exceeds critical current threshold
        new_collapse = active & (f > collapse_threshold)
        if np.any(new_collapse):
            c[new_collapse] = True
            # Cascade outward: collapsed regions weaken neighbors and snap them
            c_float = c.astype(np.float32)
            for _ in range(cascade_iterations):
                neighbor_c = convolve(c_float, kernel_3x3, mode='wrap')
                cascade_mask = (~c) & (neighbor_c > cascade_threshold) & (f > 0.03)
                if not np.any(cascade_mask):
                    break
                c[cascade_mask] = True
                c_float = c.astype(np.float32)
        
        # Compose final mixed map
        if step == 0:
            alpha = np.zeros((h, w, 1), dtype=np.float32)
        elif step == numsteps:
            alpha = np.ones((h, w, 1), dtype=np.float32)
        else:
            alpha = f[..., np.newaxis]
            alpha[c, 0] = 1.0
        
        M = (1.0 - alpha) * A + alpha * B
        M = np.clip(M * 255.0, 0, 255).astype(np.uint8)
        out_maps.append(M)
    
    return out_maps
