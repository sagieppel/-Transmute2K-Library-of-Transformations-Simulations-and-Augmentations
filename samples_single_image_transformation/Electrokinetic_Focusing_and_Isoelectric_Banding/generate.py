import numpy as np

def transform(maps, params=None, numsteps=10):
    """
    Perform a simulation of Electrokinetic Focusing and Isoelectric Banding on 
    the input multi-channel image.
    
    Parameters:
    - maps (np.ndarray): Shape [H, W, Ch], uint8. Input image/stack of maps.
    - params (dict): Custom hyperparameters for simulation (optional).
    - numsteps (int): Number of steps in the output sequence.
    
    Returns:
    - list of np.ndarray: List of transformed maps showing progress at each step.
    """
    if params is None:
        params = {}
    
    H, W, Ch = maps.shape
    
    # pI: Isoelectric points for each channel. Distributed across standard pH spectrum [0.15, 0.85]
    pI = params.get('pI', np.linspace(0.15, 0.85, Ch) if Ch > 1 else np.array([0.5]))
    pI = np.array(pI, dtype=np.float32)
    
    # beta: Local pH buffering feedback factors (how species alter the environment pH)
    beta = params.get('beta', np.linspace(-0.15, 0.15, Ch) if Ch > 1 else np.array([0.0]))
    beta = np.array(beta, dtype=np.float32)
    
    # D: Diffusion coefficient preventing collapse of bands into single-pixel widths
    D = params.get('D', 0.005)
    
    # mu: Electrophoretic mobility per channel
    mu = params.get('mu', 1.5)
    if isinstance(mu, (int, float)):
        mu = np.full(Ch, mu, dtype=np.float32)
    else:
        mu = np.array(mu, dtype=np.float32)
        
    # E_ext: Strength of external electric field
    E_ext = params.get('E_ext', 6.0)
    
    # grad_influence: Influence of the image structure (local gradient) on field lines
    grad_influence = params.get('grad_influence', 4.0)
    
    # dt: Simulation delta-time
    dt = params.get('dt', 0.25)
    
    # substeps: Inner simulation iterations per saved transition step to ensure dynamic evolution
    substeps = params.get('substeps', 5)

    # Normalize maps concentration mapping into continuous physical space [0, 1.0]
    C = maps.astype(np.float32) / 255.0
    gray = np.mean(C, axis=-1)
    
    # Precompute static spatial conductivity gradients derived from original image structure
    grad_y = (np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0)) / 2.0
    grad_x = (np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1)) / 2.0
    
    # Setup spatial coordinate meshes for Semi-Lagrangian advection and background pH scale
    Y, X = np.indices((H, W), dtype=np.float32)
    pH_bg = Y / float(max(H - 1, 1))
    
    history = []
    
    for step in range(numsteps):
        for substep in range(substeps):
            # Dynamic local pH updating (Background gradient + current species density contribution)
            pH = pH_bg + np.tensordot(C, beta, axes=((-1), (0)))
            pH = np.clip(pH, 0.0, 1.0)
            
            C_next = np.empty_like(C)
            
            for c in range(Ch):
                # Charge is relative difference from local pH environment to the species isoelectric point
                q = pH - pI[c]
                
                # Distorted Electric force field vectors
                E_x = grad_influence * grad_x
                E_y = E_ext + grad_influence * grad_y
                
                # Electrophoretic velocities driving motion
                vx = - mu[c] * q * E_x
                vy = - mu[c] * q * E_y
                
                # Semi-Lagrangian periodic advection coordinates mapping
                map_x = (X - vx * dt) % W
                map_y = (Y - vy * dt) % H
                
                x0 = np.floor(map_x).astype(np.int32)
                x0 = np.clip(x0, 0, W - 1)
                x1 = (x0 + 1) % W
                
                y0 = np.floor(map_y).astype(np.int32)
                y0 = np.clip(y0, 0, H - 1)
                y1 = (y0 + 1) % H
                
                wx = map_x - x0
                wy = map_y - y0
                
                # Standard Bilinear extraction (wrapping around boundaries naturally)
                C_chan = C[:, :, c]
                c00 = C_chan[y0, x0]
                c10 = C_chan[y0, x1]
                c01 = C_chan[y1, x0]
                c11 = C_chan[y1, x1]
                
                C_adv = (1.0 - wy) * ((1.0 - wx) * c00 + wx * c10) + wy * ((1.0 - wx) * c01 + wx * c11)
                
                # Dynamic Fickian Diffusion balance
                lap = (
                    np.roll(C_adv, 1, axis=0) + np.roll(C_adv, -1, axis=0) +
                    np.roll(C_adv, 1, axis=1) + np.roll(C_adv, -1, axis=1) - 4.0 * C_adv
                )
                
                C_next[:, :, c] = np.clip(C_adv + dt * D * lap, 0.0, 1.0)
                
            C = C_next.copy()
            
        history.append((C * 255.0).astype(np.uint8))
        
    return history