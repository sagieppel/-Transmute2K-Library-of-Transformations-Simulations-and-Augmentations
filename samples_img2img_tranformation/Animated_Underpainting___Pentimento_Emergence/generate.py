import numpy as np

def periodic_sample(img, map_y, map_x):
    """
    Performs fast, vectorized bilinear interpolation with periodic wrap-around.
    Inputs:
      img: Numpy array of shape [H, W, C]
      map_y: Numpy array of shape [H, W] representing y-coordinates to sample
      map_x: Numpy array of shape [H, W] representing x-coordinates to sample
    Returns:
      Sampled array of shape [H, W, C] with periodic boundaries.
    """
    H, W, C = img.shape
    
    # Apply periodic modulo arithmetic
    map_y = map_y % H
    map_x = map_x % W
    
    # Clip floored indices to protect against precision edge cases where modulo can return a value rounding up to H or W.
    y0 = np.floor(map_y).astype(np.int32)
    y0 = np.clip(y0, 0, H - 1)
    y1 = (y0 + 1) % H
    
    x0 = np.floor(map_x).astype(np.int32)
    x0 = np.clip(x0, 0, W - 1)
    x1 = (x0 + 1) % W
    
    dy = map_y - y0
    dx = map_x - x0
    
    # Generate bilinear interpolation weights
    wa = ((1.0 - dy) * (1.0 - dx))[..., np.newaxis]
    wb = ((1.0 - dy) * dx)[..., np.newaxis]
    wc = (dy * (1.0 - dx))[..., np.newaxis]
    wd = (dy * dx)[..., np.newaxis]
    
    # Perform bilinear mix across all channels
    out = (img[y0, x0] * wa + 
           img[y0, x1] * wb + 
           img[y1, x0] * wc + 
           img[y1, x1] * wd)
    return out

def transform(start_map, end_map, params=None, numsteps=10):
    """
    Transforms between start_map and end_map using the Animated Underpainting / Pentimento Emergence method.
    
    Parameters:
      start_map: numpy array [H, W, C] of uint8 format (source)
      end_map: numpy array [H, W, C] of uint8 format (target)
      params: dictionary with method parameters
      numsteps: number of steps in the sequence
      
    Returns:
      A list of 'numsteps' arrays representing the animation frames from start to end.
    """
    # Default parameters
    default_params = {
        'max_displacement': 25.0,        # Maximum pentimento alignment offset (in pixels)
        'underdrawing_intensity': 0.7,   # Intensity of emerging underdrawing lines
        'stagger_spread': 0.3,           # How staggered the channels are in their transition
        'reveal_sharpness': 12.0,        # Organic edge sharpness of the reveal
        'distortion_freq': 4.0           # Frequency of the underlying canvas alignment warp
    }
    
    if params is not None:
        default_params.update(params)
    params = default_params
    
    H, W, C = start_map.shape
    
    # Convert maps to float32 normalized to [0, 1]
    A = start_map.astype(np.float32) / 255.0
    B = end_map.astype(np.float32) / 255.0
    
    # Create base grid coordinates
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Step 1: Compute a structured "underdrawing" from B containing spatial details/edges
    # Calculate gradients of targeted composition B
    L_B = np.mean(B, axis=-1)
    dy = np.roll(L_B, -1, axis=0) - np.roll(L_B, 1, axis=0)
    dx = np.roll(L_B, -1, axis=1) - np.roll(L_B, 1, axis=1)
    edges_B = np.sqrt(dx**2 + dy**2)
    max_edge = np.max(edges_B)
    if max_edge > 0:
        edges_B /= max_edge
        
    # Step 2: Set up reveal mapping based on B's low frequencies and periodic sines
    # Creates organic, region-specific transition structures
    noise_field = (
        np.sin(2 * np.pi * X / W) * np.cos(2 * np.pi * Y / H) +
        np.sin(4 * np.pi * X / W - 1.0) * np.sin(4 * np.pi * Y / H + 0.5)
    ) / 2.0
    # Blend target luminance and periodic noise to follow semantic details of B
    S = 0.6 * L_B + 0.4 * (noise_field + 0.5)
    S = (S - np.min(S)) / (np.max(S) - np.min(S) + 1e-5)
    
    # Step 3: Organize channel-by-channel delay/stagger
    # Structural/detailed channels (e.g., indices > 2) reveal earlier than color channels
    stagger = np.zeros(C, dtype=np.float32)
    if C >= 3:
        # First three channels represent RGB color (transition later)
        stagger[:3] = params['stagger_spread']
        # Remaining channels transition earlier (e.g. Depth, Normals, Gloss)
        stagger[3:] = 0.0
        # Add fine-grained stagger across indices so they don't pop at the exact same frame
        stagger += np.linspace(0.0, 0.1, C)
    else:
        stagger = np.linspace(0.0, params['stagger_spread'], C)
        
    frames = []
    
    for step in range(numsteps):
        tau = step / (numsteps - 1) if numsteps > 1 else 1.0
        out_frame = np.zeros_like(A)
        
        for c in range(C):
            # Normalized channel progress
            tau_c = np.clip((tau - stagger[c]) / (1.0 - stagger[c] + 1e-5), 0.0, 1.0)
            
            # --- A: Compositional correction shift (Pentimento offset warp) ---
            # Shift decays from maximum to 0 as tau_c moves to 1
            shift_decay = (1.0 - tau_c) ** 1.5
            
            # Add a slow canvas distortion wave
            warp_amp = params['max_displacement'] * shift_decay
            wave_y = warp_amp * np.sin(params['distortion_freq'] * np.pi * X / W + tau * np.pi)
            wave_x = warp_amp * np.cos(params['distortion_freq'] * np.pi * Y / H + tau * np.pi)
            
            # Warp coordinates for target B (it's misaligned/shifting underneath in early stages)
            # We use distinct sampling angles based on the channel to create chromatic/structural fringes
            angle = (c * (2.0 * np.pi / max(C, 1))) + (tau * np.pi * 0.2)
            shift_y = Y + wave_y + (warp_amp * np.sin(angle))
            shift_x = X + wave_x + (warp_amp * np.cos(angle))
            
            # Warp coordinates for source A slightly to react to current/displaced structural pressure
            A_shift_y = Y + (5.0 * shift_decay * np.sin(Y * 0.1))
            A_shift_x = X + (5.0 * shift_decay * np.cos(X * 0.1))
            
            # Sample both domains dynamically with periodic boundaries
            B_sampled = periodic_sample(B[:, :, c:c+1], shift_y, shift_x)[:, :, 0]
            A_sampled = periodic_sample(A[:, :, c:c+1], A_shift_y, A_shift_x)[:, :, 0]
            
            # --- B: Reveal Process --- 
            # Generate Sigmoid-based spatial reveal mask for the current progress state
            mask_c = 1.0 / (1.0 + np.exp(-params['reveal_sharpness'] * (tau_c - S)))
            
            # Linear blend of distorted / conflicting composition fields
            val_c = (1.0 - mask_c) * A_sampled + mask_c * B_sampled
            
            # --- C: Underdrawing guidelines / Pentimento Search-lines ---
            # Overlay dark structural edges representing carbon sketch guidelines
            # These lines peak in mid-transition stages (tau_c around 0.25) then vanish as paint seals it
            sketch_intensity = np.clip(1.0 - np.abs(tau_c - 0.25) / 0.25, 0.0, 1.0)
            
            if sketch_intensity > 0.0 and c < 3:
                # Sample warped guideline edges to match the current shifted underneath target
                edges_warped = periodic_sample(edges_B[:, :, np.newaxis], shift_y, shift_x)[:, :, 0]
                val_c = val_c * (1.0 - edges_warped * params['underdrawing_intensity'] * sketch_intensity)
                
            out_frame[:, :, c] = val_c
            
        # Clip limits and register output
        out_uint8 = np.clip(out_frame * 255.0, 0, 255).astype(np.uint8)
        frames.append(out_uint8)
        
    return frames