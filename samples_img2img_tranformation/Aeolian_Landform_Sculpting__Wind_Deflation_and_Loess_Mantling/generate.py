import numpy as np

def transform(start_map, end_map, params=None, numsteps=10):
    """
    Performs a transition between start_map (bedrock undergoing deflation) and 
    end_map (loess undergoing mantling) inspired by Aeolian geomorphology.
    
    Parameters:
        start_map: numpy array [H, W, C] of uint8 format
        end_map: numpy array [H, W, C] of uint8 format
        params: optional dict of parameters
        numsteps: number of steps in transformation output list
    """
    if params is None:
        params = {}
        
    # Retrieve hyperparameters with physical defaults
    wind_angle = params.get('wind_angle', 0.785)          # Angle of the wind sweep front (~45 degrees)
    hardness_influence = params.get('hardness_influence', 0.35)  # Scale of bedrock structural defiance
    shelter_influence = params.get('shelter_influence', 0.25)    # Scale of loess deposition in valleys/shelters
    front_ruggedness = params.get('front_ruggedness', 0.20)      # Geological frontal waviness (noise factor)
    transition_width = params.get('transition_width', 0.08)      # Spatial softness of the actual wind line
    channel_lag = params.get('channel_lag', 0.15)                # Weathering lag between channels (strata)

    H, W, C = start_map.shape
    
    # Convert inputs to float32 for vectorized scientific pipeline
    start_f = start_map.astype(np.float32) / 255.0
    end_f = end_map.astype(np.float32) / 255.0
    
    # Vectorized gradient calculation with periodic borders supporting any channel dim
    def periodic_grad(img):
        dx = img - np.roll(img, 1, axis=0)
        dy = img - np.roll(img, 1, axis=1)
        return np.sqrt(dx**2 + dy**2)
        
    # Calculate rock "hardness" using gradient magnitudes of start_map (edges resist deflation)
    grad_start = periodic_grad(start_f)
    # Perform a local periodic box-blur to simulate structural preservation context
    hardness = grad_start.copy()
    for shift in [-1, 1]:
        hardness += np.roll(grad_start, shift, axis=0) + np.roll(grad_start, shift, axis=1)
    hardness /= 5.0
    
    # Normalize hardness per channel
    h_min = hardness.min(axis=(0, 1), keepdims=True)
    h_max = hardness.max(axis=(0, 1), keepdims=True) + 1e-6
    hardness = (hardness - h_min) / h_max
    
    # Create a "shelter" map based on the architecture of end_map (fine sand deposits in calm basins first)
    grad_end = periodic_grad(end_f)
    shelter = 1.0 - (grad_end / (grad_end.max(axis=(0,1), keepdims=True) + 1e-6))
    
    # Projection matrix aligned to sweeping wind vector
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    wind_proj = x_coords * np.cos(wind_angle) + y_coords * np.sin(wind_angle)
    wind_proj = (wind_proj - wind_proj.min()) / (wind_proj.max() - wind_proj.min() + 1e-6)
    wind_proj = np.expand_dims(wind_proj, axis=-1) # Shape: [H, W, 1]
    
    # Generate low frequency periodic background noise
    noise = np.random.randn(H, W, 1).astype(np.float32)
    for _ in range(4):
        noise = (noise + np.roll(noise, 2, axis=0) + np.roll(noise, -2, axis=0) +
                 np.roll(noise, 2, axis=1) + np.roll(noise, -2, axis=1)) / 5.0
    n_min = noise.min()
    n_max = noise.max() + 1e-6
    noise = (noise - n_min) / n_max
    
    # Formulate activation/transformation coordinate field
    # High hardness delays transformation, high deposition shelter accelerates it
    base_activation = wind_proj + (front_ruggedness * noise) + (hardness_influence * hardness) - (shelter_influence * shelter)
    
    # Shift activation across channels to simulate varying weather properties of material strata
    if C > 1:
        lag_offsets = np.linspace(-channel_lag, channel_lag, C).reshape(1, 1, C)
        activation = base_activation + lag_offsets
    else:
        activation = base_activation
        
    # Re-normalize globally per-channel to maintain start-to-end bounds
    act_min = activation.min(axis=(0,1), keepdims=True)
    act_max = activation.max(axis=(0,1), keepdims=True) + 1e-6
    activation = (activation - act_min) / (act_max - act_min)
    
    steps_list = []
    for step in range(numsteps):
        t = step / (numsteps - 1) if numsteps > 1 else 0.5
        
        # Compensate so physical front sweeps beyond borders cleanly
        t_scaled = t * (1.0 + 2.0 * transition_width) - transition_width
        
        # Local blending mask modulated by the complex geomorphology field
        val = (t_scaled - activation) / (transition_width + 1e-6)
        blend_mask = np.clip(val + 0.5, 0.0, 1.0)
        
        # Final blend composite (0 = raw bedrock, 1 = loess mantling)
        stepped_frame = (1.0 - blend_mask) * start_f + blend_mask * end_f
        
        # Convert back to uint8 representation
        stepped_frame_uint8 = np.clip(stepped_frame * 255.0, 0, 255).astype(np.uint8)
        steps_list.append(stepped_frame_uint8)
        
    return steps_list