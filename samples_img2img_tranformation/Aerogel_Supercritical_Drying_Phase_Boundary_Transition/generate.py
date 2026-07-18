import numpy as np
from scipy.ndimage import map_coordinates

def transform(start_map, end_map, params=None, numsteps=10):
    """
    Transforms start_map into end_map using the Aerogel Supercritical Drying 
    Phase-Boundary Transition method.
    
    Args:
        start_map (np.ndarray): Uint8 array of shape [W, H, Ch].
        end_map (np.ndarray): Uint8 array of shape [W, H, Ch].
        params (dict, optional): Model hyperparameters.
        numsteps (int, optional): Number of frames in the transition.
        
    Returns:
        list: List of np.ndarray of shape [W, H, Ch] in uint8 format.
    """
    # Default parameters
    default_params = {
        'ior_scale': 25.0,         # Intensity of refractive distortions
        'jitter_freq': 0.12,       # Frequency of critical phase-change boundary noise
        'max_blur': 12.0,          # Maximum boiling volumetric haze
        'blueprint_detail': 2.5    # Feature scale for the aerogel structure extraction
    }
    if params is not None:
        default_params.update(params)
        
    ior_scale = default_params['ior_scale']
    jitter_freq = default_params['jitter_freq']
    max_blur = default_params['max_blur']
    blueprint_detail = default_params['blueprint_detail']
    
    w, h, ch = start_map.shape
    
    # Convert maps to float64 for calculations
    start_map_f = start_map.astype(np.float64)
    end_map_f = end_map.astype(np.float64)
    
    # Precompute coordinate grid with matrix indexing ('ij')
    y, x = np.meshgrid(np.arange(w), np.arange(h), indexing='ij')
    
    def periodic_blur(img, sigma):
        if sigma <= 0.05:
            return img.copy()
        # FFT-based periodic Gaussian blur for perfect boundary safety and speed
        u = np.fft.fftfreq(w)[:, None]
        v = np.fft.fftfreq(h)[None, :]
        kernel = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (u**2 + v**2))[:, :, None]
        
        img_fft = np.fft.fft2(img, axes=(0, 1))
        filtered_fft = img_fft * kernel
        return np.real(np.fft.ifft2(filtered_fft, axes=(0, 1)))

    def periodic_gradient(img_2d):
        # Central differences with periodic wrap-around
        dy = (np.roll(img_2d, -1, axis=0) - np.roll(img_2d, 1, axis=0)) * 0.5
        dx = (np.roll(img_2d, -1, axis=1) - np.roll(img_2d, 1, axis=1)) * 0.5
        return dy, dx

    def distort_image(img, dx, dy):
        # Distort coordinates with periodic boundary conditions
        coords_y = np.mod(y + dy, w)
        coords_x = np.mod(x + dx, h)
        coords = np.array([coords_y, coords_x])
        
        distorted = np.zeros_like(img)
        for c in range(ch):
            distorted[:, :, c] = map_coordinates(img[:, :, c], coords, order=1, mode='wrap')
        return distorted

    # Extract structural edges of End Map (The lightweight aerogel blueprint)
    b_diff = end_map_f - periodic_blur(end_map_f, blueprint_detail)
    b_edge = np.abs(b_diff)
    # Aerogel blueprint: delicate glowing edges + light base albedo
    b_blue = b_edge * 1.8 + end_map_f * 0.15 + 15.0
    b_blue = np.clip(b_blue, 0, 255)

    output_maps = []
    
    for step in range(numsteps):
        # Setup progress indicator (0 to 1)
        tau = step / (numsteps - 1) if numsteps > 1 else 1.0
        
        # Fluctuation curves representing the critical boundary phase peaking near 0.35
        w_boil = np.exp(-((tau - 0.35) / 0.25) ** 2)
        
        # 1. Hazy Boiling of Start Map
        blur_amount = w_boil * max_blur
        a_boil = periodic_blur(start_map_f, blur_amount)
        
        # 2. Refractive Field (driven by intensity gradient of the wet gel)
        l_a = np.mean(a_boil, axis=2)
        grad_y, grad_x = periodic_gradient(l_a)
        
        # High frequency phase boundary transition jitter
        omega = 2 * np.pi * jitter_freq
        phase_y = np.sin(x * omega + tau * 20.0) * np.cos(y * omega - tau * 10.0)
        phase_x = np.cos(x * omega - tau * 15.0) * np.sin(y * omega + tau * 25.0)
        
        # Absolute distortion scaled by fluctuation boundary
        dy = (grad_y * ior_scale * 0.08 + phase_y * 8.0) * w_boil
        dx = (grad_x * ior_scale * 0.08 + phase_x * 8.0) * w_boil
        
        # Apply distortion to A and B-blueprint
        a_distorted = distort_image(a_boil, dx, dy)
        b_blue_distorted = distort_image(b_blue, dx, dy)
        
        # 3. Formulate structural transition based on phase boundary progress
        if tau <= 0.5:
            # Transition from boiling A to the ghostly blueprint of B
            mix_val = tau / 0.5
            step_map = (1.0 - mix_val) * a_distorted + mix_val * b_blue_distorted
        else:
            # Condensation: Transition from ghostly blueprint back to standard opaque B
            mix_val = (tau - 0.5) / 0.5
            
            # Decaying remaining refraction vectors as structure stabilises
            w_decay = 1.0 - mix_val
            dy_decay = dy * w_decay
            dx_decay = dx * w_decay
            b_condensing = distort_image(end_map_f, dx_decay, dy_decay)
            
            step_map = (1.0 - mix_val) * b_blue_distorted + mix_val * b_condensing
            
        # Scale and format final step
        final_frame = np.clip(step_map, 0, 255).astype(np.uint8)
        output_maps.append(final_frame)
        
    return output_maps