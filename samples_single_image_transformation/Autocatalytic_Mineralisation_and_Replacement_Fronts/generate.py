import numpy as np

def transform(maps, params=None, numsteps=10):
    """
    Simulates Autocatalytic Mineralisation and Replacement Fronts on a multi-channel map stack.
    The process behaves like a chemically reactive rock matrix undergoing hydrothermal fluid infiltration.
    
    Parameters:
        maps (np.ndarray): input image/map of shape [H, W, Ch] and uint8 type.
        params (dict): Optional parameters to configure the physical model:
            - 'D_f' (float): Fluid diffusion rate. Default: 0.15
            - 'k' (float): Base reaction rate. Default: 0.25
            - 'a' (float): Autocatalytic coefficient (enhances reaction). Default: 1.8
            - 'b' (float): Armoring/passivating coefficient (inhibits reaction over time). Default: 0.6
            - 'zoning_freq' (float): Spatial oscillation frequency for texture zoning. Default: 15.0
            - 'fluid_color' (list/tuple): RGB/channel components targeting the infiltration fluid phase.
        numsteps (int): Number of incremental steps to compute (defaults to 10).
        
    Returns:
        list of np.ndarray: States of the system over the simulation steps.
    """
    # Handle 2D/3D shapes gracefully
    is_2d = False
    if len(maps.shape) == 2:
        maps = maps[:, :, np.newaxis]
        is_2d = True
        
    H, W, C = maps.shape
    M_orig = maps.astype(np.float32) / 255.0

    # Load or initialize params
    if params is None:
        params = {}
    D_f = params.get('D_f', 0.15)
    k = params.get('k', 0.25)
    a = params.get('a', 1.8)
    b = params.get('b', 0.6)
    zoning_freq = params.get('zoning_freq', 15.0)
    
    # Generate a contrasting complementary fluid color vector if not provided
    mean_color = np.mean(M_orig, axis=(0, 1))
    fluid_color = params.get('fluid_color', 1.0 - mean_color)
    fluid_color = np.array(fluid_color, dtype=np.float32).reshape(1, 1, C)

    # Periodic boundary helper functions using np.roll
    def calc_periodic_gradient(img):
        # Central differences mapping across limits
        dx = np.roll(img, 1, axis=0) - np.roll(img, -1, axis=0)
        dy = np.roll(img, 1, axis=1) - np.roll(img, -1, axis=1)
        # Average spatial gradients across all channels to represent mechanical pathways
        mag = np.sqrt(dx**2 + dy**2)
        return np.mean(mag, axis=-1, keepdims=True)

    def periodic_laplacian(field):
        # 5-point discrete Laplacian stencil with periodic boundaries
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4.0 * field
        )

    # 1. Compute permeability (P) based on image gradient field (pathways of least resistance)
    grad = calc_periodic_gradient(M_orig)
    grad_max = np.max(grad) + 1e-8
    P = grad / grad_max
    P = np.power(P, 0.4)  # Enhance trace connectivity

    # 2. Setup dynamical variables
    # Fluid concentrated along high permeability pathways
    F = P.copy() 
    # Mineral modification progression mapping [0, 1]
    M = np.zeros((H, W, 1), dtype=np.float32)
    # Chemical reactivity field R: mapped from underlying matrix density/lightness
    R = np.mean(M_orig, axis=-1, keepdims=True)
    R = 0.2 + 0.8 * R  # Ensure even darker/low-reactive areas can slowly participate

    output_states = []

    # Numerical simulation loop
    for step in range(numsteps):
        # Fluid propagates along high-permeability channels periodically
        dF = D_f * periodic_laplacian(F) * (P + 0.05)
        F = np.clip(F + dF, 0.0, 1.0)
        F = np.maximum(F, P)  # Constant fluid injection source at primary pathways

        # Autocatalytic replacement reaction kinetics
        # Feedback: accelerates with M (autocatalysis), decelerates as (1 - b*M) (passivating armor)
        accelerator = 1.0 + a * M
        passivator = np.clip(1.0 - b * M, 0.0, 1.0)
        
        reaction_rate = k * F * R * accelerator * passivator * (1.0 - M)
        M = np.clip(M + reaction_rate, 0.0, 1.0)

        # Multi-layered rhythmic crystallization zoning effect
        zoning = 0.5 + 0.5 * np.sin(zoning_freq * M + step * 0.4)
        zoned_color = fluid_color * zoning

        # Pseudomorphic composition: original matrix slowly replaced by zoning mineral phase
        mixed_state = (1.0 - M) * M_orig + M * zoned_color
        
        # Scale back and cast to typical range
        state_uint8 = np.clip(mixed_state * 255.0, 0, 255).astype(np.uint8)
        
        if is_2d:
            output_states.append(state_uint8[:, :, 0])
        else:
            output_states.append(state_uint8)

    return output_states