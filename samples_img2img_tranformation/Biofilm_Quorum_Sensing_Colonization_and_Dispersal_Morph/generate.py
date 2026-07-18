import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
except Exception:  # pragma: no cover
    _scipy_gaussian_filter = None


def _normalize01(x, eps=1e-6):
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _blur2d(x, sigma):
    """Fast 2-D blur for scalar fields. Uses scipy if available, otherwise a cheap repeated 3x3 diffusion blur."""
    if sigma <= 0:
        return x
    if _scipy_gaussian_filter is not None:
        return _scipy_gaussian_filter(x, sigma=float(sigma), mode="nearest").astype(np.float32, copy=False)

    # Fallback: approximate Gaussian by repeated weighted 3x3 blur.
    y = x.astype(np.float32, copy=True)
    passes = max(1, int(round(float(sigma) * 1.5)))
    for _ in range(passes):
        p = np.pad(y, 1, mode="edge")
        y = (
            4.0 * p[1:-1, 1:-1]
            + 2.0 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
            + (p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:])
        ) / 16.0
    return y.astype(np.float32, copy=False)


def transform(start_map, end_map, params=None, numsteps=10):
    """
    Biofilm Quorum-Sensing Colonization and Dispersal Morph.

    Parameters
    ----------
    start_map : np.ndarray, uint8, shape (H, W, C) or (H, W)
        Source map/material channels.
    end_map : np.ndarray, uint8, same shape as start_map
        Target map/material channels.
    params : dict or None
        Optional model parameters. Important defaults are listed below.
    numsteps : int
        Number of output frames to return.

    Returns
    -------
    list[np.ndarray]
        List of uint8 maps, each with the same shape as start_map.

    Model summary
    -------------
    The simulation keeps three spatially explicit scalar fields shared by all channels:
      C = attached/immature cell coverage
      M = mature quorum-locked biofilm matrix, target-like
      D = dispersal signal halo, source-like and anti-colonizing

    Source/target pixel statistics actively define attachment affinity, nutrient availability,
    and spatial quorum threshold. All output channels are blended with the same biological state
    field, giving consistent phase-transition behavior for RGB, PBR, or arbitrary channel stacks.
    """
    if params is None:
        params = {}

    start_arr = np.asarray(start_map)
    end_arr = np.asarray(end_map)
    if start_arr.shape != end_arr.shape:
        raise ValueError("start_map and end_map must have the same shape")
    if start_arr.ndim not in (2, 3):
        raise ValueError("start_map and end_map must have shape (H,W) or (H,W,C)")
    if numsteps < 1:
        return []

    original_ndim = start_arr.ndim
    if original_ndim == 2:
        s = start_arr[:, :, None].astype(np.float32) / 255.0
        e = end_arr[:, :, None].astype(np.float32) / 255.0
    else:
        s = start_arr.astype(np.float32) / 255.0
        e = end_arr.astype(np.float32) / 255.0

    h, w, ch = s.shape

    # ----------------------------
    # Parameters
    # ----------------------------
    seed = int(params.get("seed", 12345))
    rng = np.random.default_rng(seed)

    steps_per_frame = int(params.get("steps_per_frame", 5))
    steps_per_frame = max(1, steps_per_frame)

    seed_density = float(params.get("seed_density", 0.006))
    seed_strength = float(params.get("seed_strength", 0.75))

    quorum_sigma = float(params.get("quorum_sigma", 2.2))
    colonize_sigma = float(params.get("colonize_sigma", 1.2))
    dispersal_sigma = float(params.get("dispersal_sigma", 5.0))

    base_threshold = float(params.get("base_threshold", 0.42))
    threshold_affinity_weight = float(params.get("threshold_affinity_weight", 0.25))
    threshold_nutrient_weight = float(params.get("threshold_nutrient_weight", 0.15))
    threshold_relaxation = float(params.get("threshold_relaxation", 0.45))

    growth_rate = float(params.get("growth_rate", 0.42))
    maturation_rate = float(params.get("maturation_rate", 0.28))
    quorum_sharpness = float(params.get("quorum_sharpness", 12.0))
    quorum_memory = float(params.get("quorum_memory", 0.65))
    mature_signal_boost = float(params.get("mature_signal_boost", 0.45))

    dispersal_production = float(params.get("dispersal_production", 0.55))
    dispersal_decay = float(params.get("dispersal_decay", 0.18))
    dispersal_strength = float(params.get("dispersal_strength", 0.55))
    recolonization_rate = float(params.get("recolonization_rate", 0.09))

    immature_opacity = float(params.get("immature_opacity", 0.35))
    source_halo_strength = float(params.get("source_halo_strength", 0.80))
    completion_power = float(params.get("completion_power", 4.5))

    # ----------------------------
    # Image-derived substrate chemistry/topology fields
    # ----------------------------
    src_scalar = np.mean(s, axis=2).astype(np.float32)
    dst_scalar = np.mean(e, axis=2).astype(np.float32)
    delta_scalar = np.mean(np.abs(e - s), axis=2).astype(np.float32)

    # Roughness/topology proxy: source spatial gradient magnitude.
    gy, gx = np.gradient(src_scalar)
    grad_mag = _normalize01(np.sqrt(gx * gx + gy * gy))

    src_norm = _normalize01(src_scalar)
    dst_norm = _normalize01(dst_scalar)
    delta_norm = _normalize01(delta_scalar)

    # Attachment affinity: high on bright/nutrient-like, rough, and not-too-hostile source regions.
    affinity_raw = (
        1.35 * (src_norm - 0.5)
        + 1.15 * (grad_mag - 0.5)
        + 0.65 * (1.0 - delta_norm - 0.5)
    )
    affinity = _sigmoid(2.4 * affinity_raw).astype(np.float32)

    # Nutrient availability: places where either source or target appears resource-rich, plus change-front potential.
    nutrient = _normalize01(0.45 * src_norm + 0.35 * dst_norm + 0.20 * delta_norm).astype(np.float32)
    nutrient = np.clip(0.15 + 0.85 * nutrient, 0.0, 1.0)

    # Spatial quorum threshold: easier quorum on high-affinity / high-nutrient substrate.
    threshold = (
        base_threshold
        + threshold_affinity_weight * (1.0 - affinity)
        + threshold_nutrient_weight * (1.0 - nutrient)
    ).astype(np.float32)
    threshold = np.clip(threshold, 0.05, 0.95)

    # ----------------------------
    # Biological fields
    # ----------------------------
    seed_probability = seed_density * (0.15 + 1.85 * affinity) ** 2
    seeds = (rng.random((h, w), dtype=np.float32) < seed_probability).astype(np.float32)

    C = np.clip(seed_strength * seeds + 0.025 * affinity * rng.random((h, w), dtype=np.float32), 0.0, 1.0).astype(np.float32)
    M = np.zeros((h, w), dtype=np.float32)  # mature biofilm matrix
    D = np.zeros((h, w), dtype=np.float32)  # dispersal signal
    Q = np.zeros((h, w), dtype=np.float32)  # quorum molecule concentration

    outputs = []
    total_iters = max(1, numsteps * steps_per_frame)

    for frame in range(numsteps):
        for _ in range(steps_per_frame):
            it_index = frame * steps_per_frame + _ + 1
            progress = it_index / float(total_iters)

            # Local quorum molecule: produced by coverage and mature matrix, diffuses spatially.
            biomass = np.clip(C + M, 0.0, 1.0)
            local_biomass = _blur2d(biomass, quorum_sigma)
            mature_signal = _blur2d(M, max(0.1, quorum_sigma * 0.75))
            Q_new = local_biomass + mature_signal_boost * mature_signal
            Q = quorum_memory * Q + (1.0 - quorum_memory) * Q_new

            # The global continuation pressure lowers thresholds over time, like an increasingly permissive medium.
            T_eff = threshold * (1.0 - threshold_relaxation * progress)
            T_eff = np.clip(T_eff, 0.02, 0.95)
            quorum_gate = _sigmoid(quorum_sharpness * (Q - T_eff)).astype(np.float32)

            # Mature islands emit a diffusing dispersal signal. The halo is strongest outside mature areas.
            diffused_mature = _blur2d(M, dispersal_sigma)
            halo_source = np.clip(diffused_mature - M, 0.0, 1.0)
            D = (1.0 - dispersal_decay) * D + dispersal_production * halo_source
            D = np.clip(D, 0.0, 1.0)

            # Colonization grows from existing biomass and image-derived affinity/nutrient fields.
            neighborhood = _blur2d(np.clip(C + 0.70 * M, 0.0, 1.0), colonize_sigma)
            grow = (
                growth_rate
                * (0.10 + 0.90 * affinity)
                * (0.20 + 0.80 * nutrient)
                * (0.15 + neighborhood)
                * (1.0 - C)
                * (1.0 - M)
            )
            C = C + grow

            # Dispersal dissolves immature coverage around mature islands, reverting those halos toward source.
            dissolve = dispersal_strength * D * (1.0 - M) * (1.15 - 0.85 * quorum_gate)
            dissolve = np.clip(dissolve, 0.0, 0.95)
            C = C * (1.0 - dissolve)

            # Dispersal halos become future colonization fronts after partial decay/recovery.
            recolonize = recolonization_rate * D * (1.0 - D) * affinity * nutrient * (1.0 - C) * (1.0 - M)
            C = C + recolonize

            # Quorum-locked patches mature into target-like matrix.
            mature_rate = maturation_rate * quorum_gate * (0.25 + 0.75 * nutrient) * (1.0 - 0.60 * D)
            mature_rate = np.clip(mature_rate, 0.0, 1.0)
            dM = mature_rate * C * (1.0 - M)
            M = M + dM

            # Some attached cells are consumed/encapsulated into matrix.
            C = C * (1.0 - 0.20 * mature_rate) + 0.02 * dM

            C = np.clip(C, 0.0, 1.0).astype(np.float32, copy=False)
            M = np.clip(M, 0.0, 1.0).astype(np.float32, copy=False)

        # Render this frame. Mature matrix is target-like; dispersal halos suppress target influence.
        progress = (frame + 1) / float(numsteps)
        alpha = M + immature_opacity * C * (1.0 - M)
        alpha = alpha * (1.0 - source_halo_strength * D * (1.0 - M))
        alpha = np.clip(alpha, 0.0, 1.0)

        # Ensure the process can complete as an image morph, while keeping early/mid frames biofilm-driven.
        completion = progress ** completion_power
        alpha = alpha * (1.0 - completion) + completion
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        out = s * (1.0 - alpha[:, :, None]) + e * alpha[:, :, None]
        out_u8 = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
        if original_ndim == 2:
            out_u8 = out_u8[:, :, 0]
        outputs.append(out_u8)

    return outputs
