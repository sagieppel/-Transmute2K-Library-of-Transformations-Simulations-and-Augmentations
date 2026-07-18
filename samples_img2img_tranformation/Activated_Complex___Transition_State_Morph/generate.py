import numpy as np
from scipy.ndimage import gaussian_filter, sobel, distance_transform_edt


def _to_float01(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    if x.max() > 1.0 or x.min() < 0.0:
        x = np.clip(x / 255.0, 0.0, 1.0)
    return x


def _ensure_3d(x):
    if x.ndim == 2:
        return x[..., None]
    return x


def _gauss_multich(x, sigma):
    if sigma <= 0:
        return x
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        out[..., c] = gaussian_filter(x[..., c], sigma=sigma, mode='reflect')
    return out


def _laplacian_multich(x):
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        xc = x[..., c]
        out[..., c] = (
            -4.0 * xc
            + np.roll(xc, 1, axis=0) + np.roll(xc, -1, axis=0)
            + np.roll(xc, 1, axis=1) + np.roll(xc, -1, axis=1)
        )
    return out


def _gradient_mag_scalar(img2d):
    gx = sobel(img2d, axis=1, mode='reflect')
    gy = sobel(img2d, axis=0, mode='reflect')
    return np.sqrt(gx * gx + gy * gy)


def _channelwise_edge_energy(x):
    h, w, ch = x.shape
    acc = np.zeros((h, w), dtype=np.float32)
    for c in range(ch):
        acc += _gradient_mag_scalar(x[..., c])
    acc /= max(ch, 1)
    return acc


def _make_feature_fields(start, end, edge_power=1.0, blur_sigma=2.0):
    # Difference field: where states disagree strongly, activation can happen earlier.
    diff = np.mean(np.abs(end - start), axis=2)

    # Structural complexity from edges/corners proxy.
    edges_s = _channelwise_edge_energy(start)
    edges_e = _channelwise_edge_energy(end)
    edges = 0.5 * (edges_s + edges_e)

    # Normalize.
    def norm01(a):
        a = a.astype(np.float32)
        mn = float(a.min())
        mx = float(a.max())
        if mx - mn < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return (a - mn) / (mx - mn)

    diff_n = norm01(diff)
    edges_n = norm01(edges) ** edge_power

    # Barrier field: high on strong structure, lower where maps differ strongly.
    barrier = 0.65 * edges_n + 0.35 * (1.0 - diff_n)
    barrier = norm01(_gauss_multich(barrier[..., None], blur_sigma)[..., 0])

    # Multiple possible pathways / seeds for local reconfiguration.
    seed_strength = norm01(0.7 * diff_n + 0.3 * edges_n)
    seed_strength = norm01(gaussian_filter(seed_strength, sigma=max(0.5, blur_sigma * 0.5), mode='reflect'))

    return diff_n, edges_n, barrier, seed_strength


def _build_transition_schedule(seed_strength, barrier, numsteps, front_sigma=1.2):
    h, w = seed_strength.shape

    # Choose active seeds from the strongest regions; deterministic threshold by percentile.
    perc = 92.0
    thr = np.percentile(seed_strength, perc)
    seeds = seed_strength >= thr
    if not np.any(seeds):
        seeds[np.unravel_index(np.argmax(seed_strength), seed_strength.shape)] = True

    # Distance from seed front.
    dist = distance_transform_edt(~seeds).astype(np.float32)
    dist = gaussian_filter(dist, sigma=front_sigma, mode='reflect')
    dist /= (dist.max() + 1e-8)

    # Local activation time: farther + higher barrier => later activation.
    t_activate = 0.55 * dist + 0.45 * barrier
    t_activate -= t_activate.min()
    t_activate /= (t_activate.max() + 1e-8)

    # Local dwell width in activated complex state.
    width = 0.12 + 0.18 * barrier + 0.10 * (1.0 - seed_strength)
    width = np.clip(width, 0.08, 0.35)

    # Convert to per-step schedule helper arrays.
    times = np.linspace(0.0, 1.0, numsteps, dtype=np.float32)
    return times, t_activate.astype(np.float32), width.astype(np.float32), seeds


def transform(start_map, end_map, params=None, numsteps=10):
    """
    Activated Complex / Transition-State Morph

    Inputs:
        start_map, end_map: uint8 arrays of shape [h,w,ch] or [h,w]
        params: optional dict
        numsteps: number of output frames

    Returns:
        list of uint8 arrays with same shape as input
    """
    if params is None:
        params = {}

    start_in = np.asarray(start_map)
    end_in = np.asarray(end_map)
    if start_in.shape != end_in.shape:
        raise ValueError('start_map and end_map must have the same shape')
    if start_in.ndim not in (2, 3):
        raise ValueError('Inputs must have shape [h,w] or [h,w,ch]')

    orig_2d = (start_in.ndim == 2)
    start = _ensure_3d(_to_float01(start_in))
    end = _ensure_3d(_to_float01(end_in))

    h, w, ch = start.shape

    # Parameters controlling the toy physical model.
    edge_power = float(params.get('edge_power', 1.2))
    feature_blur_sigma = float(params.get('feature_blur_sigma', 2.0))
    front_sigma = float(params.get('front_sigma', 1.0))
    transient_strength = float(params.get('transient_strength', 0.32))
    transient_smoothing = float(params.get('transient_smoothing', 1.0))
    transient_sharpen = float(params.get('transient_sharpen', 0.75))
    channel_coupling = float(params.get('channel_coupling', 0.30))
    overshoot_bias = float(params.get('overshoot_bias', 0.50))
    relax_pow = float(params.get('relax_pow', 1.6))
    noise_strength = float(params.get('noise_strength', 0.06))
    seed = int(params.get('seed', 0))

    rng = np.random.default_rng(seed)

    # Build image-derived energy landscape and local activation schedule.
    diff_n, edges_n, barrier, seed_strength = _make_feature_fields(
        start, end, edge_power=edge_power, blur_sigma=feature_blur_sigma
    )
    times, t_activate, width, seeds = _build_transition_schedule(
        seed_strength, barrier, numsteps=numsteps, front_sigma=front_sigma
    )

    # Channel-wise disagreement controls asynchronous channel conversion.
    ch_diff = np.mean(np.abs(end - start), axis=(0, 1))
    if ch_diff.max() > ch_diff.min():
        ch_phase = (ch_diff - ch_diff.min()) / (ch_diff.max() - ch_diff.min() + 1e-8)
    else:
        ch_phase = np.zeros_like(ch_diff)
    ch_phase = (ch_phase - np.mean(ch_phase)) * 0.16  # modest offset per channel

    # Shared transient third-state field: neither source nor target.
    # Derived from local average + curvature + noise, then relaxed over time.
    mean_state = 0.5 * (start + end)
    smooth_mean = _gauss_multich(mean_state, sigma=transient_smoothing)
    lap_s = _laplacian_multich(start)
    lap_e = _laplacian_multich(end)
    curvature = 0.5 * (lap_s - lap_e)

    scalar_noise = rng.standard_normal((h, w), dtype=np.float32)
    scalar_noise = gaussian_filter(scalar_noise, sigma=2.0, mode='reflect')
    scalar_noise /= (np.std(scalar_noise) + 1e-8)
    scalar_noise = scalar_noise[..., None]

    # Signed overshoot direction favors moving away from simple midpoint.
    sign_dir = np.sign((end - start) + 1e-6)
    sign_dir = overshoot_bias * sign_dir + (1.0 - overshoot_bias) * np.sign(curvature + 1e-6)

    activated_state = smooth_mean + transient_sharpen * curvature + noise_strength * scalar_noise * sign_dir
    activated_state = np.clip(activated_state, 0.0, 1.0)

    # Global scalar field for consistent coupled motion across all channels/maps.
    common_shift = _gauss_multich((end - start), sigma=1.5)

    frames = []
    for ti, t in enumerate(times):
        # Per-pixel activation envelope.
        # Before activation: source basin. Around activation: unstable third state.
        # After activation: relax to target basin.
        pre = t < t_activate
        post = t > (t_activate + width)
        mid = ~(pre | post)

        # Progress inside activated regime.
        tau = np.zeros((h, w), dtype=np.float32)
        tau[mid] = (t - t_activate[mid]) / (width[mid] + 1e-8)
        tau = np.clip(tau, 0.0, 1.0)

        # Build weights.
        w_start = np.zeros((h, w), dtype=np.float32)
        w_mid = np.zeros((h, w), dtype=np.float32)
        w_end = np.zeros((h, w), dtype=np.float32)

        w_start[pre] = 1.0
        # In the activated regime, leave source quickly, linger in third state.
        w_start[mid] = (1.0 - tau[mid]) ** 2
        w_mid[mid] = 4.0 * tau[mid] * (1.0 - tau[mid])

        # After activated regime, relax with local barrier-controlled rate.
        post_prog = np.zeros((h, w), dtype=np.float32)
        post_prog[post] = (t - (t_activate[post] + width[post])) / (1.0 - (t_activate[post] + width[post]) + 1e-8)
        post_prog = np.clip(post_prog, 0.0, 1.0)
        local_relax = post_prog ** (relax_pow + 1.2 * barrier)
        w_end[post] = local_relax[post]
        w_mid[post] = 1.0 - w_end[post]

        # Normalize within numerical tolerance.
        wsum = w_start + w_mid + w_end + 1e-8
        w_start /= wsum
        w_mid /= wsum
        w_end /= wsum

        # Base coupled state.
        frame = (
            w_start[..., None] * start +
            w_mid[..., None] * activated_state +
            w_end[..., None] * end
        )

        # Add coherent transient reorganization using image-derived field.
        # Strongest near the activated complex, weaker elsewhere.
        act_amp = (w_mid * transient_strength * (0.35 + 0.65 * diff_n) * (0.25 + 0.75 * (1.0 - barrier))).astype(np.float32)
        coherent_push = act_amp[..., None] * common_shift
        frame = frame + coherent_push

        # Channel-specific timing while still coupled to the same spatial event.
        if ch > 1 and channel_coupling > 0.0:
            ch_t = np.clip(t + ch_phase, 0.0, 1.0)
            for c in range(ch):
                local_tc = ch_t[c]
                pre_c = local_tc < t_activate
                post_c = local_tc > (t_activate + width)
                mid_c = ~(pre_c | post_c)

                tau_c = np.zeros((h, w), dtype=np.float32)
                tau_c[mid_c] = (local_tc - t_activate[mid_c]) / (width[mid_c] + 1e-8)
                tau_c = np.clip(tau_c, 0.0, 1.0)

                ws = np.zeros((h, w), dtype=np.float32)
                wm = np.zeros((h, w), dtype=np.float32)
                we = np.zeros((h, w), dtype=np.float32)
                ws[pre_c] = 1.0
                ws[mid_c] = (1.0 - tau_c[mid_c]) ** 2
                wm[mid_c] = 4.0 * tau_c[mid_c] * (1.0 - tau_c[mid_c])

                pp = np.zeros((h, w), dtype=np.float32)
                pp[post_c] = (local_tc - (t_activate[post_c] + width[post_c])) / (1.0 - (t_activate[post_c] + width[post_c]) + 1e-8)
                pp = np.clip(pp, 0.0, 1.0)
                lr = pp ** (relax_pow + 1.2 * barrier)
                we[post_c] = lr[post_c]
                wm[post_c] = 1.0 - we[post_c]
                ssum = ws + wm + we + 1e-8
                ws /= ssum
                wm /= ssum
                we /= ssum

                ch_frame = ws * start[..., c] + wm * activated_state[..., c] + we * end[..., c]
                ch_frame = ch_frame + channel_coupling * (act_amp * common_shift[..., c])
                # Mix channel-specific path with globally coupled path.
                frame[..., c] = (1.0 - channel_coupling) * frame[..., c] + channel_coupling * ch_frame

        frame = np.clip(frame, 0.0, 1.0)
        out = (frame * 255.0 + 0.5).astype(np.uint8)
        if orig_2d:
            out = out[..., 0]
        frames.append(out)

    # Guarantee exact endpoints if requested number of steps includes them.
    if len(frames) > 0:
        frames[0] = start_in.copy()
        frames[-1] = end_in.copy()

    return frames


if __name__ == '__main__':
    # Minimal self-test.
    h, w, ch = 128, 128, 6
    a = np.zeros((h, w, ch), dtype=np.uint8)
    b = np.zeros((h, w, ch), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    a[..., 0] = np.clip((xx / w) * 255, 0, 255).astype(np.uint8)
    a[..., 1] = np.clip((yy / h) * 255, 0, 255).astype(np.uint8)
    a[..., 2] = (((xx - 64) ** 2 + (yy - 64) ** 2) < 30 ** 2).astype(np.uint8) * 255
    a[..., 3] = ((xx // 8) % 2).astype(np.uint8) * 255
    a[..., 4] = ((yy // 8) % 2).astype(np.uint8) * 255
    a[..., 5] = 64

    b[..., 0] = np.clip((1 - xx / w) * 255, 0, 255).astype(np.uint8)
    b[..., 1] = np.clip((1 - yy / h) * 255, 0, 255).astype(np.uint8)
    b[..., 2] = (((xx - 32) ** 2 + (yy - 96) ** 2) < 22 ** 2).astype(np.uint8) * 255
    b[..., 3] = ((yy // 6) % 2).astype(np.uint8) * 255
    b[..., 4] = ((xx // 6) % 2).astype(np.uint8) * 255
    b[..., 5] = 192

    seq = transform(a, b, numsteps=8)
    print(len(seq), seq[0].shape, seq[-1].dtype)
