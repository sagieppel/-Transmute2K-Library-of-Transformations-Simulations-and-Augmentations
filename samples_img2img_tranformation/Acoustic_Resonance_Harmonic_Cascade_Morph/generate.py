import numpy as np


def _norm01(x, lo=1.0, hi=99.0, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    a, b = np.percentile(x, (lo, hi))
    if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(float(b - a)) < eps:
        mn = float(np.min(x))
        mx = float(np.max(x))
        if abs(mx - mn) < eps:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - mn) / (mx - mn + eps), 0.0, 1.0).astype(np.float32)
    return np.clip((x - a) / (b - a + eps), 0.0, 1.0).astype(np.float32)


def _box_blur_periodic(x, passes=1):
    y = np.asarray(x, dtype=np.float32)
    for _ in range(int(max(0, passes))):
        y = (y + np.roll(y, 1, axis=0) + np.roll(y, -1, axis=0) + np.roll(y, 1, axis=1) + np.roll(y, -1, axis=1)) * 0.2
    return y.astype(np.float32)


def _to_uint8_frame(x, original_ndim):
    y = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    if original_ndim == 2:
        y = y[..., 0]
    return y


def transform(start_map, end_map, params=None, numsteps=10):
    '''
    Acoustic Resonance Harmonic Cascade Morph, approximate image/map implementation.

    start_map and end_map are uint8 arrays of shape HxWxC, with any number of channels.
    The method derives an acoustic impedance field from the current multi-channel material,
    excites it with a fundamental and harmonic frequencies, estimates standing-wave energy
    by FFT band-pass resonators, accumulates cavitation damage at pressure antinodes, and
    uses that damage field as the spatially coherent phase-transition coordinate from start
    to end. The same local transition coordinate is applied to every channel, preserving
    consistency across RGB/PBR/arbitrary map channels.
    '''
    if params is None:
        params = {}

    defaults = {
        'fundamental_cycles': 3.5,
        'harmonics': [1, 2, 3, 4, 6],
        'q_factor': 4.0,
        'impedance_contrast': 1.35,
        'boundary_gain': 1.20,
        'nonlinear_gain': 0.55,
        'harmonic_decay': 0.75,
        'cavitation_threshold': 0.56,
        'streaming_strength': 0.35,
        'streaming_blur_passes': 2,
        'damage_memory': 0.985,
        'resonance_strength': 0.42,
        'damage_acceleration': 0.10,
        'resonance_window_power': 0.75,
        'alpha_blur_passes': 1,
        'max_lead': 0.34,
        'max_lag': 0.28,
        'monotonic': True
    }
    p = defaults.copy()
    p.update(params)

    start_arr = np.asarray(start_map)
    end_arr = np.asarray(end_map)
    if start_arr.shape != end_arr.shape:
        raise ValueError('start_map and end_map must have the same shape')
    if start_arr.ndim not in (2, 3):
        raise ValueError('start_map and end_map must be HxW or HxWxC arrays')

    original_ndim = start_arr.ndim
    if original_ndim == 2:
        start_arr = start_arr[..., None]
        end_arr = end_arr[..., None]

    h, w, ch = start_arr.shape
    numsteps = int(max(1, numsteps))

    s0 = start_arr.astype(np.float32) / 255.0
    e0 = end_arr.astype(np.float32) / 255.0
    delta = e0 - s0

    if numsteps == 1:
        return [_to_uint8_frame(e0, original_ndim)]

    eps = 1e-7

    # Deterministic channel mixing: all channels contribute to impedance, but the same
    # acoustic transition coordinate is later applied to all channels for material coherence.
    idx = np.arange(ch, dtype=np.float32)
    weights = np.sin((idx + 1.0) * 1.324717957) + 0.5 * np.cos((idx + 1.0) * 2.2360679)
    weights = weights - np.mean(weights)
    if np.sum(np.abs(weights)) > eps:
        weights = weights / (np.sum(np.abs(weights)) + eps)
    else:
        weights = np.zeros_like(weights)

    # Frequency grid in cycles per image. Each harmonic is represented by a circular
    # resonant bandpass transfer function, a cheap spectral approximation to a Helmholtz
    # standing-wave response in an inhomogeneous impedance guide.
    fy = np.fft.fftfreq(h).astype(np.float32) * float(h)
    fx = np.fft.fftfreq(w).astype(np.float32) * float(w)
    kx, ky = np.meshgrid(fx, fy)
    kr = np.sqrt(kx * kx + ky * ky).astype(np.float32)

    harmonics = np.asarray(p['harmonics'], dtype=np.float32)
    harmonics = harmonics[harmonics > 0]
    if harmonics.size == 0:
        harmonics = np.asarray([1.0], dtype=np.float32)

    fundamental = float(p['fundamental_cycles'])
    q = max(float(p['q_factor']), 0.25)
    filters = []
    for hm in harmonics:
        k0 = max(0.05, fundamental * float(hm))
        bw = max(0.45, k0 / q)
        filt = np.exp(-0.5 * ((kr - k0) / bw) ** 2).astype(np.float32)
        filt[0, 0] = 0.0
        filters.append(filt)

    def impedance_field(material):
        mean = np.mean(material, axis=2)
        if ch > 1:
            # std and modal projection allow non-RGB/PBR channels to affect impedance too.
            std = np.std(material, axis=2)
            modal = np.tensordot(material, weights, axes=([2], [0])).astype(np.float32)
            modal = _norm01(modal, 1.0, 99.0)
            field = 0.68 * mean + 0.22 * std + 0.10 * modal
        else:
            field = mean
        field = _box_blur_periodic(field, passes=1)
        z = 1.0 + float(p['impedance_contrast']) * (field - 0.5)
        return z.astype(np.float32)

    def impedance_boundary(z):
        dx = z - np.roll(z, 1, axis=1)
        dy = z - np.roll(z, 1, axis=0)
        b = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        return _norm01(b, 5.0, 99.5)

    def pressure_energy(z, boundary):
        zsrc = (z - np.mean(z)) / (np.std(z) + eps)
        bsrc = (boundary - np.mean(boundary)) / (np.std(boundary) + eps)
        base = zsrc + float(p['boundary_gain']) * bsrc
        total = np.zeros_like(z, dtype=np.float32)
        previous_wave = None

        for j, filt in enumerate(filters):
            if previous_wave is None:
                src = base
            else:
                # Harmonic cascade: nonlinear pressure at impedance discontinuities creates
                # the source term for the next harmonic generation.
                nonlinear = previous_wave * previous_wave
                nonlinear = (nonlinear - np.mean(nonlinear)) / (np.std(nonlinear) + eps)
                src = base + float(p['nonlinear_gain']) * boundary * nonlinear

            wave = np.fft.ifft2(np.fft.fft2(src) * filt).real.astype(np.float32)
            wave = (wave - np.mean(wave)) / (np.std(wave) + eps)
            amp = 1.0 / (float(harmonics[min(j, len(harmonics) - 1)]) ** float(p['harmonic_decay']) + eps)
            total += amp * wave * wave
            if previous_wave is not None:
                total += 0.15 * amp * np.abs(wave * previous_wave)
            previous_wave = wave

        return _norm01(total, 5.0, 99.3)

    frames = []
    damage = np.zeros((h, w), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)

    for i in range(numsteps):
        t = float(i) / float(numsteps - 1)

        if i == 0:
            frames.append(_to_uint8_frame(s0, original_ndim))
            continue
        if i == numsteps - 1:
            frames.append(_to_uint8_frame(e0, original_ndim))
            continue

        current = s0 + alpha[..., None] * delta
        z = impedance_field(current)
        boundary = impedance_boundary(z)
        energy = pressure_energy(z, boundary)

        threshold = float(p['cavitation_threshold'])
        cavitation = np.maximum(0.0, energy - threshold) / (1.0 - threshold + eps)
        streaming = _box_blur_periodic(cavitation, passes=int(p['streaming_blur_passes']))
        increment = (cavitation + float(p['streaming_strength']) * streaming) / (1.0 + float(p['streaming_strength']))

        # Accumulated cavitation damage: antinodes transform earlier; nodes lag behind.
        damage = float(p['damage_memory']) * damage + increment / float(max(1, numsteps - 1))
        dnorm = _norm01(damage, 4.0, 99.0)

        window = np.sin(np.pi * t) ** float(p['resonance_window_power'])
        local_alpha = t + float(p['resonance_strength']) * (dnorm - 0.5) * window
        local_alpha += float(p['damage_acceleration']) * (dnorm - 0.30) * t * window
        local_alpha = _box_blur_periodic(local_alpha, passes=int(p['alpha_blur_passes']))

        lower = max(0.0, t - float(p['max_lag']) * window - 0.02)
        upper = min(1.0, t + float(p['max_lead']) * window + 0.02)
        local_alpha = np.clip(local_alpha, lower, upper).astype(np.float32)

        if bool(p['monotonic']):
            alpha = np.maximum(alpha, local_alpha)
            alpha = np.clip(alpha, lower, upper).astype(np.float32)
        else:
            alpha = local_alpha

        # Smoothstep turns the local damage coordinate into a phase-transition fraction.
        eased = alpha * alpha * (3.0 - 2.0 * alpha)
        out = s0 + eased[..., None] * delta
        frames.append(_to_uint8_frame(out, original_ndim))

    return frames
