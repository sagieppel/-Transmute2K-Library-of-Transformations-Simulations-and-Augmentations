import numpy as np


def _periodic_lap(a):
    return (np.roll(a, 1, axis=0) + np.roll(a, -1, axis=0) +
            np.roll(a, 1, axis=1) + np.roll(a, -1, axis=1) - 4.0 * a)


def _periodic_grad(a):
    gx = 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))
    gy = 0.5 * (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0))
    return gx, gy


def _robust01(a, lo_p=2.0, hi_p=98.0):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.percentile(a, [lo_p, hi_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _spectral_gaussian_blur_periodic(a, sigma):
    a = np.asarray(a, dtype=np.float32)
    if sigma is None or sigma <= 0:
        return np.full_like(a, float(np.mean(a)), dtype=np.float32)
    h, w = a.shape
    ky = (2.0 * np.pi * np.fft.fftfreq(h)).reshape(h, 1)
    kx = (2.0 * np.pi * np.fft.rfftfreq(w)).reshape(1, w // 2 + 1)
    transfer = np.exp(-0.5 * float(sigma) * float(sigma) * (kx * kx + ky * ky))
    out = np.fft.irfft2(np.fft.rfft2(a) * transfer, s=(h, w)).real
    return out.astype(np.float32)


def _warp_periodic_bilinear(img, dx, dy, chunk=16):
    img = np.asarray(img, dtype=np.float32)
    h, w, ch = img.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing='ij')

    # Backward sampling: output pixel at (x,y) samples input at (x-dx, y-dy).
    sx = np.mod(xx - dx.astype(np.float32), float(w))
    sy = np.mod(yy - dy.astype(np.float32), float(h))

    # Prevent potential IndexError from rare float rounding precision anomalies by forcing coordinates into periodic bounds
    x0 = np.floor(sx).astype(np.int64) % w
    y0 = np.floor(sy).astype(np.int64) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h

    wx = (sx - x0.astype(np.float32)).astype(np.float32)
    wy = (sy - y0.astype(np.float32)).astype(np.float32)

    wa = ((1.0 - wx) * (1.0 - wy))[..., None]
    wb = (wx * (1.0 - wy))[..., None]
    wc = ((1.0 - wx) * wy)[..., None]
    wd = (wx * wy)[..., None]

    out = np.empty_like(img, dtype=np.float32)
    chunk = max(1, int(chunk))
    for c0 in range(0, ch, chunk):
        c1 = min(ch, c0 + chunk)
        Ia = img[y0, x0, c0:c1]
        Ib = img[y0, x1, c0:c1]
        Ic = img[y1, x0, c0:c1]
        Id = img[y1, x1, c0:c1]
        out[:, :, c0:c1] = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def transform(maps, params=None, numsteps=10):
    if params is None:
        params = {}
    if not isinstance(maps, np.ndarray):
        raise TypeError('maps must be a numpy array')
    if maps.dtype != np.uint8:
        raise TypeError('maps must have dtype uint8')

    original_ndim = maps.ndim
    if maps.ndim == 2:
        maps_in = maps[:, :, None]
    elif maps.ndim == 3:
        maps_in = maps
    else:
        raise ValueError('maps must have shape [h,w,ch] or [h,w]')

    h, w, ch = maps_in.shape
    steps = int(numsteps if numsteps is not None else 10)
    if steps <= 0:
        return []

    img01 = maps_in.astype(np.float32) / 255.0

    def pget(name, default):
        return params[name] if name in params else default

    def channel_or_none(idx):
        if idx is None:
            return None
        idx = int(idx)
        if idx < 0:
            idx = ch + idx
        if idx < 0 or idx >= ch:
            raise ValueError('channel index out of range: %s' % str(idx))
        return img01[:, :, idx]

    # A single scalar material/image signal drives the plate. By default all channels
    # contribute, which is useful for RGB, hyperspectral maps, and stacked PBR maps.
    if bool(pget('use_rgb_luma', False)) and ch >= 3:
        signal = (0.2126 * img01[:, :, 0] + 0.7152 * img01[:, :, 1] + 0.0722 * img01[:, :, 2]).astype(np.float32)
    else:
        signal = np.mean(img01, axis=2, dtype=np.float32)

    signal01 = _robust01(signal)
    sgx, sgy = _periodic_grad(signal01)
    edge = np.sqrt(sgx * sgx + sgy * sgy).astype(np.float32)
    edge01 = _robust01(edge, 5.0, 99.5)

    # External transverse load on the elastic plate.
    load_channel = pget('load_channel', None)
    load_map = channel_or_none(load_channel)
    if load_map is None:
        sigma = float(pget('load_sigma', max(4.0, min(h, w) / 32.0)))
        low = _spectral_gaussian_blur_periodic(signal01, sigma)
        load = signal01 - low
    else:
        load = _robust01(load_map) - 0.5
    load = load.astype(np.float32)
    load -= np.mean(load)
    load_std = float(np.std(load)) + 1e-6
    load = load * (float(pget('load_scale', 1.0)) / load_std)
    load -= np.mean(load)

    # Spatially varying bending stiffness D. High-gradient/material-feature zones are
    # treated as more rigid, so they resist deformation and preserve fine structure.
    stiff_channel = pget('stiffness_channel', None)
    stiff_map = channel_or_none(stiff_channel)
    if stiff_map is None:
        ew = float(pget('stiffness_edge_weight', 0.70))
        lw = float(pget('stiffness_luma_weight', 0.30))
        s = ew * edge01 + lw * signal01
        s = _robust01(s)
    else:
        s = _robust01(stiff_map)

    dmin = max(1e-5, float(pget('stiffness_min', 0.18)))
    dmax = max(dmin + 1e-5, float(pget('stiffness_max', 2.75)))
    D = (dmin * ((dmax / dmin) ** s)).astype(np.float32)

    thickness_channel = pget('thickness_channel', None)
    thickness_map = channel_or_none(thickness_channel)
    if thickness_map is not None:
        t01 = _robust01(thickness_map)
        tmin = float(pget('thickness_min', 0.55))
        tmax = float(pget('thickness_max', 1.45))
        thick = tmin + (tmax - tmin) * t01
        D = (D * thick ** 3).astype(np.float32)

    # Anchor/pinning field. Strong features act like weak constraints attached to a
    # backing substrate; this creates nonlocal stress patterns around them.
    anchor_floor = float(pget('anchor_floor', 0.012))
    anchor_strength = float(pget('anchor_strength', 0.075))
    anchor_channel = pget('anchor_channel', None)
    anchor_map = channel_or_none(anchor_channel)
    if anchor_map is None:
        anchor = anchor_floor + anchor_strength * (edge01 ** 2)
    else:
        anchor = anchor_floor + anchor_strength * (_robust01(anchor_map) ** 2)
    anchor = anchor.astype(np.float32)

    tension = float(pget('tension', 0.12))
    relaxation = float(pget('relaxation', 0.65))
    relaxation = float(np.clip(relaxation, 0.01, 1.25))
    spectral_floor = float(pget('spectral_floor', 1e-4))

    # Fourier preconditioner for the constant-coefficient periodic plate operator:
    # L[w] = Delta(D Delta w) - T Delta w + A w.
    ky = (2.0 * np.pi * np.fft.fftfreq(h)).reshape(h, 1)
    kx = (2.0 * np.pi * np.fft.rfftfreq(w)).reshape(1, w // 2 + 1)
    q = 4.0 - 2.0 * np.cos(kx) - 2.0 * np.cos(ky)  # eigenvalue of -periodic_laplacian
    dmean = float(np.mean(D))
    amean = float(np.mean(anchor))
    denom = dmean * q * q + tension * q + amean + spectral_floor
    denom[0, 0] = np.inf

    wfield = np.zeros((h, w), dtype=np.float32)
    wstates = []
    max_abs_w = pget('max_abs_plate_displacement', None)
    if max_abs_w is not None:
        max_abs_w = float(max_abs_w)

    for _ in range(steps):
        lapw = _periodic_lap(wfield)
        # Variable-stiffness plate operator; the lap(D*lap(w)) term is a compact
        # finite-difference approximation to spatially varying bending resistance.
        op = _periodic_lap(D * lapw) - tension * lapw + anchor * wfield
        residual = load - op
        residual -= np.mean(residual)
        update = np.fft.irfft2(np.fft.rfft2(residual) / denom, s=(h, w)).real.astype(np.float32)
        wfield = (wfield + relaxation * update).astype(np.float32)
        wfield -= np.mean(wfield)
        if max_abs_w is not None and max_abs_w > 0:
            wfield = np.clip(wfield, -max_abs_w, max_abs_w).astype(np.float32)
        wstates.append(wfield.copy())

    compliance_power = float(pget('compliance_power', 0.55))
    compliance = (dmean / (D + 1e-6)) ** compliance_power
    compliance = np.clip(compliance, float(pget('compliance_min', 0.20)), float(pget('compliance_max', 3.0))).astype(np.float32)
    stress_flow_weight = float(pget('stress_flow_weight', 0.15))

    def raw_displacement_and_moment(wf):
        lapw = _periodic_lap(wf)
        moment = (D * lapw).astype(np.float32)
        gx, gy = _periodic_grad(wf)
        mx, my = _periodic_grad(moment)
        dxr = compliance * (gx - stress_flow_weight * mx / (dmean + 1e-6))
        dyr = compliance * (gy - stress_flow_weight * my / (dmean + 1e-6))
        return dxr.astype(np.float32), dyr.astype(np.float32), moment

    final_dx_raw, final_dy_raw, final_moment = raw_displacement_and_moment(wstates[-1])
    final_mag = np.sqrt(final_dx_raw * final_dx_raw + final_dy_raw * final_dy_raw)
    ref = float(np.percentile(final_mag, 95.0))
    if not np.isfinite(ref) or ref < 1e-6:
        ref = float(np.std(final_mag)) + 1e-6

    default_warp = 0.035 * float(min(h, w))
    warp_pixels = float(pget('warp_pixels', default_warp))
    max_disp = float(pget('max_displacement', 0.12 * float(min(h, w))))
    value_coupling = float(pget('value_coupling', 0.035))
    sample_chunk = int(pget('sample_chunk', 16))

    stress_ref = float(np.percentile(np.abs(final_moment - np.mean(final_moment)), 95.0))
    if not np.isfinite(stress_ref) or stress_ref < 1e-6:
        stress_ref = float(np.std(final_moment)) + 1e-6

    img255 = maps_in.astype(np.float32)
    outputs = []
    for wf in wstates:
        dxr, dyr, moment = raw_displacement_and_moment(wf)
        dx = (warp_pixels * dxr / ref).astype(np.float32)
        dy = (warp_pixels * dyr / ref).astype(np.float32)

        mag = np.sqrt(dx * dx + dy * dy)
        limiter = np.minimum(1.0, max_disp / (mag + 1e-6)).astype(np.float32)
        dx *= limiter
        dy *= limiter

        warped = _warp_periodic_bilinear(img255, dx, dy, chunk=sample_chunk)

        if value_coupling != 0.0:
            stress = (moment - np.mean(moment)) / stress_ref
            stress = np.clip(stress, -3.0, 3.0).astype(np.float32)
            warped = warped + (255.0 * value_coupling) * stress[:, :, None]

        out = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
        if original_ndim == 2:
            out = out[:, :, 0]
        outputs.append(out)

    return outputs