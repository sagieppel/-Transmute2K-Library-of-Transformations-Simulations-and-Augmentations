import numpy as np


def _laplacian(a):
    return (np.roll(a, 1, axis=0) +
            np.roll(a, -1, axis=0) +
            np.roll(a, 1, axis=1) +
            np.roll(a, -1, axis=1) -
            4.0 * a)


def _grad_x(a):
    return 0.5 * (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1))


def _velocity_from_vorticity(omega, max_velocity):
    h, w = omega.shape

    kx = (2.0 * np.pi * np.fft.rfftfreq(w)).astype(np.float32)
    ky = (2.0 * np.pi * np.fft.fftfreq(h)).astype(np.float32)
    k2 = ky[:, None] * ky[:, None] + kx[None, :] * kx[None, :]

    omega_hat = np.fft.rfft2(omega)
    k2[0, 0] = 1.0

    # 2-D incompressible Boussinesq toy model:
    # omega = -laplacian(psi), u = dpsi/dy, v = -dpsi/dx.
    psi_hat = omega_hat / k2
    psi_hat[0, 0] = 0.0

    u = np.fft.irfft2(1j * ky[:, None] * psi_hat, s=(h, w)).astype(np.float32)
    v = np.fft.irfft2(-1j * kx[None, :] * psi_hat, s=(h, w)).astype(np.float32)

    u -= np.mean(u)
    v -= np.mean(v)

    vmax = float(np.max(np.sqrt(u * u + v * v)))
    if vmax > max_velocity and vmax > 1.0e-8:
        scale = max_velocity / vmax
        u *= scale
        v *= scale

    return u, v


def _advect_periodic(a, u, v, dt, yy, xx, chunk_channels=16):
    h, w = u.shape

    # Backtrace through the velocity field. Use float64 here to reduce the chance
    # of exact boundary values, then wrap integer indices explicitly for safety.
    y = np.mod(yy - float(dt) * v.astype(np.float64), float(h))
    x = np.mod(xx - float(dt) * u.astype(np.float64), float(w))

    yf = np.floor(y)
    xf = np.floor(x)

    # Important fix: even after modulo, floating point roundoff can very rarely
    # produce x == w or y == h. The base indices must be wrapped too; otherwise
    # advanced indexing can try to read index w/h, causing an out-of-bounds crash.
    y0 = np.mod(yf.astype(np.intp), h)
    x0 = np.mod(xf.astype(np.intp), w)
    y1 = (y0 + 1) % h
    x1 = (x0 + 1) % w

    wy = (y - yf).astype(np.float32)
    wx = (x - xf).astype(np.float32)

    w00 = (1.0 - wy) * (1.0 - wx)
    w01 = (1.0 - wy) * wx
    w10 = wy * (1.0 - wx)
    w11 = wy * wx

    if a.ndim == 2:
        return (w00 * a[y0, x0] +
                w01 * a[y0, x1] +
                w10 * a[y1, x0] +
                w11 * a[y1, x1]).astype(np.float32)

    out = np.empty_like(a, dtype=np.float32)
    ch = a.shape[2]
    chunk_channels = int(max(1, chunk_channels))

    for s in range(0, ch, chunk_channels):
        e = min(ch, s + chunk_channels)
        out[:, :, s:e] = (w00[:, :, None] * a[y0, x0, s:e] +
                          w01[:, :, None] * a[y0, x1, s:e] +
                          w10[:, :, None] * a[y1, x0, s:e] +
                          w11[:, :, None] * a[y1, x1, s:e])

    return out


def transform(maps, params=None, numsteps=10):
    p = {} if params is None else dict(params)

    arr = np.asarray(maps)
    squeeze_output = False

    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze_output = True

    if arr.ndim != 3:
        raise ValueError('maps must have shape [h,w,ch]')
    if arr.shape[2] < 1:
        raise ValueError('maps must contain at least one channel')

    h, w, ch = arr.shape

    if numsteps is None:
        numsteps = 10
    numsteps = int(numsteps)
    if numsteps <= 0:
        return []

    dt = float(p.get('dt', 0.85))
    buoyancy = float(p.get('buoyancy', 70.0))
    viscosity = float(p.get('viscosity', 0.08))
    thermal_diffusion = float(p.get('thermal_diffusion', 0.055))
    dye_diffusion = float(p.get('dye_diffusion', 0.004))
    heat_rate = float(p.get('heat_rate', 0.075))
    drag = float(p.get('drag', 0.018))
    max_velocity = float(p.get('max_velocity', 3.5))
    image_temperature_weight = float(p.get('image_temperature_weight', 0.70))
    vertical_gradient_weight = float(p.get('vertical_gradient_weight', 0.30))
    perturbation = float(p.get('perturbation', 0.045))
    noise_amount = float(p.get('noise_amount', 0.010))
    initial_vorticity = float(p.get('initial_vorticity', 0.018))
    channel_mix = float(p.get('channel_mix', 0.006))
    preserve_mean = bool(p.get('preserve_mean', False))
    chunk_channels = int(p.get('chunk_channels', 16))
    seed = p.get('seed', 0)

    C = arr.astype(np.float32) / 255.0
    initial_channel_mean = C.mean(axis=(0, 1), keepdims=True)

    # Float64 coordinate grids make the semi-Lagrangian backtrace more robust.
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64),
                         indexing='ij')
    yn = (yy / max(1.0, float(h - 1))).astype(np.float32)

    # Temperature is the active scalar: image brightness plus bottom-hot/top-cool gradient.
    image_temp = C.mean(axis=2)
    T = image_temperature_weight * image_temp + vertical_gradient_weight * yn

    # Weak three-wave seed encourages Rayleigh-Benard-like cells/rolls.
    cell_size = float(p.get('cell_size', max(12.0, min(h, w) / 8.0)))
    k = 2.0 * np.pi / max(2.0, cell_size)
    hex_pattern = (np.cos(k * xx) +
                   np.cos(k * (0.5 * xx + 0.8660254038 * yy)) +
                   np.cos(k * (0.5 * xx - 0.8660254038 * yy))) / 3.0
    T = T + perturbation * hex_pattern.astype(np.float32)

    rng = np.random.default_rng(seed)
    if noise_amount > 0.0:
        T = T + noise_amount * rng.standard_normal((h, w)).astype(np.float32)
    T = np.clip(T, 0.0, 1.0).astype(np.float32)

    omega = (-0.05 * buoyancy * _grad_x(T)).astype(np.float32)
    if initial_vorticity > 0.0:
        omega += initial_vorticity * rng.standard_normal((h, w)).astype(np.float32)
    omega -= np.mean(omega)

    band = float(p.get('boundary_band', max(2.0, 0.08 * h)))
    top_mask = np.exp(-((yy / band) ** 2)).astype(np.float32)
    bottom_mask = np.exp(-(((h - 1.0 - yy) / band) ** 2)).astype(np.float32)

    frames = []

    for _ in range(numsteps):
        u, v = _velocity_from_vorticity(omega, max_velocity)

        # Semi-Lagrangian transport of heat, vorticity, and all image/material channels.
        T = _advect_periodic(T, u, v, dt, yy, xx, chunk_channels)
        C = _advect_periodic(C, u, v, dt, yy, xx, chunk_channels)
        omega = _advect_periodic(omega, u, v, dt, yy, xx, chunk_channels)

        # Thermal diffusion plus virtual boundary forcing: bottom heats, top cools.
        T += dt * thermal_diffusion * _laplacian(T)
        T += dt * heat_rate * (bottom_mask * (1.0 - T) - top_mask * T)
        T = np.clip(T, 0.0, 1.0).astype(np.float32)

        # Vorticity equation approximation:
        # d omega / dt = viscosity laplacian(omega) - buoyancy dT/dx - drag omega.
        omega += dt * (viscosity * _laplacian(omega) - buoyancy * _grad_x(T) - drag * omega)
        omega -= np.mean(omega)
        omega = omega.astype(np.float32)

        # Dye/material diffusion and weak cross-channel mixing.
        if dye_diffusion > 0.0:
            C += dt * dye_diffusion * _laplacian(C)

        if channel_mix > 0.0 and ch > 1:
            local_mean = C.mean(axis=2, keepdims=True)
            C += dt * channel_mix * (local_mean - C)

        if preserve_mean:
            C += initial_channel_mean - C.mean(axis=(0, 1), keepdims=True)

        C = np.clip(C, 0.0, 1.0).astype(np.float32)

        out = np.rint(C * 255.0).clip(0, 255).astype(np.uint8)
        if squeeze_output:
            out = out[:, :, 0]
        frames.append(out)

    return frames