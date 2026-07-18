import numpy as np


def _periodic_smooth2d(a, iters):
    a = a.astype(np.float32, copy=False)
    iters = int(max(0, iters))
    for _ in range(iters):
        a = 0.25 * np.roll(a, 1, axis=0) + 0.50 * a + 0.25 * np.roll(a, -1, axis=0)
        a = 0.25 * np.roll(a, 1, axis=1) + 0.50 * a + 0.25 * np.roll(a, -1, axis=1)
    return a


def _extract_signal(q, params):
    h, w, c = q.shape
    ch = params.get('signal_channel', None)
    mode = params.get('signal_mode', 'auto')

    if ch is not None:
        return q[:, :, int(ch) % c].astype(np.float32, copy=False)

    if mode == 'mean' or (mode == 'auto' and c != 3):
        return np.mean(q, axis=2, dtype=np.float32)

    if c >= 3:
        return (0.299 * q[:, :, 0] + 0.587 * q[:, :, 1] + 0.114 * q[:, :, 2]).astype(np.float32)

    return q[:, :, 0].astype(np.float32, copy=False)


def _make_velocity_field(q, params):
    signal_smooth_iters = params.get('signal_smooth_iters', 6)
    velocity_smooth_iters = params.get('velocity_smooth_iters', 3)
    speed = float(params.get('speed', 1.25))
    gradient_floor = float(params.get('gradient_floor', 0.002))
    tangent_fraction = float(params.get('tangent_fraction', 0.15))
    direction = str(params.get('direction', 'up_gradient')).lower()

    s = _extract_signal(q, params)
    s = _periodic_smooth2d(s, signal_smooth_iters)

    gx = 0.5 * (np.roll(s, -1, axis=1) - np.roll(s, 1, axis=1))
    gy = 0.5 * (np.roll(s, -1, axis=0) - np.roll(s, 1, axis=0))

    sign = -1.0 if direction in ('down', 'down_gradient', 'minus', 'minus_end', 'depolymerizing') else 1.0

    vx0 = sign * (gx - tangent_fraction * gy)
    vy0 = sign * (gy + tangent_fraction * gx)

    mag = np.sqrt(vx0 * vx0 + vy0 * vy0).astype(np.float32)
    strength = mag / (mag + gradient_floor + 1e-12)
    vx = vx0 / (mag + 1e-12) * strength
    vy = vy0 / (mag + 1e-12) * strength

    vx = _periodic_smooth2d(vx, velocity_smooth_iters)
    vy = _periodic_smooth2d(vy, velocity_smooth_iters)

    m = np.sqrt(vx * vx + vy * vy)
    too_large = m > 1.0
    if np.any(too_large):
        vx = vx.copy()
        vy = vy.copy()
        vx[too_large] /= m[too_large]
        vy[too_large] /= m[too_large]

    return (speed * vx).astype(np.float32), (speed * vy).astype(np.float32)


def _advect_conservative_periodic(q, u_face, v_face, dt):
    q_right = np.roll(q, -1, axis=1)
    f_right = np.where(u_face[:, :, None] >= 0.0, u_face[:, :, None] * q, u_face[:, :, None] * q_right)
    q = q - dt * (f_right - np.roll(f_right, 1, axis=1))

    q_down = np.roll(q, -1, axis=0)
    g_down = np.where(v_face[:, :, None] >= 0.0, v_face[:, :, None] * q, v_face[:, :, None] * q_down)
    q = q - dt * (g_down - np.roll(g_down, 1, axis=0))

    return q


def transform(maps, params=None, numsteps=10):
    if params is None:
        params = {}
    else: 
        params = dict(params)

    arr = np.asarray(maps)
    if arr.ndim != 3:
        raise ValueError('maps must have shape [h, w, ch]')
    if arr.shape[2] < 1:
        raise ValueError('maps must contain at least one channel')

    numsteps = int(numsteps)
    if numsteps <= 0:
        return []

    if arr.dtype == np.uint8:
        q = arr.astype(np.float32) / 255.0
    else:
        q = arr.astype(np.float32)
        if np.nanmax(q) > 1.5:
            q = q / 255.0
        q = np.clip(q, 0.0, 1.0)

    recompute_field = bool(params.get('recompute_field', False))
    cfl = float(params.get('cfl', 0.45))
    cfl = min(max(cfl, 0.05), 0.95)

    if not recompute_field:
        vx, vy = _make_velocity_field(q, params)
        u_face = 0.5 * (vx + np.roll(vx, -1, axis=1))
        v_face = 0.5 * (vy + np.roll(vy, -1, axis=0))
    else:
        vx = vy = u_face = v_face = None

    states = []
    for _ in range(numsteps):
        if recompute_field:
            vx, vy = _make_velocity_field(q, params)
            u_face = 0.5 * (vx + np.roll(vx, -1, axis=1))
            v_face = 0.5 * (vy + np.roll(vy, -1, axis=0))

        max_courant = float(np.max(np.abs(vx) + np.abs(vy)))
        substeps = max(1, int(np.ceil(max_courant / cfl)))
        dt = 1.0 / float(substeps)

        for _sub in range(substeps):
            q = _advect_conservative_periodic(q, u_face, v_face, dt)

        out = np.rint(np.clip(q, 0.0, 1.0) * 255.0).astype(np.uint8)
        states.append(out)

    return states