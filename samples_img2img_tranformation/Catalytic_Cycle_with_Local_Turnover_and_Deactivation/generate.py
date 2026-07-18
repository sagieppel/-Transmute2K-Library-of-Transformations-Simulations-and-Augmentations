import numpy as np


def _normalize01(a, eps=1e-6, constant_if_nonzero=0.0):
    a = a.astype(np.float32, copy=False)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn < eps:
        if float(np.mean(np.abs(a))) > eps:
            return np.full_like(a, float(constant_if_nonzero), dtype=np.float32)
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def _laplacian4(a):
    up = np.empty_like(a)
    down = np.empty_like(a)
    left = np.empty_like(a)
    right = np.empty_like(a)
    up[0, :] = a[0, :]
    up[1:, :] = a[:-1, :]
    down[-1, :] = a[-1, :]
    down[:-1, :] = a[1:, :]
    left[:, 0] = a[:, 0]
    left[:, 1:] = a[:, :-1]
    right[:, -1] = a[:, -1]
    right[:, :-1] = a[:, 1:]
    return up + down + left + right - 4.0 * a


def _diffuse(a, amount):
    amount = float(np.clip(amount, 0.0, 0.24))
    return a + amount * _laplacian4(a)


def _gradient_mag(a):
    dx = np.zeros_like(a, dtype=np.float32)
    dy = np.zeros_like(a, dtype=np.float32)
    dx[:, :-1] = a[:, 1:] - a[:, :-1]
    dy[:-1, :] = a[1:, :] - a[:-1, :]
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _render_frame(start_f, end_f, alpha, original_ndim):
    frame = start_f + (end_f - start_f) * alpha[..., None]
    frame = np.rint(np.clip(frame * 255.0, 0.0, 255.0)).astype(np.uint8)
    if original_ndim == 2:
        frame = frame[:, :, 0]
    return frame


def transform(start_map, end_map, params=None, numsteps=10):
    if params is None:
        params = {}

    p = {
        'seed': 1234,
        'dt': 0.18,
        'substeps': 8,
        'conversion_rate': 2.2,
        'completion_pressure': 0.35,
        'catalyst_base': 0.08,
        'activation_rate': 0.45,
        'turnover_activation': 0.25,
        'poisoning_rate': 0.42,
        'fatigue_rate': 0.08,
        'recovery_rate': 0.20,
        'deactivation_rate': 0.75,
        'migration_rate': 0.07,
        'poison_diffusion': 0.015,
        'autocatalysis': 1.8,
        'catalyst_half_activity': 0.22,
        'catalyst_nonlinearity': 2.0,
        'initial_smoothing': 3,
        'force_final': True
    }
    p.update(params)

    start_arr = np.asarray(start_map)
    end_arr = np.asarray(end_map)
    if start_arr.shape != end_arr.shape:
        raise ValueError('start_map and end_map must have the same shape')
    if start_arr.ndim not in (2, 3):
        raise ValueError('maps must have shape [height,width] or [height,width,channels]')

    original_ndim = start_arr.ndim
    if original_ndim == 2:
        start_arr3 = start_arr[:, :, None]
        end_arr3 = end_arr[:, :, None]
    else:
        start_arr3 = start_arr
        end_arr3 = end_arr

    numsteps = int(numsteps)
    if numsteps <= 0:
        return []

    start_f = start_arr3.astype(np.float32) / 255.0
    end_f = end_arr3.astype(np.float32) / 255.0
    h, w, ch = start_f.shape

    if numsteps == 1:
        out = np.rint(np.clip(end_f * 255.0, 0.0, 255.0)).astype(np.uint8)
        if original_ndim == 2:
            out = out[:, :, 0]
        return [out]

    rng = np.random.default_rng(int(p['seed']))

    src_int = np.mean(start_f, axis=2).astype(np.float32)
    tgt_int = np.mean(end_f, axis=2).astype(np.float32)
    delta = np.mean(np.abs(end_f - start_f), axis=2).astype(np.float32)
    delta_n = _normalize01(delta, constant_if_nonzero=1.0)

    src_edge = _normalize01(_gradient_mag(src_int), constant_if_nonzero=0.0)
    tgt_edge = _normalize01(_gradient_mag(tgt_int), constant_if_nonzero=0.0)
    change_edge = _normalize01(_gradient_mag(delta_n), constant_if_nonzero=0.0)
    edge_surface = _normalize01(0.55 * src_edge + 0.25 * tgt_edge + 0.20 * change_edge, constant_if_nonzero=0.0)

    noise = rng.random((h, w)).astype(np.float32)
    for _ in range(int(p['initial_smoothing'])):
        noise = _diffuse(noise, 0.22)
    noise = _normalize01(noise, constant_if_nonzero=0.5)

    target_stability = _normalize01(0.45 * (1.0 - tgt_edge) + 0.35 * tgt_int + 0.20 * delta_n, constant_if_nonzero=0.5)
    activity_surface = np.clip(0.12 + 0.88 * (0.50 * edge_surface + 0.35 * delta_n + 0.15 * noise), 0.0, 1.0).astype(np.float32)
    poison_susc = np.clip(0.10 + 0.90 * (0.45 * src_edge + 0.25 * (1.0 - target_stability) + 0.20 * src_int + 0.10 * noise), 0.0, 1.0).astype(np.float32)
    regen_map = np.clip(0.10 + 0.90 * (0.50 * target_stability + 0.25 * (1.0 - poison_susc) + 0.25 * delta_n), 0.0, 1.0).astype(np.float32)
    feedstock = np.clip(0.03 + 0.97 * delta_n, 0.0, 1.0).astype(np.float32)

    catalyst_base = float(p['catalyst_base'])
    C = np.clip(catalyst_base + (1.0 - catalyst_base) * (0.55 * activity_surface + 0.25 * noise), 0.0, 1.0).astype(np.float32)
    Q = np.clip(0.04 * poison_susc * (1.0 - activity_surface), 0.0, 1.0).astype(np.float32)
    P = np.zeros((h, w), dtype=np.float32)

    outputs = [_render_frame(start_f, end_f, P, original_ndim)]

    substeps = max(1, int(p['substeps']))
    total_iters = max(1, (numsteps - 1) * substeps)
    iter_count = 0
    eps = 1e-6
    dt = float(p['dt'])
    n = float(p['catalyst_nonlinearity'])
    half = float(p['catalyst_half_activity'])
    half_n = half ** n

    def step_once(iter_index):
        nonlocal C, Q, P
        sim_frac = float(iter_index + 1) / float(total_iters)

        active = np.clip(C * (1.0 - Q), 0.0, 1.0)
        active_n = np.power(active, n, dtype=np.float32)
        competence = active_n / (half_n + active_n + eps)

        R = 1.0 - P
        surface_boost = 0.35 + 0.65 * activity_surface
        episodic_accel = 1.0 + float(p['autocatalysis']) * P * (1.0 - P) + 0.4 * P

        catalytic_rate = float(p['conversion_rate']) * feedstock * surface_boost * competence * episodic_accel * R
        late_pressure = float(p['completion_pressure']) * (sim_frac ** 2.0) * feedstock * (0.15 + 0.85 * competence) * R
        dP = dt * (catalytic_rate + late_pressure)
        dP = np.minimum(dP, R)
        P = np.clip(P + dP, 0.0, 1.0)

        turnover = dP / (dt + eps)

        Q = Q + dt * (float(p['poisoning_rate']) * turnover * poison_susc + float(p['fatigue_rate']) * active * turnover)
        Q = Q - dt * float(p['recovery_rate']) * Q * regen_map
        Q = Q + dt * float(p['poison_diffusion']) * _laplacian4(Q)
        Q = np.clip(Q, 0.0, 1.0)

        C = C + dt * (float(p['activation_rate']) * activity_surface * (1.0 - C) * (1.0 - Q))
        C = C + dt * (float(p['turnover_activation']) * turnover * (1.0 - C) * (1.0 - Q))
        C = C - dt * (float(p['deactivation_rate']) * Q * C)
        C = C + dt * float(p['migration_rate']) * _laplacian4(C)
        C = np.clip(C, 0.0, 1.0)

    for out_index in range(1, numsteps):
        target_iter = int(round(out_index * total_iters / float(numsteps - 1)))
        while iter_count < target_iter:
            step_once(iter_count)
            iter_count += 1

        if out_index == numsteps - 1 and bool(p['force_final']):
            final = np.rint(np.clip(end_f * 255.0, 0.0, 255.0)).astype(np.uint8)
            if original_ndim == 2:
                final = final[:, :, 0]
            outputs.append(final)
        else:
            outputs.append(_render_frame(start_f, end_f, P, original_ndim))

    return outputs
