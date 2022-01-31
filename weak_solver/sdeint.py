import torch
from .step_funs import em_step, srk_additive_step, rossler_step
from .step_funs import em_step_parallel, srk_additive_step_parallel, rossler_step_parallel
from .noise import em_noise, srk_additive_noise, rossler_noise
from .noise import em_noise_parallel, srk_additive_noise_parallel, rossler_noise_parallel
# from .noise import rossler_noise, em_noise, srk_additive_noise, rossler_noise_parallel
# from .step_funs import rossler_step, em_step, srk_additive_step, rossler_step_parallel

import functorch


def integrate(sde, y0, ts, method='rossler'):
    if method == 'rossler':
        step = rossler_step
        noise = rossler_noise(y0.shape[-1], y0.shape[:-1], y0.device)
    elif method == 'em':
        step = em_step
        noise = em_noise(y0.shape[-1], y0.shape[:-1], y0.device)
    elif method == 'srk':
        step = srk_additive_step
        noise = srk_additive_noise(y0.shape[-1], y0.shape[:-1], y0.device)
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return stochastic_integrate(sde, y0, ts, step, noise)


def stochastic_integrate(sde, y0: torch.Tensor, ts, step, noise):
    ys = y0.unsqueeze(0)
    ys[0] = y0
    for i in range(ts.shape[0]-1):
        delta_t = ts[i+1] - ts[i]
        noise_step = noise(delta_t)
        next_y = step(ys[i], ts[i], noise_step, delta_t, sde)
        ys = torch.cat((ys, next_y[None]), 0)

    return ys


def integrate_parallel_time_steps(sde, y0, ts, method='rossler'):
    if method == 'rossler':
        # step = rossler_step_parallel
        step = rossler_step
        noise = rossler_noise_parallel(y0.shape[-1], y0.shape[:-1], y0.device)
    elif method == 'em':
        # step = em_step_parallel
        step = em_step
        noise = em_noise_parallel(y0.shape[-1], y0.shape[:-1], y0.device)
    elif method == 'srk':
        # step = srk_additive_step_parallel
        step = srk_additive_step
        noise = srk_additive_noise_parallel(y0.shape[-1], y0.shape[:-1], y0.device)
    else:
        raise ValueError('Unknown method: {}'.format(method))

    delta_ts = ts[1:] - ts[:-1]
    noise_values = noise(delta_ts)
    step_fun = lambda y, t, n, delta: step(y, t, n, delta, sde)
    return functorch.vmap(step_fun)(y0, ts[:-1], noise_values, delta_ts)
    # return step(y0, ts[:-1], noise_values, delta_ts, sde)
