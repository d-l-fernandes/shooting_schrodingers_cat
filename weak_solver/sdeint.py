import torch
from .step_funs import rossler_step
from .noise import rossler_noise


def integrate(sde, y0, ts, method='rossler'):
    if method == 'rossler':
        step = rossler_step
        noise = rossler_noise(y0.shape[-1], y0.shape[0], y0.device)
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return stochastic_integrate(sde, y0, ts, step, noise)


def stochastic_integrate(sde, y0: torch.Tensor, ts, step, noise):
    ys = y0.unsqueeze(0)
    ys[0] = y0
    for i in range(ts.shape[0]-1):
        delta_t = ts[i+1] - ts[i]
        step_noise = noise(delta_t)
        next_y = step(ys[i], ts[i], step_noise, delta_t, sde)
        ys = torch.cat((ys, next_y[None]), 0)

    return ys
