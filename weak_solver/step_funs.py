from typing import Tuple
import torch


def rossler_step(x: torch.Tensor,
                 t: float,
                 noise: torch.Tensor,
                 delta_t: float,
                 sde) -> torch.tensor:
    drift_t = sde.f(t, x)
    diffusion_t = sde.g(t, x)
    diffusion_t_dt = sde.g(t + delta_t, x)  # Assumes additive diffusion

    beta = noise

    # Aux values
    diff_t_beta = diffusion_t * beta
    diff_t_dt_beta = diffusion_t_dt * beta

    x_2_0 = x + drift_t * delta_t + diff_t_beta

    drift_x_2_0 = sde.f(t + delta_t, x_2_0)

    x_next = x + 0.5 * delta_t * (drift_t + drift_x_2_0) + diff_t_dt_beta

    return x_next


def rossler_step_parallel(x: torch.Tensor,
                 t: torch.Tensor,
                 noise: Tuple[torch.Tensor, torch.Tensor],
                 delta_t: torch.Tensor,
                 sde) -> torch.tensor:
    drift_t = sde.f(t, x)
    diffusion_t = sde.g(t, x)
    diffusion_t_dt = sde.g(t + delta_t, x)  # Assumes additive diffusion

    beta = noise

    # Aux values
    diff_t_beta = torch.einsum("a,a...->a...", diffusion_t, beta)
    diff_t_dt_beta = torch.einsum("a,a...->a...", diffusion_t_dt, beta)

    drift_t_delta_t = torch.einsum("a,a...->a...", delta_t, drift_t)

    x_2_0 = x + drift_t_delta_t + diff_t_beta

    drift_x_2_0 = sde.f(t + delta_t, x_2_0)

    drift_x_2_0_delta_t = torch.einsum("a,a...->a...", delta_t, drift_x_2_0)

    x_next = x + 0.5 * (drift_t_delta_t + drift_x_2_0_delta_t) + diff_t_dt_beta

    return x_next


def em_step(x: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor,
            delta_t: torch.Tensor,
            sde) -> torch.tensor:
    drift_1 = sde.f(t, x)
    diffusion_1 = sde.g(t, x)

    # Aux values
    diff_1_beta = diffusion_1 * noise
    drift_1_delta_t = drift_1 * delta_t
    x_2_0 = x + drift_1_delta_t + diff_1_beta

    return x_2_0


def em_step_parallel(x: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor,
            delta_t: torch.Tensor,
            sde) -> torch.tensor:
    drift_1 = sde.f(t, x)
    diffusion_1 = sde.g(t, x)

    # Aux values
    diff_1_beta = torch.einsum("a...,a->a...", noise, diffusion_1)
    drift_1_delta_t = torch.einsum("a...,a->a...", drift_1, delta_t)
    x_2_0 = x + drift_1_delta_t + diff_1_beta

    return x_2_0


def srk_additive_step(x: torch.Tensor,
            t: float,
            noise: Tuple[torch.Tensor, torch.Tensor],
            delta_t: float,
            sde) -> torch.tensor:

    STAGES = 2
    C0 = (0, 3 / 4)
    C1 = (1, 0)
    A0 = (
        (),
        (3 / 4,),
    )
    B0 = (
        (),
        (3 / 2,),
    )

    alpha = (1 / 3, 2 / 3)
    beta1 = (1, 0)
    beta2 = (-1, 1)

    dt = delta_t
    rdt = 1 / dt
    I_k, I_k0 = noise

    y1 = x
    H0 = []

    for i in range(STAGES):
        H0i = x
        for j in range(i):
            f = sde.f(t + C0[j] * dt, H0[j])
            g_weight = B0[i][j] * I_k0 * rdt
            g_prod = sde.g(t + C1[j] * dt, x) * g_weight
            # g_prod = torch.einsum("...ab,...b->...a", sde.g(t + C1[j] * dt, x), g_weight)
            H0i = H0i + A0[i][j] * f * dt + g_prod
        H0.append(H0i)

        f = sde.f(t + C0[i] * dt, H0i)
        g_weight = beta1[i] * I_k + beta2[i] * I_k0 * rdt
        # g_prod = torch.einsum("...ab,...b->...a", sde.g(t + C1[i] * dt, x), g_weight)
        g_prod = sde.g(t + C1[i] * dt, x) * g_weight
        y1 = y1 + alpha[i] * f * dt + g_prod

    return y1


def srk_additive_step_parallel(x: torch.Tensor,
                      t: torch.Tensor,
                      noise: Tuple[torch.Tensor, torch.Tensor],
                      delta_t: torch.Tensor,
                      sde) -> torch.tensor:

    STAGES = 2
    C0 = (0, 3 / 4)
    C1 = (1, 0)
    A0 = (
        (),
        (3 / 4,),
    )
    B0 = (
        (),
        (3 / 2,),
    )

    alpha = (1 / 3, 2 / 3)
    beta1 = (1, 0)
    beta2 = (-1, 1)

    dt = delta_t
    rdt = 1 / dt
    I_k, I_k0 = noise

    y1 = x
    H0 = []

    for i in range(STAGES):
        H0i = x
        for j in range(i):
            f = sde.f(t + C0[j] * dt, H0[j])
            g_weight = torch.einsum("a...,a->a...", B0[i][j] * I_k0, rdt)
            g_prod = torch.einsum("a,a...->a...", sde.g(t + C1[j] * dt, x), g_weight)
            f_dt = torch.einsum("a,a...->a...", dt, f)
            H0i = H0i + A0[i][j] * f_dt + g_prod
        H0.append(H0i)

        f = sde.f(t + C0[i] * dt, H0i)
        g_weight = beta1[i] * I_k + torch.einsum("a...,a->a...", beta2[i] * I_k0, rdt)
        g_prod = torch.einsum("a,a...->a...", sde.g(t + C1[i] * dt, x), g_weight)
        f_dt = torch.einsum("a,a...->a...", dt, f)
        y1 = y1 + alpha[i] * f_dt + g_prod

    return y1
