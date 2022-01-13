from typing import Tuple
import torch


def rossler_step(x: torch.Tensor,
                 t: float,
                 noise: Tuple[torch.Tensor, torch.Tensor],
                 delta_t: float,
                 sde) -> torch.tensor:
    drift_1 = sde.f(t, x)
    diffusion_1 = sde.g(t, x)

    beta, chi = noise

    # Aux values
    diff_1_beta = torch.einsum("...bc,...c->...b", diffusion_1, beta)
    x_2_0 = x + drift_1 * delta_t + diff_1_beta

    diff_1_chi = torch.einsum("...bc,...cd->...bd", diffusion_1, chi)

    diff_1_chi_skip = torch.einsum(
        "...bc,...cd->...bd",
        diffusion_1[..., 1:],
        chi[..., 1:, 0:1]
    )

    for i in range(1, chi.shape[-1]):
        diff_1_chi_skip_step = torch.einsum(
            "...bc,...cd->...bd",
            torch.cat((diffusion_1[..., :i], diffusion_1[..., i+1:]), dim=-1),
            torch.cat((chi[..., :i, i:i+1], chi[..., i+1:, i:i+1]), dim=-2)
        )
        diff_1_chi_skip = torch.cat((diff_1_chi_skip, diff_1_chi_skip_step), -1)

    x_2_tilde_n = x.unsqueeze(-1) + drift_1.unsqueeze(-1) * delta_t + diff_1_chi
    x_3_tilde_n = x.unsqueeze(-1) + drift_1.unsqueeze(-1) * delta_t - diff_1_chi

    x_2_bar_n = x.unsqueeze(-1) + drift_1.unsqueeze(-1) * delta_t + diff_1_chi_skip / torch.sqrt(torch.abs(delta_t))
    x_3_bar_n = x.unsqueeze(-1) + drift_1.unsqueeze(-1) * delta_t - diff_1_chi_skip / torch.sqrt(torch.abs(delta_t))

    diff_2_tilde = torch.diagonal(sde.g(t + delta_t, x_2_tilde_n.transpose(-1, -2)), dim1=-3, dim2=-1)
    diff_3_tilde = torch.diagonal(sde.g(t + delta_t, x_3_tilde_n.transpose(-1, -2)), dim1=-3, dim2=-1)
    diff_2_bar = torch.diagonal(sde.g(t + delta_t, x_2_bar_n.transpose(-1, -2)), dim1=-3, dim2=-1)
    diff_3_bar = torch.diagonal(sde.g(t + delta_t, x_3_bar_n.transpose(-1, -2)), dim1=-3, dim2=-1)

    # Sum terms
    drift_term = delta_t / 2 * (drift_1 + sde.f(t + delta_t, x_2_0))
    first_diff_term = torch.einsum("...bc,...c->...b",
                                   0.5 * diffusion_1 + 0.25 * diff_2_tilde + 0.25 * diff_3_tilde,
                                   beta)
    second_diff_term = torch.einsum("...bc,...c->...b",
                                    0.5 * diff_2_tilde - 0.5 * diff_3_tilde,
                                    torch.diagonal(chi, dim1=-2, dim2=-1) / torch.sqrt(torch.abs(delta_t)))
    third_diff_term = torch.einsum("...bc,...c->...b",
                                   0.25 * diff_2_bar + 0.25 * diff_3_bar - 0.5 * diffusion_1,
                                   beta)
    forth_diff_term = torch.sum(0.5 * diff_2_bar - 0.5 * diff_3_bar, -1) * torch.sqrt(torch.abs(torch.abs(delta_t)))

    x_next = x + drift_term + first_diff_term + second_diff_term + third_diff_term + forth_diff_term

    return x_next


def em_step(x: torch.Tensor,
            t: float,
            noise: torch.Tensor,
            delta_t: float,
            sde) -> torch.tensor:
    drift_1 = sde.f(t, x)
    diffusion_1 = sde.g(t, x)

    # Aux values
    diff_1_beta = torch.einsum("...bc,...c->...b", diffusion_1, noise)
    x_2_0 = x + drift_1 * delta_t + diff_1_beta

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
            g_prod = torch.einsum("...ab,...b->...a", sde.g(t + C1[j] * dt, x), g_weight)
            H0i = H0i + A0[i][j] * f * dt + g_prod
        H0.append(H0i)

        f = sde.f(t + C0[i] * dt, H0i)
        g_weight = beta1[i] * I_k + beta2[i] * I_k0 * rdt
        g_prod = torch.einsum("...ab,...b->...a", sde.g(t + C1[i] * dt, x), g_weight)
        y1 = y1 + alpha[i] * f * dt + g_prod

    return y1
