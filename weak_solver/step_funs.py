from typing import Tuple
import torch


def rossler_step(x: torch.Tensor,
                 t: float,
                 beta: torch.Tensor,
                 chi: torch.Tensor,
                 delta_t: float,
                 sde) -> torch.tensor:
    drift_1 = sde.f(t, x)
    diffusion_1 = sde.g(t, x)

    # Aux values
    diff_1_beta = torch.einsum("abc,ac->ab", diffusion_1, beta)
    x_2_0 = x + drift_1 * delta_t + diff_1_beta

    diff_1_chi = torch.einsum("abc,acd->abd", diffusion_1, chi).permute(2, 0, 1)

    diff_1_chi_skip = torch.einsum(
        "abc,acd->dab",
        torch.cat((diffusion_1[:, :, :0], diffusion_1[:, :, 0 + 1:]), dim=-1),
        torch.cat((chi[:, :0, 0:0 + 1], chi[:, 0 + 1:, 0:0 + 1]), dim=-2)
    )

    for i in range(1, chi.shape[-1]):
        diff_1_chi_skip_step = torch.einsum(
            "abc,acd->dab",
            torch.cat((diffusion_1[:, :, :i], diffusion_1[:, :, i+1:]), dim=-1),
            torch.cat((chi[:, :i, i:i+1], chi[:, i+1:, i:i+1]), dim=-2)
        )
        diff_1_chi_skip = torch.cat((diff_1_chi_skip, diff_1_chi_skip_step), 0)

    x_2_tilde_n = x[None] + drift_1[None] * delta_t + diff_1_chi
    x_3_tilde_n = x[None] + drift_1[None] * delta_t - diff_1_chi

    x_2_bar_n = x[None] + drift_1[None] * delta_t + diff_1_chi_skip / torch.sqrt(torch.abs(delta_t))
    x_3_bar_n = x[None] + drift_1[None] * delta_t - diff_1_chi_skip / torch.sqrt(torch.abs(delta_t))

    diff_2_tilde = torch.diagonal(sde.g(t + delta_t, x_2_tilde_n), dim1=0, dim2=-1)
    diff_3_tilde = torch.diagonal(sde.g(t + delta_t, x_3_tilde_n), dim1=0, dim2=-1)
    diff_2_bar = torch.diagonal(sde.g(t + delta_t, x_2_bar_n), dim1=0, dim2=-1)
    diff_3_bar = torch.diagonal(sde.g(t + delta_t, x_3_bar_n), dim1=0, dim2=-1)

    # Sum terms
    drift_term = delta_t / 2 * (drift_1 + sde.f(t + delta_t, x_2_0))
    first_diff_term = torch.einsum("abc,ac->ab",
                                   0.5 * diffusion_1 + 0.25 * diff_2_tilde + 0.25 * diff_3_tilde,
                                   beta)
    second_diff_term = torch.einsum("abc,ac->ab",
                                    0.5 * diff_2_tilde - 0.5 * diff_3_tilde,
                                    torch.diagonal(chi, dim1=-2, dim2=-1) / torch.sqrt(torch.abs(delta_t)))
    third_diff_term = torch.einsum("abc,ac->ab",
                                   0.25 * diff_2_bar + 0.25 * diff_3_bar - 0.5 * diffusion_1,
                                   beta)
    forth_diff_term = torch.sum(0.5 * diff_2_bar - 0.5 * diff_3_bar, -1) * torch.sqrt(torch.abs(torch.abs(delta_t)))

    x_next = x + drift_term + first_diff_term + second_diff_term + third_diff_term + forth_diff_term

    return x_next
