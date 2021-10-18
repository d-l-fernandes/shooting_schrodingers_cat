from typing import Tuple
import torch


def rossler_step(x: torch.Tensor,
                 t: float,
                 noise: Tuple[torch.Tensor, torch.Tensor],
                 delta_t: float,
                 sde) -> torch.tensor:
    beta, chi = noise

    drift_1 = sde.f(t, x)
    diffusion_1 = sde.g(t, x)

    # Aux values
    diff_1_beta = torch.einsum("abc,ac->ab", diffusion_1, beta)
    x_2_0 = x + drift_1 * delta_t + diff_1_beta

    diff_1_chi = torch.einsum("abc,acd->abd", diffusion_1, chi).permute(2, 0, 1)

    diff_1_chi_skip = torch.empty((chi.shape[0], chi.shape[-1], chi.shape[-1], chi.shape[-1]), device=x.device)

    mask = torch.empty_like(chi, dtype=torch.bool, device=x.device).fill_(False)
    for i in range(chi.shape[-1]):
        mask[:, i] = True
        chi_skip = chi.masked_fill(mask, 0.)
        diff_1_chi_skip[:, :, i] = torch.einsum("abc,acd->abd", diffusion_1, chi_skip)
        mask[:, i] = False
    diff_1_chi_skip = torch.diagonal(diff_1_chi_skip, dim1=-2, dim2=-1).permute(2, 0, 1)

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
