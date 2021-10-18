from typing import Callable, Tuple
import torch


def rossler_noise(noise_dims: int, batch_size: int, device) \
        -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    size = (batch_size, noise_dims)

    def generate_noise(delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        beta = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))
        gamma = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))

        beta_beta = torch.einsum("ab,ac->abc", beta, beta)

        upper_tri = 0.5 * (beta_beta - torch.sqrt(torch.abs(delta_t)) * gamma[:, :, None])
        lower_tri = 0.5 * (beta_beta + torch.sqrt(torch.abs(delta_t)) * gamma[:, None])
        diag = 0.5 * torch.diag_embed(torch.diagonal(beta_beta, dim1=-2, dim2=-1) - torch.abs(delta_t))

        chi = diag + torch.triu(upper_tri, 1) + torch.tril(lower_tri, -1)
        return beta, chi

    return generate_noise
