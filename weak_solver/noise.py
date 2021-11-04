from typing import Callable, Tuple
import torch
import math


def rossler_noise(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # beta = torch.multinomial(torch.tensor([1/6, 2/3, 1/6], device=device),
        #                          math.prod(size), replacement=True).reshape(size)
        # beta = (beta - 1) * torch.sqrt(3 * torch.abs(delta_t))
        beta = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))
        # gamma = torch.multinomial(torch.tensor([1/2, 1/2], device=device),
        #                           math.prod(size), replacement=True).reshape(size)
        # gamma = (gamma * 2 - 1) * torch.sqrt(torch.abs(delta_t))
        gamma = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))

        beta_beta = torch.einsum("...b,...c->...bc", beta, beta)

        upper_tri = 0.5 * (beta_beta - torch.sqrt(torch.abs(delta_t)) * gamma.unsqueeze(-1))
        lower_tri = 0.5 * (beta_beta + torch.sqrt(torch.abs(delta_t)) * gamma.unsqueeze(-2))
        diag = 0.5 * torch.diag_embed(torch.diagonal(beta_beta, dim1=-2, dim2=-1) - torch.abs(delta_t))

        chi = diag + torch.triu(upper_tri, 1) + torch.tril(lower_tri, -1)
        return beta, chi

    return generate_noise
