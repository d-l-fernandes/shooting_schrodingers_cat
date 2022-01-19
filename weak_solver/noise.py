from typing import Callable, Tuple
import torch
import math


def rossler_noise(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], torch.Tensor]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> torch.Tensor:
        beta = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))
        return beta

    return generate_noise


def rossler_noise_parallel(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], torch.Tensor]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> torch.Tensor:
        beta = torch.einsum("a...,a->a...", torch.randn(size, device=device), torch.sqrt(torch.abs(delta_t)))
        return beta

    return generate_noise


def em_noise(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], torch.Tensor]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> torch.Tensor:
        beta = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))
        return beta

    return generate_noise


def em_noise_parallel(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], torch.Tensor]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> torch.Tensor:
        beta = torch.einsum("a...,a->a...", torch.randn(size, device=device), torch.sqrt(torch.abs(delta_t)))
        return beta

    return generate_noise


def srk_additive_noise(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))
        gamma = torch.randn(size, device=device) * torch.sqrt(torch.abs(delta_t))

        gamma = 0.5 * torch.abs(delta_t) * (beta + 1 / 3**0.5 * gamma)
        return beta, gamma

    return generate_noise


def srk_additive_noise_parallel(noise_dims: int, batch_dims: Tuple[int], device) \
        -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:

    size = batch_dims + (noise_dims, )

    def generate_noise(delta_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = torch.einsum("a...,a->a...", torch.randn(size, device=device), torch.sqrt(torch.abs(delta_t)))
        gamma = torch.einsum("a...,a->a...", torch.randn(size, device=device), torch.sqrt(torch.abs(delta_t)))
        gamma = torch.einsum("a...,a->a...",
                             beta + 1 / 3**0.5 * gamma, 0.5 * torch.abs(delta_t))
        return beta, gamma

    return generate_noise
