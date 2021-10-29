import torch
import numpy as np

Tensor = torch.Tensor


def exponential_decay_base_2(ipfp_iteration):
    return 2 / 2**ipfp_iteration


def exponential_decay_min_1(ipfp_iteration):
    return 1. * np.exp(-ipfp_iteration) + 1


schedule = exponential_decay_base_2


def stein_discrepancy(theta: Tensor, p_grad: Tensor, sigma: float, delta_t: Tensor, ipfp_iteration: int) -> Tensor:
    pairwise_dists = torch.cdist(theta.contiguous(), theta.contiguous()) * delta_t
    diffs = (theta.unsqueeze(-2) - theta.unsqueeze(-3)) * delta_t**0.5

    h = torch.flatten(pairwise_dists, -2, -1).median(dim=-1)[0].detach()
    # h = torch.sqrt(h / torch.log(torch.tensor(theta.shape[1] + 1)).to(theta.device))[:, None, None]
    h = torch.sqrt(h / torch.log(torch.tensor(theta.shape[-2] + 1).to(theta.device))).unsqueeze(-1).unsqueeze(-1)
    # h = torch.sqrt(h).unsqueeze(-1).unsqueeze(-1)
    # h = torch.sqrt(0.0001 * h / torch.log(torch.tensor(theta.shape[1] + 1)).to(theta.device))

    kxy = torch.exp(-pairwise_dists / h**2 / 2) * delta_t ** 2 / sigma**schedule(ipfp_iteration)

    h = h.unsqueeze(-1)
    dxdkxy = - 1 / h**2 * torch.einsum("...bcd,...bc->...bcd", diffs, kxy)
    h = h.unsqueeze(-1)
    dx2d2kxy = 1 / h**4 * torch.einsum("...bcd,...bce->...bcde", diffs, diffs)
    dx2d2kxy = torch.einsum("...bcde,...bc->...bcde", dx2d2kxy, kxy)
    dx2d2kxy += 1 / h**2 * torch.einsum("ab,...->...ab", torch.eye(theta.shape[-1], device=theta.device), kxy)

    trace_dx2d2lxy = torch.einsum("...ii->...", dx2d2kxy)
    first_term = torch.einsum("...ab,...ac,...cb->...ac", p_grad, kxy, p_grad)
    second_term = torch.einsum("...ab,...acb->...ac", p_grad, dxdkxy)
    third_term = torch.einsum("...acb,...cb->...ac", dxdkxy, p_grad)

    u = first_term + second_term + third_term + trace_dx2d2lxy

    # return torch.flatten(u, -2, -1).sum(-1) / theta.shape[-2]**2
    u -= torch.diag_embed(torch.diagonal(u, dim1=-1, dim2=-2))

    return 1 / (theta.shape[1] * (theta.shape[1] - 1)) * torch.flatten(u, -2, -1).sum(-1)
