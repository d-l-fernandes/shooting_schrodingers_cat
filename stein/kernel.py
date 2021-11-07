import torch
import numpy as np
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("schedule", "constant",
                  [
                      "exponential_decay_base_2",
                      "exponential_decay_min_0",
                      "inverse_exponential_decay_max_2",
                      "inverse_exponential_decay_max_1",
                      "linear",
                      "constant",
                  ],
                  "Diffusion to use.")
FLAGS = flags.FLAGS


def exponential_decay_base_2(ipfp_iteration, num_epochs):
    return 2 / 2**ipfp_iteration


def exponential_decay_min_0(ipfp_iteration, num_epochs):
    return np.exp(-ipfp_iteration)


def inverse_exponential_decay_max_alpha(alpha, ipfp_iteration, num_epochs):
    return -alpha * np.exp(-ipfp_iteration) + alpha


def linear(ipfp_iteration, num_epochs):
    return min(ipfp_iteration / 5, 1.)


def constant(ipfp_iteration, num_epochs):
    return 1.


schedule_dict = {
    "exponential_decay_base_2": exponential_decay_base_2,
    "exponential_decay_min_0": exponential_decay_min_0,
    "inverse_exponential_decay_max_2": lambda ipfp, num_epochs: inverse_exponential_decay_max_alpha(2, ipfp, num_epochs),
    "inverse_exponential_decay_max_1": lambda ipfp, num_epochs: inverse_exponential_decay_max_alpha(1, ipfp, num_epochs),
    "linear": linear,
    "constant": constant
}


def stein_discrepancy(theta: Tensor, p_grad: Tensor, sigma: float, delta_t: Tensor, ipfp_iteration: int,
                      num_epochs: int) -> Tensor:
    schedule = schedule_dict[FLAGS.schedule]
    sigma = sigma**schedule(ipfp_iteration, num_epochs)

    pairwise_dists = torch.cdist(theta.contiguous(), theta.contiguous()) * delta_t**2
    diffs = (theta.unsqueeze(-2) - theta.unsqueeze(-3)) * delta_t

    indices = torch.triu_indices(theta.shape[-2], theta.shape[-2], 1)
    h = pairwise_dists[..., indices[0], indices[1]].median(dim=-1)[0]
    # h = torch.sqrt(
    #     sigma * h / torch.log(torch.tensor(theta.shape[-2] + 1, device=theta.device))).unsqueeze(-1).unsqueeze(-1)
    h = sigma * torch.sqrt(h).unsqueeze(-1).unsqueeze(-1)

    kxy = torch.exp(-pairwise_dists / h**2 / 2) * delta_t ** 2 / sigma**2

    median_times_kxy = torch.einsum("...bc,...bc->...bc", 1 / h ** 2, kxy)

    h = h.unsqueeze(-1)
    median_times_diffs = torch.einsum("...bcd,...bcd->...bcd", -1 / h ** 2, diffs)

    dxdkxy = torch.einsum(
        "...bcd,...bc->...bcd",
        median_times_diffs,
        kxy)

    dx2d2kxy = torch.einsum("...bcd,...bce->...bcde", median_times_diffs, median_times_diffs)
    dx2d2kxy = torch.einsum("...bcde,...bc->...bcde", dx2d2kxy, kxy)
    dx2d2kxy += torch.einsum("ab,...->...ab", torch.eye(theta.shape[-1], device=theta.device), median_times_kxy)

    trace_dx2d2lxy = torch.einsum("...ii->...", dx2d2kxy)
    first_term = torch.einsum("...ab,...ac,...cb->...ac", p_grad, kxy, p_grad)
    second_term = torch.einsum("...ab,...acb->...ac", p_grad, -dxdkxy)
    third_term = torch.einsum("...acb,...cb->...ac", dxdkxy, p_grad)

    u = first_term + second_term + third_term + trace_dx2d2lxy

    # return torch.flatten(u, -2, -1).sum(-1) / theta.shape[-2]**2
    u -= torch.diag_embed(torch.diagonal(u, dim1=-1, dim2=-2))

    return 1 / (theta.shape[-2] * (theta.shape[-2] - 1)) * torch.abs(torch.flatten(u, -2, -1).sum(-1))
