from typing import Tuple
import torch
import numpy as np

from absl import flags

flags.DEFINE_bool("ksd_unbiased", True,
                  "Whether to use unbiased ksd estimator.")

flags.DEFINE_enum("ksd_scale", "mean", ["median", "mean"],
                  "Which type of scale to use for ksd estimator.")

Tensor = torch.Tensor

FLAGS = flags.FLAGS


def mean(tensor: Tensor) -> Tensor:
    return torch.mean(tensor, dim=-1)


def median(tensor: Tensor) -> Tensor:
    return torch.median(tensor, dim=-1)[0]


def unbiased(us: Tuple[Tensor, ...], num_samples: int) -> Tuple[Tensor, ...]:
    us = (i - torch.diag_embed(torch.diagonal(i, dim1=-1, dim2=-2)) for i in us)
    scale = 1 / (num_samples * (num_samples - 1))
    return tuple(torch.abs(torch.flatten(i, -2, -1).sum(-1)) * scale for i in us)


def biased(us: Tuple[Tensor, ...], num_samples: int) -> Tuple[Tensor, ...]:
    scale = 1 / num_samples**2
    return tuple(torch.flatten(i, -2, -1).sum(-1) * scale for i in us)


class KSDCalculator:
    def __init__(self):
        if FLAGS.ksd_scale == "mean":
            self.scale_function = mean
        elif FLAGS.ksd_scale == "median":
            self.scale_function = median
        else:
            raise RuntimeError("Invalid value for ksd_scale")

        if FLAGS.ksd_unbiased:
            self.biased_function = unbiased
        else:
            self.biased_function = biased

    def stein_discrepancy(self, theta: Tensor, p_grad: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        diffs = theta.unsqueeze(-2) - theta.unsqueeze(-3)
        pairwise_dists = torch.sum(diffs**2, -1)

        indices = torch.triu_indices(theta.shape[-2], theta.shape[-2], 1)

        h = self.scale_function(pairwise_dists[..., indices[0], indices[1]])

        h = torch.sqrt(h).unsqueeze(-1).unsqueeze(-1)  # / np.log(theta.shape[-2] + 1)

        kxy = torch.exp(-pairwise_dists / h**2 / 2)

        h = h.unsqueeze(-1)
        dxdkxy = - 1 / h**2 * torch.einsum("...bcd,...bc->...bcd", diffs, kxy)
        trace_dx2d2kxy = torch.einsum("...a,...->...a", -1 / h**4 * diffs**2 + 1 / h**2, kxy).sum(-1)

        first_terms = (torch.einsum("...ab,...ac,...cb->...ac", i, kxy, i) for i in p_grad)
        second_terms = (torch.einsum("...ab,...acb->...ac", i, -dxdkxy) for i in p_grad)
        third_terms = (torch.einsum("...acb,...cb->...ac", dxdkxy, i) for i in p_grad)

        us = (sum(i) + trace_dx2d2kxy for i in zip(first_terms, second_terms, third_terms))

        return self.biased_function(us, theta.shape[-2])
