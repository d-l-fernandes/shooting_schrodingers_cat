from typing import Union, Tuple
import torch
import numpy as np

from absl import flags

flags.DEFINE_bool("ksd_unbiased", True,
                  "Whether to use unbiased ksd estimator.")
flags.DEFINE_bool("ksd_detached", False,
                  "Whether to use detach h in ksd estimator.")

flags.DEFINE_enum("ksd_scale", "mean", ["median", "mean"],
                  "Which type of scale to use for ksd estimator.")

Tensor = torch.Tensor

FLAGS = flags.FLAGS


def stein_discrepancy(theta: Tensor, p_grad: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:

    diffs = theta.unsqueeze(-2) - theta.unsqueeze(-3)
    pairwise_dists = torch.sum(diffs**2, -1)

    indices = torch.triu_indices(theta.shape[-2], theta.shape[-2], 1)

    if FLAGS.ksd_scale == "median":
        h = pairwise_dists[..., indices[0], indices[1]].median(dim=-1)[0]
    elif FLAGS.ksd_scale == "mean":
        h = pairwise_dists[..., indices[0], indices[1]].mean(dim=-1)
    else:
        raise RuntimeError("Invalid value for ksd_scale.")

    if FLAGS.ksd_detached:
        h = torch.sqrt(h).unsqueeze(-1).unsqueeze(-1).detach() \
            / np.log(theta.shape[-2] + 1)
    else:
        h = torch.sqrt(h).unsqueeze(-1).unsqueeze(-1) \
            / np.log(theta.shape[-2] + 1)

    kxy = torch.exp(-pairwise_dists / h**2 / 2)

    h = h.unsqueeze(-1)
    dxdkxy = - 1 / h**2 * torch.einsum("...bcd,...bc->...bcd", diffs, kxy)
    trace_dx2d2kxy = torch.einsum("...a,...->...a", -1 / h**4 * diffs**2 + 1 / h**2, kxy).sum(-1)

    if type(p_grad) is not tuple:
        first_term = torch.einsum("...ab,...ac,...cb->...ac", p_grad, kxy, p_grad)
        second_term = torch.einsum("...ab,...acb->...ac", p_grad, -dxdkxy)
        third_term = torch.einsum("...acb,...cb->...ac", dxdkxy, p_grad)

        u = first_term + second_term + third_term + trace_dx2d2kxy

        if FLAGS.ksd_unbiased:
            u -= torch.diag_embed(torch.diagonal(u, dim1=-1, dim2=-2))

            return 1 / (theta.shape[-2] * (theta.shape[-2] - 1)) * torch.abs(torch.flatten(u, -2, -1).sum(-1))
        else:
            return torch.flatten(u, -2, -1).sum(-1) / theta.shape[-2]**2
    else:
        first_terms = (torch.einsum("...ab,...ac,...cb->...ac", i, kxy, i) for i in p_grad)
        second_terms = (torch.einsum("...ab,...acb->...ac", i, -dxdkxy) for i in p_grad)
        third_terms = (torch.einsum("...acb,...cb->...ac", dxdkxy, i) for i in p_grad)

        us = (sum(i) for i in zip(first_terms, second_terms, third_terms))

        if FLAGS.ksd_unbiased:
            us = (i - torch.diag_embed(torch.diagonal(i, dim1=-1, dim2=-2)) for i in us)
            scale = 1 / (theta.shape[-2] * (theta.shape[-2] - 1))
            return tuple(torch.abs(torch.flatten(i, -2, -1).sum(-1)) * scale for i in us)
        else:
            scale = 1 / theta.shape[-2]**2
            return tuple(torch.flatten(i, -2, -1).sum(-1) * scale for i in us)
