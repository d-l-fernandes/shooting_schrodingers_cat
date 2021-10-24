import torch
import torch.autograd.functional as functional

Tensor = torch.Tensor


def stein_discrepancy(theta: Tensor, p_grad: Tensor) -> Tensor:
    # pairwise_dists = torch.cdist(theta, theta)
    diffs = theta.unsqueeze(2) - theta.unsqueeze(1)
    pairwise_dists = torch.einsum("...a,...a->...", diffs, diffs)

    h = torch.flatten(pairwise_dists, 1, -1).median(dim=1)[0]
    # h = torch.sqrt(h / torch.log(torch.tensor(theta.shape[1] + 1)).to(theta.device))[:, None, None]
    h = torch.sqrt(0 * h + 0.01)[:, None, None]
    # h = pairwise_dists.median()
    # h = torch.sqrt(0.0001 * h / torch.log(torch.tensor(theta.shape[1] + 1)).to(theta.device))

    kxy = torch.exp(-pairwise_dists / h**2 / 2)

    h = h.unsqueeze(-1)
    dxdkxy = - 1 / h**2 * torch.einsum("abcd,abc->abcd", diffs, kxy)
    h = h.unsqueeze(-1)
    dx2d2kxy = \
        - 1 / h**2 * (-torch.eye(theta.shape[-1], device=theta.device)[None, None, None] +
                      1 / h**2 * torch.einsum("abcd,abce->abcde", diffs, diffs))
    dx2d2kxy = torch.einsum("abcde,abc->abcde", dx2d2kxy, kxy)

    trace_dx2d2lxy = torch.einsum("...ii->...", dx2d2kxy)
    first_term = torch.einsum("dab,dac,dcb->dac", p_grad, kxy, p_grad)
    second_term = torch.einsum("dab,dacb->dac", p_grad, dxdkxy)
    third_term = torch.einsum("dacb,dcb->dac", dxdkxy, p_grad)

    u = first_term + second_term + third_term + trace_dx2d2lxy

    return (1 / theta.shape[1]**2 * torch.flatten(u, 1, -1).sum(-1)).sum()
