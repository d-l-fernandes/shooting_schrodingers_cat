import torch

Tensor = torch.Tensor


def stein_discrepancy(theta: Tensor, p_grad: Tensor) -> Tensor:

    diffs = (theta.unsqueeze(-2) - theta.unsqueeze(-3)) # * delta_t**0.5
    pairwise_dists = torch.sum(diffs**2, -1)

    indices = torch.triu_indices(theta.shape[-2], theta.shape[-2], 1)
    # h = pairwise_dists[..., indices[0], indices[1]].median(dim=-1)[0]
    h = pairwise_dists[..., indices[0], indices[1]].mean(dim=-1)
    # h = \
    #     sigma * torch.sqrt(
    #         h / torch.log(torch.tensor(theta.shape[-2] + 1, device=theta.device))).unsqueeze(-1).unsqueeze(-1).detach()
    h = torch.sqrt(h).unsqueeze(-1).unsqueeze(-1).detach() # / np.log(theta.shape[-2] + 1)
    # h = torch.sqrt(h).unsqueeze(-1).unsqueeze(-1) / np.log(theta.shape[-2] + 1)

    kxy = torch.exp(-pairwise_dists / h**2 / 2)

    h = h.unsqueeze(-1)
    dxdkxy = - 1 / h**2 * torch.einsum("...bcd,...bc->...bcd", diffs, kxy)
    h = h[..., 0]
    dx2d2kxy = -1 / h**4 * torch.sum(diffs**2, -1)
    dx2d2kxy *= kxy
    dx2d2kxy += 1 / h**2 * theta.shape[-1] * kxy

    trace_dx2d2lxy = dx2d2kxy

    first_term = torch.einsum("...ab,...ac,...cb->...ac", p_grad, kxy, p_grad)
    second_term = torch.einsum("...ab,...acb->...ac", p_grad, -dxdkxy)
    third_term = torch.einsum("...acb,...cb->...ac", dxdkxy, p_grad)

    u = first_term + second_term + third_term + trace_dx2d2lxy

    # return torch.flatten(u, -2, -1).sum(-1) / theta.shape[-2]**2
    u -= torch.diag_embed(torch.diagonal(u, dim1=-1, dim2=-2))

    # return 1 / (theta.shape[-2] * (theta.shape[-2] - 1)) * torch.abs(torch.flatten(u, -2, -1).sum(-1))
    return torch.sqrt(1 / (theta.shape[-2] * (theta.shape[-2] - 1)) * torch.square(torch.flatten(u, -2, -1)).sum(-1))
