import torch
from torch import distributions
from absl import flags
from torch.autograd.functional import jacobian

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  [
                      "brownian",
                      "whirlpool",
                      "hill"
                   ],
                  "Prior to use.")
FLAGS = flags.FLAGS


class BasePriorSDE:
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        raise NotImplementedError


class Brownian(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x, device=x.device)

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.diag_embed(torch.ones_like(x, device=x.device))

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        scale_tril = torch.einsum("a, a...->a...", torch.sqrt(delta_ts), diffusions)
        return distributions.MultivariateNormal(loc=x, scale_tril=scale_tril)


class Whirlpool(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        if dims != 2:
            raise RuntimeError("Double well only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.u(x)

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.diag_embed(torch.ones_like(x, device=x.device))

    @staticmethod
    def u(x: Tensor) -> Tensor:
        y = 3.5 * torch.cat((-x[..., 1].unsqueeze(-1), x[..., 0].unsqueeze(-1)), dim=-1)
        return y

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        if forward:
            scale = -1.
        else:
            scale = 1.
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        scale_tril = torch.einsum("a, a...->a...", torch.sqrt(delta_ts), diffusions)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.u(x))
        return distributions.MultivariateNormal(
            loc=x + scale * drifts, scale_tril=scale_tril)


class Hill(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        if dims != 2:
            raise RuntimeError("Double well only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.grad_u(x)

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.diag_embed(torch.ones_like(x, device=x.device))

    @staticmethod
    def u(x: Tensor) -> Tensor:
        return 1000 * torch.exp(-(x**2).sum(-1) / (2 * 2.**2))

    def grad_u(self, x: Tensor) -> Tensor:
        return jacobian(lambda x_grad: self.u(x).sum(), x)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        # if forward:
        #     scale = -1.
        # else:
        #     scale = 1.
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        scale_tril = torch.einsum("a, a...->a...", torch.sqrt(delta_ts), diffusions)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.grad_u(x))
        return distributions.MultivariateNormal(
            loc=x + drifts, scale_tril=scale_tril)


class SDE:
    sde_type = 'ito'

    def __init__(self, drift, diffusion, noise_type="diagonal"):
        super().__init__()
        self.noise_type = noise_type
        self.f = lambda t, x: drift(x, t)
        self.g = lambda t, x: diffusion(x, t)


prior_sdes_dict = {
    "brownian": Brownian,
    "whirlpool": Whirlpool,
    "hill": Hill,
}
