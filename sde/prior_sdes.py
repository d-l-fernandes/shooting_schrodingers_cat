import torch
from torch import distributions
from absl import flags

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

    @staticmethod
    def g(t: Tensor, x: Tensor) -> Tensor:
        return torch.diag_embed(torch.ones_like(x, device=x.device))

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        raise NotImplementedError


class Brownian(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x, device=x.device)

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
        self.fac = 1.
        self.delta = 0.35
        if dims != 2:
            raise RuntimeError("Double well only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.grad_u(x[..., 0], x[..., 1])

    def u(self, x: Tensor, y: Tensor) -> Tensor:
        # return 10 * torch.exp(-(x**2).sum(-1) / (2 * 1.**2))
        z = (5 / 2.0) * (x ** 2 - 1 ** 2) ** 2 + y ** 2 + self.fac \
            * torch.exp(-(x ** 2 + y ** 2) / self.delta) / self.delta
        return z

    def grad_u(self, x: Tensor, y: Tensor) -> Tensor:
        u = -(10 * x * (x ** 2 - 1)) + self.fac * 2 * x * torch.exp(-(x ** 2 + y ** 2) / self.delta) / self.delta ** 2
        v = -(2 * y) + self.fac * 2 * y * torch.exp(-(x ** 2 + y ** 2) / self.delta) / self.delta ** 2
        return torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=-1)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        scale_tril = torch.einsum("a, a...->a...", torch.sqrt(delta_ts), diffusions)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.grad_u(x[..., 0], x[..., 1]))
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
