import torch
from absl import flags
from torch import distributions
from torch.autograd.functional import jacobian
from sklearn import datasets
import numpy as np

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  [
                      "brownian",
                      "whirlpool",
                      "hill",
                      "maze",
                      "spiral"
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
            raise RuntimeError("Whirlpool only applicable to 2D.")

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
            raise RuntimeError("Hill only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.grad_u(x[..., 0], x[..., 1])

    def u(self, x: Tensor, y: Tensor) -> Tensor:
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


class Maze(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.fac = 1.
        self.delta = 0.01
        if dims != 2:
            raise RuntimeError("Maze only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.grad_u(x[..., 0], x[..., 1])

    def u(self, x: Tensor, y: Tensor) -> Tensor:
        # First vertical
        y1_1 = \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.0) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.125) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.25) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.375) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.5) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.625) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.75) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 0.875) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x + 0.6) ** 2 + (y - 1.) ** 2) / self.delta) / self.delta

        # Second vertical
        y2_1 = \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.0) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.125) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.25) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.375) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.5) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.625) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.75) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 0.875) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.0) ** 2 + (y + 1.) ** 2) / self.delta) / self.delta

        # Third vertical
        y3_1 = \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.0) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.125) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.25) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.375) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.5) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.625) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.75) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 0.875) ** 2) / self.delta) / self.delta + \
            self.fac * torch.exp(- ((x - 0.6) ** 2 + (y - 1.) ** 2) / self.delta) / self.delta

        return y1_1 + y2_1 + y3_1

    def _u(self, x: Tensor):
        return self.u(x[..., 0], x[..., 1])

    def grad_u(self, x: Tensor, y: Tensor) -> Tensor:
        # First vertical
        u1_1 = \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.000) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.125) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.250) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.375) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.500) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.625) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.750) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.875) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.600) * torch.exp(- ((x + 0.6) ** 2 + (y - 1.000) ** 2) / self.delta) / self.delta**2
        v1_1 = \
            self.fac * 2 * (y - 0.000) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.000) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.125) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.125) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.250) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.250) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.375) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.375) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.500) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.500) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.625) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.625) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.750) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.750) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.875) * torch.exp(- ((x + 0.6) ** 2 + (y - 0.875) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 1.000) * torch.exp(- ((x + 0.6) ** 2 + (y - 1.000) ** 2) / self.delta) / self.delta**2

        # Second vertical
        u2_1 = \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.000) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.125) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.250) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.375) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.500) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.625) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.750) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.875) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 1.000) ** 2) / self.delta) / self.delta**2
        v2_1 = \
            self.fac * 2 * (y + 0.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.000) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.125) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.125) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.250) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.250) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.375) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.375) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.500) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.500) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.625) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.625) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.750) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.750) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 0.875) * torch.exp(- ((x + 0.0) ** 2 + (y + 0.875) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y + 1.000) * torch.exp(- ((x + 0.0) ** 2 + (y + 1.000) ** 2) / self.delta) / self.delta**2

        # Second vertical
        u3_1 = \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.000) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.125) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.250) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.375) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.500) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.625) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.750) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.875) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (x - 0.600) * torch.exp(- ((x - 0.6) ** 2 + (y - 1.000) ** 2) / self.delta) / self.delta**2
        v3_1 = \
            self.fac * 2 * (y - 0.000) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.000) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.125) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.125) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.250) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.250) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.375) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.375) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.500) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.500) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.625) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.625) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.750) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.750) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 0.875) * torch.exp(- ((x - 0.6) ** 2 + (y - 0.875) ** 2) / self.delta) / self.delta**2 + \
            self.fac * 2 * (y - 1.000) * torch.exp(- ((x - 0.6) ** 2 + (y - 1.000) ** 2) / self.delta) / self.delta**2

        return torch.cat(((u1_1 + u2_1 + u3_1).unsqueeze(-1), (v1_1 + v2_1 + v3_1).unsqueeze(-1)), dim=-1)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        scale_tril = torch.einsum("a, a...->a...", torch.sqrt(delta_ts), diffusions)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.grad_u(x[..., 0], x[..., 1]))
        return distributions.MultivariateNormal(
            loc=x + drifts, scale_tril=scale_tril)


class Spiral(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        if dims != 2:
            raise RuntimeError("Hill only applicable to 2D.")

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.fac = 1.
        self.delta = 0.35
        self.scaling_factor = 4.

        x, y = datasets.make_swiss_roll(1000, noise=0.1)
        self.locs = torch.tensor(x)[:, [0, 2]]
        self.locs = (self.locs - self.locs.mean()) / self.locs.std() * self.scaling_factor
        self.locs = self.locs.float()
        self.scale_tril = torch.diag_embed(torch.ones_like(self.locs) * 0.1)

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.grad_u(x)

    def u(self, x: Tensor, y: Tensor) -> Tensor:
        positions = torch.vstack([torch.flatten(x), torch.flatten(y)]).transpose(0, 1)

        return self._u(positions.float()).transpose(0, 1).reshape((x.shape[0], x.shape[-1]))

    def _u(self, x: Tensor) -> Tensor:
        norm = distributions.MultivariateNormal(loc=self.locs.to(x.device), scale_tril=self.scale_tril.to(x.device))
        return torch.exp(norm.log_prob(x.unsqueeze(-2))).sum(-1)

    def grad_u(self, x) -> Tensor:
        return -jacobian(lambda x_grad: self._u(x_grad).sum(), x)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
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
    "maze": Maze,
    "spiral": Spiral,
}
