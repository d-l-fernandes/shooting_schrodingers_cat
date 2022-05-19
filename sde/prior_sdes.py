import torch
from absl import flags
from torch import distributions

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  [
                      "brownian",
                      "whirlpool",
                      "menorah",
                      "hill",
                      "periodic"
                  ],
                  "Prior to use.")
flags.DEFINE_float("hill_scale", 1., "Scale of hill potential.")
FLAGS = flags.FLAGS


class BasePriorSDE:
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims
        self.forward = True

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
        self.noise_type = "additive"
        self.sde_type = "ito"

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x, device=x.device)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        sigma = torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), diffusions * torch.sqrt(delta_ts))
        return distributions.Independent(distributions.Normal(x, sigma), 1)


class Whirlpool(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "additive"
        self.sde_type = "ito"
        if dims != 2:
            raise RuntimeError("Whirlpool only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        if self.forward:
            scale = -1.
        else:
            scale = 1.
        return scale * self.u(x)

    @staticmethod
    def u(x: Tensor) -> Tensor:
        y = 3. * torch.cat((-x[..., 1].unsqueeze(-1), x[..., 0].unsqueeze(-1)), dim=-1)
        return y

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        if forward:
            scale = -1.
        else:
            scale = 1.
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.u(x))
        sigma = torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), diffusions * torch.sqrt(delta_ts))
        return distributions.Independent(distributions.Normal(x + scale * drifts, sigma), 1)


class Menorah(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "additive"
        self.sde_type = "ito"
        if dims != 2:
            raise RuntimeError("Menorah only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        if self.forward:
            scale = -1.
        else:
            scale = 1.
        return scale * self.u(x)

    @staticmethod
    def u(x: Tensor) -> Tensor:
        sign = torch.sign(x[..., 1])
        y = 2 * torch.cat(x[..., 0].unsqueeze(-1), (torch.sqrt(x[..., 1] * sign) * sign).unsqueeze(-1), dim=-1)
        return y

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        if forward:
            scale = -1.
        else:
            scale = 1.
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.u(x))
        sigma = torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), diffusions * torch.sqrt(delta_ts))
        return distributions.Independent(distributions.Normal(x + scale * drifts, sigma), 1)


class Hill(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "additive"
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
        return FLAGS.hill_scale * z

    def grad_u(self, x: Tensor, y: Tensor) -> Tensor:
        u = -(10 * x * (x ** 2 - 1)) + self.fac * 2 * x * torch.exp(-(x ** 2 + y ** 2) / self.delta) / self.delta ** 2
        v = -(2 * y) + self.fac * 2 * y * torch.exp(-(x ** 2 + y ** 2) / self.delta) / self.delta ** 2
        return FLAGS.hill_scale * torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=-1)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.grad_u(x[..., 0], x[..., 1]))
        sigma = torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), diffusions * torch.sqrt(delta_ts))
        return distributions.Independent(distributions.Normal(x + drifts, sigma), 1)


class Periodic(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "additive"
        self.sde_type = "ito"
        self.scale = 0.5
        if dims != 2:
            raise RuntimeError("Periodic only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.grad_u(x[..., 0], x[..., 1])

    def u(self, x: Tensor, y: Tensor) -> Tensor:
        z = (y - self.scale * torch.sin(torch.pi * x)) ** 2
        return z

    def grad_u(self, x: Tensor, y: Tensor) -> Tensor:
        u = 2 * torch.pi * self.scale * torch.cos(torch.pi * x) * (y - self.scale * torch.sin(torch.pi * x))
        v = - 2 * (y - self.scale * torch.sin(torch.pi * x))
        return torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), dim=-1)

    def transition_density(self, ts: Tensor, x: Tensor, forward: bool) -> distributions.Distribution:
        delta_ts = ts[1:] - ts[:-1]
        diffusions = self.g(ts[:-1], x)
        drifts = torch.einsum("a, a...->a...", delta_ts, self.grad_u(x[..., 0], x[..., 1]))
        sigma = torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), diffusions * torch.sqrt(delta_ts))
        return distributions.Independent(distributions.Normal(x + drifts, sigma), 1)


class SDE:
    sde_type = 'ito'

    def __init__(self, drift, diffusion, noise_type="additive"):
        super().__init__()
        self.noise_type = noise_type
        self.f = lambda t, x: drift(x, t)
        self.g = lambda t, x: diffusion(x, t)


prior_sdes_dict = {
    "brownian": Brownian,
    "whirlpool": Whirlpool,
    "menorah": Menorah,
    "hill": Hill,
    "periodic": Periodic,
}
