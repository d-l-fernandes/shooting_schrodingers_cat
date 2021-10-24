import torch
from torch import distributions
import functorch
from absl import flags
import math

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  [
                      "brownian",
                      "whirlpool",
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

    @staticmethod
    def transition_density(x: Tensor, delta_t: Tensor, forward: bool) -> distributions.Distribution:
        raise NotImplementedError


class Brownian(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.std_scale = 5.

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x, device=x.device)

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return self.std_scale * torch.diag_embed(torch.ones_like(x, device=x.device))
        # return torch.ones_like(x, device=x.device)

    def transition_density(self, x: Tensor, delta_t: Tensor, forward: bool) -> distributions.Distribution:
        if len(delta_t.shape) == 0:
            scale_tril = torch.diag_embed(torch.ones_like(x) * torch.sqrt(delta_t) * self.std_scale)
            return distributions.MultivariateNormal(loc=x, scale_tril=scale_tril)
        else:
            # TODO implement general delta_t
            pass


class Whirlpool(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.std_scale = 5.
        if dims != 2:
            raise RuntimeError("Double well only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        return self.u(x)

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.diag_embed(torch.ones_like(x, device=x.device) * self.std_scale)
        # return torch.ones_like(x, device=x.device)

    @staticmethod
    def u(x: Tensor) -> Tensor:
        y = 3 * torch.cat((-x[..., 1].unsqueeze(-1), x[..., 0].unsqueeze(-1)), dim=-1)
        return y

    def transition_density(self, x: Tensor, delta_t: Tensor, forward: bool) -> distributions.Distribution:
        if len(delta_t.shape) == 0:
            if forward:
                scale = -1.
            else:
                scale = 1.
            scale_tril = torch.diag_embed(torch.ones_like(x) * torch.sqrt(delta_t) * self.std_scale)
            return distributions.MultivariateNormal(
                loc=(x + scale * self.u(x) * torch.sqrt(delta_t)), scale_tril=scale_tril)
        else:
            # TODO implement general delta_t
            pass


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
}
