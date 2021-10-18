import torch
from torch import distributions
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  ["brownian",
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
    def transition_density(x: Tensor, delta_t: Tensor) -> distributions.Distribution:
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
        # return torch.ones_like(x, device=x.device)

    @staticmethod
    def transition_density(x: Tensor, delta_t: Tensor) -> distributions.Distribution:
        if len(delta_t.shape) == 0:
            scale_tril = torch.diag_embed(torch.ones_like(x) * torch.sqrt(delta_t))
            return distributions.MultivariateNormal(loc=x, scale_tril=scale_tril)
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
}
