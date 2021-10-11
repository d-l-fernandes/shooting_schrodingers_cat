import torch
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  ["brownian",
                   ],
                  "Prior to use.")
FLAGS = flags.FLAGS


class BasePriorSDE(torch.nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims


class Brownian(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(x, device=x.device)

    def diffusion(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.ones_like(x, device=x.device)


class SDE(torch.nn.Module):
    sde_type = 'ito'

    def __init__(self, drift, diffusion, noise_type="diagonal"):
        super().__init__()
        self.noise_type = noise_type
        self.f = lambda t, x: drift(x, t)
        self.g = lambda t, x: diffusion(x, t)


prior_sdes_dict = {
    "brownian": Brownian,
}
