import torch
from torch import distributions
import torch.autograd.functional as functional
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("prior_sde", "brownian",
                  [
                      "brownian",
                      "double_well"
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


class DoubleWell(BasePriorSDE):
    def __init__(self, dims: int):
        super().__init__(dims)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        if dims != 2:
            raise RuntimeError("Double well only applicable to 2D.")

    def f(self, t: Tensor, x: Tensor) -> Tensor:
        grad_u = functional.jacobian(
            lambda x_grad: self.u(x_grad).sum(), x, create_graph=True, vectorize=True)
        return -grad_u

    def g(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.diag_embed(torch.ones_like(x, device=x.device))
        # return torch.ones_like(x, device=x.device)

    @staticmethod
    def u(x: Tensor) -> Tensor:
        x_term = x[:, 0] ** 2
        y_term = (x[:, 1] + 5) ** 2
        exp_term = torch.exp(-(x[:, 0]**2 + x[:, 1]**2))
        return x_term + y_term + exp_term

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
    "double_well": DoubleWell,
}
