import numpy as np
import torch
import torch.nn.functional as functional
from absl import flags

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_enum("diffusion", "scalar",
                  ["scalar",
                   ],
                  "Diffusion to use.")
flags.DEFINE_float("min_gamma", 0.1, lower_bound=0., help="Minimum diffusion.")
flags.DEFINE_float("max_gamma", np.sqrt(2.), lower_bound=0., help="Maximum diffusion.")
FLAGS = flags.FLAGS


class BaseDiffusion(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, final_t: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.final_t = final_t

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill(0.)


class Scalar(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int, final_t: float):
        super().__init__(input_size, output_size, final_t)
        self.g_min = FLAGS.min_gamma
        self.g_max = FLAGS.max_gamma
        g_diff = self.g_max - self.g_min
        self.gamma_t = \
            lambda t: (self.g_min + 2 * g_diff * t / self.final_t) * (t < self.final_t / 2) + \
                      ((2 * self.g_max - self.g_min) - 2 * g_diff * t / self.final_t) * (t >= self.final_t / 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gamma = torch.tensor(self.gamma_t(t), device=x.device)
        return gamma
        # if len(gamma.shape) == 0:
        #     return torch.diag_embed(torch.ones_like(x, device=x.device) * gamma)
        #     # return torch.ones_like(x, device=x.device) * gamma
        # elif len(gamma.shape) == 1:
        #     return torch.diag_embed(torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), gamma))
        #     # return torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), gamma)
        # else:
        #     return torch.diag_embed(torch.ones_like(x, device=x.device) * gamma)
        #     # return torch.diag_embed(torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), gamma))


diffusions_dict = {
    "scalar": Scalar,
}
