import torch
from absl import flags

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_float("initial_gamma", -1, lower_bound=-1, help="Minimum diffusion.")
flags.DEFINE_float("total_gamma", -1, lower_bound=-1, help="Maximum diffusion.")
FLAGS = flags.FLAGS


class Scalar(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, final_t: float, max_diffusion: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.final_t = final_t

        if FLAGS.total_gamma > 0:
            g_max = FLAGS.total_gamma
        else:
            g_max = max_diffusion

        if FLAGS.initial_gamma > 0:
            a = FLAGS.initial_gamma
            b = 4 / self.final_t * (g_max / self.final_t - a)
        else:
            a = g_max
            b = 0.

        # self.gamma_t = \
        #     lambda t: (self.g_min + 2 * g_diff * t / self.final_t) * (t < self.final_t / 2) + \
        #               ((2 * self.g_max - self.g_min) - 2 * g_diff * t / self.final_t) * (t >= self.final_t / 2)
        self.gamma_t = \
            lambda t: (a + b * t) * (t < self.final_t / 2) + \
                      (a + b * self.final_t - b * t) * (t >= self.final_t / 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gamma = self.gamma_t(t)
        # return torch.ones_like(x, device=x.device) * gamma
        # return gamma
        if len(gamma.shape) == 0:
            return torch.ones_like(x, device=x.device) * gamma
        else:
            return torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), gamma)
