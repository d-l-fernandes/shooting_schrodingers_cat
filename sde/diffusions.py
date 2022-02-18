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
            g_max = torch.ones(self.output_size) * FLAGS.total_gamma
        else:
            g_max = max_diffusion

        self.total_diffusion = g_max

        if FLAGS.initial_gamma > 0:
            self.a = torch.ones(self.output_size) * FLAGS.initial_gamma
            self.b = 4 / self.final_t * (g_max / self.final_t - self. a)
        else:
            self.a = g_max / self.final_t
            self.b = torch.zeros(self.output_size)

        # self.gamma_t = \
        #     lambda t: (self.g_min + 2 * g_diff * t / self.final_t) * (t < self.final_t / 2) + \
        #               ((2 * self.g_max - self.g_min) - 2 * g_diff * t / self.final_t) * (t >= self.final_t / 2)
        # self.gamma_t = \
        #     lambda t: (a + b * t) * (t < self.final_t / 2) + \
        #               (a + b * self.final_t - b * t) * (t >= self.final_t / 2)

    def gamma_t(self, t):
        b = self.b.to(t.device)
        a = self.a.to(t.device)
        if len(t.shape) != 0:
            t = t.unsqueeze(-1)
        return (a + b * t) * (t < self.final_t / 2) + (a + b * self.final_t - b * t) * (t >= self.final_t / 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gamma = self.gamma_t(t)
        # return torch.ones_like(x, device=x.device) * gamma
        # return gamma
        if len(gamma.shape) == 1:
            return torch.ones_like(x, device=x.device) * gamma.to(x.device)
        else:
            return torch.einsum("a...b,ab->a...b", torch.ones_like(x, device=x.device), gamma.to(x.device))
