import torch
from absl import flags

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_float("initial_gamma", 0.1, lower_bound=-1, help="Initial diffusion.")
flags.DEFINE_float("max_gamma", 1., lower_bound=-1, help="Maximum diffusion.")
flags.DEFINE_float("total_gamma", -1, lower_bound=-1, help="Total diffusion.")
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

        self.total_diffusion = g_max

        if FLAGS.initial_gamma > 0:
            self.a = FLAGS.initial_gamma
            if FLAGS.max_gamma > 0:
                self.b = 2 * (FLAGS.max_gamma - self.a) / self.final_t
            else:
                self.b = 4 / self.final_t * (g_max / self.final_t - self.a)
        else:
            self.a = g_max / self.final_t
            self.b = 0.

    def gamma_t(self, t):
        return (self.a + self.b * t) * (t < self.final_t / 2) \
               + (self.a + self.b * self.final_t - self.b * t) * (t >= self.final_t / 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gamma = self.gamma_t(t)
        return gamma
