import torch
import torch.distributions as distributions
from torch.nn.functional import softplus
from absl import flags

Tensor = torch.Tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_enum("prior_dist", "diffusion_learnable_gaussian",
                  [
                      "gaussian",
                      "learnable_gaussian",
                      "diffusion_gaussian",
                      "diffusion_learnable_gaussian",
                  ],
                  "Prior/Likelihood distribution to use.")
flags.DEFINE_float("sigma", 0.01, "STD to use in Gaussian (fixed or initial value).")
FLAGS = flags.FLAGS


class BasePrior(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


class Gaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = FLAGS.sigma

    def forward(self, x: Tensor) -> distributions.Distribution:
        return distributions.MultivariateNormal(loc=x, scale_tril=torch.diag_embed(torch.ones_like(x) * self.sigma))


class LearnableGaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = torch.nn.Parameter(FLAGS.sigma * torch.ones((output_size,), device=device), requires_grad=True)

    def forward(self, x: Tensor, time_values: Tensor, diffusions: Tensor) -> distributions.Distribution:
        scale_tril = torch.diag_embed(torch.einsum("...a,a->...a", torch.ones_like(x), torch.sigmoid(self.sigma)))
        return distributions.MultivariateNormal(loc=x, scale_tril=scale_tril)


class DiffusionGaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = torch.nn.Parameter(FLAGS.sigma * torch.ones((output_size,), device=device), requires_grad=True)

    def forward(self, x: Tensor, delta_t: Tensor, diffusions: Tensor) -> distributions.Distribution:
        # scale_tril = torch.sigmoid(self.sigma) * torch.sqrt(delta_t) * diffusions
        scale_tril = torch.sqrt(delta_t) * diffusions
        return distributions.MultivariateNormal(loc=x, scale_tril=torch.diag_embed(scale_tril))


class DiffusionLearnableGaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 20 * input_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size), torch.nn.Sigmoid(),
        )

    def forward(self, x: Tensor, delta_t: Tensor, diffusions: Tensor) -> distributions.Distribution:
        scale_tril = self.nn(x) * torch.sqrt(delta_t) * diffusions
        return distributions.MultivariateNormal(loc=x, scale_tril=torch.diag_embed(scale_tril))


priors_dict = {
    "gaussian": Gaussian,
    "learnable_gaussian": LearnableGaussian,
    "diffusion_gaussian": DiffusionGaussian,
    "diffusion_learnable_gaussian": DiffusionLearnableGaussian
}
