import torch
import torch.distributions as distributions
from torch.nn.functional import softplus
from absl import flags

Tensor = torch.Tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_enum("prior_dist", "gaussian",
                  [
                      "gaussian",
                      "learnable_gaussian",
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

    def forward(self, x: Tensor, scale) -> distributions.Distribution:
        # return distributions.MultivariateNormal(loc=x, scale_tril=torch.diag_embed(torch.ones_like(x) * self.sigma))
        return distributions.Independent(distributions.Normal(loc=x, scale=torch.ones_like(x) * self.sigma), 1)


class LearnableGaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = torch.nn.Parameter(torch.ones((output_size,), device=device), requires_grad=True)

    def forward(self, x: Tensor, scale) -> distributions.Distribution:
        scale = torch.einsum("...a,a->...a", torch.ones_like(x), torch.sigmoid(self.sigma))
        return distributions.Independent(distributions.Normal(loc=x, scale=scale), 1)


priors_dict = {
    "gaussian": Gaussian,
    "learnable_gaussian": LearnableGaussian,
}
