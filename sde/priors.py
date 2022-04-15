import torch
import torch.distributions as distributions
from absl import flags
from sde.drifts import ScoreNetwork
import functorch

Tensor = torch.Tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_enum("prior_dist", "gaussian",
                  [
                      "gaussian",
                      "learnable_gaussian",
                      "diffusion_gaussian",
                      "score_network"
                  ],
                  "Prior/Likelihood distribution to use.")
FLAGS = flags.FLAGS


class BasePrior(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


class Gaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, x: Tensor, scale) -> distributions.Distribution:
        # return distributions.MultivariateNormal(loc=x, scale_tril=torch.diag_embed(torch.ones_like(x) * self.sigma))
        return distributions.Independent(
            distributions.Normal(loc=x, scale=torch.ones_like(x, device=x.device) * scale), 1)


class LearnableGaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = torch.nn.Parameter(torch.ones((output_size,), device=device), requires_grad=True)

    def forward(self, x: Tensor, scale) -> distributions.Distribution:
        scale = torch.einsum("...a,a->...a", torch.ones_like(x), torch.sigmoid(self.sigma))
        return distributions.Independent(distributions.Normal(loc=x, scale=scale), 1)


class DiffusionGaussian(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = torch.nn.Parameter(torch.ones((output_size,), device=device), requires_grad=True)

    def forward(self, x: Tensor, diffusion) -> distributions.Distribution:
        scale = torch.einsum("ba...,a->ba...", torch.ones_like(x), diffusion)
        return distributions.Independent(distributions.Normal(loc=x, scale=scale), 1)


class ScoreNetworkLikelihood(BasePrior):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.sigma = ScoreNetwork(input_size, output_size)

    def forward(self, x: Tensor, t: Tensor) -> distributions.Distribution:
        scale = torch.sigmoid(functorch.vmap(self.sigma)(x, t)) + 1e-8
        return distributions.Independent(distributions.Normal(loc=x, scale=scale), 1)


priors_dict = {
    "gaussian": Gaussian,
    "learnable_gaussian": LearnableGaussian,
    "diffusion_gaussian": DiffusionGaussian,
    "score_network": ScoreNetworkLikelihood
}
