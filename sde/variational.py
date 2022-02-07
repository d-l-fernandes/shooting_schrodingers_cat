import torch
import torch.distributions as distributions
from torch.nn import functional
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("variational", "gaussian",
                  [
                      "gaussian",
                      "gaussian_mnist"
                  ],
                  "Variational distribution to use.")
FLAGS = flags.FLAGS


class BaseVariational(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, sigma: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.)


class Gaussian(BaseVariational):
    def __init__(self, input_size: int, output_size: int, sigma: float):
        super().__init__(input_size, output_size, sigma)
        intermediate_size = 20 * input_size
        self.mean_nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size),
        )
        self.std_nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size),
        )

    def forward(self, x: Tensor) -> distributions.Distribution:
        mean = self.mean_nn(x)
        # out = torch.ones_like(x, device=x.device) * self.sigma
        # out = functional.softplus(self.std_nn(x)) + 1e-5
        out = torch.sigmoid(self.std_nn(x)) + 1e-8
        return distributions.Independent(distributions.Normal(loc=mean, scale=out), 1)


class GaussianMNIST(BaseVariational):
    def __init__(self, input_size: int, output_size: int, sigma: float):
        super().__init__(input_size, output_size, sigma)
        intermediate_size = 200
        self.mean_nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, self.output_size),
        )
        self.std_nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, self.output_size),
        )

    def forward(self, x: Tensor) -> distributions.Distribution:
        mean = self.mean_nn(x)
        # out = torch.ones_like(x, device=x.device) * self.sigma
        # out = functional.softplus(self.std_nn(x)) + 1e-8
        out = torch.sigmoid(self.std_nn(x)) + 1e-8
        return distributions.Independent(distributions.Normal(loc=mean, scale=out), 1)


variational_dict = {
    "gaussian": Gaussian,
    "gaussian_mnist": GaussianMNIST,
}
