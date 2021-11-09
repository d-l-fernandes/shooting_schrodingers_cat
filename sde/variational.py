import torch
import torch.distributions as distributions
from torch.nn import functional
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("variational", "gaussian",
                  [
                      "gaussian"
                  ],
                  "Variational distribution to use.")
FLAGS = flags.FLAGS


class BaseVariational(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, sigma: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma


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
        # self.std_nn = torch.nn.Linear(intermediate_size, self.output_size**2)
        # self.std_nn = torch.nn.Linear(intermediate_size, self.output_size)

    def forward(self, x: Tensor) -> distributions.Distribution:
        mean = self.mean_nn(x)
        out = torch.diag_embed(torch.ones_like(x, device=x.device) * self.sigma)
        return distributions.MultivariateNormal(loc=mean, scale_tril=out)
        # out = self.std_nn(self.nn(x))
        # out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        # diag = torch.diagonal(out, dim1=-1, dim2=-2)
        # out = out - torch.diag_embed(diag) + (torch.diag_embed(functional.softplus(diag) + 1e-4))
        # return distributions.MultivariateNormal(loc=mean, scale_tril=out.tril())
        # out = functional.softplus(self.std_nn(self.nn(x))) + 1e-8
        # out = functional.softplus(self.std_nn(x)) + 1e-6
        # return distributions.MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(out))


variational_dict = {
    "gaussian": Gaussian,
}
