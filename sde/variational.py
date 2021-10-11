import torch
import torch.distributions as distributions
from torch.nn import functional
from torch.autograd.functional import jacobian
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("variational", "gaussian",
                  [
                      "identity_gaussian",
                      "gaussian"
                  ],
                  "Variational distribution to use.")
FLAGS = flags.FLAGS


class BaseVariational(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


class IdentityGaussian(BaseVariational):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, x: Tensor, time_values: Tensor) -> distributions.Distribution:
        t_values = time_values[:, None, None].repeat(1, x.shape[1], 1)
        return distributions.MultivariateNormal(
            loc=torch.zeros_like(x),
            scale_tril=torch.diag_embed(torch.ones_like(x)))


class Gaussian(BaseVariational):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 20 * input_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.Tanh(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.Tanh(),
        )
        self.mean_nn = torch.nn.Linear(intermediate_size, self.output_size)
        self.std_nn = torch.nn.Linear(intermediate_size, self.output_size**2)

    def forward(self, x: Tensor) -> distributions.Distribution:
        mean = self.mean_nn(self.nn(x))
        out = self.std_nn(self.nn(x))
        out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        diag = torch.diagonal(out, dim1=-1, dim2=-2)
        out = out - torch.diag_embed(diag) + torch.diag_embed(functional.softplus(diag))
        return distributions.MultivariateNormal(loc=mean, scale_tril=out.tril())


variational_dict = {
    "identity_gaussian": IdentityGaussian,
    "gaussian": Gaussian,
}
