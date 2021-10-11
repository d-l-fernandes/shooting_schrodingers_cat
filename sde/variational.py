import torch
import torch.distributions as distributions
from torch.nn import functional
from torch.autograd.functional import jacobian
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("variational", "gaussian",
                  ["gaussian"],
                  "Variational distribution to use.")
FLAGS = flags.FLAGS


class BaseVariational(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


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

    def log_prob(self, x: Tensor, scale) -> Tensor:
        # mean = self.mean_nn(self.nn(x))
        mean = x
        transform = self.mean_nn(self.nn(x))
        grad_log_prob_x = jacobian(
            lambda x_jac: self.mean_nn(self.nn(x)).sum(), x, create_graph=True, vectorize=True)
        # out = self.std_nn(self.nn(x))
        # out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        # diag = torch.diagonal(out, dim1=-1, dim2=-2)
        # out = out - torch.diag_embed(diag) + torch.diag_embed(functional.softplus(diag))
        # return distributions.MultivariateNormal(loc=mean, scale_tril=out.tril())
        # return distributions.MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(scale))
        return \
            distributions.MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(scale)).log_prob(transform) \
            + grad_log_prob_x



variational_dict = {
    "gaussian": Gaussian,
}
