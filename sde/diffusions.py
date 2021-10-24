import numpy as np
import torch
import torch.nn.functional as functional
from absl import flags

Tensor = torch.Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

flags.DEFINE_enum("diffusion", "nn_space_diagonal",
                  ["scalar",
                   "constant_diagonal",
                   "nn_time_diagonal", "nn_time",
                   "nn_space_diagonal", "nn_space", "nn_space_mnist",
                   "nn_general_diagonal", "nn_general", "nn_general_mnist"
                   ],
                  "Diffusion to use.")
FLAGS = flags.FLAGS


class BaseDiffusion(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, final_t: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.final_t = final_t

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill(0.)


class Scalar(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int, final_t: float):
        super().__init__(input_size, output_size, final_t)
        # self.constant = np.sqrt(2)
        self.constant = np.sqrt(1.)
        self.g_max = np.sqrt(1.)
        self.g_min = np.sqrt(1.)
        g_diff = self.g_max - self.g_min
        self.gamma_t = \
            lambda t: (self.g_min + 2 * g_diff * t / self.final_t) * (t < self.final_t / 2) + \
                      ((2 * self.g_max - self.g_min) - 2 * g_diff * t / self.final_t) * (t >= self.final_t / 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # return torch.diag_embed(torch.ones_like(x, device=x.device) * self.constant)
        gamma = self.gamma_t(t).to(x.device)
        if len(gamma.shape) == 0:
            return torch.diag_embed(torch.ones_like(x, device=x.device) * gamma)
        else:
            return torch.diag_embed(torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), gamma))
        # return torch.ones_like(x, device=x.device) * self.constant


class ConstantDiagonal(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.constant = torch.nn.Parameter(torch.randn(self.output_size, device=device), requires_grad=True)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        diff = torch.einsum("a,...a->...a", torch.sigmoid(self.constant), torch.ones_like(x, device=x.device))
        # return diff
        return torch.diag_embed(diff)


class NNTimeDiagonal(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 5 * output_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.diag_embed(functional.softplus(self.nn(t)))


class NNTime(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 5 * output_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, self.output_size**2)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.nn(t)
        out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        diag = torch.diagonal(out, dim1=-1, dim2=-2)
        out = out - torch.diag_embed(diag) + torch.diag_embed(functional.softplus(diag))
        return out.tril()


class NNSpaceDiagonal(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 5 * output_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_size, intermediate_size), torch.nn.SELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SELU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.diag_embed(functional.softplus(self.nn(x)))


class NNSpace(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 10 * output_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, self.output_size**2)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.nn(x)
        out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        diag = torch.diagonal(out, dim1=-1, dim2=-2)
        out = out - torch.diag_embed(diag) + torch.diag_embed(torch.sigmoid(diag))
        return out.tril()


class NNSpaceMNIST(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 100
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, self.output_size**2)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        out = self.nn(x)
        out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        diag = torch.diagonal(out, dim1=-1, dim2=-2)
        out = out - torch.diag_embed(diag) + torch.diag_embed(torch.sigmoid(diag))
        return out.tril()


class NNGeneralDiagonal(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 10 * (input_size + 1)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+1, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = torch.cat((x, t), -1)
        return torch.diag_embed(functional.sigmoid(self.nn(x)))


class NNGeneral(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 20 * output_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+1, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size**2)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = torch.cat((x, t), -1)
        out = self.nn(x)
        out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        diag = torch.diagonal(out, dim1=-1, dim2=-2)
        out = out - torch.diag_embed(diag) + torch.diag_embed(torch.sigmoid(diag))
        return out.tril()


class NNGeneralMNIST(BaseDiffusion):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 300
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+1, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size**2)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = torch.cat((x, t), -1)
        out = self.nn(x)
        out = out.reshape(out.shape[:-1] + (self.output_size,) + (self.output_size,))
        diag = torch.diagonal(out, dim1=-1, dim2=-2)
        out = out - torch.diag_embed(diag) + torch.diag_embed(torch.sigmoid(diag))
        return out.tril()


diffusions_dict = {
    "scalar": Scalar,
    "constant_diagonal": ConstantDiagonal,
    "nn_time_diagonal": NNTimeDiagonal,
    "nn_time": NNTime,
    "nn_space_diagonal": NNSpaceDiagonal,
    "nn_space": NNSpace,
    "nn_space_mnist": NNSpaceMNIST,
    "nn_general_diagonal": NNGeneralDiagonal,
    "nn_general": NNGeneral,
    "nn_general_mnist": NNGeneralMNIST
}
