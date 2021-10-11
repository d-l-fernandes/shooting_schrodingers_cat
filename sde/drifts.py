import torch
from absl import flags

Tensor = torch.Tensor

flags.DEFINE_enum("drift", "nn_space",
                  ["constant",
                   "linear",
                   "nn_space",
                   "nn_space_mnist",
                   "nn_general",
                   "nn_general_mnist",
                   ],
                  "Drift to use.")
FLAGS = flags.FLAGS


class BaseDrift(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.)


class Constant(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.constant = torch.nn.Parameter(torch.randn(self.output_size, requires_grad=True))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.einsum("a,...a->...a", self.constant, torch.ones_like(x))


class Linear(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.linear = torch.nn.Linear(self.input_size, self.output_size)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.linear(x)


class NNSpace(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 10 * input_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.nn(x)


class NNSpaceMNIST(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 100
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.SELU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SELU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.nn(x)


class NNGeneral(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 20 * output_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+1, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if len(t.shape) != len(x.shape):
            t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]
        x = torch.cat((x, t), -1)
        return self.nn(x)


class NNGeneralMNIST(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 100
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+1, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if len(t.shape) != len(x.shape):
            t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]
        x = torch.cat((x, t), -1)
        return self.nn(x)


drifts_dict = {
    "constant": Constant,
    "linear": Linear,
    "nn_space": NNSpace,
    "nn_space_mnist": NNSpaceMNIST,
    "nn_general": NNGeneral,
    "nn_general_mnist": NNGeneralMNIST
}
