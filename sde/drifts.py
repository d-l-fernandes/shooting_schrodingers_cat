import torch
from absl import flags
import math
import torch.nn.functional as F

Tensor = torch.Tensor

flags.DEFINE_enum("drift", "nn_space",
                  ["constant",
                   "linear",
                   "nn_space",
                   "nn_space_mnist",
                   "nn_general",
                   "nn_general_mnist",
                   "score_network"
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
        intermediate_size = 20 * input_size
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.SiLU(),
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
            torch.nn.Linear(self.input_size + 1, intermediate_size), torch.nn.ReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.ReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.ReLU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]
        x = torch.cat((x, t), -1)
        return self.nn(x)


class NNGeneralMNIST(BaseDrift):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        intermediate_size = 200
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.input_size + 1, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, intermediate_size), torch.nn.LeakyReLU(),
            torch.nn.Linear(intermediate_size, self.output_size)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # if len(t.shape) == 0:
        #     t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]
        # else:
        #     t = (torch.einsum("a...,a->a...", torch.ones_like(x, device=x.device), t))[..., 0:1]
        t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]
        x = torch.cat((x, t), -1)
        return self.nn(x)


class ScoreNetwork(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        decoder_layers = [128, 128]
        encoder_layers = [16]
        pos_dim = 16
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, input_size]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers + [output_size],
                       activate_final=False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(input_size,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=torch.nn.LeakyReLU())

    def forward(self, x, t):
        t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]
        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out = self.net(h)
        return out


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final=False, activation_fn=F.relu):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
      From Fairseq.
      Build sinusoidal embeddings.
      This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
      https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)

    emb = timesteps * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0, 1])

    return emb


drifts_dict = {
    "constant": Constant,
    "linear": Linear,
    "nn_space": NNSpace,
    "nn_space_mnist": NNSpaceMNIST,
    "nn_general": NNGeneral,
    "nn_general_mnist": NNGeneralMNIST,
    "score_network": ScoreNetwork
}
