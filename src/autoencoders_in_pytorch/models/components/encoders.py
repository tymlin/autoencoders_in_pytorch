import math

import torch
from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.input_dim = math.prod(input_shape)

        self.layers = nn.ModuleDict()
        self.layers["flatten"] = nn.Flatten(start_dim=1)

        current_dim = self.input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            self.layers[f"linear_{index}"] = nn.Linear(current_dim, hidden_dim)
            self.layers[f"relu_{index}"] = nn.ReLU()
            current_dim = hidden_dim

        self.layers["latent"] = nn.Linear(current_dim, latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers.values():
            x = layer(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_channels: list[int],
        latent_channels: int,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels

        self.layers = nn.ModuleDict()

        kernel_size = 3
        stride = 2
        padding = kernel_size // 2
        channels, height, width = input_shape
        for index, out_channels in enumerate(hidden_channels):
            self.layers[f"conv_{index}"] = Conv2dNormActivation(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                bias=False,
            )
            channels = out_channels

        self.layers["latent"] = nn.Conv2d(
            in_channels=channels,
            out_channels=latent_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers.values():
            x = layer(x)
        return x


class MLPVariationalEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.input_dim = math.prod(input_shape)

        self.layers = nn.ModuleDict()
        self.layers["flatten"] = nn.Flatten(start_dim=1)

        current_dim = self.input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            self.layers[f"linear_{index}"] = nn.Linear(current_dim, hidden_dim)
            self.layers[f"relu_{index}"] = nn.ReLU()
            current_dim = hidden_dim

        self.mu = nn.Linear(current_dim, latent_dim)
        self.logvar = nn.Linear(current_dim, latent_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = inputs
        for layer in self.layers.values():
            x = layer(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        # reparametrization
        std = torch.exp(0.5 * logvar)  # std = e^(0.5*log(std^2)) = e^(log(sqrt(std^2))); log(std^2) = logvar
        eps = torch.randn_like(std, device=std.device)
        z = mu + eps * std  # z is our latents

        return z, mu, logvar
