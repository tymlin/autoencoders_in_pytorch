import math

from torch import Tensor, nn
from torchvision.ops import Conv2dNormActivation


class MLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_shape = output_shape

        self.output_dim = math.prod(output_shape)

        self.layers = nn.ModuleDict()

        current_dim = self.latent_dim
        for index, hidden_dim in enumerate(self.hidden_dims):
            self.layers[f"linear_{index}"] = nn.Linear(current_dim, hidden_dim)
            self.layers[f"relu_{index}"] = nn.ReLU()
            current_dim = hidden_dim

        self.layers["output"] = nn.Linear(current_dim, self.output_dim)
        self.layers["sigmoid"] = nn.Sigmoid()
        self.layers["unflatten"] = nn.Unflatten(
            dim=1,
            unflattened_size=self.output_shape,
        )

    def forward(self, latents: Tensor) -> Tensor:
        x = latents
        for layer in self.layers.values():
            x = layer(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: list[int],
        output_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.output_shape = output_shape

        self.layers = nn.ModuleDict()

        output_channels, height, width = output_shape
        output_paddings = []

        # reproduce the encoders spatial reductions, including the latent layer.
        for _ in range(len(hidden_channels) + 1):
            next_height = (height + 1) // 2
            next_width = (width + 1) // 2
            output_paddings.append(
                (
                    height - (2 * next_height - 1),
                    width - (2 * next_width - 1),
                )
            )
            height, width = next_height, next_width
        # the decoder reconstructs the sizes in reverse order.
        output_paddings.reverse()

        kernel_size = 3
        stride = 2
        padding = kernel_size // 2
        channels = latent_channels
        for index, out_channels in enumerate(self.hidden_channels):
            self.layers[f"conv_transpose_{index}"] = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_paddings[index],
            )
            self.layers[f"batch_norm_{index}"] = nn.BatchNorm2d(out_channels)
            self.layers[f"relu_{index}"] = nn.ReLU()
            channels = out_channels

        self.layers["output"] = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_paddings[-1],
        )
        self.layers["sigmoid"] = nn.Sigmoid()

    def forward(self, latents: Tensor) -> Tensor:
        x = latents
        for layer in self.layers.values():
            x = layer(x)
        return x


class CNNUpsampleDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: list[int],
        output_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.output_shape = output_shape

        self.layers = nn.ModuleDict()

        output_channels, height, width = self.output_shape
        target_sizes = [(height, width)]

        for _ in self.hidden_channels:
            height = (height + 1) // 2
            width = (width + 1) // 2
            target_sizes.append((height, width))

        target_sizes.reverse()

        kernel_size = 3
        stride = 1
        padding = kernel_size // 2
        channels = self.latent_channels
        for index, out_channels in enumerate(self.hidden_channels):
            self.layers[f"upsample_{index}"] = nn.Upsample(
                size=target_sizes[index],
                mode="nearest",
            )
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

        self.layers["upsample_output"] = nn.Upsample(
            size=target_sizes[-1],
            mode="nearest",
        )
        self.layers["output"] = nn.Conv2d(
            in_channels=channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.layers["sigmoid"] = nn.Sigmoid()

    def forward(self, latents: Tensor) -> Tensor:
        x = latents
        for layer in self.layers.values():
            x = layer(x)
        return x
