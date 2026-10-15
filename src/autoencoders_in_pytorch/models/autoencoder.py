from abc import ABC

import torch
import torchinfo
from torch import Tensor, nn

from autoencoders_in_pytorch.models.decoders import CNNDecoder, MLPDecoder
from autoencoders_in_pytorch.models.encoders import CNNEncoder, MLPEncoder, MLPVariationalEncoder


class BaseAutoencoder(nn.Module, ABC):
    name: str
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(inputs)
        outputs = self.decode(latents)
        return outputs, latents

    def summary(self, example_input_shape: tuple[int, int, int, int], depth: int = 4) -> str:
        col_names = ["input_size", "output_size", "num_params", "params_percent"]
        model_summary = str(
            torchinfo.summary(
                self,
                input_size=example_input_shape,
                depth=depth,
                col_names=col_names,
                verbose=0,
                device=self.device,
            )
        )
        return model_summary


class MLPAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        encoder_input_shape: tuple[int, int, int] = (1, 28, 28),
        encoder_hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        decoder_hidden_dims: list[int] | None = None,
        decoder_output_shape: tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()

        if encoder_hidden_dims is None:
            encoder_hidden_dims = []
        if decoder_hidden_dims is None:
            decoder_hidden_dims = []

        self.encoder = MLPEncoder(
            input_shape=encoder_input_shape,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_shape=decoder_output_shape,
        )


class CNNAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        encoder_input_shape: tuple[int, int, int] = (1, 28, 28),
        encoder_hidden_channels: list[int] | None = None,
        latent_channels: int = 32,
        decoder_hidden_channels: list[int] | None = None,
        decoder_output_shape: tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()
        if encoder_hidden_channels is None:
            encoder_hidden_channels = []
        if decoder_hidden_channels is None:
            decoder_hidden_channels = []

        self.encoder = CNNEncoder(
            input_shape=encoder_input_shape,
            hidden_channels=encoder_hidden_channels,
            latent_channels=latent_channels,
        )
        self.decoder = CNNDecoder(
            latent_channels=latent_channels,
            hidden_channels=decoder_hidden_channels,
            output_shape=decoder_output_shape,
        )


class MLPVariationalAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        encoder_input_shape: tuple[int, int, int] = (1, 28, 28),
        encoder_hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        decoder_hidden_dims: list[int] | None = None,
        decoder_output_shape: tuple[int, int, int] = (1, 28, 28),
    ):
        super().__init__()

        if encoder_hidden_dims is None:
            encoder_hidden_dims = []
        if decoder_hidden_dims is None:
            decoder_hidden_dims = []

        self.encoder = MLPVariationalEncoder(
            input_shape=encoder_input_shape,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
        )

        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_shape=decoder_output_shape,
        )

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.encoder(inputs)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        latents, mu, logvar = self.encode(inputs)  # z is our latents
        outputs = self.decode(latents)
        return outputs, latents, mu, logvar
