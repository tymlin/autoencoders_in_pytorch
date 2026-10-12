from abc import ABC

import torch
from torch import Tensor, nn

from autoencoders_in_pytorch.models.decoders import MLPDecoder
from autoencoders_in_pytorch.models.encoders import MLPEncoder


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
