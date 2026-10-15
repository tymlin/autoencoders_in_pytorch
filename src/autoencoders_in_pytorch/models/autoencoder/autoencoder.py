from autoencoders_in_pytorch.models.autoencoder.base import BaseAutoencoder
from autoencoders_in_pytorch.models.components import CNNDecoder, CNNEncoder, MLPDecoder, MLPEncoder


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
