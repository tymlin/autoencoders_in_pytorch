from autoencoders_in_pytorch.models.autoencoder.base import BaseVariationalAutoencoder
from autoencoders_in_pytorch.models.components import MLPDecoder, MLPVariationalEncoder


class MLPVariationalAutoencoder(BaseVariationalAutoencoder):
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
