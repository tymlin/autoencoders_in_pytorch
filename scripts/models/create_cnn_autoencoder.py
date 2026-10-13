import torch

from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import CNNAutoencoder

if __name__ == "__main__":
    model = CNNAutoencoder(
        encoder_input_shape=(1, 28, 28),
        encoder_hidden_channels=[8, 16],
        latent_channels=32,
        decoder_hidden_channels=[16, 8],
        decoder_output_shape=(1, 28, 28),
    )
    x = torch.rand(2, 1, 28, 28)
    outputs, latents = model(x)
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Latent dim: {latents.shape}")
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Model architecture: \n{model}")
