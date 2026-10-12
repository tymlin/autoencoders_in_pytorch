import torch

from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import MLPAutoencoder

if __name__ == "__main__":
    model = MLPAutoencoder(
        encoder_input_shape=(1, 28, 28),
        encoder_hidden_dims=[128],
        latent_dim=32,
        decoder_hidden_dims=[128],
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
