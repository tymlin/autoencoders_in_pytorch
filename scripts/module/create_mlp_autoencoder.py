import torch
from torch import nn

from autoencoders_in_pytorch.models import Device, MLPAutoencoder
from autoencoders_in_pytorch.module import AutoencoderModule
from autoencoders_in_pytorch.utils.torch_utils import get_device_and_id

DEVICE: Device = Device.CPU
BATCH_SIZE: int = 2

if __name__ == "__main__":
    model = MLPAutoencoder(
        encoder_input_shape=(1, 28, 28),
        encoder_hidden_dims=[128, 64],
        latent_dim=32,
        decoder_hidden_dims=[64, 128],
        decoder_output_shape=(1, 28, 28),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()
    module = AutoencoderModule(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    resolved_device, _ = get_device_and_id(DEVICE)
    module.setup(device=resolved_device)

    inputs = torch.rand(BATCH_SIZE, 1, 28, 28)
    labels = torch.randint(0, 2, (BATCH_SIZE,))
    batch = (inputs, labels)
    train_step_out = module.training_step(batch)
    validation_step_out = module.validation_step(batch)
    module_state_dict = module.state_dict()
