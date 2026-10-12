import torch
from torch import nn

from autoencoders_in_pytorch.data import DataModule, DatasetSplit, MNISTDataset
from autoencoders_in_pytorch.models import Device, MLPAutoencoder
from autoencoders_in_pytorch.module import AutoencoderModule
from autoencoders_in_pytorch.trainer import Trainer

DEVICE: Device = Device.CPU
BATCH_SIZE: int = 64
EPOCHS: int = 10

if __name__ == "__main__":
    train_ds = MNISTDataset(split=DatasetSplit.TRAIN)
    test_ds = MNISTDataset(split=DatasetSplit.TEST)
    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=test_ds,  # use test as validation
        test_ds=None,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    model = MLPAutoencoder(
        encoder_input_shape=(1, 28, 28),
        encoder_hidden_dims=[128],
        latent_dim=32,
        decoder_hidden_dims=[128],
        decoder_output_shape=(1, 28, 28),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.MSELoss()
    module = AutoencoderModule(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
    )

    trainer = Trainer(
        module=module,
        datamodule=datamodule,
        max_epochs=EPOCHS,
        device=DEVICE,
    )

    trainer.fit()
