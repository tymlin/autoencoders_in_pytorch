import torch
from torch import nn

from autoencoders_in_pytorch.data import DataModule, DatasetSplit, MNISTDataset
from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import Device, MLPAutoencoder
from autoencoders_in_pytorch.module import AutoencoderModule
from autoencoders_in_pytorch.utils.progress import create_progress
from autoencoders_in_pytorch.utils.torch_utils import get_device_and_id

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

    datamodule.setup_dataloaders()
    resolved_device, _ = get_device_and_id(DEVICE)
    module.setup(device=resolved_device)

    train_dataloader = datamodule.get_train_dataloader()
    validation_dataloader = datamodule.get_val_dataloader()

    num_train_batches = len(train_dataloader)
    num_val_batches = len(validation_dataloader)
    for _epoch in range(EPOCHS):
        with create_progress(unit="batches") as progress:
            task_id = progress.add_task(
                f"Training `{module.model.name}`",
                total=num_train_batches,
                metrics={},
            )
            for _batch_ix, batch in enumerate(train_dataloader):
                out = module.training_step(batch)
                out = out.to_numpy()
                progress_metrics = {
                    "loss": f"{out.loss.item():.6f}",
                }
                progress.update(
                    task_id,
                    advance=1,
                    metrics=progress_metrics,
                )

        with create_progress(unit="batches") as progress:
            task_id = progress.add_task(
                f"Evaluating `{module.model.name}`",
                total=num_val_batches,
                metrics={},
            )
            for _batch_ix, batch in enumerate(validation_dataloader):
                out = module.validation_step(batch)
                out = out.to_numpy()
                progress_metrics = {
                    "loss": f"{out.loss.item():.6f}",
                }
                progress.update(
                    task_id,
                    advance=1,
                    metrics=progress_metrics,
                )
