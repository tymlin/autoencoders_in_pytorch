import torch
from torch import nn

from autoencoders_in_pytorch.config import DataConfig, RunConfig, TrainerConfig
from autoencoders_in_pytorch.consts import NOW
from autoencoders_in_pytorch.data import DATASET_CLASSES, DataModule, DatasetSplit, DatasetType, build_transform
from autoencoders_in_pytorch.loggers import configure_logger, logger
from autoencoders_in_pytorch.losses import VAELoss
from autoencoders_in_pytorch.models import MLPVariationalAutoencoder
from autoencoders_in_pytorch.module import VariationalAutoencoderModule
from autoencoders_in_pytorch.trainer import Trainer
from autoencoders_in_pytorch.utils.files import save_json
from autoencoders_in_pytorch.utils.torch_utils import seed_everything

BATCH_SIZE: int = 128
EPOCHS: int = 100

if __name__ == "__main__":
    config = RunConfig(
        run_name_suffix="mlp-vae_mnist",
        data=DataConfig(
            dataset=DatasetType.MNIST,
            batch_size=BATCH_SIZE,
            val_split=DatasetSplit.TEST,  # we use test set as validation
        ),
        trainer=TrainerConfig(
            max_epochs=EPOCHS,
        ),
    )

    # first configure the logger with run dirpath and file handler
    run_name_suffix = f"_{config.run_name_suffix}" if config.run_name_suffix else ""
    run_dirpath = config.run_dirpath / f"{NOW}{run_name_suffix}"
    configure_logger(run_dirpath=run_dirpath)

    config.run_dirpath = run_dirpath
    save_json(run_dirpath / "config.json", config.model_dump(mode="json"))
    logger.info(f"Training configuration: \n{config.model_dump_json(indent=4)}")

    seed_everything(config.seed)
    dataset_cls = DATASET_CLASSES[config.data.dataset]
    transform = build_transform(image_size=config.data.image_size)
    train_ds = dataset_cls(
        split=config.data.train_split,
        transform=transform,
    )
    val_ds = dataset_cls(
        split=config.data.val_split,
        transform=transform,
    )
    test_ds = (
        dataset_cls(split=config.data.test_split, transform=transform) if config.data.test_split is not None else None
    )
    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.persistent_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
        shuffle=config.data.shuffle,
        seed=config.seed,
    )

    channels, height, width = train_ds[0][0].shape
    image_shape: tuple[int, int, int] = (channels, height, width)
    model = MLPVariationalAutoencoder(
        encoder_input_shape=image_shape,
        encoder_hidden_dims=[128, 64],
        latent_dim=32,
        decoder_hidden_dims=[64, 128],
        decoder_output_shape=image_shape,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    loss_fn = VAELoss(
        reconstruction_loss_cls=nn.MSELoss,
        reconstruction_loss_weight=1.0,
        reconstruction_loss_reduction="sum",
        kl_loss_weight=1.0,
        scale_by_batch_size=True,
    )
    module = VariationalAutoencoderModule(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        calculate_perceptual_metrics=config.trainer.calculate_perceptual_metrics,
    )

    trainer = Trainer(
        module=module,
        datamodule=datamodule,
        max_epochs=config.trainer.max_epochs,
        run_dirpath=run_dirpath,
        device=config.trainer.device,
        device_id=config.trainer.device_id,
        num_visu_samples=config.trainer.num_visu_samples,
        num_visu_latents=config.trainer.num_visu_latents,
    )

    trainer.fit()
