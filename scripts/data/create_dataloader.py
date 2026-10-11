from autoencoders_in_pytorch.data import DataModule, DatasetSplit, MNISTDataset
from autoencoders_in_pytorch.loggers import logger

SPLIT: DatasetSplit = DatasetSplit.TRAIN
SHOW_PLOT: bool = True
NUM_IMAGES: int = 10

if __name__ == "__main__":
    train_ds = MNISTDataset(split=DatasetSplit.TRAIN)
    test_ds = MNISTDataset(split=DatasetSplit.TEST)

    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=test_ds,
        test_ds=test_ds,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    datamodule.setup_dataloaders()
    train_dataloader = datamodule.get_train_dataloader()

    loader_iter = iter(train_dataloader)
    images, targets = next(loader_iter)
    logger.info(f"Images shape: {images.shape}, targets shape: {targets.shape}")
