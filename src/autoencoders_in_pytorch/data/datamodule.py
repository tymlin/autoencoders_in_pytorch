from typing import Callable

import torch
from torchvision.transforms import Compose

from autoencoders_in_pytorch.data.datasets import ImageDataset
from autoencoders_in_pytorch.data.types import DatasetSplit
from autoencoders_in_pytorch.loggers import logger


class DataModule:
    def __init__(
        self,
        train_ds: ImageDataset,
        val_ds: ImageDataset,
        test_ds: ImageDataset | None,
        inverse_transform: Compose | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        collate_fn_train: Callable | None = None,
        collate_fn_validation: Callable | None = None,
        collate_fn_test: Callable | None = None,
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.inverse_transform = inverse_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.train_generator = torch.Generator().manual_seed(seed)

        self.total_batches = {}
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.collate_fns: dict[DatasetSplit, Callable | None] = {
            DatasetSplit.TRAIN: collate_fn_train,
            DatasetSplit.VALIDATION: collate_fn_validation,
            DatasetSplit.TEST: collate_fn_test,
        }

        self._log_info()

    def setup_dataloaders(self) -> None:
        self.train_dataloader = self._dataloader(DatasetSplit.TRAIN)
        self.val_dataloader = self._dataloader(DatasetSplit.VALIDATION)
        if self.test_ds is not None:
            self.test_dataloader = self._dataloader(DatasetSplit.TEST)

    def get_dataset(self, split: DatasetSplit) -> ImageDataset | None:
        if split == DatasetSplit.TRAIN:
            return self.train_ds
        elif split == DatasetSplit.VALIDATION:
            return self.val_ds
        elif split == DatasetSplit.TEST:
            return self.test_ds
        else:
            msg = "Dataset split must be either 'train', 'validation' or 'test'"
            logger.error(msg)
            raise ValueError(msg)

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataloader is None:
            msg = "Training dataloader is not initialized. " "Call `setup_dataloaders()` first."
            logger.error(msg)
            raise RuntimeError(msg)

        return self.train_dataloader

    def get_val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_dataloader is None:
            msg = "Validation dataloader is not initialized. " "Call `setup_dataloaders()` first."
            logger.error(msg)
            raise RuntimeError(msg)

        return self.val_dataloader

    def get_test_dataloader(self) -> torch.utils.data.DataLoader | None:
        if self.test_ds is None:
            return None

        if self.test_dataloader is None:
            msg = "Test dataloader is not initialized. " "Call `setup_dataloaders()` first."
            logger.error(msg)
            raise RuntimeError(msg)

        return self.test_dataloader

    def _dataloader(self, split: DatasetSplit) -> torch.utils.data.DataLoader:
        shuffle = split == DatasetSplit.TRAIN and self.shuffle  # shuffle only for train
        drop_last = self.drop_last if split == DatasetSplit.TRAIN else False  # don't drop last for val/test

        dataset = self.get_dataset(split)
        if dataset is None:
            msg = f"Dataset for {split} split is None"
            logger.error(msg)
            raise ValueError(msg)

        collate_fn = self.collate_fns[split]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            generator=(self.train_generator if split == DatasetSplit.TRAIN else None),
        )

        self.total_batches[split] = len(dataloader)
        return dataloader

    def _log_info(self) -> None:
        datamodule_info = []
        datasets = {"train": self.train_ds, "val": self.val_ds, "test": self.test_ds}
        for split, ds in datasets.items():
            ds_name = "None" if ds is None else ds.name
            num_samples = 0 if ds is None else len(ds)
            ds_info = f"{split.rjust(6)}: {num_samples} ({ds_name})"
            datamodule_info.append(ds_info)
        datamodule_info = "\n".join(datamodule_info)
        logger.info(f"DataModule stats: \n{datamodule_info}")

    def state_dict(self) -> dict:
        state = {
            "train_generator_state": self.train_generator.get_state(),
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self.train_generator.set_state(state_dict["train_generator_state"])
        logger.info("Loaded datamodule state dict")
