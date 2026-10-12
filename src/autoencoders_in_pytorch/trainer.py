from pathlib import Path

import torch

from autoencoders_in_pytorch.consts import NOW, RUNS_DIRPATH
from autoencoders_in_pytorch.data import DataModule
from autoencoders_in_pytorch.loggers import configure_logger, log_breaking_point, logger
from autoencoders_in_pytorch.models import Device
from autoencoders_in_pytorch.module import AutoencoderModule
from autoencoders_in_pytorch.utils.files import create_directory, get_relpath
from autoencoders_in_pytorch.utils.progress import create_progress, progress_to_str
from autoencoders_in_pytorch.utils.torch_utils import get_device_and_id


class Trainer:
    def __init__(
        self,
        module: AutoencoderModule,
        datamodule: DataModule,
        max_epochs: int,
        device: Device,
        device_id: int = 0,
        run_dirpath: str | Path = RUNS_DIRPATH,
        run_name: str | None = None,
    ) -> None:
        self.module = module
        self.datamodule = datamodule
        self.run_dirpath = run_dirpath
        self.max_epochs = max_epochs

        run_name_postfix = f"_{run_name}" if run_name else ""
        self.run_dirpath = Path(run_dirpath) / f"{NOW}{run_name_postfix}"
        create_directory(self.run_dirpath, parents=True, exist_ok=True)
        configure_logger(run_dirpath=self.run_dirpath)

        self.device, self.device_id = get_device_and_id(device, device_id)
        self.current_epoch = 0
        self.current_step = 0
        ckpt_dirpath = Path(self.run_dirpath) / "checkpoints"
        create_directory(ckpt_dirpath, parents=True, exist_ok=True)
        self.ckpt_filepath = ckpt_dirpath / "last.pth"

        self.num_char = 150

    def fit(self):
        self.datamodule.setup_dataloaders()
        self.module.setup(device=self.device)

        train_dataloader = self.datamodule.get_train_dataloader()
        validation_dataloader = self.datamodule.get_val_dataloader()

        num_train_batches = len(train_dataloader)
        num_val_batches = len(validation_dataloader)
        logger.info(
            "Dataloader batches:\n"
            f"\tTrain: {num_train_batches} batches\n"
            f"\tValidation  : {num_val_batches}  batches"
        )

        log_breaking_point(
            logger,
            f" Starting the training of `{self.module.name}` from scratch",
            num_chars=self.num_char,
            fill_char="=",
            n_bottom=1,
            n_top=1,
        )
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            log_breaking_point(
                logger,
                f" Starting epoch {self.current_epoch + 1}/{self.max_epochs} ",
                num_chars=self.num_char,
                fill_char="=",
            )
            self.training_epoch(train_dataloader)
            self.validation_epoch(validation_dataloader)

        self.save_checkpoint()

    def training_epoch(self, dataloader: torch.utils.data.DataLoader):
        num_batches = len(dataloader)
        with create_progress(unit="batches") as progress:
            task_id = progress.add_task(
                f"Training `{self.module.model.name}`",
                total=num_batches,
                metrics={},
            )
            for _batch_ix, batch in enumerate(dataloader):
                out = self.module.training_step(batch)
                self.current_step += 1
                out = out.to_numpy()
                progress_metrics = {
                    "loss": f"{out.loss.item():.6f}",
                }
                progress.update(
                    task_id,
                    advance=1,
                    metrics=progress_metrics,
                )
            final_progress_bar_text = f"Final progress: \n{progress_to_str(progress)}"
        logger.info(final_progress_bar_text)

    def validation_epoch(self, dataloader: torch.utils.data.DataLoader):
        num_batches = len(dataloader)
        with create_progress(unit="batches") as progress:
            task_id = progress.add_task(
                f"Evaluating `{self.module.model.name}`",
                total=num_batches,
                metrics={},
            )
            for _batch_ix, batch in enumerate(dataloader):
                out = self.module.validation_step(batch)
                out = out.to_numpy()
                progress_metrics = {
                    "loss": f"{out.loss.item():.6f}",
                }
                progress.update(
                    task_id,
                    advance=1,
                    metrics=progress_metrics,
                )
            final_progress_bar_text = f"Final progress: \n{progress_to_str(progress)}"
        logger.info(final_progress_bar_text)

    def save_checkpoint(self):
        self.module.model.eval()
        epoch, step = self.current_epoch, self.current_step
        logger.info(
            f"Saving the `{self.ckpt_filepath.stem}` checkpoint "
            f"(epoch={epoch}, step={step}) to '{get_relpath(self.ckpt_filepath)}'"
        )
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()

        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "epoch": epoch,
            "step": step,
        }
        torch.save(ckpt_state, self.ckpt_filepath)
