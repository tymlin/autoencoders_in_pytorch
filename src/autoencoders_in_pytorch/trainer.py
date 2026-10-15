import random
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autoencoders_in_pytorch.data import DataModule
from autoencoders_in_pytorch.loggers import log_breaking_point, logger
from autoencoders_in_pytorch.metrics import MetricsContainer
from autoencoders_in_pytorch.models import Device, ModelStage
from autoencoders_in_pytorch.module import BaseModule
from autoencoders_in_pytorch.utils.files import create_directory, get_relpath, save_json
from autoencoders_in_pytorch.utils.progress import create_progress, progress_to_str
from autoencoders_in_pytorch.utils.torch_utils import get_device_and_id
from autoencoders_in_pytorch.visualization import plot_images, plot_latents


class Trainer:
    def __init__(
        self,
        module: BaseModule,
        datamodule: DataModule,
        max_epochs: int,
        run_dirpath: str | Path,
        device: Device,
        device_id: int = 0,
        num_visu_samples: int = 36,
        num_visu_latents: int | None = None,
    ) -> None:
        self.module = module
        self.datamodule = datamodule
        self.run_dirpath = Path(run_dirpath)
        self.max_epochs = max_epochs
        self.num_visu_samples = num_visu_samples
        self.num_visu_latents = num_visu_latents

        self.metrics: dict[ModelStage, MetricsContainer] = {
            ModelStage.TRAIN: MetricsContainer(),
            ModelStage.VALIDATION: MetricsContainer(),
        }

        self.device, self.device_id = get_device_and_id(device, device_id)
        self.current_epoch = 0
        self.current_step = 0

        ckpt_dirpath = Path(self.run_dirpath) / "checkpoints"
        create_directory(ckpt_dirpath, parents=True, exist_ok=True)
        self.ckpt_filepath = ckpt_dirpath / "last.pth"
        self.metrics_filepath = self.run_dirpath / "metrics.json"
        self.plots_dirpath = Path(self.run_dirpath) / "plots"
        create_directory(self.plots_dirpath, parents=True, exist_ok=True)

        self.visu_sample_indices = sorted(
            random.sample(
                range(len(self.datamodule.val_ds)),
                k=min(self.num_visu_samples, len(self.datamodule.val_ds)),
            )
        )
        self.visu_latent_indices = sorted(
            random.sample(
                range(len(self.datamodule.val_ds)),
                k=min(
                    self.num_visu_latents if self.num_visu_latents is not None else len(self.datamodule.val_ds),
                    len(self.datamodule.val_ds),
                ),
            )
        )
        self.visu_batch = None

        self.num_char = 150

    def fit(self):
        self.module.setup(device=self.device)
        self.log_module_summary()

        train_dataloader = self.datamodule.get_train_dataloader()
        validation_dataloader = self.datamodule.get_val_dataloader()

        # get specific batches for visualization with dataloader collate function
        self.visu_batch = validation_dataloader.collate_fn(
            [self.datamodule.val_ds[index] for index in self.visu_sample_indices]
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
                f"Starting epoch {self.current_epoch + 1}/{self.max_epochs} ",
                num_chars=self.num_char,
                fill_char="=",
            )
            self.training_epoch(train_dataloader)
            self.validation_epoch(validation_dataloader)

            self.log_epoch_summary()
            self.save_metrics()
            self.save_checkpoint()

    def training_epoch(self, dataloader: torch.utils.data.DataLoader):
        num_batches = len(dataloader)
        with create_progress(unit="batches") as progress:
            task_id = progress.add_task(
                f"Training `{self.module.model.name}`",
                total=num_batches,
                metrics={},
            )
            for batch_idx, batch in enumerate(dataloader):
                out = self.module.training_step(batch)
                self.current_step += 1

                metrics = self.module.compute_metrics(out)
                self.add_metrics(
                    ModelStage.TRAIN, epoch=self.current_epoch, step=batch_idx, num_samples=len(out), metrics=metrics
                )
                progress_metrics = {name: f"{value:.4f}" for name, value in metrics.items()}
                progress.update(
                    task_id,
                    advance=1,
                    metrics=progress_metrics,
                )
            final_progress_bar_text = f"Final progress: {progress_to_str(progress)}"
        logger.info(final_progress_bar_text)

    def validation_epoch(self, dataloader: torch.utils.data.DataLoader):
        num_batches = len(dataloader)
        with create_progress(unit="batches") as progress:
            task_id = progress.add_task(
                f"Evaluating `{self.module.model.name}`",
                total=num_batches,
                metrics={},
            )
            latent_batches = []
            label_batches = []
            for batch_idx, batch in enumerate(dataloader):
                out = self.module.validation_step(batch)
                out_numpy = out.to_numpy()
                latent_batches.append(out_numpy.latents)
                label_batches.append(out_numpy.labels)

                metrics = self.module.compute_metrics(out)
                self.add_metrics(
                    ModelStage.VALIDATION,
                    epoch=self.current_epoch,
                    step=batch_idx,
                    num_samples=len(out),
                    metrics=metrics,
                )

                progress_metrics = {name: f"{value:.4f}" for name, value in metrics.items()}
                progress.update(
                    task_id,
                    advance=1,
                    metrics=progress_metrics,
                )
            final_progress_bar_text = f"Final progress: {progress_to_str(progress)}"
        logger.info(final_progress_bar_text)

        out = self.module.validation_step(self.visu_batch).to_numpy()
        epoch = self.current_epoch + 1
        plot_images(
            images=out.inputs,
            sample_indices=self.visu_sample_indices,
            title=f"Epoch {epoch:03d} - inputs",
            save_filepath=(self.plots_dirpath / "inputs" / f"epoch_{epoch:03d}.png"),
        )
        plot_images(
            images=out.outputs,
            sample_indices=self.visu_sample_indices,
            title=f"Epoch {epoch:03d} - reconstructions",
            save_filepath=(self.plots_dirpath / "reconstructions" / f"epoch_{epoch:03d}.png"),
        )
        plot_latents(
            latents=np.concatenate(latent_batches)[self.visu_latent_indices],
            labels=np.concatenate(label_batches)[self.visu_latent_indices],
            class_names=self.datamodule.val_ds.classes_int2str,
            title=f"Epoch {epoch} - latent space",
            save_filepath=(self.plots_dirpath / "latents" / f"latents_epoch_{epoch:03d}.png"),
        )

    def add_metrics(
        self, stage: ModelStage, epoch: int, step: int, num_samples: int, metrics: dict[str, float]
    ) -> None:
        self.metrics[stage].add(
            epoch=epoch,
            step=step,
            num_samples=num_samples,
            metrics=metrics,
        )

    def get_all_metrics(self) -> dict[str, list[dict[str, Any]]]:
        metrics = {stage.value: container.to_records() for stage, container in self.metrics.items()}
        return metrics

    def log_module_summary(self):
        model_structure = str(self.module.model)
        image, _ = self.datamodule.train_ds[0]
        channels, height, width = image.shape
        input_shape = (self.datamodule.batch_size, channels, height, width)
        model_summary = self.module.model.summary(input_shape)
        optim_info = str(self.module.optimizer)
        loss_fn_info = str(self.module.loss_fn)

        module_info_dict = {
            "Model structure": model_structure,
            "Model summary": model_summary,
            "Optimizer": optim_info,
            "Loss function": loss_fn_info,
        }

        msg = f"`{self.module.name}` summary: \n"
        for name, info in module_info_dict.items():
            info = textwrap.indent(info, "\t\t")
            msg += f"\t{name}: \n{info}\n"
        logger.info(msg)

    def log_epoch_summary(self) -> None:
        log_breaking_point(
            logger, f"Finished epoch {self.current_epoch + 1}/{self.max_epochs}", num_chars=self.num_char, fill_char="="
        )

        msg = f"Epoch {self.current_epoch + 1} metrics summary: "
        for stage, container in self.metrics.items():
            msg += f"\n\t{stage.value}:"
            metrics = container.mean(epoch=self.current_epoch)
            for name, value in metrics.items():
                msg += f"\n\t\t{name}: {value:.8f}"

        logger.info(msg)

    def save_metrics(self) -> None:
        metrics = self.get_all_metrics()
        save_json(filepath=self.metrics_filepath, data=metrics)

    def save_checkpoint(self):
        self.module.model.eval()
        epoch, step = self.current_epoch, self.current_step
        logger.info(
            f"Saving the `{self.ckpt_filepath.stem}` checkpoint "
            f"(epoch={epoch}, step={step}) to '{get_relpath(self.ckpt_filepath)}'"
        )
        module_state = self.module.state_dict()
        datamodule_state = self.datamodule.state_dict()
        metrics = self.get_all_metrics()

        ckpt_state = {
            "module": module_state,
            "datamodule": datamodule_state,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        }
        torch.save(ckpt_state, self.ckpt_filepath)
