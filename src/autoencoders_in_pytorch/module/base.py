from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.image import (
    deep_image_structure_and_texture_similarity,
    learned_perceptual_image_patch_similarity,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import ModelStage

if TYPE_CHECKING:
    from autoencoders_in_pytorch.models import BaseModel

from autoencoders_in_pytorch.module.outputs import ModuleStepOut


class BaseModule(ABC):
    model: "BaseModel"
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module

    def __init__(
        self,
        model: "BaseModel",
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        calculate_perceptual_metrics: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.calculate_perceptual_metrics = calculate_perceptual_metrics

        self.device = None

        if self.calculate_perceptual_metrics:
            msg = (
                "Perceptual metrics (LPIPS and DISTS) are enabled. "
                "Computing them for every batch slows down training and validation."
            )
            logger.warning(msg)

    @property
    def name(self):
        return self.__class__.__name__

    def setup(
        self,
        device: str,
        ckpt_state: dict[str, Any] | None = None,
    ) -> None:
        self.device = device

        logger.info(f"Moving model and loss to the device: `{self.device}`")
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        if ckpt_state is not None:
            self.load_state_dict(ckpt_state)

    def training_step(self, batch: tuple) -> ModuleStepOut:
        stage: ModelStage = ModelStage.TRAIN
        self.model.train()
        out = self._common_step(batch, stage)
        return out

    def validation_step(self, batch: tuple) -> ModuleStepOut:
        stage: ModelStage = ModelStage.VALIDATION
        self.model.eval()
        with torch.no_grad():
            out = self._common_step(batch, stage)
        return out

    @abstractmethod
    def _common_step(self, batch: tuple, stage: ModelStage) -> ModuleStepOut:
        raise NotImplementedError()

    @torch.no_grad()
    def compute_metrics(self, out: ModuleStepOut) -> dict[str, float]:
        outputs = out.outputs  # reconstruction from the decoder
        targets = out.inputs  # ground-truth data from the dataset

        metrics = {"loss": out.loss}
        mse = F.mse_loss(outputs, targets)
        metrics = {**metrics, "mse": mse}
        mae = F.l1_loss(outputs, targets)
        metrics = {**metrics, "mae": mae}

        ssim = structural_similarity_index_measure(
            outputs,
            targets,
            data_range=1.0,
        )
        metrics = {**metrics, "ssim": ssim}
        psnr = peak_signal_noise_ratio(
            outputs,
            targets,
            data_range=1.0,
            dim=(1, 2, 3),
            reduction="elementwise_mean",
        )
        metrics = {**metrics, "psnr": psnr}

        if self.calculate_perceptual_metrics:
            perceptual_outputs = outputs
            perceptual_targets = targets

            if perceptual_outputs.shape[1] != 3:
                if perceptual_outputs.shape[1] == 1:
                    perceptual_outputs = perceptual_outputs.repeat(1, 3, 1, 1)
                    perceptual_targets = perceptual_targets.repeat(1, 3, 1, 1)
                else:
                    msg = f"Perceptual metrics require images with 1 or 3 channels, got {perceptual_outputs.shape[1]}."
                    logger.error(msg)
                    raise ValueError(msg)

            lpips = learned_perceptual_image_patch_similarity(
                perceptual_outputs,
                perceptual_targets,
                # AlexNet requires at least 31x31 inputs; SqueezeNet supports 17x17 and VGG supports 16x16.
                net_type="squeeze",
                reduction="mean",
                normalize=True,
            )
            metrics = {**metrics, "lpips": lpips}

            dists = deep_image_structure_and_texture_similarity(
                perceptual_outputs,
                perceptual_targets,
                reduction="mean",
            )
            metrics = {**metrics, "dists": dists}

        return {name: value.item() for name, value in metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        module_state = {
            "model": model_state,
            "optimizer": optimizer_state,
        }
        return module_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        logger.info(f"Loading `{self.name}` state dict.")

        self.model.load_state_dict(state_dict["model"])
        logger.info(f"\tLoaded `{self.model.name}` state dict")

        self.optimizer.load_state_dict(state_dict["optimizer"])
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(f"\tLoaded optimizer state with lr = {lr}")

        logger.info(f"Loaded `{self.name}` state dict")
