from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import ModelStage

if TYPE_CHECKING:
    from autoencoders_in_pytorch.models import BaseAutoencoder


@dataclass
class ModuleStepItem:
    inputs: Tensor | np.ndarray  # Tensor: (C, H, W), NumPy: (H, W, C)
    targets: Tensor | np.ndarray  # Tensor: (C, H, W), NumPy: (H, W, C)
    outputs: Tensor | np.ndarray  # Tensor: (C, H, W), NumPy: (H, W, C)
    latents: Tensor | np.ndarray  # (L,)
    label: int | None = None


@dataclass
class ModuleStepOut:
    inputs: Tensor | np.ndarray  # Tensor: (B, C, H, W), NumPy: (B, H, W, C)
    targets: Tensor | np.ndarray  # Tensor: (B, C, H, W), NumPy: (B, H, W, C)
    outputs: Tensor | np.ndarray  # Tensor: (B, C, H, W), NumPy: (B, H, W, C)
    latents: Tensor | np.ndarray  # (B, L)
    loss: Tensor | np.ndarray  # Scalar: ()
    labels: Tensor | np.ndarray | None = None  # (B,)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> ModuleStepItem:
        return ModuleStepItem(
            inputs=self.inputs[index],
            targets=self.targets[index],
            outputs=self.outputs[index],
            latents=self.latents[index],
            label=int(self.labels[index]) if self.labels is not None else None,
        )

    def to_numpy(self) -> "ModuleStepOut":
        def convert(value: Tensor | np.ndarray, image: bool = False) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            array = value.detach().cpu().numpy()
            return array.transpose(0, 2, 3, 1) if image else array

        return ModuleStepOut(
            inputs=convert(self.inputs, image=True),
            targets=convert(self.targets, image=True),
            outputs=convert(self.outputs, image=True),
            latents=convert(self.latents),
            loss=convert(self.loss),
            labels=convert(self.labels) if self.labels is not None else None,
        )


class AutoencoderModule:
    model: "BaseAutoencoder"
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module

    def __init__(
        self,
        model: "BaseAutoencoder",
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = None

    @property
    def name(self):
        return self.__class__.__name__

    def setup(
        self,
        device: str,
        ckpt_state: dict[str, Any] | None = None,
    ) -> None:
        self.device = device

        logger.info(f"Moving model and loss to device `{self.device}`")
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        if ckpt_state is not None:
            self.load_state_dict(ckpt_state)

    def training_step(self, batch) -> ModuleStepOut:
        stage: ModelStage = ModelStage.TRAIN
        self.model.train()
        out = self._common_step(batch, stage)
        return out

    def validation_step(self, batch) -> ModuleStepOut:
        stage: ModelStage = ModelStage.VALIDATION
        self.model.eval()
        with torch.no_grad():
            out = self._common_step(batch, stage)
        return out

    def _common_step(self, batch, stage: ModelStage) -> ModuleStepOut:
        inputs, labels = batch
        inputs = inputs.to(self.device)
        targets = inputs

        outputs, latents = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        if stage == ModelStage.TRAIN:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        step_out = ModuleStepOut(
            inputs=inputs,
            targets=targets,
            outputs=outputs,
            latents=latents,
            loss=loss,
            labels=labels,
        )
        return step_out

    @torch.no_grad()
    def compute_metrics(self, out: ModuleStepOut) -> dict[str, float]:
        metrics = {
            "loss": out.loss.item(),
        }
        return metrics

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
