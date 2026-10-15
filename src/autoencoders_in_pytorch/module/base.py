from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

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

    @abstractmethod
    @torch.no_grad()
    def compute_metrics(self, out: ModuleStepOut) -> dict[str, float]:
        raise NotImplementedError()

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
