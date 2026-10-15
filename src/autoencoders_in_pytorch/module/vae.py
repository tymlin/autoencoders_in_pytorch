import torch

from autoencoders_in_pytorch.models import ModelStage
from autoencoders_in_pytorch.module.base import BaseModule
from autoencoders_in_pytorch.module.outputs import ModuleStepOut


class VariationalAutoencoderModule(BaseModule):
    def _common_step(self, batch: tuple, stage: ModelStage) -> ModuleStepOut:
        pass

    @torch.no_grad()
    def compute_metrics(self, out: ModuleStepOut) -> dict[str, float]:
        pass
