from typing import TYPE_CHECKING

from autoencoders_in_pytorch.losses import VAELoss
from autoencoders_in_pytorch.models import ModelStage
from autoencoders_in_pytorch.module.base import BaseModule
from autoencoders_in_pytorch.module.outputs import ModuleStepOut

if TYPE_CHECKING:
    from autoencoders_in_pytorch.models import BaseVariationalAutoencoder


class VariationalAutoencoderModule(BaseModule):
    model: "BaseVariationalAutoencoder"
    loss_fn: VAELoss

    def _common_step(self, batch: tuple, stage: ModelStage) -> ModuleStepOut:
        inputs, labels = batch
        inputs = inputs.to(self.device)
        targets = inputs

        model_output = self.model(inputs)
        loss = self.loss_fn(model_output.outputs, targets, model_output.mu, model_output.logvar)
        if stage == ModelStage.TRAIN:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        step_out = ModuleStepOut(
            inputs=inputs,
            targets=targets,
            outputs=model_output.outputs,
            latents=model_output.latents,
            loss=loss,
            labels=labels,
        )
        return step_out
