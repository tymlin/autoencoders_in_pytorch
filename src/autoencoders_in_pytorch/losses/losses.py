from typing import Literal

import torch
from torch import Tensor, nn

from autoencoders_in_pytorch.loggers import logger

ReductionType = Literal["mean", "sum"]


class VAELoss(nn.Module):
    def __init__(
        self,
        reconstruction_loss_cls: type[nn.Module] = nn.MSELoss,
        reconstruction_loss_reduction: ReductionType = "sum",
        reconstruction_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        scale_by_batch_size: bool = True,
    ):
        super().__init__()

        if reconstruction_loss_reduction not in ("mean", "sum"):
            msg = f"Reconstruction loss reduction must be either 'mean' or 'sum', got {reconstruction_loss_reduction}."
            logger.error(msg)
            raise ValueError(msg)

        if reconstruction_loss_reduction == "mean" and scale_by_batch_size:
            msg = (
                "Using reconstruction_loss_reduction='mean' together with scale_by_batch_size=True divides the already "
                "averaged reconstruction loss by batch size again. This makes its scale relative to KL loss depend on "
                "batch size."
            )
            logger.warning(msg)

        self.reconstruction_loss_fn = reconstruction_loss_cls(reduction=reconstruction_loss_reduction)
        self.reconstruction_loss_reduction = reconstruction_loss_reduction
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.scale_by_batch_size = scale_by_batch_size

    @staticmethod
    def _kl_loss_fn(mu: Tensor, logvar: Tensor) -> Tensor:
        # D. Kingma, M. Welling; Auto-Encoding Variational Bayes
        # https://arxiv.org/abs/1312.6114
        # Appendix B
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss_value = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        return kl_loss_value

    def forward(
        self,
        outputs: Tensor,  # (B, C, H, W)
        targets: Tensor,  # (B, C, H, W)
        mu: Tensor,  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
        logvar: Tensor,  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
    ) -> Tensor:
        reconstruction_loss_value = self.reconstruction_loss_fn(outputs, targets)
        kl_loss_value = self._kl_loss_fn(mu, logvar)
        if self.scale_by_batch_size:
            batch_size = targets.shape[0]
            reconstruction_loss_value = reconstruction_loss_value / batch_size
            kl_loss_value = kl_loss_value / batch_size

        loss_value = self.reconstruction_loss_weight * reconstruction_loss_value + self.kl_loss_weight * kl_loss_value
        return loss_value
