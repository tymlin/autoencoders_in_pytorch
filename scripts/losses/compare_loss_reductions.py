import torch
import torch.nn.functional as F
from torch import Tensor, nn

batch_size = 2
latent_dim = 2
shape = (batch_size, 1, 28, 28)
inputs = torch.randn(shape)
inputs = F.sigmoid(inputs)
targets = torch.randn(shape)
targets = F.sigmoid(targets)
mu = torch.randn(batch_size, latent_dim)
logvar = torch.randn(batch_size, latent_dim)
rec_mse_sum_loss_fn = nn.MSELoss(reduction="sum")
rec_mse_mean_loss_fn = nn.MSELoss(reduction="mean")
rec_bce_sum_loss_fn = nn.BCELoss(reduction="sum")
rec_bce_mean_loss_fn = nn.BCELoss(reduction="mean")


def kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))


rec_loss_value = rec_mse_sum_loss_fn(inputs, targets)
print(f"Reconstruction MSE loss for the whole batch ({batch_size}): {rec_loss_value:.2f} (reduction: sum)")
rec_loss_value = rec_loss_value / batch_size
print(f"Reconstruction MSE loss averaged per sample/image: {rec_loss_value:.2f} (reduction: sum/batch_size)")
rec_loss_value = rec_mse_mean_loss_fn(inputs, targets)
print(f"Reconstruction MSE loss averaged over all pixels in the batch: {rec_loss_value:.2f} (reduction: mean)")

# rec_loss_value = rec_bce_sum_loss_fn(F.sigmoid(inputs), F.sigmoid(targets))
rec_loss_value = rec_bce_sum_loss_fn(inputs, targets)
print(f"Reconstruction BCE loss for the whole batch ({batch_size}): {rec_loss_value:.2f} (reduction: sum)")
rec_loss_value = rec_loss_value / batch_size
print(f"Reconstruction BCE loss averaged per sample/image: {rec_loss_value:.2f} (reduction: sum/batch_size)")
# rec_loss_value = rec_bce_mean_loss_fn(F.sigmoid(inputs), F.sigmoid(targets))
rec_loss_value = rec_bce_mean_loss_fn(inputs, targets)
print(f"Reconstruction BCE loss averaged over all pixels in the batch: {rec_loss_value:.2f} (reduction: mean)")

kl_loss_value = kl_loss(mu, logvar)
print(f"KL loss for the whole batch ({batch_size}): {kl_loss_value:.2f}")
kl_loss_value = kl_loss_value / batch_size
print(f"KL loss averaged per sample/image: {kl_loss_value:.2f} (reduction: sum/batch_size)")


# Example output:
# Reconstruction MSE loss for the whole batch (2): 141.43 (reduction: sum)
# Reconstruction MSE loss averaged per sample/image: 70.71 (reduction: sum/batch_size)
# Reconstruction MSE loss averaged over all pixels in the batch: 0.09 (reduction: mean)
# Reconstruction BCE loss for the whole batch (2): 1274.93 (reduction: sum)
# Reconstruction BCE loss averaged per sample/image: 637.46 (reduction: sum/batch_size)
# Reconstruction BCE loss averaged over all pixels in the batch: 0.81 (reduction: mean)
# KL loss for the whole batch (2): 2.06
# KL loss averaged per sample/image: 1.03 (reduction: sum/batch_size)
