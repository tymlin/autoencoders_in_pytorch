import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

predictions = torch.linspace(0.001, 0.999, 100)
targets = torch.linspace(0, 1, 100)

# --------------------------- VISU HEATMAPS---------------------------------
prediction_grid, target_grid = torch.meshgrid(
    predictions,
    targets,
    indexing="xy",
)

mse = F.mse_loss(prediction_grid, target_grid, reduction="none")
bce = F.binary_cross_entropy(prediction_grid, target_grid, reduction="none")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

extent = (
    predictions[0].item(),
    predictions[-1].item(),
    targets[0].item(),
    targets[-1].item(),
)

for ax, values, name in zip(axes, (mse, bce), ("MSE", "BCE")):
    image = ax.imshow(
        values.numpy(),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    ax.set(title=name, xlabel="Prediction", ylabel="Target")
    fig.colorbar(image, ax=ax, label="Loss")

fig.tight_layout()
plt.show()
plt.close()
# --------------------------------------------------------------------------

# ------------------------- VISU 2D TARGET=0.5 -----------------------------
target = 0.5
curve_targets = torch.full_like(predictions, target)
# curve_targets = 1 - predictions

mse_curve = F.mse_loss(predictions, curve_targets, reduction="none")
bce_curve = F.binary_cross_entropy(predictions, curve_targets, reduction="none")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(predictions.numpy(), mse_curve.numpy(), label="MSE")
ax.plot(predictions.numpy(), bce_curve.numpy(), label="BCE")

ax.set(
    title=f"Reconstruction loss for target = {target}",
    xlabel="Prediction",
    ylabel="Loss",
)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
plt.show()
plt.close()
# --------------------------------------------------------------------------

# ------------------------- VISU 2D TARGET=0.1 -----------------------------
target = 0.1
curve_targets = torch.full_like(predictions, target)
# curve_targets = 1 - predictions

mse_curve = F.mse_loss(predictions, curve_targets, reduction="none")
bce_curve = F.binary_cross_entropy(predictions, curve_targets, reduction="none")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(predictions.numpy(), mse_curve.numpy(), label="MSE")
ax.plot(predictions.numpy(), bce_curve.numpy(), label="BCE")

ax.set(
    title=f"Reconstruction loss for target = {target}",
    xlabel="Prediction",
    ylabel="Loss",
)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
plt.show()
plt.close()
# --------------------------------------------------------------------------

# ------------------------- VISU 2D TARGET=0.9 -----------------------------
target = 0.9
curve_targets = torch.full_like(predictions, target)
# curve_targets = 1 - predictions

mse_curve = F.mse_loss(predictions, curve_targets, reduction="none")
bce_curve = F.binary_cross_entropy(predictions, curve_targets, reduction="none")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(predictions.numpy(), mse_curve.numpy(), label="MSE")
ax.plot(predictions.numpy(), bce_curve.numpy(), label="BCE")

ax.set(
    title=f"Reconstruction loss for target = {target}",
    xlabel="Prediction",
    ylabel="Loss",
)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
plt.show()
plt.close()
# --------------------------------------------------------------------------
