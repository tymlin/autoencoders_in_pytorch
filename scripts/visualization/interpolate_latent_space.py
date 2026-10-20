from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from autoencoders_in_pytorch.consts import RUNS_DIRPATH
from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import (
    BaseModel,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)

RUN_DIRPATH = RUNS_DIRPATH / "run_name"
CHECKPOINT_FILEPATH = RUN_DIRPATH / "checkpoints" / "last.pth"
OUTPUT_FILEPATH = RUN_DIRPATH / "results_plots" / "latent_space_grid.png"

MODEL_CLS = MLPVariationalAutoencoder  # MLPAutoencoder
IMAGE_SHAPE = (1, 32, 32)
ENCODER_HIDDEN_DIMS = [128, 64]
DECODER_HIDDEN_DIMS = [64, 128]

DEVICE = "cpu"
X_RANGE = (-3.0, 3.0)
Y_RANGE = (-3.0, 3.0)
NUM_STEPS = 20


@torch.inference_mode()
def interpolate_space(
    model: BaseModel,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_steps: int,
) -> plt.Figure:
    if num_steps < 2:
        msg = "`num_steps` must be at least 2."
        logger.error(msg)
        raise ValueError(msg)

    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        msg = "Each coordinate range must have its lower bound before its upper bound."
        logger.error(msg)
        raise ValueError(msg)

    model.eval()

    x = torch.linspace(
        x_range[0],
        x_range[1],
        num_steps,
        device=model.device,
        dtype=model.dtype,
    )
    y = torch.linspace(
        y_range[1],
        y_range[0],
        num_steps,
        device=model.device,
        dtype=model.dtype,
    )

    # Columns: increasing z1. Rows: decreasing z2.
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    latents = torch.stack(
        [xx, yy],
        dim=-1,
    ).reshape(-1, 2)  # (num_steps^2, 2)

    images = model.decode(latents).outputs  # (num_steps^2, C, H, W)
    grid = make_grid(
        images.cpu(),
        nrow=num_steps,
        padding=0,
    )  # (3, H_grid, W_grid)

    # Each sampled coordinate lies at the center of its image tile.
    dx = (x_range[1] - x_range[0]) / (num_steps - 1)
    dy = (y_range[1] - y_range[0]) / (num_steps - 1)
    extent = (
        x_range[0] - dx / 2,
        x_range[1] + dx / 2,
        y_range[0] - dy / 2,
        y_range[1] + dy / 2,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        grid.permute(1, 2, 0).numpy(),
        extent=extent,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(f"{model.name} — latent space")

    return fig


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"

    model = MODEL_CLS(
        encoder_input_shape=IMAGE_SHAPE,
        encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
        latent_dim=2,
        decoder_hidden_dims=DECODER_HIDDEN_DIMS,
        decoder_output_shape=IMAGE_SHAPE,
    )

    checkpoint = torch.load(
        CHECKPOINT_FILEPATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["module"]["model"])
    del checkpoint

    model.to(DEVICE)

    fig = interpolate_space(
        model=model,
        x_range=X_RANGE,
        y_range=Y_RANGE,
        num_steps=NUM_STEPS,
    )

    output_filepath = Path(OUTPUT_FILEPATH)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved latent space grid to '{output_filepath}'")
