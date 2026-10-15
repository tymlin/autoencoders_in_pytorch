from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.ticker import MaxNLocator

from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.metrics import MetricsContainer
from autoencoders_in_pytorch.models import ModelStage
from autoencoders_in_pytorch.utils.files import create_directory


def plot_images(
    images: np.ndarray,  # (N, H, W, C)
    sample_indices: list[int],
    ncols: int = 6,
    title: str = "Images",
    save_filepath: str | Path | None = None,
    show_plot: bool = False,
) -> plt.Figure:
    if len(images) == 0 or len(images) != len(sample_indices):
        msg = "Images and indices must be non-empty and have equal lengths."
        logger.error(msg)
        raise ValueError(msg)

    if ncols < 1:
        msg = "Number of columns must be greater than zero."
        logger.error(msg)
        raise ValueError(msg)

    ncols = min(ncols, len(images))
    nrows = (len(images) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.5 * ncols, 2.5 * nrows),
        squeeze=False,
        layout="none",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.99,
        bottom=0.02,
        top=0.92,
        wspace=0.05,
        hspace=0.2,
    )
    fig.suptitle(title, y=0.97)

    for ax in axes.flat:
        ax.set_axis_off()

    for image, index, ax in zip(images, sample_indices, axes.flat):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"#{index}")

    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        create_directory(save_filepath.parent, parents=True, exist_ok=True)
        fig.savefig(save_filepath, dpi=200)
        logger.info(f"Saved figure to '{save_filepath}'")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig


def project_latents(
    latents: np.ndarray,  # (N, L) or (N, C, H, W)
    random_state: int = 42,
) -> np.ndarray:  # (N, 2)
    if latents.ndim > 2:
        latents = latents.reshape(latents.shape[0], -1)

    if latents.ndim != 2 or latents.shape[1] < 2:
        msg = "Expected latents with shape (N, L), where L >= 2."
        logger.error(msg)
        raise ValueError(msg)

    if latents.shape[1] == 2:
        return latents

    if len(latents) < 4:
        msg = "This UMAP projection requires at least four samples."
        logger.error(msg)
        raise ValueError(msg)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(latents) - 1),
        min_dist=0.1,
        random_state=random_state,
        n_jobs=1,
    )
    reduced = reducer.fit_transform(latents)
    return reduced


def plot_latents(
    latents: np.ndarray,  # (N, L)
    labels: np.ndarray,  # (N,)
    class_names: dict[int, str],
    title: str = "Latent space",
    save_filepath: str | Path | None = None,
    show_plot: bool = False,
) -> plt.Figure:
    points = project_latents(latents)

    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(labels):
        msg = "Expected points with shape (N, 2) and matching labels."
        logger.error(msg)
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")

    num_classes = len(class_names)
    if num_classes <= 20:
        cmap_name = "tab10" if num_classes <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name, num_classes)
    else:
        cmap = plt.get_cmap(lut=num_classes)

    for color_index, (label, class_name) in enumerate(class_names.items()):
        mask = labels == label
        if not mask.any():
            continue

        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            color=cmap(color_index),
            label=class_name,
            s=10,
            alpha=0.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Class")

    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        create_directory(save_filepath.parent, parents=True, exist_ok=True)
        fig.savefig(save_filepath, dpi=200, bbox_inches="tight")
        logger.info(f"Saved figure to '{save_filepath}'")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig


def plot_metric(
    metrics: dict[ModelStage, MetricsContainer],
    metric_name: str,
    save_filepath: str | Path | None = None,
    show_plot: bool = False,
) -> plt.Figure:
    colors = {
        ModelStage.TRAIN: "limegreen",
        ModelStage.VALIDATION: "royalblue",
    }

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for stage, container in metrics.items():
        epochs = []
        values = []

        for epoch in container.get_epochs():
            epoch_metrics = container.mean(epoch=epoch)
            if metric_name not in epoch_metrics:
                continue

            epochs.append(epoch + 1)
            values.append(epoch_metrics[metric_name])

        if values:
            ax.plot(
                epochs,
                values,
                label=stage.value,
                color=colors[stage],
                marker="o",
                markersize=4,
            )

    if not ax.lines:
        plt.close(fig)
        msg = f"No values recorded for metric {metric_name!r}."
        logger.error(msg)
        raise ValueError(msg)

    ax.set_title(metric_name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()

    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        create_directory(save_filepath.parent, parents=True, exist_ok=True)
        fig.savefig(save_filepath, dpi=200)
        logger.info(f"Saved figure to '{save_filepath}'")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig
