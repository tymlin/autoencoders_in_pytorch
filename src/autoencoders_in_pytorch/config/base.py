from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoencoders_in_pytorch.consts import RUNS_DIRPATH
from autoencoders_in_pytorch.data import DatasetSplit, DatasetType
from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import Device


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    dataset: DatasetType = Field(
        default=DatasetType.MNIST,
        description="Dataset used for the experiment.",
    )
    train_split: Literal[DatasetSplit.TRAIN] = Field(
        default=DatasetSplit.TRAIN,
        description="Source dataset split used for training. Must be 'train'.",
    )
    val_split: Literal[DatasetSplit.VALIDATION, DatasetSplit.TEST] = Field(
        default=DatasetSplit.TEST,
        description=(
            "Source dataset split used for validation. Use 'test' for datasets without a dedicated validation split."
        ),
    )
    test_split: Literal[DatasetSplit.TEST] | None = Field(
        default=None,
        description="Source dataset split used for testing. None means no separate test dataset is created.",
    )
    image_size: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] | None = Field(
        default=None,
        description="Target image size as (height, width). None preserves the original image size.",
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        description="Number of samples per batch for training, validation and testing.",
    )
    num_workers: int = Field(
        default=8,
        ge=0,
        description="Number of subprocesses used for data loading. Zero loads data in the main process.",
    )
    persistent_workers: bool = Field(
        default=False,
        description=(
            "Whether to keep data loading workers alive between epochs. Requires `num_workers` greater than zero."
        ),
    )
    pin_memory: bool = Field(
        default=True,
        description=(
            "Whether to copy loaded tensors into pinned host memory to support faster transfers to CUDA devices."
        ),
    )
    drop_last: bool = Field(
        default=True,
        description=(
            "Whether to drop the final training batch when it contains fewer samples than batch_size. "
            "Does not affect validation or testing."
        ),
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle training samples each epoch. Does not affect validation or testing.",
    )

    @model_validator(mode="after")
    def validate_persistent_workers(self) -> Self:
        if self.persistent_workers and self.num_workers <= 0:
            msg = "Persistent workers require `num_workers` greater than zero."
            logger.error(msg)
            raise ValueError(msg)

        return self


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    lr: float = Field(
        default=0.0005,
        gt=0,
        allow_inf_nan=False,
        description="Learning rate used by the optimizer.",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Weight decay coefficient used by the optimizer.",
    )


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    max_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of training epochs.",
    )
    device: Device = Field(
        default=Device.CPU,
        description="Device type used for model execution: CPU, CUDA or MPS.",
    )
    device_id: int = Field(
        default=0,
        ge=0,
        description="CUDA device index. Ignored when using CPU or MPS.",
    )
    calculate_perceptual_metrics: bool = Field(
        default=True,
        description=(
            "Whether to calculate LPIPS and DISTS in addition to basic reconstruction metrics during training "
            "and validation. Enabling this increases computation time."
        ),
    )
    num_visu_samples: int = Field(
        default=36,
        ge=1,
        description="Maximum number of fixed validation samples used to visualize inputs and reconstructions.",
    )
    num_visu_latents: int | None = Field(
        default=None,
        ge=4,
        description=(
            "Maximum number of validation samples used to visualize latent space. None uses all validation samples. "
            "At least four samples are required for UMAP projection."
        ),
    )


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    run_dirpath: Path = Field(
        default=RUNS_DIRPATH,
        description="Directory used to store run outputs.",
    )
    run_name_suffix: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional name appended to the timestamp in the run directory name. None means only the timestamp is used."
        ),
    )
    seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
        description="Random seed used by Python, NumPy, PyTorch and training data shuffling.",
    )

    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Dataset selection, source splits and data loading settings.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer hyperparameters.",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Training duration, device, visualization and metric settings.",
    )
