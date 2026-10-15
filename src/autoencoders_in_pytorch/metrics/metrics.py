from dataclasses import asdict, dataclass
from typing import Any

from autoencoders_in_pytorch.loggers import logger


@dataclass
class MetricsRecord:
    epoch: int
    step: int
    num_samples: int
    metrics: dict[str, float]


class MetricsContainer:
    def __init__(self) -> None:
        self._records: list[MetricsRecord] = []

    def add(
        self,
        epoch: int,
        step: int,
        num_samples: int,
        metrics: dict[str, float],
    ) -> None:
        if num_samples <= 0:
            msg = "Number of samples must be greater than zero."
            logger.error(msg)
            raise ValueError(msg)

        if not metrics:
            msg = "Metrics cannot be empty."
            logger.error(msg)
            raise ValueError(msg)

        self._records.append(
            MetricsRecord(
                epoch=epoch,
                step=step,
                num_samples=num_samples,
                metrics=metrics.copy(),
            )
        )

    def mean(self, epoch: int) -> dict[str, float]:
        weighted_sums: dict[str, float] = {}
        sample_counts: dict[str, int] = {}

        for record in self._records:
            if record.epoch != epoch:
                continue

            num_samples = record.num_samples

            for name, value in record.metrics.items():
                weighted_sums[name] = weighted_sums.get(name, 0.0) + value * num_samples
                sample_counts[name] = sample_counts.get(name, 0) + num_samples

        if not weighted_sums:
            msg = f"No metrics recorded for epoch {epoch}."
            logger.error(msg)
            raise ValueError(msg)

        output = {name: total / sample_counts[name] for name, total in weighted_sums.items()}
        return output

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self._records]

    def get_metric_names(self) -> list[str]:
        return sorted({name for record in self._records for name in record.metrics})

    def get_epochs(self) -> list[int]:
        return sorted({record.epoch for record in self._records})
