from enum import StrEnum, auto


class PerceptualMetrics(StrEnum):
    LPIPS = auto()
    DISTS = auto()
    BOTH = auto()
