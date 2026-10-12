from enum import StrEnum, auto


class ModelStage(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
