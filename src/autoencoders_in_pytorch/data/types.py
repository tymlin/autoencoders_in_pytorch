from enum import StrEnum, auto


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class DatasetType(StrEnum):
    MNIST = auto()
    CIFAR10 = auto()
    FashionMNIST = auto()
