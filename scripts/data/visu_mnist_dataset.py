import matplotlib.pyplot as plt

from autoencoders_in_pytorch.data import DatasetSplit, MNISTDataset
from autoencoders_in_pytorch.loggers import logger

SPLIT: DatasetSplit = "train"
SHOW_PLOT: bool = True
NUM_IMAGES: int = 10

dataset = MNISTDataset(split=SPLIT)
dataset_len = len(dataset)
logger.info(f"Length of `{dataset.name}` dataset: {dataset_len}")
dataset_class_names = dataset.get_class_names()
logger.info(f"Dataset `{dataset.name}` class names: {dataset_class_names}")
image, target = dataset[0]
logger.info(f"Image shape: {image.shape} (type: {type(image)}), Label: {target} (type: {type(target)})")
indices = list(range(NUM_IMAGES))
fig = dataset.plot_images(indices, show_plot=SHOW_PLOT)
