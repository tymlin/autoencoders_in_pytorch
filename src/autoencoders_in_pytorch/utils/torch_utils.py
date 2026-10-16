import random

import numpy as np
import torch

from autoencoders_in_pytorch.loggers import logger
from autoencoders_in_pytorch.models import Device


def seed_everything(seed: int) -> None:
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_torch_devices_availability(as_dict: bool = False) -> dict[Device, bool] | list[Device]:
    """Checks the availability of CPU, CUDA, and MPS devices in PyTorch.

    as_dict: bool: Whether to return the availability status as a dictionary.

    :return dict: A dictionary with the availability status of each device.
    """
    available_devices = {
        Device.CPU: True,
        Device.CUDA: torch.cuda.is_available(),
        Device.MPS: torch.backends.mps.is_available(),
    }
    if not as_dict:
        available_devices = [device for device, status in available_devices.items() if status]
    return available_devices


def auto_detect_device() -> Device:
    """Select an available device, preferring CUDA, then MPS, then CPU.

    :return Device: Selected device
    """
    available_devices = check_torch_devices_availability()

    if Device.CUDA in available_devices:
        return Device.CUDA

    if Device.MPS in available_devices:
        return Device.MPS

    return Device.CPU


def get_device_and_id(device: Device, device_id: int = 0) -> tuple[str, int]:
    available_devices = check_torch_devices_availability()
    device_count = torch.cuda.device_count()

    available_devices_str = ", ".join(f"`{d}`" for d in available_devices)
    logger.info(
        f"Available devices: {available_devices_str}. "
        f"Available CUDA device count: {device_count}. "
        f"Specified device: `{device.value}` with id: `{device_id}`"
    )
    if device not in available_devices:
        msg = f"Specified device `{device.value}` is not available"
        logger.error(msg)
        raise ValueError(msg)

    if device == Device.CUDA and torch.cuda.is_available():
        if not 0 <= device_id < device_count:
            msg = f"CUDA device index {device_id} is out of range."
            logger.error(msg)
            raise ValueError(msg)

        device_id = int(device_id)
        resolved_device = f"{Device.CUDA}:{device_id}"
    elif device == Device.MPS and torch.backends.mps.is_available():
        resolved_device = device.value
        device_id = 0
    else:
        resolved_device = device.value
        device_id = 0

    return resolved_device, device_id
