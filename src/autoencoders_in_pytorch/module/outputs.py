from dataclasses import dataclass

import numpy as np
from torch import Tensor


@dataclass
class ModuleStepItem:
    inputs: Tensor | np.ndarray  # Tensor: (C, H, W), NumPy: (H, W, C)
    targets: Tensor | np.ndarray  # Tensor: (C, H, W), NumPy: (H, W, C)
    outputs: Tensor | np.ndarray  # Tensor: (C, H, W), NumPy: (H, W, C)
    latents: Tensor | np.ndarray  # (L,)
    label: int | None = None


@dataclass
class ModuleStepOut:
    inputs: Tensor | np.ndarray  # Tensor: (B, C, H, W), NumPy: (B, H, W, C)
    targets: Tensor | np.ndarray  # Tensor: (B, C, H, W), NumPy: (B, H, W, C)
    outputs: Tensor | np.ndarray  # Tensor: (B, C, H, W), NumPy: (B, H, W, C)
    latents: Tensor | np.ndarray  # (B, L)
    loss: Tensor | np.ndarray  # Scalar: ()
    labels: Tensor | np.ndarray | None = None  # (B,)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> ModuleStepItem:
        return ModuleStepItem(
            inputs=self.inputs[index],
            targets=self.targets[index],
            outputs=self.outputs[index],
            latents=self.latents[index],
            label=int(self.labels[index]) if self.labels is not None else None,
        )

    def to_numpy(self) -> "ModuleStepOut":
        def convert(value: Tensor | np.ndarray, image: bool = False) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            array = value.detach().cpu().numpy()
            return array.transpose(0, 2, 3, 1) if image else array

        return ModuleStepOut(
            inputs=convert(self.inputs, image=True),
            targets=convert(self.targets, image=True),
            outputs=convert(self.outputs, image=True),
            latents=convert(self.latents),
            loss=convert(self.loss),
            labels=convert(self.labels) if self.labels is not None else None,
        )
