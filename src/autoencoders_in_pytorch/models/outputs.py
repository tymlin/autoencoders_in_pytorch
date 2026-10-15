from typing import NamedTuple

from torch import Tensor


class EncoderOutput(NamedTuple):
    latents: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)


class VariationalEncoderOutput(NamedTuple):
    latents: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
    mu: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
    logvar: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)


class DecoderOutput(NamedTuple):
    outputs: Tensor  # (B, C, H, W)


class AutoencoderOutput(NamedTuple):
    outputs: Tensor  # (B, C, H, W)
    latents: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)


class VariationalAutoencoderOutput(NamedTuple):
    outputs: Tensor  # (B, C, H, W)
    latents: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
    mu: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
    logvar: Tensor  # MLP: (B, L), CNN: (B, C_z, H_z, W_z)
