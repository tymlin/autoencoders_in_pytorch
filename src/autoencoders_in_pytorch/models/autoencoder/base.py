from abc import ABC, abstractmethod
from typing import Any

import torch
import torchinfo
from torch import Tensor, nn

from autoencoders_in_pytorch.models.outputs import (
    AutoencoderOutput,
    DecoderOutput,
    EncoderOutput,
    VariationalAutoencoderOutput,
    VariationalEncoderOutput,
)


class BaseModel(nn.Module, ABC):
    name: str
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def encode(self, inputs: Any) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, latents: Any) -> Any:
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        raise NotImplementedError()

    def summary(self, example_input_shape: tuple[int, int, int, int], depth: int = 4) -> str:
        col_names = ["input_size", "output_size", "num_params", "params_percent"]
        model_summary = str(
            torchinfo.summary(
                self,
                input_size=example_input_shape,
                depth=depth,
                col_names=col_names,
                verbose=0,
                device=self.device,
            )
        )
        return model_summary


class BaseAutoencoder(BaseModel, ABC):
    def encode(self, inputs: Tensor) -> EncoderOutput:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> DecoderOutput:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> AutoencoderOutput:
        encoder_output = self.encode(inputs)
        decoder_output = self.decode(encoder_output.latents)
        output = AutoencoderOutput(
            latents=encoder_output.latents,
            outputs=decoder_output.outputs,
        )
        return output


class BaseVariationalAutoencoder(BaseModel, ABC):
    def encode(self, inputs: Tensor) -> VariationalEncoderOutput:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> DecoderOutput:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> VariationalAutoencoderOutput:
        encoder_output = self.encode(inputs)
        decoder_output = self.decode(encoder_output.latents)
        output = VariationalAutoencoderOutput(
            outputs=decoder_output.outputs,
            latents=encoder_output.latents,
            mu=encoder_output.mu,
            logvar=encoder_output.logvar,
        )
        return output
