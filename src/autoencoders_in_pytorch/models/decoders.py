import math

from torch import Tensor, nn


class MLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        output_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_shape = output_shape
        self.output_dim = math.prod(output_shape)

        self.layers = nn.ModuleDict()

        current_dim = self.latent_dim
        for index, hidden_dim in enumerate(self.hidden_dims):
            self.layers[f"linear_{index}"] = nn.Linear(current_dim, hidden_dim)
            self.layers[f"relu_{index}"] = nn.ReLU()
            current_dim = hidden_dim

        self.layers["output"] = nn.Linear(current_dim, self.output_dim)
        self.layers["sigmoid"] = nn.Sigmoid()
        self.layers["unflatten"] = nn.Unflatten(
            dim=1,
            unflattened_size=self.output_shape,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers.values():
            x = layer(x)
        return x
