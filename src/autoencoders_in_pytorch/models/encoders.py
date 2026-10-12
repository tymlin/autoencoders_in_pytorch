import math

from torch import Tensor, nn


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = math.prod(input_shape)
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.layers = nn.ModuleDict()
        self.layers["flatten"] = nn.Flatten(start_dim=1)

        current_dim = self.input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            self.layers[f"linear_{index}"] = nn.Linear(current_dim, hidden_dim)
            self.layers[f"relu_{index}"] = nn.ReLU()
            current_dim = hidden_dim

        self.layers["latent"] = nn.Linear(current_dim, latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.layers.values():
            x = layer(x)
        return x
