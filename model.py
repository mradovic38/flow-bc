from nn_utils import init_weights

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        create_activation=lambda: nn.ReLU,
        init_type: str = "orthogonal",
        has_output_activation: bool = True,
    ):
        super().__init__()

        assert len(hidden_sizes) > 0

        layers = []
        in_dim = input_dim

        for i, h in enumerate(hidden_sizes):
            linear = nn.Linear(in_dim, h)
            activation = create_activation()

            init_weights(linear, init_type, activation=activation.__name__)

            layers.append(linear)

            is_last = i == len(hidden_sizes) - 1
            if has_output_activation or (not is_last):
                layers.append(activation())

            in_dim = h

        self.model = nn.Sequential(*layers)
        self.output_dim = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)