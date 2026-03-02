from nn_utils import init_weights

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int | None = None,
        create_activation=lambda: nn.ReLU,
        init_type: str = "orthogonal",
        output_init_gain: float = 1.0,
        activate_output: bool = False,
    ):
        super().__init__()

        assert len(hidden_sizes) > 0

        layers = []
        in_dim = input_dim

        for h in hidden_sizes:
            linear = nn.Linear(in_dim, h)
            activation = create_activation()

            init_weights(linear, init_type, activation=activation.__name__)

            layers.append(linear)
            layers.append(activation())

            in_dim = h

        if output_dim is not None:
            linear = nn.Linear(in_dim, output_dim)
            init_weights(linear, init_type, gain=output_init_gain)
            
            layers.append(linear)

            if activate_output:
                layers.append(create_activation()())

            self.output_dim = output_dim
        else:
            self.output_dim = hidden_sizes[-1]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)