from torch import nn


def init_weights(layer: nn.Linear, init_type: str, activation="relu", gain=None):
    """
    Initialize a torch.nn.Linear layer weights.
    """
    init_type = init_type.lower()
    activation = activation.lower()

    if not gain:
        gain = nn.init.calculate_gain(activation)

    match(init_type):
        case "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight, gain=gain)
        case "xavier_normal":
            nn.init.xavier_normal_(layer.weight, gain=gain)
        case "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity=activation)
        case "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity=activation)
        case "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=gain)
        case "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        case "uniform":
            nn.init.uniform_(layer.weight, a=0.0, b=0.02)
        case "zeros":
            nn.init.zeros_(layer.weight)
        case _:
            raise ValueError(f"Unknown init_type: {init_type}")
        
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)