import math

import torch
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


def safe_atanh(x, eps=1e-6):
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.original = None

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        if self.original is not None:
            return
        self.original = {name: p.data.clone() for name, p in self.model.named_parameters() if p.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        if self.original is None:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])
        self.original = None


class TimeEmbedder(nn.Module):
    """
    Turns continuous scalar time into vector representations. 

    Based on OpenAI GLIDE sinusoidal timestep embeddings:
    https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py#L87
    """
    def __init__(self, hidden_size, time_freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.time_freq_dim = time_freq_dim

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        if t.dim() == 2:
            t = t.squeeze(-1)   # (B,1) -> (B)

        t_freq = self.timestep_embedding(t, self.time_freq_dim)
        t_emb = self.mlp(t_freq)
        return t_emb
    