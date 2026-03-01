from model import MLP
from nn_utils import init_weights

import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int] = [256, 256],
        create_activation=lambda: nn.ReLU,
        weight_init_type: str = "orthogonal",
        init_log_std: float = 0.0,
    ):
        super().__init__()

        # Feature extractor
        self.backbone = MLP(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            create_activation=create_activation,
            init_type=weight_init_type,
            has_output_activation=True,
        )

        # Mean head
        self.mean = nn.Linear(self.backbone.output_dim, act_dim)
        init_weights(self.mean, "orthogonal", is_output=True)

        # Std head (learned from features)
        self.log_std_head = nn.Linear(self.backbone.output_dim, act_dim)
        # Initialize log_std around init_log_std
        nn.init.constant_(self.log_std_head.bias, init_log_std)
        nn.init.orthogonal_(self.log_std_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Normal:
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = self.log_std_head(features)
        # Clamp for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def sample(self, obs: torch.Tensor, deterministic=False):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.rsample()
        return torch.tanh(action)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.forward(obs)
        eps = 1e-6
        pre_tanh_action = torch.atanh(torch.clamp(action, -1 + eps, 1 - eps))
        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1, keepdim=True)
        # Tanh correction
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + eps), dim=-1, keepdim=True)
        return log_prob
    
        
class FlowPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.model = nn.Sequential(
            nn.Linear(hidden + action_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, a_t, t, s):
        s_embed = self.state_net(s)
        x = torch.cat([a_t, t, s_embed], dim=-1)
        return self.model(x)