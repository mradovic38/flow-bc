from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
from torchdiffeq import odeint

from model import MLP
from nn_utils import init_weights, safe_atanh, EMA, TimeEmbedder


class Policy(nn.Module, ABC):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @abstractmethod
    def forward(self, obs: torch.Tensor):
        """Return the policy distribution or internal representation."""
        pass

    @abstractmethod
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return an action in [-1, 1]."""
        pass

    @abstractmethod
    def loss(self, states, actions):
        """Return the policy loss."""
        pass
        

class GaussianPolicy(Policy):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int] = [256, 256],
        create_activation=lambda: nn.ReLU,
        weight_init_type: str = "orthogonal",
        init_log_std: float = 0.0,
    ):
        super().__init__(obs_dim, act_dim)

        # Feature extractor
        self.backbone = MLP(
            input_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            create_activation=create_activation,
            init_type=weight_init_type,
            activate_output=True,
        )

        # Mean head
        self.mean = nn.Linear(self.backbone.output_dim, act_dim)
        init_weights(self.mean, "orthogonal", activation="linear", gain=0.01)
    
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
        pre_tanh_action = safe_atanh(action)
        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1, keepdim=True)
        # Tanh correction
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + eps), dim=-1, keepdim=True)
        return log_prob
    
    def loss(self, states, actions):
        return -self.log_prob(states, actions).mean()

    
class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        backbone_hidden_sizes: list[int] = [256, 256],
        velocity_hidden_sizes: list[int] = [256, 256],
        time_embedder_hidden_size: int = 256,
        time_freq_dim: int = 64,
        ode_steps: int = 10,
        ode_method: str = "euler",
        ema_decay=0.9999,
        lognormal_mu=-1.2,
        lognormal_sigma=1.2,
    ):
        super().__init__()

        self.act_dim = act_dim

        self.ode_steps = ode_steps
        self.ode_method = ode_method

        self.lognormal_mu = lognormal_mu
        self.lognormal_sigma = lognormal_sigma

        self.backbone = MLP(
            input_dim=obs_dim,
            hidden_sizes=backbone_hidden_sizes,
            activate_output=True,
        )

        feat_dim = self.backbone.output_dim

        self.time_embed = TimeEmbedder(time_embedder_hidden_size, time_freq_dim)

        self.velocity = MLP(
            input_dim=feat_dim + act_dim + time_embedder_hidden_size,
            hidden_sizes=velocity_hidden_sizes,
            output_dim=act_dim,
        )

        self.ema = EMA(self, decay=ema_decay)

    def velocity_field(self, z, h, t):
        t_emb = self.time_embed(t)
        inp = torch.cat([z, h, t_emb], dim=-1)
        v = self.velocity(inp)
        return v
    
    def loss(self, obs, actions):
        B = obs.size(0)
        device = obs.device

        h = self.backbone(obs)
        z0 = torch.randn(B, self.act_dim, device=device)
        z1 = safe_atanh(actions)

        z = torch.randn(B, 1, device=device) * self.lognormal_sigma + self.lognormal_mu
        t = torch.exp(z)
        t = 1 / (1 + t)
        t = torch.clamp(t, 1e-4, 1.0)

        zt = (1 - t) * z0 + t * z1

        target_velocity = z1 - z0
        pred_velocity = self.velocity_field(zt, h, t)

        weight = (1 - t).squeeze(-1) ** 2
        loss = ((pred_velocity - target_velocity) ** 2).sum(dim=-1)
        loss = (weight * loss).mean()

        return loss

    @torch.no_grad()
    def sample(self, obs, deterministic=False):
        B = obs.size(0)
        device = obs.device

        h = self.backbone(obs)
        z0 = torch.zeros(B, self.act_dim, device=device) if deterministic else torch.randn(B, self.act_dim, device=device)

        def odefunc(t, z):
            t_batch = torch.full((B, 1), t, device=device)
            return self.velocity_field(z, h, t_batch)

        t_span = torch.linspace(0.0, 1.0, self.ode_steps + 1, device=device)
        z_traj = odeint(odefunc, z0, t_span, method=self.ode_method)
        z1 = z_traj[-1]

        return torch.tanh(z1)

    def forward(self, obs):
        return self.sample(obs)
