from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from model import MLP
from nn_utils import init_weights


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


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        half = dim // 2
        freqs = 2 ** torch.arange(half).float() * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor):
        # t: (B, 1)
        angles = t * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    
class _FlowODEFunc(nn.Module):
    def __init__(self, policy, h):
        super().__init__()
        self.policy = policy
        self.h = h  # precomputed state features

    def forward(self, t, x):
        B = x.shape[0]
        t = torch.full((B, 1), t, device=x.device)
        return self.policy.velocity_field(x, self.h, t)
    

class FlowMatchingPolicy(Policy):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(256, 256),
        time_dim: int = 32,
        ode_method: str = "euler",
        ode_steps: int = 20,
    ):
        super().__init__(obs_dim, act_dim)

        self.ode_method = ode_method
        self.ode_steps = ode_steps

        # State encoder
        self.backbone = MLP(
            input_dim=obs_dim,
            hidden_sizes=list(hidden_sizes),
            activate_output=True,
        )
        feat_dim = self.backbone.output_dim

        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)

        # Velocity field
        self.velocity = MLP(
            input_dim=feat_dim + act_dim + time_dim,
            hidden_sizes=[256, 256],
            output_dim=act_dim,
            output_init_gain=0.01
        )
        
    def velocity_field(self, x, h, t):
        t_emb = self.time_embed(t)
        inp = torch.cat([x, h, t_emb], dim=-1)
        return self.velocity(inp)

    @torch.no_grad()
    def sample(self, obs, deterministic=False):
        B = obs.shape[0]
        device = obs.device

        h = self.backbone(obs)

        if deterministic:
            x0 = torch.zeros(B, self.act_dim, device=device)
        else:
            x0 = torch.randn(B, self.act_dim, device=device)

        ode_func = _FlowODEFunc(self, h)

        t = torch.linspace(0, 1, self.ode_steps + 1, device=device)

        xt = odeint(
            ode_func,
            x0,
            t,
            method=self.ode_method,
            options={"step_size": 1.0 / self.ode_steps},
        )

        x1 = xt[-1]

        return torch.tanh(x1)
    
    def forward(self, obs: torch.Tensor):
        return self.sample(obs, deterministic=False)

    def loss(self, obs, actions):
        """
        obs:     (B, obs_dim)
        actions: (B, act_dim)  in [-1, 1]
        """
        B = obs.shape[0]
        device = obs.device

        h = self.backbone(obs)

        t = torch.rand(B, 1, device=device)
        eps = 1e-5
        t = t * (1 - eps)

        x0 = torch.randn_like(actions)

        xt = (1 - t) * x0 + t * actions

        target_v = (actions - xt) / (1 - t)

        pred_v = self.velocity_field(xt, h, t)

        return F.mse_loss(pred_v, target_v)