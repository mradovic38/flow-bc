import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)

        # state-independent log std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)

        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return Normal(mean, std)

    def act(self, state, deterministic=False):
        dist = self.forward(state)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        return torch.tanh(action)
        
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