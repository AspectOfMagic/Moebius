from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SquashedGaussianActor(nn.Module):
    """Gaussian policy transformed into the environment's action bounds."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, action_low: Tensor, action_high: Tensor) -> None:
        super().__init__()
        self.backbone = mlp(observation_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2)
        self.register_buffer("action_bias", (action_high + action_low) / 2)

    def forward(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        features = self.backbone(observation)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        squashed_action = torch.tanh(pre_tanh)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        correction = torch.log(self.action_scale * (1 - squashed_action.square()) + 1e-6)
        log_prob = log_prob - correction.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = mlp(observation_dim + action_dim, 1, hidden_dim)

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((observation, action), dim=-1))

