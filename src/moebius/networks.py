from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal


def mlp(input_dim: int, output_dim: int, hidden_dim: int, layer_norm: bool = False) -> nn.Sequential:
    """Two hidden ReLU layers, optionally with LayerNorm before each ReLU."""
    layers: list[nn.Module] = []
    for layer_input in (input_dim, hidden_dim):
        layers.append(nn.Linear(layer_input, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Bounded Gaussian policy, optionally conditioned on a task latent."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_low: Tensor,
        action_high: Tensor,
        latent_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = mlp(observation_dim + latent_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_scale", (action_high - action_low) / 2)
        self.register_buffer("action_bias", (action_high + action_low) / 2)

    def distribution_parameters(self, observation: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return the pre-tanh Gaussian parameters used by the policy."""
        inputs = observation if latent is None else torch.cat((observation, latent), dim=-1)
        features = self.backbone(inputs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        return mean, log_std

    def deterministic_action_and_log_std(
        self, observation: Tensor, latent: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Expose policy spread so behavior cloning can constrain exploration."""
        mean, log_std = self.distribution_parameters(observation, latent)
        return torch.tanh(mean) * self.action_scale + self.action_bias, log_std

    def forward(
        self, observation: Tensor, latent: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.distribution_parameters(observation, latent)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        squashed_action = torch.tanh(pre_tanh)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
        correction = torch.log(self.action_scale * (1 - squashed_action.square()) + 1e-6)
        log_prob = log_prob - correction.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    """State-action value function, optionally conditioned on a task latent."""

    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dim: int, latent_dim: int = 0, layer_norm: bool = False
    ) -> None:
        super().__init__()
        self.network = mlp(observation_dim + action_dim + latent_dim, 1, hidden_dim, layer_norm)

    def forward(self, observation: Tensor, action: Tensor, latent: Tensor | None = None) -> Tensor:
        inputs = (observation, action) if latent is None else (observation, action, latent)
        return self.network(torch.cat(inputs, dim=-1))


class TransitionDecoder(nn.Module):
    """Predict transition outcomes from state, action, and task latent."""

    def __init__(self, observation_dim: int, action_dim: int, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = mlp(observation_dim + action_dim + latent_dim, output_dim, hidden_dim)

    def forward(self, observation: Tensor, action: Tensor, latent: Tensor) -> Tensor:
        return self.network(torch.cat((observation, action, latent), dim=-1))


class ContextEncoder(nn.Module):
    """Combine unordered transition factors into a Gaussian task posterior.

    ``softplus`` keeps factor-variance gradients alive; an additive posterior
    floor prevents overconfidence.
    """

    _MIN_FACTOR_VARIANCE = 1e-4

    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int, min_posterior_std: float = 0.05) -> None:
        super().__init__()
        if not 0 < min_posterior_std <= 1:
            raise ValueError("min_posterior_std must be in (0, 1].")
        self.latent_dim = latent_dim
        self.min_posterior_variance = min_posterior_std**2
        self.network = mlp(context_dim, 2 * latent_dim, hidden_dim)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior parameters for ``[tasks, context, features]``."""
        if context.ndim != 3:
            raise ValueError("Context must have shape [num_tasks, context_size, context_dim].")
        parameters = self.network(context)
        means, raw_variances = parameters.chunk(2, dim=-1)
        variances = F.softplus(raw_variances) + self._MIN_FACTOR_VARIANCE
        precisions = variances.reciprocal()

        # Include the unit-normal prior as another Gaussian factor.
        posterior_precision = 1 + precisions.sum(dim=1)
        posterior_variance = posterior_precision.reciprocal()
        posterior_mean = posterior_variance * (precisions * means).sum(dim=1)
        posterior_variance = posterior_variance + self.min_posterior_variance
        return posterior_mean, posterior_variance.log()
