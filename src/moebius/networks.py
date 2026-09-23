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
    """Gaussian policy transformed into the environment's action bounds.

    ``latent_dim`` is zero for ordinary SAC and positive for PEARL, where the
    policy receives a task latent in addition to the environment observation.
    """

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
    """State-action value function, optionally conditioned on a PEARL latent."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, latent_dim: int = 0) -> None:
        super().__init__()
        self.network = mlp(observation_dim + action_dim + latent_dim, 1, hidden_dim)

    def forward(self, observation: Tensor, action: Tensor, latent: Tensor | None = None) -> Tensor:
        inputs = (observation, action) if latent is None else (observation, action, latent)
        return self.network(torch.cat(inputs, dim=-1))


class ContextEncoder(nn.Module):
    """Infer a diagonal Gaussian task posterior from unordered transitions.

    Each transition emits a Gaussian factor.  The factors are combined with a
    unit-normal prior using a product of diagonal Gaussians, which is invariant
    to the order of the context transitions and gives the prior for empty
    context.
    """

    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int, min_posterior_std: float = 0.05) -> None:
        super().__init__()
        if not 0 < min_posterior_std <= 1:
            raise ValueError("min_posterior_std must be in (0, 1].")
        self.latent_dim = latent_dim
        self.min_posterior_variance = min_posterior_std**2
        self.network = mlp(context_dim, 2 * latent_dim, hidden_dim)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior mean and log variance for ``[tasks, context, features]``."""
        if context.ndim != 3:
            raise ValueError("Context must have shape [num_tasks, context_size, context_dim].")
        parameters = self.network(context)
        means, log_variances = parameters.chunk(2, dim=-1)
        log_variances = log_variances.clamp(-10, 5)
        precisions = torch.exp(-log_variances)

        # Include p(z) = N(0, I) as one additional Gaussian factor.
        posterior_precision = 1 + precisions.sum(dim=1)
        posterior_variance = posterior_precision.reciprocal()
        posterior_mean = posterior_variance * (precisions * means).sum(dim=1)
        posterior_variance = posterior_variance.clamp_min(self.min_posterior_variance)
        return posterior_mean, posterior_variance.log()
