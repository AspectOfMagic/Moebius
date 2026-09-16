from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from moebius.networks import QNetwork, SquashedGaussianActor
from moebius.replay_buffer import Batch


class SACAgent:
    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        target_entropy: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        action_dim = int(np.prod(action_low.shape))
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.actor = SquashedGaussianActor(observation_dim, action_dim, hidden_dim, action_low_tensor, action_high_tensor).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dim).to(self.device)
        self.target_q1 = deepcopy(self.q1).eval()
        self.target_q2 = deepcopy(self.q2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.gamma = gamma
        self.tau = tau

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(observation_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def behavior_clone(self, batch: Batch) -> dict[str, float]:
        """Fit the deterministic actor action to expert actions in ``batch``."""
        observations = self._tensor(batch.observations)
        actions = self._tensor(batch.actions)
        predicted_actions, _ = self.actor(observations, deterministic=True)
        loss = F.mse_loss(predicted_actions, actions)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return {"loss/bc": loss.item()}

    def update(
        self,
        batch: Batch,
        demonstration_batch: Batch | None = None,
        bc_weight: float = 0.0,
    ) -> dict[str, float]:
        observations = self._tensor(batch.observations)
        actions = self._tensor(batch.actions)
        rewards = self._tensor(batch.rewards)
        next_observations = self._tensor(batch.next_observations)
        dones = self._tensor(batch.dones)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_observations)
            next_q = torch.minimum(self.target_q1(next_observations, next_actions), self.target_q2(next_observations, next_actions))
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_prob)

        q1_loss = F.mse_loss(self.q1(observations, actions), target_q)
        q2_loss = F.mse_loss(self.q2(observations, actions), target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_prob = self.actor(observations)
        q_value = torch.minimum(self.q1(observations, sampled_actions), self.q2(observations, sampled_actions))
        sac_actor_loss = (self.alpha.detach() * log_prob - q_value).mean()
        bc_loss = torch.zeros((), device=self.device)
        if demonstration_batch is not None and bc_weight > 0:
            demo_observations = self._tensor(demonstration_batch.observations)
            demo_actions = self._tensor(demonstration_batch.actions)
            predicted_demo_actions, _ = self.actor(demo_observations, deterministic=True)
            bc_loss = F.mse_loss(predicted_demo_actions, demo_actions)
        actor_loss = sac_actor_loss + bc_weight * bc_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._update_targets()
        return {
            "loss/actor": actor_loss.item(),
            "loss/sac_actor": sac_actor_loss.item(),
            "loss/bc": bc_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "stats/alpha": self.alpha.item(),
            "stats/q_value": q_value.mean().item(),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _update_targets(self) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)
