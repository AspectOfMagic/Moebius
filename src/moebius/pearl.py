"""Probabilistic Embeddings for Actor-critic RL (PEARL) core components."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from moebius.networks import ContextEncoder, QNetwork, SquashedGaussianActor
from moebius.replay_buffer import Batch, TaskBatch


def context_from_batch(batch: Batch, reward_scale: float = 1.0) -> np.ndarray:
    """Build PEARL context features ``[s, a, r, s']`` from a replay batch."""
    rewards = batch.rewards
    if rewards.ndim == batch.observations.ndim - 1:
        rewards = rewards[..., None]
    return np.concatenate((batch.observations, batch.actions, reward_scale * rewards, batch.next_observations), axis=-1).astype(
        np.float32, copy=False
    )


def context_transition(
    observation: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_observation: np.ndarray,
    reward_scale: float = 1.0,
) -> np.ndarray:
    """Build one context feature vector while collecting an episode."""
    return np.concatenate(
        (
            np.asarray(observation, dtype=np.float32).reshape(-1),
            np.asarray(action, dtype=np.float32).reshape(-1),
            np.asarray([reward_scale * reward], dtype=np.float32),
            np.asarray(next_observation, dtype=np.float32).reshape(-1),
        )
    )


class PEARLAgent:
    """SAC with a latent task posterior inferred from task-specific context.

    The critic loss updates both critics and the context encoder.  The actor is
    conditioned on the inferred latent but does not backpropagate into the
    encoder, matching PEARL's separation between task inference and policy
    optimization.
    """

    def __init__(
        self,
        observation_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        latent_dim: int = 5,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        encoder_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        kl_weight: float = 0.1,
        target_entropy: float | None = None,
        reward_scale: float = 0.1,
        fixed_alpha: float | None = 0.1,
        gradient_clip_norm: float = 10.0,
        min_posterior_std: float = 0.05,
        device: str = "cpu",
    ) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive for PEARL.")
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.context_dim = 2 * observation_dim + int(np.prod(action_low.shape)) + 1
        action_dim = int(np.prod(action_low.shape))
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive.")
        if fixed_alpha is not None and fixed_alpha <= 0:
            raise ValueError("fixed_alpha must be positive when supplied.")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        self.encoder = ContextEncoder(
            self.context_dim, latent_dim, hidden_dim, min_posterior_std=min_posterior_std
        ).to(self.device)
        self.actor = SquashedGaussianActor(
            observation_dim, action_dim, hidden_dim, action_low_tensor, action_high_tensor, latent_dim
        ).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dim, latent_dim).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dim, latent_dim).to(self.device)
        self.target_q1 = deepcopy(self.q1).eval()
        self.target_q2 = deepcopy(self.q2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        initial_log_alpha = 0.0 if fixed_alpha is None else math.log(fixed_alpha)
        self.log_alpha = torch.tensor(
            [initial_log_alpha], device=self.device, requires_grad=fixed_alpha is None
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=alpha_lr) if fixed_alpha is None else None
        )
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.gamma = gamma
        self.tau = tau
        self.kl_weight = kl_weight
        self.reward_scale = reward_scale
        self.gradient_clip_norm = gradient_clip_norm
        self.bc_log_std_target = -2.0
        self.bc_log_std_weight = 0.01

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def infer_latent(self, context: np.ndarray | None, deterministic: bool = False) -> np.ndarray:
        """Infer one task embedding. ``None``/empty context returns the prior."""
        if context is None or len(context) == 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        context_tensor = self._tensor(np.asarray(context, dtype=np.float32)).reshape(1, -1, self.context_dim)
        mean, log_variance = self.encoder(context_tensor)
        latent = mean if deterministic else mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return latent.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        context: np.ndarray | None = None,
        deterministic: bool = False,
        latent: np.ndarray | None = None,
    ) -> np.ndarray:
        observation_tensor = self._tensor(observation).reshape(1, -1)
        selected_latent = self.infer_latent(context, deterministic=deterministic) if latent is None else latent
        latent_tensor = self._tensor(selected_latent).reshape(1, -1)
        action, _ = self.actor(observation_tensor, latent_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def behavior_clone(self, task_batch: TaskBatch, prior_context: bool = False) -> dict[str, float]:
        """Clone expert actions under either a context prefix or the prior.

        ``prior_context`` explicitly trains the empty-context policy used for
        the first action after a reset. Otherwise, the latent comes from the
        chronological task context supplied by the caller.
        """
        observations = self._flatten(task_batch.rl.observations)
        actions = self._flatten(task_batch.rl.actions)
        _, transitions = task_batch.rl.observations.shape[:2]
        if prior_context:
            latents = torch.zeros((len(task_batch.task_ids) * transitions, self.latent_dim), device=self.device)
        else:
            context = self._tensor(context_from_batch(task_batch.context, self.reward_scale))
            posterior_mean, posterior_log_variance = self.encoder(context)
            latents = self._expand_latents(
                self._sample_latents(posterior_mean, posterior_log_variance), transitions
            )
        loss, action_loss, spread_loss = self._behavior_cloning_loss(observations, actions, latents)
        self.actor_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip_norm)
        self.actor_optimizer.step()
        self.encoder_optimizer.step()
        return {
            "loss/bc": loss.item(),
            "loss/bc_action": action_loss.item(),
            "loss/bc_spread": spread_loss.item(),
        }

    def update_critic(self, task_batch: TaskBatch) -> dict[str, float]:
        """Update critics and task inference without changing the actor.

        This is also used for demonstration-only warm-up so online collection
        does not begin with random value functions.
        """
        context = self._tensor(context_from_batch(task_batch.context, self.reward_scale))
        observations = self._flatten(task_batch.rl.observations)
        actions = self._flatten(task_batch.rl.actions)
        rewards = self.reward_scale * self._flatten(task_batch.rl.rewards)
        next_observations = self._flatten(task_batch.rl.next_observations)
        dones = self._flatten(task_batch.rl.dones)
        tasks, transitions = task_batch.rl.observations.shape[:2]

        posterior_mean, posterior_log_variance = self.encoder(context)
        latents = self._sample_latents(posterior_mean, posterior_log_variance)
        expanded_latents = self._expand_latents(latents, transitions)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_observations, expanded_latents)
            next_q = torch.minimum(
                self.target_q1(next_observations, next_actions, expanded_latents),
                self.target_q2(next_observations, next_actions, expanded_latents),
            )
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_prob)

        q1 = self.q1(observations, actions, expanded_latents)
        q2 = self.q2(observations, actions, expanded_latents)
        q1_loss = F.smooth_l1_loss(q1, target_q)
        q2_loss = F.smooth_l1_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        kl_loss = self._kl_to_standard_normal(posterior_mean, posterior_log_variance)
        inference_loss = critic_loss + self.kl_weight * kl_loss
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        inference_loss.backward()
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.gradient_clip_norm
        )
        encoder_gradient_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip_norm)
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        self._update_targets()
        return {
            "loss/critic": critic_loss.item(),
            "loss/kl": kl_loss.item(),
            "stats/critic_gradient_norm": float(critic_gradient_norm),
            "stats/encoder_gradient_norm": float(encoder_gradient_norm),
            "stats/target_q": target_q.mean().item(),
            "stats/z_mean": posterior_mean.mean().item(),
            "stats/z_std": torch.exp(0.5 * posterior_log_variance).mean().item(),
            "stats/meta_batch_tasks": float(tasks),
        }

    def update(
        self,
        task_batch: TaskBatch,
        demonstration_batch: TaskBatch | None = None,
        bc_weight: float = 0.0,
    ) -> dict[str, float]:
        critic_metrics = self.update_critic(task_batch)
        context = self._tensor(context_from_batch(task_batch.context, self.reward_scale))
        observations = self._flatten(task_batch.rl.observations)
        tasks, transitions = task_batch.rl.observations.shape[:2]

        # Recompute only to obtain a current, detached task posterior for SAC's actor update.
        with torch.no_grad():
            actor_mean, actor_log_variance = self.encoder(context)
            actor_task_latents = self._sample_latents(actor_mean, actor_log_variance)
            actor_latents = self._expand_latents(actor_task_latents, transitions)
        sampled_actions, log_prob = self.actor(observations, actor_latents)
        q_value = torch.minimum(
            self.q1(observations, sampled_actions, actor_latents), self.q2(observations, sampled_actions, actor_latents)
        )
        q_scale = q_value.detach().abs().mean().clamp_min(1.0)
        sac_actor_loss = (self.alpha.detach() * log_prob - q_value / q_scale).mean()
        bc_loss = torch.zeros((), device=self.device)
        if demonstration_batch is not None and bc_weight > 0:
            demo_observations = self._flatten(demonstration_batch.rl.observations)
            demo_actions = self._flatten(demonstration_batch.rl.actions)
            _, demo_transitions = demonstration_batch.rl.observations.shape[:2]
            # Deployment conditions the actor on the learner's current
            # context. Clone expert actions under that same latent instead of
            # a separate, clean expert-only context.
            with torch.no_grad():
                demo_latents = self._expand_latents(actor_task_latents, demo_transitions)
            bc_loss, _, _ = self._behavior_cloning_loss(demo_observations, demo_actions, demo_latents)
        actor_loss = sac_actor_loss + bc_weight * bc_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_gradient_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
        self.actor_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return {
            **critic_metrics,
            "loss/actor": actor_loss.item(),
            "loss/sac_actor": sac_actor_loss.item(),
            "loss/bc": bc_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "stats/alpha": self.alpha.item(),
            "stats/q_value": q_value.mean().item(),
            "stats/q_scale": q_scale.item(),
            "stats/actor_gradient_norm": float(actor_gradient_norm),
            "stats/meta_batch_tasks": float(tasks),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _behavior_cloning_loss(
        self, observations: Tensor, actions: Tensor, latents: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        predicted_actions, log_std = self.actor.deterministic_action_and_log_std(observations, latents)
        action_loss = F.mse_loss(predicted_actions, actions)
        spread_loss = F.mse_loss(log_std, torch.full_like(log_std, self.bc_log_std_target))
        return action_loss + self.bc_log_std_weight * spread_loss, action_loss, spread_loss

    @staticmethod
    def _sample_latents(mean: Tensor, log_variance: Tensor) -> Tensor:
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    @staticmethod
    def _kl_to_standard_normal(mean: Tensor, log_variance: Tensor) -> Tensor:
        return (-0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=-1)).mean()

    def _flatten(self, values: np.ndarray) -> Tensor:
        return self._tensor(values).reshape(-1, values.shape[-1])

    @staticmethod
    def _expand_latents(latents: Tensor, transitions_per_task: int) -> Tensor:
        return latents[:, None, :].expand(-1, transitions_per_task, -1).reshape(-1, latents.shape[-1])

    @torch.no_grad()
    def _update_targets(self) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.tau)
