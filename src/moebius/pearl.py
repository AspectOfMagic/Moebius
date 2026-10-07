"""Probabilistic Embeddings for Actor-critic RL (PEARL) core components."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from moebius.context import ContextBuilder
from moebius.networks import ContextEncoder, QNetwork, SquashedGaussianActor, TransitionDecoder
from moebius.replay_buffer import Batch, TaskBatch

# Observation entries the auxiliary decoder predicts the one-step change of:
# gripper opening (3), object position (4:7), object quaternion (7:11) and,
# with contact observations, the forces on the object (11:18; always zero,
# and so trivially predicted, without them).
DECODED_STATE = slice(3, 18)


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
        kl_weight: float = 0.05,
        kl_free_bits: float = 0.1,
        target_entropy: float | None = None,
        reward_scale: float = 0.1,
        fixed_alpha: float | None = None,
        initial_alpha: float = 0.02,
        gradient_clip_norm: float = 10.0,
        min_posterior_std: float = 0.05,
        max_reward: float | None = 10.0,
        critic_layer_norm: bool = True,
        decoder_weight: float = 0.0,
        decoder_lr: float = 3e-4,
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
        if initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive.")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if kl_free_bits < 0:
            raise ValueError("kl_free_bits must be non-negative.")
        self.encoder = ContextEncoder(
            self.context_dim, latent_dim, hidden_dim, min_posterior_std=min_posterior_std
        ).to(self.device)
        self.actor = SquashedGaussianActor(
            observation_dim, action_dim, hidden_dim, action_low_tensor, action_high_tensor, latent_dim
        ).to(self.device)
        self.q1 = QNetwork(observation_dim, action_dim, hidden_dim, latent_dim, critic_layer_norm).to(self.device)
        self.q2 = QNetwork(observation_dim, action_dim, hidden_dim, latent_dim, critic_layer_norm).to(self.device)
        self.target_q1 = deepcopy(self.q1).eval()
        self.target_q2 = deepcopy(self.q2).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        if decoder_weight < 0:
            raise ValueError("decoder_weight must be non-negative.")
        # Auxiliary transition decoder (0 disables it). It predicts the change
        # in gripper opening, object position and object orientation, plus
        # the scaled reward, from (s, a, z).
        self.decoder_weight = decoder_weight
        self.decoder = None
        self.decoder_optimizer = None
        if decoder_weight > 0:
            output_dim = len(range(observation_dim)[DECODED_STATE]) + 1
            self.decoder = TransitionDecoder(observation_dim, action_dim, latent_dim, hidden_dim, output_dim).to(
                self.device
            )
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr)
        # alpha is expressed in scaled-reward units (the same units as Q). With
        # automatic tuning, start small so the behavior-cloned policy is not
        # immediately pushed toward uniform random actions.
        initial_log_alpha = math.log(initial_alpha if fixed_alpha is None else fixed_alpha)
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
        # Nats per latent dimension that the KL term does not penalize, so the
        # posterior is never pushed all the way back onto the prior.
        self.kl_free_bits = kl_free_bits
        self.reward_scale = reward_scale
        self.gradient_clip_norm = gradient_clip_norm
        if max_reward is not None and max_reward <= 0:
            raise ValueError("max_reward must be positive when supplied.")
        # Upper bound on one step of the soft Bellman target: the largest
        # scaled reward plus the largest possible entropy bonus. The squashed
        # policy's entropy can never exceed that of a uniform distribution
        # over the action box, sum(log(high - low)).
        self.max_scaled_reward = None if max_reward is None else reward_scale * max_reward
        self.max_policy_entropy = float(torch.log(action_high_tensor - action_low_tensor).sum())
        self.bc_log_std_target = -2.0
        self.bc_log_std_weight = 0.01

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def infer_latent(self, context: np.ndarray | None, deterministic: bool = False) -> np.ndarray:
        """Infer one task embedding. ``None``/empty context returns ``z = 0``.

        The empty-context convention matches training, where ``k = 0``
        previous episodes also uses ``z = 0``.
        """
        if context is None or len(context) == 0:
            return np.zeros(self.latent_dim, dtype=np.float32)
        context_tensor = self._tensor(np.asarray(context, dtype=np.float32)).reshape(1, -1, self.context_dim)
        mean, log_variance = self.encoder(context_tensor)
        latent = mean if deterministic else mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return latent.squeeze(0).cpu().numpy()

    def latent_for_episode(
        self,
        previous_episodes: Sequence[Batch],
        builder: ContextBuilder,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Batch | None]:
        """The single deployment rule for choosing an episode's task latent.

        The context is built from every completed episode earlier in the same
        adaptation sequence (none for the first episode, giving ``z = 0``).
        Training samples contexts with the same builder and the same notion of
        "previous episodes", so the encoder sees matching inputs.
        """
        if not previous_episodes:
            return np.zeros(self.latent_dim, dtype=np.float32), None
        context = builder.build(previous_episodes, rng)
        return self.infer_latent(context_from_batch(context, self.reward_scale), deterministic=deterministic), context

    @torch.no_grad()
    def posterior(self, context: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation, for logging adaptation."""
        if context is None or len(context) == 0:
            return np.zeros(self.latent_dim, dtype=np.float32), np.ones(self.latent_dim, dtype=np.float32)
        context_tensor = self._tensor(np.asarray(context, dtype=np.float32)).reshape(1, -1, self.context_dim)
        mean, log_variance = self.encoder(context_tensor)
        return mean.squeeze(0).cpu().numpy(), torch.exp(0.5 * log_variance).squeeze(0).cpu().numpy()

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

    def behavior_clone(self, task_batch: TaskBatch) -> dict[str, float]:
        """Task-agnostic behavior cloning of the actor under ``z = 0``.

        As in the original PEARL, the encoder is trained only by the critic
        loss and KL term. Cloning through the encoder made ``z`` encode
        whatever predicted expert actions, and that meaning then shifted once
        critic training took over. ``task_batch.context`` is ignored.
        """
        observations = self._flatten(task_batch.rl.observations)
        actions = self._flatten(task_batch.rl.actions)
        latents = torch.zeros((len(observations), self.latent_dim), device=self.device)
        loss, action_loss, spread_loss = self._behavior_cloning_loss(observations, actions, latents)
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
        self.actor_optimizer.step()
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

        empty_context = context.shape[1] == 0
        if empty_context:
            # First episode of an adaptation sequence: act under z = 0 exactly
            # as rollouts do. No encoder or KL gradient in this case.
            posterior_mean = torch.zeros((tasks, self.latent_dim), device=self.device)
            posterior_log_variance = torch.zeros_like(posterior_mean)
            latents = posterior_mean
        else:
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
            clipped_fraction = torch.zeros((), device=self.device)
            q_upper_bound = float("inf")
            if self.max_scaled_reward is not None:
                # No return can exceed this, so bootstrapped values above it
                # are pure over-estimation; clipping stops them compounding.
                q_upper_bound = (
                    self.max_scaled_reward + self.alpha.item() * self.max_policy_entropy
                ) / (1 - self.gamma)
                clipped_fraction = (target_q > q_upper_bound).float().mean()
                target_q = target_q.clamp(max=q_upper_bound)

        q1 = self.q1(observations, actions, expanded_latents)
        q2 = self.q2(observations, actions, expanded_latents)
        q1_loss = F.smooth_l1_loss(q1, target_q)
        q2_loss = F.smooth_l1_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        kl_per_dimension = self._kl_per_dimension(posterior_mean, posterior_log_variance)
        kl_loss = kl_per_dimension.sum()
        penalized_kl = kl_per_dimension.clamp_min(self.kl_free_bits).sum()
        if empty_context:
            penalized_kl = torch.zeros((), device=self.device)
        inference_loss = critic_loss + self.kl_weight * penalized_kl
        decoder_metrics: dict[str, float] = {}
        use_decoder = self.decoder is not None and not empty_context
        if use_decoder:
            targets = self._decoder_targets(observations, rewards, next_observations)
            decoder_loss = self._decoder_loss(observations, actions, expanded_latents, targets)
            inference_loss = inference_loss + self.decoder_weight * decoder_loss
            with torch.no_grad():
                decoder_metrics["loss/decoder"] = decoder_loss.item()
                if tasks > 1:
                    # Loss with every task given another task's latent, minus
                    # the true loss. Positive means z carries information that
                    # predicts this task's transitions.
                    wrong_latents = self._expand_latents(torch.roll(latents, 1, dims=0), transitions)
                    wrong_loss = self._decoder_loss(observations, actions, wrong_latents, targets)
                    decoder_metrics["stats/decoder_z_gain"] = (wrong_loss - decoder_loss).item()
            self.decoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        inference_loss.backward()
        if use_decoder:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.gradient_clip_norm)
            self.decoder_optimizer.step()
        critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.gradient_clip_norm
        )
        encoder_gradient_norm = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.gradient_clip_norm)
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        self._update_targets()
        metrics = {
            "loss/critic": critic_loss.item(),
            "stats/critic_gradient_norm": float(critic_gradient_norm),
            "stats/target_q": target_q.mean().item(),
            "stats/q_max": torch.maximum(q1, q2).max().item(),
            "stats/target_q_clipped_fraction": clipped_fraction.item(),
            "stats/meta_batch_tasks": float(tasks),
            "stats/context_length": float(context.shape[1]),
        }
        if self.max_scaled_reward is not None:
            metrics["stats/q_upper_bound"] = q_upper_bound
        metrics.update(decoder_metrics)
        if not empty_context:
            # Posterior statistics are only meaningful with context; logging
            # the k = 0 prior would mask whether inference is working.
            metrics.update(
                {
                    "loss/kl": kl_loss.item(),
                    "loss/kl_penalized": penalized_kl.item(),
                    "stats/encoder_gradient_norm": float(encoder_gradient_norm),
                    "stats/z_mean": posterior_mean.mean().item(),
                    "stats/z_std": torch.exp(0.5 * posterior_log_variance).mean().item(),
                    # Spread of posterior means across tasks: near zero means
                    # every task maps to the same latent.
                    "stats/z_mean_task_std": posterior_mean.std(dim=0).mean().item() if tasks > 1 else 0.0,
                }
            )
        return metrics

    def update(
        self,
        task_batch: TaskBatch,
        demonstration_batch: TaskBatch | None = None,
        bc_weight: float = 0.0,
        clone_grip_magnitude: bool = True,
    ) -> dict[str, float]:
        critic_metrics = self.update_critic(task_batch)
        context = self._tensor(context_from_batch(task_batch.context, self.reward_scale))
        observations = self._flatten(task_batch.rl.observations)
        tasks, transitions = task_batch.rl.observations.shape[:2]

        # Recompute only to obtain a current, detached task posterior for SAC's actor update.
        with torch.no_grad():
            if context.shape[1] == 0:
                actor_task_latents = torch.zeros((tasks, self.latent_dim), device=self.device)
            else:
                actor_mean, actor_log_variance = self.encoder(context)
                actor_task_latents = self._sample_latents(actor_mean, actor_log_variance)
            actor_latents = self._expand_latents(actor_task_latents, transitions)
        sampled_actions, log_prob = self.actor(observations, actor_latents)
        q_value = torch.minimum(
            self.q1(observations, sampled_actions, actor_latents), self.q2(observations, sampled_actions, actor_latents)
        )
        sac_actor_loss, q_scale = self._normalized_sac_actor_loss(self.alpha.detach(), log_prob, q_value)
        with torch.no_grad():
            _, policy_log_std = self.actor.distribution_parameters(observations, actor_latents)
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
            bc_loss, _, _ = self._behavior_cloning_loss(
                demo_observations,
                demo_actions,
                demo_latents,
                clone_grip_magnitude=clone_grip_magnitude,
            )
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
            # Squashed-policy entropy estimate; -action_dim * log(2) would be
            # uniform random actions on [-1, 1]^action_dim.
            "stats/policy_entropy": -log_prob.mean().item(),
            "stats/policy_log_std": policy_log_std.mean().item(),
            "stats/actor_gradient_norm": float(actor_gradient_norm),
            "stats/meta_batch_tasks": float(tasks),
        }

    def _decoder_targets(self, observations: Tensor, scaled_rewards: Tensor, next_observations: Tensor) -> Tensor:
        state_change = (next_observations - observations)[:, DECODED_STATE]
        return torch.cat((state_change, scaled_rewards), dim=-1)

    def _decoder_loss(self, observations: Tensor, actions: Tensor, latents: Tensor, targets: Tensor) -> Tensor:
        """Mean squared error with each target standardized over the batch.

        Object displacements are millimetres while rewards are of order one;
        dividing by each target's (detached) batch std weights them equally.
        """
        scale = targets.std(dim=0, keepdim=True).clamp_min(1e-3).detach()
        prediction = self.decoder(observations, actions, latents)
        return (((prediction - targets) / scale) ** 2).mean()

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.decoder is not None:
            state["decoder"] = self.decoder.state_dict()
        return state

    def _tensor(self, array: np.ndarray) -> Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _behavior_cloning_loss(
        self,
        observations: Tensor,
        actions: Tensor,
        latents: Tensor,
        clone_grip_magnitude: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        predicted_actions, log_std = self.actor.deterministic_action_and_log_std(observations, latents)
        action_loss = self._behavior_cloning_action_loss(predicted_actions, actions, clone_grip_magnitude)
        spread_loss = F.mse_loss(log_std, torch.full_like(log_std, self.bc_log_std_target))
        return action_loss + self.bc_log_std_weight * spread_loss, action_loss, spread_loss

    @staticmethod
    def _behavior_cloning_action_loss(
        predicted_actions: Tensor, expert_actions: Tensor, clone_grip_magnitude: bool = True
    ) -> Tensor:
        """Clone all actions, or only XYZ and the gripper's open/close direction."""
        if clone_grip_magnitude or expert_actions.shape[-1] < 4:
            return F.mse_loss(predicted_actions, expert_actions)

        movement_error = (predicted_actions[..., :-1] - expert_actions[..., :-1]).square().sum(dim=-1)
        direction = expert_actions[..., -1].sign()
        # A small margin teaches open versus close without pulling the policy
        # toward the expert's force magnitude. Zero expert commands are ignored.
        direction_error = F.relu(0.1 - direction * predicted_actions[..., -1]).square()
        direction_error = torch.where(direction == 0, torch.zeros_like(direction_error), direction_error)
        return (movement_error + direction_error).mean() / expert_actions.shape[-1]

    @staticmethod
    def _normalized_sac_actor_loss(alpha: Tensor, log_prob: Tensor, q_value: Tensor) -> tuple[Tensor, Tensor]:
        """SAC actor objective divided by the batch Q magnitude.

        The division keeps the behavior-cloning term comparable in size to the
        RL term. It must scale the entropy term too: normalizing only Q makes
        the effective temperature ``alpha * q_scale``, which drove earlier runs
        to near-uniform random actions.
        """
        q_scale = q_value.detach().abs().mean().clamp_min(1.0)
        return ((alpha * log_prob - q_value) / q_scale).mean(), q_scale

    @staticmethod
    def _sample_latents(mean: Tensor, log_variance: Tensor) -> Tensor:
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    @staticmethod
    def _kl_per_dimension(mean: Tensor, log_variance: Tensor) -> Tensor:
        """KL(q || N(0, I)) per latent dimension, averaged over tasks."""
        return (-0.5 * (1 + log_variance - mean.square() - log_variance.exp())).mean(dim=0)

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
