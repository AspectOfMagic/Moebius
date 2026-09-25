import numpy as np
import torch

from moebius.config import TrainConfig
from moebius.envs import DynamicsTaskSampler, GoalTaskSampler
from moebius.networks import ContextEncoder
from moebius.pearl import PEARLAgent, context_from_batch, context_transition
from moebius.networks import SquashedGaussianActor
from moebius.context import AdaptationContextBuffer, ContextBuilder, episode_batch
from moebius.replay_buffer import (
    Batch,
    MultiTaskReplayBuffer,
    ReplayBuffer,
    TaskBatch,
    concatenate_batches,
    concatenate_task_batch_rl,
)
from moebius.sac import SACAgent
from moebius.train import demonstration_batch_sizes, online_bc_weight, pearl_online_bc_weight


def test_actor_respects_action_bounds() -> None:
    actor = SquashedGaussianActor(3, 2, 16, torch.tensor([-2.0, -1.0]), torch.tensor([2.0, 3.0]))
    actions, log_probabilities = actor(torch.randn(8, 3))
    assert actions.shape == (8, 2)
    assert log_probabilities.shape == (8, 1)
    assert torch.all(actions[:, 0] >= -2) and torch.all(actions[:, 0] <= 2)
    assert torch.all(actions[:, 1] >= -1) and torch.all(actions[:, 1] <= 3)


def test_phase_three_defaults_to_pick_and_place_with_disjoint_task_configs() -> None:
    assert TrainConfig().env_name == "pick-place-v3"
    sampler = GoalTaskSampler(num_train_tasks=3, num_test_tasks=2, seed=7)

    assert len(sampler.train_tasks) == 3
    assert len(sampler.test_tasks) == 2
    assert {task.reset_seed for task in sampler.train_tasks}.isdisjoint(
        task.reset_seed for task in sampler.test_tasks
    )


def test_dynamics_tasks_hide_disjoint_diverse_object_property_vectors() -> None:
    sampler = DynamicsTaskSampler(
        3,
        2,
        object_radius_min=0.01,
        object_radius_max=0.02,
        object_height_min=0.02,
        object_height_max=0.04,
        object_mass_min=0.2,
        object_mass_max=1.0,
        object_table_friction_min=0.1,
        object_table_friction_max=1.0,
        object_finger_friction_min=0.2,
        object_finger_friction_max=2.0,
        contact_softness_min=0.005,
        contact_softness_max=0.02,
        seed=9,
    )
    train_parameters = {
        (
            task.object_radius,
            task.object_height,
            task.object_mass,
            task.object_table_friction,
            task.object_finger_friction,
            task.contact_softness,
        )
        for task in sampler.train_tasks
    }
    test_parameters = {
        (
            task.object_radius,
            task.object_height,
            task.object_mass,
            task.object_table_friction,
            task.object_finger_friction,
            task.contact_softness,
        )
        for task in sampler.test_tasks
    }

    assert train_parameters.isdisjoint(test_parameters)
    assert all(task.reset_seed is None for task in sampler.train_tasks + sampler.test_tasks)
    assert all(
        0.01 <= radius <= 0.02
        and 0.02 <= height <= 0.04
        and 0.2 <= mass <= 1.0
        and 0.1 <= table_friction <= 1.0
        and 0.2 <= finger_friction <= 2.0
        and 0.005 <= softness <= 0.02
        for radius, height, mass, table_friction, finger_friction, softness in train_parameters | test_parameters
    )


def test_dynamics_task_resampling_preserves_task_id_and_bounds() -> None:
    sampler = DynamicsTaskSampler(1, 1, object_radius_min=0.01, object_radius_max=0.02, seed=4)

    replacement = sampler.resample_task(task_id=17)

    assert replacement.task_id == 17
    assert replacement.reset_seed is None
    assert 0.01 <= replacement.object_radius <= 0.02
    assert replacement.object_height is not None
    assert replacement.object_mass is not None
    assert replacement.object_table_friction is not None
    assert replacement.object_finger_friction is not None
    assert replacement.contact_softness is not None


def _episode(start: float, steps: int, observation_dim: int = 8, moving: bool = False) -> Batch:
    """A synthetic episode; the object (obs[4:7]) moves 1 cm per step if ``moving``."""
    observations = np.zeros((steps, observation_dim), dtype=np.float32)
    observations[:, 0] = start + np.arange(steps)
    next_observations = observations.copy()
    if moving:
        next_observations[:, 4] += 0.01
    return episode_batch(
        list(observations),
        list(np.zeros((steps, 2), dtype=np.float32)),
        [1.0] * steps,
        list(next_observations),
        [False] * steps,
    )


def test_context_builder_samples_distinct_transitions_and_prefers_object_motion() -> None:
    builder = ContextBuilder(context_size=10, motion_fraction=0.8, motion_threshold=1e-3)
    still, moving = _episode(0.0, 50), _episode(100.0, 50, moving=True)

    context = builder.build([still, moving], np.random.default_rng(0))

    assert context.observations.shape == (10, 8)
    assert len(np.unique(context.observations[:, 0])) == 10
    assert builder.motion_fraction_of(context) == 0.8
    # A pool smaller than the context size is used whole, without duplicates.
    short = builder.build([_episode(0.0, 4)], np.random.default_rng(0))
    assert sorted(short.observations[:, 0]) == [0.0, 1.0, 2.0, 3.0]


def test_adaptation_context_buffer_matches_deployment_history() -> None:
    """Context for k previous episodes only uses the first k episodes of a sequence."""
    builder = ContextBuilder(context_size=6, motion_fraction=0.0)
    contexts = AdaptationContextBuffer((0, 1), observation_dim=8, action_dim=2, max_sequences_per_task=2, seed=3)
    for task_id in (0, 1):
        contexts.begin_sequence(task_id)
        for episode_index in range(3):
            contexts.begin_episode(task_id)
            for step in range(5):
                observation = np.zeros(8, dtype=np.float32)
                observation[0] = 100 * episode_index + step
                contexts.add(task_id, observation, np.zeros(2), 1.0, observation, False)
            contexts.end_episode(task_id)
    assert contexts.max_previous_episodes(0) == 3

    seen_lengths = set()
    for _ in range(40):
        batch, previous_episodes = contexts.sample(np.array([0, 1]), builder, max_previous_episodes=2)
        seen_lengths.add(previous_episodes)
        assert batch.observations.shape[:2] == (2, 0 if previous_episodes == 0 else 5 if previous_episodes == 1 else 6)
        if previous_episodes:
            # Episode index = obs[0] // 100 must be below k.
            assert np.all(batch.observations[..., 0] // 100 < previous_episodes)
    assert seen_lengths == {0, 1, 2}


def test_unfinished_episode_is_not_used_as_context() -> None:
    contexts = AdaptationContextBuffer((0,), observation_dim=8, action_dim=2, seed=2)
    contexts.begin_sequence(0)
    contexts.begin_episode(0)
    contexts.add(0, np.zeros(8), np.zeros(2), 1.0, np.zeros(8), False)
    assert contexts.max_previous_episodes(0) == 0
    batch, previous_episodes = contexts.sample(np.array([0]), ContextBuilder(context_size=4), max_previous_episodes=3)
    assert previous_episodes == 0 and batch.observations.shape == (1, 0, 8)


def test_sac_update_produces_finite_metrics() -> None:
    buffer = ReplayBuffer(3, 2, capacity=32)
    for _ in range(32):
        buffer.add(np.random.randn(3), np.random.uniform(-1, 1, 2), 1.0, np.random.randn(3), False)
    agent = SACAgent(3, np.full(2, -1.0), np.full(2, 1.0), hidden_dim=16)
    metrics = agent.update(buffer.sample(16))
    assert all(np.isfinite(value) for value in metrics.values())


def test_demonstration_augmented_sac_updates_and_decays_bc_loss() -> None:
    online = ReplayBuffer(3, 2, capacity=32)
    demonstrations = ReplayBuffer(3, 2, capacity=32)
    for _ in range(32):
        online.add(np.random.randn(3), np.random.uniform(-1, 1, 2), 1.0, np.random.randn(3), False)
        demonstrations.add(np.random.randn(3), np.random.uniform(-1, 1, 2), 1.0, np.random.randn(3), False)
    agent = SACAgent(3, np.full(2, -1.0), np.full(2, 1.0), hidden_dim=16)

    pretrain_metrics = agent.behavior_clone(demonstrations.sample(8))
    demonstration_batch = demonstrations.sample(4)
    mixed_batch = concatenate_batches(online.sample(8), demonstration_batch)
    metrics = agent.update(mixed_batch, demonstration_batch, bc_weight=0.5)

    assert all(np.isfinite(value) for value in pretrain_metrics.values())
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["loss/bc"] >= 0
    assert mixed_batch.observations.shape == (12, 3)
    assert demonstration_batch_sizes(256, 0.25) == (192, 64)
    assert online_bc_weight(TrainConfig(bc_weight=1.0, bc_decay_steps=100), 0) == 1.0
    assert online_bc_weight(TrainConfig(bc_weight=1.0, bc_decay_steps=100), 50) == 0.5
    assert online_bc_weight(TrainConfig(bc_weight=1.0, bc_decay_steps=100), 100) == 0.0


def test_context_encoder_is_permutation_invariant() -> None:
    encoder = ContextEncoder(context_dim=9, latent_dim=3, hidden_dim=16)
    context = torch.randn(2, 5, 9)
    mean, log_variance = encoder(context)
    reordered_mean, reordered_log_variance = encoder(context.flip(dims=(1,)))
    assert mean.shape == (2, 3)
    assert log_variance.shape == (2, 3)
    assert torch.allclose(mean, reordered_mean)
    assert torch.allclose(log_variance, reordered_log_variance)


def test_context_encoder_respects_posterior_std_floor() -> None:
    encoder = ContextEncoder(context_dim=5, latent_dim=2, hidden_dim=8, min_posterior_std=0.2)
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        encoder.network[-1].bias[2:] = -10.0

    _, log_variance = encoder(torch.zeros(1, 32, 5))

    assert torch.all(torch.exp(0.5 * log_variance) >= 0.2 - 1e-6)


def test_pearl_context_reward_is_scaled() -> None:
    transition = context_transition(
        np.array([1.0, 2.0]), np.array([0.5]), 10.0, np.array([3.0, 4.0]), reward_scale=0.1
    )
    assert transition[3] == 1.0


def test_multitask_buffer_and_pearl_update_produce_finite_metrics() -> None:
    observation_dim, action_dim = 3, 2
    buffer = MultiTaskReplayBuffer(range(3), observation_dim, action_dim, capacity=32)
    for task_id in range(3):
        for _ in range(32):
            buffer.add(
                task_id,
                np.random.randn(observation_dim),
                np.random.uniform(-1, 1, action_dim),
                float(np.random.randn()),
                np.random.randn(observation_dim),
                False,
            )
    task_batch = buffer.sample_meta_batch(meta_batch_size=2, batch_size=8, context_size=6)
    assert task_batch.context.observations.shape == (2, 6, observation_dim)
    assert context_from_batch(task_batch.context).shape == (2, 6, 2 * observation_dim + action_dim + 1)
    agent = PEARLAgent(observation_dim, np.full(action_dim, -1.0), np.full(action_dim, 1.0), latent_dim=4, hidden_dim=16)
    metrics = agent.update(task_batch)
    assert all(np.isfinite(value) for value in metrics.values())
    action = agent.act(np.zeros(observation_dim, dtype=np.float32), deterministic=True)
    assert action.shape == (action_dim,)


def test_demonstration_augmented_pearl_updates() -> None:
    observation_dim, action_dim = 3, 2
    online = MultiTaskReplayBuffer(range(3), observation_dim, action_dim, capacity=32)
    demonstrations = MultiTaskReplayBuffer(range(3), observation_dim, action_dim, capacity=32)
    for task_id in range(3):
        for _ in range(32):
            transition = (np.random.randn(observation_dim), np.random.uniform(-1, 1, action_dim), float(np.random.randn()), np.random.randn(observation_dim), False)
            online.add(task_id, *transition)
            demonstrations.add(task_id, *transition)
    online_batch = online.sample_meta_batch(meta_batch_size=2, batch_size=8, context_size=6)
    demo_batch = demonstrations.sample_meta_batch_for_task_ids(online_batch.task_ids, batch_size=4, context_size=6)
    mixed_batch = concatenate_task_batch_rl(online_batch, demo_batch)
    agent = PEARLAgent(
        observation_dim, np.full(action_dim, -1.0), np.full(action_dim, 1.0), latent_dim=4, hidden_dim=16, fixed_alpha=0.1
    )
    initial_alpha = agent.alpha.item()

    pretrain_metrics = agent.behavior_clone(demo_batch)
    prior_pretrain_metrics = agent.behavior_clone(demo_batch, prior_context=True)
    metrics = agent.update(mixed_batch, demo_batch, bc_weight=0.5)

    assert mixed_batch.rl.observations.shape == (2, 12, observation_dim)
    assert all(np.isfinite(value) for value in pretrain_metrics.values())
    assert all(np.isfinite(value) for value in prior_pretrain_metrics.values())
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["loss/bc"] >= 0
    assert metrics["loss/alpha"] == 0
    assert metrics["stats/q_scale"] >= 1
    assert agent.alpha.item() == initial_alpha
    assert pearl_online_bc_weight(TrainConfig(bc_weight=1.0, bc_decay_steps=100, pearl_bc_weight_floor=0.1), 100) == 0.1


def test_pearl_actor_normalization_scales_entropy_with_q() -> None:
    """Dividing only Q by q_scale would inflate the effective temperature."""
    alpha = torch.tensor(0.1)
    log_prob = torch.full((4, 1), -1.0, requires_grad=True)
    q_value = torch.full((4, 1), 50.0, requires_grad=True)

    loss, q_scale = PEARLAgent._normalized_sac_actor_loss(alpha, log_prob, q_value)
    loss.backward()

    assert q_scale.item() == 50.0
    # Entropy and Q gradients keep SAC's alpha : 1 ratio after normalization.
    assert torch.allclose(log_prob.grad, torch.full_like(log_prob, 0.1 / 50.0 / 4))
    assert torch.allclose(q_value.grad, torch.full_like(q_value, -1.0 / 50.0 / 4))
    assert torch.allclose(log_prob.grad / -q_value.grad, alpha.expand_as(log_prob))


def test_pearl_tunes_alpha_automatically_by_default() -> None:
    observation_dim, action_dim = 3, 2
    buffer = MultiTaskReplayBuffer(range(2), observation_dim, action_dim, capacity=32)
    for task_id in range(2):
        for _ in range(32):
            buffer.add(
                task_id,
                np.random.randn(observation_dim),
                np.random.uniform(-1, 1, action_dim),
                float(np.random.randn()),
                np.random.randn(observation_dim),
                False,
            )
    agent = PEARLAgent(observation_dim, np.full(action_dim, -1.0), np.full(action_dim, 1.0), latent_dim=4, hidden_dim=16)
    assert agent.alpha_optimizer is not None
    assert np.isclose(agent.alpha.item(), TrainConfig().pearl_initial_alpha)

    initial_alpha = agent.alpha.item()
    metrics = agent.update(buffer.sample_meta_batch(meta_batch_size=2, batch_size=8, context_size=6))

    assert all(np.isfinite(value) for value in metrics.values())
    assert "stats/policy_entropy" in metrics and "stats/policy_log_std" in metrics
    assert agent.alpha.item() != initial_alpha


def test_context_encoder_has_gradient_when_factors_are_very_uncertain() -> None:
    """The old clamp(-10, 5) gave zero gradient here, freezing the posterior at the prior."""
    encoder = ContextEncoder(context_dim=5, latent_dim=2, hidden_dim=8)
    with torch.no_grad():
        encoder.network[-1].bias[2:] = 100.0

    mean, log_variance = encoder(torch.randn(3, 16, 5))
    (mean.sum() + log_variance.sum()).backward()

    assert torch.exp(0.5 * log_variance).mean() > 0.9
    assert encoder.network[-1].bias.grad[2:].abs().sum() > 0


def test_context_encoder_empty_context_returns_prior() -> None:
    encoder = ContextEncoder(context_dim=5, latent_dim=3, hidden_dim=8, min_posterior_std=0.05)
    mean, log_variance = encoder(torch.zeros(2, 0, 5))
    assert torch.allclose(mean, torch.zeros_like(mean))
    assert torch.allclose(log_variance.exp(), torch.full_like(log_variance, 1 + 0.05**2))


def test_pearl_update_with_empty_context_uses_zero_latent_and_skips_kl() -> None:
    observation_dim, action_dim = 8, 2
    buffer = MultiTaskReplayBuffer(range(2), observation_dim, action_dim, capacity=32)
    for task_id in range(2):
        for _ in range(32):
            buffer.add(task_id, np.random.randn(observation_dim), np.random.uniform(-1, 1, action_dim), 1.0, np.random.randn(observation_dim), False)
    contexts = AdaptationContextBuffer(range(2), observation_dim, action_dim, seed=0)
    task_batch = buffer.sample_meta_batch(meta_batch_size=2, batch_size=8, context_size=1)
    context, previous_episodes = contexts.sample(task_batch.task_ids, ContextBuilder(context_size=4), 3)
    assert previous_episodes == 0
    empty_batch = TaskBatch(task_ids=task_batch.task_ids, context=context, rl=task_batch.rl)
    agent = PEARLAgent(observation_dim, np.full(action_dim, -1.0), np.full(action_dim, 1.0), latent_dim=3, hidden_dim=16)

    metrics = agent.update(empty_batch)
    bc_metrics = agent.behavior_clone(empty_batch)

    assert all(np.isfinite(value) for value in metrics.values())
    assert all(np.isfinite(value) for value in bc_metrics.values())
    assert "loss/kl" not in metrics and metrics["stats/context_length"] == 0


def test_kl_free_bits_leave_small_posteriors_unpenalized() -> None:
    per_dimension = PEARLAgent._kl_per_dimension(torch.tensor([[0.1, 2.0]]), torch.zeros(1, 2))
    assert torch.allclose(per_dimension, torch.tensor([0.005, 2.0]))
    assert torch.isclose(per_dimension.clamp_min(0.1).sum(), torch.tensor(2.1))


def test_latent_for_episode_uses_prior_first_then_context() -> None:
    agent = PEARLAgent(8, np.full(2, -1.0), np.full(2, 1.0), latent_dim=3, hidden_dim=16)
    builder = ContextBuilder(context_size=5)
    rng = np.random.default_rng(0)

    first_latent, first_context = agent.latent_for_episode([], builder, rng)
    later_latent, later_context = agent.latent_for_episode([_episode(0.0, 20)], builder, rng, deterministic=True)

    assert first_context is None and np.all(first_latent == 0)
    assert later_context is not None and later_context.observations.shape == (5, 8)
    assert later_latent.shape == (3,)
