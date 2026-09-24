import numpy as np
import torch

from moebius.config import TrainConfig
from moebius.envs import DynamicsTaskSampler, GoalTaskSampler
from moebius.networks import ContextEncoder
from moebius.pearl import PEARLAgent, context_from_batch, context_transition
from moebius.networks import SquashedGaussianActor
from moebius.replay_buffer import (
    MultiTaskReplayBuffer,
    ReplayBuffer,
    TaskTrajectoryContextBuffer,
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


def test_task_context_buffer_preserves_chronological_prefixes() -> None:
    contexts = TaskTrajectoryContextBuffer((0, 1), max_episodes_per_task=2, seed=3)
    for task_id in (0, 1):
        contexts.begin_episode(task_id)
        for step in range(10):
            observation = np.array([float(step), float(task_id)], dtype=np.float32)
            contexts.add(task_id, observation, np.array([step], dtype=np.float32), float(step), observation + 1, False)
        contexts.end_episode(task_id)

    batch = contexts.sample_prefixes(np.array([0, 1]), max_context_size=8)

    assert batch.observations.shape[0] == 2
    assert batch.observations.shape[1] in (1, 4, 8)
    assert np.all(np.diff(batch.observations[:, :, 0], axis=1) == 1)


def test_failed_expert_trajectory_can_be_discarded_before_buffer_commit() -> None:
    """Demonstration collection must not add a trajectory until it is accepted."""
    replay = MultiTaskReplayBuffer((0,), observation_dim=2, action_dim=1, capacity=8)
    contexts = TaskTrajectoryContextBuffer((0,), max_episodes_per_task=2, seed=2)
    failed_trajectory = [
        (np.array([0.0, 0.0]), np.array([0.0]), 1.0, np.array([1.0, 0.0]), False),
        (np.array([1.0, 0.0]), np.array([0.0]), 1.0, np.array([2.0, 0.0]), False),
    ]
    successful_trajectory = [
        (np.array([3.0, 0.0]), np.array([0.0]), 1.0, np.array([4.0, 0.0]), False),
        (np.array([4.0, 0.0]), np.array([0.0]), 1.0, np.array([5.0, 0.0]), True),
    ]

    # This mirrors a failed collection attempt: it is intentionally never committed.
    assert replay.size(0) == 0
    contexts.begin_episode(0)
    for transition in successful_trajectory:
        replay.add(0, *transition)
        contexts.add(0, *transition)
    contexts.end_episode(0)

    assert len(failed_trajectory) == 2
    assert replay.size(0) == len(successful_trajectory)
    batch = contexts.sample_prefixes(np.array([0]), max_context_size=2)
    assert np.all(batch.observations[:, :, 0] >= 3.0)


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
    agent = PEARLAgent(observation_dim, np.full(action_dim, -1.0), np.full(action_dim, 1.0), latent_dim=4, hidden_dim=16)
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
