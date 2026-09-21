import numpy as np
import torch

from moebius.config import TrainConfig
from moebius.envs import DynamicsTaskSampler, GoalTaskSampler
from moebius.networks import ContextEncoder
from moebius.pearl import PEARLAgent, context_from_batch
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


def test_dynamics_tasks_hide_disjoint_mass_and_friction_pairs() -> None:
    sampler = DynamicsTaskSampler(3, 2, mass_scale_min=0.5, mass_scale_max=1.5, friction_scale_min=0.5, friction_scale_max=1.5, seed=9)
    train_parameters = {(task.mass_scale, task.friction_scale) for task in sampler.train_tasks}
    test_parameters = {(task.mass_scale, task.friction_scale) for task in sampler.test_tasks}

    assert train_parameters.isdisjoint(test_parameters)
    assert all(task.reset_seed is None for task in sampler.train_tasks + sampler.test_tasks)
    assert all(0.5 <= mass <= 1.5 and 0.5 <= friction <= 1.5 for mass, friction in train_parameters | test_parameters)


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

    pretrain_metrics = agent.behavior_clone(demo_batch)
    prior_pretrain_metrics = agent.behavior_clone(demo_batch, prior_context=True)
    metrics = agent.update(mixed_batch, demo_batch, bc_weight=0.5)

    assert mixed_batch.rl.observations.shape == (2, 12, observation_dim)
    assert all(np.isfinite(value) for value in pretrain_metrics.values())
    assert all(np.isfinite(value) for value in prior_pretrain_metrics.values())
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["loss/bc"] >= 0
    assert pearl_online_bc_weight(TrainConfig(bc_weight=1.0, bc_decay_steps=100, pearl_bc_weight_floor=0.1), 100) == 0.1
