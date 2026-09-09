import numpy as np
import torch

from moebius.networks import ContextEncoder
from moebius.pearl import PEARLAgent, context_from_batch
from moebius.networks import SquashedGaussianActor
from moebius.replay_buffer import MultiTaskReplayBuffer, ReplayBuffer
from moebius.sac import SACAgent


def test_actor_respects_action_bounds() -> None:
    actor = SquashedGaussianActor(3, 2, 16, torch.tensor([-2.0, -1.0]), torch.tensor([2.0, 3.0]))
    actions, log_probabilities = actor(torch.randn(8, 3))
    assert actions.shape == (8, 2)
    assert log_probabilities.shape == (8, 1)
    assert torch.all(actions[:, 0] >= -2) and torch.all(actions[:, 0] <= 2)
    assert torch.all(actions[:, 1] >= -1) and torch.all(actions[:, 1] <= 3)


def test_sac_update_produces_finite_metrics() -> None:
    buffer = ReplayBuffer(3, 2, capacity=32)
    for _ in range(32):
        buffer.add(np.random.randn(3), np.random.uniform(-1, 1, 2), 1.0, np.random.randn(3), False)
    agent = SACAgent(3, np.full(2, -1.0), np.full(2, 1.0), hidden_dim=16)
    metrics = agent.update(buffer.sample(16))
    assert all(np.isfinite(value) for value in metrics.values())


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
