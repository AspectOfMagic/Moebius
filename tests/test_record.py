from types import SimpleNamespace

import numpy as np
import torch

from moebius.pearl import PEARLAgent
from moebius.record import load_pearl_agent


def test_load_pearl_agent_restores_encoder_and_latent_policy() -> None:
    observation_dim, action_dim = 3, 2
    source = PEARLAgent(
        observation_dim,
        np.full(action_dim, -1.0),
        np.full(action_dim, 1.0),
        latent_dim=4,
        hidden_dim=16,
    )
    checkpoint = {
        "config": {"latent_dim": 4, "hidden_dim": 16, "pearl_critic_layer_norm": True},
        "agent": source.state_dict(),
    }
    environment = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(observation_dim,)),
        action_space=SimpleNamespace(low=np.full(action_dim, -1.0), high=np.full(action_dim, 1.0)),
    )

    restored = load_pearl_agent(checkpoint, environment, "cpu")

    assert restored.latent_dim == source.latent_dim
    assert torch.equal(restored.log_alpha, source.log_alpha)
    for restored_parameter, source_parameter in zip(restored.encoder.parameters(), source.encoder.parameters()):
        assert torch.equal(restored_parameter, source_parameter)
    action = restored.act(np.zeros(observation_dim, dtype=np.float32), deterministic=True)
    assert action.shape == (action_dim,)


def test_corner_camera_frames_are_rotated_by_default() -> None:
    from moebius.record import should_rotate_frames

    assert should_rotate_frames("corner", None) is True
    assert should_rotate_frames("topview", None) is False
    # Explicit flags override the default in both directions.
    assert should_rotate_frames("corner", False) is False
    assert should_rotate_frames("corner3", True) is True
