from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from moebius.config import TrainConfig
from moebius.envs import (
    SHAPES,
    SHAPE_ENVIRONMENTS,
    DynamicsTaskSampler,
    GoalTaskSampler,
    ObjectPropertyBounds,
    TaskSpec,
    object_geometry,
)
from moebius.experts import EXPERT_VARIANTS, dedicated_expert
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
from moebius.train import (
    demonstration_expert,
    demonstration_batch_sizes,
    grip_band_for_task,
    lowest_reliable_grip,
    evaluate_pearl,
    online_bc_weight,
    pearl_online_bc_weight,
    seed_failed_expert_attempts,
)


def test_actor_respects_action_bounds() -> None:
    actor = SquashedGaussianActor(3, 2, 16, torch.tensor([-2.0, -1.0]), torch.tensor([2.0, 3.0]))
    actions, log_probabilities = actor(torch.randn(8, 3))
    assert actions.shape == (8, 2)
    assert log_probabilities.shape == (8, 1)
    assert torch.all(actions[:, 0] >= -2) and torch.all(actions[:, 0] <= 2)
    assert torch.all(actions[:, 1] >= -1) and torch.all(actions[:, 1] <= 3)


def test_phase_three_defaults_to_pick_and_place_with_disjoint_task_configs() -> None:
    assert TrainConfig().env_name == "pick-place-v3"
    assert TrainConfig().pearl_env_names == SHAPE_ENVIRONMENTS
    assert TrainConfig().pearl_decoder_weight == 0.0
    assert not TrainConfig().pearl_vary_object_dynamics
    sampler = GoalTaskSampler(num_train_tasks=3, num_test_tasks=2, seed=7)

    assert len(sampler.train_tasks) == 3
    assert len(sampler.test_tasks) == 2
    assert {task.reset_seed for task in sampler.train_tasks}.isdisjoint(
        task.reset_seed for task in sampler.test_tasks
    )


def test_dynamics_tasks_cycle_shapes_and_respect_bounds() -> None:
    bounds = ObjectPropertyBounds(
        half_width=(0.01, 0.02),
        half_height=(0.02, 0.03),
        bar_half_length=(0.05, 0.06),
        mass=(0.2, 1.0),
        table_friction=(0.1, 1.0),
        finger_friction=(0.2, 2.0),
    )
    sampler = DynamicsTaskSampler(6, 4, bounds, seed=9)
    tasks = sampler.train_tasks + sampler.test_tasks

    assert [task.shape for task in sampler.train_tasks] == list(SHAPES) + list(SHAPES[:2])
    assert [task.shape for task in sampler.test_tasks] == list(SHAPES)
    assert all(task.reset_seed is None for task in tasks)
    for task in tasks:
        assert 0.01 <= task.object_half_width <= 0.02
        assert 0.2 <= task.object_mass <= 1.0
        assert 0.1 <= task.object_table_friction <= 1.0
        assert 0.2 <= task.object_finger_friction <= 2.0
        if task.shape == "sphere":
            assert task.object_half_height == task.object_half_width
        else:
            assert 0.02 <= task.object_half_height <= 0.03
        assert (task.object_half_length is not None) == (task.shape == "bar")
        if task.shape == "bar":
            assert 0.05 <= task.object_half_length <= 0.06
    train_masses = {task.object_mass for task in sampler.train_tasks}
    assert train_masses.isdisjoint(task.object_mass for task in sampler.test_tasks)


def test_dynamics_task_resampling_preserves_task_id_and_shape() -> None:
    sampler = DynamicsTaskSampler(4, 1, seed=4)
    original = sampler.train_tasks[3]

    replacement = sampler.resample_task(original)

    assert replacement.task_id == original.task_id
    assert replacement.shape == original.shape
    assert replacement != original


def test_shape_only_tasks_cover_environments_without_randomizing_dynamics() -> None:
    sampler = DynamicsTaskSampler(
        8,
        4,
        ObjectPropertyBounds(shapes=("cylinder", "box")),
        seed=4,
        env_names=SHAPE_ENVIRONMENTS,
        vary_dynamics=False,
    )

    assert [task.env_name for task in sampler.test_tasks] == list(SHAPE_ENVIRONMENTS)
    assert [task.shape for task in sampler.train_tasks[:4]] == ["cylinder"] * 4
    assert [task.shape for task in sampler.train_tasks[4:]] == ["box"] * 4
    assert all(task.object_mass is None for task in sampler.train_tasks + sampler.test_tasks)
    assert all(task.object_table_friction is None for task in sampler.train_tasks + sampler.test_tasks)
    assert all(task.object_finger_friction is None for task in sampler.train_tasks + sampler.test_tasks)
    assert {type(dedicated_expert(name)).__name__ for name in SHAPE_ENVIRONMENTS} == {
        "SawyerPickPlaceV3Policy",
        "SawyerPickPlaceWallV3Policy",
        "SawyerShelfPlaceV3Policy",
        "SawyerPickOutOfHoleV3Policy",
    }


def test_object_geometry_sizes_and_inertia() -> None:
    cylinder = object_geometry("cylinder", 0.02, 0.03, None, 1.0)
    assert cylinder.size == (0.02, 0.03, 0.0)
    assert cylinder.rest_height == 0.03 and cylinder.grip_half_width == 0.02
    assert np.isclose(cylinder.inertia[0], (3 * 0.02**2 + 4 * 0.03**2) / 12)
    assert np.isclose(cylinder.inertia[2], 0.02**2 / 2)

    sphere = object_geometry("sphere", 0.02, 0.5, None, 2.0)
    assert sphere.size[0] == 0.02 and sphere.rest_height == 0.02 and sphere.bounding_radius == 0.02
    assert np.allclose(sphere.inertia, 2 * 2.0 * 0.02**2 / 5)

    bar = object_geometry("bar", 0.02, 0.01, 0.06, 3.0)
    assert bar.half_extents == (0.06, 0.02, 0.01)
    assert bar.grip_half_width == 0.02 and bar.rest_height == 0.01
    assert np.isclose(bar.inertia[0], 3.0 * (0.02**2 + 0.01**2) / 3)
    assert np.isclose(bar.bounding_radius, np.linalg.norm((0.06, 0.02, 0.01)))

    box = object_geometry("box", 0.02, 0.01, 0.5, 1.0)
    assert box.half_extents == (0.02, 0.02, 0.01)  # half_length only applies to bars


def test_legacy_cylinder_tasks_convert_to_shaped_tasks() -> None:
    legacy = {
        "task_id": 3,
        "reset_seed": None,
        "object_radius": 0.02,
        "object_height": 0.05,
        "object_mass": 0.5,
        "object_table_friction": 1.0,
        "object_finger_friction": 2.0,
        "contact_softness": 0.02,
    }

    task = TaskSpec.from_dict(legacy)

    assert task.shape == "cylinder"
    assert task.object_half_width == 0.02 and task.object_half_height == 0.025
    assert task.object_mass == 0.5 and task.object_half_length is None
    assert TaskSpec.from_dict({"task_id": 1, "reset_seed": 5}) == TaskSpec(1, 5)


def test_property_bounds_read_from_config() -> None:
    config = asdict(TrainConfig(object_mass_min=0.3, object_shapes=("sphere", "bar")))
    bounds = ObjectPropertyBounds.from_config(config)
    assert bounds.mass == (0.3, TrainConfig().object_mass_max)
    assert bounds.shapes == ("sphere", "bar")
    assert ObjectPropertyBounds.from_config({}) == ObjectPropertyBounds()


def test_grip_effort_only_changes_the_grip_command() -> None:
    observation = np.zeros(39)
    observation[:3] = observation[4:7] = (0.0, 0.6, 0.05)  # hand at the object
    observation[3] = 0.5
    careful = EXPERT_VARIANTS["careful"]
    gentle = replace(careful, grip_effort=0.35)
    assert careful.get_action(observation)[3] == 1.0
    assert gentle.get_action(observation)[3] == pytest.approx(0.35)
    assert np.allclose(gentle.get_action(observation)[:3], careful.get_action(observation)[:3])
    observation[4:7] = (0.1, 0.7, 0.02)  # object far away: open
    assert careful.get_action(observation)[3] == -1.0
    assert EXPERT_VARIANTS["metaworld"].get_action(observation)[3] == 0.0  # MetaWorld parity


def test_force_control_demonstrations_use_one_full_grip_target() -> None:
    gentle = replace(EXPERT_VARIANTS["careful"], grip_effort=0.23)
    assert demonstration_expert(gentle, gripper_force_control=True).grip_effort == 1.0
    assert demonstration_expert(gentle, gripper_force_control=False) is gentle


def test_lowest_reliable_grip_picks_the_first_level_with_four_of_five_successes() -> None:
    calls = []

    def succeeds(level: float, attempt: int) -> bool:
        calls.append((level, attempt))
        return level >= 0.35

    assert lowest_reliable_grip(succeeds, (0.1, 0.2, 0.35, 0.5)) == 0.35
    assert calls == [
        (0.1, 0),
        (0.1, 1),
        (0.2, 0),
        (0.2, 1),
        (0.35, 0),
        (0.35, 1),
        (0.35, 2),
        (0.35, 3),
    ]
    assert lowest_reliable_grip(lambda level, attempt: False, (0.1, 0.2)) is None


def test_grip_bands_are_balanced_without_being_tied_to_shape() -> None:
    assignments = [grip_band_for_task(task_id, len(TrainConfig.object_shapes))[0] for task_id in range(18)]
    assert assignments.count("gentle") == 6
    assert assignments.count("medium") == 6
    assert assignments.count("strong") == 6
    pairs = [(TrainConfig.object_shapes[i % 3], assignments[i]) for i in range(18)]
    assert len(set(pairs)) == 9
    assert all(assignments[i] != "strong" or grip_band_for_task(i, 3)[1] == frozenset((0.8,)) for i in range(18))


def test_grip_penalty_charges_only_for_squeezing_beyond_the_calibrated_grip() -> None:
    pytest.importorskip("metaworld")
    from moebius.envs import MetaWorldSingleTaskEnv

    environment = MetaWorldSingleTaskEnv(
        "pick-place-v3",
        seed=0,
        max_episode_steps=120,
        gripper_force_limit=60.0,
        gripper_force_control=True,
        grip_penalty=1.0,
    )
    task = TaskSpec(
        0,
        shape="box",
        object_half_width=0.02,
        object_half_height=0.02,
        object_mass=0.2,
        object_table_friction=1.0,
        object_finger_friction=1.0,
    )

    def total_penalty(task: TaskSpec, expert) -> float:
        environment.set_task_properties(task)
        observation, _ = environment.reset()
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            observation, _, terminated, truncated, info = environment.step(expert.get_action(observation))
            total += info["grip_penalty"]
        return total

    try:
        careful = EXPERT_VARIANTS["careful"]
        calibrated = replace(task, object_grip_effort=0.2)
        assert total_penalty(task, careful) == 0.0  # no calibrated grip: no penalty
        full_grip = total_penalty(calibrated, careful)
        assert full_grip > 0.0
        assert total_penalty(calibrated, replace(careful, grip_effort=0.2)) < 0.2 * full_grip
    finally:
        environment.close()


def test_force_control_dense_reward_uses_the_continuous_effort() -> None:
    pytest.importorskip("metaworld")
    from moebius.envs import MetaWorldSingleTaskEnv

    environment = MetaWorldSingleTaskEnv(
        "pick-place-v3",
        seed=0,
        max_episode_steps=120,
        gripper_force_limit=60.0,
        gripper_force_control=True,
    )
    task = TaskSpec(
        0,
        shape="box",
        object_half_width=0.02,
        object_half_height=0.02,
        object_mass=0.2,
        object_table_friction=1.0,
        object_finger_friction=1.0,
    )
    try:
        environment.set_task_properties(task)
        observation, _ = environment.reset()
        careful = EXPERT_VARIANTS["careful"]
        low_effort = np.array([0.0, 0.0, 0.0, 0.01], dtype=np.float32)
        full_effort = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(100):
            observation, _, terminated, truncated, _ = environment.step(careful.get_action(observation))
            low_grasp = environment.unwrapped.compute_reward(low_effort, observation)[4]
            full_grasp = environment.unwrapped.compute_reward(full_effort, observation)[4]
            if full_grasp - low_grasp > 0.1:
                break
            assert not (terminated or truncated)
        else:
            pytest.fail("The expert never reached a state with an action-dependent caging reward.")

        # Force the post-contact phase under test.
        environment._gripper_contact_latched = True
        next_observation, reward, _, _, info = environment.step(low_effort)
        expected = environment.unwrapped.compute_reward(low_effort, next_observation)
        incorrectly_full = environment.unwrapped.compute_reward(full_effort, next_observation)
        assert reward == pytest.approx(expected[0])
        assert info["grasp_reward"] == pytest.approx(expected[4])
        assert info["grasp_reward"] < incorrectly_full[4]
    finally:
        environment.close()


def test_force_control_separates_precontact_closing_from_postcontact_squeeze() -> None:
    pytest.importorskip("metaworld")
    from moebius.envs import MetaWorldSingleTaskEnv

    limit = 60.0
    requested_effort = 0.2
    environment = MetaWorldSingleTaskEnv(
        "pick-place-v3",
        seed=0,
        max_episode_steps=10,
        gripper_force_limit=limit,
        gripper_force_control=True,
    )
    task = TaskSpec(
        0,
        shape="box",
        object_half_width=0.02,
        object_half_height=0.02,
    )
    try:
        environment.set_task_properties(task)
        environment.reset()
        environment._object_bilateral_finger_contact = lambda: False
        _, _, _, _, info = environment.step(np.array([0.0, 0.0, 0.0, requested_effort], dtype=np.float32))
        force_range = environment.unwrapped.model.actuator_forcerange[environment._finger_actuator_ids]
        assert info["gripper_precontact_closing"]
        assert not info["gripper_contact_latched"]
        assert np.allclose(force_range, (-limit, limit))

        environment._object_bilateral_finger_contact = lambda: True
        _, _, _, _, info = environment.step(np.array([0.0, 0.0, 0.0, requested_effort], dtype=np.float32))
        force_range = environment.unwrapped.model.actuator_forcerange[environment._finger_actuator_ids]
        assert not info["gripper_precontact_closing"]
        assert info["gripper_contact_latched"]
        assert info["gripper_bilateral_contact"]
        assert info["gripper_requested_effort"] == pytest.approx(requested_effort)
        assert info["gripper_contact_latched"] == environment.unwrapped._prev_obs[11]
        assert np.allclose(force_range, (-requested_effort * limit, requested_effort * limit))

        environment.step(np.array([0.0, 0.0, 0.0, -0.1], dtype=np.float32))
        force_range = environment.unwrapped.model.actuator_forcerange[environment._finger_actuator_ids]
        assert not environment._gripper_contact_latched
        assert np.allclose(force_range, (-limit, limit))
    finally:
        environment.close()


def test_force_control_caging_reward_saturates_at_the_calibrated_effort() -> None:
    pytest.importorskip("metaworld")
    from moebius.envs import MetaWorldSingleTaskEnv

    environment = MetaWorldSingleTaskEnv(
        "pick-place-v3",
        seed=0,
        max_episode_steps=120,
        gripper_force_limit=60.0,
        gripper_force_control=True,
    )
    task = TaskSpec(
        0,
        shape="box",
        object_half_width=0.02,
        object_half_height=0.02,
        object_mass=0.2,
        object_table_friction=1.0,
        object_finger_friction=1.0,
        object_grip_effort=0.35,
    )
    try:
        environment.set_task_properties(task)
        observation, _ = environment.reset()
        careful = EXPERT_VARIANTS["careful"]
        below = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float32)
        calibrated = np.array([0.0, 0.0, 0.0, 0.35], dtype=np.float32)
        full = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(100):
            observation, _, terminated, truncated, _ = environment.step(careful.get_action(observation))
            low_grasp = environment.unwrapped.compute_reward(below, observation)[4]
            calibrated_grasp = environment.unwrapped.compute_reward(calibrated, observation)[4]
            if calibrated_grasp - low_grasp > 0.1:
                break
            assert not (terminated or truncated)
        else:
            pytest.fail("The expert never reached a state with an action-dependent caging reward.")

        full_grasp = environment.unwrapped.compute_reward(full, observation)[4]
        assert calibrated_grasp == pytest.approx(full_grasp)
        assert low_grasp < calibrated_grasp
    finally:
        environment.close()


def test_default_expert_matches_metaworld_policy() -> None:
    policies = pytest.importorskip("metaworld.policies")
    reference = policies.SawyerPickPlaceV3Policy()
    expert = EXPERT_VARIANTS["metaworld"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        observation = rng.uniform(-0.2, 0.2, 39)
        observation[4:7] = observation[:3] + rng.normal(0, 0.05, 3)  # hand near the object
        observation[3] = rng.uniform(0, 1)
        expected = np.clip(reference.get_action(observation), -1, 1)
        assert np.allclose(expert.get_action(observation), expected, atol=1e-6)


def test_contact_observations_read_the_object_weight_from_the_table() -> None:
    pytest.importorskip("metaworld")
    from moebius.envs import CONTACT_FORCE_SCALE, CONTACT_SLICE, MetaWorldSingleTaskEnv

    environment = MetaWorldSingleTaskEnv("pick-place-v3", seed=0, max_episode_steps=20, contact_observations=True)
    try:
        for mass in (0.2, 1.0):
            environment.set_task_properties(
                TaskSpec(
                    0,
                    shape="box",
                    object_half_width=0.02,
                    object_half_height=0.02,
                    object_mass=mass,
                    object_table_friction=1.0,
                    object_finger_friction=1.0,
                )
            )
            environment.reset()
            for _ in range(5):  # hand stays above the object with the gripper open
                observation, *_ = environment.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))
            forces = observation[CONTACT_SLICE] * CONTACT_FORCE_SCALE
            assert forces[5] == pytest.approx(9.81 * mass, rel=0.2)
            assert np.allclose(forces[[0, 1, 2, 6]], 0.0, atol=1e-6)
            assert np.all(np.abs(observation[CONTACT_SLICE]) <= 5.0)
            assert observation[18 + 16] * CONTACT_FORCE_SCALE == pytest.approx(9.81 * mass, rel=0.2)
    finally:
        environment.close()


def test_pick_out_of_hole_reward_accepts_objects_wider_than_native_pad_threshold() -> None:
    pytest.importorskip("metaworld")
    from moebius.envs import MetaWorldSingleTaskEnv

    environment = MetaWorldSingleTaskEnv("pick-out-of-hole-v3", seed=0, max_episode_steps=5)
    try:
        environment.set_task_properties(
            TaskSpec(
                0,
                shape="box",
                object_half_width=TrainConfig.object_half_width_max,
                object_half_height=0.02,
                env_name="pick-out-of-hole-v3",
            )
        )
        environment.reset()
        _, reward, *_ = environment.step(np.zeros(4, dtype=np.float32))
        assert np.isfinite(reward)
    finally:
        environment.close()


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


def test_force_features_are_opt_in_for_regular_pearl() -> None:
    config = TrainConfig()
    assert config.gripper_force_limit is None
    assert not config.gripper_force_control
    assert not config.contact_observations
    assert config.pearl_grip_penalty == 0.0
    assert config.pearl_expert_variants == ("careful", "slow_transport", "low_grasp", "metaworld")


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
    metrics = agent.update(mixed_batch, demo_batch, bc_weight=0.5)

    assert mixed_batch.rl.observations.shape == (2, 12, observation_dim)
    assert all(np.isfinite(value) for value in pretrain_metrics.values())
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


def _random_multitask_buffer(observation_dim: int, action_dim: int, reward: float = 1.0) -> MultiTaskReplayBuffer:
    buffer = MultiTaskReplayBuffer(range(2), observation_dim, action_dim, capacity=32)
    for task_id in range(2):
        for _ in range(32):
            buffer.add(
                task_id,
                np.random.randn(observation_dim),
                np.random.uniform(-1, 1, action_dim),
                reward,
                np.random.randn(observation_dim),
                False,
            )
    return buffer


def test_behavior_cloning_does_not_train_the_encoder() -> None:
    """The encoder learns only from the critic loss, KL and the auxiliary decoder."""
    agent = PEARLAgent(3, np.full(2, -1.0), np.full(2, 1.0), latent_dim=4, hidden_dim=16)
    encoder_before = [parameter.detach().clone() for parameter in agent.encoder.parameters()]
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]

    agent.behavior_clone(_random_multitask_buffer(3, 2).sample_meta_batch(2, 8, 6))

    assert all(torch.equal(before, after) for before, after in zip(encoder_before, agent.encoder.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, agent.actor.parameters()))


def test_online_force_control_bc_clones_grip_direction_not_magnitude() -> None:
    expert = torch.tensor([[0.2, -0.3, 0.4, 1.0]])
    gentle_close = torch.tensor([[0.2, -0.3, 0.4, 0.2]])
    full_close = torch.tensor([[0.2, -0.3, 0.4, 1.0]])
    wrong_direction = torch.tensor([[0.2, -0.3, 0.4, -0.2]])

    gentle_loss = PEARLAgent._behavior_cloning_action_loss(gentle_close, expert, clone_grip_magnitude=False)
    full_loss = PEARLAgent._behavior_cloning_action_loss(full_close, expert, clone_grip_magnitude=False)
    wrong_loss = PEARLAgent._behavior_cloning_action_loss(wrong_direction, expert, clone_grip_magnitude=False)
    assert gentle_loss == pytest.approx(0.0)
    assert full_loss == pytest.approx(0.0)
    assert wrong_loss > gentle_loss
    assert PEARLAgent._behavior_cloning_action_loss(gentle_close, expert, clone_grip_magnitude=True) > 0


def test_decoder_loss_trains_the_encoder_and_logs_z_gain() -> None:
    torch.manual_seed(0)
    agent = PEARLAgent(
        12, np.full(2, -1.0), np.full(2, 1.0), latent_dim=4, hidden_dim=16, decoder_weight=1.0
    )
    task_batch = _random_multitask_buffer(12, 2).sample_meta_batch(2, 8, 6)
    observations = agent._flatten(task_batch.rl.observations)
    actions = agent._flatten(task_batch.rl.actions)
    targets = agent._decoder_targets(
        observations, agent._flatten(task_batch.rl.rewards), agent._flatten(task_batch.rl.next_observations)
    )
    assert targets.shape == (observations.shape[0], 10)
    mean, log_variance = agent.encoder(agent._tensor(context_from_batch(task_batch.context, agent.reward_scale)))
    latents = agent._expand_latents(agent._sample_latents(mean, log_variance), task_batch.rl.observations.shape[1])
    agent._decoder_loss(observations, actions, latents, targets).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in agent.encoder.parameters())

    metrics = agent.update_critic(task_batch)
    assert np.isfinite(metrics["loss/decoder"]) and np.isfinite(metrics["stats/decoder_z_gain"])
    assert "decoder" in agent.state_dict()


def test_decoder_weight_zero_restores_original_pearl_objective() -> None:
    agent = PEARLAgent(12, np.full(2, -1.0), np.full(2, 1.0), latent_dim=4, hidden_dim=16, decoder_weight=0.0)
    metrics = agent.update_critic(_random_multitask_buffer(12, 2).sample_meta_batch(2, 8, 6))
    assert agent.decoder is None and "decoder" not in agent.state_dict()
    assert "loss/decoder" not in metrics and "stats/decoder_z_gain" not in metrics


def test_critic_uses_layer_norm_by_default() -> None:
    agent = PEARLAgent(3, np.full(2, -1.0), np.full(2, 1.0), latent_dim=4, hidden_dim=16)
    plain = PEARLAgent(3, np.full(2, -1.0), np.full(2, 1.0), latent_dim=4, hidden_dim=16, critic_layer_norm=False)
    assert any(isinstance(layer, torch.nn.LayerNorm) for layer in agent.q1.network)
    assert not any(isinstance(layer, torch.nn.LayerNorm) for layer in plain.q1.network)


def test_td_targets_are_capped_at_the_largest_achievable_return() -> None:
    observation_dim, action_dim = 3, 2
    agent = PEARLAgent(
        observation_dim,
        np.full(action_dim, -1.0),
        np.full(action_dim, 1.0),
        latent_dim=4,
        hidden_dim=16,
        reward_scale=0.1,
        max_reward=10.0,
        fixed_alpha=0.1,
    )
    with torch.no_grad():
        for target in (agent.target_q1, agent.target_q2):
            target.network[-1].bias.fill_(1e4)

    metrics = agent.update_critic(_random_multitask_buffer(observation_dim, action_dim).sample_meta_batch(2, 8, 6))

    expected_bound = (0.1 * 10.0 + 0.1 * action_dim * np.log(2.0)) / (1 - agent.gamma)
    assert np.isclose(metrics["stats/q_upper_bound"], expected_bound, rtol=1e-5)
    assert metrics["stats/target_q_clipped_fraction"] == 1.0
    assert metrics["stats/target_q"] <= expected_bound + 1e-4


def test_failed_expert_attempts_are_replayed_as_experience_not_demonstrations() -> None:
    observation_dim, action_dim = 8, 2

    def trajectory(steps: int) -> list:
        return [
            (np.zeros(observation_dim), np.zeros(action_dim), 0.0, np.zeros(observation_dim), False)
            for _ in range(steps)
        ]

    class _Writer:
        def __init__(self) -> None:
            self.scalars: dict[str, float] = {}

        def add_scalar(self, name: str, value: float, step: int) -> None:
            self.scalars[name] = value

    buffer = MultiTaskReplayBuffer(range(3), observation_dim, action_dim, capacity=64)
    contexts = AdaptationContextBuffer(range(3), observation_dim, action_dim, seed=0)
    writer = _Writer()

    replayed = seed_failed_expert_attempts({0: [trajectory(5), trajectory(4)], 1: []}, buffer, contexts, writer)

    assert replayed == 9 and writer.scalars["demo/pearl_failed_transitions_replayed"] == 9
    assert buffer.size(0) == 9 and buffer.size(1) == 0 and buffer.size(2) == 0
    assert contexts.max_previous_episodes(0) == 2 and contexts.max_previous_episodes(1) == 0


def test_pearl_evaluation_uses_mean_actions_and_sampled_latents_at_every_step() -> None:
    from moebius.envs import TaskSpec

    observation_dim, action_dim, episode_length = 8, 2, 5

    class _FakeEnvironment:
        def set_task_properties(self, task: TaskSpec) -> None:
            pass

        def reset(self, seed: int | None = None):
            self.steps = 0
            return np.zeros(observation_dim, dtype=np.float32), {}

        def step(self, action: np.ndarray):
            self.steps += 1
            observation = np.full(observation_dim, self.steps, dtype=np.float32)
            return observation, 1.0, False, self.steps >= episode_length, {"success": True}

    agent = PEARLAgent(observation_dim, np.full(action_dim, -1.0), np.full(action_dim, 1.0), latent_dim=3, hidden_dim=16)
    action_modes, latent_modes = set(), set()
    act, latent_for_episode = agent.act, agent.latent_for_episode

    def recording_act(observation, context=None, deterministic=False, latent=None):
        action_modes.add(deterministic)
        return act(observation, context, deterministic, latent)

    def recording_latent(previous_episodes, builder, rng, deterministic=False):
        if previous_episodes:
            latent_modes.add(deterministic)
        return latent_for_episode(previous_episodes, builder, rng, deterministic)

    agent.act, agent.latent_for_episode = recording_act, recording_latent
    tasks = (TaskSpec(0), TaskSpec(1))

    evaluation = evaluate_pearl(
        agent, _FakeEnvironment(), tasks, 2, ContextBuilder(context_size=4), np.random.default_rng(0), sequences_per_task=3
    )

    assert action_modes == {True}
    assert latent_modes == {False}
    assert evaluation.episodes_per_step == 6
    assert np.allclose(evaluation.successes, 1.0) and np.allclose(evaluation.task_successes, 1.0)
    assert np.allclose(evaluation.returns, episode_length)


def test_low_grasp_stops_descending_at_the_hand_floor() -> None:
    """MetaWorld clamps the hand at z = 0.05, below which a grasp target is unreachable."""
    observation = np.zeros(39)
    observation[:3] = (0.0, 0.6, 0.052)  # hand at the floor, above the object
    observation[3] = observation[21] = 0.9  # gripper still open
    observation[4:7] = (0.005, 0.6, 0.02)
    observation[36:39] = (0.1, 0.8, 0.2)

    waiting = EXPERT_VARIANTS["low_grasp"].get_action(observation)
    observation[3] = observation[21] = 0.5  # gripper closed on the object
    carrying = EXPERT_VARIANTS["low_grasp"].get_action(observation)

    assert np.allclose(waiting[:3], 0.0) and waiting[3] == 1.0
    assert np.all(carrying[:3] > 0)


def test_low_grasp_keeps_lifting_a_short_object() -> None:
    """At the floor the hand is ~2.7 cm above a short box; lifting must not re-trigger the descent."""
    observation = np.zeros(39)
    observation[:3] = (0.0, 0.6, 0.063)  # hand lifted 1.6 cm above the floor
    observation[3] = observation[21] = 0.54  # gripper closed on the object
    observation[4:7] = (0.005, 0.6, 0.036)  # object lifted 1.6 cm, still below 4 cm
    observation[24] = 0.033  # and rising
    observation[36:39] = (0.1, 0.8, 0.2)
    for name in ("low_grasp", "careful"):
        assert EXPERT_VARIANTS[name].get_action(observation)[2] > 0, name
