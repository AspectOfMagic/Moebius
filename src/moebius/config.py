from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration shared by the single-task SAC and PEARL runs."""

    algorithm: str = "sac"
    env_name: str = "pick-place-v3"
    # SAC uses env_name; PEARL can share one policy across these environments.
    pearl_env_names: tuple[str, ...] = (
        "pick-place-v3",
        "pick-place-wall-v3",
        "shelf-place-v3",
        "pick-out-of-hole-v3",
    )
    seed: int = 1
    total_steps: int = 250_000
    start_steps: int = 10_000
    update_after: int = 1_000
    update_every: int = 50
    updates_per_step: int = 1
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float | None = None
    demo_episodes: int = 0
    bc_pretrain_updates: int = 10_000
    demo_batch_fraction: float = 0.25
    bc_weight: float = 1.0
    bc_decay_steps: int = 250_000
    pearl_task_distribution: str = "dynamics"
    # Spheres are omitted because scripted experts solve them unreliably.
    object_shapes: tuple[str, ...] = ("cylinder", "box", "bar")
    object_half_width_min: float = 0.015
    object_half_width_max: float = 0.025
    object_half_height_min: float = 0.015
    object_half_height_max: float = 0.03
    object_bar_half_length_min: float = 0.04
    object_bar_half_length_max: float = 0.07
    pearl_vary_object_dynamics: bool = False
    object_mass_min: float = 0.1
    # Heavier objects are rarely solvable under the default force limit.
    object_mass_max: float = 1.2
    object_table_friction_min: float = 0.05
    object_table_friction_max: float = 1.5
    object_finger_friction_min: float = 0.3
    object_finger_friction_max: float = 2.0
    # Newtons per finger; None preserves MetaWorld's unlimited servo.
    gripper_force_limit: float | None = None
    # After contact, the gripper action selects a fraction of the force limit.
    gripper_force_control: bool = False
    # Per-step penalty at full squeeze above the calibrated requirement.
    pearl_grip_penalty: float = 0.0
    # Levels tried in ascending order during grip calibration.
    pearl_grip_effort_levels: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.7, 0.8, 0.9, 1.0)
    # Reuse pick-place's empty second-object slots for contact forces.
    contact_observations: bool = False
    # Ordered strategies for single-environment force-control feasibility.
    pearl_expert_variants: tuple[str, ...] = ("careful", "slow_transport", "low_grasp", "metaworld")
    max_property_candidates_per_task: int = 25
    pearl_demo_episodes_per_task: int = 2
    pearl_demo_max_attempts_per_task: int = 20
    pearl_bc_pretrain_updates: int = 5_000
    pearl_critic_pretrain_updates: int = 5_000
    pearl_bc_weight_floor: float = 0.1
    pearl_reward_scale: float = 0.1
    # None enables entropy tuning; alpha uses scaled-reward units.
    pearl_fixed_alpha: float | None = None
    pearl_initial_alpha: float = 0.02
    # MetaWorld's per-step reward ceiling; zero disables TD-target clipping.
    pearl_max_reward: float = 10.0
    pearl_critic_layer_norm: bool = True
    # Optional transition-decoder loss; zero disables it.
    pearl_decoder_weight: float = 0.0
    # Failed expert attempts are replayed but never behavior-cloned.
    pearl_replay_failed_demos: bool = True
    gradient_clip_norm: float = 10.0
    min_posterior_std: float = 0.05
    # Recent adaptation sequences retained per task.
    context_sequences_per_task: int = 4
    context_motion_fraction: float = 0.75
    context_motion_threshold: float = 1e-3
    latent_dim: int = 5
    encoder_lr: float = 3e-4
    kl_weight: float = 0.05
    kl_free_bits: float = 0.1
    num_train_tasks: int = 20
    num_eval_tasks: int = 10
    meta_batch_size: int = 8
    context_size: int = 64
    adaptation_episodes: int = 3
    # Independent evaluation sequences per held-out task.
    pearl_eval_sequences_per_task: int = 1
    max_episode_steps: int = 500
    eval_interval: int = 10_000
    eval_episodes: int = 10
    log_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "auto"
