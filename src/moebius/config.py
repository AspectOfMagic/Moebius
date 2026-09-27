from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """Configuration shared by the single-task SAC and PEARL runs."""

    algorithm: str = "sac"
    # Phase 3's state-based manipulation target. Reset seeds define the
    # object/goal configurations used as PEARL tasks; only the bookkeeping
    # task identifier is omitted from the observation.
    env_name: str = "pick-place-v3"
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
    # Set demo_episodes > 0 to enable the Phase 3 demonstration-augmented SAC
    # protocol. The zero default preserves a pure-SAC baseline.
    demo_episodes: int = 0
    bc_pretrain_updates: int = 10_000
    demo_batch_fraction: float = 0.25
    bc_weight: float = 1.0
    bc_decay_steps: int = 250_000
    pearl_task_distribution: str = "dynamics"
    # Bounds for PEARL's hidden object-property distribution. These deliberately
    # live in configuration rather than the CLI: they define the experiment's
    # task family and must be saved with the checkpoint.
    object_radius_min: float = 0.015
    object_radius_max: float = 0.025
    object_height_min: float = 0.025
    object_height_max: float = 0.055
    object_mass_min: float = 0.25
    object_mass_max: float = 1.25
    object_table_friction_min: float = 0.25
    object_table_friction_max: float = 2.00
    object_finger_friction_min: float = 1.00
    object_finger_friction_max: float = 3.00
    # MuJoCo solref's positive time-constant parameter: higher values create
    # softer contacts. The lower bound remains above the simulation timestep.
    contact_softness_min: float = 0.008
    contact_softness_max: float = 0.060
    # Number of distinct property vectors tried when finding one that can
    # yield the requested expert demonstrations.
    max_property_candidates_per_task: int = 25
    pearl_demo_episodes_per_task: int = 2
    pearl_demo_max_attempts_per_task: int = 20
    pearl_bc_pretrain_updates: int = 5_000
    pearl_critic_pretrain_updates: int = 5_000
    pearl_bc_weight_floor: float = 0.1
    pearl_reward_scale: float = 0.1
    # None enables SAC's automatic entropy tuning toward target_entropy.
    # Both alpha values are in scaled-reward units (see pearl_reward_scale).
    pearl_fixed_alpha: float | None = None
    pearl_initial_alpha: float = 0.02
    # Critic stabilization. pearl_max_reward is the largest per-step reward
    # (MetaWorld: 10); TD targets are capped at the return it implies. Set 0
    # to disable. LayerNorm in the critics limits value extrapolation.
    pearl_max_reward: float = 10.0
    pearl_critic_layer_norm: bool = True
    # Store the scripted expert's failed attempts (accepted training tasks
    # only) as online experience and context. Never behavior-cloned.
    pearl_replay_failed_demos: bool = True
    gradient_clip_norm: float = 10.0
    min_posterior_std: float = 0.05
    # Encoder context: built from the completed episodes earlier in the same
    # adaptation sequence, identically in training, evaluation and recording.
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
    max_episode_steps: int = 500
    eval_interval: int = 10_000
    eval_episodes: int = 10
    log_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "auto"
