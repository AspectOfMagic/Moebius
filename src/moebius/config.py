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
    mass_scale_min: float = 0.5
    mass_scale_max: float = 1.5
    friction_scale_min: float = 0.5
    friction_scale_max: float = 1.5
    pearl_demo_episodes_per_task: int = 2
    pearl_bc_pretrain_updates: int = 5_000
    latent_dim: int = 5
    encoder_lr: float = 3e-4
    kl_weight: float = 0.1
    num_train_tasks: int = 20
    num_eval_tasks: int = 10
    meta_batch_size: int = 8
    context_size: int = 32
    adaptation_episodes: int = 3
    max_episode_steps: int = 500
    eval_interval: int = 10_000
    eval_episodes: int = 10
    log_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "auto"
