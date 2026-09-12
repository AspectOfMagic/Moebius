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
