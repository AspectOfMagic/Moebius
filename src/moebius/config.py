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
    # PEARL shares one policy and encoder across these compatible
    # pick-and-place environments. SAC continues to use only ``env_name``.
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
    # Set demo_episodes > 0 to enable the Phase 3 demonstration-augmented SAC
    # protocol. The zero default preserves a pure-SAC baseline.
    demo_episodes: int = 0
    bc_pretrain_updates: int = 10_000
    demo_batch_fraction: float = 0.25
    bc_weight: float = 1.0
    bc_decay_steps: int = 250_000
    pearl_task_distribution: str = "dynamics"
    # PEARL shape tasks: object shape and size, with optional hidden dynamics.
    # Bounds deliberately live in configuration rather than the CLI: they
    # define the experiment's task family and are saved with the checkpoint.
    # Shapes are assigned round-robin; spheres use half_width as the radius
    # and only bars use the half-length range (their half-extent along x).
    # Spheres are left out by default: the sweep found them solvable only about
    # half the time by any expert variant, adding difficulty but no contrast.
    object_shapes: tuple[str, ...] = ("cylinder", "box", "bar")
    object_half_width_min: float = 0.015
    object_half_width_max: float = 0.025
    object_half_height_min: float = 0.015
    object_half_height_max: float = 0.03
    object_bar_half_length_min: float = 0.04
    object_bar_half_length_max: float = 0.07
    # Hidden dynamics are optional. The default shape-only experiment keeps
    # each MetaWorld environment's nominal mass and friction.
    pearl_vary_object_dynamics: bool = False
    object_mass_min: float = 0.1
    # Capped at 1.2 kg: with a 60 N gripper limit, 2 kg objects were mostly
    # unsolvable by every expert variant.
    object_mass_max: float = 1.2
    object_table_friction_min: float = 0.05
    object_table_friction_max: float = 1.5
    object_finger_friction_min: float = 0.3
    object_finger_friction_max: float = 2.0
    # Newtons per finger; None keeps MetaWorld's unlimited servo, under which
    # mass and friction barely affect the grasp. Pick a value with
    # moebius-sweep.
    # Force control is an opt-in PEARL experiment; regular PEARL keeps
    # MetaWorld's original unlimited position servo.
    gripper_force_limit: float | None = None
    # Before contact, a positive gripper action closes at a fixed full drive.
    # After contact, its magnitude is the requested squeeze as a fraction of
    # gripper_force_limit. This separates approach speed from holding force.
    gripper_force_control: bool = False
    # Reward penalty for squeezing harder than the object needs, per step at
    # a full squeeze beyond what it needs (MetaWorld's reward is at most 10
    # per step). The needed grip is calibrated per task during setup (see
    # pearl_grip_effort_levels). 0 disables the penalty.
    pearl_grip_penalty: float = 0.0
    # Grip commands tried, from lowest up, to find each task's lowest reliable
    # grip (4 of 5 expert successes), including a full grip. The reward
    # penalty supplies its own measurement tolerance, so calibration does not
    # add a second margin. Demonstrations keep a common full grip so the
    # task-agnostic behavior-cloning warm start has one consistent target; RL
    # then learns to reduce force from that reliable starting policy.
    pearl_grip_effort_levels: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.7, 0.8, 0.9, 1.0)
    # Put the forces on the object (from the fingers, from the table, and the
    # grip squeeze) into observation slots 11:18, which pick-place leaves
    # empty. MetaWorld moves the hand kinematically, so without them mass and
    # friction are invisible until the object slips. PEARL runs only.
    contact_observations: bool = False
    # Ordered grasp strategies tried for the optional single-environment
    # force-control experiment. Normal multi-environment training uses one
    # dedicated MetaWorld policy per environment instead.
    pearl_expert_variants: tuple[str, ...] = ("careful", "slow_transport", "low_grasp", "metaworld")
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
    # Auxiliary encoder objective (not in original PEARL): a decoder predicts
    # each transition's object motion, gripper change and reward from
    # (s, a, z), and its error trains the encoder alongside the critic loss.
    # 0 disables it.
    pearl_decoder_weight: float = 0.0
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
    # Independent adaptation sequences per held-out task at each evaluation.
    # More sequences reduce noise but cost (adaptation_episodes + 1) episodes
    # per task each.
    pearl_eval_sequences_per_task: int = 1
    max_episode_steps: int = 500
    eval_interval: int = 10_000
    eval_episodes: int = 10
    log_dir: Path = Path("runs")
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "auto"
