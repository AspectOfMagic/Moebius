from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from moebius.config import TrainConfig
from moebius.context import AdaptationContextBuffer, ContextBuilder, episode_batch
from moebius.envs import (
    SHAPES,
    SHAPE_ENVIRONMENTS,
    DynamicsTaskSampler,
    GoalTaskSampler,
    MetaWorldSingleTaskEnv,
    MetaWorldTaskSuite,
    ObjectPropertyBounds,
    TaskSpec,
)
from moebius.experts import EXPERT_VARIANTS, PickPlaceExpert, dedicated_expert, expert_variant
from moebius.pearl import PEARLAgent, context_from_batch
from moebius.replay_buffer import (
    Batch,
    MultiTaskReplayBuffer,
    ReplayBuffer,
    TaskBatch,
    concatenate_batches,
    concatenate_task_batch_rl,
)
from moebius.sac import SACAgent


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SAC or PEARL on state-based MetaWorld manipulation tasks.")
    parser.add_argument("--algorithm", choices=("sac", "pearl"), default="sac")
    parser.add_argument(
        "--env-name",
        default="pick-place-v3",
        help="MetaWorld MT1 task (default: pick-place-v3).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=250_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=TrainConfig.max_episode_steps,
        help="Episode length (at most 500, MetaWorld's limit). Shorter episodes make slow, careful handling "
        "costly, so the best behavior differs between objects.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a PyTorch device such as 'cuda'.")
    parser.add_argument(
        "--demo-episodes",
        type=int,
        default=0,
        help="Successful scripted-expert rollouts to collect before SAC training; set to 0 for pure SAC.",
    )
    parser.add_argument(
        "--bc-pretrain-updates",
        type=int,
        default=10_000,
        help="Actor behavior-cloning updates on demonstrations before online SAC.",
    )
    parser.add_argument(
        "--demo-batch-fraction",
        type=float,
        default=0.25,
        help="Fraction of each online SAC update sampled from the permanent demonstration buffer.",
    )
    parser.add_argument(
        "--bc-weight",
        type=float,
        default=1.0,
        help="Initial behavior-cloning loss weight during online SAC fine-tuning.",
    )
    parser.add_argument(
        "--bc-decay-steps",
        type=int,
        default=250_000,
        help="Environment steps over which the online behavior-cloning loss decays to zero.",
    )
    parser.add_argument(
        "--pearl-task-distribution",
        choices=("goal", "dynamics"),
        default="dynamics",
        help="PEARL task distribution: object shapes across environments or legacy fixed goals.",
    )
    parser.add_argument(
        "--pearl-env-names",
        nargs="+",
        choices=SHAPE_ENVIRONMENTS,
        default=list(TrainConfig.pearl_env_names),
        help="Compatible MetaWorld environments shared by one PEARL agent.",
    )
    parser.add_argument(
        "--pearl-vary-object-dynamics",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.pearl_vary_object_dynamics,
        help="Also randomize object mass and friction. Disabled by default so tasks differ by environment and shape.",
    )
    parser.add_argument(
        "--pearl-demo-episodes-per-task",
        type=int,
        default=2,
        help="Expert rollout count per hidden-dynamics PEARL training task; set to 0 to disable demo warm-starting.",
    )
    parser.add_argument(
        "--pearl-demo-max-attempts-per-task",
        type=int,
        default=20,
        help="Maximum scripted-expert attempts used to collect the requested successful PEARL demonstrations per task.",
    )
    parser.add_argument(
        "--pearl-bc-pretrain-updates",
        type=int,
        default=5_000,
        help="PEARL actor/encoder behavior-cloning updates before online training.",
    )
    parser.add_argument(
        "--pearl-critic-pretrain-updates",
        type=int,
        default=5_000,
        help="Joint PEARL critic/actor/encoder updates on demonstrations before online interaction.",
    )
    parser.add_argument(
        "--pearl-bc-weight-floor",
        type=float,
        default=TrainConfig.pearl_bc_weight_floor,
        help="Minimum online PEARL behavior-cloning weight after decay. With force control, online BC anchors "
        "movement and grip direction but not force magnitude, so 1.0 preserves manipulation skill without "
        "opposing the grip penalty.",
    )
    parser.add_argument(
        "--pearl-reward-scale",
        type=float,
        default=0.1,
        help="Reward scale used by PEARL critics and context features.",
    )
    parser.add_argument(
        "--pearl-fixed-alpha",
        type=float,
        default=None,
        help="Fixed PEARL entropy coefficient in scaled-reward units. Omit to tune it automatically toward the target entropy.",
    )
    parser.add_argument(
        "--pearl-initial-alpha",
        type=float,
        default=0.02,
        help="Starting PEARL entropy coefficient (scaled-reward units) when it is tuned automatically.",
    )
    parser.add_argument(
        "--pearl-max-reward",
        type=float,
        default=10.0,
        help="Largest per-step environment reward (MetaWorld: 10). Caps PEARL's TD targets at the largest "
        "achievable return; set to 0 to disable.",
    )
    parser.add_argument(
        "--object-shapes",
        nargs="+",
        choices=SHAPES,
        default=list(TrainConfig.object_shapes),
        help="Object shapes for PEARL dynamics tasks, assigned round-robin to tasks.",
    )
    parser.add_argument(
        "--gripper-force-limit",
        type=float,
        default=TrainConfig.gripper_force_limit,
        help="Cap on each finger actuator's force in newtons for PEARL dynamics tasks. Without a cap, "
        "MetaWorld's fingers squeeze far harder than the objects weigh, so mass and friction barely matter. "
        "Choose a value with moebius-sweep.",
    )
    parser.add_argument(
        "--gripper-force-control",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.gripper_force_control,
        help="PEARL: close the fingers at a fixed drive before contact, then use the gripper action magnitude "
        "as requested squeeze force (a fraction of --gripper-force-limit).",
    )
    parser.add_argument(
        "--pearl-grip-penalty",
        type=float,
        default=TrainConfig.pearl_grip_penalty,
        help="Reward penalty per step for a full squeeze beyond what the object needs (calibrated per task "
        "during setup). 0 disables it.",
    )
    parser.add_argument(
        "--pearl-expert-variants",
        nargs="+",
        choices=tuple(EXPERT_VARIANTS),
        default=list(TrainConfig.pearl_expert_variants),
        help="Ordered expert strategies to try for the optional single-environment force-control experiment. "
        "Multi-environment training always uses its dedicated expert policies.",
    )
    parser.add_argument(
        "--pearl-replay-failed-demos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the scripted expert's failed attempts on accepted training tasks to the online replay and "
        "context buffers (never behavior-cloned).",
    )
    parser.add_argument(
        "--pearl-critic-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LayerNorm in PEARL's critics to limit value extrapolation.",
    )
    parser.add_argument(
        "--pearl-decoder-weight",
        type=float,
        default=TrainConfig.pearl_decoder_weight,
        help="Weight of the auxiliary loss that trains the encoder to predict object motion and reward "
        "(VariBAD-style). 0 disables it, leaving the original PEARL objective.",
    )
    parser.add_argument(
        "--contact-observations",
        action=argparse.BooleanOptionalAction,
        default=TrainConfig.contact_observations,
        help="PEARL: observe the forces on the object (fingers, table, grip squeeze) in pick-place's unused "
        "observation slots 11:18.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--min-posterior-std", type=float, default=0.05)
    parser.add_argument(
        "--context-sequences-per-task",
        type=int,
        default=4,
        help="Recent adaptation sequences kept per PEARL task for sampling encoder context.",
    )
    parser.add_argument(
        "--context-motion-fraction",
        type=float,
        default=0.75,
        help="Target share of context transitions in which the object moved.",
    )
    parser.add_argument(
        "--context-motion-threshold",
        type=float,
        default=1e-3,
        help="Per-step object displacement (m) that counts as the object moving.",
    )
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument(
        "--kl-free-bits",
        type=float,
        default=0.1,
        help="KL nats per latent dimension that are not penalized.",
    )
    parser.add_argument("--num-train-tasks", type=int, default=20)
    parser.add_argument("--num-eval-tasks", type=int, default=10)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--adaptation-episodes", type=int, default=3)
    parser.add_argument(
        "--pearl-eval-sequences-per-task",
        type=int,
        default=1,
        help="Adaptation sequences run per held-out task at each PEARL evaluation. Each sequence costs "
        "(adaptation_episodes + 1) episodes per task.",
    )
    args = parser.parse_args()
    values = vars(args)
    if not 1 <= values["max_episode_steps"] <= 500:
        parser.error("--max-episode-steps must be between 1 and 500 (MetaWorld's episode limit).")
    values["object_shapes"] = tuple(values["object_shapes"])
    values["pearl_env_names"] = tuple(values["pearl_env_names"])
    values["pearl_expert_variants"] = tuple(values["pearl_expert_variants"])
    return TrainConfig(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent: SACAgent, environment: MetaWorldSingleTaskEnv, episodes: int) -> tuple[float, float]:
    returns, successes = [], []
    for _ in range(episodes):
        observation, _ = environment.reset()
        terminated = truncated = False
        episode_return = 0.0
        info: dict[str, float] = {}
        while not (terminated or truncated):
            action = agent.act(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            episode_return += reward
        returns.append(episode_return)
        successes.append(float(info.get("success", False)))
    return float(np.mean(returns)), float(np.mean(successes))


def collect_demonstrations(
    config: TrainConfig,
    observation_dim: int,
    action_dim: int,
    writer: SummaryWriter,
) -> ReplayBuffer | None:
    """Collect successful pick-place demonstrations in a separate environment."""
    if config.demo_episodes == 0:
        return None
    if config.demo_episodes < 0:
        raise ValueError("--demo-episodes must be non-negative.")
    if config.env_name != "pick-place-v3":
        raise ValueError("Demonstration collection currently supports only --env-name pick-place-v3.")

    capacity = config.demo_episodes * config.max_episode_steps
    buffer = ReplayBuffer(observation_dim, action_dim, capacity, config.seed + 10_000)
    environment = MetaWorldSingleTaskEnv(config.env_name, config.seed + 2, config.max_episode_steps)
    expert = expert_variant("metaworld")
    successful_episodes = 0
    try:
        for episode in range(config.demo_episodes):
            observation, _ = environment.reset()
            terminated = truncated = False
            episode_return = 0.0
            info: dict[str, float] = {}
            while not (terminated or truncated):
                action = np.asarray(expert.get_action(observation), dtype=np.float32)
                # Store the executable action, not an expert's out-of-range command.
                action = np.clip(action, environment.action_space.low, environment.action_space.high)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                buffer.add(observation, action, reward, next_observation, terminated)
                observation = next_observation
                episode_return += reward
            success = float(info.get("success", False))
            successful_episodes += int(success)
            writer.add_scalar("demo/episode_return", episode_return, episode + 1)
            writer.add_scalar("demo/success", success, episode + 1)
    finally:
        environment.close()

    success_rate = successful_episodes / config.demo_episodes
    writer.add_scalar("demo/success_rate", success_rate, 0)
    if successful_episodes != config.demo_episodes:
        raise RuntimeError(
            f"Only {successful_episodes}/{config.demo_episodes} expert rollouts succeeded; refusing to use failed demonstrations."
        )
    return buffer


def demonstration_batch_sizes(batch_size: int, fraction: float) -> tuple[int, int]:
    if batch_size < 2:
        raise ValueError("--batch-size must be at least 2 when using demonstrations.")
    if not 0 < fraction < 1:
        raise ValueError("--demo-batch-fraction must be strictly between 0 and 1.")
    demo_size = min(batch_size - 1, max(1, round(batch_size * fraction)))
    return batch_size - demo_size, demo_size


def online_bc_weight(config: TrainConfig, step: int) -> float:
    if config.bc_weight < 0:
        raise ValueError("--bc-weight must be non-negative.")
    if config.bc_decay_steps < 1:
        raise ValueError("--bc-decay-steps must be positive.")
    return config.bc_weight * max(0.0, 1.0 - step / config.bc_decay_steps)


def train_sac(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device)
    environment = MetaWorldSingleTaskEnv(config.env_name, config.seed, config.max_episode_steps)
    evaluation_environment = MetaWorldSingleTaskEnv(config.env_name, config.seed + 1, config.max_episode_steps)
    observation_dim = int(np.prod(environment.observation_space.shape))
    action_dim = int(np.prod(environment.action_space.shape))
    agent = SACAgent(
        observation_dim=observation_dim,
        action_low=environment.action_space.low,
        action_high=environment.action_space.high,
        hidden_dim=config.hidden_dim,
        gamma=config.gamma,
        tau=config.tau,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        alpha_lr=config.alpha_lr,
        target_entropy=config.target_entropy,
        device=device,
    )
    buffer = ReplayBuffer(observation_dim, action_dim, config.replay_size, config.seed)
    run_name = f"{config.env_name}_sac_seed{config.seed}"
    writer = SummaryWriter(config.log_dir / run_name)
    writer.add_text("config", str(asdict(config)))

    demonstrations = collect_demonstrations(config, observation_dim, action_dim, writer)
    if demonstrations is not None:
        if config.bc_pretrain_updates < 0:
            raise ValueError("--bc-pretrain-updates must be non-negative.")
        if demonstrations.size < config.batch_size:
            raise ValueError(
                "Demonstration buffer is smaller than --batch-size; increase --demo-episodes or lower --batch-size."
            )
        for update in range(1, config.bc_pretrain_updates + 1):
            metrics = agent.behavior_clone(demonstrations.sample(config.batch_size))
            if update % 100 == 0 or update == config.bc_pretrain_updates:
                writer.add_scalar("pretrain/bc_loss", metrics["loss/bc"], update)
        writer.add_scalar("demo/transitions", demonstrations.size, 0)
        online_batch_size, demo_batch_size = demonstration_batch_sizes(config.batch_size, config.demo_batch_fraction)

    observation, _ = environment.reset(seed=config.seed)
    episode_return, episode_length, episode = 0.0, 0, 0
    for step in range(1, config.total_steps + 1):
        action = agent.act(observation) if demonstrations is not None or step > config.start_steps else environment.action_space.sample()
        next_observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        # Time limits bootstrap; true terminals do not.
        buffer.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        episode_return += reward
        episode_length += 1

        if done:
            writer.add_scalar("train/episode_return", episode_return, step)
            writer.add_scalar("train/episode_length", episode_length, step)
            writer.add_scalar("train/success", float(info.get("success", False)), step)
            observation, _ = environment.reset()
            episode_return, episode_length = 0.0, 0
            episode += 1

        minimum_online_batch = online_batch_size if demonstrations is not None else config.batch_size
        if step >= config.update_after and buffer.size >= minimum_online_batch and step % config.update_every == 0:
            for _ in range(config.update_every * config.updates_per_step):
                if demonstrations is None:
                    metrics = agent.update(buffer.sample(config.batch_size))
                else:
                    demo_batch = demonstrations.sample(demo_batch_size)
                    metrics = agent.update(
                        concatenate_batches(buffer.sample(online_batch_size), demo_batch),
                        demo_batch,
                        online_bc_weight(config, step),
                    )
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
            if demonstrations is not None:
                writer.add_scalar("schedule/bc_weight", online_bc_weight(config, step), step)

        if step % config.eval_interval == 0:
            evaluation_return, evaluation_success = evaluate(agent, evaluation_environment, config.eval_episodes)
            writer.add_scalar("eval/mean_return", evaluation_return, step)
            writer.add_scalar("eval/success_rate", evaluation_success, step)
            print(f"step={step:,} return={evaluation_return:.1f} success={evaluation_success:.2%}")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "agent": agent.state_dict()}, config.checkpoint_dir / f"{run_name}.pt")
    writer.close()
    environment.close()
    evaluation_environment.close()


def _reset_task(environment: MetaWorldSingleTaskEnv, task: TaskSpec) -> np.ndarray:
    environment.set_task_properties(task)
    if task.reset_seed is None:
        observation, _ = environment.reset()
    else:
        observation, _ = environment.reset(seed=task.reset_seed)
    return observation


def _pearl_task_sampler(config: TrainConfig) -> GoalTaskSampler | DynamicsTaskSampler:
    if config.pearl_task_distribution == "goal":
        return GoalTaskSampler(config.num_train_tasks, config.num_eval_tasks, config.seed)
    if config.pearl_task_distribution == "dynamics":
        if not config.pearl_env_names:
            raise ValueError("--pearl-env-names must contain at least one environment.")
        if (config.gripper_force_control or config.contact_observations or config.pearl_grip_penalty > 0) and (
            config.pearl_env_names != ("pick-place-v3",)
        ):
            raise ValueError("Force control, force penalties and contact observations require only pick-place-v3.")
        return DynamicsTaskSampler(
            config.num_train_tasks,
            config.num_eval_tasks,
            ObjectPropertyBounds.from_config(asdict(config)),
            seed=config.seed,
            env_names=config.pearl_env_names,
            vary_dynamics=config.pearl_vary_object_dynamics,
        )
    raise ValueError(f"Unsupported PEARL task distribution: {config.pearl_task_distribution}")


Trajectory = list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]

# A reliable grip succeeds in four of five paired attempts.
GRIP_SEARCH_ATTEMPTS = 5
GRIP_SEARCH_SUCCESSES = 4
GRIP_EFFORT_BANDS: tuple[tuple[str, frozenset[float]], ...] = (
    ("gentle", frozenset((0.1, 0.2, 0.35))),
    ("medium", frozenset((0.5, 0.7))),
    ("strong", frozenset((0.8,))),
)


def grip_band_for_task(task_id: int, shape_count: int) -> tuple[str, frozenset[float]]:
    """Cycle balanced grip bands across complete shape groups."""
    if shape_count < 1:
        raise ValueError("shape_count must be positive.")
    return GRIP_EFFORT_BANDS[(task_id // shape_count) % len(GRIP_EFFORT_BANDS)]


def demonstration_expert(expert: PickPlaceExpert, gripper_force_control: bool) -> PickPlaceExpert:
    """Use one reliable grip target when cloning cannot yet observe the hidden task."""
    return replace(expert, grip_effort=1.0) if gripper_force_control else expert


def lowest_reliable_grip(
    succeeds: Callable[[float, int], bool],
    levels: Sequence[float],
    attempts: int = GRIP_SEARCH_ATTEMPTS,
    required: int = GRIP_SEARCH_SUCCESSES,
) -> float | None:
    """Return the lowest level that reaches the required paired successes."""
    for level in levels:
        successes = 0
        for attempt in range(attempts):
            successes += bool(succeeds(level, attempt))
            if successes >= required:
                return level
            if successes + attempts - attempt - 1 < required:
                break
    return None


@dataclass
class FeasibleDynamicsData:
    sampler: DynamicsTaskSampler
    demonstrations: MultiTaskReplayBuffer | None
    contexts: AdaptationContextBuffer | None
    # Failed attempts are replayable experience, never demonstrations.
    failed_trajectories: dict[int, list[Trajectory]] = field(default_factory=dict)


def collect_feasible_dynamics_data(
    config: TrainConfig,
    sampler: DynamicsTaskSampler,
    observation_dim: int,
    action_dim: int,
    writer: SummaryWriter,
) -> FeasibleDynamicsData:
    """Find feasible tasks, calibrate grip when enabled, and retain demonstrations."""
    required_successes = config.pearl_demo_episodes_per_task
    max_attempts = config.pearl_demo_max_attempts_per_task
    if required_successes == 0:
        return FeasibleDynamicsData(sampler, None, None)
    if required_successes < 0:
        raise ValueError("--pearl-demo-episodes-per-task must be non-negative.")
    if max_attempts < required_successes:
        raise ValueError("--pearl-demo-max-attempts-per-task must be at least --pearl-demo-episodes-per-task.")
    if config.max_property_candidates_per_task < 1:
        raise ValueError("max_property_candidates_per_task must be positive.")

    demonstrations = MultiTaskReplayBuffer(
        (task.task_id for task in sampler.train_tasks),
        observation_dim,
        action_dim,
        required_successes * config.max_episode_steps,
        config.seed + 20_000,
    )
    # Rotations prevent every context prefix from using the same demonstrations.
    contexts = AdaptationContextBuffer(
        (task.task_id for task in sampler.train_tasks),
        observation_dim,
        action_dim,
        max_sequences_per_task=required_successes,
        seed=config.seed + 30_000,
    )
    environment = MetaWorldTaskSuite(
        config.pearl_env_names,
        config.seed + 3,
        config.max_episode_steps,
        gripper_force_limit=config.gripper_force_limit,
        contact_observations=config.contact_observations,
        gripper_force_control=config.gripper_force_control,
        grip_penalty=config.pearl_grip_penalty,
    )
    attempt_counters: dict[tuple[str, int], int] = defaultdict(int)

    def experts_for_task(task: TaskSpec) -> tuple[tuple[str, Any], ...]:
        env_name = task.env_name or config.env_name
        if config.gripper_force_control:
            if not config.pearl_expert_variants:
                raise ValueError("--pearl-expert-variants must contain at least one force-control expert.")
            return tuple((name, expert_variant(name)) for name in config.pearl_expert_variants)
        return ((env_name, dedicated_expert(env_name)),)

    def run_expert(
        task: TaskSpec, expert: Any, reset_seed: int | None = None
    ) -> tuple[Trajectory, float, float]:
        """Run one expert episode; return (trajectory, success, return)."""
        environment.set_task_properties(task)
        observation, _ = environment.reset(seed=task.reset_seed if reset_seed is None else reset_seed)
        terminated = truncated = False
        episode_return = 0.0
        info: dict[str, float] = {}
        trajectory: Trajectory = []
        while not (terminated or truncated):
            action = np.asarray(expert.get_action(observation), dtype=np.float32)
            action = np.clip(action, environment.action_space.low, environment.action_space.high)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            trajectory.append((observation, action, reward, next_observation, terminated))
            observation = next_observation
            episode_return += reward
        return trajectory, float(info.get("success", False)), episode_return

    def expert_trajectories(
        task: TaskSpec, split: str, expert: Any
    ) -> tuple[list[Trajectory], list[Trajectory], int]:
        """Return (successes, failures, attempts) for one object and expert variant."""
        trajectories: list[Trajectory] = []
        failures: list[Trajectory] = []
        for attempt in range(1, max_attempts + 1):
            attempt_counters[(split, task.task_id)] += 1
            logged_attempt = attempt_counters[(split, task.task_id)]
            trajectory, success, episode_return = run_expert(task, expert)
            metric_prefix = "demo" if split == "train" else "feasibility"
            writer.add_scalar(f"{metric_prefix}/{split}_task_{task.task_id}_return", episode_return, logged_attempt)
            writer.add_scalar(f"{metric_prefix}/{split}_task_{task.task_id}_success", success, logged_attempt)
            if success:
                trajectories.append(trajectory)
                if len(trajectories) == required_successes:
                    return trajectories, failures, attempt
            else:
                failures.append(trajectory)
        return trajectories, failures, max_attempts

    failed_trajectories: dict[int, list[Trajectory]] = {}
    chosen_experts: dict[str, list[str]] = {"train": [], "test": []}

    def calibrated_grip(task: TaskSpec, split: str, expert: PickPlaceExpert) -> float:
        """Find grip effort on paired resets known to be solvable at full grip."""
        split_offset = 0 if split == "train" else 1_000_000
        seed_start = config.seed + 100_000 + split_offset + task.task_id * 1_000
        reference_seeds: list[int] = []
        # Compare force levels on resets already proven solvable at full grip.
        for offset in range(max_attempts * 3):
            reset_seed = seed_start + offset
            if run_expert(task, replace(expert, grip_effort=1.0), reset_seed)[1] > 0:
                reference_seeds.append(reset_seed)
                if len(reference_seeds) == GRIP_SEARCH_ATTEMPTS:
                    break
        if len(reference_seeds) < GRIP_SEARCH_SUCCESSES:
            return 1.0
        level = lowest_reliable_grip(
            lambda effort, attempt: run_expert(
                task, replace(expert, grip_effort=effort), reference_seeds[attempt]
            )[1]
            > 0,
            config.pearl_grip_effort_levels,
            attempts=len(reference_seeds),
            required=GRIP_SEARCH_SUCCESSES,
        )
        return 1.0 if level is None else level

    def solve(
        task: TaskSpec, split: str, allowed_grips: frozenset[float] | None = None
    ) -> tuple[str | None, TaskSpec, list[Trajectory], list[Trajectory], int]:
        """Return the first feasible expert and its calibrated demonstrations."""
        all_failures: list[Trajectory] = []
        total_attempts = 0
        for name, expert in experts_for_task(task):
            trajectories, failures, attempts = expert_trajectories(task, split, expert)
            all_failures.extend(failures)
            total_attempts += attempts
            if len(trajectories) < required_successes:
                continue
            if not config.gripper_force_control:
                return name, task, trajectories, all_failures, total_attempts

            grip = calibrated_grip(task, split, expert)
            if allowed_grips is not None and not any(np.isclose(grip, allowed) for allowed in allowed_grips):
                continue
            gripped_task = replace(task, object_grip_effort=grip)
            trajectories, failures, attempts = expert_trajectories(
                gripped_task, split, demonstration_expert(expert, config.gripper_force_control)
            )
            all_failures.extend(failures)
            total_attempts += attempts
            if len(trajectories) == required_successes:
                return name, gripped_task, trajectories, all_failures, total_attempts
        return None, task, [], all_failures, total_attempts

    def accept_tasks(tasks: tuple[TaskSpec, ...], split: str, retain: bool) -> tuple[TaskSpec, ...]:
        accepted_tasks: list[TaskSpec] = []
        for original_task in tasks:
            grip_band: str | None = None
            allowed_grips: frozenset[float] | None = None
            if config.gripper_force_control:
                grip_band, allowed_grips = grip_band_for_task(original_task.task_id, len(config.object_shapes))
            candidate = original_task
            for candidate_index in range(1, config.max_property_candidates_per_task + 1):
                variant_name, candidate, trajectories, failures, attempts = solve(candidate, split, allowed_grips)
                if variant_name is not None:
                    accepted_tasks.append(candidate)
                    grip = "" if candidate.object_grip_effort is None else f", grip {candidate.object_grip_effort:.2f}"
                    band = "" if grip_band is None else f", {grip_band} band"
                    chosen_experts[split].append(
                        f"task {candidate.task_id} ({candidate.env_name}, {candidate.shape}): "
                        f"{variant_name}{grip}{band}"
                    )
                    print(
                        f"feasibility {split} task={candidate.task_id} env={candidate.env_name} "
                        f"shape={candidate.shape}: accepted candidate "
                        f"{candidate_index}/{config.max_property_candidates_per_task} with expert "
                        f"'{variant_name}'{grip}{band} ({required_successes}/{attempts} successes)"
                    )
                    if retain:
                        episodes = []
                        for trajectory in trajectories:
                            for transition in trajectory:
                                demonstrations.add(candidate.task_id, *transition)
                            episodes.append(episode_batch(*zip(*trajectory, strict=True)))
                        for offset in range(len(episodes)):
                            contexts.add_sequence(candidate.task_id, episodes[offset:] + episodes[:offset])
                        # Keep failures only for the accepted object.
                        failed_trajectories[candidate.task_id] = failures
                    writer.add_scalar(f"demo/{split}_task_{candidate.task_id}_attempts", attempts, 0)
                    break
                candidate = sampler.resample_task(original_task)
            else:
                raise RuntimeError(
                    f"Could not find a {split} {original_task.env_name}/{original_task.shape} task with "
                    f"{required_successes} successful "
                    f"expert rollouts{'' if grip_band is None else f' in the {grip_band} grip band'} after "
                    f"{config.max_property_candidates_per_task} candidates. Relax the property "
                    "bounds, add expert variants, or increase max_property_candidates_per_task."
                )
        return tuple(accepted_tasks)

    try:
        sampler.train_tasks = accept_tasks(sampler.train_tasks, "train", retain=True)
        sampler.test_tasks = accept_tasks(sampler.test_tasks, "test", retain=False)
    finally:
        environment.close()

    writer.add_scalar("demo/pearl_transitions", sum(demonstrations.size(task.task_id) for task in sampler.train_tasks), 0)
    writer.add_scalar("demo/pearl_failed_episodes", sum(len(failures) for failures in failed_trajectories.values()), 0)
    for split, entries in chosen_experts.items():
        writer.add_text(f"tasks/{split}_experts", "\n\n".join(entries))
    return FeasibleDynamicsData(sampler, demonstrations, contexts, failed_trajectories)


def contextualize_task_batch(
    task_batch: TaskBatch,
    contexts: AdaptationContextBuffer,
    builder: ContextBuilder,
    max_previous_episodes: int,
) -> tuple[TaskBatch, int]:
    """Replace random replay context with prior adaptation episodes."""
    context, previous_episodes = contexts.sample(task_batch.task_ids, builder, max_previous_episodes)
    return TaskBatch(task_ids=task_batch.task_ids, context=context, rl=task_batch.rl), previous_episodes


def context_builder(config: TrainConfig) -> ContextBuilder:
    return ContextBuilder(
        context_size=config.context_size,
        motion_fraction=config.context_motion_fraction,
        motion_threshold=config.context_motion_threshold,
    )


def pearl_online_bc_weight(config: TrainConfig, step: int) -> float:
    if not 0 <= config.pearl_bc_weight_floor <= config.bc_weight:
        raise ValueError("--pearl-bc-weight-floor must be between zero and --bc-weight.")
    if config.bc_decay_steps < 1:
        raise ValueError("--bc-decay-steps must be positive.")
    decayed = config.bc_weight * max(0.0, 1.0 - step / config.bc_decay_steps)
    return max(config.pearl_bc_weight_floor, decayed)


@dataclass
class PearlEpisodeResult:
    episode_return: float
    success: float
    episode: Batch
    context: Batch | None
    posterior_std: float


def _pearl_episode(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    task: TaskSpec,
    previous_episodes: list[Batch],
    builder: ContextBuilder,
    rng: np.random.Generator,
) -> PearlEpisodeResult:
    """Run one deterministic-policy episode with a posterior-sampled latent."""
    observation = _reset_task(environment, task)
    episode_latent, context = agent.latent_for_episode(previous_episodes, builder, rng, deterministic=False)
    posterior_std = 1.0
    if context is not None:
        _, std = agent.posterior(context_from_batch(context, agent.reward_scale))
        posterior_std = float(std.mean())
    terminated = truncated = False
    episode_return = 0.0
    info: dict[str, float] = {}
    transitions = []
    while not (terminated or truncated):
        action = agent.act(observation, deterministic=True, latent=episode_latent)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        transitions.append((observation, action, reward, next_observation, terminated))
        observation = next_observation
        episode_return += reward
    return PearlEpisodeResult(
        episode_return,
        float(info.get("success", False)),
        episode_batch(*zip(*transitions, strict=True)),
        context,
        posterior_std,
    )


@dataclass
class PearlEvaluation:
    """Averages per adaptation step (index k = number of earlier episodes)."""

    returns: np.ndarray
    successes: np.ndarray
    posterior_stds: np.ndarray
    context_motion_fractions: np.ndarray
    task_successes: np.ndarray
    episodes_per_step: int


def evaluate_pearl(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    tasks: tuple[TaskSpec, ...],
    adaptation_episodes: int,
    builder: ContextBuilder,
    rng: np.random.Generator,
    sequences_per_task: int = 1,
) -> PearlEvaluation:
    """Measure performance and posterior width across adaptation steps."""
    if sequences_per_task < 1:
        raise ValueError("sequences_per_task must be positive.")
    steps = adaptation_episodes + 1
    returns, successes = np.zeros(steps), np.zeros(steps)
    posterior_stds, motion_fractions = np.zeros(steps), np.zeros(steps)
    task_successes = np.zeros(len(tasks))
    for task_index, task in enumerate(tasks):
        for _ in range(sequences_per_task):
            previous_episodes: list[Batch] = []
            for adaptation_step in range(steps):
                result = _pearl_episode(agent, environment, task, previous_episodes, builder, rng)
                previous_episodes.append(result.episode)
                returns[adaptation_step] += result.episode_return
                successes[adaptation_step] += result.success
                task_successes[task_index] += result.success
                posterior_stds[adaptation_step] += result.posterior_std
                if result.context is not None:
                    motion_fractions[adaptation_step] += builder.motion_fraction_of(result.context)
    count = len(tasks) * sequences_per_task
    return PearlEvaluation(
        returns / count,
        successes / count,
        posterior_stds / count,
        motion_fractions / count,
        task_successes / (sequences_per_task * steps),
        count,
    )


def seed_failed_expert_attempts(
    failed_trajectories: dict[int, list[Trajectory]],
    buffer: MultiTaskReplayBuffer,
    contexts: AdaptationContextBuffer,
    writer: SummaryWriter,
) -> int:
    """Replay failed expert attempts as experience and adaptation context."""
    transitions = 0
    for task_id, trajectories in failed_trajectories.items():
        if not trajectories:
            continue
        for trajectory in trajectories:
            for transition in trajectory:
                buffer.add(task_id, *transition)
            transitions += len(trajectory)
        contexts.add_sequence(task_id, [episode_batch(*zip(*trajectory, strict=True)) for trajectory in trajectories])
    writer.add_scalar("demo/pearl_failed_transitions_replayed", transitions, 0)
    print(f"replayed {transitions} transitions from failed expert attempts")
    return transitions


def train_pearl(config: TrainConfig) -> None:
    if config.meta_batch_size > config.num_train_tasks:
        raise ValueError("--meta-batch-size cannot exceed --num-train-tasks.")
    set_seed(config.seed)
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device)
    task_sampler = _pearl_task_sampler(config)
    environment_options = {
        "gripper_force_limit": config.gripper_force_limit,
        "contact_observations": config.contact_observations,
        "gripper_force_control": config.gripper_force_control,
        "grip_penalty": config.pearl_grip_penalty,
    }
    if isinstance(task_sampler, DynamicsTaskSampler):
        environment = MetaWorldTaskSuite(
            config.pearl_env_names, config.seed, config.max_episode_steps, **environment_options
        )
        evaluation_environment = MetaWorldTaskSuite(
            config.pearl_env_names, config.seed + 1_000, config.max_episode_steps, **environment_options
        )
        run_environment_name = "multi-pick-place"
    else:
        environment = MetaWorldSingleTaskEnv(
            config.env_name, config.seed, config.max_episode_steps, **environment_options
        )
        evaluation_environment = MetaWorldSingleTaskEnv(
            config.env_name, config.seed + 1, config.max_episode_steps, **environment_options
        )
        run_environment_name = config.env_name
    observation_dim = int(np.prod(environment.observation_space.shape))
    action_dim = int(np.prod(environment.action_space.shape))
    run_name = f"{run_environment_name}_pearl_seed{config.seed}"
    writer = SummaryWriter(config.log_dir / run_name)
    writer.add_text("config", str(asdict(config)))
    demonstration_data: tuple[MultiTaskReplayBuffer, AdaptationContextBuffer] | None = None
    failed_trajectories: dict[int, list[Trajectory]] = {}
    if isinstance(task_sampler, DynamicsTaskSampler):
        feasible_data = collect_feasible_dynamics_data(config, task_sampler, observation_dim, action_dim, writer)
        task_sampler = feasible_data.sampler
        failed_trajectories = feasible_data.failed_trajectories
        if feasible_data.demonstrations is not None and feasible_data.contexts is not None:
            demonstration_data = feasible_data.demonstrations, feasible_data.contexts
    for split, split_tasks in (("train", task_sampler.train_tasks), ("test", task_sampler.test_tasks)):
        writer.add_text(f"tasks/{split}", "\n\n".join(str(asdict(task)) for task in split_tasks))
    agent = PEARLAgent(
        observation_dim=observation_dim,
        action_low=environment.action_space.low,
        action_high=environment.action_space.high,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        gamma=config.gamma,
        tau=config.tau,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        encoder_lr=config.encoder_lr,
        alpha_lr=config.alpha_lr,
        kl_weight=config.kl_weight,
        kl_free_bits=config.kl_free_bits,
        target_entropy=config.target_entropy,
        reward_scale=config.pearl_reward_scale,
        fixed_alpha=config.pearl_fixed_alpha,
        initial_alpha=config.pearl_initial_alpha,
        gradient_clip_norm=config.gradient_clip_norm,
        min_posterior_std=config.min_posterior_std,
        max_reward=config.pearl_max_reward if config.pearl_max_reward > 0 else None,
        critic_layer_norm=config.pearl_critic_layer_norm,
        decoder_weight=config.pearl_decoder_weight,
        device=device,
    )
    buffer = MultiTaskReplayBuffer(
        (task.task_id for task in task_sampler.train_tasks), observation_dim, action_dim, config.replay_size, config.seed
    )
    online_contexts = AdaptationContextBuffer(
        (task.task_id for task in task_sampler.train_tasks),
        observation_dim,
        action_dim,
        max_sequences_per_task=config.context_sequences_per_task,
        seed=config.seed + 40_000,
    )
    if config.pearl_replay_failed_demos:
        seed_failed_expert_attempts(failed_trajectories, buffer, online_contexts, writer)
    builder = context_builder(config)
    rollout_rng = np.random.default_rng(config.seed + 50_000)
    evaluation_rng = np.random.default_rng(config.seed + 60_000)
    demonstrations: MultiTaskReplayBuffer | None = None
    if demonstration_data is not None:
        demonstrations, demonstration_contexts = demonstration_data
        if config.pearl_bc_pretrain_updates < 0:
            raise ValueError("--pearl-bc-pretrain-updates must be non-negative.")
        for update in range(1, config.pearl_bc_pretrain_updates + 1):
            metrics = agent.behavior_clone(
                demonstrations.sample_meta_batch(config.meta_batch_size, config.batch_size, 1)
            )
            if update % 100 == 0 or update == config.pearl_bc_pretrain_updates:
                writer.add_scalar("pretrain/pearl_bc_loss", metrics["loss/bc"], update)
        if config.pearl_critic_pretrain_updates < 0:
            raise ValueError("--pearl-critic-pretrain-updates must be non-negative.")
        for update in range(1, config.pearl_critic_pretrain_updates + 1):
            demonstration_batch, _ = contextualize_task_batch(
                demonstrations.sample_meta_batch(config.meta_batch_size, config.batch_size, 1),
                demonstration_contexts,
                builder,
                config.adaptation_episodes,
            )
            metrics = agent.update(demonstration_batch, demonstration_batch, bc_weight=config.bc_weight)
            if update % 100 == 0 or update == config.pearl_critic_pretrain_updates:
                writer.add_scalar("pretrain/pearl_critic_loss", metrics["loss/critic"], update)
                if "loss/kl" in metrics:
                    writer.add_scalar("pretrain/pearl_kl_loss", metrics["loss/kl"], update)
                writer.add_scalar("pretrain/pearl_joint_bc_loss", metrics["loss/bc"], update)
        online_batch_size, demo_batch_size = demonstration_batch_sizes(config.batch_size, config.demo_batch_fraction)

    task = task_sampler.sample_train_task()
    observation = _reset_task(environment, task)
    online_contexts.begin_sequence(task.task_id)
    online_contexts.begin_episode(task.task_id)
    # Completed episodes determine the next rollout latent.
    sequence_episodes: list[Batch] = []
    episode_transitions: list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
    episode_latent, _ = agent.latent_for_episode(sequence_episodes, builder, rollout_rng)
    episodes_on_task = 0
    episode_return, episode_length, episode_grip_penalty = 0.0, 0, 0.0
    try:
        for step in range(1, config.total_steps + 1):
            if demonstrations is None and step <= config.start_steps:
                action = environment.action_space.sample()
            else:
                action = agent.act(observation, latent=episode_latent)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            done = terminated or truncated
            buffer.add(task.task_id, observation, action, reward, next_observation, terminated)
            online_contexts.add(task.task_id, observation, action, reward, next_observation, terminated)
            episode_transitions.append((observation, action, reward, next_observation, terminated))
            observation = next_observation
            episode_return += reward
            episode_length += 1
            episode_grip_penalty += info.get("grip_penalty", 0.0)

            if done:
                online_contexts.end_episode(task.task_id)
                sequence_episodes.append(episode_batch(*zip(*episode_transitions, strict=True)))
                episode_transitions = []
                writer.add_scalar("train/episode_return", episode_return, step)
                writer.add_scalar("train/episode_length", episode_length, step)
                writer.add_scalar("train/success", float(info.get("success", False)), step)
                writer.add_scalar("train/episode_grip_penalty", episode_grip_penalty, step)
                writer.add_scalar(f"train_task/task_{task.task_id}_success", float(info.get("success", False)), step)
                writer.add_scalar(f"train_task/task_{task.task_id}_return", episode_return, step)
                episodes_on_task += 1
                if episodes_on_task >= config.adaptation_episodes + 1:
                    task = task_sampler.sample_train_task()
                    sequence_episodes = []
                    episodes_on_task = 0
                    online_contexts.begin_sequence(task.task_id)
                observation = _reset_task(environment, task)
                online_contexts.begin_episode(task.task_id)
                episode_latent, rollout_context = agent.latent_for_episode(sequence_episodes, builder, rollout_rng)
                if rollout_context is not None:
                    writer.add_scalar(
                        "train/context_motion_fraction", builder.motion_fraction_of(rollout_context), step
                    )
                episode_return, episode_length, episode_grip_penalty = 0.0, 0, 0.0

            if step >= config.update_after and step % config.update_every == 0:
                try:
                    for _ in range(config.update_every * config.updates_per_step):
                        if demonstrations is None:
                            online_batch, previous_episodes = contextualize_task_batch(
                                buffer.sample_meta_batch(config.meta_batch_size, config.batch_size, 1),
                                online_contexts,
                                builder,
                                config.adaptation_episodes,
                            )
                            metrics = agent.update(online_batch)
                        else:
                            online_batch, previous_episodes = contextualize_task_batch(
                                buffer.sample_meta_batch(config.meta_batch_size, online_batch_size, 1),
                                online_contexts,
                                builder,
                                config.adaptation_episodes,
                            )
                            demo_batch = demonstrations.sample_meta_batch_for_task_ids(
                                online_batch.task_ids, demo_batch_size, 1
                            )
                            metrics = agent.update(
                                concatenate_task_batch_rl(online_batch, demo_batch),
                                demo_batch,
                                pearl_online_bc_weight(config, step),
                                # Force-control BC preserves direction, not magnitude.
                                clone_grip_magnitude=not config.gripper_force_control,
                            )
                except ValueError:
                    metrics = None
                if metrics is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, step)
                    writer.add_scalar("stats/context_previous_episodes", previous_episodes, step)
                    writer.add_scalar(
                        "stats/context_motion_fraction", builder.motion_fraction_of(online_batch.context), step
                    )
                    if demonstrations is not None:
                        writer.add_scalar("schedule/pearl_bc_weight", pearl_online_bc_weight(config, step), step)

            if step % config.eval_interval == 0:
                evaluation = evaluate_pearl(
                    agent,
                    evaluation_environment,
                    task_sampler.test_tasks,
                    config.adaptation_episodes,
                    builder,
                    evaluation_rng,
                    config.pearl_eval_sequences_per_task,
                )
                evaluation_successes = evaluation.successes
                writer.add_scalar(
                    "eval/adaptation_gain", evaluation.successes[-1] - evaluation.successes[0], step
                )
                writer.add_scalar("eval/success_mean", float(evaluation.successes.mean()), step)
                writer.add_scalar("eval/episodes_per_step", evaluation.episodes_per_step, step)
                for test_task, task_success in zip(task_sampler.test_tasks, evaluation.task_successes, strict=True):
                    writer.add_scalar(f"eval_task/task_{test_task.task_id}_success", task_success, step)
                environment_successes: dict[str, list[float]] = defaultdict(list)
                for test_task, task_success in zip(task_sampler.test_tasks, evaluation.task_successes, strict=True):
                    environment_successes[test_task.env_name or config.env_name].append(float(task_success))
                for env_name, successes in environment_successes.items():
                    writer.add_scalar(f"eval_env/{env_name}_success", float(np.mean(successes)), step)
                for adaptation_step in range(config.adaptation_episodes + 1):
                    writer.add_scalar(
                        f"eval/return_after_context_{adaptation_step}", evaluation.returns[adaptation_step], step
                    )
                    writer.add_scalar(
                        f"eval/success_after_context_{adaptation_step}", evaluation.successes[adaptation_step], step
                    )
                    if adaptation_step > 0:
                        writer.add_scalar(
                            f"eval/z_std_after_context_{adaptation_step}",
                            evaluation.posterior_stds[adaptation_step],
                            step,
                        )
                        writer.add_scalar(
                            f"eval/context_motion_fraction_{adaptation_step}",
                            evaluation.context_motion_fractions[adaptation_step],
                            step,
                        )
                print(
                    f"step={step:,} prior_success={evaluation_successes[0]:.2%} "
                    f"adapted_success={evaluation_successes[-1]:.2%}"
                )
    finally:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(config),
                "agent": agent.state_dict(),
                # Persist accepted tasks because feasibility filtering resamples them.
                "task_specs": {
                    "train": [asdict(task) for task in task_sampler.train_tasks],
                    "test": [asdict(task) for task in task_sampler.test_tasks],
                },
            },
            config.checkpoint_dir / f"{run_name}.pt",
        )
        writer.close()
        environment.close()
        evaluation_environment.close()


def train(config: TrainConfig) -> None:
    if config.algorithm == "sac":
        train_sac(config)
    elif config.algorithm == "pearl":
        train_pearl(config)
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
