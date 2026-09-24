from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from moebius.config import TrainConfig
from moebius.envs import DynamicsTaskSampler, GoalTaskSampler, MetaWorldSingleTaskEnv, TaskSpec
from moebius.pearl import PEARLAgent, context_transition
from moebius.replay_buffer import (
    MultiTaskReplayBuffer,
    ReplayBuffer,
    TaskBatch,
    TaskTrajectoryContextBuffer,
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
        help="PEARL task distribution: hidden object properties or legacy fixed goals.",
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
        default=0.1,
        help="Minimum online PEARL behavior-cloning weight after decay.",
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
        default=0.1,
        help="Fixed PEARL entropy coefficient; prevents automatic-temperature runaway.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--min-posterior-std", type=float, default=0.05)
    parser.add_argument(
        "--context-episodes-per-task",
        type=int,
        default=64,
        help="Completed chronological context episodes retained per PEARL task.",
    )
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--num-train-tasks", type=int, default=20)
    parser.add_argument("--num-eval-tasks", type=int, default=10)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--adaptation-episodes", type=int, default=3)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


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


def _pick_place_expert() -> object:
    """Create the MetaWorld scripted policy used only for Phase 3 demos."""
    try:
        from metaworld.policies import SawyerPickPlaceV3Policy
    except ImportError as error:
        raise ImportError("MetaWorld's expert policies are required for demonstration-augmented SAC.") from error
    return SawyerPickPlaceV3Policy()


def collect_demonstrations(
    config: TrainConfig,
    observation_dim: int,
    action_dim: int,
    writer: SummaryWriter,
) -> ReplayBuffer | None:
    """Collect permanent, successful pick-and-place expert trajectories.

    The expert environment is independent from evaluation and uses normal,
    unseeded resets after construction, so it samples the training reset
    distribution without exposing evaluation rollouts to the replay buffer.
    """
    if config.demo_episodes == 0:
        return None
    if config.demo_episodes < 0:
        raise ValueError("--demo-episodes must be non-negative.")
    if config.env_name != "pick-place-v3":
        raise ValueError("Demonstration collection currently supports only --env-name pick-place-v3.")

    capacity = config.demo_episodes * config.max_episode_steps
    buffer = ReplayBuffer(observation_dim, action_dim, capacity, config.seed + 10_000)
    environment = MetaWorldSingleTaskEnv(config.env_name, config.seed + 2, config.max_episode_steps)
    expert = _pick_place_expert()
    successful_episodes = 0
    try:
        for episode in range(config.demo_episodes):
            observation, _ = environment.reset()
            terminated = truncated = False
            episode_return = 0.0
            info: dict[str, float] = {}
            while not (terminated or truncated):
                action = np.asarray(expert.get_action(observation), dtype=np.float32)
                # Scripted policies may rely on MetaWorld to clip their
                # commands. Store the clipped action, since that is what the
                # environment actually executes and what the actor can emit.
                action = np.clip(action, environment.action_space.low, environment.action_space.high)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                # Preserve MetaWorld's terminal semantics; a success flag does
                # not necessarily end an episode in this environment.
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
        # Behavior cloning supplies a competent initial policy, replacing the
        # random-action warm-up used by pure SAC.
        action = agent.act(observation) if demonstrations is not None or step > config.start_steps else environment.action_space.sample()
        next_observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        # Time limits are not environmental terminals and should bootstrap.
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
        if config.env_name != "pick-place-v3":
            raise ValueError("The hidden-dynamics task distribution currently supports only pick-place-v3.")
        return DynamicsTaskSampler(
            config.num_train_tasks,
            config.num_eval_tasks,
            object_radius_min=config.object_radius_min,
            object_radius_max=config.object_radius_max,
            object_height_min=config.object_height_min,
            object_height_max=config.object_height_max,
            object_mass_min=config.object_mass_min,
            object_mass_max=config.object_mass_max,
            object_table_friction_min=config.object_table_friction_min,
            object_table_friction_max=config.object_table_friction_max,
            object_finger_friction_min=config.object_finger_friction_min,
            object_finger_friction_max=config.object_finger_friction_max,
            contact_softness_min=config.contact_softness_min,
            contact_softness_max=config.contact_softness_max,
            seed=config.seed,
        )
    raise ValueError(f"Unsupported PEARL task distribution: {config.pearl_task_distribution}")


Trajectory = list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]


@dataclass
class FeasibleDynamicsData:
    sampler: DynamicsTaskSampler
    demonstrations: MultiTaskReplayBuffer | None
    contexts: TaskTrajectoryContextBuffer | None


def collect_feasible_dynamics_data(
    config: TrainConfig,
    sampler: DynamicsTaskSampler,
    observation_dim: int,
    action_dim: int,
    writer: SummaryWriter,
) -> FeasibleDynamicsData:
    """Find solvable property vectors and retain their successful rollouts.

    A candidate must produce ``pearl_demo_episodes_per_task`` successes within
    ``pearl_demo_max_attempts_per_task`` attempts. Those same trajectories are
    committed to the demonstration/context buffers, so no second collection
    pass can fail merely because later resets are less favorable.
    """
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
    contexts = TaskTrajectoryContextBuffer(
        (task.task_id for task in sampler.train_tasks), config.context_episodes_per_task, config.seed + 30_000
    )
    environment = MetaWorldSingleTaskEnv(config.env_name, config.seed + 3, config.max_episode_steps)
    expert = _pick_place_expert()

    def successful_trajectories(task: TaskSpec, split: str) -> tuple[list[Trajectory], int]:
        trajectories: list[Trajectory] = []
        for attempt in range(1, max_attempts + 1):
            observation = _reset_task(environment, task)
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
            success = float(info.get("success", False))
            metric_prefix = "demo" if split == "train" else "feasibility"
            writer.add_scalar(f"{metric_prefix}/{split}_task_{task.task_id}_return", episode_return, attempt)
            writer.add_scalar(f"{metric_prefix}/{split}_task_{task.task_id}_success", success, attempt)
            if success:
                trajectories.append(trajectory)
                if len(trajectories) == required_successes:
                    return trajectories, attempt
        return trajectories, max_attempts

    def accept_tasks(tasks: tuple[TaskSpec, ...], split: str, retain: bool) -> tuple[TaskSpec, ...]:
        accepted_tasks: list[TaskSpec] = []
        for original_task in tasks:
            candidate = original_task
            for candidate_index in range(1, config.max_property_candidates_per_task + 1):
                trajectories, attempts = successful_trajectories(candidate, split)
                if len(trajectories) == required_successes:
                    accepted_tasks.append(candidate)
                    print(
                        f"feasibility {split} task={candidate.task_id}: accepted candidate "
                        f"{candidate_index}/{config.max_property_candidates_per_task} "
                        f"({required_successes}/{attempts} expert successes)"
                    )
                    if retain:
                        for trajectory in trajectories:
                            contexts.begin_episode(candidate.task_id)
                            for transition in trajectory:
                                demonstrations.add(candidate.task_id, *transition)
                                contexts.add(candidate.task_id, *transition)
                            contexts.end_episode(candidate.task_id)
                    writer.add_scalar(f"demo/{split}_task_{candidate.task_id}_attempts", attempts, 0)
                    break
                candidate = sampler.resample_task(original_task.task_id)
            else:
                raise RuntimeError(
                    f"Could not find a {split} task with {required_successes} successful expert rollouts after "
                    f"{config.max_property_candidates_per_task} property candidates. Relax the property bounds or "
                    "increase max_property_candidates_per_task."
                )
        return tuple(accepted_tasks)

    try:
        sampler.train_tasks = accept_tasks(sampler.train_tasks, "train", retain=True)
        sampler.test_tasks = accept_tasks(sampler.test_tasks, "test", retain=False)
    finally:
        environment.close()

    writer.add_scalar("demo/pearl_transitions", sum(demonstrations.size(task.task_id) for task in sampler.train_tasks), 0)
    return FeasibleDynamicsData(sampler, demonstrations, contexts)


def contextualize_task_batch(
    task_batch: TaskBatch, contexts: TaskTrajectoryContextBuffer, context_size: int
) -> TaskBatch:
    """Replace replay's unordered context with a rollout-like prefix context."""
    return TaskBatch(
        task_ids=task_batch.task_ids,
        context=contexts.sample_prefixes(task_batch.task_ids, context_size),
        rl=task_batch.rl,
    )


def pearl_online_bc_weight(config: TrainConfig, step: int) -> float:
    if not 0 <= config.pearl_bc_weight_floor <= config.bc_weight:
        raise ValueError("--pearl-bc-weight-floor must be between zero and --bc-weight.")
    if config.bc_decay_steps < 1:
        raise ValueError("--bc-decay-steps must be positive.")
    decayed = config.bc_weight * max(0.0, 1.0 - step / config.bc_decay_steps)
    return max(config.pearl_bc_weight_floor, decayed)


def _append_context(context: list[np.ndarray], transition: np.ndarray, context_size: int) -> np.ndarray:
    context.append(transition)
    if len(context) > context_size:
        del context[:-context_size]
    return np.asarray(context, dtype=np.float32)


def _pearl_episode(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    task: TaskSpec,
    context: list[np.ndarray],
    context_size: int,
    deterministic: bool,
    buffer: MultiTaskReplayBuffer | None = None,
) -> tuple[float, float]:
    observation = _reset_task(environment, task)
    initial_context = np.asarray(context, dtype=np.float32) if context else None
    episode_latent = agent.infer_latent(initial_context, deterministic=deterministic)
    terminated = truncated = False
    episode_return = 0.0
    info: dict[str, float] = {}
    while not (terminated or truncated):
        action = agent.act(observation, deterministic=deterministic, latent=episode_latent)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        if buffer is not None:
            # As in SAC, time-limit truncation should still bootstrap.
            buffer.add(task.task_id, observation, action, reward, next_observation, terminated)
        _append_context(
            context,
            context_transition(observation, action, reward, next_observation, agent.reward_scale),
            context_size,
        )
        observation = next_observation
        episode_return += reward
    return episode_return, float(info.get("success", False))


def evaluate_pearl(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    tasks: tuple[TaskSpec, ...],
    adaptation_episodes: int,
    context_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure return and success before/after accumulating adaptation context."""
    returns = np.zeros(adaptation_episodes + 1, dtype=np.float64)
    successes = np.zeros(adaptation_episodes + 1, dtype=np.float64)
    for task in tasks:
        context: list[np.ndarray] = []
        for adaptation_step in range(adaptation_episodes + 1):
            # The first rollout is prior-only. Subsequent rollouts use context
            # collected by previous exploratory episodes on the same task.
            deterministic = adaptation_step == adaptation_episodes
            episode_return, success = _pearl_episode(
                agent, environment, task, context, context_size, deterministic=deterministic
            )
            returns[adaptation_step] += episode_return
            successes[adaptation_step] += success
    return returns / len(tasks), successes / len(tasks)


def train_pearl(config: TrainConfig) -> None:
    if config.meta_batch_size > config.num_train_tasks:
        raise ValueError("--meta-batch-size cannot exceed --num-train-tasks.")
    set_seed(config.seed)
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device)
    environment = MetaWorldSingleTaskEnv(config.env_name, config.seed, config.max_episode_steps)
    evaluation_environment = MetaWorldSingleTaskEnv(config.env_name, config.seed + 1, config.max_episode_steps)
    task_sampler = _pearl_task_sampler(config)
    observation_dim = int(np.prod(environment.observation_space.shape))
    action_dim = int(np.prod(environment.action_space.shape))
    run_name = f"{config.env_name}_pearl_seed{config.seed}"
    writer = SummaryWriter(config.log_dir / run_name)
    writer.add_text("config", str(asdict(config)))
    demonstration_data: tuple[MultiTaskReplayBuffer, TaskTrajectoryContextBuffer] | None = None
    if isinstance(task_sampler, DynamicsTaskSampler):
        feasible_data = collect_feasible_dynamics_data(config, task_sampler, observation_dim, action_dim, writer)
        task_sampler = feasible_data.sampler
        if feasible_data.demonstrations is not None and feasible_data.contexts is not None:
            demonstration_data = feasible_data.demonstrations, feasible_data.contexts
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
        target_entropy=config.target_entropy,
        reward_scale=config.pearl_reward_scale,
        fixed_alpha=config.pearl_fixed_alpha,
        gradient_clip_norm=config.gradient_clip_norm,
        min_posterior_std=config.min_posterior_std,
        device=device,
    )
    buffer = MultiTaskReplayBuffer(
        (task.task_id for task in task_sampler.train_tasks), observation_dim, action_dim, config.replay_size, config.seed
    )
    online_contexts = TaskTrajectoryContextBuffer(
        (task.task_id for task in task_sampler.train_tasks), config.context_episodes_per_task, config.seed + 40_000
    )
    demonstrations: MultiTaskReplayBuffer | None = None
    if demonstration_data is not None:
        demonstrations, demonstration_contexts = demonstration_data
        if config.pearl_bc_pretrain_updates < 0:
            raise ValueError("--pearl-bc-pretrain-updates must be non-negative.")
        for update in range(1, config.pearl_bc_pretrain_updates + 1):
            demonstration_batch = contextualize_task_batch(
                demonstrations.sample_meta_batch(config.meta_batch_size, config.batch_size, config.context_size),
                demonstration_contexts,
                config.context_size,
            )
            metrics = agent.behavior_clone(demonstration_batch, prior_context=update % 4 == 0)
            if update % 100 == 0 or update == config.pearl_bc_pretrain_updates:
                writer.add_scalar("pretrain/pearl_bc_loss", metrics["loss/bc"], update)
        if config.pearl_critic_pretrain_updates < 0:
            raise ValueError("--pearl-critic-pretrain-updates must be non-negative.")
        for update in range(1, config.pearl_critic_pretrain_updates + 1):
            demonstration_batch = contextualize_task_batch(
                demonstrations.sample_meta_batch(config.meta_batch_size, config.batch_size, config.context_size),
                demonstration_contexts,
                config.context_size,
            )
            # Keep the cloned actor synchronized with the encoder while the
            # value functions warm up on demonstration-only data.
            metrics = agent.update(demonstration_batch, demonstration_batch, bc_weight=config.bc_weight)
            if update % 100 == 0 or update == config.pearl_critic_pretrain_updates:
                writer.add_scalar("pretrain/pearl_critic_loss", metrics["loss/critic"], update)
                writer.add_scalar("pretrain/pearl_kl_loss", metrics["loss/kl"], update)
                writer.add_scalar("pretrain/pearl_joint_bc_loss", metrics["loss/bc"], update)
        online_batch_size, demo_batch_size = demonstration_batch_sizes(config.batch_size, config.demo_batch_fraction)

    task = task_sampler.sample_train_task()
    observation = _reset_task(environment, task)
    online_contexts.begin_episode(task.task_id)
    task_context: list[np.ndarray] = []
    episode_latent = agent.infer_latent(None)
    episodes_on_task = 0
    episode_return, episode_length = 0.0, 0
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
            _append_context(
                task_context,
                context_transition(observation, action, reward, next_observation, agent.reward_scale),
                config.context_size,
            )
            observation = next_observation
            episode_return += reward
            episode_length += 1

            if done:
                online_contexts.end_episode(task.task_id)
                writer.add_scalar("train/episode_return", episode_return, step)
                writer.add_scalar("train/episode_length", episode_length, step)
                writer.add_scalar("train/success", float(info.get("success", False)), step)
                episodes_on_task += 1
                if episodes_on_task >= config.adaptation_episodes + 1:
                    task = task_sampler.sample_train_task()
                    task_context = []
                    episodes_on_task = 0
                observation = _reset_task(environment, task)
                online_contexts.begin_episode(task.task_id)
                current_context = np.asarray(task_context, dtype=np.float32) if task_context else None
                episode_latent = agent.infer_latent(current_context)
                episode_return, episode_length = 0.0, 0

            if step >= config.update_after and step % config.update_every == 0:
                try:
                    for _ in range(config.update_every * config.updates_per_step):
                        if demonstrations is None:
                            online_batch = contextualize_task_batch(
                                buffer.sample_meta_batch(config.meta_batch_size, config.batch_size, config.context_size),
                                online_contexts,
                                config.context_size,
                            )
                            metrics = agent.update(online_batch)
                        else:
                            online_batch = contextualize_task_batch(
                                buffer.sample_meta_batch(config.meta_batch_size, online_batch_size, config.context_size),
                                online_contexts,
                                config.context_size,
                            )
                            demo_batch = demonstrations.sample_meta_batch_for_task_ids(
                                online_batch.task_ids, demo_batch_size, config.context_size
                            )
                            metrics = agent.update(
                                concatenate_task_batch_rl(online_batch, demo_batch),
                                demo_batch,
                                pearl_online_bc_weight(config, step),
                            )
                except ValueError:
                    # Not enough task-specific data yet; keep collecting.
                    metrics = None
                if metrics is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, step)
                    if demonstrations is not None:
                        writer.add_scalar("schedule/pearl_bc_weight", pearl_online_bc_weight(config, step), step)

            if step % config.eval_interval == 0:
                evaluation_returns, evaluation_successes = evaluate_pearl(
                    agent,
                    evaluation_environment,
                    task_sampler.test_tasks,
                    config.adaptation_episodes,
                    config.context_size,
                )
                for adaptation_step, (evaluation_return, evaluation_success) in enumerate(
                    zip(evaluation_returns, evaluation_successes, strict=True)
                ):
                    writer.add_scalar(f"eval/return_after_context_{adaptation_step}", evaluation_return, step)
                    writer.add_scalar(f"eval/success_after_context_{adaptation_step}", evaluation_success, step)
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
                # Filtering replaces random candidates, so persist the exact
                # accepted vectors for reproducible evaluation and recording.
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
