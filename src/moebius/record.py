from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from moebius.envs import DynamicsTaskSampler, GoalTaskSampler, MetaWorldSingleTaskEnv, ObjectPropertyBounds, TaskSpec
from moebius.networks import SquashedGaussianActor
from moebius.context import ContextBuilder, episode_batch
from moebius.pearl import PEARLAgent
from moebius.replay_buffer import Batch


UPSIDE_DOWN_CAMERAS = frozenset({"corner"})


def checkpoint_environment_options(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    """Environment features saved by PEARL; SAC always uses its baseline dynamics."""
    if algorithm != "pearl":
        return {}
    return {
        "gripper_force_limit": config.get("gripper_force_limit"),
        "contact_observations": bool(config.get("contact_observations", False)),
        "gripper_force_control": bool(config.get("gripper_force_control", False)),
        "grip_penalty": float(config.get("pearl_grip_penalty", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record SAC or PEARL evaluation episodes as MP4 files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3, help="Number of SAC evaluation episodes to record.")
    parser.add_argument(
        "--adaptation-episodes",
        type=int,
        default=None,
        help="PEARL context-collection episodes; defaults to the checkpoint's training configuration.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index of the held-out PEARL task configuration to record.",
    )
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera-name",
        choices=("corner", "topview", "behindGripper", "gripperPOV", "corner2", "corner3", "corner4"),
        default="corner",
        help="MetaWorld camera to use. 'corner' frames are rotated upright automatically.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a PyTorch device such as 'cuda'.")
    return parser.parse_args()


def load_actor(
    checkpoint: dict[str, Any], environment: MetaWorldSingleTaskEnv, device: str
) -> SquashedGaussianActor:
    config = checkpoint["config"]
    observation_dim = int(np.prod(environment.observation_space.shape))
    action_dim = int(np.prod(environment.action_space.shape))
    action_low = torch.as_tensor(environment.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(environment.action_space.high, dtype=torch.float32, device=device)
    actor = SquashedGaussianActor(
        observation_dim,
        action_dim,
        int(config["hidden_dim"]),
        action_low,
        action_high,
    ).to(device)
    actor.load_state_dict(checkpoint["agent"]["actor"])
    actor.eval()
    return actor


def load_pearl_agent(checkpoint: dict[str, Any], environment: MetaWorldSingleTaskEnv, device: str) -> PEARLAgent:
    """Recreate a PEARL agent, including its task encoder, from a checkpoint."""
    config = checkpoint["config"]
    observation_dim = int(np.prod(environment.observation_space.shape))
    agent = PEARLAgent(
        observation_dim=observation_dim,
        action_low=environment.action_space.low,
        action_high=environment.action_space.high,
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        reward_scale=float(config.get("pearl_reward_scale", 1.0)),
        # State restoration below supplies the saved alpha value.
        fixed_alpha=None if config.get("pearl_fixed_alpha") is None else float(config["pearl_fixed_alpha"]),
        gradient_clip_norm=float(config.get("gradient_clip_norm", 10.0)),
        min_posterior_std=float(config.get("min_posterior_std", 0.05)),
        kl_weight=float(config.get("kl_weight", 0.05)),
        kl_free_bits=float(config.get("kl_free_bits", 0.1)),
        critic_layer_norm=bool(config.get("pearl_critic_layer_norm", False)),
        decoder_weight=float(config.get("pearl_decoder_weight", 0.0)),
        device=device,
    )
    state = checkpoint["agent"]
    for name in ("encoder", "actor", "q1", "q2", "target_q1", "target_q2"):
        getattr(agent, name).load_state_dict(state[name])
        getattr(agent, name).eval()
    agent.log_alpha.data.copy_(torch.as_tensor(state["log_alpha"], dtype=torch.float32, device=agent.device))
    return agent


@torch.no_grad()
def select_action(actor: SquashedGaussianActor, observation: np.ndarray, device: str) -> np.ndarray:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action, _ = actor(observation_tensor, deterministic=True)
    return action.squeeze(0).cpu().numpy()


def _reset_task(environment: MetaWorldSingleTaskEnv, task: TaskSpec) -> np.ndarray:
    environment.set_task_properties(task)
    if task.reset_seed is None:
        observation, _ = environment.reset()
    else:
        observation, _ = environment.reset(seed=task.reset_seed)
    return observation


def _append_frame(writer: imageio.Writer, environment: MetaWorldSingleTaskEnv, rotate_180: bool = False) -> None:
    frame = environment.render()
    if frame is None:
        raise RuntimeError("MetaWorld did not return an RGB frame. Ensure rendering dependencies are installed.")
    if rotate_180:
        frame = np.ascontiguousarray(frame[::-1, ::-1])
    writer.append_data(frame)


def record_sac(
    actor: SquashedGaussianActor, environment: MetaWorldSingleTaskEnv, config: dict[str, Any], args: argparse.Namespace, device: str
) -> None:
    rotate_180 = args.camera_name in UPSIDE_DOWN_CAMERAS
    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=int(config["seed"]) + episode)
        video_path = args.video_dir / f"{config['env_name']}_episode_{episode + 1}.mp4"
        episode_return = 0.0
        terminated = truncated = False
        with imageio.get_writer(video_path, fps=args.fps) as writer:
            _append_frame(writer, environment, rotate_180)
            while not (terminated or truncated):
                action = select_action(actor, observation, device)
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += reward
                _append_frame(writer, environment, rotate_180)
        print(f"saved {video_path} (return={episode_return:.1f})")


def record_pearl_episode(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    task: TaskSpec,
    previous_episodes: list[Batch],
    builder: ContextBuilder,
    rng: np.random.Generator,
    video_path: Path,
    fps: int,
    rotate_180: bool = False,
) -> tuple[float, bool]:
    """Record one rollout and append it to the adaptation history."""
    observation = _reset_task(environment, task)
    episode_latent, _ = agent.latent_for_episode(previous_episodes, builder, rng, deterministic=False)
    transitions = []
    episode_return = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}
    with imageio.get_writer(video_path, fps=fps) as writer:
        _append_frame(writer, environment, rotate_180)
        while not (terminated or truncated):
            action = agent.act(observation, deterministic=True, latent=episode_latent)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            transitions.append((observation, action, reward, next_observation, terminated))
            observation = next_observation
            episode_return += reward
            _append_frame(writer, environment, rotate_180)
    previous_episodes.append(episode_batch(*zip(*transitions, strict=True)))
    return episode_return, bool(info.get("success", False))


def record_pearl(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    config: dict[str, Any],
    args: argparse.Namespace,
    task_specs: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    if task_specs is not None:
        test_tasks = tuple(TaskSpec.from_dict(task) for task in task_specs["test"])
    elif config.get("pearl_task_distribution", "goal") == "dynamics":
        sampler = DynamicsTaskSampler(
            int(config["num_train_tasks"]),
            int(config["num_eval_tasks"]),
            ObjectPropertyBounds.from_config(config),
            seed=int(config["seed"]),
            env_names=tuple(config.get("pearl_env_names", (config["env_name"],))),
            vary_dynamics=bool(config.get("pearl_vary_object_dynamics", True)),
        )
        test_tasks = sampler.test_tasks
    else:
        sampler = GoalTaskSampler(
            num_train_tasks=int(config["num_train_tasks"]),
            num_test_tasks=int(config["num_eval_tasks"]),
            seed=int(config["seed"]),
        )
        test_tasks = sampler.test_tasks
    if not 0 <= args.task_index < len(test_tasks):
        raise ValueError(f"--task-index must be in [0, {len(test_tasks) - 1}].")
    adaptation_episodes = (
        int(config["adaptation_episodes"]) if args.adaptation_episodes is None else args.adaptation_episodes
    )
    if adaptation_episodes < 0:
        raise ValueError("--adaptation-episodes must be non-negative.")

    task = test_tasks[args.task_index]
    previous_episodes: list[Batch] = []
    builder = ContextBuilder(
        context_size=int(config["context_size"]),
        motion_fraction=float(config.get("context_motion_fraction", 0.75)),
        motion_threshold=float(config.get("context_motion_threshold", 1e-3)),
    )
    rng = np.random.default_rng(int(config["seed"]) + 60_000)
    name_prefix = f"{task.env_name or config['env_name']}_pearl_task{args.task_index}"
    rotate_180 = args.camera_name in UPSIDE_DOWN_CAMERAS
    for episode in range(adaptation_episodes):
        video_path = args.video_dir / f"{name_prefix}_context_{episode + 1}.mp4"
        episode_return, success = record_pearl_episode(
            agent, environment, task, previous_episodes, builder, rng, video_path, args.fps, rotate_180=rotate_180
        )
        print(f"saved {video_path} (context={episode + 1}, return={episode_return:.1f}, success={success})")

    video_path = args.video_dir / f"{name_prefix}_adapted.mp4"
    episode_return, success = record_pearl_episode(
        agent, environment, task, previous_episodes, builder, rng, video_path, args.fps, rotate_180=rotate_180
    )
    print(f"saved {video_path} (adapted, return={episode_return:.1f}, success={success})")


def record(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    algorithm = config.get("algorithm", "sac")
    if algorithm not in ("sac", "pearl"):
        raise ValueError(f"Unsupported checkpoint algorithm: {algorithm}")
    environment_name = config["env_name"]
    if algorithm == "pearl" and checkpoint.get("task_specs") is not None:
        test_specs = checkpoint["task_specs"]["test"]
        if not 0 <= args.task_index < len(test_specs):
            raise ValueError(f"--task-index must be in [0, {len(test_specs) - 1}].")
        environment_name = test_specs[args.task_index].get("env_name") or environment_name
    environment = MetaWorldSingleTaskEnv(
        env_name=environment_name,
        seed=int(config["seed"]),
        max_episode_steps=int(config["max_episode_steps"]),
        render_mode="rgb_array",
        camera_name=args.camera_name,
        **checkpoint_environment_options(config, algorithm),
    )
    args.video_dir.mkdir(parents=True, exist_ok=True)

    try:
        if algorithm == "sac":
            record_sac(load_actor(checkpoint, environment, device), environment, config, args, device)
        else:
            record_pearl(
                load_pearl_agent(checkpoint, environment, device),
                environment,
                config,
                args,
                checkpoint.get("task_specs"),
            )
    finally:
        environment.close()


def main() -> None:
    record(parse_args())


if __name__ == "__main__":
    main()
