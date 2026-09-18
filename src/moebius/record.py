from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from moebius.envs import DynamicsTaskSampler, GoalTaskSampler, MetaWorldSingleTaskEnv, TaskSpec
from moebius.networks import SquashedGaussianActor
from moebius.pearl import PEARLAgent, context_transition


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
        default="topview",
        help="MetaWorld camera to use. 'topview' provides the clearest overview of the object.",
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


def _append_context(context: list[np.ndarray], transition: np.ndarray, context_size: int) -> None:
    context.append(transition)
    if len(context) > context_size:
        del context[:-context_size]


def _reset_task(environment: MetaWorldSingleTaskEnv, task: TaskSpec) -> np.ndarray:
    environment.set_object_dynamics(task.mass_scale, task.friction_scale)
    if task.reset_seed is None:
        observation, _ = environment.reset()
    else:
        observation, _ = environment.reset(seed=task.reset_seed)
    return observation


def _append_frame(writer: imageio.Writer, environment: MetaWorldSingleTaskEnv) -> None:
    frame = environment.render()
    if frame is None:
        raise RuntimeError("MetaWorld did not return an RGB frame. Ensure rendering dependencies are installed.")
    writer.append_data(frame)


def record_sac(
    actor: SquashedGaussianActor, environment: MetaWorldSingleTaskEnv, config: dict[str, Any], args: argparse.Namespace, device: str
) -> None:
    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=int(config["seed"]) + episode)
        video_path = args.video_dir / f"{config['env_name']}_episode_{episode + 1}.mp4"
        episode_return = 0.0
        terminated = truncated = False
        with imageio.get_writer(video_path, fps=args.fps) as writer:
            _append_frame(writer, environment)
            while not (terminated or truncated):
                action = select_action(actor, observation, device)
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += reward
                _append_frame(writer, environment)
        print(f"saved {video_path} (return={episode_return:.1f})")


def record_pearl_episode(
    agent: PEARLAgent,
    environment: MetaWorldSingleTaskEnv,
    task: TaskSpec,
    context: list[np.ndarray],
    context_size: int,
    video_path: Path,
    fps: int,
    deterministic: bool,
) -> tuple[float, bool]:
    """Record one context-collection or adapted PEARL rollout on one task."""
    observation = _reset_task(environment, task)
    episode_return = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}
    with imageio.get_writer(video_path, fps=fps) as writer:
        _append_frame(writer, environment)
        while not (terminated or truncated):
            current_context = np.asarray(context, dtype=np.float32) if context else None
            action = agent.act(observation, current_context, deterministic=deterministic)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            _append_context(context, context_transition(observation, action, reward, next_observation), context_size)
            observation = next_observation
            episode_return += reward
            _append_frame(writer, environment)
    return episode_return, bool(info.get("success", False))


def record_pearl(
    agent: PEARLAgent, environment: MetaWorldSingleTaskEnv, config: dict[str, Any], args: argparse.Namespace
) -> None:
    if config.get("pearl_task_distribution", "goal") == "dynamics":
        sampler = DynamicsTaskSampler(
            num_train_tasks=int(config["num_train_tasks"]),
            num_test_tasks=int(config["num_eval_tasks"]),
            mass_scale_min=float(config.get("mass_scale_min", 0.5)),
            mass_scale_max=float(config.get("mass_scale_max", 1.5)),
            friction_scale_min=float(config.get("friction_scale_min", 0.5)),
            friction_scale_max=float(config.get("friction_scale_max", 1.5)),
            seed=int(config["seed"]),
        )
    else:
        sampler = GoalTaskSampler(
            num_train_tasks=int(config["num_train_tasks"]),
            num_test_tasks=int(config["num_eval_tasks"]),
            seed=int(config["seed"]),
        )
    if not 0 <= args.task_index < len(sampler.test_tasks):
        raise ValueError(f"--task-index must be in [0, {len(sampler.test_tasks) - 1}].")
    adaptation_episodes = (
        int(config["adaptation_episodes"]) if args.adaptation_episodes is None else args.adaptation_episodes
    )
    if adaptation_episodes < 0:
        raise ValueError("--adaptation-episodes must be non-negative.")

    task = sampler.test_tasks[args.task_index]
    context: list[np.ndarray] = []
    context_size = int(config["context_size"])
    name_prefix = f"{config['env_name']}_pearl_task{args.task_index}"
    for episode in range(adaptation_episodes):
        video_path = args.video_dir / f"{name_prefix}_context_{episode + 1}.mp4"
        episode_return, success = record_pearl_episode(
            agent, environment, task, context, context_size, video_path, args.fps, deterministic=False
        )
        print(f"saved {video_path} (context={episode + 1}, return={episode_return:.1f}, success={success})")

    video_path = args.video_dir / f"{name_prefix}_adapted.mp4"
    episode_return, success = record_pearl_episode(
        agent, environment, task, context, context_size, video_path, args.fps, deterministic=True
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
    environment = MetaWorldSingleTaskEnv(
        env_name=config["env_name"],
        seed=int(config["seed"]),
        max_episode_steps=int(config["max_episode_steps"]),
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )
    args.video_dir.mkdir(parents=True, exist_ok=True)

    try:
        if algorithm == "sac":
            record_sac(load_actor(checkpoint, environment, device), environment, config, args, device)
        else:
            record_pearl(load_pearl_agent(checkpoint, environment, device), environment, config, args)
    finally:
        environment.close()


def main() -> None:
    record(parse_args())


if __name__ == "__main__":
    main()
