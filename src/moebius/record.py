from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from moebius.envs import MetaWorldSingleTaskEnv
from moebius.networks import SquashedGaussianActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record deterministic SAC evaluation episodes as MP4 files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
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


@torch.no_grad()
def select_action(actor: SquashedGaussianActor, observation: np.ndarray, device: str) -> np.ndarray:
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    action, _ = actor(observation_tensor, deterministic=True)
    return action.squeeze(0).cpu().numpy()


def record(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    environment = MetaWorldSingleTaskEnv(
        env_name=config["env_name"],
        seed=int(config["seed"]),
        max_episode_steps=int(config["max_episode_steps"]),
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )
    actor = load_actor(checkpoint, environment, device)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    try:
        for episode in range(args.episodes):
            observation, _ = environment.reset(seed=int(config["seed"]) + episode)
            video_path = args.video_dir / f"{config['env_name']}_episode_{episode + 1}.mp4"
            episode_return = 0.0
            terminated = truncated = False
            with imageio.get_writer(video_path, fps=args.fps) as writer:
                frame = environment.render()
                if frame is None:
                    raise RuntimeError("MetaWorld did not return an RGB frame. Ensure rendering dependencies are installed.")
                writer.append_data(frame)
                while not (terminated or truncated):
                    action = select_action(actor, observation, device)
                    observation, reward, terminated, truncated, _ = environment.step(action)
                    episode_return += reward
                    writer.append_data(environment.render())
            print(f"saved {video_path} (return={episode_return:.1f})")
    finally:
        environment.close()


def main() -> None:
    record(parse_args())


if __name__ == "__main__":
    main()
