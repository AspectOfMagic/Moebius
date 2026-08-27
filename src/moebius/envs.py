from __future__ import annotations

import gymnasium as gym
import numpy as np


class MetaWorldSingleTaskEnv(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Create one goal-observable MetaWorld task through Gymnasium.

    MT1 is deliberately used for the SAC baseline. It keeps the goal in the
    state and has no latent task inference component; PEARL will replace this
    simple setting in Phase 2.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_name: str = "reach-v3",
        seed: int = 1,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
    ) -> None:
        try:
            import metaworld  # noqa: F401 - importing registers Gymnasium environments
        except ImportError as error:
            raise ImportError(
                "MetaWorld is required. Install project dependencies with "
                "`pip install -e '.[dev]'`."
            ) from error

        environment = gym.make(
            "Meta-World/MT1",
            env_name=env_name,
            seed=seed,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
        super().__init__(environment)
