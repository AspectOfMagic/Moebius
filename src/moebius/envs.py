from __future__ import annotations

from dataclasses import dataclass

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
        camera_name: str | None = None,
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
            camera_name=camera_name,
        )
        super().__init__(environment)


@dataclass(frozen=True)
class TaskSpec:
    """A fixed goal configuration identified only by its reproducible reset seed."""

    task_id: int
    reset_seed: int


class GoalTaskSampler:
    """Create a simple ML1 task distribution for the first PEARL experiment.

    MetaWorld's goal-observable MT1 wrapper draws goal configurations from its
    reset RNG.  Resetting with one seed per task makes each goal configuration
    reproducible while keeping the task id and goal itself out of the policy
    interface.  Test seeds are disjoint from train seeds.
    """

    def __init__(self, num_train_tasks: int, num_test_tasks: int, seed: int = 1) -> None:
        if num_train_tasks < 1 or num_test_tasks < 1:
            raise ValueError("num_train_tasks and num_test_tasks must both be positive.")
        self.rng = np.random.default_rng(seed)
        reset_seeds = self.rng.integers(0, np.iinfo(np.int32).max, size=num_train_tasks + num_test_tasks)
        self.train_tasks = tuple(TaskSpec(task_id, int(reset_seeds[task_id])) for task_id in range(num_train_tasks))
        self.test_tasks = tuple(
            TaskSpec(task_id, int(reset_seeds[num_train_tasks + task_id])) for task_id in range(num_test_tasks)
        )

    def sample_train_task(self) -> TaskSpec:
        return self.train_tasks[int(self.rng.integers(len(self.train_tasks)))]
