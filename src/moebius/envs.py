from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


class MetaWorldSingleTaskEnv(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Create one goal-observable MetaWorld task through Gymnasium.

    MT1 keeps the object and goal state observable. For PEARL, a reset seed
    selects one reproducible pick-and-place configuration while its task
    identifier stays outside the agent observation.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_name: str = "pick-place-v3",
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
    """A fixed reset configuration identified only by its reproducible seed."""

    task_id: int
    reset_seed: int


class GoalTaskSampler:
    """Create seeded object-and-goal configurations for one MetaWorld task.

    MetaWorld's goal-observable MT1 wrapper draws each reset configuration
    from its RNG. Resetting with one seed per task makes a pick-and-place
    object/goal configuration reproducible while keeping the task id out of
    the policy interface. The object and goal state remain part of the state
    observation. Test seeds are disjoint from train seeds.
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
