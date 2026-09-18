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
        self._object_body_id: int | None = None
        self._object_geom_id: int | None = None
        self._base_object_mass: float | None = None
        self._base_object_friction: np.ndarray | None = None

    def set_object_dynamics(self, mass_scale: float = 1.0, friction_scale: float = 1.0) -> None:
        """Set hidden pick-and-place object dynamics without changing observations.

        Only the object's mass and sliding-friction coefficient are scaled.
        The robot, object pose, and goal continue to be supplied by MetaWorld's
        regular state observation.
        """
        if self.spec is None or self.spec.kwargs.get("env_name") != "pick-place-v3":
            raise ValueError("Hidden dynamics are currently supported only for pick-place-v3.")
        if mass_scale <= 0 or friction_scale <= 0:
            raise ValueError("mass_scale and friction_scale must both be positive.")
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure hidden object dynamics.") from error

        model = self.unwrapped.model
        if self._object_body_id is None or self._object_geom_id is None:
            self._object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
            self._object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
            if self._object_body_id < 0 or self._object_geom_id < 0:
                raise RuntimeError("Could not find the pick-place object in the MuJoCo model.")
            self._base_object_mass = float(model.body_mass[self._object_body_id])
            self._base_object_friction = model.geom_friction[self._object_geom_id].copy()

        assert self._base_object_mass is not None and self._base_object_friction is not None
        model.body_mass[self._object_body_id] = self._base_object_mass * mass_scale
        model.geom_friction[self._object_geom_id] = self._base_object_friction
        model.geom_friction[self._object_geom_id, 0] *= friction_scale
        mujoco.mj_setConst(model, self.unwrapped.data)


@dataclass(frozen=True)
class TaskSpec:
    """A task identified only by replay bookkeeping, never an agent input."""

    task_id: int
    reset_seed: int | None = None
    mass_scale: float = 1.0
    friction_scale: float = 1.0


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


class DynamicsTaskSampler:
    """Sample hidden mass/friction tasks with randomized visible configurations.

    A task fixes only the physical parameters. Object and goal positions are
    re-sampled by the environment on every reset, preventing PEARL from using
    a visible placement as a task identifier. Train and test parameter pairs
    are generated independently and are never exposed to the policy.
    """

    def __init__(
        self,
        num_train_tasks: int,
        num_test_tasks: int,
        mass_scale_min: float = 0.5,
        mass_scale_max: float = 1.5,
        friction_scale_min: float = 0.5,
        friction_scale_max: float = 1.5,
        seed: int = 1,
    ) -> None:
        if num_train_tasks < 1 or num_test_tasks < 1:
            raise ValueError("num_train_tasks and num_test_tasks must both be positive.")
        if not 0 < mass_scale_min <= mass_scale_max:
            raise ValueError("Mass scales must be positive and ordered.")
        if not 0 < friction_scale_min <= friction_scale_max:
            raise ValueError("Friction scales must be positive and ordered.")
        self.rng = np.random.default_rng(seed)
        parameters = self.rng.uniform(
            low=(mass_scale_min, friction_scale_min),
            high=(mass_scale_max, friction_scale_max),
            size=(num_train_tasks + num_test_tasks, 2),
        )
        self.train_tasks = tuple(
            TaskSpec(task_id, mass_scale=float(parameters[task_id, 0]), friction_scale=float(parameters[task_id, 1]))
            for task_id in range(num_train_tasks)
        )
        self.test_tasks = tuple(
            TaskSpec(
                task_id,
                mass_scale=float(parameters[num_train_tasks + task_id, 0]),
                friction_scale=float(parameters[num_train_tasks + task_id, 1]),
            )
            for task_id in range(num_test_tasks)
        )

    def sample_train_task(self) -> TaskSpec:
        return self.train_tasks[int(self.rng.integers(len(self.train_tasks)))]
