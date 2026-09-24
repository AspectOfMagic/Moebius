from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import gymnasium as gym
import numpy as np


def _shape_aware_pick_place_caging_reward(
    environment: object,
    action: np.ndarray,
    obj_pos: np.ndarray,
    obj_radius: float = 0,
    pad_success_thresh: float = 0,
    object_reach_radius: float = 0,
    xz_thresh: float = 0,
    desired_gripper_effort: float = 1.0,
    high_density: bool = False,
    medium_density: bool = False,
) -> float:
    """MetaWorld's caging reward with the live cylinder radius.

    ``SawyerPickPlaceEnvV3`` hard-codes a 0.015 m radius even though this
    wrapper varies the object's geometry. Replacing only that constant keeps
    its reward semantics while making small and large task objects graspable.
    """
    del obj_radius, pad_success_thresh, object_reach_radius, xz_thresh, high_density, medium_density
    from metaworld.utils import reward_utils

    pad_success_margin = 0.05
    x_z_success_margin = 0.005
    radius = float(environment.model.geom("objGeom").size[0])
    tcp = environment.tcp_center
    left_pad = environment.get_body_com("leftpad")
    right_pad = environment.get_body_com("rightpad")
    delta_object_y_left_pad = left_pad[1] - obj_pos[1]
    delta_object_y_right_pad = obj_pos[1] - right_pad[1]
    right_caging_margin = abs(abs(obj_pos[1] - environment.init_right_pad[1]) - pad_success_margin)
    left_caging_margin = abs(abs(obj_pos[1] - environment.init_left_pad[1]) - pad_success_margin)
    right_caging = reward_utils.tolerance(
        delta_object_y_right_pad,
        bounds=(radius, pad_success_margin),
        margin=right_caging_margin,
        sigmoid="long_tail",
    )
    left_caging = reward_utils.tolerance(
        delta_object_y_left_pad,
        bounds=(radius, pad_success_margin),
        margin=left_caging_margin,
        sigmoid="long_tail",
    )
    y_caging = reward_utils.hamacher_product(left_caging, right_caging)
    tcp_xz = tcp + np.array([0.0, -tcp[1], 0.0])
    obj_xz = np.asarray(obj_pos) + np.array([0.0, -obj_pos[1], 0.0])
    tcp_obj_xz_distance = float(np.linalg.norm(tcp_xz - obj_xz, ord=2))
    init_obj_xz = environment.obj_init_pos + np.array([0.0, -environment.obj_init_pos[1], 0.0])
    init_tcp_xz = environment.init_tcp + np.array([0.0, -environment.init_tcp[1], 0.0])
    tcp_obj_xz_margin = np.linalg.norm(init_obj_xz - init_tcp_xz, ord=2) - x_z_success_margin
    x_z_caging = reward_utils.tolerance(
        tcp_obj_xz_distance,
        bounds=(0, x_z_success_margin),
        margin=tcp_obj_xz_margin,
        sigmoid="long_tail",
    )
    gripper_closed = min(max(0, action[-1]), desired_gripper_effort)
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
    return (caging_and_gripping + caging) / 2


class MetaWorldSingleTaskEnv(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Create one goal-observable MetaWorld task through Gymnasium.

    MT1 keeps the object and goal state observable. For PEARL, a reset seed
    selects one reproducible pick-and-place configuration while its task
    identifier stays outside the agent observation.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}
    _TABLE_CONTACT_TYPE = 1 << 1

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
        self._table_geom_ids: tuple[int, ...] | None = None
        self._finger_geom_ids: tuple[int, ...] | None = None
        self._base_object_mass: float | None = None
        self._base_object_inertia: np.ndarray | None = None
        self._base_object_size: np.ndarray | None = None
        self._base_object_friction: np.ndarray | None = None
        self._base_table_friction: np.ndarray | None = None
        self._base_finger_friction: np.ndarray | None = None
        self._base_object_solref: np.ndarray | None = None
        self._base_table_solref: np.ndarray | None = None
        self._base_finger_solref: np.ndarray | None = None
        self._base_table_contype: np.ndarray | None = None
        self._base_table_conaffinity: np.ndarray | None = None
        self._base_object_contype: int | None = None
        self._base_object_conaffinity: int | None = None
        self._shape_aware_reward_installed = False

    def _pick_place_model_ids(self) -> None:
        """Cache pick-place geometry ids and their immutable nominal values."""
        if self._object_body_id is not None:
            return
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure hidden object properties.") from error

        model = self.unwrapped.model
        self._object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self._object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        if self._object_body_id < 0 or self._object_geom_id < 0:
            raise RuntimeError("Could not find the pick-place object in the MuJoCo model.")

        table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablelink")
        if table_body_id < 0:
            raise RuntimeError("Could not find the pick-place table in the MuJoCo model.")
        # MetaWorld renders the table top separately from its invisible group-4
        # collision box. Use the latter so this task property affects contacts,
        # not merely appearance.
        self._table_geom_ids = tuple(
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == table_body_id and model.geom_group[geom_id] == 4
        )
        self._finger_geom_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("rightclaw_it", "rightpad_geom", "leftclaw_it", "leftpad_geom")
        )
        if not self._table_geom_ids or any(geom_id < 0 for geom_id in self._finger_geom_ids):
            raise RuntimeError("Could not find the pick-place table or finger collision geometries.")

        self._base_object_mass = float(model.body_mass[self._object_body_id])
        self._base_object_inertia = model.body_inertia[self._object_body_id].copy()
        self._base_object_size = model.geom_size[self._object_geom_id].copy()
        self._base_object_friction = model.geom_friction[self._object_geom_id].copy()
        self._base_table_friction = model.geom_friction[list(self._table_geom_ids)].copy()
        self._base_finger_friction = model.geom_friction[list(self._finger_geom_ids)].copy()
        self._base_object_solref = model.geom_solref[self._object_geom_id].copy()
        self._base_table_solref = model.geom_solref[list(self._table_geom_ids)].copy()
        self._base_finger_solref = model.geom_solref[list(self._finger_geom_ids)].copy()
        self._base_table_contype = model.geom_contype[list(self._table_geom_ids)].copy()
        self._base_table_conaffinity = model.geom_conaffinity[list(self._table_geom_ids)].copy()
        self._base_object_contype = int(model.geom_contype[self._object_geom_id])
        self._base_object_conaffinity = int(model.geom_conaffinity[self._object_geom_id])

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
        model = self.unwrapped.model
        self._pick_place_model_ids()
        assert self._base_object_mass is not None and self._base_object_friction is not None
        model.body_mass[self._object_body_id] = self._base_object_mass * mass_scale
        model.geom_friction[self._object_geom_id] = self._base_object_friction
        model.geom_friction[self._object_geom_id, 0] *= friction_scale
        import mujoco

        mujoco.mj_setConst(model, self.unwrapped.data)

    def set_task_properties(self, task: TaskSpec) -> None:
        """Apply a hidden, per-task pick-place object configuration.

        All supported values are numeric fields in MuJoCo's already-compiled
        model. Shape remains a cylinder, but its radius and height can vary.
        Contact friction is assigned to the table and finger collision geoms;
        MuJoCo uses the maximum of the two geom frictions for a contact, so the
        object itself is kept at a negligible friction to keep these two task
        parameters independent.
        """
        if self.spec is None or self.spec.kwargs.get("env_name") != "pick-place-v3":
            raise ValueError("Hidden object properties are currently supported only for pick-place-v3.")
        self._pick_place_model_ids()
        assert self._object_body_id is not None and self._object_geom_id is not None
        assert self._table_geom_ids is not None and self._finger_geom_ids is not None
        assert self._base_object_mass is not None and self._base_object_inertia is not None
        assert self._base_object_size is not None and self._base_object_friction is not None
        assert self._base_table_friction is not None and self._base_finger_friction is not None
        assert self._base_object_solref is not None and self._base_table_solref is not None
        assert self._base_finger_solref is not None
        assert self._base_table_contype is not None and self._base_table_conaffinity is not None
        assert self._base_object_contype is not None and self._base_object_conaffinity is not None
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure hidden object properties.") from error

        model = self.unwrapped.model
        # Goal tasks leave all physics at MetaWorld's nominal configuration.
        if task.object_radius is None:
            model.body_mass[self._object_body_id] = self._base_object_mass
            model.body_inertia[self._object_body_id] = self._base_object_inertia
            model.geom_size[self._object_geom_id] = self._base_object_size
            model.geom_friction[self._object_geom_id] = self._base_object_friction
            model.geom_friction[list(self._table_geom_ids)] = self._base_table_friction
            model.geom_friction[list(self._finger_geom_ids)] = self._base_finger_friction
            model.geom_solref[self._object_geom_id] = self._base_object_solref
            model.geom_solref[list(self._table_geom_ids)] = self._base_table_solref
            model.geom_solref[list(self._finger_geom_ids)] = self._base_finger_solref
            model.geom_contype[list(self._table_geom_ids)] = self._base_table_contype
            model.geom_conaffinity[list(self._table_geom_ids)] = self._base_table_conaffinity
            model.geom_contype[self._object_geom_id] = self._base_object_contype
            model.geom_conaffinity[self._object_geom_id] = self._base_object_conaffinity
            mujoco.mj_setConst(model, self.unwrapped.data)
            return

        if not self._shape_aware_reward_installed:
            self.unwrapped._gripper_caging_reward = MethodType(_shape_aware_pick_place_caging_reward, self.unwrapped)
            self._shape_aware_reward_installed = True

        values = (
            task.object_radius,
            task.object_height,
            task.object_mass,
            task.object_table_friction,
            task.object_finger_friction,
            task.contact_softness,
        )
        if any(value is None or value <= 0 for value in values):
            raise ValueError("All hidden object-property values must be positive.")
        radius, height, mass, table_friction, finger_friction, softness = values
        assert radius is not None and height is not None and mass is not None
        assert table_friction is not None and finger_friction is not None and softness is not None

        model.geom_size[self._object_geom_id, :2] = (radius, height / 2)
        model.body_mass[self._object_body_id] = mass
        # Cylinder principal moments about its center of mass.
        model.body_inertia[self._object_body_id] = (
            mass * (3 * radius**2 + height**2) / 12,
            mass * (3 * radius**2 + height**2) / 12,
            mass * radius**2 / 2,
        )
        model.geom_friction[self._object_geom_id] = (1e-5, 1e-6, 1e-7)
        # MetaWorld disables its invisible table collision box. Give it a
        # dedicated collision type and give only the object affinity for that
        # type. The table can therefore contact the object, but not the
        # gripper or robot geoms that use MetaWorld's ordinary collision type.
        model.geom_contype[list(self._table_geom_ids)] = self._TABLE_CONTACT_TYPE
        model.geom_conaffinity[list(self._table_geom_ids)] = 0
        model.geom_contype[self._object_geom_id] = self._base_object_contype
        model.geom_conaffinity[self._object_geom_id] = self._base_object_conaffinity | self._TABLE_CONTACT_TYPE
        model.geom_friction[list(self._table_geom_ids)] = self._base_table_friction * (
            table_friction / self._base_table_friction[:, :1]
        )
        model.geom_friction[list(self._finger_geom_ids)] = self._base_finger_friction * (
            finger_friction / self._base_finger_friction[:, :1]
        )
        for geom_ids in (self._table_geom_ids, self._finger_geom_ids):
            model.geom_solref[list(geom_ids), 0] = softness
        model.geom_solref[self._object_geom_id, 0] = softness
        mujoco.mj_setConst(model, self.unwrapped.data)

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        observation, info = super().reset(seed=seed, options=options)
        # MetaWorld's pick-place reset has a fixed z=0.02 object center. Put a
        # resized cylinder flush on the table and keep reward baselines in sync.
        if self._object_geom_id is not None and self._base_object_size is not None:
            try:
                import mujoco
            except ImportError as error:
                raise ImportError("MuJoCo is required to reset hidden object properties.") from error
            half_height = float(self.unwrapped.model.geom_size[self._object_geom_id, 1])
            joint = self.unwrapped.data.joint("objjoint")
            joint.qpos[2] = half_height
            mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)
            unwrapped = self.unwrapped
            if getattr(unwrapped, "obj_init_pos", None) is not None:
                unwrapped.obj_init_pos[2] = half_height
            if hasattr(unwrapped, "objHeight"):
                unwrapped.objHeight = half_height
            if hasattr(unwrapped, "heightTarget"):
                unwrapped.heightTarget = half_height + 0.04
            if hasattr(unwrapped, "maxPlacingDist") and getattr(unwrapped, "obj_init_pos", None) is not None:
                unwrapped.maxPlacingDist = float(
                    np.linalg.norm(
                        np.array([unwrapped.obj_init_pos[0], unwrapped.obj_init_pos[1], unwrapped.heightTarget])
                        - np.array(unwrapped._target_pos)
                    )
                    + unwrapped.heightTarget
                )
            observation = unwrapped._get_obs()
        return observation, info


@dataclass(frozen=True)
class TaskSpec:
    """A task identified only by replay bookkeeping, never an agent input."""

    task_id: int
    reset_seed: int | None = None
    object_radius: float | None = None
    object_height: float | None = None
    object_mass: float | None = None
    object_table_friction: float | None = None
    object_finger_friction: float | None = None
    contact_softness: float | None = None


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
    """Sample hidden object-property tasks with randomized visible configurations.

    A task fixes only the physical parameters. Object and goal positions are
    re-sampled by the environment on every reset, preventing PEARL from using
    a visible placement as a task identifier. Train and test parameter pairs
    are generated independently and are never exposed to the policy.
    """

    def __init__(
        self,
        num_train_tasks: int,
        num_test_tasks: int,
        object_radius_min: float = 0.008,
        object_radius_max: float = 0.028,
        object_height_min: float = 0.016,
        object_height_max: float = 0.080,
        object_mass_min: float = 0.10,
        object_mass_max: float = 2.00,
        object_table_friction_min: float = 0.05,
        object_table_friction_max: float = 4.00,
        object_finger_friction_min: float = 0.05,
        object_finger_friction_max: float = 4.00,
        contact_softness_min: float = 0.004,
        contact_softness_max: float = 0.060,
        seed: int = 1,
    ) -> None:
        if num_train_tasks < 1 or num_test_tasks < 1:
            raise ValueError("num_train_tasks and num_test_tasks must both be positive.")
        bounds = (
            ("object radius", object_radius_min, object_radius_max),
            ("object height", object_height_min, object_height_max),
            ("object mass", object_mass_min, object_mass_max),
            ("object/table friction", object_table_friction_min, object_table_friction_max),
            ("object/finger friction", object_finger_friction_min, object_finger_friction_max),
            ("contact softness", contact_softness_min, contact_softness_max),
        )
        for name, lower, upper in bounds:
            if not 0 < lower <= upper:
                raise ValueError(f"{name.capitalize()} bounds must be positive and ordered.")
        self.rng = np.random.default_rng(seed)
        self._bounds = tuple((lower, upper) for _, lower, upper in bounds)
        parameters = self.rng.uniform(
            low=tuple(lower for _, lower, _ in bounds),
            high=tuple(upper for _, _, upper in bounds),
            size=(num_train_tasks + num_test_tasks, len(bounds)),
        )
        self.train_tasks = tuple(self.task_from_parameters(task_id, parameters[task_id]) for task_id in range(num_train_tasks))
        self.test_tasks = tuple(
            self.task_from_parameters(task_id, parameters[num_train_tasks + task_id]) for task_id in range(num_test_tasks)
        )

    def task_from_parameters(self, task_id: int, parameters: np.ndarray) -> TaskSpec:
        """Build a task with a caller-provided replay id from one property vector."""
        return TaskSpec(
            task_id,
            object_radius=float(parameters[0]),
            object_height=float(parameters[1]),
            object_mass=float(parameters[2]),
            object_table_friction=float(parameters[3]),
            object_finger_friction=float(parameters[4]),
            contact_softness=float(parameters[5]),
        )

    def resample_task(self, task_id: int) -> TaskSpec:
        """Draw a new property vector while retaining its replay-buffer id."""
        lower, upper = zip(*self._bounds, strict=True)
        return self.task_from_parameters(task_id, self.rng.uniform(lower, upper))

    def sample_train_task(self) -> TaskSpec:
        return self.train_tasks[int(self.rng.integers(len(self.train_tasks)))]
