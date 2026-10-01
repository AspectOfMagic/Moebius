from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Object shapes and hidden properties
# ---------------------------------------------------------------------------

# Every shape rests with the same (identity) orientation, so the object
# quaternion in the observation does not reveal which shape a task uses.
# "bar" is a box elongated along x; the gripper closes along y, so its length
# does not change the grip width but makes off-centre grasps tilt.
SHAPES = ("cylinder", "box", "bar", "sphere")

# MuJoCo mjtGeom values. Kept as plain integers so the geometry helpers can be
# used (and tested) without importing MuJoCo; the environment checks them
# against mujoco.mjtGeom when it applies a shape.
GEOM_SPHERE = 2
GEOM_CYLINDER = 5
GEOM_BOX = 6


@dataclass(frozen=True)
class ObjectGeometry:
    """MuJoCo geom parameters and derived quantities for one object."""

    geom_type: int
    size: tuple[float, float, float]
    # Axis-aligned half extents in the object frame (x, y, z).
    half_extents: tuple[float, float, float]
    bounding_radius: float
    inertia: tuple[float, float, float]

    @property
    def rest_height(self) -> float:
        """Height of the object centre when resting on the table."""
        return self.half_extents[2]

    @property
    def grip_half_width(self) -> float:
        """Half the object's width along y, the gripper's closing axis."""
        return self.half_extents[1]


def object_geometry(
    shape: str, half_width: float, half_height: float, half_length: float | None, mass: float
) -> ObjectGeometry:
    """Build a shape's geom size, bounds and principal inertia.

    ``half_width`` is the half-width along the gripper's closing axis (y).
    Spheres use it as their radius and ignore ``half_height``; only bars use
    ``half_length`` (their half-extent along x).
    """
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape {shape!r}; expected one of {SHAPES}.")
    if min(half_width, half_height, mass) <= 0:
        raise ValueError("Object half-width, half-height and mass must be positive.")
    if shape == "cylinder":
        r, h = half_width, half_height
        return ObjectGeometry(
            GEOM_CYLINDER,
            (r, h, 0.0),
            (r, r, h),
            float(np.hypot(r, h)),
            (mass * (3 * r**2 + 4 * h**2) / 12, mass * (3 * r**2 + 4 * h**2) / 12, mass * r**2 / 2),
        )
    if shape == "sphere":
        r = half_width
        moment = 2 * mass * r**2 / 5
        return ObjectGeometry(GEOM_SPHERE, (r, 0.0, 0.0), (r, r, r), r, (moment, moment, moment))
    if shape == "bar":
        if half_length is None or half_length <= 0:
            raise ValueError("A bar needs a positive half_length.")
        hx = half_length
    else:  # box
        hx = half_width
    hy, hz = half_width, half_height
    return ObjectGeometry(
        GEOM_BOX,
        (hx, hy, hz),
        (hx, hy, hz),
        float(np.linalg.norm((hx, hy, hz))),
        (mass * (hy**2 + hz**2) / 3, mass * (hx**2 + hz**2) / 3, mass * (hx**2 + hy**2) / 3),
    )


@dataclass(frozen=True)
class ObjectPropertyBounds:
    """Sampling ranges for task shapes and hidden physical properties.

    Only properties that change how the object must be handled are hidden:
    mass together with finger friction decides whether a grasp slips, and a
    low table friction makes the object slide or roll away when touched.
    These only matter when the gripper's force is limited (see
    ``gripper_force_limit``); MetaWorld's position-servo fingers otherwise
    squeeze far harder than any of these objects weigh.
    """

    shapes: tuple[str, ...] = SHAPES
    half_width: tuple[float, float] = (0.015, 0.025)
    half_height: tuple[float, float] = (0.015, 0.03)
    bar_half_length: tuple[float, float] = (0.04, 0.07)
    mass: tuple[float, float] = (0.1, 2.0)
    table_friction: tuple[float, float] = (0.05, 1.5)
    finger_friction: tuple[float, float] = (0.3, 2.0)

    def __post_init__(self) -> None:
        if not self.shapes or any(shape not in SHAPES for shape in self.shapes):
            raise ValueError(f"shapes must be a non-empty subset of {SHAPES}.")
        for name in ("half_width", "half_height", "bar_half_length", "mass", "table_friction", "finger_friction"):
            lower, upper = getattr(self, name)
            if not 0 < lower <= upper:
                raise ValueError(f"{name} bounds must be positive and ordered.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ObjectPropertyBounds:
        """Read bounds from a TrainConfig dict; missing keys use the defaults."""
        defaults = cls()

        def pair(name: str, key: str) -> tuple[float, float]:
            default = getattr(defaults, name)
            return (float(config.get(f"{key}_min", default[0])), float(config.get(f"{key}_max", default[1])))

        return cls(
            shapes=tuple(config.get("object_shapes", defaults.shapes)),
            half_width=pair("half_width", "object_half_width"),
            half_height=pair("half_height", "object_half_height"),
            bar_half_length=pair("bar_half_length", "object_bar_half_length"),
            mass=pair("mass", "object_mass"),
            table_friction=pair("table_friction", "object_table_friction"),
            finger_friction=pair("finger_friction", "object_finger_friction"),
        )


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
    """MetaWorld's caging reward with the live object half-width.

    ``SawyerPickPlaceEnvV3`` hard-codes a 0.015 m radius. This uses the current
    object's half-width along the gripper's closing axis (y) instead, so every
    shape and size is rewarded for the same caging geometry.
    """
    del obj_radius, pad_success_thresh, object_reach_radius, xz_thresh, high_density, medium_density
    from metaworld.utils import reward_utils

    pad_success_margin = 0.05
    x_z_success_margin = 0.005
    radius = _grip_half_width(environment.model, environment.model.geom("objGeom").id)
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


def _grip_half_width(model: Any, geom_id: int) -> float:
    """Half-width along y of a sphere, cylinder or box geom."""
    size = model.geom_size[geom_id]
    return float(size[1] if int(model.geom_type[geom_id]) == GEOM_BOX else size[0])


def _rest_height(model: Any, geom_id: int) -> float:
    """Centre height of a sphere, cylinder or box geom resting on the table."""
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == GEOM_SPHERE:
        return float(size[0])
    if geom_type == GEOM_CYLINDER:
        return float(size[1])
    return float(size[2])


class MetaWorldSingleTaskEnv(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Create one goal-observable MetaWorld task through Gymnasium.

    MT1 keeps the object and goal state observable. For PEARL, a task sets the
    pick-place object's shape, size and hidden physical properties, while its
    task identifier stays outside the agent observation.

    ``gripper_force_limit`` (newtons per finger, ``None`` = MetaWorld's
    unlimited servo) caps the finger actuators. It is a fixed property of the
    robot, not of a task.
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
        gripper_force_limit: float | None = None,
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
        self._ids_cached = False
        self._base: dict[str, Any] = {}
        self._shape_aware_reward_installed = False
        if gripper_force_limit is not None:
            self.set_gripper_force_limit(gripper_force_limit)

    # -- model bookkeeping -------------------------------------------------

    def _pick_place_model_ids(self) -> None:
        """Cache pick-place ids and the nominal values this wrapper changes."""
        if self._ids_cached:
            return
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure object properties.") from error

        # The integer constants above must match this MuJoCo build.
        assert GEOM_SPHERE == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        assert GEOM_CYLINDER == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        assert GEOM_BOX == int(mujoco.mjtGeom.mjGEOM_BOX)

        model = self.unwrapped.model
        self._object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self._object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "objGeom")
        if self._object_body_id < 0 or self._object_geom_id < 0:
            raise RuntimeError("Could not find the pick-place object in the MuJoCo model.")
        table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tablelink")
        if table_body_id < 0:
            raise RuntimeError("Could not find the pick-place table in the MuJoCo model.")
        # MetaWorld renders the table top separately from its invisible group-4
        # collision box. Use the latter so table friction affects contacts.
        self._table_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == table_body_id and model.geom_group[geom_id] == 4
        ]
        self._finger_geom_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("rightclaw_it", "rightpad_geom", "leftclaw_it", "leftpad_geom")
        ]
        if not self._table_geom_ids or any(geom_id < 0 for geom_id in self._finger_geom_ids):
            raise RuntimeError("Could not find the pick-place table or finger collision geometries.")
        self._pad_geom_ids = {self._finger_geom_ids[1], self._finger_geom_ids[3]}
        finger_joint_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ("r_close", "l_close")
        }
        self._finger_actuator_ids = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trnid[actuator_id, 0]) in finger_joint_ids
        ]
        body, geom = self._object_body_id, self._object_geom_id
        bvh_address = int(model.body_bvhadr[body]) if hasattr(model, "body_bvhadr") else -1
        single_bvh = bvh_address >= 0 and int(model.body_bvhnum[body]) == 1
        self._object_bvh_address = bvh_address if single_bvh else None
        self._base = {
            "mass": float(model.body_mass[body]),
            "inertia": model.body_inertia[body].copy(),
            "geom_type": int(model.geom_type[geom]),
            "geom_size": model.geom_size[geom].copy(),
            "geom_rbound": float(model.geom_rbound[geom]),
            "geom_aabb": model.geom_aabb[geom].copy(),
            "bvh_aabb": model.bvh_aabb[bvh_address].copy() if single_bvh else None,
            "object_friction": model.geom_friction[geom].copy(),
            "table_friction": model.geom_friction[self._table_geom_ids].copy(),
            "finger_friction": model.geom_friction[self._finger_geom_ids].copy(),
            "table_contype": model.geom_contype[self._table_geom_ids].copy(),
            "table_conaffinity": model.geom_conaffinity[self._table_geom_ids].copy(),
            "object_contype": int(model.geom_contype[geom]),
            "object_conaffinity": int(model.geom_conaffinity[geom]),
            "actuator_forcelimited": model.actuator_forcelimited[self._finger_actuator_ids].copy(),
            "actuator_forcerange": model.actuator_forcerange[self._finger_actuator_ids].copy(),
        }
        self._ids_cached = True

    def _set_object_geometry(
        self, geom_type: int, size: np.ndarray, rbound: float, aabb: np.ndarray, bvh_aabb: np.ndarray | None
    ) -> None:
        """Swap the object's primitive shape in the compiled model.

        MuJoCo reads geom type and size at every step, so a primitive can be
        changed in place. Its collision bounds are updated as well: the
        bounding radius and box are what the collision broad phase tests.
        """
        model = self.unwrapped.model
        geom = self._object_geom_id
        model.geom_type[geom] = geom_type
        model.geom_size[geom] = size
        model.geom_rbound[geom] = rbound
        model.geom_aabb[geom] = aabb
        if self._object_bvh_address is not None and bvh_aabb is not None:
            model.bvh_aabb[self._object_bvh_address] = bvh_aabb

    def set_gripper_force_limit(self, limit: float | None) -> None:
        """Cap each finger actuator's force (N); ``None`` restores MetaWorld's servo."""
        self._pick_place_model_ids()
        model = self.unwrapped.model
        if limit is None:
            model.actuator_forcelimited[self._finger_actuator_ids] = self._base["actuator_forcelimited"]
            model.actuator_forcerange[self._finger_actuator_ids] = self._base["actuator_forcerange"]
            return
        if limit <= 0:
            raise ValueError("gripper_force_limit must be positive.")
        if len(self._finger_actuator_ids) != 2:
            raise RuntimeError("Could not find the two finger actuators (r_close, l_close).")
        model.actuator_forcelimited[self._finger_actuator_ids] = 1
        model.actuator_forcerange[self._finger_actuator_ids] = (-limit, limit)

    def grip_force(self) -> float:
        """Total normal force (N) between the finger pads and the object right now."""
        import mujoco

        self._pick_place_model_ids()
        model, data = self.unwrapped.model, self.unwrapped.data
        wrench = np.zeros(6)
        total = 0.0
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = {int(contact.geom1), int(contact.geom2)}
            if self._object_geom_id in geoms and geoms & self._pad_geom_ids:
                mujoco.mj_contactForce(model, data, index, wrench)
                total += abs(wrench[0])
        return total

    # -- tasks ---------------------------------------------------------------

    def set_task_properties(self, task: TaskSpec) -> None:
        """Apply a task's object shape, size and hidden physical properties.

        Friction is assigned to the table and finger collision geoms; MuJoCo
        uses the larger of the two geoms' frictions for a contact, so the
        object itself keeps a negligible friction to make these two task
        properties independent. Goal tasks (``task.shape is None``) restore
        MetaWorld's nominal object.
        """
        if self.spec is None or self.spec.kwargs.get("env_name") != "pick-place-v3":
            raise ValueError("Object properties are currently supported only for pick-place-v3.")
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure object properties.") from error
        self._pick_place_model_ids()
        model = self.unwrapped.model
        base = self._base
        body, geom = self._object_body_id, self._object_geom_id

        if task.shape is None:
            model.body_mass[body] = base["mass"]
            model.body_inertia[body] = base["inertia"]
            self._set_object_geometry(
                base["geom_type"], base["geom_size"], base["geom_rbound"], base["geom_aabb"], base["bvh_aabb"]
            )
            model.geom_friction[geom] = base["object_friction"]
            model.geom_friction[self._table_geom_ids] = base["table_friction"]
            model.geom_friction[self._finger_geom_ids] = base["finger_friction"]
            model.geom_contype[self._table_geom_ids] = base["table_contype"]
            model.geom_conaffinity[self._table_geom_ids] = base["table_conaffinity"]
            model.geom_contype[geom] = base["object_contype"]
            model.geom_conaffinity[geom] = base["object_conaffinity"]
            mujoco.mj_setConst(model, self.unwrapped.data)
            return

        if not self._shape_aware_reward_installed:
            self.unwrapped._gripper_caging_reward = MethodType(_shape_aware_pick_place_caging_reward, self.unwrapped)
            self._shape_aware_reward_installed = True

        values = (
            task.object_half_width,
            task.object_half_height,
            task.object_mass,
            task.object_table_friction,
            task.object_finger_friction,
        )
        if any(value is None or value <= 0 for value in values):
            raise ValueError("All object size and property values must be positive.")
        geometry = object_geometry(
            task.shape, task.object_half_width, task.object_half_height, task.object_half_length, task.object_mass
        )
        extents = np.asarray(geometry.half_extents)
        aabb = np.concatenate((np.zeros(3), extents))
        self._set_object_geometry(geometry.geom_type, np.asarray(geometry.size), geometry.bounding_radius, aabb, aabb)
        model.body_mass[body] = task.object_mass
        model.body_inertia[body] = geometry.inertia
        model.geom_friction[geom] = (1e-5, 1e-6, 1e-7)
        # MetaWorld disables its invisible table collision box. Give it a
        # dedicated collision type and give only the object affinity for it:
        # the table then contacts the object, not the gripper or robot.
        model.geom_contype[self._table_geom_ids] = self._TABLE_CONTACT_TYPE
        model.geom_conaffinity[self._table_geom_ids] = 0
        model.geom_contype[geom] = base["object_contype"]
        model.geom_conaffinity[geom] = base["object_conaffinity"] | self._TABLE_CONTACT_TYPE
        model.geom_friction[self._table_geom_ids] = base["table_friction"] * (
            task.object_table_friction / base["table_friction"][:, :1]
        )
        model.geom_friction[self._finger_geom_ids] = base["finger_friction"] * (
            task.object_finger_friction / base["finger_friction"][:, :1]
        )
        mujoco.mj_setConst(model, self.unwrapped.data)

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        observation, info = super().reset(seed=seed, options=options)
        # MetaWorld's reset puts the object centre at z = 0.02. Rest the
        # current shape on the table and keep reward baselines in sync.
        if self._ids_cached:
            try:
                import mujoco
            except ImportError as error:
                raise ImportError("MuJoCo is required to reset object properties.") from error
            unwrapped = self.unwrapped
            rest_height = _rest_height(unwrapped.model, self._object_geom_id)
            joint = unwrapped.data.joint("objjoint")
            joint.qpos[2] = rest_height
            mujoco.mj_forward(unwrapped.model, unwrapped.data)
            if getattr(unwrapped, "obj_init_pos", None) is not None:
                unwrapped.obj_init_pos[2] = rest_height
            if hasattr(unwrapped, "objHeight"):
                unwrapped.objHeight = rest_height
            if hasattr(unwrapped, "heightTarget"):
                unwrapped.heightTarget = rest_height + 0.04
            if hasattr(unwrapped, "maxPlacingDist") and getattr(unwrapped, "obj_init_pos", None) is not None:
                unwrapped.maxPlacingDist = float(
                    np.linalg.norm(
                        np.array([unwrapped.obj_init_pos[0], unwrapped.obj_init_pos[1], unwrapped.heightTarget])
                        - np.array(unwrapped._target_pos)
                    )
                    + unwrapped.heightTarget
                )
            # Refresh the frame-stacked previous observation too, so the first
            # observation does not pair the new object height with the old one.
            unwrapped._prev_obs = unwrapped._get_curr_obs_combined_no_goal()
            observation = unwrapped._get_obs()
        return observation, info


@dataclass(frozen=True)
class TaskSpec:
    """A task identified only by replay bookkeeping, never an agent input.

    ``shape is None`` means MetaWorld's nominal object (goal tasks). Otherwise
    the object is ``shape`` with the given sizes (metres) and hidden mass
    (kg), table friction and finger friction.
    """

    task_id: int
    reset_seed: int | None = None
    shape: str | None = None
    object_half_width: float | None = None
    object_half_height: float | None = None
    object_half_length: float | None = None
    object_mass: float | None = None
    object_table_friction: float | None = None
    object_finger_friction: float | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TaskSpec:
        """Rebuild a saved task, converting pre-shape checkpoints.

        Older checkpoints stored a cylinder's radius, full height and contact
        softness. They become a cylinder with the same size; softness is no
        longer a task property and is dropped.
        """
        values = dict(values)
        if "object_radius" in values:
            radius = values.pop("object_radius")
            height = values.pop("object_height", None)
            values.pop("contact_softness", None)
            if radius is not None:
                values.setdefault("shape", "cylinder")
                values["object_half_width"] = radius
                values["object_half_height"] = None if height is None else height / 2
        return cls(**values)


class GoalTaskSampler:
    """Create seeded object-and-goal configurations for one MetaWorld task.

    MetaWorld's goal-observable MT1 wrapper draws each reset configuration
    from its RNG. Resetting with one seed per task makes a pick-and-place
    object/goal configuration reproducible while keeping the task id out of
    the policy interface. Test seeds are disjoint from train seeds.
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
    """Sample object shapes, sizes and hidden properties for PEARL tasks.

    Shapes are assigned round-robin so every shape appears in both the train
    and test sets; sizes and hidden properties are drawn uniformly within
    ``bounds``. Object and goal positions are re-sampled by the environment
    on every reset, so a visible placement never identifies a task.
    """

    def __init__(
        self,
        num_train_tasks: int,
        num_test_tasks: int,
        bounds: ObjectPropertyBounds | None = None,
        seed: int = 1,
    ) -> None:
        if num_train_tasks < 1 or num_test_tasks < 1:
            raise ValueError("num_train_tasks and num_test_tasks must both be positive.")
        self.bounds = ObjectPropertyBounds() if bounds is None else bounds
        self.rng = np.random.default_rng(seed)
        shapes = self.bounds.shapes
        self.train_tasks = tuple(
            self._sample(task_id, shapes[task_id % len(shapes)]) for task_id in range(num_train_tasks)
        )
        self.test_tasks = tuple(
            self._sample(task_id, shapes[task_id % len(shapes)]) for task_id in range(num_test_tasks)
        )

    def _uniform(self, bounds: tuple[float, float]) -> float:
        return float(self.rng.uniform(*bounds))

    def _sample(self, task_id: int, shape: str) -> TaskSpec:
        half_width = self._uniform(self.bounds.half_width)
        half_height = half_width if shape == "sphere" else self._uniform(self.bounds.half_height)
        half_length = self._uniform(self.bounds.bar_half_length) if shape == "bar" else None
        return TaskSpec(
            task_id,
            shape=shape,
            object_half_width=half_width,
            object_half_height=half_height,
            object_half_length=half_length,
            object_mass=self._uniform(self.bounds.mass),
            object_table_friction=self._uniform(self.bounds.table_friction),
            object_finger_friction=self._uniform(self.bounds.finger_friction),
        )

    def resample_task(self, task: TaskSpec) -> TaskSpec:
        """Draw new sizes and properties for ``task``, keeping its id and shape."""
        if task.shape is None:
            raise ValueError("Only shaped tasks can be resampled.")
        return self._sample(task.task_id, task.shape)

    def sample_train_task(self) -> TaskSpec:
        return self.train_tasks[int(self.rng.integers(len(self.train_tasks)))]
