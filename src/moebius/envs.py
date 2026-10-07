from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

# All shapes use identity orientation; bars extend along x while fingers close along y.
SHAPES = ("cylinder", "box", "bar", "sphere")
SHAPE_ENVIRONMENTS = (
    "pick-place-v3",
    "pick-place-wall-v3",
    "shelf-place-v3",
    "pick-out-of-hole-v3",
)

# Plain constants keep geometry helpers independent of MuJoCo imports.
GEOM_SPHERE = 2
GEOM_CYLINDER = 5
GEOM_BOX = 6

# Reuse pick-place's empty second-object slots. Force entries are scaled newtons:
#   11     grip-control phase (0 = closing, 1 = post-contact squeeze)
#   12:15  total force from the fingers on the object (world x, y, z)
#   15     magnitude of horizontal table force
#   16     vertical table force
#   17     grip squeeze: summed normal force between finger pads and object
CONTACT_SLICE = slice(11, 18)
CONTACT_LATCH_INDEX = 11
CONTACT_FORCE_SCALE = 20.0
# Clip impact spikes to keep sensor features near the state scale.
CONTACT_FEATURE_CLIP = 5.0
# Allow measurement noise above the calibrated grip.
GRIP_SQUEEZE_TOLERANCE = 1.10


@dataclass(frozen=True)
class ObjectGeometry:
    """MuJoCo geom parameters and derived quantities for one object."""

    geom_type: int
    size: tuple[float, float, float]
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
    """Build MuJoCo geometry and inertia for an object primitive."""
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
    """Sampling ranges for object geometry and hidden dynamics."""

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
    """MetaWorld's caging reward using the live object half-width."""
    from metaworld.utils import reward_utils

    # Pick-place omits generic thresholds; restore its native defaults.
    pick_place_defaults = pad_success_thresh <= 0 or xz_thresh <= 0
    x_z_success_margin = 0.005 if xz_thresh <= 0 else xz_thresh
    radius = _grip_half_width(environment.model, environment.model.geom("objGeom").id)
    # Preserve native pad clearance as object width changes.
    native_radius = 0.015 if obj_radius <= 0 else obj_radius
    native_pad_success = 0.05 if pad_success_thresh <= 0 else pad_success_thresh
    pad_success_margin = radius + max(0.0, native_pad_success - native_radius)
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
    # Saturate grasp reward at the calibrated effort.
    effort_override = getattr(environment, "_moebius_desired_gripper_effort", None)
    if effort_override is not None:
        desired_gripper_effort = float(effort_override)
    gripper_closed = min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
    caging = reward_utils.hamacher_product(y_caging, x_z_caging)
    gripping = gripper_closed if caging > 0.97 else 0.0
    caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
    if high_density or pick_place_defaults:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if medium_density:
        tcp_to_obj = float(np.linalg.norm(np.asarray(obj_pos) - tcp))
        tcp_to_obj_init = float(np.linalg.norm(environment.obj_init_pos - environment.init_tcp))
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=abs(tcp_to_obj_init - object_reach_radius),
            sigmoid="long_tail",
        )
        caging_and_gripping = (caging_and_gripping + float(reach)) / 2
    return caging_and_gripping


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
    """MetaWorld wrapper for object geometry, dynamics, sensing, and grip control."""

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
        contact_observations: bool = False,
        gripper_force_control: bool = False,
        grip_penalty: float = 0.0,
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
        self.env_name = env_name
        self._ids_cached = False
        self._base: dict[str, Any] = {}
        self._shape_aware_reward_installed = False
        self._gripper_force_limit: float | None = None
        self._allowed_squeeze: float | None = None
        # Latching prevents a weak grip from regaining automatic full force.
        self._gripper_contact_latched = False
        self.unwrapped._moebius_desired_gripper_effort = None
        if grip_penalty < 0:
            raise ValueError("grip_penalty must be non-negative.")
        self.grip_penalty = grip_penalty
        if gripper_force_control and gripper_force_limit is None:
            raise ValueError("gripper_force_control needs a gripper_force_limit (the force of a full command).")
        self.gripper_force_control = gripper_force_control
        if gripper_force_limit is not None:
            self.set_gripper_force_limit(gripper_force_limit)
        self.contact_observations = contact_observations
        if contact_observations or gripper_force_control:
            if env_name != "pick-place-v3":
                raise ValueError("Contact sensing and force control are supported only for pick-place-v3.")
            self._pick_place_model_ids()
            unwrapped = self.unwrapped
            current_frame = unwrapped._get_curr_obs_combined_no_goal

            def current_frame_with_control_state() -> np.ndarray:
                frame = current_frame()
                self._write_control_observation(frame)
                return frame

            # MetaWorld uses this method for both frames in its observation.
            unwrapped._get_curr_obs_combined_no_goal = current_frame_with_control_state
            low = self.observation_space.low.copy()
            high = self.observation_space.high.copy()
            sensor_indices = list(range(11, 18)) if contact_observations else [CONTACT_LATCH_INDEX]
            for offset in (0, 18):
                low[np.asarray(sensor_indices) + offset] = -CONTACT_FEATURE_CLIP
                high[np.asarray(sensor_indices) + offset] = CONTACT_FEATURE_CLIP
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    def _pick_place_model_ids(self) -> None:
        """Cache pick-place ids and the nominal values this wrapper changes."""
        if self._ids_cached:
            return
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure object properties.") from error

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
        # Friction acts on MetaWorld's invisible table collision geometry.
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
        """Swap object geometry and refresh its collision bounds."""
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
        self._gripper_force_limit = limit
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

    def _object_contact_forces(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return world-frame finger/table forces and pad-normal squeeze."""
        import mujoco

        self._pick_place_model_ids()
        model, data = self.unwrapped.model, self.unwrapped.data
        finger_ids = set(self._finger_geom_ids)
        table_ids = set(self._table_geom_ids)
        finger_force = np.zeros(3)
        table_force = np.zeros(3)
        squeeze = 0.0
        wrench = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if self._object_geom_id not in (geom1, geom2):
                continue
            other = geom1 if geom2 == self._object_geom_id else geom2
            if other not in finger_ids and other not in table_ids:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            force = contact.frame.reshape(3, 3).T @ wrench[:3]
            if geom1 == self._object_geom_id:
                force = -force
            if other in finger_ids:
                finger_force += force
                if other in self._pad_geom_ids:
                    squeeze += abs(wrench[0])
            else:
                table_force += force
        return finger_force, table_force, squeeze

    def grip_force(self) -> float:
        """Total normal force (N) between the finger pads and the object right now."""
        return self._object_contact_forces()[2]

    def _object_bilateral_finger_contact(self) -> bool:
        """Return whether both finger sides currently contact the object."""
        self._pick_place_model_ids()
        data = self.unwrapped.data
        right_finger_ids = set(self._finger_geom_ids[:2])
        left_finger_ids = set(self._finger_geom_ids[2:])
        right_contact = False
        left_contact = False
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self._object_geom_id:
                other = geom2
            elif geom2 == self._object_geom_id:
                other = geom1
            else:
                continue
            right_contact |= other in right_finger_ids
            left_contact |= other in left_finger_ids
        return right_contact and left_contact

    def contact_features(self) -> np.ndarray:
        """The seven contact observation values (see ``CONTACT_SLICE``)."""
        finger_force, table_force, squeeze = self._object_contact_forces()
        forces = np.concatenate((finger_force, [np.linalg.norm(table_force[:2]), table_force[2], squeeze]))
        scaled_forces = np.clip(forces / CONTACT_FORCE_SCALE, -CONTACT_FEATURE_CLIP, CONTACT_FEATURE_CLIP)
        return np.concatenate(([float(self._gripper_contact_latched)], scaled_forces))

    def _write_control_observation(self, frame: np.ndarray) -> None:
        """Expose the force-control phase, plus forces when requested."""
        if self.contact_observations:
            frame[CONTACT_SLICE] = self.contact_features()
        elif self.gripper_force_control:
            frame[CONTACT_LATCH_INDEX] = float(self._gripper_contact_latched)

    def set_task_properties(self, task: TaskSpec) -> None:
        """Apply object geometry and dynamics; nominal tasks restore defaults."""
        if self.env_name not in SHAPE_ENVIRONMENTS:
            raise ValueError(f"Object shapes are supported only for {SHAPE_ENVIRONMENTS}.")
        if task.env_name is not None and task.env_name != self.env_name:
            raise ValueError(f"Task requires {task.env_name}, but this environment is {self.env_name}.")
        try:
            import mujoco
        except ImportError as error:
            raise ImportError("MuJoCo is required to configure object properties.") from error
        self._pick_place_model_ids()
        model = self.unwrapped.model
        base = self._base
        body, geom = self._object_body_id, self._object_geom_id
        self._allowed_squeeze = None
        self._gripper_contact_latched = False
        self.unwrapped._moebius_desired_gripper_effort = None
        if task.object_grip_effort is not None and self._gripper_force_limit is not None:
            if not 0 < task.object_grip_effort <= 1:
                raise ValueError("object_grip_effort must be in (0, 1].")
            self._allowed_squeeze = GRIP_SQUEEZE_TOLERANCE * task.object_grip_effort * 2 * self._gripper_force_limit
            if self.gripper_force_control:
                self.unwrapped._moebius_desired_gripper_effort = task.object_grip_effort

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

        if not self._shape_aware_reward_installed and hasattr(self.unwrapped, "_gripper_caging_reward"):
            self.unwrapped._gripper_caging_reward = MethodType(_shape_aware_pick_place_caging_reward, self.unwrapped)
            self._shape_aware_reward_installed = True

        sizes = (task.object_half_width, task.object_half_height)
        if any(value is None or value <= 0 for value in sizes):
            raise ValueError("All object size values must be positive.")
        optional_properties = (task.object_mass, task.object_table_friction, task.object_finger_friction)
        if any(value is not None and value <= 0 for value in optional_properties):
            raise ValueError("Object mass and friction values must be positive when supplied.")
        object_mass = base["mass"] if task.object_mass is None else task.object_mass
        geometry = object_geometry(
            task.shape, task.object_half_width, task.object_half_height, task.object_half_length, object_mass
        )
        extents = np.asarray(geometry.half_extents)
        aabb = np.concatenate((np.zeros(3), extents))
        self._set_object_geometry(geometry.geom_type, np.asarray(geometry.size), geometry.bounding_radius, aabb, aabb)
        model.body_mass[body] = object_mass
        model.body_inertia[body] = geometry.inertia
        model.geom_friction[geom] = (1e-5, 1e-6, 1e-7)
        # Enable the table collision only for the object.
        model.geom_contype[self._table_geom_ids] = self._TABLE_CONTACT_TYPE
        model.geom_conaffinity[self._table_geom_ids] = 0
        model.geom_contype[geom] = base["object_contype"]
        model.geom_conaffinity[geom] = base["object_conaffinity"] | self._TABLE_CONTACT_TYPE
        if task.object_table_friction is None:
            model.geom_friction[self._table_geom_ids] = base["table_friction"]
        else:
            model.geom_friction[self._table_geom_ids] = base["table_friction"] * (
                task.object_table_friction / base["table_friction"][:, :1]
            )
        if task.object_finger_friction is None:
            model.geom_friction[self._finger_geom_ids] = base["finger_friction"]
        else:
            model.geom_friction[self._finger_geom_ids] = base["finger_friction"] * (
                task.object_finger_friction / base["finger_friction"][:, :1]
            )
        mujoco.mj_setConst(model, self.unwrapped.data)

    def _set_finger_force(self, force: float) -> None:
        """Limit each finger's force (N)."""
        model = self.unwrapped.model
        model.actuator_forcelimited[self._finger_actuator_ids] = 1
        model.actuator_forcerange[self._finger_actuator_ids] = (-force, force)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        precontact_closing = False
        if self.gripper_force_control:
            action = np.array(action, dtype=np.float32)
            effort = float(np.clip(action[3], -1.0, 1.0))
            reward_action = action.copy()
            if effort > 0:
                if self._object_bilateral_finger_contact():
                    self._gripper_contact_latched = True
                precontact_closing = not self._gripper_contact_latched
                # Closing is fixed-speed; post-contact magnitude selects force.
                applied_effort = 1.0 if precontact_closing else effort
                self._set_finger_force(applied_effort * self._gripper_force_limit)
                action[3] = 1.0
                reward_action[3] = applied_effort
            elif effort < 0:
                self._gripper_contact_latched = False
                self._set_finger_force(self._gripper_force_limit)
                action[3] = -1.0
                reward_action[3] = -1.0
            else:
                self._set_finger_force(0.0)
                action[3] = 1.0
                reward_action[3] = 0.0
        observation, reward, terminated, truncated, info = super().step(action)
        if self.gripper_force_control:
            bilateral_contact = self._object_bilateral_finger_contact()
            if effort > 0 and bilateral_contact:
                self._gripper_contact_latched = True
            # Refresh the phase after the post-step contact check.
            self._write_control_observation(observation)
            if getattr(self.unwrapped, "_prev_obs", None) is not None:
                self._write_control_observation(self.unwrapped._prev_obs)
            # Recompute dense reward with the physically applied effort.
            reward, reward_info = self.unwrapped.evaluate_state(observation, reward_action)
            info.update(reward_info)
            info["gripper_precontact_closing"] = precontact_closing
            info["gripper_requested_effort"] = effort
            info["gripper_contact_latched"] = self._gripper_contact_latched
            info["gripper_bilateral_contact"] = bilateral_contact
        info["grip_penalty"] = 0.0
        if self.grip_penalty > 0 and self._allowed_squeeze is not None and not precontact_closing:
            # Cap impact spikes; automatic pre-contact force is not penalized.
            squeeze = min(self.grip_force(), 2 * self._gripper_force_limit)
            excess = max(0.0, squeeze - self._allowed_squeeze)
            info["grip_penalty"] = self.grip_penalty * excess / (2 * self._gripper_force_limit)
            reward -= info["grip_penalty"]
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if self.gripper_force_control:
            self._set_finger_force(self._gripper_force_limit)
            self._gripper_contact_latched = False
        observation, info = super().reset(seed=seed, options=options)
        # Rest resized objects on the table and update reward baselines.
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
            unwrapped._prev_obs = unwrapped._get_curr_obs_combined_no_goal()
            observation = unwrapped._get_obs()
        return observation, info


@dataclass(frozen=True)
class TaskSpec:
    """Task properties kept outside the agent observation."""

    task_id: int
    reset_seed: int | None = None
    shape: str | None = None
    object_half_width: float | None = None
    object_half_height: float | None = None
    object_half_length: float | None = None
    object_mass: float | None = None
    object_table_friction: float | None = None
    object_finger_friction: float | None = None
    object_grip_effort: float | None = None
    env_name: str | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TaskSpec:
        """Rebuild a task, converting legacy cylinder checkpoints."""
        values = dict(values)
        values.pop("object_crush_force", None)  # briefly-used fragile objects
        if "object_radius" in values:
            radius = values.pop("object_radius")
            height = values.pop("object_height", None)
            values.pop("contact_softness", None)
            if radius is not None:
                values.setdefault("shape", "cylinder")
                values["object_half_width"] = radius
                values["object_half_height"] = None if height is None else height / 2
        return cls(**values)


class MetaWorldTaskSuite:
    """Compatible MetaWorld environments selected by ``TaskSpec.env_name``."""

    def __init__(
        self,
        env_names: tuple[str, ...],
        seed: int,
        max_episode_steps: int,
        **environment_options: Any,
    ) -> None:
        if not env_names:
            raise ValueError("env_names must not be empty.")
        self.environments = {
            name: MetaWorldSingleTaskEnv(name, seed + index, max_episode_steps, **environment_options)
            for index, name in enumerate(env_names)
        }
        self.active = self.environments[env_names[0]]
        reference_observation = self.active.observation_space
        reference_action = self.active.action_space
        for name, environment in self.environments.items():
            if environment.observation_space.shape != reference_observation.shape:
                raise ValueError(f"{name} has an incompatible observation space.")
            if environment.action_space.shape != reference_action.shape:
                raise ValueError(f"{name} has an incompatible action space.")

    @property
    def observation_space(self):
        return self.active.observation_space

    @property
    def action_space(self):
        return self.active.action_space

    @property
    def env_name(self) -> str:
        return self.active.env_name

    def set_task_properties(self, task: TaskSpec) -> None:
        env_name = task.env_name or self.active.env_name
        try:
            self.active = self.environments[env_name]
        except KeyError as error:
            raise ValueError(f"Task requires unknown environment {env_name!r}.") from error
        self.active.set_task_properties(task)

    def reset(self, *args: Any, **kwargs: Any):
        return self.active.reset(*args, **kwargs)

    def step(self, action: np.ndarray):
        return self.active.step(action)

    def render(self):
        return self.active.render()

    def close(self) -> None:
        for environment in self.environments.values():
            environment.close()


class GoalTaskSampler:
    """Create disjoint seeded train and test configurations."""

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
    """Sample balanced environment/shape tasks with optional dynamics."""

    def __init__(
        self,
        num_train_tasks: int,
        num_test_tasks: int,
        bounds: ObjectPropertyBounds | None = None,
        seed: int = 1,
        env_names: tuple[str, ...] = ("pick-place-v3",),
        vary_dynamics: bool = True,
    ) -> None:
        if num_train_tasks < 1 or num_test_tasks < 1:
            raise ValueError("num_train_tasks and num_test_tasks must both be positive.")
        self.bounds = ObjectPropertyBounds() if bounds is None else bounds
        self.rng = np.random.default_rng(seed)
        if not env_names or any(name not in SHAPE_ENVIRONMENTS for name in env_names):
            raise ValueError(f"env_names must be a non-empty subset of {SHAPE_ENVIRONMENTS}.")
        self.env_names = tuple(env_names)
        self.vary_dynamics = vary_dynamics
        shapes = self.bounds.shapes
        # Cover environments before advancing to the next shape.
        combinations = tuple((env_name, shape) for shape in shapes for env_name in self.env_names)
        self.train_tasks = tuple(
            self._sample(task_id, *combinations[task_id % len(combinations)]) for task_id in range(num_train_tasks)
        )
        self.test_tasks = tuple(
            self._sample(task_id, *combinations[task_id % len(combinations)]) for task_id in range(num_test_tasks)
        )

    def _uniform(self, bounds: tuple[float, float]) -> float:
        return float(self.rng.uniform(*bounds))

    def _sample(self, task_id: int, env_name: str, shape: str) -> TaskSpec:
        half_width = self._uniform(self.bounds.half_width)
        half_height = half_width if shape == "sphere" else self._uniform(self.bounds.half_height)
        half_length = self._uniform(self.bounds.bar_half_length) if shape == "bar" else None
        dynamics = (
            {
                "object_mass": self._uniform(self.bounds.mass),
                "object_table_friction": self._uniform(self.bounds.table_friction),
                "object_finger_friction": self._uniform(self.bounds.finger_friction),
            }
            if self.vary_dynamics
            else {}
        )
        return TaskSpec(
            task_id,
            shape=shape,
            object_half_width=half_width,
            object_half_height=half_height,
            object_half_length=half_length,
            env_name=env_name,
            **dynamics,
        )

    def resample_task(self, task: TaskSpec) -> TaskSpec:
        """Draw a replacement while keeping task id, environment and shape."""
        if task.shape is None:
            raise ValueError("Only shaped tasks can be resampled.")
        return self._sample(task.task_id, task.env_name or self.env_names[0], task.shape)

    def sample_train_task(self) -> TaskSpec:
        return self.train_tasks[int(self.rng.integers(len(self.train_tasks)))]
