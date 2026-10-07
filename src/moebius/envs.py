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
SHAPE_ENVIRONMENTS = (
    "pick-place-v3",
    "pick-place-wall-v3",
    "shelf-place-v3",
    "pick-out-of-hole-v3",
)

# MuJoCo mjtGeom values. Kept as plain integers so the geometry helpers can be
# used (and tested) without importing MuJoCo; the environment checks them
# against mujoco.mjtGeom when it applies a shape.
GEOM_SPHERE = 2
GEOM_CYLINDER = 5
GEOM_BOX = 6

# Contact observations. Pick-place observes one object, so slots 11:18 of
# each 18-value frame (reserved by MetaWorld for a second object) are always
# zero; with ``contact_observations`` they carry the forces on the object
# instead. The force entries are in units of CONTACT_FORCE_SCALE newtons:
#   11     grip-control phase (0 = closing, 1 = post-contact squeeze)
#   12:15  total force from the fingers on the object (world x, y, z)
#   15     magnitude of horizontal table force
#   16     vertical table force
#   17     grip squeeze: summed normal force between finger pads and object
CONTACT_SLICE = slice(11, 18)
CONTACT_LATCH_INDEX = 11
CONTACT_FORCE_SCALE = 20.0
# Contact readings are clipped to +-5 units (100 N). On impact the measured
# squeeze spikes to 400-1500 N for a step; unclipped, those spikes put inputs
# of 20-75 into the networks while every other input is about 1.
CONTACT_FEATURE_CLIP = 5.0
# Grip penalty: squeeze up to this multiple of a task's calibrated grip
# (object_grip_effort) is free, which leaves room for the noise in the
# measured squeeze.
GRIP_SQUEEZE_TOLERANCE = 1.10


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
    from metaworld.utils import reward_utils

    # Pick-place's specialized helper calls this method without the generic
    # thresholds; its native values are 0.05, 0.005 and high-density shaping.
    pick_place_defaults = pad_success_thresh <= 0 or xz_thresh <= 0
    x_z_success_margin = 0.005 if xz_thresh <= 0 else xz_thresh
    radius = _grip_half_width(environment.model, environment.model.geom("objGeom").id)
    # Preserve the native clearance between the object and the successful pad
    # spacing. Using the native absolute threshold breaks when a substituted
    # object is wider than it (notably pick-out-of-hole: 0.02 m threshold).
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
    # Force-controlled tasks calibrate the lowest reliable effort. Reward a
    # complete grasp once the command reaches that effort instead of continuing
    # to pull the policy toward a full squeeze. Regular MetaWorld tasks keep the
    # default desired effort of 1.0.
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
    """Create one goal-observable MetaWorld task through Gymnasium.

    MT1 keeps the object and goal state observable. For PEARL, a task sets the
    pick-place object's shape, size and hidden physical properties, while its
    task identifier stays outside the agent observation.

    ``gripper_force_limit`` (newtons per finger, ``None`` = MetaWorld's
    unlimited servo) caps the finger actuators. It is a fixed property of the
    robot, not of a task.

    ``contact_observations`` (pick-place only) puts the forces on the object
    into observation slots 11:18 (see ``CONTACT_SLICE``), like a wrist
    force sensor and tactile finger pads. MetaWorld moves the hand
    kinematically, so without them a heavy and a light object look the same
    until one slips; with them, lifting reads out the weight directly.

    ``gripper_force_control`` separates closing motion from squeeze force. A
    positive gripper action closes the fingers with one fixed, full-force
    position-servo command until either finger first contacts the object.
    From that contact onward, the action magnitude is the requested squeeze
    force as a fraction of ``gripper_force_limit``. A negative action opens
    at the same fixed drive and starts a new approach; zero applies no drive.

    ``grip_penalty`` charges for squeezing harder than an object needs. A
    task's ``object_grip_effort`` is the lowest grip the scripted expert
    reliably succeeded with (found by the training setup); each step, the
    reward drops by ``grip_penalty`` times the squeeze above
    ``GRIP_SQUEEZE_TOLERANCE`` times that grip, as a fraction of the full
    squeeze. The right grip depends on the object's hidden mass and friction,
    and so does the reward. Tasks without a calibrated grip are not
    penalized.
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
        # Contact is latched for the duration of a grasp. Without a latch, a
        # deliberately weak squeeze that briefly loses contact would switch
        # back to full pre-contact drive and silently rescue the grasp.
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

            # MetaWorld builds both the current frame and the frame-stacked
            # previous one from this method, so both carry the control phase.
            unwrapped._get_curr_obs_combined_no_goal = current_frame_with_control_state
            low = self.observation_space.low.copy()
            high = self.observation_space.high.copy()
            sensor_indices = list(range(11, 18)) if contact_observations else [CONTACT_LATCH_INDEX]
            for offset in (0, 18):
                low[np.asarray(sensor_indices) + offset] = -CONTACT_FEATURE_CLIP
                high[np.asarray(sensor_indices) + offset] = CONTACT_FEATURE_CLIP
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

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
        """Forces on the object right now: (from fingers, from table, pad squeeze).

        The first two are world-frame vectors in newtons. MuJoCo's contact
        normal points from geom1 to geom2 and a positive normal force pushes
        geom2 along it, so the force on the object is flipped when the object
        is geom1.
        """
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
        """Whether finger geometry on both sides currently touches the object.

        A valid grasp may touch a pad or the adjacent claw geometry. Requiring
        both pad geoms simultaneously was unnecessarily brittle, while the old
        any-contact trigger switched to squeeze control after a one-sided hit.
        """
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

    # -- tasks ---------------------------------------------------------------

    def set_task_properties(self, task: TaskSpec) -> None:
        """Apply a task's object shape, size and hidden physical properties.

        Friction is assigned to the table and finger collision geoms; MuJoCo
        uses the larger of the two geoms' frictions for a contact, so the
        object itself keeps a negligible friction to make these two task
        properties independent. Goal tasks (``task.shape is None``) restore
        MetaWorld's nominal object.
        """
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
            # Both fingers push with grip_effort x the limit.
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
        # MetaWorld disables its invisible table collision box. Give it a
        # dedicated collision type and give only the object affinity for it:
        # the table then contacts the object, not the gripper or robot.
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
                # Before contact, magnitude must not change closing speed.
                # After contact, it is purely the requested squeeze force.
                applied_effort = 1.0 if precontact_closing else effort
                self._set_finger_force(applied_effort * self._gripper_force_limit)
                action[3] = 1.0
                # MetaWorld's caging reward follows the command actually sent
                # to the gripper: fixed close before contact, requested effort
                # after contact.
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
            # super().step built the observation before the post-step contact
            # check above. Refresh its current frame and the saved previous
            # frame so the actor always observes the mode actually in effect.
            self._write_control_observation(observation)
            if getattr(self.unwrapped, "_prev_obs", None) is not None:
                self._write_control_observation(self.unwrapped._prev_obs)
            # MetaWorld uses the gripper action in its dense caging reward. The
            # position-servo action and post-contact force request have
            # different meanings, so re-evaluate with the command that was
            # physically active during this step.
            reward, reward_info = self.unwrapped.evaluate_state(observation, reward_action)
            info.update(reward_info)
            info["gripper_precontact_closing"] = precontact_closing
            info["gripper_requested_effort"] = effort
            info["gripper_contact_latched"] = self._gripper_contact_latched
            info["gripper_bilateral_contact"] = bilateral_contact
        info["grip_penalty"] = 0.0
        if self.grip_penalty > 0 and self._allowed_squeeze is not None and not precontact_closing:
            # Capped at the gripper's full squeeze so impact spikes do not
            # cost more than the step's whole reward. The first contact step
            # is exempt: it used the automatic closing drive, not the agent's
            # requested post-contact squeeze, so charging it would be unfair.
            squeeze = min(self.grip_force(), 2 * self._gripper_force_limit)
            excess = max(0.0, squeeze - self._allowed_squeeze)
            info["grip_penalty"] = self.grip_penalty * excess / (2 * self._gripper_force_limit)
            reward -= info["grip_penalty"]
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if self.gripper_force_control:
            # MetaWorld's reset opens the gripper; do it with full force.
            self._set_finger_force(self._gripper_force_limit)
            self._gripper_contact_latched = False
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
    the object is ``shape`` with the given sizes (metres). Optional mass and
    friction values override the environment defaults. ``object_grip_effort``
    is a force-control calibration used by the optional grip penalty.
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
    object_grip_effort: float | None = None
    # None preserves old single-environment checkpoints.
    env_name: str | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TaskSpec:
        """Rebuild a saved task, converting pre-shape checkpoints.

        Older checkpoints stored a cylinder's radius, full height and contact
        softness. They become a cylinder with the same size; softness is no
        longer a task property and is dropped.
        """
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
    """A small pool of compatible MetaWorld environments selected by TaskSpec.

    The policy still sees one common 39D observation and 4D action space. The
    environment name is bookkeeping used only to activate the right simulator;
    it is never appended to the observation.
    """

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
    """Sample environment/shape combinations and optional dynamics for PEARL.

    Environment/shape pairs are assigned round-robin; sizes are sampled from
    ``bounds``. Mass and friction are sampled only when ``vary_dynamics`` is
    true, otherwise each environment retains its nominal values.
    """

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
        # Interleave environments first so even a small evaluation set covers
        # every environment; later cycles introduce the next shape.
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
