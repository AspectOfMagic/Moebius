"""Scripted pick-and-place experts with tunable grasp and transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MetaWorld observation layout (39 values).
_HAND = slice(0, 3)
_GRIPPER = 3
_OBJECT = slice(4, 7)
_PREVIOUS_GRIPPER = 18 + 3
_PREVIOUS_OBJECT_Z = 18 + 6
_GOAL = slice(36, 39)


@dataclass(frozen=True)
class PickPlaceExpert:
    """Stateless hover, descend, grasp, and transport controller."""

    grasp_height: float = 0.03
    descend_tolerance: float = 0.05
    max_object_height: float = 0.04
    close_threshold: float = 0.73
    settle: bool = False
    transport_gain: float = 10.0
    transport_speed: float = 1.0
    hand_floor: float | None = None
    grip_effort: float = 1.0
    open_effort: float = 0.0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float64)
        hand = observation[_HAND]
        gripper = observation[_GRIPPER]
        object_position = observation[_OBJECT]
        target_object = object_position + np.array([-0.005, 0.0, 0.0])
        goal = observation[_GOAL]
        speed = 1.0
        gain = 10.0
        if np.linalg.norm(hand[:2] - target_object[:2]) > 0.02:
            target = target_object + np.array([0.0, 0.0, 0.1])
        elif (
            abs(hand[2] - target_object[2]) > self.descend_tolerance
            and target_object[2] < self.max_object_height
            and (
                self.hand_floor is None
                or (hand[2] > self.hand_floor + 0.01 and object_position[2] <= observation[_PREVIOUS_OBJECT_Z] + 5e-4)
            )
        ):
            target = target_object + np.array([0.0, 0.0, self.grasp_height])
        elif gripper > self.close_threshold or (
            self.settle and abs(gripper - observation[_PREVIOUS_GRIPPER]) > 1e-3
        ):
            target = hand
        else:
            target = goal
            gain = self.transport_gain
            speed = self.transport_speed
        delta = np.clip(gain * (target - hand), -speed, speed)
        grab = self.grip_effort if np.linalg.norm(hand - object_position) < 0.07 else self.open_effort
        return np.clip(np.append(delta, grab), -1.0, 1.0).astype(np.float32)


EXPERT_VARIANTS: dict[str, PickPlaceExpert] = {
    "metaworld": PickPlaceExpert(),
    # Lower grasps reduce torque on tall and long objects.
    "low_grasp": PickPlaceExpert(grasp_height=0.01, descend_tolerance=0.02, hand_floor=0.05, open_effort=-1.0),
    "slow_transport": PickPlaceExpert(transport_speed=0.3, open_effort=-1.0),
    "careful": PickPlaceExpert(
        grasp_height=0.01,
        descend_tolerance=0.02,
        hand_floor=0.05,
        settle=True,
        transport_speed=0.3,
        open_effort=-1.0,
    ),
}


def expert_variant(name: str) -> PickPlaceExpert:
    try:
        return EXPERT_VARIANTS[name]
    except KeyError as error:
        raise ValueError(f"Unknown expert variant {name!r}; expected one of {tuple(EXPERT_VARIANTS)}.") from error


def dedicated_expert(env_name: str):
    """Return MetaWorld's one dedicated scripted policy for an environment."""
    from metaworld import policies

    policy_names = {
        "pick-place-v3": "SawyerPickPlaceV3Policy",
        "pick-place-wall-v3": "SawyerPickPlaceWallV3Policy",
        "shelf-place-v3": "SawyerShelfPlaceV3Policy",
        "pick-out-of-hole-v3": "SawyerPickOutOfHoleV3Policy",
    }
    try:
        return getattr(policies, policy_names[env_name])()
    except KeyError as error:
        raise ValueError(f"No dedicated expert is configured for {env_name!r}.") from error
