"""Scripted pick-and-place experts with tunable grasp and transport.

``PickPlaceExpert()`` with default arguments reproduces MetaWorld's
``SawyerPickPlaceV3Policy`` exactly. The named variants change only how the
object is grasped and carried, which is what differs between objects that are
light or heavy, grippy or slippery, and stable or prone to sliding.
"""

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
    """A stateless pick-place controller over MetaWorld observations.

    Phases, as in MetaWorld's policy: hover above the object, descend, close
    the gripper, then carry the object to the goal.

    grasp_height: target height of the hand above the object centre while descending.
    descend_tolerance: descending stops once the hand is this close to the object in z.
    max_object_height: only descend while the object centre is below this height.
    close_threshold: wait until the normalised gripper opening drops below this.
    settle: also wait until the gripper opening stops changing between frames,
        so the grip is fully established before lifting.
    transport_gain, transport_speed: proportional gain and per-axis action cap
        while carrying the object (1.0 = MetaWorld's full speed).
    hand_floor: lowest hand height MetaWorld allows (pick-place clamps the hand
        at z = 0.05). When set, descending also stops once the hand reaches it,
        so a grasp target below the floor means "as low as the hand can go"
        instead of descending forever, and never resumes while the object is
        rising: at the floor the hand can sit more than ``descend_tolerance``
        above a short object, and lifting it would otherwise re-trigger the
        descent. ``None`` keeps MetaWorld's behavior.
    grip_effort: gripper action while grasping (MetaWorld: 1.0). With the
        environment's ``gripper_force_control`` it is the fraction of the
        gripper force limit to squeeze with. Training sets it per object to
        the lowest grip that reliably works (see ``object_grip_effort``).
    open_effort: gripper action away from the object (MetaWorld: 0.0). With
        ``gripper_force_control`` an action of 0 does not drive the fingers,
        so the variants used there open with -1.0 (identical under
        MetaWorld's position control).
    """

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
    # MetaWorld's scripted policy, unchanged.
    "metaworld": PickPlaceExpert(),
    # Grasp as low as the hand can reach (about 2-3 cm lower than MetaWorld's
    # policy): closer to the object's centre, steadier for tall or long objects.
    "low_grasp": PickPlaceExpert(grasp_height=0.01, descend_tolerance=0.02, hand_floor=0.05, open_effort=-1.0),
    # Carry at 30% speed: less inertial load on a slipping or heavy object.
    "slow_transport": PickPlaceExpert(transport_speed=0.3, open_effort=-1.0),
    # Low grasp, wait for a settled grip, then carry slowly.
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
