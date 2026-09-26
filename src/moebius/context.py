"""One context-construction rule shared by PEARL training, evaluation and recording.

Before this module, training sampled short chronological windows from anywhere
in an episode, while rollouts used the last few transitions of the previous
episode. The encoder was therefore trained on one context distribution and
deployed on another. Every caller now builds the context for adaptation
episode ``k`` from the ``k`` completed episodes before it, through
:meth:`ContextBuilder.build`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from moebius.replay_buffer import Batch

# MetaWorld v3 state layout: hand xyz (0:3), gripper opening (3), object xyz (4:7).
OBJECT_POSITION_SLICE = slice(4, 7)


def episode_batch(
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    rewards: Sequence[float],
    next_observations: Sequence[np.ndarray],
    dones: Sequence[bool],
) -> Batch:
    """Pack one completed episode as ``[T, ...]`` arrays."""
    return Batch(
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        np.asarray(next_observations, dtype=np.float32),
        np.asarray(dones, dtype=np.float32).reshape(-1, 1),
    )


def empty_context(observation_dim: int, action_dim: int) -> Batch:
    """A zero-length context. PEARL maps it to the prior mean ``z = 0``."""
    return Batch(
        np.zeros((0, observation_dim), dtype=np.float32),
        np.zeros((0, action_dim), dtype=np.float32),
        np.zeros((0, 1), dtype=np.float32),
        np.zeros((0, observation_dim), dtype=np.float32),
        np.zeros((0, 1), dtype=np.float32),
    )


def object_motion_mask(observations: np.ndarray, next_observations: np.ndarray, threshold: float) -> np.ndarray:
    """Transitions in which the object moved, computed from observations only."""
    displacement = next_observations[..., OBJECT_POSITION_SLICE] - observations[..., OBJECT_POSITION_SLICE]
    return np.linalg.norm(displacement, axis=-1) > threshold


@dataclass(frozen=True)
class ContextBuilder:
    """Subsample a pool of earlier-episode transitions into one encoder context.

    Hidden mass, friction and contact softness only affect transitions in
    which the object is touched, so up to ``motion_fraction`` of the context
    is drawn from transitions where the object moved. The rest is drawn
    uniformly so the encoder still sees ordinary transitions. The encoder is
    permutation invariant, so the order of the returned transitions is
    irrelevant.
    """

    context_size: int = 64
    motion_fraction: float = 0.75
    motion_threshold: float = 1e-3

    def __post_init__(self) -> None:
        if self.context_size < 1:
            raise ValueError("context_size must be positive.")
        if not 0 <= self.motion_fraction <= 1:
            raise ValueError("motion_fraction must be in [0, 1].")
        if self.motion_threshold < 0:
            raise ValueError("motion_threshold must be non-negative.")

    def pool_size(self, episodes: Sequence[Batch]) -> int:
        return int(sum(len(episode.observations) for episode in episodes))

    def build(self, episodes: Sequence[Batch], rng: np.random.Generator, size: int | None = None) -> Batch:
        """Return ``min(size, pool)`` distinct transitions from ``episodes``.

        ``episodes`` are the completed episodes before the current one in the
        same adaptation sequence. No episodes means an empty context.
        """
        if not episodes:
            raise ValueError("Use empty_context() when there are no previous episodes.")
        size = self.context_size if size is None else size
        pool = Batch(
            *(np.concatenate([getattr(episode, field) for episode in episodes]) for field in Batch.__dataclass_fields__)
        )
        count = min(size, len(pool.observations))
        motion = object_motion_mask(pool.observations, pool.next_observations, self.motion_threshold)
        moving, still = np.flatnonzero(motion), np.flatnonzero(~motion)
        moving_count = min(len(moving), int(round(self.motion_fraction * count)))
        chosen_moving = rng.choice(moving, size=moving_count, replace=False)
        # Fill the remainder from stationary transitions first, then from any
        # moving transitions not chosen yet (when the pool is mostly motion).
        filler = np.concatenate(
            (rng.permutation(still), rng.permutation(np.setdiff1d(moving, chosen_moving, assume_unique=True)))
        )
        chosen = rng.permutation(np.concatenate((chosen_moving, filler[: count - moving_count])).astype(np.int64))
        return Batch(*(getattr(pool, field)[chosen] for field in Batch.__dataclass_fields__))

    def motion_fraction_of(self, context: Batch) -> float:
        """Share of context transitions with object motion, for logging."""
        if context.observations.shape[-2] == 0:
            return 0.0
        return float(object_motion_mask(context.observations, context.next_observations, self.motion_threshold).mean())


class AdaptationContextBuffer:
    """Recent adaptation sequences for each task, stored as completed episodes.

    One sequence is the run of consecutive episodes on one task, exactly as in
    training rollouts and evaluation. Sampling picks a number of previous
    episodes ``k`` in ``[0, max_previous_episodes]`` and builds each task's
    context from the first ``k`` episodes of one of its stored sequences, so
    the encoder is trained on the same kind of context it receives before
    adaptation episode ``k``. Keeping only a few recent sequences keeps the
    context close to the current policy's behavior.
    """

    def __init__(
        self,
        task_ids: Iterable[int],
        observation_dim: int,
        action_dim: int,
        max_sequences_per_task: int = 4,
        seed: int = 1,
    ) -> None:
        ids = tuple(task_ids)
        if not ids:
            raise ValueError("At least one task id is required.")
        if len(set(ids)) != len(ids):
            raise ValueError("Task ids must be unique.")
        if max_sequences_per_task < 1:
            raise ValueError("max_sequences_per_task must be positive.")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self._sequences: dict[int, deque[list[Batch]]] = {
            task_id: deque(maxlen=max_sequences_per_task) for task_id in ids
        }
        self._active: dict[int, list[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] | None] = {
            task_id: None for task_id in ids
        }
        self.rng = np.random.default_rng(seed)

    def begin_sequence(self, task_id: int) -> None:
        """Start a new adaptation sequence; later episodes are appended to it."""
        self._ensure_task(task_id)
        if self._active[task_id]:
            raise RuntimeError(f"Task {task_id} has an unfinished episode.")
        self._active[task_id] = None
        self._sequences[task_id].append([])

    def begin_episode(self, task_id: int) -> None:
        self._ensure_task(task_id)
        if self._active[task_id]:
            raise RuntimeError(f"Task {task_id} already has an active context episode.")
        if not self._sequences[task_id]:
            self._sequences[task_id].append([])
        self._active[task_id] = []

    def add(
        self,
        task_id: int,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._ensure_task(task_id)
        if self._active[task_id] is None:
            raise RuntimeError(f"Call begin_episode({task_id}) before adding transitions.")
        self._active[task_id].append(
            (
                np.asarray(observation, dtype=np.float32).copy(),
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def end_episode(self, task_id: int) -> None:
        self._ensure_task(task_id)
        episode = self._active[task_id]
        if episode:
            self._sequences[task_id][-1].append(episode_batch(*zip(*episode, strict=True)))
        self._active[task_id] = None

    def max_previous_episodes(self, task_id: int) -> int:
        """Longest run of completed episodes stored for ``task_id``."""
        self._ensure_task(task_id)
        return max((len(sequence) for sequence in self._sequences[task_id]), default=0)

    def sample(
        self, task_ids: np.ndarray, builder: ContextBuilder, max_previous_episodes: int
    ) -> tuple[Batch, int]:
        """Return stacked ``[tasks, n, ...]`` contexts and the shared ``k``.

        ``k = 0`` yields zero-length contexts, matching the first episode of
        every adaptation sequence, where the agent acts under ``z = 0``.
        """
        if max_previous_episodes < 0:
            raise ValueError("max_previous_episodes must be non-negative.")
        task_ids = np.asarray(task_ids, dtype=np.int64)
        if task_ids.ndim != 1 or len(task_ids) == 0:
            raise ValueError("task_ids must be a non-empty one-dimensional array.")
        available = min(self.max_previous_episodes(int(task_id)) for task_id in task_ids)
        previous_episodes = int(self.rng.integers(0, min(max_previous_episodes, available) + 1))
        if previous_episodes == 0:
            contexts = [empty_context(self.observation_dim, self.action_dim) for _ in task_ids]
        else:
            histories = []
            for task_id in task_ids:
                eligible = [sequence for sequence in self._sequences[int(task_id)] if len(sequence) >= previous_episodes]
                sequence = eligible[int(self.rng.integers(len(eligible)))]
                histories.append(sequence[:previous_episodes])
            # Tensors need one context length for every task in the meta-batch.
            size = min(builder.context_size, min(builder.pool_size(history) for history in histories))
            contexts = [builder.build(history, self.rng, size) for history in histories]
        stacked = Batch(*(np.stack([getattr(context, field) for context in contexts]) for field in Batch.__dataclass_fields__))
        return stacked, previous_episodes

    def add_sequence(self, task_id: int, episodes: Sequence[Batch]) -> None:
        """Store an already completed sequence of episodes (demonstrations)."""
        self._ensure_task(task_id)
        if not episodes:
            raise ValueError("A sequence needs at least one episode.")
        self._sequences[task_id].append(list(episodes))

    def _ensure_task(self, task_id: int) -> None:
        if task_id not in self._sequences:
            raise KeyError(f"Unknown task id {task_id}.")
