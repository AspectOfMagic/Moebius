from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


def concatenate_batches(*batches: Batch) -> Batch:
    """Join equally shaped transition batches along their sample dimension."""
    if not batches:
        raise ValueError("At least one batch is required.")
    return Batch(*(np.concatenate([getattr(batch, field) for batch in batches], axis=0) for field in Batch.__dataclass_fields__))


@dataclass
class TaskBatch:
    """Independent context and RL samples for a batch of meta-training tasks.

    All fields in ``context`` and ``rl`` have a leading task dimension.  The
    samples are deliberately independent: context identifies the task, while
    the RL batch provides the Bellman/actor updates for that inferred task.
    """

    task_ids: np.ndarray
    context: Batch
    rl: Batch


def concatenate_task_batch_rl(primary: TaskBatch, secondary: TaskBatch) -> TaskBatch:
    """Mix same-task online and demonstration samples, retaining online context."""
    if not np.array_equal(primary.task_ids, secondary.task_ids):
        raise ValueError("Task batches must use identical task ids in the same order.")
    return TaskBatch(
        task_ids=primary.task_ids,
        context=primary.context,
        rl=Batch(
            *(np.concatenate((getattr(primary.rl, field), getattr(secondary.rl, field)), axis=1) for field in Batch.__dataclass_fields__)
        ),
    )


class ReplayBuffer:
    def __init__(self, observation_dim: int, action_dim: int, capacity: int, seed: int = 1) -> None:
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation: np.ndarray, action: np.ndarray, reward: float, next_observation: np.ndarray, done: bool) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {self.size}")
        indices = self.rng.integers(self.size, size=batch_size)
        return Batch(
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )


class MultiTaskReplayBuffer:
    """One off-policy replay buffer per task, as required by PEARL.

    A task id never appears as an input to the policy.  It is only a storage
    key that makes it possible to sample a coherent context set and an RL
    minibatch from the same task during meta-training.
    """

    def __init__(
        self, task_ids: Iterable[int], observation_dim: int, action_dim: int, capacity: int, seed: int = 1
    ) -> None:
        ids = tuple(task_ids)
        if not ids:
            raise ValueError("At least one task id is required.")
        if len(set(ids)) != len(ids):
            raise ValueError("Task ids must be unique.")
        self.buffers = {
            task_id: ReplayBuffer(observation_dim, action_dim, capacity, seed + index)
            for index, task_id in enumerate(ids)
        }
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        task_id: int,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer(task_id).add(observation, action, reward, next_observation, done)

    def size(self, task_id: int) -> int:
        return self._buffer(task_id).size

    def eligible_task_ids(self, minimum_size: int) -> tuple[int, ...]:
        return tuple(task_id for task_id, buffer in self.buffers.items() if buffer.size >= minimum_size)

    def sample_meta_batch(self, meta_batch_size: int, batch_size: int, context_size: int) -> TaskBatch:
        if meta_batch_size < 1 or batch_size < 1 or context_size < 1:
            raise ValueError("meta_batch_size, batch_size, and context_size must all be positive.")
        minimum_size = max(batch_size, context_size)
        eligible = self.eligible_task_ids(minimum_size)
        if len(eligible) < meta_batch_size:
            raise ValueError(
                f"Need {meta_batch_size} tasks with at least {minimum_size} transitions; found {len(eligible)}."
            )
        task_ids = self.rng.choice(np.asarray(eligible), size=meta_batch_size, replace=False)
        return self.sample_meta_batch_for_task_ids(task_ids, batch_size, context_size)

    def sample_meta_batch_for_task_ids(
        self, task_ids: np.ndarray, batch_size: int, context_size: int
    ) -> TaskBatch:
        if batch_size < 1 or context_size < 1:
            raise ValueError("batch_size and context_size must both be positive.")
        task_ids = np.asarray(task_ids, dtype=np.int64)
        if task_ids.ndim != 1 or len(task_ids) == 0 or len(np.unique(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be a non-empty one-dimensional array of unique ids.")
        minimum_size = max(batch_size, context_size)
        undersized = [int(task_id) for task_id in task_ids if self._buffer(int(task_id)).size < minimum_size]
        if undersized:
            raise ValueError(f"Tasks {undersized} need at least {minimum_size} transitions.")
        contexts = [self._buffer(int(task_id)).sample(context_size) for task_id in task_ids]
        rl_batches = [self._buffer(int(task_id)).sample(batch_size) for task_id in task_ids]
        return TaskBatch(
            task_ids=task_ids,
            context=self._stack(contexts),
            rl=self._stack(rl_batches),
        )

    def _buffer(self, task_id: int) -> ReplayBuffer:
        try:
            return self.buffers[task_id]
        except KeyError as error:
            raise KeyError(f"Unknown task id {task_id}.") from error

    @staticmethod
    def _stack(batches: list[Batch]) -> Batch:
        return Batch(*(np.stack([getattr(batch, field) for batch in batches]) for field in Batch.__dataclass_fields__))
