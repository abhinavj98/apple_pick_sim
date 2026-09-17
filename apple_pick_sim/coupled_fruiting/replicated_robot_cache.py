"""Process-local cache for replicated FR3 USD models and SolverMuJoCo."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CachedReplicatedRobot:
    """Finalized replicated FR3 model plus the MuJoCo solver that owns it."""

    robot_model: Any
    template_tcp: int
    mj_solver: Any
    template_model: Any
    rest_joint_q: np.ndarray
    rest_joint_qd: np.ndarray

    def restore_rest_pose(self) -> None:
        """Reset template and batched joint coordinates to the cached rest pose."""
        self.template_model.joint_q.assign(self.rest_joint_q)
        self.template_model.joint_qd.assign(self.rest_joint_qd)
        from apple_pick_sim.coupled_fruiting.batched_build import (
            _broadcast_robot_state_from_template,
        )

        _broadcast_robot_state_from_template(self.template_model, self.robot_model)


@dataclass
class ReplicatedRobotMuJoCoCache:
    """In-process map from robot layout key to a live MuJoCo FR3 instance."""

    _entries: dict[tuple[Any, ...], CachedReplicatedRobot] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_create(
        self,
        key: tuple[Any, ...],
        factory: Callable[[], Any],
    ) -> Any:
        hit = self._entries.get(key)
        if hit is not None:
            self.hits += 1
            restore = getattr(hit, "restore_rest_pose", None)
            if callable(restore):
                restore()
            return hit
        self.misses += 1
        entry = factory()
        self._entries[key] = entry
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


_PROCESS_CACHE = ReplicatedRobotMuJoCoCache()


def process_replicated_robot_cache() -> ReplicatedRobotMuJoCoCache:
    return _PROCESS_CACHE


def clear_process_replicated_robot_cache() -> None:
    _PROCESS_CACHE.clear()


def make_replicated_robot_cache_key(
    *,
    num_envs: int,
    device: str,
    usd_path: Path | str | None,
    add_apple_payload: bool,
    robot_base_pos: tuple[float, float, float] | None,
    mujoco_kwargs: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """Stable key for a replicated FR3 + SolverMuJoCo configuration."""
    path = "" if usd_path is None else str(usd_path)
    kwargs_blob = json.dumps(
        mujoco_kwargs or {},
        sort_keys=True,
        default=str,
    )
    base = None if robot_base_pos is None else tuple(float(v) for v in robot_base_pos)
    return (
        int(num_envs),
        str(device),
        path,
        bool(add_apple_payload),
        base,
        kwargs_blob,
    )


def acquire_replicated_fr3_robot(
    *,
    reuse: bool,
    key: tuple[Any, ...],
    factory: Callable[[], Any],
    cache: ReplicatedRobotMuJoCoCache | None = None,
) -> Any:
    """Return a replicated robot, optionally from the process cache."""
    if not reuse:
        return factory()
    active = process_replicated_robot_cache() if cache is None else cache
    return active.get_or_create(key, factory)
