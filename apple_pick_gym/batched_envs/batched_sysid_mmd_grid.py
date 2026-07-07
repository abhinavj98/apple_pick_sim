"""Bend-stiffness grid and recorded-action tensor helpers for batched sys-ID MMD."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any, NamedTuple, Protocol

from apple_pick_gym.batched_envs.batched_sysid_collect import broadcast_structure_params
from apple_pick_sim.system_id.batched_digital_twin_init import initialize_batched_env_from_dataset

import numpy as np
import torch

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.batched_digital_twin_init import infer_base_params_for_structure
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
from apple_pick_sim.system_id.mmd_features import (
    ReplayObservationCollector,
    replay_obs_dict_from_sysid_numpy,
)

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")


class _ReplayCollectorLike(Protocol):
    @property
    def n_rows(self) -> int: ...

    def to_arrays(self) -> dict[str, Any]: ...


class _FrozenReplayCollector:
    """Replay collector backed by pre-built arrays (used after merge)."""

    def __init__(self, arrays: dict[str, Any]) -> None:
        self._arrays = arrays

    @property
    def n_rows(self) -> int:
        return int(self._arrays["action"].shape[0])

    def to_arrays(self) -> dict[str, Any]:
        return self._arrays


def _concat_replay_arrays(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    junction_names = list(left["junction_names"])
    if junction_names != list(right["junction_names"]):
        raise ValueError("junction_names mismatch between replay collectors")

    def _cat_1d_or_2d(key: str) -> np.ndarray:
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        if a.shape[0] == 0:
            return b
        if b.shape[0] == 0:
            return a
        return np.concatenate([a, b], axis=0)

    return {
        "action": _cat_1d_or_2d("action").astype(np.float32, copy=False),
        "ft_wrist": _cat_1d_or_2d("ft_wrist").astype(np.float32, copy=False),
        "tcp_velocity": _cat_1d_or_2d("tcp_velocity").astype(np.float32, copy=False),
        "tcp_pos": _cat_1d_or_2d("tcp_pos").astype(np.float32, copy=False),
        "apple_pos": _cat_1d_or_2d("apple_pos").astype(np.float32, copy=False),
        "phase": _cat_1d_or_2d("phase").astype(np.int8, copy=False),
        "dir_idx": _cat_1d_or_2d("dir_idx").astype(np.int32, copy=False),
        "excitation_type": _cat_1d_or_2d("excitation_type").astype(np.int8, copy=False),
        "excitation_direction": _cat_1d_or_2d("excitation_direction").astype(
            np.float32, copy=False
        ),
        "woody_part_start_pos": {
            name: _cat_1d_or_2d_for_woody(left["woody_part_start_pos"][name], right["woody_part_start_pos"][name])
            for name in junction_names
        },
        "woody_part_end_pos": {
            name: _cat_1d_or_2d_for_woody(left["woody_part_end_pos"][name], right["woody_part_end_pos"][name])
            for name in junction_names
        },
        "junction_names": junction_names,
    }


def _cat_1d_or_2d_for_woody(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape[0] == 0:
        return b_arr.astype(np.float32, copy=False)
    if b_arr.shape[0] == 0:
        return a_arr.astype(np.float32, copy=False)
    return np.concatenate([a_arr, b_arr], axis=0).astype(np.float32, copy=False)


class BatchedSysIdReplayCollectors:
    """One ReplayObservationCollector per parallel env for replay feature accumulation."""

    def __init__(
        self,
        num_envs: int,
        recorded_by_env: Sequence[Mapping[str, Any]],
    ) -> None:
        recorded = list(recorded_by_env)
        n = int(num_envs)
        if len(recorded) != n:
            raise ValueError(
                f"recorded_by_env length ({len(recorded)}) must match num_envs ({n})"
            )
        self._recorded_by_env = recorded
        self._collectors: list[_ReplayCollectorLike] = [
            ReplayObservationCollector(item) for item in recorded
        ]

    def record_step(self, env: Any, *, env_idx: int, frame_idx: int) -> None:
        idx = int(env_idx)
        recorded = self._recorded_by_env[idx]
        obs = env.sysid_numpy_obs(idx)
        adapted = replay_obs_dict_from_sysid_numpy(
            obs,
            junction_names=list(recorded["junction_names"]),
        )
        collector = self._collectors[idx]
        if not isinstance(collector, ReplayObservationCollector):
            raise RuntimeError("cannot record_step on merged replay collectors")
        collector.record(adapted, frame_idx=int(frame_idx))

    def to_arrays(self, env_idx: int) -> dict[str, Any]:
        return self._collectors[int(env_idx)].to_arrays()

    def n_rows(self, env_idx: int) -> int:
        return int(self._collectors[int(env_idx)].n_rows)

    def merge(self, other: BatchedSysIdReplayCollectors) -> BatchedSysIdReplayCollectors:
        """Concatenate rows per env_idx for chunking across replay batches."""
        if len(self._collectors) != len(other._collectors):
            raise ValueError("cannot merge replay collectors with different num_envs")
        merged = BatchedSysIdReplayCollectors.__new__(BatchedSysIdReplayCollectors)
        merged._recorded_by_env = list(self._recorded_by_env)
        merged._collectors = []
        for left, right in zip(self._collectors, other._collectors, strict=True):
            arrays = _concat_replay_arrays(left.to_arrays(), right.to_arrays())
            merged._collectors.append(_FrozenReplayCollector(arrays))
        return merged


def recorded_metadata_by_env(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
) -> list[dict[str, Any]]:
    """Load recorded obs arrays per env_idx for ReplayObservationCollector init."""
    num_envs = int(num_candidates) * int(num_directions)
    out: list[dict[str, Any]] = []
    for env_idx in range(num_envs):
        direction_idx = int(env_idx) % int(num_directions)
        recorded = dict(dataset.load_episode_obs_arrays(int(structure_idx), direction_idx))
        n_frames = int(np.asarray(recorded["action"]).shape[0])
        if "dir_idx" not in recorded:
            recorded["dir_idx"] = np.full(n_frames, direction_idx, dtype=np.int32)
        out.append(recorded)
    return out


class BendStiffnessCandidate(NamedTuple):
    """One grid point for segment bend stiffnesses."""

    primary: float
    secondary: float
    spur: float
    stem: float

    def to_overrides(self) -> dict[str, dict[str, float]]:
        return {
            "primary": {"bend_stiffness": float(self.primary)},
            "secondary": {"bend_stiffness": float(self.secondary)},
            "spur": {"bend_stiffness": float(self.spur)},
            "stem": {"bend_stiffness": float(self.stem)},
        }

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        out = base
        for segment, value in (
            ("primary", self.primary),
            ("secondary", self.secondary),
            ("spur", self.spur),
            ("stem", self.stem),
        ):
            if getattr(base, segment) is not None:
                out = fs.set_rod_bend_stiffness(out, segment, float(value))
        return out


def iter_bend_stiffness_candidates(
    *,
    primary_values: tuple[float, ...],
    secondary_values: tuple[float, ...],
    spur_values: tuple[float, ...],
    stem_values: tuple[float, ...],
):
    """Yield bend-stiffness grid candidates in Cartesian product order."""
    for primary, secondary, spur, stem in product(
        primary_values,
        secondary_values,
        spur_values,
        stem_values,
    ):
        yield BendStiffnessCandidate(
            primary=float(primary),
            secondary=float(secondary),
            spur=float(spur),
            stem=float(stem),
        )


def gt_bend_stiffness_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> BendStiffnessCandidate:
    """Build the GT bend-stiffness candidate from inferred structure params."""
    params = infer_base_params_for_structure(dataset, structure_idx)
    values: dict[str, float] = {}
    for segment in ROD_SEGMENTS:
        rod = getattr(params, segment)
        if rod is None:
            raise ValueError(f"Segment {segment!r} is missing in inferred params")
        values[segment] = float(rod.bend_stiffness)
    return BendStiffnessCandidate(
        primary=values["primary"],
        secondary=values["secondary"],
        spur=values["spur"],
        stem=values["stem"],
    )


def ensure_gt_candidate_in_grid(
    candidates: list[BendStiffnessCandidate],
    gt: BendStiffnessCandidate,
) -> list[BendStiffnessCandidate]:
    """Ensure ``gt`` appears in the candidate list, replacing the last entry if needed."""
    for candidate in candidates:
        if (
            candidate.primary == gt.primary
            and candidate.secondary == gt.secondary
            and candidate.spur == gt.spur
            and candidate.stem == gt.stem
        ):
            return list(candidates)
    if not candidates:
        return [gt]
    updated = list(candidates)
    updated[-1] = gt
    return updated


def build_recorded_actions_tensor(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
) -> np.ndarray:
    """Stack recorded EE actions for all candidate/direction env slots."""
    direction_actions: list[np.ndarray] = []
    n_frames: int | None = None
    for direction_idx in range(num_directions):
        arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
        action = np.asarray(arrays["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 6:
            raise ValueError(
                f"expected action shape (n_frames, 6), got {action.shape!r}"
            )
        if n_frames is None:
            n_frames = int(action.shape[0])
        elif int(action.shape[0]) != n_frames:
            raise ValueError("all direction episodes must have same n_frames")
        direction_actions.append(action)

    if n_frames is None:
        raise ValueError("num_directions must be positive")

    num_envs = int(num_candidates) * int(num_directions)
    out = np.empty((num_envs, n_frames, 6), dtype=np.float32)
    for candidate_idx in range(num_candidates):
        for direction_idx in range(num_directions):
            env_idx = candidate_idx * num_directions + direction_idx
            out[env_idx] = direction_actions[direction_idx]
    return out


def actions_tensor_from_recorded_frame(
    recorded_actions: np.ndarray,
    *,
    frame_idx: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return one recorded action frame for every env on ``device``."""
    frame = np.asarray(recorded_actions[:, frame_idx, :], dtype=np.float32)
    return torch.as_tensor(frame, device=device, dtype=torch.float32)


def replay_batched_sysid_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[BendStiffnessCandidate],
    num_directions: int,
    seed: int,
    build_env_fn: Callable[..., Any],
) -> BatchedSysIdReplayCollectors:
    """Replay recorded actions for one structure across bend-stiffness candidates."""
    num_candidates = len(candidates)
    if num_candidates < 1:
        raise ValueError("candidates must be non-empty")
    d = int(num_directions)
    if d < 1:
        raise ValueError("num_directions must be >= 1")

    num_envs = num_candidates * d
    base_params = infer_base_params_for_structure(dataset, int(structure_idx))
    per_env_params = broadcast_structure_params(
        [c.apply_to(base_params) for c in candidates],
        d,
    )
    recorded_actions = build_recorded_actions_tensor(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=d,
        num_candidates=num_candidates,
    )
    n_frames = int(recorded_actions.shape[1])

    env = build_env_fn(
        num_envs=num_envs,
        per_env_params=per_env_params,
        max_episode_steps=n_frames,
    )
    try:
        env.reset(seed=int(seed))
        initialize_batched_env_from_dataset(
            env,
            dataset,
            structure_idx=int(structure_idx),
            num_directions=d,
        )
        recorded_by_env = recorded_metadata_by_env(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=d,
            num_candidates=num_candidates,
        )
        collectors = BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

        for frame_idx in range(n_frames):
            actions = actions_tensor_from_recorded_frame(
                recorded_actions,
                frame_idx=frame_idx,
                device=env.device,
            )
            env.step(actions)
            for env_idx in range(num_envs):
                collectors.record_step(env, env_idx=env_idx, frame_idx=frame_idx)
    finally:
        env.close()

    return collectors
