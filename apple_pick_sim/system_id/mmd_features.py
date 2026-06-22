"""Feature construction for sys-ID MMD objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

STATE_VECTOR_FIELDS: tuple[str, ...] = (
    "ft_wrist",
    "tcp_velocity",
    "action",
    "tcp_pos",
    "apple_pos",
    "woody_part_start_pos",
    "woody_part_end_pos",
    "excitation_direction",
    "phase",
    "excitation_type",
)

REQUIRED_ARRAY_KEYS: tuple[str, ...] = (
    "ft_wrist",
    "tcp_velocity",
    "action",
    "tcp_pos",
    "apple_pos",
    "woody_part_start_pos",
    "woody_part_end_pos",
    "excitation_direction",
    "phase",
    "excitation_type",
    "dir_idx",
    "junction_names",
)


class ReplayObservationCollector:
    """Collect live replay observations into the dataset array contract."""

    def __init__(self, recorded: Mapping[str, Any]) -> None:
        _require_keys(
            recorded,
            (
                "action",
                "phase",
                "dir_idx",
                "excitation_type",
                "excitation_direction",
                "junction_names",
            ),
        )
        self._recorded = recorded
        self._junction_names = [str(name) for name in recorded["junction_names"]]
        self._rows: dict[str, list[np.ndarray | int]] = {
            "action": [],
            "ft_wrist": [],
            "tcp_velocity": [],
            "tcp_pos": [],
            "apple_pos": [],
            "phase": [],
            "dir_idx": [],
            "excitation_type": [],
            "excitation_direction": [],
        }
        self._woody_start: dict[str, list[np.ndarray]] = {
            name: [] for name in self._junction_names
        }
        self._woody_end: dict[str, list[np.ndarray]] = {
            name: [] for name in self._junction_names
        }

    @property
    def n_rows(self) -> int:
        return len(self._rows["action"])

    def _recorded_row(self, key: str, frame_idx: int) -> np.ndarray | int:
        values = self._recorded[key]
        return values[frame_idx]

    def _split_flat_woody(self, values: Any, *, key: str) -> dict[str, np.ndarray]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        expected = 3 * len(self._junction_names)
        if arr.size != expected:
            raise ValueError(f"{key} has {arr.size} values, expected {expected}")
        return {
            name: arr[i * 3 : (i + 1) * 3].astype(np.float32, copy=False)
            for i, name in enumerate(self._junction_names)
        }

    def record(self, obs: Mapping[str, Any], *, frame_idx: int) -> None:
        """Append one live replay observation aligned to one recorded frame."""

        n_frames = int(np.asarray(self._recorded["action"]).shape[0])
        if frame_idx < 0 or frame_idx >= n_frames:
            return
        for key in ("ft_wrist", "tcp_velocity", "tcp_pos", "apple_pos", "woody_start", "woody_end"):
            if key not in obs:
                raise KeyError(f"missing replay observation field: {key}")

        self._rows["action"].append(
            np.asarray(self._recorded_row("action", frame_idx), dtype=np.float32).reshape(6)
        )
        self._rows["ft_wrist"].append(
            np.asarray(obs["ft_wrist"], dtype=np.float32).reshape(6)
        )
        self._rows["tcp_velocity"].append(
            np.asarray(obs["tcp_velocity"], dtype=np.float32).reshape(6)
        )
        self._rows["tcp_pos"].append(
            np.asarray(obs["tcp_pos"], dtype=np.float32).reshape(3)
        )
        self._rows["apple_pos"].append(
            np.asarray(obs["apple_pos"], dtype=np.float32).reshape(3)
        )
        self._rows["phase"].append(int(self._recorded_row("phase", frame_idx)))
        self._rows["dir_idx"].append(int(self._recorded_row("dir_idx", frame_idx)))
        self._rows["excitation_type"].append(
            int(self._recorded_row("excitation_type", frame_idx))
        )
        self._rows["excitation_direction"].append(
            np.asarray(
                self._recorded_row("excitation_direction", frame_idx),
                dtype=np.float32,
            ).reshape(3)
        )

        for name, pos in self._split_flat_woody(
            obs["woody_start"], key="woody_start"
        ).items():
            self._woody_start[name].append(pos)
        for name, pos in self._split_flat_woody(obs["woody_end"], key="woody_end").items():
            self._woody_end[name].append(pos)

    def to_arrays(self) -> dict[str, Any]:
        """Return collected observations as arrays compatible with feature builders."""

        return {
            "action": np.stack(self._rows["action"], axis=0).astype(np.float32),
            "ft_wrist": np.stack(self._rows["ft_wrist"], axis=0).astype(np.float32),
            "tcp_velocity": np.stack(self._rows["tcp_velocity"], axis=0).astype(np.float32),
            "tcp_pos": np.stack(self._rows["tcp_pos"], axis=0).astype(np.float32),
            "apple_pos": np.stack(self._rows["apple_pos"], axis=0).astype(np.float32),
            "phase": np.asarray(self._rows["phase"], dtype=np.int8),
            "dir_idx": np.asarray(self._rows["dir_idx"], dtype=np.int32),
            "excitation_type": np.asarray(self._rows["excitation_type"], dtype=np.int8),
            "excitation_direction": np.stack(
                self._rows["excitation_direction"], axis=0
            ).astype(np.float32),
            "woody_part_start_pos": {
                name: np.stack(rows, axis=0).astype(np.float32)
                for name, rows in self._woody_start.items()
            },
            "woody_part_end_pos": {
                name: np.stack(rows, axis=0).astype(np.float32)
                for name, rows in self._woody_end.items()
            },
            "junction_names": list(self._junction_names),
        }


def _require_keys(arrays: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise KeyError(f"missing MMD feature field(s): {', '.join(missing)}")


def _as_2d(values: Any, *, name: str, n_frames: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape}")
    if arr.shape[0] != n_frames:
        raise ValueError(
            f"{name} has {arr.shape[0]} frames, expected {n_frames}"
        )
    return arr


def flatten_woody_positions(
    woody_by_junction: Mapping[str, np.ndarray],
    *,
    frame_idx: int,
    junction_names: list[str],
) -> np.ndarray:
    """Flatten one woody endpoint dict in metadata junction order."""

    parts: list[np.ndarray] = []
    for name in junction_names:
        if name not in woody_by_junction:
            raise KeyError(f"missing woody endpoint for junction {name!r}")
        values = np.asarray(woody_by_junction[name], dtype=np.float32)
        if values.ndim == 1:
            pos = values.reshape(3)
        else:
            pos = values[frame_idx].reshape(3)
        parts.append(pos)
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def _stack_woody(
    woody_by_junction: Mapping[str, np.ndarray],
    *,
    n_frames: int,
    junction_names: list[str],
) -> np.ndarray:
    rows = [
        flatten_woody_positions(
            woody_by_junction,
            frame_idx=frame_idx,
            junction_names=junction_names,
        )
        for frame_idx in range(n_frames)
    ]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def build_state_matrix(arrays: Mapping[str, Any]) -> np.ndarray:
    """Build per-frame observable state rows in the MMD feature order."""

    _require_keys(arrays, REQUIRED_ARRAY_KEYS)
    action = np.asarray(arrays["action"], dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"action must be a 2D array, got shape {action.shape}")
    n_frames = int(action.shape[0])
    junction_names = [str(name) for name in arrays["junction_names"]]

    columns = [
        _as_2d(arrays["ft_wrist"], name="ft_wrist", n_frames=n_frames),
        _as_2d(arrays["tcp_velocity"], name="tcp_velocity", n_frames=n_frames),
        _as_2d(arrays["action"], name="action", n_frames=n_frames),
        _as_2d(arrays["tcp_pos"], name="tcp_pos", n_frames=n_frames),
        _as_2d(arrays["apple_pos"], name="apple_pos", n_frames=n_frames),
        _stack_woody(
            arrays["woody_part_start_pos"],
            n_frames=n_frames,
            junction_names=junction_names,
        ),
        _stack_woody(
            arrays["woody_part_end_pos"],
            n_frames=n_frames,
            junction_names=junction_names,
        ),
        _as_2d(
            arrays["excitation_direction"],
            name="excitation_direction",
            n_frames=n_frames,
        ),
        _as_2d(arrays["phase"], name="phase", n_frames=n_frames),
        _as_2d(
            arrays["excitation_type"],
            name="excitation_type",
            n_frames=n_frames,
        ),
    ]
    return np.concatenate(columns, axis=1).astype(np.float32, copy=False)


def iter_kept_hold_segments(
    *,
    phase: np.ndarray,
    dir_idx: np.ndarray,
    direction: int,
) -> list[np.ndarray]:
    """Return latter-half index arrays for contiguous hold segments in one direction."""

    phase = np.asarray(phase).reshape(-1)
    dir_idx = np.asarray(dir_idx).reshape(-1)
    if phase.shape != dir_idx.shape:
        raise ValueError(
            f"phase and dir_idx must have matching shape, got {phase.shape} and {dir_idx.shape}"
        )

    kept: list[np.ndarray] = []
    current: list[int] = []
    for frame_idx, (phase_value, dir_value) in enumerate(zip(phase, dir_idx, strict=True)):
        is_hold = int(phase_value) == 1 and int(dir_value) == int(direction)
        if is_hold:
            current.append(frame_idx)
            continue
        if current:
            drop = int(math.ceil(len(current) / 2.0))
            tail = np.asarray(current[drop:], dtype=np.int32)
            if tail.size >= 2:
                kept.append(tail)
            current = []
    if current:
        drop = int(math.ceil(len(current) / 2.0))
        tail = np.asarray(current[drop:], dtype=np.int32)
        if tail.size >= 2:
            kept.append(tail)
    return kept


def build_transition_features_by_direction(
    arrays: Mapping[str, Any],
) -> dict[int, np.ndarray]:
    """Build hold-only transition feature rows keyed by direction index."""

    _require_keys(arrays, REQUIRED_ARRAY_KEYS)
    state = build_state_matrix(arrays)
    phase = np.asarray(arrays["phase"]).reshape(-1)
    dir_idx = np.asarray(arrays["dir_idx"]).reshape(-1)
    if state.shape[0] != phase.size or state.shape[0] != dir_idx.size:
        raise ValueError("state, phase, and dir_idx frame counts must match")

    out: dict[int, np.ndarray] = {}
    for direction in sorted({int(value) for value in dir_idx.tolist()}):
        rows: list[np.ndarray] = []
        for segment in iter_kept_hold_segments(
            phase=phase,
            dir_idx=dir_idx,
            direction=direction,
        ):
            for start_idx, end_idx in zip(segment[:-1], segment[1:], strict=True):
                current = state[int(start_idx)]
                delta = state[int(end_idx)] - current
                rows.append(np.concatenate([current, delta]).astype(np.float32))
        if rows:
            out[direction] = np.stack(rows, axis=0).astype(np.float32, copy=False)
    return out
