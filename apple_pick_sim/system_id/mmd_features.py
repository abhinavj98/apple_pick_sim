"""Feature construction for sys-ID MMD objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CMA_WOODY_JUNCTIONS: tuple[str, str] = ("primary_spur", "spur_stem")


def cma_woody_junctions_from_env(names: list[str]) -> list[str]:
    """Filter an env's full T-junction ``names`` down to the CMA woody subset.

    Real environments expose a larger T-junction set (e.g.
    ``primary_support_left``, ``primary_support_right``, ``stem_apple``, ...).
    CMA/MMD features only use ``CMA_WOODY_JUNCTIONS``; raise if either is missing.
    """
    have = set(names)
    missing = [n for n in CMA_WOODY_JUNCTIONS if n not in have]
    if missing:
        raise ValueError(f"env junction_names missing {missing}; got {names}")
    return list(CMA_WOODY_JUNCTIONS)

STATE_VECTOR_FIELDS: tuple[str, ...] = (
    "ft_wrist",
    "tcp_velocity",
    "tcp_pos",
    "apple_pos",
    "woody_part_start_pos",
    "woody_bending_angles",
)

_STATE_VECTOR_PREFIX_PHYS_SCALE: tuple[float, ...] = (
    # ft_wrist F
    2.0,
    2.0,
    2.0,
    # ft_wrist τ
    0.5,
    0.5,
    0.5,
    # tcp_velocity v
    0.02,
    0.02,
    0.02,
    # tcp_velocity ω
    0.02,
    0.02,
    0.02,
    # tcp_pos
    0.005,
    0.005,
    0.005,
    # apple_pos
    0.005,
    0.005,
    0.005,
)
WOODY_START_PHYS_SCALE = 0.005
BEND_ANGLE_PHYS_SCALE = 0.05


def state_vector_phys_scale(n_junctions: int) -> np.ndarray:
    """Fixed physical scales for one STATE_VECTOR row at ``n_junctions``.

    Woody XYZ and bend scales are uniform across junctions. CMA production
    uses ``n_junctions=2``; the exported ``STATE_VECTOR_PHYS_SCALE`` tuple is
    that instance.
    """
    n = int(n_junctions)
    if n < 1:
        raise ValueError(f"n_junctions must be >= 1, got {n_junctions!r}")
    prefix = np.asarray(_STATE_VECTOR_PREFIX_PHYS_SCALE, dtype=np.float64)
    woody = np.full(3 * n, WOODY_START_PHYS_SCALE, dtype=np.float64)
    bend = np.full(n, BEND_ANGLE_PHYS_SCALE, dtype=np.float64)
    return np.concatenate([prefix, woody, bend])


STATE_VECTOR_PHYS_SCALE: tuple[float, ...] = tuple(
    float(x) for x in state_vector_phys_scale(n_junctions=2)
)


def transition_feature_scale(n_features: int, *, n_junctions: int = 2) -> np.ndarray:
    """Return divisor vector for [s, Δs, trailing one-hots]."""
    state = state_vector_phys_scale(n_junctions)
    state_dim = int(state.size)
    if n_features < 2 * state_dim:
        raise ValueError(
            f"transition features width {n_features} < 2*state_dim={2 * state_dim} "
            f"(n_junctions={int(n_junctions)})"
        )
    n_extra = int(n_features) - 2 * state_dim
    return np.concatenate([state, state, np.ones(n_extra, dtype=np.float64)])


def scored_ft_wrist(arrays: Mapping[str, Any]) -> Any:
    """Prefer convert-time ``ft_wrist_lpf`` when present; else live ``ft_wrist``."""
    lpf = arrays.get("ft_wrist_lpf")
    if lpf is None:
        return arrays["ft_wrist"]
    arr = np.asarray(lpf)
    if arr.size == 0:
        return arrays["ft_wrist"]
    return lpf


def n_junctions_from_episodes(episodes: Sequence[Mapping[str, Any]]) -> int:
    """Return the shared woody-junction count from recorded/replay bags."""
    if not episodes:
        raise ValueError("need at least one episode to resolve n_junctions")
    counts: list[int] = []
    for episode in episodes:
        names = episode.get("junction_names")
        if names is None:
            raise KeyError("episode missing junction_names")
        counts.append(len(names))
    if len(set(counts)) != 1:
        raise ValueError(f"mixed junction counts across episodes: {counts}")
    n = int(counts[0])
    if n < 1:
        raise ValueError(f"n_junctions must be >= 1, got {n}")
    return n

REQUIRED_ARRAY_KEYS: tuple[str, ...] = (
    "ft_wrist",
    "tcp_velocity",
    "action",
    "tcp_pos",
    "apple_pos",
    "woody_part_start_pos",
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
        recorded_action = np.asarray(recorded["action"], dtype=np.float32)
        if recorded_action.ndim == 2:
            self._action_dim = int(recorded_action.shape[1])
        else:
            self._action_dim = 6
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
            "stable": [],
            "hold_number": [],
        }
        self._woody_start: dict[str, list[np.ndarray]] = {
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

    def record(self, obs: Mapping[str, Any], *, frame_idx: int, stable: bool = True) -> None:
        """Append one live replay observation aligned to one recorded frame."""

        n_frames = int(np.asarray(self._recorded["action"]).shape[0])
        if frame_idx < 0 or frame_idx >= n_frames:
            raise IndexError(
                f"frame_idx={frame_idx} out of range for recorded episode "
                f"with {n_frames} frames"
            )
        for key in ("ft_wrist", "tcp_velocity", "tcp_pos", "apple_pos", "woody_start"):
            if key not in obs:
                raise KeyError(f"missing replay observation field: {key}")

        self._rows["action"].append(
            np.array(self._recorded_row("action", frame_idx), dtype=np.float32, copy=True).reshape(
                self._action_dim
            )
        )
        self._rows["ft_wrist"].append(
            np.array(obs["ft_wrist"], dtype=np.float32, copy=True).reshape(6)
        )
        self._rows["tcp_velocity"].append(
            np.array(obs["tcp_velocity"], dtype=np.float32, copy=True).reshape(6)
        )
        self._rows["tcp_pos"].append(
            np.array(obs["tcp_pos"], dtype=np.float32, copy=True).reshape(3)
        )
        self._rows["apple_pos"].append(
            np.array(obs["apple_pos"], dtype=np.float32, copy=True).reshape(3)
        )
        self._rows["phase"].append(int(self._recorded_row("phase", frame_idx)))
        self._rows["dir_idx"].append(int(self._recorded_row("dir_idx", frame_idx)))
        if "hold_number" in self._recorded:
            self._rows["hold_number"].append(
                int(self._recorded_row("hold_number", frame_idx))
            )
        else:
            self._rows["hold_number"].append(-1)
        self._rows["excitation_type"].append(
            int(self._recorded_row("excitation_type", frame_idx))
        )
        self._rows["excitation_direction"].append(
            np.array(
                self._recorded_row("excitation_direction", frame_idx),
                dtype=np.float32,
                copy=True,
            ).reshape(3)
        )

        for name, pos in self._split_flat_woody(
            obs["woody_start"], key="woody_start"
        ).items():
            self._woody_start[name].append(np.array(pos, dtype=np.float32, copy=True))
        self._rows["stable"].append(bool(stable))

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
            "hold_number": np.asarray(self._rows["hold_number"], dtype=np.int32),
            "excitation_type": np.asarray(self._rows["excitation_type"], dtype=np.int8),
            "excitation_direction": np.stack(
                self._rows["excitation_direction"], axis=0
            ).astype(np.float32),
            "stable": np.asarray(self._rows["stable"], dtype=bool),
            "woody_part_start_pos": {
                name: np.stack(rows, axis=0).astype(np.float32)
                for name, rows in self._woody_start.items()
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


def replay_obs_dict_from_sysid_numpy(
    sysid_obs: Mapping[str, Any],
    *,
    junction_names: list[str],
) -> dict[str, Any]:
    """Adapt batched sysid_numpy obs into ReplayObservationCollector.record format."""

    return {
        "ft_wrist": np.asarray(sysid_obs["ft_wrist"], dtype=np.float32).reshape(6),
        "tcp_velocity": np.asarray(sysid_obs["tcp_velocity"], dtype=np.float32).reshape(6),
        "tcp_pos": np.asarray(sysid_obs["tcp_pos"], dtype=np.float32).reshape(3),
        "apple_pos": np.asarray(sysid_obs["apple_pos"], dtype=np.float32).reshape(3),
        "woody_start": flatten_woody_positions(
            sysid_obs["woody_part_start_pos"],
            frame_idx=0,
            junction_names=junction_names,
        ),
    }


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


def _tiled_positions(values: Any, *, n_frames: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.tile(arr, (n_frames, 1))
    return arr


def _bending_chords(
    arrays: Mapping[str, Any],
    *,
    n_frames: int,
    junction_names: list[str],
) -> list[np.ndarray]:
    """Per-junction chord vectors (n_frames, 3) used for bending deflection.

    ``CMA_WOODY_JUNCTIONS`` (``primary_spur``, ``spur_stem``) uses the real
    Branch/Spur/Apple-aligned chords: ``primary_spur`` chords
    ``start[spur_stem] - start[primary_spur]``; ``spur_stem`` chords
    ``apple_pos - start[spur_stem]``. Other junction orderings fall back to the
    distal rule: chord ``i`` is ``start[i+1] - start[i]`` and the last chord is
    ``apple_pos - start[last]``.
    """
    starts_by_name = arrays["woody_part_start_pos"]
    apple_pos = _tiled_positions(arrays["apple_pos"], n_frames=n_frames)

    if list(junction_names) == list(CMA_WOODY_JUNCTIONS):
        primary_spur = _tiled_positions(
            starts_by_name["primary_spur"], n_frames=n_frames
        )
        spur_stem = _tiled_positions(starts_by_name["spur_stem"], n_frames=n_frames)
        return [spur_stem - primary_spur, apple_pos - spur_stem]

    starts = [
        _tiled_positions(starts_by_name[name], n_frames=n_frames)
        for name in junction_names
    ]
    n_junctions = len(junction_names)
    return [
        starts[i + 1] - starts[i] if i < n_junctions - 1 else apple_pos - starts[i]
        for i in range(n_junctions)
    ]


def build_bending_angles(
    arrays: Mapping[str, Any],
    *,
    n_frames: int,
    junction_names: list[str],
) -> np.ndarray:
    """Compute local bending deflection angles (in radians) from rest pose (frame 0)."""
    n_junctions = len(junction_names)
    if n_junctions == 0:
        return np.zeros((n_frames, 0), dtype=np.float32)

    chords = _bending_chords(arrays, n_frames=n_frames, junction_names=junction_names)

    angles = np.zeros((n_frames, n_junctions), dtype=np.float64)
    for j_idx, vectors in enumerate(chords):
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)

        lengths_nonzero = np.where(lengths == 0.0, 1.0, lengths)
        dirs = vectors / lengths_nonzero  # (n_frames, 3)

        dir_rest = dirs[0]  # (3,)
        dot_products = np.sum(dirs * dir_rest, axis=1)  # (n_frames,)
        dot_products = np.clip(dot_products, -1.0, 1.0)

        angle_deflection = np.arccos(dot_products)
        angles[:, j_idx] = np.where(lengths.reshape(-1) == 0.0, 0.0, angle_deflection)

    # Force frame 0 deflection to be exactly 0.0
    angles[0, :] = 0.0

    return angles.astype(np.float32, copy=False)


def build_state_matrix(arrays: Mapping[str, Any]) -> np.ndarray:
    """Build per-frame observable state rows in the MMD feature order."""

    _require_keys(arrays, REQUIRED_ARRAY_KEYS)
    action = np.asarray(arrays["action"], dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"action must be a 2D array, got shape {action.shape}")
    n_frames = int(action.shape[0])
    junction_names = [str(name) for name in arrays["junction_names"]]

    columns = [
        _as_2d(scored_ft_wrist(arrays), name="ft_wrist", n_frames=n_frames),
        _as_2d(arrays["tcp_velocity"], name="tcp_velocity", n_frames=n_frames),
        _as_2d(arrays["tcp_pos"], name="tcp_pos", n_frames=n_frames),
        _as_2d(arrays["apple_pos"], name="apple_pos", n_frames=n_frames),
        _stack_woody(
            arrays["woody_part_start_pos"],
            n_frames=n_frames,
            junction_names=junction_names,
        ),
        build_bending_angles(
            arrays,
            n_frames=n_frames,
            junction_names=junction_names,
        ),
    ]
    return np.concatenate(columns, axis=1).astype(np.float32, copy=False)


def iter_kept_hold_segments(
    *,
    phase: np.ndarray,
    dir_idx: np.ndarray,
    direction: int,
    stable: np.ndarray | None = None,
    min_frames: int = 1,
) -> list[np.ndarray]:
    """Return full contiguous hold index arrays for one direction (no latter-half burn-in).

    Segmentation uses ``phase == 1`` and matching ``dir_idx`` only. ``stable`` is
    accepted for API compatibility but does **not** split segments (apply it as an
    in-hold sample mask at aggregation time instead).
    """

    phase = np.asarray(phase).reshape(-1)
    dir_idx = np.asarray(dir_idx).reshape(-1)
    if phase.shape != dir_idx.shape:
        raise ValueError(
            f"phase and dir_idx must have matching shape, got {phase.shape} and {dir_idx.shape}"
        )
    if stable is not None:
        # Validate shape for callers that still pass stable; ignore for boundaries.
        stable_arr = np.asarray(stable, dtype=bool).reshape(-1)
        if stable_arr.shape != phase.shape:
            raise ValueError(
                f"stable and phase must have matching shape, got {stable_arr.shape} and {phase.shape}"
            )

    kept: list[np.ndarray] = []
    current: list[int] = []
    min_frames = max(1, int(min_frames))

    def _flush() -> None:
        nonlocal current
        if not current:
            return
        idxs = np.asarray(current, dtype=np.int32)
        if idxs.size >= min_frames:
            kept.append(idxs)
        current = []

    for frame_idx, (phase_value, dir_value) in enumerate(zip(phase, dir_idx, strict=True)):
        is_hold = int(phase_value) == 1 and int(dir_value) == int(direction)
        if is_hold:
            current.append(frame_idx)
            continue
        _flush()
    _flush()
    return kept


def _stable_masked_segment(
    segment: np.ndarray,
    stable: np.ndarray | None,
) -> np.ndarray:
    """Keep segment frame indices that are stable (or all indices if no mask)."""
    idxs = np.asarray(segment, dtype=np.int64).reshape(-1)
    if stable is None or idxs.size == 0:
        return idxs
    mask = np.asarray(stable, dtype=bool).reshape(-1)
    return idxs[mask[idxs]]


def _one_hot_hold_id(hold_idx: int, *, n_holds: int) -> np.ndarray:
    n = int(n_holds)
    if n <= 0:
        raise ValueError(f"n_holds must be positive, got {n_holds!r}")
    i = int(hold_idx)
    if i < 0 or i >= n:
        raise ValueError(f"hold_idx {i} out of range for n_holds={n}")
    vec = np.zeros(n, dtype=np.float32)
    vec[i] = 1.0
    return vec


def _one_hot_dir_id(dir_idx: int, *, n_directions: int) -> np.ndarray:
    n = int(n_directions)
    if n <= 0:
        raise ValueError(f"n_directions must be positive, got {n_directions!r}")
    i = int(dir_idx)
    if i < 0 or i >= n:
        raise ValueError(f"dir_idx {i} out of range for n_directions={n}")
    vec = np.zeros(n, dtype=np.float32)
    vec[i] = 1.0
    return vec


def combine_transition_features(
    episodes: list[Mapping[str, Any]],
    *,
    use_median: bool = False,
    hold_id_onehot: bool = False,
    n_holds: int | None = None,
    dir_id_onehot: bool = False,
    n_directions: int | None = None,
) -> dict[int, np.ndarray]:
    """Concatenate hold-only transition features keyed by excitation direction."""
    parts: dict[int, list[np.ndarray]] = {}
    for arrays in episodes:
        for direction, features in build_transition_features_by_direction(
            arrays,
            use_median=use_median,
            hold_id_onehot=hold_id_onehot,
            n_holds=n_holds,
            dir_id_onehot=dir_id_onehot,
            n_directions=n_directions,
        ).items():
            parts.setdefault(direction, []).append(features)
    return {
        direction: np.concatenate(chunks, axis=0)
        for direction, chunks in sorted(parts.items())
        if chunks
    }


def build_transition_features_by_direction(
    arrays: Mapping[str, Any],
    *,
    use_median: bool = False,
    hold_id_onehot: bool = False,
    n_holds: int | None = None,
    dir_id_onehot: bool = False,
    n_directions: int | None = None,
) -> dict[int, np.ndarray]:
    """Build hold-only transition feature rows keyed by excitation direction.

    When ``use_median`` is True, emit one row per consecutive hold pair using
    full-hold median states: ``[s_i, s_{i+1}-s_i]`` (optionally + hold-id /
    dir-id one-hot). When False, emit frame→frame transitions on full hold
    segments.
    """

    _require_keys(arrays, REQUIRED_ARRAY_KEYS)
    state = build_state_matrix(arrays)
    phase = np.asarray(arrays["phase"]).reshape(-1)
    dir_idx = np.asarray(arrays["dir_idx"]).reshape(-1)
    stable = np.asarray(arrays.get("stable", np.ones(phase.shape[0], dtype=bool)), dtype=bool).reshape(
        -1
    )
    if state.shape[0] != phase.size or state.shape[0] != dir_idx.size:
        raise ValueError("state, phase, and dir_idx frame counts must match")

    resolved_n_holds = n_holds
    if hold_id_onehot:
        if resolved_n_holds is None:
            # Prefer recorded column max+1, else segment count across dirs later.
            if "hold_number" in arrays:
                hn = np.asarray(arrays["hold_number"], dtype=np.int32).reshape(-1)
                positive = hn[hn >= 0]
                resolved_n_holds = int(positive.max()) + 1 if positive.size else 1
            else:
                resolved_n_holds = None  # set per-direction below
        elif int(resolved_n_holds) <= 0:
            raise ValueError(f"n_holds must be positive, got {n_holds!r}")

    resolved_n_directions = n_directions
    if dir_id_onehot:
        if resolved_n_directions is None:
            resolved_n_directions = int(dir_idx.max()) + 1 if dir_idx.size else 1
        elif int(resolved_n_directions) <= 0:
            raise ValueError(f"n_directions must be positive, got {n_directions!r}")
        resolved_n_directions = int(resolved_n_directions)

    out: dict[int, np.ndarray] = {}
    for direction in sorted({int(value) for value in dir_idx.tolist()}):
        frame_indices = np.where(dir_idx == direction)[0]
        if len(frame_indices) == 0:
            continue

        segments = iter_kept_hold_segments(
            phase=phase,
            dir_idx=dir_idx,
            direction=direction,
            min_frames=1,
        )
        rows: list[np.ndarray] = []
        if use_median:
            medians: list[np.ndarray] = []
            hold_ids: list[int] = []
            for hold_i, segment in enumerate(segments):
                kept = _stable_masked_segment(segment, stable)
                if kept.size < 1:
                    continue
                medians.append(np.median(state[kept], axis=0).astype(np.float32))
                if "hold_number" in arrays:
                    hn = int(np.asarray(arrays["hold_number"])[int(kept[0])])
                    hold_ids.append(hn if hn >= 0 else hold_i)
                else:
                    hold_ids.append(hold_i)
            n_holds_dir = (
                int(resolved_n_holds)
                if resolved_n_holds is not None
                else max(len(medians), 1)
            )
            for i in range(len(medians) - 1):
                current = medians[i]
                delta = medians[i + 1] - current
                row = np.concatenate([current, delta]).astype(np.float32)
                if hold_id_onehot:
                    row = np.concatenate(
                        [row, _one_hot_hold_id(hold_ids[i], n_holds=n_holds_dir)]
                    )
                if dir_id_onehot:
                    assert resolved_n_directions is not None
                    row = np.concatenate(
                        [
                            row,
                            _one_hot_dir_id(
                                direction, n_directions=resolved_n_directions
                            ),
                        ]
                    )
                rows.append(row)
        else:
            n_holds_dir = (
                int(resolved_n_holds)
                if resolved_n_holds is not None
                else max(len(segments), 1)
            )
            for hold_i, segment in enumerate(segments):
                kept = _stable_masked_segment(segment, stable)
                if kept.size < 2:
                    continue
                for start_idx, end_idx in zip(kept[:-1], kept[1:], strict=True):
                    current = state[int(start_idx)]
                    delta = state[int(end_idx)] - current
                    row = np.concatenate([current, delta]).astype(np.float32)
                    if hold_id_onehot:
                        if "hold_number" in arrays:
                            hn = int(np.asarray(arrays["hold_number"])[int(start_idx)])
                            hid = hn if hn >= 0 else hold_i
                        else:
                            hid = hold_i
                        row = np.concatenate(
                            [row, _one_hot_hold_id(hid, n_holds=n_holds_dir)]
                        )
                    if dir_id_onehot:
                        assert resolved_n_directions is not None
                        row = np.concatenate(
                            [
                                row,
                                _one_hot_dir_id(
                                    direction, n_directions=resolved_n_directions
                                ),
                            ]
                        )
                    rows.append(row)
        if rows:
            arr = np.stack(rows, axis=0).astype(np.float32, copy=False)
            if direction in out:
                out[direction] = np.concatenate([out[direction], arr], axis=0)
            else:
                out[direction] = arr
    return out
