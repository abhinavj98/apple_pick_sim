"""Data preparation helpers for the sys-ID dataset dashboard."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PHASE_NAMES: dict[int, str] = {
    0: "move_out",
    1: "hold",
    2: "return",
}
PHASE_VALUES: dict[str, int] = {name: value for value, name in PHASE_NAMES.items()}


@dataclass(frozen=True)
class EndpointSeries:
    """One woody endpoint trajectory for 3D plotting."""

    name: str
    xyz: np.ndarray


@dataclass(frozen=True)
class HoldSummary:
    """Compact quasi-static hold summary for one direction/amplitude group."""

    direction: int
    amplitude_m: float
    n_frames: int
    mean_force_n: np.ndarray
    force_norm_n: float
    tcp_displacement_m: float
    stiffness_n_per_m: float | None


def _round_float(value: float) -> float:
    return float(round(float(value), 6))


def phase_names_for_values(values: np.ndarray | Sequence[int]) -> list[str]:
    """Return display phase names for stored phase integer codes."""

    return [PHASE_NAMES.get(int(value), f"unknown_{int(value)}") for value in values]


def _phase_values(phases: Sequence[str | int] | None) -> set[int] | None:
    if phases is None:
        return None
    out: set[int] = set()
    for phase in phases:
        if isinstance(phase, str):
            try:
                out.add(PHASE_VALUES[phase])
            except KeyError as exc:
                raise ValueError(f"unknown phase filter: {phase!r}") from exc
        else:
            out.add(int(phase))
    return out


def build_frame_mask(
    arrays: Mapping[str, Any],
    *,
    direction: int | None = None,
    phases: Sequence[str | int] | None = None,
) -> np.ndarray:
    """Build a boolean frame mask for optional direction and phase filters."""

    phase = np.asarray(arrays["phase"]).reshape(-1)
    mask = np.ones(phase.shape[0], dtype=bool)
    if direction is not None:
        dir_idx = np.asarray(arrays["dir_idx"]).reshape(-1)
        if dir_idx.shape != phase.shape:
            raise ValueError("dir_idx and phase must have matching frame counts")
        mask &= dir_idx == int(direction)
    phase_values = _phase_values(phases)
    if phase_values is not None:
        mask &= np.isin(phase, list(phase_values))
    return mask


def woody_endpoint_series(
    arrays: Mapping[str, Any],
    mask: np.ndarray | None = None,
) -> list[EndpointSeries]:
    """Return woody start/end endpoint trajectories in metadata junction order."""

    junction_names = [str(name) for name in arrays.get("junction_names", [])]
    if mask is None:
        n_frames = int(np.asarray(arrays["phase"]).reshape(-1).shape[0])
        mask = np.ones(n_frames, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)

    start_by_name = arrays["woody_part_start_pos"]
    end_by_name = arrays.get("woody_part_end_pos") or {}
    out: list[EndpointSeries] = []
    for name in junction_names:
        start = np.asarray(start_by_name[name], dtype=np.float64)
        out.append(EndpointSeries(name=f"{name} start", xyz=start[mask]))
        if name in end_by_name:
            end = np.asarray(end_by_name[name], dtype=np.float64)
            out.append(EndpointSeries(name=f"{name} end", xyz=end[mask]))
    return out


def _amplitude_values(arrays: Mapping[str, Any], n_frames: int) -> np.ndarray:
    if "amplitude_m" not in arrays:
        return np.zeros(n_frames, dtype=np.float64)
    values = np.asarray(arrays["amplitude_m"], dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(n_frames, dtype=np.float64)
    if values.size != n_frames:
        raise ValueError("amplitude_m and phase must have matching frame counts")
    return values


def hold_summaries(
    arrays: Mapping[str, Any],
    *,
    initial_tcp_pos: np.ndarray | Sequence[float] | None = None,
) -> list[HoldSummary]:
    """Compute per-direction/per-amplitude summaries over hold frames."""

    phase = np.asarray(arrays["phase"]).reshape(-1)
    dir_idx = np.asarray(arrays["dir_idx"]).reshape(-1)
    if phase.shape != dir_idx.shape:
        raise ValueError("phase and dir_idx must have matching frame counts")
    n_frames = phase.shape[0]
    amplitude = _amplitude_values(arrays, n_frames)
    ft = np.asarray(arrays["ft_wrist"], dtype=np.float64).reshape(n_frames, 6)
    tcp_pos = np.asarray(arrays["tcp_pos"], dtype=np.float64).reshape(n_frames, 3)
    if initial_tcp_pos is None:
        initial = tcp_pos[0]
    else:
        initial = np.asarray(initial_tcp_pos, dtype=np.float64).reshape(3)

    summaries: list[HoldSummary] = []
    keys = sorted(
        {
            (int(direction), _round_float(float(amp)))
            for direction, amp, phase_value in zip(dir_idx, amplitude, phase, strict=True)
            if int(phase_value) == PHASE_VALUES["hold"]
        }
    )
    for direction, amp in keys:
        mask = (phase == PHASE_VALUES["hold"]) & (dir_idx == direction) & np.isclose(
            amplitude,
            amp,
        )
        if not np.any(mask):
            continue
        mean_force = np.mean(ft[mask, :3], axis=0)
        force_norm = _round_float(float(np.linalg.norm(mean_force)))
        mean_tcp_pos = np.mean(tcp_pos[mask], axis=0)
        displacement = _round_float(float(np.linalg.norm(mean_tcp_pos - initial)))
        stiffness = None if displacement <= 0.0 else _round_float(force_norm / displacement)
        summaries.append(
            HoldSummary(
                direction=direction,
                amplitude_m=amp,
                n_frames=int(np.count_nonzero(mask)),
                mean_force_n=mean_force,
                force_norm_n=force_norm,
                tcp_displacement_m=displacement,
                stiffness_n_per_m=stiffness,
            )
        )
    return summaries
