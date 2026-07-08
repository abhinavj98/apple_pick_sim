"""Quasi-static hold diagnostics for batched sys-ID trajectory datasets.

Design notes
------------
The fruiting system **has no damping** — oscillations at hold never decay
kinematically.  This makes instantaneous-speed thresholds meaningless.
Instead, quasi-static quality is checked via **force stability**:

* ``max_force_cv`` — coefficient of variation of force-norm in the latter-half
  hold window (std / mean).  Low CV means the system oscillates around a
  stable mean elastic force — exactly what you average for stiffness ID.
* ``max_force_mean_drift`` — |late_third_mean − early_third_mean| / mean.
  Low value means the force mean has converged between entry into hold and
  the end of hold.
* ``max_tcp_excursion_m`` — TCP moved by at most this much from its
  hold-start position (controller stiffness proxy).

Speed and drift values are still recorded for diagnostic plots but are **not**
part of the pass/fail gate.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

_HOLD_PHASE = int(PHASE_TO_INT["hold"])

# ------------------------------------------------------------------
# Threshold dataclass
# ------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StiffnessIdHoldThresholds:
    """Pass/fail gates appropriate for no-damping quasi-static stiffness ID.

    ``max_force_cv``
        Max acceptable coefficient of variation (std/mean) of force-norm in
        the metrics window.  ~0.05 means ±5 % oscillation around the mean.
    ``max_force_mean_drift_frac``
        Max acceptable relative drift of mean force between the first and
        last third of the hold window.  ~0.05 means the mean changes by < 5 %.
    ``max_tcp_excursion_m``
        TCP displacement from its hold-start position.  The stiff controller
        should keep this small; large excursion means the controller gave up
        to a different equilibrium.
    ``max_force_impulse_frac``
        Frames whose force-norm exceeds ``mean + max_force_impulse_frac * std``
        are flagged as impulse outliers (solver spikes).  Reported but not a
        hard gate.
    """

    max_force_cv: float = 0.10
    max_force_mean_drift_frac: float = 0.05
    max_tcp_excursion_m: float = 0.050


# ------------------------------------------------------------------
# Report dataclasses
# ------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HoldSegmentReport:
    """Force-stability metrics for one contiguous hold at a fixed amplitude."""

    structure_idx: int
    direction_idx: int
    amplitude_m: float
    n_frames: int
    n_frames_expected: int
    is_complete: bool
    metrics_window: str
    # --- command check ---
    commanded_zero_action: bool
    # --- force stability (primary gates) ---
    force_norm_mean_n: float
    force_norm_std_n: float
    force_cv: float
    force_mean_drift_frac: float
    n_impulse_frames: int
    # --- positional (informational) ---
    max_tcp_excursion_m: float
    max_tcp_speed_m_s: float
    max_apple_speed_m_s: float
    max_apple_drift_m: float
    # --- verdict ---
    is_quasi_static: bool
    issues: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DatasetHoldSummary:
    """Aggregate hold quasi-static pass rate over a dataset."""

    dataset_dir: str
    n_episodes: int
    n_hold_segments: int
    n_complete_segments: int
    n_quasi_static_segments: int
    thresholds: dict[str, float]
    expected_hold_frames: int
    segments: tuple[HoldSegmentReport, ...]

    @property
    def quasi_static_rate(self) -> float:
        if self.n_hold_segments == 0:
            return 0.0
        return float(self.n_quasi_static_segments) / float(self.n_hold_segments)

    @property
    def complete_quasi_static_rate(self) -> float:
        if self.n_complete_segments == 0:
            return 0.0
        complete_pass = sum(1 for seg in self.segments if seg.is_complete and seg.is_quasi_static)
        return float(complete_pass) / float(self.n_complete_segments)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def expected_hold_frame_count(
    metadata: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
) -> int:
    """Return configured hold frame count from episode metadata or manifest."""
    control_hz = float(
        metadata.get("control_hz")
        or (manifest or {}).get("collection", {}).get("control_hz", 30.0)
    )
    hold_duration_s = float(metadata.get("hold_duration_s", 1.0))
    return max(0, int(math.ceil(hold_duration_s * control_hz)))


def _apple_speeds_from_positions(pos: np.ndarray, time: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if pos.shape[0] < 2:
        return np.zeros(pos.shape[0], dtype=np.float64)
    dt = np.maximum(np.diff(time), 1e-9)
    speeds = np.linalg.norm(np.diff(pos, axis=0) / dt[:, None], axis=1)
    return np.concatenate([[float(speeds[0])], speeds.astype(np.float64)])


def _iter_hold_segment_indices(
    arrays: dict[str, Any],
    *,
    use_latter_half: bool,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Yield (amplitude, full_idx, metrics_idx) for each contiguous hold segment."""
    phase = np.asarray(arrays["phase"], dtype=np.int8).reshape(-1)
    amplitude = np.asarray(
        arrays.get("amplitude_m", np.zeros(phase.shape[0])), dtype=np.float64
    ).reshape(-1)
    hold_mask = phase == _HOLD_PHASE
    if not np.any(hold_mask):
        return []

    segments: list[tuple[float, np.ndarray, np.ndarray]] = []
    current_amp: float | None = None
    current_indices: list[int] = []
    for frame_idx in np.where(hold_mask)[0].tolist():
        amp = float(amplitude[frame_idx])
        if current_amp is None or np.isclose(amp, current_amp):
            current_amp = amp
            current_indices.append(frame_idx)
            continue
        if current_indices:
            full_idx = np.asarray(current_indices, dtype=np.int64)
            metrics_idx = full_idx[len(full_idx) // 2 :] if use_latter_half else full_idx
            segments.append((float(current_amp), full_idx, metrics_idx))
        current_amp = amp
        current_indices = [frame_idx]
    if current_indices:
        full_idx = np.asarray(current_indices, dtype=np.int64)
        metrics_idx = full_idx[len(full_idx) // 2 :] if use_latter_half else full_idx
        segments.append((float(current_amp), full_idx, metrics_idx))
    return segments


def hold_metric_frame_indices(
    arrays: Mapping[str, Any],
    *,
    use_latter_half: bool = True,
) -> np.ndarray:
    """Return concatenated hold frame indices (latter-half per segment by default)."""
    segments = _iter_hold_segment_indices(arrays, use_latter_half=use_latter_half)
    parts = [metrics_idx for _, _, metrics_idx in segments if int(metrics_idx.size) > 0]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts, axis=0)


# ------------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------------

def analyze_hold_segment(
    arrays: dict[str, Any],
    *,
    structure_idx: int,
    direction_idx: int,
    amplitude_m: float,
    full_frame_indices: np.ndarray,
    metrics_frame_indices: np.ndarray,
    expected_hold_frames: int,
    thresholds: StiffnessIdHoldThresholds,
    metrics_window: str,
) -> HoldSegmentReport:
    """Compute force-stability and positional metrics for one hold window."""
    full_idx = np.asarray(full_frame_indices, dtype=np.int64).reshape(-1)
    idx = np.asarray(metrics_frame_indices, dtype=np.int64).reshape(-1)
    n_frames = int(idx.size)
    is_complete = int(full_idx.size) >= int(0.9 * expected_hold_frames)

    action = np.asarray(arrays["action"], dtype=np.float64).reshape(-1, 6)[idx]
    commanded_zero_action = bool(np.allclose(action, 0.0))

    tcp_vel = np.asarray(arrays["tcp_velocity"], dtype=np.float64).reshape(-1, 6)[idx, :3]
    tcp_pos = np.asarray(arrays["tcp_pos"], dtype=np.float64).reshape(-1, 3)[idx]
    apple_pos = np.asarray(arrays["apple_pos"], dtype=np.float64).reshape(-1, 3)[idx]
    time = np.asarray(
        arrays.get("sim_time", np.arange(arrays["phase"].shape[0])), dtype=np.float64
    ).reshape(-1)[idx]
    ft = np.asarray(arrays["ft_wrist"], dtype=np.float64).reshape(-1, 6)[idx, :3]

    force_norm = np.linalg.norm(ft, axis=1)
    force_mean = float(np.mean(force_norm)) if force_norm.size else 0.0
    force_std  = float(np.std(force_norm))  if force_norm.size else 0.0
    force_cv   = force_std / max(force_mean, 1e-9)

    # Mean drift between first third and last third
    n3 = max(1, n_frames // 3)
    early_mean = float(np.mean(force_norm[:n3]))
    late_mean  = float(np.mean(force_norm[-n3:]))
    force_mean_drift_frac = abs(late_mean - early_mean) / max(force_mean, 1e-9)

    # Impulse frames: norm > mean + 3 * std
    impulse_threshold = force_mean + 3.0 * force_std
    n_impulse = int(np.sum(force_norm > impulse_threshold)) if force_norm.size else 0

    # Positional metrics (informational)
    tcp_speed = np.linalg.norm(tcp_vel, axis=1)
    apple_speed = _apple_speeds_from_positions(apple_pos, time)
    tcp_excursion = float(np.max(np.linalg.norm(tcp_pos - tcp_pos[0], axis=1))) if n_frames >= 2 else 0.0
    max_tcp_speed = float(np.max(tcp_speed)) if tcp_speed.size else 0.0
    max_apple_speed = float(np.max(apple_speed)) if apple_speed.size else 0.0
    apple_drift = float(np.linalg.norm(apple_pos[-1] - apple_pos[0])) if n_frames >= 2 else 0.0

    issues: list[str] = []
    if not commanded_zero_action:
        issues.append("commanded_motion")
    if force_cv > thresholds.max_force_cv:
        issues.append("force_oscillation")
    if force_mean_drift_frac > thresholds.max_force_mean_drift_frac:
        issues.append("force_not_converged")
    if tcp_excursion > thresholds.max_tcp_excursion_m:
        issues.append("tcp_excursion")

    return HoldSegmentReport(
        structure_idx=int(structure_idx),
        direction_idx=int(direction_idx),
        amplitude_m=float(amplitude_m),
        n_frames=n_frames,
        n_frames_expected=int(expected_hold_frames),
        is_complete=is_complete,
        metrics_window=metrics_window,
        commanded_zero_action=commanded_zero_action,
        force_norm_mean_n=force_mean,
        force_norm_std_n=force_std,
        force_cv=force_cv,
        force_mean_drift_frac=force_mean_drift_frac,
        n_impulse_frames=n_impulse,
        max_tcp_excursion_m=tcp_excursion,
        max_tcp_speed_m_s=max_tcp_speed,
        max_apple_speed_m_s=max_apple_speed,
        max_apple_drift_m=apple_drift,
        is_quasi_static=len(issues) == 0,
        issues=tuple(issues),
    )


def analyze_episode_hold_quasi_static(
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    *,
    thresholds: StiffnessIdHoldThresholds | None = None,
    use_latter_half: bool = True,
    manifest: dict[str, Any] | None = None,
) -> list[HoldSegmentReport]:
    """Analyze each hold segment in one batched episode."""
    th = thresholds or StiffnessIdHoldThresholds()
    expected_hold = expected_hold_frame_count(metadata, manifest=manifest)
    structure_idx = int(metadata.get("structure_idx", 0))
    direction_idx = int(metadata.get("direction_idx", 0))
    window = "latter_half" if use_latter_half else "full_hold"

    reports: list[HoldSegmentReport] = []
    for amplitude_m, full_idx, metrics_idx in _iter_hold_segment_indices(arrays, use_latter_half=use_latter_half):
        if metrics_idx.size == 0:
            continue
        reports.append(
            analyze_hold_segment(
                arrays,
                structure_idx=structure_idx,
                direction_idx=direction_idx,
                amplitude_m=amplitude_m,
                full_frame_indices=full_idx,
                metrics_frame_indices=metrics_idx,
                expected_hold_frames=expected_hold,
                thresholds=th,
                metrics_window=window,
            )
        )
    return reports


def analyze_dataset_hold_quasi_static(
    dataset_dir: Path | str,
    *,
    thresholds: StiffnessIdHoldThresholds | None = None,
    use_latter_half: bool = True,
    structure_indices: Iterable[int] | None = None,
    direction_indices: Iterable[int] | None = None,
) -> DatasetHoldSummary:
    """Analyze hold quasi-static behavior across a batched dataset."""
    dataset = BatchedSysIdDataset(dataset_dir)
    th = thresholds or StiffnessIdHoldThresholds()
    entries = dataset.episode_entries()
    if structure_indices is not None:
        structure_set = {int(v) for v in structure_indices}
        entries = [e for e in entries if int(e["structure_idx"]) in structure_set]
    if direction_indices is not None:
        direction_set = {int(v) for v in direction_indices}
        entries = [e for e in entries if int(e["direction_idx"]) in direction_set]

    segments: list[HoldSegmentReport] = []
    expected_hold = 0
    for entry in entries:
        s = int(entry["structure_idx"])
        d = int(entry["direction_idx"])
        arrays = dataset.load_episode_obs_arrays(s, d)
        metadata = dataset.load_episode_metadata(s, d)
        expected_hold = max(expected_hold, expected_hold_frame_count(metadata, manifest=dataset.manifest))
        segments.extend(
            analyze_episode_hold_quasi_static(
                arrays,
                metadata,
                thresholds=th,
                use_latter_half=use_latter_half,
                manifest=dataset.manifest,
            )
        )

    return DatasetHoldSummary(
        dataset_dir=str(Path(dataset_dir)),
        n_episodes=len(entries),
        n_hold_segments=len(segments),
        n_complete_segments=sum(1 for seg in segments if seg.is_complete),
        n_quasi_static_segments=sum(1 for seg in segments if seg.is_quasi_static),
        thresholds=dataclasses.asdict(th),
        expected_hold_frames=int(expected_hold),
        segments=tuple(segments),
    )


def hold_summary_to_dict(summary: DatasetHoldSummary) -> dict[str, Any]:
    """JSON-serializable summary payload."""
    return {
        "dataset_dir": summary.dataset_dir,
        "n_episodes": summary.n_episodes,
        "n_hold_segments": summary.n_hold_segments,
        "n_complete_segments": summary.n_complete_segments,
        "n_quasi_static_segments": summary.n_quasi_static_segments,
        "quasi_static_rate": summary.quasi_static_rate,
        "complete_quasi_static_rate": summary.complete_quasi_static_rate,
        "expected_hold_frames": summary.expected_hold_frames,
        "thresholds": summary.thresholds,
        "segments": [dataclasses.asdict(seg) for seg in summary.segments],
    }


def write_dataset_hold_quasi_static_report(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    thresholds: StiffnessIdHoldThresholds | None = None,
    structure_indices: Iterable[int] | None = None,
    direction_indices: Iterable[int] | None = None,
) -> Path:
    """Write ``hold_quasi_static_summary.json`` under ``output_dir``."""
    summary = analyze_dataset_hold_quasi_static(
        dataset_dir,
        thresholds=thresholds,
        structure_indices=structure_indices,
        direction_indices=direction_indices,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hold_quasi_static_summary.json"
    out_path.write_text(
        json.dumps(hold_summary_to_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def format_hold_summary_text(summary: DatasetHoldSummary) -> str:
    """Human-readable one-screen hold quasi-static report."""
    issue_counts: dict[str, int] = {}
    for seg in summary.segments:
        for issue in seg.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    complete_pass = sum(1 for seg in summary.segments if seg.is_complete and seg.is_quasi_static)

    lines = [
        "Hold quasi-static check (no-damping stiffness-ID mode, latter-half window)",
        f"  dataset:  {summary.dataset_dir}",
        f"  episodes: {summary.n_episodes}",
        f"  hold segments: {summary.n_hold_segments}"
        f"  ({summary.n_complete_segments} complete ≥90% of {summary.expected_hold_frames} frames)",
        f"  quasi-static pass (all):      {summary.n_quasi_static_segments}/{summary.n_hold_segments}"
        f"  ({100.0 * summary.quasi_static_rate:.1f}%)",
        f"  quasi-static pass (complete): {complete_pass}/{summary.n_complete_segments}"
        f"  ({100.0 * summary.complete_quasi_static_rate:.1f}%)",
        "  thresholds:",
        f"    max_force_cv               = {summary.thresholds['max_force_cv']}   (std/mean force norm)",
        f"    max_force_mean_drift_frac  = {summary.thresholds['max_force_mean_drift_frac']}   (|late−early| / mean)",
        f"    max_tcp_excursion_m        = {summary.thresholds['max_tcp_excursion_m']} m",
    ]
    if summary.segments:
        cvs = [seg.force_cv for seg in summary.segments]
        drifts = [seg.force_mean_drift_frac for seg in summary.segments]
        excursions = [seg.max_tcp_excursion_m for seg in summary.segments]
        lines += [
            "  dataset statistics (all hold segments):",
            f"    force_cv:          median={np.median(cvs):.4f}  p90={np.percentile(cvs, 90):.4f}  max={max(cvs):.4f}",
            f"    force_mean_drift:  median={np.median(drifts):.4f}  p90={np.percentile(drifts, 90):.4f}  max={max(drifts):.4f}",
            f"    tcp_excursion:     median={np.median(excursions)*1000:.1f} mm  p90={np.percentile(excursions, 90)*1000:.1f} mm  max={max(excursions)*1000:.1f} mm",
        ]
    if issue_counts:
        lines.append("  failure counts:")
        for key in sorted(issue_counts):
            lines.append(f"    {key}: {issue_counts[key]}")
    return "\n".join(lines)
