"""Branch kinetic-energy envelope decay diagnostics for VBD settle."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from apple_pick_sim.coupled_fruiting.settle_quasi_static import (
    DEFAULT_SETTLE_MAX_BRANCH_SPEED_M_S,
    SettleStabilityReport,
    _bodies_per_world,
    _branch_body_indices,
    _max_linear_speed_m_s,
)
from apple_pick_sim.fruiting_system.coupled import CoupledCableScene
from apple_pick_sim.fruiting_system.params import FruitingSystemParams

DEFAULT_KE_SAMPLE_EVERY = 10
DEFAULT_KE_ANALYSIS_TAIL_FRACTION = 0.5
DEFAULT_KE_MIN_PEAKS = 3
DEFAULT_KE_PEAK_DECAY_RTOL = 0.10


@dataclasses.dataclass(frozen=True)
class SettleKeAnalysisConfig:
    """Parameters for peak-envelope decay analysis."""

    analysis_tail_fraction: float = DEFAULT_KE_ANALYSIS_TAIL_FRACTION
    min_peaks: int = DEFAULT_KE_MIN_PEAKS
    peak_decay_rtol: float = DEFAULT_KE_PEAK_DECAY_RTOL
    speed_threshold_m_s: float = DEFAULT_SETTLE_MAX_BRANCH_SPEED_M_S
    ke_peak_threshold_j: float | None = None


@dataclasses.dataclass(frozen=True)
class KeSample:
    """One sampled branch KE point during settle."""

    substep: int
    sim_time_s: float
    ke_j: float
    max_branch_speed_m_s: float


@dataclasses.dataclass(frozen=True)
class SettleKeDecayReport:
    """Per-env strong settle gate from branch KE peak envelope decay."""

    world: int
    peak_ke_j: tuple[float, ...]
    final_peak_ke_j: float
    ke_peak_threshold_j: float
    is_envelope_decaying: bool
    is_ke_below_threshold: bool
    issues: tuple[str, ...]
    is_ke_decay_stable: bool


def branch_linear_kinetic_energy_j(
    body_qd: np.ndarray,
    body_mass: np.ndarray,
    body_indices: Sequence[int],
) -> float:
    """Sum of translational KE [J] over listed bodies (linear speed only)."""
    rows = body_qd.reshape(-1, 6)
    masses = np.asarray(body_mass, dtype=np.float64).reshape(-1)
    total = 0.0
    for idx in body_indices:
        i = int(idx)
        v = rows[i, :3]
        m = float(masses[i])
        total += 0.5 * m * float(np.dot(v, v))
    return float(total)


def branch_total_mass_kg(body_mass: np.ndarray, body_indices: Sequence[int]) -> float:
    masses = np.asarray(body_mass, dtype=np.float64).reshape(-1)
    return float(sum(float(masses[int(idx)]) for idx in body_indices))


def default_ke_peak_threshold_j(
    branch_mass_kg: float,
    *,
    speed_threshold_m_s: float = DEFAULT_SETTLE_MAX_BRANCH_SPEED_M_S,
) -> float:
    """Equivalent peak KE if all branch mass moved at ``speed_threshold_m_s``."""
    m = max(float(branch_mass_kg), 0.0)
    v = float(speed_threshold_m_s)
    return 0.5 * m * v * v


def find_ke_peaks(ke_series: np.ndarray) -> list[int]:
    """Return indices of local maxima (interior points only)."""
    arr = np.asarray(ke_series, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n < 3:
        return []
    peaks: list[int] = []
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(i)
    return peaks


def envelope_is_decaying(
    peak_values: Sequence[float],
    *,
    min_peaks: int = DEFAULT_KE_MIN_PEAKS,
    peak_decay_rtol: float = DEFAULT_KE_PEAK_DECAY_RTOL,
) -> bool:
    """True when peak envelope shrinks with tolerance on the last ``min_peaks`` peaks."""
    peaks = [float(v) for v in peak_values]
    if len(peaks) < int(min_peaks):
        return False
    first = peaks[0]
    last = peaks[-1]
    rtol = float(peak_decay_rtol)
    if last > first * (1.0 + rtol):
        return False
    if last > first * (1.0 - rtol):
        return False
    tail = peaks[-int(min_peaks) :]
    for i in range(1, len(tail)):
        if tail[i] > tail[i - 1] * (1.0 + rtol):
            return False
    return True


def _envelope_issues(
    peak_values: Sequence[float],
    *,
    min_peaks: int,
    peak_decay_rtol: float,
    final_peak_ke_j: float,
    ke_peak_threshold_j: float,
) -> tuple[bool, bool, tuple[str, ...]]:
    peaks = [float(v) for v in peak_values]
    issues: list[str] = []
    if len(peaks) < int(min_peaks):
        issues.append("insufficient_peaks")
    is_decaying = envelope_is_decaying(
        peaks,
        min_peaks=min_peaks,
        peak_decay_rtol=peak_decay_rtol,
    )
    if len(peaks) >= 2:
        first = peaks[0]
        last = peaks[-1]
        if last > first * (1.0 + float(peak_decay_rtol)):
            issues.append("envelope_growing")
    if not is_decaying and "insufficient_peaks" not in issues and "envelope_growing" not in issues:
        issues.append("envelope_not_decaying")
    is_below = float(final_peak_ke_j) <= float(ke_peak_threshold_j)
    if not is_below:
        issues.append("ke_above_threshold")
    return is_decaying, is_below, tuple(issues)


def _tail_samples(
    samples: Sequence[KeSample],
    *,
    analysis_tail_fraction: float,
) -> list[KeSample]:
    rows = list(samples)
    if not rows:
        return []
    frac = float(analysis_tail_fraction)
    if frac >= 1.0:
        return rows
    if frac <= 0.0:
        return rows[-1:]
    start = int(len(rows) * (1.0 - frac))
    start = max(0, min(start, len(rows) - 1))
    return rows[start:]


def per_env_settle_ke_decay_reports(
    samples: Sequence[tuple[int, float, float, float] | KeSample],
    *,
    world: int,
    branch_mass_kg: float,
    config: SettleKeAnalysisConfig | None = None,
) -> list[SettleKeDecayReport]:
    """Analyze one env's KE timeseries and return a decay stability report."""
    cfg = config if config is not None else SettleKeAnalysisConfig()
    parsed: list[KeSample] = []
    for row in samples:
        if isinstance(row, KeSample):
            parsed.append(row)
        else:
            substep, sim_time_s, ke_j, max_speed = row
            parsed.append(
                KeSample(
                    substep=int(substep),
                    sim_time_s=float(sim_time_s),
                    ke_j=float(ke_j),
                    max_branch_speed_m_s=float(max_speed),
                )
            )
    tail = _tail_samples(parsed, analysis_tail_fraction=cfg.analysis_tail_fraction)
    ke_series = np.array([s.ke_j for s in tail], dtype=np.float64)
    peak_idx = find_ke_peaks(ke_series)
    peak_vals = tuple(float(ke_series[i]) for i in peak_idx)
    final_peak = float(peak_vals[-1]) if peak_vals else (float(ke_series[-1]) if ke_series.size else 0.0)
    threshold = (
        float(cfg.ke_peak_threshold_j)
        if cfg.ke_peak_threshold_j is not None
        else default_ke_peak_threshold_j(
            branch_mass_kg,
            speed_threshold_m_s=cfg.speed_threshold_m_s,
        )
    )
    is_decaying, is_below, issues = _envelope_issues(
        peak_vals,
        min_peaks=cfg.min_peaks,
        peak_decay_rtol=cfg.peak_decay_rtol,
        final_peak_ke_j=final_peak,
        ke_peak_threshold_j=threshold,
    )
    return [
        SettleKeDecayReport(
            world=int(world),
            peak_ke_j=peak_vals,
            final_peak_ke_j=final_peak,
            ke_peak_threshold_j=threshold,
            is_envelope_decaying=is_decaying,
            is_ke_below_threshold=is_below,
            issues=issues,
            is_ke_decay_stable=is_decaying and is_below and len(issues) == 0,
        )
    ]


@dataclasses.dataclass
class SettleKeRecorder:
    """Record branch KE samples during VBD settle."""

    num_envs: int
    sample_every: int = DEFAULT_KE_SAMPLE_EVERY
    _samples: dict[int, list[KeSample]] = dataclasses.field(default_factory=dict)
    _branch_masses_kg: dict[int, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        for w in range(int(self.num_envs)):
            self._samples.setdefault(w, [])

    def record_substep(
        self,
        cable: CoupledCableScene,
        params_list: Sequence[FruitingSystemParams],
        substep_idx: int,
        sim_dt: float,
        *,
        sample_every: int | None = None,
    ) -> None:
        """Sample branch KE when ``substep_idx`` hits the sample interval."""
        every = int(sample_every if sample_every is not None else self.sample_every)
        completed = int(substep_idx) + 1
        if every <= 0 or completed % every != 0:
            return
        if cable.apple_body is None:
            raise ValueError("cable scene has no apple body")
        body_qd = cable.state_0.body_qd.numpy()
        body_mass = cable.model.body_mass.numpy()
        bodies_per_world = _bodies_per_world(cable)
        sim_time_s = completed * float(sim_dt)
        for world in range(int(self.num_envs)):
            offset = int(world) * int(bodies_per_world)
            branch_indices = _branch_body_indices(
                spur_bodies=cable.spur_bodies,
                stem_bodies=cable.stem_bodies,
                apple_body=int(cable.apple_body),
                world_body_offset=offset,
            )
            if world not in self._branch_masses_kg:
                self._branch_masses_kg[world] = branch_total_mass_kg(body_mass, branch_indices)
            ke_j = branch_linear_kinetic_energy_j(body_qd, body_mass, branch_indices)
            max_speed = _max_linear_speed_m_s(body_qd, branch_indices)
            self._samples[world].append(
                KeSample(
                    substep=completed,
                    sim_time_s=sim_time_s,
                    ke_j=ke_j,
                    max_branch_speed_m_s=max_speed,
                )
            )

    def timeseries(self, env: int) -> tuple[KeSample, ...]:
        return tuple(self._samples[int(env)])

    def all_timeseries(self) -> dict[int, tuple[KeSample, ...]]:
        return {w: tuple(samples) for w, samples in self._samples.items()}

    def reports(
        self,
        *,
        config: SettleKeAnalysisConfig | None = None,
    ) -> list[SettleKeDecayReport]:
        cfg = config if config is not None else SettleKeAnalysisConfig()
        out: list[SettleKeDecayReport] = []
        for world in range(int(self.num_envs)):
            mass = float(self._branch_masses_kg.get(world, 0.0))
            report = per_env_settle_ke_decay_reports(
                self._samples[world],
                world=world,
                branch_mass_kg=mass,
                config=cfg,
            )
            out.extend(report)
        return out


def settle_ke_decay_reports_from_recorder(
    recorder: SettleKeRecorder,
    *,
    config: SettleKeAnalysisConfig | None = None,
) -> list[SettleKeDecayReport]:
    return recorder.reports(config=config)


def per_env_branch_ke_j_from_cable(cable: CoupledCableScene) -> list[float]:
    """Instantaneous branch linear KE [J] for each env."""
    if cable.apple_body is None:
        raise ValueError("cable scene has no apple body")
    body_qd = cable.state_0.body_qd.numpy()
    body_mass = cable.model.body_mass.numpy()
    bodies_per_world = _bodies_per_world(cable)
    num_worlds = int(cable.model.world_count)
    ke_values: list[float] = []
    for world in range(num_worlds):
        offset = int(world) * int(bodies_per_world)
        branch_indices = _branch_body_indices(
            spur_bodies=cable.spur_bodies,
            stem_bodies=cable.stem_bodies,
            apple_body=int(cable.apple_body),
            world_body_offset=offset,
        )
        ke_values.append(branch_linear_kinetic_energy_j(body_qd, body_mass, branch_indices))
    return ke_values


def print_settle_checkpoint_report(
    stability_reports: Sequence[SettleStabilityReport],
    branch_ke_j: Sequence[float],
    *,
    substep_idx: int,
    sim_time_s: float,
    prefix: str = "",
    verbose: bool = True,
) -> None:
    """Log combined per-env stability and instantaneous branch KE at a checkpoint."""
    stable_count = sum(1 for report in stability_reports if report.is_stable)
    total = len(stability_reports)
    print(
        f"Settle checkpoint @ substep {int(substep_idx)} ({float(sim_time_s):.3f} s sim):",
        flush=True,
    )
    print(
        f"{prefix}Post-settle stability "
        f"(branch path vs rest length, apple z, residual speed, branch KE):",
        flush=True,
    )
    for report, ke_j in zip(stability_reports, branch_ke_j, strict=True):
        if not verbose and report.is_stable:
            continue
        status = "STABLE" if report.is_stable else "UNSTABLE"
        issue_text = f"  issues: {', '.join(report.issues)}" if report.issues else ""
        print(
            f"{prefix}  env{report.world}: {status}  "
            f"path={report.path_length_m:.4f}/{report.nominal_length_m:.4f} m "
            f"({report.path_over_nominal:.2f}×)  "
            f"apple_z={report.apple_z_m:.3f} m  "
            f"|v|_max={report.max_branch_speed_m_s:.4f} m/s  "
            f"ke={float(ke_j):.4g} J"
            f"{issue_text}",
            flush=True,
        )
    print(
        f"{prefix}Summary: {stable_count}/{total} envs stable after settle",
        flush=True,
    )


def print_settle_ke_decay_report(
    reports: Sequence[SettleKeDecayReport],
    *,
    prefix: str = "",
    verbose: bool = True,
    title: str = "Post-settle KE decay (branch linear KE peak envelope)",
) -> None:
    """Log per-env KE decay stability and summary count."""
    stable_count = sum(1 for report in reports if report.is_ke_decay_stable)
    total = len(reports)
    print(
        f"{prefix}{title}:",
        flush=True,
    )
    for report in reports:
        if not verbose and report.is_ke_decay_stable:
            continue
        status = "KE_DECAY_STABLE" if report.is_ke_decay_stable else "KE_DECAY_UNSTABLE"
        first_peak = report.peak_ke_j[0] if report.peak_ke_j else float("nan")
        last_peak = report.peak_ke_j[-1] if report.peak_ke_j else float("nan")
        issue_text = f"  issues: {', '.join(report.issues)}" if report.issues else ""
        print(
            f"{prefix}  env{report.world}: {status}  "
            f"peaks={len(report.peak_ke_j)}  "
            f"first={first_peak:.4g} J  last={last_peak:.4g} J  "
            f"final_peak={report.final_peak_ke_j:.4g} J  "
            f"threshold={report.ke_peak_threshold_j:.4g} J"
            f"{issue_text}",
            flush=True,
        )
    print(
        f"{prefix}Summary: {stable_count}/{total} envs KE-decay-stable",
        flush=True,
    )


def peak_rows_from_report(
    report: SettleKeDecayReport,
    samples: Sequence[KeSample],
    *,
    analysis_tail_fraction: float = DEFAULT_KE_ANALYSIS_TAIL_FRACTION,
) -> list[tuple[int, float, float]]:
    """Map peak KE values to ``(peak_idx, sim_time_s, peak_ke_j)`` in the analysis window."""
    tail = _tail_samples(samples, analysis_tail_fraction=analysis_tail_fraction)
    ke_series = np.array([s.ke_j for s in tail], dtype=np.float64)
    peak_idx = find_ke_peaks(ke_series)
    rows: list[tuple[int, float, float]] = []
    for i in peak_idx:
        rows.append((int(i), float(tail[i].sim_time_s), float(ke_series[i])))
    return rows
