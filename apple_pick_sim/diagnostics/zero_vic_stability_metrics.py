"""Stability metrics for zero-VIC hold diagnostics and grid sweeps."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class StabilityThresholds:
    """Pass/fail gates for post-settle zero-VIC hold."""

    max_apple_drift_m: float = 0.02
    max_apple_z_drop_m: float = 0.015
    max_apple_path_length_m: float = 0.05
    max_pos_err_m: float = 0.05
    max_tcp_speed_m_s: float = 0.05
    max_apple_speed_m_s: float = 0.05
    max_harvest_force_n: float = 200.0
    apple_z_min_m: float = 0.0


@dataclasses.dataclass(frozen=True)
class EnvStabilityMetrics:
    """Per-env aggregate over a hold window."""

    env: int
    is_finite: bool
    is_stable: bool
    issues: tuple[str, ...]
    max_apple_drift_m: float
    max_apple_z_drop_m: float
    apple_path_length_m: float
    apple_pos_std_m: float
    max_pos_err_m: float
    max_tcp_speed_m_s: float
    max_apple_speed_m_s: float
    max_harvest_force_n: float
    min_apple_z_m: float


@dataclasses.dataclass(frozen=True)
class HoldSummary:
    """Aggregate pass rates for one config trial."""

    num_envs: int
    vic_pass_rate: float
    settle_pass_rate: float | None
    ik_pass_rate: float | None
    max_apple_drift_m: float
    max_apple_z_drop_m: float
    max_apple_path_length_m: float
    max_pos_err_m: float
    max_tcp_speed_m_s: float
    max_harvest_force_n: float


def parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats from CLI grid arguments."""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one float in comma-separated list")
    return [float(p) for p in parts]


def _apple_pos(row: dict) -> tuple[float, float, float]:
    return float(row["apple_x"]), float(row["apple_y"]), float(row["apple_z"])


def _speed3(row: dict, prefix: str) -> float:
    vx = float(row[f"{prefix}x"])
    vy = float(row[f"{prefix}y"])
    vz = float(row[f"{prefix}z"])
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def _harvest_force_n(row: dict) -> float:
    fx = float(row["harvest_fx"])
    fy = float(row["harvest_fy"])
    fz = float(row["harvest_fz"])
    return math.sqrt(fx * fx + fy * fy + fz * fz)


def _is_finite_row(row: dict) -> bool:
    keys = (
        "pos_err_m",
        "tcp_vx",
        "tcp_vy",
        "tcp_vz",
        "apple_x",
        "apple_y",
        "apple_z",
        "apple_vx",
        "apple_vy",
        "apple_vz",
        "harvest_fx",
        "harvest_fy",
        "harvest_fz",
    )
    for key in keys:
        if not math.isfinite(float(row[key])):
            return False
    return True


def _apple_path_length_m(positions: Sequence[tuple[float, float, float]]) -> float:
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for i in range(len(positions) - 1):
        a = positions[i]
        b = positions[i + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def _apple_pos_std_m(positions: Sequence[tuple[float, float, float]]) -> float:
    if not positions:
        return float("inf")
    n = float(len(positions))
    mx = sum(p[0] for p in positions) / n
    my = sum(p[1] for p in positions) / n
    mz = sum(p[2] for p in positions) / n
    var = sum(
        (p[0] - mx) ** 2 + (p[1] - my) ** 2 + (p[2] - mz) ** 2 for p in positions
    ) / n
    return math.sqrt(var)


def compute_env_stability_metrics(
    rows: Sequence[dict],
    *,
    env: int,
    duration_max: float,
    thresholds: StabilityThresholds | None = None,
) -> EnvStabilityMetrics:
    """Aggregate stability metrics for one env over ``t_s <= duration_max``."""
    th = thresholds if thresholds is not None else StabilityThresholds()
    env_rows = sorted(
        (
            r
            for r in rows
            if int(r["env"]) == int(env) and float(r["t_s"]) <= float(duration_max) + 1e-9
        ),
        key=lambda r: float(r["t_s"]),
    )
    if not env_rows:
        return EnvStabilityMetrics(
            env=int(env),
            is_finite=False,
            is_stable=False,
            issues=("no_rows",),
            max_apple_drift_m=float("inf"),
            max_apple_z_drop_m=float("inf"),
            apple_path_length_m=float("inf"),
            apple_pos_std_m=float("inf"),
            max_pos_err_m=float("inf"),
            max_tcp_speed_m_s=float("inf"),
            max_apple_speed_m_s=float("inf"),
            max_harvest_force_n=float("inf"),
            min_apple_z_m=float("-inf"),
        )

    is_finite = all(_is_finite_row(r) for r in env_rows)
    positions = [_apple_pos(r) for r in env_rows]
    p0 = positions[0]
    max_drift = max(
        math.sqrt(
            (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2 + (p[2] - p0[2]) ** 2
        )
        for p in positions
    )
    min_z = min(p[2] for p in positions)
    max_z_drop = max(0.0, p0[2] - min_z)
    path_len = _apple_path_length_m(positions)
    pos_std = _apple_pos_std_m(positions)

    max_pos = max(float(r["pos_err_m"]) for r in env_rows)
    max_tcp = max(_speed3(r, "tcp_v") for r in env_rows)
    max_apple = max(_speed3(r, "apple_v") for r in env_rows)
    max_harvest = max(_harvest_force_n(r) for r in env_rows)

    issues: list[str] = []
    if not is_finite:
        issues.append("non_finite")
    if max_drift > th.max_apple_drift_m:
        issues.append("apple_drift")
    if max_z_drop > th.max_apple_z_drop_m:
        issues.append("apple_sag")
    if path_len > th.max_apple_path_length_m:
        issues.append("apple_wander")
    if max_pos > th.max_pos_err_m:
        issues.append("pos_err")
    if max_tcp > th.max_tcp_speed_m_s:
        issues.append("tcp_speed")
    if max_apple > th.max_apple_speed_m_s:
        issues.append("apple_speed")
    if max_harvest > th.max_harvest_force_n:
        issues.append("harvest_force")
    if min_z < th.apple_z_min_m:
        issues.append("apple_floor")

    return EnvStabilityMetrics(
        env=int(env),
        is_finite=is_finite,
        is_stable=len(issues) == 0,
        issues=tuple(issues),
        max_apple_drift_m=max_drift,
        max_apple_z_drop_m=max_z_drop,
        apple_path_length_m=path_len,
        apple_pos_std_m=pos_std,
        max_pos_err_m=max_pos,
        max_tcp_speed_m_s=max_tcp,
        max_apple_speed_m_s=max_apple,
        max_harvest_force_n=max_harvest,
        min_apple_z_m=min_z,
    )


def summarize_hold_metrics(
    metrics: Sequence[EnvStabilityMetrics],
    *,
    settle_stable: Sequence[bool] | None = None,
    ik_inside: Sequence[bool] | None = None,
) -> HoldSummary:
    """Compute pass rates and worst-case scalars across envs."""
    n = len(metrics)
    if n == 0:
        return HoldSummary(
            num_envs=0,
            vic_pass_rate=0.0,
            settle_pass_rate=None,
            ik_pass_rate=None,
            max_apple_drift_m=float("inf"),
            max_apple_z_drop_m=float("inf"),
            max_apple_path_length_m=float("inf"),
            max_pos_err_m=float("inf"),
            max_tcp_speed_m_s=float("inf"),
            max_harvest_force_n=float("inf"),
        )

    vic_pass = sum(1 for m in metrics if m.is_stable) / n
    settle_rate = None
    if settle_stable is not None and len(settle_stable) == n:
        settle_rate = sum(1 for ok in settle_stable if ok) / n
    ik_rate = None
    if ik_inside is not None and len(ik_inside) == n:
        ik_rate = sum(1 for ok in ik_inside if ok) / n

    return HoldSummary(
        num_envs=n,
        vic_pass_rate=vic_pass,
        settle_pass_rate=settle_rate,
        ik_pass_rate=ik_rate,
        max_apple_drift_m=max(m.max_apple_drift_m for m in metrics),
        max_apple_z_drop_m=max(m.max_apple_z_drop_m for m in metrics),
        max_apple_path_length_m=max(m.apple_path_length_m for m in metrics),
        max_pos_err_m=max(m.max_pos_err_m for m in metrics),
        max_tcp_speed_m_s=max(m.max_tcp_speed_m_s for m in metrics),
        max_harvest_force_n=max(m.max_harvest_force_n for m in metrics),
    )
