"""Post-settle quasi-static checks for spur→stem→apple branch geometry."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np

from apple_pick_sim.fruiting_system.coupled import CoupledCableScene
from apple_pick_sim.fruiting_system.params import FruitingSystemParams


@dataclasses.dataclass(frozen=True)
class SettleQuasiStaticReport:
    """Per-env branch path length compared to nominal spur+stem+apple rest length."""

    world: int
    path_length_m: float
    nominal_length_m: float
    is_quasi_static: bool


DEFAULT_SETTLE_PATH_ATOL_M = 0.002
DEFAULT_SETTLE_PATH_RTOL = 0.01
DEFAULT_SETTLE_MAX_BRANCH_SPEED_M_S = 0.05


def path_length_within_nominal(
    path_length_m: float,
    nominal_length_m: float,
    *,
    atol_m: float = DEFAULT_SETTLE_PATH_ATOL_M,
    rtol: float = DEFAULT_SETTLE_PATH_RTOL,
) -> bool:
    """True when branch path is within nominal rest length plus slack."""
    slack = max(float(atol_m), float(nominal_length_m) * float(rtol))
    return float(path_length_m) <= float(nominal_length_m) + slack


@dataclasses.dataclass(frozen=True)
class SettleStabilityReport:
    """Per-env post-settle stability: geometry, apple height, and residual motion."""

    world: int
    path_length_m: float
    nominal_length_m: float
    is_quasi_static: bool
    apple_z_m: float
    apple_speed_m_s: float
    max_branch_speed_m_s: float
    issues: tuple[str, ...]
    is_stable: bool

    @property
    def path_over_nominal(self) -> float:
        if self.nominal_length_m <= 0.0:
            return float("inf")
        return self.path_length_m / self.nominal_length_m


def nominal_spur_stem_apple_length_m(params: FruitingSystemParams) -> float:
    """Nominal rest length: ``spur.length + stem.length + apple_radius`` [m]."""
    total = 0.0
    if params.spur is not None:
        total += float(params.spur.length)
    if params.stem is not None:
        total += float(params.stem.length)
    if params.apple_radius is not None:
        total += float(params.apple_radius)
    return total


def branch_path_length_m(
    body_q: np.ndarray,
    *,
    spur_bodies: Sequence[int],
    stem_bodies: Sequence[int],
    apple_body: int,
    world_body_offset: int = 0,
) -> float:
    """Polyline length along spur→stem→apple body COM positions [m]."""
    rows = body_q.reshape(-1, 7)
    chain = [*spur_bodies, *stem_bodies, int(apple_body)]
    if len(chain) < 2:
        raise ValueError("branch path requires at least spur/stem/apple bodies")
    pts = [rows[world_body_offset + int(idx), :3] for idx in chain]
    return float(
        sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
    )


def _bodies_per_world(cable: CoupledCableScene) -> int:
    starts = cable.model.body_world_start.numpy()
    return int(starts[1] - starts[0])


def _branch_body_indices(
    *,
    spur_bodies: Sequence[int],
    stem_bodies: Sequence[int],
    apple_body: int,
    world_body_offset: int,
) -> list[int]:
    chain = [*spur_bodies, *stem_bodies, int(apple_body)]
    return [world_body_offset + int(idx) for idx in chain]


def _max_linear_speed_m_s(body_qd: np.ndarray, body_indices: Sequence[int]) -> float:
    rows = body_qd.reshape(-1, 6)
    if not body_indices:
        return 0.0
    speeds = [float(np.linalg.norm(rows[int(idx), :3])) for idx in body_indices]
    return max(speeds)


def per_env_settle_stability_reports(
    body_q: np.ndarray,
    body_qd: np.ndarray,
    params_list: Sequence[FruitingSystemParams],
    *,
    spur_bodies: Sequence[int],
    stem_bodies: Sequence[int],
    apple_body: int,
    bodies_per_world: int,
    max_branch_speed_m_s: float = DEFAULT_SETTLE_MAX_BRANCH_SPEED_M_S,
    apple_z_min_m: float = 0.0,
    path_atol_m: float = DEFAULT_SETTLE_PATH_ATOL_M,
    path_rtol: float = DEFAULT_SETTLE_PATH_RTOL,
) -> list[SettleStabilityReport]:
    """Per-env settle stability: quasi-static geometry, floor height, residual speed."""
    quasi_reports = per_env_settle_quasi_static_reports(
        body_q,
        params_list,
        spur_bodies=spur_bodies,
        stem_bodies=stem_bodies,
        apple_body=apple_body,
        bodies_per_world=bodies_per_world,
        path_atol_m=path_atol_m,
        path_rtol=path_rtol,
    )
    rows_q = body_q.reshape(-1, 7)
    rows_qd = body_qd.reshape(-1, 6)
    reports: list[SettleStabilityReport] = []
    for quasi in quasi_reports:
        offset = int(quasi.world) * int(bodies_per_world)
        branch_indices = _branch_body_indices(
            spur_bodies=spur_bodies,
            stem_bodies=stem_bodies,
            apple_body=int(apple_body),
            world_body_offset=offset,
        )
        apple_idx = offset + int(apple_body)
        apple_z = float(rows_q[apple_idx, 2])
        apple_speed = float(np.linalg.norm(rows_qd[apple_idx, :3]))
        max_branch_speed = _max_linear_speed_m_s(body_qd, branch_indices)
        issues: list[str] = []
        if not quasi.is_quasi_static:
            issues.append("branch_path>nominal")
        if apple_z < float(apple_z_min_m):
            issues.append("apple_below_floor")
        if max_branch_speed > float(max_branch_speed_m_s):
            issues.append("residual_motion")
        reports.append(
            SettleStabilityReport(
                world=quasi.world,
                path_length_m=quasi.path_length_m,
                nominal_length_m=quasi.nominal_length_m,
                is_quasi_static=quasi.is_quasi_static,
                apple_z_m=apple_z,
                apple_speed_m_s=apple_speed,
                max_branch_speed_m_s=max_branch_speed,
                issues=tuple(issues),
                is_stable=len(issues) == 0,
            )
        )
    return reports


def settle_stability_reports_from_cable(
    cable: CoupledCableScene,
    params_list: Sequence[FruitingSystemParams],
    *,
    max_branch_speed_m_s: float = DEFAULT_SETTLE_MAX_BRANCH_SPEED_M_S,
    apple_z_min_m: float = 0.0,
    path_atol_m: float = DEFAULT_SETTLE_PATH_ATOL_M,
    path_rtol: float = DEFAULT_SETTLE_PATH_RTOL,
) -> list[SettleStabilityReport]:
    """Convenience wrapper using a settled :class:`CoupledCableScene`."""
    if cable.apple_body is None:
        raise ValueError("cable scene has no apple body")
    body_q = cable.state_0.body_q.numpy()
    body_qd = cable.state_0.body_qd.numpy()
    return per_env_settle_stability_reports(
        body_q,
        body_qd,
        params_list,
        spur_bodies=cable.spur_bodies,
        stem_bodies=cable.stem_bodies,
        apple_body=int(cable.apple_body),
        bodies_per_world=_bodies_per_world(cable),
        max_branch_speed_m_s=max_branch_speed_m_s,
        apple_z_min_m=apple_z_min_m,
        path_atol_m=path_atol_m,
        path_rtol=path_rtol,
    )


def per_env_settle_quasi_static_reports(
    body_q: np.ndarray,
    params_list: Sequence[FruitingSystemParams],
    *,
    spur_bodies: Sequence[int],
    stem_bodies: Sequence[int],
    apple_body: int,
    bodies_per_world: int,
    path_atol_m: float = DEFAULT_SETTLE_PATH_ATOL_M,
    path_rtol: float = DEFAULT_SETTLE_PATH_RTOL,
) -> list[SettleQuasiStaticReport]:
    """Compare settled branch path length to nominal rest length for each env."""
    if apple_body is None or int(apple_body) < 0:
        raise ValueError("apple_body is required for quasi-static settle checks")
    reports: list[SettleQuasiStaticReport] = []
    for world, params in enumerate(params_list):
        offset = int(world) * int(bodies_per_world)
        path = branch_path_length_m(
            body_q,
            spur_bodies=spur_bodies,
            stem_bodies=stem_bodies,
            apple_body=int(apple_body),
            world_body_offset=offset,
        )
        nominal = nominal_spur_stem_apple_length_m(params)
        reports.append(
            SettleQuasiStaticReport(
                world=int(world),
                path_length_m=path,
                nominal_length_m=nominal,
                is_quasi_static=path_length_within_nominal(
                    path,
                    nominal,
                    atol_m=path_atol_m,
                    rtol=path_rtol,
                ),
            )
        )
    return reports


def count_non_quasi_static_envs(
    body_q: np.ndarray,
    params_list: Sequence[FruitingSystemParams],
    *,
    spur_bodies: Sequence[int],
    stem_bodies: Sequence[int],
    apple_body: int,
    bodies_per_world: int,
) -> tuple[int, list[SettleQuasiStaticReport]]:
    """Return ``(count, reports)`` for envs with path length > nominal rest length."""
    reports = per_env_settle_quasi_static_reports(
        body_q,
        params_list,
        spur_bodies=spur_bodies,
        stem_bodies=stem_bodies,
        apple_body=apple_body,
        bodies_per_world=bodies_per_world,
    )
    count = sum(1 for report in reports if not report.is_quasi_static)
    return count, reports


def count_non_quasi_static_from_cable(
    cable: CoupledCableScene,
    params_list: Sequence[FruitingSystemParams],
) -> tuple[int, list[SettleQuasiStaticReport]]:
    """Convenience wrapper using a settled :class:`CoupledCableScene`."""
    if cable.apple_body is None:
        raise ValueError("cable scene has no apple body")
    body_q = cable.state_0.body_q.numpy()
    return count_non_quasi_static_envs(
        body_q,
        params_list,
        spur_bodies=cable.spur_bodies,
        stem_bodies=cable.stem_bodies,
        apple_body=int(cable.apple_body),
        bodies_per_world=_bodies_per_world(cable),
    )


def print_settle_quasi_static_summary(
    count: int,
    reports: Sequence[SettleQuasiStaticReport],
    *,
    prefix: str = "",
) -> None:
    """Log how many envs exceed nominal spur+stem+apple rest length after settle."""
    total = len(reports)
    print(
        f"{prefix}Post-settle quasi-static check "
        f"(branch path vs spur+stem+apple rest length): "
        f"{count}/{total} envs not quasi-static (path > rest length)",
        flush=True,
    )


def print_settle_stability_report(
    reports: Sequence[SettleStabilityReport],
    *,
    prefix: str = "",
    verbose: bool = True,
) -> None:
    """Log per-env settle stability and a summary stable/unstable count."""
    stable_count = sum(1 for report in reports if report.is_stable)
    total = len(reports)
    print(
        f"{prefix}Post-settle stability "
        f"(branch path vs rest length, apple z, residual speed):",
        flush=True,
    )
    for report in reports:
        if not verbose and report.is_stable:
            continue
        status = "STABLE" if report.is_stable else "UNSTABLE"
        issue_text = f"  issues: {', '.join(report.issues)}" if report.issues else ""
        print(
            f"{prefix}  env{report.world}: {status}  "
            f"path={report.path_length_m:.4f}/{report.nominal_length_m:.4f} m "
            f"({report.path_over_nominal:.2f}×)  "
            f"apple_z={report.apple_z_m:.3f} m  "
            f"|v|_max={report.max_branch_speed_m_s:.4f} m/s"
            f"{issue_text}",
            flush=True,
        )
    print(
        f"{prefix}Summary: {stable_count}/{total} envs stable after settle",
        flush=True,
    )


def count_apples_outside_envelope(
    ik_results: Sequence[tuple[float, float, bool]],
) -> tuple[int, float]:
    """Return ``(count_outside, pct_outside)`` from per-env IK bootstrap results.

    Each entry is ``(pos_err_m, rot_err_rad, is_inside_envelope)``.
    """
    total = len(ik_results)
    outside = sum(1 for _, _, inside in ik_results if not inside)
    pct = 100.0 * outside / max(total, 1)
    return outside, pct


def print_envelope_coverage_report(
    ik_results: Sequence[tuple[float, float, bool]],
    *,
    prefix: str = "",
) -> None:
    """Print per-env IK reachability and summary % outside FR3 working envelope."""
    outside_count, pct_outside = count_apples_outside_envelope(ik_results)
    total = len(ik_results)
    print(
        f"{prefix}Post-settle working envelope "
        f"(FR3 IK bootstrap vs settled gripper proxy):",
        flush=True,
    )
    for world, (pos_err, rot_err, inside) in enumerate(ik_results):
        status = "INSIDE" if inside else "OUTSIDE"
        print(
            f"{prefix}  env{world}: {status}  "
            f"pos_err={pos_err:.4f} m  rot_err={rot_err:.4f} rad",
            flush=True,
        )
    print(
        f"{prefix}Summary: {outside_count}/{total} envs outside working envelope "
        f"({pct_outside:.1f}%)",
        flush=True,
    )
