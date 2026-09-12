"""Post-grasp pretension diagnostics: load split and settle convergence."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np

from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device

_DEFAULT_SAMPLE_EVERY = 50
_DEFAULT_CONVERGE_RTOL = 0.05
_DEFAULT_TCP_PROXY_WARN_M = 5e-3


@dataclasses.dataclass(frozen=True)
class PlantLoadSplit:
    """Per-env stem vs weld reaction vs apple weight at a quiet pose."""

    stem_force_world: np.ndarray  # (3,)
    weld_force_world: np.ndarray  # (3,)
    apple_weight_N: float
    stem_fz_share: float
    weld_fz_share: float


@dataclasses.dataclass(frozen=True)
class PostGraspPreloadReport:
    """Stem/weld reaction trajectory sampled during post-grasp VBD settle."""

    substeps: int
    sample_every: int
    stem_force_samples: np.ndarray  # (n_samples, n_envs, 3)
    weld_force_samples: np.ndarray  # (n_samples, n_envs, 3)
    relative_change: float
    converged: bool
    apple_weight_N: float


def _joint_indices_for_preload(scene: Any) -> tuple[list[int], list[int]] | None:
    """Return (stem_joint_indices, weld_joint_indices) per env, or None if unavailable."""
    cable = scene.cable
    layout = getattr(scene, "layout", None)
    stem_tpl = getattr(scene, "stem_apple_joint_index", None)
    weld_tpl = getattr(cable, "gripper_proxy_apple_joint", None)
    if stem_tpl is None or weld_tpl is None:
        return None
    if layout is not None and int(layout.num_envs) > 1:
        n = int(layout.num_envs)
        stem = [layout.joint_index(w, int(stem_tpl)) for w in range(n)]
        weld = [layout.joint_index(w, int(weld_tpl)) for w in range(n)]
        return stem, weld
    return [int(stem_tpl)], [int(weld_tpl)]


def gather_stem_and_weld_forces(
    scene: Any,
    *,
    dt: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(stem_f, weld_f)`` each shape ``(n_envs, 3)``, or None if joints missing."""
    pairs = _joint_indices_for_preload(scene)
    if pairs is None:
        return None
    stem_ids, weld_ids = pairs
    cable = scene.cable
    n = len(stem_ids)
    joint_ids = stem_ids + weld_ids
    out_f, _out_t = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=cable.state_0.body_q,
        body_q_prev=cable.solver.body_q_prev,
        joint_indices=joint_ids,
        dt=float(dt),
        control=cable.model.control(clone_variables=False),
        include_penalty_damping=False,
    )
    f = out_f.numpy().reshape(-1, 3)
    return f[:n].copy(), f[n:].copy()


def plant_load_split(scene: Any, *, dt: float) -> list[PlantLoadSplit] | None:
    """Per-env vertical load share between stem→apple and weld reactions."""
    forces = gather_stem_and_weld_forces(scene, dt=dt)
    if forces is None:
        return None
    stem_f, weld_f = forces
    mass = float(getattr(scene, "apple_mass_kg", 0.0) or 0.0)
    if mass <= 0.0:
        cable = scene.cable
        apple = getattr(cable, "apple_body", None)
        if apple is not None:
            mass = float(cable.model.body_mass.numpy()[int(apple)])
    mg = mass * 9.81
    out: list[PlantLoadSplit] = []
    for i in range(stem_f.shape[0]):
        sf = stem_f[i]
        wf = weld_f[i]
        # Child-COM force on apple from stem (stem→apple child=apple) and on proxy
        # from weld (weld child=proxy). Vertical share vs apple weight.
        stem_fz = float(sf[2])
        weld_fz = float(wf[2])
        denom = mg if mg > 1e-6 else 1.0
        out.append(
            PlantLoadSplit(
                stem_force_world=sf.astype(np.float64),
                weld_force_world=wf.astype(np.float64),
                apple_weight_N=float(mg),
                stem_fz_share=stem_fz / denom,
                weld_fz_share=weld_fz / denom,
            )
        )
    return out


def sample_preload_every(substeps: int) -> int:
    n = max(1, int(substeps))
    return max(1, min(_DEFAULT_SAMPLE_EVERY, n // 4 if n >= 4 else 1))


def finalize_preload_report(
    *,
    substeps: int,
    sample_every: int,
    stem_samples: list[np.ndarray],
    weld_samples: list[np.ndarray],
    apple_weight_N: float,
    converge_rtol: float = _DEFAULT_CONVERGE_RTOL,
) -> PostGraspPreloadReport:
    if not stem_samples:
        empty = np.zeros((0, 0, 3), dtype=np.float64)
        return PostGraspPreloadReport(
            substeps=int(substeps),
            sample_every=int(sample_every),
            stem_force_samples=empty,
            weld_force_samples=empty,
            relative_change=float("inf"),
            converged=False,
            apple_weight_N=float(apple_weight_N),
        )
    stem = np.stack(stem_samples, axis=0)
    weld = np.stack(weld_samples, axis=0)
    if stem.shape[0] < 2:
        rel = float("inf")
        converged = False
    else:
        a = stem[-2] + weld[-2]
        b = stem[-1] + weld[-1]
        denom = max(float(np.linalg.norm(b)), float(apple_weight_N), 1e-6)
        rel = float(np.linalg.norm(b - a) / denom)
        converged = rel <= float(converge_rtol)
    return PostGraspPreloadReport(
        substeps=int(substeps),
        sample_every=int(sample_every),
        stem_force_samples=stem,
        weld_force_samples=weld,
        relative_change=rel,
        converged=converged,
        apple_weight_N=float(apple_weight_N),
    )


def warn_if_preload_not_converged(report: PostGraspPreloadReport) -> None:
    if report.converged:
        return
    warnings.warn(
        f"post-grasp settle preload not converged after {report.substeps} substeps "
        f"(relative_change={report.relative_change:.4f})",
        UserWarning,
        stacklevel=2,
    )


def warn_if_tcp_proxy_mismatch(
    scene: Any,
    *,
    tol_m: float = _DEFAULT_TCP_PROXY_WARN_M,
) -> None:
    """Warn when FR3 TCP and cable proxy disagree after rebootstrap."""
    cable = scene.cable
    proxy = getattr(cable, "gripper_proxy_body", None)
    tcp = getattr(scene, "tcp_body_index", None)
    robot_state = getattr(scene, "robot_state_0", None)
    if proxy is None or tcp is None or robot_state is None:
        return
    layout = getattr(scene, "layout", None)
    proxy_bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    tcp_bq = robot_state.body_q.numpy().reshape(-1, 7)
    if layout is not None and int(layout.num_envs) > 1:
        pairs = zip(layout.proxy_body_indices, layout.tcp_body_indices, strict=True)
    else:
        pairs = [(int(proxy), int(tcp))]
    for pid, tid in pairs:
        if int(pid) < 0 or int(tid) < 0:
            continue
        d = float(np.linalg.norm(proxy_bq[int(pid), :3] - tcp_bq[int(tid), :3]))
        if d > float(tol_m):
            warnings.warn(
                f"post-grasp rebootstrap TCP vs proxy pos differ by {d:.4e} m "
                f"(tol={tol_m})",
                UserWarning,
                stacklevel=2,
            )
