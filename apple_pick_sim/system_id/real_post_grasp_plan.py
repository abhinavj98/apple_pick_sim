"""Post-grasp plan: follow logged TCP + apple SE(3); diagnostic residuals only."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    align_proxy_body_q_prev_for_vbd,
    sync_model_body_q_rest_from_state,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    _proxy_world_pose_from_apple,
    quiet_all_cable_bodies,
)
from apple_pick_sim.fruiting_system.coupled import CoupledCableScene, generate_coupled_cable_scene
from apple_pick_sim.fruiting_system.params import FruitingSystemParams, GripperProxyConfig
from apple_pick_sim.system_id.real_pre_grasp_params import coerce_xyz

_ZERO_EPS = 1e-12
_DEFAULT_WARN_TOL_M = 0.02
_DEFAULT_APPROACH_ALIGN_MIN_DOT = 0.9
_DEFAULT_POSE_POS_MATCH_TOL_M = 1e-4
_PROXY_TCP_POS_WARN_M = 0.02
_PROXY_TCP_QUAT_MIN_ABS_DOT = 0.99


def pose_4x4_to_pos_quat(
    flat16: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Row-major 4×4 (translation at indices 3,7,11) → pos + quat (x,y,z,w)."""
    M = np.asarray(flat16, dtype=np.float64).reshape(4, 4)
    pos = (float(M[0, 3]), float(M[1, 3]), float(M[2, 3]))
    R = M[:3, :3].astype(np.float64)
    # Orthonormalize via SVD for robustness
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    quat = _rotmat_to_quat_xyzw(R)
    return pos, quat


def _rotmat_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → (x, y, z, w)."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _unit(vec: np.ndarray, *, field: str) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < _ZERO_EPS:
        raise ValueError(f"{field}: zero-length vector")
    return vec / n


@dataclass(frozen=True)
class PostGraspPlan:
    """Logged TCP + apple SE(3) for weld; no catalog surface snap."""

    tcp_pos: tuple[float, float, float]
    tcp_quat_xyzw: tuple[float, float, float, float]
    apple_pos_measured: tuple[float, float, float]
    apple_quat_xyzw: tuple[float, float, float, float]
    apple_pos_welded: tuple[float, float, float]
    """Apple center used for the FIXED weld; equals ``apple_pos_measured`` (follow data)."""
    weld_direction: tuple[float, float, float]
    apple_radius_m: float
    tcp_apple_distance_m: float
    tcp_radius_residual_m: float
    apple_shift_m: float
    """Always 0 when following measured apple (kept for API compatibility)."""
    tcp_approach_dot_weld: float
    """``+Z_tcp · unit(apple − tcp)`` — tip-out toward apple; expect near +1."""


def proxy_offset_from_apple_and_tcp(
    *,
    apple_pos: tuple[float, float, float],
    apple_quat_xyzw: tuple[float, float, float, float],
    tcp_pos: tuple[float, float, float],
    tcp_quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float, float]:
    """Apple-frame FIXED offset ``X_offset = X_apple^{-1} X_tcp`` as 7-tuple."""
    apple_tf = wp.transform(wp.vec3(*apple_pos), wp.quat(*apple_quat_xyzw))
    tcp_tf = wp.transform(wp.vec3(*tcp_pos), wp.quat(*tcp_quat_xyzw))
    offset_tf = wp.transform_multiply(wp.transform_inverse(apple_tf), tcp_tf)
    t = wp.transform_get_translation(offset_tf)
    q = wp.transform_get_rotation(offset_tf)
    return (
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )


def build_post_grasp_plan(
    *,
    tcp_pose_4x4: Any,
    apple_pose_4x4: Any,
    apple_radius_m: float,
    tcp_pos_override: tuple[float, float, float] | None = None,
    apple_pos_override: tuple[float, float, float] | None = None,
    warn_tol_m: float = _DEFAULT_WARN_TOL_M,
    approach_align_min_dot: float = _DEFAULT_APPROACH_ALIGN_MIN_DOT,
    pose_pos_match_tol_m: float = _DEFAULT_POSE_POS_MATCH_TOL_M,
    emit_warnings: bool = True,
) -> PostGraspPlan:
    """Build plan from logged poses; warn on mismatches without correcting them."""
    r = float(apple_radius_m)
    if not np.isfinite(r) or r <= 0:
        raise ValueError(f"apple_radius_m must be positive finite, got {r}")

    tcp_pos_pose, tcp_quat = pose_4x4_to_pos_quat(tcp_pose_4x4)
    apple_pos_pose, apple_quat = pose_4x4_to_pos_quat(apple_pose_4x4)

    tcp = np.asarray(tcp_pos_override if tcp_pos_override is not None else tcp_pos_pose, dtype=np.float64)
    apple_m = np.asarray(
        apple_pos_override if apple_pos_override is not None else apple_pos_pose, dtype=np.float64
    )
    if tcp.size != 3 or apple_m.size != 3:
        raise ValueError("tcp/apple positions must have length 3")

    if emit_warnings:
        d_tcp = float(np.linalg.norm(tcp - np.asarray(tcp_pos_pose, dtype=np.float64)))
        if d_tcp > pose_pos_match_tol_m:
            warnings.warn(
                f"post-grasp data mismatch: tcp_pos vs tcp_pose_4x4 translation "
                f"differ by {d_tcp:.4f} m (tol={pose_pos_match_tol_m})",
                UserWarning,
                stacklevel=2,
            )
        d_ap = float(np.linalg.norm(apple_m - np.asarray(apple_pos_pose, dtype=np.float64)))
        if d_ap > pose_pos_match_tol_m:
            warnings.warn(
                f"post-grasp data mismatch: apple_pos vs apple_pose_4x4 translation "
                f"differ by {d_ap:.4f} m (tol={pose_pos_match_tol_m})",
                UserWarning,
                stacklevel=2,
            )

    delta = tcp - apple_m
    d = float(np.linalg.norm(delta))
    w_hat = _unit(delta, field="weld_direction")
    # Follow data: do not snap apple onto the catalog r-sphere.
    apple_w = apple_m
    residual = abs(d - r)
    shift = 0.0

    # Logged TCP +Z (column 2 of R): tip-out should face apple (TCP → apple).
    M = np.asarray(tcp_pose_4x4, dtype=np.float64).reshape(4, 4)
    tcp_z = M[:3, 2].astype(np.float64)
    tcp_z = tcp_z / max(float(np.linalg.norm(tcp_z)), _ZERO_EPS)
    tcp_to_apple = -w_hat  # unit(apple − tcp)
    approach_dot = float(np.dot(tcp_to_apple, tcp_z))

    if emit_warnings:
        tol = float(warn_tol_m)
        if residual > tol:
            warnings.warn(
                f"post-grasp data mismatch: |tcp−apple|={d:.4f} m vs apple radius "
                f"r={r:.4f} m (residual={residual:.4f} m > tol={tol} m); "
                f"diagnostic only — sim still uses measured apple and TCP poses",
                UserWarning,
                stacklevel=2,
            )
        if approach_dot < float(approach_align_min_dot):
            warnings.warn(
                f"post-grasp data mismatch: logged TCP +Z poorly aligned with "
                f"TCP→apple (dot={approach_dot:.3f}, min={approach_align_min_dot}); "
                f"diagnostic only — sim still uses logged TCP quat",
                UserWarning,
                stacklevel=2,
            )

    return PostGraspPlan(
        tcp_pos=(float(tcp[0]), float(tcp[1]), float(tcp[2])),
        tcp_quat_xyzw=tcp_quat,
        apple_pos_measured=(float(apple_m[0]), float(apple_m[1]), float(apple_m[2])),
        apple_quat_xyzw=apple_quat,
        apple_pos_welded=(float(apple_w[0]), float(apple_w[1]), float(apple_w[2])),
        weld_direction=(float(w_hat[0]), float(w_hat[1]), float(w_hat[2])),
        apple_radius_m=r,
        tcp_apple_distance_m=d,
        tcp_radius_residual_m=residual,
        apple_shift_m=shift,
        tcp_approach_dot_weld=approach_dot,
    )


def post_grasp_plan_from_metadata(
    meta: dict[str, Any],
    *,
    apple_radius_m: float,
    warn_tol_m: float = _DEFAULT_WARN_TOL_M,
    approach_align_min_dot: float = _DEFAULT_APPROACH_ALIGN_MIN_DOT,
    emit_warnings: bool = True,
) -> PostGraspPlan:
    """Build plan from ``dataset_metadata`` including ``post_grasp_geometry``.

    New compiled episodes store the grasped apple under
    ``post_grasp_geometry.snapshot.apple_pose_4x4`` (and ``apple_pos``);
    TCP remains on ``post_grasp_geometry.tcp_pose_4x4``.
    """
    post = meta.get("post_grasp_geometry")
    if not isinstance(post, dict):
        raise ValueError("missing post_grasp_geometry in dataset metadata")
    if "tcp_pose_4x4" not in post:
        raise ValueError("post_grasp_geometry requires tcp_pose_4x4")

    snap = post.get("snapshot")
    if not isinstance(snap, dict) or "apple_pose_4x4" not in snap:
        raise ValueError("post_grasp_geometry.snapshot requires apple_pose_4x4")

    tcp_override = None
    apple_override = None
    if "tcp_pos" in post:
        tcp_override = tuple(float(x) for x in coerce_xyz(post["tcp_pos"], field="tcp_pos"))
    if "apple_pos" in snap:
        apple_override = tuple(float(x) for x in coerce_xyz(snap["apple_pos"], field="apple_pos"))
    elif "apple_pos" in post:
        apple_override = tuple(float(x) for x in coerce_xyz(post["apple_pos"], field="apple_pos"))

    # Prefer parts radius if present and differs — warn
    parts = (meta.get("pre_grasp_geometry") or {}).get("parts") or {}
    if emit_warnings and isinstance(parts.get("apple"), dict) and "radius_m" in parts["apple"]:
        parts_r = float(parts["apple"]["radius_m"])
        if abs(parts_r - float(apple_radius_m)) > 1e-6:
            warnings.warn(
                f"post-grasp data mismatch: apple_radius_m argument {apple_radius_m} "
                f"!= pre_grasp parts.apple.radius_m {parts_r}",
                UserWarning,
                stacklevel=2,
            )

    return build_post_grasp_plan(
        tcp_pose_4x4=post["tcp_pose_4x4"],
        apple_pose_4x4=snap["apple_pose_4x4"],
        apple_radius_m=apple_radius_m,
        tcp_pos_override=tcp_override,
        apple_pos_override=apple_override,
        warn_tol_m=warn_tol_m,
        approach_align_min_dot=approach_align_min_dot,
        emit_warnings=emit_warnings,
    )


def format_post_grasp_plan(plan: PostGraspPlan) -> str:
    """Human-readable grasp plan summary."""
    return "\n".join(
        [
            "post-grasp plan:",
            f"  tcp_pos={plan.tcp_pos}",
            f"  apple_meas={plan.apple_pos_measured}",
            f"  apple_welded={plan.apple_pos_welded}",
            f"  weld_dir={plan.weld_direction}",
            f"  |tcp−apple|={plan.tcp_apple_distance_m:.4f} m  r={plan.apple_radius_m:.4f} m  "
            f"residual={plan.tcp_radius_residual_m:.4f} m",
            f"  apple_shift={plan.apple_shift_m:.4f} m  "
            f"tcp_+Z·(tcp→apple)={plan.tcp_approach_dot_weld:.3f}",
        ]
    )


def apply_post_grasp_after_settle(
    free_scene: CoupledCableScene,
    plan: PostGraspPlan,
    *,
    ranges: dict,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
    device: str | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
    proxy_tcp_pos_warn_m: float = _PROXY_TCP_POS_WARN_M,
    emit_warnings: bool = True,
) -> CoupledCableScene:
    """Rebuild welded cable scene; seed woody from free settle; apple+proxy from plan.

    Proxy and apple world poses match logged SE(3). FIXED joint encodes
    ``X_offset = X_apple^{-1} X_tcp``. No catalog-radius surface snap.
    """
    if free_scene.apple_body is None:
        raise ValueError("free_scene has no apple_body")

    bq = free_scene.state_0.body_q.numpy().reshape(-1, 7).astype(np.float32).copy()
    apple_id = int(free_scene.apple_body)
    bq[apple_id, 0:3] = np.asarray(plan.apple_pos_welded, dtype=np.float32)
    bq[apple_id, 3:7] = np.asarray(plan.apple_quat_xyzw, dtype=np.float32)

    # Snapshot one woody body (first primary) for mismatch checks after copy
    woody_ids = list(free_scene.primary_bodies) + list(free_scene.spur_bodies) + list(
        free_scene.stem_bodies
    )
    woody_before = {i: bq[i].copy() for i in woody_ids if i != apple_id}

    offset_7 = proxy_offset_from_apple_and_tcp(
        apple_pos=plan.apple_pos_welded,
        apple_quat_xyzw=plan.apple_quat_xyzw,
        tcp_pos=plan.tcp_pos,
        tcp_quat_xyzw=plan.tcp_quat_xyzw,
    )
    welded = generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=params,
        base_pos=base_pos,
        device=device,
        robot_base_pos=robot_base_pos,
        gripper_proxy=GripperProxyConfig(
            fix_to_apple=True,
            weld_reference_pos=plan.apple_pos_welded,
            weld_reference_quat=plan.apple_quat_xyzw,
            weld_proxy_offset_in_apple_frame=offset_7,
        ),
    )
    if welded.apple_body is None or welded.gripper_proxy_offset_in_apple_frame is None:
        raise ValueError("welded scene missing apple or proxy offset")
    if int(welded.apple_body) != apple_id:
        raise ValueError(
            f"apple body index changed free={apple_id} welded={welded.apple_body}"
        )

    wbq = welded.state_0.body_q.numpy().reshape(-1, 7).astype(np.float32).copy()
    proxy = int(welded.gripper_proxy_body)
    n = min(bq.shape[0], wbq.shape[0])
    for i in range(n):
        if i == proxy:
            continue
        wbq[i] = bq[i]

    if emit_warnings:
        for i, before in woody_before.items():
            if i >= wbq.shape[0] or i == apple_id:
                continue
            # After copy, woody should match snapshot (we do not teleport woody)
            err = float(np.linalg.norm(wbq[i, :3] - before[:3]))
            if err > 1e-5:
                warnings.warn(
                    f"post-grasp apply mismatch: woody body {i} moved {err:.4e} m "
                    f"during seed (expected unchanged)",
                    UserWarning,
                    stacklevel=2,
                )

    off = welded.gripper_proxy_offset_in_apple_frame
    proxy_pos, proxy_quat = _proxy_world_pose_from_apple(wbq[apple_id], off)
    wbq[proxy, 0:3] = proxy_pos
    wbq[proxy, 3:7] = proxy_quat

    if emit_warnings:
        d_proxy = float(np.linalg.norm(proxy_pos - np.asarray(plan.tcp_pos, dtype=np.float32)))
        if d_proxy > float(proxy_tcp_pos_warn_m):
            warnings.warn(
                f"post-grasp data/sim mismatch: welded proxy pos vs tcp_pos differ by "
                f"{d_proxy:.4f} m (tol={proxy_tcp_pos_warn_m})",
                UserWarning,
                stacklevel=2,
            )
        tq = np.asarray(plan.tcp_quat_xyzw, dtype=np.float64)
        pq = np.asarray(proxy_quat, dtype=np.float64)
        q_dot = abs(float(np.dot(pq, tq)))
        if q_dot < float(_PROXY_TCP_QUAT_MIN_ABS_DOT):
            warnings.warn(
                f"post-grasp data/sim mismatch: welded proxy quat vs tcp_quat "
                f"abs-dot={q_dot:.4f} (min={_PROXY_TCP_QUAT_MIN_ABS_DOT})",
                UserWarning,
                stacklevel=2,
            )

    zeros = np.zeros((wbq.shape[0], 6), dtype=np.float32)
    welded.state_0.body_q.assign(wbq)
    welded.state_0.body_qd.assign(zeros)
    welded.state_1.body_q.assign(wbq)
    welded.state_1.body_qd.assign(zeros)
    quiet_all_cable_bodies(welded)
    body_count = int(welded.model.body_count)
    align_proxy_body_q_prev_for_vbd(welded, tuple(range(body_count)))
    sync_model_body_q_rest_from_state(welded)
    return welded
