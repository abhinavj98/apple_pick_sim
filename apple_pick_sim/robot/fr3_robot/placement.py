"""FR3 root placement and IK bootstrap from cable proxy."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.fruiting_system import CoupledCableScene
from apple_pick_sim.robot.fr3_robot.setup import np_zeros_like_joint_qd

# Post-bootstrap TCP vs gripper-proxy tolerances (straight-rod fixture seed sweep).
IK_BOOTSTRAP_POS_TOL_M = 0.05
IK_BOOTSTRAP_ROT_TOL_RAD = 0.05

# Per-frame teleop: IK solution vs integrated TCP target (after one velocity step from FK).
IK_TELEOP_POS_TOL_M = 0.005
IK_TELEOP_ROT_TOL_RAD = 0.005

# Deterministic ``joint_q`` recipes (fraction of joint-limit span) after the model default.
_IK_BOOTSTRAP_JOINT_Q_SEED_FRACS = (0.5, 0.25, 0.75, 0.1, 0.9)


class IKBootstrapConvergenceWarning(UserWarning):
    """FR3 TCP IK bootstrap missed position/orientation tolerance vs gripper proxy."""


class IKBootstrapConvergenceError(UserWarning):
    """FR3 TCP IK bootstrap missed position/orientation tolerance vs the gripper proxy."""


class IKTeleopConvergenceError(UserWarning):
    """FR3 TCP IK teleop missed position/orientation tolerance vs the integrated target."""


def enable_ik_bootstrap_warnings_for_examples() -> None:
    """Show :class:`IKBootstrapConvergenceWarning` on every occurrence (CLI / examples)."""
    warnings.simplefilter("always", IKBootstrapConvergenceWarning)


def pose_errors_q7_vs_q7(q7_a: np.ndarray, q7_b: np.ndarray) -> tuple[float, float]:
    """Return position error [m] and orientation error [rad] between two ``body_q`` rows."""
    a = np.asarray(q7_a, dtype=np.float64).reshape(7)
    b = np.asarray(q7_b, dtype=np.float64).reshape(7)
    pos_err = float(np.linalg.norm(a[:3] - b[:3]))
    qa = a[3:7] / (np.linalg.norm(a[3:7]) + 1e-12)
    qb = b[3:7] / (np.linalg.norm(b[3:7]) + 1e-12)
    rot_err = 2.0 * float(np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))
    return pos_err, rot_err


def pose_errors_q7_vs_target(q7: np.ndarray, target_tf: wp.transform) -> tuple[float, float]:
    """Return TCP position/orientation error [m, rad] vs a world-frame target transform."""
    pos = wp.transform_get_translation(target_tf)
    rot = wp.transform_get_rotation(target_tf)
    target_q7 = np.array(
        [float(pos[0]), float(pos[1]), float(pos[2]), float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],
        dtype=np.float64,
    )
    return pose_errors_q7_vs_q7(q7, target_q7)


def tcp_proxy_pose_errors(
    robot_body_q: np.ndarray,
    proxy_body_q: np.ndarray,
    *,
    tcp_body_index: int,
    proxy_body_index: int,
) -> tuple[float, float]:
    """Return TCP position error [m] and orientation error [rad] vs the proxy."""
    rq = robot_body_q.reshape(-1, 7)[tcp_body_index]
    pq = proxy_body_q.reshape(-1, 7)[proxy_body_index]
    return pose_errors_q7_vs_q7(rq, pq)


def tcp_ik_target_pose_errors(
    robot_model: newton.Model,
    state: Any,
    *,
    tcp_body_index: int,
    target_tf: wp.transform,
    joint_q: Any,
) -> tuple[float, float]:
    """FK ``joint_q`` and compare TCP pose to ``target_tf``. Restores ``state`` afterward."""
    jq_saved = state.joint_q.numpy().copy()
    jqd_saved = state.joint_qd.numpy().copy()
    jqd_zero = np_zeros_like_joint_qd(robot_model)
    jq = np.asarray(joint_q, dtype=robot_model.joint_q.dtype).reshape(-1)
    state.joint_q.assign(jq)
    state.joint_qd.assign(jqd_zero)
    newton.eval_fk(robot_model, state.joint_q, state.joint_qd, state)
    pos_err, rot_err = pose_errors_q7_vs_target(
        state.body_q.numpy().reshape(-1, 7)[tcp_body_index],
        target_tf,
    )
    state.joint_q.assign(jq_saved)
    state.joint_qd.assign(jqd_saved)
    newton.eval_fk(robot_model, state.joint_q, state.joint_qd, state)
    return pos_err, rot_err


def raise_if_ik_teleop_not_converged(
    pos_err_m: float,
    rot_err_rad: float,
    *,
    pos_tol_m: float = IK_TELEOP_POS_TOL_M,
    rot_tol_rad: float = IK_TELEOP_ROT_TOL_RAD,
    target_pos: tuple[float, float, float] | None = None,
) -> None:
    """Raise :class:`IKTeleopConvergenceError` when TCP IK teleop misses tolerance."""
    pos_ok = pos_err_m < pos_tol_m
    rot_ok = rot_err_rad < rot_tol_rad
    if pos_ok and rot_ok:
        return

    parts: list[str] = []
    if not pos_ok:
        parts.append(f"position error {pos_err_m:.4f} m (tol {pos_tol_m:.4f} m)")
    if not rot_ok:
        parts.append(f"orientation error {rot_err_rad:.4f} rad (tol {rot_tol_rad:.4f} rad)")
    target_note = ""
    if target_pos is not None:
        target_note = (
            f"; target position ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
        )
    raise IKTeleopConvergenceError(
        "FR3 TCP IK teleop did not converge: "
        + ", ".join(parts)
        + target_note
        + ". Command may be unreachable (workspace, joint limits, or singularity)."
    )


def warn_if_ik_bootstrap_not_converged(
    pos_err_m: float,
    rot_err_rad: float,
    *,
    pos_tol_m: float = IK_BOOTSTRAP_POS_TOL_M,
    rot_tol_rad: float = IK_BOOTSTRAP_ROT_TOL_RAD,
    target_pos: tuple[float, float, float] | None = None,
) -> bool:
    """Warn when TCP IK bootstrap misses tolerance. Returns True if within tolerance."""
    pos_ok = pos_err_m < pos_tol_m
    rot_ok = rot_err_rad < rot_tol_rad
    if pos_ok and rot_ok:
        return True

    parts: list[str] = []
    if not pos_ok:
        parts.append(f"position error {pos_err_m:.4f} m (tol {pos_tol_m:.4f} m)")
    if not rot_ok:
        parts.append(f"orientation error {rot_err_rad:.4f} rad (tol {rot_tol_rad:.4f} rad)")
    target_note = ""
    if target_pos is not None:
        target_note = (
            f"; target proxy position ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
        )
    warnings.warn(
        "FR3 TCP IK bootstrap did not converge: "
        + ", ".join(parts)
        + target_note
        + ". Consider increasing ik_bootstrap_iterations or adjusting base_pos / arm placement.",
        IKBootstrapConvergenceWarning,
        stacklevel=3,
    )
    return False


def raise_if_ik_bootstrap_not_converged(
    pos_err_m: float,
    rot_err_rad: float,
    *,
    pos_tol_m: float = IK_BOOTSTRAP_POS_TOL_M,
    rot_tol_rad: float = IK_BOOTSTRAP_ROT_TOL_RAD,
    target_pos: tuple[float, float, float] | None = None,
) -> None:
    """Raise :class:`IKBootstrapConvergenceError` when TCP IK bootstrap misses tolerance."""
    pos_ok = pos_err_m < pos_tol_m
    rot_ok = rot_err_rad < rot_tol_rad
    if pos_ok and rot_ok:
        return

    parts: list[str] = []
    if not pos_ok:
        parts.append(f"position error {pos_err_m:.4f} m (tol {pos_tol_m:.4f} m)")
    if not rot_ok:
        parts.append(f"orientation error {rot_err_rad:.4f} rad (tol {rot_tol_rad:.4f} rad)")
    target_note = ""
    if target_pos is not None:
        target_note = (
            f"; target proxy position ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
        )
    raise IKBootstrapConvergenceError(
        "FR3 TCP IK bootstrap did not converge: "
        + ", ".join(parts)
        + target_note
        + ". Adjust cable layout, fixture robot_base_pos, or ik_bootstrap_iterations "
        "so the proxy lies in the arm workspace from the specified FR3 base."
    )


def warn_ik_bootstrap_for_fr3_scene(scene: Any) -> bool:
    """Re-check nominal-column TCP vs proxy after build; warn if IK bootstrap missed tolerance."""
    robot_model = getattr(scene, "robot_model", None)
    robot_state_0 = getattr(scene, "robot_state_0", None)
    tcp_body_index = int(getattr(scene, "tcp_body_index", -1))
    if robot_model is None or robot_state_0 is None or tcp_body_index < 0:
        return True

    cable = scene.cable
    cable_view = (
        cable.as_single_instance_coupled(getattr(scene, "nominal_index", 0))
        if hasattr(cable, "as_single_instance_coupled")
        else cable
    )
    proxy_body = cable_view.gripper_proxy_body
    pos_err, rot_err = tcp_proxy_pose_errors(
        robot_state_0.body_q.numpy(),
        cable_view.state_0.body_q.numpy(),
        tcp_body_index=tcp_body_index,
        proxy_body_index=proxy_body,
    )
    pq = cable_view.state_0.body_q.numpy().reshape(-1, 7)[proxy_body]
    return warn_if_ik_bootstrap_not_converged(
        pos_err,
        rot_err,
        target_pos=(float(pq[0]), float(pq[1]), float(pq[2])),
    )


def placement_xform_for_proxy(
    proxy_body_q7: Any,
    *,
    vertical_reach_m: float = 0.80,
) -> wp.transform:
    """World transform to park the FR3 root so ``tcp`` can reach a high gripper proxy."""
    p = np.asarray(proxy_body_q7, dtype=np.float64).reshape(7)
    base_z = float(p[2]) - vertical_reach_m
    return wp.transform(wp.vec3(float(p[0]), float(p[1]), base_z), wp.quat_identity())




def ik_bootstrap_joint_q_candidates(
    robot_model: newton.Model,
    *,
    max_seeds: int = 4,
) -> list[np.ndarray]:
    """Return deterministic initial ``joint_q`` configs for TCP IK bootstrap retries."""
    if max_seeds < 1:
        raise ValueError(f"max_seeds must be >= 1, got {max_seeds}")

    jc = int(robot_model.joint_coord_count)
    lower = robot_model.joint_limit_lower.numpy().reshape(-1)[:jc].astype(np.float64)
    upper = robot_model.joint_limit_upper.numpy().reshape(-1)[:jc].astype(np.float64)
    default = robot_model.joint_q.numpy().reshape(-1)[:jc].astype(np.float64)
    span = upper - lower

    recipes: list[np.ndarray] = [default.copy()]
    recipes.extend(lower + frac * span for frac in _IK_BOOTSTRAP_JOINT_Q_SEED_FRACS)

    out: list[np.ndarray] = []
    for cand in recipes:
        if len(out) >= max_seeds:
            break
        arr = np.asarray(cand, dtype=np.float64).reshape(-1)
        if any(np.allclose(arr, prev, atol=1e-4, rtol=0.0) for prev in out):
            continue
        out.append(arr)
    return out


def _ik_bootstrap_within_tolerance(
    pos_err: float,
    rot_err: float,
    *,
    fix_to_apple: bool,
) -> bool:
    if fix_to_apple:
        rot_err = 0.0
    return pos_err < IK_BOOTSTRAP_POS_TOL_M and rot_err < IK_BOOTSTRAP_ROT_TOL_RAD


def _ik_bootstrap_error_score(pos_err: float, rot_err: float) -> float:
    return pos_err / IK_BOOTSTRAP_POS_TOL_M + rot_err / IK_BOOTSTRAP_ROT_TOL_RAD


def root_world_translation_for_proxy(
    proxy_body_q7: Any,
    *,
    vertical_reach_m: float = 0.80,
) -> np.ndarray:
    """World translation applied to the FR3 USD root for proxy reach placement."""
    tf = placement_xform_for_proxy(proxy_body_q7, vertical_reach_m=vertical_reach_m)
    t = wp.transform_get_translation(tf)
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def bootstrap_tcp_ik_from_proxy(
    cable_scene: Any,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
    *,
    ik_iterations: int = 48,
    max_joint_q_seeds: int = 4,
    raise_on_failure: bool = True,
) -> None:
    """Place the arm so ``tcp`` matches the cable gripper proxy pose (position + orientation).

    When the current ``joint_q`` does not converge, retries IK from deterministic alternate
    initial configurations (limit midpoints and span fractions) and keeps the best result.
    """
    import newton.ik as ik

    proxy_body = cable_scene.gripper_proxy_body
    fix_to_apple = getattr(cable_scene, "gripper_proxy_apple_joint", None) is not None
    bq = cable_scene.state_0.body_q.numpy().reshape(-1, 7)[proxy_body]
    target_pos = wp.vec3(float(bq[0]), float(bq[1]), float(bq[2]))
    target_rot = wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6]))
    target_tf = wp.transform(target_pos, target_rot)
    target_pos = wp.transform_get_translation(target_tf)
    target_rot = wp.transform_get_rotation(target_tf)
    dev = robot_model.device

    pos_obj = ik.IKObjectivePosition(
        link_index=tcp_body_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([target_pos], dtype=wp.vec3, device=dev),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=tcp_body_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array(
            [wp.vec4(target_rot[0], target_rot[1], target_rot[2], target_rot[3])],
            dtype=wp.vec4,
            device=dev,
        ),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=robot_model.joint_limit_lower,
        joint_limit_upper=robot_model.joint_limit_upper,
        weight=10.0,
    )

    joint_q_buf = robot_model.joint_q.reshape((1, int(robot_model.joint_coord_count)))
    objectives: list[Any] = [pos_obj, limits]
    objectives.insert(1, rot_obj)
    solver = ik.IKSolver(
        model=robot_model,
        n_problems=1,
        objectives=objectives,
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )

    seeds = ik_bootstrap_joint_q_candidates(robot_model, max_seeds=max_joint_q_seeds)
    best_jq: np.ndarray | None = None
    best_pos_err = float("inf")
    best_rot_err = float("inf")
    best_score = float("inf")
    best_seed_idx = 0
    seeds_tried = 0

    for seed_idx, seed_jq in enumerate(seeds):
        seeds_tried = seed_idx + 1
        seed_arr = seed_jq.reshape(1, -1).astype(robot_model.joint_q.dtype, copy=False)
        joint_q_buf.assign(seed_arr)
        solver.step(joint_q_buf, joint_q_buf, iterations=ik_iterations)

        jq = joint_q_buf.numpy().reshape(-1).astype(robot_model.joint_q.dtype)
        jqd = np_zeros_like_joint_qd(robot_model)
        robot_model.joint_q.assign(jq)
        robot_model.joint_qd.assign(jqd)
        robot_state_0.joint_q.assign(jq)
        robot_state_0.joint_qd.assign(jqd)
        newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_0)

        pos_err, rot_err = tcp_proxy_pose_errors(
            robot_state_0.body_q.numpy(),
            cable_scene.state_0.body_q.numpy(),
            tcp_body_index=tcp_body_index,
            proxy_body_index=proxy_body,
        )
        rot_err_for_check = 0.0 if fix_to_apple else rot_err
        score = _ik_bootstrap_error_score(pos_err, rot_err_for_check)
        if score < best_score:
            best_jq = jq.copy()
            best_pos_err = pos_err
            best_rot_err = rot_err
            best_score = score
            best_seed_idx = seed_idx
        if _ik_bootstrap_within_tolerance(pos_err, rot_err, fix_to_apple=fix_to_apple):
            break

    assert best_jq is not None
    jqd = np_zeros_like_joint_qd(robot_model)
    robot_model.joint_q.assign(best_jq)
    robot_model.joint_qd.assign(jqd)
    robot_state_0.joint_q.assign(best_jq)
    robot_state_0.joint_qd.assign(jqd)
    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_0)

    proxy_q7 = cable_scene.state_0.body_q.numpy().reshape(-1, 7)[proxy_body]
    tcp_q7 = robot_state_0.body_q.numpy().reshape(-1, 7)[tcp_body_index]
    pos_err = best_pos_err
    rot_err = best_rot_err
    if fix_to_apple:
        rot_err = 0.0
    rot_note = " (rotation not tracked; fix_to_apple)" if fix_to_apple else ""
    seed_note = ""
    if seeds_tried > 1:
        seed_note = f" (joint_q seed {best_seed_idx}, tried {seeds_tried}/{len(seeds)})"
    print(
        "FR3 TCP IK bootstrap pose:\n"
        f"  cable proxy  pos=({proxy_q7[0]:.4f}, {proxy_q7[1]:.4f}, {proxy_q7[2]:.4f}) "
        f"quat=({proxy_q7[3]:.4f}, {proxy_q7[4]:.4f}, {proxy_q7[5]:.4f}, {proxy_q7[6]:.4f})\n"
        f"  tcp achieved pos=({tcp_q7[0]:.4f}, {tcp_q7[1]:.4f}, {tcp_q7[2]:.4f}) "
        f"quat=({tcp_q7[3]:.4f}, {tcp_q7[4]:.4f}, {tcp_q7[5]:.4f}, {tcp_q7[6]:.4f})\n"
        f"  error pos={pos_err:.4f} m rot={rot_err:.4f} rad{rot_note}{seed_note}"
    )
    target_xyz = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
    if raise_on_failure:
        raise_if_ik_bootstrap_not_converged(pos_err, rot_err, target_pos=target_xyz)
    else:
        warn_if_ik_bootstrap_not_converged(pos_err, rot_err, target_pos=target_xyz)


