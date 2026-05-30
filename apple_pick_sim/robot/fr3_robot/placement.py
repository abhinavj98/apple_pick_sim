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
IK_BOOTSTRAP_POS_TOL_M = 0.25
IK_BOOTSTRAP_ROT_TOL_RAD = 0.15


class IKBootstrapConvergenceWarning(UserWarning):
    """FR3 TCP IK bootstrap missed position/orientation tolerance vs gripper proxy."""


def enable_ik_bootstrap_warnings_for_examples() -> None:
    """Show :class:`IKBootstrapConvergenceWarning` on every occurrence (CLI / examples)."""
    warnings.simplefilter("always", IKBootstrapConvergenceWarning)


def tcp_proxy_pose_errors(
    robot_body_q: np.ndarray,
    proxy_body_q: np.ndarray,
    *,
    tcp_body_index: int,
    proxy_body_index: int,
) -> tuple[float, float]:
    """Return TCP position error [m] and orientation error [rad] vs the proxy."""
    rq = robot_body_q.reshape(-1, 7)[tcp_body_index].astype(np.float64)
    pq = proxy_body_q.reshape(-1, 7)[proxy_body_index].astype(np.float64)
    pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
    qa = rq[3:7] / (np.linalg.norm(rq[3:7]) + 1e-12)
    qb = pq[3:7] / (np.linalg.norm(pq[3:7]) + 1e-12)
    rot_err = 2.0 * float(np.arccos(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0)))
    return pos_err, rot_err


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
    vertical_reach_m: float = 0.85,
) -> wp.transform:
    """World transform to park the FR3 root so ``tcp`` can reach a high gripper proxy."""
    import numpy as np

    p = np.asarray(proxy_body_q7, dtype=np.float64).reshape(7)
    base_z = max(0.0, float(p[2]) - vertical_reach_m)
    return wp.transform(wp.vec3(float(p[0]), float(p[1]), base_z), wp.quat_identity())


def bootstrap_tcp_ik_from_proxy(
    cable_scene: Any,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
    *,
    ik_iterations: int = 48,
) -> None:
    """Place the arm so ``tcp`` matches the cable gripper proxy pose (position + orientation)."""
    import newton.ik as ik

    proxy_body = cable_scene.gripper_proxy_body
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

    joint_q = robot_model.joint_q.reshape((1, int(robot_model.joint_coord_count)))
    solver = ik.IKSolver(
        model=robot_model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(joint_q, joint_q, iterations=ik_iterations)

    jq = joint_q.numpy().reshape(-1).astype(robot_model.joint_q.dtype)
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
    warn_if_ik_bootstrap_not_converged(
        pos_err,
        rot_err,
        target_pos=(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
    )


