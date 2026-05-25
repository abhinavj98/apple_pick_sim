"""FR3 root placement and IK bootstrap from cable proxy."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.fruiting_system import CoupledCableScene
from apple_pick_sim.robot.fr3_robot.setup import np_zeros_like_joint_qd

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
    target_tf = wp.transform(
        wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )
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


