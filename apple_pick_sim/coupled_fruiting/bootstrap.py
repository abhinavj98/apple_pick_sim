"""Align robot TCP with cable gripper proxy at scene construction."""

from __future__ import annotations

from typing import Any

import numpy as np

import newton
import warp as wp

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.fruiting_system import CoupledCableScene


def bootstrap_tcp_joint_from_proxy(
    cable_scene: CoupledCableScene,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
) -> None:
    """Align robot generalized coords / FK state with the cable gripper proxy pose and twist."""
    del tcp_body_index
    proxy_body = cable_scene.gripper_proxy_body

    bq = cable_scene.state_0.body_q.numpy().reshape(-1, 7)[proxy_body].astype(np.float32)
    bqd = cable_scene.state_0.body_qd.numpy().reshape(-1, 6)[proxy_body].astype(np.float32)

    jq = robot_model.joint_q.numpy().astype(np.float32).copy()
    jqd = robot_model.joint_qd.numpy().astype(np.float32).copy()

    jc = int(robot_model.joint_coord_count)
    jd = int(robot_model.joint_dof_count)
    jq[:jc] = bq.flatten()[:jc]
    jqd[:jd] = bqd.flatten()[:jd]

    robot_model.joint_q.assign(jq)
    robot_model.joint_qd.assign(jqd)
    robot_state_0.joint_q.assign(jq)
    robot_state_0.joint_qd.assign(jqd)

    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_0)


def bootstrap_articulated_tcp_from_proxy(
    cable_scene: CoupledCableScene,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
    *,
    ik_iterations: int = 48,
    raise_on_failure: bool = True,
) -> None:
    """IK placement for FR3: match ``tcp`` to the cable gripper proxy."""
    fr3_robot.bootstrap_tcp_ik_from_proxy(
        cable_scene,
        robot_model,
        tcp_body_index,
        robot_state_0,
        ik_iterations=ik_iterations,
        raise_on_failure=raise_on_failure,
    )


def mirror_tcp_to_welded_cable_after_bootstrap(
    cable_scene: CoupledCableScene,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
    *,
    gravity: wp.vec3,
    dt: float = 1.0 / 600.0,
) -> None:
    """Align prescribed proxy/apple bodies with the post-bootstrap TCP (``fix_to_apple``)."""
    if cable_scene.gripper_proxy_apple_joint is None:
        return
    offset = cable_scene.gripper_proxy_offset_in_apple_frame
    if offset is None or cable_scene.apple_body is None:
        return

    from apple_pick_sim.coupled_fruiting.proxy_coupling import (
        align_proxy_body_q_prev_for_vbd,
        launch_mirror_robot_to_proxy_and_apple,
    )

    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_0)
    dev = str(robot_model.device)
    registry = cable_scene.proxy_registry(tcp_body_index)
    rid, pid = registry.ids_wp(dev)
    zf = wp.zeros(robot_model.body_count, dtype=wp.spatial_vector, device=robot_model.device)
    launch_mirror_robot_to_proxy_and_apple(
        robot_ids=rid,
        proxy_ids=pid,
        src_body_q=robot_state_0.body_q,
        src_body_qd=robot_state_0.body_qd,
        dst_body_q=cable_scene.state_0.body_q,
        dst_body_qd=cable_scene.state_0.body_qd,
        proxy_forces=zf,
        cable_model=cable_scene.model,
        gravity=gravity,
        dt=float(dt),
        apple_body_id=cable_scene.apple_body,
        proxy_offset_in_apple=wp.vec3(*offset),
        device=dev,
    )
    align_proxy_body_q_prev_for_vbd(
        cable_scene,
        (cable_scene.gripper_proxy_body, cable_scene.apple_body),
    )
