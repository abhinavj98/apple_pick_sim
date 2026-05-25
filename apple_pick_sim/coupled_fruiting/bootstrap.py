"""Align robot TCP with cable gripper proxy at scene construction."""

from __future__ import annotations

from typing import Any

import numpy as np

import newton

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
) -> None:
    """IK placement for FR3: match ``tcp`` to the cable gripper proxy."""
    fr3_robot.bootstrap_tcp_ik_from_proxy(
        cable_scene,
        robot_model,
        tcp_body_index,
        robot_state_0,
        ik_iterations=ik_iterations,
    )
