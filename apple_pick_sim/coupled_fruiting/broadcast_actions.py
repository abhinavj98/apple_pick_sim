"""Broadcast world-0 robot joint commands to all batched worlds."""

from __future__ import annotations

from typing import Any

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout


def broadcast_joint_q_from_world0(scene: Any, layout: BatchedEnvLayout) -> None:
    """Copy world-0 ``joint_q`` / ``joint_qd`` to every env on model and ``robot_state_0``."""
    if layout.num_envs < 2:
        return
    if scene.robot_model is None or scene.robot_state_0 is None:
        raise ValueError("broadcast_joint_q_from_world0 requires robot model and state")

    model = scene.robot_model
    w0_q = layout.joint_q_slice(0)
    w0_qd = layout.joint_qd_slice(0)

    for target in (model, scene.robot_state_0):
        jq = target.joint_q.numpy().copy()
        jqd = target.joint_qd.numpy().copy()
        ref_q = jq[w0_q].copy()
        ref_qd = jqd[w0_qd].copy()
        for w in range(1, layout.num_envs):
            jq[layout.joint_q_slice(w)] = ref_q
            jqd[layout.joint_qd_slice(w)] = ref_qd
        target.joint_q.assign(jq)
        target.joint_qd.assign(jqd)

    newton.eval_fk(
        model,
        scene.robot_state_0.joint_q,
        scene.robot_state_0.joint_qd,
        scene.robot_state_0,
    )
