"""Utilities for "settle freely, then weld" initialization (fix_to_apple quiet start).

Newton/VBD model topology is fixed after :class:`newton.ModelBuilder` is finalized, so
the proxy↔apple FIXED joint (``joint_apple_gripper_proxy``) cannot be toggled on/off at
runtime. This module provides a robust two-build workflow:

1) Build a free-apple scene (``fix_to_apple=False``) and run VBD substeps to settle.
2) Build a welded scene (``fix_to_apple=True``) and seed its cable state from the
   settled configuration so the welded constraint starts near zero violation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.proxy_coupling import align_proxy_body_q_prev_for_vbd


def _proxy_world_position_from_apple(
    apple_body_q7: np.ndarray,
    offset_in_apple_frame: np.ndarray,
) -> np.ndarray:
    """World-frame proxy COM from apple pose and apple-frame grasp offset."""
    p = apple_body_q7[:3].astype(np.float64)
    q = wp.quat(
        float(apple_body_q7[3]),
        float(apple_body_q7[4]),
        float(apple_body_q7[5]),
        float(apple_body_q7[6]),
    )
    off = wp.vec3(
        float(offset_in_apple_frame[0]),
        float(offset_in_apple_frame[1]),
        float(offset_in_apple_frame[2]),
    )
    return (p + np.asarray(wp.quat_rotate(q, off), dtype=np.float64)).astype(np.float32)


def settle_vbd_substeps(scene: Any, *, substeps: int, dt: float) -> None:
    """Advance ``scene`` by ``substeps`` VBD-only substeps.

    Args:
        scene: A :class:`~apple_pick_sim.coupled_fruiting.scene.CoupledFruitingScene`
            or any object exposing ``vbd_substep(dt)``.
        substeps: Number of VBD substeps to advance.
        dt: Step size [s] per VBD substep.
    """
    n = int(substeps)
    if n <= 0:
        return
    h = float(dt)
    for _ in range(n):
        scene.vbd_substep(h)


def seed_fix_to_apple_from_settled(
    *,
    welded_scene: Any,
    settled_scene: Any,
    quiet_apple_proxy: bool = True,
) -> None:
    """Seed a welded (``fix_to_apple=True``) scene from a settled free-apple scene.

    This copies the *entire* cable model state (body poses + twists) from
    ``settled_scene`` into ``welded_scene`` and then enforces that the proxy starts
    at the configured grasp offset from the apple so the proxy↔apple fixed joint has
    minimal initial violation.

    Args:
        welded_scene: A coupled scene built with ``GripperProxyConfig(fix_to_apple=True)``.
        settled_scene: A coupled scene built with ``GripperProxyConfig(fix_to_apple=False)``
            that has already been advanced to a quasi-static configuration.
        quiet_apple_proxy: If True, zero the apple and proxy twists after seeding.
    """
    cable_w = welded_scene.cable
    cable_s = settled_scene.cable

    # Copy state for all bodies that exist in both models (body_count is expected to match).
    bq = cable_s.state_0.body_q.numpy().reshape(-1, 7)
    bqd = cable_s.state_0.body_qd.numpy().reshape(-1, 6)
    cable_w.state_0.body_q.assign(bq)
    cable_w.state_0.body_qd.assign(bqd)

    apple = cable_w.apple_body
    proxy = cable_w.gripper_proxy_body
    offset = cable_w.gripper_proxy_offset_in_apple_frame
    if apple is None or offset is None:
        return

    off = np.asarray(offset, dtype=np.float32).reshape(3)
    bq_w = cable_w.state_0.body_q.numpy().reshape(-1, 7).copy()
    bqd_w = cable_w.state_0.body_qd.numpy().reshape(-1, 6).copy()

    # Enforce proxy placement at the welded offset (apple-frame vector, world rotated).
    bq_w[proxy, :3] = _proxy_world_position_from_apple(bq_w[apple], off)
    bq_w[proxy, 3:] = bq_w[apple, 3:]

    if quiet_apple_proxy:
        bqd_w[apple] = 0.0
        bqd_w[proxy] = 0.0

    cable_w.state_0.body_q.assign(bq_w.reshape(-1, 7))
    cable_w.state_0.body_qd.assign(bqd_w.reshape(-1, 6))

    # Do not call eval_fk() here: the welded and settled models do not share joint-space
    # coordinates (free proxy vs fixed proxy↔apple), so FK from welded joint_q would
    # overwrite the seeded settled body poses.
    # Align the VBD previous-pose buffer so the next step doesn't see an artificial jump.
    body_count = int(cable_w.model.body_count)
    align_proxy_body_q_prev_for_vbd(cable_w, tuple(range(body_count)))

    rebootstrap_robot_from_cable_proxy(welded_scene)


def seed_mega_fix_to_apple_from_settled(
    *,
    welded_scene: Any,
    settled_scene: Any,
    quiet_apple_proxy: bool = True,
) -> None:
    """Seed welded mega plant from settled mega (``fix_to_apple`` two-build workflow)."""
    cable_w = welded_scene.cable
    cable_s = settled_scene.cable
    if int(cable_w.model.body_count) != int(cable_s.model.body_count):
        raise ValueError(
            f"mega body_count mismatch: welded={cable_w.model.body_count} "
            f"settled={cable_s.model.body_count}"
        )

    cable_w.state_0.body_q.assign(cable_s.state_0.body_q.numpy())
    cable_w.state_0.body_qd.assign(cable_s.state_0.body_qd.numpy())

    bq_w = cable_w.state_0.body_q.numpy().reshape(-1, 7).copy()
    bqd_w = cable_w.state_0.body_qd.numpy().reshape(-1, 6).copy()

    for inst in cable_w.instances:
        apple = inst.apple_body
        proxy = inst.gripper_proxy_body
        offset = inst.gripper_proxy_offset_in_apple_frame
        if apple is None or offset is None:
            continue
        off = np.asarray(offset, dtype=np.float32).reshape(3)
        bq_w[proxy, :3] = _proxy_world_position_from_apple(bq_w[apple], off)
        bq_w[proxy, 3:] = bq_w[apple, 3:]
        if quiet_apple_proxy:
            bqd_w[apple] = 0.0
            bqd_w[proxy] = 0.0

    cable_w.state_0.body_q.assign(bq_w.reshape(-1, 7))
    cable_w.state_0.body_qd.assign(bqd_w.reshape(-1, 6))

    align_ids: list[int] = []
    for inst in cable_w.instances:
        align_ids.append(inst.gripper_proxy_body)
        if inst.apple_body is not None:
            align_ids.append(inst.apple_body)
    align_proxy_body_q_prev_for_vbd(cable_w, tuple(align_ids))

    rebootstrap_robot_from_cable_proxy(welded_scene)


def rebootstrap_robot_from_cable_proxy(
    scene: Any,
    *,
    ik_iterations: int = 96,
) -> None:
    """Re-run FR3 IK bootstrap so the TCP matches the (post-seed) cable proxy pose."""
    if scene.robot_model is None or scene.robot_state_0 is None or scene.mj_solver is None:
        return

    import newton

    from apple_pick_sim.coupled_fruiting.bootstrap import bootstrap_articulated_tcp_from_proxy
    from apple_pick_sim.robot import fr3_robot

    def _cable_for_bootstrap():
        cable = scene.cable
        if hasattr(cable, "as_single_instance_coupled"):
            idx = int(getattr(scene, "nominal_index", 0))
            return cable.as_single_instance_coupled(idx)
        return cable

    def _run_bootstrap(iterations: int) -> None:
        bootstrap_articulated_tcp_from_proxy(
            _cable_for_bootstrap(),
            scene.robot_model,
            scene.tcp_body_index,
            scene.robot_state_0,
            ik_iterations=iterations,
        )
        scene.robot_state_1.joint_q.assign(scene.robot_model.joint_q)
        scene.robot_state_1.joint_qd.assign(scene.robot_model.joint_qd)
        newton.eval_fk(
            scene.robot_model,
            scene.robot_model.joint_q,
            scene.robot_model.joint_qd,
            scene.robot_state_1,
        )
        scene.mj_solver._update_mjc_data(
            scene.mj_solver.mj_data, scene.robot_model, scene.robot_state_0
        )

    def _tcp_proxy_gap_m() -> float:
        cable = _cable_for_bootstrap()
        tcp = int(scene.tcp_body_index)
        proxy = int(cable.gripper_proxy_body)
        tcp_pos = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3]
        proxy_pos = cable.state_0.body_q.numpy().reshape(-1, 7)[proxy, :3]
        return float(np.linalg.norm(tcp_pos - proxy_pos))

    _run_bootstrap(int(ik_iterations))

    # IK can converge to a poor local minimum after a large settle displacement.
    # Retry once with a larger iteration budget (raises if still out of tolerance).
    if _tcp_proxy_gap_m() > 0.15:
        _run_bootstrap(max(int(ik_iterations) * 2, 192))

    fr3_robot.init_mujoco_actuator_targets_from_model(
        scene.robot_model, scene.robot_control
    )
    if scene.proxy_forces is not None:
        scene.proxy_forces.zero_()
    if scene.coupling_forces_cache is not None:
        scene.coupling_forces_cache.zero_()
