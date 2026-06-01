"""Utilities for "settle freely, then weld" initialization (fix_to_apple quiet start).

Newton/VBD model topology is fixed after :class:`newton.ModelBuilder` is finalized, so
the proxy↔apple FIXED joint (``joint_apple_gripper_proxy``) cannot be toggled on/off at
runtime. This module provides a robust two-build workflow:

1) Build a free-apple scene (``fix_to_apple=False``) and run VBD substeps to settle.
2) Build a welded scene (``fix_to_apple=True``) and seed its cable state from the
   settled configuration so the welded constraint starts near zero violation.
3) Re-run FR3 IK bootstrap at the scene's fixed robot base (from fixture
   ``robot_base_pos`` or builder placement). Raise if the settled proxy is unreachable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.proxy_coupling import align_proxy_body_q_prev_for_vbd


def _proxy_world_pose_from_apple(
    apple_body_q7: np.ndarray,
    offset_7d: tuple | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame proxy (position, quaternion) from apple pose and 7D grasp offset.

    Returns ``(pos_f32[3], quat_f32[4])``.
    """
    apple_tf = wp.transform(
        wp.vec3(float(apple_body_q7[0]), float(apple_body_q7[1]), float(apple_body_q7[2])),
        wp.quat(float(apple_body_q7[3]), float(apple_body_q7[4]), float(apple_body_q7[5]), float(apple_body_q7[6])),
    )
    offset_tf = wp.transform(
        wp.vec3(float(offset_7d[0]), float(offset_7d[1]), float(offset_7d[2])),
        wp.quat(float(offset_7d[3]), float(offset_7d[4]), float(offset_7d[5]), float(offset_7d[6])),
    )
    # X_proxy = X_apple * X_offset
    proxy_tf = wp.transform_multiply(apple_tf, offset_tf)
    proxy_pos = wp.transform_get_translation(proxy_tf)
    proxy_rot = wp.transform_get_rotation(proxy_tf)
    pos = np.array([proxy_pos[0], proxy_pos[1], proxy_pos[2]], dtype=np.float32)
    quat = np.array([proxy_rot[0], proxy_rot[1], proxy_rot[2], proxy_rot[3]], dtype=np.float32)
    print(f"Apple pos: {apple_body_q7[:3]}, Apple quat: {apple_body_q7[3:]}, Offset: {offset_7d[:3]}")
    return pos, quat


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


def _nominal_cable_view(scene: Any) -> Any:
    cable = scene.cable
    if hasattr(cable, "as_single_instance_coupled"):
        idx = int(getattr(scene, "nominal_index", 0))
        return cable.as_single_instance_coupled(idx)
    return cable


def _bootstrap_tcp_at_fixed_origin(
    scene: Any,
    *,
    ik_iterations: int = 96,
) -> None:
    """Align TCP to the seeded cable proxy using the scene's fixed FR3 base placement."""
    if scene.robot_model is None or scene.robot_state_0 is None or scene.mj_solver is None:
        return

    import newton

    from apple_pick_sim.coupled_fruiting.bootstrap import bootstrap_articulated_tcp_from_proxy
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

    cable = _nominal_cable_view(scene)
    root = np.asarray(getattr(scene, "fr3_root_world_pos", (0.0, 0.0, 0.0)), dtype=np.float64)
    root_xyz = (float(root[0]), float(root[1]), float(root[2]))

    try:
        bootstrap_articulated_tcp_from_proxy(
            cable,
            scene.robot_model,
            scene.tcp_body_index,
            scene.robot_state_0,
            ik_iterations=ik_iterations,
            raise_on_failure=True,
        )
    except IKBootstrapConvergenceError as exc:
        raise IKBootstrapConvergenceError(
            "Settled gripper proxy is unreachable from the specified FR3 base at "
            f"({root_xyz[0]:.3f}, {root_xyz[1]:.3f}, {root_xyz[2]:.3f}): {exc}"
        ) from exc

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
    fr3_robot.init_mujoco_actuator_targets_from_model(
        scene.robot_model, scene.robot_control
    )
    if scene.proxy_forces is not None:
        scene.proxy_forces.zero_()
    if scene.coupling_forces_cache is not None:
        scene.coupling_forces_cache.zero_()


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

    off = np.zeros(3, dtype=np.float32)
    bq_w = cable_w.state_0.body_q.numpy().reshape(-1, 7).copy()
    bqd_w = cable_w.state_0.body_qd.numpy().reshape(-1, 6).copy()
    print(f"Offset: {off}, Apple: {bq_w[apple]}, Proxy: {bq_w[proxy]}")
    # Enforce proxy placement at the welded offset (7D transform from apple frame).
    proxy_pos, proxy_quat = _proxy_world_pose_from_apple(bq_w[apple], offset)
    bq_w[proxy, :3] = proxy_pos
    bq_w[proxy, 3:] = proxy_quat

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

    _bootstrap_tcp_at_fixed_origin(welded_scene)


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
        proxy_pos, proxy_quat = _proxy_world_pose_from_apple(bq_w[apple], offset)
        bq_w[proxy, :3] = proxy_pos
        bq_w[proxy, 3:] = proxy_quat
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

    _bootstrap_tcp_at_fixed_origin(welded_scene)
