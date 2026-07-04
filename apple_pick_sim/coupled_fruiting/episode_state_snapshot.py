"""GPU-resident episode baseline capture/restore for batched coupled sim."""

from __future__ import annotations

import dataclasses
from typing import Any

import warp as wp

from apple_pick_sim.coupled_fruiting.proxy_coupling import sync_solver_body_q_prev_from_state
from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance_batched import (
    Fr3BatchedEEImpedanceController,
)


def _clone_wp_array(arr: wp.array | None) -> wp.array | None:
    if arr is None:
        return None
    out = wp.empty_like(arr)
    wp.copy(out, arr)
    return out


@dataclasses.dataclass
class EpisodeStateSnapshot:
    """Post-weld episode baseline for cheap ``reset()`` without rebuild."""

    robot_body_q: wp.array
    robot_body_qd: wp.array
    robot_joint_q: wp.array
    robot_joint_qd: wp.array
    model_joint_q: wp.array
    model_joint_qd: wp.array
    cable_body_q_0: wp.array
    cable_body_qd_0: wp.array
    cable_body_q_1: wp.array
    cable_body_qd_1: wp.array
    vic_target_pos: wp.array | None = None
    vic_target_rot: wp.array | None = None
    vic_lin_vels: wp.array | None = None
    vic_ang_vels: wp.array | None = None
    vic_default_dof_pos_batched: wp.array | None = None

    @classmethod
    def capture(cls, sim: Any) -> EpisodeStateSnapshot:
        """Capture device state immediately after build / first init."""
        scene = sim.scene
        cable = scene.cable
        rs0 = scene.robot_state_0
        if rs0 is None or cable is None:
            raise RuntimeError("episode snapshot requires robot and cable state")

        vic_pos = vic_rot = vic_lin = vic_ang = None
        ee_ctrl = getattr(sim, "_ee_ctrl", None)
        if isinstance(ee_ctrl, Fr3BatchedEEImpedanceController):
            vic_pos = _clone_wp_array(ee_ctrl._target_pos_wp)
            vic_rot = _clone_wp_array(ee_ctrl._target_rot_wp)
            vic_lin = _clone_wp_array(ee_ctrl._lin_vels_wp)
            vic_ang = _clone_wp_array(ee_ctrl._ang_vels_wp)

        vic_default = getattr(scene, "vic_jt_default_dof_pos_batched", None)
        return cls(
            robot_body_q=_clone_wp_array(rs0.body_q),
            robot_body_qd=_clone_wp_array(rs0.body_qd),
            robot_joint_q=_clone_wp_array(rs0.joint_q),
            robot_joint_qd=_clone_wp_array(rs0.joint_qd),
            model_joint_q=_clone_wp_array(scene.robot_model.joint_q),
            model_joint_qd=_clone_wp_array(scene.robot_model.joint_qd),
            cable_body_q_0=_clone_wp_array(cable.state_0.body_q),
            cable_body_qd_0=_clone_wp_array(cable.state_0.body_qd),
            cable_body_q_1=_clone_wp_array(cable.state_1.body_q),
            cable_body_qd_1=_clone_wp_array(cable.state_1.body_qd),
            vic_target_pos=vic_pos,
            vic_target_rot=vic_rot,
            vic_lin_vels=vic_lin,
            vic_ang_vels=vic_ang,
            vic_default_dof_pos_batched=_clone_wp_array(vic_default),
        )

    def restore(self, sim: Any) -> None:
        """Restore physics and VIC targets to the captured episode baseline."""
        scene = sim.scene
        cable = scene.cable
        rs0 = scene.robot_state_0
        if rs0 is None or cable is None:
            raise RuntimeError("episode snapshot restore requires robot and cable state")

        wp.copy(rs0.body_q, self.robot_body_q)
        wp.copy(rs0.body_qd, self.robot_body_qd)
        wp.copy(rs0.joint_q, self.robot_joint_q)
        wp.copy(rs0.joint_qd, self.robot_joint_qd)
        wp.copy(scene.robot_model.joint_q, self.model_joint_q)
        wp.copy(scene.robot_model.joint_qd, self.model_joint_qd)

        wp.copy(cable.state_0.body_q, self.cable_body_q_0)
        wp.copy(cable.state_0.body_qd, self.cable_body_qd_0)
        wp.copy(cable.state_1.body_q, self.cable_body_q_1)
        wp.copy(cable.state_1.body_qd, self.cable_body_qd_1)

        init_robot_mujoco_step_buffers(scene)
        fr3_robot.hold_mujoco_actuator_targets_at_state(
            scene.robot_model, scene.robot_state_0, scene.robot_control
        )
        sync_solver_body_q_prev_from_state(cable, cable.state_0.body_q)

        if self.vic_default_dof_pos_batched is not None and getattr(
            scene, "vic_jt_default_dof_pos_batched", None
        ) is not None:
            wp.copy(scene.vic_jt_default_dof_pos_batched, self.vic_default_dof_pos_batched)

        ee_ctrl = getattr(sim, "_ee_ctrl", None)
        if isinstance(ee_ctrl, Fr3BatchedEEImpedanceController):
            if self.vic_target_pos is not None:
                wp.copy(ee_ctrl._target_pos_wp, self.vic_target_pos)
            if self.vic_target_rot is not None:
                wp.copy(ee_ctrl._target_rot_wp, self.vic_target_rot)
            if self.vic_lin_vels is not None:
                wp.copy(ee_ctrl._lin_vels_wp, self.vic_lin_vels)
            if self.vic_ang_vels is not None:
                wp.copy(ee_ctrl._ang_vels_wp, self.vic_ang_vels)
            ee_ctrl._sync_target_tf_from_device()
            ee_ctrl.stage_targets_to_scene(scene)

        scene.vic_target_twist = fr3_robot.EEVelocity()
        if scene.proxy_forces is not None:
            scene.proxy_forces.zero_()
        if scene.coupling_forces_cache is not None:
            scene.coupling_forces_cache.zero_()
