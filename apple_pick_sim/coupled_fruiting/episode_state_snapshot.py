"""GPU-resident episode baseline capture/restore for batched coupled sim."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import warp as wp

from apple_pick_sim.coupled_fruiting.proxy_coupling import sync_solver_body_q_prev_from_state
from apple_pick_sim.coupled_fruiting.scene import (
    init_robot_mujoco_step_buffers,
    seed_lagged_coupling_from_rest_harvest,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance_batched import (
    Fr3BatchedEEImpedanceController,
)

_DEFAULT_COUPLING_SEED_DT = 1.0 / 1800.0

_AVBD_ARRAY_ATTRS = (
    "joint_lambda_lin",
    "joint_lambda_ang",
    "joint_penalty_k",
    "joint_C0_lin",
    "joint_C0_ang",
    "body_q_prev",
)
_AVBD_DAHL_ATTRS = (
    "joint_sigma_prev",
    "joint_kappa_prev",
    "joint_dkappa_prev",
)


def _clone_wp_array(arr: wp.array | None) -> wp.array | None:
    if arr is None:
        return None
    out = wp.empty_like(arr)
    wp.copy(out, arr)
    return out


def _capture_solver_avbd(solver: Any) -> dict[str, wp.array | None]:
    out: dict[str, wp.array | None] = {}
    for name in _AVBD_ARRAY_ATTRS + _AVBD_DAHL_ATTRS:
        arr = getattr(solver, name, None)
        out[name] = _clone_wp_array(arr) if arr is not None else None
    return out


def _restore_solver_avbd(solver: Any, snap: dict[str, wp.array | None]) -> None:
    for name, cloned in snap.items():
        if cloned is None:
            continue
        target = getattr(solver, name, None)
        if target is None:
            continue
        wp.copy(target, cloned)


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
    # Cable SolverVBD AVBD warm-start / pretension state (optional for old npz).
    joint_lambda_lin: wp.array | None = None
    joint_lambda_ang: wp.array | None = None
    joint_penalty_k: wp.array | None = None
    joint_C0_lin: wp.array | None = None
    joint_C0_ang: wp.array | None = None
    solver_body_q_prev: wp.array | None = None
    joint_sigma_prev: wp.array | None = None
    joint_kappa_prev: wp.array | None = None
    joint_dkappa_prev: wp.array | None = None
    # Cable model revolute drive stiffness (T-roll uses min(penalty_k, target_ke)).
    joint_target_ke: wp.array | None = None
    joint_target_kd: wp.array | None = None

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
        avbd = _capture_solver_avbd(cable.solver)
        model = cable.model
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
            joint_lambda_lin=avbd["joint_lambda_lin"],
            joint_lambda_ang=avbd["joint_lambda_ang"],
            joint_penalty_k=avbd["joint_penalty_k"],
            joint_C0_lin=avbd["joint_C0_lin"],
            joint_C0_ang=avbd["joint_C0_ang"],
            solver_body_q_prev=avbd["body_q_prev"],
            joint_sigma_prev=avbd["joint_sigma_prev"],
            joint_kappa_prev=avbd["joint_kappa_prev"],
            joint_dkappa_prev=avbd["joint_dkappa_prev"],
            joint_target_ke=_clone_wp_array(getattr(model, "joint_target_ke", None)),
            joint_target_kd=_clone_wp_array(getattr(model, "joint_target_kd", None)),
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

        avbd_snap = {
            "joint_lambda_lin": self.joint_lambda_lin,
            "joint_lambda_ang": self.joint_lambda_ang,
            "joint_penalty_k": self.joint_penalty_k,
            "joint_C0_lin": self.joint_C0_lin,
            "joint_C0_ang": self.joint_C0_ang,
            "body_q_prev": self.solver_body_q_prev,
            "joint_sigma_prev": self.joint_sigma_prev,
            "joint_kappa_prev": self.joint_kappa_prev,
            "joint_dkappa_prev": self.joint_dkappa_prev,
        }
        if self.joint_lambda_lin is None:
            warnings.warn(
                "episode snapshot lacks AVBD lambda state; plant pretension may be lost on reset",
                UserWarning,
                stacklevel=2,
            )
            sync_solver_body_q_prev_from_state(cable, cable.state_0.body_q)
        else:
            _restore_solver_avbd(cable.solver, avbd_snap)

        if self.joint_target_ke is not None and hasattr(cable.model, "joint_target_ke"):
            wp.copy(cable.model.joint_target_ke, self.joint_target_ke)
        if self.joint_target_kd is not None and hasattr(cable.model, "joint_target_kd"):
            wp.copy(cable.model.joint_target_kd, self.joint_target_kd)

        init_robot_mujoco_step_buffers(scene)
        fr3_robot.hold_mujoco_actuator_targets_at_state(
            scene.robot_model, scene.robot_state_0, scene.robot_control
        )

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
        cfg = getattr(sim, "_config", None)
        if cfg is not None:
            seed_dt = float(cfg.runtime.sub_dt)
        else:
            seed_dt = _DEFAULT_COUPLING_SEED_DT
        seed_lagged_coupling_from_rest_harvest(scene, seed_dt)
