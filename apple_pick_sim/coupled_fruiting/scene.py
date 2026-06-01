"""CoupledFruitingScene and staggered substep orchestration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import warp as wp

import newton

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting.apply_wrench import _apply_spatial_wrench_to_body_f
from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim.fruiting_system import CoupledCableScene
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    align_proxy_body_q_prev_for_vbd,
    copy_cable_body_q_between_states,
    harvest_proxy_wrenches,
    harvest_stem_tension_for_tcp,
    launch_mirror_robot_to_proxy,
    launch_mirror_robot_to_proxy_and_apple,
    sync_solver_body_q_prev_from_state,
)

DEFAULT_STEM_COUPLING_GAIN: float = 1.0
DEFAULT_STEM_FORCE_CAP_N: float = 80.0
DEFAULT_STEM_TORQUE_CAP_NM: float = 20.0

DEFAULT_MUJOCO_SOLVER_KWARGS: dict[str, Any] = {
    "solver": "newton",
    "integrator": "implicitfast",
    "cone": "elliptic",
    "iterations": 20,
    "ls_iterations": 10,
    "ls_parallel": True,
    "impratio": 1000.0,
    "use_mujoco_contacts": False,
    "use_mujoco_cpu": True,
    "disable_contacts": True,
}

DEFAULT_FR3_MUJOCO_SOLVER_KWARGS: dict[str, Any] = {
    **DEFAULT_MUJOCO_SOLVER_KWARGS,
    "disable_contacts": False,
}


@dataclasses.dataclass
class CoupledFruitingScene:
    """Cable ``SolverVBD`` scene plus optional MuJoCo robot model and coupling buffers."""

    cable: CoupledCableScene
    cable_collision_pipeline: Any
    vbd_only: bool = False
    mujoco_only: bool = False
    robot_model: newton.Model | None = None
    tcp_body_index: int = -1
    mj_solver: newton.solvers.SolverMuJoCo | None = None
    robot_state_0: Any | None = None
    robot_state_1: Any | None = None
    robot_control: Any | None = None
    mj_contacts: Any | None = None
    proxy_registry: Any | None = None
    proxy_forces: wp.array | None = None
    coupling_forces_cache: wp.array | None = None
    last_vbd_contacts: Any | None = None
    force_debug: CouplingForceDebugRecorder | None = None
    stem_apple_joint_index: int | None = None
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM
    stem_harvest_explicit_apple_weight: bool = True
    """Add ``-m_apple * gravity`` to stem harvest (prescribed apple, ``inv_mass == 0``)."""
    apple_mass_kg: float = 0.0
    """Cached ``body_mass[apple]`` at build; avoids host sync during CUDA graph capture."""
    gravity_vec: wp.vec3 = dataclasses.field(default_factory=lambda: wp.vec3(0.0, 0.0, -9.81))
    use_mujoco_contacts: bool = False
    robot_disable_contacts: bool = True
    robot_kinematic_mode: bool = False
    qd_synced: wp.array | None = None
    """Pooled copy of cable ``body_qd`` before VBD (velocity-delta harvest)."""

    def apply_fr3_ee_teleop(
        self,
        dt: float,
        controller: fr3_robot.Fr3EEVelocityController,
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        if (
            self.robot_model is None
            or self.robot_state_0 is None
            or self.robot_control is None
            or self.mj_solver is None
        ):
            raise ValueError(
                "apply_fr3_ee_teleop requires robot model, state, control, and MuJoCo solver"
            )
        velocity = controller.run_ik_teleop_frame(
            dt,
            self.robot_state_0,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        controller.apply_ik_to_mujoco_control(
            self.robot_state_0,
            self.robot_control,
            frame_dt=dt,
            command_velocity=velocity,
        )
        self.robot_state_1.joint_q.assign(self.robot_state_0.joint_q)
        self.robot_state_1.joint_qd.assign(self.robot_state_0.joint_qd)
        self.mj_solver._update_mjc_data(
            self.mj_solver.mj_data, self.robot_model, self.robot_state_0
        )
        return velocity

    def apply_fr3_ee_teleop_direct(
        self,
        dt: float,
        controller: fr3_robot.Fr3EEDirectJointController,
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        if (
            self.robot_model is None
            or self.robot_state_0 is None
            or self.robot_control is None
            or self.mj_solver is None
        ):
            raise ValueError(
                "apply_fr3_ee_teleop_direct requires robot model, state, control, and MuJoCo solver"
            )
        velocity = controller.run_ik_teleop_frame(
            dt,
            self.robot_state_0,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        controller.apply_direct_joints(
            self.robot_state_0,
            self.robot_control,
            mj_solver=self.mj_solver,
        )
        self.robot_state_1.joint_q.assign(self.robot_state_0.joint_q)
        self.robot_state_1.joint_qd.assign(self.robot_state_0.joint_qd)
        return velocity

    def vbd_substep(
        self,
        dt: float,
        *,
        after_cable_clear_forces: Callable[[], None] | None = None,
    ) -> Any:
        self.cable.state_0.clear_forces()
        if after_cable_clear_forces is not None:
            after_cable_clear_forces()
        vbd_contacts = self.cable.model.collide(
            self.cable.state_0,
            collision_pipeline=self.cable_collision_pipeline,
        )
        self.cable.solver.step(
            self.cable.state_0,
            self.cable.state_1,
            self.cable.control,
            vbd_contacts,
            dt,
        )
        self.cable.state_0, self.cable.state_1 = self.cable.state_1, self.cable.state_0
        self.last_vbd_contacts = vbd_contacts
        return vbd_contacts

    def _mujoco_and_sync_proxy(self, dt: float) -> None:
        self.robot_state_0.clear_forces()
        if self.robot_kinematic_mode:
            self.coupling_forces_cache.zero_()
            newton.eval_fk(
                self.robot_model,
                self.robot_model.joint_q,
                self.robot_model.joint_qd,
                self.robot_state_0,
            )
            if self.mj_solver is not None:
                fr3_robot.sync_mujoco_visual_state(
                    self.mj_solver, self.robot_model, self.robot_state_0
                )
        else:
            self.coupling_forces_cache.assign(self.proxy_forces)
            if self.force_debug is not None:
                self.force_debug.record_applied_from_scene(self)
            _apply_spatial_wrench_to_body_f(
                self.robot_state_0, self.tcp_body_index, self.coupling_forces_cache
            )

            if not self.use_mujoco_contacts and not self.robot_disable_contacts:
                self.robot_model.collide(self.robot_state_0, self.mj_contacts)

            self.mj_solver.step(
                self.robot_state_0,
                self.robot_state_1,
                self.robot_control,
                self.mj_contacts,
                dt,
            )
            self.robot_state_0, self.robot_state_1 = self.robot_state_1, self.robot_state_0

        dev = self.robot_model.device
        rid, pid = self.proxy_registry.ids_wp(dev)

        cable = self.cable
        use_apple_sync = (
            cable.apple_body is not None
            and cable.gripper_proxy_apple_joint is not None
            and cable.gripper_proxy_offset_in_apple_frame is not None
        )
        if use_apple_sync:
            launch_mirror_robot_to_proxy_and_apple(
                robot_ids=rid,
                proxy_ids=pid,
                src_body_q=self.robot_state_0.body_q,
                src_body_qd=self.robot_state_0.body_qd,
                dst_body_q=cable.state_0.body_q,
                dst_body_qd=cable.state_0.body_qd,
                proxy_forces=self.coupling_forces_cache,
                cable_model=cable.model,
                gravity=self.gravity_vec,
                dt=dt,
                apple_body_id=cable.apple_body,
                proxy_offset_in_apple=wp.vec3(*cable.gripper_proxy_offset_in_apple_frame),
                device=str(dev),
            )
        else:
            launch_mirror_robot_to_proxy(
                robot_ids=rid,
                proxy_ids=pid,
                src_body_q=self.robot_state_0.body_q,
                src_body_qd=self.robot_state_0.body_qd,
                dst_body_q=cable.state_0.body_q,
                dst_body_qd=cable.state_0.body_qd,
                proxy_forces=self.coupling_forces_cache,
                cable_model=cable.model,
                gravity=self.gravity_vec,
                dt=dt,
                device=str(dev),
            )
        if use_apple_sync:
            prescribed = (
                int(cable.gripper_proxy_body),
                int(cable.apple_body),
            )
            # Keep state_1 and body_q_prev aligned with the teleported apple/proxy so
            # VBD does not infer spurious velocity on prescribed bodies.
            copy_cable_body_q_between_states(
                cable,
                src_state=cable.state_0,
                dst_state=cable.state_1,
                body_ids=prescribed,
            )
            sync_solver_body_q_prev_from_state(cable, cable.state_1.body_q)
        else:
            align_proxy_body_q_prev_for_vbd(cable, self.proxy_registry.proxy_body_ids)

        if self.use_mujoco_contacts:
            self.mj_solver.update_contacts(self.mj_contacts, self.robot_state_0)

    def mujoco_substep(self, dt: float) -> None:
        if self.vbd_only or self.robot_model is None:
            raise ValueError(
                "mujoco_substep requires a built robot model; pass vbd_only=False to the builder"
            )
        self._mujoco_and_sync_proxy(dt)

    def coupled_substep(
        self,
        dt: float,
        *,
        after_cable_clear_forces: Callable[[], None] | None = None,
    ) -> None:
        if self.vbd_only:
            raise ValueError(
                "coupled_substep requires vbd_only=False; use vbd_substep for cable-only stepping"
            )
        if self.mujoco_only:
            raise ValueError(
                "coupled_substep requires mujoco_only=False; use mujoco_substep for robot-only stepping"
            )

        self.mujoco_substep(dt)
        use_stem_harvest = self.stem_apple_joint_index is not None
        if not use_stem_harvest:
            if self.qd_synced is None:
                raise ValueError("qd_synced buffer missing; build coupled scene via builders")
            wp.copy(self.qd_synced, self.cable.state_0.body_qd)

        vbd_contacts = self.vbd_substep(dt, after_cable_clear_forces=after_cable_clear_forces)

        if use_stem_harvest:
            harvest_stem_tension_for_tcp(
                cable_model=self.cable.model,
                cable_solver=self.cable.solver,
                body_q_post=self.cable.state_0.body_q,
                body_q_prev=self.cable.state_1.body_q,
                dt=dt,
                stem_apple_joint_index=self.stem_apple_joint_index,
                tcp_body_index=self.tcp_body_index,
                out_robot_wrenches=self.proxy_forces,
                coupling_gain=self.stem_coupling_gain,
                force_cap_N=self.stem_force_cap_N,
                torque_cap_Nm=self.stem_torque_cap_Nm,
                explicit_apple_weight=self.stem_harvest_explicit_apple_weight,
                apple_body_index=self.cable.apple_body,
                apple_mass_kg=self.apple_mass_kg,
                gravity=self.gravity_vec,
                robot_body_q=self.robot_state_0.body_q,
                grasp_offset_in_apple_frame=self.cable.gripper_proxy_offset_in_apple_frame,
            )
        else:
            harvest_proxy_wrenches(
                self.cable.solver,
                self.cable.state_0,
                vbd_contacts,
                dt,
                registry=self.proxy_registry,
                model=self.cable.model,
                qd_synced=self.qd_synced,
                gravity=self.gravity_vec,
                out_robot_wrenches=self.proxy_forces,
            )
        if self.force_debug is not None:
            self.force_debug.record_harvested_from_scene(self)
