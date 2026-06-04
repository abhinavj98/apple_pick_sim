    """Mega plant (N VBD instances) + one FR3 MuJoCo arm (fd_ghost coupling)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting.apply_wrench import _apply_spatial_wrench_to_body_f
from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    ProxyBodyRegistry,
    align_proxy_body_q_prev_for_vbd,
    harvest_proxy_wrenches,
    harvest_stem_tension_for_tcp,
    launch_mirror_robot_to_proxy_offset,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)
from apple_pick_sim.fruiting_system.mega import MegaCoupledCableScene


def _co_teleport_apples_from_proxies(mega: MegaCoupledCableScene) -> None:
    """After ghost proxy sync, align welded apples to grasp offset (``fix_to_apple``)."""
    bq = mega.state_0.body_q.numpy().reshape(-1, 7).copy()
    bqd = mega.state_0.body_qd.numpy().reshape(-1, 6).copy()
    for inst in mega.instances:
        if inst.apple_body is None or inst.gripper_proxy_offset_in_apple_frame is None:
            continue
        proxy = inst.gripper_proxy_body
        apple = inst.apple_body
        off = np.asarray(inst.gripper_proxy_offset_in_apple_frame, dtype=np.float32).reshape(3)
        proxy_q = bq[proxy]
        p = proxy_q[:3].astype(np.float64)
        q = wp.quat(float(proxy_q[3]), float(proxy_q[4]), float(proxy_q[5]), float(proxy_q[6]))
        apple_pos = p - np.asarray(wp.quat_rotate(q, wp.vec3(*off)), dtype=np.float64)
        bq[apple, :3] = apple_pos.astype(np.float32)
        bq[apple, 3:7] = proxy_q[3:7]
        bqd[apple] = bqd[proxy]
    mega.state_0.body_q.assign(bq.ravel())
    mega.state_0.body_qd.assign(bqd.ravel())


def mega_ghost_position_offsets_wp(
    mega: MegaCoupledCableScene,
    *,
    nominal_index: int = 0,
    device: str | None = None,
) -> wp.array:
    """Per-instance world offset vs nominal base (for fd_ghost proxy mirroring)."""
    nom = mega.instance(nominal_index).base_pos
    dev = device if device is not None else str(mega.model.device)
    offsets = [
        wp.vec3(
            inst.base_pos[0] - nom[0],
            inst.base_pos[1] - nom[1],
            inst.base_pos[2] - nom[2],
        )
        for inst in mega.instances
    ]
    return wp.array(offsets, dtype=wp.vec3, device=dev)


@dataclasses.dataclass
class MegaCoupledFruitingScene:
    """One FR3 + mega VBD plant: ghost-sync all proxies, harvest nominal column only."""

    cable: MegaCoupledCableScene
    cable_collision_pipeline: Any
    ghost_registry: ProxyBodyRegistry
    harvest_registry: ProxyBodyRegistry
    position_offsets_wp: wp.array
    nominal_index: int = 0
    robot_model: newton.Model | None = None
    tcp_body_index: int = -1
    mj_solver: newton.solvers.SolverMuJoCo | None = None
    robot_state_0: Any | None = None
    robot_state_1: Any | None = None
    robot_control: Any | None = None
    mj_contacts: Any | None = None
    proxy_forces: wp.array | None = None
    coupling_forces_cache: wp.array | None = None
    last_vbd_contacts: Any | None = None
    force_debug: CouplingForceDebugRecorder | None = None
    gravity_vec: wp.vec3 = dataclasses.field(default_factory=lambda: wp.vec3(0.0, 0.0, -9.81))
    use_mujoco_contacts: bool = False
    robot_disable_contacts: bool = True
    robot_kinematic_mode: bool = True
    qd_synced: wp.array | None = None
    stem_apple_joint_index: int | None = None
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM
    stem_harvest_explicit_apple_weight: bool = True
    apple_mass_kg: float = 0.0

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
        rid, pid = self.ghost_registry.ids_wp(dev)
        cable = self.cable
        launch_mirror_robot_to_proxy_offset(
            robot_ids=rid,
            proxy_ids=pid,
            position_offsets=self.position_offsets_wp,
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
        align_bodies = list(cable.all_gripper_proxy_body_ids())
        if cable.gripper_proxy_config.fix_to_apple:
            _co_teleport_apples_from_proxies(cable)
            for inst in cable.instances:
                if inst.apple_body is not None:
                    align_bodies.append(inst.apple_body)
        align_proxy_body_q_prev_for_vbd(cable, tuple(align_bodies))

        if self.use_mujoco_contacts:
            self.mj_solver.update_contacts(self.mj_contacts, self.robot_state_0)

    def mujoco_substep(self, dt: float) -> None:
        if self.robot_model is None:
            raise ValueError("mujoco_substep requires a built robot model")
        self._mujoco_and_sync_proxy(dt)

    def coupled_substep(
        self,
        dt: float,
        *,
        after_cable_clear_forces: Callable[[], None] | None = None,
    ) -> None:
        self.mujoco_substep(dt)
        use_stem_harvest = self.stem_apple_joint_index is not None
        if not use_stem_harvest:
            if self.qd_synced is None:
                raise ValueError("qd_synced buffer missing; build via build_mega_coupled_fruiting_fr3")
            wp.copy(self.qd_synced, self.cable.state_0.body_qd)
        vbd_contacts = self.vbd_substep(dt, after_cable_clear_forces=after_cable_clear_forces)
        if use_stem_harvest:
            inst = self.cable.instance(self.nominal_index)
            apple_bid = inst.apple_body if inst.apple_body is not None else -1
            grasp_off = inst.gripper_proxy_offset_in_apple_frame
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
                apple_body_index=apple_bid,
                apple_mass_kg=self.apple_mass_kg,
                gravity=self.gravity_vec,
                robot_body_q=self.robot_state_0.body_q,
                grasp_offset_in_apple_frame=grasp_off,
            )
        else:
            harvest_proxy_wrenches(
                self.cable.solver,
                self.cable.state_0,
                vbd_contacts,
                dt,
                registry=self.harvest_registry,
                model=self.cable.model,
                qd_synced=self.qd_synced,
                gravity=self.gravity_vec,
                out_robot_wrenches=self.proxy_forces,
            )
        if self.force_debug is not None:
            self.force_debug.record_harvested_from_scene(self)
