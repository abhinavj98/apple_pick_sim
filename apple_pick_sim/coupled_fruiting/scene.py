"""CoupledFruitingScene, MegaCoupledFruitingScene, and staggered substep orchestration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import warp as wp

import newton

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting.apply_wrench import (
    _apply_spatial_wrench_to_body_f,
)
from apple_pick_sim.coupled_fruiting.vic_wrench import apply_vic_to_coupling_cache
from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim.fruiting_system import CoupledCableScene
from apple_pick_sim.fruiting_system.mega import MegaCoupledCableScene
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    ProxyBodyRegistry,
    align_proxy_body_q_prev_for_vbd,
    copy_cable_body_q_between_states,
    harvest_proxy_wrenches,
    harvest_stem_tension_for_tcp,
    launch_mirror_robot_to_proxy,
    launch_mirror_robot_to_proxy_and_apple,
    launch_mirror_robot_to_proxy_offset,
    launch_mirror_robot_to_proxy_offset_and_apple,
    sync_solver_body_q_prev_from_state,
)

DEFAULT_STEM_COUPLING_GAIN: float = 1.0
DEFAULT_STEM_FORCE_CAP_N: float = 1000.0
DEFAULT_STEM_TORQUE_CAP_NM: float = 1000.0

DEFAULT_MUJOCO_SOLVER_KWARGS: dict[str, Any] = {
    "solver": "newton",
    "integrator": "implicitfast",
    "cone": "elliptic",
    "iterations": 20,
    "ls_iterations": 10,
    "ls_parallel": True,
    "impratio": 1000.0,
    "use_mujoco_contacts": False,
    "use_mujoco_cpu": False,
    "disable_contacts": True,
}

DEFAULT_FR3_MUJOCO_SOLVER_KWARGS: dict[str, Any] = {
    **DEFAULT_MUJOCO_SOLVER_KWARGS,
    "disable_contacts": False,
}


def _vbd_substep(
    cable: CoupledCableScene | MegaCoupledCableScene,
    cable_collision_pipeline: Any,
    dt: float,
    *,
    after_cable_clear_forces: Callable[[], None] | None = None,
) -> Any:
    cable.state_0.clear_forces()
    if after_cable_clear_forces is not None:
        after_cable_clear_forces()
    vbd_contacts = cable.model.collide(
        cable.state_0,
        collision_pipeline=cable_collision_pipeline,
    )
    cable.solver.step(
        cable.state_0,
        cable.state_1,
        cable.control,
        vbd_contacts,
        dt,
    )
    cable.state_0, cable.state_1 = cable.state_1, cable.state_0
    return vbd_contacts


def _mujoco_robot_substep_prefix(scene: Any, dt: float) -> None:
    """Clear forces, run kinematic FK or dynamic MuJoCo step (no proxy sync)."""
    scene.robot_state_0.clear_forces()
    if scene.robot_kinematic_mode:
        scene.coupling_forces_cache.zero_()
        newton.eval_fk(
            scene.robot_model,
            scene.robot_model.joint_q,
            scene.robot_model.joint_qd,
            scene.robot_state_0,
        )
        if scene.mj_solver is not None:
            fr3_robot.sync_mujoco_visual_state(
                scene.mj_solver, scene.robot_model, scene.robot_state_0
            )
    else:
        scene.coupling_forces_cache.assign(scene.proxy_forces)
        apply_vic_to_coupling_cache(scene)
        if scene.force_debug is not None:
            scene.force_debug.record_applied_from_scene(scene)
        _apply_spatial_wrench_to_body_f(
            scene.robot_state_0, scene.tcp_body_index, scene.coupling_forces_cache
        )
        if not scene.use_mujoco_contacts and not scene.robot_disable_contacts:
            scene.robot_model.collide(scene.robot_state_0, scene.mj_contacts)
        scene.mj_solver.step(
            scene.robot_state_0,
            scene.robot_state_1,
            scene.robot_control,
            scene.mj_contacts,
            dt,
        )
        scene.robot_state_0, scene.robot_state_1 = scene.robot_state_1, scene.robot_state_0


def _apply_fr3_ee_teleop_impl(
    scene: Any,
    dt: float,
    controller: fr3_robot.Fr3EEVelocityController,
    *,
    viewer: fr3_robot._KeyViewer | None = None,
    velocity: fr3_robot.EEVelocity | None = None,
) -> fr3_robot.EEVelocity:
    if (
        scene.robot_model is None
        or scene.robot_state_0 is None
        or scene.robot_control is None
        or scene.mj_solver is None
    ):
        raise ValueError(
            "apply_fr3_ee_teleop requires robot model, state, control, and MuJoCo solver"
        )
    if getattr(scene, "vic_controller", None) is not None:
        if not getattr(scene, "vic_wrench_only_configured", False):
            fr3_robot.configure_vic_wrench_only_arm(
                scene.robot_model,
                scene.robot_state_0,
                scene.robot_control,
                scene.mj_solver,
            )
            scene.vic_wrench_only_configured = True
        velocity = controller.run_tcp_target_teleop_frame(
            dt,
            scene.robot_state_0,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        fr3_robot.hold_mujoco_actuator_targets_at_state(
            scene.robot_model,
            scene.robot_state_0,
            scene.robot_control,
        )
    else:
        velocity = controller.run_ik_teleop_frame(
            dt,
            scene.robot_state_0,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        controller.apply_ik_to_mujoco_control(
            scene.robot_state_0,
            scene.robot_control,
            frame_dt=dt,
            command_velocity=velocity,
        )
    scene.robot_state_1.joint_q.assign(scene.robot_state_0.joint_q)
    scene.robot_state_1.joint_qd.assign(scene.robot_state_0.joint_qd)
    scene.mj_solver._update_mjc_data(
        scene.mj_solver.mj_data, scene.robot_model, scene.robot_state_0
    )
    if getattr(scene, "vic_controller", None) is not None:
        scene.vic_target_tf = controller.target_tf
        scene.vic_target_twist = velocity
    return velocity


def _apply_fr3_ee_teleop_direct_impl(
    scene: Any,
    dt: float,
    controller: fr3_robot.Fr3EEDirectJointController,
    *,
    viewer: fr3_robot._KeyViewer | None = None,
    velocity: fr3_robot.EEVelocity | None = None,
) -> fr3_robot.EEVelocity:
    if (
        scene.robot_model is None
        or scene.robot_state_0 is None
        or scene.robot_control is None
        or scene.mj_solver is None
    ):
        raise ValueError(
            "apply_fr3_ee_teleop_direct requires robot model, state, control, and MuJoCo solver"
        )
    velocity = controller.run_ik_teleop_frame(
        dt,
        scene.robot_state_0,
        velocity=velocity,
        viewer=viewer,
        poll_events=True,
    )
    controller.apply_direct_joints(
        scene.robot_state_0,
        scene.robot_control,
        mj_solver=scene.mj_solver,
    )
    scene.robot_state_1.joint_q.assign(scene.robot_state_0.joint_q)
    scene.robot_state_1.joint_qd.assign(scene.robot_state_0.joint_qd)
    return velocity


def _harvest_coupling_wrenches(
    scene: Any,
    vbd_contacts: Any,
    dt: float,
    *,
    harvest_registry: ProxyBodyRegistry,
    cable: CoupledCableScene | MegaCoupledCableScene,
    apple_body_index: int | None = None,
    grasp_offset_in_apple_frame: tuple[float, ...] | None = None,
) -> None:
    use_stem_harvest = scene.stem_apple_joint_index is not None
    if use_stem_harvest:
        if isinstance(cable, MegaCoupledCableScene):
            inst = cable.instance(scene.nominal_index)
            apple_bid = (
                apple_body_index
                if apple_body_index is not None
                else (inst.apple_body if inst.apple_body is not None else -1)
            )
            grasp_off = (
                grasp_offset_in_apple_frame
                if grasp_offset_in_apple_frame is not None
                else inst.gripper_proxy_offset_in_apple_frame
            )
        else:
            apple_bid = (
                apple_body_index if apple_body_index is not None else cable.apple_body
            )
            grasp_off = (
                grasp_offset_in_apple_frame
                if grasp_offset_in_apple_frame is not None
                else cable.gripper_proxy_offset_in_apple_frame
            )
        harvest_stem_tension_for_tcp(
            cable_model=cable.model,
            cable_solver=cable.solver,
            body_q_post=cable.state_0.body_q,
            body_q_prev=cable.state_1.body_q,
            dt=dt,
            stem_apple_joint_index=scene.stem_apple_joint_index,
            tcp_body_index=scene.tcp_body_index,
            out_robot_wrenches=scene.proxy_forces,
            coupling_gain=scene.stem_coupling_gain,
            force_cap_N=scene.stem_force_cap_N,
            torque_cap_Nm=scene.stem_torque_cap_Nm,
            explicit_apple_weight=scene.stem_harvest_explicit_apple_weight,
            apple_body_index=apple_bid,
            apple_mass_kg=scene.apple_mass_kg,
            gravity=scene.gravity_vec,
            robot_body_q=scene.robot_state_0.body_q,
            grasp_offset_in_apple_frame=grasp_off,
        )
    else:
        harvest_proxy_wrenches(
            cable.solver,
            cable.state_0,
            vbd_contacts,
            dt,
            registry=harvest_registry,
            model=cable.model,
            qd_synced=scene.qd_synced,
            gravity=scene.gravity_vec,
            out_robot_wrenches=scene.proxy_forces,
        )
    if scene.force_debug is not None:
        scene.force_debug.record_harvested_from_scene(scene)


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


def _sync_single_proxy_after_mujoco(scene: CoupledFruitingScene, dt: float) -> None:
    dev = scene.robot_model.device
    rid, pid = scene.proxy_registry.ids_wp(dev)
    cable = scene.cable
    use_apple_sync = (
        cable.apple_body is not None
        and cable.gripper_proxy_apple_joint is not None
        and cable.gripper_proxy_offset_in_apple_frame is not None
    )
    if use_apple_sync:
        launch_mirror_robot_to_proxy_and_apple(
            robot_ids=rid,
            proxy_ids=pid,
            src_body_q=scene.robot_state_0.body_q,
            src_body_qd=scene.robot_state_0.body_qd,
            dst_body_q=cable.state_0.body_q,
            dst_body_qd=cable.state_0.body_qd,
            proxy_forces=scene.coupling_forces_cache,
            cable_model=cable.model,
            gravity=scene.gravity_vec,
            dt=dt,
            apple_body_id=cable.apple_body,
            proxy_offset_in_apple=wp.transform(
                wp.vec3(*cable.gripper_proxy_offset_in_apple_frame[:3]),
                wp.quat(*cable.gripper_proxy_offset_in_apple_frame[3:]),
            ),
            device=str(dev),
        )
    else:
        launch_mirror_robot_to_proxy(
            robot_ids=rid,
            proxy_ids=pid,
            src_body_q=scene.robot_state_0.body_q,
            src_body_qd=scene.robot_state_0.body_qd,
            dst_body_q=cable.state_0.body_q,
            dst_body_qd=cable.state_0.body_qd,
            proxy_forces=scene.coupling_forces_cache,
            cable_model=cable.model,
            gravity=scene.gravity_vec,
            dt=dt,
            device=str(dev),
        )
    if use_apple_sync:
        prescribed = (
            int(cable.gripper_proxy_body),
            int(cable.apple_body),
        )
        copy_cable_body_q_between_states(
            cable,
            src_state=cable.state_0,
            dst_state=cable.state_1,
            body_ids=prescribed,
        )
        sync_solver_body_q_prev_from_state(cable, cable.state_1.body_q)
    else:
        align_proxy_body_q_prev_for_vbd(cable, scene.proxy_registry.proxy_body_ids)


def _sync_mega_proxy_after_mujoco(scene: MegaCoupledFruitingScene, dt: float) -> None:
    dev = scene.robot_model.device
    rid, pid = scene.ghost_registry.ids_wp(dev)
    cable = scene.cable
    mirror_kw = dict(
        robot_ids=rid,
        proxy_ids=pid,
        position_offsets=scene.position_offsets_wp,
        src_body_q=scene.robot_state_0.body_q,
        src_body_qd=scene.robot_state_0.body_qd,
        dst_body_q=cable.state_0.body_q,
        dst_body_qd=cable.state_0.body_qd,
        proxy_forces=scene.coupling_forces_cache,
        cable_model=cable.model,
        gravity=scene.gravity_vec,
        dt=dt,
        device=str(dev),
    )
    if cable.gripper_proxy_config.fix_to_apple:
        if scene.welded_co_teleport_arrays is None:
            raise ValueError(
                "welded_co_teleport_arrays missing; rebuild with fix_to_apple=True"
            )
        apple_body_ids, proxy_offset_in_apple = scene.welded_co_teleport_arrays
        launch_mirror_robot_to_proxy_offset_and_apple(
            apple_body_ids=apple_body_ids,
            proxy_offset_in_apple=proxy_offset_in_apple,
            **mirror_kw,
        )
    else:
        launch_mirror_robot_to_proxy_offset(**mirror_kw)

    align_bodies = list(cable.all_gripper_proxy_body_ids())
    if cable.gripper_proxy_config.fix_to_apple:
        for inst in cable.instances:
            if inst.apple_body is not None:
                align_bodies.append(inst.apple_body)
    align_proxy_body_q_prev_for_vbd(cable, tuple(align_bodies))


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
    vic_controller: fr3_robot.Fr3EEImpedanceController | None = None
    vic_gains: fr3_robot.ImpedanceGains | None = None
    vic_target_tf: wp.transform | None = None
    vic_target_twist: fr3_robot.EEVelocity | None = None
    vic_wrench_only_configured: bool = False

    def apply_fr3_ee_teleop(
        self,
        dt: float,
        controller: fr3_robot.Fr3EEVelocityController,
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        return _apply_fr3_ee_teleop_impl(
            self, dt, controller, viewer=viewer, velocity=velocity
        )

    def apply_fr3_ee_teleop_direct(
        self,
        dt: float,
        controller: fr3_robot.Fr3EEDirectJointController,
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        return _apply_fr3_ee_teleop_direct_impl(
            self, dt, controller, viewer=viewer, velocity=velocity
        )

    def vbd_substep(
        self,
        dt: float,
        *,
        after_cable_clear_forces: Callable[[], None] | None = None,
    ) -> Any:
        vbd_contacts = _vbd_substep(
            self.cable,
            self.cable_collision_pipeline,
            dt,
            after_cable_clear_forces=after_cable_clear_forces,
        )
        self.last_vbd_contacts = vbd_contacts
        return vbd_contacts

    def _mujoco_and_sync_proxy(self, dt: float) -> None:
        _mujoco_robot_substep_prefix(self, dt)
        _sync_single_proxy_after_mujoco(self, dt)
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
        _harvest_coupling_wrenches(
            self,
            vbd_contacts,
            dt,
            harvest_registry=self.proxy_registry,
            cable=self.cable,
        )


@dataclasses.dataclass
class MegaCoupledFruitingScene:
    """One FR3 + mega VBD plant: ghost-sync all proxies, harvest nominal column only."""

    cable: MegaCoupledCableScene
    cable_collision_pipeline: Any
    ghost_registry: ProxyBodyRegistry
    harvest_registry: ProxyBodyRegistry
    position_offsets_wp: wp.array
    nominal_index: int = 0
    welded_co_teleport_arrays: tuple[wp.array, wp.array] | None = None
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
    vic_controller: fr3_robot.Fr3EEImpedanceController | None = None
    vic_gains: fr3_robot.ImpedanceGains | None = None
    vic_target_tf: wp.transform | None = None
    vic_target_twist: fr3_robot.EEVelocity | None = None
    vic_wrench_only_configured: bool = False
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
        return _apply_fr3_ee_teleop_direct_impl(
            self, dt, controller, viewer=viewer, velocity=velocity
        )

    def vbd_substep(
        self,
        dt: float,
        *,
        after_cable_clear_forces: Callable[[], None] | None = None,
    ) -> Any:
        vbd_contacts = _vbd_substep(
            self.cable,
            self.cable_collision_pipeline,
            dt,
            after_cable_clear_forces=after_cable_clear_forces,
        )
        self.last_vbd_contacts = vbd_contacts
        return vbd_contacts

    def _mujoco_and_sync_proxy(self, dt: float) -> None:
        _mujoco_robot_substep_prefix(self, dt)
        _sync_mega_proxy_after_mujoco(self, dt)
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
                raise ValueError(
                    "qd_synced buffer missing; build via build_mega_coupled_fruiting_fr3"
                )
            wp.copy(self.qd_synced, self.cable.state_0.body_qd)
        vbd_contacts = self.vbd_substep(dt, after_cable_clear_forces=after_cable_clear_forces)
        _harvest_coupling_wrenches(
            self,
            vbd_contacts,
            dt,
            harvest_registry=self.harvest_registry,
            cable=self.cable,
        )
