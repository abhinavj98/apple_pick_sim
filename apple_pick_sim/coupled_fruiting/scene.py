"""CoupledFruitingScene and staggered substep orchestration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import warp as wp

import newton

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting.apply_wrench import (
    _apply_registry_spatial_wrenches_to_body_f,
    _apply_spatial_wrench_to_body_f,
)
from apple_pick_sim.coupled_fruiting.vic_joint_torques import (
    allocate_vic_joint_torque_buffers,
    apply_vic_joint_torques_to_scene,
)
from apple_pick_sim.coupled_fruiting.vic_joint_torques_batched import (
    allocate_vic_joint_torque_buffers_batched,
    apply_vic_joint_torques_batched_to_scene,
)
from apple_pick_sim.coupled_fruiting.vic_wrench import apply_vic_to_coupling_cache
from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim.fruiting_system import CoupledCableScene
from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    ProxyBodyRegistry,
    align_proxy_body_q_prev_for_vbd,
    copy_cable_body_q_between_states,
    harvest_batched_stem_tension,
    harvest_proxy_wrenches,
    harvest_stem_tension_for_tcp,
    launch_mirror_robot_to_proxy,
    launch_mirror_robot_to_proxy_and_apple,
    launch_mirror_robot_to_proxy_offset_and_apple,
    sync_solver_body_q_prev_from_state,
    welded_co_teleport_arrays_for_layout,
)

DEFAULT_STEM_COUPLING_GAIN: float = 1.0
DEFAULT_STEM_FORCE_CAP_N: float = 200.0
DEFAULT_STEM_TORQUE_CAP_NM: float = 50.0

DEFAULT_MUJOCO_SOLVER_KWARGS: dict[str, Any] = {
    "solver": "newton",
    "integrator": "implicitfast",
    "cone": "elliptic",
    "iterations": 20,
    "ls_iterations": 10,
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
    cable: CoupledCableScene,
    cable_collision_pipeline: Any,
    dt: float,
    *,
    after_cable_clear_forces: Callable[[], None] | None = None,
) -> Any:
    """One VBD substep: clear forces, collide, solve, swap state buffers.

    Runs the cable ``SolverVBD`` integration step and swaps ``state_0`` /
    ``state_1``. Optional ``after_cable_clear_forces`` hook runs after
    ``clear_forces`` for external force injection.

    Called from :meth:`CoupledFruitingScene.vbd_substep` and
    :func:`~apple_pick_sim.coupled_fruiting.settle_then_weld.settle_vbd_substeps`
    during ``fix_to_apple`` quiet-start settling (cable-only, no robot step).
    """
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
        if getattr(scene, "vic_use_joint_torques", False):
            if scene.robot_control.joint_f is not None:
                scene.robot_control.joint_f.zero_()
            if (
                getattr(scene, "layout", None) is not None
                and getattr(scene, "vic_target_positions_wp", None) is not None
            ):
                apply_vic_joint_torques_batched_to_scene(scene)
            else:
                apply_vic_joint_torques_to_scene(scene)
        else:
            apply_vic_to_coupling_cache(scene)
        if scene.force_debug is not None:
            scene.force_debug.record_applied_from_scene(scene)
        if scene.proxy_registry is not None:
            dev = scene.robot_state_0.body_f.device
            _apply_registry_spatial_wrenches_to_body_f(
                scene.robot_state_0,
                scene.proxy_registry.robot_ids_wp(dev),
                scene.coupling_forces_cache,
            )
        else:
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


def init_robot_mujoco_step_buffers(scene: Any) -> None:
    """Seed ``robot_state_1`` and MuJoCo ``qpos`` from ``robot_state_0`` once at scene setup.

    Call after IK bootstrap or any operation that rewrites ``robot_state_0`` so the
    spare Newton state buffer and MuJoCo data match before the first ``mj_solver.step``.
    """
    if (
        scene.robot_model is None
        or scene.robot_state_0 is None
        or scene.robot_state_1 is None
        or scene.mj_solver is None
    ):
        raise ValueError(
            "init_robot_mujoco_step_buffers requires robot model, both states, and MuJoCo solver"
        )
    scene.robot_state_1.joint_q.assign(scene.robot_state_0.joint_q)
    scene.robot_state_1.joint_qd.assign(scene.robot_state_0.joint_qd)
    newton.eval_fk(
        scene.robot_model,
        scene.robot_state_0.joint_q,
        scene.robot_state_0.joint_qd,
        scene.robot_state_1,
    )
    scene.mj_solver._update_mjc_data(
        scene.mj_solver.mj_data, scene.robot_model, scene.robot_state_0
    )


def _update_fr3_ee_teleop_impl(
    scene: Any,
    dt: float,
    controller: (
        fr3_robot.Fr3EEVelocityController
        | fr3_robot.Fr3EEImpedanceController
        | fr3_robot.Fr3EEDirectJointController
        | fr3_robot.Fr3BatchedEEVelocityController
        | fr3_robot.Fr3BatchedEEDirectJointController
        | fr3_robot.Fr3BatchedEEImpedanceController
    ),
    *,
    viewer: fr3_robot._KeyViewer | None = None,
    velocity: fr3_robot.EEVelocity | None = None,
) -> fr3_robot.EEVelocity:
    """FR3 EE teleop: delegate per-frame staging to ``controller.run_coupled_teleop_frame``.

    When ``scene.vic_controller`` is set, stages ``vic_target_tf`` / ``vic_target_twist``
    for the next MuJoCo substep. VIC arm setup (zero PD, buffers) must be done at scene
    finalize, not here.

    Called from :meth:`CoupledFruitingScene.update_fr3_ee_teleop` and
    :meth:`CoupledFruitingScene.update_fr3_ee_teleop_direct` once per viewer frame.
    """
    if (
        scene.robot_model is None
        or scene.robot_state_0 is None
        or scene.robot_control is None
        or scene.mj_solver is None
    ):
        raise ValueError(
            "update_fr3_ee_teleop requires robot model, state, control, and MuJoCo solver"
        )
    if getattr(scene, "vic_controller", None) is not None:
        if controller is not scene.vic_controller:
            raise ValueError(
                "VIC teleop requires controller to be scene.vic_controller "
                "(single Fr3EEImpedanceController for target integration and wrench law)"
            )
    velocity = controller.run_coupled_teleop_frame(
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        dt,
        viewer=viewer,
        velocity=velocity,
    )
    if getattr(scene, "vic_controller", None) is not None:
        if isinstance(controller, fr3_robot.Fr3BatchedEEImpedanceController):
            controller.stage_targets_to_scene(scene)
            scene.vic_target_twist = velocity
        else:
            scene.vic_target_tf = controller.target_tf
            scene.vic_target_twist = velocity
    return velocity


def _harvest_coupling_wrenches(
    scene: Any,
    vbd_contacts: Any,
    dt: float,
    *,
    harvest_registry: ProxyBodyRegistry,
    cable: CoupledCableScene,
    apple_body_index: int | None = None,
    grasp_offset_in_apple_frame: tuple[float, ...] | None = None,
) -> None:
    """Dispatch stem FIXED-joint harvest or velocity-delta proxy harvest into ``proxy_forces``.

    Chooses path from ``scene.stem_apple_joint_index`` (set at build when
    ``fix_to_apple`` welds proxy to apple):

    - **Stem harvest** — constraint wrench on stem–apple FIXED joint, transferred
      to TCP with optional explicit apple weight (:func:`harvest_stem_tension_for_tcp`).
    - **Velocity-delta** — standard M1 proxy reaction from VBD twist jump
      (:func:`harvest_proxy_wrenches`).

    Called at the end of each :meth:`CoupledFruitingScene.coupled_substep`.
    Harvested wrenches feed the *next* MuJoCo substep as lagged coupling forces.
    """
    use_stem_harvest = scene.stem_apple_joint_index is not None
    if use_stem_harvest:
        layout = getattr(scene, "layout", None)
        tpl_stem = scene.stem_apple_joint_index
        offset = cable.gripper_proxy_offset_in_apple_frame
        if (
            layout is not None
            and layout.num_envs > 1
            and tpl_stem is not None
            and scene.stem_harvest_joint_indices_wp is not None
        ):
            harvest_batched_stem_tension(
                stem_joint_indices_wp=scene.stem_harvest_joint_indices_wp,
                tcp_indices_wp=scene.stem_harvest_tcp_indices_wp,
                apple_indices_wp=scene.stem_harvest_apple_indices_wp,
                grasp_offsets_wp=scene.stem_harvest_grasp_offsets_wp,
                apple_masses_wp=scene.stem_harvest_apple_masses_wp,
                use_grasp_offset_wp=scene.stem_harvest_use_grasp_offset_wp,
                cable_model=cable.model,
                cable_solver=cable.solver,
                body_q_post=cable.state_0.body_q,
                body_q_prev=cable.state_1.body_q,
                dt=dt,
                out_robot_wrenches=scene.proxy_forces,
                coupling_gain=scene.stem_coupling_gain,
                force_cap_N=scene.stem_force_cap_N,
                torque_cap_Nm=scene.stem_torque_cap_Nm,
                explicit_apple_weight=scene.stem_harvest_explicit_apple_weight,
                gravity=scene.gravity_vec,
                robot_body_q=scene.robot_state_0.body_q,
                device=str(scene.proxy_forces.device),
            )
        elif layout is not None and layout.num_envs > 1 and tpl_stem is not None:
            for w in range(layout.num_envs):
                apple_idx = int(layout.apple_body_indices[w])
                harvest_stem_tension_for_tcp(
                    cable_model=cable.model,
                    cable_solver=cable.solver,
                    body_q_post=cable.state_0.body_q,
                    body_q_prev=cable.state_1.body_q,
                    dt=dt,
                    stem_apple_joint_index=layout.joint_index(w, tpl_stem),
                    tcp_body_index=int(layout.tcp_body_indices[w]),
                    out_robot_wrenches=scene.proxy_forces,
                    coupling_gain=scene.stem_coupling_gain,
                    force_cap_N=scene.stem_force_cap_N,
                    torque_cap_Nm=scene.stem_torque_cap_Nm,
                    explicit_apple_weight=scene.stem_harvest_explicit_apple_weight,
                    apple_body_index=apple_idx if apple_idx >= 0 else None,
                    apple_mass_kg=scene.apple_mass_kg,
                    gravity=scene.gravity_vec,
                    robot_body_q=scene.robot_state_0.body_q,
                    grasp_offset_in_apple_frame=offset,
                    clear_wrenches=(w == 0),
                )
        else:
            apple_bid = apple_body_index if apple_body_index is not None else cable.apple_body
            grasp_off = (
                grasp_offset_in_apple_frame
                if grasp_offset_in_apple_frame is not None
                else offset
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


def _sync_single_proxy_after_mujoco(scene: CoupledFruitingScene, dt: float) -> None:
    """Mirror TCP to proxy (and apple when welded); align ``body_q_prev`` for AVBD.

    After each MuJoCo robot step, copies robot TCP pose/twist onto the cable
    gripper proxy (with double-integration correction for lagged forces). When
    ``fix_to_apple`` is active, also co-teleports the apple and syncs
    ``body_q_prev`` on prescribed bodies so VBD does not integrate spurious
    constraint impulses.

    Called from :meth:`CoupledFruitingScene._mujoco_and_sync_proxy` at the
    start of every coupled or robot-only substep.
    """
    dev = scene.robot_model.device
    rid, pid = scene.proxy_registry.ids_wp(dev)
    cable = scene.cable
    use_apple_sync = (
        cable.apple_body is not None
        and cable.gripper_proxy_apple_joint is not None
        and cable.gripper_proxy_offset_in_apple_frame is not None
    )
    if use_apple_sync:
        layout = getattr(scene, "layout", None)
        if layout is not None and layout.num_envs > 1:
            apple_ids, pos_off, grasp_off = welded_co_teleport_arrays_for_layout(
                layout,
                cable,
                device=str(dev),
                per_world_proxy_offsets=getattr(scene, "per_world_proxy_offsets", None),
            )
            launch_mirror_robot_to_proxy_offset_and_apple(
                robot_ids=rid,
                proxy_ids=pid,
                position_offsets=pos_off,
                apple_body_ids=apple_ids,
                proxy_offset_in_apple=grasp_off,
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
        else:
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
        layout = getattr(scene, "layout", None)
        if layout is not None and layout.num_envs > 1:
            prescribed = tuple(layout.proxy_body_indices) + tuple(
                int(i) for i in layout.apple_body_indices if int(i) >= 0
            )
        else:
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
    stem_harvest_explicit_apple_weight: bool = False
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
    vic_target_positions_wp: wp.array | None = None
    vic_target_rotations_wp: wp.array | None = None
    layout: Any | None = None
    """Optional :class:`~apple_pick_sim.coupled_fruiting.batched_layout.BatchedEnvLayout`."""
    env_spacing: tuple[float, float, float] | None = None
    """Replicate spacing for batched scenes (used when seeding settle-then-weld)."""
    ik_template_robot_model: Any | None = None
    """Single-world robot model kept for IK bootstrap in batched scenes (world_count>1)."""
    per_env_params: Sequence[Any] | None = None
    """Per-world :class:`~apple_pick_sim.fruiting_system.FruitingSystemParams` when heterogeneous."""
    per_world_proxy_offsets: tuple[tuple | None, ...] | None = None
    """Per-env apple-frame grasp offsets for heterogeneous settle-then-weld."""
    stem_harvest_joint_indices_wp: wp.array | None = None
    stem_harvest_tcp_indices_wp: wp.array | None = None
    stem_harvest_apple_indices_wp: wp.array | None = None
    stem_harvest_grasp_offsets_wp: wp.array | None = None
    stem_harvest_apple_masses_wp: wp.array | None = None
    stem_harvest_use_grasp_offset_wp: wp.array | None = None

    def update_fr3_ee_teleop(
        self,
        dt: float,
        controller: (
            fr3_robot.Fr3EEVelocityController
            | fr3_robot.Fr3EEImpedanceController
            | fr3_robot.Fr3BatchedEEImpedanceController
        ),
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        """Integrate EE teleop command (VIC or IK); stage targets for the next substeps.

        Public entry for per-frame teleop in ``example_coupled_fruiting.py`` and
        VIC regression tests. Run once per viewer frame, not per physics substep.
        """
        return _update_fr3_ee_teleop_impl(
            self, dt, controller, viewer=viewer, velocity=velocity
        )

    def update_fr3_ee_teleop_direct(
        self,
        dt: float,
        controller: fr3_robot.Fr3EEDirectJointController,
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        """Direct joint-angle teleop (no VIC); IK + joint command staging.

        Alternative teleop mode when VIC is disabled; writes joint commands
        directly instead of composing an impedance wrench at the TCP.
        """
        return _update_fr3_ee_teleop_impl(
            self, dt, controller, viewer=viewer, velocity=velocity
        )

    def vbd_substep(
        self,
        dt: float,
        *,
        after_cable_clear_forces: Callable[[], None] | None = None,
    ) -> Any:
        """Advance cable VBD one substep; store contacts in ``last_vbd_contacts``.

        Cable-only step (no robot). Used by ``vbd_only`` scenes,
        :func:`~apple_pick_sim.coupled_fruiting.settle_then_weld.settle_vbd_substeps`,
        and as the second half of :meth:`coupled_substep`.
        """
        vbd_contacts = _vbd_substep(
            self.cable,
            self.cable_collision_pipeline,
            dt,
            after_cable_clear_forces=after_cable_clear_forces,
        )
        self.last_vbd_contacts = vbd_contacts
        return vbd_contacts

    def _mujoco_and_sync_proxy(self, dt: float) -> None:
        """MuJoCo robot substep followed by TCP→proxy mirror and optional contact update.

        Combines :func:`_mujoco_robot_substep_prefix` (apply lagged wrench + step
        MuJoCo) with :func:`_sync_single_proxy_after_mujoco` (kinematic cable
        sync). Updates MuJoCo contacts when ``use_mujoco_contacts`` is enabled.
        """
        _mujoco_robot_substep_prefix(self, dt)
        _sync_single_proxy_after_mujoco(self, dt)
        if self.use_mujoco_contacts:
            self.mj_solver.update_contacts(self.mj_contacts, self.robot_state_0)

    def mujoco_substep(self, dt: float) -> None:
        """Robot-only substep: MuJoCo dynamics + proxy kinematic sync.

        For ``mujoco_only`` scenes that skip VBD. Full coupled scenes use
        :meth:`coupled_substep` instead, which calls this as its first phase.
        """
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
        """Full M1 staggered coupling: MuJoCo → snapshot ``qd_synced`` → VBD → harvest wrenches.

        One physics substep in the M1 staggered scheme (``docs/ROADMAP.md``):

        1. MuJoCo robot step + TCP→proxy mirror
        2. Snapshot cable ``body_qd`` into ``qd_synced`` (velocity-delta path only)
        3. VBD cable step
        4. Harvest reaction wrench into ``proxy_forces`` for the next robot step

        Primary integration API for ``example_coupled_fruiting.py``, gym envs,
        and coupling stability / explicit-load tests.
        """
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
