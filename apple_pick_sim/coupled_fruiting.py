"""M1 Slice 2b: staggered ``SolverMuJoCo`` + ``SolverVBD`` coupling around :class:`~apple_pick_sim.fruiting_system.CoupledCableScene`.

The **placeholder TCP robot** is a single free-floating box matching :class:`~apple_pick_sim.fruiting_system.GripperProxyConfig`
mass/shape — swap :func:`build_placeholder_tcp_robot_model` for ``ModelBuilder.add_usd(...)`` when the project USD is ready.

Spatial quantities follow ``apple_pick_sim.proxy_coupling`` conventions (world frame, linear then angular).

Staggered substep (see also ``fruiting_system`` module docstring)
-------------------------------------------------------------------
:meth:`CoupledFruitingScene.coupled_substep` is the authoritative loop.  Two harvest
paths are supported; the active one depends on whether ``fix_to_apple=True`` was used
when building the cable scene.

**Unified-sync / stem-harvest path** (``fix_to_apple=True``, ``stem_apple_joint_index`` set):

    proxy_forces  ──(lagged, step N-1)──►  robot body_f[tcp]  ──►  MuJoCo step
                                                                         │
                                          sync_proxy_and_apple_state ◄──┘
                                          (teleport Proxy AND Apple)
                                                  │
    proxy_forces  ◄── stem joint wrench ──  VBD step (zero proxy-apple violation)

The Proxy and the Apple are teleported together each substep so the FIXED joint
between them experiences zero violation.  VBD stretches only the *stem*, and the
stem-apple constraint force (harvested by :func:`~apple_pick_sim.proxy_coupling.harvest_stem_joint_wrench`)
is fed back to MuJoCo as the external load on the TCP.

**Velocity-delta / free-proxy path** (``fix_to_apple=False``, ``stem_apple_joint_index=None``):

    proxy_forces  ──(lagged, step N-1)──►  robot body_f[tcp]  ──►  MuJoCo step
                                                                         │
    proxy_forces  ◄── velocity-delta  ──  VBD step  ◄── sync_proxy_state ◄┘

- ``proxy_forces``: per-robot-body spatial wrench **harvested** after VBD; consumed
  on the **next** substep as external load on ``robot_state.body_f``.
- ``coupling_forces_cache``: copy of ``proxy_forces`` at apply/sync time so the same
  lagged wrench is both written to MuJoCo and subtracted from proxy velocity in
  the sync kernel (avoids double integration on the VBD side).

Kernels live in ``proxy_coupling.py``; the cable scene (Model B + proxy body) is built
by :func:`~apple_pick_sim.fruiting_system.generate_coupled_cable_scene`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim import fr3_robot
from apple_pick_sim.fruiting_system import (
    CoupledCableScene,
    GripperProxyConfig,
    example_collision_pipeline,
    generate_coupled_cable_scene,
)
from apple_pick_sim.coupling_force_debug import CouplingForceDebugRecorder
from apple_pick_sim.proxy_coupling import (
    align_proxy_body_q_prev_for_vbd,
    harvest_proxy_wrenches,
    harvest_stem_joint_wrench,
    launch_sync_proxy_and_apple_state,
    launch_sync_proxy_state,
)


# Stem-harvest feedback is explicit with one-step lag; under-relax to keep the placeholder loop stable.
DEFAULT_STEM_COUPLING_GAIN: float = 0.15
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
    # Newton collision feeding MuJoCo; CPU MuJoCo path (see ``disable_contacts``).
    "use_mujoco_contacts": False,
    "use_mujoco_cpu": True,
    # Floating TCP placeholder + ground plane can spike constraint residuals at moderate dt.
    "disable_contacts": True,
}

DEFAULT_FR3_MUJOCO_SOLVER_KWARGS: dict[str, Any] = {
    **DEFAULT_MUJOCO_SOLVER_KWARGS,
    "disable_contacts": False,
}


@dataclasses.dataclass
class CoupledFruitingScene:
    """Cable ``SolverVBD`` scene plus optional MuJoCo robot model and coupling buffers.

    **Model A** (``robot_model``, ``mj_solver``): rigid robot integrated by MuJoCo.
    **Model B** (``cable``): fruiting tree + gripper proxy integrated by VBD.
    ``proxy_registry`` maps ``tcp_body_index`` → ``cable.gripper_proxy_body``.
    """

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
    # Harvested VBD→MuJoCo wrench (indexed by robot body id); applied next substep.
    proxy_forces: wp.array | None = None
    # Snapshot of proxy_forces at apply/sync; passed to sync_proxy_state kernel.
    coupling_forces_cache: wp.array | None = None
    # Last contact buffer from ``vbd_substep`` / ``coupled_substep`` (viewer reuse).
    last_vbd_contacts: Any | None = None
    force_debug: CouplingForceDebugRecorder | None = None
    # Joint index of the stem-to-apple FIXED joint; when set, stem-joint harvest
    # replaces velocity-delta harvest and the unified-sync kernel co-teleports the apple.
    stem_apple_joint_index: int | None = None
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM
    gravity_vec: wp.vec3 = dataclasses.field(default_factory=lambda: wp.vec3(0.0, 0.0, -9.81))
    use_mujoco_contacts: bool = False
    robot_disable_contacts: bool = True
    # When True: skip MuJoCo arm integration and lagged wrench on TCP; pose from direct joint_q.
    robot_kinematic_mode: bool = False

    def apply_fr3_ee_teleop(
        self,
        dt: float,
        controller: fr3_robot.Fr3EEVelocityController,
        *,
        viewer: fr3_robot._KeyViewer | None = None,
        velocity: fr3_robot.EEVelocity | None = None,
    ) -> fr3_robot.EEVelocity:
        """Run FR3 TCP velocity teleop, then drive MuJoCo actuators before ``mujoco_substep``.

        IK targets are written to ``robot_control.joint_target_pos`` / ``joint_target_vel`` so
        ``mj_solver.step`` tracks them via the USD position actuators (not by teleporting
        ``joint_q``). Call once per **frame** (``frame_dt``, not substep ``dt``).
        """
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
        """TCP teleop via IK, then **write ``joint_q``** (testing / kinematic arm in coupled runs).

        Use with ``robot_kinematic_mode=True`` so substeps skip ``mj_solver.step`` and do not
        apply lagged coupling wrenches to the arm. Cable VBD + proxy sync still run each substep.
        Call once per **frame** (``frame_dt``).
        """
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
        """Advance the cable ``SolverVBD`` model only (no MuJoCo / proxy sync).

        Cable-only mode (``vbd_only=True``): no force exchange with the robot; the gripper
        proxy is integrated purely by VBD (pose not mirrored from MuJoCo).

        Returns the contact buffer from ``collide`` (for harvest in ``coupled_substep``).
        """
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
        """MuJoCo robot step + ``sync_proxy_state`` onto the cable gripper proxy.

        Coupling steps 1–3 of the staggered protocol (see module docstring):
        apply lagged wrench → integrate robot → mirror pose/vel onto VBD proxy.
        """
        # --- 1–2. Model A: dynamic MuJoCo step, or kinematic FK hold (direct-joint teleop).
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

        # --- 3. Kinematic sync: robot TCP → cable proxy (and apple when fix_to_apple).
        # Two kernels share the same double-integration guard on proxy velocity.
        cable = self.cable
        use_apple_sync = (
            cable.apple_body is not None
            and cable.gripper_proxy_apple_joint is not None
            and cable.gripper_proxy_offset_in_apple_frame is not None
        )
        if use_apple_sync:
            # Unified sync: teleport Proxy AND Apple together so the FIXED joint
            # between them experiences zero violation.  VBD then only needs to
            # accommodate the stem stretch.
            launch_sync_proxy_and_apple_state(
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
            launch_sync_proxy_state(
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
        align_bodies = self.proxy_registry.proxy_body_ids
        if cable.gripper_proxy_apple_joint is not None and cable.apple_body is not None:
            align_bodies = (*align_bodies, cable.apple_body)
        align_proxy_body_q_prev_for_vbd(cable, align_bodies)

        if self.use_mujoco_contacts:
            self.mj_solver.update_contacts(self.mj_contacts, self.robot_state_0)

    def mujoco_substep(self, dt: float) -> None:
        """Advance the placeholder robot in MuJoCo and mirror pose onto the cable proxy.

        Robot-only mode (``mujoco_only=True``): runs steps 1–3 of the staggered protocol
        without a following VBD step or harvest (``proxy_forces`` stays stale).
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
        """One staggered substep: MuJoCo → ``sync_proxy_state`` → VBD → harvest.

        Full five-step coupling (see module docstring). ``proxy_forces`` written here
        is **not** applied until the *next* call (one-step lag).

        ``after_cable_clear_forces`` runs on the cable state immediately after
        ``state_0.clear_forces()`` (e.g. viewer picking forces before the VBD collide/step).
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
        # Snapshot proxy velocity for velocity-delta harvest (free-proxy path only).
        use_stem_harvest = self.stem_apple_joint_index is not None
        qd_synced = None if use_stem_harvest else wp.clone(self.cable.state_0.body_qd)

        # --- 4. Advance Model B (cable + apple + proxy contacts/joints) ---
        vbd_contacts = self.vbd_substep(dt, after_cable_clear_forces=after_cable_clear_forces)

        # --- 5. Harvest: stem joint tension (unified-sync path) or velocity delta (free-proxy) ---
        if use_stem_harvest:
            # Proxy and Apple were co-teleported → FIXED joint has zero violation.
            # Read the stem-apple constraint force: the tension the tree exerts on the apple,
            # which equals the load the apple+stem system imposes on the robot TCP.
            harvest_stem_joint_wrench(
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
            )
        else:
            harvest_proxy_wrenches(
                self.cable.solver,
                self.cable.state_0,
                vbd_contacts,
                dt,
                registry=self.proxy_registry,
                model=self.cable.model,
                qd_synced=qd_synced,
                gravity=self.gravity_vec,
                out_robot_wrenches=self.proxy_forces,
            )
        if self.force_debug is not None:
            self.force_debug.record_harvested_from_scene(self)


def _find_stem_apple_joint(cable: CoupledCableScene) -> int | None:
    """Return the joint index of the stem-to-apple FIXED joint, or ``None``.

    Identified as the FIXED joint in ``fruiting_fixed_joints`` whose **child** body
    is ``cable.apple_body`` (i.e. the joint that attaches the apple to the last rod
    segment).  The proxy-apple joint has the apple as *parent*, so it is not matched.
    """
    if cable.apple_body is None:
        return None
    jchild = cable.model.joint_child.numpy()
    for j_idx, _label in cable.fruiting_fixed_joints:
        if int(jchild[j_idx]) == cable.apple_body:
            return j_idx
    return None


@wp.kernel
def _apply_tcp_spatial_wrench_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    tcp_index: int,
    wrenches: wp.array(dtype=wp.spatial_vector),
):
    """Zero all ``body_f`` slots and write the lagged TCP wrench (device path)."""
    i = wp.tid()
    if i == tcp_index:
        body_f[i] = wrenches[tcp_index]
    else:
        body_f[i] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _apply_spatial_wrench_to_body_f(state: Any, tcp_body_index: int, wrenches_spatial: wp.array) -> None:
    """Write lagged coupling wrench into ``state.body_f[tcp]``; clear all other body_f slots.

    MuJoCo consumes ``body_f`` as external spatial force/torque (world frame, about COM).
    Must run after ``clear_forces()`` each substep; user EE wrenches would be added here too.
    """
    n = int(state.body_f.shape[0])
    dev = state.body_f.device
    wp.launch(
        _apply_tcp_spatial_wrench_kernel,
        dim=n,
        inputs=[state.body_f, int(tcp_body_index), wrenches_spatial],
        device=dev,
    )


def bootstrap_tcp_joint_from_proxy(
    cable_scene: CoupledCableScene,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
) -> None:
    """Align robot generalized coords / FK state with the cable gripper proxy pose and twist."""
    del tcp_body_index  # pose lives in joint coords for this placeholder articulation
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


def _assemble_coupled_robot_scene(
    cable: CoupledCableScene,
    *,
    device: str,
    pipe: Any,
    gravity_vec: wp.vec3,
    robot_model: newton.Model,
    tcp_body: int,
    mj_solver: newton.solvers.SolverMuJoCo,
    mj_kw: dict[str, Any],
    bootstrap_fn: Callable[..., None],
    mujoco_only: bool,
    stem_coupling_gain: float,
    stem_force_cap_N: float | None,
    stem_torque_cap_Nm: float | None,
    init_mujoco_actuator_targets: bool = False,
) -> CoupledFruitingScene:
    robot_state_0 = robot_model.state()
    robot_state_1 = robot_model.state()
    robot_control = robot_model.control()

    bootstrap_fn(cable, robot_model, tcp_body, robot_state_0)
    if init_mujoco_actuator_targets:
        fr3_robot.init_mujoco_actuator_targets_from_model(robot_model, robot_control)
    robot_state_1.joint_q.assign(robot_model.joint_q)
    robot_state_1.joint_qd.assign(robot_model.joint_qd)
    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_1)
    mj_solver._update_mjc_data(mj_solver.mj_data, robot_model, robot_state_0)

    proxy_registry = cable.proxy_registry(tcp_body)
    proxy_forces = wp.zeros(robot_model.body_count, dtype=wp.spatial_vector, device=device)
    coupling_cache = wp.zeros_like(proxy_forces)

    use_mc = bool(mj_kw["use_mujoco_contacts"])
    robot_dc = bool(mj_kw.get("disable_contacts", False))
    # Model A (MuJoCo arm): zero g + sync into mj_model; cable VBD keeps gravity_vec / model g.
    fr3_robot.sync_robot_gravity_to_mujoco(robot_model, mj_solver)
    if use_mc:
        mj_contacts = newton.Contacts(mj_solver.get_max_contact_count(), 0)
    else:
        mj_contacts = robot_model.contacts()

    stem_joint = None
    if (
        cable.gripper_proxy_apple_joint is not None
        and cable.gripper_proxy_offset_in_apple_frame is not None
    ):
        stem_joint = _find_stem_apple_joint(cable)

    return CoupledFruitingScene(
        cable=cable,
        cable_collision_pipeline=pipe,
        vbd_only=False,
        mujoco_only=mujoco_only,
        robot_model=robot_model,
        tcp_body_index=tcp_body,
        mj_solver=mj_solver,
        robot_state_0=robot_state_0,
        robot_state_1=robot_state_1,
        robot_control=robot_control,
        mj_contacts=mj_contacts,
        proxy_registry=proxy_registry,
        proxy_forces=proxy_forces,
        coupling_forces_cache=coupling_cache,
        gravity_vec=gravity_vec,
        use_mujoco_contacts=use_mc,
        robot_disable_contacts=robot_dc,
        stem_apple_joint_index=stem_joint,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
    )


def build_placeholder_tcp_robot_model(
    *,
    gripper_cfg: GripperProxyConfig,
    device: str,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
) -> tuple[newton.Model, int, newton.solvers.SolverMuJoCo]:
    """Placeholder robot: world → FREE joint → TCP box (mass/shape match ``gripper_cfg``).

    Replace this builder with USD import once the FR3 / EE asset is available.

    ``mujoco_solver_kwargs`` is merged onto :data:`DEFAULT_MUJOCO_SOLVER_KWARGS` by this function.
    """
    hx, hy, hz = gripper_cfg.box_half_extents
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    builder.add_ground_plane()

    tcp_body = builder.add_link(
        mass=gripper_cfg.mass,
        label=f"{gripper_cfg.label}_robot_tcp",
    )
    shape_cfg = builder.default_shape_cfg.copy()
    shape_cfg.density = 0.0
    builder.add_shape_box(body=tcp_body, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)

    j_free = builder.add_joint_free(parent=-1, child=tcp_body)
    builder.add_articulation([j_free])

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))

    mj_kw = dict(DEFAULT_MUJOCO_SOLVER_KWARGS)
    mj_kw.update(mujoco_solver_kwargs or {})

    solver = newton.solvers.SolverMuJoCo(
        model,
        njmax=80,
        nconmax=80,
        **mj_kw,
    )
    return model, tcp_body, solver


def build_coupled_fruiting_placeholder(
    ranges: dict,
    seed: int,
    *,
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = True,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    vbd_only: bool = False,
    mujoco_only: bool = False,
) -> CoupledFruitingScene:
    """Construct cable scene and placeholder TCP robot + coupling state.

    Default (``vbd_only=False``, ``mujoco_only=False``): full MuJoCo + VBD staggered loop.
    ``vbd_only=True``: cable ``SolverVBD`` only (no robot model). ``mujoco_only=True``:
    MuJoCo robot + proxy sync only (cable tree not integrated).
    """
    if vbd_only and mujoco_only:
        raise ValueError("vbd_only and mujoco_only are mutually exclusive")
    if device is None:
        device = "cpu"

    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig()

    cable = generate_coupled_cable_scene(
        ranges,
        seed,
        base_pos=base_pos,
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
    )

    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )

    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
        )

    grip_cfg = cable.gripper_proxy_config

    mj_kw = dict(DEFAULT_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)

    robot_model, tcp_body, mj_solver = build_placeholder_tcp_robot_model(
        gripper_cfg=grip_cfg,
        device=device,
        mujoco_solver_kwargs=mj_kw,
    )

    return _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_body,
        mj_solver=mj_solver,
        mj_kw=mj_kw,
        bootstrap_fn=bootstrap_tcp_joint_from_proxy,
        mujoco_only=mujoco_only,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
    )


def build_coupled_fruiting_fr3(
    ranges: dict,
    seed: int,
    *,
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
    device: str | None = None,
    omit: Any | None = None,
    enable_self_collisions: bool = True,
    gripper_proxy: GripperProxyConfig | None = None,
    cable_collision_pipeline: Any | None = None,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
    usd_path: str | Path | None = None,
    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN,
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N,
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM,
    vbd_only: bool = False,
    mujoco_only: bool = False,
    ik_bootstrap_iterations: int = 96,
) -> CoupledFruitingScene:
    """Coupled cable scene + bundled FR3 / custom EE (``SolverMuJoCo`` on ``robot_model``)."""
    if vbd_only and mujoco_only:
        raise ValueError("vbd_only and mujoco_only are mutually exclusive")
    if not fr3_robot.fr3_assets_available():
        raise FileNotFoundError(
            "Bundled FR3 assets missing; see assets/fr3/README.md"
        )
    if device is None:
        device = "cpu"

    if gripper_proxy is None:
        gripper_proxy = GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        )

    cable = generate_coupled_cable_scene(
        ranges,
        seed,
        base_pos=base_pos,
        device=device,
        omit=omit,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
    )
    newton.eval_fk(cable.model, cable.model.joint_q, cable.model.joint_qd, cable.state_0)

    pipe = (
        cable_collision_pipeline
        if cable_collision_pipeline is not None
        else example_collision_pipeline(cable.model, args=None)
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    if vbd_only:
        return CoupledFruitingScene(
            cable=cable,
            cable_collision_pipeline=pipe,
            vbd_only=True,
            gravity_vec=gravity_vec,
        )

    mj_kw = dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS)
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)

    proxy_bq = cable.state_0.body_q.numpy().reshape(-1, 7)[cable.gripper_proxy_body]
    root_xform = fr3_robot.placement_xform_for_proxy(proxy_bq)

    robot_model, tcp_body, mj_solver = fr3_robot.build_fr3_robot_model_from_usd(
        device=device,
        usd_path=usd_path,
        root_xform=root_xform,
        mujoco_solver_kwargs=mj_kw,
    )

    def _bootstrap(
        cable_scene: CoupledCableScene,
        model: newton.Model,
        tcp_idx: int,
        state_0: Any,
    ) -> None:
        bootstrap_articulated_tcp_from_proxy(
            cable_scene,
            model,
            tcp_idx,
            state_0,
            ik_iterations=ik_bootstrap_iterations,
        )

    return _assemble_coupled_robot_scene(
        cable,
        device=device,
        pipe=pipe,
        gravity_vec=gravity_vec,
        robot_model=robot_model,
        tcp_body=tcp_body,
        mj_solver=mj_solver,
        mj_kw=mj_kw,
        bootstrap_fn=_bootstrap,
        mujoco_only=mujoco_only,
        stem_coupling_gain=stem_coupling_gain,
        stem_force_cap_N=stem_force_cap_N,
        stem_torque_cap_Nm=stem_torque_cap_Nm,
        init_mujoco_actuator_targets=True,
    )
