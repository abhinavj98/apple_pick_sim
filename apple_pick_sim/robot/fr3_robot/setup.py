"""FR3 USD import, body resolution, and MuJoCo setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from newton.usd import SchemaResolverMjc, SchemaResolverNewton

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.coupled_fruiting.vic_joint_torques import (
    _N_ARM_DOF,
    allocate_vic_joint_torque_buffers,
)
from apple_pick_sim.coupled_fruiting.vic_joint_torques_batched import (
    allocate_vic_joint_torque_buffers_batched,
)
from apple_pick_sim.robot.fr3_robot.paths import TESTFR3_SCENE_USD, fr3_assets_available

def resolve_tcp_body_index(model: newton.Model) -> int:
    """Return the unique body index for the tcp link. Heuristic: find ``ee``, then see if a direct child ``tcp`` exists."""
    labels = list(model.body_label)

    # First, try to find "ee"
    ee_hits = [
        i
        for i, lbl in enumerate(labels)
        if lbl.endswith("/ee") or lbl.split("/")[-1] == "ee"
    ]
    if len(ee_hits) != 1:
        raise ValueError(f"ambiguous or missing ee in body_label ({len(ee_hits)} hits): {labels}")
    ee_index = ee_hits[0]
    ee_label = labels[ee_index]

    # Now, look for a "tcp" child: "<...>/ee/tcp" or like that
    tcp_hits = [
        i
        for i, lbl in enumerate(labels)
        if lbl.endswith("/tcp") or lbl.split("/")[-1] == "tcp"
        if lbl.startswith(ee_label)
    ]
    if len(tcp_hits) == 1:
        return tcp_hits[0]
    elif len(tcp_hits) == 0:
        # fall back to returning the ee index itself (if no tcp)
        return ee_index
    else:
        raise ValueError(f"ambiguous tcp underneath ee in body_label ({len(tcp_hits)} hits): {labels}")

def sync_robot_gravity_to_mujoco(robot_model: newton.Model, mj_solver: SolverMuJoCo) -> None:
    """Zero Model A gravity and push it into the embedded MuJoCo ``mj_model``.

    Cable VBD (Model B) keeps its own ``cable.model.gravity`` and ``CoupledFruitingScene.gravity_vec``.
    After ``set_gravity``, ``notify_model_changed`` is required so ``mj_model.opt.gravity`` updates.
    """
    robot_model.set_gravity((0.0, 0.0, 0.0))
    mj_solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)


def resolve_ee_body_index(model: newton.Model) -> int:
    """Return the unique body index for the custom end-effector link ``ee``."""
    labels = list(model.body_label)
    for needle in ("/ee", "ee"):
        hits = [
            i
            for i, lbl in enumerate(labels)
            if lbl.endswith(needle) or lbl.split("/")[-1] == needle
        ]
        if len(hits) == 1:
            return hits[0]
    raise ValueError(f"ambiguous or missing ee in body_label: {labels}")


def build_fr3_robot_builder(
    *,
    usd_path: Path | str | None = None,
    root_xform: wp.transform | None = None,
    add_ground_plane: bool = False,
    add_apple_payload: bool = False,
) -> tuple[newton.ModelBuilder, int]:
    """Populate an FR3 USD scene on a builder without ``finalize``.

    When ``add_apple_payload`` is true, appends a mass-only FIXED child of the TCP
    labeled ``apple_payload`` (inertia-only dummy; no shape). See
    ``apple_pick_sim.coupled_fruiting.mujoco_apple_payload``.
    """
    if not fr3_assets_available():
        raise FileNotFoundError(
            f"Bundled FR3 scene or Omniverse subtree missing; see {TESTFR3_SCENE_USD} and assets/fr3/README.md"
        )
    path = Path(usd_path) if usd_path is not None else TESTFR3_SCENE_USD
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    SolverMuJoCo.register_custom_attributes(builder)
    usd_kw: dict[str, Any] = {
        "floating": False,
        "collapse_fixed_joints": False,
        "enable_self_collisions": False,
        "schema_resolvers": [SchemaResolverMjc(), SchemaResolverNewton()],
    }
    if root_xform is not None:
        usd_kw["xform"] = root_xform
        usd_kw["override_root_xform"] = True
    builder.add_usd(str(path), **usd_kw)
    if add_ground_plane:
        builder.add_ground_plane()
    tcp_idx = resolve_tcp_body_index_from_builder(builder)
    if add_apple_payload:
        from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import append_apple_payload_link

        append_apple_payload_link(builder, tcp_idx)
    return builder, tcp_idx


def resolve_tcp_body_index_from_builder(builder: newton.ModelBuilder) -> int:
    """Return TCP body index from ``builder.body_label`` before ``finalize``."""
    labels = list(builder.body_label)
    ee_hits = [
        i
        for i, lbl in enumerate(labels)
        if lbl.endswith("/ee") or lbl.split("/")[-1] == "ee"
    ]
    if len(ee_hits) != 1:
        raise ValueError(f"ambiguous or missing ee in body_label ({len(ee_hits)} hits): {labels}")
    ee_index = ee_hits[0]
    ee_label = labels[ee_index]
    tcp_hits = [
        i
        for i, lbl in enumerate(labels)
        if (lbl.endswith("/tcp") or lbl.split("/")[-1] == "tcp") and lbl.startswith(ee_label)
    ]
    if len(tcp_hits) == 1:
        return tcp_hits[0]
    if len(tcp_hits) == 0:
        return ee_index
    raise ValueError(f"ambiguous tcp underneath ee in body_label ({len(tcp_hits)} hits): {labels}")


def build_fr3_robot_model_from_usd(
    *,
    device: str | None = None,
    usd_path: Path | str | None = None,
    root_xform: wp.transform | None = None,
    add_ground_plane: bool = False,
    add_apple_payload: bool = False,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
) -> tuple[newton.Model, int, SolverMuJoCo]:
    """Build FR3 + Isaac-exported EE/tcp from USD for ``SolverMuJoCo``.

    Default USD is [`assets/testfr3_resolved.usda`] (paired with [`assets/testfr3.usd`] in Omni).

    **Fixed joints** from Isaac (including EE welds) are preserved --- pass
    ``collapse_fixed_joints=False`` implicitly so ``ee`` / ``tcp`` rigid bodies remain.

    To import a **patched binary** [`assets/testfr3.usd`] instead, rewrite its ``fr3``
    payload reference to `./fr3/omniverse_fr3/fr3.usd` so it resolves offline (see
    [`assets/fr3/README.md`]), fix EE joints like ``resolved`` if Newton reports a joint
    cycle, then pass ``usd_path=``.

    When ``add_apple_payload`` is true, the model includes a mass-only FIXED child of
    TCP (``apple_payload``); set inertial props via
    ``apply_mujoco_apple_payload_inertias``.

    Returns ``(model, tcp_body_index, mj_solver)``.
    """
    if not fr3_assets_available():
        raise FileNotFoundError(
            f"Bundled FR3 scene or Omniverse subtree missing; see {TESTFR3_SCENE_USD} and assets/fr3/README.md"
        )

    device = resolve_sim_device(device)
    builder, _ = build_fr3_robot_builder(
        usd_path=usd_path,
        root_xform=root_xform,
        add_ground_plane=add_ground_plane,
        add_apple_payload=add_apple_payload,
    )
    model = builder.finalize(device=device)
    # Model A: zero gravity for teleop/PD hold (cable VBD keeps -9.81 on its own model).
    model.set_gravity((0.0, 0.0, 0.0))

    tcp_idx = resolve_tcp_body_index(model)

    mj_kw: dict[str, Any] = {
        "solver": "newton",
        "integrator": "implicitfast",
        "cone": "elliptic",
        "iterations": 20,
        "ls_iterations": 10,
        "impratio": 1000.0,
        "use_mujoco_contacts": False,
        "use_mujoco_cpu": True,
        "disable_contacts": False,
    }
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)

    solver = SolverMuJoCo(
        model,
        njmax=200,
        nconmax=200,
        **mj_kw,
    )
    return model, tcp_idx, solver


def np_zeros_like_joint_qd(robot_model: newton.Model):
    import numpy as np

    return np.zeros(int(robot_model.joint_dof_count), dtype=np.float32)


def _assign_joint_target_q_from_coords(
    robot_model: newton.Model,
    control: Any,
    joint_q_src: Any,
) -> None:
    """Write ``joint_q``-shaped coords into ``control.joint_target_q`` (coord- or DOF-sized)."""
    import numpy as np

    if control.joint_target_q is None:
        return
    jq = np.asarray(joint_q_src, dtype=np.float32).reshape(-1)
    tgt = control.joint_target_q
    tgt_size = int(tgt.shape[0])
    n_coord = int(robot_model.joint_coord_count)
    n_dof = int(robot_model.joint_dof_count)
    if tgt_size == n_coord:
        tgt.assign(jq[:n_coord])
    elif tgt_size == n_dof:
        tgt.assign(jq[:n_dof])
    else:
        raise RuntimeError(
            f"control.joint_target_q size ({tgt_size}) does not match "
            f"robot joint_coord_count ({n_coord}) or joint_dof_count ({n_dof})"
        )
    if control.joint_target_qd is not None:
        control.joint_target_qd.assign(np_zeros_like_joint_qd(robot_model))


def sync_mujoco_visual_state(
    mj_solver: SolverMuJoCo,
    robot_model: newton.Model,
    state: Any,
) -> None:
    """Push Newton ``joint_q`` into MuJoCo and run forward kinematics for the passive viewer.

    ``_update_mjc_data`` alone updates ``qpos`` but not body poses; run ``mj_forward`` (CPU) or
    ``mujoco_warp.kinematics`` (GPU) before :meth:`~newton.solvers.SolverMuJoCo.render_mujoco_viewer`.
    """
    mj_solver._update_mjc_data(mj_solver.mj_data, robot_model, state)
    if mj_solver.use_mujoco_cpu:
        mj_solver._mujoco.mj_forward(mj_solver.mj_model, mj_solver.mj_data)
    else:
        with wp.ScopedDevice(robot_model.device):
            mj_solver._update_mjc_data(mj_solver.mjw_data, robot_model, state)
            mj_solver._mujoco_warp.kinematics(mj_solver.mjw_model, mj_solver.mjw_data)
            mj_solver._mujoco_warp.get_data_into(
                mj_solver.mj_data, mj_solver.mj_model, mj_solver.mjw_data
            )


def init_mujoco_actuator_targets_from_model(
    robot_model: newton.Model,
    control: Any,
) -> None:
    """Align MuJoCo position actuators with the model's current ``joint_q`` (post-bootstrap)."""
    _assign_joint_target_q_from_coords(robot_model, control, robot_model.joint_q.numpy())


def zero_mujoco_joint_pd(robot_model: newton.Model) -> None:
    """Disable MuJoCo position-actuator stiffness/damping (VIC wrench-only path)."""
    n = int(robot_model.joint_dof_count)
    robot_model.joint_target_ke.assign(np.zeros(n, dtype=np.float32))
    robot_model.joint_target_kd.assign(np.zeros(n, dtype=np.float32))


def _set_vic_passive_joint_damping(
    robot_model: newton.Model,
    vic_joint_damping: float,
    *,
    num_arm_dofs: int = _N_ARM_DOF,
) -> None:
    """Assign ``Model.joint_damping`` on arm DOFs (synced to ``mj_model.dof_damping`` on notify).

    Passive viscous damping absorbs cable-coupling disturbances in null-space modes that
    task-space VIC ``K_d`` cannot see. Matches real FR3 bearing friction (~0.5–2 N·m·s/rad).
    """
    if robot_model.joint_damping is None:
        return
    damping = robot_model.joint_damping.numpy().copy()
    n_arm = min(int(num_arm_dofs), int(damping.shape[0]))
    damping[:n_arm] = float(vic_joint_damping)
    robot_model.joint_damping.assign(damping)


def scale_mujoco_joint_pd(robot_model: newton.Model, scale: float) -> None:
    """Scale existing ``joint_target_ke`` / ``kd`` (e.g. tests that need weak joint holding)."""
    ke = robot_model.joint_target_ke.numpy().astype(np.float32) * float(scale)
    kd = robot_model.joint_target_kd.numpy().astype(np.float32) * float(scale)
    robot_model.joint_target_ke.assign(ke)
    robot_model.joint_target_kd.assign(kd)


def hold_mujoco_actuator_targets_at_state(
    robot_model: newton.Model,
    state: Any,
    control: Any,
) -> None:
    """Hold position actuators at the simulated ``joint_q`` with zero target velocity."""
    _assign_joint_target_q_from_coords(robot_model, control, state.joint_q.numpy())


def configure_vic_joint_torques_arm(
    robot_model: newton.Model,
    state: Any,
    control: Any,
    mj_solver: SolverMuJoCo,
    *,
    scene: Any | None = None,
    tcp_body_index: int | None = None,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
    singularity_damping: float = 0.0,
    vic_joint_damping: float = 1.0,
) -> None:
    """One-shot setup for post-grasp VIC via ``control.joint_f`` (J^T Λ wrench mapping)."""
    zero_mujoco_joint_pd(robot_model)
    _set_vic_passive_joint_damping(robot_model, vic_joint_damping)
    mj_solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    hold_mujoco_actuator_targets_at_state(robot_model, state, control)
    if control.joint_f is None:
        n = int(robot_model.joint_dof_count)
        control.joint_f = wp.zeros(n, dtype=float, device=robot_model.device)
    if scene is not None:
        tcp = (
            int(tcp_body_index)
            if tcp_body_index is not None
            else int(getattr(scene, "tcp_body_index", resolve_tcp_body_index(robot_model)))
        )
        allocate_vic_joint_torque_buffers(
            robot_model,
            scene,
            tcp_body_index=tcp,
            kp_null=kp_null,
            kd_null=kd_null,
            singularity_damping=singularity_damping,
        )


def configure_vic_joint_torques_arm_batched(
    robot_model: newton.Model,
    state: Any,
    control: Any,
    mj_solver: SolverMuJoCo,
    *,
    scene: Any,
    layout: Any,
    tcp_body_index: int | None = None,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
    singularity_damping: float = 0.0,
    vic_joint_damping: float = 1.0,
) -> None:
    """One-shot batched VIC setup: zero PD, ``joint_f`` for all worlds, batched J/H buffers."""
    zero_mujoco_joint_pd(robot_model)
    _set_vic_passive_joint_damping(robot_model, vic_joint_damping)
    mj_solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    hold_mujoco_actuator_targets_at_state(robot_model, state, control)
    if control.joint_f is None:
        n = int(robot_model.joint_dof_count)
        control.joint_f = wp.zeros(n, dtype=float, device=robot_model.device)
    if tcp_body_index is not None:
        tcp = int(tcp_body_index)
    elif getattr(scene, "tcp_body_index", None) is not None:
        tcp = int(scene.tcp_body_index)
    else:
        tcp = int(layout.tcp_body_indices[0])
    allocate_vic_joint_torque_buffers_batched(
        robot_model,
        scene,
        layout,
        tcp_body_index=tcp,
        kp_null=kp_null,
        kd_null=kd_null,
        singularity_damping=singularity_damping,
    )


def configure_vic_wrench_only_arm(
    robot_model: newton.Model,
    state: Any,
    control: Any,
    mj_solver: SolverMuJoCo,
) -> None:
    """One-shot setup for post-grasp VIC: no joint PD, actuator targets track current pose."""
    zero_mujoco_joint_pd(robot_model)
    # Push zeroed joint_target_ke/kd into the embedded MuJoCo model (otherwise ~3.4e6
    # N/m position actuators still lock the arm and TCP wrenches cannot integrate).
    mj_solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    hold_mujoco_actuator_targets_at_state(robot_model, state, control)


def sync_mujoco_actuator_targets_from_joint_q(
    robot_model: newton.Model,
    state: Any,
    control: Any,
    target_joint_q: Any,
    *,
    frame_dt: float,
    command_velocity: EEVelocity | None = None,
) -> None:
    """Write IK joint targets into ``control`` for ``SolverMuJoCo`` position actuators.

    MuJoCo integrates the arm toward ``joint_target_pos`` each ``mj_solver.step``; do not
    teleport ``state.joint_q`` when using this path.

    When ``command_velocity`` is zero (keyboard idle), ``joint_target_vel`` is set to zero
    so PD does not chase a perpetual ``(q_tgt - q_cur) / frame_dt`` feedforward.
    """
    import numpy as np

    n_dof = int(robot_model.joint_dof_count)
    q_tgt = np.asarray(target_joint_q, dtype=np.float32).reshape(-1)[:n_dof]
    q_cur = state.joint_q.numpy().reshape(-1).astype(np.float32)[:n_dof]
    if command_velocity is not None and command_velocity.is_zero():
        qd_tgt = np.zeros(n_dof, dtype=np.float32)
        if float(np.linalg.norm(q_tgt[:n_dof] - q_cur[:n_dof])) < 0.02:
            q_tgt = q_cur.copy()
    elif frame_dt > 1e-9:
        qd_tgt = (q_tgt - q_cur) / float(frame_dt)
    else:
        qd_tgt = np.zeros(n_dof, dtype=np.float32)
    target_q = getattr(control, "joint_target_q", None)
    if target_q is not None:
        tgt_size = int(target_q.shape[0])
        n_coord = int(robot_model.joint_coord_count)
        if tgt_size == n_coord:
            q_tgt = np.asarray(target_joint_q, dtype=np.float32).reshape(-1)[:n_coord]
            q_cur = state.joint_q.numpy().reshape(-1).astype(np.float32)[:n_coord]
            if command_velocity is not None and command_velocity.is_zero():
                if float(np.linalg.norm(q_tgt - q_cur)) < 0.02:
                    q_tgt = q_cur.copy()
            elif frame_dt > 1e-9:
                qd_tgt = (
                    np.asarray(target_joint_q, dtype=np.float32).reshape(-1)[:n_dof]
                    - state.joint_q.numpy().reshape(-1).astype(np.float32)[:n_dof]
                ) / float(frame_dt)
        target_q.assign(q_tgt)
        if control.joint_target_qd is not None:
            control.joint_target_qd.assign(qd_tgt.astype(np.float32))
        return
    control.joint_target_pos.assign(q_tgt)
    control.joint_target_vel.assign(qd_tgt.astype(np.float32))
