"""SolverVBD fixed-joint wrench readout (child body, world frame at COM).

Uses :meth:`newton.solvers.SolverVBD.gather_joint_wrench_child_com`, which evaluates the
same AVBD joint force/torque pair as the rigid VBD ``evaluate_joint_force_hessian`` path
for ``body_index == joint_child``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import warp as wp

import newton


@dataclasses.dataclass
class FixedJointWrenchRecord:
    """Wrench on the child body at COM for one fixed joint (post-step readout)."""

    joint_index: int
    label: str
    child_body: int
    force_world: np.ndarray  # (3,) float32, linear force [N]
    torque_at_child_com_world: np.ndarray  # (3,) float32, torque [N·m]


def iter_fixed_joint_indices(model: newton.Model) -> list[tuple[int, str]]:
    """Return ``(joint_index, label)`` for joints whose label starts with ``joint_`` and type is FIXED.

    Prefer :func:`apple_pick_sim.fruiting_system.iter_fruiting_fixed_joint_indices` for
    scenes built by :func:`~apple_pick_sim.fruiting_system.generate_scene`, which uses
    explicit joint metadata instead of this heuristic.
    """
    jt = model.joint_type.numpy()
    out: list[tuple[int, str]] = []
    for j, label in enumerate(model.joint_label):
        if not label.startswith("joint_"):
            continue
        if int(jt[j]) != int(newton.JointType.FIXED):
            continue
        out.append((j, label))
    return out


def fixed_joint_wrenches_child_com_vbd(
    model: newton.Model,
    solver: newton.solvers.SolverVBD,
    *,
    body_q: Any,
    body_q_prev: Any,
    dt: float,
    control: newton.Control | None = None,
    joint_pairs: list[tuple[int, str]] | None = None,
) -> list[FixedJointWrenchRecord]:
    """Return per-fixed-joint wrenches for the given joints (child at COM, world frame).

    Args:
        model: Finalized Newton model.
        solver: The :class:`~newton.solvers.SolverVBD` used for the step that produced ``body_q``.
        body_q: Post-step body transforms (numpy or ``wp.array``), same convention as
            :meth:`newton.solvers.SolverVBD.gather_joint_wrench_child_com`.
        body_q_prev: Pre-step body transforms for the same macro-step (world frame).
        dt: Step size [s].
        control: Optional control buffer passed to the gather API.
        joint_pairs: Optional explicit ``(joint_index, label)`` list (e.g. from
            :attr:`apple_pick_sim.fruiting_system.FruitingSystemScene.fruiting_fixed_joints`).
            If ``None``, uses :func:`iter_fixed_joint_indices`.

    Returns:
        One :class:`FixedJointWrenchRecord` per joint in ``joint_pairs`` order (or heuristic order).
    """
    if joint_pairs is None:
        pairs = list(iter_fixed_joint_indices(model))
    else:
        pairs = list(joint_pairs)
    if not pairs:
        return []

    indices = [j for j, _ in pairs]
    f_np, t_np = solver.gather_joint_wrench_child_com(
        model, body_q, body_q_prev, indices, dt, control=control
    )
    jchild = model.joint_child.numpy()
    out: list[FixedJointWrenchRecord] = []
    for i, (j, lab) in enumerate(pairs):
        out.append(
            FixedJointWrenchRecord(
                joint_index=j,
                label=lab,
                child_body=int(jchild[j]),
                force_world=np.asarray(f_np[i], dtype=np.float32),
                torque_at_child_com_world=np.asarray(t_np[i], dtype=np.float32),
            )
        )
    return out


def _as_body_q_on_solver_device(solver: newton.solvers.SolverVBD, x: Any) -> wp.array:
    device = solver.device
    if isinstance(x, wp.array):
        if x.dtype != wp.transform:
            raise TypeError("body_q / body_q_prev must be wp.array(dtype=wp.transform) or numpy-like.")
        return x.to(device) if x.device != device else x
    return wp.array(x, dtype=wp.transform, device=device)


def gather_joint_wrench_child_com_device(
    model: newton.Model,
    solver: newton.solvers.SolverVBD,
    *,
    body_q: Any,
    body_q_prev: Any,
    joint_indices: Sequence[int] | np.ndarray,
    dt: float,
    control: newton.Control | None = None,
) -> tuple[wp.array, wp.array]:
    """Device-resident joint wrenches (mirrors :meth:`SolverVBD.gather_joint_wrench_child_com`).

    Returns ``(force_world, torque_world)`` as ``wp.vec3`` arrays on ``solver.device``.
    """
    if model is not solver.model:
        raise ValueError("gather_joint_wrench_child_com_device: model must match the VBD solver model.")
    if model.body_count == 0 or solver.integrate_with_external_rigid_solver:
        raise ValueError("gather_joint_wrench_child_com_device requires VBD-integrated rigid bodies.")

    n = len(joint_indices)
    if n == 0:
        z = wp.zeros(0, dtype=wp.vec3, device=solver.device)
        return z, z

    if control is None:
        control = model.control(clone_variables=False)

    device = solver.device
    body_q_d = _as_body_q_on_solver_device(solver, body_q)
    body_q_prev_d = _as_body_q_on_solver_device(solver, body_q_prev)
    j_idx = wp.array(np.asarray(joint_indices, dtype=np.int32), dtype=wp.int32, device=device)
    out_f = wp.zeros(n, dtype=wp.vec3, device=device)
    out_t = wp.zeros(n, dtype=wp.vec3, device=device)

    from newton._src.solvers.vbd.rigid_vbd_kernels import (  # noqa: PLC0415
        gather_joint_wrench_child_at_com_kernel,
    )

    wp.launch(
        kernel=gather_joint_wrench_child_at_com_kernel,
        dim=n,
        inputs=[
            j_idx,
            body_q_d,
            body_q_prev_d,
            model.body_q,
            model.body_com,
            model.joint_type,
            model.joint_enabled,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            solver.joint_constraint_start,
            solver.joint_penalty_k,
            solver.joint_penalty_kd,
            solver.joint_sigma_start,
            solver.joint_C_fric,
            model.joint_target_ke,
            model.joint_target_kd,
            control.joint_target_pos,
            control.joint_target_vel,
            model.joint_limit_lower,
            model.joint_limit_upper,
            model.joint_limit_ke,
            model.joint_limit_kd,
            solver.joint_lambda_lin,
            solver.joint_lambda_ang,
            solver.joint_C0_lin,
            solver.joint_C0_ang,
            solver.joint_is_hard,
            float(solver.rigid_joint_alpha),
            model.joint_dof_dim,
            solver.joint_rest_angle,
            float(dt),
            out_f,
            out_t,
        ],
        device=device,
    )
    return out_f, out_t
