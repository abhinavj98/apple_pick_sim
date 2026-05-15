"""SolverVBD fixed-joint wrench readout (child body, world frame at COM).

Uses :meth:`newton.solvers.SolverVBD.gather_joint_wrench_child_com`, which evaluates the
same AVBD joint force/torque pair as the rigid VBD ``evaluate_joint_force_hessian`` path
for ``body_index == joint_child``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

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
