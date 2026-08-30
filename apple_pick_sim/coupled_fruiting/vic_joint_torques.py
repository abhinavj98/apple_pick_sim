"""VIC via dynamically-consistent joint torques (J^T Λ wrench + null-space).

Uses Newton GPU ``eval_jacobian`` / ``eval_mass_matrix`` and PyTorch for 6×6 /
7×7 linear algebra via ``wp.to_torch`` zero-copy views.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity

_N_ARM_DOF = 7
_TORQUE_CLAMP = 100000.0


def mass_matrix_with_armature(
    mass_matrix: np.ndarray,
    armature: np.ndarray,
) -> np.ndarray:
    """Return ``M + diag(armature)`` for the leading arm DOFs.

    ``newton.eval_mass_matrix`` is body inertia only; reflected motor inertia must
    be added so operational-space ``Λ = (J M⁻¹ Jᵀ)⁻¹`` matches MuJoCo.
    """
    M = np.asarray(mass_matrix, dtype=np.float64)
    arm = np.asarray(armature, dtype=np.float64).reshape(-1)
    n = int(arm.shape[0])
    if M.shape[-1] < n or M.shape[-2] < n:
        raise ValueError(f"mass matrix {M.shape} is smaller than armature length {n}")
    out = np.array(M, dtype=np.float64, copy=True)
    if out.ndim == 2:
        out[:n, :n] += np.diag(arm)
        return out
    if out.ndim == 3:
        out[:, :n, :n] += np.diag(arm)
        return out
    raise ValueError(f"mass matrix must be 2-D or 3-D, got shape {out.shape}")


def _mass_matrix_with_model_armature_torch(
    torch: Any,
    mass_matrix: Any,
    model: Any,
    *,
    num_arm_dofs: int = _N_ARM_DOF,
):
    """Add ``model.joint_armature[:num_arm_dofs]`` onto a torch mass matrix."""
    if getattr(model, "joint_armature", None) is None:
        return mass_matrix
    arm = torch.as_tensor(
        model.joint_armature.numpy().reshape(-1)[:num_arm_dofs],
        device=mass_matrix.device,
        dtype=mass_matrix.dtype,
    )
    return mass_matrix + torch.diag(arm)
_TORCH_INSTALL_HINT = (
    "PyTorch is required for VIC joint torques. "
    "Install from newton/: uv sync --extra torch-cu12"
)


def _require_torch():
    """Lazy-import PyTorch; raise with install hint if the torch extra is missing.

    VIC joint-torque mode needs PyTorch for 6×6 / 7×7 linear algebra on
    ``wp.to_torch`` views. Called at the start of
    :func:`compute_joint_torques_from_wrench_torch` and
    :func:`launch_apply_vic_joint_torques`. Install via
    ``uv sync --extra torch-cu12`` from ``newton/``.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_TORCH_INSTALL_HINT) from exc
    return torch


def find_tcp_link_idx(model: newton.Model, tcp_body_index: int, *, art_idx: int = 0) -> int:
    """Return articulation link row index ``L`` for ``tcp_body_index``."""
    if model.articulation_count == 0:
        raise ValueError("model has no articulations")
    joint_child = model.joint_child.numpy()
    art_start = model.articulation_start.numpy()
    js = int(art_start[art_idx])
    je = int(art_start[art_idx + 1])
    tcp = int(tcp_body_index)
    for j in range(js, je):
        if int(joint_child[j]) == tcp:
            return j - js
    raise ValueError(f"tcp body {tcp} not found in articulation {art_idx}")


def compute_joint_torques_from_wrench_numpy(
    task_wrench: np.ndarray,
    jacobian: np.ndarray,
    mass_matrix: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    default_dof_pos: np.ndarray,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
    singularity_damping: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU reference for ``J^T Λ wrench`` with null-space compensation."""
    task_wrench = np.asarray(task_wrench, dtype=np.float64).reshape(6)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
    joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
    joint_vel = np.asarray(joint_vel, dtype=np.float64).reshape(-1)
    default_dof_pos = np.asarray(default_dof_pos, dtype=np.float64).reshape(-1).copy()
    if default_dof_pos.shape[0] > 6:
        default_dof_pos[6] = 0.0

    jacobian_T = jacobian.T
    M_inv = np.linalg.inv(mass_matrix)
    JMJ_full = jacobian @ M_inv @ jacobian.T
    if singularity_damping > 0.0:
        JMJ_full = JMJ_full + singularity_damping * np.eye(6)
    M_task_full = np.linalg.inv(JMJ_full)
    jt_torque = jacobian_T @ M_task_full @ task_wrench

    J_inv = M_task_full @ jacobian @ M_inv
    dist = default_dof_pos - joint_pos
    dist = (dist + math.pi) % (2.0 * math.pi) - math.pi
    u_null = kd_null * (-joint_vel) + kp_null * dist
    u_null = mass_matrix @ u_null
    null_proj = np.eye(joint_pos.shape[0]) - jacobian_T @ J_inv
    null_torque = null_proj @ u_null
    torque = np.clip(jt_torque + null_torque, -_TORQUE_CLAMP, _TORQUE_CLAMP)
    return torque, jt_torque, null_torque


def compute_joint_torques_from_wrench_torch(
    task_wrench,
    jacobian,
    mass_matrix,
    joint_pos,
    joint_vel,
    default_dof_pos,
    *,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
    singularity_damping: float = 0.0,
    dtype=None,
):
    """PyTorch ``J^T Λ wrench`` with null-space compensation (mirrors NumPy reference)."""
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float64

    task_wrench = task_wrench.reshape(6).to(dtype=dtype)
    jacobian = jacobian.reshape(6, -1).to(dtype=dtype)
    mass_matrix = mass_matrix.reshape(jacobian.shape[1], jacobian.shape[1]).to(dtype=dtype)
    joint_pos = joint_pos.reshape(-1).to(dtype=dtype)
    joint_vel = joint_vel.reshape(-1).to(dtype=dtype)
    default_dof_pos = default_dof_pos.reshape(-1).to(dtype=dtype).clone()
    if default_dof_pos.shape[0] > 6:
        default_dof_pos[6] = 0.0

    jacobian_T = jacobian.T
    M_inv = torch.linalg.inv(mass_matrix)
    JMJ_full = jacobian @ M_inv @ jacobian.T
    if singularity_damping > 0.0:
        JMJ_full = JMJ_full + singularity_damping * torch.eye(6, device=JMJ_full.device, dtype=dtype)
    M_task_full = torch.linalg.inv(JMJ_full)
    jt_torque = jacobian_T @ M_task_full @ task_wrench

    J_inv = M_task_full @ jacobian @ M_inv
    dist = default_dof_pos - joint_pos
    dist = (dist + math.pi) % (2.0 * math.pi) - math.pi
    u_null = kd_null * (-joint_vel) + kp_null * dist
    u_null = mass_matrix @ u_null
    n_dof = joint_pos.shape[0]
    null_proj = torch.eye(n_dof, device=joint_pos.device, dtype=dtype) - jacobian_T @ J_inv
    null_torque = null_proj @ u_null
    torque = torch.clamp(jt_torque + null_torque, -_TORQUE_CLAMP, _TORQUE_CLAMP)
    return torque, jt_torque, null_torque


def allocate_vic_joint_torque_buffers(
    model: newton.Model,
    scene: Any,
    *,
    tcp_body_index: int,
    art_idx: int = 0,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
    singularity_damping: float = 0.0,
) -> None:
    """Pre-allocate Jacobian/mass buffers on ``scene`` for per-substep reuse."""
    dev = model.device
    max_links = int(model.max_joints_per_articulation)
    max_dofs = int(model.max_dofs_per_articulation)
    scene.vic_jt_J_buf = wp.empty((1, max_links * 6, max_dofs), dtype=float, device=dev)
    scene.vic_jt_H_buf = wp.empty((1, max_dofs, max_dofs), dtype=float, device=dev)
    scene.vic_jt_tcp_link_idx = int(find_tcp_link_idx(model, tcp_body_index, art_idx=art_idx))
    scene.vic_jt_art_idx = int(art_idx)
    default_q = model.joint_q.numpy().reshape(-1)[:_N_ARM_DOF].astype(np.float32).copy()
    default_q[6] = 0.0
    scene.vic_jt_default_dof_pos = wp.array(default_q, dtype=float, device=dev)
    scene.vic_jt_kp_null = float(kp_null)
    scene.vic_jt_kd_null = float(kd_null)
    scene.vic_jt_singularity_damping = float(singularity_damping)


def launch_apply_vic_joint_torques(
    scene: Any,
    *,
    target_tf: wp.transform,
    target_twist: EEVelocity,
    gains: ImpedanceGains | None = None,
) -> None:
    """Evaluate Jacobian/mass on device and write VIC joint torques to ``control.joint_f``."""
    torch = _require_torch()

    if (
        scene.robot_model is None
        or scene.robot_state_0 is None
        or scene.robot_control is None
        or scene.robot_control.joint_f is None
        or scene.vic_jt_J_buf is None
        or scene.vic_jt_H_buf is None
    ):
        return

    g = gains if gains is not None else ImpedanceGains()
    model = scene.robot_model
    state = scene.robot_state_0
    control = scene.robot_control
    dev = control.joint_f.device
    torch_device = wp.device_to_torch(dev)
    art_idx = int(scene.vic_jt_art_idx)
    link_idx = int(scene.vic_jt_tcp_link_idx)
    row0 = link_idx * 6
    row1 = row0 + 6

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    newton.eval_jacobian(model, state, J=scene.vic_jt_J_buf)
    newton.eval_mass_matrix(model, state, H=scene.vic_jt_H_buf, J=scene.vic_jt_J_buf)

    J_th = wp.to_torch(scene.vic_jt_J_buf)[art_idx, row0:row1, :_N_ARM_DOF].to(
        device=torch_device, dtype=torch.float64
    )
    M_th = wp.to_torch(scene.vic_jt_H_buf)[art_idx, :_N_ARM_DOF, :_N_ARM_DOF].to(
        device=torch_device, dtype=torch.float64
    )
    M_th = _mass_matrix_with_model_armature_torch(torch, M_th, model)
    q_th = wp.to_torch(state.joint_q)[:_N_ARM_DOF].to(device=torch_device, dtype=torch.float64)
    qd_th = wp.to_torch(state.joint_qd)[:_N_ARM_DOF].to(device=torch_device, dtype=torch.float64)
    default_q_th = wp.to_torch(scene.vic_jt_default_dof_pos).reshape(-1)[:_N_ARM_DOF].to(
        device=torch_device, dtype=torch.float64
    )

    tcp = int(scene.tcp_body_index)
    bq = state.body_q.numpy().reshape(-1, 7)[tcp]
    bqd = state.body_qd.numpy().reshape(-1, 6)[tcp]
    vic = getattr(scene, "vic_controller", None) or Fr3EEImpedanceController()
    wrench = vic.compute_applied_wrench(
        target_tf=target_tf,
        target_twist=target_twist,
        tcp_body_q=bq,
        tcp_body_qd=bqd,
        gains=g,
    )
    wrench_th = torch.as_tensor(wrench, device=torch_device, dtype=torch.float64)

    tau, _, _ = compute_joint_torques_from_wrench_torch(
        task_wrench=wrench_th,
        jacobian=J_th,
        mass_matrix=M_th,
        joint_pos=q_th,
        joint_vel=qd_th,
        default_dof_pos=default_q_th,
        kp_null=float(getattr(scene, "vic_jt_kp_null", 10.0)),
        kd_null=float(getattr(scene, "vic_jt_kd_null", 6.3246)),
        singularity_damping=float(getattr(scene, "vic_jt_singularity_damping", 0.0)),
    )

    joint_f_th = wp.to_torch(control.joint_f)
    joint_f_th.reshape(-1)[:_N_ARM_DOF] = tau.to(dtype=joint_f_th.dtype)


def apply_vic_joint_torques_to_scene(scene: Any) -> None:
    """Write VIC joint torques when ``vic_controller`` and targets are configured."""
    if getattr(scene, "vic_controller", None) is None:
        return
    target_tf = getattr(scene, "vic_target_tf", None)
    target_twist = getattr(scene, "vic_target_twist", None)
    if target_tf is None or target_twist is None:
        return
    launch_apply_vic_joint_torques(
        scene,
        target_tf=target_tf,
        target_twist=target_twist,
        gains=getattr(scene, "vic_gains", None),
    )
