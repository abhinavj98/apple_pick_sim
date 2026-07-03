"""Batched VIC joint torques: vectorized wrench kernel + batched PyTorch J^T Λ w."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.vic_joint_torques import (
    _N_ARM_DOF,
    _require_torch,
    find_tcp_link_idx,
)
from apple_pick_sim.coupled_fruiting.vic_wrench import compute_vic_spatial_wrench
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity

_TORQUE_CLAMP = 100000.0


@wp.func
def _vec4_to_quat(v: wp.vec4) -> wp.quat:
    return wp.quat(v[0], v[1], v[2], v[3])


@wp.kernel(enable_backward=False)
def _compute_vic_wrenches_batched_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    tcp_indices: wp.array(dtype=int),
    target_positions: wp.array(dtype=wp.vec3),
    target_rotations: wp.array(dtype=wp.vec4),
    v_des: wp.array(dtype=wp.vec3),
    w_des: wp.array(dtype=wp.vec3),
    linear_k: float,
    linear_d: float,
    angular_k: float,
    angular_d: float,
    wrenches_out: wp.array2d(dtype=float),
):
    """Per-env impedance wrench at TCP COM (world frame)."""
    w = wp.tid()
    tcp_idx = tcp_indices[w]
    target_tf = wp.transform(target_positions[w], _vec4_to_quat(target_rotations[w]))
    wrench = compute_vic_spatial_wrench(
        body_q[tcp_idx],
        body_qd[tcp_idx],
        target_tf,
        v_des[w],
        w_des[w],
        linear_k,
        linear_d,
        angular_k,
        angular_d,
    )
    wrenches_out[w, 0] = wrench[0]
    wrenches_out[w, 1] = wrench[1]
    wrenches_out[w, 2] = wrench[2]
    wrenches_out[w, 3] = wrench[3]
    wrenches_out[w, 4] = wrench[4]
    wrenches_out[w, 5] = wrench[5]


def compute_joint_torques_from_wrench_torch_batched(
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
    """Batched ``J^T Λ wrench`` with null-space compensation (``N`` articulations)."""
    torch = _require_torch()
    if dtype is None:
        dtype = torch.float64

    task_wrench = task_wrench.reshape(-1, 6).to(dtype=dtype)
    n = int(task_wrench.shape[0])
    jacobian = jacobian.reshape(n, 6, _N_ARM_DOF).to(dtype=dtype)
    mass_matrix = mass_matrix.reshape(n, _N_ARM_DOF, _N_ARM_DOF).to(dtype=dtype)
    joint_pos = joint_pos.reshape(n, _N_ARM_DOF).to(dtype=dtype)
    joint_vel = joint_vel.reshape(n, _N_ARM_DOF).to(dtype=dtype)
    default_dof_pos = default_dof_pos.reshape(n, _N_ARM_DOF).to(dtype=dtype).clone()
    default_dof_pos[:, 6] = 0.0

    jacobian_T = jacobian.transpose(-1, -2)
    M_inv = torch.linalg.inv(mass_matrix)
    JMJ_full = jacobian @ M_inv @ jacobian_T
    if singularity_damping > 0.0:
        eye6 = torch.eye(6, device=JMJ_full.device, dtype=dtype).expand(n, 6, 6)
        JMJ_full = JMJ_full + singularity_damping * eye6
    M_task_full = torch.linalg.inv(JMJ_full)
    jt_torque = (jacobian_T @ M_task_full @ task_wrench.unsqueeze(-1)).squeeze(-1)

    J_inv = M_task_full @ jacobian @ M_inv
    dist = default_dof_pos - joint_pos
    dist = (dist + math.pi) % (2.0 * math.pi) - math.pi
    u_null = kd_null * (-joint_vel) + kp_null * dist
    u_null = (mass_matrix @ u_null.unsqueeze(-1)).squeeze(-1)
    eye7 = torch.eye(_N_ARM_DOF, device=joint_pos.device, dtype=dtype).expand(n, _N_ARM_DOF, _N_ARM_DOF)
    null_proj = eye7 - jacobian_T @ J_inv
    null_torque = (null_proj @ u_null.unsqueeze(-1)).squeeze(-1)
    torque = torch.clamp(jt_torque + null_torque, -_TORQUE_CLAMP, _TORQUE_CLAMP)
    return torque, jt_torque, null_torque


def allocate_vic_joint_torque_buffers_batched(
    model: newton.Model,
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    tcp_body_index: int | None = None,
    kp_null: float = 10.0,
    kd_null: float = 6.3246,
    singularity_damping: float = 0.0,
) -> None:
    """Pre-allocate batched Jacobian/mass buffers for ``layout.num_envs`` articulations."""
    num_envs = int(layout.num_envs)
    dev = model.device
    max_links = int(model.max_joints_per_articulation)
    max_dofs = int(model.max_dofs_per_articulation)
    scene.vic_jt_J_buf = wp.empty((num_envs, max_links * 6, max_dofs), dtype=float, device=dev)
    scene.vic_jt_H_buf = wp.empty((num_envs, max_dofs, max_dofs), dtype=float, device=dev)
    tcp = int(tcp_body_index if tcp_body_index is not None else layout.tcp_body_indices[0])
    scene.vic_jt_tcp_link_idx = int(find_tcp_link_idx(model, tcp, art_idx=0))
    scene.vic_jt_num_envs = num_envs
    scene.vic_jt_tcp_indices_wp = wp.array(list(layout.tcp_body_indices), dtype=int, device=dev)
    scene.vic_jt_wrench_buf = wp.zeros((num_envs, 6), dtype=float, device=dev)

    default_rows = []
    jq = model.joint_q.numpy().reshape(-1)
    for w in range(num_envs):
        sl = layout.joint_q_slice(w)
        row = jq[sl][: _N_ARM_DOF].astype(np.float32).copy()
        row[6] = 0.0
        default_rows.append(row)
    scene.vic_jt_default_dof_pos_batched = wp.array(
        np.stack(default_rows, axis=0), dtype=float, device=dev
    )
    scene.vic_jt_kp_null = float(kp_null)
    scene.vic_jt_kd_null = float(kd_null)
    scene.vic_jt_singularity_damping = float(singularity_damping)


def _resolve_batched_vic_desired_twists(scene: Any, num_envs: int, dev: Any):
    """Return per-env ``(v_des, w_des)`` device arrays for the wrench kernel."""
    lin = getattr(scene, "vic_target_linear_vels_wp", None)
    ang = getattr(scene, "vic_target_angular_vels_wp", None)
    if lin is not None and ang is not None:
        return lin, ang

    twist: EEVelocity = getattr(scene, "vic_target_twist", None) or EEVelocity()
    v_broadcast = wp.full(num_envs, wp.vec3(*twist.linear), dtype=wp.vec3, device=dev)
    w_broadcast = wp.full(num_envs, wp.vec3(*twist.angular), dtype=wp.vec3, device=dev)
    return v_broadcast, w_broadcast


def launch_compute_vic_wrenches_batched(
    scene: Any,
    *,
    gains: ImpedanceGains | None = None,
) -> None:
    """Fill ``vic_jt_wrench_buf`` from device TCP state and staged per-env targets."""
    if (
        scene.robot_state_0 is None
        or scene.vic_jt_wrench_buf is None
        or scene.vic_jt_tcp_indices_wp is None
    ):
        return
    target_positions = getattr(scene, "vic_target_positions_wp", None)
    target_rotations = getattr(scene, "vic_target_rotations_wp", None)
    if target_positions is None or target_rotations is None:
        return
    g = gains if gains is not None else ImpedanceGains()
    num_envs = int(scene.vic_jt_num_envs)
    dev = scene.robot_state_0.body_q.device
    v_des_wp, w_des_wp = _resolve_batched_vic_desired_twists(scene, num_envs, dev)
    wp.launch(
        _compute_vic_wrenches_batched_kernel,
        dim=num_envs,
        inputs=[
            scene.robot_state_0.body_q,
            scene.robot_state_0.body_qd,
            scene.vic_jt_tcp_indices_wp,
            target_positions,
            target_rotations,
            v_des_wp,
            w_des_wp,
            float(g.linear_k),
            float(g.linear_d),
            float(g.angular_k),
            float(g.angular_d),
            scene.vic_jt_wrench_buf,
        ],
        device=dev,
    )


def launch_apply_vic_joint_torques_batched(
    scene: Any,
    *,
    gains: ImpedanceGains | None = None,
) -> None:
    """Vectorized VIC: batched FK/J/H, wrench kernel, batched PyTorch torques → ``joint_f``."""
    torch = _require_torch()

    if (
        scene.robot_model is None
        or scene.robot_state_0 is None
        or scene.robot_control is None
        or scene.robot_control.joint_f is None
        or scene.vic_jt_J_buf is None
        or scene.vic_jt_H_buf is None
        or scene.layout is None
    ):
        return

    launch_compute_vic_wrenches_batched(scene, gains=gains)

    model = scene.robot_model
    state = scene.robot_state_0
    control = scene.robot_control
    layout: BatchedEnvLayout = scene.layout
    num_envs = int(scene.vic_jt_num_envs)
    link_idx = int(scene.vic_jt_tcp_link_idx)
    row0 = link_idx * 6
    row1 = row0 + 6
    dev = control.joint_f.device
    torch_device = wp.device_to_torch(dev)
    dof_per = int(layout.joint_dof_count_per_world)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    newton.eval_jacobian(model, state, J=scene.vic_jt_J_buf)
    newton.eval_mass_matrix(model, state, H=scene.vic_jt_H_buf, J=scene.vic_jt_J_buf)

    J_th = wp.to_torch(scene.vic_jt_J_buf)[:num_envs, row0:row1, :_N_ARM_DOF].to(
        device=torch_device, dtype=torch.float64
    )
    M_th = wp.to_torch(scene.vic_jt_H_buf)[:num_envs, :_N_ARM_DOF, :_N_ARM_DOF].to(
        device=torch_device, dtype=torch.float64
    )

    jq_np = state.joint_q.numpy().reshape(num_envs, dof_per)[:, :_N_ARM_DOF]
    qd_np = state.joint_qd.numpy().reshape(num_envs, dof_per)[:, :_N_ARM_DOF]
    q_th = torch.as_tensor(jq_np, device=torch_device, dtype=torch.float64)
    qd_th = torch.as_tensor(qd_np, device=torch_device, dtype=torch.float64)
    default_q_th = wp.to_torch(scene.vic_jt_default_dof_pos_batched).to(
        device=torch_device, dtype=torch.float64
    )

    wrench_th = wp.to_torch(scene.vic_jt_wrench_buf).to(
        device=torch_device, dtype=torch.float64
    )

    tau, _, _ = compute_joint_torques_from_wrench_torch_batched(
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

    joint_f_th = wp.to_torch(control.joint_f).reshape(num_envs, dof_per)
    joint_f_th[:, :_N_ARM_DOF] = tau.to(dtype=joint_f_th.dtype)


def apply_vic_joint_torques_batched_to_scene(scene: Any) -> None:
    """Write batched VIC joint torques when controller and per-env targets are configured."""
    if getattr(scene, "vic_controller", None) is None:
        return
    if getattr(scene, "vic_target_positions_wp", None) is None:
        return
    if getattr(scene, "vic_target_rotations_wp", None) is None:
        return
    launch_apply_vic_joint_torques_batched(
        scene,
        gains=getattr(scene, "vic_gains", None),
    )
