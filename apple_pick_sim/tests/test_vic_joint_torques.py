"""Tests for VIC joint-torque path (J^T Λ wrench mapping + null-space)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.vic_joint_torques import (
    compute_joint_torques_from_wrench_numpy,
    compute_joint_torques_from_wrench_torch,
    find_tcp_link_idx,
    launch_apply_vic_joint_torques,
)
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity
from apple_pick_sim.tests.conftest import (
    DEFAULT_MJ_KW,
    FRAME_DT,
    RANGES_FIXTURE,
    SUB_DT,
    SUBSTEPS_PER_FRAME,
    build_coupled_fr3,
    fr3_assets_available,
)

pytestmark = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)

_N_ARM_DOF = 7


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _build_mujoco_only_fr3():
    cf = _import_cf()
    import apple_pick_sim.fruiting_system as fs

    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf,
        ranges,
        11,
        mujoco_only=True,
        enable_self_collisions=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    fr3_robot.init_mujoco_actuator_targets_from_model(scene.robot_model, scene.robot_control)
    return scene


def _configure_joint_torque_vic(scene) -> Fr3EEImpedanceController:
    scene.vic_use_joint_torques = True
    ctrl = Fr3EEImpedanceController(tcp_body_index=int(scene.tcp_body_index))
    scene.vic_controller = ctrl
    scene.vic_gains = ImpedanceGains()
    fr3_robot.configure_vic_joint_torques_arm(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        scene=scene,
    )
    ctrl.sync_target_from_state(scene.robot_state_0)
    return ctrl


def _eval_arm_kinematics(model, state):
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    J = newton.eval_jacobian(model, state)
    H = newton.eval_mass_matrix(model, state, J=J)
    return J, H


def test_find_tcp_link_idx():
    model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
    state = model.state()
    link_idx = find_tcp_link_idx(model, tcp_idx)
    J, _H = _eval_arm_kinematics(model, state)
    J_np = J.numpy()
    qd = state.joint_qd.numpy()[:_N_ARM_DOF]
    J_tcp = J_np[0, link_idx * 6 : (link_idx + 1) * 6, :_N_ARM_DOF]
    pred = J_tcp @ qd
    actual = state.body_qd.numpy()[tcp_idx]
    np.testing.assert_allclose(pred, actual, rtol=1e-5, atol=1e-5)


def test_zero_wrench_no_task_torque():
    model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
    state = model.state()
    link_idx = find_tcp_link_idx(model, tcp_idx)
    J, H = _eval_arm_kinematics(model, state)
    J_np = J.numpy()[0, link_idx * 6 : (link_idx + 1) * 6, :_N_ARM_DOF]
    M_np = H.numpy()[0, :_N_ARM_DOF, :_N_ARM_DOF]
    q = state.joint_q.numpy()[:_N_ARM_DOF]
    qd = state.joint_qd.numpy()[:_N_ARM_DOF]
    default_q = q.copy()
    default_q[6] = 0.0

    total, tau_task, tau_null = compute_joint_torques_from_wrench_numpy(
        task_wrench=np.zeros(6),
        jacobian=J_np,
        mass_matrix=M_np,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=10.0,
        kd_null=6.3246,
    )
    np.testing.assert_allclose(tau_task, 0.0, atol=1e-4)
    np.testing.assert_allclose(total, tau_null, rtol=1e-5, atol=1e-4)


def test_null_space_orthogonal_to_task():
    model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
    state = model.state()
    link_idx = find_tcp_link_idx(model, tcp_idx)
    J, H = _eval_arm_kinematics(model, state)
    J_np = J.numpy()[0, link_idx * 6 : (link_idx + 1) * 6, :_N_ARM_DOF]
    M_np = H.numpy()[0, :_N_ARM_DOF, :_N_ARM_DOF]
    q = state.joint_q.numpy()[:_N_ARM_DOF]
    qd = state.joint_qd.numpy()[:_N_ARM_DOF]
    default_q = q.copy()
    default_q[6] = 0.0

    _total, _tau_task, tau_null = compute_joint_torques_from_wrench_numpy(
        task_wrench=np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        jacobian=J_np,
        mass_matrix=M_np,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
    )
    tcp_accel = J_np @ tau_null
    np.testing.assert_allclose(tcp_accel, 0.0, atol=1e-3)


def test_torch_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    jacobian = rng.standard_normal((6, _N_ARM_DOF))
    mass_matrix = rng.standard_normal((_N_ARM_DOF, _N_ARM_DOF))
    mass_matrix = mass_matrix @ mass_matrix.T + np.eye(_N_ARM_DOF) * 0.5
    wrench = rng.standard_normal(6)
    q = rng.uniform(-0.5, 0.5, _N_ARM_DOF)
    qd = rng.standard_normal(_N_ARM_DOF)
    default_q = q.copy()
    default_q[6] = 0.0
    kp_null, kd_null = 10.0, 6.3246

    tau_np, jt_np, null_np = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=kp_null,
        kd_null=kd_null,
    )
    tau_th, jt_th, null_th = compute_joint_torques_from_wrench_torch(
        task_wrench=torch.as_tensor(wrench, dtype=torch.float64),
        jacobian=torch.as_tensor(jacobian, dtype=torch.float64),
        mass_matrix=torch.as_tensor(mass_matrix, dtype=torch.float64),
        joint_pos=torch.as_tensor(q, dtype=torch.float64),
        joint_vel=torch.as_tensor(qd, dtype=torch.float64),
        default_dof_pos=torch.as_tensor(default_q, dtype=torch.float64),
        kp_null=kp_null,
        kd_null=kd_null,
    )
    np.testing.assert_allclose(tau_th.detach().cpu().numpy(), tau_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jt_th.detach().cpu().numpy(), jt_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(null_th.detach().cpu().numpy(), null_np, rtol=1e-10, atol=1e-10)


def test_position_error_gives_correct_task_torque():
    """Single prismatic DOF: τ_task = J^T Λ f with analytic J and M."""
    jacobian = np.zeros((6, 1), dtype=np.float64)
    jacobian[1, 0] = 1.0
    mass_matrix = np.array([[2.0]], dtype=np.float64)
    wrench = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = np.zeros(1, dtype=np.float64)
    qd = np.zeros(1, dtype=np.float64)
    default_q = np.zeros(1, dtype=np.float64)
    singularity_damping = 1.0

    _total, tau_task, _tau_null = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=0.0,
        kd_null=0.0,
        singularity_damping=singularity_damping,
    )

    M_inv = np.linalg.inv(mass_matrix)
    JMJ = jacobian @ M_inv @ jacobian.T + singularity_damping * np.eye(6)
    Lambda = np.linalg.inv(JMJ)
    expected = jacobian.T @ Lambda @ wrench
    np.testing.assert_allclose(tau_task, expected, rtol=1e-5, atol=1e-5)


def test_apply_vic_joint_torques_writes_joint_f():
    pytest.importorskip("torch")
    scene = _build_mujoco_only_fr3()
    _configure_joint_torque_vic(scene)
    tcp = scene.tcp_body_index
    bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    target = wp.transform(
        wp.vec3(float(bq[0]) + 0.05, float(bq[1]), float(bq[2])),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )
    scene.robot_control.joint_f.zero_()
    launch_apply_vic_joint_torques(
        scene,
        target_tf=target,
        target_twist=EEVelocity(),
        gains=ImpedanceGains(linear_k=500.0, linear_d=50.0),
    )
    tau = scene.robot_control.joint_f.numpy().reshape(-1)[:_N_ARM_DOF]
    assert float(np.linalg.norm(tau)) > 0.5


def test_launch_joint_torques_match_numpy_reference():
    pytest.importorskip("torch")
    scene = _build_mujoco_only_fr3()
    _configure_joint_torque_vic(scene)
    model = scene.robot_model
    state = scene.robot_state_0
    tcp = scene.tcp_body_index
    link_idx = scene.vic_jt_tcp_link_idx
    bq = state.body_q.numpy().reshape(-1, 7)[tcp]
    target = wp.transform(
        wp.vec3(float(bq[0]) + 0.03, float(bq[1]), float(bq[2])),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )
    gains = ImpedanceGains(linear_k=400.0, linear_d=40.0, angular_k=20.0, angular_d=2.0)

    launch_apply_vic_joint_torques(
        scene,
        target_tf=target,
        target_twist=EEVelocity(linear=(0.1, 0.0, 0.0)),
        gains=gains,
    )
    tau_gpu = scene.robot_control.joint_f.numpy().reshape(-1)[:_N_ARM_DOF].copy()

    J, H = _eval_arm_kinematics(model, state)
    J_np = J.numpy()[0, link_idx * 6 : (link_idx + 1) * 6, :_N_ARM_DOF]
    M_np = H.numpy()[0, :_N_ARM_DOF, :_N_ARM_DOF]
    q = state.joint_q.numpy()[:_N_ARM_DOF]
    qd = state.joint_qd.numpy()[:_N_ARM_DOF]
    bqd = state.body_qd.numpy().reshape(-1, 6)[tcp]
    vic = Fr3EEImpedanceController()
    wrench = vic.compute_applied_wrench(
        target_tf=target,
        target_twist=EEVelocity(linear=(0.1, 0.0, 0.0)),
        tcp_body_q=bq,
        tcp_body_qd=bqd,
        gains=gains,
    )
    default_q = scene.vic_jt_default_dof_pos.numpy().reshape(-1)[:_N_ARM_DOF]
    tau_cpu, _, _ = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=J_np,
        mass_matrix=M_np,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=float(scene.vic_jt_kp_null),
        kd_null=float(scene.vic_jt_kd_null),
        singularity_damping=float(scene.vic_jt_singularity_damping),
    )
    np.testing.assert_allclose(tau_gpu, tau_cpu, rtol=1e-4, atol=1e-3)


def test_vic_joint_torques_moves_arm():
    pytest.importorskip("torch")
    scene = _build_mujoco_only_fr3()
    scene.robot_kinematic_mode = False
    ctrl = _configure_joint_torque_vic(scene)
    scene.vic_gains = ImpedanceGains(linear_k=800.0, linear_d=80.0)
    vel = fr3_robot.EEVelocity(linear=(0.5, 0.0, 0.0))
    tcp = scene.tcp_body_index
    x0 = float(scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, 0])
    max_dx = 0.0
    for _ in range(60):
        scene.update_fr3_ee_teleop(FRAME_DT, ctrl, velocity=vel)
        for _ in range(SUBSTEPS_PER_FRAME):
            scene.mujoco_substep(SUB_DT)
        max_dx = max(
            max_dx,
            float(scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, 0]) - x0,
        )
    assert max_dx > 0.05, f"expected joint-torque VIC to advance TCP +X, got max_dx={max_dx:.6f} m"
