"""sep_ori task-wrench mapping matches the real OSC collection path."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.vic_joint_torques import (
    compute_joint_torques_from_wrench_numpy,
    compute_joint_torques_from_wrench_torch,
)

_N_ARM_DOF = 7


def _spd_mass(rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((_N_ARM_DOF, _N_ARM_DOF))
    return a @ a.T + np.eye(_N_ARM_DOF) * 0.5


def test_sep_ori_false_keeps_coupled_lambda_on_full_wrench():
    rng = np.random.default_rng(0)
    jacobian = rng.standard_normal((6, _N_ARM_DOF))
    mass_matrix = _spd_mass(rng)
    wrench = rng.standard_normal(6)
    q = rng.uniform(-0.5, 0.5, _N_ARM_DOF)
    qd = np.zeros(_N_ARM_DOF)
    default_q = q.copy()
    default_q[6] = 0.0

    _total, tau_task, _null = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=0.0,
        kd_null=0.0,
        sep_ori=False,
    )
    m_inv = np.linalg.inv(mass_matrix)
    lam = np.linalg.inv(jacobian @ m_inv @ jacobian.T)
    expected = jacobian.T @ lam @ wrench
    np.testing.assert_allclose(tau_task, expected, rtol=1e-10, atol=1e-10)


def test_sep_ori_rotation_uses_direct_jacobian_not_lambda():
    rng = np.random.default_rng(1)
    jacobian = rng.standard_normal((6, _N_ARM_DOF))
    mass_matrix = _spd_mass(rng)
    wrench = np.array([0.0, 0.0, 0.0, 0.4, -0.2, 0.1], dtype=np.float64)
    q = np.zeros(_N_ARM_DOF)
    qd = np.zeros(_N_ARM_DOF)
    default_q = q.copy()

    _total, tau_task, _null = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=0.0,
        kd_null=0.0,
        sep_ori=True,
    )
    expected = jacobian[3:6, :].T @ wrench[3:6]
    np.testing.assert_allclose(tau_task, expected, rtol=1e-10, atol=1e-10)

    _total_c, tau_coupled, _ = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=0.0,
        kd_null=0.0,
        sep_ori=False,
    )
    assert not np.allclose(tau_task, tau_coupled, atol=1e-8)


def test_sep_ori_translation_uses_full_lambda_with_zeroed_rotation():
    rng = np.random.default_rng(2)
    jacobian = rng.standard_normal((6, _N_ARM_DOF))
    mass_matrix = _spd_mass(rng)
    wrench = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float64)
    q = np.zeros(_N_ARM_DOF)
    qd = np.zeros(_N_ARM_DOF)
    default_q = q.copy()

    _total, tau_task, _null = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=0.0,
        kd_null=0.0,
        sep_ori=True,
    )
    m_inv = np.linalg.inv(mass_matrix)
    lam = np.linalg.inv(jacobian @ m_inv @ jacobian.T)
    expected = jacobian.T @ lam @ wrench
    np.testing.assert_allclose(tau_task, expected, rtol=1e-10, atol=1e-10)


def test_sep_ori_torch_and_batched_match_numpy():
    torch = pytest.importorskip("torch")
    from apple_pick_sim.coupled_fruiting.vic_joint_torques_batched import (
        compute_joint_torques_from_wrench_torch_batched,
    )

    rng = np.random.default_rng(3)
    jacobian = rng.standard_normal((6, _N_ARM_DOF))
    mass_matrix = _spd_mass(rng)
    wrench = rng.standard_normal(6)
    q = rng.uniform(-0.4, 0.4, _N_ARM_DOF)
    qd = rng.standard_normal(_N_ARM_DOF)
    default_q = q.copy()
    default_q[6] = 0.0

    tau_np, jt_np, null_np = compute_joint_torques_from_wrench_numpy(
        task_wrench=wrench,
        jacobian=jacobian,
        mass_matrix=mass_matrix,
        joint_pos=q,
        joint_vel=qd,
        default_dof_pos=default_q,
        kp_null=10.0,
        kd_null=15.0,
        sep_ori=True,
    )
    tau_th, jt_th, null_th = compute_joint_torques_from_wrench_torch(
        task_wrench=torch.as_tensor(wrench, dtype=torch.float64),
        jacobian=torch.as_tensor(jacobian, dtype=torch.float64),
        mass_matrix=torch.as_tensor(mass_matrix, dtype=torch.float64),
        joint_pos=torch.as_tensor(q, dtype=torch.float64),
        joint_vel=torch.as_tensor(qd, dtype=torch.float64),
        default_dof_pos=torch.as_tensor(default_q, dtype=torch.float64),
        kp_null=10.0,
        kd_null=15.0,
        sep_ori=True,
    )
    tau_b, jt_b, null_b = compute_joint_torques_from_wrench_torch_batched(
        task_wrench=torch.as_tensor(wrench, dtype=torch.float64).unsqueeze(0),
        jacobian=torch.as_tensor(jacobian, dtype=torch.float64).unsqueeze(0),
        mass_matrix=torch.as_tensor(mass_matrix, dtype=torch.float64).unsqueeze(0),
        joint_pos=torch.as_tensor(q, dtype=torch.float64).unsqueeze(0),
        joint_vel=torch.as_tensor(qd, dtype=torch.float64).unsqueeze(0),
        default_dof_pos=torch.as_tensor(default_q, dtype=torch.float64).unsqueeze(0),
        kp_null=10.0,
        kd_null=15.0,
        sep_ori=True,
    )
    np.testing.assert_allclose(tau_th.detach().cpu().numpy(), tau_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jt_th.detach().cpu().numpy(), jt_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(null_th.detach().cpu().numpy(), null_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(tau_b[0].detach().cpu().numpy(), tau_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jt_b[0].detach().cpu().numpy(), jt_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(null_b[0].detach().cpu().numpy(), null_np, rtol=1e-10, atol=1e-10)
