"""Kinematic ``tcp_pose`` controller: prescribed TCP following and plant-wrench readout."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list

from conftest import requires_fr3

_TESTS_DIR = Path(__file__).resolve().parent
RANGES_FIXTURE = _TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
_NUM_ENVS = 2
# FR3 "ready" pose: bent, far from the q=0 straight-up singularity.
_READY_JOINT_Q = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


@pytest.fixture
def per_env_params(ranges):
    return sample_heterogeneous_params_list(ranges, topology_seed=21, num_envs=_NUM_ENVS)


def _config(*, mode: str, fix_to_apple: bool) -> BatchedHeterogeneousCoupledSimConfig:
    base = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS)
    controller = (
        ControllerConfig(mode="tcp_pose", action_dim=7)
        if mode == "tcp_pose"
        else ControllerConfig(mode="vic_pose", action_dim=19)
    )
    return dataclasses.replace(
        base,
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=fix_to_apple,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=8),
        domain_randomization=dataclasses.replace(base.domain_randomization, topology_seed=21),
        controller=controller,
        settle_diagnostics=None,
        obs=ObsConfig(allocate_buffers=True),
    )


def _set_ready_pose(sim) -> None:
    """Put every world's arm in the bent ready pose (model + state) and refresh FK."""
    import newton

    sc = sim.scene
    n_coords = int(sc.robot_model.joint_coord_count) // _NUM_ENVS
    q = sc.robot_model.joint_q.numpy().reshape(_NUM_ENVS, n_coords)
    q[:, : len(_READY_JOINT_Q)] = _READY_JOINT_Q
    sc.robot_model.joint_q.assign(q.reshape(-1))
    sc.robot_state_0.joint_q.assign(q.reshape(-1))
    newton.eval_fk(sc.robot_model, sc.robot_model.joint_q, sc.robot_model.joint_qd, sc.robot_state_0)


def _tcp_rows(sim) -> np.ndarray:
    bq = sim.scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    return np.stack([bq[i] for i in sim.scene.layout.tcp_body_indices])


def _pose_action(tcp7: np.ndarray, dx: float):
    """``[pos, quat_wxyz]`` row from a ``[pos, quat_xyzw]`` body_q row shifted by ``dx``."""
    return [
        float(tcp7[0]) + dx,
        float(tcp7[1]),
        float(tcp7[2]),
        float(tcp7[6]),
        float(tcp7[3]),
        float(tcp7[4]),
        float(tcp7[5]),
    ]


def test_config_rejects_wrong_tcp_pose_action_dim():
    base = _config(mode="tcp_pose", fix_to_apple=False)
    base.validate()
    bad = dataclasses.replace(base, controller=ControllerConfig(mode="tcp_pose", action_dim=19))
    with pytest.raises(ValueError, match="action_dim=7"):
        bad.validate()


@requires_fr3
def test_tcp_follows_prescribed_line_with_consistent_joint_velocity(ranges, per_env_params):
    torch = pytest.importorskip("torch")
    sim = BatchedHeterogeneousCoupledSim(
        _config(mode="tcp_pose", fix_to_apple=False), per_env_params, ranges, use_settle_cache=False
    )
    assert sim.scene.robot_kinematic_mode is True
    assert sim.scene.ft_from_proxy_forces is True
    _set_ready_pose(sim)
    tcp0 = _tcp_rows(sim)
    step_m = 0.001
    for k in range(1, 6):
        actions = torch.tensor(
            [_pose_action(tcp0[w], k * step_m) for w in range(_NUM_ENVS)], dtype=torch.float32
        )
        q_before = sim.scene.robot_model.joint_q.numpy().copy()
        sim.step(actions)
        tcp = _tcp_rows(sim)
        np.testing.assert_allclose(tcp[:, 0] - tcp0[:, 0], k * step_m, atol=5.0e-4)
        np.testing.assert_allclose(tcp[:, 1:3], tcp0[:, 1:3], atol=5.0e-4)
        residual = sim._ee_ctrl.ik_residual.numpy()
        assert float(residual[:, 0].max()) < 5.0e-4
        assert float(residual[:, 1].max()) < np.radians(0.5)
        q_after = sim.scene.robot_model.joint_q.numpy()
        qd = sim.scene.robot_model.joint_qd.numpy()
        np.testing.assert_allclose(qd, (q_after - q_before) / sim.frame_dt, atol=1.0e-4)
        # Interpolated, not teleported: the TCP moves at ~step / frame_dt during the frame.
        tcp_lin_vel = sim.scene.robot_state_0.body_qd.numpy().reshape(-1, 6)[
            list(sim.scene.layout.tcp_body_indices)
        ][:, :3]
        np.testing.assert_allclose(tcp_lin_vel[:, 0], step_m / sim.frame_dt, rtol=0.1)


@requires_fr3
def test_step_without_actions_holds_tcp(ranges, per_env_params):
    sim = BatchedHeterogeneousCoupledSim(
        _config(mode="tcp_pose", fix_to_apple=False), per_env_params, ranges, use_settle_cache=False
    )
    _set_ready_pose(sim)
    tcp0 = _tcp_rows(sim)
    for _ in range(3):
        sim.step(None)
    np.testing.assert_allclose(_tcp_rows(sim)[:, :3], tcp0[:, :3], atol=1.0e-6)


@requires_fr3
def test_welded_hold_reads_plant_wrench_from_proxy_forces(ranges, per_env_params):
    """Kinematic arm: the lagged coupling cache stays zero; the plant wrench is in tcp_force."""
    torch = pytest.importorskip("torch")
    sim = BatchedHeterogeneousCoupledSim(
        _config(mode="tcp_pose", fix_to_apple=True), per_env_params, ranges, use_settle_cache=False
    )
    tcp0 = _tcp_rows(sim)
    hold = torch.tensor(
        [_pose_action(tcp0[w], 0.0) for w in range(_NUM_ENVS)], dtype=torch.float32
    )
    for _ in range(5):
        sim.step(hold)
    obs = sim.gather_obs()
    tcp_force = np.asarray(obs["tcp_force"].numpy())
    coupling = np.asarray(obs["tcp_coupling_force"].numpy())
    assert np.all(np.isfinite(tcp_force))
    assert float(np.abs(coupling).max()) == 0.0
    # The welded plant hangs on the TCP: a nonzero (gravity + elastic) wrench.
    assert float(np.linalg.norm(tcp_force[:, :3], axis=1).min()) > 1.0e-3
