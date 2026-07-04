"""Longer-horizon stability checks for staggered MuJoCo + VBD coupling (FR3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    SUB_DT,
    apply_direct_hold,
    build_coupled_fr3,
    new_direct_controller,
    requires_fr3,
    run_coupled_substeps_direct_hold,
)

pytestmark = requires_fr3


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


def _import_pc():
    import apple_pick_sim.coupled_fruiting.proxy_coupling as pc

    return pc


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    wp.init()


def test_dt_sweep_no_nan():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    from apple_pick_sim.robot import fr3_robot

    stable_dt = SUB_DT
    for dt in (stable_dt, stable_dt / 2.0, stable_dt * 2.0):
        scene = build_coupled_fr3(
            cf, ranges, 20, mujoco_solver_kwargs=DEFAULT_MJ_KW
        )
        run_coupled_substeps_direct_hold(scene, fr3_robot, 100, sub_dt=dt)
        cq = scene.cable.state_0.body_q.numpy()
        rq = scene.robot_state_0.body_q.numpy()
        assert np.isfinite(cq).all(), f"non-finite cable state at dt={dt}"
        assert np.isfinite(rq).all(), f"non-finite robot state at dt={dt}"


def test_proxy_mass_sweep_sync_scales_inversely():
    """Linear sync correction |Δv| ∝ 1/m for fixed coupling force."""
    pc = _import_pc()
    n = 2
    rid, pid = 0, 1
    fc = np.array([6.0, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    v_in = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    corrections: list[float] = []

    for m_p in (0.05, 0.15, 0.5):
        ids = wp.array([rid], dtype=int, device="cpu")
        pids = wp.array([pid], dtype=int, device="cpu")
        src_q = wp.zeros(n, dtype=wp.transform, device="cpu")
        dst_q = wp.zeros(n, dtype=wp.transform, device="cpu")
        src_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
        dst_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")

        src_np = np.zeros((n, 7), dtype=np.float32)
        src_np[rid] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        src_q.assign(src_np.ravel())

        sqd_np = np.zeros((n, 6), dtype=np.float32)
        sqd_np[rid, :3] = v_in.astype(np.float32)
        src_qd.assign(sqd_np.ravel())

        inv_mass = wp.zeros(n, dtype=float, device="cpu")
        inv_mass.assign(np.asarray([99.0, 1.0 / m_p], dtype=np.float32))

        id33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        inv_inertia = wp.array([id33, id33], dtype=wp.mat33, device="cpu")

        pf_np = np.zeros((n, 6), dtype=np.float32)
        pf_np[rid, :3] = fc.astype(np.float32)
        pf = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
        pf.assign(pf_np.ravel())

        wp.launch(
            pc.mirror_robot_tcp_to_proxy_kernel,
            dim=1,
            inputs=[
                ids,
                pids,
                src_q,
                src_qd,
                dst_q,
                dst_qd,
                pf,
                inv_mass,
                inv_inertia,
                wp.vec3(0.0),
                dt,
            ],
            device="cpu",
        )
        out_v = dst_qd.numpy().reshape(n, 6)[pid, :3]
        corrections.append(float(np.linalg.norm(out_v - v_in)))

    expected_ratio = corrections[0] / corrections[1]
    actual_mass_ratio = 0.15 / 0.05
    np.testing.assert_allclose(expected_ratio, actual_mass_ratio, rtol=0.01, atol=0.01)

    expected_ratio2 = corrections[1] / corrections[2]
    actual_mass_ratio2 = 0.5 / 0.15
    np.testing.assert_allclose(expected_ratio2, actual_mass_ratio2, rtol=0.01, atol=0.01)


def test_stem_coupling_quiescent_forces_bounded():
    """Stem harvest at TCP stays finite and within default force cap under direct-joint hold."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    from apple_pick_sim.robot import fr3_robot

    scene = build_coupled_fr3(
        cf,
        ranges,
        20,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body
    run_coupled_substeps_direct_hold(scene, fr3_robot, 60, sub_dt=SUB_DT)
    w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
    fmag = float(np.linalg.norm(w[:3]))
    assert np.isfinite(w).all()
    assert fmag <= 1000.0 + 1e-3, f"|F|={fmag:.2f} N exceeds DEFAULT_STEM_FORCE_CAP_N"
    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
    pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
    assert pos_err < 2e-3, f"TCP-proxy position drift {pos_err}"
    assert np.isfinite(rq).all()


@pytest.mark.slow
def test_kinetic_energy_bounded_quiescent():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    from apple_pick_sim.robot import fr3_robot

    scene = build_coupled_fr3(
        cf,
        ranges,
        21,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    proxy = scene.cable.gripper_proxy_body
    ke_history: list[float] = []
    scene.robot_kinematic_mode = True
    ctrl = new_direct_controller(scene, fr3_robot)

    for step in range(500):
        if step % 30 == 0:
            apply_direct_hold(scene, fr3_robot, ctrl)
        scene.coupled_substep(SUB_DT)
        bqd = scene.cable.state_0.body_qd.numpy().reshape(-1, 6)
        m = float(scene.cable.model.body_mass.numpy()[proxy])
        v = bqd[proxy, :3]
        ke = 0.5 * m * float(np.dot(v, v))
        ke_history.append(ke)

    assert max(ke_history) < 2.0, f"proxy KE spike {max(ke_history):.2f} J"
    assert all(np.isfinite(ke_history))


@pytest.mark.slow
@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")
def test_fr3_coupled_substep_mujoco_gpu_finite():
    """Long horizon with MuJoCo Warp on CUDA (opt-in GPU backend)."""
    import apple_pick_sim.coupled_fruiting as cf
    fs = __import__("apple_pick_sim.fruiting_system", fromlist=["load_ranges"])
    fr3_robot = __import__("apple_pick_sim.robot", fromlist=["fr3_robot"]).fr3_robot
    from conftest import RANGES_FIXTURE, build_coupled_fr3, run_coupled_substeps_direct_hold

    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf,
        ranges,
        99,
        device="cuda:0",
        mujoco_use_cpu=False,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    assert scene.mj_solver.use_mujoco_cpu is False
    run_coupled_substeps_direct_hold(scene, fr3_robot, 80, sub_dt=SUB_DT)
    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    assert np.isfinite(rq).all()


@pytest.mark.slow
def test_fr3_coupled_substep_long_horizon_finite():
    """FR3 + cable coupled loop stays finite over hundreds of substeps (headless)."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf, ranges, 24, device="cpu", mujoco_solver_kwargs=DEFAULT_MJ_KW
    )
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body
    rq0 = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp].copy()
    run_coupled_substeps_direct_hold(scene, fr3_robot, 400, sub_dt=SUB_DT)
    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
    cq = scene.cable.state_0.body_q.numpy()
    assert np.isfinite(rq).all()
    assert np.isfinite(cq).all()
    assert float(np.linalg.norm(rq[:3] - rq0[:3])) < 0.02, "TCP drift under direct hold"
    pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
    assert pos_err < 2e-3, f"TCP-proxy position drift {pos_err}"
