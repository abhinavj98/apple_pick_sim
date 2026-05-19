"""Longer-horizon stability checks for staggered MuJoCo + VBD coupling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RANGES_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


def _import_pc():
    import apple_pick_sim.proxy_coupling as pc

    return pc


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    wp.init()


def test_dt_sweep_no_nan():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    stable_dt = (1.0 / 60.0) / 30.0
    for dt in (stable_dt, stable_dt / 2.0, stable_dt * 2.0):
        scene = cf.build_coupled_fruiting_placeholder(
            ranges, seed=20, mujoco_solver_kwargs={"disable_contacts": True}
        )
        for _ in range(100):
            scene.coupled_substep(dt)
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
            pc.sync_proxy_state,
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
    """Default fix_to_apple stem harvest stays O(apple weight) with gain + cap."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = cf.build_coupled_fruiting_placeholder(
        ranges, seed=20, mujoco_solver_kwargs={"disable_contacts": True}
    )
    tcp = scene.tcp_body_index
    dt = (1.0 / 60.0) / 30.0
    for _ in range(60):
        scene.coupled_substep(dt)
    w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
    fmag = float(np.linalg.norm(w[:3]))
    assert fmag < scene.stem_force_cap_N + 1.0, f"|F|={fmag:.2f} exceeds stem force cap"
    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    assert np.isfinite(rq).all()
    assert 1.0 < float(rq[2]) < 5.0, f"TCP drifted far from tree: z={rq[2]}"


def test_kinetic_energy_bounded_quiescent():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = cf.build_coupled_fruiting_placeholder(
        ranges, seed=21, mujoco_solver_kwargs={"disable_contacts": True}
    )
    proxy = scene.cable.gripper_proxy_body
    dt = (1.0 / 60.0) / 30.0
    ke_history: list[float] = []

    for _ in range(500):
        scene.coupled_substep(dt)
        bqd = scene.cable.state_0.body_qd.numpy().reshape(-1, 6)
        m = float(scene.cable.model.body_mass.numpy()[proxy])
        v = bqd[proxy, :3]
        ke = 0.5 * m * float(np.dot(v, v))
        ke_history.append(ke)

    assert max(ke_history) < 2.0, f"proxy KE spike {max(ke_history):.2f} J"
    assert all(np.isfinite(ke_history))
