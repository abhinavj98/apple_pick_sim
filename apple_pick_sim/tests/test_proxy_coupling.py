"""Tests for two-model proxy coupling primitives (M1 Slice 1).

Covers ``sync_proxy_state`` (Warp) and velocity-delta ``harvest_proxy_wrenches`` per
docs/ROADMAP.md (VBD wrench harvest option 3 when direct accumulators are not used).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp


def _import_pc():
    import apple_pick_sim.proxy_coupling as pc

    return pc


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


@pytest.fixture(scope="module", autouse=True)
def _ensure_warp_init():
    wp.init()


def test_sync_proxy_state_copies_pose_no_forces_no_gravity():
    """Coupling wrench zero and zero gravity: mirror pose and velocity."""
    pc = _import_pc()
    n = 2
    rid, pid = 0, 1

    ids = wp.array([rid], dtype=int, device="cpu")
    pids = wp.array([pid], dtype=int, device="cpu")

    src_q = wp.zeros(n, dtype=wp.transform, device="cpu")
    dst_q = wp.zeros(n, dtype=wp.transform, device="cpu")
    src_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    dst_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    pf = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    inv_mass = wp.zeros(n, dtype=float, device="cpu")
    inv_mass.assign(np.asarray([99.0, 1.25], dtype=np.float32))

    id33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inv_inertia = wp.array([id33, id33], dtype=wp.mat33, device="cpu")

    pos = wp.vec3(1.5, -0.25, 2.0)
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.31)
    v = wp.vec3(0.1, -0.2, 0.3)
    om = wp.vec3(0.4, -0.1, 0.2)
    src_np = np.zeros((n, 7), dtype=np.float32)
    src_np[rid, 0] = float(pos[0])
    src_np[rid, 1] = float(pos[1])
    src_np[rid, 2] = float(pos[2])
    src_np[rid, 3] = float(q[0])
    src_np[rid, 4] = float(q[1])
    src_np[rid, 5] = float(q[2])
    src_np[rid, 6] = float(q[3])
    src_q.assign(src_np.ravel())

    sqd_np = np.zeros((n, 6), dtype=np.float32)
    sqd_np[rid, :3] = [float(v[i]) for i in range(3)]
    sqd_np[rid, 3:] = [float(om[j]) for j in range(3)]
    src_qd.assign(sqd_np.ravel())

    gravity = wp.vec3(0.0)
    dt = 1.0e-3

    wp.launch(
        pc.sync_proxy_state,
        dim=1,
        inputs=[ids, pids, src_q, src_qd, dst_q, dst_qd, pf, inv_mass, inv_inertia, gravity, dt],
        device="cpu",
    )

    dqf = dst_q.numpy()[pid]
    np.testing.assert_allclose(dqf[:3], [float(pos[i]) for i in range(3)], rtol=1e-5, atol=1e-5)
    flat_qd = dst_qd.numpy().reshape(n, 6)[pid]
    np.testing.assert_allclose(flat_qd[:3], [float(v[i]) for i in range(3)], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(flat_qd[3:6], [float(om[i]) for i in range(3)], rtol=1e-5, atol=1e-5)


def test_sync_proxy_state_subtracts_coupling_and_gravity_linear():
    """Linear part: v_proxy = v_robot − (dt/m) F_c − g*dt (world frame)."""
    pc = _import_pc()
    n = 2
    rid, pid = 0, 1
    m_p = 2.0

    ids = wp.array([rid], dtype=int, device="cpu")
    pids = wp.array([pid], dtype=int, device="cpu")

    src_q = wp.zeros(n, dtype=wp.transform, device="cpu")
    dst_q = wp.zeros(n, dtype=wp.transform, device="cpu")
    src_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    dst_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")

    src_np = np.zeros((n, 7), dtype=np.float32)
    src_np[rid] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    src_q.assign(src_np.ravel())

    inv_mass = wp.zeros(n, dtype=float, device="cpu")
    inv_mass.assign(np.asarray([99.0, 1.0 / m_p], dtype=np.float32))

    id33_unused = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inertia_diag = wp.mat33(0.06, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.06)
    inv_inertia_diag = wp.inverse(inertia_diag)
    inv_inertia = wp.array([id33_unused, inv_inertia_diag], dtype=wp.mat33, device="cpu")

    fc = wp.vec3(3.0, 4.0, 0.0)
    tc = wp.vec3(0.0, 1.6, 0.0)
    pf_np = np.zeros((n, 6), dtype=np.float32)
    pf_np[rid, :3] = [fc[0], fc[1], fc[2]]
    pf_np[rid, 3:] = [tc[0], tc[1], tc[2]]
    pf = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    pf.assign(pf_np.ravel())

    v_in = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w_in = np.array([-0.1, 0.3, -0.2], dtype=np.float64)
    sqd_np = np.zeros((n, 6), dtype=np.float32)
    sqd_np[rid, :3] = v_in.astype(np.float32)
    sqd_np[rid, 3:] = w_in.astype(np.float32)
    src_qd.assign(sqd_np.ravel())

    gravity = wp.vec3(0.0, 0.0, -10.0)
    dt = 0.02

    wp.launch(
        pc.sync_proxy_state,
        dim=1,
        inputs=[ids, pids, src_q, src_qd, dst_q, dst_qd, pf, inv_mass, inv_inertia, gravity, dt],
        device="cpu",
    )

    fc_np = np.array([fc[i] for i in range(3)], dtype=np.float64)
    delta_v_c = dt / m_p * fc_np
    delta_g = np.array([gravity[i] for i in range(3)], dtype=np.float64) * dt
    expected_v = v_in - delta_v_c - delta_g

    tc_np = np.array([tc[j] for j in range(3)], dtype=np.float64)
    inv_I = np.linalg.inv(np.diag([0.06, 0.06, 0.06])).astype(np.float64)
    expected_w = w_in - dt * (inv_I @ tc_np)

    outqd = dst_qd.numpy().reshape(n, 6)[pid]
    np.testing.assert_allclose(outqd[:3], expected_v.astype(np.float32), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(outqd[3:6], expected_w.astype(np.float32), rtol=5e-3, atol=1e-4)


def test_harvest_velocity_delta_matches_applied_force_plus_gravity():
    """Harvest recovers commanded body_f linear part alongside gravity dynamics."""
    import newton

    pc = _import_pc()

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body = builder.add_link(mass=1.5)
    builder.add_shape_box(body, hx=0.06, hy=0.06, hz=0.06)
    j = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([j])
    builder.color()
    model = builder.finalize(device="cpu")
    model.set_gravity((0.0, 0.0, -9.81))

    solver = newton.solvers.SolverVBD(model, iterations=12, friction_epsilon=0.1)

    dt = 0.002
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()

    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    fx, fy = 37.5, -12.75
    fz = np.zeros(model.body_count * 6, dtype=np.float32)
    fz[0 * 6 + 0] = fx
    fz[0 * 6 + 1] = fy
    s0.body_f.assign(fz)

    qd_pre = wp.clone(s0.body_qd)

    solver.step(s0, s1, ctrl, None, dt)

    out = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
    robot_ids = wp.array([0], dtype=int, device="cpu")
    proxy_ids = wp.array([0], dtype=int, device="cpu")

    pc.launch_harvest_proxy_wrenches_velocity_delta(
        robot_ids=robot_ids,
        proxy_ids=proxy_ids,
        model=model,
        body_q_post=s1.body_q,
        qd_synced=qd_pre,
        qd_post=s1.body_qd,
        gravity=wp.vec3(0.0, 0.0, -9.81),
        dt=dt,
        out_robot_wrenches=out,
    )

    w = out.numpy().reshape(model.body_count, 6)[0]
    np.testing.assert_allclose(w[:2], [fx, fy], rtol=1e-1, atol=2.5)
    np.testing.assert_allclose(w[2], 0.0, atol=35.0)
    np.testing.assert_allclose(w[3:6], 0.0, atol=35.0)


def test_harvest_velocity_delta_gravity_only_net_near_zero():
    """With only gravity, net harvest linear ≈ 0 (m*dv/dt − m*g)."""
    import newton

    pc = _import_pc()

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body = builder.add_link(mass=1.2)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    j = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([j])
    builder.color()
    model = builder.finalize(device="cpu")
    model.set_gravity((0.0, 0.0, -9.81))

    solver = newton.solvers.SolverVBD(model, iterations=12)
    dt = 0.001
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    s0.clear_forces()

    qd_pre = wp.clone(s0.body_qd)
    solver.step(s0, s1, ctrl, None, dt)

    out = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
    rid = wp.array([0], dtype=int, device="cpu")
    pid = wp.array([0], dtype=int, device="cpu")
    g = wp.vec3(0.0, 0.0, -9.81)

    pc.launch_harvest_proxy_wrenches_velocity_delta(
        robot_ids=rid,
        proxy_ids=pid,
        model=model,
        body_q_post=s1.body_q,
        qd_synced=qd_pre,
        qd_post=s1.body_qd,
        gravity=g,
        dt=dt,
        out_robot_wrenches=out,
    )

    w = out.numpy().reshape(model.body_count, 6)[0]
    np.testing.assert_allclose(w[:3], 0.0, atol=12.0)
    np.testing.assert_allclose(w[3:6], 0.0, atol=35.0)


def test_align_proxy_body_q_prev_for_vbd_clears_finalize_spurious_velocity():
    """Stale body_q_prev must not make VBD finalize invent 2x gravity on the proxy."""
    pc = _import_pc()
    fs = _import_fs()
    cf = __import__("apple_pick_sim.coupled_fruiting", fromlist=["build_coupled_fruiting_placeholder"])

    ranges = fs.load_ranges(
        Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
    )
    scene = cf.build_coupled_fruiting_placeholder(
        ranges,
        seed=42,
        device="cpu",
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    proxy = scene.cable.gripper_proxy_body
    dt = 1.0 / 600.0

    scene._mujoco_and_sync_proxy(dt)
    pc.align_proxy_body_q_prev_for_vbd(scene.cable, scene.proxy_registry.proxy_body_ids)

    scene.cable.state_0.clear_forces()
    scene.cable.solver.step(
        scene.cable.state_0,
        scene.cable.state_1,
        scene.cable.control,
        None,
        dt,
    )
    qd = scene.cable.state_1.body_qd.numpy().reshape(-1, 6)[proxy, 2]
    # One gravity step from synced v0≈0 (not 2× from stale body_q_prev).
    assert abs(qd + 9.81 * dt) < 0.15 * 9.81 * dt, f"proxy vz after VBD={qd}, expected ≈ -g*dt"


def test_zero_robot_wrench_slots_helper():
    pc = _import_pc()
    n = 5
    out = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    rng = np.random.default_rng(0)
    vals = rng.standard_normal((n, 6)).astype(np.float32)
    out.assign(vals.ravel())
    pc.zero_robot_wrench_slots(out, wp.array([1, 3], dtype=int, device="cpu"))
    arr = out.numpy().reshape(n, 6)
    np.testing.assert_allclose(arr[1], 0.0, atol=1e-7)
    np.testing.assert_allclose(arr[3], 0.0, atol=1e-7)
    np.testing.assert_allclose(arr[2], vals[2], rtol=1e-6)
    np.testing.assert_allclose(arr[4], vals[4], rtol=1e-6)


def _quat_wxyz_to_rotation_matrix(qwxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix from Newton body_q quaternion (qx, qy, qz, qw)."""
    qx, qy, qz, qw = [float(qwxyz[i]) for i in range(4)]
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def test_sync_proxy_state_subtracts_coupling_with_rotated_body():
    """Angular correction uses body-frame inertia with world rotation from proxy pose."""
    pc = _import_pc()
    n = 2
    rid, pid = 0, 1
    m_p = 2.0
    dt = 0.02

    ids = wp.array([rid], dtype=int, device="cpu")
    pids = wp.array([pid], dtype=int, device="cpu")

    src_q = wp.zeros(n, dtype=wp.transform, device="cpu")
    dst_q = wp.zeros(n, dtype=wp.transform, device="cpu")
    src_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    dst_qd = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")

    angle = 0.73
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    src_np = np.zeros((n, 7), dtype=np.float32)
    src_np[rid, :3] = [0.5, -0.2, 2.1]
    src_np[rid, 3] = float(q[0])
    src_np[rid, 4] = float(q[1])
    src_np[rid, 5] = float(q[2])
    src_np[rid, 6] = float(q[3])
    src_q.assign(src_np.ravel())

    inv_mass = wp.zeros(n, dtype=float, device="cpu")
    inv_mass.assign(np.asarray([99.0, 1.0 / m_p], dtype=np.float32))

    id33_unused = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inertia_diag = wp.mat33(0.06, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.06)
    inv_inertia_diag = wp.inverse(inertia_diag)
    inv_inertia = wp.array([id33_unused, inv_inertia_diag], dtype=wp.mat33, device="cpu")

    fc = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    tc = np.array([0.0, 1.6, 0.0], dtype=np.float64)
    pf_np = np.zeros((n, 6), dtype=np.float32)
    pf_np[rid, :3] = fc.astype(np.float32)
    pf_np[rid, 3:] = tc.astype(np.float32)
    pf = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    pf.assign(pf_np.ravel())

    v_in = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w_in = np.array([-0.1, 0.3, -0.2], dtype=np.float64)
    sqd_np = np.zeros((n, 6), dtype=np.float32)
    sqd_np[rid, :3] = v_in.astype(np.float32)
    sqd_np[rid, 3:] = w_in.astype(np.float32)
    src_qd.assign(sqd_np.ravel())

    gravity = wp.vec3(0.0, 0.0, -10.0)

    wp.launch(
        pc.sync_proxy_state,
        dim=1,
        inputs=[ids, pids, src_q, src_qd, dst_q, dst_qd, pf, inv_mass, inv_inertia, gravity, dt],
        device="cpu",
    )

    delta_v_c = dt / m_p * fc
    delta_g = np.array([0.0, 0.0, -10.0], dtype=np.float64) * dt
    expected_v = v_in - delta_v_c - delta_g

    R = _quat_wxyz_to_rotation_matrix(src_np[rid, 3:7])
    inv_I = np.linalg.inv(np.diag([0.06, 0.06, 0.06]))
    tau_b = inv_I @ (R.T @ tc)
    delta_w = dt * (R @ tau_b)
    expected_w = w_in - delta_w

    outqd = dst_qd.numpy().reshape(n, 6)[pid]
    np.testing.assert_allclose(outqd[:3], expected_v.astype(np.float32), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(outqd[3:6], expected_w.astype(np.float32), rtol=5e-3, atol=1e-4)


def test_harvest_velocity_delta_torque_only_recovers_torque():
    """Pure torque on body_f: harvest angular part ≈ applied torque (world frame)."""
    import newton

    pc = _import_pc()

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body = builder.add_link(mass=1.5)
    builder.add_shape_box(body, hx=0.06, hy=0.06, hz=0.06)
    j = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([j])
    builder.color()
    model = builder.finalize(device="cpu")
    model.set_gravity((0.0, 0.0, -9.81))

    solver = newton.solvers.SolverVBD(model, iterations=20, friction_epsilon=0.1)
    dt = 0.002
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    tx, ty, tz = 2.5, -1.25, 0.75
    bf = np.zeros(model.body_count * 6, dtype=np.float32)
    bf[3:6] = [tx, ty, tz]
    s0.body_f.assign(bf)

    qd_pre = wp.clone(s0.body_qd)
    solver.step(s0, s1, ctrl, None, dt)

    out = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
    pc.launch_harvest_proxy_wrenches_velocity_delta(
        robot_ids=wp.array([0], dtype=int, device="cpu"),
        proxy_ids=wp.array([0], dtype=int, device="cpu"),
        model=model,
        body_q_post=s1.body_q,
        qd_synced=qd_pre,
        qd_post=s1.body_qd,
        gravity=wp.vec3(0.0, 0.0, -9.81),
        dt=dt,
        out_robot_wrenches=out,
    )

    w = out.numpy().reshape(model.body_count, 6)[0]
    np.testing.assert_allclose(w[:3], 0.0, atol=8.0)
    np.testing.assert_allclose(w[3:6], [tx, ty, tz], rtol=0.15, atol=3.0)


def test_harvest_velocity_delta_with_rotated_body():
    """Linear harvest recovers body_f force when the body pose is rotated."""
    import newton

    pc = _import_pc()

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    body = builder.add_link(mass=1.5)
    builder.add_shape_box(body, hx=0.06, hy=0.06, hz=0.06)
    j = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([j])
    builder.color()
    model = builder.finalize(device="cpu")
    model.set_gravity((0.0, 0.0, -9.81))

    solver = newton.solvers.SolverVBD(model, iterations=20, friction_epsilon=0.1)
    dt = 0.002
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    angle = 0.55
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    bq = s0.body_q.numpy().reshape(-1, 7).copy()
    bq[0, 3] = float(q[0])
    bq[0, 4] = float(q[1])
    bq[0, 5] = float(q[2])
    bq[0, 6] = float(q[3])
    s0.body_q.assign(bq.ravel())

    fx, fy = 22.0, -8.5
    bf = np.zeros(model.body_count * 6, dtype=np.float32)
    bf[0] = fx
    bf[1] = fy
    s0.body_f.assign(bf)

    qd_pre = wp.clone(s0.body_qd)
    solver.step(s0, s1, ctrl, None, dt)

    out = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
    pc.launch_harvest_proxy_wrenches_velocity_delta(
        robot_ids=wp.array([0], dtype=int, device="cpu"),
        proxy_ids=wp.array([0], dtype=int, device="cpu"),
        model=model,
        body_q_post=s1.body_q,
        qd_synced=qd_pre,
        qd_post=s1.body_qd,
        gravity=wp.vec3(0.0, 0.0, -9.81),
        dt=dt,
        out_robot_wrenches=out,
    )

    w = out.numpy().reshape(model.body_count, 6)[0]
    np.testing.assert_allclose(w[:2], [fx, fy], rtol=0.12, atol=3.0)
    np.testing.assert_allclose(w[2], 0.0, atol=35.0)


def test_proxy_registry_from_mapping_sorts_pairs():
    pc = _import_pc()
    reg = pc.ProxyBodyRegistry.from_mapping({5: 12, 1: 3, 9: 7})
    assert reg.robot_to_proxy == ((1, 3), (5, 12), (9, 7))
    assert reg.robot_body_ids == (1, 5, 9)
    assert reg.proxy_body_ids == (3, 12, 7)


def test_align_proxy_body_q_prev_with_multiple_bodies():
    """align_proxy_body_q_prev_for_vbd updates every listed proxy body slot."""
    pc = _import_pc()
    fs = _import_fs()
    cf = __import__("apple_pick_sim.coupled_fruiting", fromlist=["build_coupled_fruiting_placeholder"])

    ranges = fs.load_ranges(
        Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
    )
    scene = cf.build_coupled_fruiting_placeholder(
        ranges, seed=11, device="cpu", mujoco_solver_kwargs={"disable_contacts": True}
    )
    proxy = scene.cable.gripper_proxy_body
    apple = scene.cable.apple_body
    assert apple is not None

    bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    bqp = scene.cable.solver.body_q_prev.numpy().reshape(-1, 7).copy()
    bqp[proxy] = bq[proxy] + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    bqp[apple] = bq[apple] + np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scene.cable.solver.body_q_prev.assign(bqp.ravel())

    pc.align_proxy_body_q_prev_for_vbd(scene.cable, (proxy, apple))

    bqp_after = scene.cable.solver.body_q_prev.numpy().reshape(-1, 7)
    np.testing.assert_allclose(bqp_after[proxy], bq[proxy], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(bqp_after[apple], bq[apple], rtol=1e-6, atol=1e-6)