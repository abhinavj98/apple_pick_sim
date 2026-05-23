"""M1 Slice 2b: FR3 + coupled cable VBD staggered loop.

Force and stability tests use :class:`~apple_pick_sim.fr3_robot.Fr3EEDirectJointController`
with ``robot_kinematic_mode=True`` for accurate TCP pose without actuator drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import newton
import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import (
    DEFAULT_MJ_KW,
    FRAME_DT,
    RANGES_FIXTURE,
    SUB_DT,
    apply_direct_hold,
    build_coupled_fr3,
    build_vbd_only,
    new_direct_controller,
    requires_fr3,
    run_coupled_substeps_direct_hold,
    run_mujoco_substeps_direct_hold,
)

# FR3 IK bootstrap residual on the straight-rod fixture (see seed sweep in tests).
_FR3_BOOTSTRAP_POS_TOL_M = 0.25
# Quiescent velocity-delta harvest under direct-joint hold (aligned with coupling_stability).
_QUIESCENT_HARVEST_F_CAP_N = 500.0

pytestmark = requires_fr3


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


@pytest.fixture(scope="module", autouse=True)
def _warp_init():
    import warp as wp

    wp.init()


def test_build_coupled_default_includes_robot():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1)
    assert not scene.vbd_only
    assert not scene.mujoco_only
    assert scene.robot_model is not None
    assert scene.mj_solver is not None
    assert scene.cable.gripper_proxy_body >= 0


def test_build_vbd_only_skips_robot():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_vbd_only(cf, ranges, 1)
    assert scene.vbd_only
    assert not scene.mujoco_only
    assert scene.robot_model is None
    assert scene.mj_solver is None


def test_build_mujoco_only_includes_robot():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1, mujoco_only=True)
    assert not scene.vbd_only
    assert scene.mujoco_only
    assert scene.robot_model is not None


def test_vbd_and_mujoco_only_mutually_exclusive():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_coupled_fr3(cf, ranges, 1, vbd_only=True, mujoco_only=True)  # noqa: PT012


def test_vbd_substep_keeps_proxy_pose_finite():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_vbd_only(cf, ranges, 3)
    proxy = scene.cable.gripper_proxy_body
    pos_before = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy, :3].copy()
    assert np.isfinite(pos_before).all()
    dt = 1.0 / 600.0
    for _ in range(12):
        scene.vbd_substep(dt)
    pos_after = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy, :3]
    assert np.isfinite(pos_after).all()


def _quat_angle_error_rad(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Angle between two unit quaternions stored as (qx, qy, qz, qw)."""
    qa = q_a[3:7] / (np.linalg.norm(q_a[3:7]) + 1e-12)
    qb = q_b[3:7] / (np.linalg.norm(q_b[3:7]) + 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))

def test_tcp_pose_matches_proxy_after_bootstrap():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 2)
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body

    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
    pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
    assert pos_err < _FR3_BOOTSTRAP_POS_TOL_M, (
        f"TCP/proxy position mismatch after IK bootstrap: {pos_err} m"
    )
    assert _quat_angle_error_rad(rq, pq) < 0.15


def test_robot_state_matches_model_after_bootstrap():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 2)
    jq = scene.robot_model.joint_q.numpy()
    jqd = scene.robot_model.joint_qd.numpy()
    np.testing.assert_allclose(scene.robot_state_0.joint_q.numpy(), jq, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scene.robot_state_0.joint_qd.numpy(), jqd, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scene.robot_state_1.joint_q.numpy(), jq, rtol=1e-6, atol=1e-6)
    assert scene.mj_solver is not None
    assert len(scene.mj_solver.mj_data.qpos) >= int(scene.robot_model.joint_coord_count)


def test_mujoco_substep_proxy_does_not_teleport_on_first_step():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 5, mujoco_only=True)
    proxy = scene.cable.gripper_proxy_body
    scene.robot_kinematic_mode = True
    ctrl = new_direct_controller(scene, fr3_robot)
    apply_direct_hold(scene, fr3_robot, ctrl)
    scene.mujoco_substep(1.0 / 1800.0)
    z0 = float(scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy, 2])
    scene.mujoco_substep(1.0 / 1800.0)
    z1 = float(scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy, 2])
    assert abs(z1 - z0) < 0.05, f"proxy teleported on second substep: z {z0} -> {z1}"


def test_mujoco_substep_syncs_proxy_to_robot():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 5,
        mujoco_only=True,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
    )
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body
    apple = scene.cable.apple_body
    assert apple is not None

    apple_before = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3].copy()
    dt = 1.0 / 600.0
    for _ in range(12):
        scene.mujoco_substep(dt)

    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
    np.testing.assert_allclose(rq[:3], pq[:3], rtol=1e-4, atol=1e-4)
    apple_after = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3]
    np.testing.assert_allclose(apple_before, apple_after, rtol=1e-6, atol=1e-6)


def test_coupled_substeps_remain_finite():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 4,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
    )
    run_coupled_substeps_direct_hold(scene, fr3_robot, 12, sub_dt=1.0 / 600.0)
    cq = scene.cable.state_0.body_q.numpy()
    rq = scene.robot_state_0.body_q.numpy()
    assert np.isfinite(cq).all()
    assert np.isfinite(rq).all()


def test_coupled_substep_after_cable_clear_forces_hook_runs_once_per_substep():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 7)
    calls: list[int] = []

    def hook():
        calls.append(1)

    scene.coupled_substep(1e-5, after_cable_clear_forces=hook)
    assert len(calls) == 1
    scene.coupled_substep(1e-5, after_cable_clear_forces=hook)
    assert len(calls) == 2


def test_coupled_harvest_forces_stay_small_without_external_load():
    """Velocity-delta harvest stays finite/capped when the arm is held via direct joints."""
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 42,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body
    run_coupled_substeps_direct_hold(scene, fr3_robot, 60, sub_dt=SUB_DT)
    wrenches = scene.proxy_forces.numpy().reshape(-1, 6)
    fmag = float(np.linalg.norm(wrenches[tcp, :3]))
    assert fmag < _QUIESCENT_HARVEST_F_CAP_N, (
        f"spurious coupling harvest |F|={fmag:.2f} N (expected capped under hold)"
    )
    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
    pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
    assert pos_err < 2e-3, f"TCP-proxy drift under hold: {pos_err} m"


def test_measure_fruiting_forces_after_coupled_step():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 6)
    dt = 2.0e-4
    scene.coupled_substep(dt)
    q_prev = scene.cable.state_1.body_q
    out = fs.measure_fruiting_forces(scene.cable, scene.cable.state_0.body_q, q_prev, dt=dt)
    assert len(out["fixed_joints"]) == len(scene.cable.fruiting_fixed_joints)



def test_apply_spatial_wrench_zeroes_non_tcp_bodies():
    """Only the TCP body_f slot receives the lagged coupling wrench."""
    import newton

    cf = _import_cf()

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    b0 = builder.add_link(mass=0.5, label="base")
    b1 = builder.add_link(mass=0.5, label="tcp")
    j0 = builder.add_joint_free(parent=-1, child=b0)
    j1 = builder.add_joint_fixed(parent=b0, child=b1)
    builder.add_articulation([j0, j1])
    builder.color()
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    wrenches = wp.zeros(model.body_count, dtype=wp.spatial_vector, device="cpu")
    w_np = np.zeros((model.body_count, 6), dtype=np.float32)
    w_np[b0] = [9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w_np[b1] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    wrenches.assign(w_np.ravel())

    state.clear_forces()
    cf._apply_spatial_wrench_to_body_f(state, b1, wrenches)

    bf = state.body_f.numpy().reshape(model.body_count, 6)
    np.testing.assert_allclose(bf[b0], 0.0, atol=1e-7)
    np.testing.assert_allclose(bf[b1], w_np[b1], rtol=1e-6, atol=1e-6)


def test_coupling_forces_cache_is_value_snapshot():
    """assign copies proxy_forces; later proxy_forces edits must not change the cache."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf,
        ranges, 8, mujoco_solver_kwargs={"disable_contacts": True}
    )
    tcp = scene.tcp_body_index
    dt = 1.0 / 600.0

    injected = np.zeros(scene.robot_model.body_count * 6, dtype=np.float32)
    injected[tcp * 6 : tcp * 6 + 3] = [5.0, -3.0, 1.0]
    scene.proxy_forces.assign(injected)

    scene._mujoco_and_sync_proxy(dt)
    cache_after = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp].copy()

    mutated = injected.copy()
    mutated[tcp * 6 : tcp * 6 + 3] = [99.0, 99.0, 99.0]
    scene.proxy_forces.assign(mutated)

    cache_now = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp]
    np.testing.assert_allclose(cache_now, cache_after, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cache_after[:3], [5.0, -3.0, 1.0], rtol=1e-5, atol=1e-5)


def test_coupled_substep_lag_one_step():
    """Harvest at step N is applied from cache at step N+1, not the same substep."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf,
        ranges, 9, mujoco_solver_kwargs={"disable_contacts": True}
    )
    tcp = scene.tcp_body_index
    dt = 1.0 / 600.0

    lagged = np.zeros(scene.robot_model.body_count * 6, dtype=np.float32)
    lagged[tcp * 6 : tcp * 6 + 3] = [12.0, 0.0, 0.0]
    scene.proxy_forces.assign(lagged)

    scene.coupled_substep(dt)
    cache_after_step1 = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp]
    harvest_after_step1 = scene.proxy_forces.numpy().reshape(-1, 6)[tcp].copy()

    np.testing.assert_allclose(cache_after_step1[:3], lagged[tcp * 6 : tcp * 6 + 3], rtol=1e-4, atol=1e-4)
    assert not np.allclose(harvest_after_step1, lagged[tcp * 6 : tcp * 6 + 6], atol=0.5)

    scene.coupled_substep(dt)
    cache_after_step2 = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp]
    np.testing.assert_allclose(
        cache_after_step2,
        harvest_after_step1,
        rtol=0.05,
        atol=0.5,
    )


def test_tcp_pose_matches_proxy_each_coupled_substep():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 10,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    tcp = scene.tcp_body_index
    proxy = scene.cable.gripper_proxy_body
    scene.robot_kinematic_mode = True
    ctrl = new_direct_controller(scene, fr3_robot)

    for step in range(30):
        if step % 30 == 0:
            apply_direct_hold(scene, fr3_robot, ctrl)
        scene.coupled_substep(1.0 / 600.0)
        rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
        pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
        pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
        ang_err = _quat_angle_error_rad(rq, pq)
        assert pos_err < 2e-3, f"TCP-proxy position drift {pos_err}"
        assert ang_err < 5e-3, f"TCP-proxy orientation drift {ang_err}"


def test_coupled_long_horizon_harvest_bounded():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 42,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    tcp = scene.tcp_body_index
    max_f = 0.0
    max_tau = 0.0
    scene.robot_kinematic_mode = True
    ctrl = new_direct_controller(scene, fr3_robot)

    for step in range(400):
        if step % 30 == 0:
            apply_direct_hold(scene, fr3_robot, ctrl)
        scene.coupled_substep(SUB_DT)
        w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
        max_f = max(max_f, float(np.linalg.norm(w[:3])))
        max_tau = max(max_tau, float(np.linalg.norm(w[3:6])))

    assert max_f < _QUIESCENT_HARVEST_F_CAP_N, (
        f"harvest |F| grew to {max_f:.2f} N over 400 substeps"
    )
    assert max_tau < 50.0, f"harvest |τ| grew to {max_tau:.2f} N·m over 400 substeps"


def test_mujoco_only_robot_matches_coupled_mujoco_phase():
    """Steps 1–3 of coupled_substep match mujoco_substep on identical initial state."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene_mjc = build_coupled_fr3(cf,
        ranges, 13, mujoco_only=True, mujoco_solver_kwargs={"disable_contacts": True}
    )
    scene_cpl = build_coupled_fr3(cf,
        ranges, 13, mujoco_solver_kwargs={"disable_contacts": True}
    )
    dt = 1.0 / 600.0

    for _ in range(8):
        scene_mjc.mujoco_substep(dt)
        scene_cpl._mujoco_and_sync_proxy(dt)
        rq_m = scene_mjc.robot_state_0.body_q.numpy()
        rq_c = scene_cpl.robot_state_0.body_q.numpy()
        np.testing.assert_allclose(rq_m, rq_c, rtol=1e-5, atol=1e-5)


def test_coupled_substep_is_deterministic():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    dt = 1.0 / 600.0

    def _run(seed: int):
        scene = build_coupled_fr3(cf,
            ranges, seed, mujoco_solver_kwargs={"disable_contacts": True}
        )
        for _ in range(50):
            scene.coupled_substep(dt)
        return (
            scene.robot_state_0.body_q.numpy(),
            scene.cable.state_0.body_q.numpy(),
            scene.proxy_forces.numpy(),
        )

    a = _run(17)
    b = _run(17)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a[2], b[2], rtol=1e-5, atol=1e-5)


def test_stem_apple_joint_index_set_when_fix_to_apple():
    """stem_apple_joint_index is populated when the proxy is welded to the apple."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
    )
    assert scene.stem_apple_joint_index is not None
    assert isinstance(scene.stem_apple_joint_index, int)
    assert scene.stem_apple_joint_index >= 0


def test_stem_apple_joint_index_none_for_free_proxy():
    """stem_apple_joint_index is None when the proxy is not welded to the apple."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
    )
    assert scene.stem_apple_joint_index is None


def test_stem_apple_joint_index_set_by_default():
    """Default coupled build uses free proxy (velocity-delta harvest); stem index is None."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1)
    assert scene.stem_apple_joint_index is None


def test_sync_teleports_apple_with_proxy_when_fix_to_apple():
    """mujoco_only substep co-teleports the apple body alongside the proxy when fix_to_apple=True."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 5,
        mujoco_only=True,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
    )
    cable = scene.cable
    apple = cable.apple_body
    proxy = cable.gripper_proxy_body
    assert apple is not None
    assert cable.gripper_proxy_offset_in_apple_frame is not None

    apple_before = cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3].copy()
    proxy_before = cable.state_0.body_q.numpy().reshape(-1, 7)[proxy, :3].copy()

    from apple_pick_sim import fr3_robot

    ctrl = new_direct_controller(scene, fr3_robot)
    apply_direct_hold(
        scene,
        fr3_robot,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.08, 0.0, 0.0)),
    )
    run_mujoco_substeps_direct_hold(scene, fr3_robot, 30, sub_dt=1.0 / 600.0)

    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    apple_after = bq[apple, :3]
    proxy_after = bq[proxy, :3]

    # Apple must have moved with the proxy (both teleported by sync).
    apple_disp = np.linalg.norm(apple_after - apple_before)
    proxy_disp = np.linalg.norm(proxy_after - proxy_before)
    assert apple_disp > 1e-4, f"apple did not move with proxy: disp={apple_disp}"
    np.testing.assert_allclose(apple_disp, proxy_disp, rtol=0.01, atol=1e-4,
                               err_msg="apple and proxy must move by the same distance")

    # The distance between proxy and apple must equal |offset| (rigid grasp maintained).
    offset = np.linalg.norm(cable.gripper_proxy_offset_in_apple_frame)
    gap = np.linalg.norm(bq[proxy, :3] - bq[apple, :3])
    np.testing.assert_allclose(gap, offset, rtol=1e-3, atol=1e-3,
                               err_msg="proxy-apple separation must equal grasp offset")


def test_stem_harvest_replaces_velocity_delta_when_fix_to_apple():
    """With fix_to_apple=True the proxy_forces slot is populated by the stem joint."""
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 14,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    assert scene.stem_apple_joint_index is not None
    tcp = scene.tcp_body_index
    ctrl = new_direct_controller(scene, fr3_robot)
    apply_direct_hold(
        scene,
        fr3_robot,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )
    run_coupled_substeps_direct_hold(
        scene,
        fr3_robot,
        5 * 30,
        sub_dt=1.0 / 600.0,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
    assert float(np.linalg.norm(w[:3])) > 0.0, (
        f"expected nonzero stem harvest after moving TCP, got {w}"
    )


def test_coupled_fix_to_apple_harvests_nonzero_when_robot_pushed():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 14,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    tcp = scene.tcp_body_index
    ctrl = new_direct_controller(scene, fr3_robot)
    apply_direct_hold(
        scene,
        fr3_robot,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.25, 0.0, 0.0)),
    )
    scene.coupled_substep(1.0 / 600.0)

    w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
    assert float(np.linalg.norm(w[:3])) > 0.5, f"expected non-trivial harvest, got {w}"


def test_fr3_coupled_substep_finite_state():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 3,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    dt = 1.0 / 600.0
    for _ in range(5):
        scene.coupled_substep(dt)
    tcp = scene.tcp_body_index
    rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    assert np.isfinite(rq).all()


def test_coupled_fr3_robot_gravity_zero_even_with_contacts_enabled():
    """Robot Model A has no gravity regardless of ``disable_contacts`` (cable keeps VBD gravity)."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 2,
        mujoco_solver_kwargs={"disable_contacts": False},
    )
    g = scene.robot_model.gravity.numpy().reshape(-1)
    np.testing.assert_allclose(g, [0.0, 0.0, 0.0], atol=1e-6)


def test_mujoco_opt_gravity_zero_after_fr3_coupled_build():
    """MuJoCo ``mj_model.opt.gravity`` must match zero-g robot model (not just Newton array)."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 2,
        mujoco_solver_kwargs={"disable_contacts": False},
    )
    g_newton = scene.robot_model.gravity.numpy().reshape(-1)
    g_mj = np.asarray(scene.mj_solver.mj_model.opt.gravity, dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(g_newton, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(g_mj, [0.0, 0.0, 0.0], atol=1e-6)


def test_cable_vbd_gravity_unchanged_after_fr3_coupled_build():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 2)
    g_cable = scene.cable.model.gravity.numpy().reshape(-1)
    np.testing.assert_allclose(g_cable, [0.0, 0.0, -9.81], atol=1e-3)
    np.testing.assert_allclose(
        np.array(scene.gravity_vec, dtype=np.float64),
        [0.0, 0.0, -9.81],
        atol=1e-3,
    )


def test_coupling_gravity_vec_matches_cable_model():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 3)
    g_cable = scene.cable.model.gravity.numpy().reshape(3)
    g_vec = np.array(scene.gravity_vec, dtype=np.float64)
    np.testing.assert_allclose(g_vec, g_cable, atol=1e-3)


def test_fr3_idle_teleop_zeros_joint_target_vel():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 4,
        mujoco_only=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    dt = 1.0 / 60.0
    for _ in range(5):
        scene.mujoco_substep(dt / 30.0)
    scene.apply_fr3_ee_teleop(
        dt,
        ctrl,
        velocity=fr3_robot.EEVelocity(),
    )
    qd_tgt = scene.robot_control.joint_target_vel.numpy()
    assert float(np.max(np.abs(qd_tgt))) < 1e-5


def test_fr3_ee_teleop_drives_mujoco_joint_targets():
    import warp as wp

    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 4,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    x_tgt0 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    q_cur = scene.robot_state_0.joint_q.numpy().copy()

    scene.apply_fr3_ee_teleop(
        0.05,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(1.0, 0.0, 0.0)),
    )

    x_tgt1 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    assert x_tgt1 > x_tgt0 + 0.04, f"IK target x did not advance: {x_tgt0} -> {x_tgt1}"
    q_tgt = scene.robot_control.joint_target_pos.numpy()
    assert float(np.linalg.norm(q_tgt - q_cur)) > 1e-3, "MuJoCo joint_target_pos should differ from current q"


def test_idle_teleop_joint_error_bounded():
    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 5,
        mujoco_only=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    frame_dt = 1.0 / 60.0
    sub_dt = frame_dt / 30.0
    for _ in range(40):
        scene.apply_fr3_ee_teleop(
            frame_dt,
            ctrl,
            velocity=fr3_robot.EEVelocity(),
        )
        for _ in range(30):
            scene.mujoco_substep(sub_dt)
        q_tgt = scene.robot_control.joint_target_pos.numpy().reshape(-1)
        q_cur = scene.robot_state_0.joint_q.numpy().reshape(-1)
        qd_tgt = scene.robot_control.joint_target_vel.numpy().reshape(-1)
        n_dof = int(scene.robot_model.joint_dof_count)
        assert float(np.max(np.abs(qd_tgt[:n_dof]))) < 1e-4
        assert float(np.linalg.norm(q_tgt[:n_dof] - q_cur[:n_dof])) < 0.15


def test_post_nudge_settles():
    import warp as wp

    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 6,
        mujoco_only=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    tcp = scene.tcp_body_index
    frame_dt = 1.0 / 60.0
    sub_dt = frame_dt / 30.0

    scene.apply_fr3_ee_teleop(
        frame_dt,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )
    for _ in range(30):
        scene.mujoco_substep(sub_dt)

    bq0 = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3].copy()

    for _ in range(90):
        scene.apply_fr3_ee_teleop(
            frame_dt,
            ctrl,
            velocity=fr3_robot.EEVelocity(),
        )
        for _ in range(30):
            scene.mujoco_substep(sub_dt)

    bq1 = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3]
    drift = float(np.linalg.norm(bq1 - bq0))
    assert drift < 0.08, f"TCP drifted {drift} m after idle settle"


def test_example_coupled_fruiting_fix_to_apple_parser_default():
    from apple_pick_sim import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args([])
    assert ex._fix_to_apple_from_args(args) is False
    assert ex._gripper_proxy_from_args(args, robot_kind="fr3").fix_to_apple is False


def test_example_coupled_fruiting_fix_to_apple_parser_enabled():
    from apple_pick_sim import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--fix-to-apple"])
    assert ex._fix_to_apple_from_args(args) is True
    assert ex._gripper_proxy_from_args(args, robot_kind="fr3").fix_to_apple is True


def test_example_coupled_fruiting_fix_to_apple_parser_disabled_explicit():
    from apple_pick_sim import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--no-fix-to-apple"])
    assert ex._fix_to_apple_from_args(args) is False


def test_gripper_proxy_config_default_fix_to_apple_false():
    fs = _import_fs()
    assert fs.GripperProxyConfig().fix_to_apple is False


def test_fr3_direct_teleop_kinematic_substep_preserves_joint_q():
    import numpy as np

    from apple_pick_sim import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 7,
        mujoco_only=True,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    scene.robot_kinematic_mode = True
    ctrl = fr3_robot.Fr3EEDirectJointController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    frame_dt = 1.0 / 60.0
    sub_dt = frame_dt / 30.0

    scene.apply_fr3_ee_teleop_direct(
        frame_dt,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.25, 0.0, 0.0)),
    )
    q_after_teleop = scene.robot_state_0.joint_q.numpy().copy()

    for _ in range(30):
        scene.mujoco_substep(sub_dt)

    q_after_substeps = scene.robot_state_0.joint_q.numpy()
    np.testing.assert_allclose(q_after_substeps, q_after_teleop, rtol=0, atol=1e-5)


def test_example_fr3_direct_joints_parser():
    from apple_pick_sim import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--robot", "fr3", "--fr3-direct-joints"])
    assert args.fr3_direct_joints is True
