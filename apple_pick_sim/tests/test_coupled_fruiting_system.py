"""M1 Slice 2b: FR3 + coupled cable VBD staggered loop.

Force and stability tests use :class:`~apple_pick_sim.robot.fr3_robot.Fr3EEDirectJointController`
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
    COUPLED_SCENE_KW,
    DEFAULT_MJ_KW,
    FRAME_DT,
    RANGES_FIXTURE,
    SUB_DT,
    SUBSTEPS_PER_FRAME,
    apply_direct_hold,
    build_coupled_fr3,
    build_vbd_only,
    new_direct_controller,
    requires_fr3,
    run_coupled_substeps_direct_hold,
    run_mujoco_substeps_direct_hold,
)

from apple_pick_sim.robot.fr3_robot.placement import (
    IK_BOOTSTRAP_POS_TOL_M,
    IK_BOOTSTRAP_ROT_TOL_RAD,
)
# Quiescent stem-harvest TCP load under direct-joint hold (aligned with coupling_stability).
_QUIESCENT_HARVEST_F_CAP_N = 500.0
_GRAVITY_MS2 = 9.81
_SETTLE_HOLD_FRAMES = 30
_DRIVE_HOLD_FRAMES = 45
_POST_DRIVE_HOLD_FRAMES = 20
# Welded fix_to_apple: read stem reaction during drive; long post-hold lets AVBD
# lambda warm-start dominate after the stem-apple constraint has equilibrated.
_WELDED_LATERAL_DRIVE_FRAMES = 20
_WELDED_LATERAL_POST_HOLD_FRAMES = 0
_MIN_LATERAL_DISP_M = 0.025
_MIN_LATERAL_FORCE_N = 15.0
_NUDGE_LATERAL_M = 0.05
_LATERAL_DRIVE_CASES = (
    pytest.param(0, (0.2, 0.0, 0.0), id="x_pos"),
    pytest.param(0, (-0.2, 0.0, 0.0), id="x_neg"),
    pytest.param(1, (0.0, 0.2, 0.0), id="y_pos"),
    pytest.param(1, (0.0, -0.2, 0.0), id="y_neg"),
)

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


def test_qd_synced_buffer_reused_across_substeps():
    """Pooled ``qd_synced`` is allocated once and reused (no per-substep ``wp.clone``)."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 8, mujoco_solver_kwargs=DEFAULT_MJ_KW)
    assert scene.qd_synced is not None
    buf_id = id(scene.qd_synced)
    dt = SUB_DT
    scene.coupled_substep(dt)
    assert id(scene.qd_synced) == buf_id
    scene.coupled_substep(dt)
    assert id(scene.qd_synced) == buf_id


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
    assert pos_err < IK_BOOTSTRAP_POS_TOL_M, (
        f"TCP/proxy position mismatch after IK bootstrap: {pos_err} m"
    )
    assert _quat_angle_error_rad(rq, pq) < IK_BOOTSTRAP_ROT_TOL_RAD


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


def test_fr3_ee_teleop_does_not_resync_robot_state_1():
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 6, mujoco_only=True)
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    sentinel = scene.robot_state_0.joint_q.numpy().copy() + 0.1
    scene.robot_state_1.joint_q.assign(sentinel.astype(scene.robot_state_0.joint_q.dtype))

    scene.update_fr3_ee_teleop(
        FRAME_DT,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    np.testing.assert_allclose(
        scene.robot_state_1.joint_q.numpy(),
        sentinel,
        rtol=1e-6,
        atol=1e-6,
    )


def test_fr3_ee_teleop_direct_does_not_resync_robot_state_1():
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 7, mujoco_only=True)
    ctrl = new_direct_controller(scene, fr3_robot)
    sentinel = scene.robot_state_0.joint_q.numpy().copy() + 0.1
    scene.robot_state_1.joint_q.assign(sentinel.astype(scene.robot_state_0.joint_q.dtype))

    scene.update_fr3_ee_teleop_direct(
        FRAME_DT,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    np.testing.assert_allclose(
        scene.robot_state_1.joint_q.numpy(),
        sentinel,
        rtol=1e-6,
        atol=1e-6,
    )


def test_velocity_controller_run_coupled_teleop_frame():
    import warp as wp

    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 8, mujoco_only=True)
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    x_tgt0 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    q_cur = scene.robot_state_0.joint_q.numpy().copy()

    vel = ctrl.run_coupled_teleop_frame(
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        FRAME_DT,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    x_tgt1 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    expected_dx = 0.2 * FRAME_DT
    assert x_tgt1 > x_tgt0 + expected_dx * 0.9
    q_tgt = scene.robot_control.joint_target_pos.numpy()
    assert float(np.linalg.norm(q_tgt - q_cur)) > 1e-3
    assert vel.linear == (0.2, 0.0, 0.0)


def test_impedance_controller_run_coupled_teleop_frame():
    import warp as wp

    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 9, mujoco_only=True)
    fr3_robot.configure_vic_wrench_only_arm(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
    )
    ctrl = fr3_robot.Fr3EEImpedanceController(tcp_body_index=int(scene.tcp_body_index))
    ctrl.sync_target_from_state(scene.robot_state_0)
    x_tgt0 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    q_tgt_before = scene.robot_control.joint_target_pos.numpy().copy()

    vel = ctrl.run_coupled_teleop_frame(
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        FRAME_DT,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    x_tgt1 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    expected_dx = 0.2 * FRAME_DT
    assert x_tgt1 > x_tgt0 + expected_dx * 0.9
    q_tgt_after = scene.robot_control.joint_target_pos.numpy()
    np.testing.assert_allclose(q_tgt_after, q_tgt_before, rtol=1e-6, atol=1e-6)
    assert vel.linear == (0.2, 0.0, 0.0)


def test_direct_controller_run_coupled_teleop_frame():
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 10, mujoco_only=True)
    ctrl = new_direct_controller(scene, fr3_robot)
    q_before = scene.robot_state_0.joint_q.numpy().copy()

    ctrl.run_coupled_teleop_frame(
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        FRAME_DT,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    q_after = scene.robot_state_0.joint_q.numpy()
    assert float(np.linalg.norm(q_after - q_before)) > 1e-3


def test_mujoco_substep_proxy_does_not_teleport_on_first_step():
    from apple_pick_sim.robot import fr3_robot

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
    from apple_pick_sim.robot import fr3_robot

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
    """Stem-harvest TCP load stays finite/capped when the arm is held via direct joints."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 8,
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


def test_add_tcp_spatial_wrench_inplace_sums_at_tcp_only():
    """In-place add at TCP index leaves other bodies unchanged."""
    cf = _import_cf()

    n = 3
    wrenches = wp.zeros(n, dtype=wp.spatial_vector, device="cpu")
    w_np = np.zeros((n, 6), dtype=np.float32)
    w_np[0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w_np[2] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    wrenches.assign(w_np.ravel())

    delta = wp.spatial_vector(0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    cf._add_tcp_spatial_wrench_inplace(wrenches, tcp_body_index=2, delta=delta)

    out = wrenches.numpy().reshape(n, 6)
    np.testing.assert_allclose(out[0], w_np[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out[1], 0.0, atol=1e-7)
    np.testing.assert_allclose(out[2], w_np[2] + [0.5, 0, 0, 0, 0, 0], rtol=1e-6, atol=1e-6)


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
    from apple_pick_sim.robot import fr3_robot

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
    apply_direct_hold(scene, fr3_robot, ctrl)
    # First VBD substep can re-orient a free proxy once body_q_prev aligns; warm up once.
    scene.coupled_substep(1.0 / 600.0)

    for step in range(30):
        scene.coupled_substep(1.0 / 600.0)
        rq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
        pq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[proxy]
        pos_err = float(np.linalg.norm(rq[:3] - pq[:3]))
        ang_err = _quat_angle_error_rad(rq, pq)
        assert pos_err < 2e-3, f"TCP-proxy position drift {pos_err}"
        # Free proxy is VBD-integrated (not prescribed); allow ~1° steady-state ori drift.
        assert ang_err < 0.02, f"TCP-proxy orientation drift {ang_err}"




@pytest.mark.slow
def test_coupled_long_horizon_harvest_bounded():
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 4,
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


def test_stem_apple_joint_index_set_for_free_proxy():
    """stem_apple_joint_index is set whenever the scene has an apple (stem-harvest at TCP)."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
    )
    assert scene.stem_apple_joint_index is not None


def test_stem_apple_joint_index_set_by_default():
    """Default coupled build has an apple; stem index is populated for TCP harvest."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 1)
    assert scene.stem_apple_joint_index is not None


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

    from apple_pick_sim.robot import fr3_robot

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
    offset = np.linalg.norm(cable.gripper_proxy_offset_in_apple_frame[:3])
    gap = np.linalg.norm(bq[proxy, :3] - bq[apple, :3])
    np.testing.assert_allclose(gap, offset, rtol=1e-3, atol=1e-3,
                               err_msg="proxy-apple separation must equal grasp offset")


def test_stem_harvest_at_tcp_when_free_proxy():
    """Free proxy still harvests stem→apple wrench at TCP (not velocity-delta on proxy mass)."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 15,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    assert scene.stem_apple_joint_index is not None
    tcp = scene.tcp_body_index
    ctrl = new_direct_controller(scene, fr3_robot)
    run_coupled_substeps_direct_hold(scene, fr3_robot, 60, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)
    w = scene.proxy_forces.numpy().reshape(-1, 6)[tcp]
    assert float(np.linalg.norm(w[:3])) > 0.5, (
        f"expected nonzero stem harvest at TCP for free proxy, got {w}"
    )


def test_stem_harvest_replaces_velocity_delta_when_fix_to_apple():
    """With fix_to_apple=True the proxy_forces slot is populated by the stem joint."""
    from apple_pick_sim.robot import fr3_robot

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
    from apple_pick_sim.robot import fr3_robot

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


def _scene_nominal_apple_body(scene) -> int | None:
    """Apple body index for stem harvest / explicit weight."""
    cable = scene.cable
    if getattr(cable, "apple_body", None) is not None:
        return int(cable.apple_body)
    nom = int(getattr(scene, "nominal_index", 0))
    if hasattr(cable, "instance"):
        return cable.instance(nom).apple_body
    return None


def _scene_grasp_offset_in_apple_frame(scene) -> tuple | None:
    """Grasp offset in apple frame for stem harvest / explicit weight."""
    cable = scene.cable
    off = getattr(cable, "gripper_proxy_offset_in_apple_frame", None)
    if off is not None:
        return off
    if hasattr(cable, "instance"):
        nom = int(getattr(scene, "nominal_index", 0))
        return cable.instance(nom).gripper_proxy_offset_in_apple_frame
    return None


def _explicit_apple_wrench_for_scene(scene) -> tuple[np.ndarray, np.ndarray]:
    """Explicit apple-weight force/torque about TCP (matches stem harvest)."""
    apple = _scene_nominal_apple_body(scene)
    if apple is None or scene.robot_state_0 is None:
        return np.zeros(3), np.zeros(3)
    from apple_pick_sim.coupled_fruiting.explicit_load import (
        apple_mass_kg_from_model,
        explicit_apple_wrench_for_stem_harvest,
    )

    cable = scene.cable
    m = apple_mass_kg_from_model(cable.model, apple)
    return explicit_apple_wrench_for_stem_harvest(
        mass_kg=m,
        gravity=scene.gravity_vec,
        robot_body_q=scene.robot_state_0.body_q,
        cable_body_q=cable.state_0.body_q,
        tcp_body_index=scene.tcp_body_index,
        apple_body_index=apple,
        grasp_offset_in_apple_frame=_scene_grasp_offset_in_apple_frame(scene),
    )


def _stem_force_with_explicit_apple_weight(scene, force: np.ndarray) -> np.ndarray:
    """Stem gather linear force plus optional explicit apple support."""
    f = np.asarray(force, dtype=np.float64)
    if not getattr(scene, "stem_harvest_explicit_apple_weight", False):
        return f
    f_add, _ = _explicit_apple_wrench_for_scene(scene)
    return f + f_add


def _stem_torque_with_explicit_apple_weight(scene, torque: np.ndarray) -> np.ndarray:
    """Stem gather torque plus optional ``(p_apple - p_tcp) × F_support``."""
    tau = np.asarray(torque, dtype=np.float64)
    if not getattr(scene, "stem_harvest_explicit_apple_weight", False):
        return tau
    _, tau_add = _explicit_apple_wrench_for_scene(scene)
    return tau + tau_add


def _stem_wrench_after_coupling_limits(
    force: np.ndarray,
    torque: np.ndarray,
    *,
    coupling_gain: float,
    force_cap_N: float | None,
    torque_cap_Nm: float | None,
) -> np.ndarray:
    """Match ``harvest_stem_tension_for_tcp`` gain + norm caps (world frame, 6-vector)."""
    f = np.asarray(force, dtype=np.float64) * float(coupling_gain)
    tau = np.asarray(torque, dtype=np.float64) * float(coupling_gain)
    if force_cap_N is not None and force_cap_N > 0.0:
        fn = float(np.linalg.norm(f))
        if fn > force_cap_N:
            f = f * (force_cap_N / fn)
    if torque_cap_Nm is not None and torque_cap_Nm > 0.0:
        tn = float(np.linalg.norm(tau))
        if tn > torque_cap_Nm:
            tau = tau * (torque_cap_Nm / tn)
    return np.concatenate([f, tau])


def _resolve_stem_apple_joint_index(scene) -> int:
    """Stem→apple FIXED joint (``stem_apple_joint_index`` or fruiting metadata)."""
    if scene.stem_apple_joint_index is not None:
        return int(scene.stem_apple_joint_index)
    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    jchild = cable.model.joint_child.numpy()
    for j_idx, _label in cable.fruiting_fixed_joints:
        if int(jchild[j_idx]) == apple:
            return int(j_idx)
    raise AssertionError("no stem→apple fixed joint found on cable model")


def _stem_apple_wrench_from_scene(scene, *, dt: float) -> np.ndarray:
    """Raw stem→apple joint wrench (same gather inputs as stem harvest)."""
    from apple_pick_sim.vbd_fixed_joint_wrenches import fixed_joint_wrenches_child_com_vbd

    cable = scene.cable
    stem_j = _resolve_stem_apple_joint_index(scene)
    records = fixed_joint_wrenches_child_com_vbd(
        cable.model,
        cable.solver,
        body_q=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        dt=dt,
        joint_pairs=[(stem_j, "stem_apple")],
    )
    assert len(records) == 1, "expected exactly one stem_apple joint record"
    rec = records[0]
    return np.concatenate(
        [rec.force_world.astype(np.float64), rec.torque_at_child_com_world.astype(np.float64)]
    )


def _build_welded_coupled_for_stem_tests(cf, fs, *, seed: int, **stem_kw):
    return build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        seed,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=96,
        **stem_kw,
    )


def _zero_gravity_coupled(scene) -> None:
    scene.gravity_vec = wp.vec3(0.0, 0.0, 0.0)
    scene.cable.model.set_gravity((0.0, 0.0, 0.0))


def _apple_mass_kg(scene) -> float:
    """``model.body_mass`` for the apple link (physical mass; integration via ``inv_mass``)."""
    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    return float(cable.model.body_mass.numpy()[apple])


def _body_position(scene, body_id: int) -> np.ndarray:
    bq = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    return bq[body_id, :3].astype(np.float64)


def _run_coupled_hold_frames(
    scene,
    fr3_robot,
    n_frames: int,
    *,
    velocity=None,
) -> None:
    """Teleop one frame per ``FRAME_DT``, each with ``SUBSTEPS_PER_FRAME`` coupled substeps."""
    ctrl = new_direct_controller(scene, fr3_robot)
    vel = velocity if velocity is not None else fr3_robot.EEVelocity()
    for _ in range(n_frames):
        apply_direct_hold(scene, fr3_robot, ctrl, velocity=vel)
        for _ in range(SUBSTEPS_PER_FRAME):
            scene.coupled_substep(SUB_DT)


def _apple_position(scene) -> np.ndarray:
    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    return bq[apple, :3].astype(np.float64)


def _drive_apple_lateral(
    scene,
    fr3_robot,
    linear: tuple[float, float, float],
    *,
    drive_hold_frames: int | None = None,
    post_hold_frames: int | None = None,
) -> np.ndarray:
    """Settle, teleop along ``linear``, hold; return apple displacement (world)."""
    hold_zero = fr3_robot.EEVelocity()
    _run_coupled_hold_frames(scene, fr3_robot, _SETTLE_HOLD_FRAMES, velocity=hold_zero)
    p0 = _apple_position(scene)
    _run_coupled_hold_frames(
        scene,
        fr3_robot,
        _DRIVE_HOLD_FRAMES if drive_hold_frames is None else int(drive_hold_frames),
        velocity=fr3_robot.EEVelocity(linear=linear),
    )
    post_n = _POST_DRIVE_HOLD_FRAMES if post_hold_frames is None else int(post_hold_frames)
    if post_n > 0:
        _run_coupled_hold_frames(scene, fr3_robot, post_n, velocity=hold_zero)
    return _apple_position(scene) - p0


def _nudge_apple_lateral_vbd(
    scene,
    axis: int,
    delta_m: float,
    *,
    relax_substeps: int = 8,
) -> float:
    """Impose a lateral apple offset on ``state_0``, short VBD relax; return imposed ``delta_m``.

    Only ``state_0`` is nudged so ``state_1`` retains the pre-offset pose for AVBD.
    Keep ``relax_substeps`` modest: long relaxation lets warm-started lambdas dominate
    after the stem constraint equilibrates, flipping the reported restoring sign.
    """
    cf = _import_cf()
    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    bq = cable.state_0.body_q.numpy().reshape(-1, 7).copy()
    bq[apple, axis] += float(delta_m)
    cable.state_0.body_q.assign(bq.reshape(-1))
    cf.settle_vbd_substeps(scene, substeps=int(relax_substeps), dt=SUB_DT)
    scene.vbd_substep(SUB_DT)
    return float(delta_m)


def _lever_arm_tcp_to_apple_com(scene) -> np.ndarray | None:
    """World-frame vector from TCP to apple COM (matches stem harvest kernel)."""
    apple = _scene_nominal_apple_body(scene)
    if apple is None or scene.robot_state_0 is None:
        return None
    from apple_pick_sim.coupled_fruiting.explicit_load import (
        apple_com_from_tcp_grasp_offset,
        body_com_position_world,
        body_orientation_world,
    )

    cable = scene.cable
    tcp = scene.tcp_body_index
    p_tcp = body_com_position_world(scene.robot_state_0.body_q, tcp)
    grasp_off = _scene_grasp_offset_in_apple_frame(scene)
    if grasp_off is not None:
        tcp_rot = body_orientation_world(scene.robot_state_0.body_q, tcp)
        p_apple = apple_com_from_tcp_grasp_offset(p_tcp, tcp_rot, grasp_off)
    else:
        p_apple = body_com_position_world(cable.state_0.body_q, apple)
    return p_apple - p_tcp


def _expected_tcp_stem_harvest_from_gather(scene, stem: np.ndarray) -> np.ndarray:
    """Reference TCP wrench mirroring ``_limit_and_write_tcp_stem_wrench_kernel``."""
    f_stem = np.asarray(stem[:3], dtype=np.float64)
    tau_stem = np.asarray(stem[3:], dtype=np.float64)
    f_total = f_stem.copy()
    tau_total = tau_stem.copy()
    r = _lever_arm_tcp_to_apple_com(scene)
    if r is not None:
        if getattr(scene, "stem_harvest_explicit_apple_weight", False):
            f_add, _ = _explicit_apple_wrench_for_scene(scene)
            f_total = f_total + f_add
        tau_total = tau_total + np.cross(r, f_total)
    return _stem_wrench_after_coupling_limits(
        f_total,
        tau_total,
        coupling_gain=scene.stem_coupling_gain,
        force_cap_N=scene.stem_force_cap_N,
        torque_cap_Nm=scene.stem_torque_cap_Nm,
    )


def _assert_tcp_matches_stem(
    scene,
    tcp: int,
    *,
    rtol: float = 0.02,
    atol: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Stem gather and TCP harvest agree after coupling gain/caps."""
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench

    stem = _stem_apple_wrench_from_scene(scene, dt=SUB_DT)
    expected = _expected_tcp_stem_harvest_from_gather(scene, stem)
    tcp_w = read_tcp_wrench(scene.proxy_forces, tcp).astype(np.float64)
    np.testing.assert_allclose(tcp_w, expected, rtol=rtol, atol=atol)
    return stem, tcp_w


def test_fix_to_apple_tcp_harvest_matches_stem_apple_joint():
    """``proxy_forces[tcp]`` matches stem gather (+ explicit apple weight) when gain=1."""
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=21,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    tcp = scene.tcp_body_index
    run_coupled_substeps_direct_hold(scene, fr3_robot, 90, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)

    _assert_tcp_matches_stem(scene, tcp)

    stem_raw = _stem_apple_wrench_from_scene(scene, dt=SUB_DT)
    assert float(np.linalg.norm(stem_raw[:3])) > 1.0, (
        f"expected meaningful stem load under gravity, got F={stem_raw[:3]}"
    )


def test_fix_to_apple_tcp_harvest_applies_stem_coupling_gain():
    """Under-relaxation: TCP harvest equals ``stem_coupling_gain`` × stem joint wrench."""
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.robot import fr3_robot

    under_relax_gain = 0.15
    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=22,
        stem_coupling_gain=under_relax_gain,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    assert scene.stem_coupling_gain == under_relax_gain
    tcp = scene.tcp_body_index
    ctrl = new_direct_controller(scene, fr3_robot)
    apply_direct_hold(
        scene,
        fr3_robot,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.15, 0.0, 0.0)),
    )
    run_coupled_substeps_direct_hold(
        scene,
        fr3_robot,
        60,
        sub_dt=SUB_DT,
        velocity=fr3_robot.EEVelocity(linear=(0.15, 0.0, 0.0)),
    )
    scene.coupled_substep(SUB_DT)

    stem_raw = _stem_apple_wrench_from_scene(scene, dt=SUB_DT)
    tcp_w = read_tcp_wrench(scene.proxy_forces, tcp).astype(np.float64)
    _assert_tcp_matches_stem(scene, tcp)
    f_with_explicit = _stem_force_with_explicit_apple_weight(scene, stem_raw[:3])
    np.testing.assert_allclose(
        tcp_w[:3],
        f_with_explicit * under_relax_gain,
        rtol=0.03,
        atol=0.5,
    )


def test_fix_to_apple_apple_retains_body_mass():
    """Prescribed VBD integration zeros ``inv_mass`` only; ``body_mass`` stays analytic."""
    from apple_pick_sim.fruiting_system.params import analytic_apple_mass_kg

    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        44,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
    )
    apple = scene.cable.apple_body
    assert apple is not None
    m_model = float(scene.cable.model.body_mass.numpy()[apple])
    inv_m = float(scene.cable.model.body_inv_mass.numpy()[apple])
    m_exp = analytic_apple_mass_kg(scene.cable.params)
    assert m_exp is not None
    np.testing.assert_allclose(m_model, m_exp, rtol=1e-5, atol=1e-8)
    assert inv_m == 0.0, f"apple inv_mass should be 0 after prescribe, got {inv_m}"


def _assert_tcp_stem_load_order_of_apple_weight(
    scene,
    tcp: int,
    *,
    m_apple: float,
    min_ratio: float = 0.5,
    max_ratio: float | None = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stem harvest at TCP matches gather (+ explicit apple weight) and is O(m·g)."""
    expected = float(m_apple) * _GRAVITY_MS2
    stem, tcp_w = _assert_tcp_matches_stem(scene, tcp)
    f_mag = float(np.linalg.norm(tcp_w[:3]))
    assert f_mag >= min_ratio * expected, (
        f"TCP |F|={f_mag:.3f} N below {min_ratio}×m·g={min_ratio * expected:.3f} N"
    )
    if max_ratio is not None:
        assert f_mag <= max_ratio * expected, (
            f"TCP |F|={f_mag:.3f} N above {max_ratio}×m·g={max_ratio * expected:.3f} N"
        )
    return stem, tcp_w


@pytest.mark.slow
def test_coupled_fr3_tcp_stem_load_at_hold_free_proxy():
    """Free proxy: stem harvest at TCP (gain=1) with |F| on the order of m_apple·g."""
    from apple_pick_sim.fruiting_system.params import analytic_apple_mass_kg
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        41,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    cf.settle_vbd_substeps(scene, substeps=600, dt=SUB_DT)
    _run_coupled_hold_frames(scene, fr3_robot, 100)
    scene.coupled_substep(SUB_DT)
    m_apple = analytic_apple_mass_kg(scene.cable.params)
    assert m_apple is not None and m_apple > 0.01
    _assert_tcp_stem_load_order_of_apple_weight(
        scene, scene.tcp_body_index, m_apple=m_apple, min_ratio=0.5, max_ratio=3.5
    )


@pytest.mark.slow
def test_coupled_fr3_tcp_stem_load_at_hold_welded():
    """Welded: stem harvest at TCP with upward support and O(m·g) magnitude."""
    from apple_pick_sim.fruiting_system.params import analytic_apple_mass_kg
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=45,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    _run_coupled_hold_frames(scene, fr3_robot, 100)
    scene.coupled_substep(SUB_DT)
    m_apple = analytic_apple_mass_kg(scene.cable.params)
    assert m_apple is not None
    stem, tcp_w = _assert_tcp_stem_load_order_of_apple_weight(
        scene, scene.tcp_body_index, m_apple=m_apple, min_ratio=0.5, max_ratio=None
    )
    assert float(tcp_w[2]) > 0.5 * m_apple * _GRAVITY_MS2, (
        f"expected upward TCP support, Fz={tcp_w[2]:.2f} N"
    )


@pytest.mark.slow
def test_welded_coupled_holding_hanging_tree_stem_reaction_upward():
    """Welded + gravity hold: stem–apple reaction is upward; TCP harvest matches gather."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=31,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    tcp = scene.tcp_body_index
    _run_coupled_hold_frames(scene, fr3_robot, 100)
    scene.coupled_substep(SUB_DT)

    stem, _ = _assert_tcp_matches_stem(scene, tcp)
    assert stem[2] > 1.0, f"expected upward stem support under gravity, Fz={stem[2]:.2f} N"


@pytest.mark.slow
def test_welded_coupled_vertical_pull_produces_upward_stem_tension():
    """+Z teleop lifts the apple; stem–apple reaction stays upward with meaningful magnitude."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=33,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    tcp = scene.tcp_body_index
    apple = scene.cable.apple_body
    assert apple is not None
    hold_zero = fr3_robot.EEVelocity()
    _run_coupled_hold_frames(scene, fr3_robot, _SETTLE_HOLD_FRAMES, velocity=hold_zero)
    z0 = _body_position(scene, apple)[2]
    _run_coupled_hold_frames(
        scene,
        fr3_robot,
        _DRIVE_HOLD_FRAMES,
        velocity=fr3_robot.EEVelocity(linear=(0.0, 0.0, 0.2)),
    )
    _run_coupled_hold_frames(scene, fr3_robot, _POST_DRIVE_HOLD_FRAMES, velocity=hold_zero)
    scene.coupled_substep(SUB_DT)

    z1 = _body_position(scene, apple)[2]
    stem_pull, tcp_pull = _assert_tcp_matches_stem(scene, tcp)

    assert z1 > z0 + 0.01, f"apple should rise with +Z teleop: dz={z1 - z0:.4f} m"
    assert stem_pull[2] > 5.0, f"expected upward stem load while lifting, Fz={stem_pull[2]:.2f} N"
    assert tcp_pull[2] > 5.0


@pytest.mark.slow
def test_coupled_stem_vertical_force_matches_apple_weight():
    """Fixture fruiting scene at coupled base: stem–apple Fz ≈ m·g after settle (±10 %).

    Reuses the proven ``test_wrench_equilibrium`` settling path (collision pipeline +
    AVBD dual convergence). Coupled FR3/proxy transients are out of scope for this check.
    """
    from apple_pick_sim.tests.test_wrench_equilibrium import _get_wrenches_by_label, _settle

    fs = _import_fs()
    # High anchor (same as ``test_wrench_equilibrium``) so the chain clears z=0 while settling.
    scene = fs.generate_scene(
        fs.load_ranges(RANGES_FIXTURE),
        seed=31,
        device="cpu",
        enable_self_collisions=False,
        base_pos=(0.0, 0.0, 4.0),
    )
    q_prev, sim_dt = _settle(scene)
    apple = scene.apple_body
    assert apple is not None
    m_apple = float(scene.model.body_mass.numpy()[apple])
    assert m_apple > 0.01
    expected_fz = m_apple * _GRAVITY_MS2
    w = _get_wrenches_by_label(scene, q_prev, sim_dt)["joint_stem_apple"]
    fz = float(w.force_world[2])

    np.testing.assert_allclose(
        fz,
        expected_fz,
        rtol=0.10,
        atol=0.35,
        err_msg=(
            f"stem_apple Fz={fz:.3f} N should ≈ m·g={expected_fz:.3f} N "
            f"(m_apple={m_apple:.4f} kg)"
        ),
    )
    assert fz > 0.5, "hanging apple: stem reaction should be upward (+Z)"
    torque_thresh = 0.02 * m_apple * _GRAVITY_MS2 * 0.05
    tau = np.asarray(w.torque_at_child_com_world, dtype=np.float64)
    assert float(np.linalg.norm(tau)) < torque_thresh, (
        f"quasi-static: |τ|={np.linalg.norm(tau):.4f} N·m exceeds {torque_thresh:.4f} N·m"
    )


@pytest.mark.slow
@pytest.mark.parametrize("axis,linear", _LATERAL_DRIVE_CASES)
def test_free_proxy_lateral_stem_restoring_force(axis: int, linear: tuple[float, float, float]):
    """Free apple (VBD-only, zero-g): stem–apple force opposes imposed lateral offset."""
    cf = _import_cf()
    fs = _import_fs()
    scene = build_vbd_only(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        43 + axis,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
    )
    _zero_gravity_coupled(scene)
    cf.settle_vbd_substeps(scene, substeps=200, dt=SUB_DT)
    drive_sign = float(np.sign(linear[axis]))
    assert drive_sign != 0.0
    imposed = _nudge_apple_lateral_vbd(scene, axis, drive_sign * _NUDGE_LATERAL_M)

    stem = _stem_apple_wrench_from_scene(scene, dt=SUB_DT)
    f = stem[:3]
    min_force = 8.0
    assert abs(f[axis]) >= min_force, (
        f"stem force too small along axis {axis}: F={f[axis]:.2f} N (min {min_force})"
    )
    assert np.sign(f[axis]) == -np.sign(imposed), (
        f"restoring stem force: imposed Δ={imposed:.4f} m, F[{axis}]={f[axis]:.2f} N"
    )


@pytest.mark.slow
@pytest.mark.parametrize("axis,linear", _LATERAL_DRIVE_CASES)
def test_welded_coupled_lateral_drive_restoring_force(axis: int, linear: tuple[float, float, float]):
    """Zero-g lateral teleop: stem/TCP force opposes apple motion on the driven axis."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=32 + axis,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    _zero_gravity_coupled(scene)
    tcp = scene.tcp_body_index
    disp = _drive_apple_lateral(
        scene,
        fr3_robot,
        linear,
        drive_hold_frames=_WELDED_LATERAL_DRIVE_FRAMES,
        post_hold_frames=_WELDED_LATERAL_POST_HOLD_FRAMES,
    )

    stem, tcp_w = _assert_tcp_matches_stem(scene, tcp)
    f = stem[:3]

    assert abs(disp[axis]) >= _MIN_LATERAL_DISP_M, (
        f"apple moved too little along axis {axis}: disp={disp[axis]:.4f} m"
    )
    assert abs(f[axis]) >= _MIN_LATERAL_FORCE_N, (
        f"stem force too small along axis {axis}: F={f[axis]:.2f} N"
    )
    assert np.sign(f[axis]) == -np.sign(disp[axis]), (
        f"restoring force: disp[{axis}]={disp[axis]:.4f}, F[{axis}]={f[axis]:.2f}"
    )
    assert np.sign(tcp_w[axis]) == np.sign(f[axis])


@pytest.mark.slow
def test_welded_coupled_opposite_lateral_drive_flips_stem_force():
    """Reversing teleop direction reverses stem–apple force on that axis."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene_pos = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=40,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    scene_neg = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=41,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    for scene in (scene_pos, scene_neg):
        _zero_gravity_coupled(scene)

    _drive_apple_lateral(
        scene_pos,
        fr3_robot,
        (0.0, 0.2, 0.0),
        drive_hold_frames=_WELDED_LATERAL_DRIVE_FRAMES,
        post_hold_frames=_WELDED_LATERAL_POST_HOLD_FRAMES,
    )
    f_pos = _stem_apple_wrench_from_scene(scene_pos, dt=SUB_DT)[1]

    _drive_apple_lateral(
        scene_neg,
        fr3_robot,
        (0.0, -0.2, 0.0),
        drive_hold_frames=_WELDED_LATERAL_DRIVE_FRAMES,
        post_hold_frames=_WELDED_LATERAL_POST_HOLD_FRAMES,
    )
    f_neg = _stem_apple_wrench_from_scene(scene_neg, dt=SUB_DT)[1]

    assert abs(f_pos) >= _MIN_LATERAL_FORCE_N and abs(f_neg) >= _MIN_LATERAL_FORCE_N
    assert np.sign(f_pos) != np.sign(f_neg), (
        f"expected opposite F_y: pos drive={f_pos:.2f}, neg drive={f_neg:.2f}"
    )


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
    from apple_pick_sim.robot import fr3_robot

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
    scene.update_fr3_ee_teleop(
        dt,
        ctrl,
        velocity=fr3_robot.EEVelocity(),
    )
    qd_tgt = scene.robot_control.joint_target_vel.numpy()
    assert float(np.max(np.abs(qd_tgt))) < 1e-5


def test_fr3_ee_teleop_drives_mujoco_joint_targets():
    import warp as wp

    from apple_pick_sim.robot import fr3_robot

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

    scene.update_fr3_ee_teleop(
        0.05,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    x_tgt1 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    assert x_tgt1 > x_tgt0 + 0.008, f"IK target x did not advance: {x_tgt0} -> {x_tgt1}"
    q_tgt = scene.robot_control.joint_target_pos.numpy()
    assert float(np.linalg.norm(q_tgt - q_cur)) > 1e-3, "MuJoCo joint_target_pos should differ from current q"


def test_idle_teleop_joint_error_bounded():
    from apple_pick_sim.robot import fr3_robot

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
        scene.update_fr3_ee_teleop(
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

    from apple_pick_sim.robot import fr3_robot

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

    scene.update_fr3_ee_teleop(
        frame_dt,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )
    for _ in range(30):
        scene.mujoco_substep(sub_dt)

    bq0 = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3].copy()

    for _ in range(90):
        scene.update_fr3_ee_teleop(
            frame_dt,
            ctrl,
            velocity=fr3_robot.EEVelocity(),
        )
        for _ in range(30):
            scene.mujoco_substep(sub_dt)

    bq1 = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3]
    drift = float(np.linalg.norm(bq1 - bq0))
    assert drift < 0.08, f"TCP drifted {drift} m after idle settle"


def test_example_coupled_fruiting_enable_self_collision_parser_default():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args([])
    assert ex._enable_self_collisions_from_args(args) is False


def test_example_coupled_fruiting_enable_self_collision_parser_enabled():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--enable-self-collision"])
    assert ex._enable_self_collisions_from_args(args) is True


def test_example_coupled_fruiting_fix_to_apple_parser_default():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args([])
    assert ex._fix_to_apple_from_args(args) is False
    assert ex._gripper_proxy_from_args(args, robot_kind="fr3").fix_to_apple is False


def test_example_coupled_fruiting_fix_to_apple_parser_enabled():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--fix-to-apple"])
    assert ex._fix_to_apple_from_args(args) is True
    assert ex._gripper_proxy_from_args(args, robot_kind="fr3").fix_to_apple is True


def test_example_coupled_fruiting_fix_to_apple_parser_disabled_explicit():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--no-fix-to-apple"])
    assert ex._fix_to_apple_from_args(args) is False


def test_gripper_proxy_config_default_fix_to_apple_false():
    fs = _import_fs()
    assert fs.GripperProxyConfig().fix_to_apple is False


def test_fr3_direct_teleop_kinematic_substep_preserves_joint_q():
    import numpy as np

    from apple_pick_sim.robot import fr3_robot

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

    scene.update_fr3_ee_teleop_direct(
        frame_dt,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.25, 0.0, 0.0)),
    )
    q_after_teleop = scene.robot_state_0.joint_q.numpy().copy()

    for _ in range(30):
        scene.mujoco_substep(sub_dt)

    q_after_substeps = scene.robot_state_0.joint_q.numpy()
    np.testing.assert_allclose(q_after_substeps, q_after_teleop, rtol=0, atol=1e-5)


def test_example_coupled_fruiting_default_robot_is_fr3():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args([])
    assert args.robot == "fr3"
    assert args.vic_linear_k == 800.0
    assert args.vic_angular_d == 4.0


def test_example_coupled_fruiting_controller_parser_default():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args([])
    assert ex._resolve_controller_mode(args) == "vic"
    assert args.controller == "vic"


def test_example_coupled_fruiting_controller_parser_ee():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--controller", "ee"])
    assert ex._resolve_controller_mode(args) == "ee"


def test_example_coupled_fruiting_controller_parser_direct():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    args = ex._make_parser().parse_args(["--controller", "direct"])
    assert ex._resolve_controller_mode(args) == "direct"
