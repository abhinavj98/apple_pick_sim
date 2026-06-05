"""Integration tests for dynamic-arm VIC on the coupled FR3 stack."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.tests.conftest import (
    COUPLED_BASE_POS,
    COUPLED_ROBOT_BASE_POS,
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

_BUILD_KW = dict(
    base_pos=COUPLED_BASE_POS,
    robot_base_pos=COUPLED_ROBOT_BASE_POS,
    enable_self_collisions=False,
    mujoco_solver_kwargs=DEFAULT_MJ_KW,
)


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _import_fs():
    import apple_pick_sim.fruiting_system as fs

    return fs


def _tcp_pos_x(scene) -> float:
    tcp = scene.tcp_body_index
    return float(scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, 0])


def _seed_lagged_tcp_wrench(scene, force_x: float) -> None:
    tcp = scene.tcp_body_index
    w_np = scene.proxy_forces.numpy().reshape(-1, 6).copy()
    w_np[tcp] = [force_x, 0.0, 0.0, 0.0, 0.0, 0.0]
    scene.proxy_forces.assign(w_np.ravel())


def _build_mujoco_only_fr3():
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(cf, ranges, 11, mujoco_only=True, **_BUILD_KW)
    fr3_robot.init_mujoco_actuator_targets_from_model(scene.robot_model, scene.robot_control)
    return scene


def _tcp_spatial_linear_xd(scene) -> float:
    tcp = scene.tcp_body_index
    return float(scene.robot_state_0.body_qd.numpy().reshape(-1, 6)[tcp, 0])


def _soften_mujoco_pd(scene, scale: float = 0.02) -> None:
    """Reduce position-actuator stiffness so external TCP wrenches can integrate."""
    fr3_robot.scale_mujoco_joint_pd(scene.robot_model, scale)


def test_dynamic_tcp_wrench_moves_arm():
    scene = _build_mujoco_only_fr3()
    _soften_mujoco_pd(scene)

    scene.robot_kinematic_mode = False
    _seed_lagged_tcp_wrench(scene, force_x=80.0)
    for _ in range(80):
        scene.mujoco_substep(SUB_DT)

    v_dynamic = abs(_tcp_spatial_linear_xd(scene))
    assert v_dynamic > 1e-5

    scene2 = _build_mujoco_only_fr3()
    _soften_mujoco_pd(scene2)
    scene2.robot_kinematic_mode = True
    _seed_lagged_tcp_wrench(scene2, force_x=80.0)
    for _ in range(80):
        scene2.mujoco_substep(SUB_DT)
    v_kinematic = abs(_tcp_spatial_linear_xd(scene2))
    assert v_kinematic < 1e-5
    assert v_dynamic > v_kinematic * 50.0


def test_vic_teleop_integrates_tcp_motion():
    """VIC teleop must move the dynamic arm (requires MuJoCo PD sync after zeroing ke/kd)."""
    scene = _build_mujoco_only_fr3()
    scene.robot_kinematic_mode = False
    scene.vic_controller = Fr3EEImpedanceController()
    scene.vic_gains = ImpedanceGains(linear_k=800.0, linear_d=80.0)
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    vel = fr3_robot.EEVelocity(linear=(0.5, 0.0, 0.0))
    x0 = _tcp_pos_x(scene)
    for _ in range(60):
        scene.apply_fr3_ee_teleop(FRAME_DT, ctrl, velocity=vel)
        for _ in range(SUBSTEPS_PER_FRAME):
            scene.mujoco_substep(SUB_DT)
    dx = _tcp_pos_x(scene) - x0
    assert dx > 0.05, f"expected VIC teleop to advance TCP +X, got dx={dx:.6f} m"


def test_vic_teleop_is_wrench_only_no_joint_pd():
    """With ``vic_controller`` attached, teleop advances TCP target but disables joint PD."""
    scene = _build_mujoco_only_fr3()
    scene.robot_kinematic_mode = False
    scene.vic_controller = Fr3EEImpedanceController()
    scene.vic_gains = ImpedanceGains(linear_k=100.0, linear_d=10.0)
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    q_before = scene.robot_state_0.joint_q.numpy().reshape(-1).copy()
    x_tgt0 = float(wp.transform_get_translation(ctrl.target_tf)[0])

    scene.apply_fr3_ee_teleop(
        FRAME_DT,
        ctrl,
        velocity=fr3_robot.EEVelocity(linear=(0.2, 0.0, 0.0)),
    )

    x_tgt1 = float(wp.transform_get_translation(ctrl.target_tf)[0])
    expected_dx = 0.2 * FRAME_DT
    assert x_tgt1 > x_tgt0 + expected_dx * 0.9, "TCP target should advance under teleop velocity"
    ke = scene.robot_model.joint_target_ke.numpy()
    kd = scene.robot_model.joint_target_kd.numpy()
    assert float(np.max(np.abs(ke))) < 1e-6
    assert float(np.max(np.abs(kd))) < 1e-6
    q_tgt = scene.robot_control.joint_target_pos.numpy().reshape(-1)
    q_cur = scene.robot_state_0.joint_q.numpy().reshape(-1)
    n_dof = int(scene.robot_model.joint_dof_count)
    assert float(np.linalg.norm(q_tgt[:n_dof] - q_cur[:n_dof])) < 1e-5
    assert float(np.max(np.abs(scene.robot_control.joint_target_vel.numpy()))) < 1e-5
    assert float(np.linalg.norm(q_cur - q_before)) < 0.02, "teleop frame must not teleport joint_q"


def test_harvest_excludes_applied_wrench():
    cf = _import_cf()
    from apple_pick_sim.coupled_fruiting.vic_wrench import apply_vic_to_coupling_cache

    scene = _build_mujoco_only_fr3()
    scene.robot_kinematic_mode = False
    scene.vic_controller = Fr3EEImpedanceController()
    scene.vic_gains = ImpedanceGains(linear_k=500.0, linear_d=50.0, angular_k=20.0, angular_d=2.0)
    ctrl = fr3_robot.Fr3EEVelocityController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    pos = wp.transform_get_translation(ctrl.target_tf)
    ctrl.target_tf = wp.transform(
        wp.vec3(float(pos[0]) + 0.05, float(pos[1]), float(pos[2])),
        wp.transform_get_rotation(ctrl.target_tf),
    )
    scene.vic_target_tf = ctrl.target_tf
    scene.vic_target_twist = fr3_robot.EEVelocity()

    pf_before = scene.proxy_forces.numpy().copy()
    scene.coupling_forces_cache.assign(scene.proxy_forces)
    apply_vic_to_coupling_cache(scene)
    np.testing.assert_allclose(scene.proxy_forces.numpy(), pf_before, rtol=0, atol=1e-9)

    tcp = scene.tcp_body_index
    cache_tcp = scene.coupling_forces_cache.numpy().reshape(-1, 6)[tcp]
    pf_tcp = pf_before.reshape(-1, 6)[tcp]
    assert float(np.linalg.norm(cache_tcp - pf_tcp)) > 0.5


def _build_welded_direct(seed: int = 0):
    """Welded grasp without settle→weld (deterministic rebuild for paired comparisons)."""
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    gripper_kw = dict(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
    )
    return build_coupled_fr3(
        cf,
        ranges,
        seed,
        **_BUILD_KW,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, **gripper_kw),
    )


def _build_welded_scene(seed: int = 0):
    cf = _import_cf()
    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    gripper_kw = dict(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
    )
    settled = cf.build_coupled_fruiting_fr3(
        ranges,
        seed,
        vbd_only=True,
        **_BUILD_KW,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False, **gripper_kw),
    )
    cf.settle_vbd_substeps(settled, substeps=60, dt=SUB_DT)
    welded = cf.build_coupled_fruiting_fr3(
        ranges,
        seed,
        **_BUILD_KW,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, **gripper_kw),
    )
    cf.seed_fix_to_apple_from_settled(welded_scene=welded, settled_scene=settled, quiet_apple_proxy=True)
    return welded


def _tcp_target_pos_error(scene, ctrl: fr3_robot.Fr3EEVelocityController) -> float:
    tcp = scene.tcp_body_index
    bq = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    target = ctrl.target_tf
    p_des = wp.transform_get_translation(target)
    e = np.array([float(p_des[0]) - bq[0], float(p_des[1]) - bq[1], float(p_des[2]) - bq[2]])
    return float(np.linalg.norm(e))


@pytest.mark.slow
def test_vic_stem_deflection_under_load():
    """Dynamic VIC tracks less tightly than kinematic hold when stem load feeds back."""
    welded = _build_welded_direct(seed=0)
    vel = fr3_robot.EEVelocity(linear=(0.0, 0.15, 0.0))

    # Kinematic baseline (direct joints, no MuJoCo integration).
    kin = _build_welded_direct(seed=0)
    kin.robot_kinematic_mode = True
    kin_ctrl = fr3_robot.Fr3EEDirectJointController(kin.robot_model, kin.tcp_body_index)
    kin_ctrl.sync_target_from_state(kin.robot_state_0)
    kin.apply_fr3_ee_teleop_direct(FRAME_DT, kin_ctrl, velocity=vel)
    for _ in range(SUBSTEPS_PER_FRAME * 8):
        kin.coupled_substep(SUB_DT)
    kin_err = _tcp_target_pos_error(kin, kin_ctrl)

    # Dynamic VIC path.
    welded.robot_kinematic_mode = False
    welded.vic_controller = Fr3EEImpedanceController()
    welded.vic_gains = ImpedanceGains(linear_k=400.0, linear_d=60.0, angular_k=30.0, angular_d=3.0)
    vic_ctrl = fr3_robot.Fr3EEVelocityController(welded.robot_model, welded.tcp_body_index)
    vic_ctrl.sync_target_from_state(welded.robot_state_0)
    fr3_robot.init_mujoco_actuator_targets_from_model(welded.robot_model, welded.robot_control)
    welded.apply_fr3_ee_teleop(FRAME_DT, vic_ctrl, velocity=vel)
    for _ in range(SUBSTEPS_PER_FRAME * 8):
        welded.coupled_substep(SUB_DT)
    vic_err = _tcp_target_pos_error(welded, vic_ctrl)

    assert vic_err >= kin_err * 0.5
    assert vic_err > 1e-4
