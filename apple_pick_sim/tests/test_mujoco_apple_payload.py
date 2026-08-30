"""Tests for MuJoCo inertia-only apple payload (mass/COM/I, welded builds)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from apple_pick_sim.fruiting_system.params import FruitingSystemParams, analytic_apple_mass_kg
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
    Fr3EEImpedanceController,
    ImpedanceGains,
)
from apple_pick_sim.tests.test_coupled_fruiting_system import _import_cf, _import_fs
from conftest import (
    COUPLED_BASE_POS,
    COUPLED_ROBOT_BASE_POS,
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    SUB_DT,
    build_coupled_fr3,
    requires_fr3,
)


def test_solid_sphere_inertia_diag():
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import solid_sphere_inertia_diag

    m, r = 0.2, 0.04
    I = solid_sphere_inertia_diag(m, r)
    expected = 0.4 * m * r * r
    np.testing.assert_allclose(
        [I[0, 0], I[1, 1], I[2, 2]],
        [expected, expected, expected],
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [I[0, 1], I[0, 2], I[1, 0], I[1, 2], I[2, 0], I[2, 1]],
        0.0,
        atol=0.0,
    )


def test_solid_sphere_inertia_nonpositive_returns_zero():
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import solid_sphere_inertia_diag

    I = solid_sphere_inertia_diag(0.0, 0.04)
    assert float(I[0, 0]) == 0.0
    I2 = solid_sphere_inertia_diag(0.1, 0.0)
    assert float(I2[0, 0]) == 0.0


def test_apple_com_in_tcp_frame_matches_inv_offset():
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import apple_com_in_tcp_frame

    # Translation-only offset: apple COM is -t in TCP frame when offset quat = identity.
    offset = (0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0)
    com = apple_com_in_tcp_frame(offset)
    np.testing.assert_allclose(com, [0.0, 0.0, -0.05], rtol=1e-6, atol=1e-6)


def test_payload_props_from_params():
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import payload_props_from_params

    params = FruitingSystemParams(
        primary=None,
        secondary=None,
        spur=None,
        stem=None,
        apple_radius=0.04,
        apple_density=800.0,
    )
    offset = (0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 1.0)
    m, I, com = payload_props_from_params(params, offset)
    assert m == pytest.approx(analytic_apple_mass_kg(params))
    expected_I = 0.4 * m * (0.04**2)
    assert float(I[0, 0]) == pytest.approx(expected_I)
    np.testing.assert_allclose(com, [0.0, 0.0, -0.04], rtol=1e-6, atol=1e-6)


@requires_fr3
def test_welded_fr3_default_payload_mass_zero_with_harvest_inertia():
    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        0,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        robot_base_from_proxy=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=128,
    )
    assert scene.mj_apple_payload_body_index is not None
    payload = int(scene.mj_apple_payload_body_index)
    m_mj = float(scene.robot_model.body_mass.numpy()[payload])
    assert m_mj == pytest.approx(0.0, abs=1e-12)
    assert scene.stem_harvest_explicit_apple_weight is True
    assert scene.stem_harvest_explicit_apple_inertia is True
    assert scene.apple_inertia_kgm2 > 0.0
    g = scene.robot_model.gravity.numpy()
    np.testing.assert_allclose(g, 0.0, atol=0.0)
    from apple_pick_sim.robot.fr3_robot.setup import resolve_tcp_body_index

    assert resolve_tcp_body_index(scene.robot_model) == scene.tcp_body_index
    assert payload != scene.tcp_body_index


@requires_fr3
def test_welded_fr3_payload_matching_avbd_when_inertia_off():
    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        0,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        robot_base_from_proxy=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=128,
        stem_harvest_explicit_apple_inertia=False,
    )
    assert scene.mj_apple_payload_body_index is not None
    payload = int(scene.mj_apple_payload_body_index)
    apple = int(scene.cable.apple_body)
    m_avbd = float(scene.cable.model.body_mass.numpy()[apple])
    m_mj = float(scene.robot_model.body_mass.numpy()[payload])
    assert m_mj == pytest.approx(m_avbd, rel=1e-5, abs=1e-8)
    r = float(scene.cable.params.apple_radius)
    I = scene.robot_model.body_inertia.numpy()[payload]
    assert float(I[0, 0]) == pytest.approx(0.4 * m_avbd * r * r, rel=1e-5, abs=1e-10)
    com = scene.robot_model.body_com.numpy()[payload]
    offset = scene.cable.gripper_proxy_offset_in_apple_frame
    assert offset is not None
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import apple_com_in_tcp_frame

    np.testing.assert_allclose(com, apple_com_in_tcp_frame(offset), rtol=1e-5, atol=1e-6)
    # COM sits one apple radius from the TCP (surface welded at TCP, center r behind),
    # matching the robot-facing weld placement (proxy at apple surface).
    assert float(np.linalg.norm(com)) == pytest.approx(r, rel=1e-4, abs=1e-6)
    assert scene.stem_harvest_explicit_apple_weight is True
    g = scene.robot_model.gravity.numpy()
    np.testing.assert_allclose(g, 0.0, atol=0.0)
    # TCP resolve still finds tcp, not payload
    from apple_pick_sim.robot.fr3_robot.setup import resolve_tcp_body_index

    assert resolve_tcp_body_index(scene.robot_model) == scene.tcp_body_index
    assert payload != scene.tcp_body_index


@requires_fr3
def test_free_proxy_fr3_has_no_payload_body():
    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        0,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=False),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        robot_base_from_proxy=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=128,
    )
    assert scene.mj_apple_payload_body_index is None
    labels = list(scene.robot_model.body_label)
    assert all("apple_payload" not in str(lbl) for lbl in labels)


@requires_fr3
def test_hetero_welded_payload_masses_zero_with_harvest_inertia():
    from apple_pick_sim.coupled_fruiting.builders import build_heterogeneous_coupled_fruiting_fr3

    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p0 = fs.sample_params(ranges, seed=0)
    p1 = dataclasses.replace(
        p0,
        apple_radius=float(p0.apple_radius) * 1.35,
        apple_density=float(p0.apple_density) * 0.7,
    )
    assert analytic_apple_mass_kg(p0) != pytest.approx(analytic_apple_mass_kg(p1))

    scene = build_heterogeneous_coupled_fruiting_fr3(
        ranges,
        [p0, p1],
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=64,
        skip_ik_bootstrap=True,
        defer_template_robot_bootstrap=True,
    )
    layout = scene.layout
    assert layout is not None
    assert layout.template_mj_apple_payload_body is not None
    masses = scene.robot_model.body_mass.numpy()
    for idx in layout.mj_apple_payload_body_indices:
        assert float(masses[int(idx)]) == pytest.approx(0.0, abs=1e-12)
    assert scene.stem_harvest_explicit_apple_inertia is True


@requires_fr3
def test_hetero_welded_payload_masses_follow_per_env_apples_when_inertia_off():
    from apple_pick_sim.coupled_fruiting.builders import build_heterogeneous_coupled_fruiting_fr3

    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    p0 = fs.sample_params(ranges, seed=0)
    p1 = dataclasses.replace(
        p0,
        apple_radius=float(p0.apple_radius) * 1.35,
        apple_density=float(p0.apple_density) * 0.7,
    )
    assert analytic_apple_mass_kg(p0) != pytest.approx(analytic_apple_mass_kg(p1))

    scene = build_heterogeneous_coupled_fruiting_fr3(
        ranges,
        [p0, p1],
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=64,
        skip_ik_bootstrap=True,
        defer_template_robot_bootstrap=True,
        stem_harvest_explicit_apple_inertia=False,
    )
    layout = scene.layout
    assert layout is not None
    assert layout.template_mj_apple_payload_body is not None
    masses = scene.robot_model.body_mass.numpy()
    coms = scene.robot_model.body_com.numpy()
    m0 = float(masses[layout.mj_apple_payload_body_indices[0]])
    m1 = float(masses[layout.mj_apple_payload_body_indices[1]])
    assert m0 == pytest.approx(float(scene.cable.model.body_mass.numpy()[layout.apple_body_indices[0]]))
    assert m1 == pytest.approx(float(scene.cable.model.body_mass.numpy()[layout.apple_body_indices[1]]))
    assert m0 != pytest.approx(m1)
    c0 = float(np.linalg.norm(coms[layout.mj_apple_payload_body_indices[0]]))
    c1 = float(np.linalg.norm(coms[layout.mj_apple_payload_body_indices[1]]))
    assert c0 != pytest.approx(c1, abs=1e-6) or float(p0.apple_radius) != float(p1.apple_radius)


def _payload_inertia_diag(scene) -> np.ndarray:
    idx = int(scene.mj_apple_payload_body_index)
    I = scene.robot_model.body_inertia.numpy()[idx]
    return np.array([float(I[0, 0]), float(I[1, 1]), float(I[2, 2])], dtype=np.float64)


def _expected_sphere_I(scene) -> float:
    apple = int(scene.cable.apple_body)
    m = float(scene.cable.model.body_mass.numpy()[apple])
    r = float(scene.cable.params.apple_radius)
    return 0.4 * m * r * r


def _snapshot_robot(scene) -> dict:
    return {
        "joint_q": scene.robot_state_0.joint_q.numpy().copy(),
        "joint_qd": scene.robot_state_0.joint_qd.numpy().copy(),
    }


def _restore_robot(scene, snap: dict) -> None:
    from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
    import newton

    scene.robot_state_0.joint_q.assign(snap["joint_q"])
    scene.robot_state_0.joint_qd.assign(snap["joint_qd"])
    scene.robot_model.joint_q.assign(snap["joint_q"])
    scene.robot_model.joint_qd.assign(snap["joint_qd"])
    newton.eval_fk(
        scene.robot_model,
        scene.robot_state_0.joint_q,
        scene.robot_state_0.joint_qd,
        scene.robot_state_0,
    )
    init_robot_mujoco_step_buffers(scene)


def _configure_vic_soft(scene, *, linear_k: float = 600.0) -> Fr3EEImpedanceController:
    import warp as wp

    fr3_robot.scale_mujoco_joint_pd(scene.robot_model, 0.02)
    fr3_robot.configure_vic_wrench_only_arm(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
    )
    ctrl = Fr3EEImpedanceController(tcp_body_index=int(scene.tcp_body_index))
    ctrl.sync_target_from_state(scene.robot_state_0)
    # Bias target so VIC applies a sustained wrench.
    pos = wp.transform_get_translation(ctrl.target_tf)
    ctrl.target_tf = wp.transform(
        wp.vec3(float(pos[0]) + 0.08, float(pos[1]), float(pos[2])),
        wp.transform_get_rotation(ctrl.target_tf),
    )
    scene.vic_controller = ctrl
    scene.vic_gains = ImpedanceGains(linear_k=linear_k, linear_d=60.0)
    scene.vic_target_tf = ctrl.target_tf
    scene.vic_target_twist = fr3_robot.EEVelocity()
    scene.robot_kinematic_mode = False
    scene.proxy_forces.zero_()
    return ctrl


def _tcp_x(scene) -> float:
    tcp = int(scene.tcp_body_index)
    return float(scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, 0])


def _run_vic_mujoco_steps(scene, n: int = 120) -> float:
    x0 = _tcp_x(scene)
    for _ in range(n):
        scene.mujoco_substep(SUB_DT)
    return _tcp_x(scene) - x0


@requires_fr3
def test_clear_payload_zeros_inertia_vs_analytic_sphere():
    """Explicit inertia check: applied AVBD sphere I vs cleared zeros."""
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import (
        apply_mujoco_apple_payload_inertias,
        clear_mujoco_apple_payload_inertias,
    )

    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        7,
        mujoco_only=True,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        robot_base_from_proxy=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=128,
        stem_harvest_explicit_apple_inertia=False,
    )
    assert scene.mj_apple_payload_body_index is not None
    expected = _expected_sphere_I(scene)
    assert expected > 1e-8

    I_on = _payload_inertia_diag(scene)
    np.testing.assert_allclose(I_on, expected, rtol=1e-4, atol=1e-10)
    m_on = float(scene.robot_model.body_mass.numpy()[int(scene.mj_apple_payload_body_index)])
    assert m_on == pytest.approx(
        float(scene.cable.model.body_mass.numpy()[int(scene.cable.apple_body)]),
        rel=1e-5,
        abs=1e-8,
    )

    clear_mujoco_apple_payload_inertias(scene)
    I_off = _payload_inertia_diag(scene)
    np.testing.assert_allclose(I_off, 0.0, atol=1e-12)
    m_off = float(scene.robot_model.body_mass.numpy()[int(scene.mj_apple_payload_body_index)])
    assert m_off == pytest.approx(0.0, abs=1e-12)

    apply_mujoco_apple_payload_inertias(scene)
    I_restored = _payload_inertia_diag(scene)
    np.testing.assert_allclose(I_restored, expected, rtol=1e-4, atol=1e-10)


@requires_fr3
def test_vic_tcp_motion_differs_with_and_without_payload_inertia():
    """Same VIC setpoint: cleared payload (no inertia) moves TCP more than with inertia."""
    from apple_pick_sim.coupled_fruiting.mujoco_apple_payload import (
        apply_mujoco_apple_payload_inertias,
        clear_mujoco_apple_payload_inertias,
        solid_sphere_inertia_diag,
        _set_body_inertial_props,
        apple_com_in_tcp_frame,
    )
    import newton
    import warp as wp

    cf = _import_cf()
    fs = _import_fs()
    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        7,
        mujoco_only=True,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True, robot_facing_weld=True),
        enable_self_collisions=False,
        base_pos=COUPLED_BASE_POS,
        robot_base_pos=COUPLED_ROBOT_BASE_POS,
        robot_base_from_proxy=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        ik_bootstrap_iterations=128,
        stem_harvest_explicit_apple_inertia=False,
    )
    assert scene.mj_apple_payload_body_index is not None
    payload = int(scene.mj_apple_payload_body_index)
    # Explicit AVBD match first.
    expected_I = _expected_sphere_I(scene)
    assert expected_I > 1e-8
    np.testing.assert_allclose(_payload_inertia_diag(scene), expected_I, rtol=1e-4, atol=1e-10)

    # Amplify payload so reflected inertia is clearly visible under short VIC (keeps
    # COM/lever from grasp offset; same body topology for A/B).
    m_dyn = 2.0
    r_dyn = 0.08
    I_dyn = solid_sphere_inertia_diag(m_dyn, r_dyn)
    offset = scene.cable.gripper_proxy_offset_in_apple_frame
    assert offset is not None
    _set_body_inertial_props(
        scene.robot_model,
        payload,
        mass_kg=m_dyn,
        inertia=I_dyn,
        com_in_body=apple_com_in_tcp_frame(offset),
    )
    scene.mj_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    I_with = _payload_inertia_diag(scene)
    np.testing.assert_allclose(I_with, 0.4 * m_dyn * r_dyn * r_dyn, rtol=1e-4, atol=1e-10)

    snap = _snapshot_robot(scene)
    _configure_vic_soft(scene, linear_k=800.0)
    tgt = scene.vic_controller.target_tf
    gains = scene.vic_gains

    dx_with = _run_vic_mujoco_steps(scene, n=150)

    clear_mujoco_apple_payload_inertias(scene)
    I_without = _payload_inertia_diag(scene)
    np.testing.assert_allclose(I_without, 0.0, atol=1e-12)
    assert float(np.max(I_with)) > float(np.max(I_without)) + 1e-6

    _restore_robot(scene, snap)
    fr3_robot.scale_mujoco_joint_pd(scene.robot_model, 0.02)
    fr3_robot.configure_vic_wrench_only_arm(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
    )
    ctrl = Fr3EEImpedanceController(tcp_body_index=int(scene.tcp_body_index))
    ctrl.target_tf = tgt
    scene.vic_controller = ctrl
    scene.vic_gains = gains
    scene.vic_target_tf = tgt
    scene.vic_target_twist = fr3_robot.EEVelocity()
    scene.robot_kinematic_mode = False
    scene.proxy_forces.zero_()

    dx_without = _run_vic_mujoco_steps(scene, n=150)

    assert abs(dx_without) > abs(dx_with), (
        f"expected |dx_without| > |dx_with|; "
        f"dx_with={dx_with:.6e} dx_without={dx_without:.6e} "
        f"I_with={float(np.max(I_with)):.6e} I_without={float(np.max(I_without)):.6e}"
    )
    assert abs(dx_without - dx_with) > 2e-3, (
        f"amplified payload inertia should change TCP advance by >2 mm; "
        f"dx_with={dx_with:.6e} dx_without={dx_without:.6e}"
    )
    assert abs(dx_with) > 1e-4
    assert abs(dx_without) > 1e-4

    apply_mujoco_apple_payload_inertias(scene)
