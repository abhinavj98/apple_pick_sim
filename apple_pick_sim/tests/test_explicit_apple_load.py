"""Tests for explicit apple weight in stem-harvest TCP wrench transfer."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.explicit_load import (
    apple_com_from_tcp_grasp_offset,
    apple_explicit_wrench_about_tcp,
    apple_mass_kg_from_model,
    apple_support_force_world,
    body_com_position_world,
    explicit_apple_wrench_for_stem_harvest,
)
from apple_pick_sim.coupled_fruiting import proxy_coupling as pc
from apple_pick_sim.tests.test_coupled_fruiting_system import (
    SUB_DT,
    _build_welded_coupled_for_stem_tests,
    _explicit_apple_wrench_for_scene,
    _import_cf,
    _import_fs,
    _run_coupled_hold_frames,
)
from conftest import (
    COUPLED_BASE_POS,
    COUPLED_ROBOT_BASE_POS,
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    build_coupled_fr3,
    requires_fr3,
    run_coupled_substeps_direct_hold,
)

_SETTLE_WELD_BUILD_KW = dict(
    enable_self_collisions=False,
    base_pos=COUPLED_BASE_POS,
    robot_base_pos=COUPLED_ROBOT_BASE_POS,
    robot_base_from_proxy=False,
    mujoco_solver_kwargs=DEFAULT_MJ_KW,
    ik_bootstrap_iterations=256,
)


_GRAVITY = (0.0, 0.0, -9.81)


def test_apple_support_force_world_matches_apple_weight():
    m = 0.2
    f = apple_support_force_world(m, _GRAVITY)
    np.testing.assert_allclose(f, [0.0, 0.0, -m * 9.81], rtol=1e-12, atol=1e-12)


def test_apple_support_force_zero_mass_returns_zero():
    f = apple_support_force_world(0.0, _GRAVITY)
    np.testing.assert_allclose(f, 0.0, atol=0.0)
    f_neg = apple_support_force_world(-0.01, _GRAVITY)
    np.testing.assert_allclose(f_neg, 0.0, atol=0.0)


def test_apple_mass_kg_from_model_missing_index_returns_zero():
    assert apple_mass_kg_from_model(None, None) == 0.0
    assert apple_mass_kg_from_model(None, -1) == 0.0


def test_apple_explicit_torque_from_com_offset():
    m = 0.15
    p_tcp = np.array([0.0, 0.0, 1.0])
    p_apple = np.array([0.1, 0.0, 1.0])
    f, tau = apple_explicit_wrench_about_tcp(m, _GRAVITY, p_tcp, p_apple)
    f_exp = apple_support_force_world(m, _GRAVITY)
    np.testing.assert_allclose(f, f_exp, rtol=1e-12, atol=1e-12)
    tau_exp = np.cross(p_apple - p_tcp, f_exp)
    np.testing.assert_allclose(tau, tau_exp, rtol=1e-12, atol=1e-12)
    assert abs(float(tau[1])) > 0.01


def test_grasp_offset_places_apple_com_behind_tcp():
    p_tcp = np.array([1.0, 2.0, 3.0])
    offset = (0.0, 0.0, 0.12)
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5)
    p_apple = apple_com_from_tcp_grasp_offset(p_tcp, rot, offset)
    np.testing.assert_allclose(
        p_apple,
        p_tcp - np.asarray(wp.quat_rotate(rot, wp.vec3(*offset)), dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_grasp_offset_wrench_matches_body_q_when_fix_to_apple():
    """Kinematic offset path agrees with cable ``body_q`` after welded sync."""
    cf = _import_cf()
    fs = _import_fs()
    from apple_pick_sim.robot import fr3_robot

    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=52,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    run_coupled_substeps_direct_hold(scene, fr3_robot, 12, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)

    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    off = cable.gripper_proxy_offset_in_apple_frame
    assert off is not None
    m = apple_mass_kg_from_model(cable.model, apple)

    w_offset = explicit_apple_wrench_for_stem_harvest(
        mass_kg=m,
        gravity=_GRAVITY,
        robot_body_q=scene.robot_state_0.body_q,
        cable_body_q=cable.state_0.body_q,
        tcp_body_index=scene.tcp_body_index,
        apple_body_index=apple,
        grasp_offset_in_apple_frame=off,
    )
    p_tcp = body_com_position_world(scene.robot_state_0.body_q, scene.tcp_body_index)
    p_apple = body_com_position_world(cable.state_0.body_q, apple)
    w_body_q = apple_explicit_wrench_about_tcp(m, _GRAVITY, p_tcp, p_apple)
    np.testing.assert_allclose(w_offset[0], w_body_q[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(w_offset[1], w_body_q[1], rtol=0.02, atol=0.02)


def test_grasp_offset_torque_nonzero_when_tcp_tilted():
    m = 0.2
    p_tcp = np.array([0.0, 0.0, 1.0])
    offset = (0.08, 0.0, 0.0)
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.6)
    _, tau = apple_explicit_wrench_about_tcp(
        m,
        _GRAVITY,
        p_tcp,
        grasp_offset_in_apple_frame=offset,
        tcp_orientation_world=rot,
    )
    assert float(np.linalg.norm(tau)) > 0.05


def test_stem_harvest_explicit_adds_force_and_torque():
    cf = _import_cf()
    fs = _import_fs()
    from apple_pick_sim.robot import fr3_robot

    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=51,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    run_coupled_substeps_direct_hold(scene, fr3_robot, 20, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)

    cable = scene.cable
    apple = cable.apple_body
    assert apple is not None
    m = apple_mass_kg_from_model(cable.model, apple)
    assert m > 0.01
    assert abs(scene.apple_mass_kg - m) < 1e-9
    tcp = scene.tcp_body_index

    bq_cable = cable.state_0.body_q.numpy().reshape(-1, 7)
    bq_robot = scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    off = cable.gripper_proxy_offset_in_apple_frame
    f_exp, tau_exp = explicit_apple_wrench_for_stem_harvest(
        mass_kg=m,
        gravity=_GRAVITY,
        robot_body_q=scene.robot_state_0.body_q,
        cable_body_q=cable.state_0.body_q,
        tcp_body_index=tcp,
        apple_body_index=apple,
        grasp_offset_in_apple_frame=off,
    )

    dev = str(cable.model.device)
    out_off = wp.zeros(scene.robot_model.body_count, dtype=wp.spatial_vector, device=dev)
    out_on = wp.zeros_like(out_off)
    kw = dict(
        cable_model=cable.model,
        cable_solver=cable.solver,
        body_q_post=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        robot_body_q=scene.robot_state_0.body_q,
        dt=SUB_DT,
        stem_apple_joint_index=scene.stem_apple_joint_index,
        tcp_body_index=tcp,
        coupling_gain=1.0,
        force_cap_N=None,
        torque_cap_Nm=None,
        apple_body_index=apple,
        gravity=wp.vec3(*_GRAVITY),
        grasp_offset_in_apple_frame=off,
    )
    kw["apple_mass_kg"] = scene.apple_mass_kg
    pc.harvest_stem_tension_for_tcp(**kw, out_robot_wrenches=out_off, explicit_apple_weight=False)
    pc.harvest_stem_tension_for_tcp(**kw, out_robot_wrenches=out_on, explicit_apple_weight=True)

    w_off = out_off.numpy().reshape(-1, 6)[tcp]
    w_on = out_on.numpy().reshape(-1, 6)[tcp]
    np.testing.assert_allclose(w_on[:3] - w_off[:3], f_exp, rtol=0.02, atol=0.02)
    np.testing.assert_allclose(w_on[3:] - w_off[3:], tau_exp, rtol=0.02, atol=0.05)


@pytest.mark.slow
def test_coupled_substep_default_includes_explicit_apple_weight():
    cf = _import_cf()
    fs = _import_fs()
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.fruiting_system.params import analytic_apple_mass_kg
    from apple_pick_sim.robot import fr3_robot

    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=17,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    assert scene.stem_harvest_explicit_apple_weight is True
    run_coupled_substeps_direct_hold(scene, fr3_robot, 90, sub_dt=SUB_DT)
    scene.coupled_substep(SUB_DT)

    m = analytic_apple_mass_kg(scene.cable.params)
    assert m is not None
    expected_mg = float(m) * 9.81
    tcp_w = read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index)
    assert float(np.linalg.norm(tcp_w[:3])) >= 0.5 * expected_mg


@requires_fr3
def test_build_coupled_fr3_caches_apple_mass_kg_at_build():
    """``apple_mass_kg`` is filled at build so graph capture never reads ``body_mass`` per step."""
    cf = _import_cf()
    fs = _import_fs()

    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        54,
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    apple = scene.cable.apple_body
    assert apple is not None
    m_model = apple_mass_kg_from_model(scene.cable.model, apple)
    assert m_model > 0.01
    assert abs(scene.apple_mass_kg - m_model) < 1e-9


@requires_fr3
def test_harvest_zero_cached_mass_skips_explicit_support():
    """GPU stem harvest: ``apple_mass_kg=0`` must not add support even when explicit flag is on."""
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()

    scene = build_coupled_fr3(
        cf,
        fs.load_ranges(RANGES_FIXTURE),
        55,
        device="cpu",
        gripper_proxy=fs.GripperProxyConfig(fix_to_apple=True),
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    run_coupled_substeps_direct_hold(scene, fr3_robot, 6, sub_dt=SUB_DT)

    cable = scene.cable
    tcp = scene.tcp_body_index
    dev = str(cable.model.device)
    out_on = wp.zeros(scene.robot_model.body_count, dtype=wp.spatial_vector, device=dev)
    out_off = wp.zeros_like(out_on)
    kw = dict(
        cable_model=cable.model,
        cable_solver=cable.solver,
        body_q_post=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        robot_body_q=scene.robot_state_0.body_q,
        dt=SUB_DT,
        stem_apple_joint_index=scene.stem_apple_joint_index,
        tcp_body_index=tcp,
        coupling_gain=1.0,
        force_cap_N=None,
        torque_cap_Nm=None,
        apple_body_index=cable.apple_body,
        gravity=wp.vec3(*_GRAVITY),
        grasp_offset_in_apple_frame=cable.gripper_proxy_offset_in_apple_frame,
        apple_mass_kg=0.0,
    )
    pc.harvest_stem_tension_for_tcp(**kw, out_robot_wrenches=out_off, explicit_apple_weight=False)
    pc.harvest_stem_tension_for_tcp(**kw, out_robot_wrenches=out_on, explicit_apple_weight=True)
    np.testing.assert_allclose(
        out_on.numpy().reshape(-1, 6)[tcp],
        out_off.numpy().reshape(-1, 6)[tcp],
        rtol=1e-6,
        atol=1e-6,
    )


@requires_fr3
def test_coupled_substep_explicit_flag_delta_matches_explicit_wrench():
    """``coupled_substep`` passes ``stem_harvest_explicit_apple_weight`` into stem harvest."""
    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.robot import fr3_robot

    cf = _import_cf()
    fs = _import_fs()
    scene = _build_welded_coupled_for_stem_tests(
        cf,
        fs,
        seed=53,
        stem_coupling_gain=1.0,
        stem_force_cap_N=None,
        stem_torque_cap_Nm=None,
    )
    _run_coupled_hold_frames(scene, fr3_robot, 80)

    scene.stem_harvest_explicit_apple_weight = True
    scene.coupled_substep(SUB_DT)
    w_on = read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index).astype(np.float64)

    scene.stem_harvest_explicit_apple_weight = False
    scene.coupled_substep(SUB_DT)
    w_off = read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index).astype(np.float64)

    f_exp, _ = _explicit_apple_wrench_for_scene(scene)
    delta_f = w_on[:3] - w_off[:3]
    expected_mg = float(np.linalg.norm(f_exp))
    assert expected_mg > 0.5
    assert float(delta_f[2]) < -0.4 * expected_mg, (
        f"explicit-on should add env-on-robot m·g payload on TCP: "
        f"ΔFz={delta_f[2]:.2f}, m·g≈{expected_mg:.2f}"
    )
    np.testing.assert_allclose(
        float(np.linalg.norm(delta_f)),
        expected_mg,
        rtol=0.35,
        atol=1.5,
    )


@requires_fr3
@pytest.mark.slow
def test_settle_weld_hold_explicit_support_matches_mg():
    """Quiet settle→weld + hold: stem-harvest explicit term adds ≈ env-on-robot ``m·g`` on TCP."""
    import apple_pick_sim.coupled_fruiting as cf
    from apple_pick_sim.coupled_fruiting.builders import build_coupled_fruiting_fr3

    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

    fs = _import_fs()
    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = None
    last_exc: Exception | None = None
    for try_seed in (2, 3, 4, 5):
        try:
            settled = build_coupled_fruiting_fr3(
                ranges,
                try_seed,
                vbd_only=True,
                **_SETTLE_WELD_BUILD_KW,
                gripper_proxy=fs.GripperProxyConfig(
                    mass=fr3_robot.EE_MASS_KG,
                    fix_to_apple=False,
                ),
            )
            cf.settle_vbd_substeps(settled, substeps=40, dt=SUB_DT)
            scene = build_coupled_fruiting_fr3(
                ranges,
                try_seed,
                skip_ik_bootstrap=True,
                **_SETTLE_WELD_BUILD_KW,
                gripper_proxy=fs.GripperProxyConfig(
                    mass=fr3_robot.EE_MASS_KG,
                    fix_to_apple=True,
                ),
            )
            cf.seed_fix_to_apple_from_settled(
                welded_scene=scene, settled_scene=settled, quiet_apple_proxy=True
            )
            break
        except IKBootstrapConvergenceError as exc:
            last_exc = exc
            scene = None
    if scene is None:
        raise last_exc  # type: ignore[misc]

    run_coupled_substeps_direct_hold(scene, fr3_robot, 60, sub_dt=SUB_DT)

    cable = scene.cable
    tcp = scene.tcp_body_index
    dev = str(cable.model.device)
    out_on = wp.zeros(scene.robot_model.body_count, dtype=wp.spatial_vector, device=dev)
    out_off = wp.zeros_like(out_on)
    kw = dict(
        cable_model=cable.model,
        cable_solver=cable.solver,
        body_q_post=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        robot_body_q=scene.robot_state_0.body_q,
        dt=SUB_DT,
        stem_apple_joint_index=scene.stem_apple_joint_index,
        tcp_body_index=tcp,
        coupling_gain=1.0,
        force_cap_N=None,
        torque_cap_Nm=None,
        apple_body_index=cable.apple_body,
        apple_mass_kg=scene.apple_mass_kg,
        gravity=wp.vec3(*_GRAVITY),
        grasp_offset_in_apple_frame=cable.gripper_proxy_offset_in_apple_frame,
    )
    pc.harvest_stem_tension_for_tcp(**kw, out_robot_wrenches=out_off, explicit_apple_weight=False)
    pc.harvest_stem_tension_for_tcp(**kw, out_robot_wrenches=out_on, explicit_apple_weight=True)
    delta = out_on.numpy().reshape(-1, 6)[tcp, :3] - out_off.numpy().reshape(-1, 6)[tcp, :3]
    f_exp, _ = _explicit_apple_wrench_for_scene(scene)
    np.testing.assert_allclose(delta, f_exp, rtol=0.05, atol=0.15)
    assert float(delta[2]) < 0.5 * float(f_exp[2])
