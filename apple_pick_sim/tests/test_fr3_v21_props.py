"""Tests for official FR3 v2.1 inertial/dynamics YAML loading and application."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.robot.fr3_robot.fr3_v21_props import (
    load_fr3_v21_dynamics,
    load_fr3_v21_inertials,
    parse_fr3_v21_dynamics,
    parse_fr3_v21_inertials,
    resolve_fr3_link_body_index,
)
from apple_pick_sim.robot.fr3_robot.paths import EE_COM_IN_EE_LOCAL_M, EE_MASS_KG
from apple_pick_sim.robot.fr3_robot.setup import (
    FR3_DEFAULT_VIC_JOINT_DAMPING,
    FR3_REFLECTED_MOTOR_INERTIA_KGM2,
)
from apple_pick_sim.tests.conftest import (
    DEFAULT_MJ_KW,
    RANGES_FIXTURE,
    build_coupled_fr3,
    fr3_assets_available,
)

requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


def test_fr3_v21_yaml_parse_armature():
    dyn = parse_fr3_v21_dynamics()
    expected = (0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055)
    np.testing.assert_allclose(dyn.reflected_motor_inertia_kgm2, expected, rtol=0.0, atol=1e-4)
    assert dyn.mu_viscous == 16.0
    assert FR3_REFLECTED_MOTOR_INERTIA_KGM2 == dyn.reflected_motor_inertia_kgm2
    assert FR3_DEFAULT_VIC_JOINT_DAMPING == 16.0


def test_fr3_v21_yaml_parse_link1_mass():
    links = parse_fr3_v21_inertials()
    link1 = next(link for link in links if link.link_num == 1)
    assert link1.mass_kg == pytest.approx(2.4377, abs=1e-4)


def _import_cf():
    import apple_pick_sim.coupled_fruiting as cf

    return cf


def _build_mujoco_only_fr3():
    cf = _import_cf()
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot

    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf,
        ranges,
        11,
        mujoco_only=True,
        enable_self_collisions=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
    )
    fr3_robot.init_mujoco_actuator_targets_from_model(scene.robot_model, scene.robot_control)
    return scene


def _configure_joint_torque_vic(scene):
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
        Fr3EEImpedanceController,
        ImpedanceGains,
    )

    scene.vic_use_joint_torques = True
    ctrl = Fr3EEImpedanceController(tcp_body_index=int(scene.tcp_body_index))
    scene.vic_controller = ctrl
    scene.vic_gains = ImpedanceGains()
    fr3_robot.configure_vic_joint_torques_arm(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        scene=scene,
    )
    ctrl.sync_target_from_state(scene.robot_state_0)
    return ctrl


def _mj_body_for_newton(mj_solver, newton_body: int) -> int:
    mjc_map = mj_solver.mjc_body_to_newton.numpy().reshape(-1)
    hits = np.where(mjc_map == int(newton_body))[0]
    assert len(hits) == 1, f"expected one MuJoCo body for Newton index {newton_body}"
    return int(hits[0])


@requires_fr3
def test_configure_vic_applies_fr3_v21_link_inertials():
    scene = _build_mujoco_only_fr3()
    _configure_joint_torque_vic(scene)
    model = scene.robot_model
    mj_solver = scene.mj_solver
    links = load_fr3_v21_inertials()

    for link in links:
        idx = resolve_fr3_link_body_index(model, link.link_num)
        mass = float(model.body_mass.numpy()[idx])
        com = model.body_com.numpy()[idx]
        inertia = model.body_inertia.numpy()[idx]
        np.testing.assert_allclose(mass, link.mass_kg, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(com, link.com_m, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(inertia, link.inertia_kgm2, rtol=0.0, atol=1e-5)

        mjc_body = _mj_body_for_newton(mj_solver, idx)
        if mj_solver.use_mujoco_cpu:
            mj_mass = float(mj_solver.mj_model.body_mass[mjc_body])
            mj_inertia = np.asarray(mj_solver.mj_model.body_inertia[mjc_body], dtype=np.float64)
        else:
            mj_mass = float(mj_solver.mjw_model.body_mass.numpy()[0, mjc_body])
            mj_inertia = mj_solver.mjw_model.body_inertia.numpy()[0, mjc_body].astype(np.float64)
        np.testing.assert_allclose(mj_mass, link.mass_kg, rtol=0.0, atol=1e-5)
        yaml_evals = np.sort(np.linalg.eigvalsh(link.inertia_kgm2))
        mj_evals = np.sort(mj_inertia.reshape(-1)[:3])
        np.testing.assert_allclose(mj_evals, yaml_evals, rtol=0.0, atol=1e-4)


@requires_fr3
def test_configure_vic_ee_tcp_inertials_unchanged():
    from apple_pick_sim.robot import fr3_robot

    scene = _build_mujoco_only_fr3()
    model = scene.robot_model
    ee_idx = fr3_robot.resolve_ee_body_index(model)
    tcp_idx = int(scene.tcp_body_index)
    ee_mass_before = float(model.body_mass.numpy()[ee_idx])
    tcp_mass_before = float(model.body_mass.numpy()[tcp_idx])
    ee_com_before = model.body_com.numpy()[ee_idx].copy()

    _configure_joint_torque_vic(scene)

    ee_mass_after = float(model.body_mass.numpy()[ee_idx])
    tcp_mass_after = float(model.body_mass.numpy()[tcp_idx])
    ee_com_after = model.body_com.numpy()[ee_idx]

    assert ee_mass_before == pytest.approx(EE_MASS_KG, abs=1e-4)
    assert ee_mass_after == pytest.approx(EE_MASS_KG, abs=1e-4)
    assert tcp_mass_after == pytest.approx(tcp_mass_before, abs=1e-4)
    assert tcp_mass_after == pytest.approx(0.001, abs=1e-3)
    np.testing.assert_allclose(ee_com_after, ee_com_before, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(ee_com_after, EE_COM_IN_EE_LOCAL_M, rtol=0.0, atol=1e-4)
