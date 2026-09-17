"""EE ``body_parent_f`` → world-frame env-on-robot wrench at TCP."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_env_on_robot_tcp_wrench_negates_and_transports_lever_arm():
    """Parent-on-child at EE COM → env-on-robot about TCP (force + lever)."""
    from apple_pick_sim.robot.fr3_robot.tcp_parent_wrench import (
        env_on_robot_tcp_wrench_from_ee_parent_f,
    )

    # EE COM 0.077 m toward base of flange from TCP along world −Z when identity.
    p_ee_com = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    p_tcp = np.array([0.0, 0.0, -0.077], dtype=np.float64)
    parent_f = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    wrench = env_on_robot_tcp_wrench_from_ee_parent_f(
        parent_f, p_ee_com_world=p_ee_com, p_tcp_world=p_tcp
    )

    # Env-on-robot force is −parent-on-child.
    np.testing.assert_allclose(wrench[:3], [-1.0, 0.0, 0.0], atol=1e-9)
    # τ_tcp = −(τ_ee + (p_ee − p_tcp) × F_ee)
    # (0,0,0.077) × (1,0,0) = (0, 0.077, 0) → negate → (0, −0.077, 0)
    np.testing.assert_allclose(wrench[3:], [0.0, -0.077, 0.0], atol=1e-9)


def test_env_on_robot_tcp_wrench_preserves_local_torque_sign():
    from apple_pick_sim.robot.fr3_robot.tcp_parent_wrench import (
        env_on_robot_tcp_wrench_from_ee_parent_f,
    )

    p = np.zeros(3, dtype=np.float64)
    parent_f = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float64)
    wrench = env_on_robot_tcp_wrench_from_ee_parent_f(
        parent_f, p_ee_com_world=p, p_tcp_world=p
    )
    np.testing.assert_allclose(wrench, [0.0, 0.0, 0.0, 0.0, 0.0, -2.0], atol=1e-9)


def test_ee_com_world_rotates_local_com():
    from apple_pick_sim.robot.fr3_robot.tcp_parent_wrench import ee_com_world_from_body_q

    # body_q: origin at (1,2,3), identity quat (xyzw / Warp convention)
    body_q = np.zeros((2, 7), dtype=np.float64)
    body_q[1] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    body_com = np.zeros((2, 3), dtype=np.float64)
    body_com[1] = [0.0, 0.0, -0.077]

    got = ee_com_world_from_body_q(body_q, body_com, body_index=1)
    np.testing.assert_allclose(got, [1.0, 2.0, 3.0 - 0.077], atol=1e-9)


@requires_fr3
def test_mujoco_only_static_ee_parent_f_matches_weight_with_robot_gravity():
    """Env-on-robot TCP force is −parent_f; gravity increases downward load.

    Default FR3 pose is not a clean hanging pendulum, so we check force negation
    exactly and that enabling Model A gravity shifts env-on-robot F_z downward by
    a substantial fraction of ``m_ee * g``.
    """
    import newton

    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.tcp_parent_wrench import (
        read_ee_parent_f_world,
        tcp_world_wrench_from_scene,
    )
    from apple_pick_sim.tests.test_vic_joint_torques import _configure_joint_torque_vic

    ranges = fs.load_ranges(RANGES_FIXTURE)
    scene = build_coupled_fr3(
        cf,
        ranges,
        11,
        mujoco_only=True,
        enable_self_collisions=False,
        mujoco_solver_kwargs=DEFAULT_MJ_KW,
        gripper_proxy=GripperProxyConfig(fix_to_apple=False),
        skip_ik_bootstrap=True,
        request_body_parent_f=True,
    )
    assert scene.robot_state_0.body_parent_f is not None

    ee_idx = fr3_robot.resolve_ee_body_index(scene.robot_model)
    tcp_idx = int(scene.tcp_body_index)
    m_ee = float(scene.robot_model.body_mass.numpy()[ee_idx])
    m_tcp = float(scene.robot_model.body_mass.numpy()[tcp_idx])
    assert m_ee > 0.5
    # TCP is effectively massless (USD may assign a tiny placeholder after inertia fixup).
    assert m_tcp < 0.01 * m_ee

    ctrl = _configure_joint_torque_vic(scene)
    dt = 1.0 / 500.0
    for _ in range(100):
        scene.update_fr3_ee_teleop(dt, ctrl)
        scene.mujoco_substep(dt)

    parent0 = read_ee_parent_f_world(scene.robot_state_0, ee_idx)
    wrench0 = tcp_world_wrench_from_scene(scene)
    # Force transport is pure negation (lever arm affects torque only).
    np.testing.assert_allclose(wrench0[:3], -parent0[:3], atol=1e-6)

    g = 9.81
    scene.robot_model.set_gravity((0.0, 0.0, -g))
    scene.mj_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    for _ in range(300):
        scene.update_fr3_ee_teleop(dt, ctrl)
        scene.mujoco_substep(dt)

    parent1 = read_ee_parent_f_world(scene.robot_state_0, ee_idx)
    wrench1 = tcp_world_wrench_from_scene(scene)
    np.testing.assert_allclose(wrench1[:3], -parent1[:3], atol=1e-6)
    # Env-on-robot F_z becomes more negative (downward load on robot) by a
    # substantial fraction of EE weight.
    delta_fz = float(wrench1[2] - wrench0[2])
    assert delta_fz < -0.25 * m_ee * g, (
        f"expected gravity to increase downward env-on-robot load; "
        f"delta_fz={delta_fz}, -0.25*m*g={-0.25 * m_ee * g}"
    )
