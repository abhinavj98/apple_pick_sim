"""M2.1 Gymnasium environment tests (placeholder obs + parity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401
        from gymnasium.utils.env_checker import check_env  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed (expected to be provided by newton[dev])",
)


def _import_constants():
    # Reuse direct-path timing constants to avoid drift.
    from apple_pick_sim.tests.conftest import FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT, RANGES_FIXTURE, requires_fr3

    return FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT, RANGES_FIXTURE, requires_fr3


@dataclass(frozen=True)
class _StepMetrics:
    joint_q: np.ndarray
    tcp_body_q: np.ndarray
    tcp_wrench: np.ndarray
    params_fp: dict
    fruiting_link_forces: dict[str, dict[str, np.ndarray]]


def _fruiting_link_forces_from_scene(scene, sub_dt: float) -> dict[str, dict[str, np.ndarray]]:
    import apple_pick_sim.fruiting_system as fs

    cable = scene.cable
    measured = fs.measure_fruiting_forces(
        cable, cable.state_0.body_q, cable.state_1.body_q, dt=float(sub_dt)
    )
    out: dict[str, dict[str, np.ndarray]] = {}
    for rec in measured["fixed_joints"]:
        key = rec.label.removeprefix("joint_") or rec.label
        out[key] = {
            "force_world": np.asarray(rec.force_world, dtype=np.float64),
            "torque_at_child_com_world": np.asarray(rec.torque_at_child_com_world, dtype=np.float64),
        }
    return out


def _extract_metrics(scene, sub_dt: float) -> _StepMetrics:
    import apple_pick_sim.fruiting_system as fs

    tcp = scene.tcp_body_index
    joint_q = scene.robot_state_0.joint_q.numpy().reshape(-1).copy()
    tcp_body_q = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp].copy()
    tcp_wrench = scene.proxy_forces.numpy().reshape(-1, 6)[tcp].copy()
    params_fp = fs.params_fingerprint(scene.cable.params)
    return _StepMetrics(
        joint_q=joint_q,
        tcp_body_q=tcp_body_q,
        tcp_wrench=tcp_wrench,
        params_fp=params_fp,
        fruiting_link_forces=_fruiting_link_forces_from_scene(scene, sub_dt),
    )


@gymnasium_available
def test_env_observation_contract_placeholder():
    import gymnasium as gym

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    obs, info = env.reset(seed=0)

    assert isinstance(obs, dict)
    assert "dummy" in obs
    assert "schema_version" in obs
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs["dummy"], np.zeros((1,), dtype=np.float32), rtol=0, atol=0)
    assert obs["schema_version"] == 1
    assert "params_fingerprint" in info

    obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, (bool, np.bool_))
    assert "params_fingerprint" in info2
    assert "end_effector_wrench" in info2
    assert "fruiting_link_forces" in info2

    env.close()


@gymnasium_available
def test_info_exposes_fruiting_link_and_ee_forces():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import gymnasium as gym

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    _, info = env.reset(seed=0)

    ee = info["end_effector_wrench"]
    assert ee.shape == (6,)
    assert ee.dtype == np.float32

    links = info["fruiting_link_forces"]
    assert isinstance(links, dict)
    assert len(links) > 0
    for label, entry in links.items():
        assert isinstance(label, str)
        assert entry["force_world"].shape == (3,)
        assert entry["torque_at_child_com_world"].shape == (3,)
        assert entry["force_world"].dtype == np.float32

    # Apple hangs under gravity: stem→apple joint holds upward force on the apple.
    apple_key = next(k for k in links if k.endswith("_apple"))
    assert float(links[apple_key]["force_world"][2]) > 0.0

    _, _, _, _, info2 = env.step(12)
    assert info2["end_effector_wrench"].shape == (6,)
    assert len(info2["fruiting_link_forces"]) == len(links)

    env.close()


@gymnasium_available
def test_env_parity_against_direct_coupled_sim():
    FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT, RANGES_FIXTURE, requires_fr3 = _import_constants()

    if getattr(requires_fr3, "args", None):  # pytest mark object
        # If FR3 assets are missing, skip like the existing direct-path tests do.
        requires_fr3.args[0]  # touch for type check
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import gymnasium as gym
    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot

    ranges = fs.load_ranges(RANGES_FIXTURE)
    seed = 13
    mujoco_solver_kwargs = {"disable_contacts": True}

    # --- Direct reference rollout ---
    direct_scene = cf.build_coupled_fruiting_fr3(
        ranges,
        seed,
        mujoco_solver_kwargs=mujoco_solver_kwargs,
        enable_self_collisions=False,
    )
    direct_scene.robot_kinematic_mode = True
    direct_ctrl = fr3_robot.Fr3EEDirectJointController(direct_scene.robot_model, direct_scene.tcp_body_index)
    direct_ctrl.sync_target_from_state(direct_scene.robot_state_0)

    # --- Gym rollout (same seed + action schedule) ---
    env = gym.make(
        "ApplePickCoupled-v0",
        render_mode=None,
        max_episode_steps=10,
        mujoco_solver_kwargs=mujoco_solver_kwargs,
        enable_self_collisions=False,
    )
    env.reset(seed=seed)

    # Deterministic action schedule (includes noop and several axes).
    action_schedule = [12, 0, 0, 2, 10, 12, 4, 6]

    for a in action_schedule:
        # Step direct scene using the env's mapping.
        vel = env.unwrapped._action_to_velocity(a)  # intentional: parity contract surface
        direct_scene.apply_fr3_ee_teleop_direct(FRAME_DT, direct_ctrl, velocity=vel)
        for _ in range(SUBSTEPS_PER_FRAME):
            direct_scene.coupled_substep(SUB_DT)

        # Step gym env once (one frame).
        env.step(a)

    got = env.unwrapped._scene
    assert got is not None

    m_direct = _extract_metrics(direct_scene, SUB_DT)
    m_gym = _extract_metrics(got, SUB_DT)

    np.testing.assert_allclose(m_gym.joint_q, m_direct.joint_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m_gym.tcp_body_q, m_direct.tcp_body_q, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m_gym.tcp_wrench, m_direct.tcp_wrench, rtol=1e-5, atol=1e-4)
    assert m_gym.params_fp == m_direct.params_fp
    assert m_gym.fruiting_link_forces.keys() == m_direct.fruiting_link_forces.keys()
    for key in m_direct.fruiting_link_forces:
        np.testing.assert_allclose(
            m_gym.fruiting_link_forces[key]["force_world"],
            m_direct.fruiting_link_forces[key]["force_world"],
            rtol=1e-5,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            m_gym.fruiting_link_forces[key]["torque_at_child_com_world"],
            m_direct.fruiting_link_forces[key]["torque_at_child_com_world"],
            rtol=1e-5,
            atol=1e-4,
        )

    _, _, _, _, info = env.step(12)
    np.testing.assert_allclose(info["end_effector_wrench"], m_gym.tcp_wrench, rtol=1e-5, atol=1e-4)
    for key in m_gym.fruiting_link_forces:
        np.testing.assert_allclose(
            info["fruiting_link_forces"][key]["force_world"],
            m_gym.fruiting_link_forces[key]["force_world"],
            rtol=1e-5,
            atol=1e-4,
        )

    env.close()


@gymnasium_available
def test_check_env_smoke():
    _, _, _, _, _ = _import_constants()
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=3)
    check_env(env, skip_render_check=True)
    env.close()

