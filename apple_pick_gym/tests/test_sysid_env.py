"""Tests for ApplePickSysIdEnv (§2.1 quasi-static stiffness mapping)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from apple_pick_sim.tests.conftest import fr3_assets_available


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed (expected to be provided by newton[dev])",
)


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_sysid_reset_with_warmup_substeps_does_not_fail_weld_validation():
    """Weld direction must be chosen from nominal apple pose, not post-settle pose."""
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=1800,
        n_weld_hemisphere_samples=10,
    )
    _, info = env.reset(seed=2345)
    assert "weld_direction" in info
    env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_sysid_reset_reports_weld_direction():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        n_weld_hemisphere_samples=8,
    )
    _, info = env.reset(seed=3)
    assert "weld_direction" in info
    weld = np.asarray(info["weld_direction"], dtype=np.float64).reshape(3)
    assert abs(float(np.linalg.norm(weld)) - 1.0) < 1e-5

    scene = env.unwrapped._scene
    apple = int(scene.cable.apple_body)
    apple_pos = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3]
    robot_vec = np.asarray(info["robot_base_pos"], dtype=np.float64) - apple_pos

    stem_bodies = scene.cable.stem_bodies
    assert len(stem_bodies) >= 2
    body_q = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    physical_stem = body_q[int(stem_bodies[-1]), :3] - body_q[int(stem_bodies[-2]), :3]
    physical_stem /= np.linalg.norm(physical_stem)

    from apple_pick_sim.system_id import stem_perpendicular_robot_pole

    pole = stem_perpendicular_robot_pole(physical_stem, robot_vec)
    assert float(np.dot(weld, pole)) >= 0.0
    assert abs(float(np.dot(weld, physical_stem))) <= np.sin(0.5 * np.pi) + 1e-5
    env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_sysid_successive_resets_cycle_weld():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        n_weld_hemisphere_samples=8,
    )
    _, info_a = env.reset(seed=3)
    _, info_b = env.reset(seed=3)
    weld_a = np.asarray(info_a["weld_direction"], dtype=np.float64)
    weld_b = np.asarray(info_b["weld_direction"], dtype=np.float64)
    assert not np.allclose(weld_a, weld_b, atol=1e-4)
    env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_sysid_reset_weld_direction_override():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
    )
    _, info_ref = env.reset(seed=3)
    weld_ref = np.asarray(info_ref["weld_direction"], dtype=np.float64)
    scene = env.unwrapped._scene
    apple = int(scene.cable.apple_body)
    apple_pos = scene.cable.state_0.body_q.numpy().reshape(-1, 7)[apple, :3]
    robot_vec = np.asarray(info_ref["robot_base_pos"], dtype=np.float64) - apple_pos
    stem_bodies = scene.cable.stem_bodies
    body_q = scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    physical_stem = body_q[int(stem_bodies[-1]), :3] - body_q[int(stem_bodies[-2]), :3]
    physical_stem /= np.linalg.norm(physical_stem)

    from apple_pick_sim.system_id import stem_perpendicular_robot_pole

    pole = stem_perpendicular_robot_pole(physical_stem, robot_vec)
    oblique = pole + np.array([0.3, 0.2, 0.0], dtype=np.float64)
    oblique /= np.linalg.norm(oblique)
    assert float(np.dot(oblique, pole)) > 0.0

    env.close()
    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
    )
    _, info = env.reset(
        seed=3,
        options={"weld_direction": (float(oblique[0]), float(oblique[1]), float(oblique[2]))},
    )
    weld = np.asarray(info["weld_direction"], dtype=np.float64)
    cos_1deg = math.cos(math.radians(1.0))
    assert float(np.dot(weld, oblique)) > cos_1deg
    assert not np.allclose(weld, weld_ref, atol=1e-3)
    env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_sysid_step_info_includes_weld_direction():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=0,
        robot_facing_weld=True,
    )
    _, reset_info = env.reset(seed=0)
    assert "weld_direction" in reset_info
    reset_weld = np.asarray(reset_info["weld_direction"], dtype=np.float64)

    _, _, _, _, step_info = env.step(np.zeros(6, dtype=np.float32))
    assert "weld_direction" in step_info
    step_weld = np.asarray(step_info["weld_direction"], dtype=np.float64)
    np.testing.assert_allclose(step_weld, reset_weld, atol=1e-5)
    env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_sysid_zero_action_first_step_stays_quiet_after_welded_warmup():
    """Quiet-start reset must seed both VBD state buffers before the first step."""
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=1800,
        mujoco_solver_kwargs={"disable_contacts": True},
    )
    obs, _ = env.reset(seed=0)
    initial_tcp = np.asarray(obs["tcp_pos"], dtype=np.float64)
    scene = env.unwrapped._scene
    cable = scene.cable

    state_0 = cable.state_0.body_q.numpy().reshape(-1, 7)
    state_1 = cable.state_1.body_q.numpy().reshape(-1, 7)
    np.testing.assert_allclose(state_1[:, :3], state_0[:, :3], rtol=0.0, atol=1e-5)

    obs, *_ = env.step(np.zeros(6, dtype=np.float32))
    tcp_error_m = float(np.linalg.norm(np.asarray(obs["tcp_pos"], dtype=np.float64) - initial_tcp))
    ft_mag_n = float(np.linalg.norm(np.asarray(obs["ft_wrist"], dtype=np.float64)[:3]))
    assert tcp_error_m < 0.005
    assert ft_mag_n < 250.0
    env.close()


@gymnasium_available
def test_sysid_env_action_space():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(max_linear_vel=0.2, max_angular_vel=1.0)
    assert env.action_space.shape == (6,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_allclose(env.action_space.low[:3], -0.2)
    np.testing.assert_allclose(env.action_space.high[:3], 0.2)
    np.testing.assert_allclose(env.action_space.low[3:], -1.0)
    np.testing.assert_allclose(env.action_space.high[3:], 1.0)


@gymnasium_available
def test_sysid_env_obs_keys():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    obs, _ = env.reset(seed=0)

    for key in (
        "excitation_type",
        "excitation_f_inst",
        "excitation_direction",
        "tcp_pos",
        "tcp_quat",
        "apple_quat",
        "robot_joint_q",
        "ft_wrist",
        "raw_ft_wrist",
    ):
        assert key in obs
    assert obs["tcp_quat"].shape == (4,)
    assert obs["apple_quat"].shape == (4,)
    assert obs["robot_joint_q"].shape == (7,)
    assert obs["raw_ft_wrist"].shape == (6,)
    assert env.observation_space.contains(obs)
    env.close()


@gymnasium_available
def test_sysid_env_tcp_pos_is_actual():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    obs, _ = env.reset(seed=0)
    scene = env.unwrapped._scene
    tcp = int(scene.tcp_body_index)
    action = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, *_ = env.step(action)
    actual = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp, :3].astype(np.float32)
    target = np.asarray(env.unwrapped._controller.target_tf[:3], dtype=np.float32)

    np.testing.assert_allclose(obs["tcp_pos"], actual, rtol=1e-5, atol=1e-5)
    assert not np.allclose(obs["tcp_pos"], target, atol=1e-4)
    env.close()


@gymnasium_available
def test_restore_grasp_pose_returns_tcp_to_initial():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=8,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    obs, _ = env.reset(seed=0)
    initial_tcp = np.asarray(obs["tcp_pos"], dtype=np.float64).copy()

    push = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(4):
        obs, *_ = env.step(push)

    displaced = np.asarray(obs["tcp_pos"], dtype=np.float64)
    assert float(np.linalg.norm(displaced - initial_tcp)) > 0.005

    env.restore_grasp_pose()
    restored = np.asarray(env._tcp_pos(), dtype=np.float64)
    np.testing.assert_allclose(restored, initial_tcp, rtol=0, atol=0.001)

    zero = np.zeros((6,), dtype=np.float32)
    obs, *_ = env.step(zero)
    after_step = np.asarray(obs["tcp_pos"], dtype=np.float64)
    np.testing.assert_allclose(after_step, initial_tcp, rtol=0, atol=0.01)
    env.close()


@gymnasium_available
def test_sysid_env_no_force_termination():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    env.reset(seed=0)
    action = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(action)
    assert terminated is False
    env.close()


@gymnasium_available
def test_sysid_env_context_roundtrip():
    from apple_pick_sim.system_id import ExcitationContext
    from apple_pick_sim.tests.conftest import fr3_assets_available
    from apple_pick_gym.envs import ApplePickSysIdEnv

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
    )
    env.reset(seed=0)
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=direction)
    env.set_excitation_context(ctx)

    obs = env._make_obs()
    assert int(obs["excitation_type"]) == 0
    assert float(obs["excitation_f_inst"]) == 0.0
    np.testing.assert_allclose(obs["excitation_direction"], direction.astype(np.float32))


@gymnasium_available
def test_sysid_env_action_clipping():
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(max_linear_vel=0.2, max_angular_vel=1.0)
    cmd = env._action_to_command(np.array([1.0, -1.0, 0.5, 2.0, -2.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(cmd.linear, (0.2, -0.2, 0.2), rtol=0, atol=1e-6)
    np.testing.assert_allclose(cmd.angular, (1.0, -1.0, 0.0), rtol=0, atol=1e-6)


@gymnasium_available
def test_sysid_env_log_movement_direction_arrow():
    from newton.viewer import ViewerNull

    from apple_pick_gym.envs import ApplePickSysIdEnv

    class _ArrowProbe(ViewerNull):
        def __init__(self):
            super().__init__(num_frames=1)
            self.arrow_calls: list[str] = []
            self.line_calls: list[str] = []

        def log_arrows(self, name, starts, ends, colors, hidden=False):
            del starts, ends, colors, hidden
            self.arrow_calls.append(name)

        def log_lines(self, name, starts, ends, colors, hidden=False):
            del starts, ends, colors, hidden
            self.line_calls.append(name)

    class _BodyQ:
        @staticmethod
        def numpy():
            return np.array([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    scene = type(
        "Scene",
        (),
        {"tcp_body_index": 0, "robot_state_0": type("S", (), {"body_q": _BodyQ()})()},
    )()
    obs = {"excitation_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    viewer = _ArrowProbe()
    ApplePickSysIdEnv.log_movement_direction_arrow(
        viewer, obs, scene=scene, linear_velocity=(0.05, 0.0, 0.0)
    )
    assert viewer.arrow_calls == ["/gym/movement_direction"]

    viewer2 = _ArrowProbe()
    ApplePickSysIdEnv.log_movement_direction_arrow(
        viewer2, obs, scene=scene, linear_velocity=(0.0, 0.0, 0.0)
    )
    assert viewer2.arrow_calls == ["/gym/movement_direction"]
    assert viewer2.line_calls == []

    viewer3 = _ArrowProbe()
    ApplePickSysIdEnv.log_movement_direction_arrow(viewer3, {}, scene=scene)
    assert viewer3.arrow_calls == ["/gym/movement_direction"]
    assert viewer3.line_calls == []


@gymnasium_available
def test_sysid_env_default_vic_stiffness_and_100n_force_torque_caps():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv()
    assert env._vic_gains.linear_k == 2000.0
    assert env._stem_force_cap_n == 100.0
    assert env._stem_torque_cap_nm == 100.0
    build_kw = env._coupled_build_kwargs()
    assert build_kw["stem_force_cap_N"] == 100.0
    assert build_kw["stem_torque_cap_Nm"] == 100.0


@gymnasium_available
def test_gym_make_does_not_set_sysid_env_internal_max_episode_steps():
    """``gym.make(..., max_episode_steps=N)`` only wraps TimeLimit; truncation uses env cfg."""
    import gymnasium as gym

    import apple_pick_gym  # noqa: F401 — registers ApplePickSysId-v0

    from apple_pick_gym.envs import ApplePickSysIdEnv

    want = 512
    wrapped = gym.make("ApplePickSysId-v0", render_mode=None, max_episode_steps=want)
    try:
        assert wrapped._max_episode_steps == want
        assert wrapped.unwrapped._cfg.max_episode_steps == 240
    finally:
        wrapped.close()

    direct = ApplePickSysIdEnv(render_mode=None, max_episode_steps=want)
    assert direct._cfg.max_episode_steps == want
