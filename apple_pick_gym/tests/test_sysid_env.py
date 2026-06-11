"""Tests for ApplePickSysIdEnv (§2.1 quasi-static stiffness mapping)."""

from __future__ import annotations

import numpy as np
import pytest


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
        "ft_wrist",
    ):
        assert key in obs
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
def test_sysid_env_wrench_guard():
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv(
        max_episode_steps=2,
        fix_to_apple=False,
        fix_to_apple_warmup_substeps=0,
        max_tcp_force_n=0.001,
    )
    env.reset(seed=0)
    action = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(action)
    assert terminated is True
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


@gymnasium_available
def test_sysid_env_default_vic_stiffness():
    from apple_pick_gym.envs import ApplePickSysIdEnv

    env = ApplePickSysIdEnv()
    assert env._vic_gains.linear_k == 3000.0
