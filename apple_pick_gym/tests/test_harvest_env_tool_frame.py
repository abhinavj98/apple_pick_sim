"""[D17] The tool-frame interface on a real build (one build, one test).

``action_frame="tool"``: the actor reads start-relative, TCP-frame fields; the F/T sensor model
runs in the TCP frame; the critic's grasp axis is the realized one (TCP ``-z``); the realized-K
draw is privileged. Checked through the skrl wrapper, which picks the tool layout from the env.

CPU build (wiring only; the arm does not integrate on CPU). Run on its own (one build per
process): ``pytest apple_pick_gym/tests/test_harvest_env_tool_frame.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_WORLDS = Path(__file__).resolve().parent.parent / "world_sets" / "harvest_worlds_v1.jsonl"


def test_tool_frame_obs_critic_and_privileged_wiring():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
    from apple_pick_gym.batched_envs.harvest_obs import actor_obs_layout, rotate_tool_wrench_to_world
    from apple_pick_gym.batched_envs.harvest_reward import quat_rotate_vector
    from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig
    from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs
    from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper

    n = 2
    specs = load_world_set(_WORLDS)[:n]
    bounds = HarvestActionBounds(max_target_pos_offset_m=0.15, max_target_rot_offset_rad=0.5, k_lin_max=500.0)
    with pytest.warns(UserWarning, match="does not integrate"):
        env = ApplePickVicHarvestEnv(
            device="cpu",
            max_episode_steps=6,
            action_bounds=bounds,
            ft_sensor_config=FtSensorConfig.rl_training(),
            action_frame="tool",
            realized_k_scale_range=(0.8, 1.2),
            **world_specs_to_env_kwargs(specs),
        )
    try:
        assert env.obs_frame == "tool"
        obs, info = env.reset()
        q = obs["tcp_quat"]
        torch.testing.assert_close(obs["d_pos_tool"], torch.zeros(n, 3), atol=1e-6, rtol=0)
        torch.testing.assert_close(obs["d_rot_tool"], torch.zeros(n, 3), atol=1e-6, rtol=0)
        g_world = quat_rotate_vector(q, obs["gravity_tool"])
        torch.testing.assert_close(g_world, torch.tensor([[0.0, 0.0, -1.0]] * n), atol=1e-5, rtol=0)
        # the world-frame sensor value is the tool-frame reading rotated back
        torch.testing.assert_close(obs["ft_wrist"], rotate_tool_wrench_to_world(obs["ft_tool"], q), atol=1e-5, rtol=0)
        # critic geometry: the realized grasp axis (apple -> gripper) is the TCP's -z
        z_tool = quat_rotate_vector(q, torch.tensor([[0.0, 0.0, 1.0]] * n))
        torch.testing.assert_close(env.plant_geometry()["weld_direction"], -z_tool, atol=1e-5, rtol=0)
        k = env.privileged_fields()["realized_k_scale"]
        assert k.shape == (n, 6) and bool(((k >= 0.8) & (k <= 1.2)).all())

        wrapper = HarvestSkrlWrapper(env)
        flat, _ = wrapper.reset()
        assert wrapper.actor_layout == actor_obs_layout("tool")
        assert flat.shape == (n, actor_obs_layout("tool").total_width)
        assert "step_frac" in [e.name for e in wrapper.critic_layout.entries]
        assert "step_frac" not in [e.name for e in wrapper.actor_layout.entries]
        nxt, reward, term, trunc, _ = wrapper.step(torch.zeros(n, 13))
        assert bool(torch.isfinite(nxt).all()) and bool(torch.isfinite(wrapper.state()).all())
    finally:
        env.close()
