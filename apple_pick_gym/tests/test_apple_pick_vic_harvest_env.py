"""ApplePickVicHarvestEnv: 13-D delta-pose action, vic_pose (19D) controller,
arm+plant DR, sensor-realistic obs, privileged info.

CPU, N=2 -- fast smoke coverage of the acceptance criteria in
docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md Task 6. The IK
convergence gate (Step 5) and the full rollout artifact (Step 6) are
measured/generated separately (see the report), not as pytest assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from apple_pick_sim.robot import fr3_robot


def _usd_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return fr3_robot.fr3_assets_available()


requires_fr3 = pytest.mark.skipif(
    not _usd_available(), reason="Requires bundled assets/fr3 and usd-core"
)


def _make_env(num_envs: int = 2, **kwargs):
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    return ApplePickVicHarvestEnv(num_envs=num_envs, device="cpu", **kwargs)


@requires_fr3
class TestApplePickVicHarvestEnv:
    def test_support_dr_is_baked_into_build_and_snapshot(self):
        """Support DR must be in the build config (so it is settled) and in the stored
        snapshot -- not applied afterwards and reverted by restore_episode_snapshot()."""
        env = _make_env(num_envs=4)
        try:
            sample = env._last_support_dr_sample
            assert sample is not None
            fs = env._sim.config.fruiting_system
            assert fs.support_kp_per_env == tuple(float(x) for x in sample.kp)
            assert fs.support_roll_kp_per_env == tuple(float(x) for x in sample.roll_kp)
            assert fs.support_zeta_per_env == tuple(float(x) for x in sample.zeta)
            assert len(set(fs.support_kp_per_env)) > 1

            import warp as wp

            snap = env._sim.episode_snapshot
            live = wp.to_torch(env._sim.scene.cable.solver.joint_penalty_k)
            torch.testing.assert_close(wp.to_torch(snap.joint_penalty_k), live)
            env.reset()
            torch.testing.assert_close(wp.to_torch(env._sim.scene.cable.solver.joint_penalty_k), live)
        finally:
            env.close()

    def test_uses_dynamic_apple_with_weld_harvest_like_sysid(self):
        env = _make_env()
        try:
            cfg = env._sim.config
            assert cfg.robot.gripper.dynamic_apple is True
            assert cfg.fruiting_system.tcp_harvest_source == "weld"
        finally:
            env.close()

    def test_action_space_is_13d(self):
        env = _make_env()
        try:
            assert env.action_space.shape == (13,)
        finally:
            env.close()

    def test_controller_config_is_vic_pose_19d(self):
        env = _make_env()
        try:
            assert env._sim.config.controller.mode == "vic_pose"
            assert env._sim.config.controller.action_dim == 19
        finally:
            env.close()

    def test_target_pose_initializes_to_current_tcp_pose_on_reset(self):
        env = _make_env()
        try:
            env.reset()
            import warp as wp

            bufs = env._sim.obs_bufs
            tcp_pose = wp.to_torch(bufs.tcp_pose).to(device=env.device, dtype=torch.float32)
            torch.testing.assert_close(env._target_pose[:, :3], tcp_pose[:, :3], atol=1e-4, rtol=0)
            # quat: target_pose is wxyz, tcp_pose is xyzw (Warp-native) -- compare reordered.
            tcp_quat_wxyz = tcp_pose[:, [6, 3, 4, 5]]
            torch.testing.assert_close(env._target_pose[:, 3:7], tcp_quat_wxyz, atol=1e-4, rtol=0)
        finally:
            env.close()

    def test_actions_reach_sim_as_19d_vic_pose(self):
        env = _make_env()
        try:
            env.reset()
            raw_action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            raw_action[:, 6:9] = 100.0
            raw_action[:, 9:12] = 10.0
            raw_action[:, 12] = 1.0
            packed = env._actions_tensor(raw_action)
            assert packed.shape == (2, 19)
        finally:
            env.close()

    def test_info_carries_privileged_forces_not_in_obs(self):
        env = _make_env()
        try:
            obs, info = env.reset()
            assert "woody_part_force" not in obs
            assert "target_junction_force" not in obs
            action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            action[:, 6:9] = 100.0
            action[:, 9:12] = 10.0
            action[:, 12] = 1.0
            obs, reward, terminated, truncated, info = env.step(action)
            assert "woody_part_force" in info
            assert "target_junction_force" in info
            assert env._target_junction_name in info["woody_part_force"]
        finally:
            env.close()

    def test_arm_dr_resamples_on_reset(self):
        env = _make_env()
        try:
            env.reset()
            sample1 = env._last_arm_dr_sample.armature.copy()
            env.reset()
            sample2 = env._last_arm_dr_sample.armature.copy()
            assert not np.allclose(sample1, sample2)
        finally:
            env.close()

    def test_sensor_model_state_resets(self):
        # bias_std must be nonzero for this to be a meaningful check --
        # FtSensorConfig()'s own default is bias_std=0.0 (quiet unless
        # configured), so bias would trivially be 0 == 0 across resets.
        from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig

        env = _make_env(ft_sensor_config=FtSensorConfig(bias_std=1.0))
        try:
            env.reset()
            action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            action[:, 6:9] = 100.0
            action[:, 9:12] = 10.0
            action[:, 12] = 1.0
            obs1, *_ = env.step(action)
            bias_before = env._ft_sensor._bias.clone()
            env.reset()
            bias_after = env._ft_sensor._bias.clone()
            assert not torch.allclose(bias_before, bias_after)
        finally:
            env.close()

    def test_reward_and_termination_are_wired_not_stubbed(self):
        """Task 7 wiring smoke test: reward is nonzero-capable (not the Task 6
        stub), success/freeze/safety plumb through, truncated stays uniform."""
        from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
        from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig

        env = _make_env(
            reward_config=HarvestRewardConfig(f_threshold_n=5.0),
            episode_config=EpisodeConfig(success_streak_steps=2, safety_force_cap_n=1e6, safety_torque_cap_nm=1e6),
            max_episode_steps=5,
        )
        try:
            env.reset()
            action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            action[:, 2] = 1.0  # request a large +z delta (clamped) to build load
            action[:, 6:9] = 200.0
            action[:, 9:12] = 10.0
            action[:, 12] = 1.0

            rewards = []
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward.clone())
                # truncated must be identical across the batch every step.
                assert torch.all(truncated == truncated[0])

            assert any(torch.any(r != 0.0) for r in rewards), "reward stayed at the Task 6 stub (all zero)"
            assert env._success_tracker.streak.shape == (2,)
            assert env._freeze_mask.done_mask.shape == (2,)
        finally:
            env.close()

    def test_info_exposes_reward_terms_and_termination_reasons(self):
        env = _make_env()
        try:
            env.reset()
            action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            action[:, 6:9] = 200.0
            action[:, 9:12] = 10.0
            action[:, 12] = 1.0
            _obs, reward, _term, _trunc, info = env.step(action)

            rt = info["reward_terms"]
            assert set(rt["raw"]) == {"progress", "pullout", "collateral"}
            assert set(rt["weighted"]) == {"progress", "pullout", "collateral"}
            weighted_sum = sum(rt["weighted"].values())
            torch.testing.assert_close(rt["dense"], weighted_sum)
            torch.testing.assert_close(rt["total"], reward.reshape(2))
            assert rt["total"].shape == (2,)

            ep = info["episode"]
            for key in (
                "success_this_step",
                "success_achieved",
                "success_streak",
                "safety_junction",
                "safety_wrist",
                "frozen",
            ):
                assert ep[key].shape == (2,), key
            assert info["target_pose"].shape == (2, 7)
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
