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

    def test_world_specs_round_trip_rebuilds_identical_worlds(self, tmp_path):
        """export_world_specs -> world_specs_to_env_kwargs rebuilds the same plant, grasp,
        support-joint DR and build-time arm DR (what a screened world set relies on).
        The source build runs in a subprocess: one build per OS process."""
        import subprocess
        import sys

        from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs
        from apple_pick_sim.fruiting_system.params import fruiting_params_to_json

        src = tmp_path / "src.jsonl"
        subprocess.run(
            [sys.executable, "-m", "apple_pick_gym.batched_examples.example_screen_harvest_worlds",
             "sample", "--num-envs", "2", "--seed", "3", "--device", "cpu", "--out", str(src),
             "--hold-steps", "1", "--pull-rest-steps", "1", "--pull-ramp-steps", "1",
             "--pull-hold-steps", "1", "--pull-settle-steps", "1", "--num-pull-episodes", "1"],
            check=True, capture_output=True, text=True,
        )
        specs = load_world_set(src)
        assert [s.world_id for s in specs] == ["s3_e0", "s3_e1"]
        kwargs = {k: v for k, v in world_specs_to_env_kwargs(specs).items() if k != "num_envs"}
        rebuilt = _make_env(dr_seed=99, **kwargs)
        try:
            again = rebuilt.export_world_specs(prefix="s3")
            assert [fruiting_params_to_json(p) for p in rebuilt._sim.per_env_params] == [
                s.params_json for s in specs
            ]
            for a, b in zip(specs, again):
                assert a.world_id == b.world_id
                assert a.weld_direction == b.weld_direction
                assert a.support_kp == pytest.approx(b.support_kp)
                assert a.support_roll_kp == pytest.approx(b.support_roll_kp)
                assert a.support_zeta == pytest.approx(b.support_zeta)
                assert a.arm_link_mass_scale == pytest.approx(b.arm_link_mass_scale)
                assert a.arm_ee_inertia_scale == pytest.approx(b.arm_ee_inertia_scale)
        finally:
            rebuilt.close()

    def test_default_ranges_are_the_real_data_fixture(self):
        from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import (
            _RL_HARVEST_RANGES_FIXTURE,
        )

        assert _RL_HARVEST_RANGES_FIXTURE.name == "fruiting_system_ranges_rl_harvest_real_g05_m1.json"
        assert _RL_HARVEST_RANGES_FIXTURE.exists()

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

    def test_controller_matches_real_collection_osc_like_sysid_replay(self):
        """Same OSC as real_replay_sim_config's vic_pose: rotation without Lambda
        (sep_ori) and the real kd_null, so K_ang means the same thing as on the rig."""
        from apple_pick_gym.batched_envs.real_batched_replay_build import (
            _REAL_OSC_KD_NULL,
            _REAL_OSC_KP_NULL,
            _REAL_OSC_SEP_ORI,
        )

        env = _make_env()
        try:
            ctrl = env._sim.config.controller
            assert ctrl.sep_ori is _REAL_OSC_SEP_ORI
            assert ctrl.kp_null == _REAL_OSC_KP_NULL
            assert ctrl.kd_null == _REAL_OSC_KD_NULL
            assert env._sim.scene.vic_jt_sep_ori is _REAL_OSC_SEP_ORI
            assert env._sim.scene.vic_jt_kd_null == _REAL_OSC_KD_NULL
        finally:
            env.close()

    def test_frozen_env_holds_its_target_pose(self):
        """A frozen env replays its last gains but must not keep integrating its last delta."""
        from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import HoldSettleConfig

        # No env pre-frozen as an invalid grasp: only the explicit freeze below applies.
        env = _make_env(
            hold_settle=HoldSettleConfig(max_rest_pos_err_m=1e6, max_rest_wrist_force_n=1e6)
        )
        try:
            env.reset()
            action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            action[:, 0] = 0.01
            action[:, 6:9] = 100.0
            action[:, 9:12] = 10.0
            action[:, 12] = 1.0
            env._actions_tensor(action)
            env._freeze_mask.update(torch.tensor([True, False], device=env.device))
            before = env._target_pose.clone()
            env._actions_tensor(action)
            torch.testing.assert_close(env._target_pose[0], before[0])
            assert float(env._target_pose[1, 0] - before[1, 0]) == pytest.approx(0.01, abs=1e-6)
        finally:
            env.close()

    def test_broken_grasp_envs_are_flagged_and_frozen_from_reset(self):
        """Envs whose settled TCP sits far from the hold target (failed IK grasp, apple
        welded to a TCP centimetres away) are flagged invalid and frozen with zero reward."""
        from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import HoldSettleConfig

        # A negative tolerance marks every env invalid, exercising the wiring deterministically.
        env = _make_env(hold_settle=HoldSettleConfig(max_rest_pos_err_m=-1.0))
        try:
            assert env._invalid_env_mask.tolist() == [True, True]
            _obs, info = env.reset()
            assert info["invalid_env"].tolist() == [True, True]
            assert env._freeze_mask.done_mask.tolist() == [True, True]
            action = torch.zeros((2, 13), dtype=torch.float32, device=env.device)
            action[:, 6:9] = 100.0
            action[:, 9:12] = 10.0
            action[:, 12] = 1.0
            _obs, reward, _term, _trunc, info = env.step(action)
            assert torch.all(reward == 0.0)
            assert info["invalid_env"].tolist() == [True, True]
        finally:
            env.close()

    def test_broken_grasp_is_flagged_by_rest_wrist_force_alone(self):
        """A failed grasp can settle within the position tolerance while the weld still
        loads the wrist with >100 N, so rest wrist force is an independent criterion."""
        from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import HoldSettleConfig

        env = _make_env(
            hold_settle=HoldSettleConfig(max_rest_pos_err_m=1e6, max_rest_wrist_force_n=-1.0)
        )
        try:
            assert env._invalid_env_mask.tolist() == [True, True]
        finally:
            env.close()

    def test_valid_grasps_are_not_flagged(self):
        from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import HoldSettleConfig

        env = _make_env(
            hold_settle=HoldSettleConfig(max_rest_pos_err_m=1e6, max_rest_wrist_force_n=1e6)
        )
        try:
            _obs, info = env.reset()
            assert info["invalid_env"].tolist() == [False, False]
            assert env._freeze_mask.done_mask.tolist() == [False, False]
        finally:
            env.close()

    def test_target_pose_initializes_to_current_tcp_pose_on_reset_without_hold_settle(self):
        from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import HoldSettleConfig

        env = _make_env(hold_settle=HoldSettleConfig(enabled=False))
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

    def test_hold_settle_stores_equilibrium_target_used_at_reset(self):
        env = _make_env()
        try:
            assert env._hold_target_pose is not None
            env.reset()
            torch.testing.assert_close(env._target_pose, env._hold_target_pose)
            import warp as wp

            tcp = wp.to_torch(env._sim.obs_bufs.tcp_pose).to(device=env.device, dtype=torch.float32)[:, :3]
            # The arm holds near the stored target (sag = load / Kp, centimetres at most).
            assert torch.all((tcp - env._hold_target_pose[:, :3]).norm(dim=-1) < 0.1)
            # A second reset restores the same stored equilibrium, not the sagged TCP.
            env.reset()
            torch.testing.assert_close(env._target_pose, env._hold_target_pose)
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
        from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig
        from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
        from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig

        env = _make_env(
            reward_config=HarvestRewardConfig(detach=DetachEnvelopeConfig(f_max_n=5.0)),
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
            assert set(rt["raw"]) == {"progress", "pullout", "wrist", "collateral", "slack"}
            assert set(rt["weighted"]) == {"progress", "pullout", "wrist", "collateral", "slack"}
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
                "terminated_edge",
                "detach_index",
            ):
                assert ep[key].shape == (2,), key
            assert info["target_pose"].shape == (2, 7)
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
