"""TrainConfig: nested frozen dataclasses, JSON round trip, PPO_RNN batch validation."""

from __future__ import annotations

import dataclasses
import json

import pytest

from apple_pick_gym.rl.config import EnvConfig, PPOConfig, TrainConfig


def test_json_round_trip(tmp_path):
    cfg = TrainConfig(env=EnvConfig(kind="surrogate", num_envs=16, max_episode_steps=64), timesteps=640)
    p = tmp_path / "cfg.json"
    cfg.save_json(p)
    assert TrainConfig.load_json(p) == cfg
    assert json.loads(p.read_text())["env"]["kind"] == "surrogate"


def test_overrides_merge_into_nested_sections(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"env": {"num_envs": 8}, "ppo": {"rollouts": 32, "mini_batches": 4}}))
    cfg = TrainConfig.load_json(p)
    assert cfg.env.num_envs == 8 and cfg.ppo.rollouts == 32
    assert cfg.env.max_episode_steps == EnvConfig().max_episode_steps  # untouched default


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"ppo": {"rolouts": 32}}))
    with pytest.raises(ValueError, match="rolouts"):
        TrainConfig.load_json(p)


def test_rollouts_must_be_a_multiple_of_the_bptt_length():
    with pytest.raises(ValueError, match="sequence_length"):
        TrainConfig(ppo=PPOConfig(rollouts=48), actor=dataclasses.replace(TrainConfig().actor, sequence_length=32)).validate()


def test_minibatches_must_hold_whole_sequences():
    cfg = TrainConfig(env=EnvConfig(num_envs=3), ppo=PPOConfig(rollouts=32, mini_batches=4))
    with pytest.raises(ValueError, match="mini_batches"):
        cfg.validate()


def test_actor_and_critic_share_the_bptt_length():
    cfg = TrainConfig(critic=dataclasses.replace(TrainConfig().critic, sequence_length=16))
    with pytest.raises(ValueError, match="sequence_length"):
        cfg.validate()


def test_defaults_follow_the_plan():
    cfg = TrainConfig()
    cfg.validate()
    assert cfg.ppo.rollouts == 64 and cfg.actor.sequence_length == 32
    assert cfg.ppo.discount_factor == 0.99 and cfg.ppo.gae_lambda == 0.95
    assert cfg.ppo.time_limit_bootstrap is False
    assert cfg.env.f_max_n == 20.0 and cfg.env.tau_max_nm == 0.05
    assert cfg.env.torque_mode == "total"  # the split envelope is opt-in until the maintainer decides
    assert cfg.env.progress_mode == "delta" and cfg.env.w_slack > 0.0
    assert cfg.env.sensor_dr is True  # F/T bias / noise / drift on for training
    assert cfg.env.max_target_pos_offset_m is not None  # VIC target leash on for training


@pytest.mark.parametrize("name", ["surrogate_smoke.json", "sim_wiring_cpu.json", "sim_wiring_gpu.json", "sim_smoke_gpu.json", "sim_train_gpu.json", "sim_train_gpu_ep250.json", "sim_debug_kl_gpu.json", "sim_train_gpu_ep250_unbounded.json", "sim_train_gpu_d8b_unbounded.json", "sim_train_gpu_d8b_tanh1.json", "sim_train_gpu_d8b_tanh15.json", "sim_train_gpu_v2c.json", "sim_train_gpu_v2d.json"])
def test_checked_in_configs_load_and_validate(name):
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "rl" / "configs" / name
    cfg = TrainConfig.load_json(path)
    cfg.validate()
    if cfg.env.kind == "sim":
        root = Path(__file__).resolve().parents[2]
        assert (root / cfg.env.world_set).exists()
        if cfg.env.snapshot is not None:
            assert (root / cfg.env.snapshot).exists()


def test_d8_target_speed_cap_matches_the_rig_and_reaches_the_env():
    # [D8] real rig pulls: TCP speed p90 0.03 m/s, per-run peak <= 0.21 m/s; angular <= 0.55 rad/s.
    # The cap (60 Hz): 2 mm/step = 0.12 m/s, 0.01 rad/step = 0.6 rad/s (was 1.2 m/s, 6 rad/s).
    from apple_pick_gym.rl.trainer import build_env

    # [D8b] maintainer: max target speed 0.4 m/s and 0.3 rad/s -> per 60 Hz step
    cfg = TrainConfig()
    assert cfg.env.linear_delta_m == pytest.approx(0.4 / 60) and cfg.env.angular_delta_rad == pytest.approx(0.3 / 60)
    env = build_env(dataclasses.replace(cfg.env, kind="surrogate", num_envs=4, device="cpu"), seed=0)
    assert env.action_bounds.linear_delta_m == pytest.approx(0.4 / 60)
    assert env.action_bounds.angular_delta_rad == pytest.approx(0.3 / 60)


def test_d10_kl_adaptive_lr_has_a_floor():
    # [D10] the KL-adaptive schedule cut the LR ~25x within 18 updates (4e-5) on GPU; skrl's floor is 1e-6
    cfg = TrainConfig()
    assert cfg.ppo.kl_adaptive_min_lr == 3e-5


def test_d10_min_lr_reaches_the_scheduler(tmp_path):
    from pathlib import Path

    from apple_pick_gym.rl.trainer import build_training

    cfg = dataclasses.replace(
        TrainConfig.load_json(Path(__file__).resolve().parent.parent / "rl" / "configs" / "surrogate_smoke.json"),
        run_dir=str(tmp_path / "run"),
    )
    _wrapper, agent = build_training(cfg)
    assert agent.scheduler.min_lr == 3e-5
    # [D10b] ceiling = the base LR: on GPU the schedule climbed to 5e-4 (> 3e-4 base) as the run degraded
    assert agent.scheduler.max_lr == cfg.ppo.learning_rate


def test_d9_training_reward_balance_ranks_a_clean_pick_above_scripted_pull_above_zero():
    # [D9] under D8 the old weights ranked zero (-37) above scripted_pull (-126): pulling paid
    # dense per-step pull-out / collateral costs far above the +10 bonus -> passive optimum.
    cfg = TrainConfig().env
    assert (cfg.w_progress, cfg.w_pullout, cfg.pullout_threshold_n) == (10.0, 0.5, 10.0)
    assert (cfg.w_collateral, cfg.success_bonus, cfg.failure_penalty, cfg.w_slack) == (0.02, 20.0, -40.0, 0.01)
    from apple_pick_gym.rl.trainer import build_env

    env = build_env(dataclasses.replace(cfg, kind="surrogate", num_envs=2, device="cpu"), seed=0)
    r = env._reward_cfg
    assert r.pullout_threshold_n == 10.0 and r.w_progress == 10.0 and r.success_bonus == 20.0
    # GPU eval_d8 per-episode sums (old weights) re-weighted: collateral N*steps = sum/0.1, slack steps = sum/0.01
    def ret(progress_u, coll_sum_old, slack_sum_old, success, safety=0.0, pullout_above_hinge=0.0):
        return (
            cfg.w_progress * progress_u
            - cfg.w_pullout * pullout_above_hinge
            - cfg.w_collateral * (coll_sum_old / 0.1)
            - cfg.w_slack * (slack_sum_old / 0.01)
            + cfg.success_bonus * success
            + cfg.failure_penalty * safety
        )

    zero = ret(0.0, 6.23, 5.00, 0.0)
    scripted = ret(0.68, 56.03, 0.31, 1.0, pullout_above_hinge=5.0)
    clean = ret(0.67, 22.34, 0.42, 0.90, safety=0.10)
    assert clean > scripted > zero, (clean, scripted, zero)


def test_d11_training_config_turns_on_the_wrist_force_soft_cap():
    cfg = TrainConfig().env
    assert (cfg.wrist_force_soft_cap_n, cfg.w_wrist) == (25.0, 0.5)
    from apple_pick_gym.rl.trainer import build_env

    env = build_env(dataclasses.replace(cfg, kind="surrogate", num_envs=2, device="cpu"), seed=0)
    assert env._reward_cfg.wrist_force_soft_cap_n == 25.0 and env._reward_cfg.w_wrist == 0.5


def test_d13_training_config_charges_peak_collateral_at_success():
    from apple_pick_gym.rl.trainer import build_env

    cfg = TrainConfig().env
    assert cfg.w_peak_collateral == 0.5
    env = build_env(dataclasses.replace(cfg, kind="surrogate", num_envs=2, device="cpu"), seed=0)
    assert env._reward_cfg.w_peak_collateral == 0.5


def test_unbounded_fallback_config_sets_mean_bound_none():
    from pathlib import Path

    cfg = TrainConfig.load_json(Path(__file__).resolve().parent.parent / "rl" / "configs" / "sim_train_gpu_ep250_unbounded.json")
    assert cfg.actor.mean_bound is None and cfg.ppo.kl_threshold == 0.05


def test_d17_v2_sim_config_carries_every_v2_decision():
    from pathlib import Path

    from apple_pick_gym.rl.config import TrainConfig

    cfg = TrainConfig.load_json(Path(__file__).resolve().parent.parent / "rl" / "configs" / "sim_train_gpu_v2.json")
    cfg.validate()
    e = cfg.env
    assert (e.detach_criterion, e.detach_scale_range) == ("bend_pull", (0.85, 1.15))
    assert (e.action_frame, e.k_lin_max, e.realized_k_scale_range) == ("tool", 500.0, (0.8, 1.2))
    # 4-term reward: progress, clipped collateral, slack, success; no pull-out / wrist / peak / safety
    assert (e.w_progress, e.w_collateral, e.collateral_clip_n, e.w_slack, e.success_bonus) == (10.0, 0.02, 50.0, 0.01, 60.0)
    assert (e.w_pullout, e.w_wrist, e.w_peak_collateral) == (0.0, 0.0, 0.0)
    assert e.safety_force_cap_n >= 1e9 and e.safety_torque_cap_nm >= 1e9
    assert (e.linear_delta_m, e.angular_delta_rad) == (0.4 / 60.0, 0.3 / 60.0)  # D8b cap unchanged
    assert cfg.ppo.discount_factor == 0.995 and cfg.ppo.time_limit_bootstrap is False
    assert "harvest_worlds_v3" in e.world_set


def test_d17_v2b_config_is_v2_plus_peak_collateral_and_the_rig_speed_cap():
    # [D17] v2 converged to a fast yank (0.96 m/s, 21% > 40 N): charge the episode's peak collateral at
    # success (1.0 x N) and cap the target speed at 0.12 m/s (D8). Everything else as v2.
    from pathlib import Path

    from apple_pick_gym.rl.config import TrainConfig

    d = Path(__file__).resolve().parent.parent / "rl" / "configs"
    v2 = TrainConfig.load_json(d / "sim_train_gpu_v2.json")
    v2b = TrainConfig.load_json(d / "sim_train_gpu_v2b.json")
    v2b.validate()
    assert v2b.env.w_peak_collateral == 1.0 and v2b.env.success_bonus == 60.0
    assert v2b.env.linear_delta_m == 0.002 and v2b.env.angular_delta_rad == v2.env.angular_delta_rad
    import dataclasses

    same = lambda c: dataclasses.replace(c.env, w_peak_collateral=0.0, linear_delta_m=0.0)
    assert same(v2b) == same(v2)
    assert v2b.ppo == v2.ppo and v2b.actor == v2.actor
    assert v2b.run_dir != v2.run_dir


def test_d17m_v2c_config_is_v2b_plus_the_no_advance_wall_and_deviation_collateral():
    # [D17m] v2b learned a lift-and-push pendulum detach (46 mm toward the tree, lifting the apple, which
    # the relu collateral scored free). v2c: the target may not advance past the grasp along the grip
    # axis, and collateral charges any deviation from the rest force vectors. Everything else as v2b.
    from pathlib import Path

    from apple_pick_gym.rl.trainer import build_env

    d = Path(__file__).resolve().parent.parent / "rl" / "configs"
    v2b = TrainConfig.load_json(d / "sim_train_gpu_v2b.json")
    v2c = TrainConfig.load_json(d / "sim_train_gpu_v2c.json")
    v2c.validate()
    assert (v2c.env.max_advance_m, v2c.env.collateral_mode) == (0.0, "deviation")
    assert (v2b.env.max_advance_m, v2b.env.collateral_mode) == (None, "relu_norm")  # v2b reproduces
    same = lambda c: dataclasses.replace(c.env, max_advance_m=None, collateral_mode="relu_norm")
    assert same(v2c) == same(v2b)
    assert v2c.ppo == v2b.ppo and v2c.actor == v2b.actor and v2c.run_dir != v2b.run_dir
    env = build_env(dataclasses.replace(v2c.env, kind="surrogate", num_envs=2, device="cpu"), seed=0)
    assert env.action_bounds.max_advance_m == 0.0 and env._reward_cfg.collateral_mode == "deviation"


def test_d17m_bad_collateral_mode_or_negative_wall_is_refused():
    from apple_pick_gym.rl.trainer import build_env

    for bad in ({"collateral_mode": "abs"}, {"max_advance_m": -0.01}):
        with pytest.raises(ValueError):
            build_env(dataclasses.replace(TrainConfig().env, kind="surrogate", num_envs=2, device="cpu", **bad), seed=0)


def test_d17n_v2d_config_is_v2c_plus_stem_only_bend_fit_range_hold_and_wide_scale():
    # [D17n] v2c detached with a fast sideways swing: theta_b was measured in the world frame (spur
    # rotation counted as stem bending), the fit was extrapolated to 38.5 deg, a 3-step (50 ms) spike
    # counted as a pick and F_d varied only +-15 %. v2d: theta_b against the spur-carried rest chord,
    # F_d flat past 30 deg, a 20-step streak, F_d scale U(0.8, 2.0). Everything else as v2c.
    from pathlib import Path

    from apple_pick_gym.rl.trainer import build_env

    d = Path(__file__).resolve().parent.parent / "rl" / "configs"
    v2c = TrainConfig.load_json(d / "sim_train_gpu_v2c.json")
    v2d = TrainConfig.load_json(d / "sim_train_gpu_v2d.json")
    v2d.validate()
    e = v2d.env
    assert (e.bend_frame, e.bend_pull_theta_max_deg, e.success_streak_steps) == ("spur", 30.0, 20)
    assert tuple(e.detach_scale_range) == (0.8, 2.0) and e.max_commanded_force_n == 40.0
    assert (v2c.env.bend_frame, v2c.env.bend_pull_theta_max_deg, v2c.env.max_commanded_force_n) == ("world", None, None)
    same = lambda c: dataclasses.replace(
        c.env,
        bend_frame="world",
        bend_pull_theta_max_deg=None,
        success_streak_steps=3,
        detach_scale_range=(0.85, 1.15),
        max_commanded_force_n=None,
    )
    assert same(v2d) == same(v2c)
    assert v2d.ppo == v2c.ppo and v2d.actor == v2c.actor and v2d.run_dir != v2c.run_dir
    env = build_env(dataclasses.replace(e, kind="surrogate", num_envs=2, device="cpu"), seed=0)
    det = env._reward_cfg.detach
    assert (det.bend_frame, det.bend_pull_theta_max_deg, tuple(det.detach_scale_range)) == ("spur", 30.0, (0.8, 2.0))
    assert env._episode_cfg.success_streak_steps == 20
    assert env.action_bounds.max_commanded_force_n == 40.0


def test_d17n_bad_bend_frame_or_theta_max_is_refused():
    from apple_pick_gym.rl.trainer import build_env

    base = dataclasses.replace(TrainConfig().env, kind="surrogate", num_envs=2, device="cpu", detach_criterion="bend_pull")
    for bad in ({"bend_frame": "apple"}, {"bend_pull_theta_max_deg": 60.0}):
        with pytest.raises(ValueError):
            build_env(dataclasses.replace(base, **bad), seed=0)


def test_d17n_k_ang_min_reaches_the_action_bounds():
    # [D17n run 7] a soft wrist (K_ang ~11 N*m/rad) let the gripper and apple swing as a pendulum on the stem
    from apple_pick_gym.rl.trainer import build_env

    assert TrainConfig().env.k_ang_min == 2.0
    env = build_env(dataclasses.replace(TrainConfig().env, kind="surrogate", num_envs=2, device="cpu", k_ang_min=20.0), seed=0)
    assert env.action_bounds.k_ang_min == 20.0
