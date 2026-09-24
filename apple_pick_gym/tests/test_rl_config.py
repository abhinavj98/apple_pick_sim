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


@pytest.mark.parametrize("name", ["surrogate_smoke.json", "sim_wiring_cpu.json", "sim_wiring_gpu.json", "sim_smoke_gpu.json", "sim_train_gpu.json", "sim_train_gpu_ep250.json"])
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

    cfg = TrainConfig()
    assert cfg.env.linear_delta_m == 0.002 and cfg.env.angular_delta_rad == 0.01
    env = build_env(dataclasses.replace(cfg.env, kind="surrogate", num_envs=4, device="cpu"), seed=0)
    assert env.action_bounds.linear_delta_m == 0.002 and env.action_bounds.angular_delta_rad == 0.01


def test_d10_kl_adaptive_lr_has_a_floor():
    # [D10] the KL-adaptive schedule cut the LR ~25x within 18 updates (4e-5) on GPU; skrl's floor is 1e-6
    cfg = TrainConfig()
    assert cfg.ppo.kl_adaptive_min_lr == 1e-4


def test_d10_min_lr_reaches_the_scheduler(tmp_path):
    from pathlib import Path

    from apple_pick_gym.rl.trainer import build_training

    cfg = dataclasses.replace(
        TrainConfig.load_json(Path(__file__).resolve().parent.parent / "rl" / "configs" / "surrogate_smoke.json"),
        run_dir=str(tmp_path / "run"),
    )
    _wrapper, agent = build_training(cfg)
    assert agent.scheduler.min_lr == 1e-4


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
