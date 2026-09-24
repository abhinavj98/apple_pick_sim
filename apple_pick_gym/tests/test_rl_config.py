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


@pytest.mark.parametrize("name", ["surrogate_smoke.json", "sim_wiring_cpu.json", "sim_wiring_gpu.json", "sim_smoke_gpu.json"])
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
