"""Trainer + checkpointing on the surrogate env (CPU, fast)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from apple_pick_gym.rl.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from apple_pick_gym.rl.config import EnvConfig, PPOConfig, TrainConfig
from apple_pick_gym.rl.models import RecurrentNetConfig
from apple_pick_gym.rl.trainer import build_training, run_training

_NET = RecurrentNetConfig(pre_mlp=(32,), lstm_hidden=16, post_mlp=(32,), sequence_length=8)


def _cfg(tmp_path, **kw):
    base = TrainConfig(
        env=EnvConfig(kind="surrogate", num_envs=8, max_episode_steps=16, device="cpu"),
        ppo=PPOConfig(rollouts=16, mini_batches=2, learning_epochs=2),
        actor=_NET,
        critic=_NET,
        timesteps=48,
        checkpoint_every_updates=1,
        run_dir=str(tmp_path / "run"),
    )
    return dataclasses.replace(base, **kw)


def test_training_runs_logs_and_checkpoints(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_training(cfg)
    assert result.timestep == cfg.timesteps
    assert result.updates == cfg.timesteps // cfg.ppo.rollouts == 3
    hist = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    updates = [h for h in hist if h["kind"] == "update"]
    assert len(updates) == 3
    for h in updates:
        for key in ("Loss / Policy loss", "Loss / Value loss", "Policy / Standard deviation"):
            assert math.isfinite(h[key]), key
    episodes = [h for h in hist if h["kind"] == "episode"]
    assert len(episodes) == cfg.timesteps // cfg.env.max_episode_steps
    assert (tmp_path / "run" / "config.json").exists()
    ckpts = sorted((tmp_path / "run" / "checkpoints").iterdir())
    assert [p.name for p in ckpts] == ["ckpt_000000016", "ckpt_000000032", "ckpt_000000048"]
    assert latest_checkpoint(tmp_path / "run") == ckpts[-1]
    assert list((tmp_path / "run").glob("events.out.tfevents.*")) or list((tmp_path / "run").rglob("events.out.tfevents.*"))


def test_checkpoint_round_trip_restores_weights_optimizer_and_scalers(tmp_path):
    cfg = _cfg(tmp_path)
    run_training(cfg)
    ckpt = latest_checkpoint(tmp_path / "run")
    wrapper, agent = build_training(cfg)
    before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    meta = load_checkpoint(ckpt, agent, wrapper, cfg)
    assert meta["timestep"] == 48 and meta["updates"] == 3
    after = agent.policy.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    saved = torch.load(ckpt / "agent.pt", weights_only=False)
    for k, v in saved["policy"].items():
        torch.testing.assert_close(after[k], v)
    torch.testing.assert_close(agent._state_preprocessor.running_mean, saved["state_preprocessor"]["running_mean"])
    assert agent.optimizer.state_dict()["state"]  # Adam moments restored


def test_checkpoint_refuses_a_mismatched_layout_or_action_bounds(tmp_path):
    cfg = _cfg(tmp_path)
    run_training(cfg)
    ckpt = latest_checkpoint(tmp_path / "run")
    meta = json.loads((ckpt / "meta.json").read_text())
    meta["critic_layout"][-1][0] = "renamed_field"
    (ckpt / "meta.json").write_text(json.dumps(meta))
    wrapper, agent = build_training(cfg)
    with pytest.raises(ValueError, match="critic_layout"):
        load_checkpoint(ckpt, agent, wrapper, cfg)

    meta["critic_layout"][-1][0] = "invalid"
    meta["action_bounds"]["k_lin_max"] = 999.0
    (ckpt / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="action_bounds"):
        load_checkpoint(ckpt, agent, wrapper, cfg)


def test_resume_continues_from_the_saved_timestep(tmp_path):
    cfg = _cfg(tmp_path, timesteps=32)
    run_training(cfg)
    cfg2 = dataclasses.replace(cfg, timesteps=64)
    result = run_training(cfg2, resume="latest")
    assert result.start_timestep == 32 and result.timestep == 64 and result.updates == 4
    names = [p.name for p in sorted((tmp_path / "run" / "checkpoints").iterdir())]
    assert names[-1] == "ckpt_000000064"
    hist = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert max(h["timestep"] for h in hist) == 64


def test_save_checkpoint_writes_meta_sidecar(tmp_path):
    cfg = _cfg(tmp_path)
    wrapper, agent = build_training(cfg)
    path = save_checkpoint(tmp_path / "ck", agent, wrapper, cfg, timestep=5, updates=0, wandb_run_id=None)
    meta = json.loads((path / "meta.json").read_text())
    for key in ("actor_layout", "critic_layout", "action_bounds", "rnn", "timestep", "git_sha", "config"):
        assert key in meta, key
    assert meta["rnn"]["policy"]["sequence_length"] == 8


def test_resume_reseeds_episode_rng_instead_of_replaying_the_start(tmp_path):
    """A resumed segment must not replay the arm-DR / F/T sensor draws of timestep 0."""
    from apple_pick_gym.rl.trainer import reseed_for_resume

    cfg = _cfg(tmp_path)
    w0, _ = build_training(cfg)
    w0.reset()
    fresh = w0._env.privileged_fields()["arm_friction"].clone()
    fresh_bias = w0._env._ft_sensor._bias.clone()

    w1, _ = build_training(cfg)
    reseed_for_resume(w1, cfg, start_timestep=32)
    w1.reset()
    resumed = w1._env.privileged_fields()["arm_friction"]
    assert not torch.allclose(resumed, fresh)
    assert not torch.equal(w1._env._ft_sensor._bias, fresh_bias)

    w2, _ = build_training(cfg)  # same (seed, timestep) -> same draws: resumes are reproducible
    reseed_for_resume(w2, cfg, start_timestep=32)
    w2.reset()
    torch.testing.assert_close(w2._env.privileged_fields()["arm_friction"], resumed)
