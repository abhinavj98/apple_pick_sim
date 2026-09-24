"""Baselines + eval CLI on the surrogate env (CPU)."""

from __future__ import annotations

import json

import pytest
import torch

from apple_pick_gym.rl import eval_vic_harvest as ev
from apple_pick_gym.rl import train_vic_harvest as train_cli
from apple_pick_gym.rl.baselines import RandomPolicy, ScriptedPullPolicy, ZeroPolicy
from apple_pick_gym.rl.config import EnvConfig
from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper
from apple_pick_gym.rl.trainer import build_env

_ENV = ["--env", "surrogate", "--num-envs", "8", "--max-episode-steps", "120", "--device", "cpu"]


def _wrapper():
    return HarvestSkrlWrapper(build_env(EnvConfig(kind="surrogate", num_envs=8, max_episode_steps=120, device="cpu")))


def test_baselines_emit_policy_space_actions():
    w = _wrapper()
    obs, _ = w.reset()
    for pol in (ZeroPolicy(), RandomPolicy(seed=0), ScriptedPullPolicy()):
        pol.reset(w)
        a = pol.act(w, obs, w.state())
        assert a.shape == (8, 13) and float(a.abs().max()) <= 1.0 + 1e-6


def test_scripted_pull_moves_along_the_weld_axis():
    w = _wrapper()
    w.reset()
    pol = ScriptedPullPolicy(rate_m_per_step=0.002)
    pol.reset(w)
    env_units = w.action_scaler.to_env(pol.act(w, None, None))
    weld = w._env.plant_geometry()["weld_direction"]
    torch.testing.assert_close(env_units[:, :3], weld * 0.002, atol=1e-6, rtol=1e-4)


def test_eval_baselines_write_metrics_and_scripted_pull_beats_zero(tmp_path):
    zero = tmp_path / "zero.json"
    pull = tmp_path / "pull.json"
    assert ev.main([*_ENV, "--baseline", "zero", "--episodes", "1", "--out", str(zero)]) == 0
    assert ev.main([*_ENV, "--baseline", "scripted_pull", "--episodes", "1", "--out", str(pull)]) == 0
    z, p = json.loads(zero.read_text()), json.loads(pull.read_text())
    for m in (z, p):
        for key in ("success_rate", "safety_rate", "return_mean", "peak_detach_index_mean", "peak_collateral_n_mean", "episodes"):
            assert key in m, key
    assert p["success_rate"] > z["success_rate"]
    assert z["policy"] == "baseline:zero"


def test_eval_checkpoint_is_deterministic(tmp_path):
    run = str(tmp_path / "run")
    assert train_cli.main([
        *_ENV[:-2], "--device", "cpu", "--max-episode-steps", "16", "--rollouts", "16", "--mini-batches", "2",
        "--learning-epochs", "1", "--sequence-length", "8", "--hidden", "16", "--timesteps", "16", "--run-dir", run,
    ]) == 0
    ckpt = sorted((tmp_path / "run" / "checkpoints").iterdir())[-1]
    outs = []
    for i in range(2):
        out = tmp_path / f"m{i}.json"
        assert ev.main(["--checkpoint", str(ckpt), "--episodes", "1", "--max-episode-steps", "16", "--out", str(out)]) == 0
        outs.append(json.loads(out.read_text()))
    assert outs[0]["return_mean"] == pytest.approx(outs[1]["return_mean"])
    assert outs[0]["policy"].startswith("checkpoint:")
