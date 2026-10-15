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
        for key in ("success_rate", "safety_rate", "return_mean", "peak_detach_index_mean", "peak_collateral_n_mean", "peak_collateral_n_success_mean", "episodes"):
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


def test_d17_cli_offers_every_baseline():
    from apple_pick_gym.rl.baselines import BASELINES
    from apple_pick_gym.rl.eval_vic_harvest import build_parser

    for name in BASELINES:
        assert build_parser().parse_args(["--baseline", name, "--out", "x.json"]).baseline == name


def test_d17m_eval_overrides_the_wall_and_collateral_mode_and_reports_the_advance(tmp_path):
    # [D17m] score any policy (e.g. a v2b checkpoint) under v2c's wall and collateral
    args = ev.build_parser().parse_args(
        [*_ENV, "--baseline", "zero", "--out", str(tmp_path / "x.json"), "--max-advance-m", "0.0", "--collateral-mode", "deviation"]
    )
    cfg, wrapper, _policy, _label = ev.build_eval(args)
    try:
        assert (cfg.env.max_advance_m, cfg.env.collateral_mode) == (0.0, "deviation")
        assert wrapper.action_scaler.bounds.max_advance_m == 0.0
    finally:
        wrapper.close()
    args = ev.build_parser().parse_args([*_ENV, "--baseline", "zero", "--out", "x.json", "--max-advance-m", "none"])
    cfg, wrapper, _policy, _label = ev.build_eval(args)
    wrapper.close()
    assert cfg.env.max_advance_m is None
    out = tmp_path / "pull.json"
    assert ev.main([*_ENV, "--baseline", "scripted_pull", "--episodes", "1", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["peak_tcp_advance_mm_mean"] >= 0.0 and m["peak_tcp_advance_mm_max"] >= m["peak_tcp_advance_mm_mean"]


def test_d17n_eval_overrides_the_v2d_detach_rules_and_force_clamp(tmp_path):
    # [D17n] score any policy (e.g. a v2c checkpoint) under v2d's stem-only bend, fit-range hold,
    # 20-step streak, wide F_d scale and commanded-force clamp
    args = ev.build_parser().parse_args(
        [
            *_ENV, "--baseline", "zero", "--out", str(tmp_path / "x.json"),
            "--detach-criterion", "bend_pull", "--bend-frame", "spur", "--bend-theta-max-deg", "30",
            "--success-streak", "20", "--detach-scale", "0.8", "2.0", "--max-commanded-force-n", "40",
        ]
    )  # fmt: skip
    cfg, wrapper, _policy, _label = ev.build_eval(args)
    try:
        e = cfg.env
        assert (e.detach_criterion, e.bend_frame, e.bend_pull_theta_max_deg) == ("bend_pull", "spur", 30.0)
        assert (e.success_streak_steps, tuple(e.detach_scale_range), e.max_commanded_force_n) == (20, (0.8, 2.0), 40.0)
        assert wrapper.action_scaler.bounds.max_commanded_force_n == 40.0
    finally:
        wrapper.close()
    args = ev.build_parser().parse_args([*_ENV, "--baseline", "zero", "--out", "x.json", "--max-commanded-force-n", "none"])
    cfg, wrapper, _policy, _label = ev.build_eval(args)
    wrapper.close()
    assert cfg.env.max_commanded_force_n is None


def test_d17q_eval_writes_per_env_records_and_success_only_peaks(tmp_path):
    import numpy as np

    out = tmp_path / "pull.json"
    assert ev.main([*_ENV, "--baseline", "scripted_pull", "--episodes", "2", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert "per_env" not in m
    rec = np.load(tmp_path / "pull.per_env.npz")
    for ep in (0, 1):
        for k in ("success", "invalid", "steps_to_success", "peak_wrist_force_n", "peak_target_force_n"):
            assert rec[f"ep{ep}/{k}"].shape == (8,), k
    won = np.concatenate([rec[f"ep{e}/success"] & ~rec[f"ep{e}/invalid"] for e in (0, 1)])
    assert m["n_success"] == int(won.sum()) > 0
    wrist = np.concatenate([rec[f"ep{e}/peak_wrist_force_n"] for e in (0, 1)])[won]
    assert m["peak_wrist_force_n_success_mean"] == pytest.approx(float(wrist.mean()), rel=1e-5)
