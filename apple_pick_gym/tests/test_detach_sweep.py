"""detach_sweep: record raw junction wrenches with detachment disabled, replay detach rules offline."""

from __future__ import annotations

import json

import numpy as np
import torch

from apple_pick_gym.rl import detach_sweep as ds

_ENV = ["--env", "surrogate", "--num-envs", "16", "--max-episode-steps", "120", "--device", "cpu"]


def test_offline_rule_replay_first_crossing_with_streak_and_filter():
    # one env, index series crossing 1 for 2 steps, then 4 steps
    idx = torch.tensor([[0.2], [1.2], [1.1], [0.5], [1.5], [1.5], [1.5], [1.5]])
    assert ds.first_detach(idx, streak=2).tolist() == [2]
    assert ds.first_detach(idx, streak=3).tolist() == [6]
    assert ds.first_detach(idx, streak=5).tolist() == [-1]


def test_ema_filter_matches_the_sensor_alpha():
    x = torch.zeros(10, 1, 6)
    x[3:] = 1.0
    y = ds.ema(x, alpha=0.5)
    assert float(y[3, 0, 0]) == 0.5 and float(y[4, 0, 0]) == 0.75
    torch.testing.assert_close(ds.ema(x, alpha=1.0), x)


def test_sweep_cli_reports_every_rule_per_policy(tmp_path):
    out = tmp_path / "sweep.json"
    assert ds.main([*_ENV, "--policies", "zero", "scripted_pull", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    rules = rep["policies"]["scripted_pull"]["rules"]
    assert "total|raw|streak3" in rules and "force_only|ema3hz|streak10" in rules
    assert "total_tau0.2|raw|streak3" in rules
    assert rep["policies"]["zero"]["rules"]["force_only|raw|streak3"]["success"] == 0.0
    assert rules["force_only|raw|streak3"]["success"] > 0.5
    # detachment was disabled while recording: nobody froze
    assert rep["policies"]["scripted_pull"]["recorded_frozen_fraction"] == 0.0
    noise = rep["policies"]["zero"]["noise"]
    for k in ("tau_p99", "dtau_p99", "torsion_p99", "bending_p99", "force_p99"):
        assert np.isfinite(noise[k]), k
