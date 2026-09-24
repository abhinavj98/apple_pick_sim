"""End-to-end wiring of the RL stack on the *real* harvest env (CPU, one build).

Trains 2 PPO updates over 2 synchronized episodes of ``ApplePickVicHarvestEnv`` (N=2
screened v1 worlds, hold settle) through the training CLI, then checks the metrics,
the checkpoint sidecar and the critic layout against the real junction set. The arm does
not integrate on CPU (see the env's warning), so this proves plumbing, not learning --
the learning check is ``test_rl_smoke_training.py`` (surrogate) and the GPU commands in
``docs/handbook-rl-policy.md``. Run on its own: one sim build per process.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_CFG = Path(__file__).resolve().parent.parent / "rl" / "configs" / "sim_wiring_cpu.json"


def test_train_cli_runs_end_to_end_on_the_real_env(tmp_path):
    from apple_pick_gym.rl import train_vic_harvest as cli

    run = tmp_path / "run"
    rc = cli.main(["--config", str(_CFG), "--run-dir", str(run), "--allow-cpu-sim"])
    assert rc == 0
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in rows if r["kind"] == "update"]
    episodes = [r for r in rows if r["kind"] == "episode"]
    assert len(updates) == 2 and len(episodes) == 2
    for u in updates:
        assert math.isfinite(u["Loss / Policy loss"]) and math.isfinite(u["Loss / Value loss"])
        assert u["Step / nonfinite envs"] == 0.0
    ckpt = sorted((run / "checkpoints").iterdir())[-1]
    meta = json.loads((ckpt / "meta.json").read_text())
    names = [row[0] for row in meta["critic_layout"]]
    for j in ("primary_spur", "spur_stem", "stem_apple", "primary_support_left", "primary_support_right"):
        assert f"woody_part_force/{j}" in names
    assert names[-2:] == ["frozen", "invalid"]
    assert meta["timestep"] == 32
