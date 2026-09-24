"""wandb gets the training scalars explicitly (not via TensorBoard sync), keyed on ``timestep``."""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

_CFG = Path(__file__).resolve().parent.parent / "rl" / "configs" / "surrogate_smoke.json"


class FakeWandb(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.run = None
        self.init_kwargs = None
        self.defined: list[tuple] = []
        self.logged: list[dict] = []
        self.util = types.SimpleNamespace()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = types.SimpleNamespace(id=kwargs.get("id"))
        return self.run

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))

    def log(self, data, **kwargs):
        assert not kwargs, "explicit step= would fight the timestep x-axis"
        self.logged.append(dict(data))

    def Video(self, path, **kwargs):  # noqa: N802
        return ("video", str(path))

    def finish(self):
        self.run = None


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_sink_defines_timestep_axis_once_and_logs_rows(fake_wandb):
    from apple_pick_gym.rl.trainer import WandbSink

    sink = WandbSink()
    sink.log(64, {"Loss / Policy loss": 0.1})  # no run yet: dropped, no crash
    assert fake_wandb.logged == []
    fake_wandb.init(project="p")
    sink.log(64, {"Loss / Policy loss": 0.1})
    sink.log(128, {"Loss / Policy loss": 0.2})
    assert fake_wandb.defined == [("timestep", {}), ("*", {"step_metric": "timestep"})]
    assert fake_wandb.logged == [
        {"Loss / Policy loss": 0.1, "timestep": 64},
        {"Loss / Policy loss": 0.2, "timestep": 128},
    ]


def test_backfill_logs_update_rows_up_to_the_resume_point(fake_wandb, tmp_path):
    from apple_pick_gym.rl.trainer import WandbSink, backfill_wandb

    rows = [
        {"kind": "update", "timestep": 64, "Loss / Policy loss": 1.0, "Episode / success rate": 0.5},
        {"kind": "episode", "timestep": 100, "Episode / success rate": 0.5},
        {"kind": "update", "timestep": 128, "Loss / Policy loss": 2.0},
        {"kind": "update", "timestep": 192, "Loss / Policy loss": 3.0},  # after the checkpoint: re-run
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake_wandb.init(project="p")
    n = backfill_wandb(path, upto_timestep=128, sink=WandbSink())
    assert n == 2
    assert fake_wandb.logged == [
        {"timestep": 64, "Loss / Policy loss": 1.0, "Episode / success rate": 0.5},
        {"timestep": 128, "Loss / Policy loss": 2.0},
    ]


@pytest.mark.slow
def test_training_logs_scalars_to_wandb_without_tensorboard_sync(fake_wandb, tmp_path):
    from apple_pick_gym.rl.config import TrainConfig
    from apple_pick_gym.rl.trainer import run_training

    cfg = dataclasses.replace(TrainConfig.load_json(_CFG), run_dir=str(tmp_path / "run"), wandb=True)
    run_training(cfg, max_updates=2)
    assert fake_wandb.init_kwargs["sync_tensorboard"] is False
    rows = [r for r in fake_wandb.logged if "Loss / Policy loss" in r]
    assert [r["timestep"] for r in rows] == [cfg.ppo.rollouts, 2 * cfg.ppo.rollouts]
    assert all("Policy / KL (mean)" in r or "Reward / Total reward (mean)" in r for r in rows)
