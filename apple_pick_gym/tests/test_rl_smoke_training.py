"""CPU smoke: skrl recurrent PPO learns the surrogate harvest task (reward and success rise).

This is the end-to-end infrastructure check -- env contract, DR, critic state, LSTM
models, wrapper auto-reset / terminated semantics, trainer, checkpoints -- on the
analytic surrogate (the real env needs CUDA). ~1 min on 2 CPU threads. Uses the
checked-in ``apple_pick_gym/rl/configs/surrogate_smoke.json``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_CFG = Path(__file__).resolve().parent.parent / "rl" / "configs" / "surrogate_smoke.json"


def test_recurrent_ppo_learns_the_surrogate_harvest_task(tmp_path):
    from apple_pick_gym.rl.config import TrainConfig
    from apple_pick_gym.rl.trainer import run_training

    cfg = dataclasses.replace(TrainConfig.load_json(_CFG), run_dir=str(tmp_path / "run"))
    result = run_training(cfg)
    ep = result.episodes
    assert len(ep) == cfg.timesteps // cfg.env.max_episode_steps == 10
    ret = np.array([e["Episode / return (mean)"] for e in ep])
    succ = np.array([e["Episode / success rate"] for e in ep])
    safety = np.array([e["Episode / safety rate"] for e in ep])
    # learning: return and success both rise clearly from the first episodes to the last
    assert ret[-3:].mean() > ret[:2].mean() + 5.0, ret
    assert succ[-3:].mean() >= 0.6 and succ[-3:].mean() > succ[:2].mean() + 0.4, succ
    assert safety.max() <= 0.05, safety
    # optimization stayed healthy
    updates = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    updates = [u for u in updates if u["kind"] == "update"]
    assert len(updates) == cfg.timesteps // cfg.ppo.rollouts
    for u in updates:
        assert math.isfinite(u["Loss / Policy loss"]) and math.isfinite(u["Loss / Value loss"])
        assert u["Policy / Standard deviation"] > 0.05  # no premature collapse
    assert result.last_checkpoint is not None and (result.last_checkpoint / "meta.json").exists()
