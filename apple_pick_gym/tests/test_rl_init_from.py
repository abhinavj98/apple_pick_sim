"""--init-from: start a NEW run (timestep 0, own run dir / wandb id) from another run's checkpoint weights."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_CFG = Path(__file__).resolve().parent.parent / "rl" / "configs" / "surrogate_smoke.json"


def _policy_params(ckpt: Path) -> dict[str, torch.Tensor]:
    return torch.load(ckpt / "agent.pt", map_location="cpu", weights_only=False)["policy"]


def test_init_from_starts_a_fresh_run_from_checkpoint_weights(tmp_path, monkeypatch):
    import apple_pick_gym.rl.trainer as trainer
    from apple_pick_gym.rl.config import TrainConfig

    base = TrainConfig.load_json(_CFG)
    src = trainer.run_training(dataclasses.replace(base, run_dir=str(tmp_path / "src")), max_updates=1)
    src_ckpt = src.last_checkpoint

    loaded = {}
    real_load = trainer.load_checkpoint

    def spy(path, agent, wrapper, cfg):
        meta = real_load(path, agent, wrapper, cfg)
        loaded["path"] = Path(path)
        loaded["policy"] = {k: v.detach().clone().cpu() for k, v in agent.policy.state_dict().items()}
        return meta

    monkeypatch.setattr(trainer, "load_checkpoint", spy)
    dst_cfg = dataclasses.replace(base, run_dir=str(tmp_path / "dst"), seed=3)
    res = trainer.run_training(dst_cfg, init_from=str(src_ckpt), max_updates=1)
    assert loaded["path"] == src_ckpt
    for k, v in _policy_params(src_ckpt).items():
        assert torch.equal(loaded["policy"][k], v.cpu()), k
    assert res.start_timestep == 0 and res.updates == 1
    meta = json.loads((res.last_checkpoint / "meta.json").read_text())
    assert meta["timestep"] == base.ppo.rollouts and meta["init_from"] == str(src_ckpt)
    rows = [json.loads(l) for l in (tmp_path / "dst" / "metrics.jsonl").read_text().splitlines()]
    assert rows[0] == {"kind": "init_from", "checkpoint": str(src_ckpt), "source_timestep": src.timestep}


def test_init_from_and_resume_are_exclusive(tmp_path):
    from apple_pick_gym.rl.config import TrainConfig
    from apple_pick_gym.rl.trainer import run_training

    cfg = dataclasses.replace(TrainConfig.load_json(_CFG), run_dir=str(tmp_path / "r"))
    with pytest.raises(ValueError, match="init_from"):
        run_training(cfg, resume="latest", init_from="x")
