"""Checkpoints: skrl's agent modules plus a ``meta.json`` sidecar that guards compatibility.

``agent.pt`` is ``agent.save()`` -- policy, value, optimizer and the three running
standard scalers (observations, states, values). ``meta.json`` records what the weights
mean: the actor and critic ``ObsLayout`` tables, the action bounds the ``[-1, 1]`` policy
box maps onto, both towers' RNN spec, the global timestep / update count, the wandb run id
and the git SHA. :func:`load_checkpoint` refuses a checkpoint whose layouts, bounds or RNN
spec differ from the current build -- a silently misaligned observation vector would
still "load" and then act nonsensically.

Directory layout: ``<run_dir>/checkpoints/ckpt_<timestep:09d>/{agent.pt, meta.json}``.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any

from apple_pick_gym.batched_envs.harvest_obs import actor_obs_layout

_META_SCHEMA = "vic_harvest_rl_checkpoint_v1"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        )
        return out.stdout.strip()
    except Exception:
        return None


def _layout_rows(layout) -> list[list[Any]]:
    return [[e.name, e.start, e.width] for e in layout.entries]


def _rnn_spec(model) -> dict[str, Any]:
    spec = model.get_specification()["rnn"]
    return {"sequence_length": spec["sequence_length"], "sizes": [[s[0], s[2]] for s in spec["sizes"]]}


def checkpoint_meta(
    agent, wrapper, cfg, *, timestep: int, updates: int, wandb_run_id: str | None, init_from: str | None = None
) -> dict[str, Any]:
    return {
        "schema": _META_SCHEMA,
        "timestep": int(timestep),
        "updates": int(updates),
        "actor_layout": _layout_rows(actor_obs_layout()),
        "critic_layout": _layout_rows(wrapper.critic_layout),
        "action_bounds": dataclasses.asdict(wrapper.action_scaler.bounds),
        "rnn": {"policy": _rnn_spec(agent.policy), "value": _rnn_spec(agent.value)},
        "wandb_run_id": wandb_run_id,
        "init_from": init_from,
        "git_sha": _git_sha(),
        "config": cfg.to_dict(),
    }


def save_checkpoint(
    directory, agent, wrapper, cfg, *, timestep: int, updates: int, wandb_run_id: str | None, init_from: str | None = None
) -> Path:
    path = Path(directory) / f"ckpt_{int(timestep):09d}"
    path.mkdir(parents=True, exist_ok=True)
    agent.save(str(path / "agent.pt"))
    meta = checkpoint_meta(
        agent, wrapper, cfg, timestep=timestep, updates=updates, wandb_run_id=wandb_run_id, init_from=init_from
    )
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def latest_checkpoint(run_dir) -> Path | None:
    root = Path(run_dir) / "checkpoints"
    if not root.is_dir():
        return None
    ckpts = sorted(p for p in root.iterdir() if p.name.startswith("ckpt_") and (p / "meta.json").exists())
    return ckpts[-1] if ckpts else None


def load_checkpoint(path, agent, wrapper, cfg) -> dict[str, Any]:
    """Load ``path`` into ``agent`` after checking it matches this build; returns its meta."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("schema") != _META_SCHEMA:
        raise ValueError(f"{path}: unsupported checkpoint schema {meta.get('schema')!r}")
    expected = checkpoint_meta(agent, wrapper, cfg, timestep=0, updates=0, wandb_run_id=None)
    for key in ("actor_layout", "critic_layout", "action_bounds", "rnn"):
        if meta[key] != expected[key]:
            raise ValueError(f"{path}: checkpoint {key} does not match this build\n saved:   {meta[key]}\n current: {expected[key]}")
    agent.load(str(path / "agent.pt"))
    return meta
