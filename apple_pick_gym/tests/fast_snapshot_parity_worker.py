"""[D17q] Worker for test_vic_harvest_fast_snapshot_build: one env build per process (rebuilding sims in one
process is not safe).

  make <world_set.jsonl> <out_snapshot.npz>               full build + hold settle, save the snapshot
  dump <world_set.jsonl> <snapshot.npz> <on|off> <out.npz>  build with that snapshot, dump model arrays,
                                                           the restored state and a fixed-action rollout
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[1] / "rl/configs/sim_train_gpu_v2g_true.json"
STEPS = 10  # GPU contact atomics make rollouts split run to run (seen from step 12, either build path)


def _env(world_set: str, snapshot: str | None, fast: bool):
    from apple_pick_gym.rl.config import TrainConfig
    from apple_pick_gym.rl.trainer import build_env

    cfg = TrainConfig.load_json(CONFIG)
    n = sum(1 for line in open(world_set) if line.strip())
    env_cfg = dataclasses.replace(cfg.env, world_set=world_set, snapshot=snapshot, num_envs=n, fast_snapshot_build=fast)
    return build_env(env_cfg, seed=0)


def _wp_arrays(obj, prefix: str) -> dict[str, np.ndarray]:
    import warp as wp

    out = {}
    for name, val in vars(obj).items():
        if isinstance(val, wp.array) and val.size > 0:
            out[f"{prefix}.{name}"] = val.numpy()
    return out


def _flat(prefix: str, d: dict) -> dict[str, np.ndarray]:
    import torch

    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[f"{prefix}.{k}"] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            out.update(_flat(f"{prefix}.{k}", v))
    return out


def make(world_set: str, out: str) -> None:
    env = _env(world_set, None, False)
    env.reset()
    env.save_world_snapshot(out)
    env.close()


def dump(world_set: str, snapshot: str, fast: str, out: str) -> None:
    import torch

    from apple_pick_sim.coupled_fruiting.episode_state_snapshot import EpisodeStateSnapshot

    t0 = time.time()
    env = _env(world_set, snapshot, fast == "on")
    rec = {"build_s": np.array(time.time() - t0)}
    scene = env._sim.scene
    rec.update(_wp_arrays(scene.cable.model, "cable_model"))
    rec.update(_wp_arrays(scene.robot_model, "robot_model"))
    snap = EpisodeStateSnapshot.capture(env._sim)
    for f in dataclasses.fields(snap):
        v = getattr(snap, f.name)
        if v is not None:
            rec[f"snapshot.{f.name}"] = v.numpy()
    rec["env.invalid"] = env._invalid_env_mask.cpu().numpy()
    rec["env.hold_target_pose"] = env._hold_target_pose.cpu().numpy()
    rec["gravity"] = np.asarray(scene.cable.model.gravity.numpy())

    obs, info = env.reset()
    rec.update(_flat("reset.obs", obs))
    rec.update(_flat("reset.info", info))
    lo = torch.as_tensor(env.action_space.low, device=env.device)
    hi = torch.as_tensor(env.action_space.high, device=env.device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    for t in range(STEPS):
        u = torch.rand((env.num_envs, lo.numel()), generator=gen).to(env.device)
        u[:, :6] = 0.5 + 0.25 * (u[:, :6] - 0.5)  # gentle pose deltas so the rollout stays in contact
        obs, r, te, tr, info = env.step(lo + u * (hi - lo))
        rec.update(_flat(f"step{t:02d}.obs", obs))
        rec.update(_flat(f"step{t:02d}.info", info))
        rec[f"step{t:02d}.reward"] = r.detach().cpu().numpy()
        rec[f"step{t:02d}.terminated"] = te.detach().cpu().numpy()
    env.close()
    np.savez(out, **rec)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"make": make, "dump": dump}[cmd](*rest)
