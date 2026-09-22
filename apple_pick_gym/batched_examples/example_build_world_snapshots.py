"""Build one shard of a screened world set and save its settled episode snapshot.

Training builds at most ~512 envs per process, so a large world set is split into
shards. Each shard is built once here (full settle + hold settle), checked with a
short zero-action hold, and saved as:

- ``<out-dir>/shard_<k>.jsonl``            -- the shard's WorldSpecs, in env order
- ``<out-dir>/shard_<k>_snapshot.npz``     -- the settled baseline ``reset()`` restores

A training process then builds the same worlds and restores the snapshot
(``ApplePickVicHarvestEnv(episode_snapshot_path=..., **world_specs_to_env_kwargs(specs))``),
so every run starts from this vetted settled state rather than its own settle
outcome. Worlds that misbehave in this build are marked invalid in the snapshot.

One shard per process::

    uv run python apple_pick_gym/batched_examples/example_build_world_snapshots.py \\
        --world-set apple_pick_gym/world_sets/harvest_worlds_v2.jsonl \\
        --shard-size 500 --shard-index 0 --out-dir ~/.cache/apple_pick_sim/world_sets/harvest_worlds_v2
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-set", required=True, help="Screened JSONL; only passed worlds are used.")
    p.add_argument("--shard-size", type=int, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--order-seed", type=int, default=0, help="Shuffle before sharding (mixes screening seeds).")
    p.add_argument("--device", default=None)
    p.add_argument("--check-steps", type=int, default=120, help="Zero-action steps in the post-build check.")
    p.add_argument("--max-check-wrist-n", type=float, default=20.0)
    return p


def shard_specs(specs: Sequence, *, shard_size: int, shard_index: int, order_seed: int) -> list:
    order = np.random.default_rng(order_seed).permutation(len(specs))
    idx = order[shard_index * shard_size : (shard_index + 1) * shard_size]
    return [specs[i] for i in idx]


def main(argv: Sequence[str] | None = None) -> None:
    import torch

    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.world_set import load_world_set, save_world_set, world_specs_to_env_kwargs

    args = make_parser().parse_args(argv)
    specs = shard_specs(
        load_world_set(args.world_set, passed_only=True),
        shard_size=args.shard_size,
        shard_index=args.shard_index,
        order_seed=args.order_seed,
    )
    if not specs:
        raise SystemExit(f"shard {args.shard_index} is empty")
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    env = ApplePickVicHarvestEnv(device=args.device, **world_specs_to_env_kwargs(specs))
    try:
        t_build = time.time() - t0
        obs, info = env.reset()
        peak = torch.zeros(env.num_envs, device=env.device)
        zero = torch.zeros((env.num_envs, 13), dtype=torch.float32, device=env.device)
        for _ in range(args.check_steps):
            obs, _r, _te, _tr, info = env.step(zero)
            wrist = torch.linalg.norm(info["ft_wrist"][:, :3], dim=-1)
            peak = torch.maximum(peak, torch.nan_to_num(wrist, nan=float("inf")))
        failed = peak > args.max_check_wrist_n
        env._invalid_env_mask = env._invalid_env_mask | failed
        env.reset()  # back to the captured baseline before saving (the snapshot itself is untouched)
        stem = f"shard_{args.shard_index:02d}"
        save_world_set(out / f"{stem}.jsonl", specs)
        env.save_world_snapshot(out / f"{stem}_snapshot.npz")
        n_bad = int(env._invalid_env_mask.sum())
    finally:
        env.close()
    print(
        f"{stem}: {len(specs) - n_bad}/{len(specs)} valid, build {t_build:.0f} s, "
        f"total {time.time() - t0:.0f} s -> {out}"
    )


if __name__ == "__main__":
    main()
