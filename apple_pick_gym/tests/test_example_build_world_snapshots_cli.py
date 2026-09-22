"""CLI: build a world-set shard once, save its settled snapshot, restore it in a new build."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import torch

from apple_pick_gym.tests.test_apple_pick_vic_harvest_env import requires_fr3


def _run(module: str, *args: str) -> str:
    """One env build per OS process (docs/in-process-rebuild-heap-corruption.md)."""
    res = subprocess.run([sys.executable, "-m", module, *args], check=True, capture_output=True, text=True)
    return res.stdout


def test_parser():
    from apple_pick_gym.batched_examples.example_build_world_snapshots import make_parser

    a = make_parser().parse_args(
        ["--world-set", "w.jsonl", "--shard-size", "500", "--shard-index", "2", "--out-dir", "d"]
    )
    assert (a.world_set, a.shard_size, a.shard_index, a.out_dir, a.order_seed) == ("w.jsonl", 500, 2, "d", 0)


@requires_fr3
def test_shard_snapshot_restores_the_saved_settled_state_in_a_new_build(tmp_path):
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.world_set import load_world_set, save_world_set, world_specs_to_env_kwargs
    from apple_pick_sim.coupled_fruiting.episode_state_snapshot import load_snapshot_npz

    pool = tmp_path / "pool.jsonl"
    _run("apple_pick_gym.batched_examples.example_screen_harvest_worlds",
         "sample", "--num-envs", "2", "--seed", "4", "--device", "cpu", "--out", str(pool),
         "--hold-steps", "1", "--pull-rest-steps", "1", "--pull-ramp-steps", "1",
         "--pull-hold-steps", "1", "--pull-settle-steps", "1", "--num-pull-episodes", "1")
    specs = [dataclasses.replace(s, screening={**s.screening, "passed": True}) for s in load_world_set(pool)]
    save_world_set(pool, specs)

    out = tmp_path / "shards"
    _run("apple_pick_gym.batched_examples.example_build_world_snapshots",
         "--world-set", str(pool), "--shard-size", "2", "--shard-index", "0", "--out-dir", str(out),
         "--device", "cpu", "--check-steps", "2")
    shard_specs = load_world_set(out / "shard_00.jsonl")
    assert sorted(s.world_id for s in shard_specs) == ["s4_e0", "s4_e1"]
    arrays, meta = load_snapshot_npz(out / "shard_00_snapshot.npz")
    assert meta["num_envs"] == 2 and len(meta["world_fingerprints"]) == 2

    env = ApplePickVicHarvestEnv(
        device="cpu",
        episode_snapshot_path=out / "shard_00_snapshot.npz",
        **world_specs_to_env_kwargs(shard_specs),
    )
    try:
        env.reset()
        np.testing.assert_allclose(env._sim.scene.cable.state_0.body_q.numpy(), arrays["cable_body_q_0"], atol=1e-6)
        np.testing.assert_allclose(env._sim.scene.robot_state_0.joint_q.numpy(), arrays["robot_joint_q"], atol=1e-6)
        torch.testing.assert_close(env._target_pose, torch.tensor(meta["hold_target_pose"]))
        # Worlds the shard build marked invalid stay invalid after loading.
        assert all(now or not saved for now, saved in zip(env._invalid_env_mask.tolist(), meta["invalid_env"]))
    finally:
        env.close()


@requires_fr3
def test_snapshot_refuses_different_worlds(tmp_path):
    """Pairing is by exact plant params + grasp; a mismatched shard must not load."""
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    snap = tmp_path / "s.npz"
    _run("apple_pick_gym.tests.test_example_build_world_snapshots_cli", "--write-default-snapshot", str(snap))
    with pytest.raises(ValueError, match="different worlds"):
        ApplePickVicHarvestEnv(num_envs=2, device="cpu", dr_seed=123, topology_seed=123, episode_snapshot_path=snap)


if __name__ == "__main__":  # helper for test_snapshot_refuses_different_worlds (separate process)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--write-default-snapshot")
    ns = ap.parse_args()
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv

    e = ApplePickVicHarvestEnv(num_envs=2, device="cpu", dr_seed=0, topology_seed=0)
    try:
        e.save_world_snapshot(ns.write_default_snapshot)
    finally:
        e.close()
