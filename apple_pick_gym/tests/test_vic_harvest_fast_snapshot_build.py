"""[D17q] fast_snapshot_build: seeding the build from the episode snapshot (skipping the gravity and post-grasp
settles the snapshot overwrites) must give the same env as the full build + snapshot: same model, same
restored state, same rollout. The rollout is compared only over its first steps: GPU contact atomics make even
two full builds split run to run (seen from step 12)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / "apple_pick_gym/world_sets/snapshots/harvest_worlds_true_holdout/shard_00.jsonl"
N_WORLDS = 8
# per-process F/T sensor noise (unseeded), BVH flags and pointers differ between any two processes
IGNORE = ("build_s", "bvh_shape_enabled", "shape_source_ptr", ".obs.ft_tool", ".obs.ft_wrist")


def _run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "apple_pick_gym.tests.fast_snapshot_parity_worker", *args], cwd=ROOT, check=True)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not WORLDS.exists(), reason="true-holdout world set not present")
def test_fast_snapshot_build_matches_full_build(tmp_path):
    worlds = tmp_path / "worlds.jsonl"
    worlds.write_text("".join(open(WORLDS).readlines()[:N_WORLDS]))
    snap = tmp_path / "snapshot.npz"
    _run("make", str(worlds), str(snap))
    _run("dump", str(worlds), str(snap), "off", str(tmp_path / "full.npz"))
    _run("dump", str(worlds), str(snap), "on", str(tmp_path / "fast.npz"))
    full, fast = np.load(tmp_path / "full.npz"), np.load(tmp_path / "fast.npz")

    assert set(full.files) == set(fast.files)
    bad = []
    for k in sorted(full.files):
        if any(k.endswith(i) for i in IGNORE):
            continue
        a, b = full[k], fast[k]
        if a.shape != b.shape:
            bad.append(f"{k}: shape {a.shape} vs {b.shape}")
        elif a.dtype.kind in "fc":
            if not np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True):
                bad.append(f"{k}: max |diff| {np.nanmax(np.abs(a - b)):.3g}")
        elif not np.array_equal(a, b):
            bad.append(f"{k}: differs")
    assert not bad, "\n".join(bad[:60])
    print(f"build: full {float(full['build_s']):.0f} s, fast {float(fast['build_s']):.0f} s")
