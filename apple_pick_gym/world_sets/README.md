# Screened harvest world sets

Each `*.jsonl` line is a `WorldSpec` (`apple_pick_gym/batched_envs/world_set.py`): the exact
plant params, grasp weld direction, support-joint DR and build-time arm DR of one world, plus
its screening record. Worlds were screened with `example_screen_harvest_worlds.py` (build/rest
check, 2 s zero-action hold, three sys-ID-style 3 cm pulls at the real gains) in **two
independent builds**; only worlds passing both are kept.

| set | worlds | notes |
| --- | --- | --- |
| `harvest_worlds_v1.jsonl` | 587 | first screening (640 candidates) |
| `harvest_worlds_v2.jsonl` | 2000 | seeded random 2000 of 2212 accepted (2432 candidates); includes v1 worlds |

## Settled snapshots (v2)

Built by `example_build_world_snapshots.py` as 4 shards of 500 (shuffle seed 0), stored
outside the repo in `~/.cache/apple_pick_sim/world_sets/harvest_worlds_v2/`:

- `shard_0{0..3}.jsonl` -- the shard's worlds, in env order
- `shard_0{0..3}_snapshot.npz` -- the settled episode baseline (`reset()` restores it)

Valid worlds per shard: 500 / 499 / 499 / 499 (the 3 that misbehaved in their shard build are
marked invalid in the snapshot and masked at training). Train on a shard with:

```python
from pathlib import Path
from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs

d = Path.home() / ".cache/apple_pick_sim/world_sets/harvest_worlds_v2"
specs = load_world_set(d / "shard_00.jsonl")
env = ApplePickVicHarvestEnv(episode_snapshot_path=d / "shard_00_snapshot.npz",
                             **world_specs_to_env_kwargs(specs))
```

Build + restore of a 500-world shard takes ~1 min. Snapshots are tied to their worlds by a
hash of the exact params + grasp and refuse to load into any other set. Rebuild them (same
command, same `--order-seed`) if the sim/build code changes.

## Committed snapshot: all 2000 worlds in one build (RL training set)

`snapshots/harvest_worlds_v2_all2000/` is committed to the repo (~11 MB) so training does not
depend on a local cache. It is a single N=2000 shard of `harvest_worlds_v2.jsonl`:

- `shard_00.jsonl` -- all 2000 worlds, in the snapshot's env order (a reordering of v2)
- `shard_00_snapshot.npz` -- the settled baseline; 3 worlds are marked invalid and masked

Built on 2026-09-22 with `example_build_world_snapshots.py --shard-size 2000 --shard-index 0`
from branch `fix/harvest-vic-forces` at `a750dba`. Load it the same way as a shard, with
`d = Path("apple_pick_gym/world_sets/snapshots/harvest_worlds_v2_all2000")`.

**Staleness:** the snapshot is a settled sim state, valid only for the sim/build code it was
built with. Its fingerprint check only catches changed worlds, not changed physics, so after
any change to the build path, solver settings or the Newton pin, rebuild it with the command
above (using `--retries 4`) and commit the new files. Each rebuild adds ~11 MB to git history,
so rebuild only when the code actually changed.
