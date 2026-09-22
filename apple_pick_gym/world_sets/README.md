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
