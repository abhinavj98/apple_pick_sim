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
| `harvest_worlds_v3.jsonl` | 2000 | [D17] task v2: equator grasps (60 deg cone about world -y, i.i.d.; roll +-90 deg; world-frame grasp), 500/30 hold; seeded random 2000 of 2459 accepted (3072 candidates). Coverage: `harvest_worlds_v3_coverage.md` |

**v1/v2 caveat (found 2026-10-02, D17i/j):** their grasps were built in the legacy apple frame, so the
realized grasp is the stored `weld_direction` rotated by the settled apple (26 deg median), and their
Fibonacci grasp lattice repeats the same directions for every seed (v2: 192 distinct in 2000).

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

## Committed snapshot: v3 (task v2, D17)

`snapshots/harvest_worlds_v3_all2000/` -- a single N=2000 shard of `harvest_worlds_v3.jsonl`; 1998/2000
valid (2 marked invalid and masked). Its `shard_00.jsonl` pins each world's resolved apple-frame weld
(`weld_offset`): world-frame grasps resolve the weld from the build's settled apple, which is not
bit-reproducible (two builds of the same worlds differed by up to 9.9 mm / 28 deg), so a training build
must reuse the snapshot build's weld or a third of the worlds start with 30-200 N at the wrist. The
fingerprint includes the offset. Built on 2026-10-03 with
`example_build_world_snapshots.py --world-set apple_pick_gym/world_sets/harvest_worlds_v3.jsonl
--shard-size 2000 --shard-index 0 --retries 4` (first attempt, 258 s; the builder writes the resolved welds). `sim_train_gpu_v2.json` trains on it.

Screening recipe (pass 1, one batch per process; pass 2 rescreens the survivors in chunks of 128,
shuffle seed 0):

```bash
uv run python apple_pick_gym/batched_examples/example_screen_harvest_worlds.py sample --num-envs 128 --seed <300..323> \
    --grasp-pole 0 -1 0 --grasp-max-polar-deg 60 --grasp-roll-max-deg 90 --grasp-frame world --grasp-sampling random \
    --out tmp/harvest_worlds_v3/pass1_v3.jsonl
```


## vv-wide bend (D17p)

`snapshots/harvest_worlds_vvwide_{train,holdout}/` (not committed): 10,048 / 1,536 worlds from
`fruiting_system_ranges_rl_harvest_vvwide_bend.json` (spur/stem flexural modulus log-uniform 1e5..1e11 Pa, support kp 100..10,000).
Two-pass screened (14,289 / 25,216 accepted), then balanced over spur decade x stem decade x kp half-decade. Recipe and scripts:
`tmp/harvest_worlds_vvwide/` (`run_pass1.sh`, `run_pass2.sh`, `after_pass1.sh`, `select_balanced.py`); decision log D17p.
