# Sys-ID stable collect & replay (blow-up isolation)

> **Amendment (2026-07-15, matches shipped code):** soft-disable / collect
> `excluded` uses **sticky NaN/Inf + IK only** (`hard_blowup_mask`), not every
> force/speed-cap frame. Offline `exclude_unstable_episodes` uses unstable-frame
> **fraction > 0.25** (not “any `stable=False`”). Force/torque monitor caps are
> **50 N / 20 N·m**. See `docs/batched-stability-monitor-design.md` and
> `docs/batched-sysid-dataset.md`.


**Date:** 2026-07-12  
**Status:** Approved for planning  
**Roadmap:** V.5.1 (loss / grid hardening precursor — crash-safe collect/replay)  
**Related:** `docs/batched-stability-monitor-design.md`, `docs/batched-sysid-dataset.md`, `docs/ROADMAP.md`

## Problem

Batched sys-ID collection and grid replay keep stepping every env even after a world blows up (NaN, wrench/speed caps). That risks destabilizing the shared batch, wastes compute, and still writes unusable `(structure, direction)` episodes that later pollute GT ranking. Today `stable` is per-frame annotation only: scoring masks frames and can disqualify **candidates**, but nothing **disables** a bad env mid-run or **excludes** bad GT episodes before replay.

## Goals

1. **Mid-collect soft-disable:** on first stability blow-up for an env, sticky-disable it: zero its actions, stop recording further frames, leave physics state as-is, keep sibling envs running.
2. **Mid-replay soft-disable:** same policy during `replay_batched_sysid_structure` / grid replay.
3. **Persist exclusion:** still write the (partial) Parquet; mark the episode `excluded` in the manifest with a reason.
4. **Post-collect filter:** offline tool marks any episode with **any** `stable=False` frame as excluded (covers legacy datasets and anything missed at collect time).
5. **Grid load:** skip excluded episodes by default so they never enter scoring.
6. **Trustworthy `stable`:** tune/document `StabilityThresholds` enough that rule “any unstable → exclude episode” is safe; add tests against obvious false positives.
7. **GPU hot path:** disable masks and action zeroing stay device-resident `torch` ops; no per-env CPU round-trips in the step critical path.

## Non-goals

- Multi-step / horizon transition features (deferred).
- Hold quasi-static quality filters (force CV / drift / TCP excursion) as exclude criteria.
- CEM / collect-time **penalty** bookkeeping for blow-ups (deferred; disable + exclude only).
- Per-env physics removal, snapshot restore, or rebuild of a single world slot.
- Replacing `batched_hold_quasi_static.py` or folding QS plugins into the monitor (follow-up deep dive).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Placement | Mid-collect disable **and** post-collect filter |
| Disable behavior | Soft freeze: `actions=0`, stop writing, leave state |
| Episode on disk | Keep Parquet; `excluded=true` in manifest |
| Replay | Same soft-disable as collect |
| Exclude rule | Any frame with `stable=False` → exclude whole `(s,d)` |
| Thresholds | Part of this slice: document + light tune + false-positive tests |
| Features / QS / penalties | Out of scope |

## Architecture

```text
┌─ Collect / Replay step loop ─────────────────────────────┐
│  actions = traj_or_recorded                              │
│  actions = EnvDisableController.apply_actions(actions)   │
│  obs = env.step(actions)                                 │
│  report = BatchedStabilityMonitor.check(obs)             │
│  EnvDisableController.update(report.unstable)  # sticky  │
│  record_step only if not disabled[i]                     │
│  finalize → manifest.episodes[].excluded                 │
└──────────────────────────────────────────────────────────┘

┌─ Post-collect filter ────────────────────────────────────┐
│  scan frames: any stable=False → excluded=true           │
│  patch manifest (or write cleaned copy)                  │
└──────────────────────────────────────────────────────────┘

┌─ Grid load ──────────────────────────────────────────────┐
│  skip episodes with excluded=true (default)              │
└──────────────────────────────────────────────────────────┘
```

### Ownership

| Piece | Module | Role |
|-------|--------|------|
| `EnvDisableController` | `apple_pick_gym/batched_envs/` (new) | Sticky disable + vectorized action zeroing |
| Collect wiring | `batched_sysid_collect.py` | Use controller in collect loop |
| Replay wiring | `batched_sysid_mmd_grid.py` | Use controller in replay loop |
| Manifest fields | trajectory store / `docs/batched-sysid-dataset.md` | `excluded`, `excluded_reason` |
| Filter library + CLI | `apple_pick_gym/` (+ thin script or example) | Offline any-`stable=False` → exclude |
| Threshold tuning | `batched_stability_monitor.py` + stability design doc | Align `stable` with “blow-up / unsafe” |
| Grid skip | load helpers used by `example_batched_sysid_mmd_grid.py` | Default skip excluded |
| Shell glue | `scripts/collect_and_rank_sysid_gt.sh` | Run filter between collect and grid |

## Components

### `EnvDisableController`

- State (on env device): `disabled: BoolTensor[num_envs]` (sticky).
- Optional host-side bookkeeping for finalize/logging: disable step index, reason codes (updated sparingly; not in the inner tensor path).
- API:
  - `update(unstable: BoolTensor) -> None` — `disabled |= unstable` (device bool).
  - `apply_actions(actions: Tensor) -> Tensor` — zero rows where `disabled` (vectorized; preserve device/dtype).
  - `should_record_mask() -> BoolTensor` or host-safe query after step for the existing per-env record loop.
- Does **not** call `env.reset` or restore snapshots.
- **GPU rule:** `update` / `apply_actions` must not `.cpu()` / `.numpy()` / `.item()` in the hot path. Recording may still loop envs on host (I/O already does).

### Collect behavior (`collect_batched_quasi_static_dataset`)

```text
for each step:
  actions = trajectory actions
  actions = controller.apply_actions(actions)   # zeros already-disabled envs
  obs = env.step(actions)
  report = monitor.check(obs, step_idx=...)
  # Record this step for envs not yet disabled (includes the blow-up frame).
  for i where not disabled[i]:
      record_step(..., stable=not unstable[i])  # blow-up → stable=False once
  controller.update(report.unstable)            # sticky: no further frames next steps
```

- Finalize: any env ever disabled **or** any recorded frame with `stable=False` →  
  `excluded=true`, `excluded_reason="stability_blowup"`.
- Partial trajectories remain on disk for debugging (last frame may be the blow-up).

### Replay behavior (`replay_batched_sysid_structure`)

- Same soft-disable on candidate envs.
- Stop accumulating collector frames / features for disabled slots after disable.
- Existing `UNSTABLE_DISQUALIFY_THRESHOLD` (10%) candidate disqualification remains as a second line; soft-disable is the crash guard.

### Manifest schema additions (`batched_sysid_v1`)

Per `manifest.episodes[]` entry (backward compatible; missing ⇒ not excluded):

| Field | Type | Meaning |
|-------|------|---------|
| `excluded` | bool | If true, default loaders/grid skip this episode |
| `excluded_reason` | string \| null | e.g. `"stability_blowup"` |

### Post-collect filter

- Library: e.g. `exclude_unstable_episodes(dataset_dir, *, inplace=False, output_dir=None)`.
- Rule: any frame with `stable=False` → set `excluded=true` / `excluded_reason="stability_blowup"` on that episode.
- Default: do not silently destroy the only copy — prefer writing an updated manifest alongside or to `output_dir`; `--inplace` explicit for in-place manifest patch.
- CLI thin wrapper; `collect_and_rank_sysid_gt.sh` invokes it between collect and GT-rank replay.

### Grid load

- `load_recorded_episodes_for_structure` / action-tensor builders skip `excluded` episodes by default.
- Optional kwarg / CLI `--include-excluded` for debugging only.
- If a structure has **zero** usable directions after filtering → fail with a clear error (no empty silent grid).

### Stability thresholds (this slice)

- Document that `stable` means **blow-up / unsafe**, not hold QS quality.
- Light-tune defaults if a small real collect or tests show clear false positives under current caps (`StabilityThresholds` / stem force-torque caps).
- Add unit tests: nominal under-cap obs → stable; NaN or extreme force → unstable.

## Error handling

- All directions excluded for a structure → explicit error listing structure index / counts.
- Filter without `--inplace` never deletes Parquets; only updates exclusion metadata.
- Disabled envs that worsen (more NaNs): remain zero-action; no resurrect in this slice.
- Legacy manifests without `excluded`: treat as `excluded=false`; filter can annotate them.

## Testing

**Unit (fast)**

- Controller sticky OR; action zeroing only on disabled rows; device preserved.
- Manifest round-trip of `excluded` / `excluded_reason`.
- Loaders skip excluded by default; include flag loads them.
- Filter marks episode with one `stable=False`; leaves all-stable untouched.
- Monitor: nominal vs NaN / huge force.

**Integration (targeted)**

- Collect with mocked/tiny batch: after first unstable, later frames not recorded; actions for that slot are 0; finalize sets excluded.
- Replay: same disable; siblings continue.
- Script path: filter runs between collect and grid (smoke / dry invocation acceptable).

## Success criteria

1. One blown env does not require aborting the whole collect or replay batch.
2. Excluded `(s,d)` never enter default grid scoring.
3. Soft-disable hot path stays on GPU tensors.
4. Documented thresholds + tests support “any `stable=False` ⇒ exclude episode.”

## Follow-ups (explicit reminders)

1. **Deep dive into stability monitor / controller** — retune thresholds, reason persistence, possible QS plugins; treat current light tune as insufficient for long-term.
2. Multi-step / horizon transition features for MMD/Wasserstein bags.
3. Collect-time / CEM **penalty** when an env blows up (disable remains; add scored penalty later).
4. Hold QS-based outlier filtering (promote `batched_hold_quasi_static` from report-only).

## Validation commands (planned)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_env_disable_controller.py \
  apple_pick_gym/tests/test_exclude_unstable_episodes.py \
  apple_pick_gym/tests/test_batched_sysid_collect.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py -q

# after implement: filter between collect and grid
# bash scripts/collect_and_rank_sysid_gt.sh
```

(Exact test module names may match implementation; keep under `apple_pick_gym/tests/`.)
