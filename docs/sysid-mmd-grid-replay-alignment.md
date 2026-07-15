# Sys-ID MMD grid replay alignment

| Field | Value |
| ----- | ----- |
| **Status** | Shipped (V.4.3) |
| **Next** | V.5.1 loss / GT-scoring hardening — see `docs/ROADMAP.md` |

## Summary

Batched sys-ID grid replay compares recorded quasi-static trajectories against
re-simulations under candidate bend-stiffness values. This document records the
frame-alignment bug fixed in the grid pipeline, the reconciled hold-metric
semantics, and the intentional structure-level weld design.

## Pre-weld row vs replay steps

During collection (`example_batched_collect_sysid_data.py`), each episode begins
with a **pre-weld snapshot** row:

- `step_idx = -1`, `phase = -1` (`pre_weld`)
- Observations come from the settled tree **before** weld; no `env.step` ran
- `action` is zeros; `tcp_pos` / `ft_wrist` are placeholder zeros

Replay must compare **real trajectory steps only**. The grid pipeline strips
pre-weld rows once via `strip_pre_weld_rows()` in
`apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`, applied in:

- `build_recorded_actions_tensor`
- `load_recorded_episodes_for_structure`
- `recorded_metadata_by_env`

After stripping, `recorded[i]` aligns with replay step `i` (same action applied,
post-step observation compared).

Metric helpers (`trajectory_mse`, `trajectory_hold_aggregated_mse`,
`grid_viz_table.replay_vs_recorded_errors`) fail fast if a leading pre-weld row
is still present.

## Hold metrics

Default scoring uses `--use-median` (full hold windows, no latter-half burn-in).
Bag layout, one-hots, and dims: `docs/sysid-transition-features.md`.

| Path | Behavior |
|------|----------|
| Console `--score-mse` with `--use-median` | `trajectory_paired_hold_median_mse` — paired median(replay) vs median(GT) per hold |
| Wasserstein with `--use-median` | Hold→hold median bags `[s_i, Δs_i]`; optional `--hold-id-onehot` / `--pool-directions` (latter auto-appends dir one-hot) |
| `--no-use-median` | Frame-wise MSE and frame→frame bags on full holds |

Deprecated CLI (still accepted): `--mse-hold-aggregation` maps to `--use-median` /
`--no-use-median`; `--mse-hold-latter-half` is a no-op (full holds always).

Named gates: `scripts/gate_sysid_gt_sinkhorn.sh --gate gate_median_hold|gate_hold_id|gate_pooled_dirs`.

Grid viz woody metrics (`apple_pick_gym/grid_viz_metrics.py`):

- `woody_segment_pos_mse_paired_holds` — aggregate within each hold, then mean over holds (default median path)
- `woody_segment_pos_mse_hold_aggregated` — legacy flat bag over all hold frames

By default the grid skips manifest episodes with `excluded: true` (collect sticky
disable or offline `exclude_unstable_episodes`); pass `--include-excluded` only for debug.

`GridVizRow` also reports `n_directions_all` / `n_directions_hold`: the number
of directions with valid metrics. Cross-direction means require all directions
to be finite (no silent NaN dropping).

## Structure-level weld (intentional)

Each **structure** has one physical weld/grasp point, shared across:

- All recorded excitation directions for that structure
- All stiffness candidates replayed in the grid

`replay_batched_sysid_structure` reads weld metadata from direction 0 as a
representative source (`load_episode_metadata(structure_idx, 0)`). This is
**not** a GT-only shortcut: the weld anchor is a constant of the recorded pick,
not a per-candidate quantity. Non-GT candidates are evaluated against the same
grasp point by design.

## Build params vs init state (independent)

| Axis | Default | Opt-in |
|------|---------|--------|
| **Build params** | Oracle: `true_params_for_structure` (privileged recorded `FruitingSystemParams`) | `--infer-params` → `infer_base_params_for_structure` (obs digital twin) |
| **Init state** | Settle + `initialize_batched_env_from_dataset` from obs/metadata | `--use-snapshot` → restore `EpisodeStateSnapshot` |

These flags are independent. Snapshot is privileged **state**, not privileged **params**.
Grid MSE/Wasserstein debugging currently defaults to oracle params. Infer-only
fidelity floor (V.4.2.1) is **deferred** — helpers and `--infer-params` exist;
a dedicated capstone test is not Current focus (`docs/ROADMAP.md`).

## Collection / replay sim config

Grid replay reads `manifest.collection.control_hz` (falls back to module
`CONTROL_HZ = 30.0`). Collection and replay example scripts must keep
`VIC_GAINS`, settle substeps, and related module constants in sync.

## Tests

- `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py` — pre-weld strip,
  action tensor shape, frame alignment, trajectory MSE / paired-hold median /
  legacy hold aggregation, manifest `excluded` load skip
- `apple_pick_gym/tests/test_batched_sysid_grid_viz_table.py` — viz row GT ranking
- `apple_pick_gym/tests/test_batched_sysid_grid_viz_integration.py` — end-to-end
  GT preference on a tiny collected dataset
- `apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py` —
  manifest `control_hz`, `--use-median` / deprecated hold flags, `--hold-id-onehot` /
  `--pool-directions`

## Validation

From the repository root:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_sysid_grid_viz_table.py \
  apple_pick_gym/tests/test_batched_sysid_grid_viz_integration.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py -q

uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \
  --viewer null --dataset tmp/batched_sysid_dataset_xs/ --replay-only --score-mse \
  --plot-output tmp/mmd_grid_test_slow_xs_fixed --grid-values-are-gt-multipliers \
  --primary-bend-stiffness-values 0.005,100.5 --spur-bend-stiffness-values 1 \
  --stem-bend-stiffness-values 0.005,100.5 --secondary-bend-stiffness-values 1 \
  --use-median
```

Expect GT candidate (`dist_log_gt = 0`) to rank best or near-best on hold metrics
after the pre-weld alignment fix. Further GT-rank reliability is **V.5.1**.
