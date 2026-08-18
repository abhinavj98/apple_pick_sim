# One-structure multi-direction CMA with 5/3 holdout

| Field | Value |
| ----- | ----- |
| **Status** | Approved |
| **Canonical living docs** | H4 `docs/handbook-real-replay.md`; H5 `docs/handbook-youngs-cma.md`; H3 `docs/handbook-sysid-scoring.md` |
| **Date** | 2026-08-17 |
| **Roadmap** | M4.0 — one tree, eight pulls; verification is reduced train/val losses plus val pose/force magnitude and trend |
| **Extends** | `docs/superpowers/specs/2026-08-14-real-multi-structure-cmaes-design.md` (convert + per-direction replay), `docs/superpowers/plans/2026-08-17-real-vic-pose-cmaes.md` (slice 4 1×1 CMA wiring **shipped**) |
| **Reference data** | `robot_replay/new_data/s09/` — eight compiled `s09-dNN.parquet`, `NN ∈ {0…7}` |

## Purpose

Convert one real tree folder into a **1-structure × 8-direction** `batched_sysid_v1` dataset, replay every pull with its own weld/gripper, **sample five directions at random** (seeded, without replacement) for CMA-ES, freeze \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\), and **report** pooled Sinkhorn plus Cartesian F/T error on the **three leftover** directions the optimizer never saw.

This is VISION's real-data calibration *protocol* (held-out real segment). The slice **passes only if** train Sinkhorn falls over CMA generations, val Sinkhorn of the frozen fit beats the shipped initial-mean baseline on the same three dirs, and on those val dirs fitted **poses and forces match real magnitude and trend** (gates below). A leak-free JSON with diverging overlays is a fail.

## Relationship to the 2026-08-14 spec

That spec remains the contract for folder convert, canonical geometry, manifest fields (`n_holds`, `sim_config`, `topology_seed`), per-direction weld/gripper, last-action padding, and truncate-before-features.

This spec **narrows scope and replaces acceptance**:

| 2026-08-14 | This slice |
| --- | --- |
| Two trees (`s09`+`s11`), eight dirs each | **One** tree (`s09`), eight dirs |
| Pooled Sinkhorn on all dirs of a structure | Seeded sample of 5 train dirs; remaining 3 are val |
| Drop the one-structure `vic_pose` guard | **Keep** the guard (still one structure) |
| Held-out listed as a non-goal | Held-out **is** the acceptance |
| Convert default 15 Hz nearest-copy | Convert F/T: 10 Hz `filtfilt` then block-mean to **30 Hz** (already drafted in working tree) |
| Slice 4 CMA real builder | **Shipped** — reuse, do not reimplement |

Two-tree merge stays future work. Do not implement it here.

## Locked decisions

| Topic | Choice |
| --- | --- |
| Structure | One. `structure_idx = 0`. Keep the grid/CMA `vic_pose` one-structure `SystemExit`. |
| Episode set | Convert **all eight** `s09-dNN.parquet` into one dataset. Do not emit a second “train-only” convert; subsetting is a scoring/replay argument. |
| `direction_idx` | Integer `NN` from the filename `sXX-dNN.parquet`. If parquet `dump.direction_index` exists, it **must** equal `NN` or convert fails. |
| Train / val split | **Opt-in holdout.** Default CMA (no split flags) scores **all** usable dirs, same as today (1×1 real and sim-sim unchanged). When `--direction-split-seed` is passed, or when **both** `--direction-indices` and `--val-direction-indices` are passed, draw/pin a split. Seeded sample **without replacement**: `n_train=5`, `n_val=3`, `len(dirs)` must equal 8. Population is `sorted` disk `direction_idx`. Sampler is `random.Random(seed).sample(dirs, 5)` (stdlib; not NumPy). Train = that sample sorted; val = sorted complement. Default seed when the flag is present with no value: `DIRECTION_SPLIT_SEED = 17`. Seed 17 on `{0…7}` yields train `{2,4,5,6,7}`, val `{0,1,3}`. Write seed + both lists on `holdout_report.json`. Passing only one of the explicit index flags is an error. |
| Phenotype | Unchanged: \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\); primary \(E\) fixed. |
| Fit objective | Existing complete pooled Sinkhorn on **train** dirs only (`gt_candidate is None`). |
| Freeze | CMA `final_mean` (same vector the CLI already evaluates at the end of fit). No `tell()` on val dirs. |
| Val reports | (1) pooled Sinkhorn `eligible_mean` on val dirs; (2) hold-phase Cartesian MAE \(\lvert\Delta F\rvert\) (N) and \(\lvert\Delta\tau\rvert\) (N·m), mean over hold frames pooled across the three dirs. Same world-frame `ft_wrist` H3 contract; **no sim tare, no extra LPF on sim**. |
| Baseline column | Replay the **shipped** `CMA_SEARCH_PARAMS["initial_mean_log10"]` on the **same val dirs**. Fitted val Sinkhorn must be **lower** than this baseline. |
| Verification | See **Verification gates**. Overlays for every val direction (real vs fitted) are required so magnitude/trend can be inspected. |
| Leak | A val `direction_idx` in any CMA generation `tell`, train overlay used as the holdout report, or convert that omits val episodes, fails the slice. |
| Control rate | `collection.control_hz = 30` after convert. Discrete/drive fields: last sample of each window. |
| Geometry | One canonical `fruiting_system_params` / mean `fruiting_base_pos` for the tree; per-dir post-grasp SE(3), `initial_robot_joint_q`, gains, `pull_direction`. |

## Architecture

```text
robot_replay/new_data/s09/s09-dNN.parquet  (NN=0..7)
        │
        ▼
 convert --input-dir  →  batched_sysid_v1
   collection.num_structures=1, num_directions=8, n_holds=4, control_hz=30
   episodes/s00_d00 … s00_d07
        │
        ├─ CMA --direction-split-seed 17   # opt-in; omit flag → all dirs, no holdout
        │     sample 5 train dirs, remainder val
        │     ask/tell pooled Sinkhorn on train only
        │     write cmaes_report.json + freeze final_mean
        │
        └─ holdout eval (no optimizer)
              replay the three val dirs at:
                • shipped initial_mean_log10
                • CMA final_mean
              write holdout_report.json  (seed, index lists, Sinkhorn + F/T MAE)
```

GPU batch during fit is `population_size × 5` directions in one structure chunk. Shipped `population_size=15` ⇒ 75 envs (under the 25000 cap). Local smoke may shrink population/generations the same way slice 4 did; restoration before commit still required.

## Slice 1 — folder convert (one tree)

**Owners:** `apple_pick_sim/system_id/real_to_batched_sysid.py`, `robot_replay/convert_real_to_batched_sysid_metadata.py`

Keep the existing single-file 1×1 CLI. Add `--input-dir`:

- Collect compiled episodes matching `sXX-dNN.parquet` (reject `*_robot`, `*_tracking`, PNGs, videos).
- Write `episode_filename(0, NN)` / metadata `direction_idx = NN`.
- Canonicalize geometry as in the 2026-08-14 spec (identical rods; mean base pose; fail if any direction exceeds `--base-pos-tolerance-m`, default 5 mm).
- World-rotate F/T, 10 Hz zero-phase Butterworth, block-mean F/T and TCP velocity to `--control-hz` (default 30). Provenance: `collection.ft_filter` + episode `ft_filter`.
- Write `collection.n_holds` (4 for these logs), full `sim_config` via `sim_config_to_manifest_dict` from `real_replay_sim_config`, and `topology_seed`.
- `env_idx = direction_idx` (one structure).

Do not merge a second tree in this slice.

## Slice 2 — multi-direction replay

**Owners:** `apple_pick_gym/batched_envs/real_batched_replay_build.py`, `batched_sysid_multi_replay.py`, `apple_pick_sim/system_id/batched_digital_twin_init.py`, `batched_sysid_mmd_grid.py`

Unchanged from 2026-08-14 slice 2, minus structure-keyed *multi*-structure `fruiting_base_pos` (still one structure, one base pose). Required even for 1×8:

- Per-slot weld pose and gripper from **that direction's** episode metadata (today every env gets direction 0).
- Drive tensor padded to max length with the **last logged action**; recorded bags keep true lengths; truncate replay to recorded length **before** features so padded frames cannot enter Sinkhorn.

Keep the `vic_pose` one-structure guard on grid and CMA.

## Slice 3 — direction subset through prepare/evaluate

**Owner:** `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` (`prepare_youngs_modulus_structure`, `evaluate_youngs_modulus_*`), CMA CLI. Grid CLI subset flags are deferred (acceptance is CMA-only).

`resolve_direction_indices` already accepts `direction_indices=`. Thread that optional list from the CMA CLI into prepare/evaluate. Default remains “all usable dirs on disk.”

CMA CLI:

- **Default (no split flags):** all usable directions, no `holdout_report.json`, shipped 1×1 and sim-sim behavior unchanged.
- **Holdout mode:** `--direction-split-seed` (optional explicit int, default 17 when the flag is present) draws train/val as in Locked decisions. Fit loop (every `ask`/`tell` and the final-mean wave) uses only train. After a successful fit, holdout eval uses val only. Those calls must not `tell()`.
- Optional override: both `--direction-indices` and `--val-direction-indices` together pin a split (no seed required); still assert disjoint, non-empty, subset of disk.

Disjointness (holdout mode only): CMA **raise** if the two index lists overlap, if either is empty, if an index is missing from the dataset, or if `len(disk dirs) != 8`.

Scoring: `YoungsModulusScoringConfig.n_directions` is the **one-hot width**, not the loaded-slot count. Keep it at `collection.num_directions` (8, or `max(disk_id)+1`). `_one_hot_dir_id` requires `0 <= dir_idx < n_directions`, so train IDs `{2,4,5,6,7}` crash if the width is `len(selected)=5`. Empty val slots do **not** enter pooling: `WassersteinScoringContext.expected_directions` is the keys of bags that were actually loaded, and `score_candidate_wasserstein_complete` restricts the pool to that set. Collector layout (`direction_episodes_from_collectors`) already uses local `0 .. len(selected)-1` and zips onto disk IDs — do not conflate that with the one-hot width. `collection.num_directions` on disk stays 8. `collection.n_holds` stays 4. Pooled Sinkhorn pools only the selected dirs.

## Slice 4 — holdout report

Write `holdout_report.json` next to `cmaes_report.json`:

Required keys (floats filled at runtime; `baseline` phenotype is the shipped
`initial_mean_log10`):

- `structure_idx` (int, `0`)
- `direction_split_seed` (int; omitted only when both explicit index flags overrode the draw)
- `train_direction_indices` / `val_direction_indices` (int lists, sorted)
- `phenotype_log10.baseline` / `phenotype_log10.fitted` (length-3 log10 vectors)
- `train_fitted`, `val_baseline`, `val_fitted`: each has
  `eligible_mean_sinkhorn`, `mae_force_n`, `mae_torque_nm` (finite floats)
- `verification`: booleans + scalars for the gates below (`train_sinkhorn_decreased`, `val_sinkhorn_improved`, per-dir `force_magnitude_ok`, `force_trend_ok`, `tcp_pose_magnitude_ok`, `tcp_pose_trend_ok`)
- Val overlay HTML paths (one per val dir, real vs fitted)

Cartesian MAE remains diagnostic. Document the magnitude/trend reductions once in H3/H5.

## Verification gates

Hold-phase only, world `ft_wrist` and `tcp_pos`, no sim tare. Pull axis \(\hat p\) is the logged unit `pull_direction` / `excitation_direction` for that episode. Per val direction, then **all three dirs must pass**. Constants: `MAGNITUDE_RATIO ∈ [1/3, 3]`, `TREND_PEARSON_MIN = 0.5`.

**Signed series (not unsigned norms):**

- Force: \(F_\parallel = F \cdot \hat p\) (N). Torque magnitude `|τ|` stays a **magnitude-only** check (no parallel axis in this slice).
- Pose: TCP displacement along pull, \(s = (x - x_{\mathrm{hold0}}) \cdot \hat p\) (m), where \(x_{\mathrm{hold0}}\) is TCP at the first hold frame of that dir.

1. **Reduced train loss.** CMA `structures.0.generations[*].score_summary.eligible_mean`: at least two finite values and `means[-1] < means[0]`.
2. **Reduced val loss.** `val_fitted.eligible_mean_sinkhorn < val_baseline.eligible_mean_sinkhorn`.
3. **Force magnitude.** Hold-frame mean `|F_∥|` and mean `|τ|`: `μ_fit / μ_real ∈ [1/3, 3]`. If `μ_real < 0.2` N (or 0.05 N·m for torque), require `μ_fit < 3 μ_real + 0.4` in the same units instead of a raw ratio.
4. **Force trend.** Per-hold mean \(F_\parallel\) is a length-`n_holds` vector (skip empty holds; need ≥3 holds). Pearson `r(real, fitted) ≥ 0.5`. If either series has variance ≈ 0, **trend passes iff the force-magnitude gate already passed** (do not fail on NaN `r`).
5. **Pose magnitude and trend.** Same ratio gate on mean `|s|` as force. Pearson `r` of per-hold mean \(s\) vs real `≥ 0.5`, with the same zero-variance exception. Apple `pos` along \(\hat p\) is reported the same way but **does not fail** the slice if TCP already passes (diagnostic).

Overlays must show real vs fitted F/T and TCP for each val dir. Magnitude/trend fail even if Sinkhorn improved. Beating the shipped initial-mean Sinkhorn on val is required but cheap (baseline \(E \sim 3\,\mathrm{GPa}\)); the signed pose/force gates are the physics check.

## Error handling

- Convert fails loudly on: duplicate `dNN`, missing compiled parquet, `dump.direction_index` ≠ `NN`, rod mismatch, base-pose spread, empty `--input-dir`.
- Replay fails if two slots in a 1×8 batch share direction-0 weld metadata (regression of the current bug).
- CMA fails if a train `tell` loaded a val episode (assert selected `direction_indices` on every prepared structure).
- Do not guess `--allow-wrench-as-twist` on this path.

## Tests (TDD)

**Convert**

- `--input-dir` of two synthetic `s09-d00` / `s09-d01` files → `num_directions=2`, episodes `s00_d00` / `s00_d01`, `direction_idx` from the filename.
- F/T `filtfilt` + 30 Hz block-mean: expected output length `n_src // round(source_hz/30)` and `collection.control_hz == 30`.
- Last-sample phase at a pull/hold boundary is not the mean of the window.
- `dump.direction_index` mismatch with filename fails.
- Geometry spread over tolerance fails.
- `n_holds`, `sim_config.joint_damping_ratio`, `topology_seed` present.
- Single-file 1×1 CLI still writes `s00_d00` (regression).

**Replay**

- Two directions, unequal lengths: drive padded with last action; bags truncated; padded frame absent from features.
- Two directions in one batch get **distinct** logged apple/TCP weld poses.

**CMA / holdout**

- `--direction-split-seed` absent: 1×1 / sim-sim CMA still uses all dirs and does not require eight episodes.
- `direction_indices=(0,1)` on an 8-dir mock never calls `load_episode_obs_arrays(..., 5)`.
- `choose_direction_split(range(8), seed=17)` → train `{2,4,5,6,7}`, val `{0,1,3}`; disjoint; a different seed can change membership.
- Overlapping explicit train/val lists `SystemExit`; one explicit list without the other `SystemExit`.
- Holdout eval uses `evaluate_*` with frozen candidate only (no extra `tell`).
- `holdout_report.json` contains both phenotypes, both metrics, `verification` flags, and val overlay paths.
- Unit tests for the magnitude ratio and Pearson helpers (factor-of-3 pass/fail; r below 0.5 fails; zero-variance series pass trend iff magnitude passed; opposite-sign \(F_\parallel\) fails trend).
- Scoring one-hot `n_directions` stays `collection.num_directions` (8) on a subset evaluate; loaded dirs `{2,4,5,6,7}` must build features (dir 5 is legal) and val dirs must not appear in `expected_directions`.
- Sim-sim CMA (twist `vic`, all dirs, GT) unchanged.

## Acceptance (this slice)

From the repository root, GPU, `robot_replay/new_data/s09/` compiled bags present:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input-dir robot_replay/new_data/s09 \
  --dataset-out tmp/real_batched_s09 \
  --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09 \
  --output tmp/real_kp_e_cmaes_s09_holdout \
  --direction-split-seed 17 \
  --viewer null \
  --overwrite
```

Exact flag names may match existing argparse style when implemented; the semantics above are the contract.

Pass if and only if:

1. Manifest is 1 structure × 8 directions, `control_hz=30`, `n_holds=4`.
2. `cmaes_report.json` `command_status` completed (local smoke may shrink `population_size` / `max_generations`; restore shipped 15×10 before commit). Document the knobs actually run, as slice 4 did.
3. Holdout mode only: `holdout_report.json` `direction_split_seed == 17`; train `{2,4,5,6,7}`, val `{0,1,3}`; every CMA generation’s loaded dirs ⊆ train.
4. `holdout_report.json` has `val_baseline` and `val_fitted` Sinkhorn + F/T MAE, phenotype vectors, verification flags, and val overlay paths.
5. No `gt_diagnostics`. Search floor on this real run remains \(\log_{10} E = 7\) for spur/stem.
6. **Science gate (required):** train `eligible_mean` last < first; `val_fitted` Sinkhorn < `val_baseline` Sinkhorn; every val dir passes force magnitude + force trend + TCP pose magnitude + TCP pose trend (Verification gates).

Fail the slice for a leak, crash, or any failed science gate. Do not ship “report only” if val pose/force trend is wrong.

## Non-goals

- Second tree / multi-structure merge / dropping the one-structure `vic_pose` guard.
- Changing Sinkhorn features, scales, phenotype, or sim-sim CMA.
- Migrating gym collect / MMD off twist `vic`.
- Diagnosing CUDA exit 139 as a blocker (record if it recurs; shrink population as in slice 4).
- Retargeting `initial_mean_log10` (still a ROADMAP follow-up).
- V.5.3 held-out **sim-sim**.
- Treating a pass as M5-ready (this is s09 one-tree holdout, not a pick policy).

## Open risks

1. A seeded draw can still cluster in \((\theta,\phi)\); that is sampling variance, not a convert bug. Record the lists so the split is inspectable.
2. Per-direction weld is currently unimplemented; without it, eight dirs in one batch is systematically wrong.
3. ~~Uncommitted working-tree convert LPF/30 Hz should land **in this slice** (or immediately before)~~ — **resolved**: landed in `a3feddd` (10 Hz `filtfilt` → 30 Hz block-mean) and `8b9645d` (score `ft_wrist_lpf` instead of overwriting world F/T). Acceptance bags already match H3/H4.
4. Slice-4 CMA native crash (exit 139 at gen 3, pop 6) may return at 75 envs.
