# Sinkhorn fixed-scale normalization

| Field | Value |
| ----- | ----- |
| **Status** | Design approved, not yet implemented |
| **Date** | 2026-08-14 |
| **Roadmap** | M4.0 real `robot_replay` → CMA (Sinkhorn on converted GT + live sim replay) |
| **Extends / amends** | `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md` — that spec locks full 19D `action` in `STATE_VECTOR`; this spec drops `action` from the **score** only (replay still drives full 19D) |
| **Reference episode** | `robot_replay/new_data/s09/s09-d00.parquet` (converted 15 Hz, decimated from 1 kHz log; `tmp/real_batched_s09_d00`) |

## Problem

`fit_gt_normalization` (`apple_pick_sim/system_id/mmd.py`) z-scores each `STATE_VECTOR` column using **GT-only** mean/std, with a floor: `std = 1.0` when `std < 1e-6`. On real hold-phase bags, several channels are near-static (e.g. `tcp_velocity` hold std ≈ 1e-4 m/s), well above the `1e-6` floor. A ~1 cm/s sim/real residual on such a channel becomes O(100) "σ", dominating aggregate Sinkhorn (~93% of cost on the `s09-d00` grid smoke) and hiding real plant-fidelity signal (F/T, woody bend) that has healthy GT variance and would otherwise drive ranking.

Evidence (`tmp/real_kp_e_grid_s09`, winner candidate `c001`, aggregate Sinkhorn ≈ 3.59e4):

| Group | Only-this-differs Sinkhorn | Frac of total |
|---|---:|---:|
| `tcp_velocity` | 3.33e4 | 0.93 |
| `woody_start` | 2.2e3 | 0.06 |
| `apple_pos` | 161 | 0.004 |
| `ft_wrist` | 153 | 0.004 |
| `woody_bending_angles` | 38 | 0.001 |
| `tcp_pos` | 9 | 0.000 |
| `action` | 0 | 0.000 (identical GT/sim by construction) |

## Goal

Fidelity-preserving normalization (goal **B**, not ranking-only): keep real/sim residuals interpretable in physical units, stop the numerical explosion caused by near-zero GT hold variance, and remove `action` from the score (it never differs on this replay contract, so it contributes nothing to ranking regardless of scale).

## Non-goals

- Changing the **fit site**: `μ` (and today, `σ`) are still fit from GT only; candidate statistics never enter normalization.
- Removing `action` from replay drive, `REQUIRED_ARRAY_KEYS`, or the parquet/bag contract — only from `STATE_VECTOR_FIELDS` / `build_state_matrix` (score-time feature columns).
- Reworking the alignment spec's woody/F-T/hold contract (`docs/superpowers/specs/2026-08-13-...`) beyond dropping `action` from the score.
- Fixing or reviving the `biased_mmd2` / `batched_sysid_mmd_grid.py` MMD-grid path — it is stale (superseded by Sinkhorn in `wasserstein.py`) and is noted as such, not actively maintained by this change.
- Per-episode or per-structure auto-tuned scales; `s_phys` is a fixed constant table, not fit from data at runtime.

## Design

### 1. Drop `action` from `STATE_VECTOR`

`STATE_VECTOR_FIELDS` (`apple_pick_sim/system_id/mmd_features.py`) removes `"action"`. `build_state_matrix` no longer concatenates an action block. `REQUIRED_ARRAY_KEYS` is unchanged — `action` is still required in the arrays mapping (collector validation, replay drive), just not scored.

### 2. Fixed physical scale replaces GT-std normalization

`fit_gt_normalization` / `apply_normalization` (`apple_pick_sim/system_id/mmd.py`) change from:

```text
x_norm = (x − mean(GT)) / max(std(GT), eps→1.0)
```

to:

```text
x_norm = (x − mean(GT)) / s_phys
```

`mean(GT)` is still fit from GT transition features only (unchanged call site: `wasserstein.py`'s `sinkhorn_gt_preference` / pooled-GT paths). `s_phys` is a fixed vector aligned to the post-`action`-removal `STATE_VECTOR` column order, applied identically to the level (`s`) and delta (`Δs`) halves of each transition row. Hold one-hot columns (appended after `[s, Δs]`) are **not** mean-centered and use `s_phys = 1` (raw 0/1 passthrough, unchanged from today's *values* — only the surrounding blocks change scale).

The `std < 1e-6 → 1.0` floor is removed for scored blocks; it is no longer meaningful once scale is fixed rather than GT-derived.

### 3. Scale table

Derived from measured GT statistics on `s09-d00` (real hold std, real pull amplitude, marker noise floor — see chat evidence) and the user's explicit "care about this much" values:

| Block | Dims | `s_phys` | Basis |
|---|---:|---:|---|
| `ft_wrist` force (Fx,Fy,Fz) | 3 | 3 N | ~O(1) vs hold Fy std (~1 N), pull swings (few N) |
| `ft_wrist` torque (Tx,Ty,Tz) | 3 | 0.5 N·m | ~hold Tx std (0.37 N·m) |
| `tcp_velocity` linear (vx,vy,vz) | 3 | 0.02 m/s | ≫ hold std (~1 mm/s); ~pull-relevant speed |
| `tcp_velocity` angular (wx,wy,wz) | 3 | 0.02 rad/s | ≫ hold std (few mrad/s) |
| `tcp_pos` | 3 | 0.01 m | half of measured Y travel (~25 mm) |
| `apple_pos` | 3 | 0.02 m | ~measured Y travel (~28 mm) |
| `woody_start[primary_spur]` | 3 | 0.02 m | user-specified (branch moves ~1–2 mm in practice; scale is deliberately looser than noise floor) |
| `woody_start[spur_stem]` | 3 | 0.02 m | ~measured Y travel (~25 mm) |
| `woody_bending_angles` (primary_spur, spur_stem) | 2 | 0.05 rad | ~hold mean/spread (~0.05–0.1 rad) |
| hold one-hot | n_holds | 1 (no centering) | categorical, not physical |

Column order matches `STATE_VECTOR_FIELDS` after dropping `action`: `ft_wrist(6) → tcp_velocity(6) → tcp_pos(3) → apple_pos(3) → woody_part_start_pos(6, primary_spur then spur_stem) → woody_bending_angles(2)`, mirrored for the `Δs` half, then hold one-hot appended.

### 4. Blast radius / stale path note

`apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` calls the same `fit_gt_normalization` / `apply_normalization` for its `biased_mmd2` scorer. That path is stale (superseded by the Sinkhorn scorer in `wasserstein.py` / `example_youngs_modulus_sys_id.py`) and will inherit the fixed-scale behavior incidentally. No functional fix or revival of that path is in scope; add a one-line stale-path note in its module docstring.

## Tests

- `apple_pick_sim/tests/test_mmd_features.py`: `test_build_state_matrix_uses_exact_feature_order` and other width/order assertions (~6 sites) drop the `action` block from expected output.
- `apple_pick_sim/tests/test_mmd.py`: `test_gt_normalization_is_per_feature_and_uses_gt_only_statistics` rewritten for fixed-scale behavior — GT mean still fit from data, `std`/scale now asserted against the fixed table, not GT-derived variance; add a case showing a near-zero-GT-variance column no longer inflates candidate residuals into triple digits.
- New regression: reproduce the `s09-d00` winner-candidate ablation (`tcp_velocity` no longer ~93% of aggregate Sinkhorn) as a smoke assertion, if feasible with a small fixture (not full sim replay).

## How to verify

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py apple_pick_sim/tests/test_mmd.py -q -p no:launch_testing

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_grid_s09_fixedscale \
  --viewer null \
  --support-kp-values 1e3,1e4 \
  --log10-e-spur 9.0 \
  --log10-e-stem 9.0,8.0 \
  --export-replays \
  --overwrite
```

Re-run the feature-group ablation from this chat's methodology on the new ranking output; confirm `tcp_velocity` no longer dominates aggregate Sinkhorn and `action` contributes exactly 0 (still identical GT/sim by construction, now also absent from `STATE_VECTOR`).
