# Stiffness scoring via quasi-static level bags

| Field | Value |
| ----- | ----- |
| **Status** | Approved — ready for implementation |
| **Canonical living doc:** | `docs/handbook-sysid-scoring.md` (H3) |
| **Date** | 2026-08-26 |
| **Roadmap** | M4.0 — real `robot_replay` → CMA; Young's modulus sys-ID on quasi-static hold frames |
| **Extends / amends** | `docs/superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md` (fixed `s_phys`, one-hot scale 1) and `docs/superpowers/specs/2026-07-14-median-hold-features-design.md` (hold-reduced `[s, Δs]` rows) |
| **Reference dataset** | `tmp/real_batched_s09_k_frame` (8 directions, `n_holds=4`, converted from 1 kHz logs to 30 Hz) |

## Decisions (2026-08-26)

This slice targets **stiffness identification**, not global force-magnitude matching.
Stiffness is the load–deformation relationship across holds within each excitation
direction: at hold *k*, what `ft_wrist`, woody displacement, and bend angles appear
for a given commanded pose?

| Choice | Decision | Rationale |
| --- | --- | --- |
| Hold aggregation | **No mean** — all stable hold frames | Hold means mix settling transients with plateau values; real and sim settle at different rates, biasing Young's modulus (see §Why not hold mean). |
| Feature shape | **Levels only** — drop `Δs` | Quasi-static plateaus carry stiffness; frame deltas are noise and timing-sensitive. |
| Direction pooling | **Yes** — keep `pool_directions=True` | One GT normalization fit, one GeomLoss call, one CMA scalar. Safe only with categorical anchoring. |
| Categorical anchoring | **`categorical_weight = 30`** | Without it, pooled transport leaks across directions/holds for ~2 cost; hold progression is the stiffness signal. |
| Force-magnitude penalty | **Not required for this slice** | `--force-magnitude-weight` is a narrow first-moment hack; defer until anchoring is measured. |

**Target run configuration:**

```text
--hold-aggregation none --no-include-delta --categorical-weight 30 --pool-directions
```

(`--pool-directions` is already the default; listed explicitly for clarity.)

## Problem

The shipped CMA path (`--hold-aggregation mean`, transition bags, pooled directions,
`categorical_weight = 1`) optimizes a score that improves while stiffness-relevant
quantities stay wrong. Task 9 (2026-08-17) is the canonical failure: train
`eligible_mean` 22.54 → 17.08, val 23.63 → 17.13, yet val torque ratios were
0.073 / 0.014 / 0.044. A `|log(sim‖F‖/real‖F‖)|` term (`--force-magnitude-weight`)
was added 2026-08-24 as a compensating hack.

Two mechanisms explain the misspecification.

**1. Pooled transport escapes through categorical columns.**
`transition_feature_scale` gives hold and direction one-hots `s_phys = 1`, so
crossing a direction or hold boundary costs ~2 (squared L2) while a 5 N force
error costs ~100. The optimal plan pairs a too-soft sim direction with whichever
real direction has similar forces, pays ~2, and reports a low pooled distance.
That measures a **pooled marginal** of forces, not the per-direction,
per-hold stiffness curve. Hold identity is equally cheap to cross, and the
hold-to-hold load progression is precisely the stiffness signal.

**2. Hold means on transition bags discard the quasi-static samples.**
With `--hold-aggregation mean` and `n_holds=4`, each direction yields **3**
hold-to-hold transition rows. Pooling five train directions gives ~15 rows in
58 dimensions — too few for distributional scoring and too coarse for stiffness
(the across-hold curve is compressed to 3 points per direction).

Forces are not under-weighted (`s_phys = 0.5 N` makes F/T dominate ground cost).
The leak is **correspondence** and **aggregation**, not feature weighting.

## Goal

Score **stiffness** by matching quasi-static hold-frame levels under per-direction,
per-hold correspondence, while keeping Sinkhorn's latency invariance within each
hold plateau. Real F/T is 1 kHz → LPF → 30 Hz block-mean; sim harvest is one
sample per 30 Hz step. Frame *i* on the two sides is the same label, not the same
physical instant, so the objective must not assume frame-wise temporal alignment.

## Why not hold mean

`iter_kept_hold_segments` returns the **full** contiguous `phase == 1` segment
(no latter-half burn-in). Each hold therefore contains `[settling transient …
plateau …]`. `stable` masks bad samples but does not trim the approach.

A hold mean averages transient and plateau frames into one number. Real and sim
settle at different rates (actuator dynamics, 10 Hz LPF, solver substeps,
viscoelastic creep). That produces a **systematic bias** in the scored feature
that CMA interprets as stiffness error — e.g. identical equilibrium force but
different time constants yield different hold means.

With all stable frames and levels-only rows, each `(direction, hold)` cell holds
~60 real and ~60 sim rows. Most mass sits on the plateau; transient mismatch
contributes its own bounded cost without shifting the plateau comparison. A hold
mean has no such freedom — transient contamination is indistinguishable from a
stiffness error.

**Hold-mean level rows with anchored categoricals** (4 rows/direction, ~20 pooled)
degenerate further: one real row and one sim row per `(direction, hold)` cell
forces Sinkhorn into paired hold-mean MSE. That removes latency invariance
entirely and duplicates `mean_hold_block_errors`. Permitted by the API but
**explicitly not the target configuration** for this slice.

## Why yes pooling (with anchoring)

Pooling across directions is retained because:

1. **One normalization fit** — all directions share one GT `μ` and `σ`; no
   per-direction centering that would break cross-direction comparability.
2. **One GeomLoss call** on ~1000 rows instead of eight small bags.
3. **One CMA scalar** — no aggregation rule over per-direction scores.

With `categorical_weight = 30`, crossing cost is `2w² = 1800`, intended to exceed
any plausible within-cell row distance. Transport becomes effectively
block-diagonal: pooled scoring is a bookkeeping convenience over the same
per-`(direction, hold)` cells that per-direction scoring would use.

**Pooling without anchoring is rejected** — it re-introduces the correspondence
leak (§Problem, item 1).

Verify `w = 30` against actual within-cell distance scales on `s09` after
implementation; raise if large torque residuals soften the block structure.

## Non-goals

- Any change to real→sim F/T conversion, filtering, or tare (H3's "do not tare
  simulated `ft_wrist`" warning stands).
- Lag estimation, cross-correlation, DTW, or hold-tail windowing.
- Removing `mean_hold_block_errors` or holdout gates — they use their own
  mean-hold path for diagnostics and are unaffected.
- Replacing Sinkhorn with direct `mean_hold_block_errors` as the CMA fitness
  (deferred; level-bag Sinkhorn is the chosen path).
- Diagnosing torque reachability (see Risks).

## Design

### 1. Level bags: `include_delta`

Add `include_delta: bool = True` to `build_transition_features_by_direction`
and `combine_transition_features`
(`apple_pick_sim/system_id/mmd_features.py`). When `include_delta=False`, a row
is the state alone rather than a transition:

```text
include_delta=True   (today)  row = [s_i, s_{i+1} − s_i]  + one-hots
include_delta=False  (new)    row = [s_i]                 + one-hots
```

With `hold_reduce="none", include_delta=False`, every **stable hold frame**
emits one row. This also removes an off-by-one: the existing frame-to-frame
branch zips `kept[:-1]` with `kept[1:]`, so it requires `kept.size >= 2` and
drops each hold's last frame; levels mode requires `kept.size >= 1` and keeps
all frames.

Per-direction row counts go from 3 to roughly 150–250 on `s09` (4 holds,
~250 replay frames at 30 Hz). Pooled train (~5 directions): ~750–1250 rows in
`23 + 4 + 8 = 35` dimensions.

`hold_reduce="mean"|"median"` with `include_delta=False` is permitted and emits
one level row per retained hold. **Not the target configuration** (see §Why not
hold mean).

Rationale for dropping `Δs`: within a quasi-static hold, frame-to-frame deltas
are dominated by sensor and solver noise and are timing-sensitive. Levels keep
the invariant stiffness content.

### 2. Categorical anchoring: `categorical_weight`

Add `categorical_weight: float = 1.0` to `transition_feature_scale`. Trailing
one-hot columns use `s_phys = 1 / w` instead of `1`, so crossing a direction
(or hold) costs `2w²`:

| `w` | Cost to cross one one-hot block |
| ---: | ---: |
| 1 (today) | 2 |
| 10 | 200 |
| 30 | 1800 |

The weight lives in the **scale**, not in feature construction, so raw
features stay interpretable 0/1 and all weighting continues to reside in
`NormalizationStats.std`, consistent with the fixed-scale spec.

A single weight covers both hold and direction one-hots. Splitting into two
knobs is deferred (YAGNI); anchoring holds matters as much as directions
because across-hold load progression carries stiffness information.

### 3. Normalization plumbing

`fit_gt_normalization` (`apple_pick_sim/system_id/mmd.py`) accepts
`include_delta` and `categorical_weight` and forwards both to
`transition_feature_scale`. Its mean-zeroing line generalizes from a hard
`2 * state_dim` to the block count:

```text
n_blocks = 2 if include_delta else 1
mean[n_blocks * state_dim :] = 0.0
```

`transition_feature_scale` likewise validates against
`n_blocks * state_dim` instead of `2 * state_dim`, and returns
`[state] * n_blocks + [ones(n_extra) / w]`.

`n_junctions` semantics are unchanged: callers must still pass the bag's
junction count so physical woody/bend columns are not mistaken for
categoricals.

### 4. Scoring config and CLI

`wasserstein._feature_kwargs` gains both keys and threads them through
`prepare_gt_wasserstein_context`, `prepare_gt_wasserstein_scoring_context`,
`score_candidate_wasserstein`, and `score_candidate_wasserstein_complete`.
Every `fit_gt_normalization` call site inside `wasserstein.py` passes the same
two values used to build the features — GT and candidate must share one
normalization contract.

`YoungsModulusScoringConfig`
(`apple_pick_gym/batched_envs/batched_sysid_cmaes.py`) gains
`include_delta: bool = True` and `categorical_weight: float = 1.0`.

`example_youngs_modulus_cmaes.py` and `example_youngs_modulus_sys_id.py` gain:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--include-delta` / `--no-include-delta` | `True` | Append the `Δs` half to each scored row |
| `--categorical-weight` | `1.0` | Divisor reciprocal for hold/direction one-hot columns |

Both are recorded in the `cmaes_report.json` `scoring` block alongside
`hold_aggregation`, `hold_id_onehot`, and `pool_directions`.

### 5. Defaults and compatibility

All new parameters default to current behavior. Existing datasets, reports,
and tests are unaffected until a run opts in. `pool_directions` stays `True`
by default.

## Risks and known limits

- **Score magnitudes are not comparable across this change.** Level bags in
  ~35 dimensions with ~1000 pooled rows produce a different `aggregate_sinkhorn`
  scale than 58-dimensional transition bags with ~15 rows. Task 9 numbers
  cannot be compared against post-change runs. Within-run comparisons remain valid.
- **Row count is not effective sample size.** Consecutive plateau frames are
  correlated; the gain is better-conditioned within-cell transport and
  transient/plateau separation, not i.i.d. sampling.
- **`categorical_weight` may need tuning.** If within-cell distances exceed
  `2w²`, block-diagonal structure softens and cross-direction leakage returns.
- **Cost.** Pooled bags grow from ~15 to ~1000 rows (~10⁶ pairs per candidate).
  Trivial on GPU; confirm `scoring_seconds` in the report.
- **Torque reachability unchanged.** This spec makes the objective trustworthy
  enough to distinguish model limits from misspecification; it does not produce
  torque the sim cannot generate.

## Tests

Written before implementation, in the repo's TDD order.

`apple_pick_sim/tests/test_mmd.py`

- `transition_feature_scale` accepts single-block width
  (`state_dim + n_extra`) when `include_delta=False`, and still rejects widths
  below one block.
- `categorical_weight` sets trailing one-hot divisors to `1/w` and leaves
  physical block scales untouched.
- `fit_gt_normalization(..., include_delta=False)` zeroes the mean only past
  one state block, and keeps GT-fit means on the physical columns.

`apple_pick_sim/tests/test_mmd_features.py`

- `hold_reduce="none", include_delta=False` emits one row per stable hold
  frame, **including** each hold's last frame, with width
  `state_dim + n_onehots`, and each row equal to the corresponding
  `build_state_matrix` row.
- Unstable frames are still excluded, and pre-weld (`phase == -1`) rows stay
  excluded after `strip_pre_weld_rows`.
- A single-frame hold contributes one level row (where the delta path
  contributed none).
- `hold_reduce="mean", include_delta=False` emits one row per retained hold
  (API coverage only; not the target config).

`apple_pick_sim/tests/test_wasserstein.py`

- Two bags identical except for the direction one-hot yield a Sinkhorn
  distance that grows with `categorical_weight` (assert the `w²` relationship
  via the exact singleton shortcut, which is deterministic and needs no GPU).
- GT context and candidate scoring share one `include_delta` /
  `categorical_weight` contract; a mismatch raises the existing feature-width
  error rather than silently comparing different layouts.

`apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py`

- `--no-include-delta` and `--categorical-weight` parse, reach
  `YoungsModulusScoringConfig`, and appear in the `cmaes_report.json`
  `scoring` block.
- Omitting both flags preserves today's config values exactly.

## How to verify

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd.py \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  -q -p no:launch_testing
```

Then an A/B on the reference dataset, holding `--cma-seed` and the direction
split fixed. Baseline is the shipped configuration; candidate is the target
configuration from §Decisions:

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09_k_frame \
  --output tmp/real_kp_e_cmaes_s09_levels_w30 \
  --direction-split-seed 17 --cma-seed 7 \
  --hold-aggregation none --no-include-delta --categorical-weight 30 \
  --viewer null --no-isolated-eval-waves --overwrite \
  --max-envs-per-batch 50 --settle-substeps 1000 --settle-quiet-every 100
```

Acceptance for this slice is **diagnostic**, not the science gate: compared
with the baseline run at the same seed:

- `aggregate_sinkhorn` falls as fitted Young's modulus moves toward real stiffness.
- Per-direction `mean_hold_force_err_n`, `mean_hold_woody_bend_rad`, and
  `mean_hold_woody_start_m` (from `mean_hold_block_errors`) move toward zero
  as the score improves — stiffness curve alignment, not just ‖F‖ ratio.
- Per-direction force/torque norm ratios should track the score without
  `--force-magnitude-weight` doing all the work.

If Sinkhorn falls while hold-mean block errors stay flat, anchoring did not take
or the sim model cannot reach the real stiffness curve; probe transport plans
per `(direction, hold)` before retuning the objective.
