# Sys-ID scoring and bags handbook

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-08-17 |
| Code owners | `apple_pick_sim/system_id/mmd_features.py`, `apple_pick_sim/system_id/mmd.py`, `apple_pick_sim/system_id/wasserstein.py`, `apple_pick_sim/system_id/batched_trajectory_store.py`, `apple_pick_sim/system_id/real_to_batched_sysid.py` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H4 `docs/handbook-real-replay.md` (convert must emit this contract); H5 `docs/handbook-youngs-cma.md` (grid/CMA scores it); H2 `docs/handbook-variable-impedance.md` (action semantics only) |
| Archive specs | [Real/sim CMA feature alignment](superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md) — Implemented; [fixed-scale normalization](superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md) — Implemented; [median-hold features](superpowers/specs/2026-07-14-median-hold-features-design.md) — Implemented; [batched MMD grid](superpowers/specs/2026-07-06-batched-sysid-mmd-grid-design.md) — Historical; [MMD grid diagnostic](superpowers/specs/2026-06-22-mmd-grid-diagnostic-design.md) — Historical; [batched collection](superpowers/specs/2026-07-04-batched-sysid-collection-design.md) — Historical; [dataset dashboard](superpowers/specs/2026-06-22-sysid-dashboard-design.md) — Historical |

This handbook is the canonical contract for `batched_sysid_v1` bags and the
features scored from them. `docs/ROADMAP.md` owns delivery status and next work.

> **Warning — do not tare simulated `ft_wrist`.**
>
> Real compiled bags already isolate plant load as EMA−EMA:
> `ft_wrist` is loaded EMA minus the unloaded same-motion replay EMA.
> Convert that compiled column (do not subtract `ft_wrist_raw − ft_wrist_baseline`
> again). Convert keeps world-rotated `ft_wrist` as an unfiltered 30 Hz
> block-mean and writes a separate `ft_wrist_lpf` series (zero-phase
> Butterworth, default 10 Hz, then the same block-mean). Scoring uses
> `ft_wrist_lpf` when present. Never convert an uncorrected `*_robot.parquet`.
>
> Sim `ft_wrist` is already the plant TCP wrench. Gym copies
> `coupling_forces_cache` (`tcp_coupling_force`): stem-apple harvest plus
> optional explicit apple weight. EE gravity, tool inertia, and VIC effort do
> **not** enter that 6-vector on the `vic` / `vic_pose` path
> (`vic_use_joint_torques=True`; controller wrenches go to `joint_f`). A
> robot-only replay therefore reports `ft_wrist = 0`. Subtracting a simulated
> baseline is a no-op today and must not be added “for parity.” Do not copy
> the real convert LPF onto sim harvest.
>
> If harvest later becomes a sensor-like external wrench (tool weight and
> inertia included), this warning is void: then mimic the real unloaded
> same-motion tare, once per GT episode. Until then, leave candidate
> `ft_wrist` unsubtracted.

## 1. Three layers

Keep these layers separate; similarly named fields do not imply identical
storage or scoring semantics.

| Layer | Meaning | Important boundary |
| ----- | ------- | ------------------ |
| **Runtime obs** | Gym or scene observation dictionaries produced while simulation runs. | May include debug/rebuild fields such as `woody_part_end_pos`; see `docs/gym-observation-contract.md`. |
| **Bag** | Arrays loaded from a `batched_sysid_v1` episode Parquet plus episode metadata. | Carries replay inputs, including 6D or 19D `action`; trajectory frames do not carry woody ends. |
| **Score vector** | Numeric columns assembled by `mmd_features.build_state_matrix` and its transition builders. | Uses `STATE_VECTOR_FIELDS`; `action`, quaternions, joints, raw F/T, and woody ends are not scored. |

H2 owns the meaning of a 6D `vic` action versus a 19D `vic_pose_v1` action.
H4 owns real-log conversion. H5 owns candidate selection. This handbook owns
the common bag-to-score boundary.

## 2. `batched_sysid_v1` layout

`batched_trajectory_store.SCHEMA_VERSION` names this layout:

```text
<dataset>/
  manifest.json
  episodes/
    s00_d00.parquet
    s00_d01.parquet
    ...
```

### Manifest

`batched_trajectory_store.write_manifest` writes dataset provenance,
`collection` settings, light `structures[]` summaries, and an `episodes[]`
catalog. An episode entry identifies `structure_idx`, `direction_idx`,
`env_idx`, deterministic `filename`, `episode_id`, pull direction, frame count,
and optional exclusion state.

### Episode metadata

`batched_trajectory_store.EPISODE_METADATA_KEYS` defines the serialized schema
metadata allow-list. Replay-relevant entries include reset TCP/apple poses and
robot joints, fixture and weld fields, `junction_names`, physical parameters,
control rate, action declarations, and convert `ft_filter` provenance:

- `action_dim`: 6 or 19;
- `action_layout`: for example `vic_pose_v1`;
- `action_compatible_with_vic_twist`; and
- `action_semantics`.

`real_to_batched_sysid.export_real_episode_to_batched_dataset` packs a
pose-controlled real log as
`[target_pos(3), target_quat_wxyz(4), Kp(6), Kd(6)]`, sets
`action_dim=19` and `action_layout="vic_pose_v1"`, and does not reinterpret the
logged pose-control wrench as a twist.

### Frame table

`batched_trajectory_store.BATCHED_REQUIRED_FRAME_COLUMNS` requires
`step_idx`, `phase`, `excitation_type`, `excitation_direction`, `action`,
`tcp_velocity`, and `ft_wrist`. Scoring-aligned writers also provide
`hold_number`, `tcp_pos`, `apple_pos`, `stable`, and dynamic
`woody_start__<junction>` columns. Real convert also writes optional
`ft_wrist_lpf` (not required on sim bags). `hold_number` remains loader-compatible
optional data for older bags; missing values fall back to inferred segment
indices or `-1` where the collector needs a sentinel.

Each episode file represents one structure/direction pair, so trajectory rows
do not repeat `episode_id` or `dir_idx`. Higher-level loaders attach the
direction identity from the manifest/file index before score construction.
`BatchedSysIdDataset.load_episode_obs_arrays` accepts only action matrices with
width 6 or 19.

### Woody geometry and the pre-weld exception

Trajectory rows persist starts only. The batched collector's
`_woody_obs_for_junction_names` removes `woody_part_end_pos` when selecting the
CMA pair, and converted real rows never add ends. A reconstruction-only
pre-weld row may carry prefixed woody-end columns at `step_idx=-1`; it is not a
trajectory or scoring row. `BatchedSysIdDataset.load_episode_obs_arrays` tolerates that
column solely so digital-twin initialization can address the pre-weld row.

## 3. `STATE_VECTOR_FIELDS`

`mmd_features.STATE_VECTOR_FIELDS` fixes score-time order. For the production
CMA pair `J=2`, `mmd_features.CMA_WOODY_JUNCTIONS` is
`("primary_spur", "spur_stem")` and the state width is 26:

| Offset | Field | Dimensions | Meaning |
| ------ | ----- | ---------- | ------- |
| 0:6 | `ft_wrist` | 6 | World-frame force then torque, env-on-robot, about TCP. Real GT uses `ft_wrist_lpf` when that column is present; candidate replay always uses live `ft_wrist`. |
| 6:12 | `tcp_velocity` | 6 | Linear then angular TCP velocity |
| 12:15 | `tcp_pos` | 3 | TCP world position |
| 15:18 | `apple_pos` | 3 | Apple world position |
| 18:24 | `woody_part_start_pos` | `3J=6` | Starts flattened in `junction_names` order |
| 24:26 | `woody_bending_angles` | `J=2` | Rest-relative spur then stem chord angles |

In general, \(D_s = 18 + 4J\). For the CMA pair the bending chords are:

- `primary_spur`: `start[spur_stem] - start[primary_spur]`;
- `spur_stem`: `apple_pos - start[spur_stem]`.

`mmd_features.build_bending_angles` compares each normalized chord with its
frame-0 direction and forces frame 0 to exactly zero.

`mmd_features.REQUIRED_ARRAY_KEYS` still includes `action`. This validates a
complete replay bag and supplies frame count, but `action` is deliberately
absent from `STATE_VECTOR_FIELDS` and `build_state_matrix` output.

## 4. Feature alignment contract

Real GT and simulated candidates must enter scoring in the same frames and
with the same geometry:

1. **F/T conversion.**
   `real_to_batched_sysid.world_wrench_from_ee_logged` applies
   \(F_W=R_{W,TCP}F_{EE}\) and
   \(\tau_W=R_{W,TCP}\tau_{EE}\). Conversion applies this to `ft_wrist` and
   `raw_ft_wrist`, without another sign flip or lever-arm transport.
   World `ft_wrist` is then block-mean downsampled unfiltered. Convert also
   writes `ft_wrist_lpf`: the same world wrench after a 10 Hz `filtfilt`, then
   the same block-mean to `collection.control_hz`. Scoring
   (`mmd_features.scored_ft_wrist` / `build_state_matrix`) uses `ft_wrist_lpf`
   when present; `raw_ft_wrist` remains diagnostic only, unfiltered.
2. **No sim filter and no sim tare.**
   Candidate replay uses its live world-frame plant harvest as `ft_wrist`.
   There is no score-time F/T transform, no simulated EMA/LPF, and no unloaded
   baseline subtraction. See the warning at the top of this handbook.
3. **Woody points.**
   `real_to_batched_sysid.tag_poses_to_cma_woody` maps Branch and Spur tag
   translations to the two `CMA_WOODY_JUNCTIONS` starts and maps the Apple tag
   translation to `apple_pos`. It requires all three tag-pose columns.
4. **Hold identity.**
   `_scalar_hold_number` prefers scalar `hold_index`; it can decode an older
   one-hot-like `hold_number` only as a fallback. Parquet stores one scalar.
   `_one_hot_hold_id` creates categorical columns only during score feature
   construction.
5. **No action score.**
   The full action remains in each bag to drive an equivalent replay. It is
   not evidence about candidate plant fidelity and is not scored.

Other non-scored fields include `woody_part_force`, TCP/apple quaternions,
robot joints, camera calibration, raw F/T, and unfiltered real `ft_wrist`
when `ft_wrist_lpf` is present.

## 5. Normalization

`mmd.fit_gt_normalization` computes a mean from GT transition rows only, then
uses fixed physical divisors from
`mmd_features.transition_feature_scale`. Candidate statistics never affect the
fit:

\[
x_\mathrm{norm} = \frac{x-\mu_\mathrm{GT}}{s_\mathrm{phys}}.
\]

For one `J=2` state half,
`mmd_features.STATE_VECTOR_PHYS_SCALE` is:

| State block | Scale per component |
| ----------- | ------------------- |
| F/T force | 2 N |
| F/T torque | 0.5 N·m |
| TCP linear velocity | 0.02 m/s |
| TCP angular velocity | 0.02 rad/s |
| TCP position | 0.005 m |
| Apple position | 0.005 m |
| Woody start XYZ | 0.005 m |
| Bending angle | 0.05 rad |

`transition_feature_scale` repeats this table for both \(s\) and \(\Delta s\).
Trailing hold/direction one-hots have mean zero and scale one, so their raw
0/1 values pass through. The `NormalizationStats.std` attribute is a
compatibility name: it stores these fixed divisors, not GT standard deviation.
Callers must pass the bag's junction count so additional physical columns are
not mistaken for categorical extras.

## 6. Transition bags

`mmd_features.build_transition_features_by_direction` creates hold-only rows

\[
v=[s,\Delta s]
\]

and optionally appends source-hold and direction one-hots.

- **Frame-to-frame (`use_median=False`)**: within each contiguous full hold,
  use consecutive retained frames. `stable=False` removes a sample but does
  not split the hold, so a transition can bridge dropped frames.
- **Hold-to-hold median (`use_median=True`)**: compute one state median from
  stable frames in each full hold, then emit
  `[s_i, s_{i+1}-s_i]` for consecutive retained holds.

There is no latter-half burn-in in these feature builders. Quasi-static
diagnostic windows elsewhere are a different layer.

`combine_transition_features` concatenates episode rows by physical
`dir_idx`. The pooled Sinkhorn path appends a fixed-width direction one-hot,
concatenates all physical directions under internal
`wasserstein.POOLED_DIRECTION_KEY`, and fits one GT normalization on that
pooled bag. Complete scoring also maintains independently normalized
per-direction bags for diagnostics. Missing expected directions disqualify the
candidate instead of silently changing the comparison set.

## 7. Scorers

`wasserstein.sinkhorn_distance` is the production scorer used by the
Young's/CMA path. It evaluates GeomLoss
`SamplesLoss("sinkhorn", p=2, blur=1.0)` on normalized bags and has an exact
singleton-bag shortcut. `prepare_gt_wasserstein_scoring_context` prepares the
pooled optimizer context plus physical-direction diagnostics;
`score_candidate_wasserstein_complete` enforces direction completeness and
returns pooled fitness (or a transition-count weighted per-direction aggregate
when pooling is disabled).

`mmd.biased_mmd2` remains available for offline diagnostics. Its helper path in
`apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` is stale and is not the
production optimization contract. Do not infer current scoring behavior from
the historical MMD grid specs.

## 8. Replay alignment notes

Sim-collected datasets can include a settled pre-weld reconstruction row with
`step_idx=-1`, `phase=-1`. `batched_sysid_mmd_grid.strip_pre_weld_rows` removes
it before building recorded action tensors, loading score episodes, or
comparing replay frames. After stripping, recorded row \(i\) is the post-step
observation for replay action \(i\). Metric helpers fail fast if a leading
pre-weld row is still present.

A structure has one weld/grasp point shared by all directions and all
candidates. Reusing direction 0's representative weld metadata is intentional,
not a GT-parameter shortcut. Build parameters and initial dynamic state are
independent choices: `--infer-params` selects observation-derived build
parameters, while `--use-snapshot` is a separate privileged-state option.

Grid CLI defaults: `--use-median`, `--hold-id-onehot`, and `--pool-directions`
on (full hold windows). Console `--score-mse` with `--use-median` uses
`trajectory_paired_hold_median_mse`. Wasserstein with `--use-median` uses
hold→hold median bags. Deprecated `--mse-hold-aggregation` still maps onto
`--use-median`; `--mse-hold-latter-half` is a no-op. By default the grid skips
manifest episodes with `excluded: true`; pass `--include-excluded` only for
debug. Replay reads `manifest.collection.control_hz` (fallback
`CONTROL_HZ = 30.0`).

## 9. Legacy single-env Parquet

The older layout under `apple_pick_sim/system_id/trajectory_store.py` is:

```text
<output_dir>/
  metadata.parquet
  frames/<episode_id>.parquet
  initial_states/<episode_id>.npz   # optional privileged snapshot
```

It repeats `episode_id`/`dir_idx`, stores 6D twist actions, and may store both
woody starts and ends. Those properties must not be copied into
`batched_sysid_v1` scoring bags. Observation-first replay is the default;
`--use-snapshot` is sim-to-sim debug only. Use `ApplePickReplay-v0` /
`example_gym_replay.py` for this path.

## 10. Code map and tests

| Owner | Contract |
| ----- | -------- |
| `mmd_features.STATE_VECTOR_FIELDS`, `STATE_VECTOR_PHYS_SCALE`, `REQUIRED_ARRAY_KEYS`, `CMA_WOODY_JUNCTIONS`, `scored_ft_wrist` | State order, scale, bag validation, CMA geometry, real LPF wrench |
| `mmd_features.build_state_matrix`, `build_transition_features_by_direction`, `combine_transition_features` | State and transition rows |
| `mmd.fit_gt_normalization`, `apply_normalization` | GT mean plus fixed physical scale |
| `wasserstein.prepare_gt_wasserstein_scoring_context`, `score_candidate_wasserstein_complete`, `sinkhorn_distance` | Production Sinkhorn context and score |
| `batched_trajectory_store.BatchedEpisodeWriter`, `BatchedSysIdDataset` | Batched Parquet write/load contract |
| `real_to_batched_sysid.export_real_episode_to_batched_dataset` | Real F/T, hold, woody, and `vic_pose_v1` conversion |

Key tests:

- `apple_pick_sim/tests/test_mmd_features.py` — exact state order and width,
  CMA chords, no required woody end, 19D action support, stable masks,
  median-hold rows, and one-hot placement.
- `apple_pick_sim/tests/test_mmd.py` — GT-only mean, fixed divisors,
  non-centered one-hots, variable junction counts, and diagnostic MMD.
- `apple_pick_sim/tests/test_wasserstein.py` — per-direction and pooled
  contexts, completeness, singleton bags, and direction one-hots.
- `apple_pick_sim/tests/test_real_to_batched_sysid.py` — F/T rotation,
  EMA−EMA no re-tare, unfiltered `ft_wrist` plus 10 Hz `ft_wrist_lpf` + 30 Hz
  block-mean, last-sample phase, two-start tag mapping, scalar holds, no
  trajectory ends, and 19D pose action packing.
- `apple_pick_sim/tests/test_batched_trajectory_store.py` and
  `apple_pick_gym/tests/test_batched_sysid_collect.py` — storage and collector
  round trips.

### How to verify

From the repository root:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py apple_pick_sim/tests/test_mmd.py -q -p no:launch_testing
```
