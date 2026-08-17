# Real multi-structure Young's-modulus CMA-ES

| Field | Value |
| ----- | ----- |
| **Status** | Approved design; not implemented |
| **Canonical living doc** | `docs/handbook-youngs-cma.md` (H5); convert contract in `docs/handbook-real-replay.md` (H4) |
| **Date** | 2026-08-14 |
| **Roadmap** | M4.0 bit 3 — slices 3–4 (real `robot_replay` → grid ranking → CMA) |
| **Extends** | `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md` (slice 1 shipped), `docs/superpowers/specs/2026-07-16-youngs-modulus-cmaes-loop-design.md` (sim-sim loop), `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md` |
| **Reference data** | `tmp/final_data/s09/`, `tmp/final_data/s11/` — two trees, eight pull directions each |

## Purpose

Fit \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\) independently
for each real tree, using recorded `vic_pose` replay and pooled hold-phase
Sinkhorn, with one pycma optimizer per structure advancing in synchronized
generation waves.

Today this cannot run. Convert emits only a 1×1 dataset, the grid refuses more
than one structure on `vic_pose`, the CMA entry point never selects the real
builder and fails real structures for missing sim ground truth, and several
manifest fields the scoring path depends on are never written.

## What "multi-structure parallelism" means here

Physical chunks never mix structures, by deliberate design:

> Physical chunks never mix distinct `structure_idx` values. Cross-structure
> heterogeneous batches can collapse per-direction trajectories, so even an
> unlimited `max_envs_per_batch` keeps one structure per chunk.
> — `chunk_replay_candidate_blocks`, `batched_sysid_multi_replay.py:304`

So GPU parallelism is **population × directions within one structure** (for
example 15 candidates × 8 pulls = 120 envs in one Newton batch, well under the
25000-env default cap). Structures are sequential chunks inside one process,
one loop, and one report. `population_size` in `CMA_SEARCH_PARAMS` is the
GPU-utilization knob.

Cross-structure co-residency is a **non-goal**: `fruiting_base_pos`, weld
SE(3), and bootstrap joints are batch-scalar at build time, and
`_assert_uniform_topology` rejects differing segment counts.

## Locked decisions

| Topic | Choice |
| --- | --- |
| Parallelism | Orchestration-level: one pycma per structure, synchronized waves, one structure per physical chunk |
| Phenotype | Unchanged: \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\); primary \(E\) fixed |
| Dataset shape | One tree folder → one structure; `dNN` → directions; multiple trees merged into one multi-structure `batched_sysid_v1` |
| Control rate | Convert decimates 1 kHz F/T with filtfilt + block-mean to 30 Hz (discrete fields: last sample per window). 15 Hz nearest-copy is superseded. |
| Structure geometry | One canonical geometry per tree; per-pull post-grasp state stays per direction |
| Frame alignment | Bags keep true lengths; padding happens only in the replay drive tensor; replay is truncated back before scoring |
| Ranking GT | Real logged bags; no sim-oracle GT candidate |
| Scope | Convert + replay fixes + grid + CMA, delivered in that order |

## Verified facts this design rests on

Measured on `tmp/final_data` at design time; re-check if the data is regenerated.

1. **Raw logs are 1 kHz.** `dump.control_hz = 1000.0`, `dump.num_directions = 8`,
   `dump.direction_index` ∈ 0–7, four holds (`hold_index` 0–3). The only real
   dataset that replays today (`tmp/real_batched_s09_d00`) was built from a
   hand-made 15 Hz file, 102 frames.
2. **Substep cost depends on episode duration, not the frame rate.**
   `substeps_per_step = round(1/(control_hz · sub_dt))` with `sub_dt = 1/1800`,
   so a ~6.8 s episode is ~12k substeps whether logged at 15 Hz or 1 kHz.
   Decimation reduces host overhead only.
3. **Per-tree geometry is effectively constant across directions.** Base pose
   deviates at most 0.29 mm (s09) and 0.78 mm (s11) from the per-tree mean; rod
   lengths and radii are identical to the micron. Distinct `params_fingerprint`
   values across directions are a hashing artifact, not geometric disagreement.
4. **The two trees genuinely differ only in the spur.** s09: length 0.120 m,
   radius 2.5 mm. s11: length 0.100 m, radius 5.9 mm. Primary and stem come
   from the shared catalog.
5. **Episode lengths differ per direction** (s09: 6565–6711; s11: 6231–6964)
   and every episode's final hold runs to its last frame, so truncating to the
   shortest clips real hold data (up to ~44% of s11's last hold).
6. **Replay bags carry no `hold_number`.** It is written by
   `batched_sysid_collect.py` only; the replay collectors in
   `batched_sysid_mmd_grid.py` emit `phase`, `dir_idx`, `stable`, and the state
   columns, but not `hold_number`.

## Slice 1 — convert

**Owner:** `apple_pick_sim/system_id/real_to_batched_sysid.py`,
`robot_replay/convert_real_to_batched_sysid_metadata.py`

### Decimation

> **Superseded for F/T and velocity (2026-08-17).** Convert no longer uses
> nearest-timestamp 15 Hz copies. Default `--control-hz` is **30 Hz**. Scored
> continuous signals (`ft_wrist`, `tcp_velocity`, diagnostic `raw_ft_wrist`)
> are world-rotated, zero-phase Butterworth low-passed (default 10 Hz), then
> block-mean downsampled. Discrete/drive fields (`action`, poses, `phase`,
> hold, joints, woody) still take one sample per window (the last), so
> averaging does not smear pull/hold boundaries. See H3/H4.

Add a target control rate (default 30 Hz). Convert resamples the 1 kHz log
instead of inheriting its rate through `_resolve_control_hz`. The decimated
rate is written to `collection.control_hz` and episode metadata.

### Tree folder → one structure

Iterate `sXX-dNN.parquet` in a folder. `direction_idx` comes from
`dump.direction_index` (currently hardcoded to `0`, `real_to_batched_sysid.py:425`).
Episodes are written as `episode_filename(structure_idx, direction_idx)`
instead of always `s00_d00` (`real_to_batched_sysid.py:790`).

### Canonical structure geometry

All directions of one structure share build geometry:

- assert rod geometry is identical across directions (measured spread: zero);
- set `fruiting_base_pos` to the mean across directions;
- **fail conversion** if any direction's base pose deviates more than
  `--base-pos-tolerance-m` (default 0.005) from that mean, which would mean the
  tree moved between pulls;
- write the same `fruiting_system_params` and `params_fingerprint` on every
  direction of the structure.

Per-pull fields stay per direction: post-grasp apple and TCP SE(3),
`initial_robot_joint_q`, controller gains, `pull_direction`, `episode_id`.

### Multi-tree merge

Assign `structure_idx` by lexicographic folder name (`s09` → 0, `s11` → 1) and set
`env_idx = structure_idx · num_directions + direction_idx`. Write a
sim-collect-shaped manifest in one pass (today `write_manifest` is called once
with `num_structures: 1`, `num_directions: 1` and overwrites everything):

- `collection.num_structures`, `collection.num_directions` reflect reality;
- `structures[]` has one entry per tree;
- `episodes[]` carries each episode's **true** `n_frames`;
- `collection.max_steps` is the padded max (the value replay will actually step).

### Manifest fields the scoring path requires

These are read by the CMA path and are absent from real datasets today. All
must be written at convert time:

| Field | Why | Consequence if missing |
| --- | --- | --- |
| `collection.n_holds` | Fixes the hold one-hot width | See below — silent feature-width mismatch |
| `collection.sim_config` | `support_joint_zeta_from_dataset` reads `sim_config.joint_damping_ratio` for the support \(k_d = \zeta \cdot 2\sqrt{kI}\) that pairs with the free \(k_p\) dimension | Falls back to `SUPPORT_JOINT_ZETA_FALLBACK = 0.5`; equals the fixture value today, so latent rather than broken |
| `collection.topology_seed` | Scene DR seed | Grid defaults to `collection.seed` (0), CMA hardcodes 42 — the two CLIs disagree |

Write the **whole** `sim_config` block through `sim_config_to_manifest_dict`,
built from `real_replay_sim_config`, exactly as sim collection does. A partial
block containing only `joint_damping_ratio` would leave the replay-time
manifest-versus-env comparison reporting spurious mismatches.

**`n_holds` is a correctness blocker, not a cosmetic gap.** With
`scoring.n_holds = None`, `build_transition_features_by_direction` infers the
one-hot width per bag (`mmd_features.py:599`). Recorded bags have
`hold_number` and infer 4; replay bags have none and fall back to
`max(len(medians), 1)`, the number of hold segments actually detected. A
candidate that loses one hold produces a 3-wide one-hot against a 4-wide GT.
Sim datasets are unaffected because sim collection writes `n_holds`.

### CLI

Add `--input-dir` (tree folder), a multi-folder merge mode, `--control-hz`, and
`--base-pos-tolerance-m`. The existing single-file 1×1 path keeps working.

## Slice 2 — replay layer

**Owner:** `apple_pick_gym/batched_envs/real_batched_replay_build.py`,
`batched_sysid_multi_replay.py`, `apple_pick_sim/system_id/batched_digital_twin_init.py`,
`batched_sysid_mmd_grid.py`

Real replay has only ever run one direction, so these gaps have never been hit.

### Per-direction weld pose and gripper

`make_real_replay_build_env_fn` closes over a single `episode_meta` and calls
`apply_logged_post_grasp_se3_to_cable(cable, dict(episode_meta), layout=layout)`
(`real_batched_replay_build.py:289`), which writes that one logged apple/TCP
pose into **every** world. With eight directions in a batch, all eight would
weld at direction 0's grasp.

Similarly `ReplaySlot.gripper` is one gripper shared by every direction slot
(`batched_sysid_multi_replay.py:270`), resolved from direction 0 in
`prepare_youngs_modulus_structure`.

Both become per slot. The natural home is the per-slot initializer
`initialize_batched_env_from_episode_sources`, which already loops
`ReplayEpisodeSource` values and loads each episode's metadata to set per-env
joints and TCP targets.

### Structure-keyed builder

`fruiting_base_pos`, `bootstrap_joint_q`, and `control_hz` are closure
constants in the real builder. Because chunks are always single-structure,
these become structure-keyed rather than per-env: pass the chunk's
`structure_idx` into `build_env_fn` (which today receives only `num_envs`,
`per_env_params`, `per_env_grippers`, `max_episode_steps`) and resolve them
from that structure's metadata.

### Frame alignment

Bags keep their recorded lengths. Padding lives only in the drive path:

1. Relax the per-structure frame-count equality in `_validate_request`
   (`batched_sysid_multi_replay.py:167`) to take the max, and let the fusion
   signature carry that max so structures still fuse.
2. `build_recorded_actions_tensor` pads each direction to the max by repeating
   its **last logged action** instead of raising
   (`batched_sysid_mmd_grid.py:1388`). The final hold runs to the end of every
   episode, so a repeated last action continues that hold — physically
   consistent.
3. Truncate each replay episode back to its own recorded length before feature
   building, so padded frames can never enter a Sinkhorn bag.

Padded frames are therefore unrepresentable in any bag, rather than excluded by
convention.

## Slice 3 — grid

Remove the single-structure guard for `vic_pose`
(`example_youngs_modulus_sys_id.py:920`) once slice 2 lands. The grid stays the
diagnostic and acceptance path and gains nothing else.

## Slice 4 — CMA

**Owner:** `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py`

1. **Select the real builder** when `dataset_declares_vic_pose` or
   `--controller-mode vic_pose`. Today the CLI always calls the sim
   `_make_build_env_fn` (line 577) and `build_sim_config` (line 584).
2. **Stop requiring sim ground truth.**
   `state.gt_candidate = gt_support_kp_youngs_candidate_from_structure(...)`
   marks real structures `failed` at init (lines 615–622). The evaluator
   already sets `gt_candidate = None` on the real branch
   (`batched_sysid_cmaes.py:771`); the CLI must match, and the report omits GT
   error diagnostics. `_gt_error_diagnostics` already returns `None` for a
   missing GT, and `aggregate_fitted_youngs_modulus_stats` already guards the
   empty-GT case, so no library change is needed.
3. **Add `--controller-mode` and `check_action_semantics`**, which the grid has
   and CMA does not, so a mis-tagged log cannot be replayed as a twist.
4. **Widen the search box for real data.** `_CMA_SEARCH_LOG10_LOWER = [2.0, 8.0, 8.0]`
   constrains spur and stem to \(\ge 10^8\) Pa, but the real-world proxy
   fixture puts them at **2.5e7 – 1.0e8 Pa**, so the entire plausible range sits
   at or below the box floor and every fit would clamp at the bound. The grid
   sweeps already used \(10^7\)–\(10^9\). Lower the spur/stem bound to at least
   \(\log_{10} E = 7\) for real runs. Support \(k_p\) bounds are unchanged.

The optimizer contract itself is untouched: independent pycma per structure,
`ask()` → fused wave → `tell()`, per-structure failure isolation, explicit
final-mean wave, overlays and `cmaes_report.json`.

## Error handling

Structure-local failures — empty bags, instability disqualification, fusion
incompatibility, chunk failure, ask/tell failure — keep other structures
running, as they do today. `--fail-fast` preserves the strict path. Convert
fails loudly rather than guessing on: base-pose spread over tolerance,
mismatched rod geometry within a tree, missing `dump.direction_index`, and
duplicate `(structure, direction)` pairs.

## Tests

Written before the corresponding implementation, per
`.cursor/rules/test-driven-development.mdc`.

**Convert**

- tree folder → correct `episodes/sSS_dDD` layout, manifest counts, and
  `direction_idx` taken from `dump.direction_index`;
- decimation to the target rate yields the expected frame count and
  `collection.control_hz`;
- geometry canonicalization: identical params across directions, mean base pose;
- base-pose spread over tolerance fails conversion with a named error;
- `n_holds`, `sim_config.joint_damping_ratio`, and `topology_seed` are present;
- merge of two trees → `num_structures = 2` with per-direction metadata intact.

**Replay**

- unequal direction lengths produce a drive tensor padded to the max with the
  last action, while recorded bags keep their true lengths;
- replay episodes are truncated to recorded length before feature building
  (a padded frame never reaches a bag);
- two directions in one batch receive different weld poses and grippers;
- two structures with different base poses resolve per structure through
  `build_env_fn`.

**Scoring**

- with `n_holds` present, recorded and replay bags produce identical hold
  one-hot widths even when a candidate drops a hold.

**CMA**

- a `vic_pose` dataset selects the real builder and does not fail a structure
  for missing GT;
- the report omits GT diagnostics and still aggregates cross-structure stats.

**Regression**

- sim-sim grid and CMA behavior unchanged (twist `vic`, GT insertion, existing
  gates).

## Acceptance

From the repository root:

1. Convert `tmp/final_data/s09` and `tmp/final_data/s11` into one
   two-structure, eight-direction dataset at 15 Hz.
2. Run the grid on it as a smoke test — both structures rank, overlays render.
3. Run a short CMA (small `max_generations`) and confirm `cmaes_report.json`
   holds two independent fitted vectors with no GT diagnostics and no clamping
   at a search bound.

Exact `uv run` invocations are added to `README.md` and `docs/ROADMAP.md`
validation commands as part of the implementation, per
`.cursor/rules/readme-runtime-verification.mdc`.

## Non-goals

- Cross-structure co-residency in one Newton batch.
- Changing the phenotype, Sinkhorn features, normalization, or the
  one-structure-per-chunk policy.
- Migrating gym collect, MMD, or sim-sim CMA off twist `vic`.
- Updating domain-randomization fixtures from the fitted values.
- Held-out validation (V.5.3) and multi-GPU sharding.

## Open risks

1. **Throughput at 120 envs is unmeasured.** The only data point is 27 envs at
   109 s replay plus 32 s build/settle, on ~12k substeps. Whether a 120-env
   chunk costs roughly the same (GPU latency-bound) or ~4× more
   (throughput-bound) sets the wall clock for 20 structures. Measure the
   replay-time-versus-env-count curve before committing to a population size.
2. **Logged grasp geometry is already inconsistent with the sim apple.**
   Building metadata on this data warns
   `post-grasp data mismatch: |tcp−apple| = 0.0618 m vs apple radius r = 0.0400 m`.
   It is diagnostic only, but it will affect fit quality; inspect overlays
   before trusting fitted numbers.
3. **Fit quality on real data is unproven.** Sim-sim already showed mean bias on
   support \(k_p\) and spur \(E\). Real data adds model mismatch; the first
   fits should be treated as diagnostics, not calibrated values.
