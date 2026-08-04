# Young's-modulus system identification

## Document status

| Field | Value |
| --- | --- |
| **Last reviewed** | 2026-08-04 |
| **Roadmap slice** | V.5.2 (support \(k_p\) retarget) |
| **Status** | Grid + CMA migrated to support \(k_p\) × spur/stem \(E\); acceptance pending Task 8 |

This is the canonical implementation reference for dataset-driven
Young's-modulus system identification. `docs/system_identification.md` owns the
larger protocol and roadmap context; `docs/sysid-transition-features.md` owns
the shared feature-vector definition.

## Behavior summary

`example_youngs_modulus_sys_id.py` replays recorded `batched_sysid_v1` actions
over a Cartesian grid of material candidates. A candidate contains support-joint
\(k_p\) (shared angular+linear, support \(\zeta=1\)) plus spur/stem Young's
modulus in Pa. `SupportKpYoungsCandidate.apply_to()` patches support FIXED-joint
\(k_p\)/\(k_d\) and sets spur/stem \(E\) through `set_rod_youngs_modulus()`,
which re-derives geometry-consistent VBD stiffness and damping while preserving
fixed axial overrides. Primary \(E\), secondary \(E\), geometry, topology,
density, non-support joint penalties, apple parameters, and other structure
state remain fixed.

Ground truth may be inserted into each structure-local grid. Eligible
candidates rank by ascending pooled Sinkhorn fitness, with local candidate
index as the deterministic tie-breaker. The command writes `ranking.json`,
one overlay per structure, and optional candidate replay datasets.

The grid remains a diagnostic and acceptance tool. The separate CMA-ES loop is
`example_youngs_modulus_cmaes.py`; it reuses the same candidate evaluator /
fused replay path and does not replace this command. Runnable collect → fit
commands: **README.md** → **CMA-ES sim-to-sim transfer (Young's modulus)**.
Notes: `docs/youngs-modulus-cmaes-implementation.md`; design:
`docs/superpowers/specs/2026-07-16-youngs-modulus-cmaes-loop-design.md`.
V.5.2 verification (Task 8) passed 2026-07-17.

## Complete Sinkhorn scoring

The default score uses hold medians, hold-ID one-hot features, and pooled
physical directions with a fixed-width direction one-hot:

```text
[state, delta_state, hold_id_onehot, physical_direction_onehot]
```

`prepare_gt_wasserstein_scoring_context()` prepares both:

- one pooled, GT-normalized bag used for optimizer fitness and ranking; and
- independently normalized physical-direction bags used only for diagnostics.

`score_candidate_wasserstein_complete()` requires every expected physical
direction before creating pooled fitness. Missing or empty direction bags
disqualify only that candidate. The structure and later candidates continue.
When pooling is disabled, the aggregate is the transition-count-weighted mean
of complete physical-direction losses.

In `ranking.json`:

- `aggregate_sinkhorn` is the pooled optimizer/ranking fitness;
- `per_direction_sinkhorn` is keyed only by physical source direction IDs;
- internal `POOLED_DIRECTION_KEY = -1` is never serialized as a physical
  direction;
- non-finite values are strict-JSON `null`, never `NaN` or `Infinity`; and
- an empty-bag disqualified GT row may have `aggregate_sinkhorn: null`,
  `per_direction_sinkhorn: {}`, and `rank: null`.

Collection exclusion and replay disqualification share
`DEFAULT_UNSTABLE_FRACTION_THRESHOLD = 0.25`. The comparison is strict:
fractions greater than 0.25 are unstable; exactly 0.25 remains eligible.

## Fused multi-structure replay

The default `--multi-structure-batch` scheduler flattens compatible work as:

```text
structure x structure-local candidate x physical source direction
```

`ReplaySlotKey(structure_idx, local_candidate_idx, direction_idx)` is the
stable identity across parameter application, action routing, per-environment
dataset initialization, observation collection, scoring, and reporting.
Source direction IDs may be sparse and are never renumbered.

One `ReplayCandidateBlock` contains all direction slots for one
structure/candidate pair. Physical chunks preserve block boundaries and obey
`--max-envs-per-batch`; a positive cap smaller than one complete block is
rejected. A zero cap is unbounded by this scheduler and remains subject to
available GPU memory.

Structures fuse only when topology, segment counts, ordered junction names,
frame count, direction layout, controller configuration, physical gripper
configuration, and observation shapes are compatible. Geometry, material
values, recorded weld poses, and candidate lists may differ by structure.
Fusion incompatibility uses scalar replay. With default error handling, a
fused runtime failure discards partial results for affected structures and
retries those structures individually; `--fail-fast` raises instead.

Fusing changes scheduling only. Candidate scoring, ranking, GT comparison,
overlays, exports, and errors remain structure-local. Use
`--no-multi-structure-batch` for scalar parity and diagnosis.

## Gate

`scripts/gate_youngs_modulus_sysid.sh` runs collection, unstable-episode
exclusion, grid replay/ranking, per-seed reporting, and final aggregation.
Defaults are seeds `0,1,2`, five structures, and five directions.

A seed passes when every requested structure is evaluable and GT ranks first
for a strict majority (at least three of five by default). The overall gate
passes only when every seed passes. This operational gate intentionally does
not require every healthy structure to rank GT first. Logs and strict-JSON
summaries are preserved on failure.

The separate CMA integrity gate (`scripts/gate_youngs_modulus_cmaes.sh`) does
not replace this ranking gate and does not impose a GT-error threshold. See
`docs/youngs-modulus-cmaes-implementation.md`.

## Code map

| Responsibility | Owner |
| --- | --- |
| Candidate mapping and structure-local scoring | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` |
| Stable slot planning, chunking, and fused replay | `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` |
| Per-environment episode initialization | `apple_pick_sim/system_id/batched_digital_twin_init.py` |
| Complete Sinkhorn context and scoring | `apple_pick_sim/system_id/wasserstein.py` |
| Dataset/grid/report orchestration | `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` |
| Overlay generation | `apple_pick_gym/youngs_modulus_overlay_viz.py` |
| Gate report validation | `apple_pick_gym/batched_envs/youngs_modulus_gate_report.py` |
| Multi-seed orchestration | `scripts/gate_youngs_modulus_sysid.sh` |

## Run and verify

Collect → grid verification (fusion is on by default; include GT support \(k_p\)
`1e4` in the grid when using the variance fixture):

```bash
uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 --max-steps 200 \
  --output tmp/support_kp_sysid_dataset --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --viewer null --dataset tmp/support_kp_sysid_dataset \
  --output tmp/support_kp_grid \
  --support-kp-values 1e3,1e4,1e5 \
  --log10-e-spur 8.0,9.5,11.0 \
  --log10-e-stem 8.0,9.5,11.0 \
  --include-gt-candidate --overwrite
```

Grid axis flags: `--support-kp-values` (physical) **or** `--log10-support-kp`
(exclusive); plus `--log10-e-spur` and `--log10-e-stem`. Primary-\(E\) axes
(`--log10-e-primary`) are removed.

Add `--no-multi-structure-batch` to run the scalar parity path.

Focused fast tests:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_batched_digital_twin_init.py \
  apple_pick_sim/tests/test_batched_heterogeneous_build.py \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_sysid_script.py \
  -q -p no:launch_testing
```

Full multi-seed gate (expensive):

```bash
bash scripts/gate_youngs_modulus_sysid.sh
```

The existing five-structure fused warm-up is diagnostic only. The slow
acceptance node
`test_batched_sysid_replay.py::test_two_structure_youngs_grid_fused_matches_independent`
currently reports a scalar/fused aggregate-Sinkhorn mismatch, so it is not part
of the green verification block above. Numerical parity, independent/fused
timing, low-cap parity, build count, and peak-memory acceptance remain pending.
