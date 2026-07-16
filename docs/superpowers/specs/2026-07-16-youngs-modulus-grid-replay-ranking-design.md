# Young's Modulus Grid Replay and Ranking

**Date:** 2026-07-16  
**Roadmap slice:** V.5.2 prerequisite — GT recorded-action replay for Young's-modulus candidates

## Purpose

Replace the synthetic collection behavior in
`example_batched_youngs_modulus_collect_viz.py` with dataset-driven system
identification. For every structure in an existing `batched_sysid_v1` dataset,
replay all usable recorded directions over a predefined Cartesian grid of
Young's-modulus candidates, score each candidate with the validated Sinkhorn
Wasserstein loss, rank the candidates, and compare the result with the stored
ground truth.

Rename the entry point to
`apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`.
Ground-truth collection remains owned by
`example_batched_collect_sysid_data.py`.

## Scope

- Required source: one `batched_sysid_v1` dataset.
- Process all structures by default, with an optional structure subset.
- Sweep independent log10-E lists for primary, spur, and stem using their
  Cartesian product.
- Make exact-GT candidate insertion configurable and enabled by default.
- Keep secondary Young's modulus fixed at its stored GT value when present.
- Replay every usable, non-excluded recorded direction by default.
- Rank with the existing hold-phase, pooled-direction Sinkhorn contract.
- Optionally export each candidate replay as a mini dataset.
- Write machine-readable rankings and per-structure Plotly overlays.

This slice does not implement CMA-ES, infer geometry from observation-only
data, fit damping or geometry, or alter the existing bend-stiffness grid.

## Data flow

```text
batched_sysid_v1
  -> load structure GT params and recorded directions/actions
  -> Cartesian log10-E grid
  -> optionally insert exact GT E
  -> apply each candidate to the structure's GT base params
  -> candidate-major chunked batched replay
  -> hold transition features
  -> pooled-direction Sinkhorn score
  -> eligibility filter and ascending ranking
  -> ranking.json, overlays, optional replay mini-datasets
```

For each structure, `true_params_for_structure` provides the oracle base
parameters. `YoungsModulusCandidate.apply_to` changes only primary, spur, and
stem E. Geometry, topology, density, damping ratio, apple parameters, and
secondary E remain fixed.

The existing material conversion is authoritative: candidate E must flow
through `set_rod_youngs_modulus` / `rod_params_from_material`. This re-derives
bend stiffness and bend damping and re-derives beam-consistent axial terms,
while preserving fixed axial stretch overrides detected on the base rod.
Directly patching bend stiffness is incorrect.

## CLI

The renamed script is dataset-only.

Required:

- `--dataset`: source dataset directory.
- `--output`: report/output directory.

Grid and selection:

- `--log10-e-primary`
- `--log10-e-spur`
- `--log10-e-stem`
- `--structure-indices` (default: all)
- `--include-gt-candidate / --no-include-gt-candidate` (default: enabled)
- `--max-candidates`
- `--max-envs-per-batch`

Replay and output:

- Existing validated Wasserstein feature controls, defaulting to hold median,
  hold-ID one-hot, and pooled directions with direction one-hot.
- `--include-excluded` for diagnostics only.
- `--export-replays` to write candidate mini-datasets.
- `--max-overlay-candidates` to show the top-ranked candidates plus GT.
- `--fail-fast` to stop instead of recording a per-structure error.
- Explicit overwrite permission for existing outputs.

Synthetic topology sampling and standalone GT collection options are removed
from this script.

## Components

### Candidate evaluation library

Extend `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` with:

- GT Young's-modulus extraction from dataset structure parameters.
- Tight log-space candidate matching.
- Configurable exact-GT insertion without duplicates.
- Typed candidate-score and structure-ranking results.
- A thin `evaluate_youngs_modulus_candidates` adapter over existing replay and
  Wasserstein primitives.

The evaluator is intentionally reusable by the later CMA-ES ask/tell loop.

### Replay reuse

Reuse parameter-agnostic behavior from
`apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`:

- usable-direction resolution;
- recorded action and episode loading;
- candidate-major environment ordering;
- environment-budget chunking;
- replay collectors;
- instability accounting and disqualification;
- conversion to per-direction observations.

Generalize bend-specific annotations only where required by the existing
`candidate.apply_to(base_params)` runtime contract. Preserve original
direction IDs, including sparse IDs after exclusions, in scoring and exports.

### CLI orchestration

`example_youngs_modulus_sys_id.py` loads the dataset, iterates structures,
constructs candidates, invokes the evaluator, ranks results, and coordinates
reporting, overlays, and optional exports. Simulation and scoring logic stays
outside the example.

### Visualization

Extend `youngs_modulus_overlay_viz.py` with an in-memory replay adapter.
Visualization must not require exporting candidate datasets first. For large
grids, display the best-ranked candidates and include GT when available.

## Ranking and reports

Eligible finite candidates are sorted by ascending aggregate Sinkhorn loss,
with candidate index as the deterministic tie-breaker. Candidates exceeding
the existing replay instability threshold, missing required direction bags,
or producing non-finite scores are retained in the report but disqualified
from numeric ranking.

`<output>/ranking.json` contains:

- source dataset and scoring configuration;
- per structure: stored GT E/log10-E, fixed secondary E, every candidate,
  score, rank, instability fraction, disqualification reason, and GT marker;
- winning candidate and its log-space and relative error to GT;
- exact GT rank when insertion is enabled;
- aggregate GT-rank distribution, winner error summaries, and skipped
  structures.

Each structure also receives
`structure_XXX/youngs_modulus_overlay.html`. Optional candidate replay datasets
use the established
`structure_XXX/candidates/cYYY/` mini-dataset layout.

The JSON report remains authoritative if optional visualization or export
fails independently.

## Error handling

- Fail before simulation for a missing dataset, malformed/empty candidate
  lists, missing primary/spur/stem, or a candidate grid over the configured
  cap.
- Record and continue past a structure with malformed metadata or no usable
  directions unless `--fail-fast` is set.
- If every structure fails, return a non-zero exit status.
- Use tight log10-E tolerance for GT matching to avoid floating-point
  duplicates.
- Preserve manifest exclusion policy by default; `--include-excluded` is an
  explicit diagnostic override.
- Never renumber sparse source direction IDs during replay export.

## Testing

Follow red-green-refactor.

Fast tests:

1. Parser requires dataset/output, removes synthetic-only options, validates
   the Cartesian grid, and supports the GT toggle.
2. GT extraction and insertion work in log space, including no-GT mode.
3. Applying candidate E freezes non-E fields and secondary E while deriving
   physically consistent stiffness and damping.
4. Mocked replay verifies candidate-major ordering, chunking, all usable
   directions, action fidelity, and excluded/sparse direction handling.
5. Ranking covers GT preference, ties, no-GT mode, non-finite loss, and
   instability disqualification.
6. Report/export tests validate schema, top-K-plus-GT overlay selection, and
   sparse direction IDs.
7. Existing replay and Wasserstein suites remain green.

Manual CUDA smoke:

1. Collect a tiny dataset with
   `example_batched_collect_sysid_data.py`.
2. Run `example_youngs_modulus_sys_id.py` with a small E grid.
3. Verify successful exit, ranking schema, overlay generation, and optional
   replay mini-dataset readability.

The smoke reports GT rank but does not impose a universal rank-one CI
assertion on unhealthy samples.

## Success criteria

- Every selected dataset structure is evaluated independently against the
  same configured E-grid over all usable directions.
- Candidate E is converted into complete material-derived simulation knobs.
- The validated pooled hold Sinkhorn score drives ranking.
- GT insertion is configurable and GT comparison is explicit in the report.
- Secondary E and all non-E structure parameters remain fixed.
- Candidate replays can be exported without corrupting sparse direction IDs.
- The renamed dataset-only CLI and focused tests are documented and runnable.
