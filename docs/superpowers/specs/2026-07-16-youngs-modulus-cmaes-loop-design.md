# Young's-Modulus CMA-ES Loop Design

**Date:** 2026-07-16  
**Roadmap slice:** V.5.2 — continuous Young's-modulus calibration  
**Status:** Approved; implementation pending

## Purpose

Add a separate CMA-ES entry point that runs an independent pycma optimization
for each selected dataset structure. Each optimizer searches for primary,
spur, and stem Young's modulus using the existing recorded-action replay and
pooled hold-phase Sinkhorn Wasserstein objective.

The output is one fitted three-component Young's-modulus vector per structure.
The implementation also retains optimizer diagnostics and cross-structure
statistics. Turning those fits into domain-randomization ranges is deferred.
The implemented Cartesian-grid command remains the diagnostic and acceptance
path documented in `docs/youngs-modulus-sysid.md`.

## Scope

- Add a separate `example_youngs_modulus_cmaes.py` command; do not add a
  `grid|cmaes` mode switch to the existing grid command.
- Optimize
  \(\log_{10}([E_\mathrm{primary}, E_\mathrm{spur}, E_\mathrm{stem}])\).
- Run one independent CMA-ES optimizer per selected structure.
- Evaluate every generation's population in parallel with the existing
  candidate-by-direction batched replay.
- Use pooled hold-phase Sinkhorn Wasserstein as the fitness.
- Initialize at the midpoint of fixture-derived log-space bounds, not at GT.
- Stop on either a configured generation cap or pycma's native convergence
  criteria.
- Explicitly replay and score the final distribution mean, and treat it as the
  fitted estimate.
- Preserve the best sampled candidate and final covariance as diagnostics.
- Produce a final-mean-versus-recorded-GT overlay for each successful structure.
- Add pycma (`cma`) to the Gym tooling dependencies.

## Non-goals

- Updating domain-randomization fixtures from the fitted values.
- Choosing a final domain-randomization range policy. Raw min/max is reported
  only as a diagnostic; robust quantiles, margins, and fit validation remain a
  later decision.
- Fitting damping ratio, density, geometry, apple parameters, or secondary E.
- Sharing one optimizer or one fitted E vector across structures.
- Changing replay semantics, Wasserstein features, instability policy, or the
  material-to-VBD conversion.
- Changing or removing the Cartesian-grid ranking command, candidate replay
  export, or top-ranked grid overlays.
- Held-out validation, real-data acceptance criteria, or new excitation types.

## Architecture

The existing library boundary is extended:

- `example_youngs_modulus_sys_id.py` remains the Cartesian-grid diagnostic.
- `example_youngs_modulus_cmaes.py` owns optimizer CLI setup, dataset
  iteration, fit reporting, viewer integration, and artifact orchestration.
- `batched_sysid_cmaes.py` owns parameter maps, CMA-ES orchestration, result
  types, penalties, and cross-structure aggregation.
- `evaluate_youngs_modulus_structures(...)` is the fused generation evaluator.
  It owns preparation, recorded replay, instability checks, pooled Sinkhorn
  scoring, scalar fallback, and structure-local metadata.
- `batched_sysid_multi_replay.py` continues to own stable slot planning,
  physical chunking, per-environment routing, and fused replay.

The optimizer uses pycma's direct ask/tell interface:

```text
for each selected structure
  -> resolve log10-E bounds and midpoint
  -> CMAEvolutionStrategy(midpoint, sigma, bounds)
  -> ask() for one population
  -> map vectors to YoungsModulusCandidate
  -> one chunked fused replay for structure x candidate x direction
  -> convert eligible Sinkhorn scores to fitness values
  -> tell(samples, fitness)
  -> repeat until max generations or native stop
  -> ask no further samples
  -> explicitly evaluate final distribution mean
  -> write structure result and final-mean overlay
aggregate successful structure fits
```

Using a scalar pycma objective callback is rejected because it would evaluate
simulations sequentially and discard the existing GPU batching. Shelling out to
the grid CLI for every generation is rejected because it adds process and file
overhead and weakens typed error handling. Replacing the grid command is also
rejected: its exact-GT insertion, parity, and gate workflows remain useful
independent acceptance diagnostics.

## Bounds and initialization

Search bounds come from `youngs_modulus_pa.min/max` for primary, spur, and stem
in the ranges fixture associated with the dataset:

1. Use an explicit `--ranges` path when supplied.
2. Otherwise use the dataset manifest's `collection.ranges_path`.
3. If neither resolves to a loadable fixture, fail clearly. Do not silently
   substitute unrelated default bounds.

Bounds must be finite, positive, ordered, and present for all three fitted
segments. Convert each pair to log10 space. The initial CMA mean is the
component-wise midpoint of those log-space bounds. This avoids using stored GT
as optimizer input.

The initial sigma is CLI-configurable and expressed in log10 decades, with a
default of `0.25`. pycma bounds keep all samples within the fixture ranges. The
implementation records the resolved physical and log-space bounds, midpoint,
sigma, population size, seed, and stop options in the report.

## Per-generation evaluation

`ask()` returns one population of three-dimensional log-space vectors per
structure. Convert the vectors to `YoungsModulusCandidate` values in Pa and
pass all selected structure/population pairs to
`evaluate_youngs_modulus_structures(...)`. Compatible structures share the
same fused schedule; incompatible or affected failed structures retain the
implemented scalar fallback.

The evaluator stack already:

- freezes each structure's oracle geometry, topology, damping ratio, density,
  apple parameters, and secondary E;
- changes primary, spur, and stem E through
  `set_rod_youngs_modulus`, preserving fixed axial overrides;
- replays every usable recorded direction;
- preserves stable structure/local-candidate/physical-direction identity;
- chunks whole candidate-direction blocks under `--max-envs-per-batch`;
- scores pooled median hold-transition bags with hold and direction identity;
- marks unstable, incomplete, and non-finite candidates as disqualified.

Candidate order returned by pycma must remain aligned with evaluator candidate
indices when fitness values are passed to `tell()`.

For a generation with at least one finite eligible score, every disqualified or
non-finite candidate receives the deterministic finite penalty
`worst_finite + max(1.0, abs(worst_finite))`. The report records the original
disqualification reason and the substituted penalty. If no candidate in a
generation is eligible, fail that structure rather than updating CMA-ES from
an arbitrary all-penalty population.

## Stopping and fitted estimate

Each structure stops when either:

- the number of completed generations reaches `--max-generations`; or
- `CMAEvolutionStrategy.stop()` reports a native pycma convergence condition.

Record the exact stop reason. A generation cap is mandatory so command cost is
bounded; `--max-generations` defaults to `10`.

After stopping, evaluate the final CMA distribution mean once through the same
replay and scoring path. This explicitly measured point is the fitted
Young's-modulus estimate for structure-level errors and cross-structure
aggregation. If the final mean is disqualified or non-finite, fail the
structure instead of silently substituting the best sampled point.

Also report the best sampled candidate and its measured score. It is diagnostic
and does not replace the final mean. Record the final log-space covariance so
the optimizer's remaining uncertainty and parameter coupling are inspectable.

## New CLI

The separate command reuses dataset, structure selection, replay, scoring,
viewer, settling, overwrite, batching, seed, multi-structure batching, and
fail-fast controls that remain meaningful.

Add CMA-specific controls:

- `--ranges`
- `--max-generations`
- `--population-size` (optional; pycma default when omitted)
- `--initial-sigma-log10`
- `--cma-seed` (default `0`)

The existing `--seed` continues to control replay behavior and retains its
manifest fallback. `--cma-seed` controls optimizer sampling. Each structure
uses a deterministic seed derived from the base CMA seed and the structure
index so independent optimizers do not reuse identical random streams.

Do not expose grid-only and grid-artifact controls on the CMA-ES command:

- `--log10-e-primary`
- `--log10-e-spur`
- `--log10-e-stem`
- `--include-gt-candidate`
- `--max-candidates`
- `--export-replays`
- `--max-overlay-candidates`

Those controls remain unchanged on `example_youngs_modulus_sys_id.py`. The
CMA-ES command continues past a failed structure, records its error, and fits
the remaining structures by default. `--fail-fast` aborts immediately on the
first structure error for debugging.

## Reports and artifacts

Write one strict-JSON CMA-ES report under `--output`. Update it atomically after
each completed or failed structure so earlier expensive fits survive a later
structure failure.

For every successful structure report:

- resolved bounds and initial mean in log10-E and Pa;
- CMA seed, sigma, population size, generations, evaluation count, and stop
  reason;
- per-generation sample vectors, raw Sinkhorn scores, substituted fitness
  values, disqualification metadata, distribution mean, and covariance;
- final mean in log10-E and Pa;
- the final mean's explicitly evaluated aggregate and per-direction Sinkhorn
  scores;
- best sampled vector and score;
- stored GT E and log10-E for evaluation only;
- final-mean log-space and relative errors to GT.

Covariance fields represent the actual scaled search covariance in log10-E
coordinates, not pycma's unscaled internal shape matrix.

Write
`structure_XXX/youngs_modulus_overlay.html` using the final mean's explicit
replay against the recorded GT trajectory. Overlay failure is recorded without
invalidating an otherwise valid numeric fit.

Across successful structures report:

- count of requested, fitted, and failed structures;
- mean fitted log10-E and corresponding geometric-mean E;
- sample covariance and standard deviation of fitted log10-E;
- arithmetic mean of fitted E;
- per-segment minimum and maximum fitted E as diagnostics;
- comparison with mean stored GT E.

The aggregate min/max values are not automatically approved
domain-randomization bounds. A later slice must decide how to reject poor fits
and whether to use extrema, robust quantiles, covariance, or safety margins.

## Error handling

- Missing or malformed dataset-wide ranges configuration fails before
  optimization.
- Missing required rods or unusable structure metadata fails that structure.
- A structure with no usable directions fails that structure.
- An all-invalid generation fails that structure.
- A disqualified or non-finite final mean fails that structure.
- Optional overlay errors remain isolated from the numeric report.
- If every requested structure fails, the command exits nonzero.
- Existing non-empty output directories still require `--overwrite`.

## Testing

Follow red-green-refactor.

Fast tests cover:

1. Fixture-bound extraction, validation, log conversion, and midpoint
   initialization.
2. Failure when manifest bounds are absent or unresolvable, plus `--ranges`
   override behavior.
3. Candidate conversion and ask/evaluate/tell order with a deterministic fake
   optimizer/evaluator.
4. Population ordering and candidate-by-direction batching.
5. Invalid-candidate penalty ordering and all-invalid generation failure.
6. Generation-cap and native-stop behavior with recorded stop reasons.
7. Explicit final-mean evaluation and rejection of an invalid final mean.
8. Independence of per-structure optimizers and seeds.
9. Per-generation, per-structure, and aggregate report math, including strict
   JSON and diagnostic min/max.
10. Atomic report persistence and default continue-versus-fail-fast behavior.
11. Final-mean overlay selection and isolation of overlay failures.
12. Parser additions and removal of grid-only options.
13. Existing candidate application, replay, Wasserstein, and overlay tests
    remain green.

Use a deterministic synthetic bowl objective to show that the CMA mean moves
toward a known optimum without running physics in fast tests. This validates
orchestration, not pycma internals.

Manual CUDA acceptance:

1. Collect or reuse a small healthy `batched_sysid_v1` dataset.
2. Run CMA-ES for one structure, a small population, and a few generations.
3. Confirm successful exit, multiple ask/tell generations, an explicitly
   scored final mean, strict report schema, and a readable final-mean overlay.
4. Report final mean, best sample, stored GT, Sinkhorn scores, and stop reason.

The smoke need not enforce a universal fit-error threshold; held-out acceptance
belongs to V.5.3.

## Success criteria

- The new script is CMA-ES-only; the existing grid script remains available.
- Every selected structure receives an independent bounded pycma run.
- Each generation is evaluated through one chunked batched replay path.
- Stored GT is used only for reporting, never initialization or fitness.
- The final distribution mean is explicitly replayed, scored, and reported as
  the fitted estimate.
- Per-structure fits and aggregate statistics survive partial failures.
- Final-mean overlays compare each fit with recorded GT behavior.
- Fast tests pass and the documented CUDA smoke is runnable.
- Domain-randomization range selection remains explicitly deferred.
