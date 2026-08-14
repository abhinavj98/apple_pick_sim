# Young's-Modulus CMA-ES Loop Design

**Date:** 2026-07-16  
**Roadmap slice:** V.5.2 — continuous Young's-modulus calibration  
**Status:** Superseded — the loop shipped, but its primary-\(E\) phenotype was replaced by support-\(k_p\) × spur/stem \(E\).
**Canonical living doc:** `docs/handbook-youngs-cma.md`
**Implementation notes / how-to:** `docs/handbook-youngs-cma.md`; README → **CMA-ES sim-to-sim transfer**.

## Purpose

Add a separate CMA-ES entry point that runs an independent pycma optimization
for each selected dataset structure. Each optimizer searches for primary,
spur, and stem Young's modulus using the existing recorded-action replay and
pooled hold-phase Sinkhorn Wasserstein objective.

The output is one fitted three-component Young's-modulus vector per structure.
The implementation also retains optimizer diagnostics and cross-structure
statistics. Turning those fits into domain-randomization ranges is deferred.
The implemented Cartesian-grid command remains the diagnostic and acceptance
path documented in `docs/handbook-youngs-cma.md`.

## Scope

- Add a separate `example_youngs_modulus_cmaes.py` command; do not add a
  `grid|cmaes` mode switch to the existing grid command.
- Optimize
  \(\log_{10}([E_\mathrm{primary}, E_\mathrm{spur}, E_\mathrm{stem}])\).
- Run one independent CMA-ES optimizer per selected structure.
- Evaluate every generation's population in parallel with the existing
  candidate-by-direction batched replay.
- Use pooled hold-phase Sinkhorn Wasserstein as the fitness.
- Initialize from `CMA_SEARCH_PARAMS` (default: explicit log10-E start mean,
  not recorded GT; optional `"bounds_midpoint"` uses fixture midpoints).
- Constrain search with `CMA_SEARCH_PARAMS["search_bounds_log10"]` (default:
  absolute 0.1–100 GPa / \(\log_{10} E \in [8, 11]\) per role), not the narrow
  fixture `youngs_modulus_pa` ε-bands. `None` allows unbounded search.
- Stop on either a configured generation cap or pycma's native convergence
  criteria.
- Explicitly replay and score the final distribution mean, and treat it as the
  fitted estimate.
- Preserve the best sampled candidate and final covariance as diagnostics.
- Produce a final-mean-versus-recorded-GT overlay for each successful structure.
- Add pycma (`cma`) to the Gym tooling dependencies.
- Preserve the existing grid gate and add separate CMA-ES fit-integrity gate
  artifacts.

## Non-goals

- Updating domain-randomization fixtures from the fitted values.
- Choosing a final domain-randomization range policy. Raw min/max is reported
  only as a diagnostic; robust quantiles, margins, and fit validation remain a
  later decision.
- Fitting damping ratio, density, geometry, apple parameters, or secondary E.
- Sharing one optimizer or one fitted E vector across structures.
- Changing Wasserstein features, instability policy, or the material-to-VBD
  conversion (except the joint-kd ζ-hold amendment below).
- Changing or removing the Cartesian-grid ranking command, candidate replay
  export, or top-ranked grid overlays.
- Held-out validation, real-data acceptance criteria, or new excitation types.

**Amendment (2026-07-22 → 2026-07-23) — joint-kd from fixture ζ:** FIXED-joint
penalty `kd` is expanded from `sim_build.joint_damping_ratio` as absolute
`kd = ζ · 2 · √(k · I)` / `√(k · m)` per env (child mass/inertia). Weld `kd` is
**not** scaled with Young's modulus; fixture ζ is constant across CMA/grid E
candidates. (An earlier √(E/E_ref) distal scale was removed as incorrect for
constant weld ζ.) Re-baseline `scripts/gate_youngs_modulus_cmaes.sh` and the
ranking gate after damping-policy changes. See `docs/damping-tuning.md`.

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
initialize one optimizer per selected structure
while any optimizer is active
  -> remove optimizers already stopped by cap or native criteria
  -> ask() once from every active optimizer
  -> preserve each optimizer's returned sample order
  -> map vectors to per-structure YoungsModulusCandidate populations
  -> one logical fused evaluation for active structure x candidate x direction
     (physical replay may be chunked or retried through the existing fallback)
  -> route scores by structure and local candidate index
  -> compute penalties independently within each structure
  -> tell(original samples, aligned fitness) independently per successful structure
  -> fail only structures with invalid generation results
  -> snapshot newly stopped bounded phenotype means
ask no further samples
explicitly evaluate all stopped means in one fused final-mean wave
write successful structure overlays
aggregate successful structure fits
```

Using a scalar pycma objective callback is rejected because it would evaluate
simulations sequentially and discard the existing GPU batching. Shelling out to
the grid CLI for every generation is rejected because it adds process and file
overhead and weakens typed error handling. Replacing the grid command is also
rejected: its exact-GT insertion, parity, and gate workflows remain useful
independent acceptance diagnostics.

## Bounds and initialization

Two roles are separated: the ranges fixture for replay construction (and
optional midpoint init), versus the CMA search box in `CMA_SEARCH_PARAMS`.

### Ranges fixture (replay / optional midpoint)

Resolve the ranges fixture associated with the dataset:

1. Use an explicit `--ranges` path when supplied.
2. Otherwise use the dataset manifest's `collection.ranges_path`.
3. If neither resolves to a loadable fixture, fail clearly. Do not silently
   substitute an unrelated default fixture.

The fixture remains required before optimizer construction. It supplies
`sim_build` controls for replay environments and
`youngs_modulus_pa.min/max` for primary/spur/stem when
`initial_mean_log10 == "bounds_midpoint"`. Fixture bounds must be finite,
positive, ordered, and present for all three fitted segments. Relative CLI and
manifest paths resolve from the process CWD; the report stores the absolute
path. CMA-specific validation must not rely only on the generic ranges
validator: it explicitly rejects missing, null, non-numeric, non-finite,
non-positive, equal, and reversed segment bounds.

The fixture `youngs_modulus_pa` box is **not** the default CMA search
constraint. In the variance proxy fixture the primary band is essentially a
point (~0.5 Pa wide), which cannot support meaningful CMA exploration.

### Search box and start mean (`CMA_SEARCH_PARAMS`)

Search clipping and ask validation use `search_bounds_log10`:

- default: absolute safety box 0.1–100 GPa
  (`lower=[8,8,8]`, `upper=[11,11,11]` in log10-E), **not** the narrow fixture
  ε-bands;
- `None`: unbounded search (omit pycma `bounds`; ask validation skips the
  phenotype box; the integrity gate checks only `e_pa == 10**log10_e`).

When `search_bounds_log10` is set, construct pycma with
`bounds=[lower_log10, upper_log10]` and keep samples inside that box.

Default start mean is the search-box midpoint `[9.5, 9.5, 9.5]` (still not
structure GT). Setting `initial_mean_log10` to `"bounds_midpoint"` instead
derives the component-wise midpoint of the loaded fixture's log-space bounds.

Default `initial_sigma_log10` is `1.0` (log10 decades; matches library
`DEFAULT_INITIAL_SIGMA_LOG10`). Reject non-positive or
non-finite sigma. Record the active search bounds (or JSON `null` when
unbounded), start mean, sigma, population size, seed, and stop options in the
report. Structure `bounds` in `cmaes_report.json` are the **search** box, not
the fixture ε-bands.

Default `CMA_SEARCH_PARAMS` knobs (edit in
`example_youngs_modulus_cmaes.py`): `initial_mean_log10=[9.5,9.5,9.5]`,
`initial_sigma_log10=1.0`, `population_size=15`, `max_generations=10`,
`cma_seed=56`, `search_bounds_log10` = `[8,8,8]`–`[11,11,11]`.

## Per-generation evaluation

`ask()` returns one population of three-dimensional log-space vectors per
structure. Convert the vectors to `YoungsModulusCandidate` values in Pa and
pass all selected structure/population pairs to
`evaluate_youngs_modulus_structures(...)`. Compatible structures share the
same fused schedule. The current fallback is preserved exactly: a compatibility
mismatch makes the entire submitted group replay through per-structure scalar
evaluation, while a fused runtime failure retries only affected structures
from scratch through scalar evaluation.

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

The coordinator validates every asked population before replay: population
length matches the optimizer, every vector has length three, values are finite,
and (when a search box is configured) lie within that phenotype domain, and
returned evaluations contain each
candidate index exactly once. `tell()` receives the original sample objects
returned by `ask()`, in their original order.

For a generation with at least one finite eligible score, every disqualified or
non-finite candidate receives the deterministic finite penalty
`worst_finite + max(1.0, abs(worst_finite))`. The report records the original
disqualification reason and the substituted penalty. If no candidate in a
generation is eligible, **re-`ask()` / re-evaluate** up to
`DEFAULT_ALL_INVALID_REASKS` (3) times, then `tell()` with uniform flat fitness
`ALL_INVALID_FLAT_PENALTY` (`1e12`, `update_best=False`) so the structure stays
active (see Error handling amendment below). Pass `all_invalid_reasks=0` for
legacy fail-without-`tell()`. If the prescribed penalty overflows or is
otherwise non-finite, fail that structure without calling `tell()`.

## Stopping and fitted estimate

Each structure stops when either:

- the number of completed generations reaches `CMA_SEARCH_PARAMS["max_generations"]`; or
- `CMAEvolutionStrategy.stop()` reports a native pycma convergence condition.

Record all stop evidence. `stop_kind` is `generation_cap`, `pycma`, or `both`;
the report also contains the complete strict-JSON pycma stop-condition mapping
and completed-generation count. Check native criteria and the cap before each
`ask()` and after every successful `tell()`. A generation cap is mandatory so
command cost is bounded; `max_generations` defaults to `10`.

After stopping, snapshot `CMAEvolutionStrategy.result.xfavorite`, pycma's
bounded phenotype-space distribution mean. Do not use the internal genotype
`es.mean` or the best sampled point. After no optimizer remains active,
evaluate all snapshots once through one logical multi-structure replay and
scoring wave. This explicitly measured point is the fitted
Young's-modulus estimate for structure-level errors and cross-structure
aggregation. If the final mean is disqualified or non-finite, fail the
structure instead of silently substituting the best sampled point.

Also report the best sampled candidate and its measured score. It is diagnostic
and does not replace the final mean. Record the final log-space covariance so
the optimizer's remaining uncertainty and parameter coupling are inspectable.
Specifically record pycma's internal shape matrix `C`, global `sigma`,
`sigma_vec.scaling`, phenotype coordinate standard deviations, and the
effective unbounded optimizer-coordinate covariance
`sigma**2 * diag(sigma_vec.scaling) @ C @ diag(sigma_vec.scaling)`. Do not label
this matrix as the exact covariance of the nonlinearly bounded phenotype
distribution.

## New CLI

The separate command reuses dataset, structure selection, replay, scoring,
viewer, settling, overwrite, batching, seed, multi-structure batching, and
fail-fast controls that remain meaningful.

Add CMA-specific controls on the CLI only for dataset/replay wiring:

- `--ranges`

Search initialization and loop knobs live in module-level
`CMA_SEARCH_PARAMS` inside `example_youngs_modulus_cmaes.py`:

- `initial_mean_log10` (`"bounds_midpoint"` or an explicit length-3 log10-E vector;
  default `[9.5, 9.5, 9.5]`)
- `initial_sigma_log10` (default `1.0`)
- `population_size` (default `15`; `None` → pycma default)
- `max_generations` (default `10`)
- `cma_seed` (application base seed; not passed directly to pycma; default `56`)
- `search_bounds_log10` (`None` = unbounded, or `{"lower","upper"}` length-3
  log10-E vectors; default absolute 0.1–100 GPa / `[8,8,8]`–`[11,11,11]`)

The existing `--seed` continues to control replay behavior and retains its
manifest fallback. Default `cma_seed` comes from `CMA_SEARCH_PARAMS`; `--cma-seed`
overrides it (the integrity gate passes `--cma-seed "${SEED}"` so each SEEDS job
varies both collect and optimizer RNG). Each structure
uses a deterministic seed derived from the base CMA seed and the structure
index so independent optimizers do not reuse identical random streams. Because
pycma treats seed `0` as time-based and otherwise uses NumPy's global RNG by
default, do not pass the application seed directly as pycma's `seed` option. Derive a
stable positive 32-bit effective seed from the base seed and structure index,
construct a dedicated NumPy `Generator` for each optimizer, pass its
`standard_normal` method as pycma's `randn` option, and set pycma `seed` to
`np.nan`. Derive it with
`SeedSequence([base_seed % 2**32, structure_idx % 2**32]).generate_state(1,
dtype=np.uint32)[0]`, remap zero to one, and reject a collision among selected
structures rather than silently sharing streams. Record both the base and
effective seeds and test repeated interleaved optimizers.

Construct pycma with the resolved mean and sigma as positional arguments and an
options mapping containing `randn=generator.standard_normal`, `seed=np.nan`,
and disabled library verbosity. Include `bounds=[lower_log10, upper_log10]`
only when `search_bounds_log10` is set; omit bounds for unbounded search.
Include `popsize` only when `population_size` is supplied. Lock and
test the resolved `cma` version through `uv.lock`.
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

`--multi-structure-batch` remains default-on. `--no-multi-structure-batch`
selects the scalar parity/debug path. Small fused-versus-scalar Sinkhorn
differences are accepted as numerical noise and are not a release-blocking
exact-equality requirement.

## Reports and artifacts

Write `<output>/cmaes_report.json` as strict JSON. Write an initial report before
the first replay, then update it atomically after each generation wave, failure,
stop transition, final-mean evaluation, and overlay attempt. This preserves
completed histories and failure evidence even though successful stopped means
are deliberately queued for one fused final evaluation.

Each structure has one report status: `active`,
`stopped_pending_final_evaluation`, `fitted`, or `failed`. Failures include a
machine-readable stage (`prepare`, `generation_evaluation`, `all_invalid`,
`penalty`, or `final_mean`) plus a human-readable message. Overlay failure is
recorded as a separate artifact error and does not change `fitted` status.

For every successful structure report:

- active search bounds (or JSON `null` when unbounded) and initial mean in
  log10-E and Pa;
- CMA base/effective seed, sigma, population size, generations, evaluation
  counts, and complete stop evidence;
- per-generation sample vectors, raw Sinkhorn scores, substituted fitness
  values, disqualification metadata, distribution mean, and covariance;
- final mean in log10-E and Pa;
- the final mean's explicitly evaluated aggregate and per-direction Sinkhorn
  scores;
- best sampled vector and score;
- stored GT E and log10-E for evaluation only;
- final-mean log-space and relative errors to GT (`gt_diagnostics`);
- `evaluated_history_extrema`: component-wise min/max of samples actually
  submitted in CMA populations (`ask_samples_log10`), in log10-E and Pa
  (`null` before the first population).

Per-generation distribution fields explicitly distinguish the distribution
used by `ask()` from the post-`tell()` distribution. Covariance fields use the
effective unbounded optimizer-coordinate definition above, not pycma's
unscaled shape matrix and not an exact bounded-phenotype covariance.

Report separate counters:

- `completed_generations`: successful `tell()` calls;
- `optimizer_samples_told`: population samples included in successful
  `tell()` calls;
- `replay_candidate_evaluations`: population and final-mean candidates
  submitted logically for scoring;
- `final_mean_evaluations`: zero or one;
- physical environment slots and scalar retry attempts from replay
  diagnostics.

Scalar retries do not increment pycma's logical optimizer evaluation count.

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
Sample covariance uses `ddof=1`; covariance and standard deviation are JSON
`null` when fewer than two structures fit. All means, extrema, and errors are
component-wise.

The existing grid ranking gate remains unchanged. Add separate
`gate_youngs_modulus_cmaes.sh` and `youngs_modulus_cmaes_gate_report.py`
artifacts that validate report completeness, finite explicitly scored final
means, search-bounds membership when bounds are present (skip membership when
unbounded / `bounds` is null), coherent counts, and stop evidence. This integrity
gate does not impose a GT-error threshold.

## Error handling

- Missing or malformed dataset-wide ranges configuration fails before
  optimization.
- Missing required rods or unusable structure metadata fails that structure.
- A structure with no usable directions fails that structure.
- An all-invalid generation (every sample disqualified / non-finite) does
  **not** fail by default: `run_cma_generation_wave` re-`ask()`s / re-evaluates
  up to `DEFAULT_ALL_INVALID_REASKS` (3) more times at the same generation
  index, then if still all-invalid calls `tell()` with a uniform flat fitness
  `ALL_INVALID_FLAT_PENALTY` (`1e12`). The structure stays active; penalty
  metadata records `flat_penalty_tell` and `all_invalid_reasks`. Identical flat
  fitnesses give CMA no relative ranking signal — intentional so the generation
  is not aborted and the distribution can move/shrink. Pass
  `all_invalid_reasks=0` to restore legacy fail-without-`tell()`. No CLI knob
  in this slice (library default only).
- A disqualified or non-finite final mean fails that structure.
- Optional overlay errors remain isolated from the numeric report.
- If every requested structure fails, the command exits nonzero.
- Existing non-empty output directories still require `--overwrite`.
- Partial numeric success exits zero unless `--fail-fast` aborted the command.
- A global ranges/configuration error or unexpected top-level batch-evaluator
  error aborts nonzero after atomically recording available evidence.
- Closing the viewer cancels the command: do not call `tell()` with truncated
  replay, checkpoint cancellation evidence, and exit nonzero.
- `--overwrite` does not make stale files authoritative: clear CMA-owned report
  and selected-structure overlay targets before starting, and trust only
  artifacts referenced by the current report.

## Testing

Follow red-green-refactor.

Fast tests cover:

1. Fixture-bound extraction, validation, log conversion, and midpoint
   initialization.
2. Failure when manifest bounds are absent or unresolvable, plus `--ranges`
   override behavior.
3. Candidate conversion and ask/evaluate/tell order with a deterministic fake
   optimizer/evaluator.
4. Synchronized active-set waves, stable
   structure/candidate/direction routing, chunking, and scalar fallback.
5. Invalid-candidate penalties, overflow, all-invalid re-ask / flat-penalty
   `tell()`, and `all_invalid_reasks=0` fail-without-tell; include a real-pycma
   all-invalid → re-ask → tell regression (not FakeOptimizer-only).
6. Independent generation-cap/native-stop timing and complete stop evidence.
7. One fused bounded-`xfavorite` final wave and rejection of an invalid final
   mean.
8. Independence and reproducibility of interleaved optimizer-owned RNGs.
9. Per-generation, per-structure, and aggregate report math, including strict
   JSON and diagnostic min/max.
10. Atomic progress persistence, structured failures, cancellation, and
    continue-versus-fail-fast behavior.
11. Final-mean overlay selection and isolation of overlay failures.
12. New parser controls, absence of grid-only controls on the CMA command, and
    unchanged grid CLI/gate contracts.
13. Existing candidate application, replay, Wasserstein, and overlay tests
    remain green.
14. Real-pycma bounded synthetic bowls with distinct per-structure optima,
    repeated interleaved-run determinism, and NumPy/pycma strict-JSON values.
15. Separate CMA integrity-gate parsing and existing ranking-gate regression.

Use deterministic synthetic bowl objectives to show that independent CMA means
move toward distinct known optima without running physics. Fake optimizers test
the coordinator protocol; one real-pycma test validates the adapter and bounded
phenotype handling rather than re-testing pycma convergence comprehensively.

Manual CUDA acceptance:

1. Collect or reuse a small healthy `batched_sysid_v1` dataset.
2. Run CMA-ES for multiple structures, a small population, and a few
   generations with default fused batching.
3. Confirm successful exit, multiple ask/tell generations, an explicitly
   scored fused final-mean wave, independent histories/stop evidence, strict
   report schema, and readable final-mean overlays.
4. Report final mean, best sample, stored GT, Sinkhorn scores, and stop reason.
5. Run a small `--no-multi-structure-batch` debug smoke and compare report
   structure and control flow; exact Sinkhorn equality is not required because
   small fused/scalar numerical differences are accepted as noise.
6. After a full build, collect **5 structures × 5 directions**, run CMA-ES, and
   report optimized final means versus stored true parameters, each structure's
   `evaluated_history_extrema` min/max, and final covariance diagnostics.

The smoke need not enforce a universal fit-error threshold; held-out acceptance
belongs to V.5.3. **Task 8 verification passed 2026-07-17** (artifacts under
`tmp/task8_cuda_acceptance/`; see `docs/handbook-youngs-cma.md`).

## Success criteria

- The new script is CMA-ES-only; the existing grid script remains available.
- Every selected structure receives an independent bounded pycma run.
- Active structures advance in synchronized generation waves evaluated through
  the fused structure × population × direction path, subject to physical
  chunking and existing scalar retries.
- Stored GT E parameters are used only for reporting, never initialization or
  fitness; recorded observations remain the fitness target and non-fitted
  oracle structure parameters remain frozen during replay.
- The final distribution mean is explicitly replayed, scored, and reported as
  the fitted estimate.
- Per-structure fits and aggregate statistics survive partial failures.
- Final-mean overlays compare each fit with recorded GT behavior.
- Fast tests pass and the documented CUDA smoke (including the 5×5 collect
  report above) is executed and recorded; until then the design remains
  verification-pending rather than complete.
- Domain-randomization range selection remains explicitly deferred.
