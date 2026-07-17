# Young's-Modulus Multi-Structure Grid Megabatch Design

**Date:** 2026-07-16  
**Roadmap slice:** V.5.2 grid-performance prerequisite  
**Status:** Approved in brainstorming; awaiting written-spec review

## Purpose

Accelerate the existing dataset-driven Young's-modulus grid workflow by
evaluating compatible structures in one heterogeneous GPU megabatch instead of
building, settling, replaying, and closing one simulation batch per structure.

The flattened simulation work is:

```text
structure x local grid candidate x recorded direction
```

Each structure remains an independent system-identification problem. It keeps
its own candidate list, recorded observations, Sinkhorn losses, ranking, GT
comparison, overlays, exports, and failure result. This design changes replay
scheduling only; it does not aggregate loss across structures.

The resulting replay primitive is intended to be reusable by a later CMA-ES
loop, but CMA-ES implementation is explicitly deferred.

## Current bottleneck

`example_youngs_modulus_sys_id.py::_run()` currently iterates selected
structures. For each structure it calls
`evaluate_youngs_modulus_candidates()`, which calls
`replay_candidates_for_structure()`. Every candidate chunk then calls
`replay_batched_sysid_structure()`, which builds and closes a separate
`ApplePickBatchedSysIdEnv`.

The current candidate-by-direction simulation is already parallel within one
structure, but the outer structure loop repeats:

- heterogeneous model construction;
- settle and weld initialization;
- robot/controller allocation;
- CUDA graph and buffer setup where applicable; and
- environment teardown.

For five structures, 32 candidates, and five directions, the useful work is
800 worlds. When memory permits, those worlds should be built and replayed as
one model rather than five sequential 160-world models.

One process per structure is rejected for a single GPU. Multiple processes
duplicate CUDA contexts, models, and memory and can reduce throughput or cause
out-of-memory failures.

## Scope

- Optimize `example_youngs_modulus_sys_id.py` grid replay.
- Preserve each structure's exact local candidate list.
- Preserve optional structure-specific GT candidate insertion.
- Change only primary, spur, and stem Young's modulus per candidate.
- Keep each structure's geometry, topology, density, damping ratio, apple
  parameters, secondary E, and other non-fitted state fixed.
- Replay all usable recorded directions for each local candidate.
- Build one heterogeneous environment per flattened chunk.
- Preserve independent per-structure instability checks, Sinkhorn scores,
  rankings, reports, overlays, and replay exports.
- Preserve existing `--max-envs-per-batch`, `--fail-fast`, exclusion, viewer,
  overwrite, and scoring controls.
- Fall back to individual structure replay on an exceptional fused runtime
  failure when `--fail-fast` is disabled.
- Measure wall time and peak GPU memory on the target RTX 4090.

## Non-goals

- Implementing pycma, CMA-ES `ask()`/`tell()`, convergence, or optimizer
  reports.
- Aggregating losses across structures or producing one shared fitted E vector.
- Fitting geometry, damping, density, apple parameters, secondary E, or any
  parameter besides primary/spur/stem E.
- Supporting mixed topology, segment counts, junction layouts, replay frame
  counts, direction layouts, controller configurations, or gripper
  configurations in one fused run.
- Reusing one built environment across changing candidate sets or optimizer
  generations.
- Changing the Sinkhorn feature contract, instability threshold, ranking
  policy, report schema, overlay selection, or export schema.
- Adding multi-process execution for one GPU.
- Modifying the vendored `newton/` submodule.

## Compatibility contract

All structures admitted to one fused replay must come from a compatible
collection and have:

- the same topology and segment counts;
- identical ordered junction names;
- the same recorded frame count after pre-weld rows are stripped;
- the same usable direction layout and source direction-space width;
- compatible control rate and replay simulation configuration;
- the same controller and gripper configuration; and
- compatible observation array shapes.

Geometry and material values may differ by structure. Candidate lists may also
differ by structure, including a different exact-GT candidate appended to each
list.

Compatibility is validated before allocating the fused GPU model. A malformed
or incompatible structure is recorded as a structure-local failure and omitted
from the fused request unless `--fail-fast` is enabled. This preserves the
existing default behavior of continuing past bad structures.

## Architecture

### Multi-structure replay module

Add
`apple_pick_gym/batched_envs/batched_sysid_multi_replay.py`.

This module owns typed planning and replay data:

- `YoungsModulusStructureRequest`: one structure index, its local candidates,
  resolved directions, base parameters, recorded episodes, and prepared
  scoring context;
- `ReplayCandidateBlock`: one structure-local candidate and all of its
  direction slots;
- `ReplaySlot`: one exact
  `(structure_idx, local_candidate_idx, direction_idx)` world, including
  applied parameters, recorded actions, and initialization source; and
- a fused replay result keyed by the same stable indices.

The module owns:

- compatibility validation;
- deterministic flattening;
- environment-budget chunk planning;
- per-chunk heterogeneous environment construction;
- action replay and stability monitoring;
- observation collection; and
- routing collectors back to stable structure/candidate/direction keys.

It does not own Young's-modulus scoring, ranking, reporting, visualization, or
optimization.

### Per-environment dataset initialization

Extend
`apple_pick_sim/system_id/batched_digital_twin_init.py` with a per-environment
initialization API. Instead of one scalar `structure_idx` and direction
selection by `env_idx % num_directions`, the generalized API receives one
explicit `(structure_idx, direction_idx)` source for every environment.

Each world must receive the correct:

- initial robot joint position and zero joint velocity;
- VIC default joint position;
- initial TCP target position and orientation;
- episode-specific initialization metadata; and
- any existing digital-twin state applied by the scalar path.

The current `initialize_batched_env_from_dataset()` API remains as a
compatibility wrapper. It constructs the repeated per-environment sources used
by the one-structure candidate-major layout and delegates to the generalized
implementation.

### Young's-modulus evaluation

Extend `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` with a
multi-structure evaluator. The historical module name remains unchanged; this
slice adds no optimizer.

Factor the existing evaluator into two responsibilities:

1. prepare one structure request, including GT context and applied parameters;
2. score one structure from already-routed replay observations.

The existing `evaluate_youngs_modulus_candidates()` remains available and
retains its current behavior. The multi-structure evaluator prepares all valid
requests, invokes fused replay, and then uses the same structure-local scoring
logic to return one `YoungsModulusEvaluation` per structure.

### CLI orchestration

Update
`apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` to:

1. resolve the selected structures and each structure's local candidate list;
2. preflight and prepare all valid structure requests;
3. invoke one multi-structure evaluation;
4. retry affected structures individually after exceptional fused runtime
   failures when allowed;
5. preserve requested structure order in reports; and
6. pass unchanged evaluations to the existing report, overlay, and export
   paths.

The CLI remains a grid-only command. No CMA-specific options are added.

## Deterministic index and ordering contract

Stable identity is the tuple:

```text
(structure_idx, local_candidate_idx, direction_idx)
```

`local_candidate_idx` is local to one structure and remains aligned with that
structure's candidate list and report. It is not a global candidate index.

Requests are flattened in:

1. requested structure order;
2. local candidate order; and
3. resolved direction order.

The implementation must not infer identity from a post-chunk environment
offset. Each chunk carries explicit slot descriptors, and collectors are keyed
back to the stable tuple after replay.

Sparse source direction IDs remain unchanged. They are not renumbered during
flattening, scoring, overlays, or replay export.

## Candidate application

For structure `s` and local candidate `c`:

```text
per_env_params = candidate[s, c].apply_to(base_params[s])
```

`YoungsModulusCandidate.apply_to()` remains authoritative. It changes only
primary, spur, and stem E through `set_rod_youngs_modulus`, which re-derives
material-dependent VBD stiffness and damping while preserving fixed axial
overrides. Directly patching bend stiffness remains incorrect.

Every direction world for the same `(s, c)` receives the same applied
parameters. Different structures may have different geometry and fixed
materials, and different local candidates may have different fitted E.

## Chunking

One `ReplayCandidateBlock` contains all direction worlds for one
`(structure_idx, local_candidate_idx)` pair. Candidate blocks are atomic:
chunking never splits one candidate's directions across models.

For common direction count `D`, each candidate block consumes `D`
environments. The chunk planner greedily preserves deterministic flattened
order while ensuring:

```text
sum(block.num_directions for block in chunk) <= max_envs_per_batch
```

When `--max-envs-per-batch=0`, all candidate blocks use one unbounded
environment batch. When the limit is positive, it must be at least `D`;
otherwise the command fails before model construction.

For 32 candidates, five structures, and five directions, the flattened count
is 800 environments. If the configured limit and measured GPU memory permit,
the scheduler performs one build, settle, and replay. If not, it uses the
minimum deterministic number of whole-candidate chunks allowed by the cap.

The 25,000 environment default is an administrative cap, not proof that every
configuration fits GPU memory. Peak memory must be measured with the actual
model and replay settings.

## Replay loop

For each chunk:

1. build `per_env_params` from the slot descriptors;
2. build one `ApplePickBatchedSysIdEnv`;
3. reset and initialize each world from its explicit episode source;
4. create stability and soft-disable state for the entire chunk;
5. replay one common recorded action frame across all worlds using the
   per-slot action rows;
6. collect batched observations with one device-to-host download per frame;
7. route each collector to its stable tuple; and
8. close the environment.

The compatibility contract guarantees a common replay frame count. No padding,
ragged action mask, or per-world completion state is introduced in this slice.

`record_all_envs_step()` currently requires identical ordered junction names
across environments. The preflight check makes that requirement explicit
rather than allowing a late replay failure.

## Scoring and ranking

Scoring remains independent for every structure and candidate.

For each structure:

- gather that candidate's replay episodes in resolved direction order;
- compute per-direction instability fractions;
- disqualify according to the existing instability, missing-direction, and
  non-finite rules;
- score against only that structure's prepared recorded-GT Sinkhorn context;
- rank eligible candidates by ascending aggregate Sinkhorn;
- break ties by local candidate index; and
- mark the structure-specific GT candidate exactly as today.

No score, transition bag, rank, or penalty crosses a structure boundary.

The existing `YoungsModulusEvaluation` and
`YoungsModulusCandidateScore` contracts remain authoritative. This keeps
`ranking.json`, overlays, and optional replay exports backward compatible.

## Failure handling

### Preflight failures

Dataset loading, structure preparation, and compatibility errors occur before
fused model allocation. With default behavior, each bad structure is recorded
as skipped and valid structures continue. With `--fail-fast`, the first error
is raised.

### Candidate-local failures

Instability, missing replay bags, and non-finite Sinkhorn values disqualify
only the affected local candidate. They do not fail the fused chunk or another
structure.

### Fused runtime failures

A model build, reset, initialization, or replay-step exception may invalidate
every slot in that chunk. When `--fail-fast` is enabled, raise immediately.

Otherwise:

1. identify structures represented in the failed chunk;
2. discard any partial fused results for those structures;
3. exclude their remaining blocks from fused replay;
4. continue fused replay for unaffected structures; and
5. retry each affected structure independently through the existing
   single-structure evaluator.

This exceptional fallback preserves structure-local error reporting and can
recover from fused-batch memory pressure. Duplicate work on the fallback path
is acceptable because it is not the healthy performance path.

If an individual retry fails, record that structure's error as today. If every
requested structure fails, the command exits nonzero.

### Optional artifact failures

Overlay and replay-export failures remain isolated from valid numeric ranking
results. Fused replay does not change this behavior.

## Viewer behavior

Each replay chunk owns a different Newton model. Viewer integration must bind
the active chunk model before rendering it rather than retaining only the
first model. World offsets continue to use the environment's configured
spacing.

Viewer mode is for diagnosis. Performance acceptance and memory measurements
use `--viewer null` so rendering does not distort throughput.

## Diagnostics and performance measurement

Print concise run diagnostics containing:

- number of prepared, skipped, fused, and individually retried structures;
- total candidate blocks and flattened environments;
- chunk count and environments per chunk;
- model build/settle time;
- replay time;
- scoring/reporting time; and
- total wall time.

The CUDA acceptance run records:

- independent baseline wall time;
- fused wall time;
- model build count before and after;
- peak GPU memory before and after; and
- per-structure numerical parity.

GPU timing must synchronize before reading elapsed CUDA work. Memory
measurement must use a documented repeatable command or device API.

No universal speedup threshold is imposed before measurement. The structural
performance criterion is that a compatible workload fitting in one chunk
reduces model build/settle count from one per structure to one total. Measured
wall-time and memory results determine whether the default cap should change;
this slice does not change the 25,000 default speculatively.

## Testing

Follow red-green-refactor.

### Fast tests

1. Structure preparation preserves requested order and exact local candidate
   lists, including different structure-specific GT candidates.
2. Flattening produces deterministic
   `(structure, candidate, direction)` slots with correct applied parameters
   and sparse source direction IDs.
3. Whole-candidate chunking respects the environment cap and rejects a cap
   smaller than the common direction count.
4. Compatibility validation rejects mismatched topology metadata, junction
   names, frame counts, direction layouts, controller settings, grippers, and
   observation shapes before environment construction.
5. Per-environment initialization selects the correct structure/direction
   metadata, joint state, VIC defaults, and TCP target.
6. Recorded actions are routed to the correct world and frame.
7. One build occurs when every slot fits; deterministic multiple build calls
   occur when chunked.
8. Routed collectors reconstruct the exact structure-local candidate and
   direction layout.
9. Multi-structure scoring matches independent single-structure scoring for
   aggregate Sinkhorn, per-direction Sinkhorn, instability, disqualification,
   rank, and GT identity.
10. Instability and non-finite loss remain candidate- and structure-local.
11. A fused runtime failure retries only affected structures when
    `--fail-fast` is disabled and raises immediately when enabled.
12. Existing ranking JSON, overlay, export, sparse-direction, strict-JSON, and
    all-fail behavior remain unchanged.

### CUDA acceptance

Use one small healthy `batched_sysid_v1` dataset with at least two compatible
structures, multiple directions, and a small E grid.

1. Run the existing independent path as a baseline.
2. Run the fused path with one chunk.
3. Assert the same eligible/disqualified candidates and ranks per structure.
4. Compare finite aggregate and per-direction Sinkhorn losses within numerical
   tolerance.
5. Record model build count, wall time, and peak GPU memory.
6. Repeat with a low environment cap to exercise deterministic multi-chunk
   replay.
7. Confirm overlays and optional exports remain readable.

The smoke does not require universal GT rank one on unhealthy samples and does
not invent a fixed speedup target.

## Documentation

Add a short implementation note under `docs/` describing:

- the stable replay-slot index contract;
- the compatibility assumptions;
- the code ownership boundary;
- exceptional fallback semantics;
- focused test modules; and
- exact independent/fused benchmark commands and measured results.

Update `docs/ROADMAP.md` validation commands when the canonical fused smoke is
known. Keep README commands synchronized only if the documented primary run or
installation workflow changes.

## Success criteria

- Compatible selected structures are replayed through one flattened
  structure-by-candidate-by-direction schedule.
- A workload fitting under the environment cap builds and settles one model
  instead of one model per structure.
- Each structure retains its exact candidate list and independent Sinkhorn
  ranking.
- Candidate E changes only primary, spur, and stem material-derived behavior;
  structure geometry and all other fixed parameters remain structure-local.
- Stable tuple indices preserve structure, local candidate, and source
  direction identity through chunking.
- Existing report, overlay, export, exclusion, and failure semantics remain
  backward compatible.
- Exceptional fused failures fall back only affected structures unless
  `--fail-fast` is enabled.
- Fast tests pass, the two-structure CUDA parity smoke passes, and measured
  RTX 4090 wall time and peak memory are recorded.
- CMA-ES remains deferred and can later consume the typed replay primitive
  without changing this grid's scientific semantics.
