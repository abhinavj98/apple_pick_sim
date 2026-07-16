# Young's-Modulus Sys-ID Smoke and Gate Design

**Date:** 2026-07-16  
**Status:** Approved

## Purpose

Validate the dataset-driven Young's-modulus replay path against known simulated
ground truth, report the GT Sinkhorn distance and rank for each structure, and
provide a repeatable multi-seed gate analogous to
`scripts/gate_sysid_gt_sinkhorn.sh`.

This work validates the shipped `example_youngs_modulus_sys_id.py`; it does not
change the candidate model, replay semantics, or Sinkhorn feature contract.

## Success criteria

1. A diagnostic smoke run collects one seed with five structures and one pull
   direction using the collector's standard trajectory defaults.
2. The smoke replay includes each structure's exact GT Young's-modulus candidate
   and reports its aggregate Sinkhorn distance and rank.
3. A reusable gate defaults to three seeds, five structures, and five directions.
4. A seed passes when all five structures are evaluable and GT ranks first for
   at least three structures.
5. The overall gate passes only when every configured seed passes.
6. Logs and machine-readable summaries remain available on both pass and failure.

The smoke is diagnostic rather than an implementation blocker. If it exposes a
ranking problem, investigate and fix an in-scope defect when possible, but still
implement the repeatable gate and report the unresolved evidence if no safe fix
is found.

## One-off smoke

Run the following stages from the repository root:

1. Collect a `batched_sysid_v1` dataset with seed 0, five structures, one
   direction, a null viewer, overwrite enabled, and the collection script's
   normal trajectory defaults.
2. Run the existing unstable-episode exclusion command in place.
3. Run `example_youngs_modulus_sys_id.py` with its default log10-E grid,
   pooled-direction Sinkhorn defaults, exact GT insertion enabled, and a null
   viewer.
4. Parse `ranking.json` and report, for every structure:
   - GT candidate index;
   - aggregate GT Sinkhorn distance;
   - per-direction GT Sinkhorn distance;
   - GT rank;
   - winning candidate index and whether it is GT.
5. Summarize how many of the five structures place GT at rank one.

If GT does not lead for a majority, inspect dataset exclusions, replay
instability/disqualification, candidate equality, replay configuration fidelity,
and per-direction scores before changing scoring behavior.

## Gate architecture

### Shell orchestrator

Add `scripts/gate_youngs_modulus_sysid.sh`.

The script follows the operational shape of `gate_sysid_gt_sinkhorn.sh`:

- resolve and enter the repository root;
- parse environment overrides;
- create a timestamped report root;
- run seeds in parallel with one log file and exit-status file per seed;
- retry collection, exclusion, and replay independently;
- invoke the report module for each seed;
- invoke the report module once more to finalize the cross-seed result;
- exit nonzero when any seed or finalization fails.

Defaults:

- `SEEDS=0,1,2`;
- `NUM_STRUCTURES=5`;
- `NUM_DIRECTIONS=5`;
- collector trajectory values remain its normal defaults unless explicitly
  overridden;
- GT insertion remains enabled;
- Sinkhorn scoring uses median hold features, hold-ID one-hot features, and
  pooled directions;
- viewer is null.

Environment overrides cover seeds, structure and direction counts, trajectory
parameters, settle parameters, retry count and delay, dataset prefix, report
root, and the three log10-E candidate axes.

### Python report module

Add a focused module under `apple_pick_gym/batched_envs/` for the
Young's-modulus gate report. It consumes the existing `ranking.json` contract
instead of coupling the shell script to JSON traversal.

Per-seed mode:

- validate the ranking report shape;
- require exactly the configured number of evaluated structures and no skipped
  structures;
- locate exactly one GT candidate per structure;
- extract aggregate and per-direction Sinkhorn distances and rank;
- count GT rank-one structures;
- apply the configurable majority threshold, defaulting to
  `floor(expected_structures / 2) + 1`;
- write a strict-JSON seed summary;
- return nonzero on validation or gate failure.

Finalization mode:

- load every expected seed summary;
- preserve each seed's pass/fail status and per-structure evidence;
- write one strict-JSON final summary;
- return nonzero unless every expected seed passed.

The module prints a compact human-readable summary as well as writing JSON.

## Data flow

```text
collector
  -> batched_sysid_v1 dataset
  -> unstable-episode exclusion
  -> Young's-modulus replay/ranking
  -> ranking.json
  -> per-seed gate summary
  -> final multi-seed summary and process exit status
```

`ranking.json` remains the source of truth for ranks and scores. The gate report
does not recompute Sinkhorn distances or rerank candidates.

## Failure behavior

- Collection, exclusion, replay, and reporting failures are isolated by seed.
- A failed seed does not stop other parallel seeds from completing.
- Missing or malformed ranking data fails that seed with a useful diagnostic.
- Any skipped structure fails the seed because the requested denominator is five
  evaluable structures, not merely the structures that happened to finish.
- Missing GT, duplicate GT rows, null GT rank, or a non-finite/missing aggregate
  GT Sinkhorn distance fails report validation.
- Candidate disqualification is retained in the summary and naturally prevents
  an absent rank-one result.
- Existing output directories are overwritten only where the invoked commands
  explicitly support it.
- The final report is attempted even after seed failures so partial evidence is
  retained.

## Testing

Follow TDD for the report module and shell wiring.

Unit tests cover:

- extraction of GT aggregate/per-direction distances and ranks;
- a three-of-five pass;
- a two-of-five failure;
- configurable expected structure count and majority threshold;
- skipped structures;
- missing, duplicate, or unranked GT candidates;
- malformed ranking JSON and missing scores;
- strict JSON output;
- finalization across all-passing and partially failing seed summaries.

Shell-focused tests or static assertions cover:

- expected defaults of three seeds, five structures, and five directions;
- collection, exclusion, replay, per-seed report, and finalization commands;
- forwarding candidate-grid and trajectory environment overrides;
- preserving logs and aggregating parallel exit statuses.

After unit tests, run the one-direction smoke. Then run the full default gate when
runtime permits. Report the exact commands, artifact paths, GT Sinkhorn distances,
ranks, and pass/fail counts.

## Out of scope

- Changing the Wasserstein/Sinkhorn feature definition.
- Adding synthetic collection to the Young's-modulus replay CLI.
- Changing the Young's-modulus candidate parameterization.
- Reusing or broadly refactoring the stiffness-grid `sysid_gate_report.py`.
- Requiring every structure to rank GT first; the approved policy is a strict
  majority per seed.
