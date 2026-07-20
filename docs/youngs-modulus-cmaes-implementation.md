# Young's-modulus CMA-ES implementation notes

## Document status

| Field | Value |
| --- | --- |
| **Last reviewed** | 2026-07-17 |
| **Roadmap slice** | V.5.2 |
| **Status** | Implementation verified complete (Task 8 focused/full suites + CUDA 5×5 acceptance, 2026-07-17) |
| **Design** | `docs/superpowers/specs/2026-07-16-youngs-modulus-cmaes-loop-design.md` |
| **Grid contract** | Unchanged — `docs/youngs-modulus-sysid.md` |

Human-readable notes for the separate bounded pycma loop. The Cartesian-grid
CLI, ranking report, and ranking gate remain the diagnostic/acceptance path
documented in `docs/youngs-modulus-sysid.md`.

## Behavior summary

`example_youngs_modulus_cmaes.py` fits primary/spur/stem Young's modulus in
\(\log_{10} E\) for each selected `batched_sysid_v1` structure. One independent
pycma optimizer owns each structure's mean, covariance, fitness penalties, seed
stream, and stop criteria. Active optimizers advance in **synchronized
generation waves**: every active structure calls `ask()` once, all populations
go through one logical fused evaluation
(`structure × local candidate × physical direction`), then each successful
structure calls `tell()` with the original `ask()` samples in original order.

When an optimizer stops (generation cap and/or pycma native stop), the CLI
snapshots bounded phenotype mean `es.result.xfavorite` (not `es.mean` or
`xbest`). After the active set empties, all snapshots are scored in one final
logical fused wave (one candidate per structure). That measured final mean is
the fitted estimate. Best sampled candidates and covariance diagnostics remain
reporting-only.

Stored GT E is never used for initialization or fitness. Bounds and replay
`sim_build` knobs come from one resolved ranges fixture (`--ranges`, else
manifest `collection.ranges_path`). Relative paths resolve from the process
CWD; the report stores the absolute path. Missing ranges fail before optimizer
construction — there is no unrelated default-fixture fallback.

Default controls: `--max-generations=10`, `--initial-sigma-log10=1.0`,
`--cma-seed=0`, `--multi-structure-batch` on. Before bound truncation, sigma
`1.0` makes roughly two decades (`mean/100` … `mean*100`) a two-standard-
deviation exploration range; fixture bounds remain authoritative.

## Log-space math and covariance terminology

- Optimizer coordinates are \(\log_{10}([E_\mathrm{primary}, E_\mathrm{spur}, E_\mathrm{stem}])\).
- Invalid/disqualified samples in a generation with at least one finite eligible
  score receive penalty `worst_finite + max(1, abs(worst_finite))`. Overflow or
  an all-invalid generation fails that structure without `tell()`.
- Per-structure covariance in `cmaes_report.json` reports pycma `C`, global
  `sigma`, `sigma_vec.scaling`, phenotype-coordinate standard deviations, and
  effective unbounded optimizer-coordinate covariance
  \(\sigma^2 \mathrm{diag}(\mathrm{scale})\,C\,\mathrm{diag}(\mathrm{scale})\).
  Do **not** call this the exact covariance of the nonlinearly bounded phenotype.
- Cross-structure aggregate sample covariance/std of fitted log10-E use
  `ddof=1` and are JSON `null` when fewer than two structures fit.
- `evaluated_history_extrema` is the component-wise min/max of samples actually
  submitted in CMA populations (`ask_samples_log10`), in both log10-E and Pa.

## Logical versus physical counters

| Counter | Meaning |
| --- | --- |
| `completed_generations` | Successful `tell()` calls for that structure |
| `optimizer_samples_told` | Population samples included in those `tell()` calls |
| `replay_candidate_evaluations` | Logical population + final-mean candidates submitted for scoring |
| `final_mean_evaluations` | `0` or `1` |
| `physical_env_slots` | Physical replay environment slots used. Batch total prefers fused `diagnostics.flattened_envs`; per-structure counts come from prepared usable directions (`candidates × len(direction_indices)`), including scoring-failed-but-replayed structures. Scalar-only waves sum the same attributed slots. Manifest/CLI `num_directions` is not used for attribution. |
| `scalar_retries` | Affected-structure scalar retries after fused failure |

Scalar retries and physical chunking do not change pycma's logical evaluation
count. One logical wave may span multiple physical chunks.

## Routing, fallback, and accepted numerical noise

Stable identity is
`ReplaySlotKey(structure_idx, local_candidate_idx, direction_idx)`. Source
direction IDs may be sparse. Physical chunks never mix distinct
`structure_idx` values (cross-structure heterogeneous batches can collapse
per-direction trajectories); multiple candidates of one structure may still
share a chunk. Compatibility mismatch sends the whole submitted group through
per-structure scalar evaluation; a fused runtime failure retries only affected
structures. `--no-multi-structure-batch` forces the scalar debug path. Small
fused-versus-scalar Sinkhorn differences are accepted numerical noise —
compare routing, status, and schema, not exact score equality.

## Failure and checkpoint states

Structure statuses: `active`, `stopped_pending_final_evaluation`, `fitted`,
`failed`. `stop_kind` is `generation_cap`, `pycma`, or `both`. Failures carry a
stage (`prepare`, `generation_evaluation`, `all_invalid`, `penalty`,
`final_mean`) plus a message. Overlay errors are separate `artifact_errors` and
do not invalidate `fitted`.

The CLI writes `<output>/cmaes_report.json` atomically before the first replay
and after each wave / state transition / overlay attempt. `--overwrite` clears
only CMA-owned report/temp and selected-structure overlay targets. Continue
after structure-local failures by default; `--fail-fast` aborts; all-failed /
global-error / viewer-cancellation exit nonzero; partial numeric success exits
zero.

Each generation record includes `score_summary` (eligible/penalized Sinkhorn
mean, sample variance/std with `ddof=1`, best eligible) and `wave_seconds`.
Top-level `timing` stores `fit_seconds`, per-wave timings, and
`command_seconds`. After a successful fit the CLI also writes Plotly HTML:

- `generation_score_mean_variance.html` (all structures: mean + variance vs generation)
- `structure_XXX_generation_scores.html` (per structure: mean ± std and variance)
- `structure_XXX_optimizer_diagnostics.html` (mean trajectory vs GT, spur/stem mean
  path, ||mean−GT||, and sigma / phenotype std / effective-cov trace)
- `structure_XXX_spur_stem_sinkhorn_3d.html` (log10(spur), log10(stem), Sinkhorn;
  primary omitted; GT vertical marker + best/final mean marked)

Regenerate figures from an existing report with::

    uv run python -c "from pathlib import Path; from apple_pick_gym.youngs_modulus_cmaes_viz import write_cmaes_visualization_bundle; write_cmaes_visualization_bundle('tmp/task8_cuda_acceptance/cmaes_fused/cmaes_report.json', 'tmp/task8_cuda_acceptance/cmaes_fused')"

## Code map

| Responsibility | Owner |
| --- | --- |
| Bounds, seeds, pycma options, ask/tell wave, fit coordinator, snapshots, aggregates | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` |
| Fused structure×candidate×direction replay | `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` |
| CMA CLI, atomic `cmaes_report.json`, final-mean overlays | `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` |
| Generation score + spur/stem vs Sinkhorn Plotly figures | `apple_pick_gym/youngs_modulus_cmaes_viz.py` |
| Cartesian grid CLI (unchanged) | `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` |
| CMA integrity gate report | `apple_pick_gym/batched_envs/youngs_modulus_cmaes_gate_report.py` |
| Multi-seed CMA gate orchestration | `scripts/gate_youngs_modulus_cmaes.sh` |
| Grid ranking gate (unchanged) | `scripts/gate_youngs_modulus_sysid.sh`, `youngs_modulus_gate_report.py` |

Dependency: `cma` (pycma) on the `gym` extra, currently locked at `4.4.4`.

## CLI and gate interfaces

CMA fit (from repo root):

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset /tmp/batched_sysid_dataset \
  --output /tmp/youngs_cmaes \
  --max-generations 10 --cma-seed 0 --viewer null --overwrite
```

CMA-specific controls: `--ranges`, `--max-generations`, optional
`--population-size`, `--initial-sigma-log10`, `--cma-seed`. Grid-only controls
(`--log10-e-*`, `--include-gt-candidate`, `--max-candidates`, `--export-replays`,
`--max-overlay-candidates`) are absent.

Integrity gate (no GT-error threshold; defaults seeds `0,1,2`, five structures,
five directions):

```bash
bash scripts/gate_youngs_modulus_cmaes.sh
```

Optional env overrides include `SEEDS`, `NUM_STRUCTURES`, `NUM_DIRECTIONS`,
`MAX_GENERATIONS`, `INITIAL_SIGMA_LOG10`, `POPULATION_SIZE`, `RANGES`.

## Tests

| Module | Catches |
| --- | --- |
| `test_batched_sysid_cmaes_candidate.py` | Bounds, sigma, dedicated RNG/`seed=np.nan`, interleaved determinism, pycma import |
| `test_batched_sysid_cmaes_loop.py` | Wave routing, penalties, independent stop, `xfavorite`, covariance, report extrema/aggregates |
| `test_batched_sysid_multi_replay.py` | Fused slot planning, structure-homogeneous chunks, routing |
| `test_example_youngs_modulus_cmaes_cli.py` | Parser contract, ranges resolution, atomic report, continue/fail-fast, overlay isolation, exact `physical_env_slots` attribution |
| `test_youngs_modulus_cmaes_gate_report.py` | Integrity evidence, no GT threshold, strict JSON finalize |
| `test_gate_youngs_modulus_cmaes_script.py` | Shell defaults, CMA forwarding, retries, parallel seeds |
| Grid CLI/gate tests above remain the ranking regression path | Unchanged grid contracts |

Focused fast suite (does **not** claim full verification or CUDA smoke):

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_cmaes_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_cmaes_script.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_sysid_script.py \
  -q
```

## CUDA / full-build acceptance (passed 2026-07-17)

Documented acceptance collect is **5 structures × 5 directions**. Artifacts from
the verification run live under `tmp/task8_cuda_acceptance/`
(`validation_report.md`, `validation_report.json`, fused
`cmaes_fused/cmaes_report.json` + overlays, scalar smoke
`cmaes_scalar_smoke/`). For the fused CMA run, report at least:

1. Optimized final mean versus stored true (GT) parameters per structure.
2. Each structure's `evaluated_history_extrema` min/max (log10-E and Pa).
3. Final per-structure covariance diagnostics (`C`, `sigma`, scaling,
   phenotype std, effective unbounded covariance).

Also confirm synchronized structure×population×direction waves (with
structure-homogeneous physical chunks), independent histories/stop evidence,
one-candidate-per-structure final-mean wave, retry counters, report
checkpoints, and overlays. A small `--no-multi-structure-batch` smoke compares
routing/status/schema only. Exact fused/scalar Sinkhorn equality is not
required.
