# CMA per-generation sparse trajectory persist

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living docs after impl:** | `docs/handbook-youngs-cma.md` |
| **Date** | 2026-08-31 |
| **Roadmap** | Diagnostic instrumentation on `feature/real-replay-parallel-sysid`. **Not** M4.0 Task 9 science gate. |
| **Related** | H3 `docs/handbook-sysid-scoring.md`; H5 `docs/handbook-youngs-cma.md`; overlay viz `apple_pick_gym/youngs_modulus_overlay_viz.py` |

## Purpose

CMA already harvests every candidate’s replay bags in memory (`YoungsModulusEvaluation.replay_episodes`) and then drops them. Fitness is pooled Sinkhorn on the joint `STATE_VECTOR`; force MAE is a diagnostic. After a generation, we cannot tell whether a high score is wrench mismatch, woody/bend mismatch, or OT cross-matching.

Persist a **sparse** subset of each generation’s harvests (bags + full-`STATE_VECTOR` time-series HTML) so later runs can be inspected without re-replay.

## Problem

- A 30-gen 5D real-replay CMA run kept only `cmaes_report.json` scalars and a **final-mean** overlay of `‖F‖` / `‖τ‖` / `‖Δtcp‖`.
- Per-direction Sinkhorn can improve while diagonal `|F|` ratios plateau; the joint OT metric does not decompose by feature.
- Persisting all 20×8 bags per generation is unnecessary for that diagnosis and is large on disk.

## Locked decisions (do not reopen)

| Topic | Choice |
| ----- | ------ |
| Scope | Instrument **future** CMA runs only. Do not re-replay the finished 30-gen output. |
| Density | **Sparse**: three roles × all scored directions. Not the full population. |
| Roles | `best`, `mean`, `worst_force` (rules below) |
| Storage | Bags (`npz`) **and** one Plotly HTML per role, written in the CMA parent after each generation |
| Plot surface | Full physical `STATE_VECTOR` time series (real vs sim). No extra per-gen overlay-norms HTML. |
| Final-mean overlay | Unchanged (`structure_XXX/youngs_modulus_overlay.html`) |
| Default | Persist **on**; `--no-persist-generation-replays` skips bags and HTML |
| Writer process | **Parent only.** Isolated eval workers keep current pickle IPC; they do not write `generations/`. |
| Sinkhorn / fitness | Unchanged. No `force_magnitude_weight` change. |

## Roles

Eligible set = candidates that are not disqualified and have finite `aggregate_sinkhorn`.

| Role | Selection |
| ---- | --------- |
| `best` | Lowest eligible **penalized** fitness (same ranking as `best_eligible` in `score_summary`) |
| `mean` | Eligible sample **closest in Euclidean log10** to the **ask** distribution mean (`CmaGenerationRecord.ask_distribution.mean_log10`). This is the population that was just scored, not post-`tell` `xfavorite`. |
| `worst_force` | Eligible candidate with **largest** `mean_hold_force_err_n`. Missing/non-finite `mean_hold_force_err_n` is ineligible for this role only. |

Ties: lower `candidate_index`. If two roles pick the same candidate, write **both** role directories (duplicate bags allowed). If a role has no eligible candidate, omit that role directory and record the skip in `roles.json`.

## On-disk layout

Under existing `--output`:

```text
<output>/structure_XXX/generations/gen_YY/
  roles.json
  best/
    metadata.json
    dir_00.npz
    …
    features.html
  mean/
    …
  worst_force/
    …
```

- `XXX` / `YY` are zero-padded (`structure_000`, `gen_00`).
- One `npz` per scored direction id (the dataset’s `direction_idx`, not a packed 0..N-1 remap).
- Resume: existing generation folders stay. Re-evaluating the same `generation_index` **replaces** that `gen_YY/` tree.
- `--overwrite` deletes `structure_XXX/generations/` for the selected structures (same pass as overlay/checkpoint cleanup).

### `roles.json`

Index for the generation: `generation_index`, `structure_idx`, ask-mean log10, and per-role `{role, candidate_index, log10_e, fitness, mean_hold_force_err_n, path}`. Omit or null a role that was skipped.

### `metadata.json`

Role, `candidate_index`, full phenotype `log10_e` (length 3 or 5), `aggregate_sinkhorn`, `per_direction_sinkhorn`, hold-block diagnostics already on `YoungsModulusCandidateScore` (force/torque/woody/bend errors and sim/real force norms).

### `dir_XX.npz`

**Full runtime** trajectories (move + hold + other phases), not Sinkhorn hold-bags. Strip pre-weld / `step_idx < 0` the same way overlay does. Real and sim are **not** required to share `T`; do **not** interpolate. Each side is plotted against its own `sim_time`. If `stable` is missing, treat all frames as stable (`True`). If a direction is missing on either side, skip that `npz` and record an artifact error for the role.

Keys:

| Key | Shape / dtype |
| ----- | ------------- |
| `sim_time_real`, `sim_time_sim` | `(T_side,)` float64 |
| `phase_real`, `phase_sim` | `(T_side,)` int8 |
| `stable_real`, `stable_sim` | `(T_side,)` bool |
| `real_state` | `(T_real, D)` float32 — `build_state_matrix(real)` |
| `sim_state` | `(T_sim, D)` float32 — `build_state_matrix(sim)` |

`D` is `STATE_VECTOR` width (`23` at CMA’s two woody junctions). Real wrench uses `scored_ft_wrist` (convert-time `ft_wrist_lpf` when present). Direction files are `dir_{id:02d}.npz` using the dataset’s `direction_idx`.

## Feature HTML

One `features.html` per role (all directions in one file).

- **Rows:** scored directions.
- **Columns (6):** \(F_{xyz}\); \(\tau_{xyz}\); TCP \(v,\omega\) (6); TCP \(x,y,z\); woody starts (6); bend angles (2).
- Real = solid, sim = dashed, **same color per channel**.
- Hold/move phase bands as in `youngs_modulus_overlay_viz`.
- **Physical units**, not Sinkhorn-normalized divisors.
- Hover: phenotype + that direction’s Sinkhorn and hold `|F|` sim/real ratio when available.
- **Block L2 table** at the top: per-direction force / torque / woody / bend mean-hold errors from `mean_hold_block_errors` (same helpers as CMA diagnostics).

Plot/write failures append to `StructureCmaState.artifact_errors` and **do not** fail the structure (same contract as final-mean overlay).

## Hook

After a generation wave **successfully** `tell()`s and attaches `CmaGenerationRecord`, the parent calls a persist helper with `(structure_idx, record, YoungsModulusEvaluation)`.

Place the call where the CLI already sees the fused batch after isolated or in-process eval (do not add disk IO inside `cma_wave_evaluation_worker`). A small callback from `fit_youngs_modulus_cma` / the generation-wave function is allowed so tests can persist without the example CLI.

`--persist-generation-replays` default **true**; `--no-persist-generation-replays` no-ops the helper.

`cmaes_report.json` gains a compact `generation_artifacts` summary per structure (generations written, roles present, skip/error strings). Do not embed arrays in the report.

## Tests (CPU only)

- Role selection: best / nearest-to-ask-mean / worst force; disqualified excluded; missing force-err ineligible for `worst_force`; collisions write two roles with the same `candidate_index`.
- `npz` round-trip: `build_state_matrix` width, `scored_ft_wrist` on real when `ft_wrist_lpf` is present.
- HTML: two traces per channel (real/sim); row count = number of directions.
- CLI: persist default on; `--no-persist-generation-replays` does not write `generations/`; `--overwrite` removes `generations/`.
- Persist exception → `artifact_errors`, structure still `fitted`.

## Out of scope

- Re-analyzing or re-replaying `tmp/cma_5d_kp_flex_axial_30gen`.
- Persisting the full population.
- Per-generation `‖F‖`/`‖τ‖`/`‖Δtcp‖` overlay HTML (final-mean overlay stays).
- Changing Sinkhorn, pooling, `force_magnitude_weight`, or phenotype bounds.
- Snapshot video, holdout val overlays, Cartesian grid `--export-replays`.

## Handbook

After implementation, H5 documents the layout, three roles, CLI flags, and that bags are full runtime trajectories while Sinkhorn still uses hold-stable bags.
