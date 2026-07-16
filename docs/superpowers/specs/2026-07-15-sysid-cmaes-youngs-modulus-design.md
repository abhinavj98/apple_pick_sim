# Sys-ID CMA-ES over Young's modulus — design spec

**Date:** 2026-07-15  
**Roadmap slice:** V.5.2 (CEM / CMA calibration loop)  
**Builds on:** V.4.3 batched stiffness grid (`batched_sysid_mmd_grid.py`), V.5.1 hold Sinkhorn + pooled-directions scoring  
**Related:** `docs/system_identification.md` §4, `docs/material-parameter-sampling.md`, `docs/sysid-transition-features.md`

## Problem

V.5.1 shipped a GT-preferring hold-phase Sinkhorn Wasserstein score on `batched_sysid_v1` with pooled directions. V.5.2 must drive a continuous optimizer over material θ, not a manual stiffness grid.

Today’s grid (`example_batched_sysid_mmd_grid.py`) sweeps `bend_stiffness` multipliers. That is the wrong long-term θ: stiffness is geometry-entangled. For sim-to-sim calibration we want to sample **Young's modulus** \(E\) per woody segment, **re-derive** VBD bend/stretch (and bend damping from fixed \(\zeta\)) from geometry, score candidates in parallel with the existing batched replay path, and report how well fitted \(E\) recovers true \(E\).

## Goals

1. **CMA-ES (pycma)** over \(\theta = \log_{10}(E)\) for **primary, spur, stem** (no secondary in these structures).
2. Per candidate: set \(E\), keep \(\zeta\) / geometry / other fields from oracle `true_params_for_structure`, re-derive VBD knobs via `rod_params_from_material` (honor axial `vbd_stretch_fixed`-style overrides when present on the base rod).
3. **Fitness:** hold-phase Sinkhorn Wasserstein with **pooled directions** + dir one-hot (same contract as `gate_pooled_dirs` / CLI defaults `--use-median --hold-id-onehot --pool-directions`).
4. **Parallel eval:** each CMA generation packs all λ samples into one (chunked) batched replay, same spirit as the stiffness grid (`MAX_ENVS_PER_BATCH`).
5. **Independent CMA-ES per structure**; aggregate across structures: mean of fitted means, sample covariance of fitted \(\log_{10} E\), relative error of cross-structure mean vs mean true \(E\), and spread (std of fitted \(\log_{10} E\)).
6. **Sim-sim validation report** JSON (+ short GPU smoke script/CLI path).
7. Add **`cma` (pycma)** as a project dependency (appropriate uv extra).

## Non-goals

- Fitting damping ratio \(\zeta\), density, geometry, apple mass, or secondary \(E\).
- Held-out structures / chirp / discrete-tone validation (V.5.3).
- Replacing or removing the bend-stiffness grid tool (keeps diagnostic role).
- Wiring optional `--score-mmd` CLI for CMA (Wasserstein only for this slice).
- Real field data / observation-only digital-twin as the default init (oracle params remain default; `--infer-params` may stay available later but is out of path for V.5.2 acceptance).

## Approach (chosen)

**Material candidate + pycma ask/tell + batched generation eval** (not bend-stiffness-only mapping; not shelling out to the grid CLI each generation).

## Architecture

```text
batched_sysid_v1 dataset
        │
        ▼
per structure: true_params_for_structure (oracle geometry + ζ + true E)
        │
        ▼
pycma CMAEvolutionStrategy(μ0, σ0) in R^3 = log10([E_p, E_spur, E_stem])
        │
   ask() → λ vectors
        │
        ▼
YoungsModulusCandidate.apply_to(base) → FruitingSystemParams[]
        │
        ▼
batched replay (candidate × direction), hold features, pooled Sinkhorn W
        │
   losses[] (lower better)
        │
        ▼
tell(losses) → update μ, Σ
        │
   … generations …
        │
        ▼
StructureFitResult → AggregateReport (mean / cov / rel error vs true)
```

## Components

| Piece | Role |
| ----- | ---- |
| `YoungsModulusCandidate` | `NamedTuple(primary, spur, stem)` in Pa; `apply_to(base)` rebuilds each present rod |
| `candidates_from_log10_e` / `log10_e_from_params` | CMA ↔ physical / GT maps |
| `evaluate_youngs_modulus_candidates(...)` | Thin adapter into existing batched replay + hold Wasserstein pooled scorer |
| `run_cmaes_for_structure(...)` | pycma ask/tell loop; returns `StructureFitResult` |
| `aggregate_fits(...)` | Cross-structure mean, sample cov, rel error, spread |
| `example_batched_sysid_cmaes.py` | CLI entry (mirrors mmd_grid example constants / settle / ranges) |
| Optional `scripts/` wrapper | Collect (or reuse dataset) → CMA → write report (sibling spirit to `collect_and_rank_sysid_gt.sh`) |

### `apply_to` semantics

For each of `primary`, `spur`, `stem` present on `base`:

- Read geometry (`length`, `radius`, `density`, `num_segments`, `direction`) and `damping_ratio` from the base rod.
- Call `rod_params_from_material(E_candidate, ζ_base, …)`.
- If the base rod used fixed axial stretch overrides (fixture `vbd_stretch_fixed` path), pass those stretch stiffness/damping values through so bend follows \(E\) while axial pins stay as today.
- Leave absent segments as `None` (no secondary).

Do **not** only patch `bend_stiffness`; stretch and bend damping must stay consistent with the material derivation docs.

### Scoring contract (frozen for this slice)

- Hold phases only (same feature builders as V.5.1 median / hold-id bags).
- `--pool-directions` on with direction one-hot.
- Primary metric: Sinkhorn Wasserstein (minimize).
- MSE may be logged for diagnostics but does **not** drive CMA `tell`.

### CMA defaults (CLI-overridable)

| Knob | Default intent |
| ---- | -------------- |
| Dim | 3 = `log10([E_primary, E_spur, E_stem])` |
| \(μ_0\) | \(\log_{10}E\) of that structure’s **true** params (sim-sim) |
| \(σ_0\) | Broad in log decades (e.g. `0.5`–`1.0`) so search is not a trivial local check |
| Bounds | Fixture `youngs_modulus_pa` min/max (ranges JSON); clip or CMA bounds |
| Population | pycma default for `N=3`, or fixed `16`–`32` |
| Generations | Small smoke default (e.g. `5`–`10`); larger for real runs |
| Batching | Chunk by `MAX_ENVS_PER_BATCH` like the stiffness grid |

### Reporting

**Per structure** (`StructureFitResult`):

- Final CMA mean \(μ\) and covariance \(Σ\) in \(\log_{10}E\) space
- Best-ever sample + loss
- Loss history / n_evals
- True \(\log_{10}E\) and true \(E\) for comparison

**Aggregate** (`AggregateReport`):

- Mean of fitted means \(\barμ\) (and \(\bar E = 10^{\barμ}\) per segment)
- Sample covariance of the per-structure fitted means (3×3 in \(\log_{10}E\))
- Per-segment relative error: \((\bar E - \mathrm{mean}(E_{\mathrm{true}})) / \mathrm{mean}(E_{\mathrm{true}})\)
- Spread: std of fitted \(\log_{10}E\) (and optionally of \(E\)) across structures

Write JSON under `--output` (and a short human-readable summary print).

## Dependency

- Add **`cma`** (pycma) via `uv add` to the package that owns the gym/sysid tooling (likely root / gym extra). Document `uv sync` extras in README / ROADMAP validation notes if the extra set changes.

## Testing

### Unit / fast (CI)

- `apply_to` rederives bend/stretch from \(E\); \(\zeta\) and geometry frozen; secondary remains `None`.
- \(\log_{10}E\) ↔ candidate round-trip; bound clip/reject behavior.
- ask/tell loop with a **mocked** scorer (known bowl around true \(\log_{10}E^*\)): few gens, assert \(μ\) moves toward truth.
- `aggregate_fits` math on synthetic fits → expected mean / cov / relative error.
- CLI `--help` / argparse smoke.

### GPU smoke (acceptance, not flaky CI gate)

Short scripted path (documented in ROADMAP validation / script):

1. Tiny `batched_sysid_v1` dataset (e.g. 1–2 structures, few directions) **or** reuse an existing settled dataset.
2. Run CMA with small `λ` and few generations on CUDA.
3. Emit the JSON report; assert process exit 0 and report schema present (mean, cov, true \(E\), relative error fields).

Optional stronger smoke (manual): mean fitted \(E\) closer to true than a far random init would be; do **not** require GT rank guarantees under adversarial \(σ_0\).

### Sim-sim validation (manual / wrapper)

Fuller collect → CMA → inspect relative error of \(\bar E\) vs mean true \(E\) and covariance spread — same spirit as `collect_and_rank_sysid_gt.sh`, but reporting continuous fits instead of discrete grid ranks.

## Error handling

- Non-finite / disqualified candidate scores: assign a large finite loss (or pycma-compatible penalty) so ask/tell continues; log count of penalized samples.
- Empty / all-excluded structure: skip with explicit error in report, do not crash the whole multi-structure run unless `--fail-fast`.
- Missing primary/spur/stem on params: hard error for this slice (structures without secondary are expected; three rods required).

## File map (planned)

| Path | Change |
| ---- | ------ |
| `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` | Library: candidate, eval adapter, CMA loop, aggregate |
| `apple_pick_gym/batched_examples/example_batched_sysid_cmaes.py` | CLI |
| `apple_pick_gym/tests/test_batched_sysid_cmaes*.py` | Unit + mock loop + aggregate |
| `scripts/` (optional) | Collect-or-reuse → CMA → report |
| `docs/ROADMAP.md` / `docs/system_identification.md` | Mark V.5.2 progress; validation commands |
| `pyproject.toml` | `cma` dependency |

Reuse without reimplementing: `replay_batched_sysid_structure` / grid batching helpers, hold Wasserstein scorers, `true_params_for_structure`, `rod_params_from_material`.

## Success criteria

- [ ] CMA-ES samples \(\log_{10}E\) and rederives material knobs from geometry.
- [ ] Each generation evaluated via batched parallel hold Wasserstein (pooled dirs).
- [ ] Per-structure mean/cov and cross-structure aggregate report with relative error vs true \(E\).
- [ ] Fast tests green; GPU smoke documented and runnable.
- [ ] ζ and non-\(E\) fields remain fixed.

## Open points resolved in brainstorming

| Topic | Decision |
| ----- | -------- |
| Segments | primary + spur + stem only |
| Optimizer | CMA-ES via **pycma** (`cma`) |
| Representation | \(\log_{10}E\) |
| Scoring | Sinkhorn Wasserstein, **hold only**, **pooled directions** |
| Parallelism | Pack generation into batched grid-style eval |
| Organization | Per-structure CMA; then cross-structure mean/cov report |
| Validation report | Mean relative error vs true + spread/cov |
| Rederive | Full material→VBD from \(E\), not bend-only patch |
| Acceptance | Includes short **GPU smoke** (schema + exit 0; not flaky CI gate) |

## First interactive slice (shipped 2026-07-15)

Shipped ahead of the full CMA loop for visual verification:

| Piece | Location |
| ----- | -------- |
| `YoungsModulusCandidate` + log10 maps | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` |
| `set_rod_youngs_modulus` | `apple_pick_sim/fruiting_system/params.py` |
| Keyboard E-grid + soft-disable | `apple_pick_gym/batched_examples/example_batched_youngs_modulus_keyboard.py` |
| Collect × directions + faceted overlay HTML | `example_batched_youngs_modulus_collect_viz.py`, `youngs_modulus_overlay_viz.py` |

Overlay hygiene: facet by pull direction; norms ‖F‖/‖T‖/‖Δtcp‖ by default; `--max-overlay-candidates` (default 8); phase as vrects; excluded episodes omitted.

Still open for V.5.2: GT recorded-action replay eval adapter, pycma ask/tell, aggregate fit report, `cma` dependency.
