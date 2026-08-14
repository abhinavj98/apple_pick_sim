# Batched sysID MMD grid — visualization design (RMSE/MSE landscape)

> **Hold MSE note (2026-07-15):** console `--score-mse` with `--use-median` uses `trajectory_paired_hold_median_mse` (not the legacy flat `trajectory_hold_aggregated_mse` bag). See `docs/sysid-mmd-grid-replay-alignment.md`.


**Status:** Historical
**Canonical living doc:** `docs/handbook-sysid-scoring.md`

Date: 2026-07-07  
Scope: visualization + reporting only (no changes to simulation/replay correctness)

## Goal

Provide a small set of plots that make two claims easy to verify for a bend-stiffness candidate grid evaluated using `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`:

1. **GT candidate has the lowest aggregate error** under a single scalar metric.
2. **Candidates closer to GT parameters have lower error** (on average / in trend).

Primary intended use is interactive inspection during grid runs and generating figure-quality snapshots from saved results.

## Inputs and assumptions

- Candidates are bend stiffness tuples `BendStiffnessCandidate(primary, secondary, spur, stem)`.
- For the target use case, **`secondary` is null/disabled**, so visualization focuses on **(primary, spur, stem)**.
- Candidate evaluation computes replay-vs-recorded metrics in two paths:
  - Console `--score-mse`: `trajectory_mse()` / `trajectory_hold_aggregated_mse()` in `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`
  - Viz JSON/plots: `build_grid_viz_rows()` in `apple_pick_gym/grid_viz_table.py` (hold metrics use the same latter-half hold aggregation as `--score-mse` when `mse_hold_aggregation` is `mean` or `median`)
- See `docs/sysid-mmd-grid-replay-alignment.md` for pre-weld frame alignment and weld semantics.
- This slice does **not** depend on MMD scoring (`aggregate_mmd2`). If/when MMD is used, it can be added as an alternate metric.

## Scalar error definition (rank + color)

We want to support **multiple error views** so we can separately validate “GT is best” for TCP pose and for F/T.

Define the following scalar errors per candidate (per structure) from the already-available mean-over-directions outputs.

### Error (TCP pose + apple pose; unit-consistent in \(m^2\))

\[
  \mathrm{err}_{pos} = w_{\mathrm{tcp}} \cdot \mathrm{tcp\_pos\_mse}
                     + w_{\mathrm{apple}} \cdot \mathrm{apple\_pos\_mse}
\]

Defaults:
- \(w_{\mathrm{tcp}} = 1\)
- \(w_{\mathrm{apple}} = 1\)

Rationale:
- Both terms are position MSE in \(m^2\) so the base metric is unit-consistent.
- Force/torque terms can be added later with explicit unit weights if needed.

### Error (F/T readings; separate force and torque RMSE)

Use the already available mean-over-directions RMSE values:
- `ft_force_rmse` (N)
- `ft_torque_rmse` (N·m)

We keep these as **separate scalar errors** (they have different physical units):

\[
  \mathrm{err}_{F} = \mathrm{ft\_force\_rmse}
\]

\[
  \mathrm{err}_{\tau} = \mathrm{ft\_torque\_rmse}
\]

### Optional convenience score (not used for unit-consistent claims)

For a single “overall” ranking, define an *optional* normalized combination (disabled by default):

\[
  \mathrm{err}_{combo} = \alpha \cdot \frac{\mathrm{err}_{pos}}{s_{pos}}
                       + \beta  \cdot \frac{\mathrm{err}_{F}}{s_{F}}
                       + \gamma \cdot \frac{\mathrm{err}_{\tau}}{s_{\tau}}
\]

where \(s_{pos}, s_F, s_{\tau}\) are robust scales (e.g. median over candidates) to make terms dimensionless.

## Distance-to-GT definition

Let \(k = (k_{primary}, k_{spur}, k_{stem})\) and \(k_{GT}\) be the GT triple for the same structure.

Use **log-space L2 distance**:

\[
  d(k) = \left\lVert \log(k + \epsilon) - \log(k_{GT} + \epsilon) \right\rVert_2
\]

with \(\epsilon \approx 10^{-12}\) to avoid \(\log(0)\) if any disabled segment uses `0.0`.

## Visualizations

### Plot A — 3D stiffness scatter + error color (primary/spur/stem)

**Purpose**: visually confirm GT is the minimum and see the local basin.

- Axes: \(x=\) primary, \(y=\) spur, \(z=\) stem (all **log-scaled**).
- Color: choose one of \(\mathrm{err}_{pos}\), \(\mathrm{err}_{F}\), or \(\mathrm{err}_{\tau}\) (perceptually uniform colormap).
- GT candidate: distinct marker (e.g. star) + outline, plus label.
- Side annotation / title includes:
  - `min_err` (for the selected metric), argmin candidate index, and whether it equals GT
  - GT’s metric value

### Plot B — distance-to-GT vs error scatter + trend

**Purpose**: directly test “closer params → lower error”.

- \(x = d(k)\) (log-space distance)
- \(y\) is one of \(\mathrm{err}_{pos}\), \(\mathrm{err}_{F}\), or \(\mathrm{err}_{\tau}\)
- Add either a smooth trend (LOESS / spline) or binned median curve.
- Highlight GT at \(x=0\).

### Plot C — 2D heatmap slices (optional but recommended)

**Purpose**: clearer “error landscape” than 3D for publication/debug.

Small multiples:
- Heatmap over two stiffnesses (e.g. stem vs primary), color = `err`
- Fix the third stiffness (e.g. spur) at a few values:
  - slice at GT spur
  - one below GT spur
  - one above GT spur

## Reporting / acceptance checks

Per structure:
- **GT-min checks** (per metric): compute \(\mathrm{err}_{pos}\), \(\mathrm{err}_{F}\), and \(\mathrm{err}_{\tau}\) for all candidates; verify argmin equals GT tuple (or report top-k with deltas) for each metric.
- **Distance correlation** (per metric): compute Spearman correlation between \(d(k)\) and each metric; report the values.

Across structures (if multiple):
- Fraction where GT is ranked #1 (or within top-k).
- Distribution summary for correlation (median/IQR).

## Non-goals

- No changes to replay fidelity, dataset format, or simulation physics.
- No runtime optimization / GPU hot-path work.

## Test plan

- Unit test: synthetic candidate grid with constructed `err` that increases with log-distance, ensuring GT is the argmin and correlation is positive.
- Integration check: run `example_batched_sysid_mmd_grid.py --score-mse` on a small dataset and confirm Plot A/B match expectations.

