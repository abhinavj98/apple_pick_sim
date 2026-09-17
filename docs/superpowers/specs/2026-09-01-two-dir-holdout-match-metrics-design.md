# Two-direction holdout CMA with match metrics (s04 / s05 / s09)

| Field | Value |
| ----- | ----- |
| **Status** | Draft — pending review |
| **Canonical living docs after impl:** | H5 `docs/handbook-youngs-cma.md`; H3 `docs/handbook-sysid-scoring.md` |
| **Date** | 2026-09-01 |
| **Roadmap** | Experiment protocol on `feature/real-replay-parallel-sysid`. **Not** the M4.0 Task 9 science-gate pass. |
| **Extends** | `docs/superpowers/specs/2026-08-17-one-structure-multidir-holdout-cmaes-design.md` (explicit train/val indices); `docs/superpowers/specs/2026-08-31-cma-generation-trajectory-persist-design.md` (`dir_XX.npz` keys) |
| **Reference data** | `robot_replay/final_data_correct_torque/s04`, `s05`, `s09` — eight compiled `sXX-dNN.parquet`, `NN ∈ {0…7}` |

## Purpose

Fit 5D CMA-ES on **six** pull directions, hold out **one horizontal and one same-azimuth oblique** (dirs **0** and **3**), and report a time-series match table on those two val dirs. Repeat on trees **s04, s05, s09** with **three CMA seeds**, three jobs at a time on the workstation GPU.

CMA **fitness stays pooled Sinkhorn** on train dirs. The new stats are **report-only**.

## Locked decisions

| Topic | Choice |
| --- | --- |
| Trees | **s04, s05, s09**. One structure per job. Keep the `vic_pose` one-structure `SystemExit`. Do not merge trees. |
| Directions | Shared catalog. **d00** ≈ horizontal −Y; **d03** ≈ same-azimuth oblique in YZ. |
| Train | `{1, 2, 4, 5, 6, 7}` |
| Val / report | `{0, 3}` |
| Split flags | Existing `--direction-indices 1,2,4,5,6,7 --val-direction-indices 0,3`. Disk still has eight usable dirs. |
| Seeds | `--cma-seed` **56, 57, 58** (shipped default plus two neighbors). Direction split is explicit, not seed-dependent. |
| Jobs | **9** independent CMA runs (3 trees × 3 seeds). |
| GPU schedule | **3 concurrent CMA parents** on the workstation RTX 4090. Pack **one tree’s three seeds** per wave (s04 56/57/58, then s05, then s09). Isolated eval waves mean each parent also has one worker: three jobs ⇒ three GPU contexts, not nine. |
| CMA knobs | **Current real `vic_pose` defaults.** Mean: support \(k_p = 500\,\mathrm{N/m}\), all four \(E = 100\,\mathrm{MPa}\). Bounds: \(k_p \in [200, 1000]\,\mathrm{N/m}\); flexural+axial \(E \in [100\,\mathrm{kPa}, 10\,\mathrm{GPa}]\). \(\sigma=0.2\), \(\sigma_{\max}=0.5\), `population_size=20`, `max_generations=20`. Do not copy the pinned-\(k_p\) wide-\(E\) box from `tmp/cma_5d_s04_wide_e_kp15`. |
| Fitness | Unchanged complete pooled Sinkhorn on **train** dirs only. No `tell()` on val. |
| Freeze | CMA `final_mean`. Holdout already replays val at **baseline** (effective real initial mean) and **fitted**. |
| New metrics | Computed **during holdout eval** while real / baseline / fitted episodes are in memory (approach A). |
| H3 gates | Still written to `holdout_report.json`. A gate fail **must not** skip `match_metrics.json`. This experiment does not treat Task 9 gate pass as success. |
| Convert | s05 and s09 from `robot_replay/final_data_correct_torque/sXX`. Reuse `tmp/real_batched_s04` if that bag is still valid; otherwise reconvert s04 the same way. |

## Pipeline

```text
final_data_correct_torque/sXX/sXX-dNN.parquet
        │
        ▼
 convert --input-dir  →  tmp/real_batched_sXX
        │
        ▼
 CMA  --direction-indices 1,2,4,5,6,7
      --val-direction-indices 0,3
      --cma-seed {56,57,58}
        │  ask/tell on train only
        ▼
 freeze final_mean
        │
        ▼
 holdout eval (no optimizer)
   replay val {0,3} at baseline mean and fitted mean
        │
        ├─ holdout_report.json          (existing Sinkhorn / F/T MAE / H3 gates)
        ├─ structure_000/holdout/*.html (existing overlays)
        ├─ match_metrics.json           (new)
        └─ structure_000/holdout/{baseline,fitted}/dir_{00,03}.npz
```

After all nine jobs, a **post-hoc aggregator** (script, not the CMA loop) prints per-tree mean±std of val **fitted** match metrics across the three seeds.

## Signals and windows

Compare real vs sim on each val direction. Real wrench uses `scored_ft_wrist` (convert-time `ft_wrist_lpf` when present). Sim wrench is unfiltered `ft_wrist`. Woody starts are `primary_spur` and `spur_stem` XYZ in `junction_names` order. **Apple pose is not in this table.**

**Windows** (both always computed):

| Window | Frames |
| --- | --- |
| `full` | Post-weld runtime (strip `step_idx < 0` / pre-weld the same way generation persist / overlays do), minus `stable=False`. If `stable` is missing, treat all frames as stable. |
| `hold` | Stable hold segments from `iter_kept_hold_segments` (same as H3). |

Truncate pairwise to `T = min(T_real, T_sim)` after the window mask. Empty window → all stats `nan` (no throw).

**Channels:**

| Family | How |
| --- | --- |
| Force per-axis | `Fx, Fy, Fz` |
| Torque per-axis | `Tx, Ty, Tz` |
| Force combined | \(F(t)\in\mathbb{R}^3\) |
| Torque combined | \(\tau(t)\in\mathbb{R}^3\) |
| Woody | each junction as \(p(t)\in\mathbb{R}^3\) |

There is **no** 6-vector combined wrench.

## Metric formulas

Let \(x_{\mathrm{real}}(t), x_{\mathrm{sim}}(t)\) be a scalar series (one F/T axis). Residual \(e_t = x_{\mathrm{sim}}-x_{\mathrm{real}}\).

| Stat | Scalar series |
| --- | --- |
| MSE | \(\mathrm{mean}_t e_t^2\) |
| Bias | \(\mathrm{mean}_t e_t\) |
| Error variance | \(\mathrm{mean}_t (e_t - \mathrm{bias})^2\) |
| R² | \(1 - \mathrm{SS}_{res}/\mathrm{SS}_{tot}\) with \(\mathrm{SS}_{res}=\sum e_t^2\), \(\mathrm{SS}_{tot}=\sum (x_{\mathrm{real}}-\bar x_{\mathrm{real}})^2\). `nan` if \(\mathrm{SS}_{tot}=0\). |
| Cross-corr lag | Integer lag (frames) maximizing **normalized** cross-correlation of the two series. Search \(\pm 1\,\mathrm{s}\) at 30 Hz (\(\pm 30\) frames). Positive lag means sim lags real. |
| Phase-corrected RMSE | RMSE after shifting sim by that lag and trimming to the overlap. If lag is `nan`, this is `nan`. |
| Peak magnitude error | \(\lvert \max_t x_{\mathrm{sim}} - \max_t x_{\mathrm{real}} \rvert\) (signed-axis peak, not \(\lVert F\rVert\)). |

Let \(v(t)\in\mathbb{R}^3\) be combined F, combined \(\tau\), or a woody start. Residual \(e(t)=v_{\mathrm{sim}}-v_{\mathrm{real}}\).

| Stat | 3-vector series |
| --- | --- |
| MSE | \(\mathrm{mean}_t \|e(t)\|^2\) |
| Bias | \(\mathrm{mean}_t e(t)\) (JSON length-3 list) |
| Error variance | \(\mathrm{mean}_t \|e(t)-\mathrm{bias}\|^2\) (scalar) |
| R² | \(1 - \sum\|e\|^2 / \sum\|v_{\mathrm{real}}-\bar v_{\mathrm{real}}\|^2\); `nan` if the denominator is 0 |
| Cross-corr lag | Same lag search on the scalar \(\|v(t)\|\) |
| Phase-corrected RMSE | RMSE of \(\|v_{\mathrm{sim}}(t-\mathrm{lag})-v_{\mathrm{real}}(t)\|\) on the overlap (3-vector Euclidean RMSE: \(\sqrt{\mathrm{mean}\|e\|^2}\) after shift) |
| Peak magnitude error | \(\lvert \max_t \|v_{\mathrm{sim}}(t)\| - \max_t \|v_{\mathrm{real}}(t)\| \rvert\) for F and \(\tau\); for woody, \(\lvert \max_t \|p_{\mathrm{sim}}-p_{\mathrm{sim}}(0)\| - \max_t \|p_{\mathrm{real}}-p_{\mathrm{real}}(0)\| \rvert\) (displacement from the window’s first frame) |

Baseline and fitted each get a full copy of this table vs the same real traces.

## Artifacts

Each job `--output` (example `tmp/cma_s04_val03_seed56/`):

```text
<output>/
  cmaes_report.json
  holdout_report.json
  match_metrics.json
  structure_000/holdout/
    direction_000.html
    direction_003.html
    baseline/dir_00.npz
    baseline/dir_03.npz
    fitted/dir_00.npz
    fitted/dir_03.npz
```

`npz` keys match `build_direction_state_npz` in `apple_pick_gym/cma_generation_persist.py` (reuse that helper; do not invent a second schema).

`match_metrics.json` is JSON-only (no ndarrays). Required keys:

- `tree` (string, e.g. `"s04"`). Taken from the `--dataset` directory name: `real_batched_s04` → `s04`. If the folder name does not contain `sNN`, write the directory stem as-is.
- `cma_seed` (int)
- `train_direction_indices` / `val_direction_indices` (sorted int lists)
- `phenotype_log10.baseline` / `phenotype_log10.fitted` (length-5 log10 lists)
- `directions.<id>.<window>.<side>` for `id ∈ {0,3}`, `window ∈ {full, hold}`, `side ∈ {fitted, baseline}`:
  - `force.fx|fy|fz` and `torque.tx|ty|tz`: seven scalar stats
  - `force.combined` and `torque.combined`: 3-vector stats (bias is a length-3 list)
  - `woody.primary_spur` and `woody.spur_stem`: 3-vector stats

Metric or `npz` write failures append to `StructureCmaState.artifact_errors` and **do not** fail the structure (same contract as generation persist / overlays).

## CLI

No new flag is required to select dirs 0 and 3. Add `--write-match-metrics` / `--no-write-match-metrics`: default **on** when holdout mode is active; `--no-write-match-metrics` skips `match_metrics.json` (val `npz` still written unless generation-persist is independently disabled — val persist is **on** whenever holdout runs, including `--no-write-match-metrics`, so a later script can recompute stats without a GPU replay).

Example (knobs are the in-code real defaults; only seed and split are explicit):

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input-dir robot_replay/final_data_correct_torque/s05 \
  --dataset-out tmp/real_batched_s05 \
  --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s05 \
  --output tmp/cma_s05_val03_seed56 \
  --direction-indices 1,2,4,5,6,7 \
  --val-direction-indices 0,3 \
  --cma-seed 56 \
  --viewer null \
  --overwrite
```

Repeat for s04/s09 and seeds 57/58. Launch at most three of these CMA processes at once.

## Code map

| Responsibility | Module |
| --- | --- |
| Pure stats (MSE, bias, variance, R², lag, phase-corrected RMSE, peak) | **Create** `apple_pick_sim/system_id/match_metrics.py` |
| Assemble `match_metrics.json`; persist val `npz`; hook after val replay | **Modify** `apple_pick_gym/batched_envs/holdout_evaluation.py` (`run_holdout_evaluation`) |
| Reuse `npz` arrays | `apple_pick_gym/cma_generation_persist.py` (`build_direction_state_npz`) |
| CLI default-on flag | `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` |
| Seed-wise summary table | Small script under `apple_pick_gym/viz/` or `apple_pick_gym/batched_examples/` — **after** the nine jobs, not in the CMA hot path |

Sinkhorn, pycma `ask`/`tell`, isolated eval waves, and H3 gates are unchanged.

## Tests (CPU only)

- `apple_pick_sim/tests/test_match_metrics.py`:
  - identical series → MSE 0, bias 0, variance 0, R² 1, lag 0, peak 0
  - constant offset → MSE = bias², variance 0
  - sim shifted by 5 frames → lag 5; phase-corrected RMSE matches unshifted RMSE within float tolerance
  - zero real variance → R² is `nan`
  - empty series → all `nan`
  - 3-vector: bias length 3; MSE = mean \(\|e\|^2\)
- Extend `apple_pick_gym/tests/test_holdout_evaluation.py`:
  - fake real/baseline/fitted episodes for dirs 0 and 3 → JSON has `full` and `hold`, per-axis F/T, combined F/T, two woody junctions, both sides
  - missing val direction → `nan`s + artifact error, JSON still written
- CLI: holdout flags write `match_metrics.json` and val `npz`; `--no-write-match-metrics` skips JSON but still writes val `npz`

Do not put the 9 GPU jobs in pytest.

## Out of scope

- Changing CMA fitness or `force_magnitude_weight`
- Combined 6-vector wrench
- Scoring apple pose / TCP in `match_metrics.json` (overlays remain)
- Dropping the one-structure `vic_pose` guard or merging trees
- Treating H3 `force_magnitude_ok` / Task 9 as this experiment’s pass criterion
- Replaying leftover dirs `{1,2,4,5,6,7}` as extra val
- Unattended multi-machine launch (protocol allows it; this workstation is a pool of 3)

## Success

Implementation: unit tests above green; one CPU-level holdout assembly test writes `match_metrics.json` with the locked schema.

Experiment: nine jobs finish (or recorded failures); each successful job has `match_metrics.json` for dirs 0 and 3 (full + hold, baseline + fitted); seed-summary table exists per tree. **Do not** claim the M4.0 science gate passed from this table.
