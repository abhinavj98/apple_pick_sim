# Young's-modulus grid and CMA handbook

This is the canonical living reference for the support-joint \(k_p\) ×
spur/stem Young's-modulus phenotype, its Cartesian diagnostic, and the
simulation CMA-ES loop. Scoring mathematics belongs in H3; delivery status and
the next real-data acceptance work belong in `docs/ROADMAP.md`.

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-09-11 |
| Code owners | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`; `apple_pick_gym/batched_envs/cma_wave_evaluation.py`; `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py`; `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`; `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H2 `docs/handbook-variable-impedance.md`; H3 `docs/handbook-sysid-scoring.md`; H4 `docs/handbook-real-replay.md` |
| Archive specs | **Implemented:** `docs/superpowers/specs/2026-08-04-support-joint-kp-sysid-design.md`; **Superseded phenotype, implemented loop:** `docs/superpowers/specs/2026-07-16-youngs-modulus-cmaes-loop-design.md`; **Partial:** `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md`; **Implemented (holdout CLI; Task 9 science gate failed on torque):** `docs/superpowers/specs/2026-08-17-one-structure-multidir-holdout-cmaes-design.md` |

Related boundaries:

- H2 defines `vic` and `vic_pose` action/controller semantics.
- H3 defines `batched_sysid_v1`, `STATE_VECTOR`, normalization, and Sinkhorn
  scoring. This handbook does not duplicate that math.
- H4 owns real-log conversion and
  `real_batched_replay_build.make_real_replay_build_env_fn`.
- `docs/batched-stability-monitor-design.md` owns online stability monitoring
  and episode exclusion details.

## 1. Phenotype

**Cartesian grid (3D):** unchanged diagnostic axes

\[
\theta_{\mathrm{grid}} =
\left(
\log_{10} k_p^{\mathrm{support}},
\log_{10} E_{\mathrm{flex,spur}},
\log_{10} E_{\mathrm{flex,stem}}
\right).
\]

`iter_support_kp_youngs_candidates` and `candidates_from_log10_vector` length-3
samples set flexural spur/stem \(E\) only; axial moduli stay at the structure baseline.

**CMA-ES (9D):**

\[
\theta_{\mathrm{CMA}} =
\left(
\log_{10} k_p^{\mathrm{support}},
\log_{10} E_{\mathrm{flex,spur}},
\log_{10} E_{\mathrm{flex,stem}},
\log_{10} E_{\mathrm{ax,spur}},
\log_{10} E_{\mathrm{ax,stem}},
\log_{10} k_p^{\mathrm{support,roll}},
\zeta_{\mathrm{spur}},
\zeta_{\mathrm{stem}},
\zeta_{\mathrm{support}}
\right).
\]

Dims 0–5 are log10 stiffness; dims 6–8 are **linear** damping ratios in
\([0, 1]\) (spur/stem rod \(\zeta\), support-joint \(\zeta\)). Primary rod
damping stays fixture/base frozen.

`SupportKpYoungsCandidate` stores `(support_kp, spur, stem)` as **flexural** moduli
plus optional `(spur_youngs, stem_youngs)` axial moduli, optional
`support_roll_kp` (N·m/rad), and optional spur/stem/support ζ fields.
Length-9 `candidates_from_log10_vector` maps all nine axes; length-6 leaves ζ
`None`; length-5 leaves roll and ζ `None`; length-3 leaves axial, roll, and ζ
`None`.

`SupportKpYoungsCandidate.apply_to` uses **`set_rod_flexural_modulus`** on spur/stem,
then **`set_rod_youngs_modulus`** when axial slots are set, then
**`set_rod_damping_ratio`** when spur/stem ζ are set. Fused replay separately applies the
per-environment support linear/angular penalties through
`apply_per_env_support_joint_penalties` and T-roll through
`apply_per_env_support_roll_penalties` / `support_roll_kp_per_env` (before settle),
using per-candidate `support_joint_zeta` when present (else the dataset
`joint_damping_ratio`). Real `vic_pose` production builds also pass
`support_zeta_per_env` into the weld re-apply path so snapshots capture the
searched ζ.
Support \(k_d\) is derived using that ζ and each
child body's mass/inertia. T-junction world clamps (`primary_support_left/right`)
are authored **soft** revolute hinges about the primary axis (VBD penalty-only,
`vbd:joint_is_hard=0`). Linear \(k_p\) and pitch/yaw angular
\(k_p=\tfrac{3}{4}L^2 k_{\mathrm{lin}}\) use penalty slots 0–1; T-roll writes both
revolute drive `target_ke` and penalty slot `c0+2` so runtime
`min(penalty_k, target_ke)` matches the candidate (episode snapshots restore both).
Other rod welds stay hard `FIXED`.
Build-time and CMA-apply support pitch/yaw angular \(k_p\) is

\[
k_{\mathrm{ang}} = \tfrac{3}{4}\, L_{\mathrm{dowel}}^{2}\, k_{\mathrm{lin}},
\]

with \(L_{\mathrm{dowel}}\) the **primary rod length** (metres) between the T-junction
world clamps. CMA searches linear \(k_{\mathrm{lin}}\) (N/m) and T-roll
\(k_p^{\mathrm{support,roll}}\) (N·m/rad); both clamps get the derived
\(k_{\mathrm{ang}}\) (N·m/rad). Config-level snapshots that do not yet have a sampled
env use the fixture `primary.length` midpoint. Per-env apply uses that env's
`params.primary.length`. (Fixture JSON may still list the same number for angular
and linear slots; the map overwrites angular.)

The following remain fixed:

- primary and secondary **flexural** moduli (and axial unless explicitly searched);
- support-joint damping ratio;
- non-support fixed-joint penalties;
- geometry, topology, density, apple parameters, and other structure state.

Primary \(E\) is intentionally not a grid or optimizer axis. The support
mounts dominate the relevant primary compliance in the proxy fixture, so
freeing primary \(E\) would misattribute mount compliance to wood.

## 2. Cartesian and fused grid

`example_youngs_modulus_sys_id.py` builds a Cartesian product from:

- `--support-kp-values` in physical units, or mutually exclusive
  `--log10-support-kp`;
- `--log10-e-spur`; and
- `--log10-e-stem`.

`iter_support_kp_youngs_candidates` preserves Cartesian order. The default
multi-structure scheduler flattens compatible work as
`structure × local candidate × physical direction`. Stable
`ReplaySlotKey(structure_idx, local_candidate_idx, direction_idx)` identities
survive parameter application, action routing, scoring, and reporting.
Physical chunks preserve complete candidate/direction blocks under
`--max-envs-per-batch`. Incompatible structures raise
`ReplayFusionIncompatible` (use `--no-multi-structure-batch` to evaluate
structures independently). A fused chunk runtime failure is recorded as a
structure error without an automatic scalar retry.

Rankings remain structure-local. Eligible candidates sort by ascending pooled
Sinkhorn fitness, with local candidate index as the deterministic tie-breaker.
The CLI writes `ranking.json`, per-structure overlays, and optional replay
exports. For simulation datasets, `--include-gt-candidate` can insert the exact
recorded support-\(k_p\)/spur/stem candidate when it is absent.

Controller selection is data-aware:

- simulation collection and the default simulation grid use 6D `vic`;
- `action_layout=vic_pose_v1` / `action_dim=19` selects `vic_pose`;
- `--controller-mode vic|vic_pose` is an explicit grid override.

See H2 for action meaning. A pose-control wrench must never be interpreted as a
twist.

## 3. CMA-ES loop

`example_youngs_modulus_cmaes.py` is the per-structure fit command for
simulation datasets and for real 1×1 `vic_pose` bags;
it does not replace the Cartesian diagnostic. One independent pycma optimizer
owns each selected structure. Active structures advance in synchronized waves:

1. each active optimizer calls `ask()` once;
2. all populations enter one logical fused evaluation;
3. results route back in original sample order;
4. each successful structure calls `tell()` independently; and
5. stopped bounded phenotype means (`es.result.xfavorite`) enter one explicit
   final-mean evaluation wave.

The measured final mean is the fitted estimate. Best sampled points and
covariance are diagnostics only. Stored simulation GT values are used for
post-hoc reporting, never optimizer initialization or fitness.

**Process-isolated evaluation waves (2026-08-25):** by default each fused
evaluation wave (generation, re-ask, or final-mean) runs in a fresh subprocess
via [`cma_wave_evaluation.py`](../apple_pick_gym/batched_envs/cma_wave_evaluation.py).
The worker executes the same `evaluate_youngs_modulus_structures` path as the
CLI (settle → replay collectors → Sinkhorn scoring) and returns a full pickled
`YoungsModulusBatchEvaluation` (scores, replay episodes, applied params).
The parent keeps pycma `ask`/`tell` and `penalize_youngs_modulus_scores`.
`env.close()` does **not** call `wp.clear_kernel_cache()`. Disable with
`--no-isolated-eval-waves` for interactive viewer debug; holdout val evals
remain in-process after fit (same as before). A failed worker is retried up
to `--wave-max-attempts` (default 5) before the wave fails.

**In-process FR3 reuse (`--no-isolated-eval-waves`):** the in-process path sets
`RobotConfig.reuse_replicated_mujoco`. After the first fused weld, later waves
reuse the USD-imported FR3 model and `SolverMuJoCo` (reset to the cached rest
`joint_q`) and only rebuild the VBD plant. Isolated workers do **not** reuse:
each wave is a new process, so FR3 is constructed cold. Default isolation is
unchanged.

### Search defaults

`CMA_SEARCH_PARAMS` is the source of truth:

| Knob | Default |
| ---- | ------- |
| Coordinates | mixed: `log10([support_kp, E_flex_spur, E_flex_stem, E_ax_spur, E_ax_stem, support_roll_kp])` + linear `[ζ_spur, ζ_stem, ζ_support]` in `[0, 1]` (CMA 9D); grid uses first three only |
| Initial mean | sim-sim `[4.0, 9.5, 9.5, 9.5, 9.5, log10(0.75), 0.5, 0.5, 0.5]`; real `vic_pose` `[log10(1000), 8.7, 8.7, 8.7, 8.7, log10(0.75), 0.5, 0.5, 0.5]` approx (see `_REAL_CMA_MEAN_LOG10`) |
| Initial sigma | `0.2`; real/sim `max_sigma_log10=0.5` (pycma `maxstd` + post-`tell` σ clamp) |
| `cma_stds` | `[1]*6 + [0.5]*3` so initial phenotype std ≈ `sigma * cma_stds[i]`: stiffness/roll dims 0.2 decades, ζ dims 0.1 (10% of the `[0, 1]` box, not 20%) |
| Population | `20` |
| Maximum generations | `15` |
| CMA base seed | `56` |
| Bounds | sim-sim lower `[2, 4.0, 4.0, 7.0, 7.0, -1, 0, 0, 0]`, upper `[6, 10.7, 10.7, 10.7, 10.7, 2, 1, 1, 1]`; real `vic_pose` kp \(200\)–\(4\,\mathrm{kN/m}\), \(E\) \(100\,\mathrm{kPa}\)–\(10\,\mathrm{GPa}\), same roll + ζ `[0,1]` box |

The sim-sim support box is \(10^2\)–\(10^6\,\mathrm{N/m}\); real `vic_pose` support \(k_p\) is \(200\)–\(4\,\mathrm{kN/m}\) (init \(1\,\mathrm{kN/m}\)). `"bounds_midpoint"` initialization extends fixture spur/stem midpoints to 9D by duplicating flexural midpoints onto axial slots, using \(\log_{10}(0.75)\) for T-roll, and `0.5` for the three ζ dims when a 9D search box is active. Grid `"bounds_midpoint"` remains 3D.
The ranges fixture remains required for replay `sim_build` settings, but its
narrow material ranges are not the default CMA safety box.

### Invalid samples, stopping, and reports

With at least one finite eligible score, invalid samples receive
`worst_finite + max(1, abs(worst_finite))`. An all-invalid generation is
re-asked up to `DEFAULT_ALL_INVALID_REASKS` (3) times, then told a uniform
`ALL_INVALID_FLAT_PENALTY` (`1e12`) so the structure remains active. The
generation cap and native pycma stop conditions are both honored.

The CLI atomically checkpoints `<output>/cmaes_report.json` before replay and
after each wave or state transition. Structure states are `active`,
`stopped_pending_final_evaluation`, `fitted`, or `failed`. Reports distinguish:

- `completed_generations` and `optimizer_samples_told`;
- logical `replay_candidate_evaluations` and `final_mean_evaluations`;
- physical environment slots and scalar retries;
- final mean, best sample, evaluated-history extrema, and covariance
  diagnostics; and
- structure-local failures versus non-fatal artifact errors.

The CMA integrity gate checks finite, coherent fit evidence and does not impose
a GT-error threshold.

## 4. Scoring handoff to H3

Grid ranking and CMA fitness use the production complete pooled Sinkhorn path
owned by H3:

- bag and `STATE_VECTOR` fields (real GT F/T from `ft_wrist_lpf` when present);
- fixed physical scales and GT-only centering;
- hold/direction handling and completeness;
- exclusion of `action` from the score vector; and
- per-direction diagnostic versus pooled optimizer fitness.

**CMA default hold aggregation (2026-09-02):** `--hold-aggregation mean`
(arithmetic mean of stable hold frames before emitting `[s_i, s_{i+1}-s_i]` rows).
Use `--hold-aggregation none` for quasi-static level bags (every stable hold frame
is a row). The deprecated `--use-median` flag and
`YoungsModulusScoringConfig.use_median=True` both override to median when
`hold_aggregation` is not set explicitly.
Each generation `raw_scores` entry carries `mean_hold_force_err_n`,
`mean_hold_torque_err_nm`, `mean_hold_woody_start_m`, and
`mean_hold_woody_bend_rad` computed from the same mean-hold states as Sinkhorn.
The `score_summary` dict adds `eligible_mean_hold_*` / `best_eligible_mean_hold_*`
across non-disqualified candidates.

**Stiffness defaults (2026-09-02):** `--hold-aggregation mean --include-delta
--delta-weight 1 --categorical-weight 100` with pooled directions (default).
Level-bag pass: `--hold-aggregation none --no-include-delta --delta-weight 0.2`.
Legacy transition bags:
`--hold-aggregation mean --include-delta --categorical-weight 1`. Recorded under
`scoring.hold_aggregation`, `scoring.include_delta`, and `scoring.categorical_weight`
in `cmaes_report.json`.

**Full-trajectory scoring (2026-09-14, opt-in):** `--full-trajectory` scores each
direction's `move_out` ramp segments alongside its `hold` segments, instead of
hold-only. Zero-velocity holds constrain stiffness but cannot constrain damping
(`spur_damping_ratio`, `stem_damping_ratio`, `support_joint_zeta` — see §1); the
ramp is where damping shows up. Each `move_out` segment is paired 1:1, in order,
with the direction's next `hold` segment and inherits that hold's `hold_number`
for `hold_id_onehot` (sim recordings stamp `hold_number=-1` outside the hold
phase — `quasi_static_trajectory.QuasiStaticTrajectory.current_hold_number`).
Every row gets a 2-wide one-hot phase tag (`move_out`/`hold`) so the Sinkhorn
ground cost never conflates a ramp transition with a hold transition. `return`
and `pre_weld` frames stay excluded either way. Requires `--hold-aggregation
none`: reducing a ramp to a single median/mean state discards the very signal
full-trajectory scoring exists to capture (`YoungsModulusScoringConfig` raises
at construction time if this is violated). Recorded under
`scoring.full_trajectory` in `cmaes_report.json`. Default: off (hold-only,
matches prior behavior); pooled and per-direction bags both use a single fixed
physical-scale normalization, so with the default ~40%/60% pull/hold frame
split on real logs the ramp is a substantial and currently unweighted share of
the pooled fitness — there is no separate move_out/hold weight knob yet.

**Force-magnitude fitness term (2026-08-24, opt-in):** eligible CMA fitness can add
`λ · mean_d |log(sim‖F‖ / real‖F‖)|`, where ‖F‖ is
`per_direction_mean_hold_force_norm_n` and each side is floored at
`FORCE_FLOOR_N` (0.2 N). CLI `--force-magnitude-weight` sets λ (default 0;
legacy runs often used 100–300). Recorded under `scoring.force_magnitude_weight`
in `cmaes_report.json`. This is optimizer fitness only; holdout
`force_magnitude_ok` gates remain unchanged (H3).

See `docs/handbook-sysid-scoring.md`. Do not infer the current objective from
historical MMD grid or primary-\(E\) design documents.

**Generation force, torque, TCP, and woody plots.** After
`--persist-generation-replays`, plot real vs sim Fx/Fy/Fz, Tx/Ty/Tz, TCP XYZ,
and woody-start XYZ for `primary_spur` / `spur_stem` (PNG + optional HTML) from
`structure_XXX/generations/gen_YY/best`.
Sim **wrench** traces default to a 5 Hz zero-phase Butterworth (plot-only;
`--sim-lpf-hz 0` disables). TCP and woody positions are not filtered:

```bash
uv run python -m apple_pick_gym.viz.cma_force_plots \
  --run tmp/cma_5d_rotvec_delta_20gen \
  --manifest tmp/real_batched_s09_k_frame/manifest.json
```

Output is `structure_XXX/force_plots/gen_YY/` plus an index README.

## 5. Real-data path and status boundary

The real grid path reuses H4's
`make_real_replay_build_env_fn`, including open-loop FR3 initialization,
recorded control rate, logged gripper transform, post-grasp SE(3), and 19D
`vic_pose` drive. `example_youngs_modulus_sys_id.py` auto-detects
`vic_pose_v1` metadata or accepts `--controller-mode vic_pose`.

Converted real bags have no simulator-oracle recoverable phenotype.
Consequently `gt_candidate` is `None`, every row has `is_gt=false`, and
`--include-gt-candidate` is forced off with a warning. One converted tree is
one structure (1×1 or 1×N); `vic_pose` grid and CMA runs reject
multi-structure selection. Explicit `--controller-mode vic` on a
`vic_pose_v1` packed dataset is refused (twist replay is not valid for
recorded pose-control bags).

This plumbing is shipped, but a successful build/replay is not ranking
acceptance. Folder convert and per-direction weld are H4 (shipped). The
following remain ROADMAP-owned and must not be inferred as complete from this
handbook:

- GPU science-gate **pass** on s09 holdout (Task 9 **ran** 2026-08-17 at
  shipped 15×10: Sinkhorn + TCP passed; **failed** on val torque magnitude); and
- multi-tree merge / dropping the one-structure `vic_pose` guard.

Real 1×1 `vic_pose` CMA (same H4 builder as the grid):

- Real 1×1 `vic_pose` datasets auto-select `make_real_replay_build_env_fn`;
  `--controller-mode` is the explicit opt-in/override.
- Dynamic apple + weld TCP harvest is **default on** (`--dynamic-apple` /
  `--no-dynamic-apple`); isolated eval waves inherit via `CmaReplayContext`.
- Each selected structure's recorded bag must include convert-time
  `ft_wrist_lpf`; CMA refuses otherwise. Sinkhorn scores that column (H3);
  live candidate harvest stays unfiltered `ft_wrist`.
- `gt_candidate` is `None`; `cmaes_report.json` omits `gt_diagnostics`.
- Effective spur/stem floor is \(\log_{10} E = 7\) on real runs only.
- **Known issue (undiagnosed):** local `population_size=6`, `max_generations=4`
  on `s09-d00` exited native **139** while starting generation 3 after two
  completed generations (`eligible_mean` `19.46 → 18.23`).

### Opt-in holdout (5 train / 3 val)

Default CMA (no split flags) still scores **all** usable directions, writes
no `holdout_report.json`, and leaves sim-sim / 1×1 real behavior unchanged.

Holdout mode is opt-in:

| Flag | Semantics |
| ---- | --------- |
| `--direction-split-seed` | `nargs="?"`, `const=17`, `default=None`. Bare flag ⇒ seed 17. Absent ⇒ no holdout. |
| `--direction-indices` + `--val-direction-indices` | Together pin a split (comma-separated ints). Exactly one of the pair is `SystemExit`. |

Holdout always requires **eight** usable disk dirs (seed or explicit). Lists
must be disjoint, non-empty, and a subset of disk. Seeded sample uses stdlib
`random.Random(seed).sample(sorted(dirs), 5)`. Train = that sample sorted;
val = sorted complement. Seed 17 on `{0…7}` yields train `{2,4,5,6,7}`,
val `{0,1,3}`. One-hot `n_directions` stays `collection.num_directions` (8).

Fit: every `ask`/`tell` and the final-mean wave pass `direction_indices=train`
only. After a successful fit, freeze CMA `final_mean` (the same vector the
CLI already evaluates at the end of fit). Holdout eval replays **val** dirs
at shipped `CMA_SEARCH_PARAMS["initial_mean_log10"]` (baseline) and at
`final_mean` (fitted). Those calls must not `tell()`. Failed fit: no
`holdout_report.json`; non-zero exit preserved.

`holdout_report.json` is written atomically next to `cmaes_report.json`.
Required keys:

- `structure_idx` (int, `0`)
- `direction_split_seed` (int; **omitted** when both explicit index flags
  overrode the draw)
- `train_direction_indices` / `val_direction_indices` (sorted int lists)
- `phenotype_log10.baseline` / `phenotype_log10.fitted` (length-3 log10
  vectors)
- `train_fitted`, `val_baseline`, `val_fitted`: each has
  `eligible_mean_sinkhorn`, `mae_force_n`, `mae_torque_nm` (finite floats)
- `verification`: `train_sinkhorn_decreased`, `val_sinkhorn_improved`, and
  per-val-dir `force_magnitude_ok`, `force_trend_ok`,
  `tcp_pose_magnitude_ok`, `tcp_pose_trend_ok` (plus diagnostic ratios /
  Pearson \(r\); see H3)
- `val_overlay_paths`: one HTML per val dir under
  `structure_000/holdout/direction_0NN.html` (real vs **fitted**, not the
  train overlay)

`--overwrite` clears `holdout_report.json` and `structure_000/holdout/`.

**Exit code:** a successful fit still writes the report, then exits **1** if
any gate fails. The CLI prints one line naming the first failed gate (and
direction). Magnitude/trend fail even if Sinkhorn improved. Beating the
shipped initial-mean Sinkhorn on val is required but cheap (baseline
\(E \sim 3\,\mathrm{GPa}\)); the signed pose/force gates are the physics
check. **Do not infer a passing science gate from this handbook** — Task 9
ran 2026-08-17 and **failed** on val torque magnitude (see ROADMAP).

Keep the one-structure `vic_pose` `SystemExit`. Two-tree merge is out of
scope.

See `docs/ROADMAP.md` for the ordered M4.0 work.

## 6. Stability and exclusion

Online monitoring, sticky soft-disable, manifest exclusion, and the default
unstable-frame threshold are documented in
`docs/batched-stability-monitor-design.md`. Candidate replay uses the same
exclusion/disqualification policy as the grid and scoring stack.

Soft-disabled 6D rows become zero twist; soft-disabled 19D rows freeze the last
pose-and-gains command, as defined in H2. A missing direction, unstable
episode, empty bag, or non-finite score disqualifies the affected candidate
without silently changing the expected comparison set.

## 7. Commands

Run from the repository root.

### Simulation collect → grid

```bash
uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 --max-steps 200 \
  --output tmp/support_kp_sysid_dataset --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --viewer null --dataset tmp/support_kp_sysid_dataset \
  --output tmp/support_kp_grid \
  --support-kp-values 1e3,1e4,1e5 \
  --log10-e-spur 8.0,9.5,11.0 \
  --log10-e-stem 8.0,9.5,11.0 \
  --include-gt-candidate --overwrite
```

### Simulation CMA

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --viewer null --dataset tmp/support_kp_sysid_dataset \
  --output tmp/support_kp_cmaes_fit --overwrite
```

### Real convert → grid plumbing smoke

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/new_data/s09/s09-d00.parquet \
  --dataset-out tmp/real_batched_s09_d00 --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_grid \
  --viewer null \
  --support-kp-values 1e3,1e4 \
  --log10-e-spur 9.0 \
  --log10-e-stem 9.0 \
  --no-include-gt-candidate \
  --overwrite
```

This real command proves build/replay plumbing only until ROADMAP's trusted
ranking smoke is accepted.

### Real convert → CMA plumbing smoke

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_cmaes_s09_d00 \
  --viewer null \
  --overwrite
```

This is a plumbing/fit-loop smoke on one converted episode; ranking quality is
still ROADMAP-owned. Shipped `CMA_SEARCH_PARAMS` is `population_size=15`,
`max_generations=10` (~hours on an RTX 4090); that full run has **not** been
executed in verification. For a local smoke, temporarily set
`population_size=4`, `max_generations=3` in
`apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py`, run with a
distinct `--output`, then restore shipped knobs before commit. Verified reduced
run on `s09-d00` (`tmp/real_kp_e_cmaes_s09_d00_retry`): `eligible_mean`
`18.85 → 17.99 → 13.75`. A separate local run with `population_size=6`,
`max_generations=4` crashed with native exit **139** starting generation 3
(`eligible_mean` `19.46 → 18.23` in two completed generations); root cause
undiagnosed. Do not commit `tmp/` artifacts.

### Real folder convert → holdout CMA

Opt-in 5/3 holdout on one converted tree. Requires eight compiled
`s09-dNN.parquet` under `robot_replay/new_data/s09/`. Task 9 ran this recipe
(2026-08-17, pop=15 / gen=10): plumbing passed; science gate **failed** on
val torque magnitude (ROADMAP).

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input-dir robot_replay/new_data/s09 \
  --dataset-out tmp/real_batched_s09 \
  --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09 \
  --output tmp/real_kp_e_cmaes_s09_holdout \
  --direction-split-seed 17 \
  --viewer null \
  --overwrite
```

Bare `--direction-split-seed` is the same as `--direction-split-seed 17`.
Omit the flag to score all dirs with no `holdout_report.json`.

### Gates

```bash
bash scripts/gate_youngs_modulus_sysid.sh
bash scripts/gate_youngs_modulus_cmaes.sh
```

Both gates are expensive multi-seed simulation workflows. The first enforces
the ranking policy; the second validates CMA fit integrity.

## 8. Code map, tests, and verification

| Responsibility | Module / symbol |
| -------------- | --------------- |
| Candidate mapping, support penalties, evaluation, CMA coordinator | `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` |
| Process-isolated CMA wave evaluation | `apple_pick_gym/batched_envs/cma_wave_evaluation.py`; `cma_wave_evaluation_worker.py` |
| Process-isolated CMA snapshot MP4s | `spawn_isolated_cma_snapshot_video`; `cma_snapshot_video_worker.py` (fresh GL process per clip) |
| Stable slot planning, chunking, fused/scalar replay | `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` |
| Cartesian grid, ranking, real-builder opt-in | `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` |
| CMA search defaults, CLI, counters, atomic report | `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` |
| Holdout split, report, val overlays, exit on gate fail | `apple_pick_sim/system_id/holdout_gates.py`; `apple_pick_gym/batched_envs/holdout_evaluation.py` |
| Shared real replay build | `apple_pick_gym/batched_envs/real_batched_replay_build.py` |
| Ranking and CMA gates | `apple_pick_gym/batched_envs/youngs_modulus_gate_report.py`; `youngs_modulus_cmaes_gate_report.py`; `scripts/gate_youngs_modulus_*.sh` |
| Persisted-bag force/torque/TCP/woody time series | `apple_pick_gym/viz/cma_force_plots.py` |

Focused grid/controller-mode checks:

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_sysid_script.py -q
```

Focused CMA checks:

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_cma_wave_evaluation.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  apple_pick_gym/tests/test_holdout_evaluation.py \
  apple_pick_sim/tests/test_holdout_gates.py \
  apple_pick_gym/tests/test_youngs_modulus_cmaes_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_cmaes_script.py \
  apple_pick_gym/tests/test_cma_force_plots.py -q
```

The multi-replay module imports simulation fixtures by the bare name
`conftest`; isolate those imports when running it beside Gym tests:

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  --import-mode=importlib \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py -q
```
