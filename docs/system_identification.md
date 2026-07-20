# System Identification Protocol: Multi-Node Apple-Branch Dynamics

## 1. Objective

Develop a high-fidelity, tunable simulation model of an apple-branch system. The physical system is modeled as a topological network of spatial springs and masses (primary branch, secondary branch, spur, stem), solved via VBD (Variational Body Dynamics). Rod **material** parameters (Young's modulus \(E\), damping ratio \(\zeta\)) and geometry are identified from real-world kinematic and force/torque (F/T) telemetry; VBD stiffness/damping are **derived** at sample time (`docs/material-parameter-sampling.md`).

The immediate calibration path uses **CMA-ES** with complete pooled Sinkhorn
fitness over replayed hold transitions. Before real-data optimization, M3 must
verify observation-only replay in sim-to-sim: treat a differently tuned
simulator as ground truth, reconstruct the tunable simulator from collectable
observations, replay the same recorded actions, and measure the reconstruction
error floor. MMD remains available as a library diagnostic.

Observation-only replay and digital-twin reconstruction requirements live in `docs/digital-twin.md`.

## 2. Excitation Trajectories

Trajectories must excite multiple vibrational modes while the robot maintains grasp on the apple. Excitation stays within the **elastic regime** to avoid premature detachment or plastic deformation.

**Protocol order:** run **2.1 first**; its stiffness map defines safe amplitude bounds for **2.2** and **2.3**.

**Safety:** monitor wrench online and abort a sweep if force exceeds a conservative threshold (e.g. ~80% of a prior stem-break estimate for that cultivar/session). Do not assume “up to 10 cm” is safe without this guard.

**Repeatability:** bookend each session with two identical single-direction quasi-static sweeps (start and end). Drift in temperature, moisture, or prior disturbance shows up as MMD-inflating variance if uncaught.

### 2.1 Quasi-Static Stepped Mapping (isolating stiffness $K$)

**Purpose:** Map the stiffness profile $K(x)$ of the branch chain by suppressing velocity-dependent damping ($B$) and mass-dependent inertia ($M$).

**Execution:**

- Build a **forward-looking hemisphere** of unit directions toward the stem attachment point.
- Sample **~10 directions** with a **Fibonacci / golden-ratio lattice** (uniform solid angle). Avoid naive uniform azimuth/elevation, which over-samples near the pole.
- For each direction: step the end-effector from center in configurable increments (default **5 cm** per step, **10 cm** total) with a **fast move + hold** at each amplitude, then return to center before the next direction. Captures compression, shear, and coupled compressive–shear modes relevant to push-then-shear picking while limiting canopy penetration.
- **Hold** each pose **1–2 s** so transients decay before logging steady-state F/T.

**Implementation (§2.1 shipped):** trajectory generators, Parquet recording, and dataset replay live under `apple_pick_sim/system_id/` and `apple_pick_gym/`. Defaults and test commands: [Implementation notes: §2.1 quasi-static](#implementation-notes-21-quasi-static-stepped-mapping) below and `docs/sysid-trajectory-storage.md`.

**Run in sim** (one direction, 2 cm increments, 10 cm total):

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --n-directions 1 --movement-per-step-m 0.02 --total-movement-m 0.10 \
  --move-speed-mps 0.2
```

On headless Linux the example auto-appends `--viewer null`. Pass `--output <dataset_dir>` to write observation-first Parquet frames and episode metadata. Privileged `initial_states/*.npz` snapshots are default-off; add `--save-snapshot` only when collecting a sim-to-sim baseline for comparison.

**Verify pull-direction geometry** (default 90° hemisphere, matches collection):

```bash
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output pull_directions.png
```

### 2.2 Multi-Axis Log Chirps (modal excitation)

**Purpose:** Identify natural frequencies and bandwidth of each node. Primary branch resonates lower; stem/spur higher.

**Chirp vs discrete frequencies (future dynamic identification):** Use a **true continuous chirp**, not a small set of discrete sine tones.

| | Log chirp | Discrete frequencies |
| --- | --- | --- |
| Resonance capture | Sweeps through $\omega_n = \sqrt{K/M}$ automatically | $B$ is weakly identifiable unless a tone lands near $\omega_n$ |
| Objective landscape | Continuous manifold in $[s_t, \Delta s_t]$; better-conditioned distributional loss | Isolated periodic orbits; parameter degeneracy (many $\theta$ match the same orbits) |
| Cost per optimizer eval | One run per direction | ~3–4× (settling per tone) |

Reserve **discrete steady-state tones** for **held-out validation** after the dynamic optimizer converges on chirp data—not as the primary excitation.

**Execution:**

- Reuse the same **10 hemisphere directions** from §2.1.
- Per direction: **logarithmic chirp** from **~0.1 Hz → ~5+ Hz** (equal time per decade, not per Hz).
- Amplitude scales as **$A \propto 1/f$** (approximately constant velocity amplitude) to limit stem snap and kinematic faults at high frequency.
- For **nonlinearity checks**: repeat 2–3 representative directions at **50% and 100%** of the §2.1 safe amplitude. Large parameter shifts between levels imply a nonlinear stiffness term.

**Post-processing (analysis / debug, not optimizer input):** FRF peaks from wrench signals via FFT or Welch’s method help interpret fitted $K$, $B$, $M$; the solver does not “interpolate diagonal dynamics” automatically.

### 2.3 Pure Torsional Excitation (rotational impedance)

**Purpose:** Torsion at the abscission zone is often the dominant failure mode; rotational impedance differs from translational axes.

**Execution:** Hold the apple at zero Cartesian displacement.

1. **Quasi-static angular sweeps** with holds about stem-local **Z (twist)** and orthogonal **pitch/yaw** → rotational/bending stiffness $K_\theta$.
2. **Rotational log chirps** on the same axes → rotational/bending damping $B_\theta$.

Amplitude bounds from §2.1 apply here as well.

## 3. Data Representation and Feature Engineering

MMD compares **distributions** of transitions; encoding must respect causality without 1:1 time alignment.

**Shipped feature contract (dims, hold/median bags, pooling one-hots, gates):** see `docs/sysid-transition-features.md`.

### 3.1 State Definition

**Protocol intent** (field + future chirp phases) vs **shipped hold-phase bags** (`STATE_VECTOR_FIELDS` in `mmd_features.py`; dims and layout in `docs/sysid-transition-features.md`):

| Symbol | Protocol meaning | Shipped hold bags |
| --- | --- | --- |
| $W_{ee}$ | Interaction wrench (3D force, 3D torque) | `ft_wrist` (6) |
| $v_{ee}$ | End-effector velocity | `tcp_velocity` (6: linear + angular) |
| — | Recorded drive signal | `action` (6) |
| — | TCP / fruit pose | `tcp_pos` (3), `apple_pos` (3) |
| $P_{\text{nodes}}$ | Tracked branch/spring endpoints when available | `woody_part_{start,end}_pos` (\(3N_j\) each) + `woody_bending_angles` (\(N_j\)); total \(D_s=24+7N_j\) |
| \(\phi_{\text{exc}}\) | Trajectory type + chirp frequency \(f(t)\) | Not columns of \(s_t\); `excitation_type` is auxiliary / bag metadata. Instantaneous \(f(t)\) is for §2.2 chirps (not hold bags today) |
| \(\hat{u}\) | Unit excitation direction | Recorded as `excitation_direction`; bags key by `dir_idx` (optional dir one-hot only when pooling) |

**Observability:** Field data may not include all woody endpoints (occlusion, no markers). If only wrench + EE kinematics are reliable, reduce \(s_t\) explicitly—do not assume full internal node coordinates for MMD.

**Replay initialization bundle:** M3.0.3 requires a separate initial-observation bundle before transition features are built. At minimum this includes schema/episode metadata, control rate, recorded TCP actions, TCP pose/twist, bias-corrected F/T wrench, apple pose, woody endpoint observations with junction labels, grasp/weld transform, and robot/fruiting/camera/F/T calibration transforms. This bundle replaces privileged simulator arrays such as `body_q`, `body_qd`, joint buffers, VBD previous-state buffers, and controller target transforms; see `docs/digital-twin.md` for the replacement map.

### 3.2 Transition Feature Vector

Markovian flow (not absolute pose alone):

$$v = [s,\, \Delta s]$$

Shipped hold bags support frame→frame \(\Delta s\) or hold→hold median \(\Delta s\). Batched grid CLI defaults: `--use-median`, `--hold-id-onehot`, and `--pool-directions` **on** (pool appends dir one-hot and merges bags; disable with `--no-*`). Library helpers still default `use_median=False` / no pooling. **No latter-half burn-in** in the feature builders. Full contract: `docs/sysid-transition-features.md`.

### 3.3 Pre-Processing

- **Time sync:** Simulator $\Delta t$ matches real sensor polling rate.
- **Z-score normalization:** Zero mean, unit variance per feature dimension before MMD so Newton-scale wrench does not dominate meter-scale position (GT fit per direction bag, or one pooled bag when `--pool-directions`).
- **Replay fidelity:** Each optimizer rollout is driven by the **recorded EE velocity telemetry** from the source run, not a re-synthesized chirp. Phase/amplitude mismatch otherwise inflates the objective for the wrong reason.

### 3.4 Optimizer data pooling

The accepted V.5.2 objective is one complete pooled Sinkhorn loss per
structure/candidate. Physical-direction identity is retained with a fixed-width
one-hot before pooling; independently normalized per-direction losses remain
diagnostics and do not drive optimizer updates. See
`docs/youngs-modulus-sysid.md`.

## 4. Optimization: CMA-ES

The immediate V.5.2 optimizer is a separate pycma ask/tell loop for each
selected structure:

\[
\theta = \log_{10}([E_\mathrm{primary}, E_\mathrm{spur}, E_\mathrm{stem}]).
\]

Bounds come from the associated ranges fixture and initialization is the
component-wise midpoint in bounded log space, not recorded GT. Geometry,
damping ratio, density, mass, secondary E, and all other non-fitted fields
remain fixed for this slice. Derived VBD knobs are rebuilt from candidate E;
they are not independent optimizer dimensions.

1. **Ask:** obtain one bounded CMA-ES population.
2. **Replay:** evaluate the population with the fused candidate scheduler and
   identical recorded actions.
3. **Score:** use complete pooled hold-phase Sinkhorn fitness.
4. **Tell:** preserve population order; substitute deterministic finite
   penalties only when a generation retains at least one eligible candidate.
5. **Stop:** honor the generation cap or pycma's native stop criteria.
6. **Measure:** explicitly replay the final distribution mean and report it as
   the fitted estimate.
7. **Aggregate:** summarize fitted means and covariance across successful
   structures.

The Cartesian grid remains a separate diagnostic CLI. The optimizer path is
implemented as `example_youngs_modulus_cmaes.py` with notes in
`docs/youngs-modulus-cmaes-implementation.md`. Design:
`docs/superpowers/specs/2026-07-16-youngs-modulus-cmaes-loop-design.md`.
Verification (focused/full tests + CUDA acceptance) **passed** 2026-07-17 —
V.5.2 is Done in `docs/ROADMAP.md`.

## Tests and implementation

| Milestone | Status | Code / docs |
| --- | --- | --- |
| M3.0 §2.1 quasi-static | **Done** (trajectory + gym replay) | `apple_pick_sim/system_id/`, `apple_pick_gym/envs/apple_pick_sysid_env.py`; implementation notes below |
| M3.0.2 recording + privileged-state replay | **Done** | `TrajectoryWriter`, `TrajectoryDataset`, `ApplePickReplayEnv`, `example_gym_replay.py`, `docs/sysid-trajectory-storage.md` |
| M3.0.3 observation-only replay init | **Done** | Observation-only Parquet replay is the default initializer (`--use-snapshot` opts into the privileged `.npz` path instead); spec in `docs/digital-twin.md` |
| M3.1.1 MMD stiffness grid (legacy single-env) | **Done** | `apple_pick_gym/examples/run_system_identification.py --mmd-output <dir>` — prefer V.4.3 batched grid for `batched_sysid_v1` |
| M3.0.4 digital-twin fixture catalog | **Done** | `digital_twin_fixture_catalog.json`, example obs JSON, `test_digital_twin.py`; see `docs/digital-twin.md` |
| V.4.2 batched parallel collection | **Done** | `ApplePickBatchedSysIdEnv`, `batched_sysid_v1` layout; see `docs/batched-sysid-dataset.md` |
| V.4.3 in-process batched grid | **Done** | `example_batched_sysid_mmd_grid.py` + `batched_sysid_mmd_grid.py` (MSE / Sinkhorn Wasserstein + viz); library MMD via `evaluate_batched_mmd_grid`; see `docs/sysid-mmd-grid-replay-alignment.md` |
| V.4.2.1 batched digital-twin fidelity | **Done** | Helpers + CLI `--infer-params` shipped; infer-only fidelity floor optional cleanup — see `docs/ROADMAP.md` |
| M3.0 §2.2–2.3 chirps / torsion | Planned | — |
| V.5.1 loss / feature hardening | **Done** | GT should rank first on healthy samples; bad-sampling misses remain diagnostic and the operational gate uses a strict majority; Wasserstein primary |
| V.5.2 CMA-ES calibration loop | **Done** | Separate pycma entry point over primary/spur/stem log10-E using pooled Sinkhorn fitness; Task 8 verification passed 2026-07-17 — see §4, `docs/youngs-modulus-cmaes-implementation.md`, `docs/youngs-modulus-sysid.md`, and `docs/ROADMAP.md` |
| V.5.2 prerequisites: E-grid, complete scoring, ranking gate, fused replay | **Implemented; fused acceptance pending** | Dataset-driven grid, structure-local reports, strict-majority ranking gate, and multi-structure scheduler are present; clean independent/fused benchmark and low-cap acceptance remain open |

### Batched MMD grid base geometry (2026-07-06)

The batched in-process grid (`apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`) replays recorded actions while sweeping per-segment `bend_stiffness`. Its **base** `FruitingSystemParams` template (rod lengths, directions, material scalars) must match the structure that produced the recorded GT trajectories.

**Previous behavior:** `replay_batched_sysid_structure` and `gt_bend_stiffness_candidate_from_structure` called `infer_base_params_for_structure`, which chord-fits rod length/direction from post-settle pre-weld woody junction anchors (`infer_segment_geometry` in `apple_pick_sim/digital_twin/from_obs.py`). Gravity sag bends rods before that frame, so chord length/angle underestimate rest length and diverge from nominal orientation.

**Current behavior (sim-to-sim default):** both call sites use `true_params_for_structure`, which deserializes the recorded `fruiting_system_params` metadata written at collection time (`apple_pick_gym/batched_envs/batched_sysid_collect.py`). Only `bend_stiffness` is varied per grid candidate via `BendStiffnessCandidate.apply_to`.

**Opt-in digital twin:** CLI `--infer-params` switches build params to `infer_base_params_for_structure` (obs-inferred geometry). Independent of `--use-snapshot` (privileged state restore). Infer-only fidelity floor remains deferred V.4.2.1. Online unstable-env signal during collect/grid: `docs/batched-stability-monitor-design.md`.

**Shipped scoring path (V.5.1 ranking accepted):** `stable` frame mask in `mmd_features.py` (masks samples inside holds; does **not** split hold segments), CLI defaults `--use-median` / `--hold-id-onehot` / `--pool-directions` on `example_batched_sysid_mmd_grid.py`, console median hold MSE via `trajectory_paired_hold_median_mse` (legacy flat bag still in `trajectory_hold_aggregated_mse`), named gates in `scripts/gate_sysid_gt_sinkhorn.sh` (default `gate_pooled_dirs`), candidate `disqualified` flags in grid viz, hold impulse flags in `batched_hold_quasi_static.py`. GT should rank first under healthy excitation/sampling; worse bad-sampling ranks remain diagnostic. Feature contract: `docs/sysid-transition-features.md`.

**Tests:** `test_true_params_for_structure_returns_exact_sampled_params`, `test_gt_bend_stiffness_candidate_from_structure_reads_true_stiffness`, `test_replay_structure_uses_true_params_geometry`.

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_digital_twin_init.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_sysid_replay.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py -q
```

Verify §2.1: see [Implementation notes: §2.1 quasi-static stepped mapping](#implementation-notes-21-quasi-static-stepped-mapping) below (pytest + `example_gym_sysid.py`). Broader schedule: `docs/ROADMAP.md`.

Preferred batched grid smoke (`batched_sysid_v1`):

```bash
uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \
  --viewer null --dataset /tmp/batched_sysid_dataset --replay-only --score-mse \
  --plot-output /tmp/mmd_grid \
  --primary-bend-stiffness-values 1e-4,2e-4 \
  --secondary-bend-stiffness-values 1e-4 \
  --spur-bend-stiffness-values 1e-4 \
  --stem-bend-stiffness-values 1e-4,2e-4
```

### Young's-modulus E-grid replay and ranking (V.5.2)

The shipped diagnostic replays `batched_sysid_v1` actions over structure-local
Cartesian log10-E grids for primary, spur, and stem. It now uses complete
pooled Sinkhorn scoring and defaults to a fused
structure × candidate × physical-direction schedule, while preserving
structure-local rankings, artifacts, and scalar fallback. The repeatable gate
requires a strict majority of GT-rank-one structures for every configured seed.

See `docs/youngs-modulus-sysid.md` for the material conversion, replay identity,
compatibility and chunking rules, report schema, gate policy, code map, and
canonical commands. Shared transition-vector details remain in
`docs/sysid-transition-features.md`.

### Legacy single-env MMD grid

Diagnostic bend-stiffness grid on the **legacy** Parquet layout:

```bash
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null \
  --primary-bend-stiffness-values 10 \
  --secondary-bend-stiffness-values 10 \
  --spur-bend-stiffness-values 10 \
  --stem-bend-stiffness-values 10 \
  --max-candidates 1
```

Run `--list-episodes` first when a dataset contains multiple recordings:

```bash
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null --list-episodes
```

Then expand each axis to search a grid. The command below evaluates
`3^4 = 81` candidates in grid order:

```bash
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null \
  --primary-bend-stiffness-values 10,25,50 \
  --secondary-bend-stiffness-values 10,25,50 \
  --spur-bend-stiffness-values 10,25,50 \
  --stem-bend-stiffness-values 10,25,50 \
  --mmd-output /tmp/apple_pick_mmd_grid
```

For each candidate the script prints mean/max replay errors for TCP force,
TCP torque, TCP position, TCP velocity, apple position, and woody endpoints.
When `--mmd-output` is set, it also computes the stiffness diagnostic objective:
hold-phase biased MMD² over per-direction transition features from
`build_transition_features_by_direction` (**full** hold segments; no latter-half
burn-in). It writes `mmd_results.csv` plus a compact diagnostic plot bundle:
`mmd_ranked_loss.png`, `mmd_direction_heatmap.png`, and
`mmd_stiffness_sensitivity.png`. This remains a diagnostic grid search, not
simulator tuning or CMA-ES.

### Sinkhorn ranking validation (2026-07-08)

Alongside hold MSE on the batched stiffness grid, a **Geomloss Sinkhorn**
objective on the same hold-phase transition bags \(v=[s,\Delta s]\) (+ optional
one-hots) is **shipped** (`apple_pick_sim/system_id/wasserstein.py`,
`wasserstein_ranking.py`; CLI `--score-wasserstein` on
`example_batched_sysid_mmd_grid.py`). Feature/CLI contract (median, hold-id,
pool→dir one-hot): `docs/sysid-transition-features.md`. Named GT-rank gates:
`scripts/gate_sysid_gt_sinkhorn.sh`. This does not replace §4 calibration:
**V.5.1 Done** (GT preference on good samples; Wasserstein primary);
**V.5.2** CMA-ES calibration **Done**
(`docs/youngs-modulus-cmaes-implementation.md`). On the legacy single-env path, the default initializer is observation-only
Parquet replay; use `--use-snapshot` only for privileged sim-to-sim debugging against
`initial_states/*.npz`.

## Implementation notes: §2.1 quasi-static stepped mapping

§2.1 drives the EE through Fibonacci-hemisphere push directions. For each direction the trajectory repeats **fast move → hold** for each increment, then either **return** or a **grasp-pose teleport** between directions:

1. **move_out** — fast linear burst along the direction for `movement_per_step_m / move_speed_mps` seconds (default 5 cm at 0.2 m/s ≈ 0.25 s).
2. **hold** — zero velocity for `hold_duration_s` (default 1.5 s) so transients decay; steady-state `ft_wrist` is logged at each amplitude.
3. Repeat steps 1–2 for `total_movement_m / movement_per_step_m` increments (default 2 × 5 cm → 10 cm total; must be an integer multiple — see `derive_n_steps()`).
4. **return** *(optional)* — one fast reverse over `total_movement_m` back to the grasp center when `skip_return=False`.
5. **teleport** *(default)* — when `skip_return=True`, the trajectory omits return frames; the caller invokes `ApplePickSysIdEnv.restore_grasp_pose()` at each direction boundary to snap robot + cable state back to the post-`reset()` grasp pose.

Quasi-static behavior comes from **hold settling**, not slow crawl speed.

Default `QuasiStaticStepConfig`: `movement_per_step_m=0.05`, `total_movement_m=0.10`, `move_speed_mps=0.2`, `hold_duration_s=1.5`, `control_hz=60`, `skip_return=True`.

`ApplePickSysIdEnv` extends VIC with `Box(6)` EE velocity actions, excitation metadata obs, actual `tcp_pos` from `body_q` (not the VIC target), and optional robot-facing weld placement. Default VIC stiffness is `vic_linear_k=2000` N/m (not the replay-env default). Applied stem feedback defaults to `stem_force_cap_n=100` N and `stem_torque_cap_nm=100` N·m.

**Grasp-pose snapshot/restore:** `reset()` calls `snapshot_grasp_pose()`, which stores robot `body_q`/`joint_q`, cable `body_q`, and VIC `target_tf`. `restore_grasp_pose()` writes those buffers back, re-syncs MuJoCo/`robot_state_1`, aligns VBD `body_q_prev`, zeros lagged `proxy_forces`/`coupling_forces_cache`, and resets `vic_target_twist`. Use this at direction boundaries when `skip_return=True`. Full-transition logging (`[s_t, Δs_t]`) should mark or exclude teleported frames because `tcp_pos` jumps discontinuously.

**Observation-only replay boundary:** the snapshot/restore path above is a simulator convenience, not the real-data initialization path. Observation-only replay rebuilds the post-grasp state from reset observations, calibration transforms, and digital-twin fixture metadata; see `docs/digital-twin.md`.

**Robot-facing weld:** when `fix_to_apple=True` and `robot_facing_weld=True` (default), each `reset()` picks a Fibonacci-hemisphere weld direction toward the fixture robot base; successive resets cycle through `n_weld_hemisphere_samples` (default 10). Override via `reset(options={"weld_direction": (x, y, z)})`. `info["weld_direction"]` reports the unit vector used.

**Episode length:** `ApplePickSysIdEnv` defaults to `max_episode_steps=240`. A full multi-direction run needs `estimate_trajectory_frames(config, n_directions) + margin`. `gym.make(..., max_episode_steps=N)` only sets the `TimeLimit` wrapper — the env still truncates at its constructor default (240) unless you pass `max_episode_steps` into `ApplePickSysIdEnv(...)` directly.

**Wrench guard:** `ApplePickSysIdEnv` caps the applied stem-harvest feedback to the robot at 100 N and 100 N·m by default (`ft_wrist`). It also exposes/logs `raw_ft_wrist`, the uncapped stem-harvest TCP wrench, so diagnostics and objectives can still see solver spikes. `compute_terminated` still inherits the coupled-env stub (`False` always); callers should monitor force limits for abort policy.

### Code map (§2.1)

| Module | Role |
|--------|------|
| `apple_pick_sim/system_id/fibonacci_hemisphere.py` | Golden-ratio polar-cap lattice, rotation to world frame, `stem_perpendicular_robot_pole`, `sample_robot_facing_pull_directions` |
| `apple_pick_sim/system_id/pull_direction_viz.py` | Live env geometry extraction + 3-panel matplotlib figures |
| `apple_pick_gym/examples/visualize_pull_directions.py` | CLI to render pull-direction figures from `ApplePickSysIdEnv` |
| `apple_pick_sim/system_id/quasi_static_trajectory.py` | Phase machine, `derive_n_steps`, `estimate_trajectory_frames`, `iter_frames()` |
| `apple_pick_sim/system_id/excitation_state.py` | `ExcitationContext` dataclass |
| `apple_pick_gym/envs/apple_pick_sysid_env.py` | Gym env (`ApplePickSysId-v0`) |
| `apple_pick_gym/examples/example_gym_sysid.py` | Interactive demo: viewer, per-step logging, mean hold forces |
| `apple_pick_sim/system_id/run_quasi_static.py` | Headless smoke runner (no viewer) |
| `apple_pick_sim/system_id/mmd_features.py` | Hold transition bags: state matrix, median/frame modes, `stable` mask, hold/dir one-hots |
| `apple_pick_sim/system_id/wasserstein.py` | Sinkhorn scoring; internal pooled bag may use `POOLED_DIRECTION_KEY=-1`, but Young's `ranking.json` `per_direction_sinkhorn` exposes physical direction IDs only |
| `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py` | Batched grid CLI (`--use-median`, `--hold-id-onehot`, `--pool-directions`, deprecated `--mse-hold-*`) |
| `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` | Dataset-driven Young's-modulus E-grid replay + Sinkhorn ranking + overlay HTML |
| `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` | Stable slot planning, chunking, and fused multi-structure replay |
| `apple_pick_gym/batched_envs/youngs_modulus_gate_report.py` | Strict per-seed and aggregate Young's gate reports |
| `apple_pick_gym/youngs_modulus_overlay_viz.py` | Faceted Plotly overlay for E-grid replay candidates |
| `scripts/gate_sysid_gt_sinkhorn.sh` | Named Sinkhorn GT-rank gates (`gate_median_hold`, `gate_hold_id`, `gate_pooled_dirs`) |
| `scripts/gate_youngs_modulus_sysid.sh` | Multi-seed collect → exclude → rank → strict-majority gate |

### Fibonacci hemisphere

Pull excitation directions and weld orientations share the same pole geometry but serve different roles:

- **Pull directions** (TCP push during §2.1): sampled via `sample_robot_facing_pull_directions(n, physical_stem, robot_vec)`.
- **Weld directions** (gripper orientation at reset): one direction per reset from the same pole-centered cap (`ApplePickSysIdEnv`, cycles across resets).

**Pole:** `stem_perpendicular_robot_pole(physical_stem, robot_vec)` — unit vector perpendicular to the physical stem (base→tip) and facing the fixture robot base (`robot_vec = robot_base_pos − apple_pos`).

**Cap sampling:** build an area-uniform golden-ratio lattice on the `+Z` polar cap in a local frame, then rotate so `+Z` aligns with the pole:

\[
z_i = \cos\theta_{\max} + (1 - \cos\theta_{\max})\left(1 - \frac{i + \tfrac{1}{2}}{N}\right),\quad
\phi_i = \arccos(z_i),\quad
\theta_i = \frac{2\pi(i + \tfrac{1}{2})}{\varphi}
\]

where \(\varphi = (1+\sqrt{5})/2\) and \(\theta_{\max}\) is `max_polar_angle` (default \(\pi/2\), full hemisphere). Every output satisfies \(\mathbf{d}_i \cdot \hat{p} \ge \cos\theta_{\max}\) for pole \(\hat{p}\).

Optional `min_horizontal_dot` filters to the world-XY half-plane toward the pole (not used by default collection or viz). When the filtered pool is shorter than `n`, indices wrap (possible duplicates).

### Tests (§2.1)

- `apple_pick_sim/tests/test_quasi_static_sysid.py` — polar-cap geometry, pole orthogonality, optional horizontal filter, trajectory phases
- `apple_pick_sim/tests/test_visualize_pull_directions.py` — live env pull-direction sanity, weld/proxy robot-facing checks, PNG smoke
- `apple_pick_gym/tests/test_sysid_env.py` — action/obs contract, `tcp_pos` source, weld direction cycling/override, excitation context round-trip, VIC defaults, no force termination, `restore_grasp_pose`

### How to verify (§2.1)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_sim/tests/test_visualize_pull_directions.py \
  apple_pick_gym/tests/test_sysid_env.py -q
```

Headless smoke (trajectory utilities only, no gym viewer):

```bash
uv run python apple_pick_sim/system_id/run_quasi_static.py --viewer null --n-directions 1
```

Physical return (legacy): pass `--no-skip-return` to `example_gym_sysid.py` or set `skip_return=False` on `QuasiStaticStepConfig`.

Full 10-direction run needs `max_episode_steps` on the env constructor ≥ `estimate_trajectory_frames(QuasiStaticStepConfig(), 10) + 64` (default 240 truncates early).
