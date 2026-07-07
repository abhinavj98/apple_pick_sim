# System Identification Protocol: Multi-Node Apple-Branch Dynamics

## 1. Objective

Develop a high-fidelity, tunable simulation model of an apple-branch system. The physical system is modeled as a topological network of spatial springs and masses (primary branch, secondary branch, spur, stem), solved via VBD (Variational Body Dynamics). Rod **material** parameters (Young's modulus \(E\), damping ratio \(\zeta\)) and geometry are identified from real-world kinematic and force/torque (F/T) telemetry; VBD stiffness/damping are **derived** at sample time (`docs/material-parameter-sampling.md`).

Optimization uses the **Cross-Entropy Method (CEM)** against field data, with **Maximum Mean Discrepancy (MMD)** as the objective so we avoid strict time-pairing requirements of L2 regression. Before optimizer selection is finalized, M3 must verify observation-only replay in sim-to-sim: treat a differently tuned simulator as ground truth, reconstruct the tunable simulator from collectable observations, replay the same recorded actions, and measure the reconstruction error floor.

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

**Chirp vs discrete frequencies (CEM + MMD):** Use a **true continuous chirp**, not a small set of discrete sine tones.

| | Log chirp | Discrete frequencies |
| --- | --- | --- |
| Resonance capture | Sweeps through $\omega_n = \sqrt{K/M}$ automatically | $B$ is weakly identifiable unless a tone lands near $\omega_n$ |
| MMD landscape | Continuous manifold in $[s_t, \Delta s_t]$; better-conditioned CEM loss | Isolated periodic orbits; parameter degeneracy (many $\theta$ match the same orbits) |
| Cost per CEM eval | One run per direction | ~3–4× (settling per tone) |

Reserve **discrete steady-state tones** for **held-out validation** after CEM converges on chirp data—not as the primary excitation.

**Execution:**

- Reuse the same **10 hemisphere directions** from §2.1.
- Per direction: **logarithmic chirp** from **~0.1 Hz → ~5+ Hz** (equal time per decade, not per Hz).
- Amplitude scales as **$A \propto 1/f$** (approximately constant velocity amplitude) to limit stem snap and kinematic faults at high frequency.
- For **nonlinearity checks**: repeat 2–3 representative directions at **50% and 100%** of the §2.1 safe amplitude. Large parameter shifts between levels imply a nonlinear stiffness term.

**Post-processing (analysis / debug, not CEM input):** FRF peaks from wrench signals via FFT or Welch’s method help interpret fitted $K$, $B$, $M$; the solver does not “interpolate diagonal dynamics” automatically.

### 2.3 Pure Torsional Excitation (rotational impedance)

**Purpose:** Torsion at the abscission zone is often the dominant failure mode; rotational impedance differs from translational axes.

**Execution:** Hold the apple at zero Cartesian displacement.

1. **Quasi-static angular sweeps** with holds about stem-local **Z (twist)** and orthogonal **pitch/yaw** → rotational/bending stiffness $K_\theta$.
2. **Rotational log chirps** on the same axes → rotational/bending damping $B_\theta$.

Amplitude bounds from §2.1 apply here as well.

## 3. Data Representation and Feature Engineering

MMD compares **distributions** of transitions; encoding must respect causality without 1:1 time alignment.

### 3.1 State Definition

Observable state $s_t$ at time $t$:

| Symbol | Description |
| --- | --- |
| $P_{\text{nodes}}$ | 3D positions of tracked branch joints/spring endpoints (when available) |
| $v_{ee}$ | End-effector Cartesian velocity (3D) |
| $W_{ee}$ | Measured interaction wrench (3D force, 3D torque) |
| $\phi_{\text{exc}}$ | Excitation context: trajectory type (quasi-static / translational chirp / torsional chirp) and **instantaneous frequency** $f(t)$ for chirps (continuous feature, not one-hot over discrete bins) |
| $\hat{u}$ | Unit excitation direction (hemisphere sample or rotation axis) |

**Observability:** Field data may not include all $P_{\text{nodes}}$ (occlusion, no markers). If only wrench + EE kinematics are reliable, reduce $s_t$ explicitly—do not assume full internal node coordinates in $P$ for MMD.

**Replay initialization bundle:** M3.0.3 requires a separate initial-observation bundle before transition features are built. At minimum this includes schema/episode metadata, control rate, recorded TCP actions, TCP pose/twist, bias-corrected F/T wrench, apple pose, woody endpoint observations with junction labels, grasp/weld transform, and robot/fruiting/camera/F/T calibration transforms. This bundle replaces privileged simulator arrays such as `body_q`, `body_qd`, joint buffers, VBD previous-state buffers, and controller target transforms; see `docs/digital-twin.md` for the replacement map.

### 3.2 Transition Feature Vector

Markovian flow (not absolute pose alone):

$$v_t = [s_t,\, \Delta s_t], \quad \Delta s_t = s_{t+1} - s_t$$

### 3.3 Pre-Processing

- **Time sync:** Simulator $\Delta t$ matches real sensor polling rate.
- **Z-score normalization:** Zero mean, unit variance per feature dimension before MMD so Newton-scale wrench does not dominate meter-scale position.
- **Replay fidelity:** Each CEM rollout is driven by the **recorded EE velocity telemetry** from the field run, not a re-synthesized chirp. Phase/amplitude mismatch otherwise inflates MMD for the wrong reason.

### 3.4 CEM data pooling

Run CEM **per excitation direction** first (separate $P$, $Q$ per $\hat{u}$). After convergence, compare $\theta$ across directions: direction-dependent parameters (likely $K$) vs shared parameters (likely $M$). Avoid pooling all directions into one MMD pool early—that smears anisotropic stiffness.

## 4. Optimization: Cross-Entropy Method (CEM)

Tune simulation parameters $\theta$ (geometry where free, rod \(E\), \(\zeta\), apple mass scalars). Derived VBD knobs are not independent CEM dimensions unless legacy fixtures are in use.

**Reframing of $\theta$:** rather than fitting raw, geometry-entangled
$K$/$B$ per segment, fit Young's modulus $E$, damping ratio $\zeta$, and density
$\rho$ — geometry-invariant quantities that transfer correctly across domain-randomized
`radius`/`length`/`num_segments`, and which the §2.1/§2.2 excitation phases already
separate cleanly (quasi-static → $E$; chirp resonance peak location vs. width/decay →
$\rho$ vs. $\zeta$). Derivation, formulas, and a numerical `ω_n·dt` stability guard for
the resulting domain randomization: `docs/material-parameter-sampling.md` ("Derivation" section).

1. **Initialize:** $\mathcal{N}(\mu_0, \Sigma_0)$ with broad $\Sigma_0$.
2. **Sample:** $N \approx 50$–$100$ candidate $\theta_i$.
3. **Simulate:** VBD with each $\theta_i$, initialized from the observation-derived digital twin, driven by **identical** recorded $v_{ee}(t)$; extract transition samples.
4. **Evaluate:**

$$L(\theta) = \text{MMD}^2(P, Q)$$

Use an **anisotropic RBF kernel**; per-dimension bandwidth $\sigma$ via median heuristic.

5. **Update:** Elite top $\rho\%$ lowest MMD; refit $\mu$, $\Sigma$ on elites; add noise floor $\epsilon \mathbf{I}$.
6. **Iterate** until MMD plateaus.
7. **Validate** on held-out discrete-frequency or alternate-amplitude trajectories (§2.2).

## Tests and implementation

| Milestone | Status | Code / docs |
| --- | --- | --- |
| M3.0 §2.1 quasi-static | **Done** (trajectory + gym replay) | `apple_pick_sim/system_id/`, `apple_pick_gym/envs/apple_pick_sysid_env.py`; implementation notes below |
| M3.0.2 recording + privileged-state replay | **Done** | `TrajectoryWriter`, `TrajectoryDataset`, `ApplePickReplayEnv`, `example_gym_replay.py`, `docs/sysid-trajectory-storage.md` |
| M3.0.3 observation-only replay init | **Done** | Observation-only Parquet replay is the default initializer (`--use-snapshot` opts into the privileged `.npz` path instead); spec in `docs/digital-twin.md` |
| M3.1.1 MMD stiffness grid | **Done** | `apple_pick_gym/examples/run_system_identification.py --mmd-output <dir>` sweeps `primary` / `secondary` / `spur` / `stem` `bend_stiffness` values, replays recorded actions, and ranks candidates by hold-phase biased MMD². |
| M3.0.4 digital-twin fixture catalog | **Done** | `digital_twin_fixture_catalog.json`, example obs JSON, `test_digital_twin.py`; see `docs/digital-twin.md` |
| V.4.2 batched parallel collection | **Done** | `ApplePickBatchedSysIdEnv`, `batched_sysid_v1` layout; see `docs/batched-sysid-dataset.md` |
| V.4.2.1 batched digital-twin replay | **Next** | Frame-0 obs + `params_fingerprint` init on `batched_sysid_v1`; extend replay fidelity capstone |
| M3.0 §2.2–2.3 chirps / torsion | Planned | — |
| M3.1 MMD features | Planned | — |
| M3.2 CEM loop | Planned | — |

Verify §2.1: see [Implementation notes: §2.1 quasi-static stepped mapping](#implementation-notes-21-quasi-static-stepped-mapping) below (pytest + `example_gym_sysid.py`). Broader M3 schedule: `docs/ROADMAP.md`.

Diagnostic bend-stiffness grid smoke:

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
hold-phase biased MMD² over per-direction transition features, with the first
half of each hold segment discarded before feature construction. It writes
`mmd_results.csv` plus a compact diagnostic plot bundle:
`mmd_ranked_loss.png`, `mmd_direction_heatmap.png`, and
`mmd_stiffness_sensitivity.png`. This remains a diagnostic grid search, not
simulator tuning or CEM.
The default initializer is observation-only Parquet replay. Use
`--use-snapshot` only for privileged sim-to-sim debugging against
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
