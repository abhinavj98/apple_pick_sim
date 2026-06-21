# System Identification Protocol: Multi-Node Apple-Branch Dynamics

## 1. Objective

Develop a high-fidelity, tunable simulation model of an apple-branch system. The physical system is modeled as a topological network of spatial springs and masses (primary branch, secondary branch, spur, stem), solved via VBD (Variational Body Dynamics). Parameters (stiffness, damping, mass) are identified from real-world kinematic and force/torque (F/T) telemetry.

Optimization uses the **Cross-Entropy Method (CEM)** against field data, with **Maximum Mean Discrepancy (MMD)** as the objective so we avoid strict time-pairing requirements of L2 regression. Before optimizer selection is finalized, M3 must verify observation-only replay in sim-to-sim: treat a differently tuned simulator as ground truth, reconstruct the tunable simulator from collectable observations, replay the same recorded actions, and measure the reconstruction error floor.

Observation-only replay and digital-twin reconstruction requirements live in `docs/observation-replay-digital-twin.md`.

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

**Implementation (§2.1 shipped):** trajectory generators, Parquet recording, and dataset replay live under `apple_pick_sim/system_id/` and `apple_pick_gym/`. Details, defaults, and test commands: `docs/system-id-quasi-static-implementation.md` and `docs/sysid-trajectory-storage.md`.

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

**Replay initialization bundle:** M3.0.3 requires a separate initial-observation bundle before transition features are built. At minimum this includes schema/episode metadata, control rate, recorded TCP actions, TCP pose/twist, bias-corrected F/T wrench, apple pose, woody endpoint observations with junction labels, grasp/weld transform, and robot/fruiting/camera/F/T calibration transforms. This bundle replaces privileged simulator arrays such as `body_q`, `body_qd`, joint buffers, VBD previous-state buffers, and controller target transforms; see `docs/observation-replay-digital-twin.md` for the replacement map.

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

Tune simulation parameters $\theta$ (masses, spring constants $K$, damping $B$).

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
| M3.0 §2.1 quasi-static | **Done** (trajectory + gym replay) | `apple_pick_sim/system_id/`, `apple_pick_gym/envs/apple_pick_sysid_env.py`, `docs/system-id-quasi-static-implementation.md` |
| M3.0.2 recording + privileged-state replay | **Done** | `TrajectoryWriter`, `TrajectoryDataset`, `ApplePickReplayEnv`, `example_gym_replay.py`, `docs/sysid-trajectory-storage.md` |
| M3.0.3 observation-only replay init | **Next** | Reconstruct a plausible Newton initial state/equilibrium from recorded observations, without saved simulator state; spec in `docs/observation-replay-digital-twin.md` |
| M3.0.4 digital-twin fixture reconstruction | Planned | Named fixture catalog and sim-to-sim transfer validation before real-world collection |
| M3.0 §2.2–2.3 chirps / torsion | Planned | — |
| M3.1 MMD features | Planned | — |
| M3.2 CEM loop | Planned | — |

Verify §2.1: `docs/system-id-quasi-static-implementation.md` (pytest + `example_gym_sysid.py`). Broader M3 schedule: `docs/ROADMAP.md`.