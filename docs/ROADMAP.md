# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-06-25 (M3 active: MMD grid-search diagnostic; [V] V.1 done, V.2 next) |
| **Owner**        | Abhinav |
| **Vision**       | See `docs/VISION.md` |

---

## How this roadmap is structured

| Section                   | Purpose |
| ------------------------- | ------- |
| **Current focus**         | Single source of truth for what to do *now*. |
| **Milestones**            | Phased outcomes from vision to implementation. |
| **Backlog**               | Lower-priority ideas — not active work. |
| **Agent execution notes** | How to run tests, where code lives, when to stop and ask. |

---

## Sequencing

1. **Done — [P0]:** Variational fruiting geometry, JSON fixtures, fixed-joint force readouts (`measure_fruiting_forces`), standalone viewer (`example_fruiting_system.py`).
2. **Done — [M1]:** Two-`Model` `SolverMuJoCo` + `SolverVBD` staggered coupling, FR3 + gripper proxy, VIC joint-torque teleop (default in `example_coupled_fruiting.py`), settle-then-weld, explicit apple load. Architecture: `docs/mujoco-vbd-coupling-architecture.md`.
3. **Done (partial) — [M2]:** `ApplePickCoupled-v0` with real observations shipped (M2.1). Remaining RL/FID slices deferred to backlog.
4. **Now — [M3]:** Simulation parameter identification — excitation trajectories, field replay, CEM + MMD calibration (`docs/system_identification.md`).
5. **In parallel — [V]:** Batched coupled vectorization via `replicate(N)` — same topology per batch, per-env numeric θ; feeds M3.2 CEM and M2 RL (`docs/vectorized-coupled-fruiting.md`).

Later: real-data collection [M4], final pick policy [M5].

---

## Current focus

**Active milestone:** [M3] — Simulation parameter identification (MMD diagnostics now; CEM later).

**Goal:** Use the shipped observation-only replay path to build the first objective diagnostic before tuning. Collect "ground-truth" trajectories from a known simulator parameter set, replay the same recorded EE actions over a grid of alternate fruiting-system parameters, compute MMD loss between GT and candidate transition distributions, and plot the loss landscape. This is a diagnostic grid search only: do **not** update simulator parameters or run CEM yet.

**Specs:** `docs/system_identification.md`, `docs/observation-replay-digital-twin.md`

**Parallel track — [V] batched vectorization** (branch `apple-pick-sim-vecorization`, spec `docs/vectorized-coupled-fruiting.md`):

- **Batch contract:** fixed `num_segments` and `omit` set per batch; only numeric `FruitingSystemParams` vary (stiffness, lengths, directions, etc.).
- **V.1 (done):** `replicate(N)` cable + robot, batched settle→weld init, `BatchedEnvLayout`, multi-TCP wrench apply, `BatchedTemplateIK` per-env scatter teleop (example: homogeneous keyboard), co-located physics + viewer grid, `example_batched_coupled_fruiting.py` + `test_vectorized_coupled_fruiting.py`. Spec: [independent env semantics — V.1 limitations](vectorized-coupled-fruiting.md#v1-shipped-vs-v2-independent-envs).
- **V.2 (next):** Fully **independent envs** within a batch — per-env DR (numeric θ / seeds), per-env settle→weld→**IK bootstrap** (no `broadcast_joint_q_from_world0` on FR3), per-env actions at runtime; then K/B runtime scatter, recorded-action replay, `gather_transitions()` for M3.2 CEM.
- **V.3:** Per-env geometry DR on reset; batched gym `(N, act_dim)` → IK scatter; batched `apple_pick_gym` adapter → M2 RL.

**Build on (do not reimplement):**

- [M1] stack: `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`, VIC joint torques (`docs/variable-impedance-teleop.md`).
- [M2.1] shipped: `apple_pick_gym/` — `ApplePickCoupledEnv`, real `Dict` observations (woody part poses/forces, apple position, TCP wrench/velocity); observation contract `docs/gym-observation-contract.md` (`info["obs_schema"] == "v3"`); reuse θ packing / subprocess patterns when M3.2 lands.

**Next up (ordered):**

1. [x] **M3.0.1 — Quasi-static §2.1 trajectory + gym replay (shipped):** Fibonacci forward-hemisphere directions, stepped push–hold–return phase machine, `ApplePickSysId-v0`, `example_gym_sysid.py` smoke. Spec: `docs/system-id-quasi-static-implementation.md`.
2. [x] **M3.0.2 — Recording + privileged-state replay (shipped):** `example_gym_sysid.py --output` writes observation-first Parquet frames and metadata; `--save-snapshot` also writes opt-in initial-state snapshots for privileged baseline comparisons. `ApplePickReplay-v0` restores saved Newton state when present and applies recorded EE velocity actions open-loop. Docs: `docs/sysid-trajectory-storage.md`.
3. [x] **M3.0.3 — Observation-only replay initialization (shipped):** Reconstructs a plausible Newton initial state/equilibrium from recorded observations and calibration metadata only. The default replay path avoids privileged saved simulator arrays (`body_q`, `body_qd`, `joint_q`, solver previous-state buffers, controller target transforms); `--use-snapshot` remains opt-in for privileged sim-to-sim debugging.
4. [ ] **M3.1.1 — MMD grid-search diagnostic (next):** Extend `apple_pick_gym/examples/run_system_identification.py` from replay-error summaries into a GT-vs-candidate diagnostic. Collect GT trajectories from a known parameter set, replay the same trajectories over a grid of alternate `primary` / `secondary` / `spur` / `stem` parameters, build per-direction transition features $[s_t, \Delta s_t]$, z-score them using GT/candidate pooled statistics, compute MMD loss, rank grid candidates, and emit a plot/table of the loss landscape. This slice must not update simulator parameters or run CEM.
5. [ ] **M3.0.4 — Digital-twin geometry reconstruction + fixtures:** Define the observable geometry bundle (topology/junction labels, tracked woody endpoints, apple pose, stem/apple frame, robot/camera/F/T calibration transforms, grasp/weld transform, base poses), rebuild a named fixture from those observations, and verify excitation directions relative to stem attachment and fixture `fruiting_base_pos` / `robot_base_pos` (forward hemisphere toward apple, robot-facing weld framing, no sign/frame mix-ups). Deliver a small **named fixture catalog** under `apple_pick_sim/fixtures/` (e.g. straight-rod test baseline + field-twin candidate) with documented base poses, range bounds, and per-fixture sys-id smoke (`example_gym_sysid.py` + pytest). Use `apple_pick_gym/examples/visualize_pull_directions.py` for live geometry checks.
6. [ ] **M3.0.5 — Remaining excitation trajectories (§2.2–2.3):** Translational **log chirps** ($A \propto 1/f$), torsional quasi-static + chirp; trajectory type + instantaneous $f(t)$ in logged state; wrench force-limit guard; §2.1 amplitude bounds feed 2.2/2.3. Deliver: trajectory generators + sim replay smoke (recorded $v_{ee}$ drives VBD).
7. [ ] **M3.1.2 — General MMD feature pipeline:** Harden the MMD objective for broader datasets: configurable observable state fields, per-direction pooling, anisotropic RBF bandwidths, missing-signal handling, and reusable tests outside the diagnostic CLI.
8. [ ] **M3.2 — CEM loop:** Sample $\theta$, subprocess rollouts, elite update; validate on held-out discrete-frequency trajectories.

---

## Milestones (summary)

### [P0] Done

Procedural fruiting system, fixtures, force readouts, `example_fruiting_system.py`.

### [M1] Done

MuJoCo + VBD coupling, FR3 import, proxy wrench exchange, GPU hot path, VIC teleop, settle-then-weld.

Key docs: `docs/mujoco-vbd-coupling-architecture.md`, `docs/WRENCH_READOUT.md`, `docs/variable-impedance-teleop.md`, `docs/gpu-coupling-optimization.md`.

### [M2] Done (partial)

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| M2.1 | Done | `ApplePickCoupled-v0`, real observations, parity tests |
| M2.0 | Deferred | Interface ADR (θ, y, FD protocol) — see backlog |
| M2.2a | Deferred | `ApplePickFID-v0` — see backlog |
| M2.2c | Deferred | SKRL subprocess smoke — see backlog |
| M2.3 | Deferred | π_exp training smoke — see backlog |

### [M3] Active

Sim parameter identification from field trajectories: CEM + MMD over transition distributions (`docs/system_identification.md`). Reuses M2 θ packing and subprocess rollout patterns where applicable; **no** Newton autodiff through the coupled stack.

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| M3.0.1 | Done | §2.1 quasi-static trajectory + `ApplePickSysId-v0` gym replay |
| M3.0.2 | Done | Parquet recording + `ApplePickReplay-v0` privileged-state action replay |
| M3.0.3 | Done | Observation-only replay initialization; no privileged simulator state by default |
| M3.1.1 | **Next** | GT trajectory collection, grid replay, MMD loss ranking + plot via `run_system_identification.py` |
| M3.0.4 | Planned | Digital-twin geometry reconstruction + fixture catalog; sim-to-sim transfer validation |
| M3.0.5 | Planned | §2.2–2.3 log chirp + torsional trajectories + sim replay smoke |
| M3.1.2 | Planned | General MMD feature pipeline per direction |
| M3.2 | Planned | CEM calibration + held-out validation |

### [V] Batched vectorization (parallel)

Homogeneous multi-world coupled rollouts: `ModelBuilder.replicate()` on cable + robot models, fixed topology per batch, per-env θ. Spec: `docs/vectorized-coupled-fruiting.md`.

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| V.1 | **Done** | `replicate(N)`, batched settle→weld, `BatchedTemplateIK` scatter, homogeneous example teleop, layout + tests, co-located physics / viewer spacing doc |
| V.2 | **Next** | Independent envs: per-env seeds/θ DR, per-env IK weld bootstrap, per-env actions; K/B scatter; recorded-action replay; `gather_transitions` for MMD/CEM |
| V.2.1 | Planned (within V.2) | Per-env IK bootstrap after settle→weld (replace joint broadcast); tests for all-world TCP at proxy |
| V.2.2 | Planned (within V.2) | Per-env `sample_params` / stiffness scatter at build or reset |
| V.2.3 | Planned (within V.2) | Example + API for per-env actions (`velocity_for_world`, action buffer); drop placeholder broadcast on independent path |
| V.3 | Planned | Per-env geometry DR on reset; gym `(N, act_dim)` → scatter; batched `apple_pick_gym` adapter |

**Consumers:** M3.2 (batched CEM rollouts), M2.3 / M2.2c (parallel RL envs).

---

## Backlog

- **[M2] remaining:** M2.0 (interface ADR), M2.2a (`ApplePickFID-v0`), M2.2c (SKRL smoke), M2.3 (π_exp training) — resume after M3 or in parallel if maintainer directs; batched env backend depends on **[V].3**.
- **[V] remaining:** V.2 (independent envs + CEM gather) and V.3 after V.1; batched VIC deferred.
- Additional manipulators or crops — explicit scope change only.
- Triangle mesh import/export — P0 stays capsule primitives.
- Real-data pipeline [M4], final pick policy [M5] — after M3 contracts exist.

---

## Agent execution notes

**Repository layout:**

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/` | Simulation code (`fruiting_system/`, `coupled_fruiting/`, `examples/`, tests) |
| `apple_pick_gym/` | Gymnasium adapter (`ApplePickCoupled-v0`); depends on `apple_pick_sim`, not vice versa |
| `newton/` | Upstream physics submodule (vendored) |
| `docs/` | Vision, roadmap, architecture, implementation notes (`system_identification.md` and `observation-replay-digital-twin.md` for M3, `vectorized-coupled-fruiting.md` for [V], `gym-observation-contract.md` for gym obs v3) |

**How to validate changes:**

```bash
# Install / sync (repo root; path-depends on newton/)
uv sync --extra gym --extra vic --extra dev

# Fast test gate (excludes @pytest.mark.slow)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"

# Gym env tests
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/ -q

# Coupled example smoke (headless)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 60

# Coupling verification CLI
uv run python apple_pick_sim/diagnostics/verify_coupling.py --num-substeps 600 --max-force 5 --max-torque 1

# M3.0 §2.1 quasi-static sys-ID (pytest)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_sim/tests/test_visualize_pull_directions.py \
  apple_pick_gym/tests/test_sysid_env.py -q

# M3.0 §2.1 pull-direction geometry viz (PNG; matches collection defaults)
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output pull_directions.png

# M3.0 §2.1 quasi-static demo (one direction, 2 cm steps)
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --n-directions 1 --movement-per-step-m 0.02 --total-movement-m 0.10 \
  --move-speed-mps 0.2

# M3.0.4 digital-twin fixture catalog check
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q

# M3.0.4 sim-to-sim collection smoke (observation-only by default; no .npz)
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 \
  --output /tmp/apple_pick_sysid_no_snapshot

# M3 raw dataset dashboard (local browser; use --help as a non-interactive smoke)
uv run python apple_pick_gym/examples/dashboard_sysid_dataset.py --help

# M3.0.4 optional privileged baseline for comparison
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 --save-snapshot \
  --output /tmp/apple_pick_sysid_with_snapshot

# [V] Batched coupled fruiting (V.1 done; V.2 next — see docs/vectorized-coupled-fruiting.md)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_vectorized_coupled_fruiting.py -q
uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \
  --viewer null --num-frames 500 --num-envs 4 --fix-to-apple --controller direct --seed 42
uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \
  --viewer null --num-frames 120 --robot placeholder --num-envs 2 --fix-to-apple
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
# M3.1.1 MMD grid diagnostic inputs: collect GT trajectories
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 \
  --output /tmp/apple_pick_sysid_gt

# M3.1.1 MMD grid diagnostic: replay same trajectories over a small grid
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/apple_pick_sysid_gt --viewer null \
  --primary-bend-stiffness-values 10,25 \
  --secondary-bend-stiffness-values 10,25 \
  --spur-bend-stiffness-values 10,25 \
  --stem-bend-stiffness-values 10,25
```

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict.
- Policy or product decisions (scope, user-visible behavior).
- Network credentials, paid APIs, or destructive operations.

**When unsupervised is expected:**

- Complete the **next unchecked slice** in Current focus using TDD and project rules.
- Fix small blockers uncovered by that slice (tests, imports, typos).
- Do **not** start a new milestone or backlog item without maintainer direction.
