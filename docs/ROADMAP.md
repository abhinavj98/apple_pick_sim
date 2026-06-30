# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-06-30 ([V] V.2.1 done; V.2.1.1 Newton bump next; [S] sim-sim transfer after batched gym) |
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
2. **Done — [M1]:** Two-`Model` `SolverMuJoCo` + `SolverVBD` staggered coupling, FR3 + gripper proxy, VIC joint-torque teleop (default in `example_coupled_fruiting.py`), settle-then-weld, explicit apple load. Architecture: `docs/mujoco-vbd-coupling-architecture.md` (§2.5: zero-payload gravity-comp sim-to-real contract for RL).
3. **Done (partial) — [M2]:** `ApplePickCoupled-v0` with real observations shipped (M2.1). Remaining RL/FID slices deferred to backlog.
4. **Done (infra) — [M3]:** Sys-ID excitation trajectories, Parquet recording, observation-only replay, subprocess MMD grid diagnostic (`docs/system_identification.md`). CEM and batched sim-sim calibration move to **[S]** after **[V].3.4**.
5. **Now — [V]:** Batched coupled vectorization via `replicate(N)` — Newton parity, fixture hardening, independent envs, batched viz/state APIs, then batched gym + parallel collection (`docs/vectorized-coupled-fruiting.md`).
6. **Next — [S]:** Sim-sim transfer — MMD objective on batched rollouts, CEM over θ, held-out validation (exclusive focus once [V].3.4 lands).

Later: real-data collection [M4], final pick policy [M5].

---

## Current focus

**Active milestone:** [V] — Batched coupled vectorization (Newton parity + fixtures, then batched gym).

**Goal:** Finish the batched coupled stack on latest Newton with real-world-like fixtures, expose vectorized robot/woody state query APIs and debug viz (`--tcp-force-arrow`, `--mark-endpoints`), wrap a batched Gymnasium env, and parallelize sys-ID trajectory collection. **[S] sim-sim transfer** (MMD + CEM) starts only after that foundation is in place.

**Specs:** `docs/vectorized-coupled-fruiting.md`, `docs/system_identification.md`, `docs/observation-replay-digital-twin.md`

**Track — [V] batched vectorization** (spec `docs/vectorized-coupled-fruiting.md`):

- **Batch contract:** fixed `num_segments` and `omit` set per batch; only numeric `FruitingSystemParams` vary (stiffness, lengths, directions, etc.).
- **V.1 (done):** `replicate(N)` cable + robot, batched settle→weld init, `BatchedEnvLayout`, multi-TCP wrench apply, `BatchedTemplateIK` per-env scatter teleop (example: homogeneous keyboard), co-located physics + viewer grid, `example_batched_coupled_fruiting.py` + `test_vectorized_coupled_fruiting.py`. Spec: [independent env semantics — V.1 limitations](vectorized-coupled-fruiting.md#v1-shipped-vs-v2-independent-envs).
- **V.2:** Independent envs within a batch — per-env IK bootstrap (V.2.1), Newton bump (V.2.1.1), fixture hardening (V.2.1.2), build-time θ DR (V.2.2), runtime actions (V.2.3), recorded replay + `gather_transitions()` (V.2.4).
- **V.3:** Per-env geometry DR on reset (V.3.1); batched `(N, act_dim)` scatter (V.3.2); vectorized viz + state-query APIs (V.3.3); batched gym + parallel sys-ID collection (V.3.4). Train under **zero-payload gravity-comp** arm model + DR apples (`docs/vectorized-coupled-fruiting.md` § Sim-to-real and RL training contract).

**Next track — [S] sim-sim transfer:** MMD on batched candidate rollouts, CEM θ update loop, held-out validation — replaces the former M3.2 scope once [V].3.4 ships.

**Build on (do not reimplement):**

- [M1] stack: `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`, VIC joint torques (`docs/variable-impedance-teleop.md`).
- [M2.1] shipped: `apple_pick_gym/` — `ApplePickCoupledEnv`, real `Dict` observations (woody part poses/forces, apple position, TCP wrench/velocity); observation contract `docs/gym-observation-contract.md` (`info["obs_schema"] == "v3"`); reuse θ packing / subprocess patterns when M3.2 lands.

**Next up (ordered):**

**[V] batched vectorization**

1. [x] **V.2.1 — Per-env IK bootstrap (shipped):** Heterogeneous path: each world's TCP at its own settled proxy after settle→weld; no `broadcast_joint_q_from_world0` on FR3. `example_batched_heterogeneous_coupled_fruiting.py` + `test_heterogeneous_coupled_fruiting.py`.
2. [ ] **V.2.1.1 — Newton submodule bump (next):** Update `newton/` to latest upstream; fix parity regressions (coupling, VIC, batched paths); full fast + coupled test gate green. Branch: `feature/newton-parity`.
3. [ ] **V.2.1.2 — Fixture stability + real-world likeness:** Refresh named fixtures under `apple_pick_sim/fixtures/` for settle stability, plausible geometry/compliance, and field-twin alignment; per-fixture smoke (`example_gym_sysid.py`, heterogeneous batched example, pytest). Overlaps M3.0.4 digital-twin catalog goals — deliver stability-first fixture set here.
4. [x] **V.2.2 — Build-time per-env θ DR (shipped):** `add_world` heterogeneous cable build; `sample_heterogeneous_params_list`; stiffness baked at `finalize()`.
5. [ ] **V.2.3 — Per-env runtime actions:** Example + API for per-env actions (`velocity_for_world`, action buffer); placeholder broadcast only for homogeneous smoke.
6. [ ] **V.2.4 — Recorded-action replay + `gather_transitions()`:** Per-world transition extraction for batched MMD/CEM (replaces subprocess grid for [S]).
7. [ ] **V.3.1 — Per-env geometry DR on reset:** Runtime kinematics scatter per world (lengths, directions) without rebuilding batch topology.
8. [ ] **V.3.2 — Batched `(N, act_dim)` → IK scatter:** Policy-scale action tensor path for parallel rollouts.
9. [ ] **V.3.3 — Vectorized viz + state-query APIs:** Ship `--tcp-force-arrow` and `--mark-endpoints` on all batched coupled examples; expose public APIs for querying robot TCP pose/wrench, joint state, and woody endpoint poses via vectorized (GPU) paths — no host `.numpy()` loops in the hot readout path.
10. [ ] **V.3.4 — Batched gym + parallel sys-ID collection:** `apple_pick_gym` env over batched coupled stack; extend `example_gym_sysid.py` / trajectory writer for `num_envs > 1` parallel excitation collection.

**[S] sim-sim transfer** (starts after V.3.4)

11. [ ] **S.1 — Batched MMD objective:** GT vs candidate transition distributions from `gather_transitions()`; per-direction features, z-score, rank — generalize `run_system_identification.py` MMD path to batched rollouts.
12. [ ] **S.2 — CEM loop:** Sample $\theta$, batched elite rollouts, refit $\mu$/$\Sigma$; no Newton autodiff through coupled stack.
13. [ ] **S.3 — Held-out sim-sim validation:** Alternate trajectories / amplitudes; report MMD and pose/force drift metrics.

**[M3] remaining infra** (can run in parallel with [V] where independent)

14. [x] **M3.0.1 — Quasi-static §2.1 trajectory + gym replay (shipped).**
15. [x] **M3.0.2 — Recording + privileged-state replay (shipped).**
16. [x] **M3.0.3 — Observation-only replay initialization (shipped).**
17. [x] **M3.1.1 — Subprocess MMD grid-search diagnostic (shipped):** `run_system_identification.py --mmd-output`; superseded for scale by [S].1 batched path.
18. [ ] **M3.0.4 — Digital-twin geometry reconstruction:** Observable geometry bundle → named fixture rebuild; largely absorbed by V.2.1.2 unless field-twin-specific tooling remains.
19. [ ] **M3.0.5 — Remaining excitation trajectories (§2.2–2.3):** Log chirps, torsional quasi-static + chirp; wrench force-limit guard.
20. [ ] **M3.1.2 — General MMD feature pipeline:** Configurable observable fields, anisotropic bandwidths, missing-signal handling — feeds [S].1.

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

### [M3] Infra (mostly done; CEM → [S])

Sys-ID recording, replay, and subprocess MMD diagnostic (`docs/system_identification.md`). Batched calibration moves to **[S]** after [V].3.4.

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| M3.0.1 | Done | §2.1 quasi-static trajectory + `ApplePickSysId-v0` gym replay |
| M3.0.2 | Done | Parquet recording + `ApplePickReplay-v0` privileged-state action replay |
| M3.0.3 | Done | Observation-only replay initialization; no privileged simulator state by default |
| M3.1.1 | Done | Subprocess MMD grid via `run_system_identification.py --mmd-output` |
| M3.0.4 | Planned | Digital-twin geometry reconstruction (overlaps V.2.1.2 fixtures) |
| M3.0.5 | Planned | §2.2–2.3 log chirp + torsional trajectories + sim replay smoke |
| M3.1.2 | Planned | General MMD feature pipeline — feeds [S].1 |
| M3.2 | **Moved → [S].2** | CEM calibration |

### [V] Batched vectorization (active)

Homogeneous multi-world coupled rollouts: `ModelBuilder.replicate()` on cable + robot models, fixed topology per batch, per-env θ. Spec: `docs/vectorized-coupled-fruiting.md`.

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| V.1 | **Done** | `replicate(N)`, batched settle→weld, `BatchedTemplateIK` scatter, homogeneous example teleop, layout + tests |
| V.2.1 | **Done** | Per-env IK bootstrap after settle→weld (heterogeneous path) |
| V.2.1.1 | **Next** | `newton/` bump to latest upstream; parity fixes; full test gate |
| V.2.1.2 | Planned | Fixture catalog refresh — stability + real-world likeness |
| V.2.2 | **Done** | Build-time per-env `sample_params` / stiffness via `add_world` |
| V.2.3 | Planned | Per-env runtime actions (`velocity_for_world`, action buffer) |
| V.2.4 | Planned | Recorded-action replay; `gather_transitions()` per world |
| V.3.1 | Planned | Per-env geometry DR on reset |
| V.3.2 | Planned | Batched `(N, act_dim)` → IK scatter |
| V.3.3 | Planned | `--tcp-force-arrow`, `--mark-endpoints`; vectorized robot/woody state query APIs |
| V.3.4 | Planned | Batched `apple_pick_gym` env; parallel sys-ID data collection |

**Consumers:** [S] sim-sim transfer, M2.3 / M2.2c (parallel RL envs).

### [S] Sim-sim transfer (next track)

Exclusive focus after [V].3.4: MMD + CEM over batched rollouts — no field data required.

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| S.1 | Planned | Batched MMD objective from `gather_transitions()` |
| S.2 | Planned | CEM θ loop with batched elite rollouts |
| S.3 | Planned | Held-out trajectory validation; sim-sim drift metrics |

---

## Backlog

- **[M2] remaining:** M2.0 (interface ADR), M2.2a (`ApplePickFID-v0`), M2.2c (SKRL smoke), M2.3 (π_exp training) — resume after [S] or [V].3.4 if maintainer directs; batched env backend depends on **[V].3.4**.
- **[V] remaining:** V.2.3–V.2.4, V.3.1–V.3.4 after V.2.1.1/V.2.1.2; batched VIC deferred.
- **[S] remaining:** Full sim-sim CEM track — see Current focus items 11–13.
- Additional manipulators or crops — explicit scope change only.
- Triangle mesh import/export — P0 stays capsule primitives.
- Real-data pipeline [M4], final pick policy [M5] — after [S] contracts exist.

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

# [V] Batched coupled fruiting (V.2.1.1 Newton bump next — see docs/ROADMAP.md)
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

# V.3.3 viz smoke (when shipped on all batched examples)
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \
  --viewer null --num-frames 60 --num-envs 2 --tcp-force-arrow --mark-endpoints

# M3.1.1 / [S] subprocess MMD grid diagnostic inputs: collect GT trajectories
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
