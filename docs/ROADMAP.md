# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-06-11 (M3 active: sys-id excitation trajectories) |
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

Later: real-data collection [M4], final pick policy [M5].

---

## Current focus

**Active milestone:** [M3] — Simulation parameter identification (CEM + MMD).

**Goal:** Implement field excitation trajectories, replay them in sim with recorded EE velocity, and build the transition-feature + MMD pipeline for CEM calibration of fruiting-system parameters $\theta$.

**Spec:** `docs/system_identification.md`

**Build on (do not reimplement):**

- [M1] stack: `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`, VIC joint torques (`docs/variable-impedance-teleop.md`, `docs/vic-implementation.md`).
- [M2.1] shipped: `apple_pick_gym/` — `ApplePickCoupledEnv`, real `Dict` observations (woody part poses/forces, apple position, TCP wrench/velocity); reuse θ packing / subprocess patterns when M3.2 lands.

**Next up (ordered):**

1. [ ] **M3.0 — Sys-id excitation trajectories:** Implement §2.1–2.3 from `docs/system_identification.md` — Fibonacci hemisphere directions, quasi-static stepped mapping, translational **log chirps** ($A \propto 1/f$), torsional quasi-static + chirp; trajectory type + instantaneous $f(t)$ in logged state; wrench force-limit guard; §2.1 amplitude bounds feed 2.2/2.3. Deliver: trajectory generators + sim replay smoke (recorded $v_{ee}$ drives VBD).
2. [ ] **M3.1 — Transition dataset + MMD:** Per-direction transition features $[s_t, \Delta s_t]$, z-score normalization, anisotropic RBF MMD objective.
3. [ ] **M3.2 — CEM loop:** Sample $\theta$, subprocess rollouts, elite update; validate on held-out discrete-frequency trajectories.

---

## Milestones (summary)

### [P0] Done

Procedural fruiting system, fixtures, force readouts, `example_fruiting_system.py`.

### [M1] Done

MuJoCo + VBD coupling, FR3 import, proxy wrench exchange, GPU hot path, VIC teleop, settle-then-weld.

Key docs: `docs/mujoco-vbd-coupling-architecture.md`, `docs/WRENCH_READOUT.md`, `docs/variable-impedance-teleop.md`, `docs/vic-implementation.md`, `docs/vic-joint-torques-implementation.md`, `docs/gpu-coupling-optimization.md`.

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
| M3.0 | **Next** | Excitation trajectories (quasi-static, log chirp, torsional) + sim replay smoke |
| M3.1 | Planned | MMD feature pipeline per direction |
| M3.2 | Planned | CEM calibration + held-out validation |

---

## Backlog

- **[M2] remaining:** M2.0 (interface ADR), M2.2a (`ApplePickFID-v0`), M2.2c (SKRL smoke), M2.3 (π_exp training) — resume after M3 or in parallel if maintainer directs.
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
| `docs/` | Vision, roadmap, architecture, implementation notes (`system_identification.md` for M3) |

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
```

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict.
- Policy or product decisions (scope, user-visible behavior).
- Network credentials, paid APIs, or destructive operations.

**When unsupervised is expected:**

- Complete the **next unchecked slice** in Current focus using TDD and project rules.
- Fix small blockers uncovered by that slice (tests, imports, typos).
- Do **not** start a new milestone or backlog item without maintainer direction.
