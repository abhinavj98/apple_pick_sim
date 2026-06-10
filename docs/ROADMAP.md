# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-06-09 (post-cleanup: mega/FID prototype removed; M2 active) |
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
3. **Now — [M2]:** Gymnasium RL infrastructure — `ApplePickCoupled-v0` with real observations; SKRL subprocess vectorization; FID env + Fisher-information exploration (subprocess workers, not a mega `Model`).
4. **Next — [M3]:** Simulation parameter identification / calibration from rollouts (finite-difference path, no Newton autodiff through coupled stack).

Later: real-data collection [M4], final pick policy [M5].

---

## Current focus

**Active milestone:** [M2] — RL infrastructure and Fisher-information exploration.

**Goal:** Ship a production-quality **1×1 coupled Gym env** (`ApplePickCoupled-v0`) with real sensor observations, then add **SKRL** parallel rollouts and an **FID env** for finite-difference probes over subprocess workers.

**Build on (do not reimplement):**

- [M1] stack: `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`, VIC joint torques (`docs/variable-impedance-teleop.md`, `docs/vic-implementation.md`).
- [M2.1] shipped: `apple_pick_gym/` — `ApplePickCoupledEnv`, `Discrete(13)` actions, real `Dict` observations (woody part poses/forces, apple position, TCP wrench/velocity).

**Next up (ordered):**

1. [ ] **M2.0 — Interface ADR:** `pack_theta` / `unpack_theta`, observation/action/`info` contracts, SKRL runner config.
2. [ ] **M2.2a — `ApplePickFID-v0`:** FID env + subprocess FD modes (`fd_ghost`, `fd_mega_same_u` semantics via workers, not mega `ModelBuilder`).
3. [ ] **M2.2c — SKRL integration:** minimal trainer + smoke (W subprocess envs).
4. [ ] **M2.3 — Train π_exp:** SKRL + `ApplePickCoupled-v0`; FIM reward via FID env on schedule.

---

## Milestones (summary)

### [P0] Done

Procedural fruiting system, fixtures, force readouts, `example_fruiting_system.py`.

### [M1] Done

MuJoCo + VBD coupling, FR3 import, proxy wrench exchange, GPU hot path, VIC teleop, settle-then-weld.

Key docs: `docs/mujoco-vbd-coupling-architecture.md`, `docs/WRENCH_READOUT.md`, `docs/variable-impedance-teleop.md`, `docs/vic-implementation.md`, `docs/vic-joint-torques-implementation.md`, `docs/gpu-coupling-optimization.md`.

### [M2] Active

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| M2.1 | Done | `ApplePickCoupled-v0`, real observations, parity tests |
| M2.0 | Next | Interface ADR (θ, y, FD protocol) |
| M2.2a | Planned | `ApplePickFID-v0` |
| M2.2c | Planned | SKRL subprocess smoke |
| M2.3 | Planned | π_exp training smoke |

### [M3] Next

Sim parameter identification from trajectories (FD / black-box, same path as M2).

---

## Backlog

- Additional manipulators or crops — explicit scope change only.
- Triangle mesh import/export — P0 stays capsule primitives.
- Real-data pipeline [M4], final pick policy [M5] — after M2–M3 contracts exist.

---

## Agent execution notes

**Repository layout:**

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/` | Simulation code (`fruiting_system/`, `coupled_fruiting/`, `examples/`, tests) |
| `apple_pick_gym/` | Gymnasium adapter (`ApplePickCoupled-v0`); depends on `apple_pick_sim`, not vice versa |
| `newton/` | Upstream physics submodule (vendored) |
| `docs/` | Vision, roadmap, architecture, implementation notes |

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
