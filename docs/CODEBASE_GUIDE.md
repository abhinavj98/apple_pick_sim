# Codebase guide

**Purpose:** a map of this repository — what the packages do, where to look for a given piece of behavior, and which document answers which question. Read this after `docs/VISION.md` / `docs/ROADMAP.md` and before searching the codebase blind.

This guide describes **structure**, not **status**. For "what's done / what's next," always defer to `docs/ROADMAP.md` — do not infer status from a doc's prose tense alone.

## Document status

| Field | Value |
| ----- | ----- |
| **Last reviewed** | 2026-07-03 (V.3.2 close-out — see `docs/ROADMAP.md` for slice numbering conventions used below) |
| **Owner** | Abhinav |

## How to read this repository

1. **`docs/VISION.md`** — why this project exists, scope, non-goals, success criteria. Read once, revisit rarely.
2. **`docs/ROADMAP.md`** — the single source of truth for status: what's shipped, what's active, what's next, validation commands. Every other doc in this repo should defer to it for sequencing.
3. **This file** — architecture map + doc index.
4. **The specific topic doc** (see [Document index](#document-index) below) for how a subsystem works.
5. **Code** — once a doc points you at a module, read the code; docs describe intent and contracts, not line-by-line behavior.

If a doc's status claim and the actual code/tests disagree, trust the code and tests, then fix the doc (or flag it) rather than propagating the stale claim.

## Architecture overview

```text
┌─────────────────────────────────────────────────────────────────┐
│ apple_pick_gym/            Gymnasium adapter (depends on         │
│  envs/, examples/, tests/  apple_pick_sim; not vice versa)        │
└───────────────────────────────┬───────────────────────────────────┘
                                 │ builds / steps
┌───────────────────────────────▼───────────────────────────────────┐
│ apple_pick_sim/             Project-local simulation code          │
│                                                                     │
│  fruiting_system/    Scene generation: params, geometry, VBD-only  │
│                       build (P0). No MuJoCo, no substep loop.      │
│  coupled_fruiting/    Two-Model orchestration: MuJoCo (arm) +      │
│                       VBD (plant) staggered coupling loop (M1),    │
│                       batched/heterogeneous variants ([V] track),  │
│                       VIC joint-torque control.                    │
│  robot/fr3_robot/     FR3 build, controllers, batched IK.          │
│  system_id/           Excitation trajectories, Parquet storage,    │
│                       MMD features (M3).                           │
│  digital_twin/        Geometry reconstruction from observations    │
│                       (M3.0.4 — partially blocked, see below).     │
│  diagnostics/         Standalone verification/benchmark scripts.   │
│  examples/            Runnable demos (one per major capability).   │
│  fixtures/            Range-sampling JSON + (planned) digital-twin │
│                       fixture catalog.                              │
│  tests/               pytest suite.                                │
└───────────────────────────────┬───────────────────────────────────┘
                                 │ vendored physics engine
┌───────────────────────────────▼───────────────────────────────────┐
│ newton/  (git submodule)    Upstream Newton — treat as vendored;   │
│                              match its APIs, avoid drive-by edits.  │
└─────────────────────────────────────────────────────────────────┘
```

**Two-solver split (core architectural fact, M1):** `SolverMuJoCo` cannot represent `JointType.CABLE`, so the fruiting tree (plant) and the FR3 arm live on **two separate `newton.Model` instances**, coupled through proxy bodies and a one-substep-lag wrench exchange. This is the single most important thing to understand before touching simulation code — see `docs/mujoco-vbd-coupling-architecture.md`.

## Directory map

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/fruiting_system/` | `params.py` (sampling, `RodParams`/`FruitingSystemParams`), `build.py` (ModelBuilder geometry, collision filters, VBD solver setup), `scene.py`/`coupled.py` (P0 scene + M1 cable-only scene) |
| `apple_pick_sim/coupled_fruiting/` | `scene.py` (`CoupledFruitingScene.coupled_substep` — the authoritative loop), `builders.py`/`batched_build.py` (homogeneous `replicate()` + heterogeneous `add_world` builds), `proxy_coupling.py` (Warp mirror/harvest kernels, incl. batched stem harvest), `settle_then_weld.py` (two-build init), `vic_joint_torques*.py` (VIC control), `batched_layout.py` (`BatchedEnvLayout`) |
| `apple_pick_sim/robot/fr3_robot/` | FR3 USD import, controllers (direct-joint, EE velocity, impedance), `batched_template_ik.py` |
| `apple_pick_sim/system_id/` | Fibonacci-hemisphere excitation, `quasi_static_trajectory.py`, `trajectory_store.py` (Parquet), `mmd*.py` |
| `apple_pick_sim/digital_twin/` | `obs_io.py`, `from_obs.py` — rebuild scene geometry from observation JSON |
| `apple_pick_sim/diagnostics/` | `verify_coupling.py`, `benchmark_coupling.py`, `sweep_zero_vic_stability.py` — standalone checks, not pytest |
| `apple_pick_sim/examples/` | One runnable script per capability; `example_batched_heterogeneous_coupled_sim.py` is the canonical batched heterogeneous example |
| `apple_pick_sim/fixtures/` | `fruiting_system_ranges_*.json` range files. **Missing:** `digital_twin_fixture_catalog.json` and its example obs JSON — see `docs/digital-twin.md` |
| `apple_pick_gym/envs/` | `apple_pick_base_env.py` → `apple_pick_coupled_env.py` (kinematic) → `apple_pick_vic_env.py` (dynamic, joint-torque VIC) → `apple_pick_sysid_env.py`, `apple_pick_replay_env.py` |
| `newton/` | Upstream Newton submodule — vendored, match its patterns rather than inventing APIs |
| `docs/` | This documentation set (below). `docs/superpowers/specs/` holds dated point-in-time design notes for already-shipped features (MMD grid diagnostic, sys-ID dashboard) — historical, not living docs |

## Document index

Organized by question, not by filename — each doc listed once, under its primary topic.

### "What's the status / what's next?"

- **`docs/ROADMAP.md`** — the only place that should be trusted for slice status, sequencing, and validation commands.
- `docs/VISION.md` — intent, scope, non-goals, success criteria (rarely changes).

### "How does the core plant + arm coupling work?"

- `docs/mujoco-vbd-coupling-architecture.md` — the two-`Model` split, ownership table, staggered substep protocol, sim-to-real gravity contract (M1). Read this first for any coupling work.
- `docs/variable-impedance-teleop.md` — post-grasp VIC: total TCP wrench, joint-torque control, which gym envs are kinematic vs. dynamic.
- `docs/WRENCH_READOUT.md` — fixed-joint wrench semantics, sign conventions, subtree cuts.
- `docs/explicit-apple-load-tcp-harvest.md` — explicit apple-weight term in stem harvest.
- `docs/gpu-coupling-optimization.md` — single-env GPU/CPU split and benchmarks (does **not** cover the batched hot path — see next section).

### "How does batched / heterogeneous / vectorized sim work?"

- `docs/vectorized-coupled-fruiting.md` — **single source of truth for the batched build→settle→weld→teleop flow**, homogeneous vs. heterogeneous batches, co-located physics vs. viewer spacing. Does not carry its own status table — see ROADMAP for sequencing.
- `docs/heterogeneous-batched-vectorization-audit.md` — narrower audit of which parts of the heterogeneous example's hot path are/aren't GPU-vectorized (re-verified 2026-07-02: both original P0 gaps are now fixed).

### "How do I sample / randomize plant material and geometry?"

- `docs/material-parameter-sampling.md` — the shipped `(E, ζ)` → derived VBD-stiffness/damping sampling contract, **plus a "Derivation" appendix** explaining why raw independent sampling is unstable and the physics behind the fix.
- `docs/real-world-proxy.md` — bench-proxy geometry, placement, stiffness tiers, and a still-open topology mismatch between the nominal and variance proxy fixtures (see its "Topology caveat").
- `docs/damping-tuning.md` — practical damping tuning notes.

### "How does system identification / sys-ID work?"

- `docs/system_identification.md` — the full M3 protocol (excitation trajectories, MMD/CEM plan) **plus an implementation-notes appendix** for the shipped §2.1 quasi-static stepped mapping (trajectory phases, Fibonacci hemisphere, code map, tests).
- `docs/sysid-trajectory-storage.md` — Parquet dataset schema, collection/replay commands, dataset dashboard.
- `docs/digital-twin.md` — observation-only replay initialization (shipped) **and** digital-twin geometry reconstruction from observations (code shipped; named fixture catalog data files are **missing** — 2 failing tests, see its "Known gap" section).

### "How does the Gym adapter work?"

- `docs/gym-observation-contract.md` — observation schema versioning (v1→v3), shared keys, env-specific keys.
- `docs/variable-impedance-teleop.md` — which envs are kinematic (`ApplePickCoupledEnv`) vs. dynamic/VIC (`ApplePickVicEnv` and its subclasses `ApplePickSysIdEnv`, `ApplePickReplayEnv`).

## Known gaps (do not assume these are done without checking code/tests)

| Gap | Detail | Where documented |
| --- | ------ | ----------------- |
| Digital-twin fixture catalog | `apple_pick_sim/fixtures/digital_twin_fixture_catalog.json` and its example obs JSON are not committed; 2 tests in `test_digital_twin.py` currently fail | `docs/digital-twin.md` |
| `real_world_proxy.json` topology | Nominal fixture uses `linear_chain`; its variance counterpart defaults to `t_junction`. The two fixtures for the same physical proxy build different topologies | `docs/real-world-proxy.md` |

## Conventions worth knowing before editing docs or code

- **Slice IDs** (`M1`, `[V].3.1`, etc.) are owned by `docs/ROADMAP.md`. Don't invent new ones in topic docs; link to ROADMAP instead of duplicating a phase table.
- **`newton/`** is vendored — match existing patterns in `newton/newton/examples/` and `newton/newton/_src/` rather than inventing new Warp/Newton APIs.
- **TDD** — tests before implementation, per `.cursor/rules/test-driven-development.mdc`.
- **GPU hot paths** — Warp kernels for anything that runs every substep or over many bodies; CPU/NumPy is fine for one-off setup, frame-rate teleop, and debug. See `.cursor/rules/gpu-warp-parallelism.mdc`.
- **Worktrees** — substantial production changes should happen in a dedicated git worktree, not directly on `main`. See `.cursor/rules/worktree-feature-dev.mdc`.
