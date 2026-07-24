# Codebase guide

**Purpose:** a map of this repository — what the packages do, where to look for a given piece of behavior, and which document answers which question. Read this after `docs/VISION.md` / `docs/ROADMAP.md` and before searching the codebase blind.

This guide describes **structure**, not **status**. For "what's done / what's next," always defer to `docs/ROADMAP.md` — do not infer status from a doc's prose tense alone.

## Document status

| Field | Value |
| ----- | ----- |
| **Last reviewed** | 2026-07-22 (README CMA-ES sim-to-sim how-to; V.5.2 Done — see `docs/ROADMAP.md`) |
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
│  envs/, batched_envs/,     apple_pick_sim; not vice versa)        │
│  batched_examples/, tests/                                       │
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
│                       settle→weld, VIC joint-torque control.       │
│  robot/fr3_robot/     FR3 build, controllers, batched IK.          │
│  system_id/           Excitation, Parquet, MMD/Wasserstein,        │
│                       batched digital-twin init (M3 / V.4).        │
│  digital_twin/        Geometry reconstruction from observations    │
│                       (M3.0.4 catalog shipped).                    │
│  diagnostics/         Standalone verification/benchmark scripts.   │
│  examples/            Runnable demos (one per major capability).   │
│  fixtures/            Range-sampling JSON + digital-twin fixture   │
│                       catalog (`digital_twin_fixture_catalog.json`). │
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
| `apple_pick_sim/fruiting_system/` | `params.py` (sampling, `RodParams`/`FruitingSystemParams`, optional `parse_sim_build` / `sim_build` on ranges JSON), `build.py` (ModelBuilder geometry, collision filters, VBD solver setup), `scene.py`/`coupled.py` (P0 scene + M1 cable-only scene) |
| `apple_pick_sim/coupled_fruiting/` | `scene.py` (`CoupledFruitingScene.coupled_substep` — the authoritative loop), `builders.py`, `batched_heterogeneous_*` (config-driven batched API), `proxy_coupling.py`, `settle_then_weld.py`, `settle_seed_device.py`, `settle_ke_decay.py`, `settle_quasi_static.py`, `settled_checkpoint.py`, `vic_joint_torques*.py`, `batched_layout.py` |
| `apple_pick_sim/robot/fr3_robot/` | FR3 USD import, controllers (direct-joint, EE velocity, impedance), `batched_template_ik.py` |
| `apple_pick_sim/system_id/` | Fibonacci-hemisphere excitation, `quasi_static_trajectory.py`, `trajectory_store.py` (legacy Parquet), `batched_trajectory_store.py` (`batched_sysid_v1`), `batched_digital_twin_init.py` (including per-environment episode sources), `parquet_init.py`, `mmd*.py` / `mmd_features.py`, complete/pooled `wasserstein.py`, `batched_hold_quasi_static.py` |
| `apple_pick_sim/digital_twin/` | `obs_io.py`, `from_obs.py` — rebuild scene geometry from observation JSON |
| `apple_pick_sim/diagnostics/` | `verify_coupling.py`, `benchmark_coupling.py`, `sweep_zero_vic_stability.py`, `log_settle_ke_decay.py`, `sweep_settle_weld_stability.py` — standalone checks, not pytest |
| `apple_pick_sim/examples/` | One runnable script per capability; `example_batched_heterogeneous_coupled_sim.py` is the canonical batched heterogeneous example |
| `apple_pick_sim/fixtures/` | `fruiting_system_ranges_*.json` (geometry/material DR; variance proxy may include optional top-level `sim_build` for VIC + joint overrides), `digital_twin_fixture_catalog.json`, `digital_twin_obs_straight_rod_initial.json` |
| `apple_pick_gym/envs/` | Legacy single-world: `apple_pick_base_env.py` → `apple_pick_coupled_env.py` → `apple_pick_vic_env.py` → `apple_pick_sysid_env.py`, `apple_pick_replay_env.py` |
| `apple_pick_gym/batched_envs/` | Batched GPU gym (V.3.3+): batched envs/collection/grid, `batched_sysid_cmaes.py` (Young's candidates, scoring, CMA-ES orchestration), `batched_sysid_multi_replay.py` (stable fused scheduling), stability/soft-disable/exclusion, `sysid_gate_report.py`, `youngs_modulus_gate_report.py` (ranking), `youngs_modulus_cmaes_gate_report.py` (CMA integrity) |
| `apple_pick_gym/batched_examples/` | Parallel collect/grid examples plus `example_youngs_modulus_sys_id.py` (dataset-driven fused E-grid) and `example_youngs_modulus_cmaes.py` (separate CMA-ES fit) |
| `apple_pick_gym/grid_viz_*.py` | Plotly / table / report helpers for batched stiffness-grid ranking (incl. paired-hold woody MSE) |
| `scripts/` | Staged sys-ID helpers: `collect_and_rank_sysid_gt.sh`, `gate_sysid_gt_sinkhorn.sh`, `gate_youngs_modulus_sysid.sh` (multi-seed ranking gate), `gate_youngs_modulus_cmaes.sh` (multi-seed CMA integrity gate) |
| `newton/` | Upstream Newton submodule — vendored, match its patterns rather than inventing APIs |
| `docs/` | This documentation set (below). `docs/specs/` holds dated point-in-time design notes (historical once stamped Implemented) |

## Document index

Organized by question, not by filename — each doc listed once, under its primary topic.

### "What's the status / what's next?"

- **`docs/ROADMAP.md`** — the only place that should be trusted for slice status, sequencing, and validation commands.
- `docs/VISION.md` — intent, scope, non-goals, success criteria (rarely changes).

### "How does the core plant + arm coupling work?"

- **`docs/coupled-sim-api.md`** — **canonical public API** after pre-gym cleanup: runtime exports, builders, examples, FR3 requirements.
- `docs/mujoco-vbd-coupling-architecture.md` — the two-`Model` split, ownership table, staggered substep protocol, sim-to-real gravity contract (M1). Read this first for any coupling work.
- `docs/variable-impedance-teleop.md` — post-grasp VIC: total TCP wrench, joint-torque control, which gym envs are kinematic vs. dynamic.
- `docs/WRENCH_READOUT.md` — fixed-joint wrench semantics, sign conventions, subtree cuts.
- `docs/explicit-apple-load-tcp-harvest.md` — explicit apple-weight term in stem harvest.
- `docs/mujoco-apple-payload.md` — welded MuJoCo TCP inertia-only apple payload (mass / \(I\) / COM; Model A `g=0`).
- `docs/gpu-coupling-optimization.md` — single-env GPU/CPU split and benchmarks (does **not** cover the batched hot path — see next section).

### "How does batched / heterogeneous / vectorized sim work?"

- `docs/vectorized-coupled-fruiting.md` — **single source of truth for the batched build→settle→weld→teleop flow**, homogeneous vs. heterogeneous batches, co-located physics vs. viewer spacing. Defaults: controller **vic**, settle cache **off**. Does not carry its own status table — see ROADMAP for sequencing.
- `docs/heterogeneous-batched-vectorization-audit.md` — narrower audit of which parts of the heterogeneous example's hot path are/aren't GPU-vectorized (re-verified 2026-07-02: both original P0 gaps are now fixed).

### "How do I sample / randomize plant material and geometry?"

- `docs/material-parameter-sampling.md` — the shipped `(E, ζ)` → derived VBD-stiffness/damping sampling contract, **plus a "Derivation" appendix** explaining why raw independent sampling is unstable and the physics behind the fix. Also documents optional top-level `sim_build` (VIC + joint kp/kd) on ranges JSON.
- `docs/real-world-proxy.md` — bench-proxy geometry, placement, stiffness tiers, and a still-open topology mismatch between the nominal and variance proxy fixtures (see its "Topology caveat").
- `docs/damping-tuning.md` — three-layer policy: bend (real \(E,\zeta\)), FIXED-joint damping (settle), stretch (`vbd_stretch_force` max load). Variance fixture `sim_build` is the canonical copy for batched-example weld damping.

### "How does system identification / sys-ID work?"

- `docs/system_identification.md` — the full M3 protocol (excitation trajectories and CMA-ES direction) **plus an implementation-notes appendix** for the shipped §2.1 quasi-static stepped mapping (trajectory phases, Fibonacci hemisphere, code map, tests).
- **`docs/youngs-modulus-sysid.md`** — canonical implemented E-grid, complete scoring, fused replay, ranking report/gate, and verification contract; handoff pointer to the separate CMA-ES command.
- **`docs/youngs-modulus-cmaes-implementation.md`** — CMA-ES loop behavior, counters, reports, gate, tests, and uv commands (V.5.2 verified).
- **README → CMA-ES sim-to-sim transfer** — copy-paste collect → fit → gate commands for Young's-modulus CMA-ES (`example_batched_collect_sysid_data.py` then `example_youngs_modulus_cmaes.py`).
- `docs/sysid-transition-features.md` — **MMD/Wasserstein state vector and transition-bag layout** (\(s\), \([s,\Delta s]\), median/hold one-hots, pooling, gate flags).
- `docs/sysid-trajectory-storage.md` — legacy single-env Parquet schema, collection/replay commands, dataset dashboard.
- `docs/batched-sysid-dataset.md` — **batched_sysid_v1** layout for parallel collection (`example_batched_collect_sysid_data.py`).
- `docs/sysid-mmd-grid-replay-alignment.md` — pre-weld strip, hold-metric semantics, structure-level weld, oracle vs `--infer-params` for the **shipped** in-process grid (`example_batched_sysid_mmd_grid.py`).
- `docs/batched-stability-monitor-design.md` — online per-env stability monitor used during collect/grid (Implemented).
- `docs/digital-twin.md` — observation-only replay initialization (shipped) and digital-twin geometry reconstruction; batched infer-only fidelity floor is deferred V.4.2.1 in `docs/ROADMAP.md`.

### "How does the Gym adapter work?"

- `docs/gym-observation-contract.md` — observation schema versioning (v1→v3), shared keys, env-specific keys.
- `docs/variable-impedance-teleop.md` — which envs are kinematic (`ApplePickCoupledEnv`) vs. dynamic/VIC (`ApplePickVicEnv` and its subclasses `ApplePickSysIdEnv`, `ApplePickReplayEnv`). Batched VIC/SysId envs: `apple_pick_gym/batched_envs/`.

## Known gaps (do not assume these are done without checking code/tests)

| Gap | Detail | Where documented |
| --- | ------ | ----------------- |
| Loss / GT scoring hardening (V.5.1) | **Done.** Soft-disable + exclude-fraction collect, transition features, median/hold-id/pooled Sinkhorn. GT should rank first on healthy samples; bad-sampling misses remain diagnostic and the operational gate uses a strict majority. | `docs/ROADMAP.md`, `docs/sysid-transition-features.md`, `docs/system_identification.md` |
| CMA-ES calibration (V.5.2) | Done — separate pycma loop over primary/spur/stem log10-E using fused pooled-Sinkhorn evaluation | `docs/ROADMAP.md`, `docs/system_identification.md` §4, `docs/youngs-modulus-cmaes-implementation.md`, design spec |
| Fused Young's grid acceptance | Implementation present; clean independent/fused timing, low-cap parity, build count, and peak-memory evidence pending | `docs/youngs-modulus-sysid.md`, `docs/ROADMAP.md` |
| Batched digital-twin fidelity (V.4.2.1) | Done as shipped helpers + `--infer-params`; infer-only fidelity floor optional cleanup (not Current focus) | `docs/ROADMAP.md`, `docs/sysid-mmd-grid-replay-alignment.md`, `docs/digital-twin.md` |
| `real_world_proxy.json` topology | Nominal fixture uses `linear_chain`; its variance counterpart defaults to `t_junction`. The two fixtures for the same physical proxy build different topologies | `docs/real-world-proxy.md` |

## Conventions worth knowing before editing docs or code

- **Slice IDs** (`M1`, `[V].3.1`, etc.) are owned by `docs/ROADMAP.md`. Don't invent new ones in topic docs; link to ROADMAP instead of duplicating a phase table.
- **`newton/`** is vendored — match existing patterns in `newton/newton/examples/` and `newton/newton/_src/` rather than inventing new Warp/Newton APIs.
- **TDD** — tests before implementation, per `.cursor/rules/test-driven-development.mdc`.
- **GPU hot paths** — Warp kernels for anything that runs every substep or over many bodies; CPU/NumPy is fine for one-off setup, frame-rate teleop, and debug. See `.cursor/rules/gpu-warp-parallelism.mdc`.
- **Worktrees** — substantial production changes should happen in a dedicated git worktree, not directly on `main`. See `.cursor/rules/worktree-feature-dev.mdc`.
