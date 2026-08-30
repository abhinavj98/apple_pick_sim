# Codebase guide

**Purpose:** a map of this repository — what the packages do, where to look for a given piece of behavior, and which document answers which question. Read this after `docs/VISION.md` / `docs/ROADMAP.md` and before searching the codebase blind.

This guide describes **structure**, not **status**. For "what's done / what's next," always defer to `docs/ROADMAP.md` — do not infer status from a doc's prose tense alone.

## Document status

| Field | Value |
| ----- | ----- |
| **Last reviewed** | 2026-08-14 (H1–H5 index; removed handbook stubs and absorbed living duplicates) |
| **Owner** | Abhinav |

## How to read this repository

1. **`docs/VISION.md`** — why this project exists, scope, non-goals, success criteria. Read once, revisit rarely.
2. **`docs/ROADMAP.md`** — the single source of truth for status: what's shipped, what's active, what's next, validation commands. Every other doc in this repo should defer to it for sequencing.
3. **The H1–H5 handbooks** — living subsystem contracts, selected from the [Document index](#document-index) below.
4. **This file** — architecture map + full doc index.
5. **Code** — once a handbook points you at a module, read the code; docs describe intent and contracts, not line-by-line behavior.

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

**Two-solver split (core architectural fact, M1):** `SolverMuJoCo` cannot represent `JointType.CABLE`, so the fruiting tree (plant) and the FR3 arm live on **two separate `newton.Model` instances**, coupled through proxy bodies and a one-substep-lag wrench exchange. This is the single most important thing to understand before touching simulation code — see H1 `docs/handbook-coupled-simulation.md`.

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
| `robot_replay/` | Real-robot sys-ID episodes + convert CLI; pre-grasp rebuild / post-grasp weld contract in `README.md` (woody via `rest_snapshot_during_run`) |
| `newton/` | Upstream Newton submodule — vendored, match its patterns rather than inventing APIs |
| `docs/` | Living handbooks and supporting references (below); `docs/superpowers/{specs,plans}/` and `docs/specs/` are dated design archives |

## Document index

Start with the five living domain handbooks. They own current subsystem
contracts; `docs/ROADMAP.md` alone owns status and sequencing.

### Start here

- **`docs/ROADMAP.md`** — the only place that should be trusted for slice status, sequencing, and validation commands.
- `docs/VISION.md` — intent, scope, non-goals, success criteria (rarely changes).
- `docs/FEATURES.md` — short feature → handbook → code-entry lookup.

### Living domain handbooks (H1–H5)

- **H1 — `docs/handbook-coupled-simulation.md`** — two-model
  MuJoCo/VBD ownership, public batched API, settle→weld, wrench/payload
  semantics, geometry, and the GPU hot path. Code starts in
  `apple_pick_sim/coupled_fruiting/` and `apple_pick_sim/fruiting_system/`.
- **H2 — `docs/handbook-variable-impedance.md`** — `vic` and `vic_pose`,
  total TCP wrench, torque control, caps, and soft-disable behavior. Code
  starts in `apple_pick_sim/coupled_fruiting/vic_joint_torques*.py`,
  `ee_impedance*.py`, and `vic_wrench.py`.
- **H3 — `docs/handbook-sysid-scoring.md`** — `batched_sysid_v1`, aligned
  state/transition bags, fixed physical scales, and MMD/Sinkhorn scoring.
  Code starts in `apple_pick_sim/system_id/{mmd_features,mmd,wasserstein}.py`
  and the trajectory stores.
- **H4 — `docs/handbook-real-replay.md`** — real Parquet conversion,
  pre-grasp rebuild, post-grasp settle/weld, `vic_pose` replay, and the shared
  real builder. Code starts in `robot_replay/`,
  `apple_pick_sim/system_id/real_to_batched_sysid.py`, and
  `apple_pick_gym/batched_envs/real_batched_replay_build.py`.
- **H5 — `docs/handbook-youngs-cma.md`** — support-\(k_p\) × spur/stem
  Young's-\(E\) phenotype, Cartesian/fused ranking, CMA-ES, gates, and the
  real-data handoff. Code starts in
  `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` and the Young's
  examples/reports.

### Supporting references (unique math / protocol only)

Do not treat these as second copies of the handbooks. They keep derivations
and experiment notes that H1–H5 summarize rather than duplicate.

- **Coupling (H1):** `docs/WRENCH_READOUT.md`,
  `docs/explicit-apple-load-tcp-harvest.md`,
  `docs/gpu-coupling-optimization.md`, and
  `docs/heterogeneous-batched-vectorization-audit.md`.
- **Plant/material (H1):** `docs/material-parameter-sampling.md`,
  `docs/real-world-proxy.md`, and `docs/damping-tuning.md`.
- **Observations (H2/H3):** `docs/gym-observation-contract.md`.
- **Sys-ID protocol/stability (H3/H5):** `docs/system_identification.md` and
  `docs/batched-stability-monitor-design.md`.
- **Replay/twin (H4):** `docs/digital-twin.md`, `docs/connection-angles-implementation.md`,
  and `robot_replay/README.md`.
- **Open defect (H5):** `docs/in-process-rebuild-heap-corruption.md` — intermittent
  Warp/Newton host-heap corruption when rebuilding and stepping many scenes in one
  process; why `--no-isolated-eval-waves` is unsafe.

### Archived design records

- `docs/superpowers/specs/` — dated design decisions and implementation
  snapshots. Their status/canonical fields point back to H1–H5; do not treat
  an unstamped prose tense as current status. **Do not read these first.**
- `docs/superpowers/plans/` — dated execution plans and completed task
  checklists. These explain how work was staged, not how the shipped system
  behaves now.
- `docs/specs/` — older design archive retained for history.

## Known gaps (do not assume these are done without checking code/tests)

| Gap | Detail | Where documented |
| --- | ------ | ----------------- |
| M4.0 trusted real ranking → CMA | Plumbing and feature alignment are Done. Next is a trusted Cartesian Sinkhorn/grid ranking smoke on aligned real bags, followed by `example_youngs_modulus_cmaes.py` on the same shared real builder. | `docs/ROADMAP.md`, H3, H4, H5 |
| Real dataset discovery | Multi-episode manifest/discovery remains open (or the workflow must document one dataset per episode). | `docs/ROADMAP.md`, H4 |
| Fused Young's grid acceptance | Implementation is present; clean independent/fused timing, low-cap parity, build-count, and peak-memory evidence remain pending. | `docs/ROADMAP.md`, H5 |
| Batched digital-twin fidelity (V.4.2.1) | Helpers and `--infer-params` shipped; an infer-only fidelity floor is optional deferred cleanup, not Current focus. | `docs/ROADMAP.md`, `docs/digital-twin.md`, H4 |
| `real_world_proxy.json` topology | The nominal fixture uses `linear_chain`; its variance counterpart defaults to `t_junction`, so the two fixtures build different topologies. | `docs/real-world-proxy.md`, H1 |
| Held-out sim-sim validation (V.5.3) | Deferred until after M4.0 or explicit maintainer direction. | `docs/ROADMAP.md`, H5 |
| In-process rebuild heap corruption | Open, not root-caused. Rebuilding and stepping many scenes in one process intermittently corrupts host object state (SIGSEGV / bogus `code`/`function` attribute errors). Use process-isolated eval waves; `--no-isolated-eval-waves` is unsafe. | `docs/in-process-rebuild-heap-corruption.md`, H5 |

## Conventions worth knowing before editing docs or code

- **Slice IDs** (`M1`, `[V].3.1`, etc.) are owned by `docs/ROADMAP.md`. Don't invent new ones in topic docs; link to ROADMAP instead of duplicating a phase table.
- **`newton/`** is vendored — match existing patterns in `newton/newton/examples/` and `newton/newton/_src/` rather than inventing new Warp/Newton APIs.
- **TDD** — tests before implementation, per `.cursor/rules/test-driven-development.mdc`.
- **GPU hot paths** — Warp kernels for anything that runs every substep or over many bodies; CPU/NumPy is fine for one-off setup, frame-rate teleop, and debug. See `.cursor/rules/gpu-warp-parallelism.mdc`.
- **Worktrees** — substantial production changes should happen in a dedicated git worktree, not directly on `main`. See `.cursor/rules/worktree-feature-dev.mdc`.
