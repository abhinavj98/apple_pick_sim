# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-07-15 |
| **Owner**        | Abhinav |
| **Vision**       | See `docs/VISION.md` |
| **Active work**  | **[V].5.1** — loss/features shipped; GT ranks **#1** on good samples (bad ranks from bad sampling allowed); leftover: optional `--score-mmd`, invariant tests |

---

## How this roadmap is structured

| Section                   | Purpose |
| ------------------------- | ------- |
| **Sequencing**            | Milestone order from vision to implementation. |
| **Current focus**         | What to do *now* — next slice and unchecked checklist. |
| **Milestones**            | Status tables per phase. |
| **Backlog**               | Deferred work — not active. |
| **Agent execution notes** | Validation commands, layout, when to stop. |

---

## Sequencing

| Phase | Status | Outcome |
| ----- | ------ | ------- |
| **[P0]** | Done | Variational fruiting geometry, fixtures, force readouts, `example_fruiting_system.py` |
| **[M1]** | Done | MuJoCo + VBD coupling, FR3, VIC teleop, settle→weld (`docs/mujoco-vbd-coupling-architecture.md`) |
| **[M2]** | Partial | M2.1 `ApplePickCoupled-v0` shipped; RL/FID slices in backlog |
| **[M3]** | Infra done | Sys-ID recording, replay, legacy single-env MMD grid (`docs/system_identification.md`); batched paths → **V.4** / **V.5** |
| **[V].1–2** | Done | Batched `replicate(N)`, heterogeneous per-env DR, fixtures, runtime actions (`docs/vectorized-coupled-fruiting.md`) |
| **[V].3** | Partial | Sim API + batched gym (V.3.3 done; V.3.4–V.3.5 pending) |
| **[V].4** | Done | Parallel collect, batched replay, in-process MSE/Wasserstein grid; V.4.2.1 digital-twin fidelity capstone deferred |
| **[V].5** | **Now** | Harden batched grid loss (`example_batched_sysid_mmd_grid.py`) → CEM → held-out validation (absorbs former **[S]** / M3.2) |
| **[M4]** | Later | Real-data collection — after **V.5** |
| **[M5]** | Later | Final pick policy |

---

## Current focus

**Next slice:** **V.5.1** leftovers — optional CLI `--score-mmd` and documented/invariant tests. Core loss hardening is **accepted**: under good excitation/sampling, the **GT stiffness candidate constantly ranks #1** (hold MSE / Sinkhorn with median + hold-id + pooled dirs). Occasional worse ranks are attributed to **bad sampling** (e.g. wrench-saturated / non-discriminative trials) and **are allowed** — do not treat those as scoring-logic failures. Gate harness pass bar `gt_rank ≤ 2` remains a soft diagnostic for noisy seeds, not a denial of #1-on-good-data.

**Scope remaining for this slice:**

- Optional CLI `--score-mmd` exposing `evaluate_batched_mmd_grid`
- Documented invariants + tests for GT preference under known-good datasets (codify: #1 expected when sampling is healthy; bad sampling may miss #1)

**Shipped wins (do not reimplement):**

- Settle stack: quiet / zero-qd, opt-in gravity ramp, **opt-in** settle cache (off by default), KE/QS diagnostics (`docs/vectorized-coupled-fruiting.md`)
- **V.4.2** parallel GT collection (`batched_sysid_v1`, `example_batched_collect_sysid_data.py`)
- Batched recorded-action replay (`replay_batched_sysid_structure`)
- **V.4.3** in-process GPU-batched stiffness grid: MSE + Sinkhorn Wasserstein + Plotly viz (`example_batched_sysid_mmd_grid.py`; alignment notes in `docs/sysid-mmd-grid-replay-alignment.md`)
- **V.5.1 mid-slice — stable collect / replay:** soft-disable during collect (`EnvDisableController`, sticky on NaN/IK); manifest `excluded` / `excluded_reason`; offline `exclude_unstable_episodes` (exclude when unstable-frame **fraction > 0.25**, preserve already-excluded); online stability force/torque caps **50 N** / **20 N·m** (`docs/batched-stability-monitor-design.md`); scripts `scripts/collect_and_rank_sysid_gt.sh`, `scripts/gate_sysid_gt_sinkhorn.sh` (design: `docs/superpowers/specs/2026-07-12-sysid-stable-collect-replay-design.md`)
- **V.5.1 mid-slice — scoring / features:** transition-feature contract (`docs/sysid-transition-features.md`); CLI defaults `--use-median` / `--hold-id-onehot` / `--pool-directions` **on** (pool forces dir one-hot; disable with `--no-*`; deprecated `--mse-hold-aggregation` / `--mse-hold-latter-half`); named Sinkhorn gates via `scripts/gate_sysid_gt_sinkhorn.sh` (script default `GATE=gate_pooled_dirs`; also `gate_median_hold` / `gate_hold_id`); `sysid_gate_report.py` + grid-viz paired-hold woody MSE helper

**Shipped ranking policy:** On healthy samples, GT **constantly ranks #1**. Bad ranks from bad sampling are allowed (fixture/excitation issue, not loss bug).

**Existing tooling (still useful):** `mmd_features.py` `stable` mask, median hold aggregation / hold→hold median bags, grid-viz candidate `disqualified` flags, hold impulse flags in `batched_hold_quasi_static.py`, online `batched_stability_monitor` (`docs/batched-stability-monitor-design.md`), soft-disable + exclude-fraction policy above.

**Deferred (not Current focus):** **V.4.2.1** — helpers + `--infer-params` exist; default sim-sim path still uses oracle `fruiting_system_params`; no infer-only fidelity floor test yet.

**Goal:** Keep GT-preferring scores from `example_batched_sysid_mmd_grid.py` (good sampling → rank #1) → **V.5.2** CEM → **V.5.3** held-out validation → **[M4]**.

**Specs:** `docs/system_identification.md`, `docs/sysid-transition-features.md`, `docs/sysid-mmd-grid-replay-alignment.md`, `docs/batched-sysid-dataset.md`, `docs/batched-stability-monitor-design.md`, `docs/digital-twin.md`, `docs/material-parameter-sampling.md`

**Build on (do not reimplement):**

- [M1] `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`, VIC joint torques (`docs/variable-impedance-teleop.md`)
- [M2.1] `apple_pick_gym/` observation contract v3 (`docs/gym-observation-contract.md`); θ packing reused in V.4 / V.5
- [V.4.3] `example_batched_sysid_mmd_grid.py`, `batched_sysid_mmd_grid.py`, `evaluate_batched_mmd_grid` / `score_candidate_mmd` (library), MSE/Wasserstein CLI paths

### Next up (ordered)

**[V].3 — sim API + gym migration** (parallel / lower priority)

- [x] **V.3.1 — Extract batched heterogeneous sim API:** `BatchedHeterogeneousCoupledSim` + config dataclass in `apple_pick_sim/coupled_fruiting/`. Surface: `build()`, `step(per_env_actions)`, `gather_obs()`, layout/scene accessors. No `argparse` / viewer in library; tests import library, not `examples/`.
- [x] **V.3.2 — Thin heterogeneous example:** `example_batched_heterogeneous_coupled_sim.py` (canonical entry point); `test_heterogeneous_coupled_fruiting.py` imports library, not monolith example.
- [x] **V.3.3 — Batched gym base env:** `ApplePickBatchedBaseEnv`, `ApplePickBatchedVicEnv`; `num_envs` + `gather_batched_obs`; episode snapshot reset; `example_batched_gym_keyboard.py`.
- [ ] **V.3.4 — Migrate gym envs:** `ApplePickCoupledEnv`, `ApplePickVicEnv`, `ApplePickSysIdEnv`, `ApplePickReplayEnv` on batched backend; retire single-world gym build path.
- [ ] **V.3.5 — Migrate gym examples:** `example_gym_sysid.py`, `example_gym_replay.py`, `example_gym_keyboard.py` + parity tests at `num_envs=1`.

**[V].4 — batched sys-ID** (backend: V.3.1 API + V.3.3 batched gym)

- [x] **V.4.1 — Recorded-action replay:** `replay_batched_sysid_structure` drives recorded EE actions on candidate stiffnesses. (No `gather_transitions()` API symbol; transition bags live in MMD/Wasserstein feature code — backlog if a public gather API is needed.)
- [x] **V.4.2 — Parallel GT collection:** `ApplePickBatchedSysIdEnv`, `example_batched_collect_sysid_data.py`, `batched_sysid_v1` Parquet layout (`docs/batched-sysid-dataset.md`).
- [ ] **V.4.2.1 — Digital-twin replay verification (deferred):** helpers + CLI `--infer-params` exist (`batched_digital_twin_init.py`); fidelity capstone still uses full `fruiting_system_params` / oracle default. Not Current focus.
- [x] **V.4.3 — In-process batched grid:** `example_batched_sysid_mmd_grid.py` + `batched_sysid_mmd_grid.py` (MSE / Sinkhorn Wasserstein + viz). Library MMD (`evaluate_batched_mmd_grid`) exists; CLI `--score-mmd` optional leftover → V.5.1. Legacy single-env: `run_system_identification.py`.
- [ ] **V.4.4 — Sys-ID tooling at batch scale** (native v1 replay dashboard; further deprecate legacy subprocess patterns)

**[V].5 — sim-sim transfer wrap-up**

- [ ] **V.5.1 — Harden loss calculation in `example_batched_sysid_mmd_grid.py` (Next; ranking accepted, polish left):**
  - [x] Soft-disable + manifest `excluded` / offline exclude (unstable-frame fraction > 0.25) + stability caps 50 N / 20 N·m
  - [x] Documented transition-feature / Sinkhorn scoring contract (`docs/sysid-transition-features.md`) + named gate CLI (`scripts/gate_sysid_gt_sinkhorn.sh`; default `gate_pooled_dirs`, also `gate_median_hold` / `gate_hold_id`)
  - [x] GT constantly ranks **#1** on hold MSE / Wasserstein under **good** sampling; worse ranks from bad sampling are allowed (not a loss bug)
  - [ ] Optional CLI `--score-mmd` exposing `evaluate_batched_mmd_grid`
  - [ ] Documented invariants + tests for GT preference under known-good datasets
- [ ] **V.5.2 — CEM calibration loop** (M3.2)
- [ ] **V.5.3 — Held-out sim-sim validation** + [M4] handoff criteria

**[M3] parallel infra** (optional alongside [V])

- [x] **M3.0.4 — Digital-twin fixture catalog:** `digital_twin_fixture_catalog.json`, `digital_twin_obs_straight_rod_initial.json`, and catalog tests shipped. Batched infer-only fidelity floor remains V.4.2.1 (deferred).
- [ ] **M3.0.5 — §2.2–2.3 excitation trajectories** (log chirp, torsional) — wire through V.4 when added

---

## Milestones (summary)

### [P0] · [M1] · [M2]

See **Sequencing** table. M2 deferred slices: M2.0 (interface ADR), M2.2a (`ApplePickFID-v0`), M2.2c (SKRL smoke), M2.3 (π_exp training).

Key M1 docs: `docs/mujoco-vbd-coupling-architecture.md`, `docs/WRENCH_READOUT.md`, `docs/variable-impedance-teleop.md`, `docs/gpu-coupling-optimization.md`.

### [M3] Infra

| Slice | Status | Notes |
| ----- | ------ | ----- |
| M3.0.1–M3.0.3 | Done | §2.1 trajectory, Parquet recording, observation-only replay |
| M3.1.1 | Done | Legacy single-env MMD grid — superseded by V.4.3 |
| M3.0.4 | Done | Fixture catalog + example obs JSON committed; `test_digital_twin.py` passes (`docs/digital-twin.md`) |
| M3.0.5 | Planned | §2.2–2.3 trajectories → V.4 backend |
| M3.1.2 | → V.5.1 | Harden loss calc in `example_batched_sysid_mmd_grid.py` |
| M3.2 | → V.5.2 | CEM calibration |

### [V] Batched vectorization

Fixed topology per batch (`num_segments`, `omit`); per-env `FruitingSystemParams` vary. Spec: `docs/vectorized-coupled-fruiting.md`.

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| **V.1** | Done | `replicate(N)`, batched settle→weld, `BatchedTemplateIK`, homogeneous example + tests |
| **V.2.1** | Done | Per-env IK bootstrap (heterogeneous path) |
| **V.2.1.1** | Done | `newton/` bump; parity fixes |
| **V.2.1.2** | Done | Fixture stability + real-world likeness (`docs/material-parameter-sampling.md`, "Derivation" section) |
| **V.2.1.3** | Done | Material \(E\), \(\zeta\) sampling (`docs/material-parameter-sampling.md`) |
| **V.2.2** | Done | Build-time per-env θ DR via `add_world` |
| **V.2.3** | Done | Per-env runtime actions (`velocity_for_world`, action buffer) |
| **V.3.1** | Done | `BatchedHeterogeneousCoupledSim` library API |
| **V.3.2** | Done | Thin heterogeneous example (`example_batched_heterogeneous_coupled_sim.py`) |
| **V.3.3** | Done | `ApplePickBatchedBaseEnv`, `ApplePickBatchedVicEnv`, episode snapshot reset |
| **V.3.4** | Planned | Migrate gym envs |
| **V.3.5** | Planned | Migrate gym examples (`num_envs=1` parity) |
| **V.4.1** | Done | Recorded-action replay (`replay_batched_sysid_structure`) |
| **V.4.2** | Done | Parallel GT collection (`batched_sysid_v1`, `ApplePickBatchedSysIdEnv`) |
| **V.4.2.1** | Deferred | Infer-params / obs-init fidelity capstone (helpers exist; not Current focus) |
| **V.4.3** | Done | In-process batched MSE/Wasserstein grid + viz; library MMD present |
| **V.4.4** | Planned | Native v1 replay/dashboard at batch scale |
| **V.5.1** | **Next** (ranking accepted) | GT #1 on good samples; bad sampling ranks allowed; leftover `--score-mmd` + invariant tests |
| **V.5.2** | Planned | CEM θ loop |
| **V.5.3** | Planned | Held-out validation; [M4] handoff |

Canonical entry point: `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py`. Public API reference: `docs/coupled-sim-api.md`.

**Consumers after V.5:** [M4] real-data validation; M2.3 / M2.2c parallel RL envs.

> Former **[S] sim-sim transfer** is fully absorbed into **V.4** (grid diagnostic) and **V.5** (harden `example_batched_sysid_mmd_grid.py` loss → CEM → validation).

---

## Backlog

- **[M2] remaining** — resume after V.5 or V.3.5 if directed; batched backend needs V.3.3–V.3.5
- **[V] deferred** — per-env geometry DR on reset without rebuild; `(N, act_dim)` policy tensor path; batched VIC as default gym controller; public `gather_transitions()` API if needed beyond feature bags
- **V.4.2.1** — infer-only digital-twin fidelity floor on `batched_sysid_v1`
- **Scope changes** — additional manipulators/crops; triangle mesh import (P0 stays capsules)
- **[M4]** real-data pipeline, **[M5]** final pick policy — after V.5

---

## Agent execution notes

> **Warning — intra-cable collisions are disabled.** Default builds set `enable_self_collisions=False` (woody↔woody, woody↔apple, stem↔woody, apple↔proxy filtered; ground only). See `docs/vectorized-coupled-fruiting.md` and `apple_pick_sim/fruiting_system/build.py::_apply_default_fruiting_collision_filters`.

**Repository layout:**

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/` | Simulation (`fruiting_system/`, `coupled_fruiting/`, `examples/`, tests) |
| `apple_pick_sim/coupled_fruiting/` | Coupled sim; `BatchedHeterogeneousCoupledSim` + `batched_heterogeneous_config.py` (V.3.1) |
| `apple_pick_gym/` | Gymnasium adapter; `envs/` (legacy single-world), `batched_envs/` (V.3.3+ grid/collect), `batched_examples/` |
| `newton/` | Upstream physics submodule (vendored) |
| `docs/` | Vision, roadmap, architecture notes — start at `docs/CODEBASE_GUIDE.md` for a map of the codebase and this directory |

**How to validate changes:**

```bash
# Install / sync (repo root; path-depends on newton/)
uv sync --extra gym --extra vic --extra dev

# Fast test gate (excludes @pytest.mark.slow)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"

# Gym env tests
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/ -q

# M1 coupled smoke (headless)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 60

# Coupling verification
uv run python apple_pick_sim/diagnostics/verify_coupling.py --num-substeps 600 --max-force 5 --max-torque 1

# Settle stack
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_settle_then_weld.py \
  apple_pick_sim/tests/test_settled_checkpoint.py -q

# [V].3 heterogeneous batched coupled (library API + thin example; FR3 required)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_heterogeneous_build.py \
  apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_batched_heterogeneous_config.py \
  apple_pick_sim/tests/test_example_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42

# Pre-gym cleanup regression (batched FR3 builders + broadcast helpers)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_vectorized_coupled_fruiting.py \
  apple_pick_sim/tests/test_broadcast_actions.py \
  apple_pick_sim/tests/test_package_layout.py -q

# Material-parameter sampling (V.2.1.3)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py -q \
  -k "material or youngs or damping_ratio or sample_params"

# [V].4 batched sys-ID collection + replay
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_trajectory_store.py \
  apple_pick_gym/tests/test_batched_sysid_env.py \
  apple_pick_gym/tests/test_batched_sysid_collect.py \
  apple_pick_gym/tests/test_batched_sysid_replay.py \
  apple_pick_gym/tests/test_batched_sysid_replay_fidelity.py -q
uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 \
  --max-steps 200 --output /tmp/batched_sysid_dataset

# [V].4.3 in-process batched stiffness grid (MSE / Wasserstein)
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_sysid_grid_viz_table.py \
  apple_pick_gym/tests/test_batched_sysid_grid_viz_integration.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py -q
uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \
  --viewer null --dataset /tmp/batched_sysid_dataset --replay-only --score-mse \
  --plot-output /tmp/mmd_grid \
  --primary-bend-stiffness-values 1e-4,2e-4 \
  --secondary-bend-stiffness-values 1e-4 \
  --spur-bend-stiffness-values 1e-4 \
  --stem-bend-stiffness-values 1e-4,2e-4

# [V].3.3 batched gym base + VIC env
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_vic_env.py \
  apple_pick_gym/tests/test_batched_obs_torch.py -q

# M3 sys-ID (legacy single-env path; V.3.5 migrates examples to batched backend)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_gym/tests/test_sysid_env.py -q
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 --output /tmp/apple_pick_sysid_gt

# Legacy single-env MMD grid (prefer example_batched_sysid_mmd_grid.py for batched_sysid_v1)
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/apple_pick_sysid_gt --viewer null \
  --primary-bend-stiffness-values 10,25 \
  --secondary-bend-stiffness-values 10,25 \
  --spur-bend-stiffness-values 10,25 \
  --stem-bend-stiffness-values 10,25

# Digital-twin fixture catalog
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q

# [V].5.1 feature / Wasserstein unit tests
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_mmd.py -q

# [V].5.1 stability / soft-disable / exclude / gate-report tests
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_stability_monitor.py \
  apple_pick_gym/tests/test_env_disable_controller.py \
  apple_pick_gym/tests/test_exclude_unstable_episodes.py \
  apple_pick_gym/tests/test_sysid_gate_report.py -q

# Optional Sinkhorn gate wrapper (not a full slow e2e; needs GPU + long runtime)
# Default GATE=gate_pooled_dirs (matches CLI pool/hold-id defaults):
# bash scripts/gate_sysid_gt_sinkhorn.sh
# bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_median_hold
# bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_hold_id
```

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict
- Policy or product decisions (scope, user-visible behavior)
- Network credentials, paid APIs, or destructive operations

**When unsupervised is expected:**

- Complete the **next unchecked slice** in Current focus using TDD and project rules
- Fix small blockers uncovered by that slice (tests, imports, typos)
- Do **not** start a new milestone or backlog item without maintainer direction
