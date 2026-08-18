# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-08-17 |
| **Owner**        | Abhinav |
| **Vision**       | See `docs/VISION.md` |
| **Active work**  | **[M4].0** 1×8 holdout pipeline shipped; **Task 9 GPU science gate FAILED** (val torque magnitude; Sinkhorn + TCP passed) |

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
| **[M1]** | Done | MuJoCo + VBD coupling, FR3, VIC teleop, settle→weld (`docs/handbook-coupled-simulation.md`, `docs/handbook-variable-impedance.md`) |
| **[M2]** | Partial | M2.1 `ApplePickCoupled-v0` shipped; RL/FID slices in backlog |
| **[M3]** | Infra done | Sys-ID recording, replay, legacy single-env MMD grid (`docs/handbook-sysid-scoring.md`); batched paths → **V.4** / **V.5** |
| **[V].1–2** | Done | Batched `replicate(N)`, heterogeneous per-env DR, fixtures, runtime actions (`docs/handbook-coupled-simulation.md`) |
| **[V].3** | Done | Sim API + batched gym (V.3.1–V.3.5) |
| **[V].4** | Done | Parallel collect, batched replay, in-process MSE/Wasserstein grid, tooling |
| **[V].5** | Infra Done | V.5.1–V.5.2 Done; **V.5.3 held-out deferred** while **M4.0** starts |
| **[M4]** | **Now** | **M4.0** real `robot_replay` → CMA-ES (`vic_pose`); further real collection later |
| **[M5]** | Later | Final pick policy |

---

## Current focus

**Next slice:** decide how to treat the **Task 9 torque-magnitude fail** on s09 holdout (sim \(\lvert\tau\rvert\) ~15–70× too small on val dirs 0/1/3). Folder convert, per-direction weld/gripper/joints, last-action pad + truncate-before-features, and opt-in holdout CMA (`--direction-split-seed`) are shipped (Tasks 1–8). Task 9 **ran** shipped knobs (pop 15 / gen 10, ~32 min RTX 4090): train Sinkhorn 22.54→17.08, val Sinkhorn 23.63→17.13, TCP magnitude+trend pass; `force_magnitude_ok` fails all three val dirs on **torque ratio** (0.073 / 0.014 / 0.044). **Do not claim the science gate passed.** Use H3 `docs/handbook-sysid-scoring.md` for signed \(F_\parallel\) / \(x_{\mathrm{hold0}}\) gates and one-hot width, H4 `docs/handbook-real-replay.md` for convert/replay, and H5 `docs/handbook-youngs-cma.md` for holdout flags / `holdout_report.json`. Slices 0–3 delivered USD `/fr3/ee` COM + inertia → convert-time `R(tcp) @` F/T (no second negate) → two-start woody + `apple_pos` → scalar `hold_number` from `hold_index`. Convert writes unfiltered world `ft_wrist` plus scored `ft_wrist_lpf` (10 Hz `filtfilt` + 30 Hz block-mean). **No sim EMA/LPF. 19D `action` in bags/replay; not in Sinkhorn `STATE_VECTOR`.** Bit-1/2 Done (convert + open-loop FR3 + 19D pose packing + `example_replay_real_batched.py`). **Bit-3 slice 1 Done:** shared real-replay `build_env_fn` + grid opt-in (`vic_pose` / 19D from dataset metadata); sim-sim twist default preserved.

**Phenotype (unchanged):** support-joint \(k_p\) × spur/stem Young's \(E\) (primary \(E\) fixed); see H5 `docs/handbook-youngs-cma.md`.

**Slice 1 Done (plumbing — do not reimplement):**

- Convert packs `vic_pose_v1` / `action_dim=19` from real parquet (`target_pose_4x4` + `dump.controller_gains`)
- Grid / multi-replay / CMA builders **opt into** `ControllerConfig(mode="vic_pose", action_dim=19)` from dataset metadata (`--controller-mode`); twist `vic` default preserved for gym collect, MMD, sim-sim CMA
- Real GT feature bags load without sim-oracle phenotype (`gt_candidate is None`; `--include-gt-candidate` forced off on real datasets)
- Post-grasp SE(3) on shared `make_real_replay_build_env_fn` + batched `apply_logged_post_grasp_se3_to_cable` (grid/CMA real replay match example)
- README + validation commands documented; unit/CLI tests for metadata→controller mode

**Scope for feature alignment (Done — spec `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md`):**

- **Slice 0:** USD `/fr3/ee` mass 1.1 kg, COM `(0,0,−0.077)`, diagonal `I_ee`; massless `/ee/tcp`
- **Slice 1:** Convert `R(tcp) @` logged F and τ; raise if `tcp_pose_4x4` missing; score convert-time `ft_wrist_lpf` (keep unfiltered `ft_wrist`); **no sim EMA/LPF**
- **Slice 2:** Compiler Branch/Spur/Apple → two woody starts + `apple_pos`; drop `woody_end` from sys-ID bag / collector / `STATE_VECTOR`
- **Slice 3:** Scalar `hold_number` from `hold_index`; one-hot only at score time
- **Not in scope:** scoring `action` (including pose-only `action[0:7]`); sim F/T low-pass; plumbing spec score-time F/T+LPF plan (superseded)

**Out of scope (M4.0 remaining / slice 2+):**

- Migrating gym collect / MMD / default CMA sim-sim off twist `vic`
- Force-hybrid / wrench-apply mode (pose PD via `vic_pose` is the drive)
- Full [M4] new bench collection protocol (use existing `robot_replay/` logs)
- **V.5.3** held-out sim-sim validation (deferred; resume after M4.0 smoke or when directed)

**Checklist:**

- [x] `example_replay_real_batched.py`: pre-grasp init apple → logged post-grasp apple+TCP SE(3) at weld (helpers in `batched_digital_twin_init`; spec `docs/superpowers/specs/2026-08-11-batched-real-replay-post-grasp-se3-design.md`)
- [x] **CMA / shared path (slice B):** post-grasp SE(3) on shared `make_real_replay_build_env_fn` + batched `apply_logged_post_grasp_se3_to_cable(..., layout=...)` so grid/CMA real replay match the example (spec `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md`)
- [x] Dataset discovery / multi-episode manifest for real converted dirs (`--input-dir` folder convert; `sXX-dNN.parquet` → `direction_idx=NN`; 1×N `batched_sysid_v1`)
- [x] CMA / multi-replay / grid build path selects `vic_pose` + `action_dim=19` from dataset metadata (twist default preserved; `--controller-mode` opt-in)
- [x] Real GT feature bags load without requiring sim `fruiting_system_params` as the ranking oracle (`gt_candidate is None`; `--include-gt-candidate` forced off on real datasets)
- [x] Unit/CLI tests for metadata→controller mode selection; refuse wrench-as-twist regressions
- [ ] End-to-end smoke on `s02-d00` (convert → grid) with `--viewer null` — commands documented in validation block + `robot_replay/README.md`; **not executed in worktree** (parquet gitignored); acceptance: envs build, 19D steps, no wrench-as-twist, no `sim_config` crash; ranking **not** trusted until post-alignment smoke
- [x] README + validation commands updated
- [x] **Alignment slice 0:** USD `/fr3/ee` mass/COM/`I_ee` from recorded `ee_config` (spec `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md`)
- [x] **Alignment slice 1:** Convert-time `R(tcp) @` F/T; scored `ft_wrist_lpf`; no sim EMA; full 19D action in bag
- [x] **Alignment slice 2:** Two-start woody + `apple_pos`; `woody_end` dropped from sys-ID bag
- [x] **Alignment slice 3:** Scalar `hold_number` from `hold_index`
- [ ] **Post-alignment:** Trusted Cartesian ranking on aligned bags (Sinkhorn smoke / grid) — superseded as M4.0 *acceptance* by the holdout science gate below; still not claimed
- [x] **Slice 4:** CMA `example_youngs_modulus_cmaes.py` on the same real builder
  (**1×1 wiring** Done)
- [x] **Folder convert + multi-direction replay** (H4): per-dir weld/gripper/arm joints; last-action pad; truncate before features
- [x] **Opt-in holdout CMA** (H5): `--direction-split-seed` (default 17 when present); train-only fit; frozen `final_mean`; `holdout_report.json`; exit 1 on gate fail
- [x] **Task 9 GPU run (science gate recorded, not passed):** convert `s09` → holdout CMA `--direction-split-seed 17`, shipped pop=15 / gen=10. Train `eligible_mean` 22.54→17.08; val Sinkhorn 23.63→17.13; TCP mag+trend pass on `{0,1,3}`. **FAIL:** `force_magnitude_ok` false on all three val dirs because torque ratio \(\ll 1/3\) (0.073 / 0.014 / 0.044); parallel-force ratios were in \([1/3, 3]\). Fitted `log10` `[2.265, 9.120, 10.994]`. Artifacts under `tmp/real_kp_e_cmaes_s09_holdout/` (gitignored). **Do not treat M4.0 ranking as accepted.**
- [ ] **Follow-up:** Real-mode CMA still seeds from shipped
  `initial_mean_log10=[4.0, 9.5, 9.5]` while the spec fixture band is
  ~\(10^{7.4}\)–\(10^{8}\) Pa; the real spur/stem floor (\(\log_{10} E = 7\))
  makes that region reachable, but search starts ~1.5σ high — retarget mean
  later, not in this slice.
- [ ] Two-tree merge / drop the one-structure `vic_pose` guard (out of scope for this holdout slice)

**Build on (do not reimplement):**

- H4 `docs/handbook-real-replay.md` — conversion, 19D pack, post-grasp SE(3), replay CLI, and shared `make_real_replay_build_env_fn`
- H2 `docs/handbook-variable-impedance.md` — `vic_pose` action semantics, anisotropic wrench, and soft-disable behavior
- H3 `docs/handbook-sysid-scoring.md` — aligned bags, fixed physical scales, and Sinkhorn scoring contract
- H5 `docs/handbook-youngs-cma.md` — support-\(k_p\) × spur/stem-\(E\) phenotype, grid, CMA loop, and gates
- H1 `docs/handbook-coupled-simulation.md` — settle→weld builders and geometry/frame ownership
- Archived implementation records remain under `docs/superpowers/specs/`; use them for decision history, not as competing living contracts.

**Shipped wins relevant to this slice (do not reimplement):**

- Bit-1 native pre/post parity convert; bit-2 format export + open-loop FR3 placement
- `vic_pose` controller + aniso wrench kernel + soft-disable pose hold
- V.5.1–V.5.2 sim-sim CMA on twist `vic` datasets (support-\(k_p\) phenotype)

**Deferred / later (not Current focus):** **V.5.3** held-out sim-sim validation; optional CLI `--score-mmd`; V.4.2.1 infer-only fidelity floor; gym/MMD migration to `vic_pose`.

**Known issues — debug next:**

- [x] **Post-grasp apple orientation vs GT** — Fixed 2026-08-07 (pre-grasp quat seed).
- [x] **Real parquet `action` is pose-control wrench** — Fixed 2026-08-10 (`vic_pose` pack + controller).
- [ ] **Real CMA native crash (exit 139)** — Undiagnosed. With local
  `population_size=6`, `max_generations=4` on `s09-d00`, the process exited
  **139** while starting generation 3 after two completed generations
  (`eligible_mean` `19.46 → 18.23`). Root cause not established.

**Goal:** **M4.0** real CMA on `robot_replay` → (optional return to) **V.5.3** held-out sim-sim → broader **[M4]** collection → **[M5]**.

**Also build on (milestones):**

- [M1] `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, VIC joint torques (H1 `docs/handbook-coupled-simulation.md`, H2 `docs/handbook-variable-impedance.md`)
- [M2.1] `apple_pick_gym/` observation contract v3
- [V.4.3] / [V.5.2] grid + support-\(k_p\) CMA / fused multi-structure replay

### Next up (ordered)

**[V].3 — sim API + gym migration** (parallel / lower priority)

- [x] **V.3.1 — Extract batched heterogeneous sim API:** `BatchedHeterogeneousCoupledSim` + config dataclass in `apple_pick_sim/coupled_fruiting/`. Surface: `build()`, `step(per_env_actions)`, `gather_obs()`, layout/scene accessors. No `argparse` / viewer in library; tests import library, not `examples/`.
- [x] **V.3.2 — Thin heterogeneous example:** `example_batched_heterogeneous_coupled_sim.py` (canonical entry point); `test_heterogeneous_coupled_fruiting.py` imports library, not monolith example.
- [x] **V.3.3 — Batched gym base env:** `ApplePickBatchedBaseEnv`, `ApplePickBatchedVicEnv`; `num_envs` + `gather_batched_obs`; episode snapshot reset; `example_batched_gym_keyboard.py`.
- [x] **V.3.4 — Migrate gym envs:** `ApplePickCoupledEnv`, `ApplePickVicEnv`, `ApplePickSysIdEnv`, `ApplePickReplayEnv` on batched backend; retire single-world gym build path.
- [x] **V.3.5 — Migrate gym examples:** `example_gym_sysid.py`, `example_gym_replay.py`, `example_gym_keyboard.py` + parity tests at `num_envs=1`.

**[V].4 — batched sys-ID** (backend: V.3.1 API + V.3.3 batched gym)

- [x] **V.4.1 — Recorded-action replay:** `replay_batched_sysid_structure` drives recorded EE actions on candidate stiffnesses. (No `gather_transitions()` API symbol; transition bags live in MMD/Wasserstein feature code — backlog if a public gather API is needed.)
- [x] **V.4.2 — Parallel GT collection:** `ApplePickBatchedSysIdEnv`, `example_batched_collect_sysid_data.py`, `batched_sysid_v1` Parquet layout (`docs/handbook-sysid-scoring.md`).
- [x] **V.4.2.1 — Digital-twin replay verification:** helpers + CLI `--infer-params` exist (`batched_digital_twin_init.py`); infer-only fidelity floor left as optional cleanup (oracle default OK for the current CMA-ES path).
- [x] **V.4.3 — In-process batched grid:** `example_batched_sysid_mmd_grid.py` + `batched_sysid_mmd_grid.py` (MSE / Sinkhorn Wasserstein + viz). Library MMD remains; CLI `--score-mmd` is later cleanup (Wasserstein is the ranking path). Legacy single-env: `run_system_identification.py`.
- [x] **V.4.4 — Sys-ID tooling at batch scale** (gate/collect scripts + gate report; further dashboard polish optional)

**[V].5 — sim-sim transfer wrap-up**

- [x] **V.5.1 — Harden loss calculation in `example_batched_sysid_mmd_grid.py`:**
  - [x] Soft-disable + manifest `excluded` / offline exclude (unstable-frame fraction > 0.25) + batched scene/monitor stability caps 40 N / 10 N·m
  - [x] Documented transition-feature / Sinkhorn scoring contract (`docs/handbook-sysid-scoring.md`) + named gate CLI (`scripts/gate_sysid_gt_sinkhorn.sh`; default `gate_pooled_dirs`, also `gate_median_hold` / `gate_hold_id`)
  - [x] GT preference is established on healthy samples; the operational gate uses a strict majority per seed and preserves bad-sampling misses for diagnosis
  - [x] Primary scorer is **Wasserstein** (Sinkhorn); optional CLI `--score-mmd` deferred as cleanup (library MMD already exists)
- [x] **V.5.2 — CMA-ES calibration loop** (M3.2) — **Done**
  - [x] `YoungsModulusCandidate` + keyboard E-grid + dataset-driven replay/ranking overlay
  - [x] Complete pooled scoring + physical-direction diagnostics + strict-majority multi-seed ranking gate
  - [x] Fused multi-structure replay implementation (clean performance/low-cap acceptance pending)
  - [x] Separate pycma ask/tell CLI + explicit final-mean evaluation + aggregate fit report + integrity gate
  - [x] Focused/full test suites, CLI checks, and CUDA acceptance (5 structures × 5 directions; report optimized vs GT, evaluated-history min/max, final covariance)
- [ ] **V.5.3 — Held-out sim-sim validation** + [M4] handoff criteria — **Deferred** (not Current focus; resume after M4.0 or when directed)

**[M4] real-data calibration**

- [ ] **M4.0 — Real `robot_replay` → CMA-ES (`vic_pose`)** — **In progress** (plumbing + holdout pipeline shipped; Task 9 GPU science gate **failed** on val torque magnitude)
- [ ] **M4.1+** — Broader real collection / held-out real segments (after M4.0)

**[M3] parallel infra** (optional alongside [V])

- [x] **M3.0.4 — Digital-twin fixture catalog:** `digital_twin_fixture_catalog.json`, `digital_twin_obs_straight_rod_initial.json`, and catalog tests shipped.
- [ ] **M3.0.5 — §2.2–2.3 excitation trajectories** (log chirp, torsional) — wire through V.4 when added

---

## Milestones (summary)

### [P0] · [M1] · [M2]

See **Sequencing** table. M2 deferred slices: M2.0 (interface ADR), M2.2a (`ApplePickFID-v0`), M2.2c (SKRL smoke), M2.3 (π_exp training).

Key M1 docs: H1 `docs/handbook-coupled-simulation.md`, H2 `docs/handbook-variable-impedance.md`, `docs/WRENCH_READOUT.md`, `docs/gpu-coupling-optimization.md`.

### [M3] Infra

| Slice | Status | Notes |
| ----- | ------ | ----- |
| M3.0.1–M3.0.3 | Done | §2.1 trajectory, Parquet recording, observation-only replay |
| M3.1.1 | Done | Legacy single-env MMD grid — superseded by V.4.3 |
| M3.0.4 | Done | Fixture catalog + example obs JSON committed; `test_digital_twin.py` passes (`docs/digital-twin.md`) |
| M3.0.5 | Planned | §2.2–2.3 trajectories → V.4 backend |
| M3.1.2 | Done (→ V.5.1) | Harden loss calc in `example_batched_sysid_mmd_grid.py` |
| M3.2 | → V.5.2 Done | CMA-ES calibration verified |

### [V] Batched vectorization

Fixed topology per batch (`num_segments`, `omit`); per-env `FruitingSystemParams` vary. See H1 `docs/handbook-coupled-simulation.md`.

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
| **V.3.4** | Done | Migrate gym envs |
| **V.3.5** | Done | Migrate gym examples (`num_envs=1` parity) |
| **V.4.1** | Done | Recorded-action replay (`replay_batched_sysid_structure`) |
| **V.4.2** | Done | Parallel GT collection (`batched_sysid_v1`, `ApplePickBatchedSysIdEnv`) |
| **V.4.2.1** | Done | Infer-params helpers/`--infer-params` shipped; infer-only floor optional cleanup |
| **V.4.3** | Done | In-process batched MSE/Wasserstein grid + viz; library MMD present |
| **V.4.4** | Done | Gate/collect scripts + batch tooling; further dashboard polish optional |
| **V.5.1** | Done | GT #1 on good samples; Wasserstein primary; `--score-mmd` cleanup later |
| **V.5.2** | Done | CMA-ES loop verified (tests + CUDA 5×5 acceptance) |
| **V.5.3** | Deferred | Held-out sim-sim validation (after M4.0 or when directed) |

### [M4] Real-data calibration

| Slice | Status | Deliverable |
| ----- | ------ | ----------- |
| **M4.0** | **In progress** | Plumbing + holdout CMA shipped; Task 9 GPU science gate **failed** (val torque ~15–70× too small) |
| **M4.1+** | Later | Broader real collection / held-out real metrics |

Canonical entry point: `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py`. Public API: H1 `docs/handbook-coupled-simulation.md`.

**Consumers after M4.0:** resume **V.5.3** if needed; broader **[M4]** collection; **[M5]** pick policy.

> Former **[S] sim-sim transfer** is fully absorbed into **V.4** (grid diagnostic) and **V.5** (harden loss → CMA-ES). **M4.0** starts real-data calibration on existing `robot_replay/` logs.

---

## Backlog

- **[M2] remaining** — resume after V.5 if directed; batched gym backend shipped (V.3)
- **[V] deferred** — per-env geometry DR on reset without rebuild; `(N, act_dim)` policy tensor path; batched VIC as default gym controller; public `gather_transitions()` API if needed beyond feature bags
- Optional cleanup: CLI `--score-mmd`; V.4.2.1 infer-only fidelity floor on `batched_sysid_v1`
- **Scope changes** — additional manipulators/crops; triangle mesh import (P0 stays capsules)
- **V.5.3** held-out sim-sim validation — deferred behind **M4.0**
- **[M4.1+]** broader real-data pipeline, **[M5]** final pick policy — after M4.0

---

## Agent execution notes

> **Warning — woody self-collisions are filtered.** Default builds set `enable_self_collisions=False` (woody↔woody filtered; apple↔woody / proxy↔woody default on; ground unchanged). See H1 `docs/handbook-coupled-simulation.md` and `apple_pick_sim/fruiting_system/build.py::_apply_default_fruiting_collision_filters`.

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

# [V].5.2 Young's-modulus grid, complete scoring, fused replay, and ranking gate
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_batched_heterogeneous_build.py \
  apple_pick_sim/tests/test_batched_digital_twin_init.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_replay_export.py \
  apple_pick_gym/tests/test_youngs_modulus_overlay_viz.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_sysid_script.py -q

# [V].5.2 CMA-ES loop + integrity gate (Done; Task 8 verification passed 2026-07-17)
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_cmaes_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_cmaes_script.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_sysid_script.py -q

uv run python apple_pick_gym/batched_examples/example_batched_youngs_modulus_keyboard.py \
  --viewer null --max-steps 60 \
  --log10-e-primary 8.0,8.5 --log10-e-spur 7.5 --log10-e-stem 7.0
# Two-step collect → rank (GT support k_p + spur/stem E from episode metadata;
# primary E fixed; grid includes GT support k_p 1e4 when using variance fixture)
uv run --env-file pytest.env python \
  apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 --max-steps 200 \
  --output tmp/support_kp_sysid_dataset --overwrite
uv run --env-file pytest.env python \
  apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --viewer null --dataset tmp/support_kp_sysid_dataset \
  --output tmp/support_kp_grid \
  --support-kp-values 1e3,1e4,1e5 \
  --log10-e-spur 8.0,9.5,11.0 \
  --log10-e-stem 8.0,9.5,11.0 \
  --include-gt-candidate --overwrite

# Separate CMA-ES fit (collect → fit → gates): see README.md
# "CMA-ES sim-to-sim transfer (support k_p + spur/stem E)" and
# docs/handbook-youngs-cma.md
# uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
#   --viewer null --dataset tmp/support_kp_sysid_dataset \
#   --output tmp/support_kp_cmaes_fit --overwrite
# Full Young's multi-seed ranking gate (expensive; defaults to 3 seeds x 5 structures x 5 directions):
# bash scripts/gate_youngs_modulus_sysid.sh
# Full Young's multi-seed CMA integrity gate (expensive; same default collect size; no GT-error threshold):
# bash scripts/gate_youngs_modulus_cmaes.sh
# CUDA acceptance (Task 8 passed): collect 5x5, fused CMA-ES, scalar smoke, and
# validation reports under tmp/task8_cuda_acceptance/ (see implementation notes).

# [M4].0 — real robot_replay → vic_pose replay → 1×8 holdout CMA
# Folder convert + per-dir replay + opt-in holdout CLI shipped (Tasks 1–8).
# Task 9 ran 2026-08-17 at shipped knobs (pop=15, gen=10, ~32 min RTX 4090).
# Plumbing OK; science gate FAILED on val torque magnitude (see checklist).
# Requires robot_replay/new_data/s09/ compiled s09-dNN.parquet (not always in clone).
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input-dir robot_replay/new_data/s09 \
  --dataset-out tmp/real_batched_s09 --overwrite
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09 \
  --output tmp/real_kp_e_cmaes_s09_holdout \
  --direction-split-seed 17 \
  --viewer null \
  --overwrite
# Acceptance (Task 9 recorded): manifest 1×8, control_hz=30, n_holds=4;
# cmaes_report.json command_status completed; gt null; spur/stem floor log10 E=7;
# holdout_report.json seed 17, train {2,4,5,6,7}, val {0,1,3}; every generation ⊆ train.
# Sinkhorn: train eligible_mean 22.54 → 17.08; val 23.63 → 17.13.
# Phenotype log10 fitted [2.265, 9.120, 10.994]. TCP mag+trend pass.
# FAIL: torque_ratio 0.073 / 0.014 / 0.044 on val dirs 0/1/3 (force |F_|| | in [1/3, 3]).
# Shipped CMA_SEARCH_PARAMS: population_size=15, max_generations=10. No CUDA 139 on this run.

# 1×1 plumbing smoke (still valid; ranking not trusted):
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00.parquet \
  --dataset-out /tmp/real_batched_s02_d00 --overwrite
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 --viewer null --max-frames 24 \
  --settle-substeps 80 --post-grasp-settle-substeps 0
# Grid opt-in (auto-detects vic_pose_v1; --include-gt-candidate forced off on real data):
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --dataset /tmp/real_batched_s02_d00 \
  --output /tmp/real_kp_e_grid \
  --viewer null \
  --support-kp-values 1e3,1e4 \
  --log10-e-spur 9.0 \
  --log10-e-stem 9.0 \
  --no-include-gt-candidate \
  --overwrite
# Post-alignment success = build/replay without crash; ranking trusted after smoke on aligned bags.
# CMA (slice 4, 1×1 wiring): plumbing/fit-loop smoke; ranking quality still ROADMAP-owned.
# Shipped CMA_SEARCH_PARAMS: population_size=15, max_generations=10 (~hours on RTX 4090).
# That full run has NOT been executed in verification. Local smoke: temporarily set
# population_size=4, max_generations=3 in example_youngs_modulus_cmaes.py, restore before commit.
# Verified reduced run (tmp/real_kp_e_cmaes_s09_d00_retry): eligible_mean 18.85 → 17.99 → 13.75.
# pop=6 / max_generations=4 crashed exit 139 starting gen 3 (eligible_mean 19.46 → 18.23); undiagnosed.
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_cmaes_s09_d00 \
  --viewer null \
  --overwrite
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  apple_pick_gym/tests/test_holdout_evaluation.py \
  apple_pick_sim/tests/test_holdout_gates.py \
  -q -p no:launch_testing
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py \
  apple_pick_gym/tests/test_real_batched_replay_cli.py \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py \
  robot_replay/tests/test_pack_vic_pose_actions.py -q

# Optional Sinkhorn gate wrapper (not a full slow e2e; needs GPU + long runtime)
# Default GATE=gate_pooled_dirs (matches CLI pool/hold-id defaults):
# bash scripts/gate_sysid_gt_sinkhorn.sh
# bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_median_hold
# bash scripts/gate_sysid_gt_sinkhorn.sh --gate gate_hold_id
```

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict
- Policy or product decisions (scope, user-visible behavior) — especially real-vs-sim GT scoring for M4.0
- Network credentials, paid APIs, or destructive operations

**When unsupervised is expected:**

- Complete the **next unchecked slice** in Current focus using TDD and project rules
- Fix small blockers uncovered by that slice (tests, imports, typos)
- Do **not** start a new milestone or backlog item without maintainer direction
- Do **not** migrate gym collect / MMD / default sim-sim CMA off twist `vic` unless this slice’s checklist explicitly requires an opt-in path
