# Roadmap

## Document status

| Field            | Value |
| ---------------- | ----- |
| **Last updated** | 2026-07-02 |
| **Owner**        | Abhinav |
| **Vision**       | See `docs/VISION.md` |
| **Active work**  | **[V].3** — batched heterogeneous sim API + gym migration (V.1–V.2 done) |

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
| **[M3]** | Infra done | Sys-ID recording, replay, subprocess MMD grid (`docs/system_identification.md`); batched paths → **V.4** / **V.5** |
| **[V].1–2** | Done | Batched `replicate(N)`, heterogeneous per-env DR, fixtures, runtime actions (`docs/vectorized-coupled-fruiting.md`) |
| **[V].3** | **Now** | Sim API extraction + `apple_pick_gym` on batched heterogeneous backend |
| **[V].4** | Next | Batched sys-ID — `gather_transitions()`, parallel collection, in-process MMD |
| **[V].5** | Next | Sim-sim transfer wrap-up — CEM, held-out validation (absorbs former **[S]** / M3.2) |
| **[M4]** | Later | Real-data collection — after **V.5** |
| **[M5]** | Later | Final pick policy |

---

## Current focus

**Next slice:** **V.3.1** — extract `BatchedHeterogeneousCoupledSim` from `example_batched_heterogeneous_coupled_fruiting.py` into `apple_pick_sim/coupled_fruiting/`.

**Goal:** Finish **[V].3** (library API + gym migration) → **[V].4** (batched sys-ID) → **[V].5** (CEM + held-out sim-sim validation). Closes the no-field-data calibration loop before **[M4]**.

**Specs:** `docs/vectorized-coupled-fruiting.md`, `docs/material-parameter-sampling.md`, `docs/system_identification.md`, `docs/digital-twin.md`

**Build on (do not reimplement):**

- [M1] `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`, VIC joint torques (`docs/variable-impedance-teleop.md`)
- [M2.1] `apple_pick_gym/` observation contract v3 (`docs/gym-observation-contract.md`); θ packing reused in V.4 / V.5

### Next up (ordered)

**[V].3 — sim API + gym migration**

- [ ] **V.3.1 — Extract batched heterogeneous sim API:** `BatchedHeterogeneousCoupledSim` + config dataclass in `apple_pick_sim/coupled_fruiting/`. Surface: `build()`, `step(per_env_actions)`, `gather_obs()`, layout/scene accessors. No `argparse` / viewer in library; tests import library, not `examples/`.
- [ ] **V.3.2 — Thin heterogeneous example:** `example_batched_heterogeneous_coupled_fruiting.py` → CLI + viewer wrapper; port `test_heterogeneous_coupled_fruiting.py` off example imports.
- [ ] **V.3.3 — Batched gym base env:** `ApplePickBatchedBaseEnv`; `num_envs` + `gather_batched_obs`; `num_envs=1` obs v3 parity.
- [ ] **V.3.4 — Migrate gym envs:** `ApplePickCoupledEnv`, `ApplePickVicEnv`, `ApplePickSysIdEnv`, `ApplePickReplayEnv` on batched backend; retire single-world gym build path.
- [ ] **V.3.5 — Migrate gym examples:** `example_gym_sysid.py`, `example_gym_replay.py`, `example_gym_keyboard.py` + parity tests at `num_envs=1`.

**[V].4 — batched sys-ID** (backend: V.3.1 API)

- [ ] **V.4.1 — `gather_transitions()` + recorded-action replay**
- [ ] **V.4.2 — Parallel GT collection** (`num_envs > 1` pull directions / θ seeds)
- [ ] **V.4.3 — In-process batched MMD** (replaces `run_system_identification.py` subprocess grid)
- [ ] **V.4.4 — Sys-ID tooling at batch scale** (replay, dashboard; deprecate subprocess patterns)

**[V].5 — sim-sim transfer wrap-up**

- [ ] **V.5.1 — MMD feature pipeline + objective contract** (M3.1.2 gaps)
- [ ] **V.5.2 — CEM calibration loop** (M3.2)
- [ ] **V.5.3 — Held-out sim-sim validation** + [M4] handoff criteria

**[M3] parallel infra** (optional alongside [V])

- [ ] **M3.0.4 — Digital-twin geometry reconstruction:** geometry-from-obs code is implemented, but the named fixture catalog (`apple_pick_sim/fixtures/digital_twin_fixture_catalog.json`) and its example obs JSON are **not committed** — `test_digital_twin.py` currently has 2 failing tests. See `docs/digital-twin.md` ("Known gap") before assuming this is done.
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
| M3.1.1 | Done | Subprocess MMD grid — superseded by V.4.3 |
| M3.0.4 | Planned; blocked on missing fixture data | Field-twin tooling; catalog JSON + example obs file not yet committed (`docs/digital-twin.md`) |
| M3.0.5 | Planned | §2.2–2.3 trajectories → V.4 backend |
| M3.1.2 | → V.5.1 | MMD feature pipeline |
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
| **V.3.1** | **Next** | `BatchedHeterogeneousCoupledSim` library API |
| **V.3.2** | Planned | Thin heterogeneous example |
| **V.3.3** | Planned | `ApplePickBatchedBaseEnv` |
| **V.3.4** | Planned | Migrate gym envs |
| **V.3.5** | Planned | Migrate gym examples (`num_envs=1` parity) |
| **V.4.1** | Planned | `gather_transitions()` + recorded replay |
| **V.4.2** | Planned | Parallel GT sys-ID collection |
| **V.4.3** | Planned | In-process batched MMD diagnostic |
| **V.4.4** | Planned | Sys-ID replay/dashboard at batch scale |
| **V.5.1** | Planned | MMD feature contract for CEM |
| **V.5.2** | Planned | CEM θ loop |
| **V.5.3** | Planned | Held-out validation; [M4] handoff |

Reference implementation (pre–V.3.1): `apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py`.

**Consumers after V.5:** [M4] real-data validation; M2.3 / M2.2c parallel RL envs.

> Former **[S] sim-sim transfer** is fully absorbed into **V.4** (MMD diagnostic) and **V.5** (CEM + validation).

---

## Backlog

- **[M2] remaining** — resume after V.5 or V.3.5 if directed; batched backend needs V.3.3–V.3.5
- **[V] deferred** — per-env geometry DR on reset without rebuild; `(N, act_dim)` policy tensor path; batched VIC as default gym controller
- **Scope changes** — additional manipulators/crops; triangle mesh import (P0 stays capsules)
- **[M4]** real-data pipeline, **[M5]** final pick policy — after V.5

---

## Agent execution notes

> **Warning — intra-cable collisions are disabled.** Default builds set `enable_self_collisions=False` (woody↔woody, woody↔apple, stem↔woody, apple↔proxy filtered; ground only). See `docs/vectorized-coupled-fruiting.md` and `apple_pick_sim/fruiting_system/build.py::_apply_default_fruiting_collision_filters`.

**Repository layout:**

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/` | Simulation (`fruiting_system/`, `coupled_fruiting/`, `examples/`, tests) |
| `apple_pick_sim/coupled_fruiting/` | Coupled sim; **V.3.1** adds `BatchedHeterogeneousCoupledSim` |
| `apple_pick_gym/` | Gymnasium adapter; depends on `apple_pick_sim`, not vice versa |
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

# [V].2 heterogeneous batched coupled (current reference; V.3.1 extracts API from here)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42

# [V].1 homogeneous batched smoke
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_vectorized_coupled_fruiting.py -q
uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \
  --viewer null --num-frames 500 --num-envs 4 --fix-to-apple --controller direct --seed 42

# Material-parameter sampling (V.2.1.3)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py -q \
  -k "material or youngs or damping_ratio or sample_params"

# M3 sys-ID (legacy single-env path; V.3.5 / V.4 migrate to batched backend)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_gym/tests/test_sysid_env.py -q
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 --output /tmp/apple_pick_sysid_gt

# M3.1.1 subprocess MMD grid (superseded by V.4.3)
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/apple_pick_sysid_gt --viewer null \
  --primary-bend-stiffness-values 10,25 \
  --secondary-bend-stiffness-values 10,25 \
  --spur-bend-stiffness-values 10,25 \
  --stem-bend-stiffness-values 10,25

# Digital-twin fixture catalog
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q
```

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict
- Policy or product decisions (scope, user-visible behavior)
- Network credentials, paid APIs, or destructive operations

**When unsupervised is expected:**

- Complete the **next unchecked slice** in Current focus using TDD and project rules
- Fix small blockers uncovered by that slice (tests, imports, typos)
- Do **not** start a new milestone or backlog item without maintainer direction
