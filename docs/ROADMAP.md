# Roadmap

<!--
  HOW TO USE THIS FILE (human)
  - Exactly one place answers "what are we doing now?" — see "Current focus". Update it when priority shifts.
  - Milestones are ordered. Statuses: Planned | In progress | Done | Deferred | Dropped (one-line reason if Dropped).
  - Under each milestone, keep "Next actions" to small slices (ideally testable in one PR/session).

  HOW AGENTS SHOULD USE THIS FILE
  - Before work: read "Current focus" and the active milestone’s Definition of done and Constraints.
  - Pick tasks from "Next actions" in order unless blocked; if blocked, note the blocker and smallest unblock step.
  - After a slice: update checkboxes, add PR/commit links if useful, adjust "Current focus" if the milestone changed.
  - Do not start Deferred or Dropped items unless the maintainer moves them back to active.
-->

## Document status

| Field | Value |
|--------|--------|
| **Last updated** | 2026-05-25 (**[M1] Done**; **[M2] active** — **Gymnasium** env first, then FIM / [ASID](https://openreview.net/forum?id=jNR6s6OSBT). **[P0] Done**.) |
| **Owner** | Abhinav |
| **Vision** | See `docs/VISION.md` |

---

## How this roadmap is structured

| Section | Purpose |
|---------|---------|
| **Current focus** | Single source of truth for what to do *now*. Keep it short. |
| **Milestones** | Phased outcomes from vision to implementation. Each has a clear “done”. |
| **Backlog** | Unordered or lower-priority ideas — not active work. |
| **Agent execution notes** | How to run tests, where code lives, when to stop and ask. |

---

## Sequencing (how this file maps to `docs/VISION.md`)

Owner intent drives phase order (vision outcomes stay valid; **order** is explicit here):

1. **Done — Outcome 1 ([P0]):** **Variational geometry** and **joint-level force readouts** shipped; refactor (collision/readout API, ``measure_fruiting_forces``, solver damping, docs) landed. Optional P0 stretch (1.a floating EE, force-rises-with-load, richer cable scalars) **deferred**. See `docs/VISION.md` *Procedural fruiting variance*; archive under [P0].
2. **Done — Outcome 2 (manipulation stack, [M1]):** Two-`Model` **`SolverMuJoCo` + `SolverVBD`** coupling, FR3 + TCP teleop, proxy wrench exchange, structural layout (`fruiting_system/`, `coupled_fruiting/`, GPU hot path). Maintainer exit 2026-05-25; deferred stretch (EE–apple contact scenarios, formal arm readouts API) documented under [M1]. See [M1].
3. **Now — Outcome 3 (learning, [M2]):** **[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)** env over the [M1] coupled sim (**first deliverable**), then **Fisher-information** exploration per [ASID](https://openreview.net/forum?id=jNR6s6OSBT). **Dict** observations for multiple sensors; **parity tests** so canonical rollouts match **direct** `apple_pick_sim` stepping (existing M1 tests stay on the direct path). **Design constraint for M3:** log \(\theta\) and reserve **gradient / sensitivity hooks** on sim parameters even before tuning lands.
4. **Next — Outcome 5 (sim tuning, [M3]):** **Simulation parameter identification / calibration** using trajectories from \(\pi_{\mathrm{exp}}\) (and later real logs in [M4]); update \(\theta\) with documented metrics and Newton-side sensitivities where available.

Later vision phases (real-data collection [M4], final pick policy [M5]) follow once M2–M3 contracts exist.

---

## Current focus

**Active milestone:** [M2] — **RL infrastructure & Fisher-information exploration** (Gymnasium + ASID).

**In one sentence, the goal right now:** Ship a **[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)** `Env` with **`Dict` observations** (placeholder/dummy for now) and **keyboard-command actions** that drive the FR3 via the direct-joint controller — with **parity validation** against direct `CoupledFruitingScene` stepping — then add **Fisher-information** exploration ([ASID](https://openreview.net/forum?id=jNR6s6OSBT)) on top of that contract.

**Build on (do not reimplement):** [M1] stack — `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `Fr3EEVelocityController` / `EEVelocity`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`. Architecture: **`docs/mujoco-vbd-coupling-architecture.md`**. **RL API:** [Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/) (`reset` → `step` until `terminated` \| `truncated`). **Later (ASID):** [WEIRDLabUW/asid](https://github.com/WEIRDLabUW/asid) for FIM objective patterns.

**Next up (ordered — [M2]):**

    1. [ ] **M2.1 — Gymnasium environment (first):** Separate package (e.g. `apple_pick_gym/`) implementing `gymnasium.Env` using **public `apple_pick_sim` APIs only** — no coupling physics in the Gym package. Headless default (`render_mode=None`). Optional thin `apple_pick_sim` runtime facade (build / step / observe); **no `gymnasium` import** inside `apple_pick_sim/`.
    2. [ ] **M2.1a — Minimal observation contract (placeholder):** assume “no observations yet” by using a `Dict` observation space with a dummy field (e.g. `{"dummy": Box(shape=(1,), dtype=float32)}`), returning zeros for now. The observation schema must still be versioned so adding sensors later (TCP pose, joints, fruiting wrenches, …) is an additive change.
    3. [ ] **M2.1b — Keyboard-command action contract (direct drive):** actions are **keyboard-style commands** (no viewer, no polling). Start with a simple single-command-per-step interface:
       - `action_space = Discrete(13)` mapping to FR3 TCP velocity axes: `{+X,-X,+Y,-Y,+Z,-Z,+rotX,-rotX,+rotY,-rotY,+rotZ,-rotZ,noop}`.
       - Map each action to an `EEVelocity` using the existing FR3 keyboard speed constants, then use `Fr3EEDirectJointController` + `CoupledFruitingScene.apply_fr3_ee_teleop_direct` (with `robot_kinematic_mode=True`) to apply it.
       - Later in M2.0/M2.1d (after parity): extend to MultiDiscrete/MultiBinary for simultaneous keys if needed.
    4. [ ] **M2.1c — Parity validation:** For canonical scenarios (fixed `seed`, ranges fixture, action schedule, substep count), assert **direct** sim stepping matches **Gym** `reset`/`step` on agreed metrics (poses, joint q, wrench norms, `params_fingerprint`). **All existing `apple_pick_sim/tests/` remain direct-path regressions and must stay green**; new tests prove the adapter does not drift (not “re-run every M1 test through `gym.make`”).
    5. [ ] **M2.1d — Gym contract smoke:** `gymnasium.utils.env_checker.check_env` (short horizon, CPU); skip FR3 when assets missing (same spirit as `requires_fr3`).
6. [ ] **M2.0 — Interface ADR:** Lock the action/step timing contract (control frame vs. `SUBSTEPS_PER_FRAME` substeps, action mapping to `EEVelocity` and `apply_fr3_ee_teleop_direct`), then document the \(\theta\) vector and the sensor list expansion plan — feeds M2.2 FIM.
7. [ ] **M2.2 — FIM / exploration objective:** ASID-style Fisher information (or \(\mathrm{tr}(\mathrm{I}^{-1})\) proxy) from sim rollouts; \(\theta\) randomized per episode.
8. [ ] **M2.3 — Train \(\pi_{\mathrm{exp}}\):** Minimal RL loop on registered env id; training smoke (fixed seed, few steps).

**Explicitly not in M2:** Full sim-parameter optimizer ([M3]); real-robot deployment ([M4]); final pick policy ([M5]). **Do not** require Gym to execute existing M1 unit tests — require **behavioral parity** on canonical rollouts instead.

**Blockers (if any):** None for M2.1 skeleton; **M2.0** ADR can land in parallel with M2.1a once dict keys are fixed.

**Last completed milestone:** [M1] — maintainer exit **2026-05-25** (coupling, FR3 teleop, refactor layout, GPU hot path, hardening docs).

---

## Milestones

Phases below follow **Sequencing** at the top: **[P0] Done** → **[M1] Done** → **[M2] active** (Gymnasium env + FIM / ASID) → **[M3]** (sim tuning) → **[M4]** (real data) → **[M5]** (final policy).

### [P0] — Variational fruiting system generation & force telemetry

- **Status:** Done (exited 2026-05-15; optional stretch deferred to backlog / [M1] needs)
- **Links:** N/A
- **Vision:** Outcome 1 (*Visual and structural variance*); success row *Procedural fruiting variance*; plus **instrumented loads** on the same VBD-built scene.

**Objective (variance — done):** From a **JSON** file of **min/max** bounds on geometry and material-like parameters, plus a **seed**, **variationally generate** a **Newton-ready scene** for a **fixed topology**: **primary branch (stiff)** → **secondary branch (softer)** → **short spur** → **stem** → **apple**, built like **`example_apple_stem.py`** (polylines + **`add_rod`** capsule chains + primitive fruit body, **`SolverVBD`**). Run a short reproducible simulation and prove variance and determinism with tests.

**Objective (force telemetry — core shipped):** **Fixed-joint** constraint wrenches via **`newton.solvers.SolverVBD.gather_joint_wrench_child_com`** (child at COM, world frame; Newton implementation under `newton/newton/_src/solvers/vbd/`) are wrapped in **`apple_pick_sim/vbd_fixed_joint_wrenches.py`**, re-exported from **`fruiting_system.py`**, exercised in **pytest**, and visualized as **FJ** plots in **`example_fruiting_system.py`**. Per-cable-joint **penalty-style** metrics remain available through the same **`ModelBuilder` + `SolverVBD`** patterns as `example_apple_stem.py` (`get_forces()`-style joint displacement × stiffness where used). **Structured readout:** ``measure_fruiting_forces`` returns fixed-joint records plus ``cable_joint_indices`` (cable scalars still via stem example patterns). **Deferred (not required for P0 exit):** optional stretch (**1.a**, **force-rises-with-load**, richer cable scalars in ``measure_*``); maintainer may still run viewer smoke ad hoc.

**Definition of done (checklist):**

**Variance (complete):**

- [x] Documented **JSON** range format and **seed** semantics (same file + seed → same instance). See `apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json` (tests) and `fruiting_system_ranges_example_variance.json` (example default).
- [x] Generator implements the **five-part topology** above; primary vs. secondary **stiffness ordering** is respected in exported sim parameters (enforced in `sample_params`).
- [x] **Per-instance Newton model** is produced via **`ModelBuilder`** (primitive-based geometry as above); `FruitingSystemScene` dataclass carries model, solver, state, body indices.
- [x] **Newton rollout** with **`SolverVBD`** from the generated scene runs headlessly with deterministic settings for tests (`run_rollout`).
- [x] Pytest(s): fixed `(range_json, seed)` → stable summary or fingerprint; varying seed or bounds → **distinct** geometry or labels (as asserted). Fruiting tests green (see `apple_pick_sim/tests/test_fruiting_system.py`).
- [x] Documented command(s) for generate and generate+sim (see README and Agent execution notes).

**Force telemetry — core (complete):**

- [x] **Fixed-joint wrenches:** `fixed_joint_wrenches_child_com_vbd` / `iter_fixed_joint_indices` and **`SolverVBD.gather_joint_wrench_child_com`** wired for fruiting-style **FIXED** joints; viewer logging in `example_fruiting_system.py`.
- [x] **Pytest:** e.g. finite wrenches after substep, fixed-joint index iteration (`test_fruiting_system.py`).

**Exit before [M1] (process):**

- [x] **Refactor + alignment (code shipped)** — collision/readout API, ``measure_fruiting_forces``, README / WRENCH / tests; exited to [M1] 2026-05-15.
- [ ] **Manual verification** — *Deferred* (not a gate for [M1]); ad hoc: `example_fruiting_system.py` (seeds, `--no-self-collision`, **FJ** traces).

**Optional stretch (promote during refactor only if needed):**

- [ ] **1.a Floating-EE handle** — opt-in free-floating body fused to apple via `add_joint_fixed` on the same VBD `Model` (no FR3).
- [ ] **`apply_ee_wrench` + acceleration pytest** on 1.a (see `newton/newton/tests/test_body_force.py` spirit).
- [ ] **`measure_fruiting_forces` (or equivalent)** — structured dict for cable joints + fixed joints; document penalty vs. gather semantics for [M4]. *(Fixed joints + cable **indices** shipped in `measure_fruiting_forces`; cable **scalar** forces still follow `example_apple_stem.py`.)*
- [ ] **Force-rises-with-load test** — monotonic load on apple; assert ordering vs. segment stiffness intent.
- [ ] **New `uv run …` paths** — only if the refactor adds entrypoints; verify per `.cursor/rules/readme-runtime-verification.mdc`.

**Constraints / notes for implementers:**

- Prefer `apple_pick_sim/`; **`newton/` edits** are justified when exposing or hardening **VBD joint-wrench** APIs (e.g. `gather_joint_wrench_child_com`) — keep PRs **focused** and covered by `newton` tests (see `newton/newton/tests/test_solver_vbd.py` for `gather_joint_wrench_child_com`).
- TDD and `uv` conventions apply.
- Start with **scalar stiffness / thickness / length** parameters if full anisotropic materials are not yet exposed in the chosen Newton API; document the mapping from JSON fields to sim bodies.
- **Triangle mesh import/generation** (OBJ/STL, high-res render meshes) is **out of scope for P0** unless promoted from **Backlog**; P0 “geometry” means the **capsule/sphere + cable joint** representation consistent with `example_apple_stem.py`.

**Next actions:** *(none — milestone complete; work continues under [M1])*

**Completed (archive):**

- [x] JSON schema + fixtures (`apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json`, `fruiting_system_ranges_example_variance.json`)
- [x] Generator module (`apple_pick_sim/fruiting_system.py`) with `load_ranges`, `sample_params`, `generate_scene`, `geometry_fingerprint`, `run_rollout`
- [x] pytest tests in `apple_pick_sim/tests/test_fruiting_system.py` — variance + fixed-joint wrench coverage
- [x] Example viewer + **FJ** fixed-joint wrench telemetry (`example_fruiting_system.py`)
- [x] Refactor before M1: collision pipeline, ``fruiting_fixed_joints``, ``measure_fruiting_forces``, ``make_fruiting_solver_vbd``, chain collision filters, docs

---

### [M1] — FR3 manipulation stack (two-`Model` coupling)

- **Status:** Done (exited 2026-05-25; maintainer sign-off; **[P0] Done** — reuse `fruiting_system` / wrench readouts; do not duplicate P0 DoD here)
- **Links:** Reference patterns: `newton/newton/examples/cloth/example_cloth_franka.py` (Featherstone+VBD coupling on one `Model` — *adjacent* but **not** the M1 recipe, see below), `newton/newton/examples/ik/example_ik_franka.py` (FR3 URDF import + TCP body), `newton/newton/tests/test_body_force.py` (external wrench API), and the **"two-`Model` staggered coupling skeleton"** at the bottom of this section (authoritative pattern for M1).
- **Vision:** Outcome 2 (*Manipulation stack*). MuJoCo enters as **`SolverMuJoCo` inside Newton** — there is **no separate MuJoCo runtime** in this milestone.

**Objective:** Build on **[P0]** by integrating a **Franka FR3** with a **custom end-effector** (URDF/shape additions under `apple_pick_sim/` or documented assets), **rigid contact** between the **end-effector** and the **apple** in at least one validated scenario (implemented as stiff body–body contact on **`cable_model`** between the apple and a **geometry-equipped gripper proxy**, and/or on the MuJoCo robot side per design — **not** soft multi-finger grasping), and the **two-`Model` staggered coupling** (**`SolverMuJoCo`** + **`SolverVBD`**, proxy bodies + one-step lag). **Apply end-effector wrenches** on the FR3 for tests. Reuse **[P0] VBD readouts** (**fixed-joint** wrenches via **`gather_joint_wrench_child_com`** / `vbd_fixed_joint_wrenches.py`, plus cable-joint **penalty** proxies where used). **Add robot-arm force instrumentation** on the **coupled** stack (e.g. `body_f` / joint torques / agreed MuJoCo-side signals) so interaction loads are visible on **both** models. Deterministic tests for **proxy sync**, **`harvest_proxy_wrenches`**, coupling sanity, and the **EE–apple contact** load path. A **fixed** apple↔gripper-proxy joint may remain for **regression** tests alongside **contact** scenarios.

**Architecture (authoritative for M1):**

The FR3 (rigid articulated robot) and the rod backbone (cable joints) are integrated by **different solvers on two separate `Model`s**, coupled at a per-step boundary by **proxy bodies + one-step-lag staggered wrench exchange**. The motivation is concrete:

- `SolverMuJoCo` does **not support `JointType.CABLE`** (`newton/_src/solvers/mujoco/solver_mujoco.py:292`) → the rod backbone *cannot* live in the MuJoCo model.
- `SolverVBD` *can* simulate revolute chains but is not the natural fit for an articulated-robot tracking controller → the FR3 *should* live in the MuJoCo model.
- Newton's "two solvers on one `Model`" pattern (e.g. `example_cloth_franka.py`, which uses `SolverFeatherstone + SolverVBD`) is therefore inappropriate here; we use **two `Model`s** instead, mirroring the relevant robot body or bodies as **proxy rigid bodies** inside the VBD `Model`.

**Per-`Model` ownership (M1):**

| `Model` | Solver | Contents |
|---------|--------|----------|
| `robot_model` | `SolverMuJoCo` | FR3 articulation (**URDF** + **custom end-effector** geometry as needed). Apple body is **not** here. |
| `cable_model` | `SolverVBD` | Rod backbone (primary → … → stem) with cable joints, the apple rigid body, and **proxy** copies of the FR3 link(s) the cable must see (gripper / TCP) with **collision shapes** so **rigid EE–apple contact** can be simulated. **Fixed** `add_joint_fixed(gripper_proxy, apple)` may still be used for **baseline** scenarios; **contact** scenarios add/remove constraints per design. |

**FR3 URDF import** (one call, mirroring `example_ik_franka.py` and `example_cloth_franka.py`):

```python
robot_builder.add_urdf(
    newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
    floating=False,
    collapse_fixed_joints=True,
)
```

**Coupling protocol (per step, one-step lag):**

1. Apply the **lagged** VBD→MuJoCo wrench harvested from the previous step onto the robot body's `body_f` slot (cache it so step 3 can subtract it cleanly).
2. Advance the MuJoCo solver one step on `robot_model` / `robot_state_0`.
3. Mirror robot pose and velocity onto the proxy bodies inside the VBD `Model` *and* subtract the coupling-force + gravity contribution from the proxy velocity (otherwise VBD double-integrates the same wrench). The reference skeleton implements this in a single Warp kernel.
4. Advance the VBD solver one step on `cable_model` / `vbd_state_0` (rod + apple + **EE–apple and proxy** contacts/joints + any apple↔proxy **fixed** joint used for baselines).
5. **Harvest** the net wrench on each proxy body from the just-completed VBD step — read it from VBD's per-body force/torque accumulators if they're exposed (preferred), else reconstruct from the velocity delta (see Slice 4 spike below). Store as the new lagged wrench used at step 1 of the next iteration.

**Implemented harvest paths (Slice 2b, `apple_pick_sim/coupled_fruiting.py`):**

| Path | When | Harvest | Sync |
|------|------|---------|------|
| **Stem-harvest** | `GripperProxyConfig.fix_to_apple=True` (opt-in) | `harvest_stem_joint_wrench` — stem→apple FIXED joint via `gather_joint_wrench_child_com`; under-relax (`stem_coupling_gain`, force/torque caps) | `sync_proxy_and_apple_state` — co-teleport proxy + apple from TCP |
| **Velocity-delta (default)** | `fix_to_apple=False` (builder default since `GripperProxyConfig`) | `harvest_proxy_wrenches` — option 3 from Slice 4 spike | `sync_proxy_state` — proxy only |

See **`docs/mujoco-vbd-coupling-architecture.md`** for per-model ownership and the full substep diagram.

**Current progress (2026-05-22):**

- [x] Slice 1 — `proxy_coupling.py` (`sync_proxy_state`, velocity-delta harvest, `align_proxy_body_q_prev_for_vbd`); `test_proxy_coupling.py`.
- [x] Slice 2a — `generate_coupled_cable_scene` / `CoupledCableScene`; `test_coupled_cable_scene.py`.
- [x] Slice 2b — `CoupledFruitingScene` (`mujoco_substep` / `vbd_substep` / `coupled_substep`), placeholder + stem-harvest paths; `example_coupled_fruiting.py`; coupled/stability tests; `verify_coupling.py`.
- [x] Slice 2 — FR3 load (`testfr3_resolved.usda`, `build_coupled_fruiting_fr3`); `docs/fr3-usd-import-implementation.md`.
- [x] Slice 2c — TCP velocity teleop + ghost proxy tracking (`--fr3-keyboard`, `--only-mjc` and full coupled).
- [x] Slice 2d — **Force transfer accepted** (lagged harvest, apply to `body_f`, sync, debug plots).
- [x] **Slice 2f — Structural refactor** — `fruiting_system/`, `coupled_fruiting/`, `robot/fr3_robot/`; see **`docs/slice-2f-structural-refactor.md`**. Residual items in **`refactor.md`** are non-blocking.
- [x] **Slice 2g — GPU optimization** — `docs/gpu-coupling-optimization.md`, device hot path, `benchmark_coupling.py`.
- [x] Slice 2e — Hardening: `docs/slice-2e-hardening.md`, device hot-path kernels, `benchmark_coupling.py`, FR3 long-horizon + `slow` pytest marker.
- [ ] Slice 3 — README / Agent execution notes polish (optional; not a gate for M2).

**Known limitations (placeholder TCP and coupled FR3):**

- **Builders default to `fix_to_apple=False`** (velocity-delta harvest + `sync_proxy_state`). Opt in `GripperProxyConfig(fix_to_apple=True)` for stem-harvest / apple co-teleport regression tests.
- **`fix_to_apple=True` + placeholder free TCP** is often unstable (huge **QACC**, stem-harvest saturation) — see headless `test_coupling_stability` / `verify_coupling.py` notes.
- **`fix_to_apple=False` + velocity-delta harvest** stays ~mg–centi-Newton scale in headless checks — current default for FR3 keyboard smoke.
- **FR3 TCP velocity teleop + force transfer:** **Accepted** for M1 baseline (`--only-mjc` and full coupled, `--debug-coupling-forces` for wrench plots). Refactor must **preserve** staggered semantics unless explicitly changed with tests.
- **`--fr3-direct-joints`:** kinematic `joint_q` writeback for arm debugging; not the primary teleop path.
- **`test_coupling_stability.py`** asserts finiteness and cap compliance, **not** quiescent MuJoCo motion or small TCP velocities; passing tests does **not** imply a stable interactive demo.
- **`--no-self-collision` / `--mujoco-viewer`** are not the primary instability drivers; they only change cable collisions or add a second viewer window.
- **Smoke paths:** `--only-vbd` (cable only); `--robot fr3 --fr3-keyboard` (full coupled teleop); `--robot fr3 --only-mjc --fr3-keyboard` (robot + proxy sync only); `--debug-coupling-forces` for wrench plots.

**Exit note (2026-05-25):** Coupled FR3 + proxy coupling accepted for learning stack; EE–apple **contact** scenarios and formal **arm readouts API** deferred (not required for [M2] env wrapper).

**Slice 2f — definition of done (checklist):**

- [ ] **`refactor.md` tasks** implemented in agreed order; each slice keeps **`apple_pick_sim/tests/`** green and public imports stable (or documents a one-shot migration).
- [ ] **Coupling semantics preserved:** same staggered apply → MuJoCo → sync → VBD → harvest; no behavior change unless a refactor slice explicitly targets physics/API with new tests.
- [x] **Naming / layout:** `fruiting_system/` package, `coupled_fruiting/` package, `proxy_coupling` mirror/harvest kernel names — see **`docs/slice-2f-structural-refactor.md`**. (Host-sync removal is **Slice 2g**.)
- [ ] **Remaining 2f tasks** in **`refactor.md`** (2f-E and maintainer edits).

**Slice 2g — definition of done (checklist):** *(GPU path — after 2f layout stabilizes; may start profiling in parallel)*

- [ ] **Profiler harness documented:** agreed stack for this repo (at minimum: **`diagnostics/benchmark_coupling.py`** on CUDA with `wp.synchronize()`; optional **Nsight Systems** / **Nsight Compute** or Warp timing markers for kernel-level attribution). Commands live in **Agent execution notes** and a short **`docs/gpu-coupling-optimization.md`** (or a dated subsection in `docs/mujoco-vbd-coupling-architecture.md`).
- [ ] **Baseline captured on target GPU:** ms/substep (placeholder + FR3), default `sim_substeps`, warmup/bench substeps recorded in docs **before** optimization PRs land.
- [ ] **Hot-path inventory:** list host↔device syncs and CPU fallbacks per substep (`coupled_fruiting.py`, `proxy_coupling.py`, `fr3_robot.py` placement/teleop, collision triggers); prioritize by profiler cost.
- [ ] **GPU migration:** replace top-cost CPU paths with Warp kernels or device-resident arrays where Newton/Warp APIs allow; **no correctness regressions** — existing proxy/coupling/stability tests stay green; add tests when a kernel replaces logic that was only covered indirectly.
- [ ] **Measured progress:** each optimization PR cites profiler/benchmark **before → after** (same machine/GPU/driver note in doc); at least one **≥10%** ms/substep win on the coupled hot path **or** documented blocker (e.g. MuJoCo solver host-bound).
- [ ] **End-to-end device goal:** coupled substep runs with **minimal per-substep host reads** (debug/viewer paths may still `.numpy()`; pytest headless paths should not require full-state host copies unless asserting values).

**Slice 2e — definition of done (checklist):** *(hardening — after 2f / 2g per maintainer)*

- [x] **Code map / reader docs:** `docs/slice-2e-hardening.md`, `docs/mujoco-vbd-coupling-architecture.md` §5.2 (device align kernel).
- [x] **Coupling error-free (headless):** `test_fr3_coupled_substep_long_horizon_finite` (`slow`); existing stability/proxy tests green.
- [ ] **Controllers error-free (headless):** extend existing `test_fr3_*` coverage where gaps remain (deferred).
- [x] **Benchmarking:** `diagnostics/benchmark_coupling.py`; baseline table in `docs/slice-2e-hardening.md`; commands below.

Items the implementation must add on top of P0:

- `harvest_proxy_wrenches(vbd_solver, vbd_state, vbd_contacts, dt) -> wp.array(spatial_vector)` — not in Newton; lives under `apple_pick_sim/`. **Preferred implementation**: read the VBD-internal per-body force/torque accumulators populated during contact + constraint solves (see `newton/_src/solvers/vbd/rigid_vbd_kernels.py` — `accumulate_body_body_contacts_per_body`, `accumulate_body_particle_contacts_per_body`, and the `joint_lambda_lin` constraint Lagrangians). One readout per proxy, summed into a single spatial wrench. **Fixed-joint reactions on the VBD tree** (apple, handle, stem, etc.) use **[P0]’s Newton API path** — `SolverVBD.gather_joint_wrench_child_com` — not this harvester.
- A Warp kernel equivalent to `sync_proxy_state` in the reference skeleton (proxy mirror + force/gravity subtraction).
- A small registry that maps `robot_body_id → proxy_body_id` for the bodies the cable needs.

**Slice 4 spike — confirm VBD exposes a usable per-body wrench:**
The accumulators above are populated internally each step; the question Slice 4 must answer is whether they're (a) publicly readable through `state` / `solver` / `contacts`, (b) readable via a narrow public extension in `newton/`, or (c) only reconstructable. Order of preference for `harvest_proxy_wrenches`:

1. **VBD publishes the wrench.** Direct read; nothing else needed. M1 ships as designed.
2. **VBD does not publish but exposes accumulators.** Add a tiny public passthrough in `newton/` (focused PR, justified by the use case).
3. **VBD does not expose either.** Reconstruct from velocity delta: `harvest_f = body_mass * (v_proxy_post - v_proxy_synced) / dt - body_mass * gravity`. This works because `sync_proxy_state` already arranged for `v_proxy_synced` to absorb the lagged coupling force + gravity. The fixed-joint reaction is then implicit in the velocity change, so the contact + joint-reaction split is irrelevant at the API level.

The "joint reaction" split surfaces only inside option 3's derivation; **proxy** coupling still uses this single harvest channel. **Articulated-tree fixed joints** (apple↔handle, apple↔proxy, etc.) use **[P0]’s** `gather_joint_wrench_child_com` path separately.

**Reference `SolverMuJoCo` kwargs** (use as starting point; tune in tests):

```python
SolverMuJoCo(
    robot_model,
    solver="newton",
    integrator="implicitfast",
    cone="elliptic",
    iterations=20,
    ls_iterations=10,
    ls_parallel=True,
    impratio=1000.0,
)
```

**End-effector wrench input:** per substep, write the desired EE wrench (6-vec `[fx, fy, fz, tx, ty, tz]`, world frame at body CoM) into `robot_state_0.body_f[ee_body_index]` *before* step 1 of the coupling protocol; the MuJoCo solver consumes it, and the cable side sees only the resulting motion + reaction force via the proxy. Wrenches must be re-applied each substep because solvers clear forces (`test_body_force.py`).

**Force readout:** **[P0]** supplies **fixed-joint** wrenches via **`SolverVBD.gather_joint_wrench_child_com`** (`vbd_fixed_joint_wrenches.py`) and cable-joint **penalty** proxies where that convention holds. **M1 adds** **robot-side** readouts: document how to recover **joint torques**, **link wrenches**, or **`body_f`**-consistent totals on **`robot_model`** after each coupled substep so **forces on the arm** from apple interaction are **measured** alongside VBD-side reactions. Document penalty vs. gather semantics for [M3] calibration.

**Attachment / interaction matrix:**

| Mode | What it is | Solvers needed | When to use | Slice |
|------|-----------|----------------|-------------|-------|
| **1.a Floating-EE handle** | Single free-floating rigid body on `cable_model` fused to the apple via `add_joint_fixed`; wrenches on that body. **No FR3.** | `SolverVBD` only. | Optional **P0 stretch** for isolated load curves; **not** assumed to exist for M1 unless promoted during P0 refactor. | [P0] optional |
| **1.b Full FR3 + coupling** | Two `Model`s; FR3 hand mirrored as **proxy** in `cable_model`; staggered protocol below. **Rigid EE–apple contact** (and/or **fixed** apple↔proxy for regression). | `SolverMuJoCo` + `SolverVBD`. | Primary M1 manipulation scenario. | [M1] |

**Definition of done (checklist):**

- [ ] **Custom end-effector:** FR3 import extended with **project-specific EE geometry** (URDF fragment, extra shapes, or documented asset) and tested load path.
- [ ] **Rigid EE–apple contact:** At least one headless scenario where apple and EE (or **gripper proxy** on `cable_model`) interact through **rigid / stiff contact**; contact pipeline and filters documented.
- [ ] **Robot-arm force readouts:** Documented + tested extraction of **interaction forces/torques on the FR3 chain** from the **coupled** sim (agreed links / joints / `body_f` / MuJoCo diagnostics).
- [x] **Builder integration (robot half):** :func:`~apple_pick_sim.coupled_fruiting.build_coupled_fruiting_fr3` + :func:`~apple_pick_sim.fruiting_system.generate_coupled_cable_scene` produce **1.b** (`robot_model` + `cable_model`, FR3 + proxy + placement). **Remaining:** EE–apple contact scenarios and documented `(range_json, seed)` entry if not already covered by coupled example flags.
- [x] **Two-`Model` staggered step loop (placeholder):** :class:`~apple_pick_sim.coupled_fruiting.CoupledFruitingScene` + :meth:`~apple_pick_sim.coupled_fruiting.CoupledFruitingScene.coupled_substep` with placeholder TCP; `example_coupled_fruiting.py`; `test_coupled_fruiting_system.py`. **Remaining:** same loop on **FR3** with stability acceptance (Slice 2).
- [ ] **End-effector wrench on FR3:** Documented application of wrenches to the MuJoCo robot `body_f` (TCP / agreed link), once per substep, with tests that build on **[P0]** patterns where applicable.
- [ ] **Reuse [P0] telemetry on 1.b:** **[P0]** fixed-joint + cable-side readouts run on `cable_model` (including apple↔**gripper-proxy** when that joint exists); coupling / load-path test compares against an agreed baseline (e.g. fused-proxy regression, P0-only recording, or **1.a** if present).
- [ ] **Constraint-Lagrangian caveat documented:** Where cable joints still use penalty proxies, docstrings state that and link **[M4]**; fixed-joint gather path documented per Newton API.
- [ ] **Commands documented:** `README.md` and this file's "Agent execution notes" record the new `uv run …` entry-points (generate + sim with FR3 **1.b**), per `.cursor/rules/readme-runtime-verification.mdc`.

**Constraints / notes for implementers:**

- Newton-native by default; do **not** introduce a separate MuJoCo binary or runtime. `SolverMuJoCo` already lives in Newton; that is the only "MuJoCo" surface in M1.
- Keep all new logic under `apple_pick_sim/`. The only justifiable `newton/` edits are if `solver.joint_penalty_k` lookup needs a kernel for parity with the rest of the example (the existing TODO in `example_apple_stem.py::step` already notes this) — open a focused PR if it does.
- The two-`Model` coupling is **one-way per step with a one-step lag**. Do not attempt fixed-point iteration inside a step in M1 — accept the lag, document the implied stability/timestep constraint. **Slices 2c–2d are accepted**; **Slice 2f** refactors code without changing that protocol unless tests prove an intentional change. EE–apple **contact** and arm readouts stay **post-refactor** unless promoted in **Current focus**.
- Proxy bodies must mirror the *effective* inertia of the relevant robot subchain (often dominated by the hand + chain reduced-mass terms), not the raw link inertia, so that VBD-side contact/joint solutions remain well-conditioned. Tune by test, not by guess.
- Per `.cursor/rules/test-driven-development.mdc`, ship each slice **red → green**: **[P0]** stretch (1.a) owns wrench-acceleration tests **if** that path is promoted; **M1** adds failing coupling / contact tests before the full **1.b** loop and proxy-harvest unit tests before integrating MuJoCo stepping.
- Deterministic settings only (seeded P0 instance, fixed substep count, fixed wrench schedule for tests). Do not introduce viewer-coupled randomness into pytest paths.
- The cable-joint **penalty-force** readout (where used) is **adequate for break detection and qualitative load-distribution work in [P0]/M1**, but **not** a calibration-grade signal. M4 (calibration) inherits the responsibility to either move to a constraint-Lagrangian readout or document why the penalty proxy is sufficient.

**Next actions (ordered, small slices):**

1. [x] **Slice 1 — Proxy primitives (incl. VBD wrench-readout spike for *proxies*):** `apple_pick_sim/proxy_coupling.py` — velocity-delta harvest (option 3); `apple_pick_sim/tests/test_proxy_coupling.py`.
2. [x] **Slice 2 — FR3 robot loaded:** `testfr3_resolved.usda`, `build_coupled_fruiting_fr3`, placement + bootstrap tests; `--robot fr3` on coupled example.
3. [x] **Slice 2c — TCP velocity teleop:** `--only-mjc` and full coupled accepted.
4. [x] **Slice 2d — Coupled forces / transfer:** accepted (`2026-05-22`).
5. [x] **Slice 2f — Structural refactor** (2026-05-25).
6. [x] **Slice 2g — GPU optimization** (2026-05-25).
7. [x] **Slice 2e — Hardening / benchmarks** (2026-05-25).
8. [ ] **Slice 3 — Commands + docs** (optional polish).

**Next actions:** *(none — milestone complete; work continues under [M2])*

**Two-`Model` staggered coupling skeleton (authoritative reference):**

The fragment below is a *non-runnable* skeleton — placeholders (`robot_builder`, `cable_points`, `proxy_body_ids`, `effective_mass`, `harvest_proxy_wrenches`, …) are filled in by Slices 4–5. It captures the exact step ordering, the `sync_proxy_state` kernel signature, and the `SolverMuJoCo` kwargs that M1 targets.

```python
import warp as wp
import newton
from newton.solvers import SolverMuJoCo, SolverVBD

# --- Model A: MuJoCo rigid-body robot ---
robot_model = robot_builder.finalize()
mj_solver = SolverMuJoCo(
    robot_model,
    solver="newton",
    integrator="implicitfast",
    cone="elliptic",
    iterations=20,
    ls_iterations=10,
    ls_parallel=True,
    impratio=1000.0,
)
robot_state_0, robot_state_1 = robot_model.state(), robot_model.state()
control = robot_model.control()
mj_collision_pipeline = newton.CollisionPipeline(
    robot_model, reduce_contacts=True, broad_phase="explicit"
)
mj_contacts = mj_collision_pipeline.contacts()

# --- Model B: VBD deformable cable + apple + robot-link proxies ---
cable_builder = newton.ModelBuilder()
cable_builder.add_rod(
    positions=cable_points, quaternions=cable_quats, radius=0.003,
    stretch_stiffness=1e12, bend_stiffness=3.0,
    stretch_damping=1e-3, bend_damping=1.0,
)
# Mirror robot links the cable must "see" (typically the gripper / TCP):
for body_id in proxy_body_ids:
    proxy_id = cable_builder.add_body(
        xform=robot_state_0.body_q[body_id],
        mass=effective_mass[body_id],
    )
    for shape in shapes_on_body(robot_model, body_id):
        cable_builder.add_shape(body=proxy_id, **shape)
    robot_to_vbd[body_id] = proxy_id
# (apple body + apple↔gripper-proxy add_joint_fixed are added here too)

cable_model = cable_builder.finalize()
vbd_solver = SolverVBD(cable_model, iterations=10)
vbd_state_0, vbd_state_1 = cable_model.state(), cable_model.state()
vbd_control = cable_model.control()
vbd_collision_pipeline = newton.CollisionPipeline(cable_model)
vbd_contacts = vbd_collision_pipeline.contacts()

proxy_forces = wp.zeros(robot_model.body_count, dtype=wp.spatial_vector)
coupling_forces_cache = wp.zeros_like(proxy_forces)


@wp.kernel
def sync_proxy_state(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    src_body_q: wp.array(dtype=wp.transform),
    src_body_qd: wp.array(dtype=wp.spatial_vector),
    dst_body_q: wp.array(dtype=wp.transform),
    dst_body_qd: wp.array(dtype=wp.spatial_vector),
    proxy_forces: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    dt: float,
):
    i = wp.tid()
    rid = robot_ids[i]
    pid = proxy_ids[i]

    dst_body_q[pid] = src_body_q[rid]
    qd = src_body_qd[rid]

    # Undo coupling-force + gravity contribution on proxy velocity so VBD
    # does not double-integrate the same wrench in this step.
    f = proxy_forces[rid]
    delta_v = dt * body_inv_mass[pid] * wp.spatial_top(f)
    r = wp.transform_get_rotation(dst_body_q[pid])
    delta_w = dt * wp.quat_rotate(
        r, body_inv_inertia[pid] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    )
    qd = qd - wp.spatial_vector(delta_v + dt * body_inv_mass[pid] * gravity, delta_w)
    dst_body_qd[pid] = qd


# --- Coupled step (staggered, one-step lag) ---

# 1. Apply lagged VBD-to-MuJoCo wrenches
robot_state_0.clear_forces()
coupling_forces_cache.assign(proxy_forces)
robot_state_0.body_f.assign(robot_state_0.body_f + coupling_forces_cache)

# 2. Advance MuJoCo (rigid-body robot)
mj_collision_pipeline.collide(robot_state_0, mj_contacts)
mj_solver.step(robot_state_0, robot_state_1, control, mj_contacts, dt)
robot_state_0, robot_state_1 = robot_state_1, robot_state_0

# 3. Sync proxy poses/velocities and undo coupling forces (single kernel)
wp.launch(
    sync_proxy_state,
    dim=len(proxy_body_ids),
    inputs=[
        robot_ids_wp, proxy_ids_wp,
        robot_state_0.body_q, robot_state_0.body_qd,
        vbd_state_0.body_q, vbd_state_0.body_qd,
        coupling_forces_cache,
        cable_model.body_inv_mass, cable_model.body_inv_inertia,
        gravity, dt,
    ],
)

# 4. Advance VBD (cable + apple + cable-proxy contacts/joints)
vbd_collision_pipeline.collide(vbd_state_0, vbd_contacts)
vbd_solver.step(vbd_state_0, vbd_state_1, vbd_control, vbd_contacts, dt)

# 5. Harvest contact + joint-reaction wrenches on proxy bodies (used next step)
proxy_forces = harvest_proxy_wrenches(vbd_solver, vbd_state_1, vbd_contacts, dt)
vbd_state_0, vbd_state_1 = vbd_state_1, vbd_state_0
```

**Explicitly not in this milestone:**

- IK / trajectory control of the FR3 (the EE moves only under externally applied wrenches in M1).
- **Soft / multi-finger grasping** and slip-rich manipulation (deferred; rigid EE–apple **contact** is **in** scope for M1).
- Full **`SensorContact`** integration for every branch/fruit contact (defer detailed sensor plumbing unless a slice needs it; **EE–apple** may use the standard collision pipeline).
- Logging the commanded EE wrench into a versioned rollout schema (lives in M2 once the RL contract is promoted).
- True constraint-Lagrangian joint reactions (M4 calibration may promote this).

**Completed (archive as you go):**

- [x] Slice 1 — `apple_pick_sim/proxy_coupling.py` + `tests/test_proxy_coupling.py` (2026-05-15).
- [x] Slice 2a — `generate_coupled_cable_scene` / `CoupledCableScene` + `tests/test_coupled_cable_scene.py` (2026-05-15).
- [x] Slice 2b (placeholder) — `coupled_fruiting.py`, `example_coupled_fruiting.py`, stem-harvest + unified sync, `tests/test_coupled_fruiting_system.py`, `tests/test_coupling_stability.py`, `diagnostics/verify_coupling.py`, `docs/mujoco-vbd-coupling-architecture.md` (2026-05-18).
- [x] Slice 2c (core) — `Fr3EEVelocityController`, `--fr3-keyboard`, `test_fr3_ee_velocity_controller.py`, `coupling_force_debug.py` + `test_coupling_force_debug.py` (2026-05-19; `3f24330` ghost-proxy teleop).
- [x] Slice 2d — coupled force transfer accepted (2026-05-22).

---

### [M2] — RL infrastructure & Fisher-information exploration (Gymnasium + ASID)

- **Status:** In progress (active since 2026-05-25)
- **Links:** [Gymnasium — basic usage](https://gymnasium.farama.org/introduction/basic_usage/) · [ASID (ICLR 2024 oral)](https://openreview.net/forum?id=jNR6s6OSBT) · [arXiv:2404.12308](https://arxiv.org/abs/2404.12308) · [project page](https://weirdlabuw.github.io/asid/) · [reference code (FIM objective)](https://github.com/WEIRDLabUW/asid)
- **Vision:** Outcome 3 (*Learning infrastructure*); Fisher information as in `docs/VISION.md` glossary.

**Objective:** Phase **1** — **[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)** environment over the **[M1]** coupled simulator: `reset(seed, options)` → `step(action)` with a **`Dict` observation space** (placeholder/dummy for now; no sensors yet), separate from physics code. Phase **2** — **exploration policy** \(\pi_{\mathrm{exp}}\) and **Fisher information** infrastructure per ASID step (1). **Validation:** canonical rollouts via **direct** `apple_pick_sim` APIs must **match** the same schedule via **`Env.step`** (parity tests); **existing `apple_pick_sim/tests/`** continue to exercise the sim **directly** and must remain green. **Write M2 so [M3] can consume gradients:** `info` logs \(\theta\) / `params_fingerprint`; reserve sensitivity hooks — no full Newton autodiff required in M2.1.

**Architecture (M2.1):**

| Layer | Location | Role |
|-------|----------|------|
| Physics / coupling | `apple_pick_sim/` (unchanged) | `CoupledFruitingScene`, builders, controllers, `measure_fruiting_forces` |
| Optional runtime facade | `apple_pick_sim/` (small, no Gym import) | `reset` / `step` / `get_obs` / `apply_action` — numpy boundary |
| Gymnasium `Env` | `apple_pick_gym/` (planned) | `Dict` obs (placeholder allowed), `Discrete(13)` key-command actions in M2.1 (extend later); `register` id e.g. `ApplePickCoupled-v0`; `gymnasium` optional extra on Newton env |

**Default control contract (unless M2.0 ADR changes it):**
- In **M2.1**: action is a **single keyboard-style command** (`Discrete(13)` mapping to \(\pm X,\pm Y,\pm Z,\pm \mathrm{rotX},\pm \mathrm{rotY},\pm \mathrm{rotZ},noop\)) → `EEVelocity`, applied via `Fr3EEDirectJointController` + `CoupledFruitingScene.apply_fr3_ee_teleop_direct` with `robot_kinematic_mode=True`.
- One Gym **step** = one control frame = `SUBSTEPS_PER_FRAME` × `coupled_substep(SUB_DT)` (same timing as `apple_pick_sim/tests/conftest.py`).
- After parity + ADR, may extend to MultiDiscrete (simultaneous keys) and/or switch to continuous `Box(6)` twist actions for learned controllers.

**ASID mapping (apple-pick context):**

| ASID stage | This repo (target milestone) |
|------------|------------------------------|
| (0) Sim RL API (Gymnasium + parity) | **[M2]** M2.1 |
| (1) Train \(\pi_{\mathrm{exp}}\) maximizing Fisher info in sim | **[M2]** M2.2–M2.3 |
| (2) Deploy \(\pi_{\mathrm{exp}}\) in real, collect trajectories | **[M4]** (after M2 sim contract) |
| (3) System ID — refine sim parameters \(\theta\) | **[M3]** |
| (4–5) Task policy in updated sim → real | **[M5]** (+ vision Outcome 6) |

**Definition of done (checklist):**

- [ ] **Gymnasium env (M2.1):** `gymnasium.Env` with `Dict` observation space (dummy/placeholder allowed initially); documented action space (at least `Discrete(13)` for key commands in M2.1); `reset` / `step` / `close`; headless CI path; physics only via `apple_pick_sim` public API.
- [ ] **Parity tests (M2.1b):** direct `coupled_substep` path vs Gym `step` agree on fixed scenarios; **all existing `apple_pick_sim/tests/` green** on direct path.
- [ ] **`check_env` (M2.1c):** passes on agreed env id (short horizon).
- [ ] **M2.0 documented:** sensor keys, action scaling, substep count, \(\theta\) vector, FIM plan (M2.2).
- [ ] **FIM infrastructure (M2.2):** compute or approximate \(\mathrm{I}(\theta, \pi_{\mathrm{exp}})\) (or ASID \(\mathrm{tr}(\mathrm{I}^{-1})\) proxy) from rollouts.
- [ ] **\(\pi_{\mathrm{exp}}\) training (M2.3):** documented `uv run` entrypoints; smoke test on registered env.
- [ ] **M3 hooks (design):** rollouts + `info` expose \(\theta\); documented sensitivity path for [M3].

**Constraints / notes for implementers:**

- **[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)** is the RL env API: use `terminated` and `truncated` separately; optional `TimeLimit` wrapper for max episode length; `render_mode=None` for pytest.
- **Do not** move coupling logic into the Gym package; **do not** replace M1 tests with Gym-only runs — add **parity** tests instead.
- TDD: parity + `check_env` before FIM; FIM toy-\(\theta\) tests before full coupled training.
- Prefer **headless/deterministic** rollouts (`poll_events=False` on controllers); no viewer in CI.
- RL **library** choice is **M2.3** — env must stay framework-agnostic; \(\theta\) layout must not depend on a specific trainer tensor format.
- Optional `apple_pick_sim` runtime module is allowed; **`import gymnasium` only in `apple_pick_gym/`**.

**Next actions (ordered, small slices):** see **Current focus** (M2.1 → M2.1d, then M2.0 in parallel, then M2.2 → M2.3).

**Completed (archive as you go):**

- [ ] *(none yet)*

---

### [M3] — Simulation parameter tuning (system identification)

- **Status:** Planned (after [M2] env + FIM infra; **design for gradients in M2**)
- **Links:** ASID stage (3); vision Outcome 5 (*Calibration loop*)
- **Vision:** Outcome 5 — update sim parameters so sim–real (or sim–target) error drops on held-out data.

**Objective:** **Tune simulation parameters** \(\theta\) using trajectories from \(\pi_{\mathrm{exp}}\) (sim-first; then real logs from [M4]). Fit \(\theta\) so the coupled fruiting + FR3 sim matches observed poses, wrenches, contact events, or other agreed metrics. **Use gradient information where Newton/Warp allow**; otherwise document finite-difference or black-box search with the same \(\theta\) vector defined in [M2]. This is the engineering home for “simulation tuning” — not the exploration-policy trainer in [M2].

**Definition of done (checklist):**

- [ ] \(\theta\) update loop documented (optimizer, metrics, held-out eval).
- [ ] Quantitative **before/after** on held-out trajectories (sim replay or [M4] real segment).
- [ ] Sensitivity path documented: autodiff, adjoint, or FD — tied to [M2] hooks.

**Constraints / notes for implementers:**

- **Prefer gradients** from Newton-based sim when milestones allow; M2 must not paint M3 into a corner (log \(\theta\), avoid hard-coding non-identifiable combos).
- Start with **sim-only** identification (known \(\theta^\*\), recover from noisy obs) before real data.

**Next actions (ordered, small slices):**

- [ ] Stub `identify_theta` API + test on 2–3 parameter toy once [M2] rollout schema exists.
- [ ] Wire fruiting `sample_params` / builder fields into \(\theta\) with clear bounds.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

### [M4] — Real-world data collection & format alignment

- **Status:** Planned
- **Links:** ASID stage (2); vision Outcome 4
- **Vision:** Outcome 4.

**Objective:** Collect trajectories under the same \(\pi_{\mathrm{exp}}\) (or matched sensing) as simulation; formalize formats and ingestion for [M3] tuning.

**Definition of done (checklist):**

- [ ] Collection protocol documented; real logs validate against [M2] schema/versioning.

**Constraints / notes for implementers:**

- No production deployment or safety certification scope (see vision non-goals).

**Next actions (ordered, small slices):**

- [ ] Extend [M2] rollout schema for real-hardware fields once env contract is stable.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

### [M5] — Final manipulation policy (RL in calibrated sim)

- **Status:** Planned
- **Links:** N/A
- **Vision:** Outcome 6.

**Objective:** Train a final apple-picking policy in simulation after calibration tightens sim–real-relevant parameters.

**Definition of done (checklist):**

- [ ] Policy trained against [M3]-calibrated sim; eval criteria agreed with maintainer.

**Constraints / notes for implementers:**

- Builds on [M2] harness and [M3]-calibrated sim; avoid one-off policy formats.

**Next actions (ordered, small slices):**

- [ ] Defer until [M2]–[M4] exit criteria support a clear task success definition.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

## Backlog (not active)

Unordered ideas. **Do not implement** unless promoted into a milestone and “Current focus”.

- *(Promoted to Slice **2f** — see **`refactor.md`**; maintainer expands/refines the list.)*
- *(Promoted to Slice **2g** — GPU profilers, device-resident coupling, benchmark baselines; not active until **Current focus** promotes it after 2f.)*
- **Former end-to-end “thin slice” stubs** (rollout log schema v0, real-data adapter stub, calibration comparison stub, scripted policy placeholder): useful when **M2–M4** are promoted; not required to finish **P0** fruiting tests.
- *(Promoted to [M2] — Fisher-information / ASID exploration objective.)*
- Additional manipulators or crops — only with explicit scope change (vision non-goals).
- **Triangle mesh export or import** (OBJ/STL, render/FEM meshes) alongside or instead of capsule primitives — promote only if a milestone needs it; P0 stays **`ModelBuilder`** primitives per `example_apple_stem.py`.

---

## Agent execution notes

**Repository layout (this project):**

- Simulation / project code: `apple_pick_sim/` (e.g. `fruiting_system/`, `coupled_fruiting/` incl. `proxy_coupling.py`, `examples/`, `vbd_fixed_joint_wrenches.py` — wraps `SolverVBD.gather_joint_wrench_child_com` from `newton/`)
- RL / Gymnasium adapter (M2): `apple_pick_gym/` (planned) — `gymnasium.Env` only; depends on `apple_pick_sim`, not vice versa
- Physics engine (submodule, vendored): `newton/` — avoid drive-by edits; see `.cursor/rules/apple-pick-sim.mdc`

**How to validate changes:**

- Install / sync: `cd newton && uv sync --extra examples`
- Run example sim (smoke): from repo root, `uv run --directory newton python ../apple_pick_sim/examples/example_apple_stem.py`
- Tests (Newton / shared env): `uv run --directory newton python -m newton.tests` (narrow with path/file when iterating, e.g. `uv run --directory newton python -m newton.tests -k test_cable`)
- P0 fruiting-system tests: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing` (from repo root; `PYTHONPATH` ensures `apple_pick_sim` is importable; `--directory newton` sets the uv project but cwd becomes `newton/`)
- P0 wrench equilibrium (physics sanity): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_wrench_equilibrium.py -q -p no:launch_testing`
- M1 coupled cable scene (Slice 2a): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupled_cable_scene.py -q -p no:launch_testing`
- M1 coupled fruiting (Slice 2b placeholder loop): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupled_fruiting_system.py -q -p no:launch_testing`
- M1 proxy coupling (Slice 1): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_proxy_coupling.py -q -p no:launch_testing`
- M1 coupling stability (longer-horizon): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupling_stability.py -q -p no:launch_testing`
- M1 coupling verification CLI: `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/verify_coupling.py --num-substeps 600 --max-force 5 --max-torque 1`
- P0 fruiting viewer (smoke / ad hoc): `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fruiting_system.py` (see README for flags)
- M1 coupled viewer (placeholder default, `fix_to_apple=False`): `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py` (`--viewer null --num-frames …`; `--only-vbd` for cable-only).
- M1 FR3 loaded (Slice 2): `PYTHONPATH=$(pwd) uv run --directory newton python -m unittest apple_pick_sim.tests.test_fr3_usd_import -v`; `test_coupled_fruiting_system.py::test_fr3_*`; see `docs/fr3-usd-import-implementation.md`.
- M1 FR3 teleop (Slice 2c): `example_coupled_fruiting.py --robot fr3 --only-mjc --fr3-keyboard --viewer gl` (**verified**); full coupled: `--robot fr3 --fr3-keyboard --viewer gl` (ghost proxy; acceptance pending). Default `fix_to_apple=False`. WIP: `--fr3-direct-joints` with `--fr3-keyboard`.
- M1 FR3 keyboard (kinematic only): `examples/example_fr3_keyboard.py --viewer gl`.
- M1 FR3 controller unit tests: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_fr3_ee_velocity_controller.py -q -p no:launch_testing`
- M1 coupled forces debug (Slice 2d): `example_coupled_fruiting.py --robot fr3 --debug-coupling-forces`; `test_coupling_force_debug.py`; `diagnostics/verify_coupling.py` for headless checks.
- M1 coupled fruiting (FR3 integration): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupled_fruiting_system.py -k fr3 -q -p no:launch_testing`
- M1 Slice 2g (GPU): `docs/gpu-coupling-optimization.md`; `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/benchmark_coupling.py --device cuda:0 --mujoco-gpu --warmup-substeps 30 --bench-substeps 300`; examples: `--cuda-graph` with `--viewer null` on coupled/fruiting scripts.
- M1 Slice 2e (hardening): `docs/slice-2e-hardening.md`; `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/benchmark_coupling.py --robot placeholder --warmup-substeps 30 --bench-substeps 300`; slow tests: `pytest ../apple_pick_sim/tests/ -m slow -q -p no:launch_testing`
- M1 architecture doc: `docs/mujoco-vbd-coupling-architecture.md`
- M1 Slice 2f (structural refactor gates): `docs/slice-2f-structural-refactor.md` — fruiting: `pytest ../apple_pick_sim/tests/test_fruiting_system.py ../apple_pick_sim/tests/test_coupled_cable_scene.py -q`; coupled: `pytest ../apple_pick_sim/tests/test_coupled_fruiting_system.py ../apple_pick_sim/tests/test_coupling_stability.py -q`; proxy: `pytest ../apple_pick_sim/tests/test_proxy_coupling.py -q` (all with `PYTHONPATH=$(pwd) uv run --directory newton … -p no:launch_testing`)
- M1 refactor: read **`refactor.md`** and **Current focus** before structural edits; maintainer updates both when priorities change
- M1 work: follow **Current focus** and [M1] *Next actions*; add `uv run` entry-points here as slices land
- M2 Gymnasium (when `apple_pick_gym/` exists): install `gymnasium` in Newton env (optional extra TBD); parity + contract tests: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_gym/tests/ -q -p no:launch_testing` (from repo root)
- M2 regression gate: **always** run M1 direct-path suite before merging M2 slices: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing`

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict.
- A change requires **policy or product** decisions (scope, user-visible behavior, compatibility).
- A task requires **network credentials**, **paid APIs**, or **destructive** operations.

**When unsupervised is expected:**

- You may complete the **next unchecked slice** in “Current focus” using TDD and project rules — for **[M2]**, land **M2.1** (Gymnasium env + parity) before FIM / training unless the maintainer reorders; keep **`apple_pick_sim/tests/`** green on every PR; add **`apple_pick_gym/tests/`** parity + `check_env` with M2.1.
- You may fix **small obvious blockers** uncovered by that slice (tests, imports, typos) if they are necessary for the slice to be correct.
- You should **not** start a new milestone or backlog item without maintainer direction unless this file explicitly says otherwise.
