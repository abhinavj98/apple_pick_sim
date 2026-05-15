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
| **Last updated** | 2026-05-15 (**[M1] active** — FR3 + custom EE, rigid EE–apple contact, two-`Model` staggered coupling, **arm-side** force measurement. **[P0] Done** — variance, fixed-joint wrenches, refactor landed; optional stretch deferred.) |
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
2. **Now — Outcome 2 (manipulation stack, [M1]):** Integrate a **Franka FR3** with a **custom end-effector**, add **rigid contact** between the end-effector and the apple, and run the **two-`Model`** **`SolverMuJoCo` + `SolverVBD`** **staggered coupling** (proxy bodies + one-step lag — cable joints still cannot live on the MuJoCo model). **Measure forces on the robot arm** (MuJoCo / `body_f` / joint-level readouts as agreed) **and** reuse **P0 VBD telemetry** on the cable model. **End-effector wrench input** drives the arm in M1 tests. See vision *Manipulation stack*; see [M1] for architecture, DoD, and the coupling skeleton.
3. **Later — Outcome 3 (learning):** **RL infrastructure** for an **exploration policy** (and downstream training hooks). Scope, stack, and reward/exploration design are **TBD** in this file until you promote specifics from discussion into milestones and “Current focus”.

Later vision phases (real data, calibration, final pick policy) remain in milestones below; they are **not** active until the maintainer moves focus past arm integration and RL foundations.

---

## Current focus

**Active milestone:** [M1] — **FR3 manipulation stack** (two-`Model` **`SolverMuJoCo` + `SolverVBD`** coupling).

**In one sentence, the goal right now:** Extend **[P0]**’s fruiting scene with a **Franka FR3 + custom end-effector**, **rigid EE–apple contact** on the coupled stack, the **staggered proxy coupling protocol**, and **force readouts on both** the VBD cable model (reuse P0) **and** the robot arm.

**Build on (do not reimplement):** `apple_pick_sim/fruiting_system.py` — `generate_scene`, `run_rollout`, `measure_fruiting_forces`, `vbd_fixed_joint_wrenches.py`. Reference patterns: `newton/newton/examples/ik/example_ik_franka.py`, `newton/newton/examples/cloth/example_cloth_franka.py` (adjacent, not the M1 recipe), and the **two-`Model` staggered coupling skeleton** at the bottom of [M1] in this file.

**Next up (ordered — mirrors [M1] Next actions):**

1. [ ] **Slice 1 — Proxy primitives:** `sync_proxy_state` Warp kernel + `harvest_proxy_wrenches(...)` under `apple_pick_sim/`; **first sub-task:** VBD wrench-readout spike (direct read / narrow `newton/` extension / velocity-delta reconstruction); unit tests in isolation (no MuJoCo step yet).
2. [ ] **Slice 2 — FR3 + custom EE + full 1.b loop:** `robot_model` + `cable_model`, collision-equipped gripper **proxy**, **rigid EE–apple contact**, staggered protocol, FR3 TCP `body_f` wrenches, P0 VBD readouts + **arm-side** force extraction; coupling / contact tests.
3. [ ] **Slice 3 — Commands + docs:** README and Agent execution notes for **1.b** FR3 flows; verify per `.cursor/rules/readme-runtime-verification.mdc`.

**Explicitly not in this milestone:** IK / trajectory control, soft multi-finger grasping, full `SensorContact` plumbing, rollout log schema (M2), calibration (M4), RL harness (M2). See [M1] *Explicitly not* for the full list.

**Blockers (if any):** None assumed. Justified `newton/` edits only for proxy wrench readout if Slice 1 spike requires a narrow public API (see [M1] Slice 1 spike).

**Last completed slice (prior milestone):** [P0] variational fruiting + fixed-joint wrench telemetry + refactor; exited 2026-05-15.

---

## Milestones

Phases below follow **Sequencing** at the top: **[P0] Done** → **[M1] active** (FR3 + custom EE + coupled contact) → RL (details TBD) → later vision phases.

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

- **Status:** In progress (active since 2026-05-15; **[P0] Done** — reuse `fruiting_system` / wrench readouts; do not duplicate P0 DoD here)
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

**Force readout:** **[P0]** supplies **fixed-joint** wrenches via **`SolverVBD.gather_joint_wrench_child_com`** (`vbd_fixed_joint_wrenches.py`) and cable-joint **penalty** proxies where that convention holds. **M1 adds** **robot-side** readouts: document how to recover **joint torques**, **link wrenches**, or **`body_f`**-consistent totals on **`robot_model`** after each coupled substep so **forces on the arm** from apple interaction are **measured** alongside VBD-side reactions. Document penalty vs. gather semantics for [M4] calibration.

**Attachment / interaction matrix:**

| Mode | What it is | Solvers needed | When to use | Slice |
|------|-----------|----------------|-------------|-------|
| **1.a Floating-EE handle** | Single free-floating rigid body on `cable_model` fused to the apple via `add_joint_fixed`; wrenches on that body. **No FR3.** | `SolverVBD` only. | Optional **P0 stretch** for isolated load curves; **not** assumed to exist for M1 unless promoted during P0 refactor. | [P0] optional |
| **1.b Full FR3 + coupling** | Two `Model`s; FR3 hand mirrored as **proxy** in `cable_model`; staggered protocol below. **Rigid EE–apple contact** (and/or **fixed** apple↔proxy for regression). | `SolverMuJoCo` + `SolverVBD`. | Primary M1 manipulation scenario. | [M1] |

**Definition of done (checklist):**

- [ ] **Custom end-effector:** FR3 import extended with **project-specific EE geometry** (URDF fragment, extra shapes, or documented asset) and tested load path.
- [ ] **Rigid EE–apple contact:** At least one headless scenario where apple and EE (or **gripper proxy** on `cable_model`) interact through **rigid / stiff contact**; contact pipeline and filters documented.
- [ ] **Robot-arm force readouts:** Documented + tested extraction of **interaction forces/torques on the FR3 chain** from the **coupled** sim (agreed links / joints / `body_f` / MuJoCo diagnostics).
- [ ] **Builder integration:** A function in `apple_pick_sim/` extends the P0 builder to produce **1.b** (`robot_model` + `cable_model`, FR3 + proxy bodies + EE) from `(range_json, seed)`, with the FR3 base parked so the TCP / EE starts near the sampled apple.
- [ ] **Two-`Model` staggered step loop:** One reproducible script under `apple_pick_sim/` runs **`SolverMuJoCo` + `SolverVBD`** headlessly and deterministically, implementing the protocol above. The proxy sync kernel and `harvest_proxy_wrenches` are unit-tested separately.
- [ ] **End-effector wrench on FR3:** Documented application of wrenches to the MuJoCo robot `body_f` (TCP / agreed link), once per substep, with tests that build on **[P0]** patterns where applicable.
- [ ] **Reuse [P0] telemetry on 1.b:** **[P0]** fixed-joint + cable-side readouts run on `cable_model` (including apple↔**gripper-proxy** when that joint exists); coupling / load-path test compares against an agreed baseline (e.g. fused-proxy regression, P0-only recording, or **1.a** if present).
- [ ] **Constraint-Lagrangian caveat documented:** Where cable joints still use penalty proxies, docstrings state that and link **[M4]**; fixed-joint gather path documented per Newton API.
- [ ] **Commands documented:** `README.md` and this file's "Agent execution notes" record the new `uv run …` entry-points (generate + sim with FR3 **1.b**), per `.cursor/rules/readme-runtime-verification.mdc`.

**Constraints / notes for implementers:**

- Newton-native by default; do **not** introduce a separate MuJoCo binary or runtime. `SolverMuJoCo` already lives in Newton; that is the only "MuJoCo" surface in M1.
- Keep all new logic under `apple_pick_sim/`. The only justifiable `newton/` edits are if `solver.joint_penalty_k` lookup needs a kernel for parity with the rest of the example (the existing TODO in `example_apple_stem.py::step` already notes this) — open a focused PR if it does.
- The two-`Model` coupling is **one-way per step with a one-step lag**. Do not attempt fixed-point iteration inside a step in M1 — accept the lag, document the implied stability/timestep constraint, and revisit only if a scenario diverges.
- Proxy bodies must mirror the *effective* inertia of the relevant robot subchain (often dominated by the hand + chain reduced-mass terms), not the raw link inertia, so that VBD-side contact/joint solutions remain well-conditioned. Tune by test, not by guess.
- Per `.cursor/rules/test-driven-development.mdc`, ship each slice **red → green**: **[P0]** stretch (1.a) owns wrench-acceleration tests **if** that path is promoted; **M1** adds failing coupling / contact tests before the full **1.b** loop and proxy-harvest unit tests before integrating MuJoCo stepping.
- Deterministic settings only (seeded P0 instance, fixed substep count, fixed wrench schedule for tests). Do not introduce viewer-coupled randomness into pytest paths.
- The cable-joint **penalty-force** readout (where used) is **adequate for break detection and qualitative load-distribution work in [P0]/M1**, but **not** a calibration-grade signal. M4 (calibration) inherits the responsibility to either move to a constraint-Lagrangian readout or document why the penalty proxy is sufficient.

**Next actions (ordered, small slices):**

1. [ ] **Slice 1 — Proxy primitives (incl. VBD wrench-readout spike for *proxies*):** Add `sync_proxy_state` Warp kernel and `harvest_proxy_wrenches(...)` under `apple_pick_sim/`. **First sub-task**: a short, focused spike that establishes which of the three options in the "Slice 4 spike" paragraph (direct read / narrow public extension / velocity-delta reconstruction) `harvest_proxy_wrenches` will use; pick the cheapest one that passes a momentum-balance unit test on a mocked 1-body VBD scene. Unit tests exercise both helpers in isolation (synthetic robot states, no MuJoCo solver yet).
2. [ ] **Slice 2 — FR3 + custom EE + 1.b (full two-`Model` loop + EE–apple contact):** Build `robot_model` with FR3 URDF + EE, build `cable_model` with collision-equipped **proxy** and **rigid contact** to the apple (and optional **fixed** fusion for regression), run the staggered protocol headlessly for *N* frames, apply wrenches on the FR3 TCP `body_f`, wire **[P0]** VBD readouts + **arm-side** force extraction, add coupling / contact tests.
3. [ ] **Slice 3 — Commands + docs:** README and Agent execution notes for **1.b** FR3 flows updated and verified per `.cursor/rules/readme-runtime-verification.mdc`.

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

- [ ] *(none yet)*

---

### [M2] — RL infrastructure & exploration policy *(planning deferred)*

- **Status:** Planned — **detailed scope TBD** (owner will expand milestones and “Current focus” when ready).
- **Links:** N/A
- **Vision:** Outcome 3 (*Learning infrastructure*); exploration/reward design may target informative trajectories (e.g. Fisher information, per vision glossary).

**Objective:** Training/eval harness tied to the simulator; support learning an **exploration policy** (and later task policies) with stable observation/action contracts. Exact RL stack, env wrapper, and exploration objectives remain to be specified.

**Definition of done (checklist):** *(to be filled when planning lands)*

- [ ] Train/eval entrypoints documented.
- [ ] Sim rollouts use a **versioned** observation/action contract (extend rather than rewrite ad hoc when M1 exists).

**Constraints / notes for implementers:**

- Prefer headless/deterministic configurations where practical.

**Next actions (ordered, small slices):**

- [ ] Defer choosing RL stack and env wrapper until this milestone is promoted with concrete exit criteria.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

### [M3] — Real-world data collection & format alignment

- **Status:** Planned
- **Links:** N/A
- **Vision:** Outcome 4.

**Objective:** Collect trajectories under the same policy and sensing assumptions as simulation; formalize formats and ingestion.

**Definition of done (checklist):**

- [ ] Collection protocol documented; real logs validate against shared schema/versioning.

**Constraints / notes for implementers:**

- No production deployment or safety certification scope (see vision non-goals).

**Next actions (ordered, small slices):**

- [ ] Extend schema/fixtures once M2 rollout contracts exist.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

### [M4] — Calibration loop (sim ↔ real)

- **Status:** Planned
- **Links:** N/A
- **Vision:** Outcome 5 + success criterion *parameter updates improve fit*.

**Objective:** Use real observations with simulator sensitivity to update parameters; show error reduction on held-out segments.

**Definition of done (checklist):**

- [ ] Chosen metric(s) documented; quantitative before/after on held-out real segment (or agreed proxy dataset).

**Constraints / notes for implementers:**

- Prefer gradients or documented sensitivity hooks from Newton-based sim where milestones allow.

**Next actions (ordered, small slices):**

- [ ] Replace stubs with optimization or systematic search per chosen metric once M3 data path is stable.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

### [M5] — Final manipulation policy (RL in calibrated sim)

- **Status:** Planned
- **Links:** N/A
- **Vision:** Outcome 6.

**Objective:** Train a final apple-picking policy in simulation after calibration tightens sim–real-relevant parameters.

**Definition of done (checklist):**

- [ ] Policy trained against M4-calibrated sim; eval criteria agreed with maintainer.

**Constraints / notes for implementers:**

- Builds on M2 harness and prior milestone contracts; avoid one-off policy formats.

**Next actions (ordered, small slices):**

- [ ] Defer until M2–M4 exit criteria support a clear task success definition.

**Completed (archive as you go):**

- [ ] *(none yet)*

---

## Backlog (not active)

Unordered ideas. **Do not implement** unless promoted into a milestone and “Current focus”.

- **Former end-to-end “thin slice” stubs** (rollout log schema v0, real-data adapter stub, calibration comparison stub, scripted policy placeholder): useful when **M2–M4** are promoted; not required to finish **P0** fruiting tests.
- Fisher-information–shaped rewards or exploration bonuses (vision glossary) — after base RL loop exists (**M2**).
- Additional manipulators or crops — only with explicit scope change (vision non-goals).
- **Triangle mesh export or import** (OBJ/STL, render/FEM meshes) alongside or instead of capsule primitives — promote only if a milestone needs it; P0 stays **`ModelBuilder`** primitives per `example_apple_stem.py`.

---

## Agent execution notes

**Repository layout (this project):**

- Simulation / project code: `apple_pick_sim/` (e.g. `fruiting_system.py`, `vbd_fixed_joint_wrenches.py` — wraps `SolverVBD.gather_joint_wrench_child_com` from `newton/`)
- Physics engine (submodule, vendored): `newton/` — avoid drive-by edits; see `.cursor/rules/apple-pick-sim.mdc`

**How to validate changes:**

- Install / sync: `cd newton && uv sync --extra examples`
- Run example sim (smoke): from repo root, `uv run --directory newton python ../apple_pick_sim/example_apple_stem.py`
- Tests (Newton / shared env): `uv run --directory newton python -m newton.tests` (narrow with path/file when iterating, e.g. `uv run --directory newton python -m newton.tests -k test_cable`)
- P0 fruiting-system tests: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing` (from repo root; `PYTHONPATH` ensures `apple_pick_sim` is importable; `--directory newton` sets the uv project but cwd becomes `newton/`)
- P0 fruiting viewer (smoke / ad hoc): `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_fruiting_system.py` (see README for flags)
- M1 work: follow **Current focus** and [M1] *Next actions*; add `uv run` entry-points here as slices land

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict.
- A change requires **policy or product** decisions (scope, user-visible behavior, compatibility).
- A task requires **network credentials**, **paid APIs**, or **destructive** operations.

**When unsupervised is expected:**

- You may complete the **next unchecked slice** in “Current focus” using TDD and project rules.
- You may fix **small obvious blockers** uncovered by that slice (tests, imports, typos) if they are necessary for the slice to be correct.
- You should **not** start a new milestone or backlog item without maintainer direction unless this file explicitly says otherwise.
