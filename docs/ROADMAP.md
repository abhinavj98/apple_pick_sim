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
| **Last updated** | 2026-05-13 (P0 complete: variational fruiting-system generator `apple_pick_sim/fruiting_system.py` + range fixture + pytest suite in `apple_pick_sim/tests/test_fruiting_system.py` all green) |
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

1. **Now — Outcome 1 (variance):** Prove **variationally generated** fruiting-system **simulation geometry** and **Newton simulations** from a **JSON min/max range file** and **seed**, for the canonical **primary → secondary → spur → stem → apple** topology (see [P0]), using the same **procedural Newton build pattern** as `apple_pick_sim/example_apple_stem.py` (polylines + per-segment frames → capsule **rods** / cable joints + **sphere** apple under **`SolverVBD`**), with automated tests (distinct geometry or labels across seeds; deterministic where practical). See `docs/VISION.md` success row *Procedural fruiting variance*.
2. **Next — Outcome 2 (manipulation stack):** Integrate a **Franka FR3** model with the P0 scene using **Newton's `SolverMuJoCo`** for the FR3 articulation while keeping **`SolverVBD`** for the cable/rod backbone — **two separate `Model`s** with **proxy bodies + one-step-lag staggered coupling** (the rod uses `JointType.CABLE`, which `SolverMuJoCo` does not support, so the split is required and a single shared `Model` is not viable). Add an **end-effector wrench input** and **per-segment force readout** along the fruiting system. See vision *Manipulation stack*; see [M1] for slices, DoD, and the authoritative coupling skeleton.
3. **Later — Outcome 3 (learning):** **RL infrastructure** for an **exploration policy** (and downstream training hooks). Scope, stack, and reward/exploration design are **TBD** in this file until you promote specifics from discussion into milestones and “Current focus”.

Later vision phases (real data, calibration, final pick policy) remain in milestones below; they are **not** active until the maintainer moves focus past arm integration and RL foundations.

---

## Current focus

**Active milestone:** [M1] — MuJoCo + Franka FR3 manipulation stack (P0 complete; waiting for maintainer to start M1)

**In one sentence, the goal right now (P0 complete):** P0 fruiting-system generator (`apple_pick_sim/fruiting_system.py`) is done and tests are green. Next focus is M1 (FR3 + MuJoCo integration) when the maintainer is ready.

**Implementation pattern (authoritative for P0 “geometry”):** Follow `apple_pick_sim/example_apple_stem.py`: sample parameters → build **control polylines** and per-segment **quaternions** → `ModelBuilder` **`add_rod`** (capsule segments + **cable** joints for stretch/bend) for each woody segment chain; **stem** + **apple** as link(s) with **`add_shape_sphere`** (or agreed primitive) and a suitable joint; integrate with **`newton.solvers.SolverVBD`**. This is **analytic / primitive-based** scene construction, **not** generation of external **triangle meshes** (OBJ/STL) unless a later milestone promotes that.

**Canonical fruiting system (design target for P0):**

| Segment | Role | Compliance / stiffness (intent) |
|---------|------|----------------------------------|
| **Primary branch** | Main structural member | **Stiffer** (baseline “tree” limb) |
| **Secondary branch** | Lateral continuation | **Softer** than primary (lower bending stiffness or equivalent material) |
| **Spur** | Short segment before fruit | Short; carries stem attachment |
| **Stem** | Fruit attachment | Connects spur to apple |
| **Apple** | Terminal body | Rigid or softly stiff fruit body per sim conventions |

**Variation inputs:** A **JSON** file describes allowed ranges (e.g. **min / max**) for geometric and material-style parameters — examples include segment **counts**, **lengths**, **radii** (capsule thickness), **bend / stretch stiffness**, **density**, and apple size/mass. The generator samples (deterministically from **seed**) within those bounds and instantiates the **ModelBuilder** graph (bodies, capsules, joints, articulations) plus **`SolverVBD`** settings for each instance.

**Next up (ordered):**

1. [x] **JSON schema + fixture:** `apple_pick_sim/fixtures/fruiting_system_ranges.json` — documented min/max ranges for all five segments and apple.
2. [x] **Generator + Newton geometry:** `apple_pick_sim/fruiting_system.py` — `generate_scene(ranges, seed)` → `FruitingSystemScene` with full primary → secondary → spur → stem → apple topology via rod/capsule + sphere + `SolverVBD`.
3. [x] **Sim hook:** `run_rollout(scene, num_steps, sim_substeps)` runs a deterministic headless VBD rollout in-place.
4. [x] **Tests:** pytest tests in `apple_pick_sim/tests/test_fruiting_system.py` — fixture schema, parameter bounds, stiffness ordering, body counts, geometry fingerprint stability/variance, rollout crash-safety, rollout determinism, self-collision toggle — all green.
5. [x] **Document commands:** README updated with generate-only and generate+sim `uv` commands and test command.

**Explicitly not in this milestone:** MuJoCo/FR3 wiring, rollout log schema, real-data adapters, calibration stubs, and RL harnesses — those follow in later milestones or backlog unless you reprioritize.

**Blockers (if any):** None assumed; if Newton-only geometry is required for variance, keep logic in `apple_pick_sim/` and avoid drive-by `newton/` edits unless patching the submodule is in scope.

**Last completed slice:** *(update when a slice lands)*

---

## Milestones

Phases below follow **Sequencing** at the top: fruiting variance tests → FR3 manipulation stack & force telemetry (`SolverMuJoCo` + `SolverVBD` in Newton) → RL (details TBD) → later vision phases.

### [P0] — Variational fruiting system generation

- **Status:** Done
- **Links:** N/A
- **Vision:** Outcome 1 (*Visual and structural variance*); success row *Procedural fruiting variance*.

**Objective:** From a **JSON** file of **min/max** bounds on geometry and material-like parameters, plus a **seed**, **variationally generate** a **Newton-ready scene** for a **fixed topology**: **primary branch (stiff)** → **secondary branch (softer)** → **short spur** → **stem** → **apple**, built like **`example_apple_stem.py`** (polylines + **`add_rod`** capsule chains + primitive fruit body, **`SolverVBD`**). Run a short reproducible simulation and prove variance and determinism with tests.

**Definition of done (checklist):**

- [x] Documented **JSON** range format and **seed** semantics (same file + seed → same instance). See `apple_pick_sim/fixtures/fruiting_system_ranges.json`.
- [x] Generator implements the **five-part topology** above; primary vs. secondary **stiffness ordering** is respected in exported sim parameters (enforced in `sample_params`).
- [x] **Per-instance Newton model** is produced via **`ModelBuilder`** (primitive-based geometry as above); `FruitingSystemScene` dataclass carries model, solver, state, body indices.
- [x] **Newton rollout** with **`SolverVBD`** from the generated scene runs headlessly with deterministic settings for tests (`run_rollout`).
- [x] Pytest(s): fixed `(range_json, seed)` → stable summary or fingerprint; varying seed or bounds → **distinct** geometry or labels (as asserted). 17 tests, all green.
- [x] Documented command(s) for generate and generate+sim (see README and Agent execution notes).

**Constraints / notes for implementers:**

- Prefer `apple_pick_sim/`; avoid drive-by `newton/` edits unless the milestone explicitly patches the submodule.
- TDD and `uv` conventions apply.
- Start with **scalar stiffness / thickness / length** parameters if full anisotropic materials are not yet exposed in the chosen Newton API; document the mapping from JSON fields to sim bodies.
- **Triangle mesh import/generation** (OBJ/STL, high-res render meshes) is **out of scope for P0** unless promoted from **Backlog**; P0 “geometry” means the **capsule/sphere + cable joint** representation consistent with `example_apple_stem.py`.

**Next actions (ordered, small slices):**

- All slices complete — see Definition of done above. Proceed to M1 when ready.

**Completed (archive as you go):**

- [x] JSON schema + fixture (`apple_pick_sim/fixtures/fruiting_system_ranges.json`)
- [x] Generator module (`apple_pick_sim/fruiting_system.py`) with `load_ranges`, `sample_params`, `generate_scene`, `geometry_fingerprint`, `run_rollout`
- [x] pytest tests in `apple_pick_sim/tests/test_fruiting_system.py` — all green

---

### [M1] — FR3 manipulation stack & fruiting-system force telemetry

- **Status:** Planned (starts after P0 exit criteria are green)
- **Links:** Reference patterns: `newton/newton/examples/cloth/example_cloth_franka.py` (Featherstone+VBD coupling on one `Model` — *adjacent* but **not** the M1 recipe, see below), `newton/newton/examples/ik/example_ik_franka.py` (FR3 URDF import + TCP body), `newton/newton/tests/test_body_force.py` (external wrench API), and the **"two-`Model` staggered coupling skeleton"** at the bottom of this section (authoritative pattern for M1).
- **Vision:** Outcome 2 (*Manipulation stack*). MuJoCo enters as **`SolverMuJoCo` inside Newton** — there is **no separate MuJoCo runtime** in this milestone.

**Objective:** Build on the P0 variational fruiting system by **attaching a Franka FR3** at the apple, **applying end-effector wrenches** to drive the system, and **measuring forces at each segment** of the fruiting backbone, with deterministic tests. Default scenario: arm gripper is **rigidly fused** to the apple (no contact-driven grasping), to isolate the abscission / load-transfer mechanics.

**Architecture (authoritative for M1):**

The FR3 (rigid articulated robot) and the rod backbone (cable joints) are integrated by **different solvers on two separate `Model`s**, coupled at a per-step boundary by **proxy bodies + one-step-lag staggered wrench exchange**. The motivation is concrete:

- `SolverMuJoCo` does **not support `JointType.CABLE`** (`newton/_src/solvers/mujoco/solver_mujoco.py:292`) → the rod backbone *cannot* live in the MuJoCo model.
- `SolverVBD` *can* simulate revolute chains but is not the natural fit for an articulated-robot tracking controller → the FR3 *should* live in the MuJoCo model.
- Newton's "two solvers on one `Model`" pattern (e.g. `example_cloth_franka.py`, which uses `SolverFeatherstone + SolverVBD`) is therefore inappropriate here; we use **two `Model`s** instead, mirroring the relevant robot body or bodies as **proxy rigid bodies** inside the VBD `Model`.

**Per-`Model` ownership (M1):**

| `Model` | Solver | Contents |
|---------|--------|----------|
| `robot_model` | `SolverMuJoCo` | FR3 articulation (URDF). Apple body is **not** here. |
| `cable_model` | `SolverVBD` | Rod backbone (primary → … → stem) with cable joints, the apple rigid body, and **proxy** copies of the FR3 link(s) the cable needs to "see" (gripper hand / TCP). The apple↔gripper-proxy `add_joint_fixed` attachment lives here. |

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
4. Advance the VBD solver one step on `cable_model` / `vbd_state_0` (rod + apple + contacts against the proxy bodies + the apple↔proxy fixed joint).
5. **Harvest** the net wrench on each proxy body from the just-completed VBD step — read it from VBD's per-body force/torque accumulators if they're exposed (preferred), else reconstruct from the velocity delta (see Slice 4 spike below). Store as the new lagged wrench used at step 1 of the next iteration.

Items the implementation must add on top of P0:

- `harvest_proxy_wrenches(vbd_solver, vbd_state, vbd_contacts, dt) -> wp.array(spatial_vector)` — not in Newton; lives under `apple_pick_sim/`. **Preferred implementation**: read the VBD-internal per-body force/torque accumulators populated during contact + constraint solves (see `newton/_src/solvers/vbd/rigid_vbd_kernels.py` — `accumulate_body_body_contacts_per_body`, `accumulate_body_particle_contacts_per_body`, and the `joint_lambda_lin` constraint Lagrangians). One readout per proxy, summed into a single spatial wrench, no separate "joint reaction" channel exposed in the M1 API.
- A Warp kernel equivalent to `sync_proxy_state` in the reference skeleton (proxy mirror + force/gravity subtraction).
- A small registry that maps `robot_body_id → proxy_body_id` for the bodies the cable needs.

**Slice 4 spike — confirm VBD exposes a usable per-body wrench:**
The accumulators above are populated internally each step; the question Slice 4 must answer is whether they're (a) publicly readable through `state` / `solver` / `contacts`, (b) readable via a narrow public extension in `newton/`, or (c) only reconstructable. Order of preference for `harvest_proxy_wrenches`:

1. **VBD publishes the wrench.** Direct read; nothing else needed. M1 ships as designed.
2. **VBD does not publish but exposes accumulators.** Add a tiny public passthrough in `newton/` (focused PR, justified by the use case).
3. **VBD does not expose either.** Reconstruct from velocity delta: `harvest_f = body_mass * (v_proxy_post - v_proxy_synced) / dt - body_mass * gravity`. This works because `sync_proxy_state` already arranged for `v_proxy_synced` to absorb the lagged coupling force + gravity. The fixed-joint reaction is then implicit in the velocity change, so the contact + joint-reaction split is irrelevant at the API level.

The "joint reaction" split surfaces only inside option 3's derivation; the M1 public API stays single-channel either way.

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

**Force readout:** generalize the existing `ExampleAppleStem.get_forces()` from "last cable joint only" to "every cable joint along primary → secondary → spur → stem", plus a separate readout for the apple↔gripper-proxy fixed joint reaction. Both use the VBD `solver.joint_penalty_k * displacement` proxy; document that this is the penalty-force approximation and **not** the true constraint Lagrangian — fine for M1's break-detection / shape-of-load-curve use cases, flagged for M4 calibration to revisit.

**Attachment-mode matrix (both scenarios in scope; ship 1.a first):**

| Mode | What it is | Solvers needed | When to use | Slice |
|------|-----------|----------------|-------------|-------|
| **1.a Floating-EE handle** | A single free-floating rigid body added to `cable_model`, fused to the apple via `add_joint_fixed`; wrenches written directly on it. **No FR3, no MuJoCo, no proxies.** | `SolverVBD` only — single-`Model`. | Characterize the fruiting system in isolation (force-vs-displacement curves, break thresholds) without arm dynamics or coupling overhead. | First. |
| **1.b Full FR3 chain** | Two `Model`s per the table above; FR3 hand mirrored as a proxy in `cable_model`; `add_joint_fixed(gripper_proxy, apple)` lives on the VBD side; staggered coupling per the protocol above. | `SolverMuJoCo` (robot) + `SolverVBD` (cable). | End-to-end manipulation-relevant scenarios; sets up future control / IK loops. | After 1.a force readouts pass tests. |

**Definition of done (checklist):**

- [ ] **Builder integration:** A function in `apple_pick_sim/` extends the P0 builder to produce both **1.a** (single-`Model`, VBD-only with a floating-EE handle) and **1.b** (`robot_model` + `cable_model`, FR3 + proxy bodies) variants from the same `(range_json, seed)`, with the FR3 base parked so its TCP starts at the sampled apple position.
- [ ] **Two-`Model` staggered step loop:** One reproducible script under `apple_pick_sim/` runs the **`SolverMuJoCo` + `SolverVBD`** coupled scene headlessly and deterministically, implementing the 6-step protocol above. The proxy sync kernel and `harvest_proxy_wrenches` are unit-tested separately.
- [ ] **End-effector wrench API:** A documented helper, e.g. `apply_ee_wrench(state, ee_body, wrench_world)`, applied once per substep. Pytest: a known constant wrench on a stripped-down **1.a** model produces the analytically expected EE body acceleration (mirrors `test_body_force.py`).
- [ ] **Per-segment force telemetry:** A documented helper, e.g. `measure_fruiting_forces(state, model, solver)`, returns a per-joint dict for every cable joint along primary → secondary → spur → stem (`{joint_key: {"linear_force_N": ..., "bending_torque_Nm": ..., "stretch_displacement_m": ..., "bend_angle_rad": ...}}`), **plus** a separate entry for the apple↔gripper-proxy fixed-joint linear reaction. Replaces the inline `print` loop in today's `get_forces()`.
- [ ] **Force-rises-with-load test:** Pytest fixture applies a known monotonically increasing pull on the apple in **1.a**, integrates a fixed number of frames, asserts (i) all backbone segment linear forces are non-decreasing within tolerance, (ii) stiffer primary segment carries less strain than the softer secondary at matched load — both follow from the canonical stiffness ordering set in P0.
- [ ] **Coupling-consistency test (1.b):** Same pull schedule applied in 1.a vs 1.b at matched gripper kinematics produces per-segment forces that agree within a documented tolerance — sanity-checks that proxy sync + wrench harvest are not leaking energy across the boundary.
- [ ] **Constraint-Lagrangian caveat documented:** `measure_fruiting_forces` docstring explicitly states it returns VBD penalty forces, not solved-constraint reactions, and links to the M4 calibration milestone.
- [ ] **Commands documented:** `README.md` and this file's "Agent execution notes" record the new `uv run …` entry-points (generate + sim with FR3, both 1.a and 1.b), per `.cursor/rules/readme-runtime-verification.mdc`.

**Constraints / notes for implementers:**

- Newton-native by default; do **not** introduce a separate MuJoCo binary or runtime. `SolverMuJoCo` already lives in Newton; that is the only "MuJoCo" surface in M1.
- Keep all new logic under `apple_pick_sim/`. The only justifiable `newton/` edits are if `solver.joint_penalty_k` lookup needs a kernel for parity with the rest of the example (the existing TODO in `example_apple_stem.py::step` already notes this) — open a focused PR if it does.
- The two-`Model` coupling is **one-way per step with a one-step lag**. Do not attempt fixed-point iteration inside a step in M1 — accept the lag, document the implied stability/timestep constraint, and revisit only if a scenario diverges.
- Proxy bodies must mirror the *effective* inertia of the relevant robot subchain (often dominated by the hand + chain reduced-mass terms), not the raw link inertia, so that VBD-side contact/joint solutions remain well-conditioned. Tune by test, not by guess.
- Per `.cursor/rules/test-driven-development.mdc`, ship each slice **red → green**: write the failing wrench-acceleration test before the helper, the failing force-rises-with-load test before generalizing `get_forces()`, and the failing coupling-consistency test before wiring 1.b.
- Deterministic settings only (seeded P0 instance, fixed substep count, fixed wrench schedule for tests). Do not introduce viewer-coupled randomness into pytest paths.
- The penalty-force readout is **adequate for break detection and qualitative load-distribution work in M1**, but **not** a calibration-grade signal. M4 (calibration) inherits the responsibility to either move to a constraint-Lagrangian readout or document why the penalty proxy is sufficient.

**Next actions (ordered, small slices):**

1. [ ] **Slice 1 — Floating-EE handle in 1.a (single-`Model`, VBD-only):** Extend `apple_pick_sim/fruiting_system.py` to add an opt-in free-floating rigid body (the "handle") fused to the apple via `add_joint_fixed`. No FR3, no proxies, no MuJoCo. Verify the existing P0 deterministic rollout still passes with the handle present.
2. [ ] **Slice 2 — End-effector wrench helper + acceleration test:** Add `apply_ee_wrench(state, ee_body, wrench_world)` and the analytic-acceleration pytest on 1.a. Demonstrate driving the apple in 1.a with a constant pull.
3. [ ] **Slice 3 — Generalize `get_forces` to `measure_fruiting_forces`:** Per-segment cable-joint readout + apple↔handle reaction; add the force-rises-with-load pytest on 1.a. Replace the inline print loop in `step()` with a structured return.
4. [ ] **Slice 4 — Proxy primitives (incl. VBD wrench-readout spike):** Add `sync_proxy_state` Warp kernel and `harvest_proxy_wrenches(...)` under `apple_pick_sim/`. **First sub-task**: a short, focused spike that establishes which of the three options above (direct read / narrow public extension / velocity-delta reconstruction) `harvest_proxy_wrenches` will use; pick the cheapest one that passes a momentum-balance unit test on a mocked 1-body VBD scene. Unit tests exercise both helpers in isolation (synthetic robot states, no MuJoCo solver yet). This isolates the trickiest part of 1.b.
5. [ ] **Slice 5 — FR3 chain scenario (1.b, full two-`Model` staggered loop):** Build `robot_model` with the FR3 URDF, build `cable_model` with the proxy hand body fixed-jointed to the apple, run the 6-step protocol headlessly for *N* frames, port the wrench helper to write on the FR3 TCP `body_f`, confirm 1.a tests still pass on 1.a and add the coupling-consistency test on 1.b.
6. [ ] **Slice 6 — Commands + docs:** README and Agent execution notes updated and verified per `.cursor/rules/readme-runtime-verification.mdc`.

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
- Contact-driven grasping with the gripper fingers (deferred; revisit when M5 needs it, or earlier as a backlog item if M1 force scenarios are insufficient).
- `SensorContact` integration for apple/gripper/branch contacts (deferred — rigid attach makes it unnecessary for the M1 success criteria).
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

- Simulation / project code: `apple_pick_sim/`
- Physics engine (submodule, vendored): `newton/` — avoid drive-by edits; see `.cursor/rules/apple-pick-sim.mdc`

**How to validate changes:**

- Install / sync: `cd newton && uv sync --extra examples`
- Run example sim (smoke): from repo root, `uv run --directory newton python ../apple_pick_sim/example_apple_stem.py`
- Tests (Newton / shared env): `uv run --directory newton python -m newton.tests` (narrow with path/file when iterating, e.g. `uv run --directory newton python -m newton.tests -k test_cable`)
- P0 fruiting-system tests: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing` (from repo root; `PYTHONPATH` ensures `apple_pick_sim` is importable; `--directory newton` sets the uv project but cwd becomes `newton/`)

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict.
- A change requires **policy or product** decisions (scope, user-visible behavior, compatibility).
- A task requires **network credentials**, **paid APIs**, or **destructive** operations.

**When unsupervised is expected:**

- You may complete the **next unchecked slice** in “Current focus” using TDD and project rules.
- You may fix **small obvious blockers** uncovered by that slice (tests, imports, typos) if they are necessary for the slice to be correct.
- You should **not** start a new milestone or backlog item without maintainer direction unless this file explicitly says otherwise.
