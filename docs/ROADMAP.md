# Roadmap



## Document status


| Field            | Value                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Last updated** | 2026-05-28 (**[M1] Done** + coupling hardening landed; **[M2] active** — **M2.1 Done**; next: **M2.0** ADR → **FID Gym env** + **SKRL** → FIM / \pi_{\mathrm{exp}}; **FD gradient toys** (θ recovery + FIM action grid) documented under [M2]. **VIC** spec: `**docs/variable-impedance-teleop.md`**. **In-process mega plant** = prototype only (not Gym critical path). **[P0] Done**.) |
| **Owner**        | Abhinav                                                                                                                                                                                                                                                                                                                                                                                   |
| **Vision**       | See `docs/VISION.md`                                                                                                                                                                                                                                                                                                                                                                      |


---

## How this roadmap is structured


| Section                   | Purpose                                                                 |
| ------------------------- | ----------------------------------------------------------------------- |
| **Current focus**         | Single source of truth for what to do *now*. Keep it short.             |
| **Milestones**            | Phased outcomes from vision to implementation. Each has a clear “done”. |
| **Backlog**               | Unordered or lower-priority ideas — not active work.                    |
| **Agent execution notes** | How to run tests, where code lives, when to stop and ask.               |


---

## Sequencing (how this file maps to `docs/VISION.md`)

Owner intent drives phase order (vision outcomes stay valid; **order** is explicit here):

1. **Done — Outcome 1 ([P0]):** **Variational geometry** and **joint-level force readouts** shipped; refactor (collision/readout API, `measure_fruiting_forces`, solver damping, docs) landed. Optional P0 stretch (1.a floating EE, force-rises-with-load, richer cable scalars) **deferred**. See `docs/VISION.md` *Procedural fruiting variance*; archive under [P0].
2. **Done — Outcome 2 (manipulation stack, [M1]):** Two-`Model` `**SolverMuJoCo` + `SolverVBD`** coupling, FR3 + TCP teleop, proxy wrench exchange, structural layout (`fruiting_system/`, `coupled_fruiting/`, GPU hot path). Maintainer exit 2026-05-25; deferred stretch (EE–apple contact scenarios, formal arm readouts API) documented under [M1]. See [M1].
3. **Now — Outcome 3 (learning, [M2]):** **Two Gymnasium envs** over [M1] physics — **1×1 coupled** (`ApplePickCoupled-v0`, M2.1) for rollouts / \pi_{\mathrm{exp}}, and a **FID env** for finite-difference probes (`**fd_ghost`**, `**fd_mega_same_u**` semantics via **subprocess parallelism**, not a single mega `Model`). **[SKRL](https://skrl.readthedocs.io/)** (or equivalent) **subprocess vectorization** for both envs. Then **Fisher-information** exploration per [ASID](https://openreview.net/forum?id=jNR6s6OSBT) with **finite-difference** sensitivities (no Newton autodiff through coupled **VBD + MuJoCo**). **Dict** observations; **parity tests** on the 1×1 env. **M3 handoff:** flat \theta in rollouts/`info`.
4. **Next — Outcome 5 (sim tuning, [M3]):** **Simulation parameter identification / calibration** using trajectories from \pi_{\mathrm{exp}} (and later real logs in [M4]); update \theta with documented metrics via the **same finite-difference / black-box** path as [M2] (no reliance on Newton autodiff through the coupled stack).

Later vision phases (real-data collection [M4], final pick policy [M5]) follow once M2–M3 contracts exist.

---

## Current focus

**Active milestone:** [M2] — **RL infrastructure & Fisher-information exploration** (Gymnasium + ASID).

**In one sentence, the goal right now:** Lock **M2.0** (\theta, y, FD protocol), ship **two Gymnasium envs** (1×1 coupled + FID), parallelize both with **SKRL subprocess workers**, implement **FIM / FD** without a **mega-model**, then train **\pi_{\mathrm{exp}}** (**M2.3**).

**Build on (do not reimplement):** [M1] stack — `CoupledFruitingScene.coupled_substep`, `build_coupled_fruiting_fr3`, `Fr3EEVelocityController` / `EEVelocity`, `measure_fruiting_forces`, `sample_params` / `params_fingerprint`. **M2.1** — `apple_pick_gym/` (`ApplePickCoupledEnv`, `ApplePickCoupled-v0`, parity + `check_env`). Architecture: `**docs/mujoco-vbd-coupling-architecture.md`**. **Post-grasp VIC (planned):** `**docs/variable-impedance-teleop.md`** — \mathbf{w}*{\mathrm{total}} = \mathbf{w}*{\mathrm{transferred}} + \mathbf{w}_{\mathrm{applied}} on TCP `body_f`, `robot_kinematic_mode=False`. **ASID reference:** [WEIRDLabUW/asid](https://github.com/WEIRDLabUW/asid). **Parallel RL:** **[SKRL](https://skrl.readthedocs.io/)** `num_envs` / subprocess env runners over Gymnasium. **Sensitivity:** **finite differences** only (no Newton autodiff through coupled **VBD + MuJoCo**).

**Strategic choice — skip mega-model (for now):**


| Approach                                                                                               | Status                                     | Notes                                                                                                                       |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **Single `Model` with N instances** (one `robot_model` + one `cable_model`, batched `coupled_substep`) | **Prototype landed** (not Gym FID default) | `MegaCoupledFruitingScene`, `mega_fd_step`, tests + keyboard example; revisit for FID env only if subprocess FD is too slow |
| **W subprocess × 1 VBD × 1 MuJoCo**                                                                    | **Active**                                 | Default for \pi_{\mathrm{exp}} rollouts and `**fd_mega_same_u*`*                                                            |
| **FID env + `fd_ghost`**                                                                               | **Active**                                 | 1 arm drives EE; plant-side FD via coordinated workers (see below) — not a unified mega builder                             |


**Two Gymnasium environments (M2.2):**


| Env (registered id TBD)                  | Physics                  | Role                                                                                  |
| ---------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `**ApplePickCoupled-v0`** (shipped M2.1) | **1 VBD × 1 MuJoCo**     | Nominal coupled rollouts, \pi_{\mathrm{exp}} training, parity / `check_env`           |
| `**ApplePickFID-v0`** (new)              | FD probe layout per mode | Finite-difference columns for FIM; **not** a drop-in replacement for coupled training |


**FID env — FD modes** (implement as `fd_mode` or equivalent; semantics unchanged, **implementation = subprocess / SKRL**, not mega `ModelBuilder`):


| Mode                            | Intended physics semantics                                                                                               | Subprocess layout (target)                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**fd_ghost`**                  | 1 MuJoCo arm; perturbed plants \theta_k; **same** x^{\mathrm{ee}}_{0:H} on all instances; **nominal-only** harvest → arm | **1 leader** coupled worker (arm + \theta_0) + **(W−1)** plant workers receiving synced TCP / proxy state each step (or agreed in-process plant batch **only inside FID env** — still **no** N-arm mega model) |
| `**fd_mega_same_u`**            | N arm↔plant pairs; **same** u_t every step; y_k = y(\theta_k, u_{0:t})                                                   | **W** × 1×1 coupled workers; SKRL (or wrapper) **broadcasts** the same `action` to all; per-worker `info["theta"]` differs                                                                                     |
| `**fd_replay`** (optional gold) | Record u_{0:H} on nominal env; replay on perturbed \theta_k                                                              | Sequential or parallel **1×1** rollouts — regression / spot-check only                                                                                                                                         |


**Parallelization:** Use **SKRL** (subprocess env pool) for **both** `ApplePickCoupled-v0` and `ApplePickFID-v0` training / FD sweeps. Learner batches policy inference; workers own GPU context (document device pinning per worker). Gym remains **one env instance = one 1×1 coupled scene** unless FID `fd_ghost` documents a narrow multi-plant exception.

**Next up (ordered — [M2]):**

1. [ ] **M2.0 — Interface ADR** (`docs/`):
  - **Control:** Gym `step` = `SUBSTEPS_PER_FRAME` × `coupled_substep(SUB_DT)`; **M2.1 default:** kinematic teleop (`robot_kinematic_mode=True`); same u_{0:H} ⇒ same x^{\mathrm{ee}}_{0:H} when one arm leads. **VIC path (post-grasp):** see `**docs/variable-impedance-teleop.md`** — dynamic arm, total TCP wrench, action u may include K,D (or presets); FD mode `**fd_mega_same_u**` when VIC is on.
  - **\theta:** `pack_theta` / `unpack_theta`; build-time perturbation per worker; `info["theta"]` + `params_fingerprint`.
  - **y, \Sigma, FD protocol:** fixed `seed`, fixed action schedule, headless; how FID env aggregates y_k across workers into J columns.
  - **Env split:** document `ApplePickCoupled-v0` vs `ApplePickFID-v0` obs/action/`info` contracts; SKRL runner config (W, devices, `fd_mode`).
  - **Deferred:** mega-model builder, `benchmark_mega_fd`, `train_vector_mega` (backlog pointer).
2. [ ] **M2.2a — `ApplePickFID-v0`:** `apple_pick_gym/` FID env + registration; modes `**fd_ghost`**, `**fd_mega_same_u**`; tests vs sequential 1×1 FD on toy |\theta| (2–3 dims).
3. [ ] **M2.2b — Identification module (no Gym):** `apple_pick_sim/identification/` — [x] `theta_recovery` toy; [ ] `rollout_features` / FIM scalar \(\mathcal{I} \approx J^\top \Sigma^{-1} J\) for FID env outputs.
4. [ ] **M2.2c — SKRL integration:** minimal trainer script + smoke (W subprocess envs); document `uv` deps and run command; verify both env ids under SKRL.
5. [ ] **M2.3 — Train \pi_{\mathrm{exp}}:** SKRL + `ApplePickCoupled-v0` (W workers); FIM reward via `ApplePickFID-v0` / identification module on schedule; headless smoke.

**M2.1 (done — archive):**

- **M2.1 — Gymnasium environment:** `apple_pick_gym/` — `ApplePickCoupledEnv`, registered `ApplePickCoupled-v0`; physics via public `apple_pick_sim` APIs only.
- **M2.1a — Placeholder `Dict` obs:** `dummy` + `schema_version` in `observation_space`.
- **M2.1b — `Discrete(13)` keyboard commands** → `EEVelocity` → direct-joint teleop.
- **M2.1c — Parity:** `apple_pick_gym/tests/test_apple_pick_coupled_env.py::test_env_parity_against_direct_coupled_sim`.
- **M2.1d — `check_env` smoke:** `test_check_env_smoke`.
- **M2.1e — Settle-then-weld on `reset`:** `fix_to_apple_warmup_substeps` + `settle_then_weld.py`; `reset(..., options={"ranges_path": ...})` for parity (2026-05-28).

**Explicitly not in M2:** Full sim-parameter optimizer ([M3]); real-robot deployment ([M4]); final pick policy ([M5]); Newton/Warp autodiff through coupled **VBD + MuJoCo** rollouts. **Do not** require Gym to execute existing M1 unit tests — require **behavioral parity** on canonical rollouts instead.

**Blockers (if any):** None for FID env spike on toy |\theta| (2–3 dims). FIM-on-reward wiring waits on M2.0 draft; SKRL dep pin in Newton `uv` env TBD in M2.2c.

**Landed in tree (2026-05-28, not a milestone exit — builds on [M1] / preps [M2]):**


| Slice                               | Location / docs                                                                                         | Role                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Explicit apple weight at TCP**    | `coupled_fruiting/explicit_load.py`, `harvest_stem_tension_for_tcp`                                     | Quasi-static support + moment when apple is prescribed (`fix_to_apple`); default on — `**docs/explicit-apple-load-tcp-harvest.md`**                                    |
| **Settle → weld init**              | `coupled_fruiting/settle_then_weld.py`; Gym `fix_to_apple_warmup_substeps`; `example_gym_keyboard.py`   | Two-build workflow; seeds welded scene from settled free-apple VBD                                                                                                     |
| **Stem harvest when apple exists**  | `builders` + `stem.py`; `DEFAULT_STEM_COUPLING_GAIN=1.0`                                                | TCP stem gather for any scene with an apple (not only `fix_to_apple`); velocity-delta only when **no** apple — `**docs/mujoco-vbd-coupling-architecture.md`** §4.2–4.3 |
| **Mega plant + fd_ghost prototype** | `fruiting_system/mega.py`, `mega_fd.py`, `MegaCoupledFruitingScene`, `example_mega_coupled_keyboard.py` | In-process N-column FD / FIM smoke; **not** the planned Gym FID path (subprocess + SKRL remains default) — `**docs/mega-coupled-cable-implementation.md`**             |


**Last completed milestone:** [M1] — maintainer exit **2026-05-25** (coupling, FR3 teleop, refactor layout, GPU hot path, hardening docs).

### Dual envs, FD modes, SKRL (reference)

**Do not conflate** the two Gym envs or the FD mode names:


| Artifact                  | When to use                                                                   |
| ------------------------- | ----------------------------------------------------------------------------- |
| `**ApplePickCoupled-v0`** | Standard RL episode: one robot, one plant, \pi_{\mathrm{exp}} / task rollouts |
| `**ApplePickFID-v0**`     | FD column sweep / FIM reward; `fd_mode` selects ghost vs same_u layout        |
| **SKRL W workers**        | Parallel envs for **both** ids; policy batching in learner                    |


**VIC / \pi_{\mathrm{exp}} note:** Full design: `**docs/variable-impedance-teleop.md`**.

- **Scenario:** settle → `fix_to_apple` weld → teleop with **dynamic** arm (`robot_kinematic_mode=False`). Grasp = kinematic weld + co-teleport (not finger slip).
- **Move accordingly:** \mathbf{w}*{\mathrm{total}} = \mathbf{w}*{\mathrm{transferred}} (lagged stem harvest + `explicit_load`) **+** \mathbf{w}_{\mathrm{applied}} (VIC) → TCP `body_f` → `mj_solver.step`. Harvest writes **plant-only** wrench to `proxy_forces` for the next lag step.
- **FD:** variable-impedance exploration needs **input-fixed** FD (`**fd_mega_same_u`** or `**fd_replay**`), not `**fd_ghost**` alone. Kinematic M2.1 path: same u ≡ same x^{\mathrm{ee}} — `**fd_ghost**` suffices for early FIM smoke.
- **Implementation (physics slice done):** `ee_impedance.py`, wrench sum in `_mujoco_robot_substep_prefix`, `example_coupled_fruiting.py --dynamic-arm --vic`; tests in `test_vic_dynamic.py`. **Gym VIC deferred** to M2.0 ADR.

**Mega-model (deferred — backlog):** N instances in one `coupled_substep` remains a possible optimization; not on the critical path while subprocess + SKRL is sufficient.

---

## Milestones

Phases below follow **Sequencing** at the top: **[P0] Done** → **[M1] Done** → **[M2] active** (Gymnasium env + FIM / ASID) → **[M3]** (sim tuning) → **[M4]** (real data) → **[M5]** (final policy).

### [P0] — Variational fruiting system generation & force telemetry

- **Status:** Done (exited 2026-05-15; optional stretch deferred to backlog / [M1] needs)
- **Links:** N/A
- **Vision:** Outcome 1 (*Visual and structural variance*); success row *Procedural fruiting variance*; plus **instrumented loads** on the same VBD-built scene.

**Objective (variance — done):** From a **JSON** file of **min/max** bounds on geometry and material-like parameters, plus a **seed**, **variationally generate** a **Newton-ready scene** for a **fixed topology**: **primary branch (stiff)** → **secondary branch (softer)** → **short spur** → **stem** → **apple**, built like `**example_apple_stem.py`** (polylines + `**add_rod**` capsule chains + primitive fruit body, `**SolverVBD**`). Run a short reproducible simulation and prove variance and determinism with tests.

**Objective (force telemetry — core shipped):** **Fixed-joint** constraint wrenches via `**newton.solvers.SolverVBD.gather_joint_wrench_child_com`** (child at COM, world frame; Newton implementation under `newton/newton/_src/solvers/vbd/`) are wrapped in `**apple_pick_sim/vbd_fixed_joint_wrenches.py**`, re-exported from `**fruiting_system.py**`, exercised in **pytest**, and visualized as **FJ** plots in `**example_fruiting_system.py`**. Per-cable-joint **penalty-style** metrics remain available through the same `**ModelBuilder` + `SolverVBD`** patterns as `example_apple_stem.py` (`get_forces()`-style joint displacement × stiffness where used). **Structured readout:** `measure_fruiting_forces` returns fixed-joint records plus `cable_joint_indices` (cable scalars still via stem example patterns). **Deferred (not required for P0 exit):** optional stretch (**1.a**, **force-rises-with-load**, richer cable scalars in `measure_*`); maintainer may still run viewer smoke ad hoc.

**Definition of done (checklist):**

**Variance (complete):**

- Documented **JSON** range format and **seed** semantics (same file + seed → same instance). See `apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json` (tests) and `fruiting_system_ranges_example_variance.json` (example default).
- Generator implements the **five-part topology** above; primary vs. secondary **stiffness ordering** is respected in exported sim parameters (enforced in `sample_params`).
- **Per-instance Newton model** is produced via `**ModelBuilder`** (primitive-based geometry as above); `FruitingSystemScene` dataclass carries model, solver, state, body indices.
- **Newton rollout** with `**SolverVBD`** from the generated scene runs headlessly with deterministic settings for tests (`run_rollout`).
- Pytest(s): fixed `(range_json, seed)` → stable summary or fingerprint; varying seed or bounds → **distinct** geometry or labels (as asserted). Fruiting tests green (see `apple_pick_sim/tests/test_fruiting_system.py`).
- Documented command(s) for generate and generate+sim (see README and Agent execution notes).

**Force telemetry — core (complete):**

- **Fixed-joint wrenches:** `fixed_joint_wrenches_child_com_vbd` / `iter_fixed_joint_indices` and `**SolverVBD.gather_joint_wrench_child_com`** wired for fruiting-style **FIXED** joints; viewer logging in `example_fruiting_system.py`.
- **Pytest:** e.g. finite wrenches after substep, fixed-joint index iteration (`test_fruiting_system.py`).

**Exit before [M1] (process):**

- **Refactor + alignment (code shipped)** — collision/readout API, `measure_fruiting_forces`, README / WRENCH / tests; exited to [M1] 2026-05-15.
- **Manual verification** — *Deferred* (not a gate for [M1]); ad hoc: `example_fruiting_system.py` (seeds, `--enable-self-collision`, **FJ** traces).

**Optional stretch (promote during refactor only if needed):**

- **1.a Floating-EE handle** — opt-in free-floating body fused to apple via `add_joint_fixed` on the same VBD `Model` (no FR3).
- `**apply_ee_wrench` + acceleration pytest** on 1.a (see `newton/newton/tests/test_body_force.py` spirit).
- `**measure_fruiting_forces` (or equivalent)** — structured dict for cable joints + fixed joints; document penalty vs. gather semantics for [M4]. *(Fixed joints + cable **indices** shipped in `measure_fruiting_forces`; cable **scalar** forces still follow `example_apple_stem.py`.)*
- **Force-rises-with-load test** — monotonic load on apple; assert ordering vs. segment stiffness intent.
- **New `uv run …` paths** — only if the refactor adds entrypoints; verify per `.cursor/rules/readme-runtime-verification.mdc`.

**Constraints / notes for implementers:**

- Prefer `apple_pick_sim/`; `**newton/` edits** are justified when exposing or hardening **VBD joint-wrench** APIs (e.g. `gather_joint_wrench_child_com`) — keep PRs **focused** and covered by `newton` tests (see `newton/newton/tests/test_solver_vbd.py` for `gather_joint_wrench_child_com`).
- TDD and `uv` conventions apply.
- Start with **scalar stiffness / thickness / length** parameters if full anisotropic materials are not yet exposed in the chosen Newton API; document the mapping from JSON fields to sim bodies.
- **Triangle mesh import/generation** (OBJ/STL, high-res render meshes) is **out of scope for P0** unless promoted from **Backlog**; P0 “geometry” means the **capsule/sphere + cable joint** representation consistent with `example_apple_stem.py`.

**Next actions:** *(none — milestone complete; work continues under [M1])*

**Completed (archive):**

- JSON schema + fixtures (`apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json`, `fruiting_system_ranges_example_variance.json`)
- Generator module (`apple_pick_sim/fruiting_system.py`) with `load_ranges`, `sample_params`, `generate_scene`, `geometry_fingerprint`, `run_rollout`
- pytest tests in `apple_pick_sim/tests/test_fruiting_system.py` — variance + fixed-joint wrench coverage
- Example viewer + **FJ** fixed-joint wrench telemetry (`example_fruiting_system.py`)
- Refactor before M1: collision pipeline, `fruiting_fixed_joints`, `measure_fruiting_forces`, `make_fruiting_solver_vbd`, chain collision filters, docs

---

### [M1] — FR3 manipulation stack (two-`Model` coupling)

- **Status:** Done (exited 2026-05-25; maintainer sign-off; **[P0] Done** — reuse `fruiting_system` / wrench readouts; do not duplicate P0 DoD here)
- **Links:** Reference patterns: `newton/newton/examples/cloth/example_cloth_franka.py` (Featherstone+VBD coupling on one `Model` — *adjacent* but **not** the M1 recipe, see below), `newton/newton/examples/ik/example_ik_franka.py` (FR3 URDF import + TCP body), `newton/newton/tests/test_body_force.py` (external wrench API), and the **"two-`Model` staggered coupling skeleton"** at the bottom of this section (authoritative pattern for M1).
- **Vision:** Outcome 2 (*Manipulation stack*). MuJoCo enters as `**SolverMuJoCo` inside Newton** — there is **no separate MuJoCo runtime** in this milestone.

**Objective:** Build on **[P0]** by integrating a **Franka FR3** with a **custom end-effector** (URDF/shape additions under `apple_pick_sim/` or documented assets), **rigid contact** between the **end-effector** and the **apple** in at least one validated scenario (implemented as stiff body–body contact on `**cable_model`** between the apple and a **geometry-equipped gripper proxy**, and/or on the MuJoCo robot side per design — **not** soft multi-finger grasping), and the **two-`Model` staggered coupling** (`**SolverMuJoCo`** + `**SolverVBD**`, proxy bodies + one-step lag). **Apply end-effector wrenches** on the FR3 for tests. Reuse **[P0] VBD readouts** (**fixed-joint** wrenches via `**gather_joint_wrench_child_com`** / `vbd_fixed_joint_wrenches.py`, plus cable-joint **penalty** proxies where used). **Add robot-arm force instrumentation** on the **coupled** stack (e.g. `body_f` / joint torques / agreed MuJoCo-side signals) so interaction loads are visible on **both** models. Deterministic tests for **proxy sync**, `**harvest_proxy_wrenches`**, coupling sanity, and the **EE–apple contact** load path. A **fixed** apple↔gripper-proxy joint may remain for **regression** tests alongside **contact** scenarios.

**Architecture (authoritative for M1):**

The FR3 (rigid articulated robot) and the rod backbone (cable joints) are integrated by **different solvers on two separate `Model`s**, coupled at a per-step boundary by **proxy bodies + one-step-lag staggered wrench exchange**. The motivation is concrete:

- `SolverMuJoCo` does **not support `JointType.CABLE`** (`newton/_src/solvers/mujoco/solver_mujoco.py:292`) → the rod backbone *cannot* live in the MuJoCo model.
- `SolverVBD` *can* simulate revolute chains but is not the natural fit for an articulated-robot tracking controller → the FR3 *should* live in the MuJoCo model.
- Newton's "two solvers on one `Model`" pattern (e.g. `example_cloth_franka.py`, which uses `SolverFeatherstone + SolverVBD`) is therefore inappropriate here; we use **two `Model`s** instead, mirroring the relevant robot body or bodies as **proxy rigid bodies** inside the VBD `Model`.

**Per-`Model` ownership (M1):**


| `Model`       | Solver         | Contents                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `robot_model` | `SolverMuJoCo` | FR3 articulation (**URDF** + **custom end-effector** geometry as needed). Apple body is **not** here.                                                                                                                                                                                                                                                                                    |
| `cable_model` | `SolverVBD`    | Rod backbone (primary → … → stem) with cable joints, the apple rigid body, and **proxy** copies of the FR3 link(s) the cable must see (gripper / TCP) with **collision shapes** so **rigid EE–apple contact** can be simulated. **Fixed** `add_joint_fixed(gripper_proxy, apple)` may still be used for **baseline** scenarios; **contact** scenarios add/remove constraints per design. |


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


| Path                    | When                                                        | Harvest                                                                                                                                                                          | Sync                                                                                                                                |
| ----------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Stem-harvest at TCP** | Cable scene has an **apple** (`stem_apple_joint_index` set) | `harvest_stem_tension_for_tcp` — stem→apple FIXED via `gather_joint_wrench_child_com`; optional explicit apple weight; `stem_coupling_gain` (default **1.0**), force/torque caps | With `fix_to_apple=True`: co-teleport proxy + apple from TCP; with `fix_to_apple=False`: **proxy-only** sync, dynamic apple on stem |
| **Velocity-delta**      | **No apple** on cable model                                 | `harvest_proxy_wrenches` — option 3 from Slice 4 spike                                                                                                                           | `sync_proxy_state` — proxy only                                                                                                     |


See `**docs/mujoco-vbd-coupling-architecture.md`** for per-model ownership and the full substep diagram.

**Current progress (2026-05-22):**

- Slice 1 — `proxy_coupling.py` (`sync_proxy_state`, velocity-delta harvest, `align_proxy_body_q_prev_for_vbd`); `test_proxy_coupling.py`.
- Slice 2a — `generate_coupled_cable_scene` / `CoupledCableScene`; `test_coupled_cable_scene.py`.
- Slice 2b — `CoupledFruitingScene` (`mujoco_substep` / `vbd_substep` / `coupled_substep`), placeholder + stem-harvest paths; `example_coupled_fruiting.py`; coupled/stability tests; `verify_coupling.py`.
- Slice 2 — FR3 load (`testfr3_resolved.usda`, `build_coupled_fruiting_fr3`); `docs/fr3-usd-import-implementation.md`.
- Slice 2c — TCP velocity teleop + ghost proxy tracking (`--fr3-keyboard`, `--only-mjc` and full coupled).
- Slice 2d — **Force transfer accepted** (lagged harvest, apply to `body_f`, sync, debug plots).
- **Slice 2f — Structural refactor** — `fruiting_system/`, `coupled_fruiting/`, `robot/fr3_robot/`; see `**docs/slice-2f-structural-refactor.md`**. Residual items in `**refactor.md**` are non-blocking.
- **Slice 2g — GPU optimization** — `docs/gpu-coupling-optimization.md`, device hot path, `benchmark_coupling.py`.
- Slice 2e — Hardening: `docs/slice-2e-hardening.md`, device hot-path kernels, `benchmark_coupling.py`, FR3 long-horizon + `slow` pytest marker.
- Slice 3 — README / Agent execution notes polish (optional; not a gate for M2).

**Known limitations (placeholder TCP and coupled FR3):**

- **Builders default to `fix_to_apple=False`** (proxy-only sync; **stem harvest at TCP** whenever an apple exists). Opt in `GripperProxyConfig(fix_to_apple=True)` for welded grasp + apple co-teleport; use `**settle_then_weld`** / Gym `**fix_to_apple_warmup_substeps**` for quieter welded starts.
- `**fix_to_apple=True` + placeholder free TCP** is often unstable (huge **QACC**, stem-harvest saturation) — see headless `test_coupling_stability` / `verify_coupling.py` notes.
- **Scenes without an apple** still use velocity-delta harvest (~mg–centi-Newton in headless checks). **FR3 keyboard smoke** defaults to `fix_to_apple=False` with stem path when the fixture includes an apple.
- **FR3 TCP velocity teleop + force transfer:** **Accepted** for M1 baseline (`--only-mjc` and full coupled, `--debug-coupling-forces` for wrench plots). Refactor must **preserve** staggered semantics unless explicitly changed with tests.
- `**--fr3-direct-joints`:** kinematic `joint_q` writeback for arm debugging; not the primary teleop path. **Post-grasp VIC** requires `**robot_kinematic_mode=False`** and TCP `body_f` wrench sum — see `**docs/variable-impedance-teleop.md**` (builders/examples default kinematic).
- `**test_coupling_stability.py**` asserts finiteness and cap compliance, **not** quiescent MuJoCo motion or small TCP velocities; passing tests does **not** imply a stable interactive demo.
- `**--enable-self-collision` / `--mujoco-viewer`** are not the primary instability drivers; they only change cable collisions or add a second viewer window.
- **Smoke paths:** `--only-vbd` (cable only); `--robot fr3 --fr3-keyboard` (full coupled teleop); `--robot fr3 --only-mjc --fr3-keyboard` (robot + proxy sync only); `--debug-coupling-forces` for wrench plots.

**Exit note (2026-05-25):** Coupled FR3 + proxy coupling accepted for learning stack; EE–apple **contact** scenarios and formal **arm readouts API** deferred (not required for [M2] env wrapper).

**Slice 2f — definition of done (checklist):**

- `**refactor.md` tasks** implemented in agreed order; each slice keeps `**apple_pick_sim/tests/`** green and public imports stable (or documents a one-shot migration).
- **Coupling semantics preserved:** same staggered apply → MuJoCo → sync → VBD → harvest; no behavior change unless a refactor slice explicitly targets physics/API with new tests.
- **Naming / layout:** `fruiting_system/` package, `coupled_fruiting/` package, `proxy_coupling` mirror/harvest kernel names — see `**docs/slice-2f-structural-refactor.md`**. (Host-sync removal is **Slice 2g**.)
- **Remaining 2f tasks** in `**refactor.md`** (2f-E and maintainer edits).

**Slice 2g — definition of done (checklist):** *(GPU path — after 2f layout stabilizes; may start profiling in parallel)*

- **Profiler harness documented:** agreed stack for this repo (at minimum: `**diagnostics/benchmark_coupling.py`** on CUDA with `wp.synchronize()`; optional **Nsight Systems** / **Nsight Compute** or Warp timing markers for kernel-level attribution). Commands live in **Agent execution notes** and a short `**docs/gpu-coupling-optimization.md`** (or a dated subsection in `docs/mujoco-vbd-coupling-architecture.md`).
- **Baseline captured on target GPU:** ms/substep (placeholder + FR3), default `sim_substeps`, warmup/bench substeps recorded in docs **before** optimization PRs land.
- **Hot-path inventory:** list host↔device syncs and CPU fallbacks per substep (`coupled_fruiting.py`, `proxy_coupling.py`, `fr3_robot.py` placement/teleop, collision triggers); prioritize by profiler cost.
- **GPU migration:** replace top-cost CPU paths with Warp kernels or device-resident arrays where Newton/Warp APIs allow; **no correctness regressions** — existing proxy/coupling/stability tests stay green; add tests when a kernel replaces logic that was only covered indirectly.
- **Measured progress:** each optimization PR cites profiler/benchmark **before → after** (same machine/GPU/driver note in doc); at least one **≥10%** ms/substep win on the coupled hot path **or** documented blocker (e.g. MuJoCo solver host-bound).
- **End-to-end device goal:** coupled substep runs with **minimal per-substep host reads** (debug/viewer paths may still `.numpy()`; pytest headless paths should not require full-state host copies unless asserting values).

**Slice 2e — definition of done (checklist):** *(hardening — after 2f / 2g per maintainer)*

- **Code map / reader docs:** `docs/slice-2e-hardening.md`, `docs/mujoco-vbd-coupling-architecture.md` §5.2 (device align kernel).
- **Coupling error-free (headless):** `test_fr3_coupled_substep_long_horizon_finite` (`slow`); existing stability/proxy tests green.
- **Controllers error-free (headless):** extend existing `test_fr3_`* coverage where gaps remain (deferred).
- **Benchmarking:** `diagnostics/benchmark_coupling.py`; baseline table in `docs/slice-2e-hardening.md`; commands below.

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

**Force readout:** **[P0]** supplies **fixed-joint** wrenches via `**SolverVBD.gather_joint_wrench_child_com`** (`vbd_fixed_joint_wrenches.py`) and cable-joint **penalty** proxies where that convention holds. **M1 adds** **robot-side** readouts: document how to recover **joint torques**, **link wrenches**, or `**body_f`**-consistent totals on `**robot_model**` after each coupled substep so **forces on the arm** from apple interaction are **measured** alongside VBD-side reactions. Document penalty vs. gather semantics for [M3] calibration.

**Attachment / interaction matrix:**


| Mode                        | What it is                                                                                                                                                         | Solvers needed                | When to use                                                                                                           | Slice         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------- |
| **1.a Floating-EE handle**  | Single free-floating rigid body on `cable_model` fused to the apple via `add_joint_fixed`; wrenches on that body. **No FR3.**                                      | `SolverVBD` only.             | Optional **P0 stretch** for isolated load curves; **not** assumed to exist for M1 unless promoted during P0 refactor. | [P0] optional |
| **1.b Full FR3 + coupling** | Two `Model`s; FR3 hand mirrored as **proxy** in `cable_model`; staggered protocol below. **Rigid EE–apple contact** (and/or **fixed** apple↔proxy for regression). | `SolverMuJoCo` + `SolverVBD`. | Primary M1 manipulation scenario.                                                                                     | [M1]          |


**Definition of done (checklist):**

- **Custom end-effector:** FR3 import extended with **project-specific EE geometry** (URDF fragment, extra shapes, or documented asset) and tested load path.
- **Rigid EE–apple contact:** At least one headless scenario where apple and EE (or **gripper proxy** on `cable_model`) interact through **rigid / stiff contact**; contact pipeline and filters documented.
- **Robot-arm force readouts:** Documented + tested extraction of **interaction forces/torques on the FR3 chain** from the **coupled** sim (agreed links / joints / `body_f` / MuJoCo diagnostics).
- **Builder integration (robot half):** :func:`~apple_pick_sim.coupled_fruiting.build_coupled_fruiting_fr3` + :func:`~apple_pick_sim.fruiting_system.generate_coupled_cable_scene` produce **1.b** (`robot_model` + `cable_model`, FR3 + proxy + placement). **Remaining:** EE–apple contact scenarios and documented `(range_json, seed)` entry if not already covered by coupled example flags.
- **Two-`Model` staggered step loop (placeholder):** :class:`~apple_pick_sim.coupled_fruiting.CoupledFruitingScene` + :meth:`~apple_pick_sim.coupled_fruiting.CoupledFruitingScene.coupled_substep` with placeholder TCP; `example_coupled_fruiting.py`; `test_coupled_fruiting_system.py`. **Remaining:** same loop on **FR3** with stability acceptance (Slice 2).
- **End-effector wrench on FR3:** Documented application of wrenches to the MuJoCo robot `body_f` (TCP / agreed link), once per substep, with tests that build on **[P0]** patterns where applicable.
- **Reuse [P0] telemetry on 1.b:** **[P0]** fixed-joint + cable-side readouts run on `cable_model` (including apple↔**gripper-proxy** when that joint exists); coupling / load-path test compares against an agreed baseline (e.g. fused-proxy regression, P0-only recording, or **1.a** if present).
- **Constraint-Lagrangian caveat documented:** Where cable joints still use penalty proxies, docstrings state that and link **[M4]**; fixed-joint gather path documented per Newton API.
- **Commands documented:** `README.md` and this file's "Agent execution notes" record the new `uv run …` entry-points (generate + sim with FR3 **1.b**), per `.cursor/rules/readme-runtime-verification.mdc`.

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
- Full `**SensorContact`** integration for every branch/fruit contact (defer detailed sensor plumbing unless a slice needs it; **EE–apple** may use the standard collision pipeline).
- Logging the commanded EE wrench into a versioned rollout schema (lives in M2 once the RL contract is promoted).
- True constraint-Lagrangian joint reactions (M4 calibration may promote this).

**Completed (archive as you go):**

- Slice 1 — `apple_pick_sim/proxy_coupling.py` + `tests/test_proxy_coupling.py` (2026-05-15).
- Slice 2a — `generate_coupled_cable_scene` / `CoupledCableScene` + `tests/test_coupled_cable_scene.py` (2026-05-15).
- Slice 2b (placeholder) — `coupled_fruiting.py`, `example_coupled_fruiting.py`, stem-harvest + unified sync, `tests/test_coupled_fruiting_system.py`, `tests/test_coupling_stability.py`, `diagnostics/verify_coupling.py`, `docs/mujoco-vbd-coupling-architecture.md` (2026-05-18).
- Slice 2c (core) — `Fr3EEVelocityController`, `--fr3-keyboard`, `test_fr3_ee_velocity_controller.py`, `coupling_force_debug.py` + `test_coupling_force_debug.py` (2026-05-19; `3f24330` ghost-proxy teleop).
- Slice 2d — coupled force transfer accepted (2026-05-22).

---

### [M2] — RL infrastructure & Fisher-information exploration (Gymnasium + ASID)

- **Status:** In progress (active since 2026-05-25)
- **Links:** [Gymnasium — basic usage](https://gymnasium.farama.org/introduction/basic_usage/) · [ASID (ICLR 2024 oral)](https://openreview.net/forum?id=jNR6s6OSBT) · [arXiv:2404.12308](https://arxiv.org/abs/2404.12308) · [project page](https://weirdlabuw.github.io/asid/) · [reference code (FIM objective)](https://github.com/WEIRDLabUW/asid)
- **Vision:** Outcome 3 (*Learning infrastructure*); Fisher information as in `docs/VISION.md` glossary.

**Objective:** Phase **1** — **[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)** environment over the **[M1]** coupled simulator (**done — M2.1**). Phase **2** — **exploration policy** \pi_{\mathrm{exp}} and **Fisher information** per ASID, using **finite-difference** sensitivities of rollout features y(\theta) w.r.t. sim parameters \theta (ASID’s approach when the simulator is not differentiable). **Validation:** parity tests (M2.1c) and **existing `apple_pick_sim/tests/`** stay on the direct path. **M3 handoff:** same flat \theta layout and FD machinery reused for calibration; episode `info` logs `**theta`** and `params_fingerprint`.

**Architecture (M2):**


| Layer              | Location                                   | Role                                                                                        |
| ------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------- |
| Physics / coupling | `apple_pick_sim/`                          | `CoupledFruitingScene` — **1 VBD × 1 MuJoCo** per worker (no mega `Model` on critical path) |
| FD / FIM           | `apple_pick_sim/identification/` (planned) | `rollout_features`, J, FIM scalar from FID / worker outputs — **no Gym import**             |
| Gymnasium `Env`    | `apple_pick_gym/`                          | `**ApplePickCoupled-v0`** (1×1, M2.1); `**ApplePickFID-v0**` (FD modes)                     |
| Parallel RL        | SKRL + Gymnasium                           | Subprocess **W** envs for **both** env ids; batched \pi in learner                          |


**Default control contract (unless M2.0 ADR changes it):**

- In **M2.1**: action is a **single keyboard-style command** (`Discrete(13)` mapping to \pm X,\pm Y,\pm Z,\pm \mathrm{rotX},\pm \mathrm{rotY},\pm \mathrm{rotZ},noop) → `EEVelocity`, applied via `Fr3EEDirectJointController` + `CoupledFruitingScene.apply_fr3_ee_teleop_direct` with `robot_kinematic_mode=True`.
- **VIC (planned, post-grasp):** extend action with impedance params; dynamic arm + \mathbf{w}_{\mathrm{total}} at TCP per `**docs/variable-impedance-teleop.md`**; Gym parity tests remain kinematic until a separate dynamic contract is added.
- One Gym **step** = one control frame = `SUBSTEPS_PER_FRAME` × `coupled_substep(SUB_DT)` (same timing as `apple_pick_sim/tests/conftest.py`).
- After parity + ADR, may extend to MultiDiscrete (simultaneous keys) and/or switch to continuous `Box(6)` twist actions for learned controllers.

**ASID mapping (apple-pick context):**


| ASID stage                                                  | This repo (target milestone)     |
| ----------------------------------------------------------- | -------------------------------- |
| (0) Sim RL API (Gymnasium + parity)                         | **[M2]** M2.1                    |
| (1) Train \pi_{\mathrm{exp}} maximizing Fisher info in sim  | **[M2]** M2.2–M2.3               |
| (2) Deploy \pi_{\mathrm{exp}} in real, collect trajectories | **[M4]** (after M2 sim contract) |
| (3) System ID — refine sim parameters \theta                | **[M3]**                         |
| (4–5) Task policy in updated sim → real                     | **[M5]** (+ vision Outcome 6)    |


**Definition of done (checklist):**

- **Gymnasium env (M2.1):** `ApplePickCoupledEnv`, `ApplePickCoupled-v0`; headless default; physics only via `apple_pick_sim`.
- **Parity tests (M2.1c):** `test_env_parity_against_direct_coupled_sim`; M1 direct-path suite remains the regression gate.
- `**check_env` (M2.1d):** `test_check_env_smoke`.
- **M2.0 documented:** \theta, y, \Sigma, dual-env + SKRL layout, `fd_*` modes (short ADR under `docs/`).
- **FID Gym env (M2.2a):** `ApplePickFID-v0`; `**fd_ghost`**, `**fd_mega_same_u**`; FD tests on toy |\theta|.
- **FIM / identification (M2.2b):** \mathcal{I} or ASID \mathrm{tr}(\mathcal{I}^{-1}) proxy from J; module in `apple_pick_sim/identification/`.
- **FD gradient toys (M2.2b):** [x] `toy_theta_recovery.py` (1D k recovery, free + welded; `docs/toy-theta-recovery-implementation.md`) + [ ] `toy_fim_action_grid.py`; pytest: `test_toy_theta_recovery.py`.
- **SKRL smoke (M2.2c):** W subprocess workers for both env ids; documented `uv run` entrypoints.
- **\pi_{\mathrm{exp}} training (M2.3):** SKRL + `ApplePickCoupled-v0`; FIM via FID env on schedule.
- **M3 handoff:** flat `theta` in rollouts/`info`; FD module reusable for [M3] `identify_theta`.

**Constraints / notes for implementers:**

- **[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)** is the RL env API: use `terminated` and `truncated` separately; optional `TimeLimit` wrapper; `render_mode=None` for pytest.
- **Do not** move coupling logic into the Gym package; **do not** replace M1 tests with Gym-only runs — add **parity** tests instead.
- **Sensitivities:** **finite differences** only (ASID); no Newton autodiff through coupled VBD + MuJoCo for M2.
- **Order:** M2.0 ADR → FID env + toy FD tests → identification / FIM → SKRL smoke → \pi_{\mathrm{exp}}. **Do not** block on mega-model or mega benchmark.
- TDD: sequential 1×1 FD gold → FID env columns → FIM scalar → **toy optimizations (below)** → SKRL W-worker smoke.
- `**import gymnasium` only in `apple_pick_gym/`**; FD/FIM math lives in `apple_pick_sim/identification/`.
- See **Dual envs, FD modes, SKRL** under **Current focus** for `fd_ghost` vs `fd_mega_same_u`.

### FD / FIM gradient verification (toy optimization)

Beyond pytest (Jacobian vs sequential gold, `J == (y_i - y_0)/ε`), run **small end-to-end optimizations** to confirm FD columns are useful for descent—not only algebraically consistent. Planned headless entrypoints: `apple_pick_sim/examples/toy_theta_recovery.py`, `apple_pick_sim/examples/toy_fim_action_grid.py` (land in **M2.2b** with `identification/`).


| Toy                            | What it validates                                                                   | Stack                                                                                                                                                                                                                                                                                                           | Success criteria                                                                                                                                                                                                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(1) Parameter recovery**     | \partial y / \partial \theta for **identification** ([M3] preview)                  | **Cable-only OK:** `MegaCoupledCableScene` + `mega_fd_step` / sequential 1×1 gold (`test_mega_fd.py`); start with **one** dim, e.g. `primary.bend_stiffness` k                                                                                                                                                  | Pick true k^; record y^ = y(k^) after fixed substeps; from wrong k_0, run 5–10 **Gauss–Newton** steps k \leftarrow k - (J^\top J + \lambda I)^{-1} J^\top (y(k)-y^) using FD J (same \varepsilon as `fd_stiffness_param_columns`); **loss** |y(k)-y^| decreases; final |
| **(2) FIM action grid search** | \mathcal{I} = J^\top \Sigma^{-1} J ranks **informative motion** (exploration smoke) | **Coupled FR3:** `MegaCoupledFruitingScene`, welded + zero-g pattern in `test_mega_fd_kinematics.py`; grid one teleop knob (e.g. lateral `EEVelocity` v_y \in -0.15, 0, +0.15 m/s for one frame) → drive → `reset_perturbed_instances_to_nominal` → FD substeps → `trace(fim_step)` with documented `sigma_inv` | **Deflecting** action achieves **higher** `trace(fim_step)` than noop when stem–apple wrench rows dominate (`fix_to_apple`); consistent with nonzero |J| on force rows in `test_welded_restoring_force_and_fd_jacobian`                                                |


**Protocol (both toys):** fixed `seed`, fixed `SUB_DT` / `SUBSTEPS_PER_FRAME`, same state restore + same control across FD columns (`fd_ghost` kinematic path). Optional: forward vs **central** FD on \theta at the best point; \varepsilon sweep if columns are noisy.

**Not in scope for these toys:** full RL / SKRL, \mathrm{tr}(\mathcal{I}^{-1}) optimization, or action gradients \partial \mathcal{I}/\partial u (toy 2 uses grid search over u, not ascent).

**Next actions (ordered, small slices):** see **Current focus** (M2.0 → M2.2a → M2.2b → M2.2c → M2.3).

**Completed (archive as you go):**

- M2.1 — `apple_pick_gym/` env + registration (2026-05-27).
- M2.1a–M2.1d — placeholder obs, `Discrete(13)`, parity, `check_env` (2026-05-27).

---

### [M3] — Simulation parameter tuning (system identification)

- **Status:** Planned (after [M2] env + FIM infra; reuse **M2** \theta layout and **finite-difference** machinery)
- **Links:** ASID stage (3); vision Outcome 5 (*Calibration loop*)
- **Vision:** Outcome 5 — update sim parameters so sim–real (or sim–target) error drops on held-out data.

**Objective:** **Tune simulation parameters** \theta using trajectories from \pi_{\mathrm{exp}} (sim-first; then real logs from [M4]). Fit \theta so the coupled fruiting + FR3 sim matches observed poses, wrenches, contact events, or other agreed metrics. Use the **same finite-difference / black-box** sensitivity path as [M2] (coupled **VBD + MuJoCo** stack is not end-to-end differentiable in Newton). This is the engineering home for “simulation tuning” — not the exploration-policy trainer in [M2].

**Definition of done (checklist):**

- \theta update loop documented (optimizer, metrics, held-out eval).
- Quantitative **before/after** on held-out trajectories (sim replay or [M4] real segment).
- Sensitivity path documented: **finite differences** (and optional black-box search) using [M2] \theta / rollout protocol.

**Constraints / notes for implementers:**

- Reuse [M2] `**pack_theta` / `unpack_theta`** and FD module; avoid a second parameterization. Log flat `**theta**` on rollouts; avoid hard-coding non-identifiable combos.
- Start with **sim-only** identification (known \theta^, recover from noisy obs) before real data; first gate: **[M2] toy (1) parameter recovery** (FD / FIM gradient verification).

**Next actions (ordered, small slices):**

- Stub `identify_theta` API + test on 2–3 parameter toy once [M2] rollout schema exists.
- Wire fruiting `sample_params` / builder fields into \theta with clear bounds.

**Completed (archive as you go):**

- *(none yet)*

---

### [M4] — Real-world data collection & format alignment

- **Status:** Planned
- **Links:** ASID stage (2); vision Outcome 4
- **Vision:** Outcome 4.

**Objective:** Collect trajectories under the same \pi_{\mathrm{exp}} (or matched sensing) as simulation; formalize formats and ingestion for [M3] tuning.

**Definition of done (checklist):**

- Collection protocol documented; real logs validate against [M2] schema/versioning.

**Constraints / notes for implementers:**

- No production deployment or safety certification scope (see vision non-goals).

**Next actions (ordered, small slices):**

- Extend [M2] rollout schema for real-hardware fields once env contract is stable.

**Completed (archive as you go):**

- *(none yet)*

---

### [M5] — Final manipulation policy (RL in calibrated sim)

- **Status:** Planned
- **Links:** N/A
- **Vision:** Outcome 6.

**Objective:** Train a final apple-picking policy in simulation after calibration tightens sim–real-relevant parameters.

**Definition of done (checklist):**

- Policy trained against [M3]-calibrated sim; eval criteria agreed with maintainer.

**Constraints / notes for implementers:**

- Builds on [M2] harness and [M3]-calibrated sim; avoid one-off policy formats.
- **Default training layout:** SKRL **W** × `ApplePickCoupled-v0` (1 VBD × 1 MuJoCo per worker). Mega-model vector training remains **deferred** unless promoted from backlog.

**Next actions (ordered, small slices):**

- Defer until [M2]–[M4] exit criteria support a clear task success definition.

**Completed (archive as you go):**

- *(none yet)*

---

## Backlog (not active)

Unordered ideas. **Do not implement** unless promoted into a milestone and “Current focus”.

- *(Promoted to Slice **2f** — see `**refactor.md`**; maintainer expands/refines the list.)*
- *(Promoted to Slice **2g** — GPU profilers, device-resident coupling, benchmark baselines; not active until **Current focus** promotes it after 2f.)*
- **Former end-to-end “thin slice” stubs** (rollout log schema v0, real-data adapter stub, calibration comparison stub, scripted policy placeholder): useful when **M2–M4** are promoted; not required to finish **P0** fruiting tests.
- *(Promoted to [M2] — Fisher-information / ASID exploration objective.)*
- Additional manipulators or crops — only with explicit scope change (vision non-goals).
- **Triangle mesh export or import** (OBJ/STL, render/FEM meshes) alongside or instead of capsule primitives — promote only if a milestone needs it; P0 stays `**ModelBuilder*`* primitives per `example_apple_stem.py`.
- **Mega-model FD (Gym / SKRL critical path)** — batched `train_vector_mega` / `benchmark_mega_fd` as default FID backend. **Prototype exists** (`MegaCoupledCableScene`, `mega_fd_step`, `example_mega_coupled_keyboard.py`); promote to FID env only if subprocess + SKRL is too slow or maintainer re-scopes M2.
- **Variable-impedance teleop (dynamic arm)** — implement per `**docs/variable-impedance-teleop.md`**; promote into M2.0 ADR + example flags when slicing.

---

## Agent execution notes

**Repository layout (this project):**

- Simulation / project code: `apple_pick_sim/` (e.g. `fruiting_system/`, `coupled_fruiting/` incl. `proxy_coupling.py`, `examples/`, `vbd_fixed_joint_wrenches.py` — wraps `SolverVBD.gather_joint_wrench_child_com` from `newton/`)
- RL / Gymnasium adapter (M2): `apple_pick_gym/` — `ApplePickCoupled-v0`, `ApplePickFID-v0` (planned); depends on `apple_pick_sim`, not vice versa
- FD / FIM (M2.2, planned): `apple_pick_sim/identification/` — Jacobian / FIM from FID env or worker batches; **no** mega `coupled_substep` on critical path
- SKRL (M2.2c+): trainer scripts under `apple_pick_gym/` or `apple_pick_sim/examples/` (TBD in ADR); pin dependency in Newton `uv` env when added
- Physics engine (submodule, vendored): `newton/` — avoid drive-by edits; see `.cursor/rules/apple-pick-sim.mdc`

**How to validate changes:**

- Install / sync: `cd newton && uv sync --extra examples`
- Run example sim (smoke): from repo root, `uv run --directory newton python ../apple_pick_sim/examples/example_apple_stem.py`
- Tests (Newton / shared env): `uv run --directory newton python -m newton.tests` (narrow with path/file when iterating, e.g. `uv run --directory newton python -m newton.tests -k test_cable`)
- P0 fruiting-system tests: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing` (from repo root; `PYTHONPATH` ensures `apple_pick_sim` is importable; `--directory newton` sets the uv project but cwd becomes `newton/`)
- **Fast default (excludes `@pytest.mark.slow` long-horizon / mega FD / settle tests):** `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing -m "not slow"`
- **Slow-only gate:** `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -m slow -q -p no:launch_testing`
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
- Post-grasp VIC: `docs/variable-impedance-teleop.md`, `docs/vic-implementation.md`; example `--dynamic-arm --vic`
- VIC pytest gate: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_ee_impedance.py ../apple_pick_sim/tests/test_vic_dynamic.py -q -p no:launch_testing`
- M1 Slice 2f (structural refactor gates): `docs/slice-2f-structural-refactor.md` — fruiting: `pytest ../apple_pick_sim/tests/test_fruiting_system.py ../apple_pick_sim/tests/test_coupled_cable_scene.py -q`; coupled: `pytest ../apple_pick_sim/tests/test_coupled_fruiting_system.py ../apple_pick_sim/tests/test_coupling_stability.py -q`; proxy: `pytest ../apple_pick_sim/tests/test_proxy_coupling.py -q` (all with `PYTHONPATH=$(pwd) uv run --directory newton … -p no:launch_testing`)
- M1 refactor: read `**refactor.md`** and **Current focus** before structural edits; maintainer updates both when priorities change
- M1 work: follow **Current focus** and [M1] *Next actions*; add `uv run` entry-points here as slices land
- M1 explicit apple load: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_explicit_apple_load.py -q -p no:launch_testing`
- M1 settle-then-weld: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_settle_then_weld.py -q -p no:launch_testing`
- Mega plant / FD prototype (not Gym critical path): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_mega_coupled_fruiting.py ../apple_pick_sim/tests/test_mega_coupled_cable_scene.py ../apple_pick_sim/tests/test_mega_fd.py ../apple_pick_sim/tests/test_mega_fd_kinematics.py -q -p no:launch_testing`; keyboard smoke: `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_mega_coupled_keyboard.py --viewer null --num-frames 1` (see `**docs/mega-coupled-cable-implementation.md`**)
- M2 Gymnasium: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_gym/tests/ -q -p no:launch_testing` (from repo root; requires `gymnasium` in Newton env, e.g. `newton[dev]`)
- M2 FID env + identification tests (when present): `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_gym/tests/ ../apple_pick_sim/tests/ -k 'fid or finite_difference' -q -p no:launch_testing`
- M2 FD gradient toys: `docs/toy-theta-recovery-implementation.md`; `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_toy_theta_recovery.py -q -p no:launch_testing`; `PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/toy_theta_recovery.py` (add `--fix-to-apple` for welded wrench mode); [ ] `toy_fim_action_grid.py` (FR3)
- M2 SKRL smoke (when present): documented in M2.0 ADR / README (W workers, headless)
- M2 regression gate: **always** run M1 direct-path suite before merging M2 slices: `PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing`

**Stop and ask the maintainer when:**

- Vision vs roadmap vs code conflict.
- A change requires **policy or product** decisions (scope, user-visible behavior, compatibility).
- A task requires **network credentials**, **paid APIs**, or **destructive** operations.

**When unsupervised is expected:**

- You may complete the **next unchecked slice** in “Current focus” using TDD and project rules — for **[M2]**, follow **M2.0 → FID env → identification / SKRL → M2.3**; **do not** implement mega-model unless promoted from backlog; keep `**apple_pick_sim/tests/`** green; run `**apple_pick_gym/tests/**` when touching the adapter.
- You may fix **small obvious blockers** uncovered by that slice (tests, imports, typos) if they are necessary for the slice to be correct.
- You should **not** start a new milestone or backlog item without maintainer direction unless this file explicitly says otherwise.

