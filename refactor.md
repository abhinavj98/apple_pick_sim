# Refactor backlog (maintainer-owned)

**Last updated:** 2026-05-22  
**ROADMAP mapping:** [M1] **Slice 2f** = this file (structure, naming, import stability). **Slice 2g** = GPU / host-sync removal (separate checklist below; do not mix into 2f PRs unless a kernel is required for correctness).

**Read before large PRs:** `docs/ROADMAP.md` (Current focus), `docs/mujoco-vbd-coupling-architecture.md`, `docs/fr3-usd-import-implementation.md`, `AGENTS.md`.

---

## Summary (for the next agent)

### What exists today

The M1 **two-Model staggered coupling** stack is **shipped and accepted** (Slice 2d, 2026-05-22). Physics semantics must not change during 2f unless a slice adds explicit tests.

| Module | ~lines | Role |
|--------|--------|------|
| `apple_pick_sim/fruiting_system.py` | ~1315 | P0 cable tree generator + `generate_coupled_cable_scene`, `CoupledCableScene`, params/sampling, `run_rollout`, `measure_fruiting_forces` |
| `apple_pick_sim/proxy_coupling.py` | ~500 | Warp kernels: robot→proxy sync, VBD→robot harvest (velocity-delta + stem path helpers), `ProxyBodyRegistry`, `align_proxy_body_q_prev_for_vbd` |
| `apple_pick_sim/coupled_fruiting.py` | ~824 | `CoupledFruitingScene`, `coupled_substep`, builders (`build_coupled_fruiting_placeholder` / `_fr3`), bootstrap, MuJoCo defaults |
| `apple_pick_sim/fr3_robot.py` | ~677 | USD import, IK bootstrap, teleop controllers, keyboard helpers |
| `apple_pick_sim/vbd_fixed_joint_wrenches.py` | ~98 | P0 + stem-harvest wrapper around `SolverVBD.gather_joint_wrench_child_com` |
| `assets/testfr3_resolved.usda` + `assets/fr3/` | — | FR3 scene + Omniverse subtree (`fr3_robot.TESTFR3_SCENE_USD`) |

**Authoritative substep** (do not reorder without physics review):

1. Apply **lagged** harvest → `robot_state.body_f[tcp]` (via `_apply_tcp_spatial_wrench_kernel`)
2. MuJoCo step on **Model A** (`robot_model`)
3. **Sync** robot TCP → cable **proxy** (and **apple** when `fix_to_apple=True`); subtract same lagged wrench + gravity from proxy twist; `align_proxy_body_q_prev_for_vbd`
4. VBD step on **Model B** (`cable.model`)
5. **Harvest** for *next* substep → `proxy_forces` (velocity-delta **or** stem joint wrench)

Implemented in `CoupledFruitingScene.coupled_substep` (`coupled_fruiting.py`); kernels in `proxy_coupling.py`.

### What this refactor is for

1. **Split monoliths** into packages with stable **public import paths** (README, pytest, examples unchanged at the top level).
2. **Rename** coupling symbols so names reflect **data direction** (robot↔proxy↔MuJoCo), not vague “sync/harvest”.
3. **Thin orchestrators** — `coupled_fruiting` should own the loop, not 1.3k lines of fruiting build logic.
4. **Prepare Slice 2g** — document every host `.numpy()` / CPU fallback on the hot path; GPU work follows 2f layout.

### What not to do in 2f

- Start refactoring the **next** module while the **current** module’s gate tests are failing or skipped.
- Change staggered lag, gravity subtraction, or harvest formulas without new/updated tests.
- Patch `newton/` for GPU gather unless justified and covered by Newton tests.
- Move `assets/` without updating `TESTFR3_SCENE_USD`, `fr3_assets_available()`, docs, and tests in one slice.
- Rename public symbols without re-exports or a documented one-shot migration.

### Suggested PR order

1. Re-export shims only (`fruiting_system/` package, optional `robot/fr3_robot/` package) — **zero behavior change**
2. `proxy_coupling` naming + shared `@wp.func` for duplicate sync math
3. Split `coupled_fruiting/` (builders, bootstrap, scene)
4. Split `fruiting_system/` (params, build, scene, coupled)
5. FR3 package + asset paths (with compatibility shim at old paths if needed)
6. Slice **2g**: stem-harvest GPU, unified `device=`, `use_mujoco_cpu=False` when validated

**Gate between slices:** Finish and merge one module’s refactor (or shim) before starting the **next** module in the list. **Tests for the module you just touched must pass** (see [Validation gate](#validation-gate-run-before-the-next-module)); do not stack multiple module splits in one PR unless each area’s gate was already green on the branch base.

---

## Invariants (must preserve in 2f)

| Invariant | Where enforced |
|-----------|----------------|
| One-step lag: `proxy_forces` applied on **next** substep | `coupled_substep`, `coupling_forces_cache` |
| World-frame spatial wrench: linear then angular `[N, N·m]` | `proxy_coupling` module docstring |
| Double-integration guard on proxy `body_qd` after sync | `sync_proxy_state`, `sync_proxy_and_apple_state` |
| `align_proxy_body_q_prev_for_vbd` after kinematic sync | `coupled_substep`, `fruiting_system._align_coupled_scene_chain_from_reference` |
| Default `fix_to_apple=False` → velocity-delta harvest | `GripperProxyConfig`, builders |
| `fix_to_apple=True` → `sync_proxy_and_apple_state` + `harvest_stem_joint_wrench` | `CoupledFruitingScene` |
| Model A gravity zeroed on robot; cable keeps VBD gravity | `fr3_robot.sync_robot_gravity_to_mujoco` |
| `robot_model.device == cable.model.device == proxy_forces.device` | Builders default `device="cpu"` today |

**Harvest paths (do not merge):**

| Path | Sync kernel | Harvest |
|------|-------------|---------|
| Free proxy (default) | `launch_sync_proxy_state` | `harvest_proxy_wrenches` (velocity-delta) |
| Stem / co-teleport (`fix_to_apple=True`) | `launch_sync_proxy_and_apple_state` | `harvest_stem_joint_wrench` |

---

## Validation gate (run before the next module)

**Rule:** After each 2f sub-slice (2f-A … 2f-D, or `proxy_coupling`-only work in step 2), run that slice’s **gate tests** and require **exit code 0** before opening or continuing work on the **next** module. If a slice only adds re-exports with no behavior change, the gate is still the same test files — they must stay green.

| Module / slice | Gate tests (minimum) | Also run when |
|----------------|----------------------|---------------|
| `proxy_coupling` (2f-C or step 2) | `test_proxy_coupling.py` | Any sync/harvest/align rename or kernel dedup |
| `fruiting_system` (2f-A) | `test_fruiting_system.py`, `test_coupled_cable_scene.py` | Package split, `generate_*`, build helpers |
| `coupled_fruiting` (2f-B) | `test_coupled_fruiting_system.py`, `test_coupling_stability.py`, `verify_coupling.py` | Scene loop, builders, bootstrap |
| `fr3_robot` (2f-D) | `test_fr3_usd_import.py`, `test_fr3_ee_velocity_controller.py`, `test_coupled_fruiting_system.py -k fr3` | USD paths, controllers, coupled FR3 |
| Cross-cutting / unsure | Full block below | Touched imports used by another module |

When a slice touches two modules (e.g. rename in `proxy_coupling` plus call sites in `coupled_fruiting`), run **both** gates before moving on.

---

## Validation (full suite — periodic or pre-merge)

From **repository root** (`PYTHONPATH` as in `docs/ROADMAP.md` Agent execution notes):

```bash
# Proxy / coupling unit tests
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_proxy_coupling.py -q -p no:launch_testing

# Coupled scene + stability
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_coupled_fruiting_system.py \
  ../apple_pick_sim/tests/test_coupled_cable_scene.py \
  ../apple_pick_sim/tests/test_coupling_stability.py -q -p no:launch_testing

# P0 fruiting (after fruiting_system package split)
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_fruiting_system.py -q -p no:launch_testing

# FR3 (if robot package touched)
PYTHONPATH=$(pwd) uv run --directory newton python -m unittest \
  apple_pick_sim.tests.test_fr3_usd_import -v
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_fr3_ee_velocity_controller.py -q -p no:launch_testing

# Headless coupling sanity
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/verify_coupling.py
```

Optional: `diagnostics/benchmark_coupling.py --device cuda:0` (2g baselines).

---

## Slice 2f — structural tasks (ordered)

Each subsection below is a **separate PR** unless maintainer says otherwise. **Do not start 2f-B until 2f-A’s gate is green**, and so on (see [Validation gate](#validation-gate-run-before-the-next-module)).

### 2f-A — `fruiting_system/` package

**Goal:** Split `fruiting_system.py` (~1315 lines) without breaking:

```python
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint
from apple_pick_sim.fruiting_system import generate_coupled_cable_scene, CoupledCableScene, GripperProxyConfig
```

**Proposed layout:**

```
apple_pick_sim/fruiting_system/
  __init__.py      # re-export all symbols currently imported from fruiting_system.py
  params.py        # RodParams, FruitingSystemParams, load_ranges, sample_params,
                   # params_fingerprint, _validate_ranges, _coerce_omit, _direction_*, _deflect_*
  build.py         # _FruitingChainArtifacts, _new_fruiting_builder, _build_fruiting_chain_into_builder,
                   # _make_rod_geometry, _connect_rod_*, _add_gripper_proxy, _finalize_fruiting_builder,
                   # _apply_all_chain_collision_filters, _pin_body_vbd_prescribed
  scene.py         # FruitingSystemScene, generate_scene, _build_scene, _scene_states_from_model,
                   # run_rollout, make_fruiting_solver_vbd, example_collision_pipeline,
                   # measure_fruiting_forces, iter_fruiting_fixed_joint_indices, geometry_fingerprint
  coupled.py       # GripperProxyConfig, CoupledCableScene, generate_coupled_cable_scene,
                   # _build_coupled_cable_scene, geometry_fingerprint_coupled,
                   # _align_coupled_scene_chain_from_reference, proxy_registry on CoupledCableScene
```

**Notes:**

- Keep **`vbd_fixed_joint_wrenches.py`** at `apple_pick_sim/` (shared by P0 and stem harvest); `scene.py` / `fruiting_system/__init__.py` re-export as today.
- M1 staggered protocol prose in `fruiting_system.py` (lines ~32–62) can move to `coupled.py` or stay as a short pointer to `docs/mujoco-vbd-coupling-architecture.md`.
- `fruiting_system.py` becomes a thin shim: `from apple_pick_sim.fruiting_system import *` or explicit re-exports (deprecation comment optional).

**Tests:** `test_fruiting_system.py`, `test_coupled_cable_scene.py`.

**Gate before next slice:** both green → then start 2f-B (or 2f-C if doing `proxy_coupling` before coupled split per maintainer order).

---

### 2f-B — `coupled_fruiting/` package

**Goal:** `coupled_fruiting.py` (~824 lines) = orchestration only.

**Proposed layout:**

```
apple_pick_sim/coupled_fruiting/
  __init__.py           # CoupledFruitingScene, build_coupled_fruiting_placeholder,
                        # build_coupled_fruiting_fr3, DEFAULT_*_MUJOCO_SOLVER_KWARGS
  scene.py              # CoupledFruitingScene: coupled_substep, mujoco_substep, vbd_substep,
                        # _mujoco_and_sync_proxy, teleop helpers
  builders.py           # build_placeholder_tcp_robot_model, _assemble_coupled_robot_scene
  bootstrap.py          # bootstrap_tcp_joint_from_proxy, bootstrap_articulated_tcp_from_proxy
  stem.py               # _find_stem_apple_joint (run once at build; cache on CoupledFruitingScene)
  apply_wrench.py       # _apply_tcp_spatial_wrench_kernel, _apply_spatial_wrench_to_body_f
```

**Notes:**

- `stem_apple_joint_index` is already resolved in `_assemble_coupled_robot_scene` via `_find_stem_apple_joint` — ensure it is **never** re-scanned per substep (today it uses `joint_child.numpy()` once at build).
- Do **not** import `fr3_robot` from `fruiting_system/`; only `coupled_fruiting` and examples depend on robot builders.
- Top-level `coupled_fruiting.py` shim preserves README import:
  `from apple_pick_sim.coupled_fruiting import build_coupled_fruiting_placeholder`.

**Tests:** `test_coupled_fruiting_system.py`, `test_coupling_stability.py`, `verify_coupling.py`, `benchmark_coupling.py`.

**Gate before next slice:** coupled + stability + `verify_coupling.py` green → then start next module (e.g. 2f-D FR3, or 2f-A if order differs).

---

### 2f-C — `proxy_coupling` naming and deduplication

**Goal:** Names reflect **direction**; reduce duplicated warp logic between the two sync kernels.

**Naming (maintainer draft — revise before merge):**

| Current | Proposed | Direction / meaning |
|---------|----------|-------------------|
| `sync_proxy_state` | `mirror_robot_tcp_to_proxy_kernel` | Model A → proxy on Model B |
| `sync_proxy_and_apple_state` | `mirror_robot_tcp_to_proxy_and_apple_kernel` | A → proxy + apple (co-teleport) |
| `launch_sync_proxy_state` | `launch_mirror_robot_to_proxy` | wrapper |
| `launch_sync_proxy_and_apple_state` | `launch_mirror_robot_to_proxy_and_apple` | wrapper |
| `harvest_proxy_wrenches_velocity_delta_kernel` | `compute_proxy_reaction_wrench_kernel` | B → lagged load for A |
| `launch_harvest_proxy_wrenches_velocity_delta` | `launch_compute_proxy_reaction_wrench` | wrapper |
| `harvest_proxy_wrenches` | keep or `harvest_proxy_reaction_for_mujoco` | high-level entry |
| `harvest_stem_joint_wrench` | `harvest_stem_tension_for_tcp` | stem FIXED → TCP slot |
| `align_proxy_body_q_prev_for_vbd` | keep (already clear) | VBD finalize guard |

**Avoid:** `extract_forces_vbd_to_mujoco` for the **sync** kernel — sync copies **robot → proxy**, not VBD → MuJoCo.

**Dedup:** Extract shared `@wp.func` for lagged-wrench + gravity correction on proxy twist (blocks duplicated in `sync_proxy_state` and `sync_proxy_and_apple_state`, ~lines 89–105 and 409–418).

**Compatibility:** Keep old names as aliases in `proxy_coupling.py` for one release if grep shows wide doc/test references; update `docs/mujoco-vbd-coupling-architecture.md` in the same PR.

**Tests:** `test_proxy_coupling.py` (import `apple_pick_sim.proxy_coupling as pc` — update aliases or test both names once).

**Gate before next slice:** `test_proxy_coupling.py` green; if call sites in `coupled_fruiting` changed, also run 2f-B gate before moving on.

---

### 2f-D — `robot/fr3_robot/` package

**Goal:** Split `fr3_robot.py`; optional asset relocation.

**Proposed layout:**

```
apple_pick_sim/robot/fr3_robot/
  __init__.py              # re-export public API used by coupled_fruiting / examples / tests
  paths.py                 # _REPO_ROOT, TESTFR3_SCENE_USD, OMNIVERSE_FR3_*, fr3_assets_available()
  setup.py                 # build_fr3_robot_model_from_usd, sync_robot_gravity_to_mujoco,
                           # resolve_tcp_body_index, resolve_ee_body_index,
                           # init_mujoco_actuator_targets_*, sync_mujoco_visual_state
  placement.py             # placement_xform_for_proxy, bootstrap_tcp_ik_from_proxy, integrate_tcp_target
  controllers/
    ee_velocity.py         # Fr3EEVelocityController, EEVelocity, _make_tcp_ik_solver, apply_fr3_ee_teleop
    ee_direct_joint.py     # Fr3EEDirectJointController
    keyboard.py            # print_fr3_keyboard_bindings, read_keyboard_ee_velocity, poll_viewer_events
```

**Assets (optional sub-slice):**

```
apple_pick_sim/robot/fr3_robot/assets/
  testfr3.usda
  testfr3_resolved.usda   # canonical sim scene
  fr3/                    # omniverse subtree (or symlink to repo assets/fr3)
```

- Update `TESTFR3_SCENE_USD` in `paths.py` only.
- Leave **`assets/testfr3_resolved.usda`** at repo root as symlink or re-export path until Slice 3 README pass, **or** document one-shot path change in `docs/fr3-usd-import-implementation.md`.
- `fr3_robot.py` → shim: `from apple_pick_sim.robot.fr3_robot import ...` or `from apple_pick_sim import fr3_robot` unchanged via `apple_pick_sim/fr3_robot.py` re-export.

**Public symbols to re-export (grep before deleting shim):**

`fr3_assets_available`, `build_fr3_robot_model_from_usd`, `resolve_tcp_body_index`, `resolve_ee_body_index`, `TESTFR3_SCENE_USD`, `EE_MASS_KG`, `EE_BOX_HALF_EXTENTS`, `bootstrap_tcp_ik_from_proxy`, `placement_xform_for_proxy`, `sync_robot_gravity_to_mujoco`, `sync_mujoco_visual_state`, `init_mujoco_actuator_targets_from_model`, `Fr3EEVelocityController`, `Fr3EEDirectJointController`, `EEVelocity`, `integrate_tcp_target`, `apply_fr3_ee_teleop`, keyboard helpers.

**Tests:** `test_fr3_usd_import.py`, `test_fr3_ee_velocity_controller.py`, FR3 tests in `test_coupled_fruiting_system.py`.

**Gate before next slice:** FR3 unit tests + `test_coupled_fruiting_system.py -k fr3` green → then 2f-E / full validation block / 2g.

**Do not** pull FR3 into `fruiting_system/` — keeps cable generator independent of robot USD.

---

### 2f-E — Docs touch-up (same PR or follow-up)

- Update symbol names in `docs/mujoco-vbd-coupling-architecture.md` §3 module map if 2f-C lands.
- Add one paragraph to `refactor.md` “Completed” when a sub-slice merges (maintainer).
- Slice 3 (`README.md`, ROADMAP Agent execution notes) waits until entrypoints stabilize.

---

## Slice 2g — GPU optimization (after 2f layout)

**Not 2f** unless a host roundtrip is provably wrong (e.g. missing `align` on device). Track here; implement in dedicated PRs with profiler numbers.

### Hot-path host sync inventory (2026-05-22)

| Location | Operation | Priority |
|----------|-----------|----------|
| `harvest_stem_joint_wrench` | `fixed_joint_wrenches_child_com_vbd` → Newton GPU gather → **`.numpy()`** → `limit_stem_coupling_wrench` (CPU) → `out_robot_wrenches.assign` | **High** — default stem path when `fix_to_apple=True` |
| `bootstrap_tcp_joint_from_proxy` | `body_q` / `joint_q` host copy | Low — once at init |
| `build_coupled_fruiting_fr3` | `proxy_bq.numpy()` for placement | Low — once at init |
| `_find_stem_apple_joint` | `joint_child.numpy()` | Low — once at build; cache on scene |
| `CouplingForceDebugRecorder` | `.numpy()` for viewer | OK — debug only |
| `coupled_substep` (free-proxy) | `wp.clone(body_qd)` for `qd_synced` | Medium — pool buffer on scene |
| Builders | `device="cpu"`, `use_mujoco_cpu: True` in `DEFAULT_MUJOCO_SOLVER_KWARGS` | **High** for e2e GPU goal |

**Already on GPU (Slice 2e):** `ProxyBodyRegistry.ids_wp`, `_apply_tcp_spatial_wrench_kernel`, velocity-delta harvest kernel, `_align_body_q_prev_kernel`, sync/harvest launch wrappers.

**Stem-harvest GPU plan:**

1. Write TCP wrench directly on device (warp kernel or thin wrapper around Newton’s `gather_joint_wrench_child_at_com_kernel` for **one** joint index).
2. Port `limit_stem_coupling_wrench` (gain + caps) to warp.
3. Parity test: CPU harvest vs GPU harvest on same state, `test_proxy_coupling` or `test_coupled_fruiting_system`.

**Newton note:** `SolverVBD.gather_joint_wrench_child_com` launches on device but returns **numpy** (`solver_vbd.py`). End-to-end GPU may need a project-local gather-to-`wp.array` path or a small Newton API extension (submodule change — higher bar).

**MuJoCo GPU:** Thread `device="cuda:0"` (or CLI `--device`) through both models; set `use_mujoco_cpu: False` when scene + tests pass. Document in `docs/gpu-coupling-optimization.md` (create in 2g).

**Benchmark:** `diagnostics/benchmark_coupling.py` — capture ms/substep before/after each 2g PR.

---

## Import stability checklist

These must keep working through 2f (shim or `__init__.py` re-export):

| Consumer | Imports |
|----------|---------|
| `README.md` | `fruiting_system.load_ranges`, `generate_scene`, `geometry_fingerprint`, `coupled_fruiting.build_coupled_fruiting_placeholder` |
| `example_fruiting_system.py` | `fruiting_system.*` |
| `example_coupled_fruiting.py` | `coupled_fruiting.*`, `fruiting_system.*`, `fr3_robot.*` |
| `example_fr3_keyboard.py` | `fr3_robot.*` |
| Tests | `import apple_pick_sim.fruiting_system as fs`, `proxy_coupling as pc`, `fr3_robot`, `coupled_fruiting` |

---

## Completed / changelog

*(Maintainer: move items here when slices merge.)*

- [ ] 2f-A `fruiting_system/` package
- [ ] 2f-B `coupled_fruiting/` package
- [ ] 2f-C `proxy_coupling` naming + sync dedup
- [ ] 2f-D `robot/fr3_robot/` package (+ optional assets move)
- [ ] 2f-E docs sync
- [ ] 2g GPU stem harvest + device defaults + benchmark baselines

---

## Open questions (maintainer)

1. Final coupling symbol names — table in 2f-C is a proposal; confirm before wide rename.
2. Asset location: keep `assets/` at repo root vs `robot/fr3_robot/assets/` only.
3. Whether `apple_pick_sim/fr3_robot.py` stays forever as shim or deprecate in Slice 3.
4. Priority: 2f-A vs 2f-B first (either order works if shims are in place).
