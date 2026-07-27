# Heterogeneous batched vectorization audit

**Last updated:** 2026-07-27

Audit of the batched heterogeneous coupled stack and its dependency chain against the goal of **fully vectorized** GPU stepping. Canonical batched heterogeneous API: `apple_pick_sim.coupled_fruiting.BatchedHeterogeneousCoupledSim` (thin example: `example_batched_heterogeneous_coupled_sim.py`). Flow reference: [`vectorized-coupled-fruiting.md`](vectorized-coupled-fruiting.md); slice status: `docs/ROADMAP.md`.

---

## Executive summary

The heterogeneous example's substep hot path is **fully vectorized** on CUDA + FR3 for physics (VBD, MuJoCo, mirror, stem harvest, VIC). Remaining gaps: one-time init cost (heterogeneous `add_world` build), frame-rate keyboard callbacks, and (until fixed) per-substep re-allocation of co-teleport / harvest flag arrays.

| Priority | Gap | Status | Impact |
| -------- | --- | ------ | ------ |
| ~~P0~~ | Stem harvest ran a Python loop over envs every substep | **Fixed** — `harvest_batched_stem_tension` (single batched launch, `proxy_coupling.py`) is used whenever `layout.num_envs > 1` | — |
| ~~P0~~ | Per-env grasp offsets and apple mass not wired into runtime coupling | **Fixed** — `prepare_batched_stem_harvest_arrays` bakes per-env `stem_harvest_grasp_offsets_wp` / `stem_harvest_apple_masses_wp` at build time, consumed by the batched harvest launch above; covered by `test_batched_stem_harvest.py` | — |
| ~~P0~~ | Per-env IK bootstrap sequential over envs | **Fixed** — `_bootstrap_tcp_per_env` uses `BatchedTemplateIK` (`settle_then_weld.py`) | Init only |
| ~~P1~~ | Action→teleop used per-env `.cpu().tolist()` and Python velocity loops | **Fixed (PR2)** — `run_coupled_teleop_frame_from_actions` + `upload_batched_twists_from_actions`; `BatchedHeterogeneousCoupledSim.step()` uses `_action_buffer` when `allocate_action_buffer=True` | Frame-rate only |
| ~~P1~~ | VIC joint torques read `joint_q.numpy()` each substep | **Fixed (PR2)** — `wp.to_torch(state.joint_q)` in `vic_joint_torques_batched.py` | Substep hot path |
| **P1** | Keyboard / `velocity_for_world` callback teleop | Still true | Frame-rate only; acceptable per project GPU rules |
| ~~P1~~ | Co-teleport / harvest flag arrays rebuilt every substep | **Fixed** — cached in `prepare_batched_stem_harvest_arrays` (`co_teleport_*_wp`, `stem_harvest_use_explicit_wp`); `coupled_substep` reuses them | — |
| **P2** | Init/build uses sequential `add_world`; settle seed host copies on CPU builds | Partially fixed (PR2) — GPU kernels for proxy alignment, cable copy, quiet twists, robot template broadcast (CUDA), legacy 1→N cable broadcast (CUDA); CPU builds keep host reference paths | One-time cost |

---

## What is already vectorized

| Layer | Mechanism | Module |
| ----- | --------- | ------ |
| VBD settle + step | Single model, `world_count=N`; all worlds step together | `coupled_fruiting/scene.py` → `vbd_substep` |
| MuJoCo robot step | Batched model, `separate_worlds=True` when `N > 1` | `coupled_fruiting/batched_build.py` |
| TCP→proxy+apple mirror | Batched Warp kernel, `dim=num_envs` | `coupled_fruiting/proxy_coupling.py` → `launch_mirror_robot_to_proxy_offset_and_apple` |
| Wrench apply to robot | Registry-based multi-TCP write | `coupled_fruiting/apply_wrench.py` |
| FR3 teleop IK | `IKSolver(n_problems=N)` + GPU gather / advance / scatter | `robot/fr3_robot/batched_template_ik.py` |
| Action→teleop (RL/gym) | Device `(N, 6)` upload + batched target advance | `robot/fr3_robot/controllers/batched_action_twists.py`, `ee_*_batched.run_coupled_teleop_frame_from_actions` |
| VIC joint torques (batched) | `wp.to_torch(joint_q)` each substep | `coupled_fruiting/vic_joint_torques_batched.py` |
| Settle→weld seed (batched) | GPU proxy alignment, cable copy, zero twists | `coupled_fruiting/settle_seed_device.py` |
| Joint broadcast (init / placeholder) | Warp scatter kernels on CUDA | `coupled_fruiting/broadcast_device.py` |
| Velocity-delta harvest | Fully batched via registry (not used when `fix_to_apple=True`) | `proxy_coupling.py` → `harvest_proxy_wrenches` |
| Stem harvest (multi-env) | Single batched launch over per-env stem/TCP/apple index arrays and per-env grasp offset / apple mass arrays (baked at build time) | `proxy_coupling.py` → `harvest_batched_stem_tension`, `prepare_batched_stem_harvest_arrays`; dispatched from `coupled_fruiting/scene.py` → `_harvest_coupling_wrenches` when `layout.num_envs > 1` |

The example inner loop structure is correct: teleop once per viewer frame, then `sim_substeps` calls to `CoupledFruitingScene.coupled_substep`:

```python
def simulate(self) -> None:
    self._teleop_world0()
    if self._robot_kind == "placeholder":
        broadcast_joint_q_from_world0(self.scene, self.layout)
    for _ in range(self.sim_substeps):
        self.scene.coupled_substep(self.sim_dt)
```

---

## Resolved since original audit (2026-06-26)

### 3. Batched GPU hot path (PR2, 2026-07-03) — **fixed for CUDA + FR3**

`BatchedHeterogeneousCoupledSim.step()` on CUDA with FR3 assets and `defaults()`:

| Stage | Before | After |
| ----- | ------ | ----- |
| Action clip | Python loop over envs | `clip_action_tensor` (Torch vectorized) |
| Action→twist | `_velocity_for_world` → `.cpu().tolist()` | `upload_batched_twists_from_actions` → `_lin_vels_wp` / `_ang_vels_wp` |
| Teleop frame | `advance_target` Python loop from callback | `run_coupled_teleop_frame_from_actions` when `_action_buffer` set |
| VIC torques | `joint_q.numpy()` / `joint_qd.numpy()` | `wp.to_torch(...)` |
| Settle capture | `body_q.numpy().copy()` | `capture_body_q_numpy` (one host read for checkpoint only) |
| Settle seed | NumPy proxy loop + host cable copy | `align_batched_proxy_poses_device`, `copy_cable_state_device`, `zero_all_body_qd_device` |
| Robot template broadcast | Host loop in `_broadcast_robot_state_from_template` | `broadcast_robot_state_from_template_device` on CUDA builds |
| Placeholder broadcast | Host loop every frame | `broadcast_joint_q_from_world0_device` (all devices); CPU placeholder nudge remains host-only |

**Acceptance rule:** zero `.numpy()`/`.cpu()` in `coupled_substep`, VIC apply, and action→teleop during `step()` when CUDA + FR3 + `defaults()`. Init may sync once at end of build for checkpoint/diagnostics.

**Owner:** `coupled_fruiting/batched_heterogeneous_coupled_sim.py`, `broadcast_device.py`, `settle_seed_device.py`, `batched_action_twists.py`.

### 1. Stem harvest: per-env Python loop — **fixed**

`_harvest_coupling_wrenches` in `coupled_fruiting/scene.py` now dispatches a **single** batched launch (`harvest_batched_stem_tension`, `dim=num_envs`) whenever `layout is not None and layout.num_envs > 1`, using precomputed per-env index/offset/mass arrays (`scene.stem_harvest_*_wp`, populated by `prepare_batched_stem_harvest_arrays` at build time). The single-env `harvest_stem_tension_for_tcp` path remains for `num_envs == 1` / non-batched scenes only. Test: `apple_pick_sim/tests/test_batched_stem_harvest.py`.

### 2. Heterogeneous per-env offsets and mass not used at runtime — **fixed**

The per-env grasp offset and apple mass arrays prepared at build time (`prepare_batched_stem_harvest_arrays` → `stem_harvest_grasp_offsets_wp`, `stem_harvest_apple_masses_wp`, `stem_harvest_use_grasp_offset_wp`) are consumed directly by `harvest_batched_stem_tension` above, so heterogeneous per-env grasp geometry and apple mass now do feed runtime stem-harvest coupling, not just settle→weld init.

**If re-auditing:** confirm `welded_co_teleport_arrays_for_layout` (the co-teleport/mirror path, as opposed to the harvest path audited here) also uses per-env offsets before declaring this fully closed — this document only re-verified the harvest side.

---

## Not vectorized — frame-rate teleop (P1, residual)

Per `.cursor/rules/gpu-warp-parallelism.mdc`, frame-rate teleop on CPU is acceptable. These remain scalar when using **keyboard** or **`velocity_for_world` callbacks** (not the device action-buffer path):

| Location | What loops |
| -------- | ---------- |
| `Fr3BatchedEEVelocityController.advance_target` | Python loop over envs filling `_lin_vels_wp` / `_ang_vels_wp` from `velocity_for_world(w)` — **bypassed** when `run_coupled_teleop_frame_from_actions` is used |
| `_sync_target_tf_from_device` | `.numpy()` round-trip + Python loop rebuilding `target_tf` list (debug/viewer; synced after `advance_target_from_actions`) |
| `BatchedTemplateIK.max_pose_error` / `pose_errors_per_world` | Per-world CPU FK for IK diagnostics |
| Legacy example `_velocity_for_world` | Per-env math when `--demo-per-env-actions` or `--noisy-action` |

Default **library** behavior (`BatchedHeterogeneousCoupledSim` with `allocate_action_buffer=True`) uploads batched actions on device. Keyboard teleop in examples still uses scalar paths.

**Owner:** `robot/fr3_robot/controllers/ee_velocity_batched.py`, `BatchedHeterogeneousCoupledSim`.

---

## Not vectorized — init / build (P2, one-time)

| Step | Location | Notes |
| ---- | -------- | ----- |
| Heterogeneous cable build | `batched_build.py` → `build_heterogeneous_coupled_cable_scene` | `for params_w in params_list: outer.add_world(sub)` |
| Param sampling | `fruiting_system/params.py` → `sample_heterogeneous_params_list` | CPU loop over `num_envs` |
| Settle→weld seed | `settle_then_weld.py` → `seed_fix_to_apple_from_settled` | GPU on CUDA via `settle_seed_device.py`; CPU builds use host reference |
| Per-env IK bootstrap | `settle_then_weld.py` → `_bootstrap_tcp_per_env` | Uses `BatchedTemplateIK` when `per_env_ik=True`; iterations from `robot.ik_bootstrap_iterations` |
| Robot state broadcast | `batched_build.py` → `_broadcast_robot_state_from_template` | GPU scatter on CUDA (`broadcast_device.py`); host loop on CPU |
| Legacy 1→N cable broadcast | `batched_build.py` → `broadcast_settled_cable_state_to_batched_worlds` | GPU kernel on CUDA; NumPy on CPU (legacy path only) |

Init-time loops matter less than substep loops but block “complete” vectorization if interpreted strictly.

### Initial per-env IK bootstrap after settle→weld — **fixed (uses BatchedTemplateIK)**

The heterogeneous path passes `per_env_ik=True` into `seed_fix_to_apple_from_settled`, which calls `_bootstrap_tcp_per_env`. That function now uses **`BatchedTemplateIK`** (`n_problems=N`, gather proxy targets → `step` → `scatter_to_model`) — **not** a Python loop over `bootstrap_articulated_tcp_from_proxy`.

| Property | Init bootstrap (`_bootstrap_tcp_per_env`) | Runtime teleop (`BatchedTemplateIK`) |
| -------- | ---------------------------------------------- | ------------------------------------ |
| `IKSolver.n_problems` | **`num_envs`** | **`num_envs`** |
| Env loop for solve | None — one batched solve | None — one batched solve |
| Seeds | `n_seeds` (Roberts sampler) | Current joint rows / teleop |
| Host sync | Pose-error diagnostics may `.numpy()` | GPU gather/scatter; minimal host I/O |

**Follow-up (not blocking):** `_bootstrap_tcp_per_env` does not call `seed_from_state` before solving (relies on solver-internal / multi-seed init). Worth a convergence check later.

**Owner:** `coupled_fruiting/settle_then_weld.py` (`_bootstrap_tcp_per_env`), `robot/fr3_robot/batched_template_ik.py`.

Test: `test_batched_ik_bootstrap_aligns_all_proxy_targets` in `test_heterogeneous_coupled_fruiting.py`.

---

## Example-specific / debug (intentionally scalar)

- `_print_status`, `test_final` in the example — host `.numpy()` + loops (debug only)
- `batched_viz.py` — per-env loops for endpoint markers and TCP force arrows (viewer only)
- Placeholder robot path — world-0 nudge (CPU host on CPU device; GPU nudge on CUDA) + `broadcast_joint_q_from_world0` (Warp kernel). **Not used** in default FR3 `defaults()` path; explicit `robot.kind='placeholder'` or FR3 asset fallback only.

---

## Still planned (not yet implemented)

Runtime K/stiffness scatter: the heterogeneous build still bakes θ at `finalize()` only — there is no runtime re-scatter of stiffness without a rebuild. Geometry DR on reset without rebuild is tracked in the **`[V]` track of `docs/ROADMAP.md`**.

**Shipped (do not list as planned):** per-env runtime actions via `step((N,6))`, recorded-transition gathering in feature code, batched gym adapter (`ApplePickBatched*`).

**Known inert config:** `FruitingSystemConfig.stem_harvest_explicit_apple_weight` is not wired through `_builder_kwargs`; welded builds always enable explicit apple load via builders.

---

## Recommended implementation order (remaining items)

1. **Optional: seed-from-state on init IK bootstrap** — May improve `_bootstrap_tcp_per_env` convergence.
2. **Runtime K/stiffness scatter** — See ROADMAP `[V]` track; out of scope for hot-path hygiene.

---

## Tests

Existing coverage:

| Module | Key tests |
| ------ | --------- |
| `test_heterogeneous_coupled_fruiting.py` | Build-time DR, per-env stiffness, per-env IK bootstrap, per-world offsets at weld |
| `test_vectorized_coupled_fruiting.py` | Batched settle→weld, IK scatter, substep stability, per-env velocity divergence |
| `test_batched_template_ik.py` | GPU gather/advance/scatter unit tests |
| `test_batched_action_twists.py` | Device action upload and vectorized clip |
| `test_broadcast_device.py` | GPU joint broadcast and template scatter parity vs host |
| `test_batched_heterogeneous_coupled_sim.py` | FR3 per-env action divergence, settle cache, placeholder warnings |

The original P0 fixes are covered by `test_batched_stem_harvest.py` (batched launch, per-env grasp offset, per-env apple mass). Suggested test for the remaining P2 item:

- `test_batched_ik_bootstrap_aligns_all_proxy_targets` — one batched init IK pass; all TCPs within bootstrap tolerance at their per-env proxies

Run:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_batched_action_twists.py \
  apple_pick_sim/tests/test_broadcast_device.py \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py \
  apple_pick_sim/tests/test_vectorized_coupled_fruiting.py -q

uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
```

---

## Code map (audit scope)

| Module | Role in heterogeneous batched sim |
| ------ | --------------------------------- |
| `coupled_fruiting/batched_heterogeneous_coupled_sim.py` | Canonical library API (`BatchedHeterogeneousCoupledSim`) |
| `examples/example_batched_heterogeneous_coupled_sim.py` | Canonical thin CLI + viewer entry point |
| `examples/example_batched_heterogeneous_coupled_sim.py` | Canonical batched heterogeneous CLI + viewer |
| `coupled_fruiting/batched_build.py` | `add_world` heterogeneous cable build |
| `coupled_fruiting/builders.py` | `build_heterogeneous_coupled_fruiting_fr3` (FR3-only) |
| `coupled_fruiting/scene.py` | `coupled_substep`; stem harvest dispatch |
| `coupled_fruiting/proxy_coupling.py` | Mirror kernels; stem / velocity-delta harvest |
| `coupled_fruiting/settle_then_weld.py` | Init; per-env IK bootstrap |
| `coupled_fruiting/settle_seed_device.py` | GPU settle→weld seed kernels (PR2) |
| `coupled_fruiting/broadcast_device.py` | GPU joint/cable broadcast kernels (PR2) |
| `robot/fr3_robot/controllers/batched_action_twists.py` | Device action upload + clip (PR2) |
| `robot/fr3_robot/batched_template_ik.py` | Batched IK (vectorized) |
| `robot/fr3_robot/controllers/ee_velocity_batched.py` | Batched teleop; device action fast path |
| `batched_viz.py` | Viewer overlays (scalar; not hot path) |
