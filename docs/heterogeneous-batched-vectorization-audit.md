# Heterogeneous batched vectorization audit

**Last updated:** 2026-06-26

Audit of `apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py` and its dependency chain against the goal of **fully vectorized** GPU stepping. Canonical batched flow and phased delivery live in [`vectorized-coupled-fruiting.md`](vectorized-coupled-fruiting.md).

---

## Executive summary

The heterogeneous example is **mostly vectorized** for the main physics loop (VBD + MuJoCo + batched IK scatter), but it is **not completely vectorized**.

| Priority | Gap | Impact |
| -------- | --- | ------ |
| **P0** | Stem harvest runs a Python loop over envs every substep | Performance; violates hot-path hygiene |
| **P0** | Per-env grasp offsets and apple mass not wired into runtime coupling | Correctness for heterogeneous DR |
| **P1** | Teleop velocity staging uses per-env Python loops | Frame-rate only; acceptable per project GPU rules |
| **P2** | Init/build uses sequential `add_world`, per-env IK bootstrap | One-time cost; lower priority |

---

## What is already vectorized

| Layer | Mechanism | Module |
| ----- | --------- | ------ |
| VBD settle + step | Single model, `world_count=N`; all worlds step together | `coupled_fruiting/scene.py` → `vbd_substep` |
| MuJoCo robot step | Batched model, `separate_worlds=True` when `N > 1` | `coupled_fruiting/batched_build.py` |
| TCP→proxy+apple mirror | Batched Warp kernel, `dim=num_envs` | `coupled_fruiting/proxy_coupling.py` → `launch_mirror_robot_to_proxy_offset_and_apple` |
| Wrench apply to robot | Registry-based multi-TCP write | `coupled_fruiting/apply_wrench.py` |
| FR3 teleop IK | `IKSolver(n_problems=N)` + GPU gather / advance / scatter | `robot/fr3_robot/batched_template_ik.py` |
| Velocity-delta harvest | Fully batched via registry (not used when `fix_to_apple=True`) | `proxy_coupling.py` → `harvest_proxy_wrenches` |

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

## Not vectorized — runtime hot path (substep loop)

### 1. Stem harvest: per-env Python loop (P0)

When `fix_to_apple=True` (the heterogeneous example default), `_harvest_coupling_wrenches` in `coupled_fruiting/scene.py` loops over worlds and calls `harvest_stem_tension_for_tcp` once per env per substep. Each call launches a **`dim=1`** Warp kernel.

With `--num-envs 4`, `sim_substeps=30`, and 30 Hz, that is roughly **3,600 sequential harvest invocations per second**.

The main vectorization spec states that mirror and stem harvest launch with `dim = len(registry)`. That holds for **velocity-delta** harvest (`harvest_proxy_wrenches`) but **not** for stem harvest on the welded path.

**Owner:** `apple_pick_sim/coupled_fruiting/scene.py` (`_harvest_coupling_wrenches`), `proxy_coupling.py` (`harvest_stem_tension_for_tcp`).

### 2. Heterogeneous per-env offsets and mass not used at runtime (P0)

Build stores per-env weld geometry on the scene:

- `CoupledFruitingScene.per_env_params`
- `CoupledFruitingScene.per_world_proxy_offsets`

Runtime coupling ignores these after settle→weld init:

| Runtime path | Current behavior | Heterogeneous need |
| ------------ | ---------------- | ------------------ |
| `welded_co_teleport_arrays_for_layout` | Repeats template (world-0) grasp offset for all envs | Per-env offset from `per_world_proxy_offsets` |
| Stem harvest | Single `cable.gripper_proxy_offset_in_apple_frame` | Per-env offset array |
| Explicit apple weight | Single `scene.apple_mass_kg` from world-0 apple via `_cached_apple_mass_kg` | Per-env mass when apple radius/density differ |

`per_world_proxy_offsets` is consumed only in `settle_then_weld.py` → `seed_fix_to_apple_from_settled` during init, not in `coupled_substep`.

A per-instance pattern already exists in `proxy_coupling.py` → `mega_welded_co_teleport_arrays_wp` and can be adapted for batched layouts.

**Owner:** `coupled_fruiting/proxy_coupling.py`, `coupled_fruiting/scene.py`, `coupled_fruiting/builders.py`.

---

## Not vectorized — frame-rate teleop (P1)

Per `.cursor/rules/gpu-warp-parallelism.mdc`, frame-rate teleop on CPU is acceptable. These remain scalar:

| Location | What loops |
| -------- | ---------- |
| `Fr3BatchedEEVelocityController.advance_target` | Python loop over envs filling `_lin_vels_wp` / `_ang_vels_wp` from `velocity_for_world(w)` |
| `_sync_target_tf_from_device` | `.numpy()` round-trip + Python loop rebuilding `target_tf` list |
| `BatchedTemplateIK.max_pose_error` / `pose_errors_per_world` | Per-world CPU FK for IK diagnostics |
| Example `_velocity_for_world` | Per-env math when `--demo-per-env-actions` or `--noisy-action` |

Default example behavior (without `--demo-per-env-actions`) applies the **same** scripted velocity to all envs. IK still scatters one row per env; only the velocity **input** is homogeneous.

**Owner:** `robot/fr3_robot/controllers/ee_velocity_batched.py`, `example_batched_heterogeneous_coupled_fruiting.py`.

---

## Not vectorized — init / build (P2, one-time)

| Step | Location | Notes |
| ---- | -------- | ----- |
| Heterogeneous cable build | `batched_build.py` → `build_heterogeneous_coupled_cable_scene` | `for params_w in params_list: outer.add_world(sub)` |
| Param sampling | `fruiting_system/params.py` → `sample_heterogeneous_params_list` | CPU loop over `num_envs` |
| Settle→weld seed | `settle_then_weld.py` → `seed_fix_to_apple_from_settled` | NumPy host copies; per-env apple/proxy pose loop |
| Per-env IK bootstrap | `settle_then_weld.py` → `_bootstrap_tcp_per_env` | Sequential template IK per world; heterogeneous example uses `per_env_ik=True` |
| Robot state broadcast | `batched_build.py` → `_broadcast_robot_state_from_template` | Host loop copying template `joint_q` into each world |

Init-time loops matter less than substep loops but block “complete” vectorization if interpreted strictly.

### Initial per-env IK bootstrap after settle→weld (P0 — often the slowest startup step)

The heterogeneous example passes `per_env_ik=True` into `seed_fix_to_apple_from_settled`. That calls `_bootstrap_tcp_per_env`, which is **not vectorized**.

```python
# settle_then_weld.py — sequential over envs
for w in range(layout.num_envs):
    bootstrap_articulated_tcp_from_proxy(view, tpl_robot, ..., ik_iterations=256)
    batched_jq[c0 : c0 + coord_per] = tpl_robot.joint_q.numpy()
```

Each call goes through `bootstrap_tcp_ik_from_proxy` in `robot/fr3_robot/placement.py`, which:

| Property | Init bootstrap (`bootstrap_tcp_ik_from_proxy`) | Runtime teleop (`BatchedTemplateIK`) |
| -------- | ---------------------------------------------- | ------------------------------------ |
| `IKSolver.n_problems` | **1** (single-world template) | **`num_envs`** |
| Env loop | Python `for w in range(N)` | None — one batched solve |
| Seed retries | Up to **4** joint-q seeds per env, sequential | Single seed row per env (from current `joint_q`) |
| Iterations | **256** per seed attempt (heterogeneous weld path) | 128 per teleop frame (example default) |
| Host sync | `.numpy()` on `body_q`, `joint_q` after each attempt | GPU gather/scatter; minimal host I/O |

**Rough cost for `N` envs:** up to `N × 4 × 256` sequential IK iteration batches, plus NumPy copies and FK per attempt. For `--num-envs 4`, that is on the order of **4,000+ IK solves at init**, compared with **one** batched solve of shape `(N, dof)` if implemented like runtime teleop.

The single-env bootstrap path explicitly rejects batched robot models:

```python
if int(robot_model.world_count) > 1:
    raise ValueError(
        "bootstrap_tcp_ik_from_proxy does not support replicated robot models; "
        "solve IK on the single-world ik_template_robot_model and broadcast joint_q."
    )
```

So init deliberately uses the template model one env at a time rather than `BatchedTemplateIK`.

**Recommended fix:** replace `_bootstrap_tcp_per_env` with batched placement:

1. GPU-gather each env’s settled **proxy** pose from `cable.state_0.body_q` (same kernel as `BatchedTemplateIK.gather_tcp_targets_from_state`, but indexed on **proxy** bodies).
2. Set `BatchedTemplateIK` target rows (template frame) from those poses.
3. Optionally batched multi-seed: `(N, n_seeds, dof)` or vectorized seed loop — harder but still better than `N` sequential Python loops.
4. One `solver.step(joint_q, joint_q, iterations=256)` with `n_problems=N`.
5. `scatter_to_model` into the batched `robot_model`.

This matches the V.2 note in [`vectorized-coupled-fruiting.md`](vectorized-coupled-fruiting.md) (“replace template-only bootstrap + broadcast with `BatchedTemplateIK` per-row target”).

**Owner:** `coupled_fruiting/settle_then_weld.py` (`_bootstrap_tcp_per_env`), `robot/fr3_robot/placement.py` (`bootstrap_tcp_ik_from_proxy`).

---

## Example-specific / debug (intentionally scalar)

- `_print_status`, `test_final` in the example — host `.numpy()` + loops (debug only)
- `batched_viz.py` — per-env loops for endpoint markers and TCP force arrows (viewer only)
- Placeholder robot path — world-0 nudge + `broadcast_joint_q_from_world0` (not used in default FR3 path)

---

## Planned but not yet implemented

From [`vectorized-coupled-fruiting.md`](vectorized-coupled-fruiting.md):

| Slice | Status | Remaining work |
| ----- | ------ | -------------- |
| **V.2.3** | Planned | Per-env runtime actions as first-class example/API default |
| **V.2.4** | Planned | `gather_transitions()` per world |
| **V.3** | Planned | Runtime geometry DR on reset; gym `(N, act_dim)` scatter |
| Runtime K/stiffness scatter | Planned | Heterogeneous build bakes θ at `finalize()` only today |

---

## Recommended implementation order

1. **Batch stem harvest** — Replace the `for w in range(num_envs)` loop with one launch (`dim=N`) taking arrays of stem joint indices, TCP indices, apple indices, per-env grasp offsets, and per-env apple masses. Mirror the registry pattern used by `harvest_proxy_wrenches`.

2. **Wire heterogeneous offsets/mass into runtime coupling** — Feed `scene.per_world_proxy_offsets` and per-env apple mass into `welded_co_teleport_arrays_for_layout` and stem harvest. Cache per-env masses at build time (no `.numpy()` in the hot path).

3. **Vectorize teleop velocity staging** — When actions are already `(N, 6)`, upload directly; reserve `velocity_for_world` callbacks for interactive/debug only.

4. **Optional: vectorize per-env IK bootstrap** — Replace `_bootstrap_tcp_per_env` sequential loop with `BatchedTemplateIK` + per-env settled-proxy target rows (see [Initial per-env IK bootstrap](#initial-per-env-ik-bootstrap-after-settleweld-p0--often-the-slowest-startup-step)).

5. **Example flags (V.2.3)** — Clarify or default per-env action scatter in the heterogeneous example.

---

## Tests

Existing coverage:

| Module | Key tests |
| ------ | --------- |
| `test_heterogeneous_coupled_fruiting.py` | Build-time DR, per-env stiffness, per-env IK bootstrap, per-world offsets at weld |
| `test_vectorized_coupled_fruiting.py` | Batched settle→weld, IK scatter, substep stability, per-env velocity divergence |
| `test_batched_template_ik.py` | GPU gather/advance/scatter unit tests |

Suggested tests after P0 fixes:

- `test_batched_stem_harvest_no_per_env_python_loop` — assert single launch / no per-substep host sync (or parity vs reference loop)
- `test_heterogeneous_per_env_grasp_offset_used_at_runtime` — different offsets → different mirror/harvest lever arms per env
- `test_heterogeneous_per_env_apple_mass_in_stem_harvest` — explicit weight matches per-env baked mass
- `test_batched_ik_bootstrap_aligns_all_proxy_targets` — one batched init IK pass; all TCPs within bootstrap tolerance at their per-env proxies

Run:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py \
  apple_pick_sim/tests/test_vectorized_coupled_fruiting.py -q

uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
```

---

## Code map (audit scope)

| Module | Role in heterogeneous example |
| ------ | ----------------------------- |
| `examples/example_batched_heterogeneous_coupled_fruiting.py` | Entry point; settle→weld + teleop loop |
| `coupled_fruiting/batched_build.py` | `add_world` heterogeneous cable build |
| `coupled_fruiting/builders.py` | `build_heterogeneous_coupled_fruiting_{fr3,placeholder}` |
| `coupled_fruiting/scene.py` | `coupled_substep`; stem harvest dispatch |
| `coupled_fruiting/proxy_coupling.py` | Mirror kernels; stem / velocity-delta harvest |
| `coupled_fruiting/settle_then_weld.py` | Init; per-env IK bootstrap |
| `robot/fr3_robot/batched_template_ik.py` | Batched IK (vectorized) |
| `robot/fr3_robot/controllers/ee_velocity_batched.py` | Batched teleop (partially scalar velocity fill) |
| `batched_viz.py` | Viewer overlays (scalar; not hot path) |
