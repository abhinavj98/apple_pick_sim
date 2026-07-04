# GPU coupling architecture and optimization

**Last updated:** 2026-07-03 (adds batched heterogeneous hot-path notes; single-env sections unchanged)

**Scope note:** Single-env sections below describe the original coupled picking path. For the **multi-env batched heterogeneous** GPU hot path (`BatchedHeterogeneousCoupledSim`), see `docs/vectorized-coupled-fruiting.md`, `docs/heterogeneous-batched-vectorization-audit.md`, and design spec `docs/superpowers/specs/2026-07-03-batched-gpu-hot-path-design.md`.

## Behavior summary

Apple Pick Sim runs on **Newton + NVIDIA Warp**. The **coupled picking hot path** (`CoupledFruitingScene.coupled_substep`) keeps cable state, proxy mirror/harvest, and (optionally) MuJoCo Warp on the GPU. Setup, IK teleop, Gym observations, and debug readouts use the CPU with explicit `.numpy()` sync points.

Coupling semantics (unchanged): **apply lagged wrench → MuJoCo robot step → mirror TCP to proxy (± apple) → VBD cable step → harvest wrench at TCP**.

**Default device:** `cuda:0` when available (`apple_pick_sim/sim_device.py`). Override with `--device cpu` or `APPLE_PICK_SIM_DEVICE=cpu`.

---

## Batched heterogeneous hot path (PR2, CUDA + FR3)

`BatchedHeterogeneousCoupledSim.step()` with `defaults()` and FR3 assets keeps the substep loop fully on device. Frame-rate teleop reads batched actions from a Torch buffer and uploads twists without per-env host round-trips.

| Stage | Module | Mechanism |
|-------|--------|-----------|
| Action clip | `batched_action_twists.clip_action_tensor` | Torch vectorized clamp |
| Action→twist | `batched_action_twists.upload_batched_twists_from_actions` | `wp.from_torch` + `wp.copy` |
| Teleop frame | `ee_*_batched.run_coupled_teleop_frame_from_actions` | Batched IK + scatter |
| VIC torques | `vic_joint_torques_batched.py` | `wp.to_torch(joint_q)` |
| Settle seed | `settle_seed_device.py` | Proxy alignment, cable copy, zero twists |
| Joint broadcast | `broadcast_device.py` | Warp scatter (init + placeholder) |
| Physics substep | `scene.coupled_substep` | Unchanged — GPU |

**Still CPU (acceptable):** keyboard teleop, `velocity_for_world` callbacks, placeholder world-0 nudge on CPU device, debug/viewer `.numpy()` readouts, checkpoint `body_q` capture (once at build).

**Placeholder robot:** test-only / explicit `robot.kind='placeholder'` or FR3 asset fallback; not used in production `defaults()` when assets are present.

---

## What runs on GPU (per substep)

| Stage | Module | Mechanism |
|-------|--------|-----------|
| MuJoCo robot | `newton.solvers.SolverMuJoCo` | GPU when `use_mujoco_cpu=False` (default on CUDA) |
| TCP → proxy mirror | `proxy_coupling.launch_mirror_robot_to_proxy[_and_apple]` | Warp kernels |
| VBD cable | `SolverVBD` + collision | Device-resident `body_q`, `body_qd` |
| Velocity-delta harvest | `proxy_coupling.launch_compute_proxy_reaction_wrench` | Warp kernel over proxy registry |
| Stem harvest (`fix_to_apple`) | `proxy_coupling.harvest_stem_tension_for_tcp` | Device gather + `_limit_and_write_tcp_stem_wrench_kernel` |
| Wrench apply (dynamic arm) | `apply_wrench._apply_spatial_wrench_to_body_f` | Warp kernel on `body_f` |
| VIC joint torques | `vic_joint_torques.apply_vic_joint_torques_to_scene` | PyTorch + Warp (`joint_f`) |
| VIC spatial (legacy) | `vic_wrench.launch_apply_vic_to_coupling_cache` | Warp kernel |
| Pooled sync buffer | `scene.qd_synced` | `wp.copy` (no per-step alloc) |

## What stays on CPU

| Path | Why |
|------|-----|
| Scene build, JSON fixtures, `sample_params` | One-off setup |
| FR3 IK bootstrap & keyboard teleop | Frame-rate; writes `joint_q` on host/model |
| `example_coupled_fruiting.py` viewer / plots | Debug; `.numpy()` after step |
| `ApplePickCoupledEnv` observations | `measure_fruiting_forces` + `.numpy()` readouts each step |
| `CouplingForceDebugRecorder` | Opt-in debug |
| `settle_then_weld.seed_fix_to_apple_from_settled` | Copies cable state via NumPy between two builds (CPU builds); CUDA uses `settle_seed_device.py` |
| Tests (most) | Correctness checks; parity tests compare CPU reference vs GPU |

**Build note:** `build_coupled_fruiting_fr3` calls `wp.synchronize()` after cable FK before reading proxy pose for FR3 placement (avoids stale GPU reads).

## Rough code split (apple_pick_sim/)

| Category | Approx. share of coupling-related logic | Notes |
|----------|----------------------------------------|-------|
| GPU hot path (Warp kernels + launches) | **~70%** of substep work | `proxy_coupling.py`, `apply_wrench.py`, `vic_*`, Newton/Warp solvers |
| CPU orchestration | **~20%** | `scene.py` substep ordering, builders, bootstrap |
| CPU-only / I/O | **~10%** | Examples, gym adapter, diagnostics CLI |

---

## Hot-path optimization inventory (per substep)

| Location | Operation | Status |
|----------|-----------|--------|
| `coupled_substep` | `wp.clone(body_qd)` | **Fixed** — pooled `qd_synced` + `wp.copy` |
| `harvest_stem_tension_for_tcp` | Full-buffer `.numpy()` + NumPy limit | **Fixed** — device gather + Warp limit kernel |
| `DEFAULT_MUJOCO_SOLVER_KWARGS` | `use_mujoco_cpu` | **Default `False`** in solver kwargs; builders resolve to MuJoCo CPU on CPU Warp devices and MuJoCo Warp on CUDA unless explicitly overridden |
| `CouplingForceDebugRecorder` | `.numpy()` | Debug only (unchanged) |
| FR3 teleop / IK | Host `joint_q` / keyboard | Frame-rate path (acceptable) |

---

## Logic correctness

### Staggered two-model loop

Documented in `docs/mujoco-vbd-coupling-architecture.md`. `coupled_substep` applies **lagged** `proxy_forces` to the robot, advances MuJoCo, mirrors TCP motion to the cable proxy (and apple when welded), runs VBD, then **harvests** fresh coupling wrench into `proxy_forces` for the next substep. Tests: `test_coupled_substep_lag_one_step`, `test_qd_synced_buffer_reused_across_substeps`.

### Stem harvest vs velocity delta

When the scene has an apple, `stem_apple_joint_index` is set and harvest uses the **stem–apple fixed joint** reaction (not velocity delta). With `fix_to_apple=True`, proxy and apple co-teleport with the TCP; harvest includes optional **explicit apple weight** (`explicit_load.py`) for prescribed apples (`inv_mass == 0`).

**GPU/CPU parity fix (2026-06-09):** `_limit_and_write_tcp_stem_wrench_kernel` now always transfers stem force from apple COM to TCP via lever arm when `robot_body_q` and apple index are provided; explicit weight adds support force and `(r × F_apple)` separately—matching `_harvest_stem_tension_for_tcp_cpu`. Verified by `test_stem_harvest_cpu_gpu_parity` and `test_stem_harvest_explicit_adds_force_and_torque`.

### Settle-then-weld

Two-build workflow (`settle_then_weld.py`): settle free apple on VBD, build welded scene, copy settled `body_q`/`body_qd`, align proxy offset, re-run IK at fixed FR3 base. Tests use `COUPLED_BASE_POS` + `COUPLED_ROBOT_BASE_POS` with adjacent-seed retry on IK bootstrap flakiness.

### Gym env (M2.1)

`ApplePickCoupledEnv` uses **kinematic direct-joint** control (not VIC). Physics still runs on GPU; observations are assembled on CPU from `measure_fruiting_forces` and TCP buffers.

---

## Profiler harness

Primary metric: **ms/substep** from [`apple_pick_sim/diagnostics/benchmark_coupling.py`](../apple_pick_sim/diagnostics/benchmark_coupling.py).

- Always call with `wp.synchronize()` on CUDA before/after timed loops (script does this).
- Fixture: `fruiting_system_ranges_straight_rod_test.json`, `disable_contacts=True`, `dt = (1/60)/30` s.
- Optional: Nsight Systems on `example_coupled_fruiting.py --viewer null --cuda-graph`.

## Benchmark commands

From repository root:

```bash
# CPU + MuJoCo CPU (default stability path)
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cpu --mujoco-cpu \
  --warmup-substeps 30 --bench-substeps 300

# CUDA + MuJoCo CPU
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-cpu \
  --warmup-substeps 30 --bench-substeps 300

# CUDA + MuJoCo Warp (GPU robot)
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-gpu \
  --warmup-substeps 30 --bench-substeps 300

# Stem-harvest path (fix_to_apple)
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --fix-to-apple \
  --warmup-substeps 30 --bench-substeps 300
```

## Baselines (reference machine)

**Linux**, fixture `fruiting_system_ranges_straight_rod_test.json`, `disable_contacts=True`, `dt ≈ 0.000556` s, warmup 30 / bench 300 unless noted.

| Robot | Device | MuJoCo | fix_to_apple | ms/substep | substeps/s | Notes |
|-------|--------|--------|--------------|------------|------------|-------|
| placeholder | cpu | CPU | false | ~8.5 | ~118 | RTX 4090 host |
| placeholder | cuda:0 | Warp | false | ~12.2 | ~82 | default CUDA path (`--mujoco-gpu` explicit) |
| placeholder | cuda:0 | CPU | false | ~10.1 | ~99 | `--mujoco-cpu` on CUDA (robot CPU, cable GPU) |
| fr3 | cpu | CPU | false | ~12.3 | ~81 | FR3 chain |

Re-run after hardware or dependency changes; update this table when reporting regressions.

---

## CUDA graph capture constraints

CUDA graphs record a **fixed** GPU launch sequence. Safe for headless examples when:

- `--cuda-graph` and CUDA device with mempool enabled
- No `--debug-coupling-forces`, no FR3 keyboard teleop
- Velocity-delta coupling (`fix_to_apple=False`); stem path needs device harvest inside graph

**Breaks capture:** per-substep `.numpy()`, viewer `apply_forces` inside captured loop, changing contact capacity mid-run, teleop inside substeps.

Pattern: `example_apple_stem.py` — capture `simulate()` loop, host readouts **after** `capture_launch`.

---

## Tests

| Module | What it catches |
|--------|-----------------|
| `test_proxy_coupling.py` | `test_stem_harvest_cpu_gpu_parity`, mirror/harvest kernels |
| `test_coupled_fruiting_system.py` | `test_qd_synced_buffer_reused_across_substeps`, `test_coupled_substep_lag_one_step`, FR3 CUDA slow test |
| `test_explicit_apple_load.py` | Explicit apple weight force/torque |
| `test_cuda_graph.py` | Headless graph smoke (CUDA only) |
| `test_settle_then_weld.py` | Two-build initialization |
| `test_batched_action_twists.py` | Device action upload + clip |
| `test_broadcast_device.py` | GPU joint broadcast parity |
| `test_batched_heterogeneous_coupled_sim.py` | Batched sim smoke, FR3 per-env actions, settle cache |
| `apple_pick_gym/tests/` | Env API + observation shapes |

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_batched_action_twists.py \
  apple_pick_sim/tests/test_broadcast_device.py -q

uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -m slow -q
```

## How to verify

```bash
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-gpu \
  --warmup-substeps 30 --bench-substeps 300

uv run python apple_pick_sim/diagnostics/verify_coupling.py \
  --num-substeps 600 --max-force 5 --max-torque 1
```
