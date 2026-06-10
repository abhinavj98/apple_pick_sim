# GPU architecture report (post-cleanup, 2026-06-09)

## Behavior summary

Apple Pick Sim runs on **Newton + NVIDIA Warp**. The **coupled picking hot path** (`CoupledFruitingScene.coupled_substep`) keeps cable state, proxy mirror/harvest, and (optionally) MuJoCo Warp on the GPU. Setup, IK teleop, Gym observations, and debug readouts use the CPU with explicit `.numpy()` sync points.

Coupling semantics (unchanged): **apply lagged wrench → MuJoCo robot step → mirror TCP to proxy (± apple) → VBD cable step → harvest wrench at TCP**.

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

**Default device:** `cuda:0` when available (`apple_pick_sim/sim_device.py`). Override with `--device cpu` or `APPLE_PICK_SIM_DEVICE=cpu`.

---

## What stays on CPU

| Path | Why |
|------|-----|
| Scene build, JSON fixtures, `sample_params` | One-off setup |
| FR3 IK bootstrap & keyboard teleop | Frame-rate; writes `joint_q` on host/model |
| `example_coupled_fruiting.py` viewer / plots | Debug; `.numpy()` after step |
| `ApplePickCoupledEnv` observations | `measure_fruiting_forces` + `.numpy()` readouts each step |
| `CouplingForceDebugRecorder` | Opt-in debug |
| `settle_then_weld.seed_fix_to_apple_from_settled` | Copies cable state via NumPy between two builds |
| Tests (most) | Correctness checks; parity tests compare CPU reference vs GPU |

**Build note:** `build_coupled_fruiting_fr3` calls `wp.synchronize()` after cable FK before reading proxy pose for FR3 placement (avoids stale GPU reads).

---

## Rough code split (apple_pick_sim/)

| Category | Approx. share of coupling-related logic | Notes |
|----------|----------------------------------------|-------|
| GPU hot path (Warp kernels + launches) | **~70%** of substep work | `proxy_coupling.py`, `apply_wrench.py`, `vic_*`, Newton/Warp solvers |
| CPU orchestration | **~20%** | `scene.py` substep ordering, builders, bootstrap |
| CPU-only / I/O | **~10%** | Examples, gym adapter, diagnostics CLI |

Line counts are dominated by tests and fruiting **build** code (also GPU-finalized models, but not per-substep).

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

## Tests

| Module | What it catches |
|--------|-----------------|
| `test_proxy_coupling.py` | `test_stem_harvest_cpu_gpu_parity`, mirror/harvest kernels |
| `test_coupled_fruiting_system.py` | Lag, quiescent caps, stem harvest integration |
| `test_explicit_apple_load.py` | Explicit apple weight force/torque |
| `test_cuda_graph.py` | Headless CUDA graph smoke (CUDA only) |
| `test_settle_then_weld.py` | Two-build initialization |
| `apple_pick_gym/tests/` | Env API + observation shapes |

```bash
# Fast gate
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/ -q -p no:launch_testing -m "not slow"

PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_gym/tests/ -q -p no:launch_testing
```

Benchmark: `apple_pick_sim/diagnostics/benchmark_coupling.py` (see `docs/gpu-coupling-optimization.md`).

---

## How to verify

```bash
cd newton && uv sync --extra examples --extra dev --extra torch-cu12 && cd ..

PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-gpu \
  --warmup-substeps 30 --bench-substeps 300

PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/verify_coupling.py \
  --num-substeps 600 --max-force 5 --max-torque 1
```
