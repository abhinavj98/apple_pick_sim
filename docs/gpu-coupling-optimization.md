# Slice 2g — GPU coupling optimization

**Last updated:** 2026-05-25

## Behavior summary

Staggered **MuJoCo + VBD** coupling (`CoupledFruitingScene.coupled_substep`) is optimized to keep state on the GPU: pooled `qd_synced`, device stem harvest/limit, optional MuJoCo Warp (`use_mujoco_cpu=False`), and optional CUDA graph replay in headless examples.

Coupling semantics are unchanged: apply lagged wrench → MuJoCo → sync proxy → VBD → harvest.

## Hot-path inventory (per substep)

| Location | Operation | Status (2g) |
|----------|-----------|-------------|
| `coupled_substep` | `wp.clone(body_qd)` | **Fixed** — pooled `qd_synced` + `wp.copy` |
| `harvest_stem_tension_for_tcp` | Full-buffer `.numpy()` + NumPy limit | **Fixed** — device gather + Warp limit kernel |
| `DEFAULT_MUJOCO_SOLVER_KWARGS` | `use_mujoco_cpu` | **Default `True`**; opt-in GPU via `mujoco_use_cpu=False` on CUDA |
| `CouplingForceDebugRecorder` | `.numpy()` | Debug only (unchanged) |
| FR3 teleop / IK | Host `joint_q` / keyboard | Frame-rate path (acceptable) |

## Profiler harness

Primary metric: **ms/substep** from [`apple_pick_sim/diagnostics/benchmark_coupling.py`](../apple_pick_sim/diagnostics/benchmark_coupling.py).

- Always call with `wp.synchronize()` on CUDA before/after timed loops (script does this).
- Fixture: `fruiting_system_ranges_straight_rod_test.json`, `disable_contacts=True`, `dt = (1/60)/30` s.
- Optional: Nsight Systems on `example_coupled_fruiting.py --viewer null --cuda-graph`.

## Benchmark commands

From repository root:

```bash
# CPU + MuJoCo CPU (default stability path)
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cpu --mujoco-cpu \
  --warmup-substeps 30 --bench-substeps 300

# CUDA + MuJoCo CPU
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-cpu \
  --warmup-substeps 30 --bench-substeps 300

# CUDA + MuJoCo Warp (GPU robot)
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-gpu \
  --warmup-substeps 30 --bench-substeps 300

# Stem-harvest path (fix_to_apple)
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --fix-to-apple \
  --warmup-substeps 30 --bench-substeps 300
```

## Baselines (reference machine)

**Linux**, fixture `fruiting_system_ranges_straight_rod_test.json`, `disable_contacts=True`, `dt ≈ 0.000556` s, warmup 30 / bench 300 unless noted.

| Robot | Device | MuJoCo | fix_to_apple | ms/substep | substeps/s | Notes |
|-------|--------|--------|--------------|------------|------------|-------|
| placeholder | cpu | CPU | false | ~10.2 | ~98 | See [slice-2e-hardening.md](slice-2e-hardening.md) |
| placeholder | cpu | CPU | false | ~8.5 | ~118 | RTX 4090 host, 2026-05-25 Slice 2g |
| placeholder | cuda:0 | Warp | false | ~12.2 | ~82 | default CUDA path (`--mujoco-gpu` explicit) |
| placeholder | cuda:0 | CPU | false | ~10.1 | ~99 | `--mujoco-cpu` on CUDA (robot CPU, cable GPU) |
| fr3 | cpu | CPU | false | ~12.3 | ~81 | slice-2e table |

Re-run after hardware or dependency changes; update this table when reporting regressions.

## CUDA graph capture constraints

CUDA graphs record a **fixed** GPU launch sequence. Safe for headless examples when:

- `--cuda-graph` and CUDA device with mempool enabled
- No `--debug-coupling-forces`, no FR3 keyboard teleop
- Velocity-delta coupling (`fix_to_apple=False`); stem path needs device harvest inside graph

**Breaks capture:** per-substep `.numpy()`, viewer `apply_forces` inside captured loop, changing contact capacity mid-run, teleop inside substeps.

Pattern: `example_apple_stem.py` — capture `simulate()` loop, host readouts **after** `capture_launch`.

## Tests

| Module | What it catches |
|--------|-----------------|
| `test_coupled_fruiting_system.py` | `test_qd_synced_buffer_reused_across_substeps`, `test_coupled_substep_lag_one_step`, FR3 CUDA slow test |
| `test_proxy_coupling.py` | `test_stem_harvest_cpu_gpu_parity` |
| `test_cuda_graph.py` | Headless graph smoke (CUDA only) |

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -m slow -q -p no:launch_testing
```
