# Slice 2f — structural refactor (packages)

**Last updated:** 2026-05-25 (layout cleanup: examples folder, shim removal)  
**ROADMAP:** [M1] Slice **2f** (layout and naming; physics unchanged). **Slice 2g** (GPU) is separate.

## Behavior summary

M1 staggered coupling semantics are unchanged: **apply lagged wrench → MuJoCo → mirror robot→proxy → VBD → harvest for next substep**. This slice only **splits monolith modules** and **renames** proxy Warp symbols so names reflect data direction.

**Canonical imports** (no top-level shims):

```python
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, generate_coupled_cable_scene
from apple_pick_sim.coupled_fruiting import build_coupled_fruiting_placeholder, CoupledFruitingScene
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.coupled_fruiting import proxy_coupling  # or .proxy_coupling.ProxyBodyRegistry
```

Runnable scripts live under **`apple_pick_sim/examples/`** (see README). There is **no** top-level `fruiting_system.py`, `coupled_fruiting.py`, `fr3_robot.py`, or `proxy_coupling.py`.

## Code map

| Package / module | Role |
|------------------|------|
| `apple_pick_sim/fruiting_system/` | P0 + M1 cable scene generation |
| `params.py` | `load_ranges`, `sample_params`, `GripperProxyConfig`, fingerprints |
| `build.py` | Rod chain + gripper proxy into `ModelBuilder`, `make_fruiting_solver_vbd` |
| `scene.py` | `generate_scene`, `run_rollout`, `measure_fruiting_forces`, collision pipeline |
| `coupled.py` | `generate_coupled_cable_scene`, `CoupledCableScene` |
| `apple_pick_sim/coupled_fruiting/` | Two-model orchestration |
| `builders.py` | `build_coupled_fruiting_placeholder`, `build_coupled_fruiting_fr3` |
| `bootstrap.py` | TCP alignment from proxy pose |
| `scene.py` | `CoupledFruitingScene.coupled_substep`, `mujoco_substep`, `vbd_substep` |
| `apply_wrench.py` | Lagged TCP wrench apply on device |
| `stem.py` | Stem–apple joint index (resolved once at build) |
| `proxy_coupling.py` | Warp mirror/harvest kernels (2f-C renames; see below) |
| `apple_pick_sim/robot/fr3_robot/` | USD import, placement, teleop controllers |
| `apple_pick_sim/examples/` | `example_apple_stem`, `example_fruiting_system`, `example_coupled_fruiting`, `example_fr3_keyboard` |
| `apple_pick_sim/vbd_fixed_joint_wrenches.py` | Shared P0 + stem-harvest wrench gather |

### Proxy coupling renames (2f-C)

| Previous (roadmap / old code) | Current |
|-----------------------------|---------|
| `sync_proxy_state` / `launch_sync_proxy_state` | `mirror_robot_tcp_to_proxy_kernel` / `launch_mirror_robot_to_proxy` |
| `sync_proxy_and_apple_state` | `mirror_robot_tcp_to_proxy_and_apple_kernel` / `launch_mirror_robot_to_proxy_and_apple` |
| `harvest_proxy_wrenches_velocity_delta_kernel` | `compute_proxy_reaction_wrench_kernel` |
| `launch_harvest_proxy_wrenches_velocity_delta` | `launch_compute_proxy_reaction_wrench` |
| `harvest_stem_joint_wrench` | `harvest_stem_tension_for_tcp` |

High-level harvest entry points `harvest_proxy_wrenches` and `align_proxy_body_q_prev_for_vbd` are unchanged.

## Tests

| Gate | Modules touched | Key tests |
|------|-----------------|-----------|
| 2f-A | `fruiting_system/` | `test_fruiting_system.py`, `test_coupled_cable_scene.py` |
| 2f-B | `coupled_fruiting/` | `test_coupled_fruiting_system.py`, `test_coupling_stability.py` |
| 2f-C | `coupled_fruiting/proxy_coupling.py` | `test_proxy_coupling.py` |
| 2f-D | `robot/fr3_robot/` | `test_fr3_usd_import.py`, `test_fr3_ee_velocity_controller.py`, `test_coupled_fruiting_system.py -k fr3` |
| Layout | shims removed, `examples/` | `test_package_layout.py` |

FR3 coupled tests use **direct joint hold** and relaxed harvest caps (~500 N); see `docs/test-migration-report.md`.

## How to verify

From **repository root** (`cd newton && uv sync --extra examples` once if needed):

```bash
# 2f-A — fruiting_system package
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_fruiting_system.py \
  ../apple_pick_sim/tests/test_coupled_cable_scene.py -q -p no:launch_testing

# 2f-B — coupled_fruiting package
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_coupled_fruiting_system.py \
  ../apple_pick_sim/tests/test_coupling_stability.py -q -p no:launch_testing

# 2f-C — proxy_coupling kernels
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_proxy_coupling.py -q -p no:launch_testing

# Layout — no top-level shims; examples import path
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_package_layout.py -q -p no:launch_testing

# Import smoke (same paths as README)
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene
from apple_pick_sim.coupled_fruiting import build_coupled_fruiting_placeholder
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json')
scene = build_coupled_fruiting_placeholder(ranges, seed=0)
scene.coupled_substep(1e-4)
print('slice_2f_import_ok')
"

# Full project tests (periodic / pre-merge)
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing
```

**Device:** Builders default to **`cuda:0`** when Warp reports CUDA (`apple_pick_sim/sim_device.py`). Override with `--device cpu` on examples/diagnostics or `APPLE_PICK_SIM_DEVICE=cpu` in the environment.

**`diagnostics/verify_coupling.py`:** still uses the **placeholder** builder; default `--max-force 5` may fail when harvest plateaus at ~300 N under gravity (pose sync remains tight). Use pytest gates above for refactor sign-off, or relax thresholds when inspecting traces only.

## Related docs

- `refactor.md` — full 2f backlog and remaining tasks (2f-E cleanup, 2g GPU)
- `docs/mujoco-vbd-coupling-architecture.md` — staggered protocol and per-model ownership
- `docs/ROADMAP.md` — Agent execution notes
