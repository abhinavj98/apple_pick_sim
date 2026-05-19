# FR3 USD import (M1 Slice 2)

## Behavior summary

**Model A** (`robot_model`, `SolverMuJoCo`) is built by importing the **Isaac scene described by your `testfr3` export**, via the offline-composed file **`assets/testfr3_resolved.usda`**. That layer references the bundled arm at `assets/fr3/omniverse_fr3/fr3.usd` and carries the same **EE** / **tcp** prims and fixed joints as `assets/testfr3.usd`, with graph fixes so Newton can parse the articulation (EE welded to **link7**, `tcp` rigid body with mass).

Newton has no `import_articulation`; use **`ModelBuilder.add_usd(..., collapse_fixed_joints=False)`** then **`finalize()`**.

To use a **binary-only** `assets/testfr3.usd`: repoint its `fr3` reference to `./fr3/omniverse_fr3/fr3.usd`, apply the same joint/MassAPI fixes as in `testfr3_resolved.usda` if Newton reports a joint-cycle error, then pass `usd_path=` to `build_fr3_robot_model_from_usd`.

## Asset layout

See [`assets/fr3/README.md`](../assets/fr3/README.md).

| File | Purpose |
|------|---------|
| **`assets/testfr3_resolved.usda`** | Default import path (`TESTFR3_SCENE_USD`) |
| **`assets/testfr3.usd`** | Original Isaac export (metadata may name this as authoring layer) |
| **`assets/fr3/omniverse_fr3/`** | Local `fr3.usd` + `configuration/fr3_robot_schema.usd` |

## Code map

| Module | Role |
|--------|------|
| [`apple_pick_sim/fr3_robot.py`](../apple_pick_sim/fr3_robot.py) | `build_fr3_robot_model_from_usd`, `resolve_tcp_body_index`, IK bootstrap, `Fr3EEVelocityController` |
| [`apple_pick_sim/example_fr3_keyboard.py`](../apple_pick_sim/example_fr3_keyboard.py) | Standalone FR3 keyboard TCP teleop |
| [`apple_pick_sim/example_coupled_fruiting.py`](../apple_pick_sim/example_coupled_fruiting.py) | ``--robot fr3 --fr3-keyboard``: per-frame IK → ``apply_fr3_ee_teleop`` → ``mujoco_substep`` or ``coupled_substep`` |
| [`apple_pick_sim/coupled_fruiting.py`](../apple_pick_sim/coupled_fruiting.py) | `build_coupled_fruiting_fr3`, `apply_fr3_ee_teleop`, root placement + bootstrap |

**TCP vs EE:** coupling, `body_f`, and `proxy_registry` use the **`tcp`** link (`/fr3/ee/tcp`). **`ee`** is the parent link with EE collision geometry.

**Gravity (coupled runs):** Model A (`robot_model` / `SolverMuJoCo`) uses **zero** gravity for keyboard teleop and PD hold (`sync_robot_gravity_to_mujoco` + `notify_model_changed(MODEL_PROPERTIES)`). Model B (`cable.model` / `SolverVBD`) keeps **−9.81 m/s²** on Z. Proxy sync/harvest kernels use `CoupledFruitingScene.gravity_vec` (cable gravity), not `robot_model.gravity`.

## Keyboard teleop (Slice 2c)

| Entry | MuJoCo steps? | Cable VBD? | Status |
|-------|---------------|------------|--------|
| `example_fr3_keyboard.py` | No (FK + `joint_q` writeback only) | No | Unit tests + viewer smoke |
| `example_coupled_fruiting.py --robot fr3 --only-mjc --fr3-keyboard` | Yes (`mujoco_substep` + proxy sync) | No integration per substep | **Manually verified** (2026-05-19); default **`fix_to_apple=False`** |
| `example_coupled_fruiting.py --robot fr3 --fr3-keyboard` (full coupled) | Yes + VBD | Yes | **Not verified** interactively |

Keys (Newton **ViewerGL**, focus window): **I/K J/L R/F** world translate; **U/O T/G Z/X** rotate. **W/S** move the camera, not the arm.

## Tests

- `apple_pick_sim/tests/test_fr3_usd_import.py` — import smoke, EE + TCP bodies, MuJoCo construct
- `apple_pick_sim/tests/test_fr3_ee_velocity_controller.py` — TCP twist integration, keyboard mapping, IK step
- `apple_pick_sim/tests/test_coupled_fruiting_system.py::test_fr3_*`

## How to verify

From repository root:

```bash
cd newton && uv sync --extra examples
cd ..

PYTHONPATH=$(pwd) uv run --directory newton python -m unittest apple_pick_sim.tests.test_fr3_usd_import -v

PYTHONPATH=$(pwd) uv run --directory newton pytest \
  ../apple_pick_sim/tests/test_coupled_fruiting_system.py::test_fr3_tcp_pose_matches_proxy_after_bootstrap \
  ../apple_pick_sim/tests/test_coupled_fruiting_system.py::test_fr3_coupled_substep_finite_state -q

PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_coupled_fruiting.py --robot fr3 --viewer null --num-frames 30

# Interactive FR3 teleop (MuJoCo-only; see table above)
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_coupled_fruiting.py \
  --robot fr3 --only-mjc --fr3-keyboard --viewer gl

PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_fr3_keyboard.py --viewer null --num-frames 60
```

Placeholder remains default on the coupled example; FR3 tests skip without `usd-core` or bundled assets.
