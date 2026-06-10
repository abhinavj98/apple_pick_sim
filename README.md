# Apple Pick Sim

This repository contains simulation code for robotic apple picking using the [Newton](https://github.com/newton-physics/newton) physics engine.

## Installation

### 1. Clone the repository

Clone this repository and initialize the submodules to pull in the Newton dependency.

```bash
git clone --recursive https://github.com/abhinavj98/apple_pick_sim.git
```

If you have already cloned the repository without the recursive flag, you can initialize the submodule manually:

```bash
git submodule update --init --recursive
```

The Newton submodule is cloned from [abhinavj98/newton](https://github.com/abhinavj98/newton) (this repo’s fork of upstream Newton). Inside `newton/`, `origin` is that fork and `upstream` is [newton-physics/newton](https://github.com/newton-physics/newton); use `git fetch upstream` there when you want changes from the official project.

### 2. Install Newton

From the repository root, sync the Newton submodule’s environment. Use **`examples`** for viewers and scripts, **`dev`** for pytest/gymnasium, and **`torch-cu12`** for VIC joint-torque teleop in `example_coupled_fruiting.py`:

```bash
cd newton && uv sync --extra examples --extra dev --extra torch-cu12 && cd ..
```

Minimal install (P0 fruiting only, no gym/VIC):

```bash
cd newton && uv sync --extra examples && cd ..
```

All `uv run` commands below assume the **repository root** as the current working directory unless noted otherwise. Set **`PYTHONPATH=$(pwd)`** so `apple_pick_sim` and `apple_pick_gym` import from the repo root.

## Running the simulation

### `example_apple_stem.py`

```bash
uv run --directory newton python ../apple_pick_sim/examples/example_apple_stem.py
```

This runs the apple simulation with three branch stiffness presets. The terminal prints forces and torques on the stem. To apply forces on the apple, use right-click and drag on the apple in the viewer.

### `example_fruiting_system.py` (variational fruiting)

Procedural **primary → secondary → spur → stem → apple** layout from `apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json` (default for
`example_fruiting_system.py`); each run draws a new sample unless you pass `--seed`. Unit tests
use `fruiting_system_ranges_straight_rod_test.json` for deterministic, nearly vertical chains.
Uses the same Newton viewer pattern as the stem example.

The script imports the `apple_pick_sim` package, so set `PYTHONPATH` to the repository root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fruiting_system.py
```

Useful options (see also the script docstring):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fruiting_system.py \
  --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 123
```

**Collisions:** The library default `generate_scene(..., enable_self_collisions=True)` only relies on Newton’s joint **parent/child** collision filters (adjacent rod segments do not collide). **Non-adjacent** chain capsules can still collide with each other and with the apple.

The interactive examples **`example_fruiting_system.py`** and **`example_coupled_fruiting.py`** disable intra-chain self collisions by default (`enable_self_collisions=False`). Pass **`--enable-self-collision`** to opt in to non-adjacent link–link contacts (same semantics as `enable_self_collisions=True` above). **Ground contact is unchanged** in either mode.

From Python, call `ExampleFruitingSystem.regenerate()` (optional seed) to rebuild while keeping the viewer. See `apple_pick_sim/examples/example_fruiting_system.py`.

## P0 variational fruiting (JSON + seed)

Range fixtures live under `apple_pick_sim/fixtures/`: **`fruiting_system_ranges_example_variance.json`**
(wide angles; default for the viewer example) and **`fruiting_system_ranges_straight_rod_test.json`**
(nearly −Z chain; default for tests). The generator is the **`apple_pick_sim/fruiting_system/`** package (`params.py`, `build.py`, `scene.py`, `coupled.py`; public API via `apple_pick_sim.fruiting_system`).

**Geometry-only smoke check** (no viewer; paths assume `uv`’s working directory is `newton/`):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json')
scene  = generate_scene(ranges, seed=42)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

**Short headless VBD rollout** (optional: pass ``collision_pipeline=example_collision_pipeline(scene.model)`` to match the viewer’s ``create_collision_pipeline`` path):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import (
    load_ranges, generate_scene, geometry_fingerprint, run_rollout,
    example_collision_pipeline,
)
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json')
scene  = generate_scene(ranges, seed=42)
pipe = example_collision_pipeline(scene.model, args=None)
run_rollout(scene, num_steps=20, sim_substeps=10, collision_pipeline=pipe)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

**Structured force readout** (fixed-joint wrenches plus ``cable_joint_indices`` metadata; cable scalar forces follow ``example_apple_stem.py`` when needed): call ``measure_fruiting_forces`` from ``apple_pick_sim.fruiting_system`` with post-step ``body_q``, pre-step ``body_q_prev``, and ``dt`` after a ``SolverVBD`` substep.

**Device:** Scene builders default to **`cuda:0`** when CUDA is available (`apple_pick_sim/sim_device.py`). Pass ``device="cpu"`` or set ``APPLE_PICK_SIM_DEVICE=cpu`` to force CPU. Interactive examples accept ``--device`` (e.g. ``--device cpu``).

### M1 two-model coupling (FR3 + VBD cable)

Headless **staggered** ``SolverMuJoCo`` + ``SolverVBD`` step via the **`apple_pick_sim/coupled_fruiting/`** package (``scene.py``, ``builders.py``, …; import ``apple_pick_sim.coupled_fruiting``). Gripper proxy defaults to **`fix_to_apple=False`** (velocity-delta harvest + proxy-only sync); pass ``GripperProxyConfig(fix_to_apple=True)`` in code for stem-harvest / apple co-teleport tests.

- **FR3 + custom EE (default):** ``build_coupled_fruiting_fr3`` imports ``assets/testfr3_resolved.usda`` (Isaac **`testfr3`** EE/tcp + bundled ``assets/fr3/omniverse_fr3/fr3.usd``); see ``assets/fr3/README.md``.
- **Placeholder:** ``build_coupled_fruiting_placeholder`` — free-floating TCP box; use ``--robot placeholder`` if FR3 assets are missing.
- **Control:** ``example_coupled_fruiting.py`` defaults to **FR3 + VIC joint-torque teleop** (dynamic arm, plant wrenches on TCP ``body_f``). Requires PyTorch: ``cd newton && uv sync --extra torch-cu12``. Tune with ``--vic-linear-k``, ``--vic-linear-d``, ``--vic-angular-k``, ``--vic-angular-d``.
- **Step modes:** default = full coupled loop; ``--only-vbd`` = cable only; ``--only-mjc`` = MuJoCo robot + proxy sync.

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m unittest apple_pick_sim.tests.test_fr3_usd_import -v
```

Smoke (paths assume repo root + ``uv`` project ``newton/``, i.e. fixture path is relative to ``newton/``):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.coupled_fruiting import build_coupled_fruiting_placeholder
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json')
scene = build_coupled_fruiting_placeholder(ranges, seed=0)
scene.coupled_substep(1e-4)
print('coupled_substep_ok')
"
```

Interactive **Newton viewer** (shows the **cable** scene: rods + apple + gripper proxy, which mirrors the coupling). Optional **`--mujoco-viewer`** opens MuJoCo’s passive viewer for the **TCP placeholder** rigid body (**second window**).

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 120
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --mujoco-viewer --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 0
# Staggered coupling wrench debug (Plots panel in ViewerGL): lagged → MuJoCo vs fresh ← VBD harvest
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --debug-coupling-forces --seed 42
# TCP force as a yellow arrow at the robot TCP (scale: --tcp-force-scale, --tcp-force-arrow-gain, min/max length)
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --viewer gl --tcp-force-arrow --seed 42
# Bundled FR3 + custom EE (default; requires usd-core + assets/fr3/)
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 60
# FR3 keyboard teleop (VIC joint torques; focus ViewerGL window)
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py \
  --fr3-keyboard --viewer gl
# Optional second window for the MuJoCo robot model
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py \
  --fr3-keyboard --mujoco-viewer --viewer gl
# Stem-harvest path: weld proxy to apple (default is --no-fix-to-apple / velocity-delta)
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --fix-to-apple --seed 42
# Placeholder TCP (no FR3 assets)
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py --robot placeholder --viewer null --num-frames 60
```

**FR3 keyboard teleop** (TCP velocity + IK; ``--viewer gl``, focus the window — **I/K J/L R/F** translate, **U/O T/G Z/X** rotate; **not W/S**, those move the camera):

- **Coupled fruiting + arm (default):** ``example_coupled_fruiting.py`` with ``--fr3-keyboard --viewer gl`` (VIC joint torques).
- **Robot only (kinematic FK, no MuJoCo step):** ``example_fr3_keyboard.py`` — useful for IK/viewer smoke without the fruiting tree.

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fr3_keyboard.py --viewer gl
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fr3_keyboard.py --viewer null --num-frames 120
```

### Validation (fast test gate)

After changes to fruiting, coupling, or gym code:

```bash
# Fast sim tests (excludes @pytest.mark.slow)
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/ -q -p no:launch_testing -m "not slow"

# Gym env tests
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_gym/tests/ -q -p no:launch_testing

# Coupled example smoke (headless; requires torch-cu12 for default FR3+VIC path)
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 60
```

GPU coupling inventory and benchmarks: `docs/gpu-coupling-optimization.md`, `docs/gpu-architecture-report.md`.

## Tests

From the repository root (requires `uv sync --extra dev` in `newton/`):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing -m "not slow"
```

Full suite including slow stability tests:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing
```

M1 coupling stability (longer-horizon; includes ``slow`` tests):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupling_stability.py -q -p no:launch_testing
```

Optional slow tests only (500+ substep stability, FR3 long horizon):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -m slow -q -p no:launch_testing
```

M1 coupling benchmark (ms/substep; see ``docs/gpu-coupling-optimization.md``):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-gpu --warmup-substeps 30 --bench-substeps 300
# CPU MuJoCo baseline:
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cpu --mujoco-cpu --warmup-substeps 30 --bench-substeps 300
```

Headless CUDA graph (coupled example):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_coupled_fruiting.py \
  --viewer null --cuda-graph --num-frames 200
```

Headless **coupling verification** (applied vs harvested wrench, TCP–proxy pose drift; exit 1 on threshold breach):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/verify_coupling.py \
  --num-substeps 600 --max-force 5 --max-torque 1
```

### Gymnasium environment (`ApplePickCoupled-v0`)

Headless env over the coupled FR3 stack; `Discrete(13)` keyboard-style actions; real observations (woody part poses/forces, apple position, TCP wrench/velocity). See `apple_pick_gym/envs/apple_pick_coupled_env.py`.

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_gym/tests/ -q -p no:launch_testing
```

`PYTHONPATH` must include the repo root so `apple_pick_sim` imports resolve; `--directory newton` selects Newton’s `pyproject.toml` and virtual environment.
