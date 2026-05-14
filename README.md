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

From the repository root, sync the Newton submodule’s environment (examples extra matches the viewer and script dependencies used here):

```bash
cd newton && uv sync --extra examples && cd ..
```

All `uv run` commands below assume the **repository root** as the current working directory unless noted otherwise.

## Running the simulation

### `example_apple_stem.py`

```bash
uv run --directory newton python ../apple_pick_sim/example_apple_stem.py
```

This runs the apple simulation with three branch stiffness presets. The terminal prints forces and torques on the stem. To apply forces on the apple, use right-click and drag on the apple in the viewer.

### `example_fruiting_system.py` (variational fruiting)

Procedural **primary → secondary → spur → stem → apple** layout from `apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json` (default for
`example_fruiting_system.py`); each run draws a new sample unless you pass `--seed`. Unit tests
use `fruiting_system_ranges_straight_rod_test.json` for deterministic, nearly vertical chains.
Uses the same Newton viewer pattern as the stem example.

The script imports the `apple_pick_sim` package, so set `PYTHONPATH` to the repository root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_fruiting_system.py
```

Useful options (see also the script docstring):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_fruiting_system.py \
  --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 123
```

**Collisions:** By default, `generate_scene(..., enable_self_collisions=True)` only relies on Newton’s joint **parent/child** collision filters (adjacent rod segments do not collide). **Non-adjacent** chain capsules can still collide with each other and with the apple; that is physically meaningful but can be stiff if segment length is small relative to capsule radius (next-but-one overlap) or if contacts fight cable constraints.

Pass **`--no-self-collision`** to set `enable_self_collisions=False`, which registers **shape collision filter pairs between every pair of distinct chain bodies** (primary through apple), so the tree does not self-collide; **ground contact is unchanged**. Use this if you need a more stable run without intra-chain contacts.

From Python, call `ExampleFruitingSystem.regenerate()` (optional seed) to rebuild while keeping the viewer. See `apple_pick_sim/example_fruiting_system.py`.

## P0 variational fruiting (JSON + seed)

Range fixtures live under `apple_pick_sim/fixtures/`: **`fruiting_system_ranges_example_variance.json`**
(wide angles; default for the viewer example) and **`fruiting_system_ranges_straight_rod_test.json`**
(nearly −Z chain; default for tests). The generator is `apple_pick_sim/fruiting_system.py` (module docstring describes the API).

**Geometry-only smoke check** (no viewer; paths assume `uv`’s working directory is `newton/`):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json')
scene  = generate_scene(ranges, seed=42)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

**Short headless VBD rollout**:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint, run_rollout
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json')
scene  = generate_scene(ranges, seed=42)
run_rollout(scene, num_steps=20, sim_substeps=10)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

## Tests

From the repository root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing
```

`PYTHONPATH` must include the repo root so `apple_pick_sim` imports resolve; `--directory newton` selects Newton’s `pyproject.toml` and virtual environment.
