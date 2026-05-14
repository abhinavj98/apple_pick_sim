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

### 2. Install Newton


#### Setup using `uv` (Recommended)
Navigate to the `newton` directory and create a virtual environment:

```bash
cd newton && uv sync --extra examples && cd ..
```


## Running the Simulation

To run the `example_apple_stem.py` simulation, execute the following command from the root of this repository:

```bash
uv run --directory newton python ../apple_pick_sim/example_apple_stem.py
```

This command runs apple simulation with 3 different branch stiffnesses. The terminal prints the forces and torques experienced by the stem

To apply forces on the apple, use your right click and drag on the apple.

**Variational fruiting system** (new random layout each run unless you pass ``--seed``; same viewer pattern as the stem example):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/example_fruiting_system.py
```

Append ``--no-self-collision`` for  collision filtering.

From Python, call ``ExampleFruitingSystem.regenerate()`` (optional seed) to build another instance while keeping the viewer. See ``apple_pick_sim/example_fruiting_system.py``.

## P0 variational fruiting (JSON + seed)

Range bounds live in `apple_pick_sim/fixtures/fruiting_system_ranges.json`.
The generator is `apple_pick_sim/fruiting_system.py` (module docstring has full API details).

**Geometry-only smoke check** (no viewer required; run from repo root):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges.json')
scene  = generate_scene(ranges, seed=42)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

**Generate and run a short headless VBD rollout** (run from repo root):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint, run_rollout
ranges = load_ranges('../apple_pick_sim/fixtures/fruiting_system_ranges.json')
scene  = generate_scene(ranges, seed=42)
run_rollout(scene, num_steps=20, sim_substeps=10)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

**Tests** (from repository root):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -v -p no:launch_testing
```
