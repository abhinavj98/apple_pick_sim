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
Follow the quickstart instructions in the [Newton README](https://github.com/newton-physics/newton/blob/main/README.md) to set up the environment and install dependencies.


#### Example Setup using `uv` (Recommended)
Navigate to the `newton` directory and creating a virtual environment:

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
