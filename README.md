# Apple Pick Sim

This repository contains simulation code for robotic apple picking using the [Newton](https://github.com/newton-physics/newton) physics engine.

**New to this codebase?** Start with [`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md) for a map of the packages, the doc set, and where to read code first. For project intent and current priorities, see [`docs/VISION.md`](docs/VISION.md) and [`docs/ROADMAP.md`](docs/ROADMAP.md).

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

### 2. Install dependencies

From the **repository root**, sync the project environment (`pyproject.toml` path-depends on the `newton/` submodule):

```bash
uv sync --extra gym --extra vic --extra dev
```

| Extra | Purpose |
| ----- | ------- |
| *(base)* | `newton[examples]` — sim, viewers, scripts |
| `gym` | Gymnasium envs (`apple_pick_gym/`) and Dash dataset dashboard |
| `vic` | VIC joint-torque teleop (`newton[torch-cu12]`, PyTorch) |
| `dev` | pytest + gymnasium |

Minimal install (P0 fruiting only, no gym/VIC/tests):

```bash
uv sync
```

All `uv run` commands below assume the **repository root** as the current working directory. `apple_pick_sim` and `apple_pick_gym` are installed editable from the root package.

## Running the simulation

### `example_apple_stem.py`

```bash
uv run python apple_pick_sim/examples/example_apple_stem.py
```

This runs the apple simulation with three branch stiffness presets. The terminal prints forces and torques on the stem. To apply forces on the apple, use right-click and drag on the apple in the viewer.

### `example_fruiting_system.py` (variational fruiting)

Default fixture: **`fruiting_system_ranges_real_world_proxy.json`**. This fixture is the bench-proxy
geometry described in `docs/real-world-proxy.md`, but its JSON currently sets
`"topology": "linear_chain"` (serial primary → … → apple), **not** the T-junction topology
(primary simply supported along ±X, spur/stem/apple from the **T center**) that document
specifies — see the topology caveat in `docs/real-world-proxy.md`. The companion
`fruiting_system_ranges_real_world_proxy_variance.json` (no explicit `"topology"` key) does
default to the T-junction builder. Each run draws a new sample unless you pass `--seed`. Pass
`--json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json` for the wide-angle
procedural variance chain. Unit tests use `fruiting_system_ranges_straight_rod_test.json` for
deterministic, nearly vertical chains. Uses the same Newton viewer pattern as the stem example.

```bash
uv run python apple_pick_sim/examples/example_fruiting_system.py
```

Useful options (see also the script docstring):

```bash
uv run python apple_pick_sim/examples/example_fruiting_system.py \
  --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 123  # legacy variance chain
```

**Collisions:** The library default `generate_scene(..., enable_self_collisions=True)` only relies on Newton’s joint **parent/child** collision filters (adjacent rod segments do not collide). **Non-adjacent** chain capsules can still collide with each other and with the apple.

The interactive examples **`example_fruiting_system.py`** and **`example_coupled_fruiting.py`** disable intra-chain self collisions by default (`enable_self_collisions=False`). Pass **`--enable-self-collision`** to opt in to non-adjacent link–link contacts (same semantics as `enable_self_collisions=True` above). **Ground contact is unchanged** in either mode.

From Python, call `ExampleFruitingSystem.regenerate()` (optional seed) to rebuild while keeping the viewer. See `apple_pick_sim/examples/example_fruiting_system.py`.

## P0 variational fruiting (JSON + seed)

Range fixtures live under `apple_pick_sim/fixtures/`: **`fruiting_system_ranges_real_world_proxy.json`**
(bench proxy geometry, `linear_chain` topology — see topology note above; default for
`example_fruiting_system.py` and `example_coupled_fruiting.py`),
**`fruiting_system_ranges_example_variance.json`** (wide-angle procedural variance), and
**`fruiting_system_ranges_straight_rod_test.json`** (nearly −Z chain; default for tests). The
generator is the **`apple_pick_sim/fruiting_system/`** package (`params.py`, `build.py`, `scene.py`,
`coupled.py`; public API via `apple_pick_sim.fruiting_system`).

**Geometry-only smoke check** (no viewer):

```bash
uv run python -c "
from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint
ranges = load_ranges('apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json')
scene  = generate_scene(ranges, seed=42)
import json; print(json.dumps(geometry_fingerprint(scene), indent=2))
"
```

**Short headless VBD rollout** (optional: pass ``collision_pipeline=example_collision_pipeline(scene.model)`` to match the viewer’s ``create_collision_pipeline`` path):

```bash
uv run python -c "
from apple_pick_sim.fruiting_system import (
    load_ranges, generate_scene, geometry_fingerprint, run_rollout,
    example_collision_pipeline,
)
ranges = load_ranges('apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json')
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
- **Control:** ``example_coupled_fruiting.py`` defaults to **FR3 + VIC joint-torque teleop** (dynamic arm, plant wrenches on TCP ``body_f``). Requires PyTorch: ``uv sync --extra vic``. Tune with ``--vic-linear-k``, ``--vic-linear-d``, ``--vic-angular-k``, ``--vic-angular-d``.
- **Step modes:** default = full coupled loop; ``--only-vbd`` = cable only; ``--only-mjc`` = MuJoCo robot + proxy sync.

```bash
uv run python -m unittest apple_pick_sim.tests.test_fr3_usd_import -v
```

Smoke:

```bash
uv run python -c "
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.coupled_fruiting import build_coupled_fruiting_placeholder
ranges = load_ranges('apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json')
scene = build_coupled_fruiting_placeholder(ranges, seed=0)
scene.coupled_substep(1e-4)
print('coupled_substep_ok')
"
```

Interactive **Newton viewer** (shows the **cable** scene: rods + apple + gripper proxy, which mirrors the coupling). Optional **`--mujoco-viewer`** opens MuJoCo’s passive viewer for the **TCP placeholder** rigid body (**second window**).

```bash
uv run python apple_pick_sim/examples/example_coupled_fruiting.py
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 120
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --mujoco-viewer --json apple_pick_sim/fixtures/fruiting_system_ranges_example_variance.json --seed 0
# Staggered coupling wrench debug (Plots panel in ViewerGL): lagged → MuJoCo vs fresh ← VBD harvest
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --debug-coupling-forces --seed 42
# TCP force as a yellow arrow at the robot TCP (scale: --tcp-force-scale, --tcp-force-arrow-gain, min/max length)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --viewer gl --tcp-force-arrow --seed 42
# Bundled FR3 + custom EE (default; requires usd-core + assets/fr3/)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 60
# FR3 keyboard teleop (VIC joint torques; focus ViewerGL window)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py \
  --fr3-keyboard --viewer gl
# Optional second window for the MuJoCo robot model
uv run python apple_pick_sim/examples/example_coupled_fruiting.py \
  --fr3-keyboard --mujoco-viewer --viewer gl
# Stem-harvest path: weld proxy to apple (default is --no-fix-to-apple / velocity-delta)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --fix-to-apple --seed 42
# Placeholder TCP (no FR3 assets)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --robot placeholder --viewer null --num-frames 60
```

### `example_batched_coupled_fruiting.py` (homogeneous batches)

Batched coupled fruiting: **N** worlds via ``replicate()``, settle→weld init, then FR3 teleop via ``BatchedTemplateIK`` per-env scatter. This reference example uses the same keyboard velocity on all envs (homogeneous smoke). For **independent** per-env seeds, per-env material θ, and per-env actions (shipped), see ``example_batched_heterogeneous_coupled_fruiting.py`` and **`docs/vectorized-coupled-fruiting.md`**. Current active work (batched sim API extraction + gym migration) is tracked in **`docs/ROADMAP.md`**.

```bash
# Headless smoke (settle→weld)
uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \
  --viewer null --num-frames 500 --num-envs 4 --fix-to-apple --seed 42

# Interactive keyboard teleop
uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \
  --num-envs 4 --env-spacing 2.5 2.5 0 --fix-to-apple --controller direct \
  --fr3-keyboard --viewer gl --seed 42

# Fast robot for CI
uv run python apple_pick_sim/examples/example_batched_coupled_fruiting.py \
  --viewer null --num-frames 120 --robot placeholder --num-envs 2 --fix-to-apple

# Heterogeneous batches: independent per-env material θ, per-env IK bootstrap, per-env actions
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
```

**FR3 keyboard teleop** (TCP velocity + IK; ``--viewer gl``, focus the window — **I/K J/L R/F** translate, **U/O T/G Z/X** rotate; **not W/S**, those move the camera):

- **Coupled fruiting + arm (default):** ``example_coupled_fruiting.py`` with ``--fr3-keyboard --viewer gl`` (VIC joint torques).
- **Robot only (kinematic FK, no MuJoCo step):** ``example_fr3_keyboard.py`` — useful for IK/viewer smoke without the fruiting tree.

```bash
uv run python apple_pick_sim/examples/example_fr3_keyboard.py --viewer gl
uv run python apple_pick_sim/examples/example_fr3_keyboard.py --viewer null --num-frames 120
```

### Validation (fast test gate)

After changes to fruiting, coupling, or gym code:

```bash
# Fast sim tests (excludes @pytest.mark.slow)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"

# Gym env tests
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/ -q

# Coupled example smoke (headless; requires --extra vic for default FR3+VIC path)
uv run python apple_pick_sim/examples/example_coupled_fruiting.py --viewer null --num-frames 60
```

GPU coupling inventory and benchmarks: `docs/gpu-coupling-optimization.md`.

## Tests

From the repository root (requires `uv sync --extra dev`):

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"
```

Full suite including slow stability tests:

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -v
```

M1 coupling stability (longer-horizon; includes ``slow`` tests):

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_coupling_stability.py -q
```

Optional slow tests only (500+ substep stability, FR3 long horizon):

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -m slow -q
```

M1 coupling benchmark (ms/substep; see ``docs/gpu-coupling-optimization.md``):

```bash
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cuda:0 --mujoco-gpu --warmup-substeps 30 --bench-substeps 300
# CPU MuJoCo baseline:
uv run python apple_pick_sim/diagnostics/benchmark_coupling.py \
  --robot placeholder --device cpu --mujoco-cpu --warmup-substeps 30 --bench-substeps 300
```

Headless CUDA graph (coupled example):

```bash
uv run python apple_pick_sim/examples/example_coupled_fruiting.py \
  --viewer null --cuda-graph --num-frames 200
```

Headless **coupling verification** (applied vs harvested wrench, TCP–proxy pose drift; exit 1 on threshold breach):

```bash
uv run python apple_pick_sim/diagnostics/verify_coupling.py \
  --num-substeps 600 --max-force 5 --max-torque 1
```

### Gymnasium environments (`apple_pick_gym/`)

Headless Gymnasium wrappers over the coupled FR3 stack. Registered envs include `ApplePickCoupled-v0`, `ApplePickVic-v0`, `ApplePickSysId-v0`, and `ApplePickReplay-v0`. Coupled/VIC/SysID/Replay envs expose `Dict` observations and set `info["obs_schema"] == "v3"` on `reset()` / `step()`.

**Observation contract:** key names, shapes, units, and env-specific semantics are documented in [`docs/gym-observation-contract.md`](docs/gym-observation-contract.md). Check `obs_schema` when loading rollouts recorded before the v1 replay key rename (`woody_start` / `woody_end` → `woody_part_*`).

`ApplePickCoupled-v0` uses `Discrete(13)` keyboard-style actions with woody part poses/forces, apple position, and TCP wrench/velocity. See `apple_pick_gym/envs/apple_pick_coupled_env.py`.

```bash
uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/ -q
```

The root `pyproject.toml` installs `apple_pick_sim` and `apple_pick_gym` editable and path-depends on the `newton/` submodule.

### M3.0 §2.1 quasi-static sys-ID (`ApplePickSysId-v0`)

Stepped push–hold excitation along Fibonacci-hemisphere pull directions. Spec: `docs/system_identification.md` (see "Implementation notes: §2.1 quasi-static stepped mapping").

```bash
# One-direction smoke (2 cm steps, 10 cm total)
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --n-directions 1 --movement-per-step-m 0.02 --total-movement-m 0.10 \
  --move-speed-mps 0.2

# Pull-direction geometry figure (default 90° hemisphere; matches collection)
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output pull_directions.png

# Tests
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_sim/tests/test_visualize_pull_directions.py \
  apple_pick_gym/tests/test_sysid_env.py -q
```

### M3 replay and digital-twin setup

Sys-ID recordings can be replayed with `ApplePickReplay-v0`. Parquet recordings are observation-first; privileged `.npz` snapshots are opt-in (`--save-snapshot`) for exact sim-to-sim baseline comparisons. A named digital-twin fixture catalog (`apple_pick_sim/fixtures/digital_twin_fixture_catalog.json`) is planned to list fixture names, base poses, observation fixtures, and smoke commands, but **that catalog file is not committed yet** — see the "Known gap" in `docs/digital-twin.md` before relying on it. Specs: [`docs/sysid-trajectory-storage.md`](docs/sysid-trajectory-storage.md) and [`docs/digital-twin.md`](docs/digital-twin.md).

```bash
# Collect a short observation-only dataset (no privileged snapshot by default)
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 \
  --output /tmp/sysid_dataset

# Optional privileged baseline for sim-to-sim comparison
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 --save-snapshot \
  --output /tmp/sysid_dataset_with_snapshot

# Replay and print dataset-vs-live observation errors
uv run python apple_pick_gym/examples/example_gym_replay.py \
  --dataset /tmp/sysid_dataset --viewer null

# Inspect collected raw trajectories in a local browser dashboard
uv run python apple_pick_gym/examples/dashboard_sysid_dataset.py \
  --dataset /tmp/sysid_dataset

# List recorded episodes before running parameter sweeps
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null --list-episodes

# Diagnostic bend-stiffness smoke: one candidate, observation-only replay
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null \
  --primary-bend-stiffness-values 10 \
  --secondary-bend-stiffness-values 10 \
  --spur-bend-stiffness-values 10 \
  --stem-bend-stiffness-values 10 \
  --max-candidates 1

# Expand each axis to sweep primary/secondary/spur/stem bend stiffnesses and rank by MMD
uv run python apple_pick_gym/examples/run_system_identification.py \
  --dataset /tmp/sysid_dataset --viewer null \
  --primary-bend-stiffness-values 10,25,50 \
  --secondary-bend-stiffness-values 10,25,50 \
  --spur-bend-stiffness-values 10,25,50 \
  --stem-bend-stiffness-values 10,25,50 \
  --mmd-output /tmp/apple_pick_mmd_grid

# Digital-twin reconstruction tests (2 known failures pending catalog fixture files — see docs/digital-twin.md)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q

# Pull-direction geometry figure
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output /tmp/apple_pick_pull_directions.png

# Storage + replay tests
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_trajectory_store.py \
  apple_pick_gym/tests/test_replay_env.py -q
```

The MMD grid command writes `mmd_results.csv` plus compact diagnostics:
`mmd_ranked_loss.png`, `mmd_direction_heatmap.png`, and
`mmd_stiffness_sensitivity.png`.
