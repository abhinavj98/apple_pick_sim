# Coupled simulation API (post pre-gym cleanup)

**Purpose:** Canonical entry points for batched and single-env MuJoCo + VBD coupled fruiting after the V.3.3 pre-gym scope narrowing (2026-07). For architecture and settle→weld flow, see [`vectorized-coupled-fruiting.md`](vectorized-coupled-fruiting.md). For status and validation commands, see [`ROADMAP.md`](ROADMAP.md).

## Requirements

- **FR3 assets** under `assets/fr3/` (and `usd-core`) are required for all coupled robot builds. Missing assets fail fast at build or example startup — there is no placeholder TCP fallback.
- Install from repo root: `uv sync --extra gym --extra vic --extra dev`.

## Public runtime API (`import apple_pick_sim.coupled_fruiting as cf`)

Use this for batched heterogeneous simulation, gym migration (V.3.3+), and tests.

| Symbol | Role |
| ------ | ---- |
| `BatchedHeterogeneousCoupledSim` | Runtime parent: `step()`, `gather_obs()`, scene/layout accessors |
| `BatchedHeterogeneousCoupledSimConfig` | Config dataclass + presets (`defaults()` / `gym_defaults()` → controller `vic`; `test_minimal()` → `direct` for CPU tests) |
| `build_batched_heterogeneous_scene` | Config-driven build (settle, weld, diagnostics) |
| `settle_vbd_substeps`, `quiet_all_cable_bodies`, `seed_fix_to_apple_from_settled*` | Settle-then-weld helpers |
| `SettledCheckpoint`, `settle_cache_path_for` | Optional disk cache for free-proxy settle state (`use_settle_cache=False` by default; opt in with `--use-settle-cache`) |

`RobotConfig.kind` is **`"fr3"` only**. Batched config validation rejects other values.

Low-level `build_*` functions are **not** re-exported from `coupled_fruiting.__init__`. Import them explicitly from `apple_pick_sim.coupled_fruiting.builders` when needed (gym, diagnostics, single-env scripts).

## Low-level builders (`apple_pick_sim.coupled_fruiting.builders`)

| Function | Use when |
| -------- | -------- |
| `build_heterogeneous_coupled_fruiting_fr3(ranges, params_list, …)` | N worlds with per-env `FruitingSystemParams` (heterogeneous DR) |
| `build_coupled_fruiting_fr3(ranges, seed, …)` | Single world; seed-based sampling; **gym envs** until V.3.4 |

**Removed (do not import):** `build_coupled_fruiting_placeholder`, `build_batched_coupled_fruiting_*`, `build_heterogeneous_coupled_fruiting_placeholder`, and placeholder TCP helpers.

Homogeneous batched tests use `build_heterogeneous_coupled_fruiting_fr3` with identical params per env (see `build_homogeneous_batched_fr3` in `apple_pick_sim/tests/conftest.py`).

## Runnable examples

| Script | Role |
| ------ | ---- |
| **`example_batched_heterogeneous_coupled_sim.py`** | **Canonical** batched entry: CLI + viewer only; logic in library. Defaults: `--controller vic`, settle cache **off** (`--use-settle-cache` to opt in). |
| `example_coupled_fruiting.py` | Single-env FR3 + VIC/EE teleop (diagnostics, keyboard); default controller `vic` |
| `example_fr3_keyboard.py` | FR3 kinematic keyboard smoke (no fruiting tree) |

Legacy monoliths under `examples/legacy/` were removed in the pre-gym cleanup.

## Typical batched workflow

```python
from apple_pick_sim.coupled_fruiting import (
    BatchedHeterogeneousCoupledSim,
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.fruiting_system import (
    load_ranges,
    parse_sim_build,
    sample_heterogeneous_params_list,
)

ranges = load_ranges("apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json")
params = sample_heterogeneous_params_list(ranges, topology_seed=42, num_envs=4)
cfg = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=4)
# Optional: ranges may include top-level ``sim_build`` (VIC + joint overrides).
# Default variance fixture ships it; ``parse_sim_build(ranges)`` is ``None`` when absent.
_ = parse_sim_build(ranges)
cfg.validate()

sim = BatchedHeterogeneousCoupledSim(cfg, params, ranges)  # use_settle_cache=False by default
sim.step(None)  # or per-env actions when controller allocates buffer
```

Canonical batched examples (`example_batched_heterogeneous_coupled_sim.py`,
sys-ID collect / MMD grid) apply `sim_build` from the loaded ranges path when
present; controller default is **vic**. Settle disk cache is **off** unless
`use_settle_cache=True` / `--use-settle-cache`. See
`docs/material-parameter-sampling.md` and `docs/damping-tuning.md`.

Headless smoke (from repo root):

```bash
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
```

## Gym (V.3 Done)

Gym envs still build via explicit import:

```python
from apple_pick_sim.coupled_fruiting.builders import build_coupled_fruiting_fr3
```

Batched gym (`ApplePickBatchedBaseEnv` / Vic / SysId) already uses
`BatchedHeterogeneousCoupledSim`. Legacy single-world gym envs/examples on that
backend at `num_envs=1` are marked **Done** in ROADMAP **V.3.4–V.3.5**.

## Diagnostics

FR3-only scripts under `apple_pick_sim/diagnostics/` (`verify_coupling.py`, `benchmark_coupling.py`, `benchmark_batched_heterogeneous.py`, `sweep_settle_weld_stability.py`, `log_settle_ke_decay.py`). They import `build_coupled_fruiting_fr3` or `build_heterogeneous_coupled_fruiting_fr3` directly — not the package `__init__` re-exports.
