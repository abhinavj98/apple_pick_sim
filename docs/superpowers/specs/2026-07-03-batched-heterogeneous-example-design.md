# V.3.2 — Thin heterogeneous batched example (design spec)

**Date:** 2026-07-03  
**Slice:** ROADMAP **V.3.2** — thin heterogeneous example  
**Status:** Implemented
**Canonical living doc:** `docs/handbook-coupled-simulation.md`

---

## Goal

Replace the monolithic `example_batched_heterogeneous_coupled_fruiting.py` with a **thin CLI + viewer wrapper** around the library runtime introduced in **V.3.1**. The new example demonstrates how callers wire argparse, teleop UX, and Newton viewer loops against `BatchedHeterogeneousCoupledSim` without embedding build/settle/controller logic.

**Success criteria (from brainstorming):**

1. New file `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py` runs headless and interactive smoke paths.
2. `test_heterogeneous_coupled_fruiting.py` (and any tests importing the old example) are ported to use **library** `BatchedHeterogeneousCoupledSim` / build API — **not** example imports.
3. Documented `uv run` smoke commands land in this spec and (after implementation) in ROADMAP validation notes.
4. Old example remains until parity is proven; docs then mark it deprecated and schedule removal.

---

## Prerequisites

| Prerequisite | Spec / artifact |
|--------------|-----------------|
| **V.3.1 step A — config + build** | [`2026-07-03-batched-heterogeneous-build-design.md`](2026-07-03-batched-heterogeneous-build-design.md) (config dataclasses, `build_batched_heterogeneous_scene`) |
| **V.3.1 step B — runtime** | [`2026-07-03-batched-heterogeneous-coupled-sim-design.md`](2026-07-03-batched-heterogeneous-coupled-sim-design.md) (`BatchedHeterogeneousCoupledSim`, `step`, `gather_obs`, settle cache) |
| **Library on branch** | `feature/batched-sim-config` (or merged main) must expose `BatchedHeterogeneousCoupledSim` + `BatchedHeterogeneousCoupledSimConfig` |
| **Reference behavior** | `apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py` (pre-extraction monolith) |

Throughout this doc, **CoupledSim** means `apple_pick_sim.coupled_fruiting.BatchedHeterogeneousCoupledSim`.

---

## Boundaries

### Example owns

| Concern | Notes |
|---------|-------|
| `argparse` via `newton.examples.create_parser()` + example-only flags | Private `_config_from_args(args) -> BatchedHeterogeneousCoupledSimConfig` |
| DR sampling orchestration | `load_ranges` + `sample_heterogeneous_params_list` before `CoupledSim(...)` |
| Teleop UX | Keyboard / scripted velocity → `(num_envs, 6)` `torch.float32` actions each frame |
| Viewer render loop | `viewer.begin_frame` / `log_state` / `end_frame`; world offsets for multi-env GL |
| Minimal status printing | Startup banners, optional periodic apple-z / TCP lines (v1 minimal) |
| `test_final()` smoke | Apple height sanity after run (mirror old example) |

### Library owns (CoupledSim)

| Concern | Notes |
|---------|-------|
| `build()` / `__init__` | Settle-then-weld, IK bootstrap, settle cache load/save |
| `step(actions)` | Controller dispatch, clipping, coupled / VBD substeps |
| `gather_obs()` | GPU batched obs when `config.obs.allocate_buffers=True` |
| `validate_actions` | Shape/dtype/device checks via `ControllerConfig` |
| Physics | No viewer, no argparse, no keyboard |

### Explicit non-goals (this slice)

- **No** library `Config.from_args()` or argparse in `apple_pick_sim/coupled_fruiting/`.
- **No** port of v2 debug/viz flags (see [Out of scope](#out-of-scope-v2-follow-up)).
- **No** deletion of `example_batched_heterogeneous_coupled_fruiting.py` until tests + smoke parity.

---

## File layout

```
apple_pick_sim/examples/
  example_batched_heterogeneous_coupled_sim.py   # NEW — thin wrapper (this spec)
  example_batched_heterogeneous_coupled_fruiting.py  # KEEP until parity; then deprecate

apple_pick_sim/coupled_fruiting/
  batched_heterogeneous_config.py              # BatchedHeterogeneousCoupledSimConfig (V.3.1)
  batched_heterogeneous_build.py               # build_batched_heterogeneous_scene (V.3.1 step A)
  batched_heterogeneous_coupled_sim.py         # CoupledSim runtime (V.3.1 step B)

apple_pick_sim/tests/
  test_heterogeneous_coupled_fruiting.py         # MIGRATE off example imports
  test_batched_heterogeneous_coupled_sim.py      # Already library-scoped (V.3.1 step B)
  test_batched_heterogeneous_build.py            # Already library-scoped (V.3.1 step A)
```

Target size: **~250–400 lines** for the new example (parser, `_config_from_args`, teleop helpers, render loop, `main`). Compare: current monolith **~1377 lines**.

---

## Runtime architecture

```
┌─────────────────────────────────────────────────────────────┐
│  example_batched_heterogeneous_coupled_sim.py               │
│  _make_parser() → newton.examples.init() → viewer, args     │
│  ranges, per_env_params = sample from args                  │
│  config = _config_from_args(args)                           │
│  sim = CoupledSim(config, per_env_params, ranges,           │
│                   viewer=viewer, use_settle_cache=...,      │
│                   force_settle=...)                         │
│  loop: actions = build_actions(...) → sim.step(actions)     │
│        render(sim.scene, viewer) → sleep(frame_dt)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  BatchedHeterogeneousCoupledSim (library)                   │
│  build / settle cache / controllers / step / gather_obs       │
└─────────────────────────────────────────────────────────────┘
```

---

## Main loop pseudocode

```python
def main():
    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    config = _config_from_args(args)
    ranges = load_ranges(config.domain_randomization.resolved_ranges_path())
    seed = config.domain_randomization.topology_seed or secrets.randbelow(2**31 - 1)
    per_env_params = sample_heterogeneous_params_list(
        ranges, topology_seed=seed, num_envs=config.runtime.num_envs
    )

    _print_startup(config, ranges, per_env_params)  # minimal v1

    sim = BatchedHeterogeneousCoupledSim(
        config,
        per_env_params,
        ranges,
        viewer=viewer,
        use_settle_cache=args.use_settle_cache,
        force_settle=args.force_settle,
    )

    _setup_viewer(viewer, sim)  # set_model, world_offsets from config.runtime.env_spacing

    frame = 0
    while viewer.is_running():
        actions = build_frame_actions(sim, viewer, args)  # (num_envs, 6) float32 on sim.device
        sim.step(actions if config.robot.step_mode != "vbd_only" else None)
        render_frame(viewer, sim)
        if args.status_every and frame % args.status_every == 0:
            print_minimal_status(sim)
        frame += 1
        time.sleep(max(0.0, sim.frame_dt))

    test_final(sim)
    cleanup(viewer, sim)


def build_frame_actions(sim, viewer, args) -> torch.Tensor:
    """Example-owned teleop → per-env action tensor."""
    n = sim.num_envs
    device = sim.device
    vel = read_scripted_or_keyboard_velocity(viewer, args, sim)  # EEVelocity-like 6-vector source
    row = torch.tensor(
        [*vel.linear, *vel.angular], dtype=torch.float32, device=device
    )
    return row.unsqueeze(0).expand(n, -1).contiguous()
```

**Differences from monolith:**

- No `ExampleBatchedHeterogeneousCoupledFruiting.simulate()` / inline `update_fr3_ee_teleop*` — library `step(actions)` owns controller + substeps.
- No deferred visible settle loop in v1 (monolith `run_visible_settle` + `inspect_settle` deferred to v2).
- Settle cache: monolith always settles inline; new example exposes `use_settle_cache` / `force_settle` (library feature).

**Headless default (Linux):** preserve monolith behavior — if no `DISPLAY`/`WAYLAND_DISPLAY` and no `--viewer`, append `--viewer null --num-frames 200`.

---

## `_config_from_args` field mapping

Private function in the example only. Starts from `BatchedHeterogeneousCoupledSimConfig.defaults()` and applies overrides. Call `config.validate()` before returning (or rely on `CoupledSim` constructor validation).

### v1 CLI flags → config

| CLI flag | Config field(s) | Notes |
|----------|-----------------|-------|
| *(newton.examples)* `--device` | `runtime.device` | Passed through `resolve_sim_device` inside config |
| `--num-envs` | `runtime.num_envs` | Default `4` (match monolith) |
| `--env-spacing X Y Z` | `runtime.env_spacing` | Viewer grid only; default `(2, 2, 2)` |
| `--hz` | `runtime.control_hz` | Default `30.0`. Keep `runtime.sub_dt = 1/1800` (monolith physics rate). Then `substeps_per_step = round(1/(hz*sub_dt))` → 60 at 30 Hz (parity with monolith `sim_substeps=60`) |
| `--seed` | `domain_randomization.topology_seed` | `None` → example picks random seed before sampling |
| `--json PATH` | `domain_randomization.ranges_path` | `None` → default variance fixture |
| `--robot {placeholder,fr3}` | `robot.kind` | FR3 falls back to placeholder when assets missing (library warns) |
| `--only-vbd` | `robot.step_mode = "vbd_only"` | Mutually exclusive with `--only-mjc` (reject) |
| `--only-mjc` | — | `SystemExit` (unsupported, same as monolith) |
| `--controller {direct,ee,vic}` | `controller.mode` | `vic` requires `uv sync --extra vic` + FR3 coupled |
| `--fix-to-apple` / `--no-fix-to-apple` | `robot.fix_to_apple` | Default `True`. Weld flags on gripper applied at build via library |
| `--enable-self-collision` | `scene.enable_self_collisions` | Default `False` |
| `--apple-woody-collision` / `--no-apple-woody-collision` | `scene.enable_apple_woody_collisions` | Default `True` |
| `--proxy-woody-collision` / `--no-proxy-woody-collision` | `scene.enable_proxy_woody_collisions` | Default `True` |
| `--settle-substeps` | `scene.settle_substeps` | Default `5000` |
| `--settle-gravity-ramp` / `--no-settle-gravity-ramp` | `scene.settle_gravity_ramp` | Default `False` |
| `--use-settle-cache` / `--no-use-settle-cache` | *(constructor)* `use_settle_cache` | **New flag.** Default `True`. Not part of frozen config |
| `--force-settle` | *(constructor)* `force_settle` | **New flag.** Default `False`. Bypasses cache read |
| `--scripted-ee-vel VX VY VZ` | *(example state)* | Not in config; drives `build_frame_actions` when keyboard off |
| `--fr3-keyboard` | *(example state)* | Uses `fr3_robot.read_keyboard_ee_velocity(viewer, ...)` |
| `--vic-linear-k`, `--vic-linear-d`, `--vic-angular-k`, `--vic-angular-d` | `controller.vic_gains` | Only when `--controller vic` |
| `--status-every` | *(example)* | Default `60`; `0` disables |
| `--print-robot-state` / `--no-print-robot-state` | *(example startup)* | One-shot post-build status via `print_batched_robot_status` |

### Fixed / derived (not exposed in v1 CLI)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `runtime.sub_dt` | `1/1800` | Monolith `sim_dt`; keeps parity at `--hz 30` |
| `controller.linear_speed`, `angular_speed` | `0.1`, `0.1` | Monolith `_FR3_TELEOP_*` |
| `controller.ik_iterations` | `128` | Monolith |
| `fruiting_system.joint_angular_kd_overrides` | defaults in config module | Applied at build in library |
| `settle_diagnostics` | `SettleDiagnosticsConfig()` defaults when settle runs | v1: no CLI tuning of KE knobs |
| `obs.allocate_buffers` | `True` | Enables `gather_obs` for future v2 viz; cheap in v1 |
| `mujoco` | defaults | Unchanged |

### `_resolve_step_mode` helper (example)

```python
def _resolve_step_mode(args) -> Literal["coupled", "vbd"]:
    if args.only_vbd:
        return "vbd"
    if args.only_mjc:
        raise SystemExit("--only-mjc is not supported for heterogeneous batched builds.")
    return "coupled"
```

Maps to `robot.step_mode`: `"vbd"` → `"vbd_only"`, else `"coupled"`.

---

## Teleop → action tensor (example sketch)

v1 supports **one shared command** broadcast to all envs (monolith default without `--demo-per-env-actions` / `--noisy-action`).

```python
@dataclass
class TeleopState:
    scripted_linear: tuple[float, float, float]
    use_keyboard: bool

def read_scripted_or_keyboard_velocity(viewer, args, sim) -> fr3_robot.EEVelocity:
    linear = tuple(args.scripted_ee_vel)  # default (0.05, 0, 0)
    angular = (0.0, 0.0, 0.0)
    if args.fr3_keyboard and hasattr(viewer, "is_key_down"):
        return fr3_robot.read_keyboard_ee_velocity(
            viewer,
            linear_speed=sim.config.controller.linear_speed,
            angular_speed=sim.config.controller.angular_speed,
        )
    return fr3_robot.EEVelocity(linear=linear, angular=angular)


def build_frame_actions(sim, viewer, args, teleop: TeleopState) -> torch.Tensor:
    import torch
    vel = read_scripted_or_keyboard_velocity(viewer, args, sim)
    row = torch.tensor([*vel.linear, *vel.angular], dtype=torch.float32, device=sim.device)
    return row.unsqueeze(0).expand(sim.num_envs, -1).contiguous()
```

**Placeholder robot:** same tensor shape; library `step` nudges world-0 joint from `actions[0, 0]` and broadcasts joint state.

**VBD-only:** `sim.step(None)`; ignore actions.

**Controller modes:** library configures `Fr3BatchedEEDirectJointController` / `Fr3BatchedEEVelocityController` / VIC from `config.controller.mode`; example never calls `update_fr3_ee_teleop*` directly.

---

## Parser surface (v1)

Extend `newton.examples.create_parser()` (inherits `--viewer`, `--num-frames`, `--device`, benchmark flags, etc.).

**Included in v1** (core parity with monolith minus deferred items):

- `--json`, `--hz`, `--seed`, `--num-envs`, `--env-spacing`
- `--enable-self-collision`, `--apple-woody-collision`, `--proxy-woody-collision`
- `--only-vbd`, `--only-mjc` (reject), `--robot`, `--controller`
- `--fr3-keyboard`, `--fix-to-apple`, `--settle-substeps`, `--settle-gravity-ramp`
- `--use-settle-cache` / `--no-use-settle-cache`, `--force-settle`
- `--scripted-ee-vel`, `--status-every`, `--print-robot-state`
- VIC gains when `--controller vic`

---

## Testing and migration plan

### Tests to migrate off example imports

`apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py` currently imports from the monolith in **11 places** (example class, `_make_parser`, `_resolve_step_mode`, `_print_per_env_params`, settle defer helpers, etc.).

| Test / group | Current dependency | Migration target |
|--------------|-------------------|------------------|
| `test_batched_heterogeneous_example_timing_matches_frame_dt` | Instantiates `ExampleBatchedHeterogeneousCoupledFruiting` | Assert `BatchedHeterogeneousCoupledSimConfig` + `runtime.frame_dt` from test config; or move to `test_batched_heterogeneous_config.py` |
| `test_batched_heterogeneous_only_vbd_parser_flag` | `_make_parser`, `_resolve_step_mode` | **Delete** or move to optional `test_example_batched_heterogeneous_coupled_sim.py` if parser tests are kept at all |
| `test_batched_heterogeneous_only_vbd_and_only_mjc_mutually_exclusive` | `_resolve_step_mode` | Same |
| `test_print_per_env_params_includes_all_rod_stiffnesses` | `_print_per_env_params` | Move helper to library test util or inline assertion on `per_env_params` fields |
| `test_batched_heterogeneous_only_mjc_rejected` | parser / step mode | Example test file or drop if covered by `_config_from_args` unit test |
| `test_batched_heterogeneous_only_vbd_builds_cable_only_scene` | Example build path | `BatchedHeterogeneousCoupledSim` with `robot.step_mode="vbd_only"`, `use_settle_cache=False` |
| `test_defer_settle_to_viewer_*` | Monolith defer-settle | **Defer** until v2 or reimplement against library if still needed |
| `test_batched_heterogeneous_only_vbd_runs_settle_with_gravity_ramp` | Example settle | `CoupledSim` + capsys on library settle logs, or build-only test |
| Parser / example integration tests | Example module | Relocate minimal parser tests to `test_example_batched_heterogeneous_coupled_sim.py` (optional, low priority) |

**Keep library-native tests unchanged:**

- Joint KD/KP batched kernels, DR sampling, stiffness per env, IK bootstrap (`_make_hetero_settle_then_weld`) — already use `build_heterogeneous_coupled_fruiting_fr3` or fruiting_system APIs.
- Long-term: refactor `_make_hetero_settle_then_weld` to use `BatchedHeterogeneousCoupledSimConfig.test_minimal` + `build_batched_heterogeneous_scene` (follow-on cleanup; not blocking V.3.2).

### `test_batched_heterogeneous_build.py` coupling

Imports `_make_hetero_settle_then_weld` from `test_heterogeneous_coupled_fruiting.py`. When migrating, extract shared fixture to `conftest.py` or a `tests/batched_hetero_fixtures.py` module to avoid cross-test imports.

### TDD order (implementation phase)

1. Add failing test: `CoupledSim` smoke with `test_minimal` config replaces example-based only-vbd test.
2. Implement `example_batched_heterogeneous_coupled_sim.py` + `_config_from_args`.
3. Migrate/remove example imports in `test_heterogeneous_coupled_fruiting.py`; extract shared fixtures.
4. Run full heterogeneous test suite + smoke commands below.
5. Update ROADMAP / vectorized doc reference to new example path; mark old example deprecated.

---

## Out of scope (v2 follow-up)

Deferred CLI / behavior from monolith (explicit user decision):

| Deferred flag / behavior | Monolith reference |
|--------------------------|-------------------|
| `--mark-endpoints` | Endpoint debug viz + `gather_batched_obs` console |
| `--tcp-force-arrow` (+ scale/gain/min/max) | `log_batched_tcp_force_arrows` |
| `--inspect-settle` | Post-settle pause before weld build |
| `--demo-per-env-actions` | Per-env scripted velocity scatter |
| `--noisy-action`, `--noisy-action-std` | Per-env Gaussian EE noise |
| `--mujoco-viewer` | Second passive MuJoCo window |
| Full settle KE CLI | `--settle-max-speed`, `--settle-ke-decay`, `--ke-sample-every`, `--ke-analysis-tail-fraction`, `--ke-min-peaks`, `--ke-peak-decay-rtol`, `--ke-peak-threshold-j`, `--settle-report-brief` |
| Visible settle loop | `run_visible_settle` progressive viewer settle |
| `--only-mjc` support | N/A (still rejected) |

v2 may add example helpers that call `sim.gather_obs()` + `batched_viz` without moving viz into the library.

---

## Validation commands

Run from repository root after implementation:

```bash
# Library regression (V.3.1)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_heterogeneous_config.py \
  apple_pick_sim/tests/test_batched_heterogeneous_build.py \
  apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py -q

# Heterogeneous suite (post migration — must not import old example)
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q

# Headless smoke — new thin example
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42

# FR3 keyboard (interactive; requires assets + display)
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --num-envs 4 --viewer gl --fr3-keyboard --seed 42

# VBD-only cable smoke
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 100 --num-envs 2 --only-vbd --settle-substeps 50 --seed 42

# Settle cache bypass (forces full settle)
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 50 --num-envs 2 --settle-substeps 20 \
  --force-settle --no-use-settle-cache --seed 42

# VIC (extra)
uv sync --extra vic
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 100 --controller vic --num-envs 2 --seed 42
```

**Parity check vs monolith (manual, pre-deprecation):**

```bash
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
```

Compare apple heights, world counts, and absence of tracebacks. After parity: add deprecation banner to monolith docstring and point to new example in `docs/ROADMAP.md` + `docs/handbook-coupled-simulation.md`.

---

## Documentation updates (post-implementation)

- `docs/ROADMAP.md` — mark V.3.2 done; swap smoke command to new example path.
- `docs/handbook-coupled-simulation.md` — reference thin example + library API.
- Monolith module docstring — `.. deprecated::` pointer to `example_batched_heterogeneous_coupled_sim.py`.

---

## Self-review and open items

| Item | Resolution / owner |
|------|-------------------|
| **`control_hz` default mismatch** — config module defaults `60` Hz; monolith defaults `30` Hz | `_config_from_args` must set `runtime.control_hz=args.hz` (default 30), not rely on bare `defaults()` timing |
| **Settle diagnostics in v1** — library enables KE reports when `settle_diagnostics` set; monolith prints heavily | v1: pass `settle_diagnostics=SettleDiagnosticsConfig()` when `settle_substeps > 0`, accept default console noise; trim in v2 CLI |
| **Visible settle / inspect-settle** — monolith defers settle to viewer when GL + no fix-to-apple | v1: library always settles at build; defer viewer-deferred settle to v2 |
| **Example parser tests** — worth keeping? | Prefer one small `test_example_batched_heterogeneous_coupled_sim.py` for `_resolve_step_mode` only; delete duplicate coverage from heterogeneous physics test file |
| **`test_final` apple tolerance** | Keep `z > -0.05` per env in example class |
| **CoupledSim naming in user docs** | Public API name is `BatchedHeterogeneousCoupledSim`; examples may alias locally for brevity but exports stay verbose |
| **Branch dependency** | Implementation blocked until V.3.1 merges or is available on working branch |
| **README sync** | Update `README.md` example commands when new script lands (per readme-runtime-verification rule) |

---

## Related specs

- [V.3.1 step A — build](2026-07-03-batched-heterogeneous-build-design.md)
- [V.3.1 step B — CoupledSim runtime](2026-07-03-batched-heterogeneous-coupled-sim-design.md)
- ROADMAP § **[V].3** — `docs/ROADMAP.md`
- Behavior reference — `docs/handbook-coupled-simulation.md`
