# Batched Stability Monitor — Design

**Date:** 2026-07-08
**Status:** Implemented
**Branch:** `feature/batched-sysid-mmd`

## Motivation

While inspecting sys-ID hold quasi-static reports (`apple_pick_sim/system_id/batched_hold_quasi_static.py`),
some `(structure, direction, amplitude)` combinations showed clear physical instability
(oscillating or non-converging force, large TCP excursion) traced to under-damped structure
parameters and pull directions that excite bending/swing modes. That analysis is currently
**post-hoc**: it runs after a full batched collection finishes, over recorded Parquet frames,
using CPU/NumPy. There is no way to detect instability **online**, per env, while the batched
sim is stepping — so unstable trials still get simulated, recorded, and written to disk before
anyone knows they should be discarded.

This spec defines a **parallelized, online stability monitor** that runs inside any batched
gym env's step loop, checking every parallel env each step using vectorized tensor ops (no
per-env Python loop), and reports which envs have gone unstable and why. It does not replace
the existing post-hoc sys-ID hold-quality analysis — it's a generic, cheaper, earlier signal
that any batched pipeline (sys-ID collection, MMD grid sweeps, VIC training, teleop) can use to
decide what to do with a misbehaving env (skip recording it, log it, flag it in a manifest,
etc.).

## Goals

- Detect instability **every control step**, across all parallel envs at once, using vectorized
  torch ops on the batched obs dict already produced by `apple_pick_gym` batched envs.
- Provide a small, generic set of **core safety checks** (NaN/Inf, force/torque cap, TCP speed
  bound) that work against any batched env's obs dict, with no sys-ID-specific assumptions.
- Provide a **plugin protocol** so callers can register additional, custom stability checks
  against any field present in that env's obs dict (e.g. a future sys-ID hold-quality check
  using `force_cv`/mean-drift, phase-gated).
- Fail fast and clearly when a plugin is used with an env whose obs dict doesn't have the
  fields that plugin needs.
- Stay a **standalone utility** — no changes to `ApplePickBatchedBaseEnv` or its subclasses.
  Callers explicitly call the monitor each step and decide what to do with the report
  (report-only; the monitor does not freeze, terminate, or otherwise mutate env state).

## Non-Goals

- Not shipping a concrete sys-ID hold-quality plugin (force_cv / mean-drift / TCP-excursion)
  in this project. The plugin protocol is designed to support it, but the plugin itself is a
  follow-up once this core lands.
- Not changing collection behavior (`collect_batched_quasi_static_dataset`, `on_step`, etc.) to
  act on stability reports (e.g. skip writes, abort per-env/per-structure). That's a follow-up
  consumer of this monitor, not part of this spec.
- Not wiring the monitor into `ApplePickBatchedBaseEnv.step()` or `info[...]` automatically.
- Not a replacement for the existing post-hoc `batched_hold_quasi_static.py` analysis, which
  remains the source of truth for sys-ID "quasi-static quality" gating.

## Architecture

New module: `apple_pick_gym/batched_envs/batched_stability_monitor.py`.

**Why `apple_pick_gym/` and not `apple_pick_sim/`:** the monitor's input contract is the batched
obs **dict** produced by `apple_pick_gym/batched_envs/obs_torch.py` (`ft_wrist`, `tcp_velocity`,
`apple_pos`, `woody_part_info`, plus per-env-type extras like `tcp_pos`, `robot_joint_q`,
`excitation_type` for the sys-ID env). That dict shape is a gym-layer convention, not something
`apple_pick_sim` itself defines — `apple_pick_sim.batched_obs.BatchedObsBuffers` is the
underlying raw warp-array state. Binding the monitor directly to the obs dict (rather than
explicit named-tensor arguments) is a deliberate choice: it lets custom plugins use **any**
field a specific env's `_gather_obs()` produces, not just the fields the core checks use. The
trade-off, accepted explicitly: plugins written against one env type's extra fields (e.g.
`excitation_type`) are not portable to env types that don't produce them — see "Plugin
compatibility" below for how this is caught early rather than silently.

```
caller's step loop
  → env.step(actions) → obs (torch, batched, from env._gather_obs())
  → monitor.check(obs, step_idx=...) → BatchedStabilityReport
  → caller decides what to do (log, skip write, flag manifest, etc.)
```

The monitor is **stateless with respect to the core checks** (pure function of the current
`obs`) — no rolling buffers are needed since NaN/Inf, force-cap, and speed-bound violations are
all instantaneous per-step conditions. Plugins, if stateful, own their own internal buffers
(e.g. a rolling per-env force-history tensor for a future windowed check); the monitor itself
does not accumulate anything across calls beyond the one small internal cache needed for the
apple-speed check (see below).

## Components

### `StabilityThresholds`

```python
@dataclass(frozen=True)
class StabilityThresholds:
    max_force_n: float = 300.0       # reuse DEFAULT_STEM_FORCE_CAP_N (scene.py)
    max_torque_nm: float = 100.0     # reuse DEFAULT_STEM_TORQUE_CAP_NM (scene.py)
    max_tcp_speed_mps: float = 5.0
    max_apple_speed_mps: float = 5.0
```

Defaults intentionally reuse the existing wrench-cap constants from
`apple_pick_sim/coupled_fruiting/scene.py` (`DEFAULT_STEM_FORCE_CAP_N = 300.0`,
`DEFAULT_STEM_TORQUE_CAP_NM = 100.0`) so "force/torque at cap" and "monitor-flagged unstable"
refer to the same physical limit already enforced elsewhere in the sim.

**Meaning of `stable`:** blow-up / unsafe (NaN, caps, speed), not hold quasi-static quality.
Episode exclude rules treat any `stable=False` frame as grounds to drop the `(structure, direction)`.
A deeper stability-monitor retune remains a follow-up (see sys-ID stable collect/replay design).

### `BatchedStabilityReport`

```python
@dataclass(frozen=True)
class BatchedStabilityReport:
    step_idx: int
    unstable: torch.Tensor          # (num_envs,) bool
    reasons: list[list[str]]        # per-env list of triggered reason codes (empty if stable)
```

Core check reason codes: `"nan_or_inf:<field>"`, `"force_cap_exceeded"`,
`"torque_cap_exceeded"`, `"tcp_speed_exceeded"`, `"apple_speed_exceeded"`. Plugin reason codes
are the plugin's own strings, appended into the same per-env list.

### `StabilityCheckPlugin` protocol

```python
@dataclass(frozen=True)
class PluginCheckResult:
    unstable: torch.Tensor          # (num_envs,) bool
    reasons: list[str | None]       # per-env reason when True, else None

class StabilityCheckPlugin(Protocol):
    name: str
    required_obs_keys: frozenset[str]

    def check(self, obs: Mapping[str, Any], *, step_idx: int) -> PluginCheckResult: ...
```

### `BatchedStabilityMonitor`

```python
class BatchedStabilityMonitor:
    def __init__(
        self,
        num_envs: int,
        *,
        known_obs_keys: set[str],
        thresholds: StabilityThresholds | None = None,
        plugins: Sequence[StabilityCheckPlugin] = (),
    ) -> None: ...

    def check(self, obs: Mapping[str, Any], *, step_idx: int) -> BatchedStabilityReport: ...
```

- `known_obs_keys` is the set of keys the caller's env's `obs` dict is known to contain
  (typically `set(obs.keys())` from one `reset()`/`step()` call, or a hardcoded list for a
  known env type).
- At construction, every plugin's `required_obs_keys` is checked against `known_obs_keys`.
  If any plugin requires a key not present, construction raises `ValueError` naming the plugin
  and the missing key(s) — **before** any stepping happens. See "Plugin compatibility" below.
- `check()` runs the core safety checks against `obs`, then calls every plugin's `.check(obs,
  step_idx=...)`, and merges all results into one `BatchedStabilityReport` (`unstable` is the
  elementwise OR across core + all plugins; `reasons[i]` is the concatenation of all triggered
  reason codes for env `i`).

### Core safety checks (run unconditionally, every `check()` call)

All vectorized across the batch dimension; no per-env Python loop.

1. **NaN/Inf:** `torch.isnan(x) | torch.isinf(x)`, reduced with `.any(dim=-1)`, for `ft_wrist`,
   `tcp_velocity`, `apple_pos`, and each junction's `anchor_force` in `obs["woody_part_info"]`.
2. **Force/torque cap:** `‖ft_wrist[:, :3]‖ > thresholds.max_force_n` or
   `‖ft_wrist[:, 3:]‖ > thresholds.max_torque_nm`.
3. **TCP speed bound:** `‖tcp_velocity[:, :3]‖ > thresholds.max_tcp_speed_mps`.
4. **Apple speed bound:** derived from `apple_pos` frame-to-frame delta if a previous-step
   cache is available; otherwise this check is a no-op for the first call. (Implementation
   detail: this is the one core check that needs one step of memory — a single `(num_envs, 3)`
   "previous apple_pos" tensor owned by the monitor instance, not exposed as report state.)

## Plugin compatibility (fail-fast construction)

Chosen behavior: **fail fast at construction**, not at first `check()` call, and not silently
skipped.

```python
monitor = BatchedStabilityMonitor(
    num_envs=env.num_envs,
    known_obs_keys=set(obs.keys()),
    plugins=[my_custom_plugin],
)
# raises ValueError immediately if my_custom_plugin.required_obs_keys - known_obs_keys
# is non-empty, e.g.:
#   ValueError: plugin 'hold_quality' requires missing obs keys: {'phase'}
```

Rationale: this catches a mismatched plugin/env pairing (e.g. a sys-ID-only plugin registered
against a VIC-training env) before any stepping starts, rather than discovering it mid-run or
silently getting a monitor that's quietly not checking what the caller thinks it's checking.

## Data flow example (illustrative, not part of this project's shipped code)

```python
obs, _info = env.reset(seed=0)
monitor = BatchedStabilityMonitor(num_envs=env.num_envs, known_obs_keys=set(obs.keys()))

for step_idx in range(n_steps):
    obs, _reward, _terminated, _truncated, _info = env.step(actions)
    report = monitor.check(obs, step_idx=step_idx)
    if bool(report.unstable.any()):
        unstable_envs = report.unstable.nonzero(as_tuple=True)[0].tolist()
        for i in unstable_envs:
            print(f"env {i} unstable at step {step_idx}: {report.reasons[i]}")
```

Callers remain fully in control of what "report only" means for them (print, log, skip a
Parquet write, mark a manifest row, etc.) — this spec does not prescribe that behavior.

## Testing Plan (TDD)

New test file: `apple_pick_gym/tests/test_batched_stability_monitor.py`, following the existing
synthetic-tensor pattern used in `apple_pick_sim/tests/test_batched_hold_quasi_static.py`.

1. **Core checks, one assertion focus per test:**
   - NaN in `ft_wrist` for one env → only that env flagged, reason includes
     `"nan_or_inf:ft_wrist"`.
   - Inf in `tcp_velocity` → flagged with `"nan_or_inf:tcp_velocity"`.
   - Force norm over `max_force_n` → flagged with `"force_cap_exceeded"`.
   - Torque norm over `max_torque_nm` → flagged with `"torque_cap_exceeded"`.
   - TCP speed over `max_tcp_speed_mps` → flagged with `"tcp_speed_exceeded"`.
   - Fully nominal obs across all envs → `report.unstable.any() is False`, all `reasons` empty.
2. **Vectorization correctness:** a 5-env batch where only envs `{1, 3}` are unstable (for
   different reasons each) → assert the exact boolean mask `[False, True, False, True, False]`
   and that `reasons[1]`/`reasons[3]` contain the expected codes while others are empty lists.
3. **Plugin merging:** a fake plugin (`required_obs_keys={"tcp_velocity"}`) that always flags
   env index 2 with reason `"fake_reason"` → report includes core reasons (if any) plus
   `"fake_reason"` for env 2 only.
4. **Fail-fast construction:** a fake plugin requiring `{"phase"}` constructed with
   `known_obs_keys={"ft_wrist", "tcp_velocity"}` → `pytest.raises(ValueError, match="phase")`,
   and the error message names the plugin's `name`.
5. **Threshold overrides:** `StabilityThresholds(max_force_n=10.0)` flags a force magnitude
   (e.g. 15 N) that would pass under the default `200.0` threshold.
6. **Apple-speed memory check:** first `check()` call does not raise/flag due to missing
   previous-position state; second call with a large `apple_pos` jump does flag
   `"apple_speed_exceeded"`.

All tests operate on plain CPU `torch.Tensor`s in a hand-built `obs` dict (no live env,
gymnasium marker, or GPU required) — fast, deterministic, and consistent with how
`test_batched_hold_quasi_static.py` tests its synthetic arrays.

Validation command (once implemented): `uv run --env-file pytest.env python -m pytest
apple_pick_gym/tests/test_batched_stability_monitor.py -q`

## Follow-ups (explicitly out of scope here)

- A concrete `HoldQualityStabilityPlugin` implementing the sys-ID `force_cv` / mean-drift /
  TCP-excursion checks online (phase-gated on `obs["phase"]`), reusing
  `StiffnessIdHoldThresholds` from `batched_hold_quasi_static.py`.
- Wiring a stability-aware `on_step` callback into `collect_batched_quasi_static_dataset` (or a
  sibling helper) that consumes `BatchedStabilityReport` to skip recording/writing frames for
  envs flagged unstable, and annotates manifest rows accordingly.
- Any automatic action (freeze/terminate) beyond report-only, if a future use case needs it.
