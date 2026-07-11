# Fixture `sim_build` knobs (VIC + joint overrides)

**Date:** 2026-07-10  
**Status:** Implemented  
**Branch:** `feature/batched-sysid-mmd`

## Problem

Three entrypoints duplicate (and partially disagree on) sim-build knobs:

| Script | VIC `linear_k/d`, `angular_k/d` | Joint overrides |
|--------|--------------------------------|-----------------|
| `example_batched_heterogeneous_coupled_sim.py` | 600 / 200 / 20 / 4 | `EXAMPLE_JOINT_*` |
| `example_batched_collect_sysid_data.py` | 200 / 10 / 10 / 1 | `EXAMPLE_JOINT_*` |
| `example_batched_sysid_mmd_grid.py` | 200 / 10 / 10 / 1 | `EXAMPLE_JOINT_*` |

Joint overrides already share Python constants in `batched_heterogeneous_config.py`. VIC does not. We want one stable source of truth colocated with the default ranges fixture.

## Goals

- Put VIC gains and joint kp/kd overrides in an optional top-level `sim_build` block on ranges JSON.
- Ship the block on `fruiting_system_ranges_real_world_proxy_variance.json` with **sys-ID VIC values** (200 / 10 / 10 / 1) and current `EXAMPLE_JOINT_*` values.
- Have the three examples above read knobs from that fixture.
- **Do not require** updating every other fixture; absent `sim_build` must keep existing behavior.

## Non-goals

- Moving settle / `control_hz` / `sub_dt` / env spacing into the fixture.
- Removing `EXAMPLE_JOINT_*` or Python VIC fallbacks.
- Auto-injecting `sim_build` into `BatchedHeterogeneousCoupledSimConfig` construction for all callers.
- Changing other range fixtures in this slice.

## Schema

Optional top-level key on a ranges fixture:

```json
"sim_build": {
  "vic_gains": {
    "linear_k": 200.0,
    "linear_d": 10.0,
    "angular_k": 10.0,
    "angular_d": 1.0
  },
  "joint_angular_kd_overrides": {
    "support": 0.3,
    "primary_spur": 0.3,
    "spur_stem": 0.3,
    "stem_apple": 0.3
  },
  "joint_linear_kd_overrides": {
    "support": 0.3,
    "primary_spur": 0.3,
    "spur_stem": 0.3,
    "stem_apple": 0.3
  },
  "joint_angular_kp_overrides": { "support": 2000.0 },
  "joint_linear_kp_overrides": { "support": 2000.0 }
}
```

### Validation rules (when `sim_build` is present)

- `sim_build` must be a JSON object.
- `vic_gains` required; all four keys required; each a finite float ≥ 0.
- Each `joint_*_overrides` key optional; if present must be an object mapping **known joint roles** (`support`, `primary_spur`, `spur_stem`, `stem_apple`) to finite floats ≥ 0.
- Unknown top-level keys under `sim_build` or unknown joint-role keys → `ValueError`.
- Missing `sim_build` → no error (optional).

## API

Mirror `parse_fixture_args`:

1. Extend `_validate_ranges` / `load_ranges` to validate `sim_build` when present.
2. Add frozen dataclass `SimBuildConfig` with:
   - `vic_gains: ImpedanceGains` (or four floats + a small converter)
   - `joint_angular_kd_overrides: dict[str, float]`
   - `joint_linear_kd_overrides: dict[str, float]`
   - `joint_angular_kp_overrides: dict[str, float]`
   - `joint_linear_kp_overrides: dict[str, float]`
3. Add `parse_sim_build(ranges: dict) -> SimBuildConfig | None`:
   - returns `None` when `sim_build` absent;
   - otherwise returns a validated config (empty override dicts when those keys omitted).

Preferred home: `apple_pick_sim/fruiting_system/params.py` next to `FixtureArgs` / `parse_fixture_args`, re-exported from `fruiting_system`.

## Call-site behavior

### Shared pattern

```text
ranges = load_ranges(path)
sim_build = parse_sim_build(ranges)
vic = sim_build.vic_gains if sim_build else FALLBACK_VIC
joint_* = sim_build.joint_* if sim_build else EXAMPLE_JOINT_*
```

### Examples

- **collect + mmd grid:** replace module-level hard-coded `VIC_GAINS` / `JOINT_*` assignments with values from `parse_sim_build` on the default (or CLI) ranges path. Keep module constants only as fallbacks when `sim_build` is missing.
- **heterogeneous coupled sim:** default `--vic-*` from fixture when present, else fallback; CLI still overrides. Align fallback VIC to sys-ID (200 / 10 / 10 / 1) so code defaults match the fixture even if someone strips `sim_build`.

### Fallback constants

Keep `EXAMPLE_JOINT_*` in `batched_heterogeneous_config.py` as the no-`sim_build` fallback. Update `_VIC_DEFAULT_*` there (and the heterogeneous example) to 200 / 10 / 10 / 1 for consistency with the chosen single truth.

## Fixture change

Only update:

`apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json`

Add the `sim_build` block with the schema values above. Leave all other fixtures unchanged.

## Tests

- `parse_sim_build` / `load_ranges`: absent → `None` / still loads; present good → expected values; bad types / negative / unknown keys → `ValueError`.
- Variance fixture loads and exposes the expected VIC + override values.
- Lightweight example/config wiring tests (existing CLI config builders) assert joint overrides and VIC come from fixture when using the variance path.

## Docs

Short note in `docs/damping-tuning.md` (or `docs/material-parameter-sampling.md`) that example VIC + joint overrides may live under optional `sim_build` on ranges JSON; variance fixture is the canonical copy for batched examples.

## Success criteria

- Editing VIC or joint overrides in the variance fixture changes all three examples without editing their module constants (when they use that fixture).
- Fixtures without `sim_build` continue to load and run with prior Python fallbacks.
- Heterogeneous demo VIC defaults match sys-ID (200 / 10 / 10 / 1) unless CLI overrides.
`)