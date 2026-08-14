# Batched heterogeneous build API (V.3.1 step A)

**Status:** Implemented
**Canonical living doc:** `docs/handbook-coupled-simulation.md`
**Scope:** Build-only layer; full `BatchedHeterogeneousCoupledSim` (step B) deferred  
**Config:** `apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py` (done; `ControllerConfig.action_dim` / `validate_actions` — [runtime spec](2026-07-03-batched-heterogeneous-coupled-sim-design.md#config-extensions-batched_heterogeneous_configpy))

## Goal

Extract scene construction from `example_batched_heterogeneous_coupled_fruiting.py` into a library function that:

- Takes frozen config + pre-sampled params + ranges
- Runs settle-then-weld when `robot.fix_to_apple=True`
- Returns a physics-ready scene without argparse, printing, or mandatory viewer coupling

Sampling stays a separate caller concern. Gym / large-scale training never pays diagnostic or viewer costs.

## Boundaries

| Layer | Responsibility |
| ----- | -------------- |
| **Sampling** (caller) | `load_ranges` + `sample_heterogeneous_params_list` (or inject `per_env_params` via config for tests) |
| **Build** (library) | Scene assembly, settle-then-weld, post-build kd overrides |
| **Runtime sim** (later, step B) | `step()`, `gather_obs()`, controllers — not in this slice |
| **Example** (V.3.2) | argparse → config, optional `viewer` to build, print diagnostics from `BuildResult` |

## Public API

### Module

`apple_pick_sim/coupled_fruiting/batched_heterogeneous_build.py`

### Function

```python
def build_batched_heterogeneous_scene(
    config: BatchedHeterogeneousCoupledSimConfig,
    per_env_params: Sequence[FruitingSystemParams],
    ranges: dict,
    *,
    viewer: Any | None = None,
) -> BatchedHeterogeneousBuildResult:
    ...
```

- `config.validate()` called at entry.
- `ranges` passed explicitly so sampling and build share the same topology fixture (no implicit path reload).
- `viewer=None` → headless settle (training, tests, CI).
- `viewer` provided → render between settle substeps (example GL sag animation). No viewer type imported at module level beyond `Any`.

### Build internals

1. Resolve device; call `_resolve_robot_kind` (FR3 → placeholder fallback with `UserWarning` when assets missing); pick `build_heterogeneous_coupled_fruiting_*`.
2. Map config sub-fields → existing builder kwargs (scene, robot, mujoco, fruiting_system).
3. **Settle-then-weld** when `robot.fix_to_apple=True` and not `vbd_only`:
   - Build free-proxy scene (`fix_to_apple=False`)
   - Run VBD settle (headless or with viewer)
   - Build welded scene (`fix_to_apple=True`, `skip_ik_bootstrap=True`, `defer_template_robot_bootstrap=True`)
   - `seed_fix_to_apple_from_settled(..., per_env_ik=True)`
4. **Otherwise:** single build + optional settle substeps from `scene.settle_substeps`.
5. Apply `set_fruiting_joint_angular_kd_batched` from `fruiting_system.joint_angular_kd_overrides`.

Existing low-level builders in `builders.py` remain unchanged; this module orchestrates them.


## Robot kind resolution and warnings

`build_batched_heterogeneous_scene` resolves robot kind via `_resolve_robot_kind(config.robot.kind)` before picking `build_heterogeneous_coupled_fruiting_fr3` vs `build_heterogeneous_coupled_fruiting_placeholder`.

| Input `config.robot.kind` | `fr3_assets_available()` | Resolved kind | Warning |
| ------------------------- | ------------------------ | ------------- | ------- |
| `"fr3"` | yes | `"fr3"` | none |
| `"fr3"` | no | `"placeholder"` | `UserWarning`: `"FR3 assets not found; building with placeholder TCP."` |
| `"placeholder"` | (ignored) | `"placeholder"` | none at build layer (step B CoupledSim emits placeholder hot-path warning) |

Implementation reference: [`batched_heterogeneous_build._resolve_robot_kind`](../../apple_pick_sim/coupled_fruiting/batched_heterogeneous_build.py).

**Policy:** warnings are **non-fatal** — build always returns `BatchedHeterogeneousBuildResult` on success. Callers (including `BatchedHeterogeneousCoupledSim`) must **not** filter or suppress these warnings. Step B emits an additional placeholder hot-path `UserWarning` whenever the resolved kind is `"placeholder"` (see [runtime spec §5](2026-07-03-batched-heterogeneous-coupled-sim-design.md#5-robot-kind-resolution-and-warnings)).

## `BatchedHeterogeneousBuildResult`

```python
@dataclasses.dataclass(frozen=True)
class BatchedHeterogeneousBuildResult:
    scene: CoupledFruitingScene
    per_env_params: tuple[FruitingSystemParams, ...]
    joint_angular_kd_overrides: dict[str, float]

    # Populated only when config.settle_diagnostics is not None
    settle_stability_reports: tuple[SettleStabilityReport, ...] | None = None
    settle_ke_decay_reports: tuple[SettleKeDecayReport, ...] | None = None
    ik_envelope_results: tuple[tuple[float, float, bool], ...] | None = None
```

- Diagnostics fields are `None` when `settle_diagnostics` is off (default for `gym_defaults()` / training).
- Build does **not** print reports; callers (example, tests) print if needed.

## Out of scope (this slice)

- `BatchedHeterogeneousCoupledSim` class (`step`, `gather_obs`, controllers)
- argparse / `from_args()` config factory
- Refactoring the example (V.3.2)
- Gym env migration (V.3.3+)
- Deferred settle as a separate code path (replaced by optional `viewer` on build)

## Testing

New module: `apple_pick_sim/tests/test_batched_heterogeneous_build.py`

| Test | Intent |
| ---- | ------ |
| `test_build_minimal_smoke` | `test_minimal()` config + injected params → `layout.num_envs` match, apples above ground |
| `test_fr3_missing_assets_fallback_warns` | Unavailable FR3 assets + `kind="fr3"` → `pytest.warns(UserWarning, match="placeholder TCP")`; resolved placeholder scene builds |
| `test_settle_then_weld_fix_to_apple` | `fix_to_apple=True` → welded scene, `per_world_proxy_offsets` set |
| `test_diagnostics_gated` | `settle_diagnostics=None` → report fields `None`; enabled → populated |
| `test_kd_overrides_applied` | `joint_angular_kd_overrides` on result matches config |

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_heterogeneous_build.py -q`

## Follow-up (step B)

Wrap `BuildResult` in `BatchedHeterogeneousCoupledSim.build(config)` with `step()`, `gather_obs()`, and controller attachment per `docs/ROADMAP.md` V.3.1.
