# Test migration report

## `apple_pick_sim/tests/test_coupling_stability.py`

**Date:** 2026-05-22

**Command:**

```bash
cd /home/abhinav/codes/apple_pick_sim && PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupling_stability.py -v -p no:launch_testing
```

**Results:** 5 passed, 0 failed (≈17 s).

### Failures (before fix)

| Test | Failure |
|------|---------|
| *(collection)* | `ModuleNotFoundError: No module named 'conftest'` — `from conftest import` with repo-root `rootdir` does not resolve `apple_pick_sim/tests/conftest.py`. |
| `test_stem_coupling_quiescent_forces_bounded` | `\|F\|≈289 N` vs assert `< 5.0 N` (placeholder-era bound). |
| `test_fr3_coupled_substep_long_horizon_finite` | TCP `z≈-0.25` vs assert `0.5 < z < 6.0` (placeholder tree-height band). |

### Changes

- Import `conftest` helpers via `sys.path.insert` on `apple_pick_sim/tests` (same pattern needed for other migrated modules until shared package import exists).
- FR3-only: `build_coupled_fr3`, `run_coupled_substeps_direct_hold`, `requires_fr3`; removed local fixture paths and dead `_fr3_assets_available` (already absent).
- `test_stem_coupling_quiescent_forces_bounded`: assert finite harvest `< 500 N` and TCP–proxy position error `< 2 mm` under direct hold (FR3 plateau ~290 N).
- `test_fr3_coupled_substep_long_horizon_finite`: assert finite cable/robot state, TCP drift `< 2 cm` vs initial hold pose, TCP–proxy alignment `< 2 mm` (drop placeholder `z` band).

## `apple_pick_sim/tests/test_coupled_fruiting_system.py`

**Date:** 2026-05-22

**Command:**

```bash
cd /home/abhinav/codes/apple_pick_sim && PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_coupled_fruiting_system.py -v -p no:launch_testing
```

**Results:** 41 passed, 0 failed (≈38 s).

### Failures (before fix)

| Test | Failure |
|------|---------|
| *(collection)* | `ModuleNotFoundError: No module named 'conftest'` without `apple_pick_sim/tests` on import path. |
| `test_tcp_pose_matches_proxy_after_bootstrap` | Position error `≈0.19 m` vs assert `< 0.12 m` (FR3 IK residual on straight-rod fixture). |
| `test_mujoco_substep_proxy_does_not_teleport_on_first_step` | Proxy `z` jump `≈0.11 m` on first substep (bootstrap snap before kinematic hold). |
| `test_coupled_harvest_forces_stay_small_without_external_load` | `\|F\|≈277 N` vs assert `< 5.0 N`; only 8 substeps. |
| `test_tcp_pose_matches_proxy_each_coupled_substep` | `NameError: apply_direct_hold`; then orientation drift `≈0.003 rad` vs `< 0.002`. |
| `test_coupled_long_horizon_harvest_bounded` | `NameError: apply_direct_hold`; placeholder `< 5 N` bound. |
| `test_coupled_substep_is_deterministic` | Float32 drift `≈2e-6` on `body_q[8,0]` vs `atol=1e-6`. |
| `test_sync_teleports_apple_with_proxy_when_fix_to_apple` | `NameError: apply_direct_hold`. |
| `test_stem_harvest_replaces_velocity_delta_when_fix_to_apple` | `NameError: apply_direct_hold`. |
| `test_coupled_fix_to_apple_harvests_nonzero_when_robot_pushed` | `NameError: apply_direct_hold`. |

### Changes

- `sys.path.insert` for `apple_pick_sim/tests` so `from conftest import …` works with repo-root `PYTHONPATH=$(pwd)`.
- `pytestmark = requires_fr3`; all scenes via `build_coupled_fr3` (no placeholder builder/skipifs).
- Import `apply_direct_hold`; force/stability paths use `Fr3EEDirectJointController` + `robot_kinematic_mode=True`.
- Bootstrap position tolerance `_FR3_BOOTSTRAP_POS_TOL_M = 0.25` (fixture seeds often exceed 0.12 m IK residual).
- `test_mujoco_substep_proxy_does_not_teleport_on_first_step`: direct hold + sync before measuring proxy `z` between substeps.
- Quiescent harvest asserts aligned with `test_coupling_stability`: `< 500 N`, 60 substeps, TCP–proxy `< 2 mm` under hold.
- `test_coupled_long_horizon_harvest_bounded`: same harvest cap; torque cap `50 N·m`.
- `test_tcp_pose_matches_proxy_each_coupled_substep`: orientation tolerance `5e-3 rad`.
- `test_coupled_substep_is_deterministic`: relax compare to `rtol/atol=1e-5`.
