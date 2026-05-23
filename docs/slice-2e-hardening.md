# Slice 2e — Hardening and hot-path optimization

**Last updated:** 2026-05-22

## Behavior summary

Slice 2e removes per-substep host roundtrips on the staggered coupling hot path while preserving apply → MuJoCo → sync → VBD → harvest semantics.

| Change | Module | What it does |
|--------|--------|--------------|
| Cached proxy ID arrays | `proxy_coupling.ProxyBodyRegistry.ids_wp` | `robot_ids` / `proxy_ids` `wp.array` built once per device |
| Device `body_q_prev` align | `proxy_coupling._align_body_q_prev_kernel` | Indexed copy after kinematic sync (no full `body_q` host pull) |
| Device TCP wrench apply | `coupled_fruiting._apply_tcp_spatial_wrench_kernel` | Zero all `body_f`, write lagged wrench at TCP on device |
| Viewer contacts reuse | `CoupledFruitingScene.last_vbd_contacts` | `example_coupled_fruiting.render()` skips extra `collide()` when substeps ran |

Stem-harvest (`harvest_stem_joint_wrench`) still uses a host path via `fixed_joint_wrenches_child_com_vbd` until Newton exposes device readout.

## Tests

| Test module | Key tests |
|-------------|-----------|
| `test_proxy_coupling.py` | `test_proxy_registry_ids_wp_cached_per_device`, `test_coupled_substep_reuses_cached_proxy_ids`, `test_align_proxy_body_q_prev_*` |
| `test_coupled_fruiting_system.py` | `test_apply_spatial_wrench_zeroes_non_tcp_bodies` |
| `test_coupling_stability.py` | `test_kinetic_energy_bounded_quiescent` (`slow`), `test_fr3_coupled_substep_long_horizon_finite` (`slow`) |

## How to verify

From repository root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -q -p no:launch_testing

PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/ -m slow -q -p no:launch_testing

PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/verify_coupling.py --num-substeps 600 --max-force 5 --max-torque 1

PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/diagnostics/benchmark_coupling.py --robot placeholder --warmup-substeps 30 --bench-substeps 300
```

## Baseline (reference machine)

Recorded on **Linux**, **CPU** device, fixture `fruiting_system_ranges_straight_rod_test.json`, `disable_contacts=True`, `dt = (1/60)/30` s:

| Robot | ms/substep | substeps/s | Frame @ 30 substeps |
|-------|------------|------------|---------------------|
| placeholder | ~10.2 | ~98 | ~305 ms (~3.3 fps) |
| fr3 | ~12.3 | ~81 | ~370 ms (~2.7 fps) |

Re-run the benchmark after hardware or dependency changes; update this table when reporting regressions.
