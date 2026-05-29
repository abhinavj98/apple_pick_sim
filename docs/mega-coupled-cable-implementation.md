# Mega coupled cable model (plant-side)

## Behavior summary

`MegaCoupledCableScene` packs **N** independent fruiting trees (rod chain + apple + gripper proxy) into one VBD `Model` for batched finite-difference / FIM work (`fd_ghost`-style plant batching). Each instance:

- Uses its own `FruitingSystemParams` (typically same geometry, **different branch `bend_stiffness`** per column).
- Is placed at `base_pos + i * instance_spacing` (default 1.5 m along +Y).
- Has **no collisions** with other instances (filter pairs between chain body groups).
- Has its own articulation block (chain joints + separate FREE proxy articulation), matching the single-instance coupled layout.

**Solvers:** one shared `SolverVBD` for all plant instances; **no** MuJoCo arm in this module.

Build entry points:

- `MegaCoupledCableScene.build(params_list, ...)`
- `generate_mega_coupled_cable_scene(ranges, seed, stiffness_epsilon=ε)` → `fd_stiffness_param_columns(nominal, ε)`

When `len(params_list) > 1`, `build` calls `sync_all_instances_from_nominal` so every column starts from the same dynamical state as instance `0` (required for FD).

Stiffness helpers live in `apple_pick_sim/fruiting_system/params.py`: `copy_fruiting_params`, `perturb_rod_stiffness`, `fd_stiffness_param_columns`.

Plant-only build lives in `fruiting_system/mega.py`. **Coupled fd_ghost** (one FR3, offset ghost-sync to all proxies, nominal-only harvest) is in `coupled_fruiting/` — see below.

## Mega coupled FR3 (`coupled_fruiting/mega_scene.py`)

**Not on the M2 critical path** (subprocess FID remains default); interactive prototype for batched FD columns.

One `SolverMuJoCo` FR3 + one mega `SolverVBD` model. Per `coupled_substep`:

1. Kinematic FR3 teleop (`robot_kinematic_mode=True` by default).
2. `launch_mirror_robot_to_proxy_offset` — `p_proxy_k = p_tcp + (base_pos_k - base_pos_0)` for every column; same TCP orientation.
3. Shared VBD substep on all instances.
4. `harvest_stem_tension_for_tcp` on the nominal column when an apple exists (`stem_apple_joint_index`); else velocity-delta on `harvest_registry` (no apple).

| Symbol | Role |
|--------|------|
| `build_mega_coupled_fruiting_fr3` | Build mega plant + FR3 + ghost/harvest registries |
| `MegaCoupledFruitingScene` | `coupled_substep`, `apply_fr3_ee_teleop_direct` |
| `ProxyBodyRegistry.from_repeated_robot` | 1 TCP → N proxies |
| `launch_mirror_robot_to_proxy_offset` | Offset-aware ghost sync |

Keyboard demo: `apple_pick_sim/examples/example_mega_coupled_keyboard.py` (multi-column: sync → `--fd-substeps` × `coupled_substep` → Jacobian/FIM → reset; `--fd-print-interval` / `--fd-fim`; `--fix-to-apple` / `--fix-to-apple-warmup-substeps` for settle-then-weld via `seed_mega_fix_to_apple_from_settled`).

## FD step driver (`mega_fd.py`)

Per-step protocol for batched plant FD (one transition sensitivity, no cross-step drift):

1. `apply_control(mega, dt)` — pluggable hook; default no-op (gravity only).
2. `mega_vbd_substep` — one `solver.step` on the shared model.
3. Extract features `y_k` per instance; form `J[:, i] = (y_{i+1} - y_0) / ε`.
4. `reset_perturbed_instances_to_nominal` — copy instance `0` dynamics onto columns `1..N-1`.

**Offset-aware reset:** world poses are **not** copied verbatim (instances are spatially separated). Paired bodies get `p_dst = p_src + (base_pos_dst - base_pos_src)` with the same quaternion.

**Full internal state** (not just FIM features): both `state_0` / `state_1` (`body_q`, `body_qd`, `body_f`, `joint_q`, `joint_qd`) and shared `SolverVBD` buffers for that instance—`body_q_prev`, `body_inertia_q`, AVBD forces/hessians, per-body contact CSR slots, joint friction (`joint_sigma_prev`, …), penalty stiffness blocks. Global rigid-contact warm-start history is invalidated after reset so the next substep does not reuse a stale manifold from another column.

| Symbol | Role |
|--------|------|
| `copy_mega_instance_state(mega, src, dst)` | Offset copy + `body_q_prev` align |
| `sync_all_instances_from_nominal` / `reset_perturbed_instances_to_nominal` | FD init / post-step reset |
| `mega_fd_step(...)` | Full step; returns `MegaFdStepResult` (`features`, `jacobian`, optional `fim_step`) |
| `default_mega_fd_features` | Apple + proxy positions in instance-local frame; stem wrench when welded (`state_1.body_q` pre-step) |
| `extract_mega_fd_jacobian` | Jacobian/FIM from current state (coupled keyboard path) |
| `copy_coupled_scene_from_nominal` | Sequential gold oracle (standalone `CoupledCableScene`) |

Optional per-step FIM: pass `sigma_inv` to `mega_fd_step`; computes `J.T @ sigma_inv @ J`.

## Code map

| Module | Symbols |
|--------|---------|
| `apple_pick_sim/fruiting_system/mega.py` | `MegaCoupledCableScene`, `FruitingInstanceLayout`, `generate_mega_coupled_cable_scene` |
| `apple_pick_sim/fruiting_system/mega_fd.py` | `mega_fd_step`, `copy_mega_instance_state`, `mega_vbd_substep`, `MegaFdStepResult` |
| `apple_pick_sim/fruiting_system/build.py` | `_build_fruiting_chain_into_builder`, collision filter helpers |
| `apple_pick_sim/fruiting_system/params.py` | FD column params, `params_fingerprint` (includes spur/stem bend) |
| `apple_pick_sim/coupled_fruiting/mega_scene.py` | `MegaCoupledFruitingScene`, `mega_ghost_position_offsets_wp` |
| `apple_pick_sim/coupled_fruiting/builders.py` | `build_mega_coupled_fruiting_fr3` |
| `apple_pick_sim/coupled_fruiting/proxy_coupling.py` | `mirror_robot_tcp_to_proxy_offset_kernel`, `from_repeated_robot` |

## Tests

- `apple_pick_sim/tests/test_mega_coupled_cable_scene.py` — FD column count, body-count scaling, distinct fingerprints, spatial offset, short VBD rollout finiteness.
- `apple_pick_sim/tests/test_mega_fd.py` — offset copy, per-step reset, multi-step drift, Jacobian vs sequential gold (per column and full mega), FIM (`test_fim_equals_jt_sigma_inv_j`, `test_fim_scales_linearly_with_sigma_inv`, identity / no-`sigma_inv` smoke).
- `apple_pick_sim/tests/test_mega_coupled_fruiting.py` — `test_build_two_instance_mega_coupled_finite`, `test_ghost_mirror_offsets`, `test_nominal_harvest_only`, `test_mega_instance0_parity_vs_1x1`.
- `apple_pick_sim/tests/test_proxy_coupling.py` — `test_mirror_robot_tcp_to_proxy_offset_kernel`, `test_proxy_registry_from_repeated_robot_pairs_order`.
- `apple_pick_sim/tests/test_mega_fd_kinematics.py` — welded zero-g stem wrench vs gather, lateral restoring force, FD Jacobian sign/column checks.
- `apple_pick_sim/tests/test_coupled_fruiting_system.py` — `test_coupled_fr3_tcp_fz_matches_apple_weight_*` (full FR3 coupled TCP ≈ m·g), `test_coupled_stem_vertical_force_matches_apple_weight` (VBD-only m·g), `test_free_proxy_lateral_stem_restoring_force`.
- `apple_pick_sim/tests/test_mega_coupled_fruiting.py` — `test_mega_coupled_tcp_fz_matches_apple_weight_at_hold`.
- `apple_pick_sim/tests/test_fruiting_system.py` — `test_perturb_rod_stiffness_*`, `test_fd_stiffness_param_columns_nominal_first_and_epsilon_guard` (FD column param helpers).

Run from repo root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_mega_coupled_cable_scene.py \
  ../apple_pick_sim/tests/test_mega_fd.py \
  ../apple_pick_sim/tests/test_mega_coupled_fruiting.py \
  ../apple_pick_sim/tests/test_proxy_coupling.py::test_mirror_robot_tcp_to_proxy_offset_kernel \
  ../apple_pick_sim/tests/test_proxy_coupling.py::test_proxy_registry_from_repeated_robot_pairs_order \
  -q -p no:launch_testing

PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/examples/example_mega_coupled_keyboard.py --viewer null --num-frames 1
```
