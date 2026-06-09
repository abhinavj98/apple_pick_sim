# Variable-impedance controller (VIC) — implementation

## Behavior summary

Post-grasp pulling uses a **dynamic** FR3 (`robot_kinematic_mode=False`). Each MuJoCo substep the scene writes the **total** TCP wrench:

\[
\mathbf{w}_{\mathrm{total}} = \mathbf{w}_{\mathrm{transferred}} + \mathbf{w}_{\mathrm{applied}}
\]

- **Transferred** (lagged): previous substep’s stem harvest in `proxy_forces[tcp]`.
- **Applied** (fresh): variable-impedance law from `Fr3EEImpedanceController` using teleop `target_tf` and live TCP pose/twist.

Frames and layout: world frame at TCP COM; spatial vector `[fx, fy, fz, tx, ty, tz]` (same as `apply_wrench.py`).

Impedance law (isotropic gains):

- \(\mathbf{F} = K_p (\mathbf{p}_{\mathrm{des}} - \mathbf{p}) + D_p (\mathbf{v}_{\mathrm{des}} - \mathbf{v})\)
- \(\boldsymbol{\tau} = K_r \mathbf{e}_R + D_r (\boldsymbol{\omega}_{\mathrm{des}} - \boldsymbol{\omega})\) where \(\mathbf{e}_R\) is axis-angle from \(\mathbf{q}_{\mathrm{des}} \otimes \mathbf{q}^{-1}\).

Harvest remains **plant-only** — `proxy_forces` is never updated with \(\mathbf{w}_{\mathrm{applied}}\).

**Joint-torque teleop (default with `--vic`):** `apply_fr3_ee_teleop` advances `target_tf` only (`run_tcp_target_teleop_frame`), zeros `joint_target_ke`/`kd`, and holds `joint_target_pos` at simulated `joint_q`. VIC maps the impedance wrench to `control.joint_f` via dynamically-consistent joint torques (`vic_joint_torques.py`); plant/proxy loads remain on TCP `body_f`. See `docs/vic-joint-torques-implementation.md`.

## Code map

| Piece | Location |
|-------|----------|
| Impedance law | `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance.py` — `Fr3EEImpedanceController`, `ImpedanceGains` |
| Wrench sum at TCP | `apple_pick_sim/coupled_fruiting/apply_wrench.py` — `_add_tcp_spatial_wrench_inplace` |
| Substep hook | `apple_pick_sim/coupled_fruiting/scene.py` — `_mujoco_robot_substep_prefix`; with `--vic`: `vic_joint_torques.apply_vic_joint_torques_to_scene`; wrench-only fallback: `vic_wrench.apply_vic_to_coupling_cache` |
| Scene fields | `CoupledFruitingScene.vic_controller`, `vic_gains`, `vic_target_tf`, `vic_target_twist`, `vic_use_joint_torques` |
| Joint-torque arm setup | `fr3_robot/setup.py` — `configure_vic_joint_torques_arm` (zeros PD, allocates J/H buffers, `joint_f`) |
| Teleop target sync | `apply_fr3_ee_teleop` — VIC path: `run_tcp_target_teleop_frame` + hold joint targets |
| Example CLI | `apple_pick_sim/examples/example_coupled_fruiting.py` — `--dynamic-arm`, `--vic`, `--vic-*-k/d` |

Design reference: `docs/variable-impedance-teleop.md`.

## Tests

| Module | Key tests |
|--------|-----------|
| `tests/test_ee_impedance.py` | Wrench law unit tests (zero at target, linear K, damping) |
| `tests/test_vic_dynamic.py` | Dynamic arm integration (`test_vic_teleop_integrates_tcp_motion`); `tests/test_vic_wrench_device.py` — GPU vs CPU parity, CUDA graph |
| `tests/test_coupled_fruiting_system.py` | `test_add_tcp_spatial_wrench_inplace_sums_at_tcp_only` |

## How to verify

From repository root (worktree or main):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_ee_impedance.py \
  ../apple_pick_sim/tests/test_vic_dynamic.py \
  ../apple_pick_sim/tests/test_vic_wrench_device.py \
  -q -p no:launch_testing -m "not slow"
```

Example smoke (headless; use test fixture + seed for reliable settle→weld bootstrap):

```bash
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/examples/example_coupled_fruiting.py \
  --robot fr3 --fix-to-apple --dynamic-arm --vic --viewer null --num-frames 2 \
  --seed 0 --json ../apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json
```

Interactive post-grasp demo:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python \
  ../apple_pick_sim/examples/example_coupled_fruiting.py \
  --robot fr3 --fix-to-apple --dynamic-arm --vic --fr3-keyboard --viewer gl
```

**Out of scope (this slice):** Gym `ApplePickCoupledEnv` remains kinematic; M2.0 ADR will extend action/obs for dynamic VIC and FD modes.
