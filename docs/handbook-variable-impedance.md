# Variable impedance control handbook

This is the canonical living reference for FR3 variable-impedance control (VIC),
including the 6D twist controller (`vic`) and 19D absolute-pose controller
(`vic_pose`). Sequencing and milestone status belong in `docs/ROADMAP.md`.

## Document status

| Field | Value |
| --- | --- |
| Last reviewed | 2026-08-14 |
| Code owners | `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance*.py`; `apple_pick_sim/coupled_fruiting/vic_wrench.py`; `apple_pick_sim/coupled_fruiting/vic_joint_torques*.py` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H1 `docs/handbook-coupled-simulation.md`; H4 `docs/handbook-real-replay.md`; H5 `docs/handbook-youngs-cma.md` |
| Archive specs | **Implemented:** `docs/superpowers/specs/2026-08-10-vic-pose-action-controller-design.md` → H2; **Superseded by undocumented retunes:** `docs/superpowers/specs/2026-07-17-wrench-cap-retune-design.md` → H2; **Historical:** `docs/specs/2026-07-10-vic-wrench-caps-design.md` → H2 |

Related boundaries:

- See H1 for settle → weld and where VIC runs in the coupled substep.
- See H4 for the real-data conversion and replay pipeline that defaults to
  `vic_pose`.
- See H5 for grid/CMA controller selection through `--controller-mode`.

## 1. Total TCP wrench and joint-torque control

Post-grasp runs use a dynamic FR3 (`robot_kinematic_mode=False`) inside the
staggered MuJoCo + VBD simulation. The plant reaction harvested at the end of
substep \(n\) is applied to the arm at substep \(n+1\). The controller task
wrench is computed from the current TCP state and target every MuJoCo substep.

`CoupledFruitingScene._mujoco_robot_substep_prefix` owns the distinction between
the two actuation paths:

| Path | Plant/proxy load | Controller effort |
| --- | --- | --- |
| Joint-torque VIC (default for `vic`) | Lagged `proxy_forces` copied through `coupling_forces_cache` to TCP `body_f` | Task wrench mapped to arm torques in `joint_f` |
| Legacy spatial-wrench VIC | Lagged plant wrench plus fresh VIC wrench accumulated in `coupling_forces_cache` and applied to TCP `body_f` | Included in that TCP body wrench |
| Kinematic/direct control | Coupling cache is zeroed; MuJoCo does not integrate the stem load | Direct joint target update |

In joint-torque mode, `vic_joint_torques.py` and
`vic_joint_torques_batched.py` implement

\[
\tau_{\mathrm{task}} = J^\mathsf{T}\Lambda w,\qquad
\Lambda=(J M^{-1}J^\mathsf{T}+\lambda I)^{-1},
\]

plus a dynamically consistent null-space term. Joint 7's null-space target is
forced to 0 rad. Joint position/velocity gains are zeroed by
`fr3_robot.configure_vic_joint_torques_arm`, so MuJoCo position actuators do not
compete with VIC. The coupling load remains an external TCP body wrench.

The plant-only invariant matters: harvest functions write only the transferred
plant wrench to `proxy_forces`; they must not feed the controller's applied
wrench back into the next lag step. The full settle, weld, mirror, VBD, and
harvest ordering is documented in H1.

### Kinematic and dynamic gym environments

`ApplePickCoupledEnv` uses direct, kinematic joint teleop. `ApplePickVicEnv` and
its `ApplePickSysIdEnv` / `ApplePickReplayEnv` subclasses use the dynamic FR3
and joint-torque VIC by default. Kinematic mode is useful for pose-exact smoke
tests, but it does not model compliance under harvested stem load.

## 2. Twist mode (`vic`)

`ControllerConfig(mode="vic", action_dim=6)` interprets each action as a world
frame TCP twist:

```text
[linear_xyz(3), angular_xyz(3)]
```

`Fr3BatchedEEImpedanceController.run_coupled_teleop_frame_from_actions`
uploads per-environment twists and integrates each environment's target pose.
The controller stages device-resident target position, rotation, linear
velocity, and angular velocity buffers:

- `vic_target_positions_wp`
- `vic_target_rotations_wp`
- `vic_target_linear_vels_wp`
- `vic_target_angular_vels_wp`

The batched wrench kernel indexes the desired velocity for each world; it does
not broadcast world 0's command. Action clipping uses the controller's
`linear_speed` and `angular_speed`.

Twist VIC uses isotropic `ImpedanceGains`:

\[
f=K_{\mathrm{lin}}(p_d-p)+D_{\mathrm{lin}}(v_d-v),
\qquad
\tau=K_{\mathrm{ang}}e_R+D_{\mathrm{ang}}(\omega_d-\omega),
\]

where \(e_R\) is the world-frame axis-angle error from
\(q_d q^{-1}\). Batched collect and grid builders read gain values from the
ranges fixture's `sim_build.vic_gains` when present; their local constants are
fallbacks. There is therefore no single gain tuple that is correct for every
entry point.

## 3. Pose mode (`vic_pose`)

`ControllerConfig(mode="vic_pose", action_dim=19)` is the batched absolute-pose
controller used to reproduce real pose-PD commands. Configuration validation
requires coupled stepping and exactly 19 action values:

```text
action[0:3]   = target_position in world coordinates [m]
action[3:7]   = target_quaternion [w, x, y, z]
action[7:13]  = Kp [Fx, Fy, Fz, Tx, Ty, Tz]
action[13:19] = Kd [Fx, Fy, Fz, Tx, Ty, Tz]
```

`Fr3BatchedEEImpedanceController.unpack_pose_action` writes the target pose
directly. Unlike `vic`, it performs no velocity integration and
`BatchedHeterogeneousCoupledSim._clip_actions` does not apply twist speed
clipping to the pose or gains.

The external quaternion contract is `wxyz`. Ingest normalizes it, maps a
near-zero quaternion to identity, and reorders it to Warp-native `xyzw` before
writing `_target_rot_wp`. Desired linear and angular velocities are both zero.

### Anisotropic pose wrench

The 12 gain values are split into four per-environment device buffers:

| Scene buffer | Action values |
| --- | --- |
| `vic_kp_lin_wp` | `Kp[0:3]` |
| `vic_kp_ang_wp` | `Kp[3:6]` |
| `vic_kd_lin_wp` | `Kd[0:3]` |
| `vic_kd_ang_wp` | `Kd[3:6]` |

When all four buffers are staged,
`vic_joint_torques_batched.launch_compute_vic_wrenches_batched` selects
`compute_vic_spatial_wrench_aniso`:

\[
f=K_p^{\mathrm{lin}}\odot(p_d-p)-K_d^{\mathrm{lin}}\odot v,
\qquad
\tau=K_p^{\mathrm{ang}}\odot e_R-K_d^{\mathrm{ang}}\odot\omega.
\]

The multiplication is component-wise and all quantities are world-frame at
the TCP COM. These per-action gains take precedence over the fixture's
isotropic `scene.vic_gains`. If any anisotropic buffer is absent, the launcher
uses the isotropic twist kernel instead.

### Soft-disable freeze semantics

`EnvDisableController` treats disabled rows according to action semantics:

- 6D twist: replace the row with zeros.
- 19D `vic_pose`: freeze and reissue the last commanded pose-and-gains row.

Zeroing a pose row would command the world origin with zero gains, leaving a
limp arm; holding the previous row preserves the last target and impedance.
The disabled mask is sticky, and disabled environments are excluded from
recording. `apple_pick_gym/tests/test_env_disable_controller.py` verifies the
initial freeze, persistence across later steps, and unchanged 6D behavior.

## 4. Stem wrench caps

The current batched coupled-scene defaults are code values:

```text
DEFAULT_STEM_FORCE_CAP_N = 40.0
DEFAULT_STEM_TORQUE_CAP_NM = 10.0
```

They are defined in `apple_pick_sim/coupled_fruiting/scene.py` and passed by
`_harvest_coupling_wrenches` to the single- or batched stem-harvest function.
They cap the transferred stem reaction at **40 N force** and **10 N·m torque**.
They do not cap the VIC task wrench or the final arm joint torques.

The 100 N / 40 N·m values proposed by
`2026-07-17-wrench-cap-retune-design.md` are not current behavior; that spec is
**Superseded by undocumented retunes**. The older
`docs/specs/2026-07-10-vic-wrench-caps-design.md` is historical.

## 5. Who uses which mode

| Consumer | Controller contract |
| --- | --- |
| Batched gym collect | `vic`, 6D twist |
| Legacy/default MMD and simulation grid paths | `vic`, 6D twist |
| Default CMA/synthetic-data path | `vic`, 6D twist |
| Converted real replay (`vic_pose_v1`) | `vic_pose`, 19D pose + gains |
| Real-data Young's grid | Infers `vic_pose` from dataset metadata; may opt in with `--controller-mode vic_pose` |
| Controller-selecting grid/CMA CLIs | `--controller-mode` is the explicit override; see H5 |

Converted real datasets identify pose actions with
`action_layout=vic_pose_v1` and `action_dim=19`. A 6D real pose-controller
wrench is not a twist and must not be replayed as `vic`; H4 owns conversion and
semantic checks.

### Finite-difference implications

Kinematic teleop gives the same end-effector trajectory for the same action and
can use an `fd_ghost` layout for early smoke tests. Dynamic VIC trajectories
depend on each plant parameterization through compliance, so parameter
finite-difference work must use independently coupled workers such as
`fd_mega_same_u` or replay the same action sequence with `fd_replay`. See the
ROADMAP's FD-mode discussion for current sequencing.

## 6. Code map

| Responsibility | Module / symbol |
| --- | --- |
| Isotropic impedance law and gains | `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance.py` — `Fr3EEImpedanceController`, `ImpedanceGains` |
| Batched target integration and pose unpack | `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance_batched.py` — `Fr3BatchedEEImpedanceController` |
| Isotropic and anisotropic wrench laws | `apple_pick_sim/coupled_fruiting/vic_wrench.py` — `compute_vic_spatial_wrench*` |
| Single-env task-wrench-to-torque path | `apple_pick_sim/coupled_fruiting/vic_joint_torques.py` |
| Batched wrench kernel and torques | `apple_pick_sim/coupled_fruiting/vic_joint_torques_batched.py` |
| Mode/action validation | `apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py` — `ControllerConfig`, `BatchedHeterogeneousCoupledSimConfig.validate` |
| Action dispatch | `apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py` — `_run_fr3_teleop_from_actions` |
| Dynamic substep and current stem caps | `apple_pick_sim/coupled_fruiting/scene.py` — `_mujoco_robot_substep_prefix`, `DEFAULT_STEM_*` |
| Sticky disable behavior | `apple_pick_gym/batched_envs/env_disable_controller.py` — `EnvDisableController` |

## 7. Tests and verification

Key regression coverage:

- `apple_pick_sim/tests/test_ee_impedance.py` — isotropic wrench law.
- `apple_pick_sim/tests/test_batched_vic.py` — per-environment target/twist
  buffers and batched wrench/torque behavior.
- `apple_pick_sim/tests/test_vic_wrench_aniso.py` — per-axis stiffness,
  angular scaling, damping, and zero-at-target behavior.
- `apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py` — the real
  `vic_pose` test module: direct targets, zero desired twists, quaternion
  fallback/order conversion, and gain staging.
- `apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py::test_vic_pose_step_moves_tcp_toward_target`
  — 19D end-to-end motion.
- `apple_pick_gym/tests/test_env_disable_controller.py` — pose-row freeze and
  6D twist zeroing.

Focused fast checks:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_vic_wrench_aniso.py \
  apple_pick_gym/tests/test_env_disable_controller.py \
  -q -p no:launch_testing
```

Pose unpack and coupled integration require the FR3 asset and PyTorch:

```bash
uv run --directory apple_pick_sim/tests --env-file ../../pytest.env python -m pytest \
  test_ee_impedance_batched_pose_actions.py \
  test_batched_heterogeneous_coupled_sim.py::test_vic_pose_step_moves_tcp_toward_target \
  -q -p no:launch_testing
```

The broader ROADMAP gates remain canonical:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/ -q -p no:launch_testing -m "not slow"
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/ -q -p no:launch_testing
```
