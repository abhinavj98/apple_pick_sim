# VIC pose-action controller (`vic_pose`)

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living doc:** | `docs/handbook-variable-impedance.md` |
| **Date** | 2026-08-10 |
| **Depends on** | Batched VIC joint-torque path (`Fr3BatchedEEImpedanceController`, `vic_joint_torques_batched`) |
| **Related** | `docs/handbook-variable-impedance.md`, `robot_replay/example_replay_real_batched.py`, real-robot `compute_pose_task_wrench` |

## Purpose

Match real-robot **pose PD** control in sim: actions are TCP **target pose + per-axis `Kp`/`Kd`**, not EE twists. First consumer is **`robot_replay/example_replay_real_batched.py` only**; gym collect / MMD / other examples stay on twist `vic` until a later integration.

**Why this blocks real replay:** compiled real parquets (`dump.action_semantics`) store a pose-control **wrench** `[Fx…Tz]` in `action`, and also log `target_pose_4x4` + `dump.controller_gains`. Bit-2 export previously copied `action` into batched datasets and twist `mode=vic` would misinterpret wrench as twist.

**Export packing:**
`export_real_episode_to_batched_dataset` packs 19D `vic_pose` actions
`[pos(3), quat_wxyz(4), Kp(6), Kd(6)]` from `target_pose_4x4` +
`dump.controller_gains` and stamps `action_layout=vic_pose_v1`. Replay those
datasets with `--controller-mode vic_pose`; the twist path (`mode="vic"`) rejects
them on action semantics. Legacy 6D-twist datasets can be converted with
`robot_replay/pack_vic_pose_actions.py` (which refuses already-19D sources
without `--force`).

## Decisions (locked)

| Topic | Choice |
| ----- | ------ |
| Approach | **B** — extend `Fr3BatchedEEImpedanceController` (no new controller class) |
| Mode coexistence | Keep twist `vic` (`action_dim=6`); add `vic_pose` (`action_dim=19`) |
| Action packing | All in `action`: `[pos(3), quat_wxyz(4), Kp(6), Kd(6)]` |
| Quat convention (action-level) | **wxyz** (external contract; converted to Warp-native xyzw internally — see note below) |
| Wrench law | Pose-only: `w = Kp ⊙ e + Kd ⊙ (0 − v)` (`v_des = 0`) |
| Force hybrid | **Out of scope** this slice |
| First caller | `example_replay_real_batched.py` only |

## Action contract

```text
action[0:3]   = target_position  (world, m)
action[3:7]   = target_quaternion (w, x, y, z), unit-normalized on ingest
action[7:13]  = Kp  for [Fx, Fy, Fz, Tx, Ty, Tz]
action[13:19] = Kd  for [Fx, Fy, Fz, Tx, Ty, Tz]
```

- Near-zero quat → identity `(1, 0, 0, 0)` in **wxyz**.
- **Internal storage note (verified against code):** the action-level quat is **wxyz** (external contract only). Newton's `body_q` and `Fr3BatchedEEImpedanceController._target_rot_wp` store quats **xyzw** (Warp's native `wp.quat(x, y, z, w)` — confirmed via `batched_template_ik.py`'s `_vec4_to_quat`/`wp.transform_get_rotation` usage). `unpack_pose_action` must convert **wxyz → xyzw** before writing `_target_rot_wp`; nothing else in the action contract changes.
- Gains are **anisotropic** per spatial axis; fixture isotropic `ImpedanceGains` are **not** used when per-env gain buffers are staged.
- Twist speed clipping (`linear_speed` / `angular_speed`) applies only to `mode="vic"`. For `vic_pose`: normalize quat; do not clip pose or gains.

## Control law (parity with real robot)

Matches factory-style `compute_pose_task_wrench`:

```text
pos_error, aa_error = pose_error(ee_pos, ee_quat, target_pos, target_quat)
wrench[:3] = Kp[:3] * pos_error + Kd[:3] * (0 - ee_linvel)
wrench[3:] = Kp[3:] * aa_error + Kd[3:] * (0 - ee_angvel)
```

Orientation error remains the existing world-frame axis-angle from `q_des * q_act^{-1}` (`vic_wrench._orientation_error_axis_angle`).

## Architecture

```text
env.step(action_19)
  │
  ├─ mode == "vic"      → upload twists → advance_targets_batch → isotropic gains
  └─ mode == "vic_pose" → unpack pose+gains → set targets directly → zero v_des
                              → stage vic_target_* + vic_kp/kd buffers
  │
  └─ substeps: launch_compute_vic_wrenches_batched
         if gain buffers present → anisotropic kernel (v_des=0)
         else → existing isotropic kernel
```

### Method B surface

On `Fr3BatchedEEImpedanceController`:

- Keep `run_coupled_teleop_frame_from_actions` for twist integrate.
- Add `run_coupled_teleop_frame_from_pose_actions(...)` that:
  1. Uploads `pos` / `quat_wxyz` into `_target_pos_wp` / `_target_rot_wp` (no `advance_targets_batch`).
  2. Zeros `_lin_vels_wp` / `_ang_vels_wp`.
  3. Uploads `Kp`/`Kd` into new device buffers.
- Extend `stage_targets_to_scene` to wire gain buffers onto the scene.

### Config / sim step

- `ControllerMode = Literal["direct", "ee", "vic", "vic_pose"]`.
- `vic_pose` requires `robot.step_mode="coupled"` (same as `vic`).
- Prefer `action_dim=19` when `mode="vic_pose"` (validate in `BatchedHeterogeneousCoupledSimConfig.__post_init__` or document as required pair).
- `BatchedHeterogeneousCoupledSim._clip_actions`: skip twist clip when `mode="vic_pose"`.
- `_run_fr3_teleop_from_actions`: branch on mode to pose vs twist method.

### Wrench kernel

- Add batched path reading per-env `wp.vec3` or `(N,6)` `Kp`/`Kd` (or two `wp.vec3` pairs for linear/angular).
- Gate: if `scene.vic_kp_wp` (name TBD) is set, use anisotropic law; else keep today’s isotropic scalars from `scene.vic_gains` / `ImpedanceGains`.

## Real replay consumer (this slice)

`robot_replay/example_replay_real_batched.py`:

- Set `controller.mode="vic_pose"`, `action_dim=19`.
- Drive `replay_batched_sysid_structure` (or equivalent) with **19D** actions from the dataset.

**Dataset note:** Today’s exporter / `replay_batched_sysid_structure` assume `action.shape[1] == 6`. This slice must:

1. Allow `(n_frames, 19)` when the sim controller is `vic_pose`, **or**
2. Provide a thin packer in the example that builds 19D actions from named columns (`target_pos`, `target_quat`, `Kp`, `Kd`) until new parquets ship packed `action`.

Prefer: pack into `action` at convert or at replay entry so `env.step` still sees one tensor. Full gym/MMD migration is **later**.

## Out of scope

- `compute_force_task_wrench` / hybrid force control
- Migrating gym collect, MMD grid, CMA, or default `gym_defaults` to `vic_pose`
- Replacing twist teleop / keyboard VIC
- Changing `fill_actions_from_tcp_velocity` (twist fill remains for old zero-action logs)
- Single-env non-batched VIC anisotropic gains (unless needed for tests on CPU with `num_envs=1`)

## Success criteria

1. Unit tests: 19D unpack → target buffers + gain buffers; wrench equals `Kp⊙e − Kd⊙v` for a synthetic TCP state.
2. Config accepts `vic_pose` + `action_dim=19`; twist `vic` regressions still green.
3. `example_replay_real_batched.py` constructs with `vic_pose` and can step 19D actions (synthetic or new parquet).
4. No change to default twist collect paths unless they explicitly opt into `vic_pose`.

## Verification (canonical)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py \
  apple_pick_sim/tests/test_controller_config_actions.py \
  apple_pick_sim/tests/test_batched_heterogeneous_config.py -q

# Optional smoke (when 19D episode available):
uv run python robot_replay/example_replay_real_batched.py \
  --dataset <batched_sysid_v1_with_19d_action> --viewer null --max-frames 5 \
  --settle-substeps 50 --post-grasp-settle-substeps 10
```

## Spec self-review

- No force hybrid placeholders left in scope.
- Action layout, quat order, and first-caller constraint are explicit.
- Hardcoded `(n, 6)` in replay helpers is called out as a required touch.
