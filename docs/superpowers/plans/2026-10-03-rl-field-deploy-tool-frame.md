# RL Field Deploy: Port the v2b (Tool-Frame) Policy Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `~/codes/Continuous_Force_RL/real_robot_exps/rl_policy_env.py` run the v2b checkpoint
(`runs/vic_harvest/v2b_s0/checkpoints/ckpt_000016000`) on the FR3 with the same observation, action and
impedance law the policy was trained with.

**Architecture:** The deploy file is a torch-only re-implementation of the training math. It does not
import `apple_pick_gym`, `warp` or `skrl`. Since the deploy was written (rig commit `527eb88`, Oct 1) training
moved to the **[D17] tool-frame contract**. The policy now observes start-relative quantities in the TCP
frame, commands pose deltas in the TCP frame, and its per-axis stiffness acts along the TCP axes
(`K_world = R diag(k) R^T`). This plan ports those three changes and makes the 1 kHz controller apply
gains in the EE frame. It also fixes a `last_action` unit bug that predates D17. Parity is pinned by a JSON
fixture produced by the *training* code, so the ported math is checked against the real thing rather than
against a second copy of it.

**Tech Stack:** Python 3, torch (CPU), gymnasium, unittest (rig repo; run with system `python3`). The
fixture generator runs in the sim repo's `uv` environment (it imports `apple_pick_gym` and `skrl`).

**Spec:** this plan. The training-side sources of truth are (sim worktree `feature/rl-skrl-ppo` @ `2563a91`):
`apple_pick_gym/batched_envs/harvest_obs.py` (`_ACTOR_TOOL_FIELDS`, `tool_frame_obs`),
`apple_pick_gym/batched_envs/harvest_action.py` (`tool_delta_to_world`, `leash_target_pose`,
`pack_vic_pose_action`), `apple_pick_gym/batched_envs/apple_pick_vic_harvest_env.py` (`_gather_obs`,
`_actions_tensor`, `reset`), `apple_pick_sim/coupled_fruiting/vic_wrench.py`
(`compute_vic_spatial_wrench_aniso_tool`), `apple_pick_gym/rl/skrl_wrapper.py::step` (`to_env` before the env).

## What changed in training since the deploy was written

| # | Training now (v2b) | Deploy today | Action |
|---|---|---|---|
| 1 | Actor obs = `d_pos_tool(3) d_rot_tool(3) gravity_tool(3) twist_tool(6) ft_tool(6) target_offset_tool(6) last_action(13)` = 40-D | world layout: `tcp_pos tcp_quat tcp_velocity ft_wrist joint_q last_action step_frac` | **port** (Task 3) |
| 2 | Pose delta is in the TCP frame. It is rotated to world by the **last observed** TCP quat, then integrated | world-frame delta | **port** (Task 3) |
| 3 | Per-axis K/D act along the TCP axes (`vic_gain_frame="tool"`), recomputed from the current TCP rotation every substep | rig controller: diagonal gains along base axes | **controller change** (Task 1) |
| 4 | `last_action` in the obs is the **env-unit** action (`scaler.to_env(u)`: K in N/m, zeta, metre/rad deltas). Checkpoint normaliser: K mean 107–193, std 98–149 | feeds back the raw `[-1, 1]` action | **bug fix, predates D17** (Task 3) |
| 5 | `step_frac` is critic-only; no absolute pose or joint angles in the actor | builds `step_frac`, quat sign hint, `+ (0, 0.2, 0)` base offset | **delete** (Task 3) |
| 6 | Speed cap `linear_delta_m = 0.002` (0.12 m/s), `k_lin_max = 500`, leash 0.15 m / 0.5 rad | read from `meta.json` `action_bounds` | none (already data-driven) |
| 7 | F/T sensor model runs in the TCP frame (EMA fc 8.16 Hz = rig's alpha 0.05 at 1 kHz, bias, noise, drift) | rig F/T is already EE-frame, already EMA'd | use `snap.force_torque` directly; drop the world rotation |
| 8 | `realized_k_scale` U(0.8, 1.2), bend-pull `detach_scale`, arm DR | n/a | none (training-only DR) |
| 9 | Reset target = settled hold target. Measured on v3: TCP-to-target offset median 0.19 mm, p90 0.9 mm, 0.25° / 1.5° | reset target = current TCP | none (equivalent) |
| 10 | Actor arch unchanged (pre 256, LSTM 256, post 256/128, `mean_bound` 1.5); eval uses clamped mean | same | none |
| 11 | [D17m] v2c checkpoints: no-advance wall. After the leash, the integrated target's position is projected back onto the plane through the grasp TCP normal to the grasp grip axis (TCP +z at the start) wherever it is more than `action_bounds.max_advance_m` past it (`harvest_action.clamp_target_advance`) | absent | **port** (Task 3) when deploying a v2c checkpoint: record start pos + TCP +z at grasp; v2b checkpoints carry `max_advance_m` absent/null |
| 12 | [D17n] v2d checkpoints: commanded-force clamp. After the leash and **before** the no-advance wall, the integrated target's position is pulled toward the TCP (same direction, orientation untouched) until `‖K_lin ⊙ R_tcpᵀ (target − tcp)‖ ≤ action_bounds.max_commanded_force_n` (40 N), with the **commanded** per-axis gains in the TCP frame (`harvest_action.clamp_commanded_force`) | absent | **port** when deploying a v2d checkpoint: same order leash → force clamp → wall; v2c and earlier checkpoints carry it absent/null |

## Global Constraints

- Rig repo `~/codes/Continuous_Force_RL`, branch `rl-policy-deploy`. Do not import `apple_pick_gym`, `apple_pick_sim`,
  `warp` or `skrl` from rig runtime code. The fixture generator is the only exception, and it is a dev tool run under the sim's `uv`.
- Deploy runs on the CPU, single-threaded inference (unchanged from rig commit `f10fee1`).
- Every existing `ControlTargets(...)` call site must keep working unchanged (`apple_pullto_static.py:442`,
  `hybrid_controller.py:1124`, `hybrid_controller.py:1206`, `controller_test.py:109`). The new field defaults to base-frame gains.
- Only tool-frame checkpoints load. A checkpoint whose `actor_layout` or `config.env.action_frame` differs
  from the tool contract is refused with a message (decision D-1 below).
- No file deletion outside `/tmp` (AGENTS.md).
- Tests: `cd ~/codes/Continuous_Force_RL && python3 -m unittest real_robot_exps.<module> -v`.

## Review Focus

1. **Quaternion double cover.** `-q` for the TCP, start or target quat must give the identical obs. Expect bit-identical
   `d_rot_tool` and `target_offset_tool`. Covered by `test_obs_sign_invariant` (Task 3).
2. **The rotation used for the action is one step stale.** Training rotates the tool delta and leashes against the
   *previous* gather's TCP (`_last_tcp_pose_wxyz`), not the post-action pose. The deploy must use `self._snap` from
   before `wait_for_policy_step()`. Covered by `test_step_uses_pre_action_tcp` (Task 3).
3. **EE-frame gains at a large tilt.** With a 90° TCP rotation, the stiff tool axis must land on the expected base axis.
   This is the case a "diag in base frame" bug would get wrong while still passing at identity. Covered by
   `test_ee_gains_quarter_turn` (Task 1).
4. **A missing `gain_frame` key from an older packer** must unpack as `"base"`, not crash the 1 kHz compute process.
   Covered by `test_unpack_without_gain_frame_defaults_to_base` (Task 1).
5. **Wrong-contract checkpoint.** Loading `checkpoint_cache/vic_harvest/d8b_tanh15_s1/ckpt_000030400` (world layout)
   must raise before any robot motion. Covered by `test_refuses_world_layout_checkpoint` (Task 4).

## Decisions to confirm (defaults used by this plan)

- **D-1: drop world-frame checkpoints.** Default: refuse them. Keeping both paths doubles the obs/action code for
  checkpoints (d8b) that did not pass the v2 gate.
- **D-2: sim TCP frame = real EE frame.** The rig's `F_T_EE = Rz(-45°) · Tz(0.1034)` (Franka hand TCP). Sys-ID already
  feeds real `initial_tcp_quat` straight into the sim TCP, and the old deploy's absolute-quat OOD check passed. So the plan
  uses the EE frame as the tool frame with no fixed rotation. The OOD gate on `gravity_tool`/`ft_tool` at reset is the
  field check. A 45° mismatch would show up as a constant offset in `gravity_tool.x/y`.
- **D-3: no detach stop.** Training freezes an env after the bend-pull success (3-step streak), so the policy never saw
  post-detach states. On the arm the policy keeps running after the apple comes off, until `--steps`, truncation or the
  cage. Default: out of scope; the operator stops the run. An F/T-drop auto-hold could be a follow-up.

---

### Task 1: EE-frame gains in the 1 kHz controller

**Files:**
- Modify: `~/codes/Continuous_Force_RL/real_robot_exps/hybrid_controller.py` (`ControlTargets` ~L28, `compute_pose_task_wrench` ~L168, `compute_torques_from_targets` ~L393, `pack_control_targets` / `unpack_control_targets` ~L1163–1227)
- Test: `~/codes/Continuous_Force_RL/real_robot_exps/test_hybrid_controller_gain_frame.py` (create)

**Interfaces:**
- Produces: `ControlTargets.gain_frame: str` (`"base"` default | `"ee"`); `compute_pose_task_wrench(..., gains_in_ee_frame: bool = False)`; `rotate_wxyz(q, v) -> Tensor` (module-level helper, `q` wxyz `[...,4]`, `v` `[...,3]`).

- [ ] **Step 1: Write the failing tests**

```python
"""EE-frame (tool-frame) pose gains: K_base = R diag(k) R^T, matching the sim's
compute_vic_spatial_wrench_aniso_tool (apple_pick_sim/coupled_fruiting/vic_wrench.py)."""

import math
import unittest

import torch

from real_robot_exps.hybrid_controller import (
    ControlTargets,
    compute_pose_task_wrench,
    pack_control_targets,
    quat_from_angle_axis,
    unpack_control_targets,
)


def _rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.tolist()
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class EeFrameGainsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.kp = torch.tensor([60.0, 80.0, 500.0, 5.0, 6.0, 30.0])
        self.kd = 2.0 * torch.sqrt(self.kp)

    def _wrench(self, ee_quat, ee_frame, pos_err, linvel, angvel, target_quat=None):
        return compute_pose_task_wrench(
            torch.zeros(3), ee_quat, linvel, angvel,
            pos_err, ee_quat if target_quat is None else target_quat,
            self.kp, self.kd, gains_in_ee_frame=ee_frame,
        )

    def test_identity_rotation_matches_base_frame(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        e, v, w = torch.randn(3), torch.randn(3), torch.randn(3)
        torch.testing.assert_close(self._wrench(q, True, e, v, w), self._wrench(q, False, e, v, w))

    def test_ee_gains_quarter_turn(self):
        # TCP rotated +90 deg about base z: tool x = base y, tool y = -base x, tool z = base z.
        q = quat_from_angle_axis(torch.tensor(math.pi / 2), torch.tensor([0.0, 0.0, 1.0]))
        e = torch.tensor([0.01, 0.0, 0.0])  # base-x error lies along tool -y -> k_y = 80
        out = self._wrench(q, True, e, torch.zeros(3), torch.zeros(3))
        torch.testing.assert_close(out[:3], torch.tensor([0.8, 0.0, 0.0]), atol=1e-6, rtol=0)

    def test_random_rotation_equals_R_diag_RT(self):
        for _ in range(20):
            q = torch.randn(4)
            q = q / q.norm()
            R = _rotmat_wxyz(q)
            e, v, w = torch.randn(3) * 0.02, torch.randn(3) * 0.1, torch.randn(3) * 0.5
            out = self._wrench(q, True, e, v, w)
            k_lin = R @ torch.diag(self.kp[:3]) @ R.T
            d_lin = R @ torch.diag(self.kd[:3]) @ R.T
            d_ang = R @ torch.diag(self.kd[3:]) @ R.T
            torch.testing.assert_close(out[:3], k_lin @ e - d_lin @ v, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out[3:], -(d_ang @ w), atol=1e-5, rtol=1e-5)  # target_quat == ee_quat

    def test_pack_roundtrip_keeps_gain_frame(self):
        t = _targets(gain_frame="ee")
        self.assertEqual(unpack_control_targets(pack_control_targets(t)).gain_frame, "ee")

    def test_unpack_without_gain_frame_defaults_to_base(self):
        d = pack_control_targets(_targets(gain_frame="ee"))
        del d["gain_frame"]
        self.assertEqual(unpack_control_targets(d).gain_frame, "base")

    def test_existing_call_sites_default_to_base(self):
        self.assertEqual(_targets().gain_frame, "base")


def _targets(**extra) -> ControlTargets:
    z6, z3 = torch.zeros(6), torch.zeros(3)
    return ControlTargets(
        target_pos=z3, target_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]), target_force=z6, sel_matrix=z6,
        task_prop_gains=z6, task_deriv_gains=z6, force_kp=z6, force_di_wrench=z6, pose_ki=z6,
        pose_integral_clamp=0.0, pose_integral_reset_on_target=True, default_dof_pos=torch.zeros(7),
        kp_null=0.0, kd_null=0.0, pos_bounds=z3, goal_position=z3, ctrl_mode="force_only",
        singularity_damping=0.0, partial_inertia_decoupling=False, sep_ori=False, **extra,
    )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them, expect failure**

Run: `cd ~/codes/Continuous_Force_RL && python3 -m unittest real_robot_exps.test_hybrid_controller_gain_frame -v`
Expected: FAIL / ERROR (`unexpected keyword argument 'gains_in_ee_frame'`, `unexpected keyword argument 'gain_frame'`).

- [ ] **Step 3: Implement**

In `ControlTargets`, append the field and give it a default so every existing constructor still works:

```python
ControlTargets = namedtuple('ControlTargets', [
    ...,
    'sep_ori',  # bool — position via full Lambda (zeroed rot), rotation via direct J_rot^T (no Lambda)
    'gain_frame',  # str — "base": gains along base axes; "ee": along the EE axes (K = R diag(k) R^T)
], defaults=('base',))
```

Add a helper next to the quaternion utilities and extend `compute_pose_task_wrench`:

```python
def rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``[..., 3]`` vectors by unit ``[..., 4]`` (w, x, y, z) quaternions."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def compute_pose_task_wrench(
    ee_pos, ee_quat, ee_linvel, ee_angvel, target_pos, target_quat,
    task_prop_gains, task_deriv_gains, gains_in_ee_frame: bool = False,
) -> torch.Tensor:
    """Compute task-space wrench for pose control.

    Matches factory_control_utils.compute_pose_task_wrench(). With ``gains_in_ee_frame`` the
    per-axis gains act along the EE axes: error and velocity are rotated into the EE frame,
    scaled, and the wrench rotated back (K_base = R diag(k) R^T) -- the apple_pick_sim VIC law
    for tool-frame policies (vic_wrench.py::compute_vic_spatial_wrench_aniso_tool).

    Returns:
        [6] wrench (Fx, Fy, Fz, Tx, Ty, Tz) in the base frame.
    """
    pos_error, aa_error = compute_pose_error(ee_pos, ee_quat, target_pos, target_quat)
    linvel, angvel = ee_linvel, ee_angvel
    if gains_in_ee_frame:
        q_inv = quat_conjugate(ee_quat)
        pos_error, aa_error = rotate_wxyz(q_inv, pos_error), rotate_wxyz(q_inv, aa_error)
        linvel, angvel = rotate_wxyz(q_inv, ee_linvel), rotate_wxyz(q_inv, ee_angvel)

    wrench = torch.zeros(*pos_error.shape[:-1], 6, dtype=pos_error.dtype, device=pos_error.device)
    wrench[..., :3] = task_prop_gains[..., :3] * pos_error + task_deriv_gains[..., :3] * (0.0 - linvel)
    wrench[..., 3:6] = task_prop_gains[..., 3:6] * aa_error + task_deriv_gains[..., 3:6] * (0.0 - angvel)
    if gains_in_ee_frame:
        wrench = torch.cat([rotate_wxyz(ee_quat, wrench[..., :3]), rotate_wxyz(ee_quat, wrench[..., 3:])], dim=-1)
    return wrench
```

In `compute_torques_from_targets` step 1, pass `gains_in_ee_frame=(targets.gain_frame == "ee")`. Leave the
pose-integral block alone (the RL env sends `pose_ki = 0`; the integral stays base-frame for other callers).
In `pack_control_targets` add `'gain_frame': str(targets.gain_frame)`. In `unpack_control_targets` add
`gain_frame=data.get('gain_frame', 'base')`.

- [ ] **Step 4: Run tests, expect pass**

Run: `cd ~/codes/Continuous_Force_RL && python3 -m unittest real_robot_exps.test_hybrid_controller_gain_frame -v`
Expected: 6 tests OK. Also run `python3 -m unittest real_robot_exps.test_field_verify real_robot_exps.test_field_pull -v` and expect the same result as before the change.

- [ ] **Step 5: Commit** (rig repo)

```bash
git add real_robot_exps/hybrid_controller.py real_robot_exps/test_hybrid_controller_gain_frame.py
git commit -m "hybrid_controller: optional EE-frame pose gains (K = R diag(k) R^T) for tool-frame VIC policies"
```

---

### Task 2: Parity fixture generated by the training code

**Files:**
- Create: `~/codes/Continuous_Force_RL/real_robot_exps/tools/make_vic_harvest_parity_fixture.py`
- Create (generated, committed): `~/codes/Continuous_Force_RL/real_robot_exps/testdata/vic_harvest_tool_parity.json`
- Copy (gitignored): `checkpoint_cache/vic_harvest/v2b_s0/ckpt_000016000/{agent.pt,meta.json}` from the sim run dir

**Interfaces:**
- Produces: JSON with keys `obs_cases` (list of `{tcp_pos, tcp_quat_wxyz, linvel, angvel, ft_tool, start_pose_wxyz, target_pose_wxyz, last_action_env, expected_obs}`), `action_cases` (list of `{u, tcp_pose_wxyz, target_pose_wxyz, expected_env_action, expected_target_pose_wxyz, expected_vic_action}`), `bounds` (the checkpoint's `action_bounds`), `policy_rollout` (`{checkpoint: "v2b_s0/ckpt_000016000", raw_obs: [T][40], expected_action: [T][13]}`).

- [ ] **Step 1: Write the generator** (runs under the sim's `uv`; imports the training modules directly)

```python
"""Dev tool: dump training-side outputs for the rl_policy_env parity tests.

Run from the rig repo root with the sim's environment (needs apple_pick_gym + skrl):
    uv run --project ~/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo \
        python real_robot_exps/tools/make_vic_harvest_parity_fixture.py \
        --checkpoint checkpoint_cache/vic_harvest/v2b_s0/ckpt_000016000
"""

import argparse
import json
from pathlib import Path

import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler

from apple_pick_gym.batched_envs.harvest_action import (
    HarvestActionBounds, integrate_delta_pose, leash_target_pose, pack_vic_pose_action,
    split_harvest_action, tool_delta_to_world,
)
from apple_pick_gym.batched_envs.harvest_obs import tool_frame_obs
from apple_pick_gym.rl.action_scaling import HarvestActionScaler
from apple_pick_gym.rl.config import RecurrentNetConfig
from apple_pick_gym.rl.models import _RecurrentTower

N_OBS, N_ACT, T = 64, 64, 40
OUT = Path(__file__).resolve().parents[1] / "testdata" / "vic_harvest_tool_parity.json"


def _unit_quat(n, g):
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def wxyz_to_xyzw(q):
    return q[:, [1, 2, 3, 0]]


def obs_cases(g, scaler):
    pos = torch.randn(N_OBS, 3, generator=g) * 0.3
    q = _unit_quat(N_OBS, g)
    vel = torch.randn(N_OBS, 6, generator=g) * 0.3
    start_pos = pos + torch.randn(N_OBS, 3, generator=g) * 0.05
    start_q = _unit_quat(N_OBS, g)
    tgt_pos = pos + torch.randn(N_OBS, 3, generator=g) * 0.02
    tgt_q = _unit_quat(N_OBS, g)
    ft_tool = torch.randn(N_OBS, 6, generator=g) * 5.0
    last_env = scaler.to_env(torch.rand(N_OBS, 13, generator=g) * 2 - 1)
    tool = tool_frame_obs(
        tcp_pos=pos, tcp_quat=wxyz_to_xyzw(q), tcp_velocity=vel, ft=torch.zeros(N_OBS, 6),
        start_pos=start_pos, start_quat=wxyz_to_xyzw(start_q), target_pos=tgt_pos, target_quat=wxyz_to_xyzw(tgt_q),
    )
    tool["ft_tool"] = ft_tool  # the env passes the sensor-frame reading through unchanged
    fields = ["d_pos_tool", "d_rot_tool", "gravity_tool", "twist_tool", "ft_tool", "target_offset_tool"]
    expected = torch.cat([tool[f] for f in fields] + [last_env], dim=-1)
    return [
        dict(tcp_pos=pos[i].tolist(), tcp_quat_wxyz=q[i].tolist(), linvel=vel[i, :3].tolist(),
             angvel=vel[i, 3:].tolist(), ft_tool=ft_tool[i].tolist(),
             start_pose_wxyz=torch.cat([start_pos[i], start_q[i]]).tolist(),
             target_pose_wxyz=torch.cat([tgt_pos[i], tgt_q[i]]).tolist(),
             last_action_env=last_env[i].tolist(), expected_obs=expected[i].tolist())
        for i in range(N_OBS)
    ]


def action_cases(g, scaler, b):
    u = torch.rand(N_ACT, 13, generator=g) * 2.4 - 1.2  # includes out-of-box values
    tcp = torch.cat([torch.randn(N_ACT, 3, generator=g) * 0.3, _unit_quat(N_ACT, g)], dim=-1)
    target = tcp.clone()
    target[:, :3] += torch.randn(N_ACT, 3, generator=g) * 0.1  # some beyond the 0.15 m leash
    env_action = scaler.to_env(u)
    split = split_harvest_action(env_action, b)
    delta = tool_delta_to_world(split.delta, tcp[:, 3:7])
    new_target = integrate_delta_pose(target, delta)
    new_target = leash_target_pose(new_target, tcp, max_pos_offset_m=b.max_target_pos_offset_m,
                                   max_rot_offset_rad=b.max_target_rot_offset_rad)
    vic = pack_vic_pose_action(new_target, split.linear_k, split.angular_k, split.zeta)
    return [
        dict(u=u[i].tolist(), tcp_pose_wxyz=tcp[i].tolist(), target_pose_wxyz=target[i].tolist(),
             expected_env_action=env_action[i].tolist(), expected_target_pose_wxyz=new_target[i].tolist(),
             expected_vic_action=vic[i].tolist())
        for i in range(N_ACT)
    ]


def policy_rollout(ckpt: Path, meta: dict, g):
    """The training actor + skrl's scaler, eval path (RecurrentPolicyRunner.act): clamped mean."""
    mods = torch.load(ckpt / "agent.pt", map_location="cpu", weights_only=False)
    a = meta["config"]["actor"]
    cfg = RecurrentNetConfig(**{k: a[k] for k in ("pre_mlp", "lstm_hidden", "lstm_layers", "post_mlp", "sequence_length")})
    tower = _RecurrentTower(40, cfg, num_envs=1)
    head = torch.nn.Linear(tower.out_dim, 13)
    pol = mods["policy"]
    tower.load_state_dict({k[len("tower."):]: v for k, v in pol.items() if k.startswith("tower.")})
    head.load_state_dict({k[len("mean_head."):]: v for k, v in pol.items() if k.startswith("mean_head.")})
    scaler = RunningStandardScaler(size=40, device="cpu")
    scaler.load_state_dict(mods["observation_preprocessor"])
    tower.eval()
    mean, std = scaler.running_mean, scaler.running_variance.sqrt()
    raw = mean + std * torch.randn(T, 40, generator=g)  # in-distribution inputs
    h = c = torch.zeros(1, 1, a["lstm_hidden"])
    acts = []
    with torch.no_grad():
        for t in range(T):
            x = scaler(raw[t : t + 1], train=False)
            feats, (h, c) = tower(x, {"rnn": [h, c]}, training=False)
            m = a["mean_bound"] * torch.tanh(head(feats) / a["mean_bound"])
            acts.append(m.clamp(-1.0, 1.0)[0])
    return dict(checkpoint="v2b_s0/ckpt_000016000", raw_obs=raw.tolist(), expected_action=torch.stack(acts).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    ckpt = Path(args.checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    b = HarvestActionBounds(**meta["action_bounds"])
    scaler = HarvestActionScaler(b)
    g = torch.Generator().manual_seed(1234)
    out = dict(bounds=meta["action_bounds"], obs_cases=obs_cases(g, scaler),
               action_cases=action_cases(g, scaler, b), policy_rollout=policy_rollout(ckpt, meta, g))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Cache the checkpoint and generate**

```bash
cd ~/codes/Continuous_Force_RL
mkdir -p checkpoint_cache/vic_harvest/v2b_s0
cp -r ~/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo/runs/vic_harvest/v2b_s0/checkpoints/ckpt_000016000 checkpoint_cache/vic_harvest/v2b_s0/
uv run --project ~/codes/apple_pick_sim/.claude/worktrees/rl-skrl-ppo python real_robot_exps/tools/make_vic_harvest_parity_fixture.py \
    --checkpoint checkpoint_cache/vic_harvest/v2b_s0/ckpt_000016000
```
Expected: `wrote .../testdata/vic_harvest_tool_parity.json`, about 200 KB. If `RecurrentNetConfig` or
`RunningStandardScaler`'s call signature differs, adapt the generator only. It is the training-side reference.

- [ ] **Step 3: Commit** (rig repo; the checkpoint itself stays gitignored)

```bash
git add real_robot_exps/tools/make_vic_harvest_parity_fixture.py real_robot_exps/testdata/vic_harvest_tool_parity.json
git commit -m "rl_policy_env: training-side parity fixture for the v2b tool-frame contract"
```

---

### Task 3: Tool-frame observation and action in `rl_policy_env.py`

**Files:**
- Modify: `~/codes/Continuous_Force_RL/real_robot_exps/rl_policy_env.py`
- Test: `~/codes/Continuous_Force_RL/real_robot_exps/test_rl_policy_env.py` (create)

**Interfaces:**
- Consumes: `rotate_wxyz`, `ControlTargets.gain_frame` (Task 1); fixture (Task 2).
- Produces: `build_tool_actor_obs(snap, start_pose, target_pose, last_env_action) -> Tensor[40]`;
  `tool_delta_to_world(delta: Tensor[6], tcp_quat_wxyz: Tensor[4]) -> Tensor[6]`;
  `FrankaVicHarvestEnv.step` stores `self._last_action = env_action` (env units).
  Removed: `_build_actor_obs`, `_obs_quat_xyzw`, `_quat_wxyz_to_xyzw`, `quat_hint_xyzw`, `sim_base_pos`, `_SIM_ROBOT_BASE_POS`, `training_quat_mean_xyzw`.

- [ ] **Step 1: Write the failing tests**

```python
"""rl_policy_env vs the training code (fixture from tools/make_vic_harvest_parity_fixture.py)."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from real_robot_exps import rl_policy_env as rpe
from real_robot_exps.pro_robot_interface import StateSnapshot

FIXTURE = json.loads((Path(__file__).parent / "testdata" / "vic_harvest_tool_parity.json").read_text())
BOUNDS = rpe.HarvestActionBounds(**FIXTURE["bounds"])
T = lambda x: torch.tensor(x, dtype=torch.float32)


def snap(pos, quat, linvel=(0, 0, 0), angvel=(0, 0, 0), ft=(0,) * 6):
    z = torch.zeros
    return StateSnapshot(T(pos), T(quat), T(linvel), T(angvel), T(ft), z(7), z(7), z(7), z(7), z(7), z(7), z(6, 7), z(7, 7))


class ToolObsParityTest(unittest.TestCase):
    def test_obs_matches_training(self):
        for c in FIXTURE["obs_cases"]:
            s = snap(c["tcp_pos"], c["tcp_quat_wxyz"], c["linvel"], c["angvel"], c["ft_tool"])
            obs = rpe.build_tool_actor_obs(s, T(c["start_pose_wxyz"]), T(c["target_pose_wxyz"]), T(c["last_action_env"]))
            torch.testing.assert_close(obs, T(c["expected_obs"]), atol=1e-5, rtol=1e-5)

    def test_obs_sign_invariant(self):
        c = FIXTURE["obs_cases"][0]
        flip = lambda p: torch.cat([T(p)[:3], -T(p)[3:]])
        a = rpe.build_tool_actor_obs(snap(c["tcp_pos"], c["tcp_quat_wxyz"]), T(c["start_pose_wxyz"]), T(c["target_pose_wxyz"]), torch.zeros(13))
        b = rpe.build_tool_actor_obs(snap(c["tcp_pos"], (-T(c["tcp_quat_wxyz"])).tolist()), flip(c["start_pose_wxyz"]), flip(c["target_pose_wxyz"]), torch.zeros(13))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


class ToolActionParityTest(unittest.TestCase):
    def test_action_chain_matches_training(self):
        scaler = rpe.HarvestActionScaler(BOUNDS)
        for c in FIXTURE["action_cases"]:
            env_action = scaler.to_env(T(c["u"]))
            torch.testing.assert_close(env_action, T(c["expected_env_action"]), atol=1e-5, rtol=1e-5)
            split = rpe.split_harvest_action(env_action, BOUNDS)
            tcp = T(c["tcp_pose_wxyz"])
            target = rpe.integrate_delta_pose(T(c["target_pose_wxyz"]), rpe.tool_delta_to_world(split.delta, tcp[3:7]))
            target = rpe.leash_target_pose(target, tcp, BOUNDS)
            # q and -q are the same pose; compare position + |dot|
            exp = T(c["expected_target_pose_wxyz"])
            torch.testing.assert_close(target[:3], exp[:3], atol=1e-5, rtol=1e-5)
            self.assertAlmostEqual(abs(float(torch.dot(target[3:], exp[3:]))), 1.0, places=5)
            vic = rpe.pack_vic_pose_action(target, split.linear_k, split.angular_k, split.zeta)
            torch.testing.assert_close(vic[7:], T(c["expected_vic_action"])[7:], atol=1e-4, rtol=1e-5)


class EnvStepContractTest(unittest.TestCase):
    """FrankaVicHarvestEnv against a fake robot: what it sends and what it feeds back."""

    def _env(self):
        gains = {k: 0.0 for k in (
            "pose_integral_clamp", "pose_integral_reset_on_target", "kp_null", "kd_null",
            "singularity_damping", "partial_inertia_decoupling", "sep_ori")}
        with patch.object(rpe, "load_gains_from_config", return_value=gains):
            env = rpe.FrankaVicHarvestEnv({"robot": {}}, BOUNDS, 250)
        env.robot = MagicMock()
        q90z = [0.7071068, 0.0, 0.0, 0.7071068]  # TCP rotated +90 deg about base z
        env._snap = snap([0.4, 0.0, 0.3], q90z)
        env._target_pose = torch.cat([env._snap.ee_pos, env._snap.ee_quat])
        env._start_pose = env._target_pose.clone()
        env._cage_origin = env._target_pose.clone()
        env._default_dof_pos = torch.zeros(7)
        env.robot.get_state_snapshot.return_value = snap([0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0])  # post-action pose
        return env

    def test_step_uses_pre_action_tcp(self):
        env = self._env()
        u = torch.zeros(13)
        u[0] = 1.0  # full-speed +x_tool
        env.step(u.numpy())
        sent = env.robot.set_control_targets.call_args[0][0]
        # +x_tool under the pre-action +90 deg z TCP is +y_base, 2 mm
        torch.testing.assert_close(sent.target_pos, T([0.4, 0.002, 0.3]), atol=1e-6, rtol=0)
        self.assertEqual(sent.gain_frame, "ee")

    def test_last_action_is_env_units(self):
        env = self._env()
        u = torch.zeros(13)
        obs, *_ = env.step(u.numpy())
        expected = rpe.HarvestActionScaler(BOUNDS).to_env(u)
        torch.testing.assert_close(T(obs[27:40]), expected, atol=1e-5, rtol=1e-5)
        self.assertGreater(float(obs[33]), 50.0)  # u=0 -> geometric-mean K, ~100 N/m, not 0


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them, expect failure**

Run: `cd ~/codes/Continuous_Force_RL && python3 -m unittest real_robot_exps.test_rl_policy_env -v`
Expected: ERROR `module 'real_robot_exps.rl_policy_env' has no attribute 'build_tool_actor_obs'`.

- [ ] **Step 3: Implement in `rl_policy_env.py`**

Constants: replace `_ACTOR_LAYOUT`, `_PROPRIO_NAMES`, `_N_PROPRIO` and delete `_SIM_ROBOT_BASE_POS`:

```python
# [name, start, width] rows exactly as rl/checkpoint.py records them in meta.json -- the [D17]
# tool-frame actor layout (harvest_obs._ACTOR_TOOL_FIELDS + last_action).
_ACTOR_LAYOUT = [
    ["d_pos_tool", 0, 3],
    ["d_rot_tool", 3, 3],
    ["gravity_tool", 6, 3],
    ["twist_tool", 9, 6],
    ["ft_tool", 15, 6],
    ["target_offset_tool", 21, 6],
    ["last_action", 27, 13],
]
_N_PROPRIO = 27  # obs dims before last_action
_PROPRIO_NAMES = (
    ["d_pos.x", "d_pos.y", "d_pos.z", "d_rot.x", "d_rot.y", "d_rot.z", "gravity.x", "gravity.y", "gravity.z"]
    + ["twist.vx", "twist.vy", "twist.vz", "twist.wx", "twist.wy", "twist.wz"]
    + ["ft.Fx", "ft.Fy", "ft.Fz", "ft.Tx", "ft.Ty", "ft.Tz"]
    + ["tgt_off.x", "tgt_off.y", "tgt_off.z", "tgt_off.rx", "tgt_off.ry", "tgt_off.rz"]
)
```

Loader: after the layout check, also require the tool action frame:

```python
    if meta["config"]["env"].get("action_frame") != "tool":
        raise RuntimeError(f"{checkpoint_dir}: action_frame={meta['config']['env'].get('action_frame')!r}; "
                           "this controller runs [D17] tool-frame checkpoints only")
```

Action helper (next to `integrate_delta_pose`):

```python
def tool_delta_to_world(delta: torch.Tensor, tcp_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """[D17] Rotate a TCP-frame ``[6]`` ``(dp, drot)`` delta into the base frame. Norm-preserving, so
    the speed cap applied in TCP-frame units holds. See ``harvest_action.tool_delta_to_world``."""
    return torch.cat([rotate_wxyz(tcp_quat_wxyz, delta[0:3]), rotate_wxyz(tcp_quat_wxyz, delta[3:6])])
```

Observation (replaces `_build_actor_obs`, `_quat_wxyz_to_xyzw` and `_rotate`):

```python
_GRAVITY_BASE = torch.tensor([0.0, 0.0, -1.0])


def build_tool_actor_obs(
    snap: StateSnapshot, start_pose: torch.Tensor, target_pose: torch.Tensor, last_env_action: torch.Tensor
) -> torch.Tensor:
    """``[40]`` actor observation in ``harvest_obs._ACTOR_TOOL_FIELDS`` order + ``last_action``,
    see ``harvest_obs.tool_frame_obs``. Poses are ``[7]`` ``(pos, quat_wxyz)`` in the base frame.

    - Everything is start-relative and in the current TCP (= Franka EE) frame, so the base
      offset of the sim's world is irrelevant and quaternion signs cancel (rotvecs are shortest-arc).
    - ``ft_tool`` is ``snap.force_torque`` as is: ``-K_F_ext_hat_K``, EE frame, bias-subtracted and
      EMA'd at 1 kHz (alpha 0.05) -- what ``FtSensorConfig.rl_training`` models in the TCP frame.
    - ``last_env_action`` is the previous action in env units (``HarvestActionScaler.to_env``), as the
      sim env stores it (the skrl wrapper scales before ``env.step``).
    """
    q = snap.ee_quat
    q_inv = quat_conjugate(q)
    to_tool = lambda v: rotate_wxyz(q_inv, v)
    rot_err = axis_angle_from_quat(quat_mul(target_pose[3:7], q_inv))
    return torch.cat([
        to_tool(snap.ee_pos - start_pose[0:3]),
        axis_angle_from_quat(quat_mul(quat_conjugate(start_pose[3:7]), q)),
        to_tool(_GRAVITY_BASE.to(q)),
        to_tool(snap.ee_linvel),
        to_tool(snap.ee_angvel),
        snap.force_torque,
        to_tool(target_pose[0:3] - snap.ee_pos),
        to_tool(rot_err),
        last_env_action,
    ])
```

Import `rotate_wxyz` from `hybrid_controller` next to the existing quaternion imports.

Env changes in `FrankaVicHarvestEnv`:
- `__init__`: drop `sim_base_pos`, `quat_hint_xyzw`, `_prev_obs_quat`; add `self._start_pose: torch.Tensor | None = None`.
- `_obs(self, snap)`: `return build_tool_actor_obs(snap, self._start_pose, self._target_pose, self._last_action).numpy().astype(np.float32)`.
- `reset`: after the settle sleep and the second snapshot, set `self._start_pose = torch.cat([snap.ee_pos, snap.ee_quat])`
  and `self._target_pose = self._start_pose.clone()`, so the start, target and first obs come from the same snapshot
  (sim: the start is recorded at the reset gather). Keep `_cage_origin` from the first snapshot. Return `self._obs(snap), {}`.
- `step`:

```python
        action_t = torch.as_tensor(action, dtype=torch.float32)
        env_action = self.action_scaler.to_env(action_t)
        split = split_harvest_action(env_action, self.action_bounds)
        # [D17] training rotates the tool-frame delta, and leashes, against the TCP of the last
        # observation (the env's _last_tcp_pose_wxyz) -- self._snap, from before this action ran
        tcp_pose = torch.cat([self._snap.ee_pos, self._snap.ee_quat])
        target = integrate_delta_pose(self._target_pose, tool_delta_to_world(split.delta, tcp_pose[3:7]))
        target = leash_target_pose(target, tcp_pose, self.action_bounds)
        target = cage_target_pose(...)  # unchanged
        ...
        self._last_action = env_action  # env units, as the sim env stores it
        self._step_count += 1
        obs = self._obs(snap)
```

- `_build_control_targets`: add `gain_frame="ee"` with the comment
  `# [D17] per-axis K/D along the TCP axes (sim vic_gain_frame="tool")`.
- CLI `main()`: drop `quat_hint_xyzw=`; log `env_action` per step (`info["env_action"] = env_action.numpy()` in `step`);
  print `obs[15:21]` as the F/T in the step line (was `obs[13:19]`).
- Update the module docstring's "40-D observation" description to the tool layout, and say that world-frame checkpoints are refused.

- [ ] **Step 4: Run tests, expect pass**

Run: `cd ~/codes/Continuous_Force_RL && python3 -m unittest real_robot_exps.test_rl_policy_env real_robot_exps.test_hybrid_controller_gain_frame -v`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add real_robot_exps/rl_policy_env.py real_robot_exps/test_rl_policy_env.py
git commit -m "rl_policy_env: [D17] tool-frame obs/action/gains for v2b; last_action in env units"
```

---

### Task 4: Checkpoint-level checks and mock run

**Files:**
- Modify: `~/codes/Continuous_Force_RL/real_robot_exps/test_rl_policy_env.py` (append)
- Modify: `~/codes/Continuous_Force_RL/real_robot_exps/FIELD_PIPELINE.md` (add the RL policy run line, if that doc lists run commands)

- [ ] **Step 1: Write the tests**

```python
REPO = Path(__file__).resolve().parents[1]
V2B = REPO / "checkpoint_cache" / "vic_harvest" / "v2b_s0" / "ckpt_000016000"
D8B = REPO / "checkpoint_cache" / "vic_harvest" / "d8b_tanh15_s1" / "ckpt_000030400"


@unittest.skipUnless(V2B.exists(), "v2b checkpoint not cached (see Task 2 step 2)")
class CheckpointParityTest(unittest.TestCase):
    def test_policy_matches_training_actor(self):
        roll = FIXTURE["policy_rollout"]
        policy = rpe.HarvestPolicy(V2B)
        for raw, exp in zip(roll["raw_obs"], roll["expected_action"]):
            torch.testing.assert_close(policy.act(T(raw)), T(exp), atol=1e-5, rtol=1e-5)

    def test_bounds_come_from_meta(self):
        policy = rpe.HarvestPolicy(V2B)
        self.assertEqual(policy.action_bounds.linear_delta_m, 0.002)
        self.assertEqual(policy.action_bounds.k_lin_max, 500.0)
        self.assertEqual(policy.max_episode_steps, 250)


@unittest.skipUnless(D8B.exists(), "d8b checkpoint not cached")
class WrongContractTest(unittest.TestCase):
    def test_refuses_world_layout_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "actor_layout"):
            rpe.HarvestPolicy(D8B)
```

- [ ] **Step 2: Run, expect pass** (Tasks 1–3 already implement this; these tests pin the checkpoint itself)

Run: `cd ~/codes/Continuous_Force_RL && python3 -m unittest real_robot_exps.test_rl_policy_env -v`
Expected: all OK, none skipped.

- [ ] **Step 3: Mock end-to-end run**

Run: `cd ~/codes/Continuous_Force_RL && python3 -m real_robot_exps.rl_policy_env --checkpoint checkpoint_cache/vic_harvest/v2b_s0/ckpt_000016000 --mock --steps 60 --allow-ood --log /tmp/v2b_mock.npz`
Expected: 60 step lines, no exception, `wrote 60 steps to /tmp/v2b_mock.npz`. Then check that the logged
`vic_action[:, 7:10]` (K_lin) stays in [20, 500] and that consecutive `vic_action[:, 0:3]` differ by ≤ 2 mm.

- [ ] **Step 4: Commit**

```bash
git add real_robot_exps/test_rl_policy_env.py real_robot_exps/FIELD_PIPELINE.md
git commit -m "rl_policy_env: v2b checkpoint parity + wrong-contract refusal tests"
```

---

## First field run checklist (not code; for the operator)

1. `calibrate_ft_bias` with the gripper free, then grasp (unchanged).
2. At reset, read the OOD report. `gravity_tool` and `ft_tool` are the start-pose dims that can be off (the rest are ~0 at reset).
   A large constant `gravity.x/y` z-score on every apple points at D-2 (EE-frame vs sim-TCP-frame rotation).
3. The policy's linear K now reaches 500 N/m along the grip axis, and the target speed is capped at 0.12 m/s. The cage stays at 0.12 m / 0.5 rad around the start.
4. The policy runs past detach (D-3). Stop the run once the apple is off.
