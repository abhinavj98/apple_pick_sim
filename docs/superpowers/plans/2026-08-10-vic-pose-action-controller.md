# VIC pose-action controller (`vic_pose`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pose-setpoint + per-axis-gains VIC action path (`mode="vic_pose"`, `action_dim=19`) alongside the existing twist VIC (`mode="vic"`, `action_dim=6`), and wire it into `robot_replay/example_replay_real_batched.py` only.

**Architecture:** Extend `Fr3BatchedEEImpedanceController` (Method B — no new controller class) with a pose-action unpack path that writes TCP targets directly (no twist integration) and stages per-env anisotropic `Kp`/`Kd` onto the scene. Extend the batched VIC wrench kernel to use those gains with `v_des=0` when present, else keep today's isotropic scalar path untouched. Wire `ControllerConfig`/`BatchedHeterogeneousCoupledSim` to branch on `mode`. Add a narrow, additive `action_dim` parameter to the shared replay-action-stacking helpers in `batched_sysid_mmd_grid.py` (default stays `6`) so the real-replay path can use 19D actions without touching MMD/CMA callers. Add a `robot_replay/` packer that rewrites a converted dataset's `action` column from 6D twist to 19D `[pos(3), quat_wxyz(4), Kp(6), Kd(6)]` using `target_pose_4x4` + constant gains, and a `--controller-mode` flag on the real-replay example.

**Tech Stack:** Python, NVIDIA Warp (`wp.func`/`wp.kernel`), PyTorch (action tensors), pyarrow (parquet), pytest, uv.

## Global Constraints

- Real-robot parity law: `wrench = Kp ⊙ pose_error + Kd ⊙ (0 − ee_velocity)` (`v_des = 0` always for `vic_pose`).
- Action packing: `action[0:3]=pos`, `action[3:7]=quat (w,x,y,z)`, `action[7:13]=Kp[Fx,Fy,Fz,Tx,Ty,Tz]`, `action[13:19]=Kd[Fx,Fy,Fz,Tx,Ty,Tz]`.
- Quaternion convention is **wxyz** everywhere in this feature (matches `body_q`/Newton). `pose_4x4_to_pos_quat` in this repo returns **xyzw** — convert explicitly when packing.
- `mode="vic"` (twist, `action_dim=6`) must remain byte-for-byte unchanged in behavior; all new tests for it must stay green.
- `vic_pose` only reaches `robot_replay/example_replay_real_batched.py` in this slice. Do not touch `apple_pick_gym/batched_examples/*`, `ApplePickBatchedSysIdEnv` defaults, `gym_defaults()`, or CMA-ES.
- `stacked_recorded_actions_for_structure` / `build_recorded_actions_tensor` / `replay_batched_sysid_structure` get an **additive** `action_dim: int = 6` parameter; default behavior for existing MMD/CMA callers must not change.
- Run tests with `uv run --env-file pytest.env python -m pytest ...` from repo root.
- Tests touching `newton`/`fr3_robot` model builders are `@pytest.mark.slow`; follow `.cursor/rules/multitask-pytest.mdc` when running the slow suite.

---

## File map

| Path | Role |
|------|------|
| `apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py` | `ControllerMode` gains `"vic_pose"`; `validate()` enforces `action_dim==19` + `step_mode=="coupled"` for it |
| `apple_pick_sim/coupled_fruiting/vic_wrench.py` | New `compute_vic_spatial_wrench_aniso` (`wp.func`), per-axis `Kp`/`Kd`, `v_des=w_des=0` |
| `apple_pick_sim/coupled_fruiting/vic_joint_torques_batched.py` | New aniso kernel + launcher; gate in `launch_compute_vic_wrenches_batched` on `scene.vic_kp_lin_wp` |
| `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance_batched.py` | Pose-gain buffers, `unpack_pose_action`, `run_coupled_teleop_frame_from_pose_actions`, `stage_pose_gains_to_scene` |
| `apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py` | Branch `_run_fr3_teleop_from_actions` on `mode`; skip twist clip for `vic_pose` |
| `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` | Additive `action_dim` param on 3 helpers (default `6`) |
| `robot_replay/pack_vic_pose_actions.py` | New CLI: rewrite converted dataset's `action` column 6D→19D from `target_pose_4x4` + constant gains |
| `robot_replay/example_replay_real_batched.py` | New `--controller-mode {vic,vic_pose}` flag; `vic_pose` sets `mode="vic_pose"`, `action_dim=19` |
| Tests | `apple_pick_sim/tests/test_controller_config_actions.py`, new `apple_pick_sim/tests/test_vic_wrench_aniso.py`, `apple_pick_sim/tests/test_batched_vic.py` (new cases), new `apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py`, `apple_pick_gym/tests/*` (new case for `action_dim` param), new `robot_replay/tests/test_pack_vic_pose_actions.py` |

---

### Task 1: `ControllerConfig` accepts `vic_pose` + 19D action validation

**Files:**
- Modify: `apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py`
- Test: `apple_pick_sim/tests/test_controller_config_actions.py`

**Interfaces:**
- Produces: `ControllerMode = Literal["direct", "ee", "vic", "vic_pose"]`; `BatchedHeterogeneousCoupledSimConfig.validate()` raises `ValueError` (message containing `"vic_pose"`) when `mode="vic_pose"` and `action_dim != 19`, or when `mode="vic_pose"` and `robot.step_mode != "coupled"`.

- [ ] **Step 1: Write the failing tests**

Append to `apple_pick_sim/tests/test_controller_config_actions.py`:

```python
def test_vic_pose_requires_action_dim_19():
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2),
        controller=ControllerConfig(mode="vic_pose", action_dim=6),
    )
    with pytest.raises(ValueError, match="vic_pose"):
        cfg.validate()


def test_vic_pose_requires_coupled_step_mode():
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2),
        controller=ControllerConfig(mode="vic_pose", action_dim=19),
        robot=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2).robot,
            step_mode="vbd_only",
        ),
    )
    with pytest.raises(ValueError, match="vic_pose"):
        cfg.validate()


def test_vic_pose_action_dim_19_accepted():
    base = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2)
    cfg = dataclasses.replace(
        base,
        controller=ControllerConfig(mode="vic_pose", action_dim=19),
        robot=dataclasses.replace(base.robot, step_mode="coupled"),
    )
    cfg.validate()  # must not raise
    assert cfg.controller.expected_action_shape(2) == (2, 19)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_controller_config_actions.py -q`
Expected: `test_vic_pose_requires_action_dim_19` and the two new tests FAIL (`Literal["direct", "ee", "vic"]` rejects `"vic_pose"` only if you also type-check; practically the first failure is that `validate()` does not raise since no `vic_pose` branch exists yet).

- [ ] **Step 3: Implement**

In `apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py`, change:

```python
ControllerMode = Literal["direct", "ee", "vic"]
```

to:

```python
ControllerMode = Literal["direct", "ee", "vic", "vic_pose"]
```

Then, in `BatchedHeterogeneousCoupledSimConfig.validate()`, right after the existing `if self.controller.mode == "vic": ...` block (around line 328), add:

```python
        if self.controller.mode == "vic_pose":
            if self.robot.step_mode != "coupled":
                raise ValueError(
                    "controller.mode='vic_pose' requires robot.step_mode='coupled'"
                )
            if self.controller.action_dim != 19:
                raise ValueError(
                    "controller.mode='vic_pose' requires action_dim=19 "
                    f"([pos(3), quat_wxyz(4), Kp(6), Kd(6)]), got {self.controller.action_dim}"
                )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_controller_config_actions.py -q`
Expected: all PASS, including pre-existing tests.

- [ ] **Step 5: Regression check**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_heterogeneous_config.py -q`
Expected: PASS (unchanged `mode="vic"` behavior).

- [ ] **Step 6: Commit**

```bash
git add apple_pick_sim/coupled_fruiting/batched_heterogeneous_config.py apple_pick_sim/tests/test_controller_config_actions.py
git commit -m "feat: add vic_pose controller mode with 19D action validation"
```

---

### Task 2: Anisotropic pose-PD wrench function (`vic_wrench.py`)

**Files:**
- Modify: `apple_pick_sim/coupled_fruiting/vic_wrench.py`
- Test: `apple_pick_sim/tests/test_vic_wrench_aniso.py` (new)

**Interfaces:**
- Consumes: existing `_orientation_error_axis_angle(q_des, q_act) -> wp.vec3` (already in `vic_wrench.py`).
- Produces: `compute_vic_spatial_wrench_aniso(tcp_tf: wp.transform, tcp_qd: wp.spatial_vector, target_tf: wp.transform, kp_lin: wp.vec3, kp_ang: wp.vec3, kd_lin: wp.vec3, kd_ang: wp.vec3) -> wp.spatial_vector` (`wp.func`), law: `force = kp_lin * e_p - kd_lin * v_act`, `torque = kp_ang * e_r - kd_ang * w_act` (i.e. `v_des = w_des = 0`).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for anisotropic (per-axis Kp/Kd) VIC pose wrench law."""

from __future__ import annotations

import warp as wp

from apple_pick_sim.coupled_fruiting.vic_wrench import compute_vic_spatial_wrench_aniso


def test_zero_wrench_at_target_zero_velocity():
    tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
    qd = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(100.0, 100.0, 100.0)
    kp_ang = wp.vec3(10.0, 10.0, 10.0)
    kd_lin = wp.vec3(5.0, 5.0, 5.0)
    kd_ang = wp.vec3(1.0, 1.0, 1.0)
    w = compute_vic_spatial_wrench_aniso(tf, qd, tf, kp_lin, kp_ang, kd_lin, kd_ang)
    for i in range(6):
        assert abs(float(w[i])) < 1e-6


def test_per_axis_gain_scales_only_that_axis():
    tcp_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    target_tf = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity())
    qd = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(100.0, 400.0, 100.0)
    kp_ang = wp.vec3(0.0, 0.0, 0.0)
    kd_lin = wp.vec3(0.0, 0.0, 0.0)
    kd_ang = wp.vec3(0.0, 0.0, 0.0)
    w = compute_vic_spatial_wrench_aniso(tcp_tf, qd, target_tf, kp_lin, kp_ang, kd_lin, kd_ang)
    assert abs(float(w[0]) - 10.0) < 1e-3  # 100 * 0.1
    assert abs(float(w[1])) < 1e-6  # error is on x only
    assert abs(float(w[2])) < 1e-6


def test_damping_opposes_velocity_with_v_des_zero():
    tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    qd = wp.spatial_vector(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    kp_lin = wp.vec3(0.0, 0.0, 0.0)
    kp_ang = wp.vec3(0.0, 0.0, 0.0)
    kd_lin = wp.vec3(10.0, 10.0, 10.0)
    kd_ang = wp.vec3(1.0, 1.0, 1.0)
    w = compute_vic_spatial_wrench_aniso(tf, qd, tf, kp_lin, kp_ang, kd_lin, kd_ang)
    assert float(w[0]) < -1.9  # -kd * v = -10 * 0.2
```

Save as `apple_pick_sim/tests/test_vic_wrench_aniso.py`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_vic_wrench_aniso.py -q`
Expected: FAIL with `ImportError: cannot import name 'compute_vic_spatial_wrench_aniso'`.

- [ ] **Step 3: Implement**

In `apple_pick_sim/coupled_fruiting/vic_wrench.py`, add after `compute_vic_spatial_wrench` (after its closing `return` around line 96):

```python
@wp.func
def compute_vic_spatial_wrench_aniso(
    tcp_tf: wp.transform,
    tcp_qd: wp.spatial_vector,
    target_tf: wp.transform,
    kp_lin: wp.vec3,
    kp_ang: wp.vec3,
    kd_lin: wp.vec3,
    kd_ang: wp.vec3,
) -> wp.spatial_vector:
    """Per-axis pose-PD wrench in world frame at TCP COM, ``v_des = w_des = 0``.

    Matches real-robot ``compute_pose_task_wrench``: ``w = Kp * e - Kd * v_actual``.
    """
    p_des = wp.transform_get_translation(target_tf)
    q_des = wp.transform_get_rotation(target_tf)
    p_act = wp.transform_get_translation(tcp_tf)
    q_act = wp.transform_get_rotation(tcp_tf)

    e_p = p_des - p_act
    e_r = _orientation_error_axis_angle(q_des, q_act)

    v_act = wp.spatial_top(tcp_qd)
    w_act = wp.spatial_bottom(tcp_qd)

    force = wp.cw_mul(kp_lin, e_p) - wp.cw_mul(kd_lin, v_act)
    torque = wp.cw_mul(kp_ang, e_r) - wp.cw_mul(kd_ang, w_act)
    return wp.spatial_vector(force[0], force[1], force[2], torque[0], torque[1], torque[2])
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_vic_wrench_aniso.py -q`
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/coupled_fruiting/vic_wrench.py apple_pick_sim/tests/test_vic_wrench_aniso.py
git commit -m "feat: add anisotropic pose-PD wrench law for vic_pose"
```

---

### Task 3: Batched aniso wrench kernel + gated launcher

**Files:**
- Modify: `apple_pick_sim/coupled_fruiting/vic_joint_torques_batched.py`
- Test: `apple_pick_sim/tests/test_batched_vic.py`

**Interfaces:**
- Consumes: `compute_vic_spatial_wrench_aniso` (Task 2); scene fields `vic_kp_lin_wp`, `vic_kp_ang_wp`, `vic_kd_lin_wp`, `vic_kd_ang_wp` (each `wp.array(dtype=wp.vec3)`, one entry per env; `None` when not staged).
- Produces: `launch_compute_vic_wrenches_batched(scene, *, gains=None)` now checks `scene.vic_kp_lin_wp` first; if present, launches the new aniso kernel (ignoring `gains`/isotropic `ImpedanceGains`); else falls back to the existing isotropic kernel unchanged.

- [ ] **Step 1: Write the failing test**

Append to `apple_pick_sim/tests/test_batched_vic.py`:

```python
@requires_fr3
@pytest.mark.slow
def test_aniso_wrench_used_when_gain_buffers_staged():
    """Per-env anisotropic Kp/Kd buffers override isotropic ImpedanceGains."""
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    dev = scene.robot_model.device

    pos = ctrl._target_pos_wp.numpy().copy()
    pos[:, 0] += 0.1
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)

    scene.vic_kp_lin_wp = wp.full(_NUM_ENVS, wp.vec3(1000.0, 0.0, 0.0), dtype=wp.vec3, device=dev)
    scene.vic_kp_ang_wp = wp.zeros(_NUM_ENVS, dtype=wp.vec3, device=dev)
    scene.vic_kd_lin_wp = wp.zeros(_NUM_ENVS, dtype=wp.vec3, device=dev)
    scene.vic_kd_ang_wp = wp.zeros(_NUM_ENVS, dtype=wp.vec3, device=dev)

    launch_compute_vic_wrenches_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    for w in range(_NUM_ENVS):
        assert abs(float(wrenches[w, 0]) - 100.0) < 1.0, f"world {w} expected Kp_x*0.1=100"
        assert abs(float(wrenches[w, 1])) < 1e-3, f"world {w} expected zero on y (kp_lin_y=0)"


@requires_fr3
@pytest.mark.slow
def test_isotropic_path_unchanged_without_gain_buffers():
    """No aniso buffers staged -> existing isotropic kernel path still used (regression guard)."""
    scene = _build_batched_scene()
    ctrl = _configure_batched_vic(scene)
    assert getattr(scene, "vic_kp_lin_wp", None) is None
    pos = ctrl._target_pos_wp.numpy().copy()
    pos[:, 0] += 0.05
    ctrl._target_pos_wp.assign(pos.astype(np.float32))
    ctrl.stage_targets_to_scene(scene)
    launch_compute_vic_wrenches_batched(scene)
    wp.synchronize()
    wrenches = scene.vic_jt_wrench_buf.numpy()
    for w in range(_NUM_ENVS):
        assert float(np.linalg.norm(wrenches[w, :3])) > 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_vic.py::test_aniso_wrench_used_when_gain_buffers_staged -q -m slow`
Expected: FAIL — `scene.vic_kp_lin_wp` staged but the wrench kernel still uses isotropic `ImpedanceGains` (wrench does not match Kp_x=1000 expectation).

- [ ] **Step 3: Implement**

In `apple_pick_sim/coupled_fruiting/vic_joint_torques_batched.py`:

Add import:

```python
from apple_pick_sim.coupled_fruiting.vic_wrench import compute_vic_spatial_wrench_aniso
```

Add a new kernel after `_compute_vic_wrenches_batched_kernel` (after its closing line, around line 65):

```python
@wp.kernel(enable_backward=False)
def _compute_vic_wrenches_batched_aniso_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    tcp_indices: wp.array(dtype=int),
    target_positions: wp.array(dtype=wp.vec3),
    target_rotations: wp.array(dtype=wp.vec4),
    kp_lin: wp.array(dtype=wp.vec3),
    kp_ang: wp.array(dtype=wp.vec3),
    kd_lin: wp.array(dtype=wp.vec3),
    kd_ang: wp.array(dtype=wp.vec3),
    wrenches_out: wp.array2d(dtype=float),
):
    """Per-env anisotropic pose-PD wrench at TCP COM (world frame), ``v_des = 0``."""
    w = wp.tid()
    tcp_idx = tcp_indices[w]
    target_tf = wp.transform(target_positions[w], _vec4_to_quat(target_rotations[w]))
    wrench = compute_vic_spatial_wrench_aniso(
        body_q[tcp_idx],
        body_qd[tcp_idx],
        target_tf,
        kp_lin[w],
        kp_ang[w],
        kd_lin[w],
        kd_ang[w],
    )
    wrenches_out[w, 0] = wrench[0]
    wrenches_out[w, 1] = wrench[1]
    wrenches_out[w, 2] = wrench[2]
    wrenches_out[w, 3] = wrench[3]
    wrenches_out[w, 4] = wrench[4]
    wrenches_out[w, 5] = wrench[5]
```

Then change `launch_compute_vic_wrenches_batched` (currently starting `def launch_compute_vic_wrenches_batched(...)`, around line 167) to gate on the new buffers before the existing isotropic launch:

```python
def launch_compute_vic_wrenches_batched(
    scene: Any,
    *,
    gains: ImpedanceGains | None = None,
) -> None:
    """Fill ``vic_jt_wrench_buf`` from device TCP state and staged per-env targets."""
    if (
        scene.robot_state_0 is None
        or scene.vic_jt_wrench_buf is None
        or scene.vic_jt_tcp_indices_wp is None
    ):
        return
    target_positions = getattr(scene, "vic_target_positions_wp", None)
    target_rotations = getattr(scene, "vic_target_rotations_wp", None)
    if target_positions is None or target_rotations is None:
        return
    num_envs = int(scene.vic_jt_num_envs)
    dev = scene.robot_state_0.body_q.device

    kp_lin = getattr(scene, "vic_kp_lin_wp", None)
    kp_ang = getattr(scene, "vic_kp_ang_wp", None)
    kd_lin = getattr(scene, "vic_kd_lin_wp", None)
    kd_ang = getattr(scene, "vic_kd_ang_wp", None)
    if kp_lin is not None and kp_ang is not None and kd_lin is not None and kd_ang is not None:
        wp.launch(
            _compute_vic_wrenches_batched_aniso_kernel,
            dim=num_envs,
            inputs=[
                scene.robot_state_0.body_q,
                scene.robot_state_0.body_qd,
                scene.vic_jt_tcp_indices_wp,
                target_positions,
                target_rotations,
                kp_lin,
                kp_ang,
                kd_lin,
                kd_ang,
                scene.vic_jt_wrench_buf,
            ],
            device=dev,
        )
        return

    g = gains if gains is not None else ImpedanceGains()
    v_des_wp, w_des_wp = _resolve_batched_vic_desired_twists(scene, num_envs, dev)
    wp.launch(
        _compute_vic_wrenches_batched_kernel,
        dim=num_envs,
        inputs=[
            scene.robot_state_0.body_q,
            scene.robot_state_0.body_qd,
            scene.vic_jt_tcp_indices_wp,
            target_positions,
            target_rotations,
            v_des_wp,
            w_des_wp,
            float(g.linear_k),
            float(g.linear_d),
            float(g.angular_k),
            float(g.angular_d),
            scene.vic_jt_wrench_buf,
        ],
        device=dev,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_vic.py -q -m slow`
Expected: all PASS, including the two new tests and all pre-existing ones (`test_zero_wrench_at_target`, etc.) unchanged.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_sim/coupled_fruiting/vic_joint_torques_batched.py apple_pick_sim/tests/test_batched_vic.py
git commit -m "feat: gate batched VIC wrench kernel on per-env anisotropic gain buffers"
```

---

### Task 4: `Fr3BatchedEEImpedanceController` pose-action unpack path

**Files:**
- Modify: `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance_batched.py`
- Test: `apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py` (new)

**Interfaces:**
- Consumes: nothing new outside this file/Task 3's scene fields.
- Produces on `Fr3BatchedEEImpedanceController`:
  - `unpack_pose_action(actions) -> None` — reads a `(num_envs, 19)` `torch.Tensor`, writes `_target_pos_wp`/`_target_rot_wp` directly (quat normalized to unit; near-zero → `(1,0,0,0)` wxyz), zeros `_lin_vels_wp`/`_ang_vels_wp`, and stores `_kp_lin_wp`/`_kp_ang_wp`/`_kd_lin_wp`/`_kd_ang_wp` (each `wp.array(dtype=wp.vec3, shape=(num_envs,))`, allocated lazily on first call).
  - `run_coupled_teleop_frame_from_pose_actions(state, control, mj_solver, dt, actions) -> EEVelocity` — calls `unpack_pose_action`, syncs `self.target_tf` from device, returns `EEVelocity()`.
  - `stage_pose_gains_to_scene(scene) -> None` — sets `scene.vic_kp_lin_wp`, `scene.vic_kp_ang_wp`, `scene.vic_kd_lin_wp`, `scene.vic_kd_ang_wp` from the controller's buffers (`None` if `unpack_pose_action` was never called).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for Fr3BatchedEEImpedanceController pose-action unpack (vic_pose)."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.robot import fr3_robot
from conftest import COUPLED_SCENE_KW, RANGES_FIXTURE, build_homogeneous_batched_fr3, requires_fr3

_NUM_ENVS = 2


def _build_ctrl():
    ranges = load_ranges(RANGES_FIXTURE)
    scene = build_homogeneous_batched_fr3(
        ranges, 42, device="cpu", num_envs=_NUM_ENVS, **COUPLED_SCENE_KW
    )
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(scene)
    ctrl = fr3_robot.Fr3BatchedEEImpedanceController(scene.robot_model, **ik_kw)
    return scene, ctrl


def _pose_action_row(pos, quat_wxyz, kp, kd):
    return list(pos) + list(quat_wxyz) + list(kp) + list(kd)


@requires_fr3
@pytest.mark.slow
def test_unpack_pose_action_sets_targets_directly():
    scene, ctrl = _build_ctrl()
    row0 = _pose_action_row((0.3, 0.4, 0.5), (1.0, 0.0, 0.0, 0.0), [1.0] * 6, [2.0] * 6)
    row1 = _pose_action_row((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0), [3.0] * 6, [4.0] * 6)
    actions = torch.tensor([row0, row1], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    pos = ctrl._target_pos_wp.numpy()
    np.testing.assert_allclose(pos[0], [0.3, 0.4, 0.5], atol=1e-6)
    np.testing.assert_allclose(pos[1], [0.1, 0.2, 0.3], atol=1e-6)
    lin = ctrl._lin_vels_wp.numpy()
    ang = ctrl._ang_vels_wp.numpy()
    assert np.allclose(lin, 0.0)
    assert np.allclose(ang, 0.0)


@requires_fr3
@pytest.mark.slow
def test_unpack_pose_action_near_zero_quat_defaults_to_identity():
    scene, ctrl = _build_ctrl()
    row = _pose_action_row((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), [0.0] * 6, [0.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    rot = ctrl._target_rot_wp.numpy()  # stored xyzw (Warp-native); identity is (0,0,0,1)
    for w in range(_NUM_ENVS):
        np.testing.assert_allclose(rot[w], [0.0, 0.0, 0.0, 1.0], atol=1e-6)


@requires_fr3
@pytest.mark.slow
def test_stage_pose_gains_to_scene_wires_buffers():
    scene, ctrl = _build_ctrl()
    row = _pose_action_row((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [1.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.unpack_pose_action(actions)
    ctrl.stage_pose_gains_to_scene(scene)
    assert scene.vic_kp_lin_wp is not None
    kp_lin = scene.vic_kp_lin_wp.numpy()
    np.testing.assert_allclose(kp_lin[0], [5.0, 6.0, 7.0], atol=1e-6)
    kp_ang = scene.vic_kp_ang_wp.numpy()
    np.testing.assert_allclose(kp_ang[0], [8.0, 9.0, 10.0], atol=1e-6)


@requires_fr3
@pytest.mark.slow
def test_run_coupled_teleop_frame_from_pose_actions_syncs_target_tf():
    scene, ctrl = _build_ctrl()
    row = _pose_action_row((0.7, 0.8, 0.9), (1.0, 0.0, 0.0, 0.0), [1.0] * 6, [1.0] * 6)
    actions = torch.tensor([row, row], dtype=torch.float32)
    ctrl.run_coupled_teleop_frame_from_pose_actions(
        scene.robot_state_0, scene.robot_control, scene.mj_solver, 1.0 / 15.0, actions
    )
    import warp as wp

    p0 = wp.transform_get_translation(ctrl.target_tf[0])
    np.testing.assert_allclose([p0[0], p0[1], p0[2]], [0.7, 0.8, 0.9], atol=1e-6)
```

Save as `apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py -q -m slow`
Expected: FAIL with `AttributeError: 'Fr3BatchedEEImpedanceController' object has no attribute 'unpack_pose_action'`.

- [ ] **Step 3: Implement**

In `apple_pick_sim/robot/fr3_robot/controllers/ee_impedance_batched.py`, add near the top-level imports:

```python
import torch
```

(Guarded the same way the rest of the batched teleop stack assumes torch is available at call time — mirror `batched_action_twists.py`'s local `import torch` inside methods instead of module-level, to avoid a hard import-time dependency.)

In `__init__`, after the existing buffer allocations (`self._ang_vels_wp = wp.zeros(...)`, around line 53), add:

```python
        self._kp_lin_wp: wp.array | None = None
        self._kp_ang_wp: wp.array | None = None
        self._kd_lin_wp: wp.array | None = None
        self._kd_ang_wp: wp.array | None = None
```

Add new methods after `sync_target_from_state` (after its closing line, around line 84):

```python
    def unpack_pose_action(self, actions) -> None:
        """Unpack ``(num_envs, 19)`` pose-action rows: set targets directly, zero twists.

        Layout: ``[pos(3), quat_wxyz(4), Kp(6), Kd(6)]``. ``Kp``/``Kd`` split as
        ``[Fx,Fy,Fz,Tx,Ty,Tz]`` into linear (0:3) and angular (3:6) halves.
        """
        n = self.layout.num_envs
        dev = self.robot_model.device
        torch_device = wp.device_to_torch(dev)
        a = actions.to(device=torch_device, dtype=torch.float32)
        if a.shape != (n, 19):
            raise ValueError(f"pose action must have shape ({n}, 19), got {tuple(a.shape)}")

        pos = a[:, 0:3].contiguous()
        quat_wxyz = a[:, 3:7].contiguous()
        quat_norm = torch.linalg.norm(quat_wxyz, dim=1, keepdim=True)
        identity_wxyz = torch.zeros_like(quat_wxyz)
        identity_wxyz[:, 0] = 1.0
        quat_wxyz = torch.where(
            quat_norm > 1.0e-9, quat_wxyz / quat_norm.clamp_min(1.0e-9), identity_wxyz
        )
        # Action contract is wxyz; Newton body_q / _target_rot_wp store Warp-native
        # xyzw (verified via batched_template_ik.py's wp.quat(x, y, z, w) usage).
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]].contiguous()

        kp = a[:, 7:13].contiguous()
        kd = a[:, 13:19].contiguous()

        wp.copy(self._target_pos_wp, wp.from_torch(pos, dtype=wp.vec3))
        wp.copy(self._target_rot_wp, wp.from_torch(quat_xyzw, dtype=wp.vec4))
        zeros = torch.zeros((n, 3), device=torch_device, dtype=torch.float32)
        wp.copy(self._lin_vels_wp, wp.from_torch(zeros.clone(), dtype=wp.vec3))
        wp.copy(self._ang_vels_wp, wp.from_torch(zeros, dtype=wp.vec3))

        if self._kp_lin_wp is None:
            self._kp_lin_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
            self._kp_ang_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
            self._kd_lin_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
            self._kd_ang_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
        wp.copy(self._kp_lin_wp, wp.from_torch(kp[:, 0:3].contiguous(), dtype=wp.vec3))
        wp.copy(self._kp_ang_wp, wp.from_torch(kp[:, 3:6].contiguous(), dtype=wp.vec3))
        wp.copy(self._kd_lin_wp, wp.from_torch(kd[:, 0:3].contiguous(), dtype=wp.vec3))
        wp.copy(self._kd_ang_wp, wp.from_torch(kd[:, 3:6].contiguous(), dtype=wp.vec3))
        self._sync_target_tf_from_device()

    def run_coupled_teleop_frame_from_pose_actions(
        self,
        state: Any,
        control: Any,
        mj_solver: Any,
        dt: float,
        actions,
    ) -> EEVelocity:
        """Per-frame ``vic_pose`` teleop: set target pose + gains directly (no integration)."""
        del state, control, mj_solver, dt  # target comes entirely from ``actions``
        self.unpack_pose_action(actions)
        return EEVelocity()

    def stage_pose_gains_to_scene(self, scene: Any) -> None:
        """Wire per-env anisotropic gain buffers onto ``scene`` (no-op before first pose action)."""
        scene.vic_kp_lin_wp = self._kp_lin_wp
        scene.vic_kp_ang_wp = self._kp_ang_wp
        scene.vic_kd_lin_wp = self._kd_lin_wp
        scene.vic_kd_ang_wp = self._kd_ang_wp
```

Quaternion unpack note (confirmed, not just flagged): `_target_rot_wp` stores Warp-native **xyzw** (`batched_template_ik.py` writes it as `wp.vec4(body_rot[0], body_rot[1], body_rot[2], body_rot[3])` where `body_rot = wp.transform_get_rotation(tf)` is a `wp.quat(x, y, z, w)`). The action contract's `quat` field is **wxyz** (external, per this plan's Global Constraints). `unpack_pose_action` reorders `wxyz → xyzw` via `quat_wxyz[:, [1, 2, 3, 0]]` before writing `_target_rot_wp` (already reflected in the Step 3 code above) — no further fix needed here.

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py -q -m slow`
Expected: all 4 PASS. If `test_run_coupled_teleop_frame_from_pose_actions_syncs_target_tf` fails on quat ordering, fix per the note above and rerun.

- [ ] **Step 5: Regression check**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_vic.py -q -m slow`
Expected: PASS (twist path untouched).

- [ ] **Step 6: Commit**

```bash
git add apple_pick_sim/robot/fr3_robot/controllers/ee_impedance_batched.py apple_pick_sim/tests/test_ee_impedance_batched_pose_actions.py
git commit -m "feat: add vic_pose action unpack to Fr3BatchedEEImpedanceController"
```

---

### Task 5: Wire `vic_pose` into `BatchedHeterogeneousCoupledSim.step()`

**Files:**
- Modify: `apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py`
- Test: `apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py`

**Interfaces:**
- Consumes: `run_coupled_teleop_frame_from_pose_actions`, `stage_pose_gains_to_scene` (Task 4).
- Produces: `BatchedHeterogeneousCoupledSim.step(actions)` accepts a `(num_envs, 19)` tensor when `config.controller.mode == "vic_pose"`, applies it without twist speed clipping, and drives the aniso wrench path end to end.

- [ ] **Step 1: Write the failing test**

Append to `apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py` (adapt fixture helpers already used in that file for a `mode="vic"` 2-env sim — mirror the existing `mode="vic"` construction test, swapping `controller=ControllerConfig(mode="vic_pose", action_dim=19)`):

```python
@requires_fr3
@pytest.mark.slow
def test_vic_pose_step_moves_tcp_toward_target():
    import torch

    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=1),
        controller=ControllerConfig(mode="vic_pose", action_dim=19),
        robot=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=1).robot,
            kind="fr3",
            step_mode="coupled",
        ),
    )
    sim = BatchedHeterogeneousCoupledSim(cfg)
    tcp0 = sim.scene.robot_state_0.body_q.numpy().reshape(-1, 7)[
        sim.scene.layout.tcp_body_indices[0]
    ][:3].copy()
    row = [
        float(tcp0[0]) + 0.05, float(tcp0[1]), float(tcp0[2]),
        1.0, 0.0, 0.0, 0.0,
        800.0, 800.0, 800.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 2.0, 2.0, 2.0,
    ]
    actions = torch.tensor([row], dtype=torch.float32)
    for _ in range(50):
        sim.step(actions)
    tcp1 = sim.scene.robot_state_0.body_q.numpy().reshape(-1, 7)[
        sim.scene.layout.tcp_body_indices[0]
    ][:3]
    assert float(tcp1[0] - tcp0[0]) > 0.01
```

(If `test_minimal` in this test module does not already build an FR3-coupled scene with a batched VIC controller pre-wired for `mode="vic"`, reuse whatever helper the existing `mode="vic"` tests in this file use to construct `BatchedHeterogeneousCoupledSim` — copy that construction path and only change `controller`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py::test_vic_pose_step_moves_tcp_toward_target -q -m slow`
Expected: FAIL — either `validate()` doesn't yet route to the pose teleop, or `_run_fr3_teleop_from_actions` calls the twist path and mis-sizes the 19D action against `action_dim=6`'s old speed-clip logic.

- [ ] **Step 3: Implement**

In `apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py`:

1. `_clip_actions` — skip clipping for `vic_pose` (pose/gains must pass through unmodified):

```python
    def _clip_actions(self, actions):
        if self._config.controller.mode == "vic_pose":
            return actions
        return clip_action_tensor(
            actions,
            linear_speed=float(self._config.controller.linear_speed),
            angular_speed=float(self._config.controller.angular_speed),
        )
```

2. `_run_fr3_teleop_from_actions` — branch on mode:

```python
    def _run_fr3_teleop_from_actions(self) -> None:
        cfg = self._config
        assert self._ee_ctrl is not None
        if cfg.controller.mode == "vic_pose":
            assert isinstance(self._ee_ctrl, fr3_robot.Fr3BatchedEEImpedanceController)
            velocity = self._ee_ctrl.run_coupled_teleop_frame_from_pose_actions(
                self._scene.robot_state_0,
                self._scene.robot_control,
                self._scene.mj_solver,
                self.frame_dt,
                self._action_buffer,
            )
            self._ee_ctrl.stage_targets_to_scene(self._scene)
            self._ee_ctrl.stage_pose_gains_to_scene(self._scene)
            self._scene.vic_target_twist = velocity
            return
        velocity = self._ee_ctrl.run_coupled_teleop_frame_from_actions(
            self._scene.robot_state_0,
            self._scene.robot_control,
            self._scene.mj_solver,
            self.frame_dt,
            self._action_buffer,
        )
        if getattr(self._scene, "vic_controller", None) is not None:
            if isinstance(self._ee_ctrl, fr3_robot.Fr3BatchedEEImpedanceController):
                self._ee_ctrl.stage_targets_to_scene(self._scene)
                self._scene.vic_target_twist = velocity
```

3. Wherever the controller is constructed for `mode == "vic"` (around line 213/237), confirm `mode == "vic_pose"` reuses the same `Fr3BatchedEEImpedanceController` construction branch (extend the `elif mode == "vic":` condition to `elif mode in ("vic", "vic_pose"):` so both share setup — `configure_vic_joint_torques_arm_batched` etc. do not depend on the action layout).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py -q -m slow`
Expected: all PASS, including pre-existing `mode="vic"` tests.

- [ ] **Step 5: Regression check**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_batched_heterogeneous_config.py apple_pick_sim/tests/test_example_batched_heterogeneous_coupled_sim.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py
git commit -m "feat: wire vic_pose mode into BatchedHeterogeneousCoupledSim.step"
```

---

### Task 6: Additive `action_dim` param on shared replay-action helpers

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`
- Test: `apple_pick_gym/tests/test_batched_sysid_replay.py` (add one case) or a new focused test file if that module doesn't already exercise `stacked_recorded_actions_for_structure` directly

**Interfaces:**
- Produces: `stacked_recorded_actions_for_structure(..., action_dim: int = 6)`, `build_recorded_actions_tensor(..., action_dim: int = 6)`, `replay_batched_sysid_structure(..., action_dim: int = 6)` — all three plumb the same `action_dim` through; default `6` preserves every existing caller's behavior exactly.

- [ ] **Step 1: Write the failing test**

Add to `apple_pick_gym/tests/test_batched_sysid_replay.py` (or create `apple_pick_gym/tests/test_batched_sysid_action_dim.py` if that file lacks direct dataset fixtures — reuse whatever `BatchedSysIdDataset` fixture the existing replay tests in that module already build, and write a 19-wide `action` column into a temp episode via the same writer helpers used elsewhere in the file):

```python
def test_stacked_recorded_actions_accepts_action_dim_19(tmp_dataset_with_wide_action):
    dataset, structure_idx = tmp_dataset_with_wide_action
    stacked = stacked_recorded_actions_for_structure(
        dataset,
        structure_idx=structure_idx,
        num_directions=1,
        num_candidates=1,
        action_dim=19,
    )
    assert stacked.shape[-1] == 19


def test_stacked_recorded_actions_default_still_requires_6():
    with pytest.raises(ValueError, match="shape"):
        stacked_recorded_actions_for_structure(
            dataset_with_19d_action_fixture,
            structure_idx=0,
            num_directions=1,
            num_candidates=1,
        )
```

(Wire `tmp_dataset_with_wide_action` as a small pytest fixture in the same file: write one `BatchedEpisodeWriter` episode with `action` rows of length 19 to a temp directory, matching the pattern the file already uses to build fixture datasets for its other replay tests — reuse `write_manifest` / episode writer calls already imported there rather than adding new I/O helpers.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_batched_sysid_replay.py -k action_dim -q`
Expected: FAIL — `stacked_recorded_actions_for_structure()` has no `action_dim` parameter yet (`TypeError: unexpected keyword argument`).

- [ ] **Step 3: Implement**

In `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`:

Change the function starting `def stacked_recorded_actions_for_structure(` (around line 1319) to accept and use `action_dim`:

```python
def stacked_recorded_actions_for_structure(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
    action_dim: int = 6,
) -> np.ndarray:
    """Stack recorded EE actions for all candidate/direction env slots."""
    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    direction_actions: list[np.ndarray] = []
    n_frames: int | None = None
    for direction_idx in dirs:
        arrays = strip_pre_weld_rows(
            dataset.load_episode_obs_arrays(structure_idx, int(direction_idx))
        )
        action = np.asarray(arrays["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != action_dim:
            raise ValueError(
                f"expected action shape (n_frames, {action_dim}), got {action.shape!r}"
            )
        if n_frames is None:
            n_frames = int(action.shape[0])
        elif int(action.shape[0]) != n_frames:
            raise ValueError("all direction episodes must have same n_frames")
        direction_actions.append(action)

    if n_frames is None:
        raise ValueError("num_directions must be positive")

    d = len(dirs)
    num_envs = int(num_candidates) * d
    out = np.empty((num_envs, n_frames, action_dim), dtype=np.float32)
    for candidate_idx in range(num_candidates):
        for local_dir, _direction_idx in enumerate(dirs):
            env_idx = candidate_idx * d + local_dir
            out[env_idx] = direction_actions[local_dir]
    return out
```

Change `build_recorded_actions_tensor` to accept and forward `action_dim: int = 6`:

```python
def build_recorded_actions_tensor(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
    action_dim: int = 6,
) -> np.ndarray:
    return stacked_recorded_actions_for_structure(
        dataset,
        structure_idx=structure_idx,
        num_directions=num_directions,
        num_candidates=num_candidates,
        direction_indices=direction_indices,
        include_excluded=include_excluded,
        action_dim=action_dim,
    )
```

(If `build_recorded_actions_tensor` already has a different body than a thin wrapper, keep its existing body and just add the `action_dim: int = 6` parameter, forwarding it to its inner `stacked_recorded_actions_for_structure` call instead of replacing the whole function — check the current implementation before editing.)

Change `replay_batched_sysid_structure` (around line 1376) to accept `action_dim: int = 6` and forward it to `build_recorded_actions_tensor`:

```python
def replay_batched_sysid_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[SysIdReplayCandidate],
    num_directions: int,
    seed: int | None = None,
    build_env_fn: Callable[..., Any],
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
    use_snapshot: bool = False,
    use_oracle_params: bool = True,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
    action_dim: int = 6,
) -> BatchedSysIdReplayCollectors:
    ...
    recorded_actions = build_recorded_actions_tensor(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=d,
        num_candidates=num_candidates,
        direction_indices=dirs,
        include_excluded=bool(include_excluded),
        action_dim=action_dim,
    )
    ...
```

(Keep every other line in `replay_batched_sysid_structure` unchanged; only add the parameter and thread it through this one call site.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_batched_sysid_replay.py -q`
Expected: all PASS, including new `action_dim` tests and every pre-existing 6D test (default unchanged).

- [ ] **Step 5: Regression check**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_gym/tests/test_batched_sysid_mmd_grid* apple_pick_gym/tests/test_batched_sysid_collect.py -q` (adjust glob to actual filenames in `apple_pick_gym/tests/` for MMD-grid coverage)
Expected: PASS (MMD/CMA callers still pass no `action_dim`, still get `6`).

- [ ] **Step 6: Commit**

```bash
git add apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py apple_pick_gym/tests/test_batched_sysid_replay.py
git commit -m "feat: add additive action_dim param to batched sysid replay helpers"
```

---

### Task 7: `robot_replay/pack_vic_pose_actions.py` — 6D→19D action packer

**Files:**
- Create: `robot_replay/pack_vic_pose_actions.py`
- Test: `robot_replay/tests/test_pack_vic_pose_actions.py` (new; create `robot_replay/tests/__init__.py` if the directory has no test package yet)

**Interfaces:**
- Consumes: a converted `batched_sysid_v1` dataset directory (`manifest.json` + `episodes/*.parquet`, `action` column shape `(n, 6)`, `target_pose_4x4` present per bit-2 export — see `apple_pick_sim/system_id/real_to_batched_sysid.py`).
- Produces: `pack_vic_pose_actions(src_dir: Path, dst_dir: Path, *, kp: tuple[float, ...], kd: tuple[float, ...], overwrite: bool = False) -> dict` — copies the dataset to `dst_dir`, rewrites every episode's `action` column to `(n, 19)` `[pos(3), quat_wxyz(4), Kp(6), Kd(6)]` built from that row's `target_pose_4x4` (falling back to `tcp_pose_4x4` if `target_pose_4x4` is absent) plus the constant `kp`/`kd`, and returns stats (`{"episodes": int, "frames": int}`). A CLI wraps this with `--dataset-in`, `--dataset-out`, `--kp` (6 floats), `--kd` (6 floats), `--overwrite`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for robot_replay/pack_vic_pose_actions.py (6D twist -> 19D pose+gains action)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_replay.pack_vic_pose_actions import pack_vic_pose_actions


def _write_minimal_dataset(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "episodes").mkdir(parents=True)
    (src / "manifest.json").write_text('{"schema_version": "batched_sysid_v1"}')

    n = 3
    target_pose = np.tile(np.eye(4, dtype=np.float32).reshape(-1), (n, 1))
    target_pose[:, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # x translation per row varies below
    for i in range(n):
        target_pose[i, 3] = 0.1 + 0.01 * i
        target_pose[i, 7] = 0.2
        target_pose[i, 11] = 0.3

    table = pa.table(
        {
            "action": pa.array(
                [[float(j) for j in range(6)] for _ in range(n)],
                type=pa.list_(pa.float32(), 6),
            ),
            "target_pose_4x4": pa.array(
                target_pose.tolist(), type=pa.list_(pa.float32(), 16)
            ),
            "step_idx": pa.array(list(range(n)), type=pa.int64()),
        }
    )
    pq.write_table(table, src / "episodes" / "s00_d00.parquet")
    return src


def test_pack_vic_pose_actions_writes_19_wide_action(tmp_path):
    src = _write_minimal_dataset(tmp_path)
    dst = tmp_path / "dst"
    stats = pack_vic_pose_actions(
        src, dst, kp=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), kd=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )
    assert stats["episodes"] == 1
    assert stats["frames"] == 3

    out = pq.read_table(dst / "episodes" / "s00_d00.parquet")
    actions = np.stack([out.column("action")[i].as_py() for i in range(out.num_rows)])
    assert actions.shape == (3, 19)
    np.testing.assert_allclose(actions[0, 0:3], [0.1, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(actions[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(actions[0, 7:13], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(actions[0, 13:19], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)


def test_pack_vic_pose_actions_refuses_existing_dst_without_overwrite(tmp_path):
    src = _write_minimal_dataset(tmp_path)
    dst = tmp_path / "dst"
    pack_vic_pose_actions(src, dst, kp=(1.0,) * 6, kd=(1.0,) * 6)
    with pytest.raises(FileExistsError):
        pack_vic_pose_actions(src, dst, kp=(1.0,) * 6, kd=(1.0,) * 6)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --env-file pytest.env python -m pytest robot_replay/tests/test_pack_vic_pose_actions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'robot_replay.pack_vic_pose_actions'`.

- [ ] **Step 3: Implement**

```python
#!/usr/bin/env python3
"""Rewrite a converted ``batched_sysid_v1`` dataset's ``action`` column from a
6D EE twist to a 19D ``vic_pose`` action: ``[pos(3), quat_wxyz(4), Kp(6), Kd(6)]``.

Position/orientation come from each frame's ``target_pose_4x4`` (falling back
to ``tcp_pose_4x4`` when absent); ``Kp``/``Kd`` are constant across the episode,
supplied by the caller (temporary until real parquets ship per-frame gains).

Example::

    uv run python robot_replay/pack_vic_pose_actions.py \\
      --dataset-in /tmp/real_batched_s02_d00 \\
      --dataset-out /tmp/real_batched_s02_d00_vic_pose \\
      --kp 800 800 800 40 40 40 \\
      --kd 80 80 80 4 4 4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_POS_DIM = 3
_QUAT_DIM = 4
_GAIN_DIM = 6
_ACTION_DIM = _POS_DIM + _QUAT_DIM + 2 * _GAIN_DIM  # 19


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation -> unit quaternion (w, x, y, z); orthonormalized via SVD."""
    u, _, vt = np.linalg.svd(rot.astype(np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return (q / n) if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _pose_action_from_flat16(flat16, kp: tuple[float, ...], kd: tuple[float, ...]) -> list[float]:
    m = np.asarray(flat16, dtype=np.float64).reshape(4, 4)
    pos = m[:3, 3].tolist()
    quat_wxyz = _rotmat_to_quat_wxyz(m[:3, :3]).tolist()
    return [*pos, *quat_wxyz, *kp, *kd]


def pack_vic_pose_actions(
    src_dir: Path,
    dst_dir: Path,
    *,
    kp: tuple[float, ...],
    kd: tuple[float, ...],
    overwrite: bool = False,
) -> dict:
    """Copy ``src_dir`` to ``dst_dir`` with 19D ``vic_pose`` actions; return ``{episodes, frames}``."""
    if len(kp) != _GAIN_DIM or len(kd) != _GAIN_DIM:
        raise ValueError(f"kp/kd must each have {_GAIN_DIM} entries")

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_dir} already exists (pass --overwrite to replace)")
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    episode_paths = sorted((dst_dir / "episodes").glob("*.parquet"))
    n_episodes = 0
    n_frames = 0
    for path in episode_paths:
        table = pq.read_table(path)
        if "target_pose_4x4" in table.column_names:
            pose_col = "target_pose_4x4"
        elif "tcp_pose_4x4" in table.column_names:
            pose_col = "tcp_pose_4x4"
        else:
            raise ValueError(f"{path}: missing target_pose_4x4 and tcp_pose_4x4")

        rows = [
            _pose_action_from_flat16(table.column(pose_col)[i].as_py(), kp, kd)
            for i in range(table.num_rows)
        ]
        action_col = pa.array(rows, type=pa.list_(pa.float32(), _ACTION_DIM))
        idx = table.column_names.index("action")
        new_table = table.set_column(idx, "action", action_col)
        pq.write_table(new_table, path, use_dictionary=False)
        n_episodes += 1
        n_frames += table.num_rows

    manifest_path = dst_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["action_dim"] = _ACTION_DIM
        manifest["action_layout"] = "vic_pose_v1"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    return {"episodes": n_episodes, "frames": n_frames}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-in", type=Path, required=True)
    parser.add_argument("--dataset-out", type=Path, required=True)
    parser.add_argument("--kp", type=float, nargs=6, required=True, metavar="Kp")
    parser.add_argument("--kd", type=float, nargs=6, required=True, metavar="Kd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    stats = pack_vic_pose_actions(
        args.dataset_in,
        args.dataset_out,
        kp=tuple(args.kp),
        kd=tuple(args.kd),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --env-file pytest.env python -m pytest robot_replay/tests/test_pack_vic_pose_actions.py -q`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add robot_replay/pack_vic_pose_actions.py robot_replay/tests/test_pack_vic_pose_actions.py
git commit -m "feat: add 6D-to-19D vic_pose action packer for real replay"
```

---

### Task 8: `example_replay_real_batched.py` — `--controller-mode vic_pose`

**Files:**
- Modify: `robot_replay/example_replay_real_batched.py`
- Modify: `robot_replay/README.md` (document the new flag + packer step, per `.cursor/rules/readme-runtime-verification.mdc`)

**Interfaces:**
- Consumes: `pack_vic_pose_actions` (Task 7, run by the user beforehand); `ControllerConfig(mode="vic_pose", action_dim=19)` (Task 1); `replay_batched_sysid_structure(..., action_dim=19)` (Task 6).
- Produces: `--controller-mode {vic,vic_pose}` CLI flag (default `vic`); when `vic_pose`, `_test_sim_config` builds a `vic_pose` controller and the replay call passes `action_dim=19`.

- [ ] **Step 1: Confirm current default behavior is covered by an existing test**

Run: `uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_real_to_batched_sysid.py -q` (or whichever test currently exercises this example's config builder — check `robot_replay/README.md` "Testing" section for the pinned command) to have a documented green baseline before touching the file.

- [ ] **Step 2: Implement**

In `robot_replay/example_replay_real_batched.py`:

1. Add a module-level default and thread a `controller_mode` parameter through `_test_sim_config` (near the other `_DEFAULT_*` constants, around line 70):

```python
_DEFAULT_CONTROLLER_MODE = "vic"
```

2. In `_test_sim_config(...)` (starting around line 124), add a `controller_mode: str = _DEFAULT_CONTROLLER_MODE` parameter, and change the controller construction (currently `dataclasses.replace(gym_cfg.controller, mode="vic", linear_speed=1.0, angular_speed=1.0)`, around line 146) to:

```python
    controller = dataclasses.replace(
        gym_cfg.controller,
        mode=controller_mode,
        action_dim=19 if controller_mode == "vic_pose" else gym_cfg.controller.action_dim,
        linear_speed=1.0,
        angular_speed=1.0,
    )
```

3. In `_build_env_fn(...)` and `build_env_fn(...)` (around lines 193-244), add `controller_mode: str = _DEFAULT_CONTROLLER_MODE` to `_build_env_fn`'s signature and forward it into the `_test_sim_config(...)` call inside `build_env_fn`.

4. In `_make_parser()` (around line 289), add:

```python
    p.add_argument(
        "--controller-mode",
        choices=["vic", "vic_pose"],
        default=_DEFAULT_CONTROLLER_MODE,
        help="'vic' (recorded EE twist, 6D action) or 'vic_pose' (pose+gains, 19D action; "
        "dataset must already carry 19D actions, e.g. via pack_vic_pose_actions.py).",
    )
```

5. Find the `main()` (or equivalent) call site that builds `_build_env_fn(...)` and calls `replay_batched_sysid_structure(...)`; pass `controller_mode=args.controller_mode` into `_build_env_fn`, and pass `action_dim=19 if args.controller_mode == "vic_pose" else 6` into the `replay_batched_sysid_structure(...)` call.

- [ ] **Step 3: Manual smoke (documented, not auto-asserted — matches this repo's existing real-replay smoke pattern)**

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00_action.parquet \
  --dataset-out /tmp/real_batched_s02_d00 --overwrite

uv run python robot_replay/pack_vic_pose_actions.py \
  --dataset-in /tmp/real_batched_s02_d00 \
  --dataset-out /tmp/real_batched_s02_d00_vic_pose \
  --kp 800 800 800 40 40 40 \
  --kd 80 80 80 4 4 4 \
  --overwrite

uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00_vic_pose \
  --controller-mode vic_pose \
  --viewer null --max-frames 5 \
  --settle-substeps 50 --post-grasp-settle-substeps 10
```

Expected: exits 0, prints `OK: TCP moved under recorded actions` (existing exit check in the file), confirming `vic_pose` end to end. If it fails, check the quaternion-order note left in Task 4 (Step 3) first.

- [ ] **Step 4: Regression check (default mode unchanged)**

```bash
uv run python robot_replay/example_replay_real_batched.py \
  --dataset /tmp/real_batched_s02_d00 --viewer null --max-frames 5 \
  --settle-substeps 50 --post-grasp-settle-substeps 10
```

Expected: same output as before this task (default `--controller-mode vic`, 6D actions, twist path).

- [ ] **Step 5: Update README**

In `robot_replay/README.md`, add a short subsection documenting `pack_vic_pose_actions.py` and `--controller-mode vic_pose`, next to the existing real-replay CLI docs (mirror the existing `s02-d00_action.parquet` → convert → replay block style).

- [ ] **Step 6: Commit**

```bash
git add robot_replay/example_replay_real_batched.py robot_replay/README.md
git commit -m "feat: add --controller-mode vic_pose to real replay example"
```

---

## Self-review notes (already applied above)

- **Spec coverage:** Action contract (Task 4), wrench law (Tasks 2-3), config/mode (Task 1), sim wiring (Task 5), first-caller-only scope (Tasks 7-8), shared-helper additive change (Task 6) — all sections of the design doc map to a task.
- **Quat convention risk:** flagged explicitly in Task 4 Step 3 and Task 8 Step 3 because `_target_rot_wp`'s exact Warp storage order wasn't independently re-verified line-by-line during planning; the plan tells the implementer exactly how to check and fix it in one place rather than leaving it a silent bug.
- **Twist-path regression:** every task with a wrench/controller/sim-step change has an explicit "Regression check" step re-running the pre-existing `mode="vic"` test suite.
- **No placeholders:** every step has runnable code or an exact command; no "TODO"/"handle appropriately" language.
