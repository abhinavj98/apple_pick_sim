# VIC joint torques implementation

## Behavior summary

Post-grasp VIC can apply the impedance law either as a **TCP body wrench** (`body_f`) or as **joint torques** (`joint_f` → MuJoCo `qfrc_applied`). In joint-torque mode:

- **VIC command** is mapped with dynamically consistent control:
  \(\tau = J^T \Lambda w + (I - J^T \Lambda J M^{-1})(M u_{\text{null}})\),
  \(\Lambda = (J M^{-1} J^T + \lambda I)^{-1}\).
- **Plant / proxy coupling forces** stay on the TCP as `body_f` (unchanged from wrench-only mode).

The controller runs **every MuJoCo substep** when `scene.vic_use_joint_torques = True`.

Joint 7 null-space default is forced to **0 rad** (gripper open posture).

## Code map

| Module | Role |
|--------|------|
| [`apple_pick_sim/coupled_fruiting/vic_joint_torques.py`](../apple_pick_sim/coupled_fruiting/vic_joint_torques.py) | `find_tcp_link_idx`, NumPy/PyTorch torque math, `launch_apply_vic_joint_torques` |
| [`apple_pick_sim/coupled_fruiting/scene.py`](../apple_pick_sim/coupled_fruiting/scene.py) | Branches on `vic_use_joint_torques`: VIC → `joint_f`, plant → `body_f` |
| [`apple_pick_sim/robot/fr3_robot/setup.py`](../apple_pick_sim/robot/fr3_robot/setup.py) | `configure_vic_joint_torques_arm` — zeros joint PD, allocates J/H buffers |
| [`apple_pick_sim/coupled_fruiting/vic_wrench.py`](../apple_pick_sim/coupled_fruiting/vic_wrench.py) | PD wrench law (reused via `Fr3EEImpedanceController.compute_applied_wrench`) |

### Per-substep flow

1. `newton.eval_fk` / `eval_jacobian` / `eval_mass_matrix` on GPU buffers (`vic_jt_J_buf`, `vic_jt_H_buf`).
2. Slice TCP Jacobian rows and 7×7 mass matrix via `wp.to_torch`.
3. Compute task wrench on CPU from TCP pose (small read).
4. `compute_joint_torques_from_wrench_torch` — `torch.linalg.inv` and matrix multiply.
5. Write torques in-place: `wp.to_torch(control.joint_f)[:7] = tau`.

## Tests

| Test | What it checks |
|------|----------------|
| `test_find_tcp_link_idx` | TCP link row matches `J @ qd ≈ body_qd` |
| `test_zero_wrench_no_task_torque` | Zero wrench → task torque zero; total = null-space only |
| `test_null_space_orthogonal_to_task` | `J @ tau_null ≈ 0` |
| `test_position_error_gives_correct_task_torque` | Analytic 1-DOF `J^T Λ f` |
| `test_torch_matches_numpy_reference` | PyTorch helper matches NumPy reference |
| `test_apply_vic_joint_torques_writes_joint_f` | Launcher writes non-zero `joint_f` |
| `test_launch_joint_torques_match_numpy_reference` | End-to-end launcher vs NumPy |
| `test_vic_joint_torques_moves_arm` | Integration: teleop peak TCP +X displacement `max_dx > 0.05` |

Regression: `test_vic_dynamic.py` (wrench-only path unchanged).

## How to verify

From repository root:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_vic_joint_torques.py \
  ../apple_pick_sim/tests/test_vic_dynamic.py -q -p no:launch_testing
```

PyTorch is an **optional** dependency. Install from the Newton submodule:

```bash
cd newton && uv sync --extra torch-cu12
```

Integration tests call `pytest.importorskip("torch")` and skip when PyTorch is absent.

## Enable in simulation

`example_coupled_fruiting.py --vic` sets this up automatically. Manual wiring:

```python
scene.vic_use_joint_torques = True
scene.vic_controller = Fr3EEImpedanceController()
scene.vic_gains = ImpedanceGains(...)
fr3_robot.configure_vic_joint_torques_arm(
    scene.robot_model, scene.robot_state_0, scene.robot_control,
    scene.mj_solver, scene=scene,
)
```

Teleop sets `vic_target_tf` / `vic_target_twist` each frame; substeps call `apply_vic_joint_torques_to_scene`.
