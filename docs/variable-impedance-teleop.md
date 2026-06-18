# Variable-impedance teleop (post-grasp, dynamic arm)

Design and implementation reference for **grasped-apple pulling** with a **variable-impedance controller (VIC)** on the FR3, using the existing staggered **MuJoCo + VBD** coupling. Complements **`docs/mujoco-vbd-coupling-architecture.md`** (staggered loop) and **`docs/ROADMAP.md`** (M2 \(\pi_{\mathrm{exp}}\) / FD modes).

**Status:** Implemented for physics + `example_coupled_fruiting.py` (`--dynamic-arm`, `--vic`). M2.1 Gym remains **kinematic** direct-joint teleop.

---

## Scenario

1. **Settle** the plant with `fix_to_apple=False` (optional warmup substeps).
2. **Weld** — rebuild with `GripperProxyConfig(fix_to_apple=True)` and seed from settled state (`settle_then_weld.py`).
3. **Teleop** while the arm is **dynamic** (`robot_kinematic_mode=False`): user commands TCP motion; stem load feeds back through the lagged harvest path; **applied** VIC wrench shapes compliance.

**"Tight grasp" in sim** is a **FIXED** proxy↔apple weld plus **kinematic co-teleport** of the apple from the TCP each substep — not finger contact or slip. See **`docs/mujoco-vbd-coupling-architecture.md`** §2.2.

---

## Control law (total wrench at TCP)

Each MuJoCo substep, before `mj_solver.step`:

\[
\mathbf{w}_{\mathrm{total}} = \mathbf{w}_{\mathrm{transferred}} + \mathbf{w}_{\mathrm{applied}}
\]

| Symbol | Source | When computed |
|--------|--------|----------------|
| \(\mathbf{w}_{\mathrm{transferred}}\) | `proxy_forces[tcp]` after previous substep's **stem harvest** (`harvest_stem_tension_for_tcp`) plus optional **explicit apple weight** (`explicit_load.py`) | **Lagged** one substep (M1 design) |
| \(\mathbf{w}_{\mathrm{applied}}\) | VIC impedance law (see below) using teleop `target_tf` and live TCP pose/twist | **Fresh** each MuJoCo substep |
| \(\mathbf{w}_{\mathrm{total}}\) | Sum | Written to `coupling_forces_cache[tcp]` → `body_f[tcp]` → MuJoCo `xfrc_applied` |

**Harvest output must stay plant-only:** `harvest_stem_tension_for_tcp` writes **only** \(\mathbf{w}_{\mathrm{transferred}}\) into `proxy_forces` for the **next** lag step. Do **not** store \(\mathbf{w}_{\mathrm{applied}}\) in `proxy_forces`.

Use the **same** `coupling_forces_cache` for mirror correction (`launch_mirror_robot_to_proxy*`) so proxy velocity correction matches what the arm felt.

Spatial layout matches **`apple_pick_sim/coupled_fruiting/apply_wrench.py`**: force `[:3]`, torque `[3:6]`, world frame.

### Impedance law (isotropic gains)

- \(\mathbf{F} = K_p (\mathbf{p}_{\mathrm{des}} - \mathbf{p}) + D_p (\mathbf{v}_{\mathrm{des}} - \mathbf{v})\)
- \(\boldsymbol{\tau} = K_r \mathbf{e}_R + D_r (\boldsymbol{\omega}_{\mathrm{des}} - \boldsymbol{\omega})\) where \(\mathbf{e}_R\) is axis-angle from \(\mathbf{q}_{\mathrm{des}} \otimes \mathbf{q}^{-1}\).

---

## "Move accordingly": MuJoCo integration (yes)

**Writing \(\mathbf{w}_{\mathrm{total}}\) to TCP `body_f` and calling `mj_solver.step` is the intended dynamic path.** Newton's `SolverMuJoCo` copies `state.body_f` into MuJoCo `xfrc_applied` (`apply_mjc_body_f_kernel` in `newton/newton/_src/solvers/mujoco/`). MuJoCo integrates the articulated arm under **external wrench + joint actuators**.

You do **not** need a hand-rolled \(\dot{\mathbf{x}} = M^{-1}\mathbf{F}\) unless you want a **kinematic admittance** loop (advance `target_tf` from wrench) instead of physics-based compliance.

### Per-substep sequence (`coupled_substep`)

```text
(1) clear_forces on robot
(2) w_coupling ← proxy_forces[tcp]           # lagged plant → arm
(3) w_applied  ← vic(state, target, K, D)    # controller
(4) coupling_forces_cache[tcp] ← w_coupling + w_applied
(5) _apply_spatial_wrench_to_body_f(robot_state_0, tcp, cache)
(6) mj_solver.step(...)
(7) mirror TCP → proxy (+ apple if fix_to_apple)
(8) VBD substep
(9) harvest → proxy_forces (plant only, for step N+1)
```

Implementation hook: **`CoupledFruitingScene._mujoco_and_sync_proxy`** (dynamic branch only).

### Required mode switch

| Setting | Kinematic (M2.1 Gym) | Dynamic (VIC) |
|---------|----------------------------------------------|-----------------|
| `robot_kinematic_mode` | `True` (builders default) | **`False`** |
| Teleop | `Fr3EEDirectJointController` + `update_fr3_ee_teleop_direct` | `Fr3EEImpedanceController` (`scene.vic_controller`) + `update_fr3_ee_teleop` |
| `body_f[tcp]` | Zeroed / ignored | **Active** |

While `robot_kinematic_mode=True`, `coupling_forces_cache.zero_()` runs and the arm does not integrate stem load — fine for FD smoke and pose-exact ghost columns, **not** for VIC.

---

## Joint-torque mode (default with `--vic`)

VIC can apply the impedance law either as a **TCP body wrench** (`body_f`) or as **joint torques** (`joint_f` → MuJoCo `qfrc_applied`). Joint-torque mode is the default with `--vic`.

### Dynamically-consistent control

\(\tau = J^T \Lambda w + (I - J^T \Lambda J M^{-1})(M u_{\text{null}})\),
\(\Lambda = (J M^{-1} J^T + \lambda I)^{-1}\).

- **Plant / proxy coupling forces** stay on the TCP as `body_f` (unchanged from wrench-only mode).
- Joint 7 null-space default is forced to **0 rad** (gripper open posture).
- The controller runs **every MuJoCo substep** when `scene.vic_use_joint_torques = True`.

### Per-substep flow (joint-torque path)

1. `newton.eval_fk` / `eval_jacobian` / `eval_mass_matrix` on GPU buffers (`vic_jt_J_buf`, `vic_jt_H_buf`).
2. Slice TCP Jacobian rows and 7×7 mass matrix via `wp.to_torch`.
3. Compute task wrench on CPU from TCP pose (small read).
4. `compute_joint_torques_from_wrench_torch` — `torch.linalg.inv` and matrix multiply.
5. Write torques in-place: `wp.to_torch(control.joint_f)[:7] = tau`.

### Teleop target sync

`update_fr3_ee_teleop` advances `target_tf` only (`run_tcp_target_teleop_frame`), zeros `joint_target_ke`/`kd`, and holds `joint_target_pos` at simulated `joint_q`. VIC maps the impedance wrench to `control.joint_f` via dynamically-consistent joint torques; plant/proxy loads remain on TCP `body_f`.

### Enable manually

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

**PyTorch dependency:** Install from the Newton submodule (`cd newton && uv sync --extra torch-cu12`). Integration tests call `pytest.importorskip("torch")` and skip when PyTorch is absent.

---

## Teleop vs joint PD (hybrid)

If `apply_ik_to_mujoco_control` still sets `joint_target_pos` / `joint_target_vel`, MuJoCo **position actuators** compete with \(\mathbf{w}_{\mathrm{total}}\).

| Mode | Behavior |
|------|----------|
| Wrench + **stiff** PD | Little deflection; stem load fights actuators |
| Wrench + **soft** PD | Closer to flange impedance; tune `joint_target_ke` / `joint_target_kd` |
| **Wrench-only** | Hold `joint_target_pos ≈ joint_q`, teleop only moves `target_tf` for \(\mathbf{w}_{\mathrm{applied}}\) |

Typical post-grasp VIC (this repo's `--vic` default): teleop advances `target_tf` only; \(\mathbf{w}_{\mathrm{applied}} = K\Delta\mathbf{x} + D\Delta\mathbf{v}\) is mapped to **joint torques** (`joint_f`) via dynamically-consistent control; **joint PD off** (`joint_target_ke/kd = 0`, targets held at current `joint_q`). Plant loads stay on TCP `body_f`.

---

## Code map

| Responsibility | Module / symbol |
|----------------|-----------------|
| Impedance law | `robot/fr3_robot/controllers/ee_impedance.py` — `Fr3EEImpedanceController`, `ImpedanceGains` |
| IK teleop (kinematic mode) | `robot/fr3_robot/controllers/ee_velocity.py` — `Fr3EEVelocityController` |
| VIC → joint torques | `coupled_fruiting/vic_joint_torques.py` — `find_tcp_link_idx`, `compute_joint_torques_from_wrench_torch`, `launch_apply_vic_joint_torques` |
| VIC → spatial wrench (legacy) | `coupled_fruiting/vic_wrench.py` — `apply_vic_to_coupling_cache`, `launch_apply_vic_to_coupling_cache` |
| Wrench sum at TCP | `coupled_fruiting/apply_wrench.py` — `_apply_spatial_wrench_to_body_f`, `_add_tcp_spatial_wrench_inplace` |
| Dynamic substep gate | `coupled_fruiting/scene.py` → `_mujoco_robot_substep_prefix`; `--vic` uses `vic_joint_torques.apply_vic_joint_torques_to_scene` |
| Scene fields | `CoupledFruitingScene.vic_controller`, `vic_gains`, `vic_target_tf`, `vic_target_twist`, `vic_use_joint_torques` |
| Joint-torque arm setup | `robot/fr3_robot/setup.py` — `configure_vic_joint_torques_arm` (zeros PD, allocates J/H buffers, `joint_f`) |
| Lagged plant wrench | `coupled_fruiting/proxy_coupling.py` — `harvest_stem_tension_for_tcp`, `explicit_load.explicit_apple_wrench_for_stem_harvest` |
| Settle → weld | `coupled_fruiting/settle_then_weld.py` |
| Example CLI | `examples/example_coupled_fruiting.py` — `--dynamic-arm`, `--vic`, `--vic-*-k/d` |

**Not** the home for arm VIC: `explicit_load.py` (quasi-static apple weight in **harvest** only).

---

## FD / FIM (\(\pi_{\mathrm{exp}}\))

| Arm mode | Same action \(u\) ⇒ same \(x^{\mathrm{ee}}\)? | FD layout |
|----------|-----------------------------------------------|-----------| 
| Kinematic teleop | Yes (one leader arm, ghost plants) | **`fd_ghost`** OK for early smoke |
| Dynamic VIC | **No** — compliance depends on plant \(\theta\) | **`fd_mega_same_u`** or **`fd_replay`** (W × 1×1 coupled workers, same \(u_{0:H}\)) |

Rod **`stiffness_epsilon`** mega columns are **plant** \(\theta\) FD, orthogonal to arm \(K,D\) unless bundled in one action vector.

See **`docs/ROADMAP.md`** — *Dual envs, FD modes, SKRL*.

---

## Tests

| Test | Intent |
|------|--------|
| `test_ee_impedance.py` | Wrench law unit tests (zero at target, linear K, damping) |
| `test_dynamic_tcp_wrench_moves_arm` | Dynamic mode integrates lagged TCP wrench (velocity) |
| `test_vic_stem_deflection_under_load` | Welded pull: VIC tracking error ≥ kinematic baseline |
| `test_harvest_excludes_applied_wrench` | VIC adds to cache only, not `proxy_forces` |
| `test_vic_wrench_device.py` | GPU vs CPU parity, CUDA graph |
| `test_find_tcp_link_idx` | TCP link row matches `J @ qd ≈ body_qd` |
| `test_zero_wrench_no_task_torque` | Zero wrench → task torque zero; total = null-space only |
| `test_null_space_orthogonal_to_task` | `J @ tau_null ≈ 0` |
| `test_position_error_gives_correct_task_torque` | Analytic 1-DOF `J^T Λ f` |
| `test_torch_matches_numpy_reference` | PyTorch helper matches NumPy reference |
| `test_apply_vic_joint_torques_writes_joint_f` | Launcher writes non-zero `joint_f` |
| `test_launch_joint_torques_match_numpy_reference` | End-to-end launcher vs NumPy |
| `test_vic_joint_torques_moves_arm` | Integration: teleop peak TCP +X displacement `max_dx > 0.05` |

## How to verify

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_ee_impedance.py \
  apple_pick_sim/tests/test_vic_dynamic.py \
  apple_pick_sim/tests/test_vic_wrench_device.py \
  apple_pick_sim/tests/test_vic_joint_torques.py -q -m "not slow"
```

Example smoke (headless):

```bash
uv run python apple_pick_sim/examples/example_coupled_fruiting.py \
  --robot fr3 --fix-to-apple --dynamic-arm --vic --viewer null --num-frames 2 \
  --seed 0 --json apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json
```

Interactive post-grasp demo:

```bash
uv run python apple_pick_sim/examples/example_coupled_fruiting.py \
  --robot fr3 --fix-to-apple --dynamic-arm --vic --fr3-keyboard --viewer gl
```

---

## Stability notes

- One-substep **lag** on \(\mathbf{w}_{\mathrm{transferred}}\) is intentional (`docs/mujoco-vbd-coupling-architecture.md` §4.2).
- High \(K\) + `stem_coupling_gain=1` + small `SUB_DT` can ring — use \(D\), `stem_force_cap_N` / `stem_torque_cap_Nm`, or `stem_coupling_gain < 1`.
- Dynamic-mode regression: tests listed above.

---

## Related docs

- `docs/mujoco-vbd-coupling-architecture.md` — ownership, stem harvest, explicit apple weight
- `docs/ROADMAP.md` — M2.0 ADR, VIC / \(\pi_{\mathrm{exp}}\) FD modes
