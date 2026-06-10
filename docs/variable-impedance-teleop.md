# Variable-impedance teleop (post-grasp, dynamic arm)

Design note for **grasped-apple pulling** with a **variable-impedance controller (VIC)** on the FR3, using the existing staggered **MuJoCo + VBD** coupling. Complements **`docs/mujoco-vbd-coupling-architecture.md`** (staggered loop) and **`docs/ROADMAP.md`** (M2 \(\pi_{\mathrm{exp}}\) / FD modes).

**Status:** Implemented for physics + `example_coupled_fruiting.py` (`--dynamic-arm`, `--vic`). M2.1 Gym remains **kinematic** direct-joint teleop; see `docs/vic-implementation.md`.

---

## Scenario

1. **Settle** the plant with `fix_to_apple=False` (optional warmup substeps).
2. **Weld** — rebuild with `GripperProxyConfig(fix_to_apple=True)` and seed from settled state (`settle_then_weld.py`).
3. **Teleop** while the arm is **dynamic** (`robot_kinematic_mode=False`): user commands TCP motion; stem load feeds back through the lagged harvest path; **applied** VIC wrench shapes compliance.

**“Tight grasp” in sim** is a **FIXED** proxy↔apple weld plus **kinematic co-teleport** of the apple from the TCP each substep — not finger contact or slip. See **`docs/mujoco-vbd-coupling-architecture.md`** §2.2.

---

## Control law (total wrench at TCP)

Each MuJoCo substep, before `mj_solver.step`:

\[
\mathbf{w}_{\mathrm{total}} = \mathbf{w}_{\mathrm{transferred}} + \mathbf{w}_{\mathrm{applied}}
\]

| Symbol | Source | When computed |
|--------|--------|----------------|
| \(\mathbf{w}_{\mathrm{transferred}}\) | `proxy_forces[tcp]` after previous substep’s **stem harvest** (`harvest_stem_tension_for_tcp`) plus optional **explicit apple weight** (`explicit_load.py`) | **Lagged** one substep (M1 design) |
| \(\mathbf{w}_{\mathrm{applied}}\) | VIC, e.g. \(\mathbf{F} = K(\mathbf{x}_{\mathrm{des}}-\mathbf{x}) + D(\mathbf{v}_{\mathrm{des}}-\mathbf{v})\) in world frame at TCP | **Fresh** each MuJoCo substep |
| \(\mathbf{w}_{\mathrm{total}}\) | Sum | Written to `coupling_forces_cache[tcp]` → `body_f[tcp]` → MuJoCo `xfrc_applied` |

**Harvest output must stay plant-only:** `harvest_stem_tension_for_tcp` writes **only** \(\mathbf{w}_{\mathrm{transferred}}\) into `proxy_forces` for the **next** lag step. Do **not** store \(\mathbf{w}_{\mathrm{applied}}\) in `proxy_forces`.

Use the **same** `coupling_forces_cache` for mirror correction (`launch_mirror_robot_to_proxy*`) so proxy velocity correction matches what the arm felt.

Spatial layout matches **`apple_pick_sim/coupled_fruiting/apply_wrench.py`**: force `[:3]`, torque `[3:6]`, world frame.

---

## “Move accordingly”: MuJoCo integration (yes)

**Writing \(\mathbf{w}_{\mathrm{total}}\) to TCP `body_f` and calling `mj_solver.step` is the intended dynamic path.** Newton’s `SolverMuJoCo` copies `state.body_f` into MuJoCo `xfrc_applied` (`apply_mjc_body_f_kernel` in `newton/newton/_src/solvers/mujoco/`). MuJoCo integrates the articulated arm under **external wrench + joint actuators**.

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

Implementation hook: **`CoupledFruitingScene._mujoco_and_sync_proxy`** and **`MegaCoupledFruitingScene._mujoco_and_sync_proxy`** (dynamic branch only).

### Required mode switch

| Setting | Kinematic (today’s mega keyboard / M2.1 Gym) | Dynamic (VIC) |
|---------|-----------------------------------------------|-----------------|
| `robot_kinematic_mode` | `True` (builders default) | **`False`** |
| Teleop | `Fr3EEDirectJointController` + `apply_fr3_ee_teleop_direct` | `Fr3EEVelocityController` + `apply_fr3_ee_teleop` (+ VIC sum) |
| `body_f[tcp]` | Zeroed / ignored | **Active** |

While `robot_kinematic_mode=True`, `coupling_forces_cache.zero_()` runs and the arm does not integrate stem load — fine for FD smoke and pose-exact ghost columns, **not** for VIC.

---

## Teleop vs joint PD (hybrid)

If `apply_ik_to_mujoco_control` still sets `joint_target_pos` / `joint_target_vel`, MuJoCo **position actuators** compete with \(\mathbf{w}_{\mathrm{total}}\).

| Mode | Behavior |
|------|----------|
| Wrench + **stiff** PD | Little deflection; stem load fights actuators |
| Wrench + **soft** PD | Closer to flange impedance; tune `joint_target_ke` / `joint_target_kd` |
| **Wrench-only** | Hold `joint_target_pos ≈ joint_q`, teleop only moves `target_tf` for \(\mathbf{w}_{\mathrm{applied}}\) |

Typical post-grasp VIC (this repo’s `--vic` default): teleop advances `target_tf` only; \(\mathbf{w}_{\mathrm{applied}} = K\Delta\mathbf{x} + D\Delta\mathbf{v}\) is mapped to **joint torques** (`joint_f`) via dynamically-consistent control; **joint PD off** (`joint_target_ke/kd = 0`, targets held at current `joint_q`). Plant loads stay on TCP `body_f`.

---

## Code map

| Responsibility | Module / symbol |
|----------------|-----------------|
| Lagged plant wrench | `proxy_coupling.harvest_stem_tension_for_tcp`, `explicit_load.explicit_apple_wrench_for_stem_harvest` |
| Apply total wrench | `apply_wrench._apply_spatial_wrench_to_body_f`, `_add_tcp_spatial_wrench_inplace` |
| Dynamic substep gate | `scene.py` → `_mujoco_robot_substep_prefix`; `--vic` uses `vic_joint_torques.apply_vic_joint_torques_to_scene` |
| TCP target + IK | `robot/fr3_robot/controllers/ee_velocity.py` |
| VIC wrench law | `robot/fr3_robot/controllers/ee_impedance.py` — `Fr3EEImpedanceController` |
| VIC → joint torques | `coupled_fruiting/vic_joint_torques.py` — `launch_apply_vic_joint_torques` |
| Settle → weld | `coupled_fruiting/settle_then_weld.py` |
| Example entry | `examples/example_coupled_fruiting.py` — `--dynamic-arm`, `--vic`, `--vic-*-k/d` |

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
| `test_ee_impedance.py` | Unit tests for K/D wrench law |
| `test_dynamic_tcp_wrench_moves_arm` | Dynamic mode integrates lagged TCP wrench (velocity) |
| `test_vic_stem_deflection_under_load` | Welded pull: VIC tracking error ≥ kinematic baseline |
| `test_harvest_excludes_applied_wrench` | VIC adds to cache only, not `proxy_forces` |

Run gate:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
  ../apple_pick_sim/tests/test_ee_impedance.py \
  ../apple_pick_sim/tests/test_vic_dynamic.py -q -p no:launch_testing
```

See also `docs/vic-implementation.md`.

---

## Stability notes

- One-substep **lag** on \(\mathbf{w}_{\mathrm{transferred}}\) is intentional (`docs/mujoco-vbd-coupling-architecture.md` §4.2).
- High \(K\) + `stem_coupling_gain=1` + small `SUB_DT` can ring — use \(D\), `stem_force_cap_N` / `stem_torque_cap_Nm`, or `stem_coupling_gain < 1`.
- Dynamic-mode regression: `test_vic_dynamic.py`, `test_ee_impedance.py` (see **Tests** above).

---

## Related docs

- `docs/mujoco-vbd-coupling-architecture.md` — ownership, stem harvest, explicit apple weight
- `docs/mega-coupled-cable-implementation.md` — mega `fd_ghost` columns (kinematic default)
- `docs/ROADMAP.md` — M2.0 ADR, VIC / \(\pi_{\mathrm{exp}}\) FD modes
