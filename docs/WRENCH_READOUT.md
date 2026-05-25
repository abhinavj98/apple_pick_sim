# Fixed-Joint Wrench Readout — Reference

This document explains what `fixed_joint_wrenches_child_com_vbd` (and the
underlying `SolverVBD.gather_joint_wrench_child_com`) actually computes, which
reference frame the result is in, and what sign convention is used.  It then
derives the analytic equilibrium expectations that the tests in
`apple_pick_sim/tests/test_wrench_equilibrium.py` assert.

---

## 1. What the API returns

```python
wrenches = fixed_joint_wrenches_child_com_vbd(
    model, solver,
    body_q=state_0.body_q.numpy(),   # post-step transforms
    body_q_prev=q_prev,              # pre-step transforms (same macro-step)
    dt=sim_dt,
    joint_pairs=list(scene.fruiting_fixed_joints),  # explicit fruiting joints (recommended)
)
# Or omit joint_pairs to fall back to iter_fixed_joint_indices(model) (label heuristic).
```

Each element is a `FixedJointWrenchRecord` with two fields:

| Field | Type | Meaning |
|---|---|---|
| `force_world` | `np.ndarray (3,) float32` | Linear force on the **child** body from the joint [N], **world frame** |
| `torque_at_child_com_world` | `np.ndarray (3,) float32` | Total torque on the **child** about its COM [N·m], **world frame** |

For scenes from ``apple_pick_sim.fruiting_system.generate_scene``, pass
``joint_pairs=list(scene.fruiting_fixed_joints)`` (or use
``apple_pick_sim.fruiting_system.measure_fruiting_forces``). The legacy helper
``apple_pick_sim.vbd_fixed_joint_wrenches.iter_fixed_joint_indices`` matches
today’s fruiting labels (``joint_*`` prefix + FIXED) but is not ideal once extra
fixed joints exist (e.g. M1 proxies).

---

## 2. Reference frame

The simulation uses a **world frame** with:

| Axis | Direction |
|---|---|
| X | right |
| Y | forward |
| **Z** | **up** |

Gravity is `(0, 0, −9.81)` m/s² (set in `_build_scene` via
`model.set_gravity((0.0, 0.0, -9.81))`).  All wrench components are expressed
in this world frame.

---

## 3. Sign convention — "force on the child from the joint"

The returned `force_world` is the constraint force that the joint exerts **on
the child body**.  For a body hanging under gravity the joint holds it up, so
`force_world[Z] > 0` (upward).

### Kernel derivation

`gather_joint_wrench_child_com` calls `evaluate_joint_force_hessian` with
`body_index = joint_child[j]`.  Inside
`evaluate_linear_constraint_force_hessian`
(`newton/_src/solvers/vbd/rigid_vbd_kernels.py`, line ~449):

```
C = x_c − x_p              # constraint: child anchor − parent anchor
f_attachment = k·C + λ + damping_term

force = −f_attachment       # child side (is_parent = False)
r     = x_c_world − COM_child_world
torque = cross(r, force)    # linear lever-arm contribution
```

The `torque_at_child_com_world` field is the **sum** of:
- the lever-arm torque above, and
- the angular-constraint contribution from `evaluate_angular_constraint_force_hessian`.

At equilibrium `C ≈ 0`, so `f_attachment ≈ λ_lin` (the accumulated
augmented-Lagrangian multiplier), and `force = −λ_lin`.  The Lagrange
multiplier converges to the value that satisfies the static balance, making
`force_world[Z]` positive for a hanging body.

> **Summary**: `force_world` is the force that the **parent body pushes up on
> the child**.  Newton's 3rd law: the equal-and-opposite reaction is the force
> the child pushes down on the parent.

---

## 4. Free-body balance on the apple body

The **apple** is the terminal body in the chain.  It has exactly **one**
constraint joint: `joint_stem_apple` (type FIXED, child = apple_body).

Forces on apple:

| Source | Z-component |
|---|---|
| Gravity | `−m_apple · g` |
| `joint_stem_apple.force_world[Z]` | `+F_Z` |
| Ground contact | 0 (apple hangs in the air) |

Newton's second law at quasi-static rest (`|a| ≈ 0`):

```
ΣF_Z = 0
F_Z − m_apple · g = 0
⟹  force_world[Z]  =  m_apple · g            (Eq. 1)
```

For the torque balance about the apple's COM (`|α| ≈ 0`):

```
Στ_COM = 0
torque_at_child_com_world  =  0               (Eq. 2)
```

Equation 2 holds because gravity acts at the COM (zero moment arm), leaving
the joint torque as the only source.  For a straight vertical chain the
linear lever-arm term is zero because `r ∥ F`, and the angular-constraint
term is also zero because no external torque is applied to the apple.

---

## 5. Subtree cut theorem for serial chains

Consider the full chain  
`primary → secondary → spur → stem → apple`.  
Label the four FIXED inter-segment joints as:

```
joint_primary_secondary   (parent = primary_tip,   child = secondary_base)
joint_secondary_spur      (parent = secondary_tip,  child = spur_base)
joint_spur_stem           (parent = spur_tip,       child = stem_base)
joint_stem_apple          (parent = stem_tip,       child = apple)
```

**Theorem**: at quasi-static rest, with no ground contacts in the subtree:

```
force_world[Z] of joint_j  =  M_subtree_j · g
```

where `M_subtree_j` is the sum of body masses of **all bodies strictly below
the cut** (i.e. descendants of the child of joint j).

**Proof sketch**:

1. Take the subtree below joint `j` as a free-body system `S_j`.
2. Internal forces inside `S_j` (cable joints between rod segments, the
   `joint_stem_apple` FIXED joint) are Newton's-3rd-law action–reaction pairs
   and cancel in the vector sum.
3. External forces on `S_j`:
   - Gravity: `Σ (0, 0, −m_i · g)` for each body `i ∈ S_j`.
   - The constraint wrench from joint `j` on the child body: `(0, 0, +F_Z)`.
   - Ground contact: assumed absent (verified in tests by checking all body Z
     positions).
4. ΣF = 0 gives `F_Z = M_subtree_j · g`.

### Expected values for the full-chain scene

| Joint | Bodies in subtree | Expected `force_world[Z]` |
|---|---|---|
| `joint_stem_apple` | apple | `m_apple · g` |
| `joint_spur_stem` | stem_bodies + apple | `(M_stem + m_apple) · g` |
| `joint_secondary_spur` | spur_bodies + stem_bodies + apple | `(M_spur + M_stem + m_apple) · g` |
| `joint_primary_secondary` | secondary_bodies + spur_bodies + stem_bodies + apple | `(M_sec + M_spur + M_stem + m_apple) · g` |

`primary_bodies[0]` has `mass = 0` (it is the pin anchor) and sits **above**
every cut, so it never enters any subtree sum.  The remaining primary bodies
also sit above the cut for `joint_primary_secondary`.

---

## 6. How masses are computed

| Segment | Mass computation |
|---|---|
| Rod bodies | `ShapeConfig(density=rod.density)` passed to `builder.add_rod`; Newton integrates capsule geometry × density internally → read back as `model.body_mass` |
| Apple body | `m = (4/3)π r³ ρ` computed explicitly in `_build_scene`, passed as `mass` to `builder.add_link`. The collision sphere is added with a **copy** of `default_shape_cfg` and `density = 0.0`, so the shape does **not** add a second lump of mass from the default ~1000 kg/m³ shape density (which would otherwise stack on top of `add_link` per Newton’s mass accumulation rules). |

So `model.body_mass[apple]` matches the analytic sphere mass used in
`add_link`.  Tests still read `model.body_mass` from the built model so they
stay aligned with whatever the builder finalized.

### Nearly linear range fixture

`apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json` is tuned for **hanging,
almost vertical** trees (unit tests and wrench checks): primary uses `elevation_deg = −90°` (direction along
world **−Z**, same as gravity), and child segments only apply ±0.5°
deflection deltas.  The interactive example defaults to
``fruiting_system_ranges_example_variance.json`` (wider angles).  The straight fixture keeps subtree wrench checks easy to
interpret while preserving seed-to-seed variation in lengths and stiffness.

---

## 7. Tolerances

**Rigid joint damping:** ``apple_pick_sim.fruiting_system.make_fruiting_solver_vbd``
applies small ``rigid_joint_linear_kd`` / ``rigid_joint_angular_kd`` (constant
``FRUITING_VBD_RIGID_JOINT_KD`` in ``fruiting_system/build.py`` via ``make_fruiting_solver_vbd``) so inter-segment FIXED
joints settle; the viewer, headless rollouts, and ``test_wrench_equilibrium.py``
share this solver configuration.

The AVBD solver uses an augmented-Lagrangian penalty method; constraint
violations are small but not exactly zero.  The expected tolerances at
quasi-static rest (vertical chain, lightly damped joints, ≥ ~200 settling
frames at 60 fps / 10 substeps in `test_wrench_equilibrium.py`) are:

| Assertion | Tolerance |
|---|---|
| `force_world[Z]` vs. analytic (full chain, mean over last frame) | ≤ 20 % |
| `force_world[Z]` vs. analytic (minimal stem + apple) | ≤ 5 % |
| `\|torque_at_child_com_world\|` vs. zero | < 2 % of `m_apple · g · r_apple` |
| Force linearity (mass ratio) | ≤ 5 % |

These bounds were chosen conservatively; the solver typically achieves < 2 %
error on the minimal stem+apple scene.  The full-chain subtree check uses
the mean of ``force_world[Z]`` over the last frame’s substeps to reduce
single-substep noise, and a wider band because the stem–apple anchor is
off the apple COM.

---

## 8. Quick reference — running the tests

```bash
# From repo root:
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest \
    ../apple_pick_sim/tests/test_wrench_equilibrium.py -v \
    -p no:launch_testing
```

---

## 9. Glossary

| Term | Meaning |
|---|---|
| AVBD | Augmented Variational Body Dynamics — the constrained rigid-body algorithm inside `SolverVBD` |
| λ (lambda_lin) | Per-joint augmented-Lagrangian multiplier for the linear constraint; accumulates over time steps |
| Child body | The body designated as `joint_child[j]` in Newton's model |
| COM | Centre of mass (world position = `body_q[b].translation` for Newton rigid bodies with `body_com = 0` in local frame) |
| Subtree | All bodies reachable from the child of a given joint by following child links downward |
