# Explicit apple load in TCP stem harvest

## Sim-to-real role

Welded builds include an explicit apple term in the harvested TCP wrench so the arm
(Model A, zero gravity) feels fruit **dead weight as an external load** — analogous
to a real FR3 with **gravity compensation assuming zero payload** after grasp.
Per-env `apple_mass_kg` from build/DR scales this term; it is **not** applied via
`robot_model.gravity`. See `docs/mujoco-vbd-coupling-architecture.md` §2.5.

**Complementary:** welded builds also attach a mass-only MuJoCo TCP child for
reflected inertia (\(I=\tfrac{2}{5}mr^{2}\)); that body does **not** replace this
harvest weight path. See `docs/mujoco-apple-payload.md`.

## Behavior

When `fix_to_apple=True`, the apple is **prescribed** for VBD (`inv_mass == 0`)
while `body_mass` stays analytic. Stem joint gather alone may under-represent
fruit weight at quasi-static hold.

**When enabled** (`stem_harvest_explicit_apple_weight=True`), stem harvest forms a
**child-side** plant wrench (before gain/caps):

\[
\mathbf{F}_{\mathrm{support}} = -m_{\mathrm{apple}}\,\mathbf{g}, \quad
\boldsymbol{\tau}_{\mathrm{support}} = (\mathbf{p}_{\mathrm{apple}} - \mathbf{p}_{\mathrm{tcp}}) \times \mathbf{F}_{\mathrm{support}}
\]

With \(\mathbf{g}=(0,0,-9.81)\), \(\mathbf{F}_{\mathrm{support}}=(0,0,+m g)\) (upward
support the stem would supply on the apple). The harvest kernels then **negate**
the total plant wrench (stem reaction + optional support) when writing
`proxy_forces[tcp]`, so TCP `body_f` uses a **wrist F/T dead-weight convention**:
hanging fruit → downward pull on the tool.

When `gripper_proxy_offset_in_apple_frame` is set (`fix_to_apple`), apple COM is
derived from the TCP pose:

\[
\mathbf{p}_{\mathrm{apple}} = \mathbf{p}_{\mathrm{tcp}} - R_{\mathrm{tcp}}\,\mathbf{o}_{\mathrm{apple}},
\]

matching `mirror_robot_tcp_to_proxy_and_apple_kernel`. Otherwise positions come
from `robot_state_0.body_q` (TCP) and cable `state_0.body_q` (apple).
\(\mathbf{g} =\) `gravity_vec` (typically `(0, 0, -9.81)`).

## Build-time default

| Layer | Default |
|-------|---------|
| `CoupledFruitingScene.stem_harvest_explicit_apple_weight` field | `False` (`scene.py`) |
| `build_coupled_fruiting_*` via `_resolve_stem_harvest_explicit_apple_weight` | `True` when `gripper_proxy.fix_to_apple` (prescribed apple); `False` for free proxy (`fix_to_apple=False`) |

Resolver (`builders.py`): `override=None` uses the welded/free rule above;
`override=True` forces on (raises if `fix_to_apple=False` — VBD already
integrates apple gravity and explicit correction would double-count);
`override=False` forces off.

**Note:** `FruitingSystemConfig.stem_harvest_explicit_apple_weight` on the batched
heterogeneous config is currently **inert** (not passed through builders).

Free-proxy scenes should keep explicit load **off**; welded (settle-then-weld)
scenes turn it **on** at build so quasi-static TCP readouts include fruit weight
under the F/T convention above.

## Code

| Symbol | Role |
|--------|------|
| `explicit_load.apple_support_force_world` | Child-side support \(-\,m\mathbf{g}\) |
| `explicit_load.explicit_apple_wrench_for_stem_harvest` | Support force + torque about TCP |
| `proxy_coupling.harvest_stem_tension_for_tcp` / `harvest_batched_stem_tension` | Gather + support + **negate** → TCP |

**CUDA graphs:** explicit apple load is computed on device inside the stem harvest
kernel (no `body_q.numpy()` / `body_mass.numpy()` per substep).
`CoupledFruitingScene.apple_mass_kg` is filled in `builders._cached_apple_mass_kg`
at scene build.

## Tests

- `apple_pick_sim/tests/test_explicit_apple_load.py` — support formula, on/off TCP
  delta (expects **downward** ΔFz under F/T convention), settle→weld hold
- `apple_pick_sim/tests/test_coupled_fruiting_system.py` — TCP harvest vs stem
  gather reference (includes negation)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_explicit_apple_load.py \
  apple_pick_sim/tests/test_coupled_fruiting_system.py -q -k "explicit or tcp_harvest or stem_load"
```
