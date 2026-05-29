# Explicit apple weight in TCP stem harvest

## Behavior

When `fix_to_apple=True`, the apple is **prescribed** for VBD (`inv_mass == 0`) while `body_mass` stays analytic. Stem joint gather alone may under-represent fruit weight at quasi-static hold.

**Default:** `CoupledFruitingScene.stem_harvest_explicit_apple_weight = True` adds (before `stem_coupling_gain` and caps):

\[
\mathbf{F}_{\mathrm{add}} = -m_{\mathrm{apple}}\,\mathbf{g}, \quad
\boldsymbol{\tau}_{\mathrm{add}} = (\mathbf{p}_{\mathrm{apple}} - \mathbf{p}_{\mathrm{tcp}}) \times \mathbf{F}_{\mathrm{add}}
\]

When `gripper_proxy_offset_in_apple_frame` is set (`fix_to_apple`), apple COM is derived from the TCP pose:

\[
\mathbf{p}_{\mathrm{apple}} = \mathbf{p}_{\mathrm{tcp}} - R_{\mathrm{tcp}}\,\mathbf{o}_{\mathrm{apple}},
\]

matching `mirror_robot_tcp_to_proxy_and_apple_kernel`. Otherwise positions come from `robot_state_0.body_q` (TCP) and cable `state_0.body_q` (apple). \(\mathbf{g} =\) `gravity_vec` (typically `(0, 0, -9.81)`).

## Code

| Module | Role |
|--------|------|
| `apple_pick_sim/coupled_fruiting/explicit_load.py` | `apple_support_force_world`, `apple_com_from_tcp_grasp_offset`, `explicit_apple_wrench_for_stem_harvest` |
| `proxy_coupling.harvest_stem_tension_for_tcp` | GPU path (explicit wrench in `_limit_and_write_tcp_stem_wrench_kernel`) + CPU fallback |
| `CoupledFruitingScene.coupled_substep` | Passes `apple_body`, `apple_mass_kg` (cached at build), `gravity_vec`, flag |

**CUDA graphs:** explicit apple weight is computed on device inside the stem harvest kernel (no `body_q.numpy()` / `body_mass.numpy()` per substep). `CoupledFruitingScene.apple_mass_kg` is filled in `builders._cached_apple_mass_kg` at scene build.

Disable for raw stem-only readouts:

```python
scene.stem_harvest_explicit_apple_weight = False
```

## F/T sensor mimic

TCP wrench for a virtual flange sensor:

```python
from apple_pick_sim.coupling_force_debug import read_tcp_wrench

w_world = read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index)
```

After `coupled_substep`, with default explicit load this includes quasi-static apple support.

## Tests

- `apple_pick_sim/tests/test_explicit_apple_load.py` — unit helpers, harvest on/off delta, `apple_mass_kg=0` skip, build-time cache, coupled flag, settle→weld explicit ≈ `m·g`
- `test_coupled_fruiting_system.py` — `_stem_force_with_explicit_apple_weight`, TCP/stem parity helpers
- `test_settle_then_weld.py` — quiet seed invariants (`body_q_prev`, twists, cleared wrenches)
- `test_proxy_coupling.py::test_stem_limit_kernel_launch_input_count_regression` — stem limit kernel launch arity
- `test_cuda_graph.py::test_coupled_cuda_graph_welded_explicit_stem_harvest_finite` — captured loop with explicit load (CUDA)

Run:

```bash
PYTHONPATH=$(pwd) uv run --directory newton python -m pytest ../apple_pick_sim/tests/test_explicit_apple_load.py -q -p no:launch_testing
```
