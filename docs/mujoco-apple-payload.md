# MuJoCo apple payload (inertia-only)

## Behavior summary

On **welded** FR3 builds (`GripperProxyConfig.fix_to_apple=True`), Model A gains a mass-only FIXED child of the TCP labeled `apple_payload`. Model A **gravity stays zero**; quasi-static fruit weight still enters via stem harvest (`-m · g`). The dummy supplies **reflected inertia** only:

\[
m = m_{\mathrm{AVBD\,apple}},\quad
I = \tfrac{2}{5} m r^{2}\,\mathbf{1},\quad
\mathbf{c}_{\mathrm{TCP}} = \mathrm{trans}\big(X_{\mathrm{offset}}^{-1}\big)
\]

so \(X_{\mathrm{apple}} = X_{\mathrm{tcp}} \cdot X_{\mathrm{offset}}^{-1}\) matches co-teleport / explicit-load COM placement. No MuJoCo sphere geom (AVBD owns collision radius).

Heterogeneous envs share one replicated topology; per-world `body_mass` / `body_inertia` / `body_com` are patched then synced with `notify_model_changed(BODY_INERTIAL_PROPERTIES)`.

## Code map

| Symbol | Role |
|--------|------|
| `coupled_fruiting/mujoco_apple_payload.py` | Helpers, `append_apple_payload_link`, `apply_mujoco_apple_payload_inertias` |
| `robot/fr3_robot/setup.py` | `build_fr3_robot_builder(..., add_apple_payload=)` |
| `coupled_fruiting/builders.py` | Welded builds pass `add_apple_payload=True`; apply after assemble / per-env offsets |
| `BatchedEnvLayout.mj_apple_payload_body_indices` | Per-world payload body indices |
| `CoupledFruitingScene.mj_apple_payload_body_index` | World-0 / single-env payload index |

## Tests

- `apple_pick_sim/tests/test_mujoco_apple_payload.py` — unit math; welded mass/I/COM match AVBD; free proxy has no payload; hetero per-env masses differ; **clear vs analytic \(I\)**; **VIC TCP motion with vs without payload inertia**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mujoco_apple_payload.py \
  apple_pick_sim/tests/test_explicit_apple_load.py -q
```
