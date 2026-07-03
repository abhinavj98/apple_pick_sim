# Material-parameter sampling (E, ζ)

## Document status

| Field | Value |
| ----- | ----- |
| **Last updated** | 2026-07-01 |
| **Roadmap slice** | [V].2.1.3 |
| **Owner** | Abhinav |

## Summary

Replace **independent** sampling of VBD rod knobs (`bend_stiffness`, `stretch_stiffness`, `bend_damping`) with sampling of **material** properties **Young's modulus** \(E\) [Pa] and **damping ratio** \(\zeta\) [–], then **derive** simulation stiffness/damping from geometry at sample time.

Geometry (`length`, `radius`, `density`, directions, `num_segments`) continues to be sampled (or fixed) as today. The VBD build path in `apple_pick_sim/fruiting_system/build.py` consumes derived `RodParams` stiffness/damping fields — only the **sampler** and **range JSON contract** change.

## Motivation

| Today | Problem |
| ----- | ------- |
| Independent min/max bands on `bend_stiffness`, `stretch_stiffness`, `bend_damping` | Stiffness and damping can be physically inconsistent with sampled `radius` / `length` / `density` |
| Sys-ID / CEM over raw VBD knobs | Hard to interpret identified values; weak transfer across geometry changes |
| Fixture tiers in N/m (`docs/real-world-proxy.md`) | Useful as **targets**, but should map through \(E\) and geometry rather than direct `bend_stiffness` bands |

Sampling \(E\) and \(\zeta\) keeps domain randomization and calibration tied to interpretable material parameters while preserving the existing Newton `add_rod` API.

## Sampling contract

### Range JSON (per rod segment)

**New primary keys** (replace `bend_stiffness`, `stretch_stiffness`, `bend_damping` ranges):

| Key | Unit | Meaning |
| --- | ---- | ------- |
| `youngs_modulus_pa` | Pa | Young's modulus \(E\) |
| `damping_ratio` | – | Modal damping ratio \(\zeta\) (fraction of critical) |

**Unchanged** (still sampled or fixed per segment): `num_segments`, `length`, `radius`, `density`, angle keys (`azimuth_deg`, `elevation_deg`, deltas).

**Apple** segment unchanged (`radius`, `density` only).

### Derived VBD knobs (at `sample_params` time)

For a circular rod cross-section with sampled `radius` \(r\), `density` \(\rho\), total `length` \(L\), and `num_segments` \(N\):

\[
A = \pi r^2,\quad I = \frac{\pi r^4}{4},\quad L_{\mathrm{seg}} = \frac{L}{N},\quad m_{\mathrm{seg}} = \rho A L_{\mathrm{seg}},\quad J_{\mathrm{seg}} = \frac{m_{\mathrm{seg}}(3r^2 + L_{\mathrm{seg}}^2)}{12}
\]

Default constitutive mapping (segment-local):

| VBD field | Formula | Units |
| --------- | ------- | ----- |
| `stretch_stiffness` | \(E A / L_{\mathrm{seg}}\) | N/m |
| `stretch_damping` | \(2 \zeta \sqrt{k_{\mathrm{stretch}}\, m_{\mathrm{seg}}}\) | N·s/m |
| `bend_stiffness` | \(E I / L_{\mathrm{seg}}\) | N·m/rad |
| `bend_damping` | \(2 \zeta \sqrt{k_{\mathrm{bend}}\, J_{\mathrm{seg}}}\) | N·m·s/rad |

\(J_{\mathrm{seg}}\) is the solid-cylinder segment moment of inertia about a transverse axis through the midpoint, making `bend_damping` dimensionally consistent. A single \(\zeta\) governs both axial and bending modes.

### Optional `vbd_stretch_fixed` override (batched VBD settling)

For stability-sensitive batched simulations (e.g. `example_batched_heterogeneous_coupled_fruiting.py` with `--only-vbd`), fixtures may pin the **axial** VBD knobs while bend DR stays on material keys:

```json
"vbd_stretch_fixed": {
  "stretch_stiffness": 500000.0,
  "stretch_damping": 30.0
}
```

Per rod segment, optional. When present, both keys are required (strictly positive). `youngs_modulus_pa` and `damping_ratio` min/max bands are still required and still drive `bend_stiffness` and `bend_damping`. Fixed stretch values are **VBD tuning constants**, not beam-theory-consistent with sampled geometry — see `fruiting_system_ranges_real_world_proxy_variance.json`.

### Tier constraints

When both **primary** and **secondary** are enabled, enforce **`primary.youngs_modulus_pa >= secondary.youngs_modulus_pa`** (replaces the current `primary.bend_stiffness >= secondary.bend_stiffness` check on derived values).

### Serialization

- Bump `FRUITING_SYSTEM_PARAMS_SCHEMA` to `fruiting_system_params_v2`.
- Store sampled **`youngs_modulus_pa`** and **`damping_ratio`** on each `RodParams` row in episode metadata (in addition to derived VBD scalars for replay fidelity).
- `params_fingerprint` adds `*_youngs_modulus_pa` and `*_damping_ratio` fields.

### Episode metadata (v1 read-only)

- New episodes serialize as `fruiting_system_params_v2` with `youngs_modulus_pa` and `damping_ratio`.
- `fruiting_params_from_dict` still reads **v1** Parquet metadata (derived stiffness only); \(E\) and \(\zeta\) are back-computed from geometry + VBD scalars.
- Range JSON **rejects** legacy `bend_stiffness` / `stretch_stiffness` / `bend_damping` keys.

## Code map

| Module | Change |
| ------ | ------ |
| `apple_pick_sim/fruiting_system/params.py` | `RodParams` + `sample_params` / `_validate_ranges`; `rod_params_from_material(E, ζ, geometry)` helper |
| `apple_pick_sim/fixtures/*.json` | Replace stiffness/damping bands with `youngs_modulus_pa` / `damping_ratio` bands |
| `apple_pick_sim/fruiting_system/build.py` | **No change** — still passes derived `RodParams` into `add_rod` |
| `apple_pick_gym/examples/run_system_identification.py` | Follow-up: grid over \(E\) or derived stiffness (legacy CLI flags remain until [S]) |

## Tests (TDD)

Add / extend in `apple_pick_sim/tests/test_fruiting_system.py`:

| Test | Intent |
| ---- | ------ |
| `test_sample_params_material_mode_derives_stiffness_from_E_zeta` | Known \(E\), \(\zeta\), \(r\), \(L\), \(\rho\) → expected `stretch_stiffness`, `bend_stiffness`, `bend_damping` |
| `test_sample_params_material_mode_primary_E_ge_secondary` | Tier ordering on \(E\) when both segments enabled |
| `test_load_ranges_material_keys_validate` | Fixture JSON with `youngs_modulus_pa` / `damping_ratio` passes validation |
| `test_load_ranges_legacy_stiffness_still_valid` | Old JSON keys still load during migration |
| `test_fruiting_params_v2_roundtrip` | Serialize / deserialize includes \(E\), \(\zeta\) |

Regression: existing deterministic-seed tests updated to material-mode fixtures or legacy path smoke.

## Verification

```bash
# Fast gate after implementation
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_fruiting_system.py -q -k "material or youngs or damping_ratio or sample_params"

# Fixture + proxy policy
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_real_world_proxy_fixture.py -q

# Heterogeneous batched DR still samples per env
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q
```

## Related docs

- `docs/real-world-proxy.md` — map proxy N/m tier **targets** to \(E\) bands via nominal geometry
- `docs/vectorized-coupled-fruiting.md` — batched DR still varies per-env θ; material keys replace raw stiffness bands
- `docs/system_identification.md` — CEM search space moves to \(E\), \(\zeta\) (plus geometry when fixed)
- `docs/sysid-trajectory-storage.md` — episode `fruiting_system_params` schema v2
