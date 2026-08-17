# Real rod mass → density override

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living doc** | `docs/handbook-real-replay.md` (H4) |
| **Date** | 2026-08-14 |
| **Roadmap** | Real replay / convert geometry fidelity |
| **Reference data** | `tmp/final_data/s09/`, `tmp/final_data/s11/` |

## Problem

Compiled real episodes now carry measured spur mass as
`pre_grasp_geometry.parts.spur.mass_kg` (s09: 0.026 kg, s11: 0.027 kg) alongside
catalog `density_kg_m3 = 1200`. Ingest copies only length, radius, and density.
Newton rods take mass from `ShapeConfig(density)` × capsule volume, so replay
builds a ~2.8 g s09 spur instead of 26 g (9× light) and a ~13 g s11 spur instead
of 27 g (2× light).

`add_rod` has no `mass=` argument. Density is the only mass channel that also
feeds stretch \(m_\mathrm{seg}\) and joint \(k_d\).

## Goal

When a rod part includes `mass_kg`, convert it to an effective cylinder density
and keep catalog `radius_m` so bending \(I \propto r^4\) is unchanged.

\[
\rho = \frac{m}{\pi r^{2} L}
\]

## Non-goals

- Patching `builder.body_mass` after `add_rod` to hit the kilogram exactly
  (capsule caps make Newton mass slightly larger than \(\pi r^{2} L \rho\)).
- Changing apple mass (already encoded as `density_kg_m3`).
- Adding lump-mass bodies or Newton API changes.
- Inflating radius to keep wood-like density.

## Design

`map_pre_grasp_geometry` (`apple_pick_sim/system_id/real_pre_grasp_params.py`)
owns the override inside the per-rod geometry helper:

- If `mass_kg` is present and not `None`, \(\rho = m / (\pi r^{2} L)\) replaces
  catalog `density_kg_m3`. Mass wins when both are present.
- If `mass_kg` is absent, keep catalog `density_kg_m3` (legacy episodes).
- Reject non-finite or non-positive `mass_kg`, and volume too small to invert.
- Leave `radius_m`, `length_m`, and Young's modulus unchanged.
- Record per overridden rod: logged mass, catalog density, derived density,
  `density_source="mass_kg"`.

`build_fruiting_params_from_real(..., use_parts_density=True)` is unchanged: it
already consumes `rod_geometry[*].density_kg_m3` for Newton, stretch knobs, and
serialized `fruiting_system_params`.

## Tests

`apple_pick_sim/tests/test_real_pre_grasp_params.py`:

- spur with `mass_kg` → density \(m/(\pi r^{2} L)\), radius unchanged;
- no `mass_kg` → catalog density;
- non-positive `mass_kg` raises;
- `fruiting_params_from_pre_grasp_meta` writes the derived density onto
  `params.spur.density`.

## Verify

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_pre_grasp_params.py \
  -q -p no:launch_testing
```
