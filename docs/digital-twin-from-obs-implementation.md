# Digital twin from field observations

## Behavior summary

The digital-twin path reconstructs a quasi-static fruiting cable scene from **partial
real-world observations** instead of sampling geometry from variance fixture JSON.
Observations use the same keys as the gym contract (`woody_part_start_pos`,
`woody_part_end_pos`, `weld_direction`) plus placement metadata.

Workflow (analogous to settle-then-weld, but obs-driven):

1. Load a `digital_twin_v2` JSON file (`apple_pick_sim/digital_twin/obs_io.py`; v1 files still load for compatibility).
2. Infer per-rod **direction** and **length** from junction anchor positions under the
   straight-rod approximation (`infer_segment_geometry`).
3. Use observed `rod_radii` when present. Take remaining non-geometric scalars
   (stiffness, damping, density, `num_segments`, apple density) from the **midpoint**
   of a base fixture JSON (`params_from_ranges_median`).
4. Build a `CoupledCableScene` via `build_digital_twin_scene` (VBD-only; no FR3 arm).

### Geometry recovery

For junction index `i` and rod name from `junction_names`:

| Segment | Origin | Target (parent-side anchor) |
|---------|--------|-----------------------------|
| First rod | `fruiting_base_pos` | `woody_part_start_pos[0]` |
| Rod `i>0` | `woody_part_end_pos[i-1]` | `woody_part_start_pos[i]` |

```text
direction = normalize(target - origin)
length    = |target - origin|
```

Topology is encoded by `junction_names` (gym labels without `joint_` prefix), e.g.
`primary_secondary`, `stem_apple`. The last `*_apple` junction enables the apple sphere;
`apple_radius` comes from the obs file or the base fixture midpoint.

`weld_direction` is applied only when `fix_to_apple=True` on
`GripperProxyConfig` (same rule as `build.py`).

## Code map

| Module | Responsibility |
|--------|----------------|
| `apple_pick_sim/digital_twin/obs_io.py` | `DigitalTwinObs`, `load_digital_twin_obs`, `save_digital_twin_obs` |
| `apple_pick_sim/digital_twin/from_obs.py` | `infer_params_from_obs`, `build_digital_twin_scene`, `params_from_ranges_median` |
| `apple_pick_sim/fixtures/digital_twin_obs_example.json` | Example obs recorded from the straight-rod fixture |
| `apple_pick_sim/fixtures/digital_twin_fixture_catalog.json` | Named fixture manifest for sim-to-sim replay/reconstruction smoke checks |
| `apple_pick_sim/examples/example_digital_twin.py` | Interactive viewer + optional VBD settle |

## Named fixture catalog

`apple_pick_sim/fixtures/digital_twin_fixture_catalog.json` is the small M3.0.4
catalog used to keep fixture names, base poses, observation files, and smoke
commands together. The initial entries are:

- `straight_rod_test` — deterministic straight-rod baseline with
  `digital_twin_obs_straight_rod_initial.json` for observation-derived scene
  reconstruction checks.
- `example_variance` — the procedural variance fixture used by the default
  gym/sys-ID scene builder.

Catalog paths are repository-root relative. The catalog test loads each referenced
range fixture, parses its `fruiting_base_pos` / `robot_base_pos`, loads any
referenced observation file, and checks that listed smoke commands use `uv run`.
This is intentionally a manifest-level check; it does not add a new fixture
selection framework.

## Tests

- `apple_pick_sim/tests/test_digital_twin.py::test_load_save_roundtrip` — JSON schema round-trip
- `apple_pick_sim/tests/test_digital_twin.py::test_round_trip_junction_positions` — median-built scene → obs → twin; anchors within 1 mm
- `apple_pick_sim/tests/test_digital_twin.py::test_topology_primary_only` — single-rod + apple topology
- `apple_pick_sim/tests/test_digital_twin.py::test_topology_full_chain` — four-rod chain inference
- `apple_pick_sim/tests/test_digital_twin.py::test_infer_params_uses_observed_rod_radii` — measured radii override fixture midpoints
- `apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets` — catalog paths stay valid
- `apple_pick_sim/tests/test_digital_twin.py::test_infer_params_rejects_mismatched_anchor_lengths` — validation

## How to verify

```bash
# Unit tests
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q

# Catalog + sim-to-sim fixture manifest check
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets -q

# Pull-direction geometry check (writes outside the repo)
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output /tmp/apple_pick_pull_directions.png

# Smoke (no viewer; import + build only)
uv run --env-file pytest.env python -c "
from pathlib import Path
from apple_pick_sim.digital_twin import load_digital_twin_obs, build_digital_twin_scene
obs = load_digital_twin_obs('apple_pick_sim/fixtures/digital_twin_obs_example.json')
build_digital_twin_scene(obs, 'apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json', device='cpu')
print('ok')
"

# Interactive viewer
uv run python apple_pick_sim/examples/example_digital_twin.py \\
  --obs apple_pick_sim/fixtures/digital_twin_obs_example.json \\
  --base-fixture apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json \\
  --settle-substeps 0 --device cpu
```

## Observation file format (`digital_twin_v2`)

```json
{
  "schema": "digital_twin_v2",
  "fruiting_base_pos": [0.0, 0.2, 1.3],
  "weld_direction": [x, y, z],
  "junction_names": ["primary_secondary", "secondary_spur", "spur_stem", "stem_apple"],
  "woody_part_start_pos": ["N*3 flat floats, meters"],
  "woody_part_end_pos": ["N*3 flat floats, meters"],
  "apple_radius": 0.054,
  "rod_radii": {
    "primary": 0.012,
    "secondary": 0.010,
    "spur": 0.008,
    "stem": 0.004
  }
}
```

`weld_direction` must be a unit vector. Arrays must have length `3 * len(junction_names)`.
`rod_radii` is optional for old observations; when omitted, inference falls back to
fixture midpoint radii.
