# Real pre-grasp → FruitingSystemParams settle viewer

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living doc:** | `docs/handbook-real-replay.md` |
| **Date** | 2026-07-24 |
| **Scope** | Plant-only: pre-grasp → `FruitingSystemParams` → settle → Newton viewer |
| **Non-goals** | Robot, weld, trajectory/`action`, CMA-ES / full `batched_sysid_v1` writer |

Related: `docs/handbook-real-replay.md`, `robot_replay/README.md`,
`docs/digital-twin.md`, `docs/real-world-proxy.md`.

## Problem

Real episodes (`robot_replay/s00-d00.parquet`) store non-bending plant geometry in
`dataset_metadata.pre_grasp_geometry`. We need a path that:

1. Converts that geometry into sim-native **`FruitingSystemParams`** (same type
   scene build / later oracle replay use).
2. Builds a **VBD plant-only** scene, settles under gravity, and opens a visualizer.

This is the first consumer of a params-first real→sim conversion. A later slice
will embed the same params blob into a full sim-supported dataset for complete
replay (weld + trajectory).

## Approach

**A — Thin `robot_replay` CLI + library helper** (chosen):

- Library under `apple_pick_sim/system_id/` builds `FruitingSystemParams` +
  `fruiting_base_pos` from real parquet pre-grasp metadata.
- `robot_replay/example_view_pre_grasp_settle.py` builds, settles, views.
- Reuse `generate_coupled_cable_scene` and settle loop pattern from
  `example_digital_twin.py`.
- `fix_to_apple=False` (no gripper weld).

Deferred: plant+robot, post-grasp weld, full dataset conversion.

## Pipeline

```text
real parquet
  dataset_metadata.pre_grasp_geometry (+ topology, parts)
       │
       ▼
FruitingSystemParams + fruiting_base_pos
       │  (materials from fixture midpoints)
       ▼
generate_coupled_cable_scene(params, base_pos, fix_to_apple=False)
       ▼
settle N VBD substeps
       ▼
Newton viewer (GL) / null for smoke
```

Optional `--dump-params out.json` writes:
- `fruiting_base_pos`
- `fruiting_system_params` (`fruiting_params_to_dict`)
- `diagnostics`: catalog-vs-chord length errors, spur/stem chord lengths,
  `apple_pos` vs Spur→Apple chord-end error, directions used

## Pre-grasp → params mapping

### Placement

For T-junction / real-world proxy:

- **`fruiting_base_pos`** = measured **Branch** xyz in the parquet frame (`franka_base_o`)
  = start of spur = spur–primary T-junction. Do not substitute the fixture default
  base for this viewer slice.
- **Primary** is built **horizontal through that T-junction** (proxy ±X / fixture
  primary axis), with length/radius from `parts.primary`.
- **Spur** tracker = **spur distal end** (end of the spur chord).
- **Apple** tracker / apple pose = fruit **center** (mid-point of the apple).

Chord layout (`shared_endpoints`): part0 Branch→Spur (spur chord), part2
Spur→Apple (stem chord).

### Rods (Branch / Spur / Apple, `shared_endpoints`)

| Sim field | Source |
| --------- | ------ |
| `primary.direction` | Same as sim fixture sampling: midpoint `azimuth_deg` / `elevation_deg` from the variance fixture (real-world proxy: both 0 → horizontal **+X**) |
| `primary.length` / `radius` | `parts.primary` (catalog); report length error vs hang chords where applicable |
| `spur.direction` | Unit vector along spur chord: **T-junction → spur end** (Branch→Spur) |
| `spur.length` / `radius` | `parts.spur` |
| `stem.direction` | Unit vector **spur end → `snapshot.apple_pos`** (apple COM). There is no separate stem-end tracker. Report error vs Spur→Apple woody chord end when both exist. |
| `stem.length` / `radius` | `parts.stem` |
| `secondary` | `null` |
| `apple_radius` | `parts.apple.radius_m` else fixture midpoint |
| `apple_density` | Fixture midpoint |
| `topology` | `t_junction` |
| `spur_attach_fraction` | Fixture (default 0.5) |

Materials: **density** (and apple density) from `parts.*.density_kg_m3`;
`youngs_modulus_pa`, `damping_ratio`, stretch knobs, and `num_segments` from
variance fixture midpoints via `rod_params_from_material` /
`build_fruiting_params_from_real` (extend that helper to accept optional
per-rod densities from parts).

Lengths come from `parts.*` (catalog / lengthened-state). Always **print**
absolute and relative error vs measured chords (`|Branch→Spur|` vs
`parts.spur.length_m`, `|Spur→Apple|` vs `parts.stem.length_m`; also
`apple_pos` vs Spur→Apple chord end when present). Length mismatch never fails
the run — `--strict` applies only to pre-grasp bend≈0. Directions from the
chords / `apple_pos` as above. Do **not** share spur/stem direction.




### Validation

- Require `pre_grasp_geometry` + woody + topology + usable `parts`.
- Prefer `woody_bending_angles ≈ 0`; warn by default, `--strict` fails.
- Coerce string-encoded `apple_pos` (known quirk); **required** for stem direction
  (spur end → apple COM).
- Unknown topology (not Branch/Spur/Apple shared_endpoints) → fail with a clear
  message until another map is registered.

## Files

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/system_id/real_pre_grasp_params.py` | Load metadata; map pre-grasp → `FruitingSystemParams` + `fruiting_base_pos` |
| `apple_pick_sim/tests/test_real_pre_grasp_params.py` | Unit tests (no GPU viewer) |
| `robot_replay/example_view_pre_grasp_settle.py` | CLI: build → settle → view |
| `robot_replay/README.md` | Document command + params-first intent |

Plant-only means no FR3 and no weld (`fix_to_apple=False`). The default free
gripper proxy body from `generate_coupled_cable_scene` **remains** (same as
digital-twin plant builds); do not special-case hiding it in this slice.

## CLI

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --settle-substeps 5000 \
  --viewer gl
```

Default `--settle-substeps` is **5000**. **Show settling in the viewer**: do not
hide a long offline settle before the first frame; drive settle VBD steps while
rendering (or interleave settle substeps with viewer frames) so the plant is
visible as it comes to rest. After that budget, keep simulating each frame as
above. Initial camera matches `example_digital_twin.py` (no auto look-at).

Flags: `--dump-params`, `--strict`, `--viewer null` (CI/smoke), device via
existing sim-device conventions.

## Tests

- Synthetic metadata: Branch → `fruiting_base_pos`; spur/stem dirs from
  Branch→Spur and Spur→`apple_pos`; parts L/r; length-error report.
- Optional smoke on `robot_replay/s00-d00.parquet` metadata (parse only).
- Serialization: `fruiting_params_to_dict` / `fruiting_params_from_dict` round-trip.
- No interactive viewer in pytest.

## Success criteria

1. From `s00-d00.parquet`, converter yields valid `FruitingSystemParams` and
   Branch-based `fruiting_base_pos`.
2. Viewer shows a plant-only cable scene (no FR3/weld), settles, then continues
   simulating under gravity in the interactive loop.
3. `--dump-params` JSON is embeddable later as episode
   `fruiting_system_params` for sim-native replay.

## Deferred (explicit)

- Post-grasp weld / robot attach
- Trajectory frames, `action` derivation, CMA-ES dataset layout
- Hardening all quirks in `docs/handbook-real-replay.md` beyond what
  this loader needs
