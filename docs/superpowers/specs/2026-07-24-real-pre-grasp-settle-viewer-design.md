# Real pre-grasp → FruitingSystemParams settle viewer

| Field | Value |
| ----- | ----- |
| **Status** | Draft (approved in brainstorming 2026-07-24) |
| **Date** | 2026-07-24 |
| **Scope** | Plant-only: pre-grasp → `FruitingSystemParams` → settle → Newton viewer |
| **Non-goals** | Robot, weld, trajectory/`action`, CMA-ES / full `batched_sysid_v1` writer |

Related: `docs/real-sysid-pre-post-grasp-fixes.md`, `robot_replay/README.md`,
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

Optional `--dump-params out.json` writes `fruiting_params_to_dict` for inspection
and future dataset embedding.

## Pre-grasp → params mapping

### Placement

For T-junction / real-world proxy:

- **`fruiting_base_pos`** = **Spur** tracker position = T-center =
  mid-span `primary_spur` junction = where spur meets primary.
- Not the Branch-only end of the Branch→Spur chord (that is along the primary
  toward/away from the T, not the hang root).

Resolve Spur as the shared endpoint of Branch→Spur and Spur→Apple
(`topology.start_nodes` / `end_nodes` with `shared_endpoints: true`).

### Rods (Branch / Spur / Apple, `shared_endpoints`)

| Sim field | Source |
| --------- | ------ |
| `primary.direction` | Unit vector along Branch↔Spur (primary axis through T-center) |
| `primary.length` / `radius` | `pre_grasp_geometry.parts.primary` |
| `spur.direction` | Unit vector along Spur→Apple (hang toward fruit) |
| `spur.length` / `radius` | `parts.spur` |
| `stem.direction` | Same hang axis as spur ( Spur→Apple ) |
| `stem.length` / `radius` | `parts.stem` |
| `secondary` | `null` |
| `apple_radius` | `parts.apple.radius_m` else fixture midpoint |
| `apple_density` | Fixture midpoint |
| `topology` | `t_junction` |
| `spur_attach_fraction` | Fixture (default 0.5) |

Materials (`youngs_modulus_pa`, `damping_ratio`, stretch knobs, `density`,
`num_segments`) from variance fixture midpoints via existing
`rod_params_from_material` / `build_fruiting_params_from_real` pattern.

**Do not** treat the three tracker chords as a naive 1:1 primary/spur/stem
length map: Branch→Apple is not a single rod; Branch→Spur is primary-axis, not
the hanging spur length.

### Validation

- Require `pre_grasp_geometry` + woody + topology + usable `parts`.
- Prefer `woody_bending_angles ≈ 0`; warn by default, `--strict` fails.
- Coerce string-encoded vectors if present (known `apple_pos` quirk); plant-only
  path does not require apple_pos for build.
- Unknown topology (not Branch/Spur/Apple shared_endpoints) → fail with a clear
  message until another map is registered.

## Files

| Path | Role |
| ---- | ---- |
| `apple_pick_sim/system_id/real_pre_grasp_params.py` | Load metadata; map pre-grasp → `FruitingSystemParams` + `fruiting_base_pos` |
| `apple_pick_sim/tests/test_real_pre_grasp_params.py` | Unit tests (no GPU viewer) |
| `robot_replay/example_view_pre_grasp_settle.py` | CLI: build → settle → view |
| `robot_replay/README.md` | Document command + params-first intent |

Reuse/extend helpers in `real_to_batched_sysid.py` where they already assemble
params from measured L/r + fixture midpoints; do not require
`step_idx == -1` or full episode ingest for this slice.

## CLI

```bash
uv run python robot_replay/example_view_pre_grasp_settle.py \
  --parquet robot_replay/s00-d00.parquet \
  --fixture apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json \
  --settle-substeps 5000 \
  --viewer gl
```

Flags: `--dump-params`, `--strict`, `--viewer null` (CI/smoke), device via
existing sim-device conventions.

## Tests

- Synthetic metadata: Spur → `fruiting_base_pos`; primary axis from Branch↔Spur;
  hang axis from Spur→Apple; parts L/r applied.
- Optional smoke on `robot_replay/s00-d00.parquet` metadata (parse only).
- Serialization: `fruiting_params_to_dict` / `fruiting_params_from_dict` round-trip.
- No interactive viewer in pytest.

## Success criteria

1. From `s00-d00.parquet`, converter yields valid `FruitingSystemParams` and
   Spur-based `fruiting_base_pos`.
2. Viewer shows a settled plant-only cable scene without FR3/weld.
3. `--dump-params` JSON is embeddable later as episode
   `fruiting_system_params` for sim-native replay.

## Deferred (explicit)

- Post-grasp weld / robot attach
- Trajectory frames, `action` derivation, CMA-ES dataset layout
- Hardening all quirks in `docs/real-sysid-pre-post-grasp-fixes.md` beyond what
  this loader needs
