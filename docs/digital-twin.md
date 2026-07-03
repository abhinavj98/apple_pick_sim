# Digital twin: observation-only replay and field-twin reconstruction

## Document status

| Field | Value |
| ----- | ----- |
| **Roadmap slice** | M3.0.3 (observation-only replay init) **Done**; M3.0.4 (digital-twin fixture reconstruction) **Planned / partially blocked** — see [Known gap](#known-gap-catalog-and-example-obs-files-are-missing) |
| **Related docs** | `docs/system_identification.md`, `docs/sysid-trajectory-storage.md`, `docs/gym-observation-contract.md`, `docs/real-world-proxy.md` |

This document merges the observation-only-replay spec and the geometry-reconstruction implementation notes that used to live in two separate files (`observation-replay-digital-twin.md`, `digital-twin-from-obs-implementation.md`).

## Purpose

M3 needs a replay path that can run from real-world logs, not from Newton internals. The sim-to-sim recorder can optionally save a privileged `.npz` snapshot (`body_q`, `body_qd`, joint buffers, VBD previous-state buffers, and VIC targets). That snapshot proves action replay, but it cannot exist for field data.

**Observation-only replay** is the shipped default: rebuild a plausible initial Newton scene from a real-world-observable bundle plus calibration metadata, then drive it with the recorded end-effector velocity sequence. "Exact replayability" means exact replay of the observed experiment inputs and reconstruction contract; it does not mean recovering hidden simulator buffers perfectly from partial observations.

**Digital-twin reconstruction** is the companion geometry path: rebuild a quasi-static fruiting cable scene from partial real-world observations instead of sampling geometry from variance fixture JSON. This part is implemented in code but the fixture *data files* it depends on for the named catalog are not yet committed — see [Known gap](#known-gap-catalog-and-example-obs-files-are-missing).

## Validation strategy

Use sim-to-sim before real data:

1. Pick a "ground-truth" sim fixture and parameter set.
2. Collect sys-ID observations from it with `example_gym_sysid.py --output`.
3. Withhold `initial_states/<episode_id>.npz`.
4. Rebuild a tunable sim from observations and fixture metadata only.
5. Replay the recorded `action` sequence and compare against:
   - privileged replay initialized from the `.npz` snapshot;
   - recorded observations from the ground-truth sim.
6. Only after the replay initializer is stable, choose the tuning objective/optimizer for the differently tuned sim.

Primary drift metrics should include TCP position, TCP velocity, apple position, woody endpoint positions, `ft_wrist`, and per-junction wrench where available. MMD/CEM remains the planned calibration route, but optimizer choice should wait until the observation-only replay error floor is measured.

## Part 1 — Observation-only replay initialization

### Initial observation bundle

These fields are the minimum practical bundle to reconstruct the post-grasp initial condition and replay an episode without privileged simulator arrays.

| Bundle item | Real-world source | Sim/log equivalent | Used for |
|-------------|-------------------|--------------------|----------|
| Episode identity and schema version | Logger metadata | `episode_id`, `obs_schema` | Parser/version selection |
| Control rate and timestamps | Robot/log clock | `control_hz`, `sim_time`, `step_idx` | Time alignment |
| Recorded TCP action sequence | Robot command log or measured EE twist | `action` | Open-loop replay input |
| TCP pose and twist | Robot FK, calibrated base frame | `tcp_pos`, `tcp_velocity` | Initial robot/TCP state and features |
| F/T wrench with bias info | Wrist F/T sensor, bias captured before contact | `ft_wrist` | Replay error and MMD features |
| Apple pose | Camera/marker estimate | `apple_pos`, future apple orientation | Fruit body placement and grasp frame |
| Woody endpoint observations | Markers, tracked keypoints, or reconstructed geometry | `woody_part_start_pos`, `woody_part_end_pos` / Parquet `woody_start__*`, `woody_end__*` | Rod/junction placement and drift metrics |
| Junction labels/topology | Fixture file or perception topology output | `junction_names`, fixture JSON | Map observations to Newton bodies/joints |
| Grasp/weld transform | Robot grasp planner, calibration, or marker on gripper/apple | `weld_direction`, `weld_reference_pos`, `weld_reference_quat` | Recreate post-grasp coupling |
| Base/calibration transforms | Robot-to-world, camera-to-world, F/T frame transforms | `fruiting_base_pos`, `robot_base_pos`, fixture metadata | Put all observations in one world frame |
| Candidate or sampled physical parameters | Measurement/calibration pass or sim metadata | `fruiting_system_params`, fixture JSON/ranges | Build the digital twin geometry and dynamics without relying on a procedural seed |

Recommended optional fields: apple orientation (not just position); per-node confidence/covariance for occluded/noisy woody endpoints; F/T bias and temperature/session metadata; fixture/cultivar/session tags.

### Privileged-state replacement map

| Current privileged field | Why it is privileged | Observation-derived replacement |
|--------------------------|----------------------|---------------------------------|
| `robot_body_q`, `robot_joint_q` | Internal simulator robot state | Robot FK from measured joint positions plus TCP pose calibration |
| `robot_body_qd`, `robot_joint_qd` | Internal velocities | Measured joint velocities or finite-differenced FK/TCP twist |
| `cable_body_q` | Newton body transforms for fruiting rods/apple | Digital-twin geometry reconstruction from topology, woody endpoints, apple pose, and fixture base pose |
| `cable_body_qd` | Hidden plant velocities at reset | Initialize to zero after a settle/equilibrium solve unless measured marker velocities are available |
| `cable_state_1_body_q`, `cable_state_1_body_qd` | VBD previous-state buffers | Rebuild by copying settled `state_0` after observation-derived initialization |
| `vic_target_tf` | Controller internal target | TCP pose at reset, converted to the controller target frame |
| `weld_reference_pos`, `weld_reference_quat` | Sim apple body transform | Apple pose plus calibrated grasp transform |
| solver/contact caches | Internal numerical state | Clear and let the first settled substeps rebuild them |

The initializer prefers invariants over hidden-state guesses: known topology, calibrated frames, measured poses, zero or measured initial velocities, and a deterministic settle pass before replay starts.

For sim-to-sim datasets, `fruiting_system_params` records the exact sampled `FruitingSystemParams` used during collection. This removes seed-dependent resampling from no-snapshot replay, but it is still metadata for θ, not a privileged simulator state restore. Full-state replay still requires the optional `.npz` snapshot.

### Sim-to-sim smoke commands

Collect a short observation-only dataset without privileged simulator snapshots (this is the default; omit `--save-snapshot`):

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 \
  --output /tmp/apple_pick_sysid_no_snapshot
```

When a privileged baseline is needed for comparison, opt in explicitly:

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 --save-snapshot \
  --output /tmp/apple_pick_sysid_with_snapshot
```

Check pull-direction geometry against the current fixture/base-pose conventions:

```bash
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output /tmp/apple_pick_pull_directions.png
```

### Relation to current code

| Module/doc | Current role |
|------------|--------------|
| `apple_pick_sim/system_id/trajectory_store.py` | Writes Parquet frames plus optional privileged `.npz` snapshots |
| `apple_pick_gym/envs/apple_pick_replay_env.py` | Replays recorded actions; defaults to observation-only init, restores privileged snapshot only when `--use-snapshot` / a snapshot file is explicitly requested |
| `apple_pick_gym/examples/example_gym_replay.py` | Reports dataset-vs-live replay errors |
| `docs/sysid-trajectory-storage.md` | Dataset storage contract — keep the observation bundle synchronized with this document |
| `docs/gym-observation-contract.md` | Runtime observation schema — bump schema only when observation keys or semantics break |

### Tests and verification (Part 1)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_trajectory_store.py \
  apple_pick_gym/tests/test_replay_env.py -q
```

## Part 2 — Digital-twin geometry reconstruction from observations

### Behavior summary

Observations use the same keys as the gym contract (`woody_part_start_pos`, `woody_part_end_pos`, `weld_direction`) plus placement metadata.

Workflow (analogous to settle-then-weld, but obs-driven):

1. Load a `digital_twin_v2` JSON file (`apple_pick_sim/digital_twin/obs_io.py`; v1 files still load for compatibility).
2. Infer per-rod **direction** and **length** from junction anchor positions under the straight-rod approximation (`infer_segment_geometry`).
3. Use observed `rod_radii` when present. Take remaining non-geometric scalars (stiffness, damping, density, `num_segments`, apple density) from the **midpoint** of a base fixture JSON (`params_from_ranges_median`).
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

Topology is encoded by `junction_names` (gym labels without `joint_` prefix), e.g. `primary_secondary`, `stem_apple`. The last `*_apple` junction enables the apple sphere; `apple_radius` comes from the obs file or the base fixture midpoint.

`weld_direction` is applied only when `fix_to_apple=True` on `GripperProxyConfig` (same rule as `build.py`).

### Code map

| Module | Responsibility |
|--------|----------------|
| `apple_pick_sim/digital_twin/obs_io.py` | `DigitalTwinObs`, `load_digital_twin_obs`, `save_digital_twin_obs` |
| `apple_pick_sim/digital_twin/from_obs.py` | `infer_params_from_obs`, `build_digital_twin_scene`, `params_from_ranges_median` |
| `apple_pick_sim/examples/example_digital_twin.py` | Interactive viewer + optional VBD settle |

### Known gap: catalog and example obs files are missing

**The reconstruction code above is implemented and its non-catalog unit tests pass.** However, the named fixture catalog concept — a small JSON manifest tying fixture names to base poses, observation files, and smoke commands — is only partially delivered: the manifest file and the example observation JSON files it (and the interactive example) point to are **not present in the working tree**:

- `apple_pick_sim/fixtures/digital_twin_fixture_catalog.json` — missing.
- `apple_pick_sim/fixtures/digital_twin_obs_straight_rod_initial.json` — missing.

As a result, two tests in `apple_pick_sim/tests/test_digital_twin.py` **currently fail**:

```bash
$ uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q
FAILED apple_pick_sim/tests/test_digital_twin.py::test_example_digital_twin_registers_model_with_viewer
FAILED apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets
2 failed, 7 passed
```

Treat M3.0.4 as **not done** until these fixture files are authored (or the tests are updated to match a smaller committed scope) and both tests pass. This is a data/fixture gap, not a code gap — do not re-derive the catalog schema without checking `test_digital_twin.py` for the exact expected shape first.

Intended catalog shape (once authored), for reference:

- `apple_pick_sim/fixtures/digital_twin_fixture_catalog.json` — schema `apple_pick_sim_fixture_catalog_v1`; each entry names a fixture, its base range JSON, an optional observation JSON, and `uv run`-prefixed smoke commands.
- Expected initial entries: `straight_rod_test` (deterministic straight-rod baseline) and `example_variance` (the procedural variance fixture used by the default gym/sys-ID scene builder).
- Catalog paths are repository-root relative. The catalog test loads each referenced range fixture, parses its `fruiting_base_pos` / `robot_base_pos`, loads any referenced observation file, and checks that listed smoke commands use `uv run`. This is intentionally a manifest-level check, not a new fixture-selection framework.

### Tests (Part 2)

- `apple_pick_sim/tests/test_digital_twin.py::test_load_save_roundtrip` — JSON schema round-trip (passes)
- `apple_pick_sim/tests/test_digital_twin.py::test_round_trip_junction_positions` — median-built scene → obs → twin; anchors within 1 mm (passes)
- `apple_pick_sim/tests/test_digital_twin.py::test_topology_primary_only` — single-rod + apple topology (passes)
- `apple_pick_sim/tests/test_digital_twin.py::test_topology_full_chain` — four-rod chain inference (passes)
- `apple_pick_sim/tests/test_digital_twin.py::test_infer_params_uses_observed_rod_radii` — measured radii override fixture midpoints (passes)
- `apple_pick_sim/tests/test_digital_twin.py::test_infer_params_rejects_mismatched_anchor_lengths` — validation (passes)
- `apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets` — **currently fails** (missing catalog file)
- `apple_pick_sim/tests/test_digital_twin.py::test_example_digital_twin_registers_model_with_viewer` — **currently fails** (missing obs example file)

### How to verify (Part 2)

```bash
# Unit tests (2 known failures — see Known gap above)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin.py -q
```

### Observation file format (`digital_twin_v2`)

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

`weld_direction` must be a unit vector. Arrays must have length `3 * len(junction_names)`. `rod_radii` is optional for old observations; when omitted, inference falls back to fixture midpoint radii.

## Digital-twin fixture contract (target shape)

A digital-twin fixture is the bridge between observations and Newton geometry. Each named fixture should document:

- topology and junction labels matching `junction_names`;
- `fruiting_base_pos`, `robot_base_pos`, and any camera/F/T/world-frame transforms used during collection;
- nominal geometry parameters and ranges for rods, stem, and apple;
- known fixed quantities versus tunable theta parameters;
- expected observation keys and which real sensors produce them;
- a sys-ID smoke command and replay drift thresholds.

The first implementation should use a sim-generated ground-truth fixture, then rebuild a separate tunable fixture from exported observations. This verifies the reconstruction and replay machinery before the project depends on real perception quality.

## Expected follow-up tests (M3.0.3/M3.0.4, beyond current gap)

- dataset fixture with no `.npz` snapshot resets through the observation initializer (already exercised by the default `example_gym_sysid.py` / `ApplePickReplayEnv` path);
- privileged replay and observation-only replay run the same recorded `action` sequence;
- drift metrics are reported for TCP/apple/woody/F/T observations;
- a sim-generated digital-twin fixture can rebuild topology and base poses from metadata (covered for ad hoc obs files; blocked for the named catalog by the gap above);
- per-fixture sys-ID smoke can use catalog `smoke_commands` once the catalog exists.
