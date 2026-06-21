# Observation-only replay and digital-twin setup

## Purpose

M3 needs a replay path that can run from real-world logs, not from Newton internals. The current sim-to-sim recorder can save a privileged `.npz` snapshot (`body_q`, `body_qd`, joint buffers, VBD previous-state buffers, and VIC targets). That snapshot proves action replay, but it cannot exist for field data.

The next target is therefore **observation-only replay initialization**: use a real-world observable bundle plus calibration metadata to rebuild a plausible initial Newton scene, then drive it with the recorded end-effector velocity sequence. "Exact replayability" means exact replay of the observed experiment inputs and reconstruction contract; it does not mean recovering hidden simulator buffers perfectly from partial observations.

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

## Initial observation bundle

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

Recommended optional fields:

- apple orientation, not just position, so weld/grasp replay does not infer orientation from sim-only defaults;
- per-node confidence/covariance, so occluded or noisy woody endpoints can be weighted lower;
- F/T bias and temperature/session metadata, so repeatability checks can separate physics drift from sensor drift;
- fixture/cultivar/session tags, so parameter ranges can be scoped to a physical specimen.

## Privileged-state replacement map

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

The initializer should prefer invariants over hidden-state guesses: known topology, calibrated frames, measured poses, zero or measured initial velocities, and a deterministic settle pass before replay starts.

For sim-to-sim datasets, `fruiting_system_params` records the exact sampled `FruitingSystemParams` used during collection. This removes seed-dependent resampling from no-snapshot replay, but it is still metadata for θ, not a privileged simulator state restore. Full-state replay still requires the optional `.npz` snapshot.

## Digital-twin fixture contract

A digital-twin fixture is the bridge between observations and Newton geometry. Each named fixture should document:

- topology and junction labels matching `junction_names`;
- `fruiting_base_pos`, `robot_base_pos`, and any camera/F/T/world-frame transforms used during collection;
- nominal geometry parameters and ranges for rods, stem, and apple;
- known fixed quantities versus tunable theta parameters;
- expected observation keys and which real sensors produce them;
- a sys-ID smoke command and replay drift thresholds.

The first implementation should use a sim-generated ground-truth fixture, then rebuild a separate tunable fixture from exported observations. This verifies the reconstruction and replay machinery before the project depends on real perception quality.

The current named catalog lives at
`apple_pick_sim/fixtures/digital_twin_fixture_catalog.json`. `straight_rod_test`
is the deterministic reconstruction baseline, and `example_variance` matches the
default gym/sys-ID fixture. Catalog entries are checked by
`apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets`,
which verifies that paths exist, referenced range/observation files load, base
poses agree with the range fixture `args`, and smoke commands use `uv run`.

## Sim-to-sim smoke commands

Collect a short observation-only dataset without privileged simulator snapshots
(this is the default; omit `--save-snapshot`):

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

## Relation to current code

| Module/doc | Current role | Next responsibility |
|------------|--------------|---------------------|
| `apple_pick_sim/system_id/trajectory_store.py` | Writes Parquet frames plus optional privileged `.npz` snapshots | Store enough initial observations and metadata to run without `.npz` |
| `apple_pick_gym/envs/apple_pick_replay_env.py` | Replays recorded actions, restoring privileged snapshots when present | Add an observation-derived initialization path when snapshots are absent |
| `apple_pick_gym/examples/example_gym_replay.py` | Reports dataset-vs-live replay errors | Add a snapshot-withheld mode and drift summary for observation-only replay |
| `docs/sysid-trajectory-storage.md` | Dataset storage contract | Keep the required observation bundle synchronized with this document |
| `docs/gym-observation-contract.md` | Runtime observation schema | Bump schema only when observation keys or semantics break |

## Tests and verification

Existing checks for the shipped pieces:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_trajectory_store.py \
  apple_pick_gym/tests/test_replay_env.py -q
```

Expected tests for M3.0.3/M3.0.4:

- dataset fixture with no `.npz` snapshot resets through the observation initializer;
- privileged replay and observation-only replay run the same recorded `action` sequence;
- drift metrics are reported for TCP/apple/woody/F/T observations;
- a sim-generated digital-twin fixture can rebuild topology and base poses from metadata;
- fixture catalog references are covered by `test_fixture_catalog_references_existing_assets`;
- per-fixture sys-ID smoke can use catalog `smoke_commands`; adding a named fixture selection CLI remains separate from the current manifest check.
