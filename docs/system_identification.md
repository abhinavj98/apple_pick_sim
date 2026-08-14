# System identification protocol

This document preserves the experiment protocol and its intent. It does not
define trajectory bags, score-vector fields, normalization, Young's-modulus
candidate semantics, or delivery status.

## Document status

| Field | Value |
| ----- | ----- |
| Last reviewed | 2026-08-14 |
| Status | Protocol overview — defer sequencing to `docs/ROADMAP.md` |
| Bag and scoring contract | H3 `docs/handbook-sysid-scoring.md` |
| Grid and CMA implementation | H5 `docs/handbook-youngs-cma.md` |
| Real conversion and replay | H4 `docs/handbook-real-replay.md` |

## 1. Objective

Calibrate a tunable VBD apple-branch model against replayable observations from
simulation and, ultimately, the physical robot. The plant is a topological
network of primary branch, secondary branch, spur, stem, and apple. Material
parameters are expressed as physical Young's modulus \(E\) and damping ratio
\(\zeta\); simulation stiffness and damping are derived from them as described
in `docs/material-parameter-sampling.md`.

The calibration protocol must:

1. collect actions and observations under repeatable excitation;
2. reconstruct the candidate simulator without privileged dynamic arrays;
3. replay the same recorded actions for every candidate;
4. compare candidate and target trajectories under the H3 scoring contract;
5. establish sim-to-sim recovery before trusting real-data optimization; and
6. evaluate improvement on held-out structures, directions, or trajectory
   segments before turning fitted values into domain-randomization policy.

Observation-only initialization and fixture reconstruction are described in
`docs/digital-twin.md`. Current milestone order and acceptance status belong
only in `docs/ROADMAP.md`.

## 2. Excitation protocol

Excitation should cover multiple deformation modes while the robot maintains
its grasp. Stay in the elastic regime and use an online wrench guard derived
from the cultivar/session rather than assuming a fixed displacement is safe.
Bookend a physical collection session with repeated reference sweeps so drift
from temperature, moisture, or prior disturbance is observable.

### 2.1 Quasi-static stepped mapping

Run this first. It maps stiffness-dominated behavior and establishes safe
amplitudes for dynamic excitation.

- Sample forward-looking pull directions with the Fibonacci/golden-ratio
  hemisphere implementation in
  `apple_pick_sim/system_id/fibonacci_hemisphere.py`.
- For each direction, alternate a fast displacement increment and a 1–2 second
  hold. Return or restore to the grasp center before the next direction.
- Use hold settling, not an artificially slow crawl, to suppress velocity and
  inertia effects.
- Record the applied action, TCP/apple motion, F/T, tracked woody geometry,
  phase, hold identity, direction, and stability state required by H3.

`apple_pick_sim/system_id/quasi_static_trajectory.py` owns the phase machine.
`example_batched_collect_sysid_data.py` is the current parallel collection
entry point; the older `example_gym_sysid.py` remains a single-env diagnostic.

### 2.2 Multi-axis logarithmic chirps

This is future dynamic-identification protocol, not the current hold-bag
optimizer input.

- Reuse representative safe directions from the quasi-static map.
- Sweep continuously on a logarithmic frequency axis so low- and
  high-frequency modes receive useful dwell.
- Scale displacement amplitude approximately as \(1/f\) to keep velocity and
  high-frequency loads bounded.
- Repeat selected directions at multiple amplitudes to expose nonlinear
  stiffness.
- Reserve discrete steady-state tones for held-out validation and
  interpretation rather than the primary identification trajectory.

FRF plots from FFT/Welch analysis are useful diagnostics, but they do not
replace the replay objective.

### 2.3 Torsional excitation

Hold translational displacement near zero and excite rotations about the
stem-local twist axis plus orthogonal pitch/yaw axes:

1. quasi-static angular sweeps with holds for rotational stiffness; then
2. rotational logarithmic chirps for damping and resonance.

Use the quasi-static wrench envelope to choose safe angular amplitudes.

## 3. Replay and scoring boundary

Every candidate must replay the source episode's recorded action sequence at
the recorded control rate. Do not regenerate a nominal trajectory during
optimization; command mismatch would be scored as plant mismatch.

The three data layers must stay distinct:

- runtime observations used by gym/simulation and reconstruction;
- `batched_sysid_v1` trajectory bags used for replay; and
- H3 score vectors and transition bags.

The canonical field list, dimensions, frame transforms, fixed physical scales,
hold/direction pooling, and action-exclusion rule live exclusively in
`docs/handbook-sysid-scoring.md`. Real conversion and `vic_pose` replay live in
`docs/handbook-real-replay.md`. This protocol intentionally carries no copied
feature table.

## 4. Calibration path

The current phenotype, Cartesian grid, fused replay scheduler, CMA ask/tell
loop, failure policy, reports, gates, and runnable commands are documented in
`docs/handbook-youngs-cma.md`.

At protocol level:

1. use a Cartesian search to diagnose identifiability and ranking behavior;
2. run continuous optimization only after the ranking path is credible;
3. keep target parameters out of initialization and fitness, using them only
   for sim-to-sim evaluation;
4. replay and score the optimizer's final distribution mean explicitly;
5. inspect per-direction residuals, instability, and parameter covariance
   rather than trusting one aggregate value; and
6. validate on held-out data before changing simulation priors.

For real data there is no simulator-oracle phenotype to insert into the
candidate set. A real replay that builds and steps successfully proves
plumbing, not trustworthy ranking or calibration. See ROADMAP for the open
ranking and CMA acceptance work.

## 5. Verification entry points

Quasi-static protocol utilities:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_quasi_static_sysid.py \
  apple_pick_sim/tests/test_visualize_pull_directions.py \
  apple_pick_gym/tests/test_sysid_env.py -q
```

Current bag/scoring contract:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_mmd.py \
  apple_pick_sim/tests/test_wasserstein.py \
  -q -p no:launch_testing
```

Use H5 for collect → grid → CMA commands and the expensive ranking/CMA gate
scripts. Use H4 for real parquet conversion and replay commands.
