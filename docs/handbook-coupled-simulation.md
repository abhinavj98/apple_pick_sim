# Coupled simulation handbook

This is the canonical living reference for the MuJoCo + VBD coupled fruiting
architecture, its public single- and multi-environment APIs, and the batched
settle → weld → teleop flow. Sequencing and milestone status belong in
[`ROADMAP.md`](ROADMAP.md).

## Document status

| Field | Value |
| --- | --- |
| Last reviewed | 2026-08-14 |
| Code owners | `apple_pick_sim/coupled_fruiting/`; `apple_pick_sim/fruiting_system/`; `apple_pick_sim/robot/fr3_robot/`; `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py` |
| Status | Living handbook — defer sequencing to `docs/ROADMAP.md` |
| Related handbooks | H2 [`handbook-variable-impedance.md`](handbook-variable-impedance.md) (VIC inside the coupled step); H3 [`handbook-sysid-scoring.md`](handbook-sysid-scoring.md) (bags produced by collect/replay); H4 [`handbook-real-replay.md`](handbook-real-replay.md) (rebuild, settle, and logged-pose weld) |
| Archive specs | **Implemented / historical:** [`2026-07-03-batched-heterogeneous-build-design.md`](superpowers/specs/2026-07-03-batched-heterogeneous-build-design.md), [`2026-07-03-batched-heterogeneous-coupled-sim-design.md`](superpowers/specs/2026-07-03-batched-heterogeneous-coupled-sim-design.md), [`2026-07-03-batched-heterogeneous-example-design.md`](superpowers/specs/2026-07-03-batched-heterogeneous-example-design.md), [`2026-07-03-batched-gpu-hot-path-design.md`](superpowers/specs/2026-07-03-batched-gpu-hot-path-design.md), [`2026-07-03-v32-close-out-design.md`](superpowers/specs/2026-07-03-v32-close-out-design.md), [`2026-07-03-pre-gym-scope-narrowing-design.md`](superpowers/specs/2026-07-03-pre-gym-scope-narrowing-design.md), [`2026-07-04-batched-gym-base-env-design.md`](superpowers/specs/2026-07-04-batched-gym-base-env-design.md); **Implemented:** [`2026-08-04-true-tcp-pose-weld-design.md`](superpowers/specs/2026-08-04-true-tcp-pose-weld-design.md), [`2026-08-07-pre-grasp-apple-orientation-design.md`](superpowers/specs/2026-08-07-pre-grasp-apple-orientation-design.md), EE alignment slice 0 in [`2026-08-13-real-sim-cma-feature-alignment-design.md`](superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md); **Superseded in part:** [`2026-08-04-tcp-tip-flange-geometry-design.md`](superpowers/specs/2026-08-04-tcp-tip-flange-geometry-design.md) (tool length), [`2026-08-05-apple-position-only-post-grasp-weld-design.md`](superpowers/specs/2026-08-05-apple-position-only-post-grasp-weld-design.md) (default orientation workaround) |

## 1. Purpose and non-goals

The fruiting system cannot share one Newton `Model` with the arm:

- `SolverMuJoCo` owns the articulated FR3 but does not support
  `JointType.CABLE`.
- `SolverVBD` owns the cable rods, fixed fruiting joints, apple, contacts, and
  gripper proxy.

The shipped design therefore uses two `Model` objects in one Newton/Warp
process. They exchange state and a spatial wrench through proxy bodies; there
is no separate MuJoCo process and no cross-model joint.

This handbook does not reproduce the fixed-joint wrench derivation, material
formulas, or controller mathematics. Those remain in the linked satellite
documents and H2. It also does not own delivery status; consult
[`ROADMAP.md`](ROADMAP.md).

## 2. Model ownership and coupling lag

| Quantity | Model A: robot | Model B: plant |
| --- | --- | --- |
| Solver | `newton.solvers.SolverMuJoCo` | `newton.solvers.SolverVBD` |
| Bodies | FR3, EE/TCP, optional mass-only `apple_payload` | Primary/secondary/spur/stem rods, apple, gripper proxy, ground |
| Gravity | Zero, matching ideal arm-only gravity compensation | Normally `(0, 0, -9.81)` m/s² |
| Pose authority | MuJoCo owns FR3 and TCP `body_q` / `body_qd` | VBD owns rods and a free apple; robot TCP is mirrored onto the proxy and, after weld, the prescribed apple |
| Collision authority | Optional robot contacts only | Fruiting, apple, proxy, and ground contacts |
| Cross-model channel | Receives lagged plant wrench in TCP `body_f` | Receives TCP pose/twist through `ProxyBodyRegistry` |

`CoupledFruitingScene.coupled_substep(dt)` is an explicit staggered exchange:

```text
substep n
  1. copy proxy_forces[n-1] to coupling_forces_cache and robot TCP body_f
  2. step MuJoCo
  3. mirror TCP pose/twist to proxy (and welded apple)
  4. step VBD
  5. harvest the plant reaction into proxy_forces[n] for substep n+1
```

The wrench therefore has a one-substep lag. `coupling_forces_cache` preserves
the exact lagged value used by MuJoCo so proxy velocity correction can avoid
double integration. `align_proxy_body_q_prev_for_vbd` aligns VBD history after
the kinematic pose overwrite.

With an apple, harvest reads the stem–apple fixed-joint reaction. Without an
apple, the fallback reconstructs proxy reaction from its VBD velocity change.
`ProxyBodyRegistry` and `BatchedEnvLayout` map each robot TCP to the matching
plant proxy; worlds never exchange state or wrench.

## 3. Public API

### Package-level batched API

The supported batched entry points are exported by
`apple_pick_sim.coupled_fruiting`:

| Symbol | Responsibility |
| --- | --- |
| `BatchedHeterogeneousCoupledSimConfig` | Composes runtime, FR3, settle/collision, domain-randomization, fruiting, controller, MuJoCo, diagnostics, and observation configuration. `validate()` enforces cross-field contracts. |
| `BatchedHeterogeneousCoupledSim` | Builds the scene and exposes `build`, `step`, `gather_obs`, `scene`, and `layout`. |
| `build_batched_heterogeneous_scene` | Config-driven free settle and optional welded rebuild. |
| `settle_vbd_substeps`, `quiet_all_cable_bodies` | Parallel VBD settling and quiescence helpers. |
| `seed_fix_to_apple_from_settled`, `seed_fix_to_apple_from_settled_body_q` | Copy settled cable state into a welded build and bootstrap the FR3. |
| `SettledCheckpoint`, `settle_cache_path_for` | Optional settled-state disk cache. |
| `CoupledFruitingScene` | Low-level two-model scene; `coupled_substep` is the authoritative physics step. |

`RobotConfig.kind` must be `"fr3"`. Coupled builds require the FR3 assets under
`assets/fr3/` and the resolved USD asset; missing assets fail rather than
falling back to a placeholder robot. `defaults()` and `gym_defaults()` use VIC.
`test_minimal()` is the CPU-oriented direct-control preset.

Constructing `BatchedHeterogeneousCoupledSim` performs the build immediately;
the `build` classmethod is a convenience alias. `step(actions)` advances one
control interval, and `gather_obs()` uses the allocated batched observation
buffers.

### Low-level builders

Import these explicitly from `apple_pick_sim.coupled_fruiting.builders`; they
are intentionally not package-level exports:

| Builder | Use |
| --- | --- |
| `build_coupled_fruiting_fr3` | One FR3 + one fruiting system, sampled from a seed. |
| `build_heterogeneous_coupled_fruiting_fr3` | Multiple worlds with per-environment `FruitingSystemParams` and common topology. |

Removed placeholder and old `build_batched_*` entry points are not supported.

## 4. Settle → weld → teleop

The proxy–apple fixed joint is build-time topology, so a stable post-grasp run
uses two builds:

1. **Build free:** set `fix_to_apple=False`; construct the plant worlds and,
   when needed, FR3 worlds.
2. **Settle:** call `settle_vbd_substeps`; optionally ramp gravity, periodically
   quiet twists, then quiet all cable bodies.
3. **Build welded:** reconstruct the same worlds with `fix_to_apple=True` and
   copy settled world \(i\) into welded world \(i\).
4. **Bootstrap and run:** align each FR3 TCP to its proxy, then run VIC, direct
   control, replay, or policy actions through repeated `coupled_substep` calls.

`BatchedHeterogeneousCoupledSim` and `build_batched_heterogeneous_scene` own
this flow for the canonical batched path. `settle_gravity_ramp` defaults off;
periodic quiet defaults to every 300 settle substeps. Settle length is
configuration-owned rather than a handbook constant.

The settled-state disk cache is **off by default**:
`BatchedHeterogeneousCoupledSim(..., use_settle_cache=False)`. Opt in with the
constructor argument or `--use-settle-cache`; cache loading validates the
configuration, ranges, and per-env parameters.

Single-env real replay uses the same state transition but may supply measured
apple/TCP poses and an explicit weld offset through H4's post-grasp plan.

### Co-located physics and viewer spacing

All batched worlds use the same simulation-frame origin. Homogeneous builds
call `ModelBuilder.replicate(N, spacing=(0, 0, 0))`; heterogeneous builds add
independent worlds without translating their physical state. Newton world
indices prevent cross-world collisions, and MuJoCo uses `separate_worlds=True`
for multiple worlds.

`RuntimeConfig.env_spacing` / `--env-spacing` controls viewer world offsets,
not plant or robot physics. Keeping the cable and robot for world \(i\)
co-located is required by TCP mirror and wrench harvest. Do not physically
space one model and leave the other at the origin.

## 5. Homogeneous and heterogeneous batches

| Contract | Homogeneous batch | Heterogeneous batch |
| --- | --- | --- |
| Build mechanism | One template plus `replicate(N)` | One `add_world` per `FruitingSystemParams` |
| Parameters | Same sampled values in every world | Different numeric material, mass, and geometry values per world |
| Topology | Shared | Still shared: enabled rods and segment counts must yield identical array structure |
| Settle seed | World \(i\) → welded world \(i\) | Same |
| FR3 weld bootstrap | Template/world-0 result may be broadcast | `BatchedTemplateIK` solves one row per environment and scatters; no world-0 joint broadcast |
| Runtime actions | May intentionally repeat one command | Device action buffer supports distinct per-env commands |

`sample_heterogeneous_params_list` freezes topology from a topology seed while
sampling continuous parameters per environment. `BatchedEnvLayout` records
per-world body and joint indices. Build-time parameter derivation and valid DR
ranges are documented in
[`material-parameter-sampling.md`](material-parameter-sampling.md) and
[`damping-tuning.md`](damping-tuning.md).

Default builds set `enable_self_collisions=False`: woody↔woody contacts are
filtered; apple↔woody and proxy↔woody stay on; ground is unchanged. See
`apple_pick_sim/fruiting_system/build.py::_apply_default_fruiting_collision_filters`.

The batch cannot mix different body counts. Rebuild-free geometry
randomization and other future work are ROADMAP-owned; this handbook does not
claim them as shipped.

## 6. Wrench, payload, and TCP harvest

Spatial coupling buffers use world-frame `[linear, angular]` values in N and
N·m. The implemented stem-harvest sign is the **child-side** fixed-joint
reaction written to `proxy_forces[tcp]` **without negation**. Welded builds add
the child-side apple support term `-m_apple * gravity` and its moment about the
TCP before gain and caps. Free builds keep that explicit term off to avoid
double-counting VBD gravity.

The code defaults are:

- `DEFAULT_STEM_COUPLING_GAIN = 1.0`;
- `DEFAULT_STEM_FORCE_CAP_N = 40.0`;
- `DEFAULT_STEM_TORQUE_CAP_NM = 10.0`.

Model A remains zero-gravity. Welded FR3 builds (`GripperProxyConfig.fix_to_apple=True`)
add a mass-only FIXED child of the TCP labeled `apple_payload`. Quasi-static fruit
weight still enters via stem harvest (`-m · g`). The dummy supplies **reflected
inertia** only:

\[
m = m_{\mathrm{AVBD\,apple}},\quad
I = \tfrac{2}{5} m r^{2}\,\mathbf{1},\quad
\mathbf{c}_{\mathrm{TCP}} = \mathrm{trans}\big(X_{\mathrm{offset}}^{-1}\big)
\]

so \(X_{\mathrm{apple}} = X_{\mathrm{tcp}} \cdot X_{\mathrm{offset}}^{-1}\) matches
co-teleport / explicit-load COM placement. There is no MuJoCo sphere geom (AVBD
owns collision radius). Heterogeneous worlds share one replicated topology;
per-world `body_mass` / `body_inertia` / `body_com` are patched then synced with
`notify_model_changed(BODY_INERTIAL_PROPERTIES)`. Helpers live in
`coupled_fruiting/mujoco_apple_payload.py`.

Use these focused references for the detailed contracts:

- [`WRENCH_READOUT.md`](WRENCH_READOUT.md) — VBD fixed-joint readout frames,
  points, and sign.
- [`explicit-apple-load-tcp-harvest.md`](explicit-apple-load-tcp-harvest.md) —
  child-side support force/moment and welded/free defaults.
- H2 [`handbook-variable-impedance.md`](handbook-variable-impedance.md) —
  controller wrench and joint-torque paths; controller effort is not fed back
  as the next plant wrench.

## 7. TCP, flange, apple orientation, and weld geometry

These are the implemented outcomes; dated specs retain the design history.

### Tool and TCP geometry

`apple_pick_sim.robot.fr3_robot.paths` and
`fruiting_system.gripper_proxy_shape` agree on this contract:

| Quantity | Implemented value / meaning |
| --- | --- |
| TCP origin | Center of the distal cylinder tip face |
| TCP local +Z | Tip-out, toward the apple at a normal grasp |
| Tool bulk | Extends from TCP toward the flange along local −Z |
| Cylinder radius | `0.05` m |
| Cylinder half-height | `0.09` m |
| Total tool length | **0.18 m (180 mm)** |
| FR3 EE/TCP orientation | `EE_TCP_ORIENT_WXYZ = (0, 1, 0, 0)`, RotX(180°) |

The 140 mm / `hh=0.07` table in the August 4 geometry spec is superseded by
alignment slice 0 and current code. Proxy geometry is centered at
`(0, 0, -hh)`, keeping the distal face coincident with the TCP.

### EE mass properties

The measured alignment constants live beside the geometry constants in
`robot/fr3_robot/paths.py` and are regression-tested against both USDA assets:

| Property | Value |
| --- | --- |
| `EE_MASS_KG` | `1.1` kg |
| COM in flange frame | `(0, 0, 0.077)` m |
| COM in EE-local frame after RotX(180°) | `(0, 0, -0.077)` m |
| `EE_INERTIA_DIAG_KGM2` about COM | `(0.00215219194, 0.00215219194, 0.00119125005)` kg·m² |

Mass, COM, and `I_ee` belong on `/fr3/ee`; the TCP remains the wrench
application point and does not carry those properties.

### Generic and real-replay welds

Generic `robot_facing_weld=True` builds choose a robot-facing apple surface pole
and construct a tip-out look-at orientation. They require
`fix_to_apple=True` and `robot_base_pos`.

Real post-grasp replay instead uses the logged TCP and apple SE(3). It computes
the apple-frame transform
`X_offset = inverse(X_apple) * X_tcp` and stores it as
`GripperProxyConfig.weld_proxy_offset_in_apple_frame`; the fixed joint is built
from that relative pose. No catalog-radius surface snap or look-at quaternion
replaces the measured poses.

`FruitingSystemParams.apple_quat_xyzw` seeds the free build from the logged
pre-grasp apple orientation, so the stem–apple child anchor and post-grasp
tracker frame are consistent. Full logged apple SE(3) is the default after
grasp. Position-only apple orientation remains an explicit diagnostic escape
hatch, not the canonical path. See H4
[`handbook-real-replay.md`](handbook-real-replay.md) for replay orchestration.

## 8. GPU hot path

On CUDA, the steady-state path keeps MuJoCo, VBD state, TCP mirror, wrench
application, batched stem harvest, and VIC torques on device. Build-time
sampling, IK/bootstrap setup, keyboard callbacks, checkpoints, and debug
readouts may synchronize on the host.

`BatchedHeterogeneousCoupledSim.step()` uses a device action buffer; batched
mirror and harvest kernels launch once over the registry/layout and reuse
build-time arrays. There must be no `.numpy()` / `.cpu()` round-trip in
`coupled_substep`, batched VIC apply, or normal action-to-teleop hot paths.

Detailed kernel ownership, device defaults, benchmarks, and resolved gaps live
in:

- [`gpu-coupling-optimization.md`](gpu-coupling-optimization.md);
- [`heterogeneous-batched-vectorization-audit.md`](heterogeneous-batched-vectorization-audit.md).

The default simulation device is CUDA when available; `--device cpu` and
`APPLE_PICK_SIM_DEVICE=cpu` are supported for debugging and tests.

## 9. Code map

| Module / symbol | Responsibility |
| --- | --- |
| `fruiting_system/coupled.py::generate_coupled_cable_scene` | Build Model B and its proxy registry. |
| `fruiting_system/build.py::_add_gripper_proxy` | Proxy shape, apple weld, explicit relative-pose and look-at paths. |
| `fruiting_system/params.py::{FruitingSystemParams, GripperProxyConfig}` | Apple orientation and proxy/weld configuration. |
| `coupled_fruiting/scene.py::CoupledFruitingScene.coupled_substep` | Authoritative staggered loop and coupling defaults. |
| `coupled_fruiting/proxy_coupling.py` | Mirror, VBD-history alignment, reaction harvest, and batched harvest kernels. |
| `coupled_fruiting/apply_wrench.py` | Registry-indexed TCP `body_f` writes. |
| `coupled_fruiting/builders.py` | Single and heterogeneous FR3 builders; explicit-load and payload setup. |
| `coupled_fruiting/mujoco_apple_payload.py` | Welded TCP `apple_payload` mass, COM, and spherical inertia. |
| `coupled_fruiting/batched_heterogeneous_build.py::build_batched_heterogeneous_scene` | Config-driven settle/weld build. |
| `coupled_fruiting/batched_heterogeneous_coupled_sim.py::BatchedHeterogeneousCoupledSim` | Public batched runtime. |
| `coupled_fruiting/batched_heterogeneous_config.py::BatchedHeterogeneousCoupledSimConfig` | Public configuration and presets. |
| `coupled_fruiting/settle_then_weld.py` | Free settle, quiet, state seed, and FR3 bootstrap. |
| `coupled_fruiting/batched_layout.py::BatchedEnvLayout` | Per-world body/joint/TCP/proxy/payload indices. |
| `robot/fr3_robot/batched_template_ik.py::BatchedTemplateIK` | Per-env FR3 IK and scatter. |
| `robot/fr3_robot/paths.py` | Tool geometry and measured EE mass properties. |
| `system_id/real_post_grasp_plan.py` | Logged full-SE(3) post-grasp weld plan. |
| `examples/example_batched_heterogeneous_coupled_sim.py` | Canonical thin batched CLI. |

Additional physical context and fitting guidance:

- [`real-world-proxy.md`](real-world-proxy.md);
- [`material-parameter-sampling.md`](material-parameter-sampling.md);
- [`damping-tuning.md`](damping-tuning.md).

## 10. Tests and verification

The primary regression groups are:

- `test_coupled_fruiting_system.py`, `test_proxy_coupling.py` — staggered
  ordering, sync, harvest, signs, and caps.
- `test_settle_then_weld.py`, `test_settled_checkpoint.py` — free settle,
  welded seed, quiescence, and cache.
- `test_batched_heterogeneous_{config,build,coupled_sim}.py`,
  `test_heterogeneous_coupled_fruiting.py` — public batched API and per-env
  parameters/actions.
- `test_vectorized_coupled_fruiting.py`, `test_batched_stem_harvest.py` —
  registry/layout vectorization.
- `test_ee_cylinder_geometry.py`, `test_real_post_grasp_plan.py`,
  `test_real_pre_grasp_params.py` — TCP geometry, EE inertials, logged pose, and
  apple-frame continuity.
- `test_explicit_apple_load.py`, `test_mujoco_apple_payload.py` — support wrench
  and reflected inertia.

Run from the repository root:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_heterogeneous_build.py \
  apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_batched_heterogeneous_config.py \
  apple_pick_sim/tests/test_example_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py -q

uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_settle_then_weld.py \
  apple_pick_sim/tests/test_settled_checkpoint.py -q

uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 200 --num-envs 4 --settle-substeps 100 --seed 42
```

For the repository-wide fast gate, use the canonical ROADMAP command:

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"
```
