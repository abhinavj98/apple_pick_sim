# Real-world bench proxy — placement, geometry, and fixture decisions

## Purpose

This document records **decisions** for aligning the apple-pick simulation with the
physical bench proxy: robot placement, branch layout, segment topology, stiffness
tiers, end-effector geometry, and observation-frame conventions. It is the spec
for implementing:

- `apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy.json` (nominal)
- `apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json` (DR)
- Catalog entry `real_world_proxy` in `digital_twin_fixture_catalog.json`

**Goals:**

1. Eliminate FR3 IK bootstrap failures caused by inconsistent placement (tree too
   high, arm auto-parked away from fixture `robot_base_pos`).
2. Match the physical proxy layout closely enough for M3 sys-ID and digital-twin
   work.

Related docs: `docs/digital-twin.md`,
`docs/gym-observation-contract.md`, `docs/system_identification.md`.

---

## Coordinate conventions

The simulation world frame matches `docs/WRENCH_READOUT.md`:

| Axis | Direction |
|------|-----------|
| X | right |
| Y | forward |
| Z | up |

Gravity: `(0, 0, −9.81)` m/s².

**Robot base** is at the world origin with identity root orientation:

```text
robot_base_pos = (0, 0, 0)   # FR3 USD root translation
```

**Observation frame policy:** Parquet and gym observations remain stored in
**world frame** (`tcp_pos`, `apple_pos`, woody endpoints, `action`,
`tcp_velocity`, `excitation_direction`, `ft_wrist`). With the robot root fixed at
the origin and no root rotation, **world frame equals robot-base frame** for
positions and world-frame twists. The doc and fixture metadata must state this
explicitly; no `obs_schema` change is required for this slice.

Episode metadata continues to record `robot_base_pos` and `fruiting_base_pos` for
replay and digital-twin calibration (`docs/sysid-trajectory-storage.md`).
For this fixture, `fruiting_base_pos` is the **T center** (mid-span spur junction),
not a cantilever root.

---

## Bench layout

### Placement

| Quantity | Value | Notes |
|----------|-------|-------|
| `robot_base_pos` | `(0, 0, 0)` m | Fixed; **do not** use `robot_base_from_proxy` for this fixture |
| `fruiting_base_pos` | `(0, 0.5, 0.95)` m | **T center** — mid-span spur attach on the primary; physical offset **(0, 50, 95) cm** from robot base |
| Robot reach / forward | **+Y** | Arm approaches apple along +Y |
| Primary branch axis | **±X** | Horizontal beam through the T center; `azimuth_deg = 0`, `elevation_deg = 0` |
| Primary span | **0.10 – 0.20** m total | Extends **±L/2** from `fruiting_base_pos` along **±X** |
| Side supports | Both ends | World-fixed clamps at the **left and right** primary endpoints (not a single cantilever pin) |
| Spur attach | Mid-span | Vertical shoot from the **center** of the primary (T junction), not from a primary tip |

### T-junction topology (target)

The physical bench holds the branch at **both ends**; the spur and apple hang from the **middle**:

```text
                         +Z
                          |
              side        |        side
             support      |       support
                │         │         │
    −X ◄════════●═════════●═════════●════════► +X
              left      T center    right
                          │
                          │  spur (nominal hang ≈ −Z)
                          ▼
                       stem → apple

              robot base @ origin
                          │
                          v +Y   (reach toward apple)
```

**Convention:** `fruiting_base_pos` is the **T center** (world position of the mid-span
`primary_spur` junction), not the left support or a cantilever root. Primary rod
geometry is centered on that point; side supports sit at `fruiting_base_pos ± (L/2, 0, 0)`.

**Default topology:** T-junction is the **default** fruiting builder mode when fixture
JSON omits `"topology"`. Legacy serial chains opt in with `"topology": "linear_chain"`.
Implemented in `apple_pick_sim/fruiting_system/build.py` (`_build_t_junction_into_builder`).
World support joints are finalized in **separate articulations** (same pattern as the
gripper proxy FREE joint) so endpoint bodies are not multi-parented in one tree.

### Arm placement rule

For `real_world_proxy`, the FR3 root is placed only via `robot_base_pos` in the
fixture `args` block (or an explicit build override). The sys-ID path must **not**
set `robot_base_from_proxy=True` when using this fixture, so IK targets a
consistent base at the origin instead of auto-parking ~0.8 m below the gripper
proxy.

When this fixture is selected, builders and envs should prefer these `args` over
legacy `COUPLED_ROBOT_BASE_POS` / `COUPLED_BASE_POS` in
`apple_pick_sim/coupled_fruiting/defaults.py`.

---

## Physical proxy ↔ simulation mapping

The physical proxy is described as four rigid links (branch, bourse, peduncle,
apple) with ball-socket and magnetic breakaway joints, plus **two bench clamps**
that pin the branch at both ends. The simulation uses **three deformable rods** plus
apple on a **T-shaped** topology; the bourse is not modeled as a separate segment.

| Physical component | Sim segment | Decision |
|--------------------|-------------|----------|
| Primary branch (simply supported) | `primary` | Horizontal rod **±L/2** from `fruiting_base_pos` along **±X** |
| Bench clamps (left / right) | world FIXED joints | Rigid supports at primary **endpoint** bodies (`parent = −1`) |
| Bourse (stem base / socket housing) | *(none)* | **Ignored** — treated as negligible compliance |
| Spur (shoot from branch center) | `spur` | Attaches at **mid-span** on primary (`joint_primary_spur`), hangs toward **−Z** |
| Peduncle / abscission layer | `stem` | Magnetic detach behavior — see torsion follow-up below |
| Apple | `apple` | Sphere at stem tip |

**Sim topology (target):** `"secondary": null`; primary forms the **crossbar** of the T;
spur → stem → apple form the **stem** of the T. Not a single serial chain from one
primary tip.

```text
  [world]──FIXED── primary endpoint … primary mid ──FIXED──[world]
                              │
                         FIXED (joint_primary_spur)
                              │
                            spur → stem → apple
```

**Junction labels** (woody observations and digital twin):

| Label | Role |
|-------|------|
| `primary_support_left` | World-fixed clamp at primary **−X** endpoint |
| `primary_support_right` | World-fixed clamp at primary **+X** endpoint |
| `primary_spur` | Mid-span branch; spur base welded to primary center body |
| `spur_stem` | Spur tip → stem base |
| `stem_apple` | Stem tip → apple pole |

Wrench subtree cuts differ from the legacy linear chain: `joint_primary_spur` supports
only spur + stem + apple mass; each support joint carries a share of primary + branch
load (not a simple serial subtree cut — see `docs/WRENCH_READOUT.md` §5.1).

### Bench ArUco ↔ sim junctions (real episodes)

Markers stay on **visible surfaces**; compile / ingest applies the known
**real-world offset** so stored junctions match **sim centerline / CoM** frames
(see `docs/real-sysid-pre-post-grasp-fixes.md` and
`apple_pick_sim/system_id/real_pre_grasp_params.py`).

| Tracked point | Meaning (robot frame) | Sim use |
|---------------|------------------------|---------|
| **Spur-start ArUco** | **Dowel–spur junction** on the primary **surface** (visible marker) | Ingest derives **`fruiting_base_pos`** on the primary **centerline**; spur **start** when `spur_surface_offset` is on (default) |
| **Spur-end ArUco** | **Spur–stem junction** on the spur **centerline** | Spur **end** = **stem start** |
| **Apple markers** | Apple **CoM** | `apple_pos` |

There is **no separate dowel ArUco** — the spur-start tag is the dowel–spur junction on the
dowel **surface**.

**Ingest:** ignore any stored `fruiting_base_pos` in parquet metadata. Derive the sim
T-center from the spur-start **surface** junction:

```text
radial_hat = normalize(spur_dir − (spur_dir·primary_axis)·primary_axis)
fruiting_base_pos = spur_start_surface − r_primary · radial_hat
```

**Build:** `spur_surface_offset` defaults to **`true`**. The sim re-applies
`+r_primary · radial_hat` at the primary→spur joint so the spur rod grows from the
**surface** (round-trip with the marker). Set `"spur_surface_offset": false` in a
fixture to restore legacy centerline attach.

```text
dowel–spur junction  = spur start   (spur-start ArUco, offset)
spur–stem junction   = spur end     (spur-end ArUco, offset)
                     = stem start
û = normalize(apple_CoM − stem_start)
stem_end             = apple_CoM − r_apple · û   # toward spur–stem junction
```

**Catalog `parts.spur.length_m`:** true measured axis length from **dowel–spur junction**
→ **spur–stem junction** (surface start → stem–spur junction). Do **not** inflate by
primary radius — sim `spur_surface_offset` (default on) handles surface attach at build time.

---

## Nominal geometry (proxy fixture)

Values below are **targets** for `fruiting_system_ranges_real_world_proxy.json`.
Variance bounds live in the companion `*_variance.json` file.

| Segment | Parameter | Nominal / range | Unit |
|---------|-----------|-----------------|------|
| Primary | `length` | **0.10 – 0.20** (full span; centered on T) | m |
| Primary | `azimuth_deg` | 0 (fixed in nominal) | deg |
| Primary | `elevation_deg` | 0 (fixed in nominal) | deg |
| Primary | `attach_mode` | `t_junction_centered` (default builder) | — |
| Primary supports | `both_ends` | World FIXED at endpoint bodies | — |
| Spur | `attach_along_primary` | **0.5** (mid-span via `spur_attach_fraction`) | fraction |
| Spur | `length` | 0.01 – 0.10 | m |
| Spur | nominal hang | `elevation_delta_deg` ≈ **−90°** from primary tangent → world **−Z** | deg |
| Spur | lateral variation | `lateral_delta_deg` — see variance fixture | deg |
| Stem | `length` | **0.01 – 0.06** | m |
| Apple | `radius` | 0.04 – 0.08 | m |

Optional top-level fixture keys: `"topology"` (`t_junction` default, or `linear_chain`);
`"spur_attach_fraction"` (default **0.5**, fixed in variance for `real_world_proxy`).

Segment `num_segments`, `radius`, and rod `density` use proxy-appropriate values
in JSON; rod radii should be filled from measurement in a later pass if not yet
available.

---

## Mechanical parameters

### Branch stiffness (proxy spring constants)

Physical proxy targets (spring constant at branch joint):

| Tier | Real apple tree [N/m] | Proxy target [N/m] |
|------|----------------------|-------------------|
| Low | 182 | 210 |
| Med | 414 | 455 |
| High | 711 | 736 |

**Fixture policy:** **continuous tier bands** in variance JSON, not three separate
preset files. Map proxy **210 – 736 N/m** onto **`youngs_modulus_pa`** min/max at
nominal primary geometry (see `docs/material-parameter-sampling.md`); legacy
`primary.bend_stiffness` bands are deprecated.

**Calibration note:** VBD `bend_stiffness` on discretized rods is not literally
N/m. Treat the table as **tier centers**; derive from sampled \(E\) and geometry,
then validate with quasi-static holds and
`apple_pick_gym/examples/run_system_identification.py` grid search before
treating numbers as ground truth. Conversion to a geometry-invariant
Young's modulus `E` (and a derived, geometry-consistent `bend_stiffness` per env),
plus the matching damping-ratio (`ζ`) and density (`ρ`) sys-ID targets and a
numerical stability guard for domain randomization, is shipped — see
`docs/material-parameter-sampling.md` ("Derivation" section).

### Spur stiffness

Vary spur **`youngs_modulus_pa`** (and \(\zeta\)) in the variance fixture (short
segment, compliant shoot). Exact N/m mapping is TBD; keep order-of-magnitude below
primary and above stem unless sys-ID dictates otherwise.

### Stem detach / magnet strength

Physical proxy magnet-holding targets (stem / abscission):

| Tier | Real stem strength [N] | Proxy magnet [N] |
|------|------------------------|------------------|
| Low | 8.1 | 9.8 |
| Med | 15.8 | 16.4 |
| High | 26.6 | 26.6 |

**Decision:** Use these magnet values to tune **stem torsional stiffness** at the
peduncle / abscission layer (twist-wedge weakening in hardware).

**Follow-up slice (not this doc’s implementation):** Expose **torsional stiffness**
(or an equivalent angular constraint parameter) on the stem rod or
`spur_stem` / `stem_apple` junction in Newton/VBD. Until that exists, stem
`bend_stiffness` and force caps are **interim** knobs only; do not claim
magnetic detach parity in validation.

Document in code comments and tests when the torsion API lands.

---

## Variance fixture scope

`fruiting_system_ranges_real_world_proxy_variance.json` should randomize:

- Primary **`youngs_modulus_pa`** and **`damping_ratio`** (branch tier band → \(E\))
- Spur length
- Spur yaw / roll off the nominal **−Z** hang (`elevation_delta_deg`, `lateral_delta_deg`)
- Spur **`youngs_modulus_pa`** / **`damping_ratio`**
- Stem material (\(\zeta\), \(E\); pending torsion API; interim bend-derived range allowed)
- Apple `radius` and `density` (see placeholders)

Material sampling contract: `docs/material-parameter-sampling.md`.

The variance fixture also ships an optional top-level **`sim_build`** block (VIC
gains, `joint_damping_ratio` and/or FIXED-joint kp/kd overrides) consumed by the
batched heterogeneous / sys-ID examples. Prefer `joint_damping_ratio` (ζ) over
hand-tuned absolute kd maps; the two are mutually exclusive. That block is
**not** domain-randomized; it is a stable sim-build snapshot colocated with the
DR ranges. See `docs/material-parameter-sampling.md` (§ Optional top-level
`sim_build`) and `docs/damping-tuning.md`.

Mid-span spur attach fraction stays **0.5** in variance (fixed topology per batch).
Nominal fixture may use fixed angles for reproducible IK smoke tests.

---

## End-effector (FR3 tool)

| Parameter | Value | Status |
|-----------|-------|--------|
| Tool length | 0.14 m (140 mm) | **Specified** |
| Tool radius | 0.10 m (100 mm) | **Specified** |
| TCP location | Distal tip along **+Z** from link7 / `ee` flange | **Specified** |
| Tool mass | See placeholders | **TBD — measure on hardware** |

**Assets:** Update `assets/testfr3_resolved.usda` (cylinder geometry, TCP offset,
mass on `ee` / `tcp`).

**Coupling:** Match `GripperProxyConfig.mass` and `box_half_extents` to the USD
TCP so VBD proxy harvest uses consistent inertia
(`apple_pick_sim/fruiting_system/params.py`).

---

## Placeholder values

The following are explicitly **not final**. Implementation should use documented
stand-ins until measured; replace when data is available and note the change in
fixture `_comment` and this table.

| Item | Placeholder | Replace with | Blocked by |
|------|-------------|--------------|------------|
| EE mass | `PLACEHOLDER_EE_MASS_KG` → use **0.5 kg** in JSON/code until measured | Scale measurement on physical tool | Hardware weigh-in |
| Apple `density` | **700 kg/m³** (mid of legacy variance fixtures) | Measured proxy fruit or water-displacement | Fruit specimen |
| Stem torsional stiffness | Interim `stem.bend_stiffness` band only | Magnet-tier → torsion [N·m/rad] mapping | Follow-up slice: expose torsion in VBD |
| Rod `radius` (primary/spur/stem) | Midpoints from `example_variance` until measured | Caliper / CAD on proxy | Measurement |
| `bend_stiffness` ↔ N/m | JSON bands centered on proxy table | Sys-ID identified coefficients | M3 grid / CEM |
| Rod `density` (`ρ`) | Sampled independently of stiffness; not measured on these specimens | Paired `(E, ρ)` measurement per specimen, or a regime-matched literature range (real wood vs. rigid-link proxy material) | Measurement or literature source — see `docs/material-parameter-sampling.md` ("Derivation" section) |

When editing fixtures, add a `_comment` field listing active placeholders.

**See also:** `docs/material-parameter-sampling.md` ("Derivation" section) — analysis of why
independent sampling of `bend_stiffness`/`bend_damping`/`radius`/`length`/`density`
produces unstable domain-randomization draws, and the shipped derived-sampling scheme
(`E`, `ζ`, `ρ` as the sys-ID/DR targets instead) plus a numerical `ω_n·dt` stability
guard.

---

## Observation / Parquet compatibility

Recorded sys-ID datasets (`example_gym_sysid.py` → `trajectory_store.py`) store:

| Column / metadata | Frame under this fixture |
|-------------------|--------------------------|
| `tcp_pos`, `tcp_quat` | World (= robot base) |
| `tcp_velocity`, `action` | World spatial twist |
| `apple_pos`, `apple_quat` | World |
| `woody_start__*`, `woody_end__*` | World junction anchors |
| `excitation_direction` | World unit vector |
| `ft_wrist` | World wrench at TCP COM |
| `robot_joint_q` | Joint space (intrinsic) |
| Metadata `robot_base_pos` | `(0, 0, 0)` |
| Metadata `fruiting_base_pos` | `(0, 0.5, 0.95)` — **T center** (mid-span spur junction) |

`run_system_identification.py` replays these logs; no schema change required when
migrating to this fixture if metadata base poses are updated consistently.

---

## Known gaps (non-goals for initial fixture)

1. **Bourse link** — not modeled; compliance folded into “ignore bourse” decision.
2. **Magnetic airgap / twist wedge** — hardware weakens hold under torsion; sim
   needs stem torsion DOF (follow-up slice).
3. **Literal N/m on rods** — requires sys-ID calibration from proxy tiers.
4. **Ball-and-socket joint limits** — spur angles are sampled via rod direction
   deltas, not explicit joint stops.
5. **Support compliance** — bench clamps modeled as rigid world FIXED joints; no
   separate spring stiffness on `primary_support_*` until measured.
6. **Support-joint wrench balance** — full analytic load split at `primary_support_*`
   under asymmetric branch loading is not validated (branch cut `primary_spur` is tested).
7. **Intra-cable collisions** — default builds disable all fruiting-chain shape contacts
   (`enable_self_collisions=False`): woody, stem, apple, and gripper proxy do not collide
   with each other. Ground contact only. See `docs/vectorized-coupled-fruiting.md` (warning).

---

## Acceptance criteria

Before marking the fixture slice done:

1. **T-junction builder** matches this doc: dual endpoint supports, mid-span spur,
   junction labels including `primary_support_left` / `primary_support_right`.
2. **IK bootstrap** converges without `IKBootstrapConvergenceWarning` on a seed
   sweep across variance bounds (nominal + worst-case samples).
3. **`visualize_pull_directions.py`** smoke with this fixture — pull directions
   and weld hemisphere consistent with +Y reach and T-center `fruiting_base_pos`.
4. **One sys-ID episode** records cleanly:
   `example_gym_sysid.py --viewer null --n-directions 1 --max-steps 200`
   with the new ranges path and catalog entry.
5. **Catalog test** passes:
   `apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets`
6. **Wrench smoke** (post-T builder): quasi-static subtree checks updated for
   mid-span `primary_spur` and split support loads.

---

## Implementation checklist

- [x] Add `fruiting_system_ranges_real_world_proxy.json` — **shipped, but see topology caveat below**
- [x] Add `fruiting_system_ranges_real_world_proxy_variance.json`
- [x] Add `real_world_proxy` to `digital_twin_fixture_catalog.json`
- [x] Wire gym/sys-ID default or explicit `--ranges` path to proxy fixture where intended
- [ ] Disable `robot_base_from_proxy` for builds using this fixture — not re-verified since this doc was written
- [ ] Update `assets/testfr3_resolved.usda` (140 mm × Ø200 mm, TCP at +Z tip)
- [ ] Sync `GripperProxyConfig` with EE mass placeholder
- [ ] Align `COUPLED_*` defaults or document override when proxy fixture is active
- [x] **T-junction builder:** centered primary, world FIXED at both endpoints,
      mid-span spur attach, junction labels `primary_support_left` /
      `primary_support_right` / `primary_spur`; wrench smoke on `primary_spur`
- [ ] **Follow-up:** stem torsional stiffness API + magnet-tier mapping

**Topology caveat (found during 2026-07 doc audit):** the shipped
`fruiting_system_ranges_real_world_proxy.json` sets `"topology": "linear_chain"`
— it opts **out** of the T-junction builder this document specifies. The
*variance* fixture (`fruiting_system_ranges_real_world_proxy_variance.json`)
omits `"topology"` and so **does** default to `t_junction`. This means the
nominal and variance fixtures for the same physical proxy currently build
**different topologies**. This is a code/fixture-data question, not something
this doc audit resolved — flag to the maintainer before relying on
`real_world_proxy.json` (non-variance) as a T-junction bench twin.

---

## How to verify

From repository root:

```bash
uv run python apple_pick_gym/examples/visualize_pull_directions.py \
  --seed 0 --n-directions 10 --fix-to-apple-warmup-substeps 0 \
  --output /tmp/real_world_proxy_pull_directions.png
```

```bash
uv run python apple_pick_gym/examples/example_gym_sysid.py \
  --viewer null --n-directions 1 --max-steps 200 \
  --output /tmp/real_world_proxy_sysid
```

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_digital_twin.py::test_fixture_catalog_references_existing_assets -q
```

(Use the actual `--ranges` / env flags once the fixture files and catalog entry exist.)

---

## Decision log

| Date | Decision |
|------|----------|
| 2026-06-26 | Robot at origin; `fruiting_base_pos = (0, 0.5, 0.95)` m; world = base frame |
| 2026-06-26 | Topology **T-junction**: primary simply supported at both ends; spur mid-span; omit secondary / bourse |
| 2026-06-26 | `fruiting_base_pos` = T center (mid-span `primary_spur`), not cantilever root |
| 2026-06-26 | Primary horizontal ±X through T center; spur nominal hang −Z; reach +Y |
| 2026-06-26 | Primary length 0.10–0.20 m (full span); stem length 0.01–0.06 m; apple radius 0.04–0.08 m |
| 2026-06-26 | Branch stiffness: continuous bands from proxy 210–736 N/m |
| 2026-06-26 | Magnet detach → stem **torsion** (follow-up slice); interim bend only |
| 2026-06-26 | EE 140 mm × 100 mm radius, TCP +Z; mass placeholder 0.5 kg until measured |
| 2026-06-26 | Apple density placeholder 700 kg/m³ until measured |
| 2026-06-26 | T builder shipped; default topology `t_junction`; opt-in `linear_chain` |
