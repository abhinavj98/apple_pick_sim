# Real sys-ID — pre_grasp / post_grasp collection fixes

## Purpose

Checklist of **data-collection and compile bugs / contract gaps** in
`pre_grasp_geometry` and `post_grasp_geometry` on real-robot static sys-ID
episodes (`schema_name: real_static_sysid_episode`, e.g.
`robot_replay/s00-d00.parquet`). Use this when fixing the upstream collector /
`compile_static_sysid` pipeline or hardening consumers
(`apple_pick_sim/system_id/real_to_batched_sysid.py`, `robot_replay/`).

Observed against: `robot_replay/s00-d00.parquet` (recompiled ~2026-07-27,
`schema_version` 1.0.0; woody rebuild snapshot on `rest_snapshot_during_run`).

Related: [`robot_replay/README.md`](../robot_replay/README.md),
`docs/digital-twin.md`, `docs/real-world-proxy.md`,
`docs/batched-sysid-dataset.md`, `docs/sysid-trajectory-storage.md`.

**Shipped viewer (plant-only):** `robot_replay/example_view_pre_grasp_settle.py`
rebuilds params from `pre_grasp_geometry` via
`select_pre_grasp_woody_snapshot` (prefer `rest_snapshot_during_run`, else legacy
`snapshot`), settles, and optionally applies a post-grasp look-at weld
(`--grasp-after-settle`) — see
[`docs/superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md`](superpowers/specs/2026-07-24-real-pre-grasp-settle-viewer-design.md)
and
[`docs/superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md`](superpowers/specs/2026-07-24-real-post-grasp-viewer-design.md).

**Stem length diagnostics (viewer only):** catalog `parts.stem.length_m` still
builds the sim rod; diagnostics compare against
`‖spur_end − apple_CoM‖ − apple_radius` (`stem_chord_formula` in
`map_pre_grasp_geometry` diagnostics). Large catalog-vs-chord gaps are a
**data** issue (see **P4**), not a mapping bug.

**Consumer warnings (until corrected data):** `|tcp−apple|≈r`, logged TCP **+Z∥ŵ**,
and pose-vs-`*_pos` consistency are checked in `real_post_grasp_plan.py` (warn and
continue; tol default 2 cm for radius/shift).

---

## Intended semantics (target contract)

These two metadata blocks are a **two-phase digital-twin init**, not synonyms for
“before/after the pull trajectory.”

```text
pre_grasp_geometry  ──►  rebuild fruiting_system geometry
                              (straight / non-bending rods + apple placement)
                              then settle under gravity in sim
                                    │
post_grasp_geometry ──►  weld / attach the robot TCP to that
                              settled plant at the measured grasp
                              pose and approach direction
```

| Block | Meaning | Downstream use |
| ----- | ------- | -------------- |
| **`pre_grasp_geometry`** | Apple in a **correct spatial pose** with **branches not bending** (woody chords at the unloaded / gravity-opposed reference). Correct static forces largely cancel gravity so the plant is held without visible deflection. This is **not** the gravity-settled hanging shape. | Rebuild `fruiting_system` geometry (rod directions/lengths, apple placement, topology). Prefer a dedicated table row (`step_idx == -1`) and/or a validated metadata snapshot. |
| **`post_grasp_geometry`** | **Settled apple under grasp**: fruit pose after the robot has grasped from a known **direction and position** (TCP / weld frame). Woody may show bend relative to the pre-grasp chords. | Attach (weld) the robot to the plant that was built from pre-grasp and then settled in sim. Supplies `weld_reference_*`, grasp TCP, and approach direction. |

**Invariant:** pre-grasp defines the **plant construction**; post-grasp defines the
**robot–plant attachment** on the settled plant. Do not build rods from
post-grasp bent chords, and do not weld using only the pre-grasp apple if a
settled grasp frame exists.

---

## Known quirks (consume defensively until fixed)

### Q1 — Pre-grasp `apple_pos` may be a string

In current `s00-d00.parquet`, woody + `apple_pos` for rebuild live on
`pre_grasp_geometry.rest_snapshot_during_run` (legacy compilers used
`pre_grasp_geometry.snapshot`). `post_grasp_geometry.apple_pos` is a JSON list of
floats, but the rebuild snapshot may still serialize:

```text
pre_grasp_geometry.rest_snapshot_during_run.apple_pos
  == "[... ... ...]"   # numpy ndarray __str__, not JSON
```

(`snapshot` on the current file is TCP-only — no woody fields.)

Any loader that does `list(apple_pos)` or assumes `list[float]` will break or
silently mis-parse.

**Workaround:** `coerce_xyz` / detect `str`, parse with `np.fromstring(..., sep=" ")`
(or `ast.literal_eval` after normalizing spaces), then validate length 3.
`select_pre_grasp_woody_snapshot` prefers `rest_snapshot_during_run` then
`snapshot`.

**Fix:** serialize all vector fields with the same encoder (JSON list / Arrow
list); add a compile-time assert that `apple_pos` is `list` of 3 numbers.

### Q2 — Top-level `episode_id` is empty

Schema metadata `episode_id` is `""`. The real UUID lives only under
`dump.episode_id` (e.g. `fd8bc9ad-…`). Consumers that read
`dataset_metadata.episode_id` get an empty string.

**Fix:** copy `dump.episode_id` (or collection UUID) into the top-level field at
compile time; reject empty on write.

---

## Fixes — pre_grasp_geometry

| ID | Severity | Issue | Desired fix |
| -- | -------- | ----- | ----------- |
| P1 | High | Non-bending woody lives in metadata (`rest_snapshot_during_run` preferred; legacy `snapshot`; plus top-level `rest_woody_*`); table has **no** `step_idx == -1` pre-grasp row. Downstream `split_pregrasp_and_trajectory` expects exactly one such row. | Emit a pre-grasp row in the episode table (and/or teach the converter to read `rest_snapshot_during_run` / `snapshot` / `rest_woody_*` explicitly). |
| P2 | High | Rebuild-snapshot `apple_pos` string quirk (**Q1**). | Uniform list[float] serialization + schema validation. |
| P3 | Medium | Rest / post timestamps can collide with row 0 while woody/apple are the **non-bending** reference (bend ≡ 0), not the settled grasped state. Timestamps do not distinguish “geometry rebuild capture” vs “grasp/weld frame”. | Record the true pre-grasp capture time (see also `dump.rest_reference_timestamp`, which may differ from top-level `rest_reference_timestamp`). Document which clock each field uses; do not reuse grasp `t₀` for pre-grasp unless they are actually the same sample. |
| P4 | Medium | `parts` catalog is `default_template` with zero `connection_rpy_deg` and placeholder `connection_source` values — not measured on this structure. On current `s00-d00`: primary catalog length **0.827 m**, stem catalog **0.005 m** vs measured surface stem chord **~0.017 m** (~243% rel err), spur catalog 0.12 m vs chord ~0.110 m (~8%). Sim rods use **catalog** lengths; large gaps hurt rebuilt-plant fidelity. | Either omit unused catalog fields, or fill from the real lengthened-state / bench measurement and set `structure_name` to a non-template id. Mark placeholders explicitly (`measured: false`). |
| P5 | Medium | Junction naming in topology / tracking (`Branch`, `Spur`, `Apple`) does not match sim twin names (`primary_spur`, `spur_stem`, `stem_apple`). Mapping is implicit. | Persist an explicit `junction_name_map` in metadata; keep tracker names and sim names side by side. |
| P6 | Low | Snapshots store `*_pose_4x4` as flat length-16 without stating row- vs column-major. Table columns use the same layout as robot dumps (translation at indices 3, 7, 11). | Add `field_layout` entries for snapshot poses (reshape + major) identical to table `field_layout`. |
| P7 | Low | `camera_frame_count` in the rest snapshot may differ from table rows’ aggregation frame count. Easy to confuse “raw selected stamps” with “aggregation frame count”. | Rename or split fields (`rest_selected_camera_timestamps` vs `camera_aggregation.requested_frame_count`). |
| P8 | Medium | Collection must keep **visible bend ≈ 0** while apple pose is still the intended rebuild pose (gravity opposed). If pre-grasp chords already bend, twin rod directions are wrong. Current rest snap on `s00-d00` has bend `[0,0,0]` (OK for `--strict`). | Collector checklist: confirm `woody_bending_angles ≈ 0` and document opposing-gravity / fixture support condition. |

---

## Fixes — post_grasp_geometry

| ID | Severity | Issue | Desired fix |
| -- | -------- | ----- | ----------- |
| G1 | High | Post-grasp must be the **settled grasped apple** plus robot grasp **position and direction** for welding. On current `s00-d00`, post woody bend is small (~0.006 / 0.008 / 0.014) and apple radius catalog is **0.035 m**; TCP–radius residual ~0.002 m (within 2 cm warn). Ensure weld fields are explicit and not confused with the non-bending rebuild snapshot. | Always store grasp TCP, apple pose, and unit weld/approach direction in `post_grasp_geometry`; assert they are the weld attachment frame after settle, not a second copy of pre-grasp plant geometry. |
| G2 | Medium | Duplicates row 0 of the table (verified equal for tcp/apple/bend/timestamp). Useful as a freeze-frame, but can drift if compilers rewrite metadata without rewriting row 0. | On compile, assert `post_grasp_geometry` == row 0 for all shared fields; or stop duplicating and point consumers at `grasp_row_index: 0`. |
| G3 | Medium | `source_metadata_summary.post_grasp_geometry` only keeps TCP/target poses — a **lossy** summary vs full `post_grasp_geometry` (no apple/woody). | Either drop the partial copy or store the same schema; never treat the summary as authoritative. |
| G4 | Medium | No `tcp_quat` / `apple_quat` / `robot_joint_q` in this block (poses are 4×4 only). Batched converter currently requires quat + `robot_joint_q` columns. | Add quat (and joint_q) to post-grasp metadata **and** ensure table columns exist under stable names (`joint_pos` vs `robot_joint_q`). |
| G5 | Low | Bending angles are correct vs `rest_chord_vectors` / pre-grasp chords (verified). Post-grasp bend must always be measured against the **same** pre-grasp chords used to rebuild geometry. | Single source of truth for pre-grasp chords; post-grasp bend recomputed only from that reference. |

---

## Cross-cutting collection / compile issues

| ID | Severity | Issue | Desired fix |
| -- | -------- | ----- | ----------- |
| C1 | High | Schema mismatch vs in-repo consumer: real episode has `joint_pos`, `tcp_pose_4x4`, metadata `pre_grasp`/`post_grasp`; converter wants `robot_joint_q`, `tcp_quat`, `apple_quat`, `step_idx=-1`, `rod_geometry`, `fruiting_base_pos`, `source_metadata.robot.control_hz`. | Define one **ingest contract** (column set + metadata keys) and validate at compile; adapt `real_to_batched_sysid` or the compiler so `s00-d00`-class files round-trip. |
| C2 | Medium | `reference_tag_to_base_4x4` source is `"hardcoded default in compile_static_sysid.py"` — not measured for the episode. Affects any interpretation that mixes AprilTag and `franka_base_o`. | Measure/log the tag→base used on the bench; refuse hardcoded defaults in “collect” mode without an explicit override flag. |
| C3 | Medium | Dual pre-grasp / rest timestamps: `dump.rest_reference_timestamp` ≠ top-level `rest_reference_timestamp` / snapshot time. | One canonical pre-grasp capture time; others are aliases or removed. |
| C4 | Low | Empty top-level `episode_id` (**Q2**). | Populate from collection UUID. |

---

## Suggested validation (compile gate)

Before writing `*.parquet`, assert:

1. Every metadata vector field is JSON-serializable `list` of numbers (no `str` poses/positions) — catches **Q1**.
2. `episode_id` non-empty and equals `dump.episode_id`.
3. `pre_grasp_geometry.snapshot.woody_bending_angles` ≈ `0` (non-bending rebuild reference) and matches top-level `rest_woody_*`.
4. `post_grasp_geometry` equals table row at `grasp_row_index` for shared keys.
5. Pre-grasp and post-grasp timestamps differ **or** an explicit `same_sample_as_grasp: true` flag is set.
6. Post-grasp includes (or can derive) a unit weld/approach direction and grasp TCP distinct from “plant-only” rebuild fields.
7. Optional: presence of `step_idx == -1` row iff the ingest contract requires it.

---

## Document status

| Field | Value |
| ----- | ----- |
| **Created** | 2026-07-24 |
| **Updated** | 2026-07-24 — clarified pre = non-bending rebuild geometry; post = settled grasp weld |
| **Evidence file** | `robot_replay/s00-d00.parquet` |
| **Owner** | Abhinav |
