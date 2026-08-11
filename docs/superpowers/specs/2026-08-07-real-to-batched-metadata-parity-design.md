# Real → batched metadata parity (pre/post grasp)

| Field | Value |
| ----- | ----- |
| **Status** | Bits 1–2 Done (incl. `vic_pose` pack + controller); **bit 3 = ROADMAP M4.0 (Current focus)** |
| **Date** | 2026-08-07 (status revised 2026-08-11) |
| **Source of truth** | `robot_replay/example_view_pre_grasp_settle.py` (native pre/post stack) |
| **Related** | `docs/ROADMAP.md` M4.0, `docs/real-sysid-pre-post-grasp-fixes.md` (C1), `robot_replay/README.md`, `docs/batched-sysid-dataset.md`, `docs/superpowers/specs/2026-08-10-vic-pose-action-controller-design.md` |

## Purpose

Bridge real-robot parquet episodes into batched-style **episode metadata JSON**
so later slices can reuse batched FR3 replay and CMA-ES. Step 1 only
proves **pre-grasp and post-grasp geometry/weld init** match the settle viewer.

## Sequencing (bits)

| Bit | Scope | Status |
| --- | ----- | ------ |
| **1** | Fix converter via shared native builders; numeric parity; batched-meta viewer | **Done** |
| **2** | Full `batched_sysid_v1` trajectory export + open-loop FR3 + 19D `vic_pose` packing + replay | **Done** |
| **3** | Feed converted episodes into existing sys-ID CMA-ES under `vic_pose` | **Next** (= ROADMAP **M4.0**) |

## Architecture (bit 1)

```text
real parquet + fixture
        │
        ▼
 native builders (trusted)
   fruiting_params_from_pre_grasp_parquet
   post_grasp_plan_from_metadata
        │
        ├──────────────► example_view_pre_grasp_settle.py
        │
        └─ thin adapter (real_to_batched_sysid)
               → episode_meta.json (batched-style keys)
                     │
                     ├─ pytest native↔convert parity
                     └─ new script: view from converted JSON
```

**Approach:** do not keep a second geometry rebuild in the converter. Call the
same APIs the settle viewer uses, then package outputs into batched metadata
keys. Schema mismatches (C1) become adapter details.

## Parity contract

Native reference (same inputs as the settle script):

```text
params, base_pos, _ = fruiting_params_from_pre_grasp_parquet(parquet, fixture)
plan = post_grasp_plan_from_metadata(load_dataset_metadata(parquet), …)
```

Converted metadata **must** match native on:

| Field | Source |
| ----- | ------ |
| `fruiting_system_params` | `fruiting_params_to_dict(params)` |
| `fruiting_base_pos` | native `base_pos` |
| `params_fingerprint` | from those params |
| `initial_tcp_pos` / `initial_tcp_quat` | plan TCP (`pose_4x4_to_pos_quat`) |
| `initial_apple_pos` / `initial_apple_quat` | plan apple |
| `weld_reference_pos` / `weld_reference_quat` | same as initial apple (current batched convention) |
| `weld_direction` | unit(TCP − apple) × `--weld-direction-sign` |
| `initial_robot_joint_q` | grasp-row `joint_pos` (row 0 when no `step_idx==-1`) |
| `apple_radius` | native params (from `pre_grasp_geometry.parts.apple.radius_m`) |
| `rod_radii` | native params radii (from `pre_grasp_geometry.parts.{primary,spur,stem}.radius_m`) |

**Tolerances:** positions ~`1e-9`–`1e-6` m; quats abs-dot ≥ `1 - 1e-9` (sign flip
OK); params / radii exact after shared builder.

**Radii note:** radii **are** in current parquets under
`pre_grasp_geometry.parts.*.radius_m`. They are **not** under top-level
`dataset_metadata.rod_geometry`. Do not require `rod_geometry`; use parts via
the native mapper.

**Adapter-only (not parity-critical):**

- `control_hz` — prefer metadata; else documented default (e.g. 15 Hz) + warning
- `episode_id` — prefer non-empty metadata / `dump.episode_id` / path stem

## Scripts, tests, errors

**Library:** rewrite `build_episode_metadata_from_real` to call native pre/post
builders; map `joint_pos` → `initial_robot_joint_q`; derive quats via
`pose_4x4_to_pos_quat`; fill `rod_radii` / `apple_radius` from native params.
Keep CLI `--fixture` / `--weld-direction-sign`.

**Tests (TDD):**

1. Parity smoke on `robot_replay/s00-d00.parquet` (skip if missing): native vs
   convert for all strict fields above.
2. Keep or slim synthetic helper tests; primary ingest gate is real episodes.

**Scripts:**

- Keep `example_view_pre_grasp_settle.py` as native reference (`--dump-params`
  optional for humans).
- Add a small viewer that loads **converted metadata JSON** and runs the same
  settle / optional post-grasp weld path for eyeballing.
- Convert CLI remains the JSON producer.

**Errors:** fail loud on missing `pre_grasp_geometry.parts` radii or post-grasp
poses (same as native). Warn on missing `control_hz`. Fall back empty
`episode_id`.

**Docs:** update `robot_replay/README.md`; note in
`docs/real-sysid-pre-post-grasp-fixes.md` that C1 metadata ingest uses shared
native builders.

## Out of scope (bit 1)

- Trajectory row export / full `batched_sysid_v1` dataset layout
- FR3 + open-loop EE-twist replay
- CMA-ES wiring
- Fixing collector compile issues beyond what the adapter needs for metadata

## Known bug — real-world replay drive signal (deferred)

**ID:** real-replay-action-zero (track with C1 / real sys-ID fixes)

**Symptom:** On current episodes (`s00-d00`, `s00-d02`, …), the parquet `action`
column is all zeros while `tcp_velocity` carries non-zero 6D twists. Phases are
`pull`/`hold`, but the logged **command** channel usable by batched replay is
empty.

**Impact:** A naive trajectory export that copies `action` would produce a
stationary FR3 replay. Bit 2 must not treat zero `action` as a valid drive
without an explicit policy.

**Intent (not a permanent convert “feature”):**

- Record this as a **real-world replay / collection bug**.
- **Future parquets** will include the **entire trajectory** (not hold-only /
  incomplete command logs) with a correct non-zero `action` (or an agreed
  command column) for open-loop EE-twist replay.
- Until then, any interim bit-2 workaround (e.g. fall back to `tcp_velocity`
  when `action` is identically zero) must be labeled as a **temporary
  mitigation**, gated and documented — not the long-term contract.

## Success criteria (bit 1)

1. `convert_real_to_batched_sysid_metadata.py` succeeds on `s00-d00`-class
   parquet without requiring `step_idx==-1`, `robot_joint_q`, `tcp_quat`, or
   top-level `rod_geometry`.
2. Pytest parity: native builders vs converted JSON on the strict field list
   (including `rod_radii` / `apple_radius`).
3. New batched-metadata viewer can settle/weld from that JSON for visual check.
4. README documents the convert + parity commands.

## Non-goals

Changing Newton, CMA-ES search math, or re-collecting episodes in this bit.
