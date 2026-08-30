# Real vs sim CMA feature alignment

| Field | Value |
| ----- | ----- |
| **Status** | Implemented |
| **Canonical living doc:** | `docs/handbook-sysid-scoring.md` |
| **Date** | 2026-08-13 |
| **Roadmap** | M4.0 real `robot_replay` → CMA (Sinkhorn on converted GT + live sim replay) |
| **Extends** | `docs/superpowers/specs/2026-08-12-real-replay-cmaes-plumbing-design.md` |
| **Amends** | Plumbing spec **Slice 2** (score-time F/T frame + LPF). This spec replaces that plan: convert-time F/T rotate, no LPF, plus woody/hold/USD COM. |
| **Reference episode** | `robot_replay/new_data/s01-d01.parquet` (`ee_config.source` = live `pylibfranka RobotState`, not `config.yaml`) |

## Purpose

Make converted real bags and live sim-replay bags share one Sinkhorn `STATE_VECTOR` convention so CMA-ES on real trajectories is scoring the same quantities.

GT bags come from convert (`apple_pick_sim/system_id/real_to_batched_sysid.py`). Candidate bags come from live sim replay of the same 19D `vic_pose` actions. Today those bags disagree on F/T frame, woody points, and (less so) hold encoding. EE mass properties on the USD arm also disagree with the recorded Desk load.

## Locked context (do not reopen)

| Topic | Fact |
| ----- | ---- |
| Sim `ft_wrist` | World-frame spatial wrench, env-on-robot, reduced to `/ee/tcp` (tip). Gym copies `coupling_forces_cache`. MuJoCo `body_f[tcp]` is that transported wrench. Real F/T is **not** injected into MuJoCo. |
| Real parquet `ft_wrist` | **Current collections** (`final_data_correct_torque/`, e.g. s09): world-frame env-on-robot wrench at TCP (force and torque about TCP) after collection-time frame correction; compiled EMA−EMA with unloaded replay **without apple**. Older logger path: `O_F_ext_hat_K` → bias → EMA → EE frame → negate; convert rotates with `R(tcp)` only when source is not already world/TCP. |
| Hold `cos(F, pull)` | After correct world/TCP alignment, real hold force is anti-parallel to pull (≈ −0.8 on s09 upward dirs). Early s01 alignment work used EE-as-stored ≈ +0.67 before world rotate. |
| USD tool length | Cylinder 180 mm (`EE_CYLINDER_HALF_HEIGHT = 0.09`). Old 140 mm / `0.1034` panda-hand default is stale vs this recording. |
| `F_x_Cee` | Tool **mass COM** in flange F, **not** the wrench point. Do not use 0.077 m as an `ft_wrist` convert offset. |
| Sim woody start/end | Parent/child anchors of the **same FIXED joint** (gap ~µm–0.8 mm), not rod chords. |
| Real woody | Tag poses `branch_pose_4x4` / `spur_pose_4x4` / `apple_pose_4x4` (translations → CMA starts + `apple_pos`). Packed source `woody_part_*` is not a convert path. Bit-1 metadata snapshots may still carry length-9 packing. |

## Slice order

```text
0. USD EE mass properties from recorded ee_config
1. Convert F/T: R(tcp) @ logged wrench; no second negate; no lever-arm transport
2. Woody schema: two starts + apple_pos; drop woody_end from the sys-ID bag
3. Hold: keep scalar hold_number from hold_index; do not pack real 4-vector one-hot
```

Implement in that order. Slice 0 does not change Sinkhorn features. Slices 1–3 do.

## Non-goals (all slices)

- LPF / EMA on sim `ft_wrist`. Real `α=0.2` runs at **1 kHz** (~4 ms). Rate-matched at 15–30 Hz is `α≈1` (passthrough). Literal `α=0.2` at `control_hz` would over-smooth holds. CMA scores holds, where the 4 ms filter has already settled. Do not copy torque slew (`_MAX_TORQUE_DELTA`) into F/T obs.
- Torque-point transport along 0.1034 m or 0.18 m
- Second negate of real F/T
- Restricting Sinkhorn `action` to `action[0:7]` (stay full 19D). **Amended 2026-08-14:** `action` is removed from score-time `STATE_VECTOR`; replay and bag contracts still carry full 19D `vic_pose`. See `docs/superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md`.
- Injecting real F/T into MuJoCo
- Writing hold one-hot into parquet
- Support-joint or `stem_apple` woody feature columns
- Changing `ft_wrist` reduction point away from `/ee/tcp`
- Putting COM or `I_ee` on `/ee/tcp`
- Using `m_load` / `I_load` / `F_x_Cload` (all zero on this episode)

---

## Slice 0 — USD EE mass properties

**Source of truth:** recorded `ee_config` on `s01-d01` (and the same Desk tool on later episodes in this collection). Not `config.yaml` defaults.

| Desk field | Value | USD on `/fr3/ee` |
| ---------- | ----- | ---------------- |
| `m_ee` | 1.1 kg | `physics:mass = 1.1` |
| `F_x_Cee` | (0, 0, 0.077) m in flange **F** | `physics:centerOfMass = (0, 0, −0.077)` m in **ee** local |
| `I_ee` | diag(0.00215219, 0.00215219, 0.00119125) kg·m² about COM in **F** | `physics:diagonalInertia` same triple; `physics:principalAxes` identity |

**Frame:** `fr3_joint8` is RotX(180). F+Z is tip-out; USD ee tip-out is −Z. Origins coincide at the flange / `ee` body origin (not the ~6 mm link7-mesh gap). RotX(180) leaves a diagonal `I_ee` unchanged. Author COM in **meters**, same convention as other FR3 link `centerOfMass` values — do **not** divide by ee xform scale `(0.2, 0.2, 0.18)`.

**Files:** `assets/testfr3_resolved.usda` (runtime) and `assets/testfr3.usda` (authoring; mass is still 1.5). Constants next to `EE_MASS_KG` in `apple_pick_sim/robot/fr3_robot/paths.py`.

**TCP:** `/fr3/ee/tcp` stays `physics:mass = 0.001`, no COM, no `I_ee`. Coupling `body_f` stays at the tip.

**Do not author from this blob:** `F_T_EE` (geometry; 180 mm cylinder already matches `T_z = 0.180`); `EE_T_K` (wrench point; slice 1); payload fields (zero).

**Docs:** one row in `docs/real-world-proxy.md` EE table + decision-log line.

**Tests:** `apple_pick_sim/tests/test_ee_cylinder_geometry.py` scrapes both USDA files for ee mass, COM, diagonal inertia, and tcp mass. Helper: flange `F_x_Cee` → ee-local via RotX(180).

**Does not change** convert F/T, `STATE_VECTOR`, or Sinkhorn.

---

## Slice 1 — F/T frame at convert

**Rule:** at convert time, for each row:

```text
F_world = R(tcp) @ F_logged
τ_world = R(tcp) @ τ_logged
```

`R(tcp)` is the 3×3 of logged TCP orientation from `tcp_pose_4x4`. Vic-pose convert already carries that column; **raise** if it is missing — do not skip the rotate or fall back to identity. Same rotation on force and torque. **No** second negate. **No** transport from K to another point.

Write the rotated 6-vector into converted `ft_wrist`. Prefer the logger’s **`ft_wrist`** (dynamic-baseline-corrected when the unified file applied it). Rotate `ft_wrist_raw` the same way if that column is copied, but **do not score `raw_ft_wrist`**. Convert the **unified** episode parquet (`s01-d01.parquet`), not `*_robot.parquet` (no woody/apple). Do not leave EE-frame F/T in the converted bag. Raise if `tcp_pose_4x4` is missing.

**Why convert-time, not score-time:** CMA GT is the converted bag; candidates are sim world wrenches. One write, both consumers match. Plumbing-spec score-time transform is **rejected**.

**Live sim replay:** unchanged. `ft_wrist` is already world / env-on-robot / TCP.

**Acceptance:** on a converted real hold, `cos(F_world, pull)` is negative (same sign as sim), without an extra flip. Unit test: known `R` and EE-frame wrench → expected world wrench.

---

## Slice 2 — woody bag schema (breaking)

Sim `woody_start[j]` / `woody_end[j]` are the two anchors of FIXED joint `j`, almost the same point. Real length-9 packs three distinct tags. Scoring those as the same `STATE_VECTOR` fields is wrong.

**CMA woody junctions (both real convert and sim collect bags):**

```text
CMA_WOODY_JUNCTIONS = ("primary_spur", "spur_stem")
```

No `support`. No `stem_apple`. Plant still has support FIXED joints; they are not Sinkhorn woody columns.

**Points that enter the bag / `STATE_VECTOR`:**

| Feature | Real convert | Sim collect / live replay |
| ------- | ------------ | ------------------------- |
| `woody_start__primary_spur` | `branch_pose_4x4` translation | parent anchor of `primary_spur` |
| `woody_start__spur_stem` | `spur_pose_4x4` translation | parent anchor of `spur_stem` |
| `apple_pos` | `apple_pose_4x4` translation | apple body translation (already a column) |

**Drop from the sys-ID bag contract:** `woody_part_end_pos` / `woody_end__*` parquet columns, `ReplayObservationCollector` `woody_end`, `STATE_VECTOR_FIELDS` / `REQUIRED_ARRAY_KEYS` entry `woody_part_end_pos`.

Live gym obs **may** still expose `woody_part_end_pos` for debug / force viz. Collect writers and the collector **must not** persist it. CMA scoring **must not** read it.

**Bending:** `build_bending_angles` no longer uses per-junction `end − start` (those chords are ~0 in sim). After this slice:

| Angle column | Chord |
| ------------ | ----- |
| spur (`primary_spur`) | `woody_start[spur_stem] − woody_start[primary_spur]` |
| stem (`spur_stem`) | `apple_pos − woody_start[spur_stem]` |

Rest pose remains frame 0 of that bag. Two angles, in `CMA_WOODY_JUNCTIONS` order.

**Convert source (2026-08-14):** Sinkhorn woody/apple come from tag SE(3) translations via `tag_poses_to_cma_woody`. Convert **requires** table columns `branch_pose_4x4`, `spur_pose_4x4`, `apple_pose_4x4` (null cells raise). Source `woody_part_start_pos` / `woody_part_end_pos` packing is **not** read (`compiler_woody_to_cma_starts` removed). Converted bag still writes `woody_start__primary_spur` / `woody_start__spur_stem` / `apple_pos`.

Map:

- `woody_start__primary_spur` ← `branch_pose_4x4` translation (T / top of spur)
- `woody_start__spur_stem` ← `spur_pose_4x4` translation (bottom of spur / top of stem)
- `apple_pos` ← `apple_pose_4x4` translation (apple center)

Do not emit `stem_apple` or any `woody_end__*` column. Bit-1 pre-grasp rebuild may still unpack packed woody from **metadata** snapshots.

Must change **sim collect + collector + `STATE_VECTOR`**, not convert-only. Otherwise live candidate bags still emit three joints × start+end.

**Grid woody MSE** that iterates end positions: either use the two starts + `apple_pos`, or skip end-based MSE. Do not resurrect `woody_end` in the bag to keep old MSE.

---

## Slice 3 — hold id

Converted parquet already has scalar `hold_number` via `_scalar_hold_number(..., hold_index=...)`. **Keep that.** Prefer `hold_index` when present.

Do **not** write the real logger’s 4-vector hold one-hot into parquet.

CMA scorer keeps `hold_id_onehot=True` and one-hots from the scalar at score time (`_one_hot_hold_id`). Same for sim-sim bags.

Mostly tests: convert of a 4-vector `hold_number` column plus `hold_index` still yields a scalar; collector / `combine_transition_features` still one-hot from that scalar.

---

## `STATE_VECTOR` after slices 1–2

Unchanged keys except woody ends removed:

```text
ft_wrist                  # world, env-on-robot, about TCP (both bags)
tcp_velocity
tcp_pos
apple_pos                 # also the stem distal point
woody_part_start_pos      # primary_spur, spur_stem only
woody_bending_angles      # 2 chords: spur then stem
```

**Amended 2026-08-14:** `action` dropped from score-time `STATE_VECTOR` only; replay still drives full 19D `vic_pose` and bags still require `action`. See `docs/superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md`.

Bag still concatenates `[s, Δs, hold_id_onehot, dir_onehot]` at score time. Not scored: `woody_part_force`, quats, joints, cameras, raw F/T.

## Tests (by slice)

| Slice | Tests |
| ----- | ----- |
| 0 | USDA scrape: ee mass 1.1, COM (0,0,−0.077), `I_ee` diagonal, tcp mass 0.001, on both `testfr3.usda` and `testfr3_resolved.usda`. RotX(180) helper. |
| 1 | Convert unit: `R @ wrench` on F and τ; no sign flip. Hold `cos(F, pull) < 0` on a fixture row with known TCP R. Converted bag `ft_wrist` ≠ EE-frame input. |
| 2 | Convert maps tag 4×4 translations → two starts + `apple_pos` (`tag_poses_to_cma_woody`). Requires `branch_pose_4x4` / `spur_pose_4x4` / `apple_pose_4x4`. No `woody_end__*` / `stem_apple` / `support` woody columns. `build_bending_angles` uses the two chords. Collector does not require `woody_end`. `STATE_VECTOR` width matches 2 junctions × start + 2 angles. Sim collect parquet `junction_names == ["primary_spur", "spur_stem"]`. |
| 3 | `_scalar_hold_number` from `hold_index` even if `hold_number` is length-4. Scorer one-hot still works. |

Existing woody-end tests (`test_mmd_features`, `test_trajectory_store`, `test_real_to_batched_sysid`, gym MSE helpers) update to the new contract; do not keep dual start+end scoring paths.

## Out of scope follow-ups (named, not this work)

- Match sim F/T filtering to real 1 kHz EMA (rate-matched equivalent is a no-op at `control_hz`)
- ~~Mimic unloaded dynamic baseline subtraction in sim~~ **Closed 2026-08-14 as not applicable:** sim `ft_wrist` is plant harvest (`coupling_forces_cache`); a robot-only tare is identically zero on `vic`/`vic_pose`. Real compiled bags already tared. Do not implement. Canonical warning: `docs/handbook-sysid-scoring.md`. Reopen only if harvest becomes a sensor-like external wrench.
- Reconcile logger metadata string (“force in EE, torque in base”) with the interface that rotates both
- Drop Kp/Kd from Sinkhorn `action`
- Multi-episode / multi-structure fused real CMA
