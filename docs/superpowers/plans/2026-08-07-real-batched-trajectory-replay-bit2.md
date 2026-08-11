# Real → batched trajectory export + format gate (bit 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This session: inline execution.**

**Goal:** Turn a real-world parquet into a 1×1 `batched_sysid_v1` dataset, prove format compatibility with `example_batched_sysid_trajectory_viz.py`, then smoke FR3 open-loop **placement** (not yet correct wrench/pose drive).

**Architecture:** Bit-1 metadata (`build_episode_metadata_from_real`) plus mapped trajectory rows → `manifest.json` + `episodes/s00_d00.parquet`. **Primary real source:** `robot_replay/s02-d00_action.parquet` when available. Legacy zero-action episodes may still use `fill_actions_from_tcp_velocity.py` as a temporary mitigation (**tcp_velocity ≠ real wrench**). **Format gate first.** Physics smoke under twist VIC is **not** correct for wrench-logged episodes — see `2026-08-10-vic-pose-action-controller-design.md`. No CMA (bit 3).

**Tech Stack:** Python, pyarrow, `BatchedSysIdDataset` / `BatchedEpisodeWriter` / `write_manifest`, `write_dataset_trajectory_viz`, `replay_batched_sysid_structure`, pytest, uv.

## Global Constraints

- Bit 1 metadata builders remain the source of truth for rebuild + grasp init.
- Real `action` is often a **pose-control wrench** (`dump.action_semantics`); do not treat it as EE twist.
- Export refuses wrench-as-twist unless `--allow-wrench-as-twist`.
- `fill_actions_from_tcp_velocity` is temporary only for older zero-action files.
- Format consumer pinned: `apple_pick_gym/batched_examples/example_batched_sysid_trajectory_viz.py`.
- Do not wire CMA-ES in this bit.

## End-to-end flow

```text
robot_replay/s02-d00_action.parquet     # preferred when present
    │ export_real_episode_to_batched_dataset / --dataset-out
    ▼
/tmp/real_batched_s02_d00/
  manifest.json
  episodes/s00_d00.parquet
    │
    ├─ FORMAT GATE
    │    example_batched_sysid_trajectory_viz.py --dataset … --output …
    │
    └─ PLACEMENT / GL SMOKE (drive signal may still be wrong)
         example_replay_real_batched.py
```

## File map

| Path | Role |
|------|------|
| `robot_replay/s02-d00_action.parquet` | Primary real episode (gitignored locally) |
| `robot_replay/fill_actions_from_tcp_velocity.py` | Optional mitigation for older zero-action files |
| `apple_pick_sim/system_id/real_to_batched_sysid.py` | `export_real_episode_to_batched_dataset` |
| `robot_replay/convert_real_to_batched_sysid_metadata.py` | `--dataset-out` |
| `robot_replay/example_replay_real_batched.py` | FR3 replay CLI (was `*_smoke` in early drafts) |
| `apple_pick_sim/tests/test_real_to_batched_sysid.py` | Export load + zero-action / wrench refuse |
| `apple_pick_gym/tests/test_real_batched_replay.py` | Slow FR3 TCP-motion smoke |
| `robot_replay/README.md` | export → **trajectory_viz** → physics commands |
| Existing | `example_batched_sysid_trajectory_viz.py` (format gate) |

---

### Task 1: Fill-helper unit test (fixture script already shipped)

**Files:**
- Create: `apple_pick_sim/tests/test_fill_actions_from_tcp_velocity.py`
- Existing: `robot_replay/fill_actions_from_tcp_velocity.py`

- [x] Fill CLI + `s00-d03_with_actions.parquet` generated locally
- [x] **Step 1: Failing/green test** — synthetic zero `action` + non-zero `tcp_velocity` → filled equals velocity; `drive_fill` in metadata
- [x] **Step 2: pytest green**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_fill_actions_from_tcp_velocity.py -q
```

---

### Task 2: Export `batched_sysid_v1` dataset from real parquet

**Files:**
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`
- Modify: `robot_replay/convert_real_to_batched_sysid_metadata.py` (`--dataset-out`)
- Test: `apple_pick_sim/tests/test_real_to_batched_sysid.py`

**Produces:**

```text
<out_dir>/
  manifest.json
  episodes/s00_d00.parquet
```

Episode frames must satisfy `BATCHED_REQUIRED_FRAME_COLUMNS` and enough bonus columns for viz (`tcp_pos`, `apple_pos`, `sim_time`, woody if available, `phase`, `action`, …). Schema metadata = bit-1 episode meta (+ movement fields optional).

Refuse all-zero `action` unless `drive_fill` present or `--allow-zero-action`.

- [x] **Step 1: Failing test** `test_export_real_to_batched_dataset_loads`
- [x] **Step 2: Implement exporter** using `BatchedEpisodeWriter` / `write_manifest` / bit-1 meta
- [x] **Step 3: CLI `--dataset-out`**
- [x] **Step 4: pytest green** (+ `test_export_refuses_all_zero_action`)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py -q
```

---

### Task 3: Format gate — trajectory viz (pinned)

**Files:** docs/README only; reuse existing example.

**Consumes:** dataset from Task 2  
**Gate command:**

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00_action.parquet \
  --dataset-out /tmp/real_batched_s02_d00 \
  --overwrite

uv run python apple_pick_gym/batched_examples/example_batched_sysid_trajectory_viz.py \
  --dataset /tmp/real_batched_s02_d00 \
  --output /tmp/real_batched_s02_d00_viz \
  --no-hold-check
```

Expected: HTML written; no schema/load errors; TCP path shows motion.

- [x] **Step 1: Run gate on exported `s02-d00_action` dataset**
- [x] **Step 2: Document commands in `robot_replay/README.md`**

---

### Task 4: Physics smoke — existing batched replay

**Files:** `robot_replay/example_replay_real_batched_smoke.py` + `apple_pick_gym/tests/test_real_batched_replay_smoke.py`.

- [x] **Step 1: Smoke** — export → `replay_batched_sysid_structure` (oracle GT stiffness); fixture span clamp + fixture `fruiting_base_pos` (C6); assert TCP motion
- [x] **Step 2: README** — physics smoke command after viz gate
- [x] Prefer reusing gym helpers over a large new example

Also fixed: `download_batched_replay_obs_numpy` / collector row copies so replay collectors do not alias live torch buffers.

---

### Task 5: Docs close-out

- [x] Update bit-1 design “sequencing” bit-2 status
- [x] Note `real-replay-action-zero` + `drive_fill` + catalog primary span (C5/C6) in fix doc
- [ ] Commit when user asks

---

## Success criteria

1. Prefer `s02-d00_action.parquet` (real non-zero `action`); optional fill only for older zero-action files.
2. Exporter writes loadable 1×1 `batched_sysid_v1` dataset.
3. **`example_batched_sysid_trajectory_viz.py` runs successfully on that dataset** (format gate) — verified on s02.
4. Physics replay smoke via existing batched replay APIs — verified.
5. Zero-action inputs fail loud unless override / `drive_fill`.
6. CMA untouched.

## Out of scope

- CMA-ES (bit 3)
- Fixing upstream collector
- Multi-structure real batches
- Permanent productization of `tcp_velocity` fill
