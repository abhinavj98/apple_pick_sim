# Real → batched trajectory export + FR3 replay (bit 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a real-world episode (metadata from bit 1 + trajectory rows with EE twists) into `batched_sysid_v1` layout and replay it with existing FR3 open-loop batched replay.

**Architecture:** Extend `real_to_batched_sysid` to write a one-structure / one-direction `batched_sysid_v1` dataset directory (manifest + `episodes/s00_d00.parquet`) using bit-1 metadata plus table rows. Drive signal is parquet `action`; for local testing until collection is fixed, use `s00-d03_with_actions.parquet` produced by filling zeros from `tcp_velocity`. Replay via `replay_batched_sysid_structure` / a thin real-episode smoke example — no CMA yet (bit 3).

**Tech Stack:** Python, pyarrow, `BatchedSysIdDataset` / `batched_trajectory_store`, `replay_batched_sysid_structure`, pytest, uv.

## Global Constraints

- Bit 1 metadata builders remain the source of truth for rebuild + grasp init.
- Long-term contract: real parquets ship non-zero `action` for the **full trajectory** (not hold-only / empty command).
- `real-replay-action-zero` is a **collection bug**; `tcp_velocity` fill is a **temporary local mitigation** only (`drive_fill` metadata stamp).
- Primary local fixture: `robot_replay/s00-d03_with_actions.parquet` (gitignored `*.parquet`; regenerate with fill script).
- Do not wire CMA-ES in this bit.

## File map

| Path | Role |
|------|------|
| `robot_replay/fill_actions_from_tcp_velocity.py` | Local fixture: copy parquet, fill `action` ← `tcp_velocity` |
| `apple_pick_sim/system_id/real_to_batched_sysid.py` | Add trajectory dataset export on top of bit-1 metadata |
| `robot_replay/convert_real_to_batched_sysid_metadata.py` or new CLI | Export full dataset dir (or extend flags) |
| `apple_pick_sim/tests/test_real_to_batched_sysid.py` | Trajectory export + action presence tests |
| `robot_replay/example_*` or gym batched example | FR3 / batched replay smoke on exported dataset |
| `robot_replay/README.md` | Commands for fill → convert dataset → replay |

---

### Task 1: Local drive-filled fixture (`s00-d03`)

**Files:**
- Create: `robot_replay/fill_actions_from_tcp_velocity.py` (if not already present)
- Test: small unit test on synthetic table OR script smoke in README
- Docs: `robot_replay/README.md` note (parquet gitignored)

**Produces:** `fill_actions_from_tcp_velocity(...)` → `s00-d03_with_actions.parquet` with non-zero `action` and `dataset_metadata.drive_fill`.

- [ ] **Step 1: Write failing test** for fill helper (synthetic 2-row table: zero action, non-zero tcp_velocity → filled action equals velocity; metadata stamp present).

```python
def test_fill_actions_from_tcp_velocity_only_zeros(tmp_path: Path):
    # write mini parquet with action=0, tcp_velocity=[1,0,0,0,0,0]
    # call fill → assert action matches velocity and drive_fill.rows_filled == 1
```

- [ ] **Step 2: Run test — expect FAIL** (helper missing or incomplete).

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_fill_actions_from_tcp_velocity.py -q
```

- [ ] **Step 3: Implement fill helper + CLI** (copy schema/metadata; fill zeros only by default).

- [ ] **Step 4: Generate local fixture**

```bash
uv run python robot_replay/fill_actions_from_tcp_velocity.py \
  --input robot_replay/s00-d03.parquet \
  --out robot_replay/s00-d03_with_actions.parquet
```

Expected: `action_nonzero_after` ≈ rows with non-zero `tcp_velocity`.

- [ ] **Step 5: Commit** script + test + README note (not the `.parquet`).

---

### Task 2: Export full `batched_sysid_v1` episode from real parquet

**Files:**
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`
- Modify: convert CLI (add `--dataset-out` or sibling script)
- Test: `apple_pick_sim/tests/test_real_to_batched_sysid.py`

**Consumes:** bit-1 `build_episode_metadata_from_real`  
**Produces:** `export_real_episode_to_batched_dataset(input, fixture, out_dir) -> Path` writing:

```text
<out_dir>/
  manifest.json          # schema_version batched_sysid_v1, 1 structure × 1 direction
  episodes/s00_d00.parquet
```

Episode parquet must include bit-1 metadata keys **and** trajectory columns expected by `BatchedSysIdDataset` / `build_recorded_actions_tensor` (at least `action`, plus whatever loaders require: `step_idx`, observations used by replay collectors — match an existing sim-collected episode schema as closely as practical).

- [ ] **Step 1: Failing test** — export from synthetic or `s00-d03_with_actions` (skip if missing); `BatchedSysIdDataset(out_dir)` loads; `action` norms > 0; metadata `fruiting_system_params` matches bit-1 convert.

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_real_to_batched_sysid.py::test_export_real_to_batched_dataset_actions -q
```

- [ ] **Step 3: Implement exporter** — reuse bit-1 meta; map real columns → batched trajectory columns; set `action` from real `action` (assert non-all-zero or warn + refuse unless `--allow-zero-action` / explicit drive_fill).

- [ ] **Step 4: CLI** — e.g. `--dataset-out /tmp/real_batched_s00_d03`

- [ ] **Step 5: Commit**

---

### Task 3: FR3 / batched open-loop replay smoke

**Files:**
- Create or extend: `robot_replay/example_replay_real_batched_episode.py` (or thin wrapper around gym batched replay)
- Test: optional short CPU/null smoke if feasible; else documented command + headless check

**Consumes:** dataset from Task 2  
**Produces:** recorded-action replay of structure 0 / direction 0 via `replay_batched_sysid_structure` (single candidate = exported params).

- [ ] **Step 1: Failing smoke test or script `--help` + import wiring test**

- [ ] **Step 2: Implement minimal replay example** (viewer null, few frames or full episode)

```bash
uv run python robot_replay/example_replay_real_batched_episode.py \
  --dataset /tmp/real_batched_s00_d03 \
  --viewer null
```

- [ ] **Step 3: Manual visual** with `--viewer gl` when display available

- [ ] **Step 4: README** — fill → export → replay command chain using `s00-d03_with_actions.parquet`

- [ ] **Step 5: Commit**

---

### Task 4: Docs + bug tracking

**Files:**
- `docs/superpowers/specs/2026-08-07-real-to-batched-metadata-parity-design.md` (bit 2 status)
- `docs/real-sysid-pre-post-grasp-fixes.md` (`real-replay-action-zero`)
- `robot_replay/README.md`

- [ ] Document that `s00-d03_with_actions.parquet` is local-only mitigation
- [ ] Document expected future collector contract: full trajectory + non-zero `action`
- [ ] Mark bit 2 success criteria: export loads in `BatchedSysIdDataset`; replay runs without stationary EE; bit 1 parity still green

---

## Success criteria

1. Regenerable local fixture `s00-d03_with_actions.parquet` has non-zero `action`.
2. Exporter writes a 1×1 `batched_sysid_v1` dataset from that file (or future real fixed parquet).
3. Existing batched replay path drives FR3/proxy with those actions (smoke).
4. Zero-action inputs fail loud or require an explicit override; `drive_fill` is labeled temporary.
5. CMA-ES untouched (bit 3).

## Out of scope

- CMA-ES / Young's search (bit 3)
- Fixing upstream collector (human parallel work)
- Multi-structure real batches
- Treating `tcp_velocity` fill as permanent product behavior
