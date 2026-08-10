# Real batched GL replay (FR3 + VIC)

| Field | Value |
| ----- | ----- |
| **Status** | Design approved; implementation pending |
| **Date** | 2026-08-10 |
| **Depends on** | Bit-2 export + physics smoke (`batched_sysid_v1` from real parquet) |
| **Related** | `docs/real-sysid-pre-post-grasp-fixes.md` (C6), `robot_replay/example_replay_real_batched_smoke.py`, `apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py` |

## Purpose

Watch a **complete open-loop FR3 + VIC trajectory** in Newton GL after rebuild and
settle, using a real episode already converted to `batched_sysid_v1`. Settle
remains off-screen; the window shows trajectory frames only.

## Sequencing

| Phase | Scope | Status |
| ----- | ----- | ------ |
| **B** | Extend `example_replay_real_batched_smoke.py` with `--viewer gl\|null` and MMD-grid-style `on_step` render; keep C6 clamps + TCP motion gate | **Next** |
| **A** | Later: teach `example_batched_sysid_mmd_grid.py` the same real clamps / flags so grids, scoring, and GL work on exported real datasets | Follow-on |

## Architecture (phase B)

```text
real parquet → convert --dataset-out → batched_sysid_v1
        │
        ▼
example_replay_real_batched_smoke.py
  • C6: clamp primary length to fixture span
  • C6: fixture fruiting_base_pos
  • FR3 + VIC env (settle during setup, not rendered)
  • replay_batched_sysid_structure(on_step=…)
        │
        ├─ viewer null: existing headless smoke / pytest
        └─ viewer gl: set_model once; begin_frame / log_state / end_frame per control step
```

**Shared physics:** same library call as MMD-grid / CMA paths
(`replay_batched_sysid_structure`). Phase B only adds viewer wiring to the thin
real-world smoke CLI.

## C6 clamps (reminder)

Real catalog primary length can be full rod stock; T-junction sim expects the
fixture support span. Episode `fruiting_base_pos` may sit outside FR3 reach when
`robot_base_pos=(0,0,0)`. Smoke overrides both from the variance fixture so IK
bootstrap succeeds. These are replay alignment overrides, not logged truth
(see fix-doc C6).

## CLI (phase B)

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/s02-d00_action.parquet \
  --dataset-out /tmp/real_batched_s02_d00 --overwrite

uv run python robot_replay/example_replay_real_batched_smoke.py \
  --dataset /tmp/real_batched_s02_d00 \
  --viewer gl \
  --max-frames 0
```

- `--viewer {gl,null}` via `newton.examples.init` (same pattern as MMD-grid).
- No `DISPLAY` / `WAYLAND_DISPLAY` → auto-fallback to `null` (project convention).
- Keep: `--dataset`, `--fixture`, `--structure-idx`, `--seed`, `--max-frames`
  (`<=0` = full episode; default stays short for smoke).
- Exit: print `tcp_motion_m`; non-zero if TCP stationary (same as today).

## Out of scope (phase B)

- Rendering settle or post-grasp weld phases
- Changing `replay_batched_sysid_structure` internals
- MMD-grid real clamps (phase A)
- CMA-ES / scoring / candidate grids
- Single-env `example_gym_replay.py` (different dataset contract)

## Testing

- Keep `test_real_exported_s02_replay_moves_tcp` as the headless physics gate (no GL in CI).
- Add a lightweight test that the smoke CLI accepts `--viewer gl|null` and attaches
  an `on_step` when the viewer is non-null (mock / no window), without requiring a
  GPU display.
- No settle-render tests.

## Docs

- This spec.
- `robot_replay/README.md`: GL command for the smoke script.
- Phase A backlog note: MMD-grid + real clamps.

## Phase A (follow-on, not this slice)

Optional flag or auto-detect on `example_batched_sysid_mmd_grid.py` to apply the
same C6 clamps so `--dataset <real export> --replay-only --viewer gl` works with
grids and scoring without going through the smoke CLI.
