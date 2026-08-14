# V.3.2 close-out design

| Field | Value |
| ----- | ----- |
| **Status** | Historical |
| **Canonical living doc:** | `docs/handbook-coupled-simulation.md` |
| **Date** | 2026-07-03 |
| **Slice** | V.3.2 close-out (housekeeping after thin example + library API ship) |
| **Next slice (out of scope here)** | V.3.3 — `ApplePickBatchedBaseEnv` |

---

## Goal

Finish the V.3.2 migration so the repository has a single, unambiguous canonical entry point for batched heterogeneous coupled simulation:

**`apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py`**

The pre–V.3.2 monolith (`example_batched_heterogeneous_coupled_fruiting.py`) moves under `examples/legacy/`. All user-facing and agent-facing documentation points at the thin example and the `BatchedHeterogeneousCoupledSim` library API. The legacy script remains runnable for `--inspect-settle` and other flags not yet ported; it is not deleted.

This change set is **atomic**: file move, test repoint, and full documentation sweep land together in one PR.

---

## Layout and file moves

### Target directory structure

```text
apple_pick_sim/examples/
├── example_batched_heterogeneous_coupled_sim.py   # canonical entry point (unchanged path)
├── inspect_batched_heterogeneous_coupled_sim.py   # dev-only inspector (unchanged path)
├── example_batched_coupled_fruiting.py            # homogeneous batched (unchanged)
├── example_coupled_fruiting.py                    # single-world coupled (unchanged)
├── …                                              # other examples (unchanged)
└── legacy/
    └── example_batched_heterogeneous_coupled_fruiting.py   # moved monolith
```

### Move operation

| From | To |
| ---- | -- |
| `apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py` | `apple_pick_sim/examples/legacy/example_batched_heterogeneous_coupled_fruiting.py` |

Use `git mv` so history is preserved. Do **not** add `legacy/__init__.py` — examples are run as scripts, not imported as a package.

### Deprecation banner (legacy monolith)

Replace the legacy file's module docstring opening with a clear deprecation block. Keep the existing run-command examples below it, but update every path to `apple_pick_sim/examples/legacy/example_batched_heterogeneous_coupled_fruiting.py`.

```python
"""DEPRECATED — pre–V.3.2 monolith retained for unmigrated CLI flags.

Canonical entry point: ``example_batched_heterogeneous_coupled_sim.py`` (sibling of
``legacy/``). Library API: ``apple_pick_sim.coupled_fruiting.BatchedHeterogeneousCoupledSim``.

This script remains for ``--inspect-settle`` and other flags not yet ported to the thin
example. Do not extend; new work belongs in the library + thin example.

Run from the repository root::

    uv run python apple_pick_sim/examples/legacy/example_batched_heterogeneous_coupled_fruiting.py \\
      …
"""
```

### Canonical example docstring (thin wrapper)

Update `example_batched_heterogeneous_coupled_sim.py` module docstring so the legacy reference uses the new path:

- Change “The monolithic `example_batched_heterogeneous_coupled_fruiting.py` is deprecated” to “The legacy monolith lives at `legacy/example_batched_heterogeneous_coupled_fruiting.py` (deprecated; `--inspect-settle` only).”
- Keep existing headless smoke and interactive run examples unchanged (they already point at the canonical script).

### `inspect_batched_heterogeneous_coupled_sim.py`

Leave at `apple_pick_sim/examples/inspect_batched_heterogeneous_coupled_sim.py`. No move. Document as **dev-only** (see Documentation updates); do not add to README.

---

## Documentation updates

| Document | Required changes |
| -------- | ---------------- |
| **`README.md`** | Remove the standalone **Legacy** paragraph that advertises the monolith for `--inspect-settle`. Under the heterogeneous batched section, state that `example_batched_heterogeneous_coupled_sim.py` is the canonical entry point. Add a single sentence that unmigrated flags (including `--inspect-settle`) live under `examples/legacy/` without listing legacy run commands. Do **not** mention `inspect_batched_heterogeneous_coupled_sim.py`. |
| **`docs/ROADMAP.md`** | V.3.1 and V.3.2 remain checked done. **Current focus** stays V.3.3. Update the canonical/legacy line under [V] milestones to: canonical = `example_batched_heterogeneous_coupled_sim.py`; legacy = `examples/legacy/example_batched_heterogeneous_coupled_fruiting.py`. Agent execution notes validation block already targets the thin example — confirm paths are correct after the move. |
| **`docs/CODEBASE_GUIDE.md`** | **Directory map** (`apple_pick_sim/examples/` row): name `example_batched_heterogeneous_coupled_sim.py` as the canonical batched heterogeneous example; note `legacy/` for the deprecated monolith. **Remove** the “Known gaps → Batched sim API extraction” row (V.3.1 shipped `BatchedHeterogeneousCoupledSim`). Add one line for `inspect_batched_heterogeneous_coupled_sim.py`: dev-only visual inspector for the library API, not a user entry point. Bump **Last reviewed** to 2026-07-03. |
| **`docs/vectorized-coupled-fruiting.md`** | Replace all references to `example_batched_heterogeneous_coupled_fruiting.py` at the old path with either `example_batched_heterogeneous_coupled_sim.py` (canonical flows, run commands, file table) or `legacy/example_batched_heterogeneous_coupled_fruiting.py` (behaviour documented only on the legacy script, e.g. animated settle with `--no-fix-to-apple`, `--only-vbd`, `--inspect-settle`). Update the code-map table row for the heterogeneous entry point. |
| **`AGENTS.md`** | Under **Repository layout**, add that the canonical batched heterogeneous demo is `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py` and that `apple_pick_sim/examples/legacy/` holds deprecated scripts. |
| **`.cursor/rules/apple-pick-sim.mdc`** | Under **Layout**, note `apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py` as the canonical batched heterogeneous example; `examples/legacy/` for deprecated monoliths. Keep the existing `example_apple_stem.py` run example as the generic “how to run” snippet. |

### Grep cleanup (secondary docs)

Run `rg example_batched_heterogeneous_coupled_fruiting` from repo root after edits; expect hits only in legacy script, legacy path references, and `test_inspect_settle_continue.py`. Update these files as part of the same change set:

| File | Action |
| ---- | ------ |
| `docs/damping-tuning.md` | Point batched-example references at `legacy/example_batched_heterogeneous_coupled_fruiting.py` (joint override tables describe legacy-only constants). |
| `docs/material-parameter-sampling.md` | Same: legacy path for `--only-vbd` stability note. |
| `docs/heterogeneous-batched-vectorization-audit.md` | Reframe as an audit of the legacy monolith's hot path; canonical API is `BatchedHeterogeneousCoupledSim`. Update paths and entry-point table. |

Do **not** change `docs/ROADMAP.md` sequencing beyond path/status wording — V.3.3+ slices are untouched.

---

## Tests

### `test_inspect_settle_continue.py` — legacy path only

This test covers `_settle_inspect_continue_requested`, a helper used only by the legacy monolith's `--inspect-settle` flow. **Do not port `--inspect-settle` to the thin example.**

Changes:

1. Set `sys.path` insertion to `apple_pick_sim/examples/legacy/` (not `examples/`).
2. Import from `example_batched_heterogeneous_coupled_fruiting` unchanged (module name stays the same; only directory moves).
3. Add a one-line module docstring note: tests legacy `--inspect-settle` helper only.

No new tests required for the move itself. Existing V.3.2 tests (`test_example_batched_heterogeneous_coupled_sim.py`, `test_heterogeneous_coupled_fruiting.py`) already target the library and thin example — they must remain green.

---

## Validation commands

Run from repository root after implementation:

```bash
# Fast heterogeneous + thin-example gate
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py \
  apple_pick_sim/tests/test_example_batched_heterogeneous_coupled_sim.py \
  apple_pick_sim/tests/test_inspect_settle_continue.py -q

# Headless smoke — canonical entry point
uv run python apple_pick_sim/examples/example_batched_heterogeneous_coupled_sim.py \
  --viewer null --num-frames 10 --settle-substeps 50 --num-envs 2

# Broader fast gate (recommended before merge)
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q -m "not slow"

# Grep sanity — no stale top-level monolith path
rg 'examples/example_batched_heterogeneous_coupled_fruiting\.py' \
  --glob '!apple_pick_sim/examples/legacy/**' \
  --glob '!docs/superpowers/**'
# Expected: zero matches
```

---

## Out of scope

| Item | Rationale |
| ---- | --------- |
| Port `--inspect-settle` to `example_batched_heterogeneous_coupled_sim.py` | Explicitly deferred; legacy script retains the flag |
| Delete `legacy/example_batched_heterogeneous_coupled_fruiting.py` | Retained for unmigrated CLI surface |
| V.3.3 — `ApplePickBatchedBaseEnv`, `gather_batched_obs`, obs v3 parity | Next roadmap slice; separate PR |
| V.3.4 / V.3.5 gym migration | Later slices |
| Move or promote `inspect_batched_heterogeneous_coupled_sim.py` to README | Dev-only; CODEBASE_GUIDE one-liner only |
| Changes under `newton/` submodule | No Newton edits required |

---

## Implementation checklist

Execute in order within a single atomic change set:

1. **Worktree** — If on `main`/`master`, create or use a feature worktree per `.cursor/rules/worktree-feature-dev.mdc`.
2. **Move** — `git mv apple_pick_sim/examples/example_batched_heterogeneous_coupled_fruiting.py apple_pick_sim/examples/legacy/`.
3. **Legacy docstring** — Apply deprecation banner; update all embedded run paths to `examples/legacy/…`.
4. **Canonical docstring** — Update thin example legacy pointer to `legacy/…` path.
5. **Test repoint** — `test_inspect_settle_continue.py` → import from `examples/legacy/`.
6. **README** — Canonical emphasis; remove legacy paragraph; no inspect script mention.
7. **ROADMAP** — Update canonical/legacy paths; confirm V.3.3 is next focus.
8. **CODEBASE_GUIDE** — Directory map, remove stale V.3.1 gap row, add inspect one-liner, bump review date.
9. **vectorized-coupled-fruiting.md** — Full path sweep (canonical vs legacy).
10. **AGENTS.md** — Canonical + legacy layout note.
11. **apple-pick-sim.mdc** — Canonical example + `legacy/` note.
12. **Secondary docs** — `damping-tuning.md`, `material-parameter-sampling.md`, `heterogeneous-batched-vectorization-audit.md`.
13. **Grep pass** — `rg example_batched_heterogeneous_coupled_fruiting`; fix any remaining stale top-level paths.
14. **Validate** — Run all commands in [Validation commands](#validation-commands); all must pass.
15. **Self-check** — No TBD/TODO placeholders; scope is close-out only (not V.3.3).

---

## Approval record

| Decision | Choice |
| -------- | ------ |
| Monolith location | `examples/legacy/` |
| `test_inspect_settle_continue` | Repoint to legacy import path only; no `--inspect-settle` port |
| Documentation sweep | Full: README, ROADMAP, CODEBASE_GUIDE, vectorized-coupled-fruiting, AGENTS.md, `.cursor/rules`, grep cleanup |
| `inspect_batched_heterogeneous_coupled_sim.py` | Dev-only; one-liner in CODEBASE_GUIDE; not in README |
| Delivery | Single atomic PR (move + docs together) |
