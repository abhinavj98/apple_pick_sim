# Pre-gym scope narrowing — design spec

| Field | Value |
| ----- | ----- |
| **Date** | 2026-07-03 |
| **Status** | Historical |
| **Canonical living doc:** | `docs/handbook-coupled-simulation.md` |
| **Goal** | Delete redundant examples/builders; FR3-only batched path; narrow public API before V.3.3 gym |
| **Execution** | Phased PRs (5 PRs), worktree `chore/pre-gym-cleanup` |

---

## Summary

Remove deprecated and duplicate entry points so the batched coupled stack has **one human-facing example** and **one runtime API** (`BatchedHeterogeneousCoupledSim` + config). Require **FR3 assets** for all batched and single-env coupled builds. Delete homogeneous `replicate(N)` builders and all placeholder TCP builders after migrating tests. Demote low-level `build_*` functions from `coupled_fruiting.__init__` exports; gym continues to use `build_coupled_fruiting_fr3` via explicit import until V.3.4.

No new repository. Gym behavior unchanged except import paths.

---

## Decisions (brainstorming outcomes)

| Topic | Decision |
| ----- | -------- |
| Deletion aggressiveness | **Moderate** — remove examples + exports; keep `diagnostics/` |
| Batched robot | **FR3 required** — no placeholder fallback in build or example |
| Homogeneous batch (`build_batched_coupled_*`) | **Delete** after migrating tests to heterogeneous API |
| Single-env | **FR3-only** — remove `build_coupled_fruiting_placeholder` |
| Public exports | **Narrow** — runtime API only in `__init__.py`; builders explicit-import |
| Gym migration | **Out of scope** (V.3.3+) except import fixes |
| `--inspect-settle` | **Dropped** with legacy monolith (not ported) |
| Execution | **Phased PRs** (not big-bang) |

---

## Target end state

### Public API (`coupled_fruiting/__init__.py`)

**Export:**

- `BatchedHeterogeneousCoupledSim`, `BatchedHeterogeneousCoupledSimConfig`
- `BatchedHeterogeneousBuildResult`, `build_batched_heterogeneous_scene`
- `CoupledFruitingScene` (until gym migrates in V.3.4)
- Existing settle/diagnostics symbols already in `__all__`

**Do not export:**

- `build_coupled_fruiting_fr3` (gym: `from apple_pick_sim.coupled_fruiting.builders import build_coupled_fruiting_fr3`)
- All `build_*_placeholder`, `build_batched_coupled_*`, `build_heterogeneous_coupled_fruiting_fr3`

### Human entry points

| Keep | Remove |
| ---- | ------ |
| `examples/example_batched_heterogeneous_coupled_sim.py` | `examples/example_batched_coupled_fruiting.py` |
| `examples/example_coupled_fruiting.py` (FR3-only) | `examples/inspect_batched_heterogeneous_coupled_sim.py` |
| | `examples/legacy/*` |

### Batched build path (only)

```
BatchedHeterogeneousCoupledSimConfig
  → build_batched_heterogeneous_scene (batched_heterogeneous_build.py)
  → build_heterogeneous_coupled_fruiting_fr3 (builders.py)
  → build_heterogeneous_coupled_cable_scene (batched_build.py)
  → CoupledFruitingScene
```

FR3 assets missing → `FileNotFoundError` with pointer to `assets/fr3/README.md`.

---

## Files to delete

| Path | Reason |
| ---- | ------ |
| `apple_pick_sim/examples/legacy/example_batched_heterogeneous_coupled_fruiting.py` | Deprecated monolith |
| `apple_pick_sim/examples/legacy/README.md` | Empty after monolith removal |
| `apple_pick_sim/examples/inspect_batched_heterogeneous_coupled_sim.py` | Subset of canonical example |
| `apple_pick_sim/examples/example_batched_coupled_fruiting.py` | Homogeneous V.1 example |
| `apple_pick_sim/tests/test_inspect_settle_continue.py` | Legacy `--inspect-settle` only |

---

## Functions to delete from `builders.py`

| Function |
| -------- |
| `build_coupled_fruiting_placeholder` |
| `build_placeholder_tcp_robot_builder` |
| `build_placeholder_tcp_robot_model` |
| `build_batched_coupled_fruiting_placeholder` |
| `build_batched_coupled_fruiting_fr3` |
| `build_heterogeneous_coupled_fruiting_placeholder` |

**Keep:** `build_coupled_fruiting_fr3`, `build_heterogeneous_coupled_fruiting_fr3`, `_assemble_coupled_robot_scene`.

---

## Phased PR plan

Work in worktree: `../apple_pick_sim-pre-gym-cleanup`, branch `chore/pre-gym-cleanup`.

| PR | Scope | Validation |
|----|-------|------------|
| **PR1** | Delete deprecated examples + doc refs | Example parser tests + headless smoke |
| **PR2** | Narrow `__init__.py` exports; fix gym/diagnostic imports | Fast pytest + gym import |
| **PR3** | FR3-only examples + `batched_heterogeneous_build` | Heterogeneous build tests |
| **PR4** | Delete builders; migrate tests + diagnostics | Full `apple_pick_sim/tests/` |
| **PR5** | Add `docs/coupled-sim-api.md`; ROADMAP validation update | ROADMAP commands |

---

## Out of scope

- V.3.3 gym env migration
- Splitting `builders.py`
- New git repository
- Porting `--inspect-settle`

---

## Success criteria

- One batched example remains
- No placeholder/homogeneous builders in `builders.py`
- Narrow `__init__.py` exports
- Test gates pass with FR3 assets
- ROADMAP validation commands updated
