# Batched API GPU hot path and defaults alignment

**Date:** 2026-07-03  
**Status:** Approved design (brainstorming)  
**Scope:** `BatchedHeterogeneousCoupledSim` — init/build, frame, substep; `defaults()` preset  
**Roadmap context:** V.3.1–V.3.2 done; supports V.3.3 gym migration  

---

## Summary

Align `BatchedHeterogeneousCoupledSimConfig.defaults()` with the canonical example CLI (example CLI wins as source of truth), wire disconnected config knobs, and make the full batched hot path GPU-resident from init through `step()` substeps on CUDA + FR3.

Approach: **phased** — PR 1 defaults/knob wiring, PR 2 GPU residency, PR 3 docs.

---

## Goals

1. **`defaults()` matches `example_batched_heterogeneous_coupled_sim.py` with no CLI flags.**
2. Config fields that affect bootstrap/build are wired through (not ignored or hardcoded elsewhere).
3. On CUDA + FR3 + `defaults()`: no `.numpy()` / `.cpu()` inside `coupled_substep`, VIC apply, or action→teleop staging during `step()`.
4. Init settle→weld seed uses device kernels; host read only for checkpoint save and optional diagnostics.
5. Audit docs reflect actual bootstrap status (already batched via `BatchedTemplateIK`) and remaining gaps.

## Non-goals

| Item | Reason |
|------|--------|
| **Runtime K/stiffness scatter (θ without rebuild)** | Separate roadmap slice (V.4 sys-ID / CEM). Today θ is baked at `ModelBuilder.finalize()`; runtime scatter would write new `bend_stiffness` / `stretch_stiffness` / `bend_damping` into VBD GPU arrays per world without full rebuild — not part of this work. |
| Heterogeneous topology within one batch | Backlog |
| Batched gym env migration (V.3.3) | Consumes this API; separate slice |
| Keyboard teleop on CPU | Allowed per GPU rules (frame-rate I/O) |
| Viewer / debug readouts after step | Intentionally scalar |
| CUDA graph capture | Future optimization |
| Newton submodule changes | Stay in `apple_pick_sim/` unless blocked |

---

## Section 1: `defaults()` alignment

**Source of truth:** `example_batched_heterogeneous_coupled_sim.py` with no extra flags.

| Field | Current `defaults()` | Target |
|-------|---------------------|--------|
| `runtime.control_hz` | 60 | **30** |
| `settle_diagnostics` | `None` | **`SettleDiagnosticsConfig()`** (`settle_substeps=5000`) |
| `obs` | `ObsConfig()` | **`None`** (allocate only when viz/gym explicitly requests) |
| `robot.ik_bootstrap_iterations` | 96 | **128** (match `IK_BOOTSTRAP_DEFAULT_ITERATIONS` in `placement.py`) |

### Knob wiring

1. **`ik_bootstrap_iterations`** — pass `robot_cfg.ik_bootstrap_iterations` into `_bootstrap_tcp_per_env(..., ik_iterations=...)` from build / `seed_fix_to_apple_from_settled`.
2. **`robot.per_env_ik: bool = True`** — new config field; replace hardcoded `per_env_ik=True` in `batched_heterogeneous_build.py`.
3. **`skip_ik_bootstrap` / `defer_template_robot_bootstrap`** — document as settle→weld invariants (build forces `True` on weld path); do not expose as user-tunable without validation changes.
4. **`gripper.fix_to_apple` vs `robot.fix_to_apple`** — add `validate()` check or warning when they disagree; build uses `robot.fix_to_apple` via `_gripper_proxy()`.

### Example CLI refactor

`_config_from_args` uses `dataclasses.replace(BatchedHeterogeneousCoupledSimConfig.defaults(), …)` for CLI-only overrides (num_envs, seed, device, viz-driven `obs`). Fields not on CLI inherit from `defaults()`.

### Tests (Phase 1)

- Update `test_batched_heterogeneous_config.py` for new preset values.
- Add `test_defaults_matches_example_cli_config` (synthetic minimal args parity).

---

## Section 2: Hot-path map and GPU residency

### Flow

```
build()
  → sample params (CPU, once)
  → free-apple scene + VBD settle (GPU, all worlds)
  → capture settled body_q
  → welded scene build
  → seed_fix_to_apple_from_settled (target: GPU kernels)
  → _bootstrap_tcp_per_env via BatchedTemplateIK (already batched)
  → prepare_batched_stem_harvest_arrays

step(actions)  [per control frame]
  → validate/clip actions (target: torch vectorized)
  → FR3 teleop or placeholder nudge (target: device upload)
  → substeps × coupled_substep (GPU)

coupled_substep  [per substep]
  → MuJoCo robot (GPU on CUDA)
  → mirror TCP→proxy+apple (Warp)
  → VBD step (GPU)
  → harvest_batched_stem_tension (Warp)
  → wrenches / VIC joint torques (target: no joint_q.numpy in VIC)

gather_obs()  [optional, GPU kernels]
```

### GPU status and targets

| Stage | Module | Today | Phase 2 target |
|-------|--------|-------|----------------|
| Settle capture | `batched_heterogeneous_build.py` | `body_q.numpy().copy()` | `wp.copy` to pooled buffer; checkpoint read once |
| Seed settled→welded | `settle_then_weld.py` | Host reshape + proxy loop | Warp: world i→i copy, per-env grasp offset, zero twists |
| Legacy 1→N broadcast | `batched_build.py` | NumPy | Warp kernel (legacy path only) |
| Quiet cable bodies | `settle_then_weld.py` | NumPy zero | `wp.launch` zero `body_qd` |
| IK bootstrap | `settle_then_weld._bootstrap_tcp_per_env` | Batched IK; CPU diagnostics loop | Wire iterations from config |
| Robot template broadcast | `batched_build._broadcast_robot_state_from_template` | Host loop | Warp scatter (P2) |
| Action clip | `batched_heterogeneous_coupled_sim.py` | Python loop | Torch vectorized clamp |
| Action→velocity | `_velocity_for_world` | `.cpu().tolist()` per env | Upload from `_action_buffer` to `_lin_vels_wp` / `_ang_vels_wp` |
| Teleop advance | `ee_velocity_batched.py` | Python loop from callback | Fast path when actions on device |
| `target_tf` sync | `_sync_target_tf_from_device` | `.numpy()` + rebuild list | Remove from production path; keep debug/viewer |
| Physics substep | `scene.coupled_substep` | GPU | No change |
| VIC torques | `vic_joint_torques_batched.py` | `joint_q.numpy()` each substep | `wp.to_torch(state.joint_q)` |
| Placeholder | `broadcast_joint_q_from_world0` | Host each frame | Test-only / explicit placeholder |
| gather_obs | `batched_obs.py` | GPU | No change |

### Phase 2 priority

1. **P0** — VIC `wp.to_torch`; vectorized clip; device action→teleop.
2. **P1** — GPU seed kernel; device settle capture.
3. **P2** — `quiet_all_cable_bodies` kernel; robot joint broadcast; legacy 1→N kernel.
4. **P3** — Gate placeholder host path to `test_minimal` / explicit `robot.kind='placeholder'`.

### Acceptance rule

CUDA + FR3 + `defaults()`: zero `.numpy()`/`.cpu()` in `coupled_substep`, VIC apply, and action→teleop during `step()`. Init may sync once at end of build for checkpoint/diagnostics only.

### Tests (Phase 2)

| Test | Intent |
|------|--------|
| Extend `test_batched_heterogeneous_coupled_sim.py` | No `.numpy()` on `joint_q` during substeps |
| `test_batched_ik_bootstrap_aligns_all_proxy_targets` | Keep green after iteration wiring |
| New `test_seed_fix_to_apple_device_parity` | CPU ref vs GPU kernel for proxy pose |
| Benchmark | `benchmark_batched_heterogeneous.py` — init + ms/substep |

---

## Section 3: Rollout

Work in isolated worktree: `feature/batched-gpu-hot-path` (or reuse `feature/batched-sim-config`).

### PR 1 — Defaults + knob wiring

Files: `batched_heterogeneous_config.py`, `batched_heterogeneous_build.py`, `settle_then_weld.py`, `example_batched_heterogeneous_coupled_sim.py`, config tests.

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_batched_heterogeneous_config.py \
  apple_pick_sim/tests/test_batched_heterogeneous_build.py \
  apple_pick_sim/tests/test_batched_heterogeneous_coupled_sim.py -q
```

### PR 2 — GPU residency

Files: `settle_then_weld.py`, `batched_heterogeneous_coupled_sim.py`, `vic_joint_torques_batched.py`, `ee_velocity_batched.py`, `ee_impedance_batched.py`, new kernels in `proxy_coupling.py` or `batched_build.py`.

```bash
uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/ -q
uv run python apple_pick_sim/diagnostics/benchmark_batched_heterogeneous.py \
  --num-envs 4 --settle-substeps 100 --robot fr3
```

### PR 3 — Docs

- Update `docs/heterogeneous-batched-vectorization-audit.md`
- Add batched section to `docs/gpu-coupling-optimization.md`
- Optional: `docs/batched-hot-path-map.md`

---

## Error handling

- **Missing FR3 assets:** Keep placeholder fallback + warning; GPU acceptance criteria apply when FR3 assets present.
- **Bootstrap IK failure:** Keep `warn_if_ik_bootstrap_not_converged`; iterations tunable via config after wiring.
- **Actions wrong device:** Existing `validate_actions` — no change.
- **GPU kernel parity:** CPU reference tests (pattern: `test_stem_harvest_cpu_gpu_parity`).

---

## Success criteria

1. `defaults()` matches example CLI with no flags.
2. Section 1 knobs wired or documented as build invariants.
3. CUDA + FR3: no host sync in substep/frame hot paths listed above.
4. Init seed uses device kernels.
5. Audit docs updated.
6. Listed tests green; no ms/substep regression in benchmark.

---

## References

- `docs/vectorized-coupled-fruiting.md` — θ application table, config map
- `docs/heterogeneous-batched-vectorization-audit.md` — prior audit (bootstrap section stale)
- `docs/gpu-coupling-optimization.md` — single-env GPU doc (needs batched section)
- `docs/ROADMAP.md` — V.3.3+ sequencing
- `apple_pick_sim/coupled_fruiting/batched_heterogeneous_coupled_sim.py` — runtime API

---

## Follow-up: `GripperGeometryConfig` (post–PR 1)

**Problem:** `RobotConfig` nests full `GripperProxyConfig`, which includes `fix_to_apple` and `robot_facing_weld`. Those weld flags are owned by `robot.fix_to_apple` and applied in `_gripper_proxy()` at build time, so two knobs expose the same concept and confuse readers.

**Target shape:**

```python
@dataclasses.dataclass(frozen=True)
class GripperGeometryConfig:
    """Gripper proxy shape/mass/placement only — no weld topology."""
    mass: float = PLACEHOLDER_EE_MASS_KG
    # ... other non-weld GripperProxyConfig fields (weld_direction, etc.) as needed

@dataclasses.dataclass(frozen=True)
class RobotConfig:
    fix_to_apple: bool = True  # canonical weld / settle→weld mode
    gripper: GripperGeometryConfig = dataclasses.field(default_factory=GripperGeometryConfig)
```

**Build helper** (replaces ad-hoc `_gripper_proxy` flag patching):

```python
def gripper_proxy_for_build(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    fix_to_apple: bool | None = None,
) -> GripperProxyConfig:
    fix = config.robot.fix_to_apple if fix_to_apple is None else fix_to_apple
    return GripperProxyConfig(
        **dataclasses.asdict(config.robot.gripper),
        fix_to_apple=fix,
        robot_facing_weld=fix,
    )
```

**Migration (single PR after PR 2 or alongside V.3.3 gym):**

1. Add `GripperGeometryConfig` in `batched_heterogeneous_config.py`.
2. Change `RobotConfig.gripper` type; remove gripper weld `validate()` warning (no duplicate field).
3. Update `_gripper_proxy` → `gripper_proxy_for_build`.
4. Update tests and any `dataclasses.replace(..., gripper=GripperProxyConfig(...))` call sites.
5. Keep `GripperProxyConfig` unchanged in `fruiting_system/` (builders still consume it).

**Non-goals:** Do not split or rename `GripperProxyConfig` in the fruiting-system layer; only the batched sim config surface changes.
