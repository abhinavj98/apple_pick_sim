# Support-joint \(k_p\) Sys-ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace primary-\(E\) fitting in the batched Young's grid + CMA path with free support-joint \(k_p\) (shared angular+linear on wire ties, \(\zeta=1\)), while keeping spur/stem \(E\) free; verify with collect → grid and collect → PyCMA sim-to-sim transfer.

**Architecture:** Introduce `SupportKpYoungsCandidate(support_kp, spur, stem)`. `apply_to` updates only spur/stem \(E\) on `FruitingSystemParams`. Per-env support \(k_p\)/\(k_d\) are patched after scene build inside fused replay (fixture support overrides are overwritten; other FIXED joints keep build-time \(\zeta\)). Evolve grid/CMA CLIs, reports, and gates to the new 3-vector. Acceptance is sim-to-sim: fresh collect, then Cartesian grid ranking, then PyCMA fit/overlays.

**Tech Stack:** Python, pycma, existing `batched_sysid_cmaes` / `batched_sysid_multi_replay` / Sinkhorn scoring, `set_fruiting_joint_*_kp(_batched)`, `joint_kd_from_damping_ratio`, `uv` / pytest.

**Spec:** [docs/superpowers/specs/2026-08-04-support-joint-kp-sysid-design.md](../specs/2026-08-04-support-joint-kp-sysid-design.md)

## Global Constraints

- CMA/grid vector: \(\bigl(\log_{10}(k_p^{\mathrm{support}}),\;\log_{10}(E_{\mathrm{spur}}),\;\log_{10}(E_{\mathrm{stem}})\bigr)\)
- Primary \(E\): **fixed** from structure true/fixture params (never a free dim)
- Support \(\zeta\): **fixed at 1** → \(k_d = 2\sqrt{k_p I}\) / \(2\sqrt{k_p m}\) on **support role only**
- Shared numeric \(k_p\) → both angular and linear slots on left+right `primary_support_*`
- Other FIXED joints: fixture `joint_damping_ratio` / defaults unchanged
- Scoring/features/soft-disable: unchanged from V.5.2
- Init: search-box midpoints / `CMA_SEARCH_PARAMS`, **not** GT
- Default support \(k_p\) log10 box: **lower 2.0, upper 6.0** (100 … \(10^6\)), mean **4.0** (\(10^4\), matches fixture); spur/stem keep log10 **[8, 11]**
- TDD: failing tests before production edits
- Run tests: `uv run --env-file pytest.env python -m pytest …` from repo root
- Commits are **approved** for this subagent-driven execution (plan commit steps run as written)

---

## File map

| File | Role |
|------|------|
| `apple_pick_sim/fruiting_system/build.py` | Add per-env angular/linear **kp** batched patch APIs (mirror existing `label_kd_per_env`) |
| `apple_pick_sim/fruiting_system/joint_kd_scaling.py` | Reuse `joint_kd_from_damping_ratio(..., roles=("support",))` for ζ=1 support kd |
| `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` | `SupportKpYoungsCandidate`, log10 maps, GT from manifest, CMA dim semantics |
| `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` | Carry `support_kp` on slots; post-build per-env support kp/kd patch |
| `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py` | Grid CLI: `--support-kp-values` / log10; drop primary-\(E\) axes |
| `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` | `CMA_SEARCH_PARAMS` 3-vector = kp + spur + stem |
| `apple_pick_gym/batched_envs/youngs_modulus_gate_report.py` (+ CMA gate report) | Report support \(k_p\) errors; drop primary-\(E\) fit columns |
| `scripts/gate_youngs_modulus_*.sh` | Point at new collect → grid / CMA recipes (in-place semantic update OK) |
| `README.md`, `docs/youngs-modulus-*.md`, `docs/ROADMAP.md` | Collect → grid and collect → CMA verification commands |
| Tests under `apple_pick_gym/tests/` + `apple_pick_sim/tests/` | Candidate, apply/patch, CLI, gate |

**Rename policy:** Prefer **in-place evolution** of the Young’s tooling (keep CLI entrypoint filenames) with updated help text and report keys. Add `SupportKpYoungsCandidate`; migrate call sites off `YoungsModulusCandidate` (delete or thin-wrap deprecated type once unused).

---

### Task 1: Per-env batched joint kp APIs

**Files:**
- Modify: `apple_pick_sim/fruiting_system/build.py` (extend existing kp batched setters — mirror `label_kd_per_env` on kd)
- Modify: `apple_pick_sim/fruiting_system/__init__.py` if symbols are re-exported
- Test: `apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py` (extend; kd per-env tests already live here)

**Interfaces:**
- Extend existing signatures (do **not** invent `*_batched_per_env` names):

```python
def set_fruiting_joint_angular_kp_batched(
    solver, template_fruiting_fixed_joints,
    label_kp: dict[str, float] | None = None, *,
    label_kp_per_env: Sequence[Mapping[str, float]] | None = None,
    num_envs: int, joints_per_world: int,
) -> dict[str, list[int]]:
    """Exactly one of label_kp (broadcast) or label_kp_per_env (per-env maps)."""

def set_fruiting_joint_linear_kp_batched(...):  # same optional label_kp_per_env
```

- Implementation must match kd (`build.py:1237-1338`):
  - Reuse / generalize `_normalize_batched_label_kd` → shared normalize for kp/kd, or add `_normalize_batched_label_kp`
  - Pack `(num_envs, n_templates)` values like `_batched_kd_values_from_per_env`
  - Extend `_apply_batched_joint_angular_kp_kernel` so `kp_values` is indexed as `w * n_templates + k` (today it only uses `kp_values[k]` — broadcast-only). Also keep widening `joint_penalty_k_min/max`
  - Existing broadcast callers that pass only `label_kp=` must keep working

- [ ] **Step 1: Write failing tests**

```python
def test_set_fruiting_joint_angular_kp_batched_per_env_writes_distinct_values():
    # Build num_envs>=2 T-junction scene; patch env0 support=1e3, env1 support=2e4
    # Assert angular penalty_k at support joints differs per world
    ...

def test_set_fruiting_joint_linear_kp_batched_per_env_writes_distinct_values():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py \
  -q -k "kp_batched_per_env"
```

Expected: FAIL (TypeError / wrong behavior — current kernel ignores per-env)

- [ ] **Step 3: Implement per-env kp on existing batched setters**

- [ ] **Step 4: Run tests to verify they pass** (include existing broadcast kp tests still green)

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_heterogeneous_coupled_fruiting.py \
  -q -k "joint_angular_kp_batched or joint_linear_kp_batched or kp_batched_per_env"
```

Expected: PASS

- [ ] **Step 5: Commit** (approved for this subagent-driven execution)

```bash
git add apple_pick_sim/fruiting_system/build.py apple_pick_sim/tests/
git commit -m "Add per-env label_kp_per_env to batched FIXED-joint kp setters."
```

---

### Task 2: Support-only ζ=1 kd helper + post-build applicator

**Files:**
- Create or extend: `apple_pick_gym/batched_envs/support_joint_penalties.py` (gym-side thin wrapper keeps Newton build APIs free of sys-ID)
- Test: `apple_pick_gym/tests/test_support_joint_penalties.py`

**Interfaces:**
- Produces:
  - `SUPPORT_JOINT_ZETA: float = 1.0`
  - `def apply_per_env_support_joint_penalties(scene: CoupledFruitingScene, support_kp_per_env: Sequence[float], *, num_envs: int, joints_per_world: int, zeta: float = SUPPORT_JOINT_ZETA) -> None`
- Behavior:
  1. Validate `len(support_kp_per_env) == num_envs` and every value `> 0.0` (raise `ValueError` naming `support_kp` otherwise)
  2. Build per-env angular+linear kp maps `[{"support": support_kp_per_env[w]} for w in range(num_envs)]`; apply via Task 1's extended setters:

```python
set_fruiting_joint_angular_kp_batched(
    scene.cable.solver, scene.cable.fruiting_fixed_joints,
    label_kp_per_env=per_env_ang_kp,
    num_envs=num_envs, joints_per_world=joints_per_world,
)
set_fruiting_joint_linear_kp_batched(..., label_kp_per_env=per_env_lin_kp, ...)
```
  3. Read `body_mass = scene.cable.model.body_mass.numpy()`, `body_inertia = scene.cable.model.body_inertia.numpy()`, `joint_child = scene.cable.model.joint_child.numpy()`; `bodies_per_world = scene.layout.bodies_per_world`
  4. For each env `w`, call `joint_kd_from_damping_ratio(zeta=zeta, roles=("support",), fruiting_fixed_joints=scene.cable.fruiting_fixed_joints, body_mass=body_mass, body_inertia=body_inertia, joint_child=joint_child, angular_kp_by_role={"support": support_kp_per_env[w]}, linear_kp_by_role={"support": support_kp_per_env[w]}, body_offset=w*bodies_per_world)` → per-env `(angular_kd, linear_kd)` dicts
  5. Apply resulting per-env support kd via `set_fruiting_joint_angular_kd_batched(..., label_kd_per_env=per_env_ang)` / `set_fruiting_joint_linear_kd_batched(..., label_kd_per_env=per_env_lin)` (existing APIs, confirmed at `apple_pick_sim/fruiting_system/build.py:1237-1338`)
  6. Do **not** touch non-support roles (`primary_spur`, `spur_stem`, `stem_apple` keep their build-time values)

- [ ] **Step 1: Write failing unit tests** (can use a small mock of `joint_kd_from_damping_ratio` math without full FR3 if possible; otherwise `@requires_fr3` scene)

```python
def test_apply_per_env_support_joint_penalties_sets_kp_and_critical_kd():
    # After apply with kp=[1e3, 2e4], zeta=1:
    # support angular/linear kp match; kd ≈ 2*sqrt(kp*I) / 2*sqrt(kp*m)
    # spur_stem kd unchanged from pre-apply snapshot
    ...

def test_apply_rejects_nonpositive_support_kp():
    with pytest.raises(ValueError, match="support_kp"):
        apply_per_env_support_joint_penalties(scene, [0.0])
```

- [ ] **Step 2: Run to verify fail → implement → pass**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_support_joint_penalties.py -q
```

- [ ] **Step 3: Commit** (if approved)

```bash
git commit -m "Add support-only joint kp/kd applicator with zeta=1."
```

---

### Task 3: `SupportKpYoungsCandidate` + log10 / GT helpers

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`
- Modify: `apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py`

**Interfaces:**
- Produces:

```python
class SupportKpYoungsCandidate(NamedTuple):
    support_kp: float
    spur: float
    stem: float

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        """Update spur/stem E only; leave primary (and secondary) unchanged."""
        ...

def candidates_from_log10_vector(x: Sequence[float]) -> SupportKpYoungsCandidate:
    # len 3: log10(support_kp), log10(E_spur), log10(E_stem)

def log10_vector_from_candidate(c: SupportKpYoungsCandidate) -> tuple[float, float, float]: ...

def gt_support_kp_from_dataset(dataset: BatchedSysIdDataset) -> float:
    """Read dataset-level GT support kp from ``manifest['collection']['sim_config']``.

    ``joint_angular_kp_overrides``/``joint_linear_kp_overrides`` are a single
    build-time config for the **whole collection run** (like
    ``sim_config_to_manifest_dict``), not per-structure like
    ``fruiting_system_params``. Every structure in one dataset shares this
    value. Assert angular=="support" and linear=="support" agree within
    ``rel_tol=1e-9`` (design mandates a single shared scalar); raise
    ``ValueError`` naming the dataset path if the key is missing or the two
    disagree.
    """

def gt_support_kp_youngs_candidate_from_structure(
    dataset: BatchedSysIdDataset, structure_idx: int,
) -> SupportKpYoungsCandidate:
    # support_kp from gt_support_kp_from_dataset(dataset) (dataset-wide);
    # spur/stem E from true_params_for_structure(dataset, structure_idx)
```

**Architecture note (confirmed against code):** unlike primary/spur/stem \(E\)
(sampled per-structure via `FruitingSystemParams` DR and stored in
per-episode `fruiting_system_params` metadata), support \(k_p\) today is a
single `sim_build` scalar applied uniformly to every env at build time
(`BatchedHeterogeneousCoupledSimConfig.fruiting_system.joint_angular_kp_overrides`).
It is **not** currently a per-structure DR quantity. This plan does not add
per-structure support-\(k_p\) collection (out of scope); GT is one value per
dataset, and grid/CMA candidates still need **per-env** override capability
(Task 1/2) purely to evaluate *different candidate* \(k_p\) values within one
fused batched replay — GT itself stays uniform across structures in a given
collect run.

- Deprecate/remove `YoungsModulusCandidate` fields usage for primary; migrate helpers that assumed 3×E.

- [ ] **Step 1: Write failing tests**

```python
def test_support_kp_candidate_apply_to_leaves_primary_e_unchanged():
    base = _base_primary_spur_stem()
    e0 = base.primary.youngs_modulus_pa
    out = SupportKpYoungsCandidate(support_kp=5e3, spur=1e8, stem=1e7).apply_to(base)
    assert out.primary.youngs_modulus_pa == pytest.approx(e0)
    assert out.spur.youngs_modulus_pa == pytest.approx(1e8)

def test_candidates_from_log10_vector_round_trip():
    x = (4.0, 9.0, 8.5)
    c = candidates_from_log10_vector(x)
    assert c.support_kp == pytest.approx(1e4)
    assert log10_vector_from_candidate(c) == pytest.approx(x)
```

- [ ] **Step 2: Implement candidate + helpers; migrate unit tests that still construct `YoungsModulusCandidate(primary=...)`**

- [ ] **Step 3: Run**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py -q
```

Expected: PASS

- [ ] **Step 4: Commit** (if approved)

```bash
git commit -m "Add SupportKpYoungsCandidate and drop primary E from phenotype."
```

---

### Task 4: Wire `support_kp` through fused multi-replay

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py`
- Test: `apple_pick_gym/tests/test_batched_sysid_multi_replay.py` (extend)

**Interfaces:**
- Extend `ReplaySlot` with `support_kp: float | None = None`
- In `build_replay_candidate_blocks`, if candidate has `.support_kp`, copy onto each slot in the block
- After successful `build_env_fn(...)` in `replay_multi_structure_candidate_blocks`, if any slot has `support_kp is not None`, call:

```python
apply_per_env_support_joint_penalties(
    env._sim.scene,                              # CoupledFruitingScene (env._sim: BatchedHeterogeneousCoupledSim)
    [slot.support_kp for slot in slots],
    num_envs=env._sim.layout.num_envs,
    joints_per_world=env._sim.layout.joints_per_world,
)
```

**before** `env.reset(seed=replay_seed)`. Verified attribute paths (matches
`apple_pick_gym/batched_envs/batched_sysid_collect.py:123` `env._sim.scene`
and `apple_pick_gym/batched_envs/apple_pick_batched_sysid_env.py:156-159`
`self._sim.layout`, `self._sim.scene.cable`):
- `env._sim.scene.cable.solver` — `SolverVBD`
- `env._sim.scene.cable.model` — for `body_mass` / `body_inertia` (Task 2's ζ=1 kd expansion)
- `env._sim.scene.cable.fruiting_fixed_joints` — world-0 template `(joint_index, label)` pairs
- `env._sim.layout.num_envs`, `env._sim.layout.joints_per_world`
`apply_per_env_support_joint_penalties` (Task 2) internally calls the Task 1
extended batched kp setters (`label_kp_per_env=...`) plus
`set_fruiting_joint_*_kd_batched(..., label_kd_per_env=...)` using these same
handles.

**Settle timing (verified OK against code):** `build_env_fn` settles with
fixture/dataset `sim_build` support \(k_p\) *before* this patch. That is fine
for this scoring path: `env.reset` restores the build snapshot, then
`initialize_batched_env_from_episode_sources` overwrites poses from recorded
GT; subsequent action-replay steps use the patched candidate support \(k_p\)/\(k_d\).
Do **not** move the patch before settle unless a later slice needs
candidate-dependent settle equilibria (Young's \(E\) already differs per env at
build via `per_env_params`; support \(k_p\) does not).

- [ ] **Step 1: Failing test** — fused build with two candidates differing only in `support_kp` yields different support `joint_penalty_k` per env after build (can assert via solver arrays before/after reset)

- [ ] **Step 2: Implement slot plumbing + post-build apply**

- [ ] **Step 3: Run multi-replay tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_multi_replay.py -q
```

- [ ] **Step 4: Commit** (if approved)

```bash
git commit -m "Apply per-env support joint kp during fused sys-ID replay."
```

---

### Task 5: Grid CLI + evaluation migration

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`
- Modify: `apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py`
- Modify: report JSON serializers in the same example / `youngs_modulus_gate_report.py` as needed

**Interfaces:**
- Replace `--log10-e-primary` with `--support-kp-values` (linear Pa-like N/m / N·m/rad shared scalar list) and/or `--log10-support-kp`
- Cartesian product: `support_kp × spur_E × stem_E`
- GT candidate via `gt_support_kp_youngs_candidate_from_structure`
- Winner / error columns: `support_kp` log10 error + spur/stem \(E\) (no primary \(E\))

- [ ] **Step 1: Update CLI tests to expect new flags; assert `--help` text mentions support kp, not primary E**

- [ ] **Step 2: Implement CLI + grid iteration using `SupportKpYoungsCandidate`**

- [ ] **Step 3: Run**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_gate_report.py -q
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py --help
```

- [ ] **Step 4: Commit** (if approved)

```bash
git commit -m "Retarget Young's grid CLI to support kp × spur/stem E."
```

---

### Task 6: CMA CLI + loop + integrity gate

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` (`CMA_SEARCH_PARAMS`)
- Modify: `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` (`fit_youngs_modulus_structures` / ask maps)
- Modify: `apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py`, `test_example_youngs_modulus_cmaes_cli.py`, CMA gate report tests
- Modify: `scripts/gate_youngs_modulus_cmaes.sh` / ranking gate script comments as needed

**Interfaces:**
- `CMA_SEARCH_PARAMS` vector meaning: `[log10_support_kp, log10_spur_E, log10_stem_E]`
- Defaults:

```python
_CMA_SEARCH_LOG10_LOWER = [2.0, 8.0, 8.0]
_CMA_SEARCH_LOG10_UPPER = [6.0, 11.0, 11.0]
# midpoint mean ≈ [4.0, 9.5, 9.5]
```

- Replace / evolve `YoungsModulusCmaBounds` so slot 0 is support-\(k_p\) log10
  bounds (absolute `[2,6]`), **not** `extract_youngs_modulus_cma_bounds(...).primary`
  (fixture primary \(E\) ε-bands). Spur/stem may still come from fixture or keep
  absolute `[8,11]` as today. `resolve_initial_mean_log10` / ask maps must use
  `candidates_from_log10_vector`.
- `candidates_from_log10_e` → `candidates_from_log10_vector` at ask/tell boundaries
- Final-mean overlay / report: support \(k_p\) vs GT from
  `gt_support_kp_from_dataset`; spur/stem \(E\) vs `true_params_for_structure`
- Integrity gate: still no hard GT-error threshold unless already present; update schema keys

- [ ] **Step 1: Failing CLI/loop tests for new vector semantics**

- [ ] **Step 2: Implement CMA mapping + reports**

- [ ] **Step 3: Run focused suite**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_loop.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  apple_pick_gym/tests/test_youngs_modulus_cmaes_gate_report.py \
  apple_pick_gym/tests/test_gate_youngs_modulus_cmaes_script.py -q
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py --help
```

- [ ] **Step 4: Commit** (if approved)

```bash
git commit -m "Retarget Young's CMA-ES to support kp with fixed zeta=1."
```

---

### Task 7: Docs + verification recipe

**Files:**
- Modify: `README.md` (CMA-ES sim-to-sim section)
- Modify: `docs/handbook-youngs-cma.md`, `docs/handbook-youngs-cma.md`
- Modify: `docs/ROADMAP.md` Agent execution notes if those commands are listed
- Modify: `docs/superpowers/specs/2026-08-04-support-joint-kp-sysid-design.md` status → Implemented when done
- Modify: `docs/damping-tuning.md` short note that support \(k_p\) is a sys-ID target with ζ=1 on support only during this path

- [ ] **Step 1: Document both verification paths**

Path 1 — collect + grid:

```bash
uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 --max-steps 200 \
  --output tmp/support_kp_sysid_dataset --overwrite

uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --viewer null \
  --dataset tmp/support_kp_sysid_dataset \
  --output tmp/support_kp_grid \
  --support-kp-values 1e3,1e4,1e5 \
  --log10-e-spur 8.0,9.5,11.0 \
  --log10-e-stem 8.0,9.5,11.0 \
  --overwrite
```

Path 2 — collect + PyCMA:

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --viewer null \
  --dataset tmp/support_kp_sysid_dataset \
  --output tmp/support_kp_cmaes_fit \
  --overwrite
```

(Adjust flag names to match Task 5/6 final CLI.)

- [ ] **Step 2: Commit docs** (if approved)

```bash
git commit -m "Document support-kp sys-ID collect/grid/CMA verification."
```

---

### Task 8: Sim-to-sim acceptance run

**Files:** none (runtime artifacts under `tmp/`)

- [ ] **Step 1: Collect fresh GT dataset** (CUDA; use fixture with known `sim_build` support kp=10000)

```bash
uv run python apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null --num-structures 2 --num-directions 3 --max-steps 200 \
  --output tmp/support_kp_sysid_dataset --overwrite
```

Expected: dataset writes; episodes record `joint_angular_kp_overrides.support`

- [ ] **Step 2: Grid ranking**

Run Task 7 Path 1 grid command. Expected: GT support \(k_p\) (or nearest node) ranks first / within ranking-gate majority policy; inspect overlay/report.

- [ ] **Step 3: PyCMA fit**

Run Task 7 Path 2. Expected: `cmaes_report.json` + overlays; integrity gate smoke green; final-mean support \(k_p\) and spur/stem \(E\) look right vs GT.

- [ ] **Step 4: Record evidence** under `tmp/support_kp_sysid_acceptance/` (or update `.superpowers/sdd/` task report if that workflow is in use). Mark design spec **Implemented**.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Drop primary \(E\); free support \(k_p\) + spur/stem \(E\) | 3, 5, 6 |
| Shared angular+linear support \(k_p\); ζ=1 support-only \(k_d\) | 2, 4 |
| Extend grid + CMA (Approach 1) | 5, 6 |
| Fused replay apply path | 1, 4 |
| GT from `sim_build` overrides | 3 |
| Search box defaults / not GT init | 6 |
| Sim-to-sim collect → grid and collect → CMA | 7, 8 |
| Non-goals (free ζ, split ang/lin, real-data gate) | respected — not tasked |

## Placeholder / consistency self-check

- No TBD steps; default log10 support \(k_p\) bounds fixed at [2, 6] / mean 4.
- Candidate type name `SupportKpYoungsCandidate` is consistent across tasks.
- Post-build applicator is the sole place support ζ=1 is enforced for candidates (fixture global ζ may still damp other welds).
