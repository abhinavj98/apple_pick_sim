# Sinkhorn Fixed-Scale Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop near-static hold `tcp_velocity` from dominating Sinkhorn by replacing GT-std z-score with fixed physical scales, and drop `action` from score-time `STATE_VECTOR` (replay still drives 19D).

**Architecture:** Keep GT-only centering (`μ_GT`). Change the divisor from `std(GT)` (with `1e-6→1` floor) to a fixed `STATE_VECTOR_PHYS_SCALE` vector mirrored onto `[s, Δs]`, with trailing hold/dir one-hot columns uncentered and scaled by 1. Drop `"action"` from `STATE_VECTOR_FIELDS` / `build_state_matrix` only.

**Tech Stack:** Python, NumPy, existing `NormalizationStats` / `wasserstein.py` Sinkhorn path, pytest + `uv run --env-file pytest.env`.

**Spec:** `docs/superpowers/specs/2026-08-14-sinkhorn-fixed-scale-normalization-design.md`

## Global Constraints

- Work on `feature/real-replay-parallel-sysid` (do not edit `main`)
- TDD: failing test before production code
- Run tests: `uv run --env-file pytest.env python -m pytest -p no:launch_testing <path> -q`
- Candidate statistics never enter normalization (GT-only `μ`)
- Do **not** remove `action` from bags, `REQUIRED_ARRAY_KEYS`, collector, or replay drive
- Do **not** revive or redesign the stale `biased_mmd2` MMD grid path; one-line stale note only
- `s_phys` is a fixed constant table (not fit at runtime)
- Same `s_phys` on level and Δ halves
- Hold (and any trailing) one-hot: no mean-centering, scale = 1

## File map

| Path | Responsibility |
| --- | --- |
| `apple_pick_sim/system_id/mmd_features.py` | Drop `action` from `STATE_VECTOR_FIELDS`; stop concatenating it in `build_state_matrix`; add `STATE_VECTOR_PHYS_SCALE` + `transition_feature_scale(n_features)` |
| `apple_pick_sim/system_id/mmd.py` | `fit_gt_normalization` uses fixed scale (keep `NormalizationStats.std` as the divisor field for API stability); zero mean on trailing one-hot cols |
| `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` | Module docstring: mark `biased_mmd2` scoring path stale |
| `apple_pick_sim/tests/test_mmd_features.py` | Update state-matrix order/width tests (no action block) |
| `apple_pick_sim/tests/test_mmd.py` | Rewrite normalization tests for fixed scale + no z-score explosion |
| `docs/handbook-sysid-scoring.md` | Document fixed-scale norm + action dropped from score |
| `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md` | Note amendment: action no longer in Sinkhorn `STATE_VECTOR` |

## Verification notes (pre-plan)

Confirmed against current code:

- Column order today: `ft(6) + tcp_vel(6) + action(N) + tcp(3) + apple(3) + woody(3×J) + bend(J)`. After drop: **26** dims for `J=2` (CMA).
- Woody/bend scales are uniform across junctions (all woody `0.02`, both bends `0.05`), so non-CMA test names (`joint_a`/`joint_b`) still match the scale vector length.
- `fit_gt_normalization` is called only on **transition** matrices (`[s, Δs, extras…]`) from `wasserstein.py` (and stale MMD grid).
- `build_state_matrix` currently uses `action` for `n_frames`; keep reading it for shape, do not concatenate.

---

### Task 1: Drop `action` from score-time `STATE_VECTOR`

**Files:**
- Modify: `apple_pick_sim/system_id/mmd_features.py`
- Test: `apple_pick_sim/tests/test_mmd_features.py`

**Interfaces:**
- Consumes: existing `REQUIRED_ARRAY_KEYS` (still includes `"action"`)
- Produces: `STATE_VECTOR_FIELDS` without `"action"`; `build_state_matrix` width = `6+6+3+3+3*J+J` (= 26 for `J=2`)

- [ ] **Step 1: Update the failing order test (remove action block from expected)**

In `test_build_state_matrix_uses_exact_feature_order`, delete the `# action` expected values (`20..25`) so expected concatenates `tcp_velocity` → `tcp_pos` directly. Keep fixture `arrays["action"]` present (still required).

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py::test_build_state_matrix_uses_exact_feature_order \
  -q -p no:launch_testing
```

Expected: FAIL — actual state still contains the action columns / length mismatch.

- [ ] **Step 3: Minimal production change**

In `mmd_features.py`:

1. Remove `"action"` from `STATE_VECTOR_FIELDS` only (leave `REQUIRED_ARRAY_KEYS`).
2. In `build_state_matrix`, keep using `action` to resolve `n_frames`, but remove `_as_2d(arrays["action"], ...)` from the `columns` list.

```python
STATE_VECTOR_FIELDS: tuple[str, ...] = (
    "ft_wrist",
    "tcp_velocity",
    "tcp_pos",
    "apple_pos",
    "woody_part_start_pos",
    "woody_bending_angles",
)

# build_state_matrix columns (action used only for n_frames):
columns = [
    _as_2d(arrays["ft_wrist"], name="ft_wrist", n_frames=n_frames),
    _as_2d(arrays["tcp_velocity"], name="tcp_velocity", n_frames=n_frames),
    _as_2d(arrays["tcp_pos"], name="tcp_pos", n_frames=n_frames),
    _as_2d(arrays["apple_pos"], name="apple_pos", n_frames=n_frames),
    _stack_woody(...),
    build_bending_angles(...),
]
```

- [ ] **Step 4: Fix remaining feature-width tests in the same file**

Update any assertion that assumed `state.shape[1]` included action dims (search for comments/`20.0` action offsets / `* 2` transition widths). Collector tests that only check recorded `action` arrays must remain green without changes.

- [ ] **Step 5: Run feature tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd_features.py -q -p no:launch_testing
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add apple_pick_sim/system_id/mmd_features.py apple_pick_sim/tests/test_mmd_features.py
git commit -m "$(cat <<'EOF'
Drop action from Sinkhorn STATE_VECTOR score features

Replay bags still require action for drive; scoring no longer
concatenates it into build_state_matrix.
EOF
)"
```

---

### Task 2: Fixed physical scale in `fit_gt_normalization`

**Files:**
- Modify: `apple_pick_sim/system_id/mmd_features.py` (add scale constants + helper)
- Modify: `apple_pick_sim/system_id/mmd.py`
- Test: `apple_pick_sim/tests/test_mmd.py`

**Interfaces:**
- Consumes: post-Task-1 state width (26 for `J=2`)
- Produces:
  - `STATE_VECTOR_PHYS_SCALE: tuple[float, ...]` length 26
  - `transition_feature_scale(n_features: int) -> np.ndarray`
  - `fit_gt_normalization(gt) -> NormalizationStats` with `stats.std` = physical scale divisor (field name kept)

Scale table (exact values from spec):

```python
STATE_VECTOR_PHYS_SCALE: tuple[float, ...] = (
    # ft_wrist F
    3.0, 3.0, 3.0,
    # ft_wrist τ
    0.5, 0.5, 0.5,
    # tcp_velocity v
    0.02, 0.02, 0.02,
    # tcp_velocity ω
    0.02, 0.02, 0.02,
    # tcp_pos
    0.01, 0.01, 0.01,
    # apple_pos
    0.02, 0.02, 0.02,
    # woody primary_spur, spur_stem (junction order; both 0.02)
    0.02, 0.02, 0.02,
    0.02, 0.02, 0.02,
    # woody_bending_angles
    0.05, 0.05,
)
```

Helper:

```python
def transition_feature_scale(n_features: int) -> np.ndarray:
    """Return divisor vector for [s, Δs, trailing one-hots]."""
    state = np.asarray(STATE_VECTOR_PHYS_SCALE, dtype=np.float64)
    state_dim = int(state.size)
    if n_features < 2 * state_dim:
        raise ValueError(
            f"transition features width {n_features} < 2*state_dim={2 * state_dim}"
        )
    n_extra = int(n_features) - 2 * state_dim
    return np.concatenate([state, state, np.ones(n_extra, dtype=np.float64)])
```

- [ ] **Step 1: Write failing normalization tests**

Replace `test_gt_normalization_is_per_feature_and_uses_gt_only_statistics` with tests that encode the new contract. Example:

```python
def test_fit_gt_normalization_uses_fixed_physical_scale_not_gt_std():
    from apple_pick_sim.system_id.mmd_features import (
        STATE_VECTOR_PHYS_SCALE,
        transition_feature_scale,
    )

    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    # Minimal transition row width: [s, Δs] only (no one-hot)
    n = 2 * state_dim
    gt = np.zeros((3, n), dtype=np.float64)
    # Column 0 is Fx: give GT real variance that must NOT become the divisor
    gt[:, 0] = [0.0, 3.0, 6.0]
    # Matching Δ column (index state_dim) left at 0

    stats = fit_gt_normalization(gt)
    scale = transition_feature_scale(n)
    np.testing.assert_allclose(stats.std, scale)
    np.testing.assert_allclose(stats.mean[0], 3.0)
    # Candidate residual 3 N on Fx → 3/3 = 1.0 after apply, not 3/std(GT)=3/sqrt(6)
    cand = np.zeros((1, n), dtype=np.float64)
    cand[0, 0] = 6.0  # 3 N above GT mean
    out = apply_normalization(cand, stats)
    assert out[0, 0] == pytest.approx(1.0)


def test_near_zero_gt_velocity_does_not_explode_candidate_residual():
    from apple_pick_sim.system_id.mmd_features import STATE_VECTOR_PHYS_SCALE

    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    n = 2 * state_dim
    vx = 6  # index of tcp_velocity vx in STATE_VECTOR after dropping action
    assert STATE_VECTOR_PHYS_SCALE[vx] == pytest.approx(0.02)

    gt = np.zeros((4, n), dtype=np.float64)
    gt[:, vx] = [0.0, 1e-4, -1e-4, 0.0]  # tiny hold variance
    stats = fit_gt_normalization(gt)
    cand = np.zeros((1, n), dtype=np.float64)
    cand[0, vx] = 0.01  # 1 cm/s residual
    out = apply_normalization(cand, stats)
    # Fixed scale 0.02 → 0.5; old GT-std path would be O(100)
    assert out[0, vx] == pytest.approx(0.01 / 0.02)
    assert abs(out[0, vx]) < 2.0


def test_trailing_onehot_is_not_mean_centered():
    from apple_pick_sim.system_id.mmd_features import STATE_VECTOR_PHYS_SCALE

    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    n_holds = 4
    n = 2 * state_dim + n_holds
    gt = np.zeros((2, n), dtype=np.float64)
    gt[0, -4:] = [1, 0, 0, 0]
    gt[1, -4:] = [0, 1, 0, 0]
    stats = fit_gt_normalization(gt)
    np.testing.assert_allclose(stats.mean[-4:], 0.0)
    np.testing.assert_allclose(stats.std[-4:], 1.0)
    out = apply_normalization(gt, stats)
    np.testing.assert_allclose(out[0, -4:], [1, 0, 0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd.py -q -p no:launch_testing
```

Expected: FAIL (still GT-std behavior / missing helpers).

- [ ] **Step 3: Add scale constants + helper in `mmd_features.py`**

Add `STATE_VECTOR_PHYS_SCALE` and `transition_feature_scale` as above. Optionally assert `len(STATE_VECTOR_PHYS_SCALE)` matches a 2-junction `build_state_matrix` width in a unit test (already covered indirectly).

- [ ] **Step 4: Implement `fit_gt_normalization`**

```python
def fit_gt_normalization(gt: np.ndarray, eps: float = 1.0e-6) -> NormalizationStats:
    """Fit GT mean; use fixed physical scales as divisors.

    ``eps`` is retained for call-site compatibility but ignored: scale no longer
    comes from GT std. Trailing columns beyond ``2 * len(STATE_VECTOR_PHYS_SCALE)``
    (hold/dir one-hots) use mean=0 and scale=1.
    """
    del eps  # unused; kept for signature compatibility
    from apple_pick_sim.system_id.mmd_features import (
        STATE_VECTOR_PHYS_SCALE,
        transition_feature_scale,
    )

    gt_arr = _as_feature_matrix(gt, name="gt")
    mean = np.mean(gt_arr, axis=0)
    scale = transition_feature_scale(gt_arr.shape[1])
    state_dim = len(STATE_VECTOR_PHYS_SCALE)
    mean = mean.copy()
    mean[2 * state_dim :] = 0.0
    return NormalizationStats(mean=mean, std=scale)
```

Leave `apply_normalization` unchanged (`(x - mean) / std`).

- [ ] **Step 5: Run mmd + feature + wasserstein unit tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_sim/tests/test_mmd.py \
  apple_pick_sim/tests/test_mmd_features.py \
  apple_pick_sim/tests/test_wasserstein.py \
  -q -p no:launch_testing
```

Expected: PASS. If `test_wasserstein.py` fails only on absolute Sinkhorn magnitudes (scale change), update assertions to the new scale or assert relative ranking only — do not revert the scale table.

- [ ] **Step 6: Commit**

```bash
git add apple_pick_sim/system_id/mmd_features.py apple_pick_sim/system_id/mmd.py apple_pick_sim/tests/test_mmd.py
git commit -m "$(cat <<'EOF'
Use fixed physical scales for Sinkhorn GT normalization

Replace GT-std z-score with STATE_VECTOR_PHYS_SCALE so near-static
hold velocity cannot dominate; leave one-hot columns uncentered.
EOF
)"
```

---

### Task 3: Mark stale MMD path + update docs

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` (module docstring)
- Modify: `docs/handbook-sysid-scoring.md`
- Modify: `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md`

**Interfaces:**
- Consumes: Task 1–2 behavior
- Produces: docs/comments only (no API change)

- [ ] **Step 1: Stale note on MMD grid module**

At the top of `batched_sysid_mmd_grid.py` docstring, add one sentence:

```text
Note: the biased_mmd2 scoring helpers in this module are stale; active
Young's / CMA ranking uses Sinkhorn in wasserstein.py. They still call
fit_gt_normalization and therefore inherit fixed physical scales.
```

- [ ] **Step 2: Update `docs/handbook-sysid-scoring.md`**

Replace the “GT z-score / clamp tiny std to eps” bullet with fixed physical scale + link to the 2026-08-14 design spec. State that `action` is required in bags but not present in score-time `STATE_VECTOR`.

- [ ] **Step 3: Amend alignment spec**

In `docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md`, under the `STATE_VECTOR` section / non-goals about “full 19D action”, add a short amendment pointing to `2026-08-14-sinkhorn-fixed-scale-normalization-design.md`: action removed from Sinkhorn features; replay still 19D.

- [ ] **Step 4: Commit**

```bash
git add apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py \
  docs/handbook-sysid-scoring.md \
  docs/superpowers/specs/2026-08-13-real-sim-cma-feature-alignment-design.md
git commit -m "$(cat <<'EOF'
Document fixed-scale Sinkhorn norm and stale MMD scoring path
EOF
)"
```

---

### Task 4: Optional smoke on converted s09 (manual)

**Files:** none (runtime verification)

- [ ] **Step 1: Re-run the tiny kp/E grid with export** (requires existing `tmp/real_batched_s09_d00`)

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_grid_s09_fixedscale \
  --viewer null \
  --support-kp-values 1e3,1e4 \
  --log10-e-spur 9.0 \
  --log10-e-stem 9.0,8.0 \
  --export-replays \
  --overwrite
```

- [ ] **Step 2: Feature-group ablation sanity**

Re-run the leave-one-group-out script from the s09 investigation (or equivalent). Expect:

- Aggregate Sinkhorn ≪ 3e4 (order-of-magnitude drop)
- `tcp_velocity` no longer ~93% of cost
- No `action` block in state features

Do not commit `tmp/` outputs.

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| Drop `action` from `STATE_VECTOR_FIELDS` / `build_state_matrix` | 1 |
| Keep `action` in `REQUIRED_ARRAY_KEYS` / bags / replay | 1 (explicit non-change) |
| `(x − μ_GT) / s_phys` with locked table | 2 |
| Same scale on `s` and `Δs` | 2 (`transition_feature_scale`) |
| Hold one-hot: no centering, scale 1 | 2 |
| Remove GT-std `1e-6→1` floor for scored blocks | 2 |
| Mark MMD/`biased_mmd2` path stale | 3 |
| Unit tests for order + fixed scale + no velocity explosion | 1, 2 |
| s09 grid smoke | 4 (manual) |
