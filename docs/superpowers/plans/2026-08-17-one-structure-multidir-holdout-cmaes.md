# One-Structure Multi-Direction CMA with 5/3 Holdout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. One task per subagent; do not start a task before the previous task's tests pass.

**Goal:** Convert `robot_replay/new_data/s09/` (eight compiled `s09-dNN.parquet`) into one 1-structure × 8-direction `batched_sysid_v1` dataset, fit \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\) with CMA-ES on a seeded random sample of **five** directions, freeze the fit, and report Sinkhorn + Cartesian F/T error plus magnitude/trend gates on the **three** directions the optimizer never saw.

**Spec (contract):** `docs/superpowers/specs/2026-08-17-one-structure-multidir-holdout-cmaes-design.md` (Approved). Where this plan and the spec disagree, the spec wins — stop and report.

**Architecture:** Keep the pycma ask/tell loop (`fit_youngs_modulus_structures`) and the shipped real `vic_pose` CMA wiring untouched. Add (a) a folder convert that emits eight episodes with per-direction metadata, (b) per-slot weld pose / gripper / arm joints in the batched real replay build so each env replays *its own* pull, (c) an optional `direction_indices` argument threaded from the CLIs through `prepare_youngs_modulus_structure` / `evaluate_youngs_modulus_*`, and (d) an opt-in holdout mode in the CMA CLI that fits on train dirs, then evaluates the frozen `final_mean` and the shipped `initial_mean_log10` baseline on val dirs and writes `holdout_report.json`.

**Tech Stack:** Python, NVIDIA Warp / Newton, pycma via `batched_sysid_cmaes.py`, pytest via `uv run --env-file pytest.env`.

## Global Constraints

- Work on `feature/real-replay-parallel-sysid` in this worktree. Do not create another worktree; do not edit `main`.
- TDD per `.cursor/rules/test-driven-development.mdc`: failing test first, confirm it fails for the expected reason, then the smallest production change.
- Test command shape (always `-p no:launch_testing`):

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing <path> -q
```

- **Default behavior must not change.** No split flags ⇒ CMA scores all usable dirs, writes no `holdout_report.json`. Sim-sim twist-`vic` CMA (GT, all dirs) and the shipped 1×1 real path stay byte-compatible in behavior.
- Keep the one-structure `vic_pose` `SystemExit` in both the grid and CMA CLIs. This slice is still one structure.
- Phenotype unchanged; primary \(E\) fixed; real search floor stays \(\log_{10} E = 7\) for spur/stem; `CMA_SEARCH_PARAMS` dict values unchanged.
- Non-goals (do not implement): second tree / multi-structure merge, Sinkhorn feature or scale changes, moving gym collect / MMD off twist `vic`, retargeting `initial_mean_log10`.
- Convert F/T LPF + 30 Hz decimation already landed (`a3feddd`, `8b9645d`); scoring reads `ft_wrist_lpf` via `mmd_features.scored_ft_wrist`. Spec Open risk 3 is closed — do not redo it.
- CUDA `exit 139` is a known non-blocker: shrink `population_size` / `max_generations` for local smoke, record the knobs actually run, restore shipped values before commit.
- Grid CLI direction-subset flags are **out of scope** (spec slice 3 mentioned them; this slice's acceptance is CMA-only). Thread `direction_indices` through `prepare`/`evaluate` so the CMA CLI can call them; do not add grid flags.
- Production `apple_pick_sim` must not import `apple_pick_gym`. Write `collection.sim_config` via `sim_config_to_manifest_dict` on a `BatchedHeterogeneousCoupledSimConfig` built in `apple_pick_sim` from `gym_defaults` + fixture `parse_sim_build` (same knobs as `real_replay_sim_config`, no gym import).

## Review amendments (2026-08-17)

Verified against current convert/replay/CMA code. These override earlier draft wording in this file:

1. **One-hot `n_directions` stays 8.** Do not set it to `len(selected)`. See Task 5.
2. **Per-dir gripper is set on `ReplaySlot.gripper`**, not re-derived inside the builder (the driver already passes `per_env_grippers=[slot.gripper …]`, which wins). See Task 3.
3. Convert CLI flag is `--input` (currently `required=True`), not a positional path. See Task 2.
4. `force_magnitude_ok` includes the torque-magnitude ratio. TCP displacement uses the first **hold** frame as \(x_{\mathrm{hold0}}\).
5. Holdout mode always requires 8 usable disk dirs, including when both explicit index flags pin the split.

## File map

| Path | Responsibility |
| --- | --- |
| `apple_pick_sim/system_id/holdout_gates.py` | **New.** `choose_direction_split`, magnitude-ratio + Pearson-trend gate helpers, signed per-hold series |
| `apple_pick_sim/tests/test_holdout_gates.py` | **New.** Pure unit tests for the above |
| `apple_pick_sim/system_id/real_to_batched_sysid.py` | Folder convert (`export_real_tree_folder_to_batched_dataset`), per-direction episode rows, canonical geometry |
| `robot_replay/convert_real_to_batched_sysid_metadata.py` | `--input-dir` CLI path |
| `apple_pick_gym/batched_envs/real_batched_replay_build.py` | Per-slot episode metadata in `make_real_replay_build_env_fn` |
| `apple_pick_sim/system_id/batched_digital_twin_init.py` | Per-env logged post-grasp SE(3) |
| `apple_pick_sim/coupled_fruiting/settle_then_weld.py` | Per-world open-loop `joint_q` (no broadcast) |
| `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` | Per-direction metadata on requests/slots; padding + truncate-before-features |
| `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` | `load_episode_metadata_for_directions` helper |
| `apple_pick_gym/batched_envs/batched_sysid_cmaes.py` | `direction_indices` through prepare/evaluate; one-hot `n_directions` stays collection width |
| `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` | Split flags, train-only fit, holdout eval, `holdout_report.json`, val overlays |
| `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py` | CLI/`_run` contract tests |
| `docs/handbook-real-replay.md`, `docs/handbook-youngs-cma.md`, `docs/handbook-sysid-scoring.md`, `docs/ROADMAP.md`, `README.md` | Docs |

**Do not modify:** `fit_youngs_modulus_structures` ask/tell logic; `CMA_SEARCH_PARAMS` values; Sinkhorn feature builders in `mmd_features.py` (read-only reuse of `scored_ft_wrist`, `iter_kept_hold_segments`).

---

### Task 1: Split sampler and verification-gate helpers (pure)

Pure NumPy/stdlib module first so every later task can assert against it without GPU.

**Files:**
- Create: `apple_pick_sim/system_id/holdout_gates.py`
- Create: `apple_pick_sim/tests/test_holdout_gates.py`

**Interfaces:**

```python
DIRECTION_SPLIT_SEED = 17
MAGNITUDE_RATIO_MIN = 1.0 / 3.0
MAGNITUDE_RATIO_MAX = 3.0
TREND_PEARSON_MIN = 0.5
FORCE_FLOOR_N = 0.2
TORQUE_FLOOR_NM = 0.05
FLOOR_SLACK_FACTOR = 3.0
FORCE_SLACK_N = 0.4  # also the torque additive slack, in N·m

def choose_direction_split(
    directions: Iterable[int], *, seed: int, n_train: int = 5
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (train, val) sorted disjoint direction indices."""

def magnitude_ratio_ok(
    *, real_mean: float, fitted_mean: float, floor: float, slack: float
) -> tuple[bool, float]:
    """Return (passed, ratio). Uses the additive floor rule when real_mean < floor."""

def trend_pearson_ok(
    real: Sequence[float], fitted: Sequence[float], *, magnitude_passed: bool
) -> tuple[bool, float | None]:
    """Pearson r >= TREND_PEARSON_MIN; zero-variance passes iff magnitude passed."""

def signed_parallel_series(
    values: np.ndarray, pull_direction: Sequence[float]
) -> np.ndarray:
    """Project (T, 3) rows onto the unit pull axis -> (T,) signed scalars."""

def per_hold_means(
    series: np.ndarray, *, phase: np.ndarray, dir_idx: np.ndarray, direction: int
) -> np.ndarray:
    """Mean of `series` over each contiguous hold segment of one direction."""

def tcp_displacement_along_pull(
    tcp_pos: np.ndarray,
    *,
    phase: np.ndarray,
    dir_idx: np.ndarray,
    direction: int,
    pull_direction: Sequence[float],
) -> np.ndarray:
    """Hold-frame signed TCP displacement: s = (x - x_hold0) · p_hat.

    ``x_hold0`` is TCP at the **first hold frame** of this direction, not
    episode frame 0 (that frame is still on the pull-in).
    """
```

- [ ] **Step 1: Write the failing tests**

`apple_pick_sim/tests/test_holdout_gates.py` — cover the spec's "Tests (TDD)" bullets:

```python
import numpy as np
import pytest

from apple_pick_sim.system_id.holdout_gates import (
    DIRECTION_SPLIT_SEED,
    FORCE_FLOOR_N,
    choose_direction_split,
    magnitude_ratio_ok,
    per_hold_means,
    signed_parallel_series,
    tcp_displacement_along_pull,
    trend_pearson_ok,
)


def test_choose_direction_split_seed_17_is_pinned():
    train, val = choose_direction_split(range(8), seed=DIRECTION_SPLIT_SEED)
    assert train == (2, 4, 5, 6, 7)
    assert val == (0, 1, 3)
    assert not set(train) & set(val)


def test_choose_direction_split_is_seed_sensitive_and_covers_population():
    train, val = choose_direction_split(range(8), seed=7)
    assert sorted(train + val) == list(range(8))
    assert train != (2, 4, 5, 6, 7)


def test_choose_direction_split_rejects_bad_population():
    with pytest.raises(ValueError, match="n_train"):
        choose_direction_split(range(4), seed=17, n_train=5)
    with pytest.raises(ValueError, match="duplicate"):
        choose_direction_split([0, 0, 1], seed=17, n_train=1)


def test_magnitude_ratio_passes_within_factor_three_and_fails_outside():
    ok, ratio = magnitude_ratio_ok(real_mean=2.0, fitted_mean=5.0, floor=FORCE_FLOOR_N, slack=0.4)
    assert ok and ratio == pytest.approx(2.5)
    ok, ratio = magnitude_ratio_ok(real_mean=2.0, fitted_mean=7.0, floor=FORCE_FLOOR_N, slack=0.4)
    assert not ok and ratio == pytest.approx(3.5)
    ok, _ = magnitude_ratio_ok(real_mean=2.0, fitted_mean=0.5, floor=FORCE_FLOOR_N, slack=0.4)
    assert not ok


def test_magnitude_ratio_uses_additive_rule_below_floor():
    # real below floor: pass iff fitted < 3*real + slack, ratio is still reported
    ok, _ = magnitude_ratio_ok(real_mean=0.1, fitted_mean=0.6, floor=FORCE_FLOOR_N, slack=0.4)
    assert ok
    ok, _ = magnitude_ratio_ok(real_mean=0.1, fitted_mean=0.9, floor=FORCE_FLOOR_N, slack=0.4)
    assert not ok


def test_trend_requires_pearson_half():
    real = [1.0, 2.0, 3.0, 4.0]
    ok, r = trend_pearson_ok(real, [1.1, 2.2, 2.9, 4.4], magnitude_passed=True)
    assert ok and r > 0.9
    ok, r = trend_pearson_ok(real, [4.0, 3.0, 2.0, 1.0], magnitude_passed=True)
    assert not ok and r < 0.0


def test_trend_zero_variance_defers_to_magnitude():
    flat = [1.0, 1.0, 1.0]
    ok, r = trend_pearson_ok(flat, flat, magnitude_passed=True)
    assert ok and r is None
    ok, r = trend_pearson_ok(flat, flat, magnitude_passed=False)
    assert not ok and r is None


def test_trend_requires_three_points():
    ok, _ = trend_pearson_ok([1.0, 2.0], [1.0, 2.0], magnitude_passed=True)
    assert not ok


def test_signed_parallel_series_is_signed_not_norm():
    vals = np.array([[0.0, 0.0, -3.0], [0.0, 0.0, 4.0]])
    out = signed_parallel_series(vals, (0.0, 0.0, 2.0))  # non-unit axis is normalized
    assert out.tolist() == [-3.0, 4.0]


def test_per_hold_means_averages_each_contiguous_hold():
    phase = np.array([0, 1, 1, 0, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(6, dtype=np.int32)
    series = np.array([9.0, 1.0, 3.0, 9.0, 10.0, 20.0])
    out = per_hold_means(series, phase=phase, dir_idx=dir_idx, direction=0)
    assert out.tolist() == [2.0, 15.0]


def test_tcp_displacement_references_first_hold_frame_not_episode_start():
    # Episode frame 0 is a pull-in at the origin; first hold is at +1 m along z.
    tcp = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.2],
        ]
    )
    phase = np.array([0, 1, 1], dtype=np.int8)
    dir_idx = np.zeros(3, dtype=np.int32)
    s = tcp_displacement_along_pull(
        tcp, phase=phase, dir_idx=dir_idx, direction=0, pull_direction=(0.0, 0.0, 1.0)
    )
    assert s.tolist() == [0.0, 0.2]
```

- [ ] **Step 2: Run and confirm collection failure**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests/test_holdout_gates.py -q
```

Expected: FAIL — `ModuleNotFoundError: apple_pick_sim.system_id.holdout_gates`.

- [ ] **Step 3: Implement `holdout_gates.py`**

Requirements the tests pin down:

- `choose_direction_split` uses **stdlib** `random.Random(seed).sample(sorted(dirs), n_train)`; raises `ValueError` on duplicates or `n_train >= len(dirs)`; returns sorted tuples.
- `magnitude_ratio_ok` returns `ratio = fitted/real` (`inf` when `real == 0` and `fitted > 0`, `1.0` when both are 0). When `real_mean < floor`, pass iff `fitted_mean < FLOOR_SLACK_FACTOR * real_mean + slack`; otherwise pass iff `MAGNITUDE_RATIO_MIN <= ratio <= MAGNITUDE_RATIO_MAX`.
- `trend_pearson_ok` needs `len >= 3`; returns `(magnitude_passed, None)` when either series has `std == 0` (or `r` is not finite); else `(r >= TREND_PEARSON_MIN, r)` using `np.corrcoef`.
- `signed_parallel_series` normalizes the axis and raises `ValueError` on a zero-norm axis.
- `per_hold_means` delegates segmentation to `mmd_features.iter_kept_hold_segments(phase=…, dir_idx=…, direction=…, min_frames=1)` (do not re-derive hold boundaries) and skips empty segments.
- `tcp_displacement_along_pull` uses the first hold-frame TCP as \(x_{\mathrm{hold0}}\) and returns one signed scalar per **hold** frame (same length as the concatenated hold indices). Raise if that direction has no hold frames.

- [ ] **Step 4: Run to green**, then `ReadLints` on both files.

---

### Task 2: Folder convert — one tree, eight directions

**Files:**
- Modify: `apple_pick_sim/system_id/real_to_batched_sysid.py`
- Modify: `robot_replay/convert_real_to_batched_sysid_metadata.py`
- Test: `apple_pick_sim/tests/test_real_to_batched_sysid.py` (extend; find the module's existing 1×1 tests and their synthetic-parquet fixture helper and reuse it)

**Interfaces:**

```python
def export_real_tree_folder_to_batched_dataset(
    input_dir: str | Path,
    *,
    fixture_path: str | Path,
    output_dir: str | Path,
    weld_direction_sign: float = 1.0,
    overwrite: bool = False,
    allow_zero_action: bool = False,
    command_argv: list[str] | None = None,
    control_hz: float | None = None,
    ft_lpf_hz: float = DEFAULT_FT_LPF_CUTOFF_HZ,
    ft_lpf_order: int = DEFAULT_FT_LPF_ORDER,
    base_pos_tolerance_m: float = 5e-3,
) -> Path
```

Implementation shape: factor today's `export_real_episode_to_batched_dataset` body into a private `_build_real_episode(path, *, fixture, direction_idx, …) -> _ConvertedEpisode` (dataclass carrying `traj`, `episode_meta`, `ft_filter`, `junction_names`, `n_frames`, `pull_direction`, `fruiting_base_pos`). Both the 1×1 function and the folder function then only differ in the loop and manifest assembly. `export_real_episode_to_batched_dataset` must keep its current signature and output (regression test below).

- [ ] **Step 1: Write the failing tests**

Add to `apple_pick_sim/tests/test_real_to_batched_sysid.py`:

1. `test_folder_convert_writes_one_structure_per_direction` — write synthetic `s09-d00.parquet` and `s09-d01.parquet` into a tmp dir; assert `collection["num_structures"] == 1`, `collection["num_directions"] == 2`, files `episodes/s00_d00.parquet` and `episodes/s00_d01.parquet` exist, and each episode row's `direction_idx` / metadata `direction_idx` equals the filename `NN`.
2. `test_folder_convert_maps_sparse_direction_numbers` — files `s09-d03` + `s09-d05` ⇒ episodes `s00_d03` / `s00_d05` and `num_directions == 6` (dense upper bound `max(NN)+1`) with only two episode rows; both rows non-excluded. (Locks `env_idx = direction_idx`.) Holdout CMA still requires 8 usable dirs; this test is convert-only. Any later `range(num_directions)` walker (including `_require_ft_wrist_lpf_per_structure`) must iterate **episode rows**, not `range(6)`, or this bag will look like a missing-LPF failure.
3. `test_folder_convert_rejects_direction_index_mismatch` — parquet `dump.direction_index = 4` in `s09-d01.parquet` ⇒ `ValueError` mentioning `direction_index`.
4. `test_folder_convert_rejects_duplicate_and_empty_inputs` — two files claiming `d00` ⇒ `ValueError`; empty dir ⇒ `ValueError`.
5. `test_folder_convert_ignores_uncompiled_siblings` — `s09-d00_robot.parquet`, `s09-d00_tracking.parquet`, `frame.png` present ⇒ still one episode.
6. `test_folder_convert_rejects_base_pose_spread` — second file's tree base offset by 2 cm ⇒ `ValueError` mentioning tolerance; and within tolerance (2 mm) ⇒ both episodes share the **same** `fruiting_base_pos` (the mean) **and** the same `fruiting_system_params` / `params_fingerprint` (copy the first episode's rebuilt params onto every direction after the rod-geometry assert).
7. `test_folder_convert_writes_n_holds_sim_config_and_topology_seed` — `collection["n_holds"]` equals `max(hold_number)+1` from the converted bags (4 when `hold_index` is 0–3), `collection["sim_config"]["joint_damping_ratio"]` present, `collection["sim_config"]["controller"]["mode"] == "vic_pose"`, `collection["topology_seed"]` present, `collection["control_hz"] == 30.0`, `collection["max_steps"] == max(n_frames)`.
8. `test_single_file_convert_still_writes_s00_d00` — regression on the existing 1×1 entry point.
9. `test_convert_cli_input_dir_does_not_require_input` — `build_parser().parse_args(["--input-dir", "/tmp/s09", "--dataset-out", "/tmp/out"])` succeeds. `--input` alone still works. Both `--input` and `--input-dir` ⇒ argparse error.

- [ ] **Step 2: Run to confirm failures** (expect `AttributeError` / missing function first).

- [ ] **Step 3: Implement**

- Discovery: `sorted(Path(input_dir).glob("*.parquet"))` filtered by `re.fullmatch(r"(?P<tree>s\d+)-d(?P<dir>\d+)\.parquet", p.name)`. Reject when the tree prefix differs between files.
- Per-file convert via `_build_real_episode(..., direction_idx=NN)`; save to `episode_filename(0, NN)`.
- Canonical geometry (2026-08-14 contract): collect each episode's `fruiting_base_pos`, assert `max |p - mean| <= base_pos_tolerance_m` per axis, then rewrite every episode's metadata with the mean before `traj.save`. Assert identical rod geometry (`junction_names` and the rebuilt `fruiting_system_params` rods: lengths/radii/segment counts). Copy one canonical `fruiting_system_params` + `params_fingerprint` onto every direction (`true_params_for_structure` reads direction 0).
- Manifest: one `structures` row (`structure_idx=0`), one `episodes` row per direction with `env_idx=direction_idx`, `collection` as in the 1×1 path plus `num_directions = max(NN)+1`, `n_holds = max(hold_number)+1`, `topology_seed` (fixture / collection seed, default 0), `sim_config` from `sim_config_to_manifest_dict` on a `BatchedHeterogeneousCoupledSimConfig` built **in `apple_pick_sim`** (`gym_defaults` + `parse_sim_build` fixture knobs, `controller.mode="vic_pose"`). Do **not** import `real_replay_sim_config` / `apple_pick_gym`. `max_steps = max(n_frames)`. Replace `source_real_parquet` with a `source_real_parquets` list (keep the singular key on the 1×1 path).
- `ft_filter` must be identical across directions; assert and write once at collection level.
- CLI (`robot_replay/convert_real_to_batched_sysid_metadata.py`): `--input` is currently `required=True`. Make `--input` and `--input-dir` a required mutually exclusive group (`parser.add_mutually_exclusive_group(required=True)`), add `--base-pos-tolerance-m` (default `5e-3`), and dispatch to the folder function. Keep every other existing flag working.

- [ ] **Step 4: Run to green**; run the whole convert test module to catch regressions.

---

### Task 3: Per-direction weld pose, gripper, and arm joints in one batch

Today every env in a batch gets direction 0's weld metadata and one broadcast `bootstrap_joint_q`. With eight directions in one batch that is systematically wrong — this is the correctness core of the slice.

**Files:**
- Modify: `apple_pick_sim/coupled_fruiting/settle_then_weld.py` (per-world open-loop joints)
- Modify: `apple_pick_sim/system_id/batched_digital_twin_init.py` (per-env logged SE(3))
- Modify: `apple_pick_gym/batched_envs/real_batched_replay_build.py` (accept per-env metadata)
- Modify: `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` (carry metadata to slots, pass to builder)
- Modify: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py` (metadata loader)
- Tests: `apple_pick_sim/tests/test_batched_digital_twin_init.py`, `apple_pick_sim/tests/test_open_loop_joint_bootstrap.py`, `apple_pick_gym/tests/test_batched_sysid_multi_replay.py` (extend existing modules)

**Interfaces:**

```python
# settle_then_weld.py
def apply_open_loop_fr3_joint_q_per_world(
    scene: Any, per_world_joint_q: Sequence[Sequence[float]]
) -> None: ...

# batched_digital_twin_init.py — new keyword; existing calls unchanged
def apply_logged_post_grasp_se3_to_cable(
    cable, meta, *, layout=None, per_env_meta: Sequence[Mapping[str, Any]] | None = None
) -> None: ...

# batched_sysid_mmd_grid.py
def load_episode_metadata_for_directions(
    dataset, *, structure_idx: int, direction_indices: Sequence[int]
) -> dict[int, dict]: ...

# real_batched_replay_build.py — factory signature unchanged.
# The *inner* build_env_fn accepts an optional kwarg:
#   per_env_episode_meta: Sequence[Mapping] | None = None
```

`ReplayStructureRequest` gains `meta_by_direction: Mapping[int, dict] | None = None`; `ReplaySlot` gains `episode_meta: dict | None = None` (from `request.meta_by_direction[direction_idx]`).

**Gripper channel (this is the one that actually runs):** `build_replay_candidate_blocks` currently does `gripper=request.gripper` for every slot, and `prepare` sets `request.gripper` from the first selected direction. The driver always passes `per_env_grippers=[slot.gripper for slot in slots]`, and `make_real_replay_build_env_fn` **prefers that list** over deriving grippers from meta. So per-dir weld offsets only take effect if slot construction uses each direction's meta:

```python
if request.meta_by_direction is not None:
    slot_gripper = gripper_proxy_for_real_batched_replay(
        dict(request.meta_by_direction[direction_idx])
    )
else:
    slot_gripper = request.gripper
```

`wants_per_env_meta` is only for cable SE(3) and `per_world_bootstrap_joint_q`. Do not "re-derive grippers from metas" inside the builder when `per_env_grippers` is already set.

Slot order already defines env order. The driver passes the new kwarg only when the builder opts in:

```python
build_kwargs = {}
if getattr(build_env_fn, "wants_per_env_meta", False) and all(
    slot.episode_meta is not None for slot in slots
):
    build_kwargs["per_env_episode_meta"] = [dict(slot.episode_meta) for slot in slots]
env = build_env_fn(..., **build_kwargs)
```

`make_real_replay_build_env_fn` sets `build_env_fn.wants_per_env_meta = True` before returning it.

- [ ] **Step 1: Write the failing tests**

1. `test_apply_open_loop_joint_q_per_world_writes_distinct_rows` — fake scene with a 2-world `robot_model` (mirror the stub style already in `test_open_loop_joint_bootstrap.py`); assert world 0 and world 1 keep different joint coords and that `eval_fk` ran once.
2. `test_apply_open_loop_joint_q_per_world_rejects_length_mismatch` — `len(per_world_joint_q) != layout.num_envs` ⇒ `ValueError`.
3. `test_apply_logged_post_grasp_se3_per_env_uses_each_meta` — 2-env layout, two metas with different `initial_apple_pos`; assert the two apple body rows in `body_q` differ and each equals its own meta.
4. `test_apply_logged_post_grasp_se3_per_env_rejects_wrong_count` ⇒ `ValueError`.
5. `test_real_build_env_fn_advertises_per_env_meta` — `getattr(fn, "wants_per_env_meta") is True`.
6. `test_two_direction_batch_gets_distinct_weld_poses` (`apple_pick_gym/tests/test_batched_sysid_multi_replay.py`) — build slots for directions 0 and 1 with distinct `meta_by_direction`; assert `ReplaySlot.episode_meta` differs per slot, `slot.gripper.weld_reference_pos` differs per slot (not both equal to direction 0), and the recorded `per_env_episode_meta` handed to the builder is `[meta_d0, meta_d1]`. This is the spec's "replay fails if two slots share direction-0 weld metadata" regression.
7. `test_slots_without_meta_do_not_pass_per_env_meta` — sim-sim path (no `meta_by_direction`) ⇒ builder called without the kwarg; every `slot.gripper` equals `request.gripper`.

- [ ] **Step 2: Run and confirm the expected failures.**

- [ ] **Step 3: Implement**

- `apply_open_loop_fr3_joint_q_per_world`: reuse the existing template write for world 0 to get `coord_per`, then assign each world's slice `batched_jq[w*coord_per:(w+1)*coord_per] = q_w` (pad/truncate to `coord_per` exactly as `_write` does), zero `joint_qd`, one `newton.eval_fk`, then the same `init_robot_mujoco_step_buffers` / actuator-target / force-cache cleanup as `apply_open_loop_fr3_joint_q`. **Do not** call `broadcast_joint_q_from_world0` on this path.
- Route it: `SimRobotConfig` gains `per_world_bootstrap_joint_q: tuple[tuple[float, ...], ...] | None = None`; `_bootstrap_tcp_at_fixed_origin` / `seed_fix_to_apple_from_settled` prefer it over the scalar. Keep the scalar path untouched when it is `None`.
- `apply_logged_post_grasp_se3_to_cable(..., per_env_meta=…)`: when provided, zip `layout.apple_body_indices` / `layout.proxy_body_indices` with the metas and use each meta's `initial_apple_pos/quat` (falling back to `weld_reference_*` as today) plus that meta's proxy offset. Raise when the counts disagree.
- `make_real_replay_build_env_fn`: factory signature stays as today (closes over one `episode_meta` / scalar `bootstrap_joint_q` for the 1×1 fallback). The **inner** `build_env_fn` gains `per_env_episode_meta=None`. When that kwarg is passed: keep using the already-supplied `per_env_grippers` (do not overwrite them), set `per_world_bootstrap_joint_q` from each meta's `initial_robot_joint_q`, and call the cable helper with `per_env_meta`. When the kwarg is absent, today's scalar cable + broadcast joints stay unchanged. Per-direction controller gains are already inside the 19D `vic_pose` action — no extra gains channel.
- `load_episode_metadata_for_directions` is a thin loop over `dataset.load_episode_metadata`. Do **not** stuff metadata into the `recorded` arrays dict (it is consumed as arrays).
- Thread `meta_by_direction` from `prepare_youngs_modulus_structure` (Task 5) — for now accept it as optional and leave callers unchanged so this task stays green on its own.

- [ ] **Step 4: Run the three test modules to green** plus `apple_pick_sim/tests/test_open_loop_joint_bootstrap.py` and any settle/weld tests that touch `bootstrap_joint_q`.

---

### Task 4: Unequal episode lengths — pad the drive, truncate before features

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_multi_replay.py` (drive tensor + post-replay truncation)
- Test: `apple_pick_gym/tests/test_batched_sysid_multi_replay.py`

**Interfaces:** unchanged public API; `build_recorded_actions_tensor` (or the inline `np.stack` at the `build_env_fn` call site) must accept ragged lengths.

- [ ] **Step 1: Write the failing tests**

1. `test_drive_tensor_pads_short_directions_with_last_action` — direction 0 has 5 frames, direction 1 has 3; assert the tensor is `(2, 5, A)` and rows 3–4 of direction 1 equal its last logged action (not zeros).
2. `test_replay_arrays_truncate_to_recorded_length` — after a stubbed replay producing 5 frames per env, the arrays handed to feature extraction for direction 1 have 3 frames.
3. `test_padded_frames_absent_from_features` — the padded tail's sentinel value never appears in the per-direction arrays used for scoring.

Use the existing stub/collector harness in that test module; do not build a GPU env.

- [ ] **Step 2: Run and confirm failure** (today's `np.stack` raises on ragged input, or padding is silently zeros).

- [ ] **Step 3: Implement**

- Compute `T_max = max(n_frames)`; allocate `(num_slots, T_max, A)` and fill each slot with its actions then `np.repeat(last_action, pad)`.
- Record each slot's true `n_frames`; after replay, slice every collected array to that length **before** any feature/Sinkhorn call.
- If `_validate_request` currently requires equal frame counts across directions, relax it to "equal action width and junction names" and keep a clear error for width/topology mismatches.

- [ ] **Step 4: Run the module to green.**

---

### Task 5: Direction subsets through prepare/evaluate

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`
- Test: `apple_pick_gym/tests/test_batched_sysid_cmaes.py` (extend; reuse its dataset mock)

**Interfaces:**

```python
def prepare_youngs_modulus_structure(
    *, dataset, structure_idx, candidates, num_directions,
    scoring, include_excluded=False,
    direction_indices: Sequence[int] | None = None,
) -> PreparedYoungsModulusStructure: ...
```

Same optional keyword on `evaluate_youngs_modulus_structures` / `evaluate_youngs_modulus_candidates` (pass through). Default `None` ⇒ today's "all usable dirs".

- [ ] **Step 1: Write the failing tests**

1. `test_prepare_uses_only_requested_directions` — 8-dir mock; `direction_indices=(0, 1)`; assert `load_episode_obs_arrays` was called exactly for dirs 0 and 1 and never for 5.
2. `test_prepare_onehot_width_stays_collection_num_directions` — select `(2, 4, 5, 6, 7)` on an 8-dir mock whose `dir_idx` columns use those disk IDs. The scoring config / `gt_context` one-hot width must be **8** (or `max(disk_id)+1`), not 5. Building transition features must succeed for dir 5 (legal under width 8). `gt_context.expected_directions` must equal `{2,4,5,6,7}` — val dirs 0, 1, 3 must not appear.
3. `test_prepare_attaches_meta_by_direction_for_selection` — the replay request carries metadata for exactly the selected dirs (feeds Task 3).
4. `test_prepare_defaults_to_all_usable_directions` — regression: no `direction_indices` ⇒ unchanged calls.
5. `test_prepare_rejects_direction_not_on_disk` ⇒ `ValueError` naming the index.
6. `test_collector_local_index_zips_to_disk_ids` — `direction_episodes_from_collectors(..., num_directions=len(selected))` returns local slots `0..4`; zipping onto `prepared.direction_indices` must recover disk IDs `(2,4,5,6,7)` (this mapping already exists — lock it with an assert).

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement**

- Forward `direction_indices=` into the existing `resolve_direction_indices` call and into `load_recorded_episodes_for_structure`.
- **Do not** `replace(scoring, n_directions=len(selected))`. Leave `scoring.n_directions` at the collection width the CLI already sets (`int(num_directions)`). `prepare` already computes `scoring_n_directions` from that field; `expected_directions` comes from loaded bags, so val dirs never enter the pool.
- Attach `meta_by_direction=load_episode_metadata_for_directions(...)` to the `ReplayStructureRequest`.
- Validate every requested index appears in `list_usable_direction_indices` (or the episode rows) before loading.
- Collector indexing is already local: `direction_episodes_from_collectors` uses `c * d + local` with `d = len(prepared.direction_indices)`, then the existing `zip(..., prepared.direction_indices)` maps back to disk IDs. Add the assert in test 6; do not invent a second mapping.

- [ ] **Step 4: Run to green**, then the full `apple_pick_gym/tests/test_batched_sysid_cmaes.py` module.

---

### Task 6: CMA CLI holdout mode — split flags and train-only fit

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py`
- Test: `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py`

**Interfaces:**

- `--direction-split-seed` — `nargs="?"`, `type=int`, `const=DIRECTION_SPLIT_SEED`, `default=None`. Present without a value ⇒ 17. Absent ⇒ no holdout.
- `--direction-indices`, `--val-direction-indices` — `type=parse_comma_separated_ints`, `default=None`. Both together pin a split; exactly one ⇒ `SystemExit`.
- `_run` result gains `train_direction_indices` / `val_direction_indices` (both `None` outside holdout mode).

- [ ] **Step 1: Write the failing tests**

1. `test_parser_direction_split_seed_defaults_to_seventeen_when_bare` and `..._absent_is_none`.
2. `test_run_without_split_flags_uses_all_directions` — existing 1×1 real `_run` stub; assert no `direction_indices` restriction and no `holdout_report.json`.
3. `test_run_holdout_seed_selects_pinned_split` — 8-dir stub dataset; assert every prepare/evaluate call during fit received `direction_indices=(2, 4, 5, 6, 7)`.
4. `test_run_holdout_never_loads_val_directions_during_fit` — record all `direction_indices` seen by the evaluate stub across generations **and** the final-mean wave; assert `{0, 1, 3}` is never among them (the spec's leak gate).
5. `test_run_rejects_partial_explicit_split` — only `--direction-indices` ⇒ `SystemExit`.
6. `test_run_rejects_overlapping_or_empty_explicit_split` ⇒ `SystemExit`.
7. `test_run_rejects_holdout_on_non_eight_direction_dataset` — 4 dirs + `--direction-split-seed` ⇒ `SystemExit` naming `8`. Same `SystemExit` when both explicit index flags pin a 2/2 split on that 4-dir bag (holdout mode always requires 8 usable disk dirs).
8. `test_run_holdout_keeps_one_structure_guard` — two structures + holdout ⇒ existing `SystemExit`.
9. `test_require_ft_wrist_lpf_iterates_episode_rows` — a 2-episode bag with `num_directions=6` (sparse d03/d05) must not `SystemExit` for missing dirs 0–2.

- [ ] **Step 2: Run and confirm failures.**

- [ ] **Step 3: Implement**

- Resolve the split right after `structure_indices` / mode resolution. **Both** seed and explicit-pair holdout modes assert `len(usable disk dirs) == 8` first. Explicit pair then also validates disjoint, non-empty, subset of disk. Seed path calls `choose_direction_split(disk_dirs, seed=seed)`.
- Pass `direction_indices=train` into every fit-time prepare/evaluate call, including the final-mean wave. Leave `YoungsModulusScoringConfig.n_directions` at `collection.num_directions` (8).
- Change `_require_ft_wrist_lpf_per_structure` to iterate usable episode rows (or the selected train+val dirs), not `range(num_directions)`.
- Leave every code path unchanged when the split is `None`.

- [ ] **Step 4: Run the CLI test module to green** (it also guards sim-sim `_run`).

---

### Task 7: Holdout evaluation, gates, and `holdout_report.json`

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py`
- Create: `apple_pick_gym/batched_envs/holdout_evaluation.py` (report builder, so the CLI stays thin)
- Test: `apple_pick_gym/tests/test_holdout_evaluation.py` (new), plus CLI wiring tests in `test_example_youngs_modulus_cmaes_cli.py`

**Interfaces:**

```python
def cartesian_ft_mae(
    *, real: Mapping[str, Any], fitted: Mapping[str, Any], direction: int
) -> tuple[float, float]:
    """Hold-frame mean |ΔF| (N) and |Δτ| (N·m); world ft_wrist via scored_ft_wrist."""

def direction_verification(
    *, real: Mapping[str, Any], fitted: Mapping[str, Any],
    direction: int, pull_direction: Sequence[float],
) -> dict[str, Any]:
    """force_magnitude_ok / force_trend_ok / tcp_pose_magnitude_ok / tcp_pose_trend_ok
    plus apple-pose diagnostics. ``force_magnitude_ok`` is True only if BOTH
    mean |F_∥| and mean |τ| pass ``magnitude_ratio_ok`` (torque is not diagnostic).
    TCP series must use ``tcp_displacement_along_pull`` (first hold frame)."""

def build_holdout_report(
    *, structure_idx: int, direction_split_seed: int | None,
    train_direction_indices, val_direction_indices,
    baseline_log10, fitted_log10,
    train_fitted, val_baseline, val_fitted,   # each: sinkhorn + MAE + per-dir arrays
    train_eligible_means: Sequence[float],
    val_overlay_paths: Mapping[int, str],
) -> dict[str, Any]: ...

def write_holdout_report(output_dir: Path, report: Mapping[str, Any]) -> Path: ...
```

- [ ] **Step 1: Write the failing tests**

`test_holdout_evaluation.py` (pure, synthetic arrays — no GPU):

1. `test_cartesian_ft_mae_uses_hold_frames_only` — a pull frame with a huge error is excluded; MAE matches the hand-computed hold-frame value; `ft_wrist_lpf` is preferred when present.
2. `test_direction_verification_passes_matching_signed_series` — fitted ≈ real ⇒ all four flags `True`; report includes `force_ratio`, `torque_ratio`, `force_pearson_r`, `tcp_ratio`, `tcp_pearson_r`.
3. `test_direction_verification_fails_flipped_force_sign` — fitted `F_∥` is real negated ⇒ `force_trend_ok is False` (magnitude may still pass). This is the gate that catches "right magnitude, wrong physics".
4. `test_direction_verification_fails_ten_times_stiff_pose` — fitted TCP displacement is 0.1× real ⇒ `tcp_pose_magnitude_ok is False`. Use an episode whose frame 0 TCP is far from the first hold frame so a wrong \(x_{\mathrm{hold0}}\) would fail this test.
5. `test_direction_verification_fails_torque_magnitude` — `|τ|` fitted is 10× real while `F_∥` matches ⇒ `force_magnitude_ok is False`.
6. `test_direction_verification_apple_pose_is_diagnostic_only` — bad apple series with good TCP ⇒ no required `*_ok` flag flips to `False`.
7. `test_build_holdout_report_has_required_keys` — assert exactly the spec's Slice 4 key list, `train_sinkhorn_decreased` from `train_eligible_means[-1] < [0]`, `val_sinkhorn_improved` from the two Sinkhorn floats, sorted index lists, `direction_split_seed` omitted when `None`, and JSON round-trips through `to_strict_jsonable`.
8. `test_build_holdout_report_requires_finite_metrics` — NaN Sinkhorn ⇒ `ValueError`.

CLI wiring tests in `test_example_youngs_modulus_cmaes_cli.py`:

8. `test_run_holdout_evaluates_baseline_and_fitted_on_val_only` — assert two extra evaluate calls, both with `direction_indices=(0, 1, 3)`, one at `CMA_SEARCH_PARAMS["initial_mean_log10"]` and one at the state's `final_mean_log10`.
9. `test_run_holdout_does_not_tell_optimizer_on_val` — stub optimizer counts `tell` calls; unchanged across holdout eval.
10. `test_run_writes_holdout_report_with_val_overlays` — `holdout_report.json` exists, has one overlay path per val dir, and the paths are **not** the train overlay.
11. `test_run_skips_holdout_report_when_fit_failed` — failed state ⇒ no `holdout_report.json`, non-zero exit preserved.
12. `test_run_holdout_report_absent_without_split_flags`.

- [ ] **Step 2: Run and confirm failures.**

- [ ] **Step 3: Implement**

- After a successful fit (state `fitted`), call the existing evaluator twice on val dirs with a one-candidate structure list: baseline `candidates_from_log10_vector(CMA_SEARCH_PARAMS["initial_mean_log10"])` and fitted `state.final_mean_log10`. Never touch the optimizer.
- `train_fitted` Sinkhorn + F/T MAE come from `state.final_evaluation` (already a train-dir replay). Use `evaluation.replay_episodes[0]` plus the recorded train bags — do **not** add a third evaluate for train MAE.
- Pull `eligible_mean` per generation from the same records the CMA report uses (`structures.0.generations[*].score_summary.eligible_mean`), so gate 1 and `cmaes_report.json` cannot disagree.
- Reuse Task 1 helpers for every gate. `force_magnitude_ok` requires both `|F_∥|` and `|τ|` ratios. TCP uses `tcp_displacement_along_pull`. `pull_direction` comes from the val episode's metadata (fall back to the logged `excitation_direction` of the first hold frame). If a bag has no `dir_idx` column, fill it with the episode's `direction` (same as `load_recorded_episodes_for_structure`) before calling the helpers.
- Render one overlay per val dir from the **val fitted** evaluation (not `_write_final_mean_overlay`, which is the train overlay) into `structure_000/holdout/direction_0NN.html`. Record those paths.
- Write via a temp file + atomic replace, matching `cmaes_report.json`'s pattern. `_clear_cma_owned_artifacts` must delete `holdout_report.json` **and** `structure_000/holdout/`.
- Exit non-zero when any gate fails, with a one-line summary naming the failed gate and direction. Still write the report.

- [ ] **Step 4: Run both test modules to green**, then `ReadLints`.

---

### Task 8: Documentation

**Files:** `docs/handbook-real-replay.md` (H4), `docs/handbook-youngs-cma.md` (H5), `docs/handbook-sysid-scoring.md` (H3), `docs/ROADMAP.md`, `README.md`

- [ ] **Step 1: H4** — folder convert contract: filename→`direction_idx`, canonical geometry + tolerance, per-direction weld/gripper/arm joints, drive padding and truncate-before-features.
- [ ] **Step 2: H3** — the magnitude-ratio and Pearson-trend reductions, signed \(F_\parallel\) and TCP-displacement (\(x_{\mathrm{hold0}}\) = first hold frame) definitions, torque folded into force magnitude, hold-phase-only scope, no sim tare, Cartesian MAE diagnostic, and that one-hot `n_directions` is collection width (not `len(selected)`).
- [ ] **Step 3: H5** — opt-in holdout mode, flag semantics and default seed 17, train-only fit, frozen `final_mean`, baseline column, `holdout_report.json` schema, and the exit-code contract on gate failure.
- [ ] **Step 4: ROADMAP** — M4.0 checklist, the acceptance commands from the spec, and (after Task 9) the knobs actually run plus results.
- [ ] **Step 5: README** — the two-command convert → holdout CMA recipe.

Cross-check: no doc may claim a passing science gate before Task 9 reports one.

---

### Task 9: Acceptance run (GPU) and results

Runtime verification is part of the slice, not optional. Run from the repository root with `robot_replay/new_data/s09/` present.

- [ ] **Step 1: Convert**

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input-dir robot_replay/new_data/s09 \
  --dataset-out tmp/real_batched_s09 \
  --overwrite
```

Check: 1 structure × 8 directions, `control_hz == 30`, `n_holds == 4`, eight episode files.

- [ ] **Step 2: Holdout CMA**

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09 \
  --output tmp/real_kp_e_cmaes_s09_holdout \
  --direction-split-seed 17 \
  --viewer null \
  --overwrite
```

Shipped knobs are `population_size=15`, `max_generations=10` ⇒ 75 envs. If it crashes (`exit 139`) or exceeds the session budget, shrink to `population_size=4`, `max_generations=3`, record exactly what ran, and restore the shipped values in code before committing.

- [ ] **Step 3: Check every acceptance item**

`cmaes_report.json`: `command_status` completed; no `gt_diagnostics`; spur/stem search floor \(\log_{10} E = 7\); every generation's loaded dirs ⊆ `{2,4,5,6,7}`.

`holdout_report.json`: `direction_split_seed == 17`; train `{2,4,5,6,7}`, val `{0,1,3}`; both phenotypes; `train_fitted` / `val_baseline` / `val_fitted` Sinkhorn + F/T MAE; `verification` flags; three val overlay paths.

Science gate: train `eligible_mean` last < first; `val_fitted` Sinkhorn < `val_baseline`; all four per-direction flags `True` for all three val dirs.

- [ ] **Step 4: Record**

Append to `.superpowers/sdd/progress.md` and the ROADMAP: knobs run, wall time, the three phenotype vectors, both Sinkhorn columns, F/T MAE, and each gate's scalars. If a gate fails, **do not** paper over it: report the failing direction and scalars and stop for a decision.

- [ ] **Step 5: Full test sweep**

```bash
uv run --env-file pytest.env python -m pytest -p no:launch_testing \
  apple_pick_sim/tests apple_pick_gym/tests -q
```

Consult `.cursor/rules/multitask-pytest.mdc` for the slow/fast split before running the whole suite serially.

---

## Definition of done

- [ ] Tasks 1–9 complete; every new test green; no knowingly red worktree.
- [ ] Default (no split flags) CMA behavior provably unchanged: sim-sim and 1×1 real tests pass untouched.
- [ ] `holdout_report.json` proves no val direction entered a `tell`.
- [ ] Science gate passed and its numbers recorded, or a clean stop with the failing scalars reported.
- [ ] Shipped `population_size` / `max_generations` restored; docs match what actually ran.
