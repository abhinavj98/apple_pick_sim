# Young's Modulus Grid Replay and Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Young's-modulus collect/viz example into a renamed, dataset-only system-identification CLI that replays a Cartesian E-grid for every recorded structure, ranks candidates with pooled hold-phase Sinkhorn loss, compares against stored GT, and optionally exports candidate replay datasets.

**Architecture:** Keep material candidate and evaluation logic in `batched_sysid_cmaes.py`, reuse the parameter-agnostic recorded-action replay path, and leave the renamed example as orchestration. Build reports and overlays from in-memory replay episodes; disk replay export remains optional.

**Tech Stack:** Python 3.11+, dataclasses/typing, NumPy, PyTorch, NVIDIA Warp/Newton, PyArrow Parquet, GeomLoss Sinkhorn, Plotly, pytest, uv.

## Global Constraints

- Follow TDD for every production change: add a failing test, confirm the expected failure, implement minimally, and rerun.
- Run Python and tests from the repository root through `uv run --env-file pytest.env`.
- Before production edits, follow the repository worktree rule and ask whether to create an isolated worktree.
- Do not alter the `newton/` submodule.
- Treat `fruiting_system_params` in episode Parquet metadata as the lossless GT source.
- Sweep primary, spur, and stem E only; freeze secondary E and every non-E parameter at stored GT.
- Route candidate E through `set_rod_youngs_modulus`; never patch bend stiffness directly.
- Default scoring is hold median + hold-ID one-hot + pooled directions with direction one-hot.
- Skip manifest-excluded directions unless `--include-excluded` is explicit.
- Preserve original sparse direction IDs through replay, scoring, visualization, and export.
- Do not implement CMA-ES in this slice.
- Do not modify the user's existing `.cursorignore` or `2026-07-15-sysid-cmaes-youngs-modulus-design.md` changes.

---

### Task 1: Ground-Truth Young's-Modulus Grid Helpers

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`
- Modify: `apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py`

**Interfaces:**
- Consumes: `BatchedSysIdDataset`, `true_params_for_structure`, and existing `youngs_modulus_candidate_from_params`.
- Produces:
  - `gt_youngs_modulus_candidate_from_structure(dataset, structure_idx) -> YoungsModulusCandidate`
  - `youngs_modulus_values_match(left, right, *, log10_atol=1e-9) -> bool`
  - `maybe_include_gt_candidate(candidates, gt, *, include_gt) -> list[YoungsModulusCandidate]`

- [ ] **Step 1: Write failing tests for GT extraction, tolerant matching, insertion, and secondary preservation**

Add tests equivalent to:

```python
def test_gt_candidate_reads_lossless_structure_params(monkeypatch):
    gt_params = _base_primary_spur_stem_with_secondary()
    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _dataset, _idx: gt_params)

    candidate = cmaes.gt_youngs_modulus_candidate_from_structure(object(), 3)

    assert candidate == cmaes.YoungsModulusCandidate(
        primary=gt_params.primary.youngs_modulus_pa,
        spur=gt_params.spur.youngs_modulus_pa,
        stem=gt_params.stem.youngs_modulus_pa,
    )
    assert gt_params.secondary is not None


def test_maybe_include_gt_candidate_is_configurable_and_deduplicates_in_log_space():
    gt = cmaes.YoungsModulusCandidate(1e8, 10**7.5, 1e7)
    near = cmaes.YoungsModulusCandidate(
        10 ** (8.0 + 5e-10), 10**7.5, 1e7
    )

    assert cmaes.maybe_include_gt_candidate([near], gt, include_gt=True) == [near]
    assert cmaes.maybe_include_gt_candidate([], gt, include_gt=False) == []
    assert cmaes.maybe_include_gt_candidate([], gt, include_gt=True) == [gt]
```

Extend the existing `apply_to` test with a non-`None` secondary rod and assert the secondary object values, including `youngs_modulus_pa`, are unchanged.

- [ ] **Step 2: Run the focused tests and confirm the missing-symbol failures**

Run:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py -q
```

Expected: FAIL because the three GT/grid helper symbols do not exist.

- [ ] **Step 3: Implement the helpers using log10 comparisons**

Add the following behavior:

```python
def gt_youngs_modulus_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> YoungsModulusCandidate:
    return youngs_modulus_candidate_from_params(
        true_params_for_structure(dataset, int(structure_idx))
    )


def youngs_modulus_values_match(
    left: YoungsModulusCandidate,
    right: YoungsModulusCandidate,
    *,
    log10_atol: float = 1e-9,
) -> bool:
    return all(
        math.isclose(math.log10(a), math.log10(b), rel_tol=0.0, abs_tol=log10_atol)
        for a, b in zip(left, right, strict=True)
    )


def maybe_include_gt_candidate(
    candidates: Sequence[YoungsModulusCandidate],
    gt: YoungsModulusCandidate,
    *,
    include_gt: bool,
) -> list[YoungsModulusCandidate]:
    items = list(candidates)
    if not include_gt or any(youngs_modulus_values_match(item, gt) for item in items):
        return items
    return [*items, gt]
```

Use imports that avoid a runtime cycle; place dataset-only imports behind `TYPE_CHECKING` or inside the extraction function if needed.

- [ ] **Step 4: Run the candidate suite**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit the independently testable helper slice**

```bash
git add \
  apple_pick_gym/batched_envs/batched_sysid_cmaes.py \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py
git commit -m "feat: add Young's modulus GT grid helpers"
```

### Task 2: Parameter-Agnostic Replay and Sparse-Direction Export

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`
- Modify: `apple_pick_sim/system_id/batched_replay_export.py`
- Modify: `apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py`
- Modify: `apple_pick_gym/tests/test_batched_replay_export.py`

**Interfaces:**
- Consumes: any candidate implementing `apply_to(FruitingSystemParams) -> FruitingSystemParams`.
- Produces:
  - `SysIdReplayCandidate` protocol used by `replay_candidates_for_structure` and `replay_batched_sysid_structure`.
  - `export_replay_candidates_for_structure` with a `source_direction_indices: Sequence[int] | None = None` keyword and `int` return value.
  - `_resolve_source_direction_indices(specs_and_replays, requested) -> tuple[int, ...]`.

- [ ] **Step 1: Write a failing replay test using `YoungsModulusCandidate`**

Add a mocked replay helper test that passes two Young's candidates and usable disk directions `[0, 2]`. Assert:

```python
assert build_calls[0]["per_env_params"] == [
    candidate_0.apply_to(base), candidate_0.apply_to(base),
    candidate_1.apply_to(base), candidate_1.apply_to(base),
]
assert collectors.to_arrays(0)["dir_idx"][0] == 0
assert collectors.to_arrays(1)["dir_idx"][0] == 2
```

The assertion must verify candidate-major environment order and original disk direction IDs, not dense local IDs.

- [ ] **Step 2: Write a failing export test for sparse source directions**

Construct source episodes `s00_d00.parquet` and `s00_d02.parquet`, call:

```python
n = export_replay_candidates_for_structure(
    export_dir,
    source_dataset=source,
    source_structure_idx=0,
    specs_and_replays=[(spec, [replay_d0, replay_d2])],
    source_direction_indices=[0, 2],
)
```

Assert the exported manifest and files retain direction IDs `0` and `2`; no `s00_d01.parquet` should be created.

- [ ] **Step 3: Run both focused test modules**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_replay_export.py -q
```

Expected: at least the sparse export test FAILS because the exporter currently assumes dense `0..N-1` source directions.

- [ ] **Step 4: Introduce the candidate protocol and preserve runtime behavior**

Add:

```python
class SysIdReplayCandidate(Protocol):
    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        raise NotImplementedError
```

Change only candidate annotations from `Sequence[BendStiffnessCandidate]` to
`Sequence[SysIdReplayCandidate]` in replay/chunking functions. Keep
`BendStiffnessCandidate` and bend-grid APIs intact.

- [ ] **Step 5: Add explicit source direction IDs to export**

Update the exporter:

```python
def _resolve_source_direction_indices(
    specs_and_replays: Sequence[
        tuple[ReplayCandidateSpec, Sequence[Mapping[str, Any]]]
    ],
    requested: Sequence[int] | None,
) -> tuple[int, ...]:
    if not specs_and_replays:
        return ()
    replay_count = len(specs_and_replays[0][1])
    direction_ids = (
        tuple(range(replay_count))
        if requested is None
        else tuple(int(value) for value in requested)
    )
    for _spec, replays in specs_and_replays:
        if len(replays) != len(direction_ids):
            raise ValueError(
                "source_direction_indices length must match replay episode count"
            )
    return direction_ids


def export_replay_candidates_for_structure(
    export_dir: Path | str,
    *,
    source_dataset: BatchedSysIdDataset,
    source_structure_idx: int,
    specs_and_replays: Sequence[
        tuple[ReplayCandidateSpec, Sequence[Mapping[str, Any]]]
    ],
    source_direction_indices: Sequence[int] | None = None,
    command_argv: Sequence[str] | None = None,
    skip_existing: bool = False,
) -> int:
    direction_ids = _resolve_source_direction_indices(
        specs_and_replays,
        source_direction_indices,
    )
```

When `source_direction_indices` is `None`, retain current dense behavior for
backward compatibility. Otherwise validate that its length equals every
candidate's replay episode count and use each supplied source ID for metadata,
source episode lookup, filename generation, and manifest entries in the
existing export loop.

- [ ] **Step 6: Run replay/export tests and the bend-grid CLI regression**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_replay_export.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the generic replay contract**

```bash
git add \
  apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py \
  apple_pick_sim/system_id/batched_replay_export.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_replay_export.py
git commit -m "refactor: generalize sys-id candidate replay"
```

### Task 3: Young's-Modulus Evaluation and Ranking Library

**Files:**
- Modify: `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`
- Create: `apple_pick_gym/tests/test_batched_sysid_youngs_grid.py`

**Interfaces:**
- Consumes:
  - `replay_candidates_for_structure`
  - `resolve_direction_indices`
  - `load_recorded_episodes_for_structure`
  - `direction_episodes_from_collectors`
  - `prepare_gt_wasserstein_context`
  - `score_candidate_wasserstein`
- Produces:

```python
@dataclass(frozen=True)
class YoungsModulusScoringConfig:
    use_median: bool = True
    hold_id_onehot: bool = True
    pool_directions: bool = True
    n_holds: int | None = None
    n_directions: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class YoungsModulusCandidateScore:
    candidate_index: int
    candidate: YoungsModulusCandidate
    aggregate_sinkhorn: float
    per_direction_sinkhorn: dict[int, float]
    instability_fraction: float
    disqualified: bool
    disqualification_reason: str | None
    rank: int | None
    is_gt: bool


@dataclass
class YoungsModulusEvaluation:
    structure_idx: int
    gt_candidate: YoungsModulusCandidate
    fixed_secondary_e_pa: float | None
    direction_indices: tuple[int, ...]
    scores: list[YoungsModulusCandidateScore]
    replay_episodes: list[list[dict[str, Any]]]
    applied_params: list[FruitingSystemParams]


def evaluate_youngs_modulus_candidates(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[YoungsModulusCandidate],
    num_directions: int,
    build_env_fn: Callable[..., Any],
    scoring: YoungsModulusScoringConfig,
    max_envs_per_batch: int = 0,
    seed: int | None = None,
    include_excluded: bool = False,
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
) -> YoungsModulusEvaluation
```

- [ ] **Step 1: Write failing ranking tests with mocked replay and scoring**

Cover:

```python
def test_evaluator_ranks_finite_eligible_scores_and_marks_gt(monkeypatch):
    # Candidate losses: far=3.0, GT=0.2, unstable=0.1.
    # The unstable candidate exceeds UNSTABLE_DISQUALIFY_THRESHOLD.
    evaluation = evaluate_youngs_modulus_candidates(
        dataset=dataset,
        structure_idx=0,
        candidates=[far, gt, unstable],
        num_directions=2,
        build_env_fn=build_env_fn,
        scoring=YoungsModulusScoringConfig(n_holds=5, n_directions=2),
    )
    assert [score.rank for score in evaluation.scores] == [2, 1, None]
    assert evaluation.scores[1].is_gt is True
    assert evaluation.scores[2].disqualification_reason == "replay_instability"


def test_evaluator_uses_all_original_usable_direction_ids(monkeypatch):
    evaluation = evaluate_youngs_modulus_candidates(
        dataset=dataset_with_directions_0_and_2,
        structure_idx=0,
        candidates=[gt],
        num_directions=3,
        build_env_fn=build_env_fn,
        scoring=YoungsModulusScoringConfig(n_holds=5, n_directions=3),
    )
    assert evaluation.direction_indices == (0, 2)
    assert replay_call["direction_indices"] == [0, 2]
```

Also test non-finite scores and missing Sinkhorn direction bags are
disqualified, and ties are ordered by candidate index.

- [ ] **Step 2: Run the new module and verify red**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py -q
```

Expected: FAIL because the result dataclasses and evaluator do not exist.

- [ ] **Step 3: Implement context preparation, replay, and candidate scoring**

The evaluator must:

```python
direction_indices = resolve_direction_indices(
    dataset,
    structure_idx=structure_idx,
    num_directions=num_directions,
    include_excluded=include_excluded,
)
recorded = load_recorded_episodes_for_structure(
    dataset,
    structure_idx=structure_idx,
    num_directions=len(direction_indices),
    direction_indices=direction_indices,
)
gt_context = prepare_gt_wasserstein_context(
    recorded,
    use_median=scoring.use_median,
    hold_id_onehot=scoring.hold_id_onehot,
    n_holds=scoring.n_holds,
    pool_directions=scoring.pool_directions,
    n_directions=scoring.n_directions,
)
collectors = replay_candidates_for_structure(
    dataset=dataset,
    structure_idx=structure_idx,
    candidates=candidates,
    num_directions=len(direction_indices),
    direction_indices=direction_indices,
    seed=seed,
    build_env_fn=build_env_fn,
    max_envs_per_batch=max_envs_per_batch,
    on_step=on_step,
    replay_sim_config=replay_sim_config,
    use_oracle_params=True,
)
```

For each candidate, gather replay episodes, compute its maximum per-direction
instability fraction against the corresponding recorded episode, call
`score_candidate_wasserstein` with keys
`primary_e_pa`, `spur_e_pa`, and `stem_e_pa`, then assign ranks only after all
disqualifications are known.

- [ ] **Step 4: Implement deterministic ranking and GT flags**

Use:

```python
eligible = [
    score for score in provisional
    if not score.disqualified and math.isfinite(score.aggregate_sinkhorn)
]
ordered = sorted(
    eligible,
    key=lambda score: (score.aggregate_sinkhorn, score.candidate_index),
)
rank_by_index = {
    score.candidate_index: rank
    for rank, score in enumerate(ordered, start=1)
}
```

Mark `is_gt` through `youngs_modulus_values_match`, not candidate position.
Calculate `applied_params` from the oracle base once per candidate so export
uses the exact simulated material values.

- [ ] **Step 5: Run the evaluator and existing Wasserstein tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_mmd_features.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the reusable evaluator**

```bash
git add \
  apple_pick_gym/batched_envs/batched_sysid_cmaes.py \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py
git commit -m "feat: evaluate and rank Young's modulus candidates"
```

### Task 4: Rename and Convert the CLI to Dataset-Only Replay

**Files:**
- Rename: `apple_pick_gym/batched_examples/example_batched_youngs_modulus_collect_viz.py` → `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`
- Create: `apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py`
- Modify: `apple_pick_gym/tests/test_example_batched_youngs_modulus_cli.py`

**Interfaces:**
- Consumes: Task 1 grid helpers, Task 3 evaluator, and setup patterns from `example_batched_sysid_mmd_grid.py`.
- Produces:
  - `_make_parser()`
  - `_resolve_structure_indices(dataset, requested) -> list[int]`
  - `_resolve_n_holds(dataset, collection) -> int | None`
  - `_resolve_n_directions(dataset, collection) -> int`
  - `_make_build_env_fn(*, ranges_path, topology_seed, control_hz, device, settle_config)`
  - `_run(args, parser, *, viewer) -> dict[str, Any]`

- [ ] **Step 1: Write failing parser and orchestration tests against the new filename**

Assert:

```python
parser = module._make_parser()
args = parser.parse_args([
    "--dataset", "/tmp/gt",
    "--output", "/tmp/rank",
])
assert args.include_gt_candidate is True
assert args.use_median is True
assert args.hold_id_onehot is True
assert args.pool_directions is True
assert args.export_replays is False

with pytest.raises(SystemExit):
    parser.parse_args(["--output", "/tmp/rank"])

option_strings = {
    option
    for action in parser._actions
    for option in action.option_strings
}
assert "--topology-seed" not in option_strings
assert "--num-structures" not in option_strings
```

Mock the evaluator and assert `_run` iterates every manifest structure,
constructs the per-segment Cartesian grid, and conditionally inserts GT.

- [ ] **Step 2: Run CLI tests and confirm the new module is absent**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py -q
```

Expected: FAIL because `example_youngs_modulus_sys_id.py` does not exist.

- [ ] **Step 3: Rename the script and replace synthetic collection arguments**

Retain reusable simulation setup functions but remove:

- topology/structure sampling;
- `--num-directions` as the source of truth;
- synthetic `per_env_params`;
- `collect_batched_quasi_static_dataset`;
- standalone output-dataset creation.

Add required dataset/output arguments, structure selection, grid controls,
Boolean GT insertion, scoring defaults, replay export, overlay cap,
`--include-excluded`, `--fail-fast`, and overwrite controls.

- [ ] **Step 4: Implement per-structure dataset orchestration**

For every selected structure:

```python
gt = gt_youngs_modulus_candidate_from_structure(dataset, structure_idx)
candidates = candidates_from_log10_cli(
    log10_e_primary=args.log10_e_primary,
    log10_e_spur=args.log10_e_spur,
    log10_e_stem=args.log10_e_stem,
)
candidates = maybe_include_gt_candidate(
    candidates,
    gt,
    include_gt=bool(args.include_gt_candidate),
)
if args.max_candidates > 0 and len(candidates) > args.max_candidates:
    parser.error(
        f"candidate grid has {len(candidates)} entries, exceeding "
        f"--max-candidates={args.max_candidates}"
    )
```

Resolve collection control rate, ranges, topology seed, settle/sim config,
hold count, and direction count from the source manifest using the same
fallback policy as the validated MMD-grid CLI. Invoke the evaluator once per
structure and continue past structure errors unless `--fail-fast`.

- [ ] **Step 5: Preserve viewer behavior without making it authoritative**

Keep optional GL/null rendering through the replay callback. Headless Linux
must default to the null viewer. Closing the viewer must remain in `finally`.

- [ ] **Step 6: Run CLI tests and import/help smoke**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_example_batched_youngs_modulus_cli.py -q
uv run --env-file pytest.env python \
  apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py --help
```

Expected: tests PASS and help exits 0.

- [ ] **Step 7: Commit the dataset-only renamed CLI**

```bash
git add \
  apple_pick_gym/batched_examples/example_batched_youngs_modulus_collect_viz.py \
  apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  apple_pick_gym/tests/test_example_batched_youngs_modulus_cli.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py
git commit -m "feat: add dataset-driven Young's modulus sys-id CLI"
```

### Task 5: Ranking Report, Top-K Overlay, and Optional Replay Export

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`
- Modify: `apple_pick_gym/youngs_modulus_overlay_viz.py`
- Modify: `apple_pick_gym/tests/test_youngs_modulus_overlay_viz.py`
- Modify: `apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py`

**Interfaces:**
- Consumes: `YoungsModulusEvaluation` and `export_replay_candidates_for_structure` with explicit `source_direction_indices`.
- Produces:
  - `select_overlay_candidate_indices(scores, *, max_candidates) -> list[int]`
  - `overlay_episodes_from_replay_evaluation(evaluation, candidate_indices) -> list[OverlayEpisode]`
  - `_structure_result_to_json(evaluation) -> dict[str, Any]`
  - `_aggregate_ranking_report(structure_rows, errors) -> dict[str, Any]`

- [ ] **Step 1: Write failing top-K-plus-GT overlay tests**

Create five ranked candidates with GT ranked fifth and request
`max_candidates=3`. Assert:

```python
selected = select_overlay_candidate_indices(scores, max_candidates=3)
assert selected == [best_idx, second_idx, gt_idx]
```

Assert the in-memory adapter preserves each replay episode's original
`dir_idx`, candidate label, log10-E triple, stability exclusion, force/torque,
and TCP arrays.

- [ ] **Step 2: Write failing report and export orchestration tests**

Assert `ranking.json` rows contain:

```python
{
    "candidate_index": 0,
    "youngs_modulus_pa": {
        "primary": 1e8,
        "spur": 10**7.5,
        "stem": 1e7,
    },
    "log10_e": [8.0, 7.5, 7.0],
    "aggregate_sinkhorn": 0.125,
    "rank": 1,
    "is_gt": True,
    "instability_fraction": 0.0,
    "disqualified": False,
    "disqualification_reason": None,
}
```

Verify winner log-space error and per-segment relative error, exact GT rank,
fixed secondary E, aggregate GT-rank distribution, skipped structures, and
non-zero failure when all structures fail.

Mock export and assert it receives:

```python
source_direction_indices=evaluation.direction_indices
```

- [ ] **Step 3: Run visualization and CLI tests to verify red**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_youngs_modulus_overlay_viz.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py -q
```

Expected: FAIL because selection, in-memory overlay, and report helpers do not exist.

- [ ] **Step 4: Implement top-K selection and in-memory overlay conversion**

Selection rules:

1. Take eligible ranked candidates in ascending rank.
2. Reserve one slot for GT if GT exists and is not already selected.
3. Fill remaining slots deterministically by rank, then candidate index.
4. If every candidate is disqualified, return an empty overlay selection and
   still write the JSON report.

Build `OverlayEpisode` directly from `evaluation.replay_episodes`; do not
require writing or reopening mini-datasets.

- [ ] **Step 5: Implement atomic report output and optional exports**

Serialize dataclasses explicitly to JSON-safe values. Write `ranking.json` to
a sibling temporary file and replace the destination only after successful
serialization. Require `--overwrite` when the destination exists.

For each candidate export:

```python
ReplayCandidateSpec(
    candidate_index=candidate_index,
    params=evaluation.applied_params[candidate_index],
    stiffnesses={
        "primary_e_pa": candidate.primary,
        "spur_e_pa": candidate.spur,
        "stem_e_pa": candidate.stem,
    },
)
```

Call the exporter with the evaluation's original direction IDs. Catch and
record optional overlay/export errors without discarding a successfully
computed ranking.

- [ ] **Step 6: Run focused reporting/visualization tests**

Run the command from Step 3.

Expected: PASS.

- [ ] **Step 7: Commit the output layer**

```bash
git add \
  apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  apple_pick_gym/youngs_modulus_overlay_viz.py \
  apple_pick_gym/tests/test_youngs_modulus_overlay_viz.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py
git commit -m "feat: report Young's modulus replay rankings"
```

### Task 6: Documentation, Regression Verification, and CUDA Smoke

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/system_identification.md`
- Test: focused modules from Tasks 1–5

**Interfaces:**
- Consumes: final CLI and report schema.
- Produces: canonical uv commands for dataset collection and E-grid ranking.

- [ ] **Step 1: Update documentation references and commands**

Replace references to
`example_batched_youngs_modulus_collect_viz.py` with
`example_youngs_modulus_sys_id.py`. Document the two-step workflow:

```bash
uv run --env-file pytest.env python \
  apple_pick_gym/batched_examples/example_batched_collect_sysid_data.py \
  --viewer null \
  --num-structures 1 \
  --num-directions 2 \
  --max-steps 80 \
  --output tmp/youngs_gt_smoke \
  --overwrite

uv run --env-file pytest.env python \
  apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \
  --viewer null \
  --dataset tmp/youngs_gt_smoke \
  --output tmp/youngs_grid_rank_smoke \
  --log10-e-primary 8.0,8.5 \
  --log10-e-spur 7.5 \
  --log10-e-stem 7.0 \
  --include-gt-candidate \
  --max-candidates 8 \
  --overwrite
```

Explain that GT values come from episode `fruiting_system_params`, secondary
E is fixed, and rank-one is expected only for healthy samples.

- [ ] **Step 2: Run the complete focused fast suite**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_batched_sysid_cmaes_candidate.py \
  apple_pick_gym/tests/test_batched_sysid_youngs_grid.py \
  apple_pick_gym/tests/test_batched_sysid_mmd_grid_helpers.py \
  apple_pick_gym/tests/test_batched_replay_export.py \
  apple_pick_gym/tests/test_youngs_modulus_overlay_viz.py \
  apple_pick_gym/tests/test_example_youngs_modulus_sys_id_cli.py \
  apple_pick_gym/tests/test_example_batched_sysid_mmd_grid_cli.py \
  apple_pick_sim/tests/test_wasserstein.py \
  apple_pick_sim/tests/test_mmd_features.py -q
```

Expected: PASS with no unexpected skips or failures.

- [ ] **Step 3: Check lints for all edited Python files**

Use IDE diagnostics for:

- `apple_pick_gym/batched_envs/batched_sysid_cmaes.py`
- `apple_pick_gym/batched_envs/batched_sysid_mmd_grid.py`
- `apple_pick_sim/system_id/batched_replay_export.py`
- `apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py`
- `apple_pick_gym/youngs_modulus_overlay_viz.py`

Expected: no new diagnostics.

- [ ] **Step 4: Run the documented CUDA smoke**

Run both commands from Step 1 on CUDA. Verify:

```bash
uv run python -c "
import json
from pathlib import Path
p = Path('tmp/youngs_grid_rank_smoke/ranking.json')
r = json.loads(p.read_text())
assert r['structures']
assert 'aggregate' in r
assert Path('tmp/youngs_grid_rank_smoke/structure_000/youngs_modulus_overlay.html').exists()
"
```

Expected: commands exit 0, report schema assertions pass, and overlay exists.
If CUDA is unavailable, run the smallest CPU smoke supported by the fixture
and record the unrun CUDA acceptance check in the handoff.

- [ ] **Step 5: Run final diff review and commit docs**

Confirm documentation commands exactly match the tested invocation. Do not
stage unrelated `.cursorignore` or pre-existing spec changes.

```bash
git add README.md docs/ROADMAP.md docs/system_identification.md
git commit -m "docs: document Young's modulus grid ranking"
```

- [ ] **Step 6: Request code review before integration**

Run the repository's code-review workflow against the complete branch diff,
resolve only findings within this feature's scope, and rerun the focused suite
after any fixes.

