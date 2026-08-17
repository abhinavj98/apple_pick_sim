# Real `vic_pose` CMA-ES Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `example_youngs_modulus_cmaes.py` replay converted real `vic_pose` bags through the same H4 builder the Cartesian grid uses, without requiring sim-oracle GT, on this branch's current 1×1 convert.

**Architecture:** Keep the existing pycma ask/tell loop (`fit_youngs_modulus_structures`) unchanged. In `_run`, resolve controller mode the same way the grid does (`dataset_declares_vic_pose` / `--controller-mode`), then opt into `make_real_replay_build_env_fn` + `real_replay_sim_config`, skip `gt_support_kp_youngs_candidate_from_structure`, pass `action_dim=19`, and widen only the real spur/stem search floor to \(\log_{10} E = 7\). Twist-`vic` sim-sim CMA stays on `_make_build_env_fn` and the shipped `[8, 11]` E box.

**Tech Stack:** Python, pycma via `batched_sysid_cmaes.py`, pytest + `uv run --env-file pytest.env`.

## Global Constraints

- Work on `feature/real-replay-parallel-sysid` (already an isolated worktree; do not create another; do not edit `main`).
- TDD: failing test before production code; run `uv run --env-file pytest.env python -m pytest -p no:launch_testing <path> -q`.
- Spec: `docs/superpowers/specs/2026-08-14-real-multi-structure-cmaes-design.md` slice 4 only.
- Phenotype unchanged: \(\log_{10}(k_p^{\mathrm{support}}, E_{\mathrm{spur}}, E_{\mathrm{stem}})\); primary \(E\) fixed.
- Ranking GT is the converted real bag; `gt_candidate is None` on real structures.
- Do **not** migrate gym collect / MMD / default sim-sim CMA off twist `vic`.
- Do **not** implement folder convert, per-direction weld, frame padding, or the grid multi-structure `vic_pose` guard removal (spec slices 1–3).
- Keep `CMA_SEARCH_PARAMS` sim-sim box `[2,8,8]–[6,11,11]` as the dict tests assert; apply the real E floor only when mode is `vic_pose`.
- Mirror the grid's one-structure `vic_pose` guard (1×1 convert is the only supported real CMA input in this slice).
- `evaluate_youngs_modulus_*` already infers `action_dim` from metadata if omitted; still pass it explicitly like the grid.

## File map

| Path | Responsibility |
| --- | --- |
| `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` | `--controller-mode`; real vs sim builder; skip GT; real search floor; pass `action_dim` |
| `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py` | CLI + `_run` contract tests; stub `load_episode_metadata` on existing sim `_run` tests |
| `docs/handbook-youngs-cma.md` | Status boundary: CMA real path shipped for 1×1 |
| `docs/ROADMAP.md` | Slice 4 checklist + validation command |
| `README.md` | Real convert → CMA smoke command |

**Do not modify:** `batched_sysid_cmaes.py` evaluator (already sets `gt_candidate = None` on `dataset_declares_vic_pose`); `example_youngs_modulus_sys_id.py` except as a copy-paste reference.

---

### Task 1: `--controller-mode` on the CMA parser

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` (`_make_parser`)
- Test: `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py`

**Interfaces:**
- Consumes: `newton.examples.create_parser()` as today
- Produces: `args.controller_mode` is `None` (default), `"vic"`, or `"vic_pose"`

- [ ] **Step 1: Write the failing parser test**

Add next to the existing `--cma-seed` parser tests in `test_example_youngs_modulus_cmaes_cli.py`:

```python
def test_parser_accepts_controller_mode_vic_pose(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/ds",
            "--output",
            "/tmp/out",
            "--controller-mode",
            "vic_pose",
        ]
    )
    assert args.controller_mode == "vic_pose"


def test_parser_controller_mode_defaults_to_none(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(["--dataset", "/tmp/ds", "--output", "/tmp/out"])
    assert args.controller_mode is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py::test_parser_accepts_controller_mode_vic_pose \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py::test_parser_controller_mode_defaults_to_none \
  -q -p no:launch_testing
```

Expected: FAIL — `unrecognized arguments: --controller-mode` / `args` has no `controller_mode`.

- [ ] **Step 3: Add the flag (copy the grid help text)**

In `_make_parser`, after the `--seed` argument (around line 470), add:

```python
    p.add_argument(
        "--controller-mode",
        choices=("vic", "vic_pose"),
        default=None,
        help="Replay controller mode (default: infer vic_pose from dataset, else vic).",
    )
```

- [ ] **Step 4: Re-run the two parser tests**

Same command as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py
git commit -m "$(cat <<'EOF'
Add CMA --controller-mode so real vic_pose datasets can opt in.

The grid already infers or accepts vic_pose; CMA could only drive twist vic.
EOF
)"
```

---

### Task 2: Real builder, refuse wrench-as-twist, skip sim GT

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` (imports, `_run` builder block, `evaluate_fn` `action_dim`)
- Modify: `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py` (new tests + `load_episode_metadata` stubs on existing `_run` tests)

**Interfaces:**
- Consumes: `dataset_declares_vic_pose`, `check_action_semantics`, `make_real_replay_build_env_fn`, `real_replay_sim_config`, `control_hz_from_episode_metadata`, `fruiting_base_pos_from_episode_metadata`, `bootstrap_joint_q_from_episode_metadata` from `apple_pick_gym.batched_envs.real_batched_replay_build`; `_grid.SETTLE_SUBSTEPS`
- Produces: `_run` on a `vic_pose_v1` dataset calls `make_real_replay_build_env_fn` (not `_make_build_env_fn`), sets `state.gt_candidate = None` without calling `gt_support_kp_youngs_candidate_from_structure`, passes `action_dim=19` into `evaluate_youngs_modulus_structures` / `evaluate_youngs_modulus_candidates`

**Existing-test trap:** `_run` will call `dataset.load_episode_metadata(structure_indices[0], 0)`. A bare `MagicMock` metadata makes `int(episode_meta.get("action_dim") or 0)` raise inside `dataset_declares_vic_pose`. Every current `_run` test must stub a 6D episode dict (Step 1 helper) **before** the new real-path tests can be the only failures.

- [ ] **Step 1: Add a sim-episode stub helper and apply it to every existing `_run` test**

At the top of `test_example_youngs_modulus_cmaes_cli.py` (after `_valid_ranges_dict`):

```python
def _sim_episode_meta() -> dict:
    return {"action_dim": 6, "action_compatible_with_vic_twist": True}


def _attach_sim_episode_meta(dataset: MagicMock) -> MagicMock:
    dataset.load_episode_metadata.return_value = _sim_episode_meta()
    return dataset
```

In **every** test that builds a `dataset = MagicMock()` and later calls `module._run`, add `_attach_sim_episode_meta(dataset)` immediately after setting `dataset.manifest` / `structure_summaries`. There are 10 `_run` call sites in this file.

Do **not** change production code yet. Re-run the file — still green (helper is unused-effect until `_run` loads metadata):

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py -q -p no:launch_testing
```

Expected: PASS (production still does not call `load_episode_metadata`).

- [ ] **Step 2: Write the failing real-builder test**

Mirror `test_run_vic_pose_dataset_uses_real_builder_and_skips_gt` from `test_example_youngs_modulus_sys_id_cli.py`, adapted to CMA (`fit_youngs_modulus_structures` instead of `evaluate_youngs_modulus_candidates`):

```python
def test_run_vic_pose_dataset_uses_real_builder_and_skips_gt(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "action_layout": "vic_pose_v1",
            "action_dim": 19,
            "control_hz": 15.0,
            "num_directions": 1,
            "ranges_path": str(ranges_path),
            "topology_seed": 9,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
        "control_hz": 15.0,
        "fruiting_base_pos": [1.0, 2.0, 3.0],
        "initial_robot_joint_q": [0.1, 0.2],
        "action_compatible_with_vic_twist": False,
    }

    real_builder = MagicMock()
    real_builder_calls: list[dict] = []
    evaluate_calls: list[dict] = []

    def fake_make_real_builder(**kwargs):
        real_builder_calls.append(dict(kwargs))
        return real_builder

    def fake_real_config(**kwargs):
        return SimpleNamespace(
            controller=SimpleNamespace(mode=kwargs["controller_mode"], action_dim=19),
            runtime=SimpleNamespace(control_hz=kwargs["control_hz"]),
        )

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        del max_generations
        for state in states.values():
            assert state.gt_candidate is None
            state.status = "fitted"
            state.final_mean_log10 = (4.0, 9.0, 9.0)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
                [0.1],
                direction_indices=(0,),
            )
            state.final_evaluation.gt_candidate = None
            state.gt_candidate = None
        batch = evaluate_fn(
            structures=[
                (0, (cmaes.candidates_from_log10_vector((4.0, 9.0, 9.0)),))
            ],
            wave_kind="final_mean",
        )
        evaluate_calls.append(batch)
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(0,),
            failed_structure_indices=(),
            generation_waves=1,
            final_mean_batch=None,
        )

    def fake_evaluate_structures(**kwargs):
        evaluate_calls.append(dict(kwargs))
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={
                0: _evaluation(
                    0,
                    [cmaes.SupportKpYoungsCandidate(1e4, 1e9, 1e7)],
                    [0.1],
                    direction_indices=(0,),
                )
            },
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
            prepared_structures=1,
            physical_slots_by_structure={0: 1},
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "make_real_replay_build_env_fn", fake_make_real_builder)
    monkeypatch.setattr(module, "real_replay_sim_config", fake_real_config)
    monkeypatch.setattr(module, "evaluate_youngs_modulus_structures", fake_evaluate_structures)
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: pytest.fail("real CMA must not load sim GT"),
    )
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)

    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        multi_structure_batch=True,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=False,
        settle_quiet_every=None,
        show_pull_direction=False,
        viewer="null",
    )
    result = module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert result["exit_nonzero"] is False
    assert len(real_builder_calls) == 1
    assert real_builder_calls[0]["controller_mode"] == "vic_pose"
    assert real_builder_calls[0]["control_hz"] == pytest.approx(15.0)
    assert real_builder_calls[0]["fruiting_base_pos"] == pytest.approx((1.0, 2.0, 3.0))
    assert real_builder_calls[0]["bootstrap_joint_q"] == pytest.approx((0.1, 0.2))
    struct_kwargs = [c for c in evaluate_calls if isinstance(c, dict)]
    assert struct_kwargs[0]["build_env_fn"] is real_builder
    assert struct_kwargs[0]["action_dim"] == 19
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert "gt_diagnostics" not in report["structures"]["0"]
```

Add a second failing test for the one-structure guard (copy the grid's `test_run_rejects_multiple_structures_for_vic_pose` but call CMA `_run`):

```python
def test_run_rejects_multiple_structures_for_vic_pose(monkeypatch, tmp_path):
    module = _load_module()
    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "action_layout": "vic_pose_v1",
            "action_dim": 19,
            "control_hz": 15.0,
            "num_directions": 1,
            "ranges_path": "/tmp/ranges.json",
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
        "fruiting_base_pos": [0.0, 0.0, 0.0],
        "initial_robot_joint_q": [0.0],
        "control_hz": 15.0,
        "action_compatible_with_vic_twist": False,
    }
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(tmp_path / "out"),
        structure_indices=(0, 1),
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode="vic",
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        multi_structure_batch=True,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=False,
        settle_quiet_every=None,
        show_pull_direction=False,
        viewer="null",
    )
    with pytest.raises(
        SystemExit, match="one converted episode / one structure per run"
    ):
        module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
```

Note: `controller_mode="vic"` on a packed `vic_pose_v1` dataset must still trip the structure guard **or** `check_action_semantics`. Prefer matching the grid: guard is `(mode == "vic_pose" or dataset_is_vic_pose) and len(structure_indices) > 1`, so even `--controller-mode vic` on a packed dataset is rejected as multi-structure real replay. `check_action_semantics` then also refuses wrench-as-twist; the SystemExit message from the guard is enough.

- [ ] **Step 3: Run the new tests to verify they fail**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py::test_run_vic_pose_dataset_uses_real_builder_and_skips_gt \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py::test_run_rejects_multiple_structures_for_vic_pose \
  -q -p no:launch_testing
```

Expected: FAIL — `_make_build_env_fn` is used / `gt_support_kp_youngs_candidate_from_structure` is called / no SystemExit for two structures.

- [ ] **Step 4: Minimal production change**

Imports at the top of `example_youngs_modulus_cmaes.py` (with the other gym imports):

```python
from apple_pick_gym.batched_envs.real_batched_replay_build import (
    bootstrap_joint_q_from_episode_metadata,
    check_action_semantics,
    control_hz_from_episode_metadata,
    dataset_declares_vic_pose,
    fruiting_base_pos_from_episode_metadata,
    make_real_replay_build_env_fn,
    real_replay_sim_config,
)
```

Re-export settle default used by the grid real path:

```python
SETTLE_SUBSTEPS = _grid.SETTLE_SUBSTEPS
```

In `_run`, **after** `structure_indices` is resolved and **before** `build_env_fn = _make_build_env_fn(...)` (currently ~line 577), insert the same mode resolution as `example_youngs_modulus_sys_id.py` `_run` lines 909–974. Replace the unconditional sim builder with:

```python
    episode_meta = dataset.load_episode_metadata(structure_indices[0], 0)
    dataset_is_vic_pose = dataset_declares_vic_pose(collection, episode_meta)
    mode = getattr(args, "controller_mode", None)
    if mode is None:
        mode = "vic_pose" if dataset_is_vic_pose else "vic"
    check_action_semantics(
        controller_mode=mode,
        collection=collection,
        episode_meta=episode_meta,
        allow_wrench_as_twist=False,
    )
    if (mode == "vic_pose" or dataset_is_vic_pose) and len(structure_indices) > 1:
        raise SystemExit(
            "vic_pose real replay currently supports one converted episode / "
            "one structure per run; select exactly one --structure-index."
        )
    action_dim = 19 if mode == "vic_pose" else 6

    settle_config = _settle_config_kwargs(args=args)
    if mode == "vic_pose":
        control_hz = control_hz_from_episode_metadata(
            episode_meta,
            collection=collection,
        )
        fruiting_base_pos = fruiting_base_pos_from_episode_metadata(episode_meta)
        bootstrap_joint_q = bootstrap_joint_q_from_episode_metadata(episode_meta)
        real_topology_seed = int(
            collection.get("topology_seed", collection.get("seed", 0))
        )
        build_env_fn = make_real_replay_build_env_fn(
            ranges_path=Path(ranges_path),
            ranges=ranges,
            topology_seed=real_topology_seed,
            fruiting_base_pos=fruiting_base_pos,
            episode_meta=episode_meta,
            settle_substeps=settle_config.get("settle_substeps") or SETTLE_SUBSTEPS,
            settle_quiet_every=settle_config.get("settle_quiet_every"),
            settle_gravity_ramp=bool(settle_config.get("settle_gravity_ramp")),
            post_grasp_settle_substeps=500,
            bootstrap_joint_q=bootstrap_joint_q,
            controller_mode="vic_pose",
            control_hz=control_hz,
        )
        replay_sim_config = real_replay_sim_config(
            num_envs=1,
            topology_seed=real_topology_seed,
            fruiting_base_pos=fruiting_base_pos,
            ranges=ranges,
            settle_substeps=settle_config.get("settle_substeps") or SETTLE_SUBSTEPS,
            settle_quiet_every=settle_config.get("settle_quiet_every"),
            settle_gravity_ramp=bool(settle_config.get("settle_gravity_ramp")),
            post_grasp_settle_substeps=500,
            bootstrap_joint_q=bootstrap_joint_q,
            controller_mode="vic_pose",
            control_hz=control_hz,
        )
    else:
        build_env_fn = _make_build_env_fn(
            ranges_path=str(ranges_path),
            topology_seed=topology_seed,
            control_hz=control_hz,
            device=device,
            settle_config=settle_config,
        )
        replay_sim_config = build_sim_config(num_envs=1, ranges=ranges, **settle_config)
```

Delete the old unconditional `_make_build_env_fn` / `build_sim_config` pair that this block replaces. Keep `replay_seed` resolution where it already lives; if it currently sits between builder construction and scoring, leave it — only the builder/config assignment moves.

Replace the GT try/except (~615–622) with:

```python
        if mode == "vic_pose":
            state.gt_candidate = None
        else:
            try:
                state.gt_candidate = gt_support_kp_youngs_candidate_from_structure(
                    dataset, int(structure_idx)
                )
            except Exception as exc:
                state.status = "failed"
                state.failure = CmaGenerationFailure("prepare", str(exc))
```

Pass `action_dim=action_dim` into both `evaluate_youngs_modulus_structures` and `evaluate_youngs_modulus_candidates` inside `evaluate_fn`.

`_gt_error_diagnostics` already returns `None` when `gt is None`; do not change the report builder.

- [ ] **Step 5: Run new tests + the full CMA CLI file**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  -q -p no:launch_testing
```

Expected: PASS, including the pre-existing sim-sim `_run` tests now that they stub 6D metadata.

- [ ] **Step 6: Commit**

```bash
git add apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py
git commit -m "$(cat <<'EOF'
Drive real vic_pose CMA through the shared replay builder.

Skip sim-oracle GT on packed datasets and refuse wrench-as-twist so CMA matches the grid path.
EOF
)"
```

---

### Task 3: Real search floor \(\log_{10} E = 7\)

**Files:**
- Modify: `apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py` (`CMA_SEARCH_PARAMS` comments + `_effective_search_bounds_log10`)
- Test: `apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py`

**Interfaces:**
- Consumes: resolved `mode` from Task 2; `CMA_SEARCH_PARAMS["search_bounds_log10"]`
- Produces: `create_structure_cma_optimizer(..., search_bounds_log10=((2.0, 7.0, 7.0), (6.0, 11.0, 11.0)))` on `vic_pose`; sim-sim still `((2.0, 8.0, 8.0), (6.0, 11.0, 11.0))`

Do **not** change `_CMA_SEARCH_LOG10_LOWER` / `test_cma_search_params_dict_is_sole_search_truth_source`.

- [ ] **Step 1: Write the failing bounds test**

Extend `test_run_vic_pose_dataset_uses_real_builder_and_skips_gt` **or** add:

```python
def test_run_vic_pose_lowers_spur_stem_search_floor(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "action_layout": "vic_pose_v1",
            "action_dim": 19,
            "control_hz": 15.0,
            "num_directions": 1,
            "ranges_path": str(ranges_path),
            "topology_seed": 9,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
        "control_hz": 15.0,
        "fruiting_base_pos": [1.0, 2.0, 3.0],
        "initial_robot_joint_q": [0.1, 0.2],
        "action_compatible_with_vic_twist": False,
    }

    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        return cmaes.create_structure_cma_optimizer(bounds, **kwargs)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        del max_generations, evaluate_fn
        for state in states.values():
            state.status = "fitted"
            state.gt_candidate = None
            state.final_mean_log10 = (4.0, 9.0, 9.0)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
                [0.1],
                direction_indices=(0,),
            )
            state.final_evaluation.gt_candidate = None
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(0,),
            failed_structure_indices=(),
            generation_waves=1,
            final_mean_batch=None,
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "make_real_replay_build_env_fn", lambda **_k: MagicMock())
    monkeypatch.setattr(
        module,
        "real_replay_sim_config",
        lambda **_k: SimpleNamespace(controller=SimpleNamespace(mode="vic_pose")),
    )
    monkeypatch.setattr(module, "create_structure_cma_optimizer", fake_create)
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)

    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        multi_structure_batch=True,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=False,
        settle_quiet_every=None,
        show_pull_direction=False,
        viewer="null",
    )
    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert create_calls[0]["search_bounds_log10"] == (
        (2.0, 7.0, 7.0),
        (6.0, 11.0, 11.0),
    )
```

Keep `test_run_passes_shipped_search_bounds_to_optimizer` asserting the sim box `(2,8,8)–(6,11,11)`.

- [ ] **Step 2: Run the new test to verify it fails**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py::test_run_vic_pose_lowers_spur_stem_search_floor \
  -q -p no:launch_testing
```

Expected: FAIL — optimizer still receives `(2.0, 8.0, 8.0)`.

- [ ] **Step 3: Apply the real floor after mode is known**

Add constants next to `_CMA_SEARCH_LOG10_LOWER` (do not mutate that list):

```python
_REAL_CMA_SEARCH_LOG10_LOWER = [2.0, 7.0, 7.0]  # support_kp 1e2, spur/stem 10 MPa
_REAL_CMA_SEARCH_LOG10_UPPER = [6.0, 11.0, 11.0]
```

Add:

```python
def _effective_search_bounds_log10(
    mode: str,
    search: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Sim-sim uses CMA_SEARCH_PARAMS; vic_pose lowers spur/stem floor to 1e7 Pa."""
    if mode == "vic_pose":
        return (
            tuple(float(x) for x in _REAL_CMA_SEARCH_LOG10_LOWER),
            tuple(float(x) for x in _REAL_CMA_SEARCH_LOG10_UPPER),
        )
    raw = search.get("search_bounds_log10")
    return normalize_search_bounds_log10(raw)
```

In `_run`, replace the existing `normalize_search_bounds_log10(search.get("search_bounds_log10"))` call with `_effective_search_bounds_log10(mode, search)` **after** `mode` is resolved. `create_structure_cma_optimizer` already receives `search_bounds_log10`.

Comment above `_CMA_SEARCH_LOG10_LOWER`: sim-sim box is 0.1–100 GPa; real `vic_pose` overrides spur/stem floor to \(10^7\) Pa via `_effective_search_bounds_log10`.

- [ ] **Step 4: Run CMA CLI tests**

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  -q -p no:launch_testing
```

Expected: PASS. `test_cma_search_params_dict_is_sole_search_truth_source` still sees `[2,8,8]–[6,11,11]` in `CMA_SEARCH_PARAMS`.

- [ ] **Step 5: Commit**

```bash
git add apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py
git commit -m "$(cat <<'EOF'
Lower real CMA spur/stem search floor to 10 MPa.

The sim-sim box still starts at 0.1 GPa; vic_pose runs must be able to reach the proxy-fixture range.
EOF
)"
```

---

### Task 4: Handbook, ROADMAP, README

**Files:**
- Modify: `docs/handbook-youngs-cma.md` §5 and §7
- Modify: `docs/ROADMAP.md` Current focus checklist + M4.0 validation block
- Modify: `README.md` (after the sim-sim CMA commands)
- Modify: `docs/superpowers/specs/2026-08-14-real-multi-structure-cmaes-design.md` slice 4 status line only if the spec has a per-slice stamp; otherwise leave the spec as Approved and point at H5

**Interfaces:** none (docs). Commands must match README/ROADMAP (`uv run`).

- [ ] **Step 1: Update H5 status boundary**

In `docs/handbook-youngs-cma.md` §5, replace the sentence that CMA has no `--controller-mode` with:

- Real 1×1 `vic_pose` datasets auto-select `make_real_replay_build_env_fn`; `--controller-mode` is the explicit opt-in/override.
- `gt_candidate` is `None`; `cmaes_report.json` omits `gt_diagnostics`.
- Effective spur/stem floor is \(\log_{10} E = 7\) on real runs only.
- Multi-episode convert / per-direction weld remain ROADMAP-owned (spec slices 1–3).

Add under §7 **Real convert → CMA plumbing smoke** (after the grid command):

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_cmaes_s09_d00 \
  --viewer null \
  --overwrite
```

State that this is a plumbing/fit-loop smoke on one converted episode; ranking quality is still ROADMAP-owned.

- [ ] **Step 2: Update ROADMAP**

Check the Slice 4 box as done **for 1×1 wiring**, and keep a remaining bullet: folder convert + multi-direction replay (spec slices 1–3) before multi-tree CMA.

Replace the comment `# CMA (slice 4): ... not yet wired.` with the same `uv run` command as H5, pointed at `tmp/real_batched_s09_d00` (current local 1×1 dataset), and add the pytest file:

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  -q -p no:launch_testing
```

- [ ] **Step 3: Update README** after the sim-sim CMA block with the same real command, noting auto-detect of `vic_pose_v1`.

- [ ] **Step 4: No pytest for markdown.** Re-run CMA tests once more after docs-only edits (no code change expected).

```bash
uv run --env-file pytest.env python -m pytest \
  apple_pick_gym/tests/test_example_youngs_modulus_cmaes_cli.py \
  -q -p no:launch_testing
```

- [ ] **Step 5: Commit**

```bash
git add docs/handbook-youngs-cma.md docs/ROADMAP.md README.md
git commit -m "$(cat <<'EOF'
Document real 1x1 vic_pose CMA wiring.

Keep multi-episode convert and fused real ranking as later M4.0 work.
EOF
)"
```

---

## Out of scope (do not do in this plan)

- `--input-dir` tree convert, `n_holds` / full `sim_config` manifest fields (slice 1).
- Per-direction weld/gripper, last-action padding, replay truncate-before-score (slice 2).
- Removing the grid/CMA one-structure `vic_pose` guard (slice 3).
- Changing Sinkhorn scales, phenotype, or sim-sim `CMA_SEARCH_PARAMS`.
- A long CUDA CMA on s09 (optional smoke after Task 4; not a gate). If run, use `--viewer null`, existing `tmp/real_batched_s09_d00`, and a locally reduced `CMA_SEARCH_PARAMS["max_generations"]` / `population_size` — do not commit `tmp/`.

## Spec coverage

| Spec slice 4 requirement | Task |
| --- | --- |
| Select real builder on `dataset_declares_vic_pose` / `--controller-mode vic_pose` | 2 |
| Do not fail structures for missing sim GT; omit GT diagnostics | 2 |
| `--controller-mode` + `check_action_semantics` | 1, 2 |
| Widen real spur/stem box to \(\log_{10} E = 7\) | 3 |
| Sim-sim grid/CMA unchanged | 2 stubs + 3 keeps `CMA_SEARCH_PARAMS` |
| README / ROADMAP `uv run` commands | 4 |

---

### Task 5: Runtime CMA-ES — Sinkhorn must decrease over generations

**Files:** none committed. Local knobs may be edited then restored. Output under `tmp/real_kp_e_cmaes_s09_d00/` (gitignored).

**Interfaces:**
- Consumes: Tasks 1–4 wiring; converted dataset `tmp/real_batched_s09_d00` (rebuild from `robot_replay/new_data/s09/s09-d00.parquet` if missing)
- Produces: `tmp/real_kp_e_cmaes_s09_d00/cmaes_report.json` whose per-generation `score_summary.eligible_mean` **decreases** (last eligible gen < first eligible gen)

This is the maintainer-required acceptance, not an optional smoke.

- [ ] **Step 1: Confirm dataset**

If `tmp/real_batched_s09_d00/manifest.json` is missing, convert:

```bash
uv run python robot_replay/convert_real_to_batched_sysid_metadata.py \
  --input robot_replay/new_data/s09/s09-d00.parquet \
  --dataset-out tmp/real_batched_s09_d00 --overwrite
```

- [ ] **Step 2: Local-only shrink of search knobs (do not commit)**

In `example_youngs_modulus_cmaes.py` `CMA_SEARCH_PARAMS` only for this run:

- `population_size`: `6`
- `max_generations`: `4`

Leave bounds/mean/sigma as in the tree. Restore these two integers after the run. Do not `git add` this file for a knob-only change.

- [ ] **Step 3: Run CMA**

```bash
uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \
  --dataset tmp/real_batched_s09_d00 \
  --output tmp/real_kp_e_cmaes_s09_d00 \
  --viewer null \
  --overwrite
```

GPU expected. Fail the task if wrench-as-twist, missing GT, or crash. Do not claim success without this command's output.

- [ ] **Step 4: Assert loss decreased**

Read `structures.0.generations` (or equivalent) `score_summary.eligible_mean` in order. Record the list. Pass only if at least two finite means exist and `means[-1] < means[0]`. Also confirm no `gt_diagnostics` and `cma.search_bounds_log10.lower` includes spur/stem `7.0`.

If loss does not decrease, status `DONE_WITH_CONCERNS` with the series — do not tune the scorer or phenotype in this task.

- [ ] **Step 5: Restore `CMA_SEARCH_PARAMS` and do not commit tmp/**

```bash
git restore apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py
```

only if the only diffs are `population_size` / `max_generations`. If Task 2–3 production diffs are still uncommitted, restore just those two keys by hand.

No commit unless you also finish leftover Task 4 docs.

---

## Manual smoke (superseded by Task 5)

Task 5 is the required verification: run CMA-ES on `tmp/real_batched_s09_d00` and show eligible-mean Sinkhorn falling across generations.
