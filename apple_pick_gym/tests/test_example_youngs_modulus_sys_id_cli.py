"""CLI contract tests for the dataset-driven Young's-modulus sys-id example."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "batched_examples"
        / "example_youngs_modulus_sys_id.py"
    )
    spec = importlib.util.spec_from_file_location(
        "example_youngs_modulus_sys_id_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_and_required_args(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/gt",
            "--output",
            "/tmp/rank",
        ]
    )
    assert args.include_gt_candidate is True
    assert args.use_median is True
    assert args.hold_id_onehot is True
    assert args.pool_directions is True
    assert args.export_replays is False
    assert args.log10_e_primary == "8.0,8.5"
    assert args.log10_e_spur == "7.5"
    assert args.log10_e_stem == "7.0"
    assert args.max_candidates == 0
    assert args.max_overlay_candidates == 8
    assert args.fail_fast is False

    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/rank"])

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--topology-seed" not in option_strings
    assert "--num-structures" not in option_strings
    assert "--num-directions" not in option_strings


def test_resolve_structure_indices_defaults_to_all_structures():
    module = _load_module()
    dataset = MagicMock()
    dataset.structure_summaries.return_value = [{}, {}, {}]

    assert module._resolve_structure_indices(dataset, None) == [0, 1, 2]
    assert module._resolve_structure_indices(dataset, (1,)) == [1]


def test_resolve_n_holds_and_directions_from_manifest():
    module = _load_module()
    dataset = MagicMock()
    dataset.episode_entries.return_value = []

    collection = {
        "n_holds": 5,
        "num_directions": 4,
    }
    assert module._resolve_n_holds(dataset, collection) == 5
    assert module._resolve_n_directions(dataset, collection) == 4


def test_resolve_n_directions_falls_back_to_episode_indices():
    module = _load_module()
    dataset = MagicMock()
    dataset.episode_entries.return_value = [
        {"direction_idx": 0},
        {"direction_idx": 2},
    ]

    assert module._resolve_n_directions(dataset, {}) == 3


def test_run_iterates_structures_and_invokes_evaluator(monkeypatch):
    module = _load_module()

    gt = cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7)
    grid_candidate = cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)
    evaluation = cmaes.YoungsModulusEvaluation(
        structure_idx=0,
        gt_candidate=gt,
        fixed_secondary_e_pa=5.0e7,
        direction_indices=(0, 1),
        scores=[],
        replay_episodes=[],
        applied_params=[],
    )

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 4,
            "ranges_path": "/tmp/ranges.json",
            "topology_seed": 42,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [grid_candidate],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda _dataset, structure_idx: cmaes.YoungsModulusCandidate(
            1.0e8 + float(structure_idx),
            10**7.5,
            1.0e7,
        ),
    )

    include_gt_calls: list[bool] = []

    def fake_maybe_include(candidates, gt_candidate, *, include_gt):
        include_gt_calls.append(bool(include_gt))
        if include_gt:
            return [*candidates, gt_candidate]
        return list(candidates)

    monkeypatch.setattr(module, "maybe_include_gt_candidate", fake_maybe_include)

    evaluator_calls: list[dict] = []

    def fake_evaluate(**kwargs):
        evaluator_calls.append(dict(kwargs))
        return evaluation

    monkeypatch.setattr(module, "evaluate_youngs_modulus_candidates", fake_evaluate)
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output="/tmp/rank",
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=True,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=False,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )
    parser = argparse.ArgumentParser()
    viewer = MagicMock()

    result = module._run(args, parser, viewer=viewer)

    assert len(evaluator_calls) == 2
    assert evaluator_calls[0]["structure_idx"] == 0
    assert evaluator_calls[1]["structure_idx"] == 1
    assert evaluator_calls[0]["num_directions"] == 4
    assert evaluator_calls[1]["num_directions"] == 4
    assert len(evaluator_calls[0]["candidates"]) == 2
    assert include_gt_calls == [True, True]
    assert result["structure_indices"] == [0, 1]
    assert len(result["structure_results"]) == 2
    assert result["structure_results"][0]["evaluation"] is evaluation
    assert result["structure_results"][0]["error"] is None


def test_run_without_gt_candidate_skips_insertion(monkeypatch):
    module = _load_module()

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": "/tmp/ranges.json",
            "topology_seed": 42,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    grid_candidate = cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [grid_candidate],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7),
    )
    monkeypatch.setattr(
        module,
        "maybe_include_gt_candidate",
        lambda candidates, _gt, *, include_gt: list(candidates),
    )

    evaluator_calls: list[dict] = []

    def fake_evaluate(**kwargs):
        evaluator_calls.append(dict(kwargs))
        return cmaes.YoungsModulusEvaluation(
            structure_idx=0,
            gt_candidate=grid_candidate,
            fixed_secondary_e_pa=None,
            direction_indices=(0,),
            scores=[],
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(module, "evaluate_youngs_modulus_candidates", fake_evaluate)
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output="/tmp/rank",
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=False,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=False,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )

    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())

    assert len(evaluator_calls[0]["candidates"]) == 1


def test_run_records_structure_errors_unless_fail_fast(monkeypatch):
    module = _load_module()

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": "/tmp/ranges.json",
            "topology_seed": 42,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7),
    )
    monkeypatch.setattr(
        module,
        "maybe_include_gt_candidate",
        lambda candidates, _gt, *, include_gt: list(candidates),
    )

    def fake_evaluate(**kwargs):
        if int(kwargs["structure_idx"]) == 0:
            raise ValueError("structure 0 failed")
        return cmaes.YoungsModulusEvaluation(
            structure_idx=1,
            gt_candidate=cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7),
            fixed_secondary_e_pa=None,
            direction_indices=(0,),
            scores=[],
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(module, "evaluate_youngs_modulus_candidates", fake_evaluate)
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )

    base_args = dict(
        dataset="/tmp/gt",
        output="/tmp/rank",
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=False,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )

    result = module._run(
        SimpleNamespace(**base_args, fail_fast=False),
        argparse.ArgumentParser(),
        viewer=MagicMock(),
    )
    assert result["structure_results"][0]["error"] == "structure 0 failed"
    assert result["structure_results"][1]["evaluation"] is not None

    with pytest.raises(ValueError, match="structure 0 failed"):
        module._run(
            SimpleNamespace(**base_args, fail_fast=True),
            argparse.ArgumentParser(),
            viewer=MagicMock(),
        )


def test_run_rejects_nonempty_output_without_overwrite(tmp_path):
    module = _load_module()

    output_dir = tmp_path / "rank"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("x")

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        overwrite=False,
        device="cpu",
    )

    with pytest.raises(SystemExit, match="non-empty"):
        module._run(args, argparse.ArgumentParser(), viewer=MagicMock())


def test_main_exits_nonzero_when_all_structures_fail(monkeypatch):
    module = _load_module()
    import newton.examples

    viewer = MagicMock()

    def fake_init(parser=None):
        return viewer, SimpleNamespace(
            dataset="/tmp/gt",
            output="/tmp/rank",
            device="cpu",
            viewer="null",
        )

    def fake_run(_args, _parser, *, viewer):
        return {
            "structure_results": [
                {"structure_idx": 0, "evaluation": None, "error": "structure 0 failed"},
                {"structure_idx": 1, "evaluation": None, "error": "structure 1 failed"},
            ],
        }

    monkeypatch.setattr(newton.examples, "init", fake_init)
    monkeypatch.setattr(module, "_make_parser", lambda: argparse.ArgumentParser())
    monkeypatch.setattr(module, "_run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    viewer.close.assert_called_once()


def test_run_propagates_manifest_collection_metadata(monkeypatch):
    module = _load_module()

    manifest_ranges = "/manifest/ranges.json"
    manifest_topology_seed = 99
    manifest_control_hz = 45.0
    manifest_seed = 123

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": manifest_control_hz,
            "num_directions": 2,
            "ranges_path": manifest_ranges,
            "topology_seed": manifest_topology_seed,
            "seed": manifest_seed,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    grid_candidate = cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)
    captured_build_env_fn = MagicMock()
    build_env_fn_kwargs: list[dict] = []
    evaluator_calls: list[dict] = []

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [grid_candidate],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7),
    )
    monkeypatch.setattr(
        module,
        "maybe_include_gt_candidate",
        lambda candidates, _gt, *, include_gt: list(candidates),
    )

    def fake_make_build_env_fn(**kwargs):
        build_env_fn_kwargs.append(dict(kwargs))
        return captured_build_env_fn

    def fake_evaluate(**kwargs):
        evaluator_calls.append(dict(kwargs))
        return cmaes.YoungsModulusEvaluation(
            structure_idx=0,
            gt_candidate=grid_candidate,
            fixed_secondary_e_pa=None,
            direction_indices=(0,),
            scores=[],
            replay_episodes=[],
            applied_params=[],
        )

    monkeypatch.setattr(module, "_make_build_env_fn", fake_make_build_env_fn)
    monkeypatch.setattr(module, "evaluate_youngs_modulus_candidates", fake_evaluate)
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output="/tmp/rank",
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=False,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=False,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )

    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())

    assert len(build_env_fn_kwargs) == 1
    assert build_env_fn_kwargs[0]["ranges_path"] == manifest_ranges
    assert build_env_fn_kwargs[0]["topology_seed"] == manifest_topology_seed
    assert build_env_fn_kwargs[0]["control_hz"] == manifest_control_hz
    assert len(evaluator_calls) == 1
    assert evaluator_calls[0]["seed"] == manifest_seed
    assert evaluator_calls[0]["build_env_fn"] is captured_build_env_fn


def test_candidates_for_structure_enforces_max_candidates(monkeypatch):
    module = _load_module()

    grid_candidates = [
        cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7),
        cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7),
    ]
    gt = cmaes.YoungsModulusCandidate(3.0e8, 10**7.5, 1.0e7)

    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: list(grid_candidates),
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: gt,
    )

    args = SimpleNamespace(
        log10_e_primary="8.0,8.5",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=True,
        max_candidates=2,
    )
    parser = argparse.ArgumentParser()

    with pytest.raises(SystemExit):
        module._candidates_for_structure(
            MagicMock(),
            args,
            structure_idx=0,
            parser=parser,
        )


def _evaluation_with_scores(*, structure_idx: int = 0):
    from apple_pick_sim.fruiting_system import params as fs
    from apple_pick_sim.tests.conftest import RANGES_FIXTURE

    base = fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=0)
    gt = cmaes.YoungsModulusCandidate(1.0e8, 10**7.5, 1.0e7)
    winner = cmaes.YoungsModulusCandidate(1.2e8, 10**7.5, 1.0e7)
    scores = [
        cmaes.YoungsModulusCandidateScore(
            candidate_index=0,
            candidate=winner,
            aggregate_sinkhorn=0.125,
            per_direction_sinkhorn={0: 0.125},
            instability_fraction=0.0,
            disqualified=False,
            disqualification_reason=None,
            rank=1,
            is_gt=False,
        ),
        cmaes.YoungsModulusCandidateScore(
            candidate_index=1,
            candidate=gt,
            aggregate_sinkhorn=0.2,
            per_direction_sinkhorn={0: 0.2},
            instability_fraction=0.0,
            disqualified=False,
            disqualification_reason=None,
            rank=2,
            is_gt=True,
        ),
    ]
    return cmaes.YoungsModulusEvaluation(
        structure_idx=int(structure_idx),
        gt_candidate=gt,
        fixed_secondary_e_pa=5.0e7,
        direction_indices=(0, 2),
        scores=scores,
        replay_episodes=[[], []],
        applied_params=[winner.apply_to(base), gt.apply_to(base)],
    )


def _task5_run_args(module, output, *, export_replays: bool) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output),
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=True,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=bool(export_replays),
        max_overlay_candidates=8,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )


def _configure_task5_run(monkeypatch, module, evaluation, *, num_structures: int = 1):
    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": "/tmp/ranges.json",
            "topology_seed": 42,
        }
    }
    dataset.structure_summaries.return_value = [{} for _ in range(num_structures)]
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: evaluation.gt_candidate,
    )
    monkeypatch.setattr(
        module,
        "maybe_include_gt_candidate",
        lambda candidates, _gt, *, include_gt: list(candidates),
    )
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "write_youngs_modulus_overlay_html",
        lambda *_args, **_kwargs: Path("/tmp/overlay.html"),
    )
    return dataset


def test_structure_result_to_json_schema():
    module = _load_module()
    evaluation = _evaluation_with_scores()

    row = module._structure_result_to_json(evaluation)

    assert row["structure_idx"] == 0
    assert row["fixed_secondary_e_pa"] == pytest.approx(5.0e7)
    assert row["direction_indices"] == [0, 2]
    assert row["gt_log10_e"] == pytest.approx([8.0, 7.5, 7.0])
    assert row["gt_rank"] == 2
    assert row["winner"]["candidate_index"] == 0
    assert row["winner"]["log10_error"]["primary"] == pytest.approx(
        math.log10(1.2e8) - math.log10(1.0e8)
    )
    assert row["winner"]["relative_error"]["primary"] == pytest.approx(0.2)

    gt_row = next(c for c in row["candidates"] if c["is_gt"])
    assert gt_row == {
        "candidate_index": 1,
        "youngs_modulus_pa": {
            "primary": 1e8,
            "spur": 10**7.5,
            "stem": 1e7,
        },
        "log10_e": [8.0, 7.5, 7.0],
        "aggregate_sinkhorn": 0.2,
        "rank": 2,
        "is_gt": True,
        "instability_fraction": 0.0,
        "disqualified": False,
        "disqualification_reason": None,
    }


def test_winner_summary_uses_authoritative_rank_one():
    module = _load_module()
    evaluation = _evaluation_with_scores()
    evaluation.scores[0] = dataclasses.replace(
        evaluation.scores[0],
        aggregate_sinkhorn=0.5,
        rank=1,
    )
    evaluation.scores[1] = dataclasses.replace(
        evaluation.scores[1],
        aggregate_sinkhorn=0.1,
        rank=2,
    )

    assert module._winner_summary(evaluation)["candidate_index"] == 0


def test_structure_result_serializes_non_finite_floats_as_null():
    module = _load_module()
    evaluation = _evaluation_with_scores()
    non_finite = cmaes.YoungsModulusCandidate(float("nan"), float("inf"), 1.0e7)
    evaluation.gt_candidate = non_finite
    evaluation.scores[0] = dataclasses.replace(
        evaluation.scores[0],
        candidate=non_finite,
        aggregate_sinkhorn=float("nan"),
        instability_fraction=float("inf"),
        rank=1,
    )

    row = module._structure_result_to_json(evaluation)
    encoded = json.dumps(row, allow_nan=False)

    assert encoded
    assert row["gt_youngs_modulus_pa"]["primary"] is None
    assert row["gt_youngs_modulus_pa"]["spur"] is None
    assert row["gt_log10_e"][:2] == [None, None]
    assert row["candidates"][0]["youngs_modulus_pa"]["primary"] is None
    assert row["candidates"][0]["log10_e"][:2] == [None, None]
    assert row["winner"]["log10_error"]["primary"] is None
    assert row["winner"]["relative_error"]["primary"] is None


def test_aggregate_ranking_report_summaries_and_skips():
    module = _load_module()
    evaluation = _evaluation_with_scores(structure_idx=0)
    structure_rows = [module._structure_result_to_json(evaluation)]
    errors = [{"structure_idx": 1, "error": "structure 1 failed"}]

    report = module._aggregate_ranking_report(
        structure_rows,
        errors,
        dataset="/tmp/gt",
        output="/tmp/rank",
        scoring=cmaes.YoungsModulusScoringConfig(use_median=True),
    )

    assert report["dataset"] == "/tmp/gt"
    assert report["output"] == "/tmp/rank"
    assert len(report["structures"]) == 1
    assert report["skipped_structures"] == errors
    assert report["aggregate"]["n_evaluated"] == 1
    assert report["aggregate"]["n_skipped"] == 1
    assert report["aggregate"]["gt_rank_histogram"]["2"] == 1


def test_run_writes_ranking_json_and_calls_export_with_direction_indices(
    monkeypatch, tmp_path
):
    module = _load_module()
    evaluation = _evaluation_with_scores()

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": "/tmp/ranges.json",
            "topology_seed": 42,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: evaluation.gt_candidate,
    )
    monkeypatch.setattr(
        module,
        "maybe_include_gt_candidate",
        lambda candidates, _gt, *, include_gt: list(candidates),
    )
    monkeypatch.setattr(
        module,
        "evaluate_youngs_modulus_candidates",
        lambda **_kwargs: evaluation,
    )
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "write_youngs_modulus_overlay_html",
        lambda *_args, **_kwargs: tmp_path / "overlay.html",
    )

    export_calls: list[dict] = []

    def fake_export(*_args, **kwargs):
        export_calls.append(dict(kwargs))
        return 1

    monkeypatch.setattr(module, "export_replay_candidates_for_structure", fake_export)

    output_dir = tmp_path / "rank"
    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=True,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=True,
        max_overlay_candidates=8,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )

    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())

    ranking_path = output_dir / "ranking.json"
    assert ranking_path.is_file()
    report = __import__("json").loads(ranking_path.read_text(encoding="utf-8"))
    assert report["structures"]
    assert report["aggregate"]["n_evaluated"] == 1
    assert len(export_calls) == 1
    export_call = export_calls[0]
    assert export_call["source_direction_indices"] == evaluation.direction_indices
    specs_and_replays = export_call["specs_and_replays"]
    assert len(specs_and_replays) == len(evaluation.scores)
    for candidate_index, (spec, replays) in enumerate(specs_and_replays):
        score = evaluation.scores[candidate_index]
        assert spec.candidate_index == candidate_index
        assert spec.params is evaluation.applied_params[candidate_index]
        assert spec.stiffnesses == {
            "primary_e_pa": score.candidate.primary,
            "spur_e_pa": score.candidate.spur,
            "stem_e_pa": score.candidate.stem,
        }
        assert replays is evaluation.replay_episodes[candidate_index]


def test_run_records_overlay_error_without_discarding_ranking(monkeypatch, tmp_path):
    module = _load_module()
    evaluation = _evaluation_with_scores()

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": "/tmp/ranges.json",
            "topology_seed": 42,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_cli",
        lambda **_kwargs: [cmaes.YoungsModulusCandidate(2.0e8, 10**7.5, 1.0e7)],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_args, **_kwargs: evaluation.gt_candidate,
    )
    monkeypatch.setattr(
        module,
        "maybe_include_gt_candidate",
        lambda candidates, _gt, *, include_gt: list(candidates),
    )
    monkeypatch.setattr(
        module,
        "evaluate_youngs_modulus_candidates",
        lambda **_kwargs: evaluation,
    )
    monkeypatch.setattr(module, "load_ranges", lambda _path: {})
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("overlay failed")

    monkeypatch.setattr(module, "write_youngs_modulus_overlay_html", boom)

    output_dir = tmp_path / "rank"
    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        log10_e_primary="8.0",
        log10_e_spur="7.5",
        log10_e_stem="7.0",
        include_gt_candidate=True,
        max_candidates=0,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        export_replays=False,
        max_overlay_candidates=8,
        fail_fast=False,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
        settle_quiet_every=module.SETTLE_QUIET_EVERY,
        show_pull_direction=False,
        viewer="null",
    )

    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())

    report = __import__("json").loads(
        (output_dir / "ranking.json").read_text(encoding="utf-8")
    )
    assert report["structures"][0]["overlay_error"] == "overlay failed"


def test_run_records_export_error_without_discarding_ranking(monkeypatch, tmp_path):
    module = _load_module()
    evaluation = _evaluation_with_scores()
    _configure_task5_run(monkeypatch, module, evaluation)
    monkeypatch.setattr(
        module,
        "evaluate_youngs_modulus_candidates",
        lambda **_kwargs: evaluation,
    )

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(module, "export_replay_candidates_for_structure", fail_export)
    output_dir = tmp_path / "rank"

    module._run(
        _task5_run_args(module, output_dir, export_replays=True),
        argparse.ArgumentParser(),
        viewer=MagicMock(),
    )

    report = json.loads((output_dir / "ranking.json").read_text(encoding="utf-8"))
    assert report["structures"][0]["export_error"] == "export failed"


def test_run_all_fail_writes_authoritative_skipped_report(monkeypatch, tmp_path):
    module = _load_module()
    evaluation = _evaluation_with_scores()
    _configure_task5_run(monkeypatch, module, evaluation, num_structures=2)

    def fail_evaluation(**kwargs):
        raise RuntimeError(f"structure {kwargs['structure_idx']} failed")

    monkeypatch.setattr(module, "evaluate_youngs_modulus_candidates", fail_evaluation)
    output_dir = tmp_path / "rank"

    result = module._run(
        _task5_run_args(module, output_dir, export_replays=False),
        argparse.ArgumentParser(),
        viewer=MagicMock(),
    )

    report = json.loads((output_dir / "ranking.json").read_text(encoding="utf-8"))
    assert report["structures"] == []
    assert report["skipped_structures"] == [
        {"structure_idx": 0, "error": "structure 0 failed"},
        {"structure_idx": 1, "error": "structure 1 failed"},
    ]
    assert result["ranking"] == report
