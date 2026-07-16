"""CLI contract tests for the dataset-driven Young's-modulus sys-id example."""

from __future__ import annotations

import argparse
import importlib.util
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
