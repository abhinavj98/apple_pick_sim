"""CLI contract tests for the separate Young's-modulus CMA-ES example."""

from __future__ import annotations

import argparse
import json
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
        / "example_youngs_modulus_cmaes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "example_youngs_modulus_cmaes_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_grid_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "batched_examples"
        / "example_youngs_modulus_sys_id.py"
    )
    spec = importlib.util.spec_from_file_location(
        "example_youngs_modulus_sys_id_regression", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_ranges_dict() -> dict:
    return {
        "primary": {"youngs_modulus_pa": {"min": 1.0e7, "max": 1.0e9}},
        "spur": {"youngs_modulus_pa": {"min": 1.0e6, "max": 1.0e8}},
        "stem": {"youngs_modulus_pa": {"min": 1.0e5, "max": 1.0e7}},
    }


def _score(candidate_index: int, candidate, sinkhorn: float):
    return cmaes.YoungsModulusCandidateScore(
        candidate_index=candidate_index,
        candidate=candidate,
        aggregate_sinkhorn=float(sinkhorn),
        per_direction_sinkhorn={0: float(sinkhorn)},
        instability_fraction=0.0,
        disqualified=False,
        disqualification_reason=None,
        rank=1 if candidate_index == 0 else None,
        is_gt=False,
    )


def _evaluation(
    structure_idx: int,
    candidates,
    scores,
    *,
    direction_indices: tuple[int, ...] = (0, 1),
):
    dirs = tuple(int(d) for d in direction_indices)
    return cmaes.YoungsModulusEvaluation(
        structure_idx=int(structure_idx),
        gt_candidate=cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
        fixed_secondary_e_pa=5e7,
        direction_indices=dirs,
        scores=[
            _score(i, cand, float(scores[i])) for i, cand in enumerate(candidates)
        ],
        replay_episodes=[[{} for _ in dirs] for _ in candidates],
        applied_params=[MagicMock() for _ in candidates],
    )


def test_cma_search_params_dict_is_sole_search_truth_source():
    module = _load_module()
    params = module.CMA_SEARCH_PARAMS
    assert set(params) >= {
        "initial_mean_log10",
        "initial_sigma_log10",
        "population_size",
        "max_generations",
        "cma_seed",
    }
    # Absolute box: all roles 0.1–100 GPa [8,11]; mean at mid.
    assert params["initial_mean_log10"] == [9.5, 9.5, 9.5]
    assert params["initial_sigma_log10"] == 0.5
    assert params["population_size"] == 15
    assert params["max_generations"] == 10
    assert params["cma_seed"] == 56
    assert params["search_bounds_log10"] == {
        "lower": [8.0, 8.0, 8.0],
        "upper": [11.0, 11.0, 11.0],
    }


def test_run_reads_search_knobs_from_cma_search_params_only(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": str(ranges_path),
            "topology_seed": 42,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    fit_calls: list[dict] = []
    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        es, seed, rng = cmaes.create_structure_cma_optimizer(bounds, **kwargs)
        return es, seed, rng

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        fit_calls.append({"max_generations": max_generations})
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_e(state.final_mean_log10)],
                [0.1],
            )
            state.gt_candidate = state.final_evaluation.gt_candidate
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=tuple(sorted(states)),
            failed_structure_indices=(),
            generation_waves=1,
            final_mean_batch=None,
        )

    monkeypatch.setattr(
        module,
        "CMA_SEARCH_PARAMS",
        {
            "initial_mean_log10": "bounds_midpoint",
            "initial_sigma_log10": 0.4,
            "population_size": 5,
            "max_generations": 3,
            "cma_seed": 11,
            "search_bounds_log10": None,
        },
    )
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "create_structure_cma_optimizer", fake_create)
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
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
    assert fit_calls[0]["max_generations"] == 3
    assert create_calls
    assert create_calls[0]["initial_sigma_log10"] == 0.4
    assert create_calls[0]["population_size"] == 5
    assert create_calls[0]["base_seed"] == 11
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["cma"]["max_generations"] == 3
    assert report["cma"]["initial_sigma_log10"] == 0.4
    assert report["cma"]["base_seed"] == 11
    assert report["cma"]["population_size"] == 5
    assert report["cma"]["search_params_source"] == "CMA_SEARCH_PARAMS"


def test_run_cli_cma_seed_overrides_cma_search_params(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": str(ranges_path),
            "topology_seed": 42,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        es, seed, rng = cmaes.create_structure_cma_optimizer(bounds, **kwargs)
        return es, seed, rng

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_e(state.final_mean_log10)],
                [0.1],
            )
            state.gt_candidate = state.final_evaluation.gt_candidate
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=tuple(sorted(states)),
            failed_structure_indices=(),
            generation_waves=1,
            final_mean_batch=None,
        )

    monkeypatch.setattr(
        module,
        "CMA_SEARCH_PARAMS",
        {
            "initial_mean_log10": "bounds_midpoint",
            "initial_sigma_log10": 0.4,
            "population_size": 5,
            "max_generations": 3,
            "cma_seed": 11,
            "search_bounds_log10": None,
        },
    )
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "create_structure_cma_optimizer", fake_create)
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=2,
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
    assert create_calls[0]["base_seed"] == 2
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["cma"]["base_seed"] == 2


def test_parser_cma_defaults_and_required_args(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "/tmp/gt",
            "--output",
            "/tmp/cma",
        ]
    )
    assert not hasattr(args, "max_generations")
    assert not hasattr(args, "initial_sigma_log10")
    assert not hasattr(args, "population_size")
    assert args.cma_seed is None
    assert args.ranges is None
    assert args.multi_structure_batch is True
    assert args.fail_fast is False
    assert args.use_median is True
    assert args.hold_id_onehot is True
    assert args.pool_directions is True
    assert module.SETTLE_QUIET_EVERY == 350
    assert args.settle_quiet_every == 350

    assert parser.parse_args(
        [
            "--dataset",
            "/tmp/gt",
            "--output",
            "/tmp/cma",
            "--no-multi-structure-batch",
        ]
    ).multi_structure_batch is False
    assert (
        parser.parse_args(
            ["--dataset", "/tmp/gt", "--output", "/tmp/cma", "--cma-seed", "7"]
        ).cma_seed
        == 7
    )

    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/cma"])

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--ranges" in option_strings
    # Mean/sigma/popsize/gens live in CMA_SEARCH_PARAMS; cma_seed is CLI-overridable.
    assert "--max-generations" not in option_strings
    assert "--population-size" not in option_strings
    assert "--initial-sigma-log10" not in option_strings
    assert "--cma-seed" in option_strings
    # Grid-only controls must stay off the CMA command.
    assert "--log10-e-primary" not in option_strings
    assert "--log10-e-spur" not in option_strings
    assert "--log10-e-stem" not in option_strings
    assert "--include-gt-candidate" not in option_strings
    assert "--max-candidates" not in option_strings
    assert "--export-replays" not in option_strings
    assert "--max-overlay-candidates" not in option_strings


def test_grid_parser_still_exposes_grid_only_controls(monkeypatch):
    """Unchanged grid CLI regression: grid-only flags remain available."""
    module = _load_grid_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--log10-e-primary" in option_strings
    assert "--include-gt-candidate" in option_strings
    assert "--max-candidates" in option_strings
    assert "--export-replays" in option_strings
    assert "--max-overlay-candidates" in option_strings
    assert "--cma-seed" not in option_strings
    assert "--max-generations" not in option_strings


def test_resolve_ranges_path_prefers_cli_then_manifest(tmp_path):
    module = _load_module()
    cli_ranges = tmp_path / "cli_ranges.json"
    manifest_ranges = tmp_path / "manifest_ranges.json"
    cli_ranges.write_text("{}", encoding="utf-8")
    manifest_ranges.write_text("{}", encoding="utf-8")

    resolved = module._resolve_ranges_path(
        SimpleNamespace(ranges=str(cli_ranges)),
        {"ranges_path": str(manifest_ranges)},
        cwd=tmp_path,
    )
    assert resolved == cli_ranges.resolve()

    resolved = module._resolve_ranges_path(
        SimpleNamespace(ranges=None),
        {"ranges_path": str(manifest_ranges)},
        cwd=tmp_path,
    )
    assert resolved == manifest_ranges.resolve()


def test_resolve_ranges_path_rejects_missing_and_unrelated_default(tmp_path):
    module = _load_module()
    with pytest.raises(SystemExit, match="ranges"):
        module._resolve_ranges_path(
            SimpleNamespace(ranges=None),
            {},
            cwd=tmp_path,
        )
    with pytest.raises(SystemExit, match="ranges"):
        module._resolve_ranges_path(
            SimpleNamespace(ranges="missing_ranges.json"),
            {},
            cwd=tmp_path,
        )


def test_clear_cma_owned_artifacts_removes_report_temp_and_selected_overlays(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    report = output_dir / "cmaes_report.json"
    report.write_text("{}", encoding="utf-8")
    tmp = output_dir / ".cmaes_report.json.123.tmp"
    tmp.write_text("{}", encoding="utf-8")
    keep = output_dir / "unrelated.txt"
    keep.write_text("keep", encoding="utf-8")

    selected = output_dir / "structure_001"
    selected.mkdir()
    (selected / "youngs_modulus_overlay.html").write_text("old", encoding="utf-8")
    (selected / "other.txt").write_text("keep", encoding="utf-8")

    other = output_dir / "structure_002"
    other.mkdir()
    (other / "youngs_modulus_overlay.html").write_text("keep", encoding="utf-8")

    module._clear_cma_owned_artifacts(output_dir, structure_indices=[1])

    assert not report.exists()
    assert not tmp.exists()
    assert keep.read_text(encoding="utf-8") == "keep"
    assert not (selected / "youngs_modulus_overlay.html").exists()
    assert (selected / "other.txt").exists()
    assert (other / "youngs_modulus_overlay.html").exists()


def test_accumulate_batch_counters_uses_structures_and_exact_planned_slots():
    """Fused diagnostics must not NameError; attribute exact planned slots."""
    from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
        MultiStructureReplayDiagnostics,
    )

    module = _load_module()
    cand_a = cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6)
    cand_b = cmaes.YoungsModulusCandidate(2e8, 2e7, 2e6)
    structures = [
        (4, (cand_a, cand_b)),  # 2 candidates × 3 usable directions = 6
        (1, (cand_a,)),  # 1 × 3 = 3
    ]
    counters = {
        "replay_candidate_evaluations": 0,
        "final_mean_evaluations": 0,
        "physical_env_slots": 0,
        "scalar_retries": 0,
    }
    dirs = (0, 1, 2)
    batch = cmaes.YoungsModulusBatchEvaluation(
        evaluations={
            4: _evaluation(4, [cand_a, cand_b], [0.1, 0.2], direction_indices=dirs),
            1: _evaluation(1, [cand_a], [0.3], direction_indices=dirs),
        },
        errors={},
        replay_diagnostics=MultiStructureReplayDiagnostics(
            candidate_blocks=3,
            flattened_envs=9,
            chunk_env_counts=(9,),
            failed_chunk_indices=(),
            build_seconds=0.0,
            replay_seconds=0.0,
        ),
        retried_structures=(),
        physical_slots_by_structure={4: 6, 1: 3},
    )
    # Must not raise NameError on undefined structure_list.
    module.accumulate_cma_batch_counters(
        counters,
        batch,
        structures=structures,
        wave_kind="generation",
        num_directions=3,
    )
    assert counters["physical_env_slots"] == 9
    assert counters["physical_env_slots:4"] == 6
    assert counters["physical_env_slots:1"] == 3
    assert counters["replay_candidate_evaluations"] == 3
    assert counters["final_mean_evaluations"] == 0
    assert counters.get("replay_candidate_evaluations:4") == 2
    assert counters.get("replay_candidate_evaluations:1") == 1

    module.accumulate_cma_batch_counters(
        counters,
        batch,
        structures=[(4, (cand_a,))],
        wave_kind="final_mean",
        num_directions=3,
    )
    assert counters["final_mean_evaluations"] == 1
    assert counters["final_mean_evaluations:4"] == 1


def test_accumulate_batch_counters_uses_actual_usable_dirs_not_cli_num_directions():
    """Per-structure slots must follow usable directions, not manifest width."""
    from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
        MultiStructureReplayDiagnostics,
    )

    module = _load_module()
    cand_a = cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6)
    cand_b = cmaes.YoungsModulusCandidate(2e8, 2e7, 2e6)
    # Manifest/CLI width is 5, but exclusions leave 2 and 1 usable directions.
    structures = [
        (4, (cand_a, cand_b)),  # 2 × 2 usable = 4
        (1, (cand_a,)),  # 1 × 1 usable = 1
    ]
    counters = {
        "replay_candidate_evaluations": 0,
        "final_mean_evaluations": 0,
        "physical_env_slots": 0,
        "scalar_retries": 0,
    }
    batch = cmaes.YoungsModulusBatchEvaluation(
        evaluations={
            4: _evaluation(
                4, [cand_a, cand_b], [0.1, 0.2], direction_indices=(0, 2)
            ),
            1: _evaluation(1, [cand_a], [0.3], direction_indices=(1,)),
        },
        errors={},
        replay_diagnostics=MultiStructureReplayDiagnostics(
            candidate_blocks=3,
            flattened_envs=5,
            chunk_env_counts=(5,),
            failed_chunk_indices=(),
            build_seconds=0.0,
            replay_seconds=0.0,
        ),
        retried_structures=(),
    )
    module.accumulate_cma_batch_counters(
        counters,
        batch,
        structures=structures,
        wave_kind="generation",
        num_directions=5,
    )
    assert counters["physical_env_slots"] == 5
    assert counters["physical_env_slots:4"] == 4
    assert counters["physical_env_slots:1"] == 1
    assert (
        counters["physical_env_slots:4"] + counters["physical_env_slots:1"]
        == counters["physical_env_slots"]
    )


def test_accumulate_batch_counters_attributes_scoring_failed_replayed_structure():
    """Fused-replayed structures that fail scoring still own their physical slots."""
    from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
        MultiStructureReplayDiagnostics,
    )

    module = _load_module()
    cand_a = cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6)
    cand_b = cmaes.YoungsModulusCandidate(2e8, 2e7, 2e6)
    structures = [
        (4, (cand_a, cand_b)),  # 2 × 2 = 4
        (1, (cand_a,)),  # 1 × 1 = 1 (replayed, scoring failed)
    ]
    counters = {
        "replay_candidate_evaluations": 0,
        "final_mean_evaluations": 0,
        "physical_env_slots": 0,
        "scalar_retries": 0,
    }
    batch = cmaes.YoungsModulusBatchEvaluation(
        evaluations={
            4: _evaluation(
                4, [cand_a, cand_b], [0.1, 0.2], direction_indices=(0, 2)
            ),
        },
        errors={1: "scoring failed"},
        replay_diagnostics=MultiStructureReplayDiagnostics(
            candidate_blocks=3,
            flattened_envs=5,
            chunk_env_counts=(5,),
            failed_chunk_indices=(),
            build_seconds=0.0,
            replay_seconds=0.0,
        ),
        retried_structures=(),
        physical_slots_by_structure={4: 4, 1: 1},
    )
    module.accumulate_cma_batch_counters(
        counters,
        batch,
        structures=structures,
        wave_kind="generation",
        num_directions=5,
    )
    assert counters["physical_env_slots"] == 5
    assert counters["physical_env_slots:4"] == 4
    assert counters["physical_env_slots:1"] == 1


def test_accumulate_batch_counters_counts_scalar_only_wave_from_evaluations():
    """Scalar-only waves (no fused diagnostics) still count exact physical slots."""
    module = _load_module()
    cand_a = cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6)
    cand_b = cmaes.YoungsModulusCandidate(2e8, 2e7, 2e6)
    structures = [
        (4, (cand_a, cand_b)),  # 2 × 2 = 4
        (1, (cand_a,)),  # 1 × 1 = 1
    ]
    counters = {
        "replay_candidate_evaluations": 0,
        "final_mean_evaluations": 0,
        "physical_env_slots": 0,
        "scalar_retries": 0,
    }
    batch = cmaes.YoungsModulusBatchEvaluation(
        evaluations={
            4: _evaluation(
                4, [cand_a, cand_b], [0.1, 0.2], direction_indices=(0, 2)
            ),
            1: _evaluation(1, [cand_a], [0.3], direction_indices=(1,)),
        },
        errors={},
        replay_diagnostics=None,
        retried_structures=(),
        physical_slots_by_structure={4: 4, 1: 1},
    )
    module.accumulate_cma_batch_counters(
        counters,
        batch,
        structures=structures,
        wave_kind="generation",
        num_directions=5,
    )
    assert counters["physical_env_slots"] == 5
    assert counters["physical_env_slots:4"] == 4
    assert counters["physical_env_slots:1"] == 1


def test_run_viewer_cancel_checkpoints_cancelled_and_exits_nonzero(
    monkeypatch, tmp_path
):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": str(ranges_path),
            "topology_seed": 42,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        raise module.ViewerCancelled("viewer closed; cancelling CMA-ES fit")

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
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
    assert result["exit_nonzero"] is True
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["command_status"] == "cancelled"
    assert "viewer closed" in str(report.get("command_error", ""))


def test_write_cmaes_report_atomic_is_strict_and_includes_extrema_covariance(tmp_path):
    module = _load_module()
    path = tmp_path / "cmaes_report.json"
    bounds = cmaes.extract_youngs_modulus_cma_bounds(_valid_ranges_dict())
    es, seed, _ = cmaes.create_structure_cma_optimizer(
        bounds, base_seed=0, structure_idx=0, population_size=4
    )
    state = cmaes.StructureCmaState(
        structure_idx=0,
        optimizer=es,
        bounds=bounds,
        effective_seed=seed,
        population_size=4,
        status="active",
    )
    payload = module._build_cmaes_report_payload(
        states={0: state},
        dataset="/tmp/gt",
        output=str(tmp_path),
        ranges_path="/tmp/ranges.json",
        base_seed=0,
        initial_mean_log10=bounds.log10_midpoint,
        initial_sigma_log10=1.0,
        max_generations=10,
        scoring=cmaes.YoungsModulusScoringConfig(),
        command_status="running",
        population_size=4,
    )
    module._write_cmaes_report_atomic(path, payload)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    structure = loaded["structures"]["0"]
    assert "evaluated_history_extrema" in structure
    assert structure["evaluated_history_extrema"]["min_log10_e"] is None
    assert "covariance" in structure
    assert structure["covariance"] is not None
    assert "effective_unbounded_covariance" in structure["covariance"]
    json.dumps(loaded, allow_nan=False)

    with pytest.raises(ValueError):
        module._write_cmaes_report_atomic(
            path,
            {"bad": float("nan")},
        )


def test_run_writes_initial_report_before_fit_and_progress_updates(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": str(ranges_path),
            "topology_seed": 42,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    write_times: list[str] = []
    original_write = module._write_cmaes_report_atomic

    def tracking_write(path, payload):
        write_times.append(str(payload.get("command_status")))
        return original_write(path, payload)

    fit_calls: list[dict] = []

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        fit_calls.append(
            {
                "structure_indices": sorted(states),
                "max_generations": max_generations,
            }
        )
        assert (output_dir / "cmaes_report.json").is_file()
        if on_progress is not None:
            on_progress(states)
            for state in states.values():
                state.status = "fitted"
                state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
                state.final_evaluation = _evaluation(
                    state.structure_idx,
                    [cmaes.candidates_from_log10_e(state.final_mean_log10)],
                    [0.5],
                )
                state.gt_candidate = state.final_evaluation.gt_candidate
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=tuple(sorted(states)),
            failed_structure_indices=(),
            generation_waves=1,
            final_mean_batch=None,
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(
        module,
        "CMA_SEARCH_PARAMS",
        {
            "initial_mean_log10": "bounds_midpoint",
            "initial_sigma_log10": 1.0,
            "population_size": 4,
            "max_generations": 3,
            "cma_seed": 0,
            "search_bounds_log10": None,
        },
    )
    monkeypatch.setattr(
        module,
        "build_sim_config",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(module, "_write_cmaes_report_atomic", tracking_write)
    monkeypatch.setattr(
        module,
        "write_youngs_modulus_overlay_html",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "overlay_episodes_from_replay_evaluation",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
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
    assert fit_calls and fit_calls[0]["structure_indices"] == [0, 1]
    assert fit_calls[0]["max_generations"] == 3
    assert "running" in write_times
    assert result["exit_nonzero"] is False
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["aggregate"]["fitted_structures"] == 2
    for key in ("0", "1"):
        structure = report["structures"][key]
        assert "evaluated_history_extrema" in structure
        assert "covariance" in structure


def test_run_continues_after_structure_failure_unless_fail_fast(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 1,
            "ranges_path": str(ranges_path),
            "topology_seed": 1,
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        states[0].status = "failed"
        states[0].failure = cmaes.CmaGenerationFailure("prepare", "boom")
        states[1].status = "fitted"
        states[1].final_mean_log10 = tuple(states[1].bounds.log10_midpoint)
        states[1].final_evaluation = _evaluation(
            1,
            [cmaes.candidates_from_log10_e(states[1].final_mean_log10)],
            [0.2],
        )
        states[1].gt_candidate = states[1].final_evaluation.gt_candidate
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(1,),
            failed_structure_indices=(0,),
            generation_waves=1,
            final_mean_batch=None,
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "write_youngs_modulus_overlay_html",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "overlay_episodes_from_replay_evaluation",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
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
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["structures"]["0"]["status"] == "failed"
    assert report["structures"]["1"]["status"] == "fitted"
    assert report["aggregate"]["fitted_structures"] == 1
    assert report["aggregate"]["failed_structures"] == 1


def test_run_all_failed_sets_exit_nonzero(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 1,
            "ranges_path": str(ranges_path),
            "topology_seed": 1,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        states[0].status = "failed"
        states[0].failure = cmaes.CmaGenerationFailure("all_invalid", "none valid")
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(),
            failed_structure_indices=(0,),
            generation_waves=1,
            final_mean_batch=None,
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
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
    assert result["exit_nonzero"] is True


def test_run_records_overlay_error_without_invalidating_fitted(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 1,
            "ranges_path": str(ranges_path),
            "topology_seed": 1,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        state = states[0]
        state.status = "fitted"
        state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
        state.final_evaluation = _evaluation(
            0,
            [cmaes.candidates_from_log10_e(state.final_mean_log10)],
            [0.3],
        )
        state.gt_candidate = state.final_evaluation.gt_candidate
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
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "write_youngs_modulus_overlay_html",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("overlay boom")),
    )
    monkeypatch.setattr(
        module,
        "overlay_episodes_from_replay_evaluation",
        lambda *_a, **_k: [MagicMock()],
    )
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
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
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["structures"]["0"]["status"] == "fitted"
    assert any("overlay boom" in err for err in report["structures"]["0"]["artifact_errors"])


def test_run_rejects_nonempty_output_without_overwrite(tmp_path):
    module = _load_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        overwrite=False,
    )
    with pytest.raises(SystemExit, match="overwrite"):
        module._run(args, argparse.ArgumentParser(), viewer=MagicMock())


def test_run_fail_fast_aborts_on_global_evaluator_error(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 1,
            "ranges_path": str(ranges_path),
            "topology_seed": 1,
        }
    }
    dataset.structure_summaries.return_value = [{}]

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None):
        raise RuntimeError("batch exploded")

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_youngs_modulus_candidate_from_structure",
        lambda *_a, **_k: cmaes.YoungsModulusCandidate(1e8, 1e7, 1e6),
    )

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        include_excluded=False,
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        multi_structure_batch=True,
        fail_fast=True,
        overwrite=True,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=False,
        settle_quiet_every=None,
        show_pull_direction=False,
        viewer="null",
    )
    result = module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert result["exit_nonzero"] is True
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["command_status"] in {"global_error", "failed"}
    assert "batch exploded" in str(report.get("command_error", ""))
