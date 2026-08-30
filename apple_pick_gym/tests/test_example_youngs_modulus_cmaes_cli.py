"""CLI contract tests for the separate Young's-modulus CMA-ES example."""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs import cma_wave_evaluation as cma_wave_evaluation_mod


def _patch_execute_cma_wave_evaluation(
    monkeypatch,
    module,
    *,
    dataset,
    fake_evaluate_structures,
    ranges_dict: dict | None = None,
    build_env_fn=None,
    replay_sim_config=None,
) -> None:
    """Route wave eval through CLI test doubles (module + source bindings)."""

    def fake_execute(spec):
        kwargs = {
            "dataset": dataset,
            "structures": list(spec.structures),
            "num_directions": spec.num_directions,
            "direction_indices": spec.direction_indices,
            "max_envs_per_batch": spec.max_envs_per_batch,
            "seed": spec.seed,
            "include_excluded": spec.include_excluded,
            "fail_fast": spec.fail_fast,
            "action_dim": spec.action_dim,
            "scoring": spec.scoring,
        }
        if build_env_fn is not None:
            kwargs["build_env_fn"] = build_env_fn
            kwargs["replay_sim_config"] = replay_sim_config
        elif ranges_dict is not None:
            built_env_fn, built_replay_sim_config = (
                cma_wave_evaluation_mod.build_cma_replay_artifacts(
                    spec.replay_context,
                    ranges=ranges_dict,
                )
            )
            kwargs["build_env_fn"] = built_env_fn
            kwargs["replay_sim_config"] = built_replay_sim_config
        return fake_evaluate_structures(**kwargs)

    def fake_spawn(spec, **kwargs):
        return fake_execute(spec)

    monkeypatch.setattr(
        cma_wave_evaluation_mod,
        "execute_cma_wave_evaluation",
        fake_execute,
    )
    monkeypatch.setattr(
        cma_wave_evaluation_mod,
        "spawn_isolated_cma_wave_evaluation",
        fake_spawn,
    )
    monkeypatch.setattr(module, "execute_cma_wave_evaluation", fake_execute)
    monkeypatch.setattr(module, "spawn_isolated_cma_wave_evaluation", fake_spawn)


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


def _sim_episode_meta() -> dict:
    return {"action_dim": 6, "action_compatible_with_vic_twist": True}


def _attach_sim_episode_meta(dataset: MagicMock) -> MagicMock:
    dataset.load_episode_metadata.return_value = _sim_episode_meta()
    return dataset


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
    replay_episodes=None,
):
    dirs = tuple(int(d) for d in direction_indices)
    if replay_episodes is None:
        replay_episodes = [[{} for _ in dirs] for _ in candidates]
    return cmaes.YoungsModulusEvaluation(
        structure_idx=int(structure_idx),
        gt_candidate=cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
        fixed_secondary_e_pa=5e7,
        direction_indices=dirs,
        scores=[
            _score(i, cand, float(scores[i])) for i, cand in enumerate(candidates)
        ],
        replay_episodes=replay_episodes,
        applied_params=[MagicMock() for _ in candidates],
    )


def _synthetic_recorded_episode(*, direction: int) -> dict:
    n = 9
    phase = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    ft = np.zeros((n, 6), dtype=np.float32)
    ft[:, 2] = 2.0
    ft[:, 4] = 0.2
    tcp = np.zeros((n, 3), dtype=np.float32)
    tcp[0, 2] = 0.0
    hold_idx = np.where(phase == 1)[0]
    for j, idx in enumerate(hold_idx):
        tcp[idx, 2] = 1.0 + 0.05 * j
    return {
        "action": np.zeros((n, 6), dtype=np.float32),
        "phase": phase,
        "dir_idx": np.full(n, int(direction), dtype=np.int32),
        "ft_wrist": ft,
        "ft_wrist_lpf": ft.copy(),
        "tcp_pos": tcp,
        "apple_pos": tcp + 0.1,
        "excitation_direction": np.tile(
            np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
        ),
    }


def _candidate_log10(candidate) -> tuple[float, float, float]:
    import math

    return (
        math.log10(float(candidate.support_kp)),
        math.log10(float(candidate.spur)),
        math.log10(float(candidate.stem)),
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
        "max_sigma_log10",
    }
    # Vector is (log10 support_kp, log10 E_spur, log10 E_stem). Support k_p
    # uses an absolute safety box [2, 6] (never fixture primary-E bands);
    # spur/stem keep the [8, 11] box; mean sits at each axis midpoint.
    assert params["initial_mean_log10"] == [4.0, 9.5, 9.5]
    assert params["initial_sigma_log10"] == 0.2
    assert params["population_size"] == 20
    assert params["max_generations"] == 20
    assert params["cma_seed"] == 56
    assert params["max_sigma_log10"] == 0.5
    assert params["search_bounds_log10"] == {
        "lower": [2.0, 8.0, 8.0],
        "upper": [6.0, 11.0, 11.0],
    }
    normalized = cmaes.normalize_search_bounds_log10(params["search_bounds_log10"])
    assert normalized == ((2.0, 8.0, 8.0), (6.0, 11.0, 11.0))


def test_run_passes_shipped_search_bounds_to_optimizer(monkeypatch, tmp_path):
    """Default CMA_SEARCH_PARAMS box must reach create_structure_cma_optimizer."""
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
    dataset.structure_summaries.return_value = [{}]
    _attach_sim_episode_meta(dataset)

    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        return cmaes.create_structure_cma_optimizer(bounds, **kwargs)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        del max_generations, evaluate_fn
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = (4.0, 9.5, 9.5)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
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

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "create_structure_cma_optimizer", fake_create)
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    assert create_calls
    assert create_calls[0]["search_bounds_log10"] == (
        (2.0, 8.0, 8.0),
        (6.0, 11.0, 11.0),
    )
    assert create_calls[0]["initial_mean_log10"] == pytest.approx([4.0, 9.5, 9.5])
    assert create_calls[0]["max_sigma_log10"] == 0.5
    report = json.loads((output_dir / "cmaes_report.json").read_text(encoding="utf-8"))
    assert report["cma"]["search_bounds_log10"] == {
        "lower": [2.0, 8.0, 8.0],
        "upper": [6.0, 11.0, 11.0],
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
    _attach_sim_episode_meta(dataset)

    fit_calls: list[dict] = []
    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        es, seed, rng = cmaes.create_structure_cma_optimizer(bounds, **kwargs)
        return es, seed, rng

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        fit_calls.append({"max_generations": max_generations})
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
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
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    _attach_sim_episode_meta(dataset)

    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        es, seed, rng = cmaes.create_structure_cma_optimizer(bounds, **kwargs)
        return es, seed, rng

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
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
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    assert args.max_generations is None
    assert not hasattr(args, "initial_sigma_log10")
    assert not hasattr(args, "population_size")
    assert args.cma_seed is None
    assert args.ranges is None
    assert args.multi_structure_batch is True
    assert args.fail_fast is False
    assert args.use_median is None
    assert args.hold_aggregation == "none"
    assert args.hold_id_onehot is True
    assert args.pool_directions is True
    assert args.include_delta is False
    assert args.categorical_weight == pytest.approx(30.0)
    assert args.force_magnitude_weight == pytest.approx(0.0)
    assert args.isolated_eval_waves is True
    assert args.wave_max_attempts == 5
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset",
                "/tmp/gt",
                "--output",
                "/tmp/cma",
                "--wave-max-attempts",
                "0",
            ]
        )
    assert module.SETTLE_QUIET_EVERY == 100
    assert args.settle_quiet_every == 100

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
    # Mean/sigma/popsize live in CMA_SEARCH_PARAMS; cma_seed/max_generations are CLI-overridable.
    assert "--max-generations" in option_strings
    assert "--population-size" not in option_strings
    assert "--initial-sigma-log10" not in option_strings
    assert "--cma-seed" in option_strings
    # Grid-only controls must stay off the CMA command.
    assert "--support-kp-values" not in option_strings
    assert "--log10-support-kp" not in option_strings
    assert "--log10-e-spur" not in option_strings
    assert "--log10-e-stem" not in option_strings
    assert "--include-gt-candidate" not in option_strings
    assert "--max-candidates" not in option_strings
    assert "--export-replays" not in option_strings
    assert "--max-overlay-candidates" not in option_strings


def test_parser_accepts_legacy_transition_scoring_flags(monkeypatch):
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
            "--hold-aggregation",
            "mean",
            "--include-delta",
            "--categorical-weight",
            "1",
            "--force-magnitude-weight",
            "100",
        ]
    )
    assert args.hold_aggregation == "mean"
    assert args.include_delta is True
    assert args.categorical_weight == pytest.approx(1.0)
    assert args.force_magnitude_weight == pytest.approx(100.0)


def test_build_cmaes_report_records_level_bag_scoring_flags():
    module = _load_module()
    scoring = cmaes.YoungsModulusScoringConfig(
        hold_aggregation="none",
        include_delta=False,
        categorical_weight=30.0,
    )
    payload = module._build_cmaes_report_payload(
        {},
        dataset="/tmp/gt",
        output="/tmp/out",
        ranges_path="/tmp/ranges.json",
        base_seed=0,
        initial_mean_log10=None,
        initial_sigma_log10=0.5,
        max_generations=1,
        scoring=scoring,
        command_status="completed",
        population_size=2,
        force_magnitude_weight=0.0,
        isolated_eval_waves=True,
    )
    assert payload["scoring"]["include_delta"] is False
    assert payload["scoring"]["categorical_weight"] == pytest.approx(30.0)
    assert payload["scoring"]["hold_aggregation"] == "none"
    assert payload["scoring"]["wave_max_attempts"] == 5

    custom = module._build_cmaes_report_payload(
        {},
        dataset="/tmp/gt",
        output="/tmp/out",
        ranges_path="/tmp/ranges.json",
        base_seed=0,
        initial_mean_log10=None,
        initial_sigma_log10=0.5,
        max_generations=1,
        scoring=scoring,
        command_status="completed",
        population_size=2,
        force_magnitude_weight=0.0,
        isolated_eval_waves=True,
        wave_max_attempts=7,
    )
    assert custom["scoring"]["wave_max_attempts"] == 7


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
    assert "--support-kp-values" in option_strings
    assert "--log10-support-kp" in option_strings
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
    holdout = selected / "holdout"
    holdout.mkdir()
    (holdout / "direction_000.html").write_text("old", encoding="utf-8")
    (output_dir / "holdout_report.json").write_text("{}", encoding="utf-8")
    (output_dir / ".holdout_report.json.9.tmp").write_text("{}", encoding="utf-8")
    (selected / "other.txt").write_text("keep", encoding="utf-8")

    other = output_dir / "structure_002"
    other.mkdir()
    (other / "youngs_modulus_overlay.html").write_text("keep", encoding="utf-8")

    module._clear_cma_owned_artifacts(output_dir, structure_indices=[1])

    assert not report.exists()
    assert not tmp.exists()
    assert not (output_dir / "holdout_report.json").exists()
    assert not (output_dir / ".holdout_report.json.9.tmp").exists()
    assert keep.read_text(encoding="utf-8") == "keep"
    assert not (selected / "youngs_modulus_overlay.html").exists()
    assert not (holdout / "direction_000.html").exists()
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
    _attach_sim_episode_meta(dataset)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        raise module.ViewerCancelled("viewer closed; cancelling CMA-ES fit")

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    _attach_sim_episode_meta(dataset)

    write_times: list[str] = []
    original_write = module._write_cmaes_report_atomic

    def tracking_write(path, payload):
        write_times.append(str(payload.get("command_status")))
        return original_write(path, payload)

    fit_calls: list[dict] = []

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
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
                    [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
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
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    _attach_sim_episode_meta(dataset)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        states[0].status = "failed"
        states[0].failure = cmaes.CmaGenerationFailure("prepare", "boom")
        states[1].status = "fitted"
        states[1].final_mean_log10 = tuple(states[1].bounds.log10_midpoint)
        states[1].final_evaluation = _evaluation(
            1,
            [cmaes.candidates_from_log10_vector(states[1].final_mean_log10)],
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
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    _attach_sim_episode_meta(dataset)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
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
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    _attach_sim_episode_meta(dataset)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        state = states[0]
        state.status = "fitted"
        state.final_mean_log10 = tuple(state.bounds.log10_midpoint)
        state.final_evaluation = _evaluation(
            0,
            [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
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
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    _attach_sim_episode_meta(dataset)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        raise RuntimeError("batch exploded")

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
    dataset.load_episode_obs_arrays.return_value = {
        "ft_wrist": np.zeros((4, 6), dtype=np.float32),
        "ft_wrist_lpf": np.ones((4, 6), dtype=np.float32),
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

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
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
            prepared_structures=1,
            physical_slots_by_structure={0: 1},
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "make_real_replay_build_env_fn", fake_make_real_builder)
    monkeypatch.setattr(module, "real_replay_sim_config", fake_real_config)
    monkeypatch.setattr(module, "evaluate_youngs_modulus_structures", fake_evaluate_structures)
    _patch_execute_cma_wave_evaluation(
        monkeypatch,
        module,
        dataset=dataset,
        fake_evaluate_structures=fake_evaluate_structures,
        build_env_fn=real_builder,
        replay_sim_config=fake_real_config(
            controller_mode="vic_pose",
            control_hz=15.0,
        ),
    )
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
    dataset.load_episode_obs_arrays.return_value = {
        "ft_wrist": np.zeros((4, 6), dtype=np.float32),
        "ft_wrist_lpf": np.ones((4, 6), dtype=np.float32),
    }

    create_calls: list[dict] = []

    def fake_create(bounds, **kwargs):
        create_calls.append(dict(kwargs))
        return cmaes.create_structure_cma_optimizer(bounds, **kwargs)

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
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
    assert create_calls[0]["initial_mean_log10"] == pytest.approx([4.0, 8.0, 8.0])


def test_run_rejects_multiple_structures_for_vic_pose(monkeypatch, tmp_path):
    module = _load_module()
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
        }
    }
    dataset.structure_summaries.return_value = [{}, {}]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
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


def test_run_rejects_vic_on_one_structure_vic_pose_dataset(monkeypatch, tmp_path):
    module = _load_module()
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
        }
    }
    dataset.structure_summaries.return_value = [{}]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
    }
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: pytest.fail(
            "must not load sim GT for packed vic_pose with vic mode"
        ),
    )
    monkeypatch.setattr(
        module,
        "_make_build_env_fn",
        lambda **_kwargs: pytest.fail(
            "must not use sim builder for packed vic_pose with vic mode"
        ),
    )
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(tmp_path / "out"),
        structure_indices=None,
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
        SystemExit,
        match="vic_pose datasets must use --controller-mode vic_pose",
    ):
        module._run(args, argparse.ArgumentParser(), viewer=MagicMock())


def test_run_vic_pose_refuses_dataset_without_ft_wrist_lpf(monkeypatch, tmp_path):
    module = _load_module()
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
    dataset.load_episode_obs_arrays.return_value = {
        "ft_wrist": np.zeros((4, 6), dtype=np.float32),
    }
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(tmp_path / "out"),
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
    with pytest.raises(SystemExit, match="ft_wrist_lpf"):
        module._run(args, argparse.ArgumentParser(), viewer=MagicMock())


def test_require_ft_wrist_lpf_refuses_when_later_direction_omits_column():
    module = _load_module()
    dataset = MagicMock()
    dataset.manifest = {"collection": {"num_directions": 2}}
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 0},
        {"structure_idx": 0, "direction_idx": 1},
    ]

    def _load(_structure_idx, direction_idx):
        if int(direction_idx) == 0:
            return {
                "ft_wrist": np.zeros((4, 6), dtype=np.float32),
                "ft_wrist_lpf": np.ones((4, 6), dtype=np.float32),
            }
        return {"ft_wrist": np.zeros((4, 6), dtype=np.float32)}

    dataset.load_episode_obs_arrays.side_effect = _load
    with pytest.raises(SystemExit, match="ft_wrist_lpf"):
        module._require_ft_wrist_lpf_per_structure(dataset, [0])


def _eight_dir_vic_pose_dataset(*, ranges_path: Path) -> MagicMock:
    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "action_layout": "vic_pose_v1",
            "action_dim": 19,
            "control_hz": 15.0,
            "num_directions": 8,
            "ranges_path": str(ranges_path),
            "topology_seed": 9,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}]
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": d} for d in range(8)
    ]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
        "control_hz": 15.0,
        "fruiting_base_pos": [1.0, 2.0, 3.0],
        "initial_robot_joint_q": [0.1, 0.2],
        "action_compatible_with_vic_twist": False,
    }
    dataset.load_episode_obs_arrays.return_value = {
        "ft_wrist": np.zeros((4, 6), dtype=np.float32),
        "ft_wrist_lpf": np.ones((4, 6), dtype=np.float32),
    }
    return dataset


def _holdout_run_stub(monkeypatch, module, *, dataset: MagicMock, tmp_path: Path):
    """Patch _run for holdout tests; returns (evaluate_calls, output_dir)."""
    output_dir = tmp_path / "cma_out"
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")

    evaluate_calls: list[dict] = []

    def fake_create_optimizer(*_args, **_kwargs):
        es = MagicMock()
        es.popsize = 15
        return es, 56, None

    def fake_evaluate_structures(**kwargs):
        evaluate_calls.append(dict(kwargs))
        structures = kwargs["structures"]
        dirs = kwargs.get("direction_indices")
        if dirs is None:
            dirs = tuple(range(8))
        dirs = tuple(int(d) for d in dirs)
        evaluations = {}
        for structure_idx, candidates in structures:
            replay_eps = [_synthetic_recorded_episode(direction=d) for d in dirs]
            sinkhorn = 0.1
            if dirs == (0, 1, 3) and len(candidates) == 1:
                cand_log10 = _candidate_log10(candidates[0])
                baseline = tuple(module.CMA_SEARCH_PARAMS["initial_mean_log10"])
                sinkhorn = 2.0 if cand_log10 == baseline else 1.0
            evaluations[int(structure_idx)] = _evaluation(
                int(structure_idx),
                list(candidates),
                [sinkhorn] * len(candidates),
                direction_indices=dirs,
                replay_episodes=[replay_eps for _ in candidates],
            )
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations=evaluations,
            errors={},
            replay_diagnostics=None,
            prepared_structures=len(evaluations),
            physical_slots_by_structure={
                int(idx): len(cands) * len(evaluations[int(idx)].direction_indices)
                for idx, cands in structures
            },
        )

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        del max_generations
        cand = cmaes.candidates_from_log10_vector((4.0, 9.0, 9.0))
        for _ in range(2):
            evaluate_fn(structures=[(0, (cand, cand))], wave_kind="generation")
        batch = evaluate_fn(
            structures=[(0, (cand,))],
            wave_kind="final_mean",
        )
        dist = cmaes.CmaDistributionSnapshot(mean_log10=(4.0, 9.0, 9.0), sigma=1.0)
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = (4.0, 9.0, 9.0)
            state.final_evaluation = batch.evaluations[0]
            state.gt_candidate = None
            state.generations = [
                cmaes.CmaGenerationRecord(
                    generation_index=0,
                    structure_idx=0,
                    ask_samples_log10=((4.0, 9.0, 9.0),),
                    candidates=(cand,),
                    raw_scores=(),
                    penalized_fitness=(2.0,),
                    penalty_metadata=({"penalized": False, "raw_aggregate_sinkhorn": 2.0},),
                    ask_distribution=dist,
                    post_tell_distribution=dist,
                ),
                cmaes.CmaGenerationRecord(
                    generation_index=1,
                    structure_idx=0,
                    ask_samples_log10=((4.0, 9.0, 9.0),),
                    candidates=(cand,),
                    raw_scores=(),
                    penalized_fitness=(1.0,),
                    penalty_metadata=({"penalized": False, "raw_aggregate_sinkhorn": 1.0},),
                    ask_distribution=dist,
                    post_tell_distribution=dist,
                ),
            ]
        if on_progress is not None:
            on_progress(states)
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(0,),
            failed_structure_indices=(),
            generation_waves=2,
            final_mean_batch=batch,
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "create_structure_cma_optimizer", fake_create_optimizer)
    monkeypatch.setattr(module, "make_real_replay_build_env_fn", lambda **_k: MagicMock())
    monkeypatch.setattr(
        module,
        "real_replay_sim_config",
        lambda **_k: SimpleNamespace(controller=SimpleNamespace(mode="vic_pose")),
    )
    import apple_pick_gym.batched_envs.holdout_evaluation as holdout_eval

    monkeypatch.setattr(
        holdout_eval,
        "load_recorded_episodes_for_structure",
        lambda _dataset, **kwargs: [
            _synthetic_recorded_episode(direction=int(d))
            for d in kwargs["direction_indices"]
        ],
    )
    monkeypatch.setattr(
        holdout_eval,
        "load_episode_metadata_for_directions",
        lambda _dataset, **kwargs: {
            int(d): {"pull_direction": [0.0, 0.0, 1.0]}
            for d in kwargs["direction_indices"]
        },
    )
    monkeypatch.setattr(module, "evaluate_youngs_modulus_structures", fake_evaluate_structures)
    _patch_execute_cma_wave_evaluation(
        monkeypatch,
        module,
        dataset=dataset,
        fake_evaluate_structures=fake_evaluate_structures,
        ranges_dict=_valid_ranges_dict(),
    )
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: pytest.fail("real CMA must not load sim GT"),
    )
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: [])
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)
    return evaluate_calls, output_dir, ranges_path


def test_parser_direction_split_seed_defaults_to_seventeen_when_bare(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(
        ["--dataset", "/tmp/ds", "--output", "/tmp/out", "--direction-split-seed"]
    )
    assert args.direction_split_seed == 17
    args = parser.parse_args(
        ["--dataset", "/tmp/ds", "--output", "/tmp/out", "--direction-split-seed", "42"]
    )
    assert args.direction_split_seed == 42


def test_parser_direction_split_seed_absent_is_none(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(["--dataset", "/tmp/ds", "--output", "/tmp/out"])
    assert args.direction_split_seed is None
    assert args.direction_indices is None
    assert args.val_direction_indices is None


def test_run_without_split_flags_uses_all_directions(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    evaluate_calls, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        direction_split_seed=None,
        direction_indices=None,
        val_direction_indices=None,
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
    assert result["train_direction_indices"] is None
    assert result["val_direction_indices"] is None
    struct_calls = [c for c in evaluate_calls if "structures" in c]
    assert struct_calls
    assert all(c.get("direction_indices") is None for c in struct_calls)
    assert not (output_dir / "holdout_report.json").exists()


def test_run_holdout_seed_selects_pinned_split(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    evaluate_calls, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        direction_split_seed=17,
        direction_indices=None,
        val_direction_indices=None,
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
    assert result["train_direction_indices"] == (2, 4, 5, 6, 7)
    assert result["val_direction_indices"] == (0, 1, 3)
    struct_calls = [c for c in evaluate_calls if "structures" in c]
    assert struct_calls
    fit_calls = [c for c in struct_calls if c.get("direction_indices") != (0, 1, 3)]
    for call in fit_calls:
        assert call["direction_indices"] == (2, 4, 5, 6, 7)


def test_run_holdout_never_loads_val_directions_during_fit(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    evaluate_calls, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        direction_split_seed=17,
        direction_indices=None,
        val_direction_indices=None,
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
    seen_dirs: set[int] = set()
    for call in evaluate_calls:
        dirs = call.get("direction_indices")
        if dirs is not None and dirs != (0, 1, 3):
            seen_dirs.update(int(d) for d in dirs)
    assert seen_dirs.isdisjoint({0, 1, 3})


def test_run_rejects_partial_explicit_split(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    args = SimpleNamespace(
        dataset="/tmp/real",
        output=str(tmp_path / "out"),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        direction_split_seed=None,
        direction_indices=(0, 1, 2),
        val_direction_indices=None,
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
    with pytest.raises(SystemExit, match="direction-indices"):
        module._run(args, argparse.ArgumentParser(), viewer=MagicMock())


def test_run_rejects_overlapping_or_empty_explicit_split(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())

    def _args(**overrides):
        base = dict(
            dataset="/tmp/real",
            output=str(tmp_path / "out"),
            structure_indices=None,
            ranges=None,
            max_envs_per_batch=0,
            seed=None,
            cma_seed=None,
            controller_mode=None,
            direction_split_seed=None,
            direction_indices=(0, 1, 2, 3, 4),
            val_direction_indices=(3, 4, 5),
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
        base.update(overrides)
        return SimpleNamespace(**base)

    with pytest.raises(SystemExit, match="disjoint"):
        module._run(_args(), argparse.ArgumentParser(), viewer=MagicMock())
    with pytest.raises(SystemExit, match="non-empty"):
        module._run(
            _args(direction_indices=(), val_direction_indices=(0, 1, 2)),
            argparse.ArgumentParser(),
            viewer=MagicMock(),
        )


def test_run_rejects_holdout_on_non_eight_direction_dataset(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "action_layout": "vic_pose_v1",
            "action_dim": 19,
            "control_hz": 15.0,
            "num_directions": 4,
            "ranges_path": str(ranges_path),
        }
    }
    dataset.structure_summaries.return_value = [{}]
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": d} for d in range(4)
    ]
    dataset.load_episode_metadata.return_value = {
        "action_layout": "vic_pose_v1",
        "action_dim": 19,
    }
    dataset.load_episode_obs_arrays.return_value = {
        "ft_wrist": np.zeros((4, 6), dtype=np.float32),
        "ft_wrist_lpf": np.ones((4, 6), dtype=np.float32),
    }
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())

    def _args(**overrides):
        base = dict(
            dataset="/tmp/real",
            output=str(tmp_path / "out"),
            structure_indices=None,
            ranges=None,
            max_envs_per_batch=0,
            seed=None,
            cma_seed=None,
            controller_mode=None,
            direction_split_seed=17,
            direction_indices=None,
            val_direction_indices=None,
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
        base.update(overrides)
        return SimpleNamespace(**base)

    with pytest.raises(SystemExit, match="8"):
        module._run(_args(), argparse.ArgumentParser(), viewer=MagicMock())
    with pytest.raises(SystemExit, match="8"):
        module._run(
            _args(
                direction_split_seed=None,
                direction_indices=(0, 1),
                val_direction_indices=(2, 3),
            ),
            argparse.ArgumentParser(),
            viewer=MagicMock(),
        )


def test_run_holdout_keeps_one_structure_guard(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    dataset.structure_summaries.return_value = [{}, {}]
    dataset.episode_entries.return_value = [
        {"structure_idx": s, "direction_idx": d}
        for s in range(2)
        for d in range(8)
    ]
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
        controller_mode=None,
        direction_split_seed=17,
        direction_indices=None,
        val_direction_indices=None,
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


def test_require_ft_wrist_lpf_iterates_episode_rows():
    module = _load_module()
    dataset = MagicMock()
    dataset.manifest = {"collection": {"num_directions": 6}}
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 3},
        {"structure_idx": 0, "direction_idx": 5},
    ]
    dataset.load_episode_obs_arrays.return_value = {
        "ft_wrist": np.zeros((4, 6), dtype=np.float32),
        "ft_wrist_lpf": np.ones((4, 6), dtype=np.float32),
    }
    module._require_ft_wrist_lpf_per_structure(dataset, [0])


def _holdout_args(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="/tmp/real",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        controller_mode=None,
        direction_split_seed=17,
        direction_indices=None,
        val_direction_indices=None,
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


def test_run_holdout_evaluates_baseline_and_fitted_on_val_only(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    evaluate_calls, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    module._run(_holdout_args(output_dir), argparse.ArgumentParser(), viewer=MagicMock())

    val_calls = [
        c
        for c in evaluate_calls
        if c.get("direction_indices") == (0, 1, 3)
    ]
    assert len(val_calls) == 2
    baseline = (4.0, 8.0, 8.0)
    seen_log10 = {
        _candidate_log10(c["structures"][0][1][0]) for c in val_calls
    }
    assert _candidate_log10(cmaes.candidates_from_log10_vector(baseline)) in seen_log10
    assert _candidate_log10(cmaes.candidates_from_log10_vector((4.0, 9.0, 9.0))) in seen_log10


def test_run_holdout_does_not_tell_optimizer_on_val(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    tell_calls: list[int] = []

    def fake_create_optimizer(*_args, **_kwargs):
        es = MagicMock()
        es.popsize = 15

        def _tell(*_a, **_k):
            tell_calls.append(1)

        es.tell = _tell
        return es, 56, None

    evaluate_calls, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    monkeypatch.setattr(module, "create_structure_cma_optimizer", fake_create_optimizer)
    stub_fit = module.fit_youngs_modulus_structures

    def fake_fit(*args, **kwargs):
        result = stub_fit(*args, **kwargs)
        for state in args[0].values():
            state.optimizer.tell([], [])
        return result

    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    module._run(_holdout_args(output_dir), argparse.ArgumentParser(), viewer=MagicMock())
    assert len(tell_calls) == 1


def test_run_writes_holdout_report_with_val_overlays(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    train_overlay = tmp_path / "train_overlay.html"
    monkeypatch.setattr(
        module,
        "_write_final_mean_overlay",
        lambda *_a, **_k: train_overlay.write_text("train", encoding="utf-8"),
    )
    _, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    module._run(_holdout_args(output_dir), argparse.ArgumentParser(), viewer=MagicMock())

    report_path = output_dir / "holdout_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["val_overlay_paths"]) == {"0", "1", "3"}
    for path in report["val_overlay_paths"].values():
        assert Path(path).is_file()
        assert "holdout/direction_" in path
        assert path != str(train_overlay)


def test_run_skips_holdout_report_when_fit_failed(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    evaluate_calls, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )

    def failed_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        del max_generations, evaluate_fn, on_progress
        for state in states.values():
            state.status = "failed"
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(),
            failed_structure_indices=(0,),
            generation_waves=0,
            final_mean_batch=None,
        )

    monkeypatch.setattr(module, "fit_youngs_modulus_structures", failed_fit)
    result = module._run(
        _holdout_args(output_dir), argparse.ArgumentParser(), viewer=MagicMock()
    )
    assert result["exit_nonzero"] is True
    assert not (output_dir / "holdout_report.json").exists()
    assert not any(
        c.get("direction_indices") == (0, 1, 3) for c in evaluate_calls
    )


def test_run_holdout_report_absent_without_split_flags(monkeypatch, tmp_path):
    module = _load_module()
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    dataset = _eight_dir_vic_pose_dataset(ranges_path=ranges_path)
    _, output_dir, _ = _holdout_run_stub(
        monkeypatch, module, dataset=dataset, tmp_path=tmp_path
    )
    args = _holdout_args(output_dir)
    args.direction_split_seed = None
    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert not (output_dir / "holdout_report.json").exists()


@pytest.mark.parametrize("isolated,reuse", [(True, False), (False, True)])
def test_run_enables_robot_reuse_only_on_inprocess_eval(monkeypatch, tmp_path, isolated, reuse):
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
    dataset.structure_summaries.return_value = [{}]
    _attach_sim_episode_meta(dataset)

    captured: list[bool] = []

    def fake_execute(spec, **_kwargs):
        captured.append(bool(spec.replay_context.reuse_replicated_mujoco))
        return cmaes.YoungsModulusBatchEvaluation(
            evaluations={},
            errors={},
            replay_diagnostics=None,
        )

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        del max_generations
        evaluate_fn(
            structures=[(0, (cmaes.SupportKpYoungsCandidate(1e4, 1e8, 1e7),))],
            wave_kind="generation",
        )
        for state in states.values():
            state.status = "fitted"
            state.final_mean_log10 = (4.0, 9.5, 9.5)
            state.final_evaluation = _evaluation(
                state.structure_idx,
                [cmaes.candidates_from_log10_vector(state.final_mean_log10)],
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

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
    )
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "execute_cma_wave_evaluation", fake_execute)
    monkeypatch.setattr(module, "spawn_isolated_cma_wave_evaluation", fake_execute)

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
        isolated_eval_waves=isolated,
        force_magnitude_weight=0.0,
    )
    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert captured == [reuse]


def test_parser_exposes_resume_and_process_restart_defaults(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(
        ["--dataset", "/tmp/gt", "--output", "/tmp/cma"]
    )
    assert args.resume is False
    assert args.max_process_restarts == 10
    assert args.cma_process_attempt == 0
    option_strings = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--resume" in option_strings


def test_run_resume_allows_nonempty_output_without_overwrite(monkeypatch, tmp_path):
    from apple_pick_gym.tests.test_batched_sysid_cmaes_loop import _make_pycma_state

    module = _load_module()
    output_dir = tmp_path / "cma_out"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("x", encoding="utf-8")
    checkpoint_path = output_dir / module.CMA_OPTIMIZER_CHECKPOINT_FILENAME

    state = _make_pycma_state()
    samples = state.optimizer.ask()
    state.optimizer.tell(samples, [1.0] * len(samples))
    state.completed_generations = 1
    cmaes.dump_cma_optimizer_checkpoint(
        checkpoint_path, {0: state}, counters={"replay_candidate_evaluations": 5}
    )

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
    _attach_sim_episode_meta(dataset)

    resumed_generations: list[int] = []

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        for state in states.values():
            resumed_generations.append(int(state.completed_generations))
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=tuple(sorted(states)),
            failed_structure_indices=(),
            generation_waves=0,
            final_mean_batch=None,
        )

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
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
        overwrite=False,
        resume=True,
        max_process_restarts=10,
        cma_process_attempt=0,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=False,
        settle_quiet_every=None,
        show_pull_direction=False,
        viewer="null",
    )
    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert resumed_generations == [1]


def test_cma_should_reexec_and_argv_helpers():
    module = _load_module()
    args = SimpleNamespace(max_process_restarts=10, cma_process_attempt=0)
    assert module.cma_should_reexec(
        {"exit_nonzero": True, "command_status": "failed"}, args
    )
    assert not module.cma_should_reexec(
        {"exit_nonzero": True, "command_status": "cancelled"}, args
    )
    argv = [
        "python",
        "-m",
        "example",
        "--dataset",
        "/tmp/gt",
        "--output",
        "/tmp/cma",
        "--overwrite",
        "--cma-process-attempt",
        "2",
    ]
    rebuilt = module.cma_reexec_argv(argv, args)
    script = str(module._cma_entry_script())
    assert script in rebuilt
    assert "--overwrite" not in rebuilt
    assert "--resume" in rebuilt
    assert rebuilt[-2:] == ["--cma-process-attempt", "1"]

    wrapper_argv = [
        "/usr/bin/python",
        "-c",
        "ignored",
        "--dataset",
        "/tmp/gt",
        "--output",
        "/tmp/cma",
        "--overwrite",
        "--max-generations",
        "50",
    ]
    rebuilt_wrapper = module.cma_reexec_argv(wrapper_argv, args)
    assert script in rebuilt_wrapper
    assert "--dataset" in rebuilt_wrapper
    assert "--max-generations" in rebuilt_wrapper
    assert rebuilt_wrapper[rebuilt_wrapper.index("--max-generations") + 1] == "50"


def test_cma_reexec_argv_uses_uv_when_available(monkeypatch):
    module = _load_module()
    args = SimpleNamespace(max_process_restarts=10, cma_process_attempt=2)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    argv = [
        "/venv/bin/python3",
        "-X",
        "faulthandler",
        str(module._cma_entry_script()),
        "--dataset",
        "/tmp/gt",
        "--output",
        "/tmp/cma",
        "--overwrite",
        "--cma-process-attempt",
        "2",
    ]
    rebuilt = module.cma_reexec_argv(argv, args)
    assert rebuilt[:5] == [
        "/usr/bin/uv",
        "run",
        "--directory",
        str(module._cma_repo_root()),
        "python",
    ]
    assert rebuilt[5].endswith("example_youngs_modulus_cmaes.py")
    assert "faulthandler" not in rebuilt
    assert "--resume" in rebuilt
    assert rebuilt[-2:] == ["--cma-process-attempt", "3"]


def test_parser_exposes_snapshot_video_interval(monkeypatch):
    module = _load_module()
    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(["--dataset", "/tmp/gt", "--output", "/tmp/cma"])
    assert args.snapshot_video_every == 0
    every = parser.parse_args(
        [
            "--dataset",
            "/tmp/gt",
            "--output",
            "/tmp/cma",
            "--snapshot-video-every",
            "5",
        ]
    )
    assert every.snapshot_video_every == 5
    disabled = parser.parse_args(
        [
            "--dataset",
            "/tmp/gt",
            "--output",
            "/tmp/cma",
            "--snapshot-video-every",
            "0",
        ]
    )
    assert disabled.snapshot_video_every == 0
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset",
                "/tmp/gt",
                "--output",
                "/tmp/cma",
                "--snapshot-video-every",
                "-1",
            ]
        )


def test_should_record_cma_snapshot_video():
    module = _load_module()
    assert not module.should_record_cma_snapshot_video(0, 0, None)
    assert module.should_record_cma_snapshot_video(5, 0, None)
    assert not module.should_record_cma_snapshot_video(5, 1, None)
    assert not module.should_record_cma_snapshot_video(5, 4, None)
    assert module.should_record_cma_snapshot_video(5, 5, None)
    assert not module.should_record_cma_snapshot_video(5, 5, 5)
    assert not module.should_record_cma_snapshot_video(5, 6, 5)


def test_choose_random_cma_snapshot_sample_is_deterministic():
    module = _load_module()
    record = SimpleNamespace(
        generation_index=4,
        structure_idx=0,
        ask_samples_log10=(
            (4.0, 8.0, 8.0),
            (4.2, 7.1, 9.0),
            (4.1, 7.5, 8.5),
        ),
        penalized_fitness=(500.0, 412.0, 430.0),
    )
    state = SimpleNamespace(generations=[record], structure_idx=0)
    first = module.choose_random_cma_snapshot_sample({0: state}, seed=17)
    second = module.choose_random_cma_snapshot_sample({0: state}, seed=17)
    other = module.choose_random_cma_snapshot_sample({0: state}, seed=18)
    assert first == second
    assert first.generation_index == 4
    assert first.structure_idx == 0
    assert first.candidate_index in {0, 1, 2}
    assert first.log10_vector == record.ask_samples_log10[first.candidate_index]
    assert {first.candidate_index, other.candidate_index} <= {0, 1, 2}


def test_cma_snapshot_video_path_uses_generation_and_sample():
    module = _load_module()
    sample = module.CmaSnapshotSample(
        structure_idx=0,
        generation_index=10,
        candidate_index=7,
        log10_vector=(4.4, 7.0, 9.1),
        fitness=412.0,
    )
    path = module.cma_snapshot_video_path(Path("/tmp/cma"), sample, direction_idx=3)
    assert path == Path(
        "/tmp/cma/videos/structure_000_gen_010_dir_003_sample_007.mp4"
    )


def test_frame_snapshot_camera_uses_plant_bounds_and_look_direction():
    module = _load_module()
    camera = SimpleNamespace(fov=45.0, pos=None, pitch=None, yaw=None)
    placed: list[tuple] = []

    class Viewer:
        def __init__(self):
            self.camera = camera
            self._last_state = None

        def set_camera(self, pos, pitch, yaw):
            placed.append((tuple(float(x) for x in pos), float(pitch), float(yaw)))

    body_q = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    env = SimpleNamespace(
        _sim=SimpleNamespace(
            scene=SimpleNamespace(
                cable=SimpleNamespace(state_0=SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q)))
            )
        )
    )
    pose = module.frame_snapshot_camera_on_structure(
        Viewer(),
        env,
        look_pitch_deg=-20.0,
        look_yaw_deg=45.0,
        padding=1.5,
    )
    assert pose is not None
    pos, pitch, yaw = pose
    assert pitch == pytest.approx(-20.0)
    assert yaw == pytest.approx(45.0)
    assert pos[2] > 0.3
    assert placed
    assert camera.fov == pytest.approx(45.0)


def test_frame_snapshot_camera_uses_parquet_pose_relative_to_structure():
    module = _load_module()
    placed: list[tuple] = []

    class Viewer:
        def set_camera(self, pos, pitch, yaw):
            placed.append((tuple(float(x) for x in pos), float(pitch), float(yaw)))

    env = SimpleNamespace(_sim=None)
    camera_to_base = [
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pose = module.frame_snapshot_camera_on_structure(
        Viewer(),
        env,
        camera_to_base_4x4=camera_to_base,
        world_offset=(2.0, 0.0, 0.0),
    )
    assert pose is not None
    pos, pitch, yaw = pose
    assert pos == pytest.approx((2.1, 0.2, 0.3))
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert placed[0][0] == pytest.approx((2.1, 0.2, 0.3))


def test_choose_random_snapshot_direction_is_deterministic():
    module = _load_module()
    first = module.choose_random_snapshot_direction((0, 3, 7), seed=17, generation_index=5)
    second = module.choose_random_snapshot_direction((0, 3, 7), seed=17, generation_index=5)
    other = module.choose_random_snapshot_direction((0, 3, 7), seed=18, generation_index=5)
    assert first == second
    assert first in {0, 3, 7}
    assert other in {0, 3, 7}


def test_choose_random_snapshot_direction_varies_by_generation():
    module = _load_module()
    pool = tuple(range(8))
    chosen = {
        module.choose_random_snapshot_direction(pool, seed=0, generation_index=generation)
        for generation in (0, 5, 10, 15, 20)
    }
    assert len(chosen) > 1


def test_resolve_cma_snapshot_direction_indices_uses_train_split():
    module = _load_module()
    dataset = MagicMock()
    resolved = module.resolve_cma_snapshot_direction_indices(
        (1, 3, 5),
        dataset=dataset,
        structure_idx=0,
        num_directions=8,
        include_excluded=False,
    )
    assert resolved == (1, 3, 5)
    dataset.structure_summaries.assert_not_called()


def test_resolve_cma_snapshot_direction_indices_uses_all_usable_without_holdout(
    monkeypatch,
):
    module = _load_module()
    dataset = MagicMock()
    monkeypatch.setattr(
        module,
        "list_usable_direction_indices",
        lambda _dataset, _structure_idx, *, include_excluded: [0, 2, 4, 6],
    )
    resolved = module.resolve_cma_snapshot_direction_indices(
        None,
        dataset=dataset,
        structure_idx=0,
        num_directions=8,
        include_excluded=False,
    )
    assert resolved == (0, 2, 4, 6)


def test_reset_headless_gl_event_loop_clears_pyglet_exit_flag():
    pyglet = pytest.importorskip("pyglet")
    module = _load_module()
    pyglet.app.event_loop.has_exit = True
    module._reset_headless_gl_event_loop()
    assert pyglet.app.event_loop.has_exit is False


def test_open_headless_gl_viewer_resets_event_loop_before_construct(monkeypatch):
    module = _load_module()
    calls: list[str] = []

    def fake_reset() -> None:
        calls.append("reset")

    class FakeViewer:
        pass

    monkeypatch.setattr(module, "_reset_headless_gl_event_loop", fake_reset)
    monkeypatch.setattr(
        module.newton.viewer,
        "ViewerGL",
        lambda **kwargs: FakeViewer(),
    )
    viewer = module._open_headless_gl_viewer()
    assert isinstance(viewer, FakeViewer)
    assert calls == ["reset"]


def test_make_snapshot_on_step_headless_ignores_stale_is_running():
    module = _load_module()
    captures = 0

    class Viewer:
        renderer = SimpleNamespace(headless=True)

        def set_model(self, model):
            del model

        def begin_frame(self, t):
            del t

        def log_state(self, state):
            del state

        def log_contacts(self, contacts, state):
            del contacts, state

        def end_frame(self):
            pass

        def get_frame(self):
            return object()

        def is_running(self):
            return False

    class Recorder:
        fps = None

        def set_fps(self, fps: float) -> None:
            self.fps = float(fps)

        def capture(self, viewer) -> None:
            nonlocal captures
            del viewer
            captures += 1

    cable = SimpleNamespace(model="MODEL", state_0="STATE")
    scene = SimpleNamespace(cable=cable, last_vbd_contacts="CONTACTS")
    sim = SimpleNamespace(
        scene=scene,
        layout=None,
        config=SimpleNamespace(runtime=SimpleNamespace(control_hz=20.0)),
        obs_bufs=None,
    )
    env = SimpleNamespace(_sim=sim, num_envs=1, _last_obs=None)
    on_step = module.make_snapshot_on_step(
        Viewer(),
        control_hz=30.0,
        recorder=Recorder(),
    )
    assert on_step(frame_idx=0, env=env) is True
    assert captures == 1


def test_make_snapshot_on_step_frames_camera_once():
    module = _load_module()
    framed: list[int] = []
    order: list[str] = []

    class Viewer:
        def set_model(self, model):
            order.append("set_model")

        def set_world_offsets(self, spacing):
            order.append("offsets")

        def hide_loading_splash(self):
            order.append("splash")

        def begin_frame(self, t):
            order.append("begin")

        def log_state(self, state):
            order.append("log")

        def log_contacts(self, contacts, state):
            order.append("contacts")

        def end_frame(self):
            order.append("end")

        def get_frame(self):
            return object()

        def is_running(self):
            return True

    class Recorder:
        def __init__(self):
            self.fps = None
            self.captures = 0

        def set_fps(self, fps: float) -> None:
            self.fps = float(fps)

        def capture(self, viewer) -> None:
            del viewer
            self.captures += 1

    cable = SimpleNamespace(model="MODEL", state_0="STATE")
    scene = SimpleNamespace(cable=cable, last_vbd_contacts="CONTACTS")
    sim = SimpleNamespace(
        scene=scene,
        layout=None,
        config=SimpleNamespace(
            runtime=SimpleNamespace(control_hz=20.0, env_spacing=(2.0, 2.0, 2.0))
        ),
        obs_bufs=None,
    )
    env = SimpleNamespace(_sim=sim, num_envs=2, _last_obs=None)
    recorder = Recorder()
    on_step = module.make_snapshot_on_step(
        Viewer(),
        control_hz=30.0,
        recorder=recorder,
        frame_camera=lambda *_a, **_k: framed.append(1),
    )
    assert on_step(frame_idx=0, env=env) is True
    assert on_step(frame_idx=1, env=env) is True
    assert framed == [1]
    assert recorder.captures == 2


def test_record_cma_snapshot_video_uses_make_snapshot_on_step(monkeypatch, tmp_path):
    module = _load_module()
    calls: list[dict] = []

    def fake_make_snapshot_on_step(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return lambda **_k: True

    class Viewer:
        def close(self):
            pass

    class Recorder:
        def __init__(self, path):
            del path
            self.frame_count = 1

        def close(self):
            pass

    monkeypatch.setattr(module, "make_snapshot_on_step", fake_make_snapshot_on_step)
    monkeypatch.setattr(module, "GlVideoRecorder", Recorder)
    monkeypatch.setattr(module, "require_gl_frame_capture", lambda _viewer: None)
    monkeypatch.setattr(module, "make_cma_wave_evaluation_spec", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(module, "execute_cma_wave_evaluation", lambda _spec: None)
    monkeypatch.setattr(
        module,
        "candidates_from_log10_vector",
        lambda _vec: (1.0, 2.0, 3.0),
    )
    sample = module.CmaSnapshotSample(
        structure_idx=0,
        generation_index=5,
        candidate_index=1,
        log10_vector=(4.0, 8.0, 8.0),
        fitness=1.0,
    )
    module.record_cma_snapshot_video(
        sample,
        output_dir=tmp_path,
        replay_context=SimpleNamespace(),
        scoring=SimpleNamespace(),
        dataset_dir=tmp_path,
        num_directions=2,
        direction_indices=(0, 1),
        max_envs_per_batch=0,
        seed=0,
        include_excluded=False,
        fail_fast=False,
        action_dim=6,
        show_pull_direction=False,
        control_hz=30.0,
        open_viewer=Viewer,
    )
    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["control_hz"] == pytest.approx(30.0)
    assert kwargs["recorder"] is not None


def test_record_cma_snapshot_video_uses_structure_parquet_camera(monkeypatch, tmp_path):
    module = _load_module()
    camera = [
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    snapshot_kwargs: list[dict] = []
    spec_kwargs: list[dict] = []

    def fake_make_snapshot_on_step(*args, **kwargs):
        del args
        snapshot_kwargs.append(kwargs)
        return lambda **_k: True

    class Viewer:
        def close(self):
            pass

    class Recorder:
        def __init__(self, path):
            del path
            self.frame_count = 1

        def close(self):
            pass

    class Dataset:
        def load_episode_metadata(self, structure_idx, direction_idx):
            assert structure_idx == 2
            return {"camera_to_base_4x4": camera}

    monkeypatch.setattr(module, "make_snapshot_on_step", fake_make_snapshot_on_step)
    monkeypatch.setattr(module, "GlVideoRecorder", Recorder)
    monkeypatch.setattr(module, "require_gl_frame_capture", lambda _viewer: None)
    monkeypatch.setattr(
        module,
        "make_cma_wave_evaluation_spec",
        lambda **kw: spec_kwargs.append(kw) or SimpleNamespace(**kw),
    )
    monkeypatch.setattr(module, "execute_cma_wave_evaluation", lambda _spec: None)
    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: Dataset())
    monkeypatch.setattr(
        module,
        "candidates_from_log10_vector",
        lambda _vec: (1.0, 2.0, 3.0),
    )
    sample = module.CmaSnapshotSample(
        structure_idx=2,
        generation_index=5,
        candidate_index=1,
        log10_vector=(4.0, 8.0, 8.0),
        fitness=1.0,
    )
    module.record_cma_snapshot_video(
        sample,
        output_dir=tmp_path,
        replay_context=SimpleNamespace(),
        scoring=SimpleNamespace(),
        dataset_dir=tmp_path,
        num_directions=8,
        direction_indices=(0, 3, 7),
        max_envs_per_batch=0,
        seed=17,
        include_excluded=False,
        fail_fast=False,
        action_dim=6,
        show_pull_direction=False,
        control_hz=30.0,
        open_viewer=Viewer,
    )
    assert snapshot_kwargs[0]["camera_to_base_4x4"] == camera
    dirs = spec_kwargs[0]["direction_indices"]
    assert dirs is not None and len(dirs) == 1
    assert dirs[0] in {0, 3, 7}


def test_run_records_snapshot_video_on_selected_generations(monkeypatch, tmp_path):
    module = _load_module()
    output_dir = tmp_path / "cma_out"
    dataset = MagicMock()
    dataset.manifest = {
        "collection": {
            "control_hz": 30.0,
            "num_directions": 2,
            "ranges_path": str(tmp_path / "ranges.json"),
            "topology_seed": 42,
            "seed": 7,
        }
    }
    dataset.structure_summaries.return_value = [{}]
    _attach_sim_episode_meta(dataset)
    (tmp_path / "ranges.json").write_text(json.dumps(_valid_ranges_dict()), encoding="utf-8")
    recorded: list[tuple[int, tuple[float, float, float]]] = []

    def fake_fit(states, *, max_generations, evaluate_fn, on_progress=None, **_kwargs):
        del max_generations, evaluate_fn
        dist = cmaes.CmaDistributionSnapshot(mean_log10=(4.0, 8.0, 8.0), sigma=0.2)
        for generation_index in range(6):
            samples = ((4.0, 8.0, 8.0), (4.1, 7.5, 8.5))
            candidates = tuple(
                cmaes.candidates_from_log10_vector(row) for row in samples
            )
            record = cmaes.CmaGenerationRecord(
                generation_index=generation_index,
                structure_idx=0,
                ask_samples_log10=samples,
                candidates=candidates,
                raw_scores=(),
                penalized_fitness=(500.0, 412.0),
                penalty_metadata=(
                    {"penalized": False, "raw_aggregate_sinkhorn": 500.0},
                    {"penalized": False, "raw_aggregate_sinkhorn": 412.0},
                ),
                ask_distribution=dist,
                post_tell_distribution=dist,
            )
            states[0].generations.append(record)
            states[0].completed_generations = generation_index + 1
            if on_progress is not None:
                on_progress(states)
        states[0].status = "fitted"
        states[0].final_mean_log10 = (4.0, 8.0, 8.0)
        states[0].final_evaluation = _evaluation(
            0,
            [cmaes.candidates_from_log10_vector((4.0, 8.0, 8.0))],
            [0.1],
        )
        return cmaes.YoungsModulusCmaFitResult(
            states=dict(states),
            fitted_structure_indices=(0,),
            failed_structure_indices=(),
            generation_waves=6,
            final_mean_batch=None,
        )

    snapshot_direction_pools: list[tuple[int, ...]] = []

    def fake_spawn_snapshot(job, **kwargs):
        del kwargs
        recorded.append((int(job.generation_index), tuple(job.log10_vector)))
        snapshot_direction_pools.append(tuple(job.direction_indices or ()))
        return tmp_path / f"gen_{int(job.generation_index)}.mp4"

    monkeypatch.setattr(module, "BatchedSysIdDataset", lambda _path: dataset)
    monkeypatch.setattr(
        module,
        "list_usable_direction_indices",
        lambda _dataset, _structure_idx, *, include_excluded: [0, 1],
    )
    monkeypatch.setattr(module, "load_ranges", lambda _path: _valid_ranges_dict())
    monkeypatch.setattr(module, "build_sim_config", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(module, "_make_build_env_fn", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(module, "fit_youngs_modulus_structures", fake_fit)
    monkeypatch.setattr(
        module,
        "spawn_isolated_cma_snapshot_video",
        fake_spawn_snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "gt_support_kp_youngs_candidate_from_structure",
        lambda *_a, **_k: cmaes.SupportKpYoungsCandidate(1e4, 1e7, 1e6),
    )
    monkeypatch.setattr(module, "write_cmaes_visualization_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_write_final_mean_overlay", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "execute_cma_wave_evaluation", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "spawn_isolated_cma_wave_evaluation", lambda *_a, **_k: None)

    args = SimpleNamespace(
        dataset="/tmp/gt",
        output=str(output_dir),
        structure_indices=None,
        ranges=None,
        max_envs_per_batch=0,
        seed=None,
        cma_seed=None,
        include_excluded=False,
        use_median=None,
        hold_id_onehot=True,
        pool_directions=True,
        multi_structure_batch=True,
        fail_fast=False,
        overwrite=True,
        resume=False,
        device="cpu",
        settle_substeps=None,
        settle_gravity_ramp=False,
        settle_quiet_every=None,
        show_pull_direction=False,
        viewer="null",
        isolated_eval_waves=True,
        force_magnitude_weight=0.0,
        snapshot_video_every=5,
    )
    module._run(args, argparse.ArgumentParser(), viewer=MagicMock())
    assert [generation for generation, _log10 in recorded] == [0, 5]
    assert snapshot_direction_pools == [(0, 1), (0, 1)]
