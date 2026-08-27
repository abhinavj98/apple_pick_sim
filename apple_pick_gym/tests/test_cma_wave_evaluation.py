"""Tests for process-isolated CMA evaluation waves."""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import pytest

from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs.cma_wave_evaluation import (
    WORKER_MODULE,
    CmaReplayContext,
    CmaWaveEvaluationSpec,
    build_cma_replay_context_from_cli,
    make_cma_wave_evaluation_spec,
    reuse_replicated_mujoco_for_cma,
    spawn_isolated_cma_wave_evaluation,
)


def _sample_context(tmp_path: Path) -> CmaReplayContext:
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text("{}", encoding="utf-8")
    return build_cma_replay_context_from_cli(
        mode="sim",
        ranges_path=ranges_path,
        topology_seed=42,
        control_hz=60.0,
        device="cpu",
        settle_config={
            "settle_substeps": 8,
            "settle_gravity_ramp": False,
            "settle_quiet_every": 300,
        },
    )


def _sample_spec(tmp_path: Path) -> CmaWaveEvaluationSpec:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    candidate = cmaes.SupportKpYoungsCandidate(1e4, 1e8, 1e7)
    return make_cma_wave_evaluation_spec(
        dataset_dir=dataset_dir,
        structures=[(0, (candidate,))],
        wave_kind="generation",
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=1),
        replay_context=_sample_context(tmp_path),
        num_directions=1,
        direction_indices=(0,),
        max_envs_per_batch=0,
        seed=7,
        include_excluded=False,
        fail_fast=False,
        action_dim=6,
        multi_structure_batch=True,
    )


def test_pickle_spec_roundtrip(tmp_path):
    spec = _sample_spec(tmp_path)
    restored = pickle.loads(pickle.dumps(spec, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored.dataset_dir == spec.dataset_dir
    assert restored.structures == spec.structures
    assert restored.wave_kind == spec.wave_kind
    assert restored.replay_context.reuse_replicated_mujoco is False


def test_replay_context_defaults_reuse_off():
    assert reuse_replicated_mujoco_for_cma(isolated_eval_waves=True) is False
    assert reuse_replicated_mujoco_for_cma(isolated_eval_waves=False) is True


def test_build_cma_replay_context_forwards_reuse_flag(tmp_path):
    ranges_path = tmp_path / "ranges.json"
    ranges_path.write_text("{}", encoding="utf-8")
    ctx = build_cma_replay_context_from_cli(
        mode="sim",
        ranges_path=ranges_path,
        topology_seed=1,
        control_hz=30.0,
        device="cpu",
        settle_config={
            "settle_substeps": 8,
            "settle_gravity_ramp": False,
            "settle_quiet_every": 300,
        },
        reuse_replicated_mujoco=True,
    )
    assert ctx.reuse_replicated_mujoco is True


def test_build_cma_replay_artifacts_forwards_reuse_to_sim_config(tmp_path, monkeypatch):
    from apple_pick_gym.batched_envs import cma_wave_evaluation as wave_mod

    captured: dict[str, bool] = {}

    def fake_build_sim_config(**kwargs):
        captured["reuse"] = bool(kwargs.get("reuse_replicated_mujoco"))
        return object()

    monkeypatch.setattr(wave_mod, "build_sim_config", fake_build_sim_config)
    monkeypatch.setattr(wave_mod, "_make_build_env_fn", lambda **_k: object())
    ctx = build_cma_replay_context_from_cli(
        mode="sim",
        ranges_path=tmp_path / "ranges.json",
        topology_seed=1,
        control_hz=30.0,
        device="cpu",
        settle_config={
            "settle_substeps": 8,
            "settle_gravity_ramp": False,
            "settle_quiet_every": 300,
        },
        reuse_replicated_mujoco=True,
    )
    wave_mod.build_cma_replay_artifacts(ctx, ranges={"primary": {}})
    assert captured["reuse"] is True


def test_spawn_invokes_worker_module(tmp_path):
    spec = _sample_spec(tmp_path)
    calls: list[list[str]] = []

    def fake_spawn(argv, **kwargs):
        calls.append(list(argv))
        batch = cmaes.YoungsModulusBatchEvaluation(
            evaluations={},
            errors={},
            replay_diagnostics=None,
            retried_structures=(),
        )
        result_path = Path(argv[4])
        with result_path.open("wb") as handle:
            pickle.dump(batch, handle)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    batch = spawn_isolated_cma_wave_evaluation(
        spec,
        output_dir=tmp_path,
        spawn_fn=fake_spawn,
    )
    assert isinstance(batch, cmaes.YoungsModulusBatchEvaluation)
    assert calls
    argv = calls[0]
    module_index = argv.index("-m")
    assert argv[module_index + 1] == WORKER_MODULE
    assert Path(argv[module_index + 2]).suffix == ".pkl"
    assert Path(argv[module_index + 3]).suffix == ".pkl"


def test_spawn_maps_segfault_exit_code(tmp_path):
    spec = _sample_spec(tmp_path)

    def crashing_spawn(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 139, stdout="", stderr="segfault")

    with pytest.raises(cmaes.CmaGenerationFailure, match="SIGSEGV"):
        spawn_isolated_cma_wave_evaluation(
            spec,
            output_dir=tmp_path,
            spawn_fn=crashing_spawn,
        )


def test_parser_default_isolated_eval_waves_true(monkeypatch):
    module = pytest.importorskip("apple_pick_gym.batched_examples.example_youngs_modulus_cmaes")
    import argparse

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)
    parser = module._make_parser()
    args = parser.parse_args(["--dataset", "/tmp/gt", "--output", "/tmp/cma"])
    assert args.isolated_eval_waves is True


def test_isolated_batch_matches_inprocess_via_spawn_stub(tmp_path, monkeypatch):
    """Spawn path returns the same batch object as in-process execute."""
    spec = _sample_spec(tmp_path)
    expected = cmaes.YoungsModulusBatchEvaluation(
        evaluations={
            0: cmaes.YoungsModulusEvaluation(
                structure_idx=0,
                gt_candidate=None,
                fixed_secondary_e_pa=None,
                direction_indices=(0,),
                scores=[],
                replay_episodes=[[]],
                applied_params=[],
            )
        },
        errors={},
        replay_diagnostics=None,
        retried_structures=(),
        prepared_structures=1,
    )

    def fake_execute(_spec):
        assert _spec.structures == spec.structures
        return expected

    monkeypatch.setattr(
        "apple_pick_gym.batched_envs.cma_wave_evaluation.execute_cma_wave_evaluation",
        fake_execute,
    )

    def worker_spawn(argv, **kwargs):
        module_index = argv.index("-m")
        job_path = Path(argv[module_index + 2])
        result_path = Path(argv[module_index + 3])
        with job_path.open("rb") as handle:
            loaded = pickle.load(handle)
        batch = fake_execute(loaded)
        with result_path.open("wb") as handle:
            pickle.dump(batch, handle)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    batch = spawn_isolated_cma_wave_evaluation(
        spec,
        output_dir=tmp_path,
        spawn_fn=worker_spawn,
    )
    assert batch.prepared_structures == expected.prepared_structures
    assert batch.evaluations.keys() == expected.evaluations.keys()


def test_isolated_generation_penalized_fitness_matches_inprocess():
    """Penalized fitness is computed in parent from full score payloads."""
    candidate = cmaes.SupportKpYoungsCandidate(1e4, 1e8, 1e7)
    score = cmaes.YoungsModulusCandidateScore(
        candidate_index=0,
        candidate=candidate,
        aggregate_sinkhorn=1.5,
        per_direction_sinkhorn={0: 1.5},
        instability_fraction=0.0,
        disqualified=False,
        disqualification_reason=None,
        rank=1,
        is_gt=False,
        per_direction_mean_hold_force_norm_n={
            0: {"real": 2.0, "sim": 4.0},
        },
    )
    fitness, metadata = cmaes.penalize_youngs_modulus_scores(
        [score],
        force_magnitude_weight=100.0,
    )
    assert fitness[0] > score.aggregate_sinkhorn
    assert metadata[0]["force_magnitude_penalty"] > 0.0
