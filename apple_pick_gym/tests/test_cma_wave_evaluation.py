"""Tests for process-isolated CMA evaluation waves."""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import pytest

from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs.cma_wave_evaluation import (
    SNAPSHOT_VIDEO_WORKER_MODULE,
    WORKER_MODULE,
    CmaReplayContext,
    CmaSnapshotVideoJob,
    CmaSnapshotVideoResult,
    CmaWaveEvaluationSpec,
    build_cma_replay_context_from_cli,
    make_cma_wave_evaluation_spec,
    reuse_replicated_mujoco_for_cma,
    spawn_isolated_cma_snapshot_video,
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


def test_build_cma_replay_context_forwards_self_collision_flag(tmp_path):
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
        enable_self_collisions=True,
    )
    assert ctx.enable_self_collisions is True


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


def _success_batch() -> cmaes.YoungsModulusBatchEvaluation:
    return cmaes.YoungsModulusBatchEvaluation(
        evaluations={},
        errors={},
        replay_diagnostics=None,
    )


def _write_result_pickle(argv: list[str], batch: cmaes.YoungsModulusBatchEvaluation) -> None:
    result_path = Path(argv[4])
    with result_path.open("wb") as handle:
        pickle.dump(batch, handle)


def test_spawn_maps_segfault_exit_code(tmp_path):
    spec = _sample_spec(tmp_path)

    def crashing_spawn(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 139, stdout="", stderr="segfault")

    with pytest.raises(cmaes.CmaGenerationFailure, match="SIGSEGV"):
        spawn_isolated_cma_wave_evaluation(
            spec,
            output_dir=tmp_path,
            spawn_fn=crashing_spawn,
            max_attempts=1,
        )


def test_spawn_retries_on_segfault_then_succeeds(tmp_path):
    spec = _sample_spec(tmp_path)
    calls: list[int] = []

    def flaky_spawn(argv, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 139, stdout="", stderr="segfault")
        _write_result_pickle(argv, _success_batch())
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    batch = spawn_isolated_cma_wave_evaluation(
        spec,
        output_dir=tmp_path,
        spawn_fn=flaky_spawn,
        max_attempts=3,
    )
    assert isinstance(batch, cmaes.YoungsModulusBatchEvaluation)
    assert len(calls) == 2


def test_spawn_retries_on_missing_result_then_succeeds(tmp_path):
    spec = _sample_spec(tmp_path)
    calls: list[int] = []

    def flaky_spawn(argv, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        _write_result_pickle(argv, _success_batch())
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    batch = spawn_isolated_cma_wave_evaluation(
        spec,
        output_dir=tmp_path,
        spawn_fn=flaky_spawn,
        max_attempts=3,
    )
    assert isinstance(batch, cmaes.YoungsModulusBatchEvaluation)
    assert len(calls) == 2


def test_spawn_retries_on_nonzero_exit_then_succeeds(tmp_path):
    spec = _sample_spec(tmp_path)
    calls: list[int] = []

    def flaky_spawn(argv, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")
        _write_result_pickle(argv, _success_batch())
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    batch = spawn_isolated_cma_wave_evaluation(
        spec,
        output_dir=tmp_path,
        spawn_fn=flaky_spawn,
        max_attempts=3,
    )
    assert isinstance(batch, cmaes.YoungsModulusBatchEvaluation)
    assert len(calls) == 2


def test_spawn_exhausts_max_attempts(tmp_path):
    spec = _sample_spec(tmp_path)
    calls: list[int] = []

    def always_fail(argv, **kwargs):
        calls.append(1)
        return subprocess.CompletedProcess(argv, 139, stdout="", stderr="segfault")

    with pytest.raises(cmaes.CmaGenerationFailure, match="SIGSEGV"):
        spawn_isolated_cma_wave_evaluation(
            spec,
            output_dir=tmp_path,
            spawn_fn=always_fail,
            max_attempts=3,
        )
    assert len(calls) == 3


def test_spawn_retries_on_unexpected_result_type_then_succeeds(tmp_path):
    spec = _sample_spec(tmp_path)
    calls: list[int] = []

    def flaky_spawn(argv, **kwargs):
        calls.append(1)
        result_path = Path(argv[4])
        if len(calls) == 1:
            with result_path.open("wb") as handle:
                pickle.dump({"not": "a batch"}, handle)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        _write_result_pickle(argv, _success_batch())
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    batch = spawn_isolated_cma_wave_evaluation(
        spec,
        output_dir=tmp_path,
        spawn_fn=flaky_spawn,
        max_attempts=3,
    )
    assert isinstance(batch, cmaes.YoungsModulusBatchEvaluation)
    assert len(calls) == 2


def test_spawn_rejects_non_positive_max_attempts(tmp_path):
    spec = _sample_spec(tmp_path)
    with pytest.raises(ValueError, match="max_attempts"):
        spawn_isolated_cma_wave_evaluation(
            spec,
            output_dir=tmp_path,
            spawn_fn=lambda *_a, **_k: subprocess.CompletedProcess([], 0),
            max_attempts=0,
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


def _sample_snapshot_job(tmp_path: Path) -> CmaSnapshotVideoJob:
    output_dir = tmp_path / "cma_out"
    output_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    return CmaSnapshotVideoJob(
        structure_idx=0,
        generation_index=5,
        candidate_index=3,
        log10_vector=(4.0, 8.0, 8.0),
        fitness=412.0,
        output_dir=output_dir,
        replay_context=_sample_context(tmp_path),
        scoring=cmaes.YoungsModulusScoringConfig(n_directions=1),
        dataset_dir=dataset_dir,
        num_directions=8,
        direction_indices=(0, 3, 7),
        max_envs_per_batch=200,
        seed=0,
        include_excluded=False,
        fail_fast=False,
        action_dim=19,
        show_pull_direction=False,
        control_hz=30.0,
    )


def test_snapshot_video_job_pickle_roundtrip(tmp_path):
    job = _sample_snapshot_job(tmp_path)
    restored = pickle.loads(pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored.generation_index == 5
    assert restored.log10_vector == (4.0, 8.0, 8.0)
    assert restored.direction_indices == (0, 3, 7)
    assert restored.replay_context.topology_seed == 42


def test_spawn_isolated_snapshot_video_invokes_worker_module(tmp_path):
    job = _sample_snapshot_job(tmp_path)
    calls: list[list[str]] = []
    video_path = tmp_path / "videos" / "structure_000_gen_005_sample_003.mp4"

    def fake_spawn(argv, **kwargs):
        calls.append(list(argv))
        result_path = Path(argv[4])
        with result_path.open("wb") as handle:
            pickle.dump(
                CmaSnapshotVideoResult(path=video_path, frame_count=12),
                handle,
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    path = spawn_isolated_cma_snapshot_video(
        job,
        output_dir=tmp_path,
        spawn_fn=fake_spawn,
    )
    assert path == video_path
    assert calls
    argv = calls[0]
    module_index = argv.index("-m")
    assert argv[module_index + 1] == SNAPSHOT_VIDEO_WORKER_MODULE


def test_spawn_isolated_snapshot_video_raises_on_zero_frames(tmp_path):
    job = _sample_snapshot_job(tmp_path)

    def zero_frame_spawn(argv, **kwargs):
        result_path = Path(argv[4])
        with result_path.open("wb") as handle:
            pickle.dump(
                CmaSnapshotVideoResult(path=tmp_path / "empty.mp4", frame_count=0),
                handle,
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="0 frames"):
        spawn_isolated_cma_snapshot_video(
            job,
            output_dir=tmp_path,
            spawn_fn=zero_frame_spawn,
        )


def test_spawn_isolated_snapshot_video_raises_on_worker_failure(tmp_path):
    job = _sample_snapshot_job(tmp_path)

    def failing_spawn(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="snapshot video wrote 0 frames"
        )

    with pytest.raises(RuntimeError, match="snapshot video worker"):
        spawn_isolated_cma_snapshot_video(
            job,
            output_dir=tmp_path,
            spawn_fn=failing_spawn,
        )


def test_snapshot_video_worker_main_writes_result(tmp_path, monkeypatch):
    from apple_pick_gym.batched_envs import cma_snapshot_video_worker as worker

    job = _sample_snapshot_job(tmp_path)
    job_path = tmp_path / "job.pkl"
    result_path = tmp_path / "result.pkl"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-empty")

    monkeypatch.setattr(worker, "record_cma_snapshot_video", lambda *_a, **_k: video)
    with job_path.open("wb") as handle:
        pickle.dump(job, handle)
    assert worker.main([str(job_path), str(result_path)]) == 0
    with result_path.open("rb") as handle:
        result = pickle.load(handle)
    assert isinstance(result, CmaSnapshotVideoResult)
    assert result.path == video
    assert result.frame_count == 1


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
