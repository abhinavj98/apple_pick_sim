"""Process-isolated CMA evaluation waves with full-fidelity pickle IPC."""

from __future__ import annotations

import pickle
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    CmaGenerationFailure,
    SupportKpYoungsCandidate,
    YoungsModulusBatchEvaluation,
    YoungsModulusScoringConfig,
    evaluate_youngs_modulus_candidates,
    evaluate_youngs_modulus_structures,
)
from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
    SysIdReplayCancelled,
)
from apple_pick_gym.batched_envs.real_batched_replay_build import (
    make_real_replay_build_env_fn,
    real_replay_sim_config,
)
from apple_pick_gym.batched_examples.example_youngs_modulus_sys_id import (
    SETTLE_GRAVITY_RAMP,
    SETTLE_QUIET_EVERY,
    SETTLE_SUBSTEPS,
    _make_build_env_fn,
    build_sim_config,
)
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

WORKER_MODULE = "apple_pick_gym.batched_envs.cma_wave_evaluation_worker"
SNAPSHOT_VIDEO_WORKER_MODULE = (
    "apple_pick_gym.batched_envs.cma_snapshot_video_worker"
)
_SIGSEGV_EXIT_CODES = frozenset({-11, 139, 245})
DEFAULT_WAVE_MAX_ATTEMPTS = 5


def reuse_replicated_mujoco_for_cma(*, isolated_eval_waves: bool) -> bool:
    """Reuse USD/MuJoCo only in-process; isolated waves always cold-construct FR3."""
    return not bool(isolated_eval_waves)


@dataclass(frozen=True)
class CmaReplayContext:
    """Serializable inputs to rebuild CMA replay env builders in a worker."""

    mode: str  # "sim" | "vic_pose"
    ranges_path: Path
    topology_seed: int
    control_hz: float
    device: str | None
    settle_substeps: int | None
    settle_gravity_ramp: bool
    settle_quiet_every: int | None
    post_grasp_settle_substeps: int
    # vic_pose-only (optional)
    real_topology_seed: int | None = None
    fruiting_base_pos: tuple[float, float, float] | None = None
    bootstrap_joint_q: tuple[float, ...] | None = None
    episode_meta: dict[str, Any] | None = None
    reuse_replicated_mujoco: bool = False
    enable_self_collisions: bool = False


@dataclass(frozen=True)
class CmaWaveEvaluationSpec:
    """Pickled job for one CMA evaluation wave (generation, re-ask, or final_mean)."""

    dataset_dir: Path
    structures: tuple[tuple[int, tuple[SupportKpYoungsCandidate, ...]], ...]
    wave_kind: str
    scoring: YoungsModulusScoringConfig
    replay_context: CmaReplayContext
    num_directions: int
    direction_indices: tuple[int, ...] | None
    max_envs_per_batch: int
    seed: int | None
    include_excluded: bool
    fail_fast: bool
    action_dim: int
    multi_structure_batch: bool
    on_step: Callable[..., bool] | None = field(default=None, compare=False, hash=False)


@dataclass(frozen=True)
class CmaSnapshotVideoJob:
    """Pickled job for one CMA snapshot MP4 (fresh GL process per clip)."""

    structure_idx: int
    generation_index: int
    candidate_index: int
    log10_vector: tuple[float, float, float]
    fitness: float | None
    output_dir: Path
    replay_context: CmaReplayContext
    scoring: YoungsModulusScoringConfig
    dataset_dir: Path
    num_directions: int
    direction_indices: tuple[int, ...] | None
    max_envs_per_batch: int
    seed: int | None
    include_excluded: bool
    fail_fast: bool
    action_dim: int
    show_pull_direction: bool
    control_hz: float


@dataclass(frozen=True)
class CmaSnapshotVideoResult:
    """Pickled worker result; ``frame_count`` must be > 0."""

    path: Path
    frame_count: int


def build_cma_replay_artifacts(
    context: CmaReplayContext,
    *,
    ranges: dict | None = None,
) -> tuple[Any, Any]:
    """Return ``(build_env_fn, replay_sim_config)`` matching the CMA CLI paths."""
    ranges_path = Path(context.ranges_path)
    loaded_ranges = ranges if ranges is not None else load_ranges(str(ranges_path))
    settle_substeps = (
        SETTLE_SUBSTEPS if context.settle_substeps is None else int(context.settle_substeps)
    )
    settle_config = {
        "settle_substeps": settle_substeps,
        "settle_gravity_ramp": bool(context.settle_gravity_ramp),
        "settle_quiet_every": context.settle_quiet_every,
    }
    if context.mode == "vic_pose":
        if context.episode_meta is None:
            raise ValueError("vic_pose replay context requires episode_meta")
        if context.fruiting_base_pos is None:
            raise ValueError("vic_pose replay context requires fruiting_base_pos")
        real_topology_seed = int(
            context.real_topology_seed
            if context.real_topology_seed is not None
            else context.topology_seed
        )
        build_env_fn = make_real_replay_build_env_fn(
            ranges_path=ranges_path,
            ranges=loaded_ranges,
            topology_seed=real_topology_seed,
            fruiting_base_pos=context.fruiting_base_pos,
            episode_meta=context.episode_meta,
            settle_substeps=settle_substeps,
            settle_quiet_every=context.settle_quiet_every,
            settle_gravity_ramp=bool(context.settle_gravity_ramp),
            post_grasp_settle_substeps=int(context.post_grasp_settle_substeps),
            bootstrap_joint_q=context.bootstrap_joint_q,
            controller_mode="vic_pose",
            control_hz=float(context.control_hz),
            reuse_replicated_mujoco=bool(context.reuse_replicated_mujoco),
            enable_self_collisions=bool(context.enable_self_collisions),
        )
        replay_sim_config = real_replay_sim_config(
            num_envs=1,
            topology_seed=real_topology_seed,
            fruiting_base_pos=context.fruiting_base_pos,
            ranges=loaded_ranges,
            settle_substeps=settle_substeps,
            settle_quiet_every=context.settle_quiet_every,
            settle_gravity_ramp=bool(context.settle_gravity_ramp),
            post_grasp_settle_substeps=int(context.post_grasp_settle_substeps),
            bootstrap_joint_q=context.bootstrap_joint_q,
            controller_mode="vic_pose",
            control_hz=float(context.control_hz),
            reuse_replicated_mujoco=bool(context.reuse_replicated_mujoco),
            enable_self_collisions=bool(context.enable_self_collisions),
        )
        return build_env_fn, replay_sim_config

    build_env_fn = _make_build_env_fn(
        ranges_path=str(ranges_path),
        topology_seed=int(context.topology_seed),
        control_hz=float(context.control_hz),
        device=context.device,
        settle_config=settle_config,
        reuse_replicated_mujoco=bool(context.reuse_replicated_mujoco),
    )
    replay_sim_config = build_sim_config(
        num_envs=1,
        ranges=loaded_ranges,
        device=context.device,
        reuse_replicated_mujoco=bool(context.reuse_replicated_mujoco),
        **settle_config,
    )
    if context.enable_self_collisions:
        import dataclasses

        replay_sim_config = dataclasses.replace(
            replay_sim_config,
            scene=dataclasses.replace(
                replay_sim_config.scene,
                enable_self_collisions=True,
            ),
        )
    return build_env_fn, replay_sim_config


def build_cma_replay_context_from_cli(
    *,
    mode: str,
    ranges_path: Path | str,
    topology_seed: int,
    control_hz: float,
    device: str | None,
    settle_config: Mapping[str, Any],
    post_grasp_settle_substeps: int = 500,
    real_topology_seed: int | None = None,
    fruiting_base_pos: tuple[float, float, float] | None = None,
    bootstrap_joint_q: tuple[float, ...] | None = None,
    episode_meta: Mapping[str, Any] | None = None,
    reuse_replicated_mujoco: bool = False,
    enable_self_collisions: bool = False,
) -> CmaReplayContext:
    """Construct replay context from CMA CLI ``_run`` settle/build inputs."""
    settle_substeps = settle_config.get("settle_substeps")
    settle_gravity_ramp = settle_config.get("settle_gravity_ramp")
    settle_quiet_every = settle_config.get("settle_quiet_every")
    return CmaReplayContext(
        mode=str(mode),
        ranges_path=Path(ranges_path).resolve(),
        topology_seed=int(topology_seed),
        control_hz=float(control_hz),
        device=device,
        settle_substeps=None if settle_substeps is None else int(settle_substeps),
        settle_gravity_ramp=(
            SETTLE_GRAVITY_RAMP if settle_gravity_ramp is None else bool(settle_gravity_ramp)
        ),
        settle_quiet_every=(
            SETTLE_QUIET_EVERY if settle_quiet_every is None else settle_quiet_every
        ),
        post_grasp_settle_substeps=int(post_grasp_settle_substeps),
        real_topology_seed=real_topology_seed,
        fruiting_base_pos=fruiting_base_pos,
        bootstrap_joint_q=bootstrap_joint_q,
        episode_meta=None if episode_meta is None else dict(episode_meta),
        reuse_replicated_mujoco=bool(reuse_replicated_mujoco),
        enable_self_collisions=bool(enable_self_collisions),
    )


def make_cma_wave_evaluation_spec(
    *,
    dataset_dir: Path | str,
    structures: Sequence[tuple[int, Sequence[SupportKpYoungsCandidate]]],
    wave_kind: str,
    scoring: YoungsModulusScoringConfig,
    replay_context: CmaReplayContext,
    num_directions: int,
    direction_indices: Sequence[int] | None,
    max_envs_per_batch: int,
    seed: int | None,
    include_excluded: bool,
    fail_fast: bool,
    action_dim: int,
    multi_structure_batch: bool,
    on_step: Callable[..., bool] | None = None,
) -> CmaWaveEvaluationSpec:
    """Build a wave spec with absolute dataset path and pickled candidate tuples."""
    normalized = tuple(
        (int(structure_idx), tuple(candidates))
        for structure_idx, candidates in structures
    )
    direction_tuple = (
        None
        if direction_indices is None
        else tuple(int(direction_idx) for direction_idx in direction_indices)
    )
    return CmaWaveEvaluationSpec(
        dataset_dir=Path(dataset_dir).resolve(),
        structures=normalized,
        wave_kind=str(wave_kind),
        scoring=scoring,
        replay_context=replay_context,
        num_directions=int(num_directions),
        direction_indices=direction_tuple,
        max_envs_per_batch=int(max_envs_per_batch),
        seed=None if seed is None else int(seed),
        include_excluded=bool(include_excluded),
        fail_fast=bool(fail_fast),
        action_dim=int(action_dim),
        multi_structure_batch=bool(multi_structure_batch),
        on_step=on_step,
    )


def execute_cma_wave_evaluation(spec: CmaWaveEvaluationSpec) -> YoungsModulusBatchEvaluation:
    """Run one evaluation wave in-process (same code path as the CMA CLI)."""
    dataset = BatchedSysIdDataset(str(spec.dataset_dir))
    build_env_fn, replay_sim_config = build_cma_replay_artifacts(spec.replay_context)
    structure_list = list(spec.structures)
    if spec.multi_structure_batch:
        return evaluate_youngs_modulus_structures(
            dataset=dataset,
            structures=structure_list,
            num_directions=int(spec.num_directions),
            build_env_fn=build_env_fn,
            scoring=spec.scoring,
            max_envs_per_batch=int(spec.max_envs_per_batch),
            seed=spec.seed,
            include_excluded=bool(spec.include_excluded),
            fail_fast=bool(spec.fail_fast),
            on_step=spec.on_step,
            replay_sim_config=replay_sim_config,
            action_dim=int(spec.action_dim),
            direction_indices=spec.direction_indices,
        )

    evaluations: dict[int, Any] = {}
    errors: dict[int, str] = {}
    for structure_idx, candidates in structure_list:
        try:
            evaluations[int(structure_idx)] = evaluate_youngs_modulus_candidates(
                dataset=dataset,
                structure_idx=int(structure_idx),
                candidates=list(candidates),
                num_directions=int(spec.num_directions),
                build_env_fn=build_env_fn,
                scoring=spec.scoring,
                max_envs_per_batch=int(spec.max_envs_per_batch),
                seed=spec.seed,
                include_excluded=bool(spec.include_excluded),
                on_step=spec.on_step,
                replay_sim_config=replay_sim_config,
                action_dim=int(spec.action_dim),
                direction_indices=spec.direction_indices,
            )
        except SysIdReplayCancelled:
            raise
        except Exception as exc:
            if bool(spec.fail_fast):
                raise
            errors[int(structure_idx)] = str(exc)
    cand_by_idx = {
        int(structure_idx): candidates for structure_idx, candidates in structure_list
    }
    physical_slots_by_structure = {
        int(idx): len(cand_by_idx[int(idx)]) * len(evaluation.direction_indices)
        for idx, evaluation in evaluations.items()
    }
    return YoungsModulusBatchEvaluation(
        evaluations=evaluations,
        errors=errors,
        replay_diagnostics=None,
        prepared_structures=len(evaluations),
        physical_slots_by_structure=physical_slots_by_structure,
    )


def _wave_jobs_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir) / ".cma_wave_jobs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    import tempfile

    return Path(tempfile.mkdtemp(prefix="cma_wave_jobs_"))


def _worker_failure_detail(
    *,
    completed: subprocess.CompletedProcess[str],
    job_path: Path,
) -> str:
    stderr_tail = (completed.stderr or "").strip()[-2000:]
    if int(completed.returncode) in _SIGSEGV_EXIT_CODES:
        return f"worker segfault (SIGSEGV); job={job_path}"
    return (
        f"worker exit {completed.returncode}; job={job_path}; "
        f"stderr={stderr_tail!r}"
    )


def _run_isolated_cma_wave_once(
    spec: CmaWaveEvaluationSpec,
    *,
    jobs_dir: Path,
    spawn_fn: Callable[..., subprocess.CompletedProcess[str]],
    timeout_s: float | None,
) -> YoungsModulusBatchEvaluation:
    """Spawn one worker subprocess and return the pickled batch."""
    token = uuid.uuid4().hex
    job_path = jobs_dir / f"job_{token}.pkl"
    result_path = jobs_dir / f"result_{token}.pkl"
    try:
        with job_path.open("wb") as handle:
            pickle.dump(spec, handle, protocol=pickle.HIGHEST_PROTOCOL)
        completed = spawn_fn(
            [
                sys.executable,
                "-m",
                WORKER_MODULE,
                str(job_path.resolve()),
                str(result_path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise CmaGenerationFailure(
                "generation_evaluation",
                _worker_failure_detail(completed=completed, job_path=job_path),
            )
        if not result_path.is_file():
            raise CmaGenerationFailure(
                "generation_evaluation",
                f"worker produced no result file; job={job_path}",
            )
        try:
            with result_path.open("rb") as handle:
                batch = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            raise CmaGenerationFailure(
                "generation_evaluation",
                f"worker result unreadable; job={job_path}: {exc}",
            ) from exc
        if not isinstance(batch, YoungsModulusBatchEvaluation):
            raise CmaGenerationFailure(
                "generation_evaluation",
                f"worker returned unexpected type {type(batch)!r}",
            )
        return batch
    finally:
        job_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def spawn_isolated_cma_wave_evaluation(
    spec: CmaWaveEvaluationSpec,
    *,
    output_dir: Path | str | None = None,
    spawn_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_s: float | None = None,
    max_attempts: int = DEFAULT_WAVE_MAX_ATTEMPTS,
) -> YoungsModulusBatchEvaluation:
    """Spawn a worker subprocess, execute the wave, and return the pickled batch."""
    attempts = int(max_attempts)
    if attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
    jobs_dir = _wave_jobs_dir(None if output_dir is None else Path(output_dir))
    runner = spawn_fn or subprocess.run
    for attempt in range(1, attempts + 1):
        try:
            return _run_isolated_cma_wave_once(
                spec,
                jobs_dir=jobs_dir,
                spawn_fn=runner,
                timeout_s=timeout_s,
            )
        except CmaGenerationFailure as exc:
            if attempt >= attempts:
                raise CmaGenerationFailure(
                    exc.stage,
                    f"{exc.message} (failed after {attempts} attempt(s))",
                ) from exc
            print(
                f"warning: isolated CMA wave {spec.wave_kind!r} failed on attempt "
                f"{attempt}/{attempts}: {exc.message}; retrying",
                file=sys.stderr,
            )
    raise CmaGenerationFailure(
        "generation_evaluation",
        f"worker failed after {attempts} attempt(s)",
    )


def spawn_isolated_cma_snapshot_video(
    job: CmaSnapshotVideoJob,
    *,
    output_dir: Path | str | None = None,
    spawn_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    timeout_s: float | None = None,
) -> Path:
    """Spawn a worker subprocess that records one snapshot MP4 and returns its path.

    Each snapshot gets a fresh process (and GL context). Reusing ViewerGL in the
    CMA parent after ``close()`` leaves pyglet/CUDA-GL in a state where later
    captures write 0 frames.
    """
    jobs_dir = _wave_jobs_dir(None if output_dir is None else Path(output_dir))
    runner = spawn_fn or subprocess.run
    token = uuid.uuid4().hex
    job_path = jobs_dir / f"snapshot_job_{token}.pkl"
    result_path = jobs_dir / f"snapshot_result_{token}.pkl"
    try:
        with job_path.open("wb") as handle:
            pickle.dump(job, handle, protocol=pickle.HIGHEST_PROTOCOL)
        completed = runner(
            [
                sys.executable,
                "-m",
                SNAPSHOT_VIDEO_WORKER_MODULE,
                str(job_path.resolve()),
                str(result_path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if int(completed.returncode) != 0:
            stderr_tail = (completed.stderr or "").strip()[-2000:]
            raise RuntimeError(
                "snapshot video worker failed "
                f"(exit {completed.returncode}); stderr={stderr_tail!r}"
            )
        if not result_path.is_file():
            raise RuntimeError(
                f"snapshot video worker produced no result file; job={job_path}"
            )
        try:
            with result_path.open("rb") as handle:
                result = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"snapshot video worker result unreadable; job={job_path}: {exc}"
            ) from exc
        if not isinstance(result, CmaSnapshotVideoResult):
            raise RuntimeError(
                f"snapshot video worker returned unexpected type {type(result)!r}"
            )
        if int(result.frame_count) <= 0:
            raise RuntimeError(f"snapshot video wrote 0 frames ({result.path})")
        return Path(result.path)
    finally:
        job_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)


def wrap_isolated_cma_evaluate_fn(
    *,
    dataset_dir: Path | str,
    scoring: YoungsModulusScoringConfig,
    replay_context: CmaReplayContext,
    num_directions: int,
    direction_indices: Sequence[int] | None,
    max_envs_per_batch: int,
    seed: int | None,
    include_excluded: bool,
    fail_fast: bool,
    action_dim: int,
    multi_structure_batch: bool,
    output_dir: Path | str,
    on_step: Callable[..., bool] | None = None,
    spawn_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    max_attempts: int = DEFAULT_WAVE_MAX_ATTEMPTS,
    counter_callback: Callable[
        [YoungsModulusBatchEvaluation, list[tuple[int, tuple[Any, ...]]], str],
        None,
    ]
    | None = None,
) -> Callable[..., YoungsModulusBatchEvaluation]:
    """Return an ``evaluate_fn`` compatible with ``fit_youngs_modulus_structures``."""

    def evaluate_fn(
        *,
        structures: Sequence[tuple[int, Sequence[SupportKpYoungsCandidate]]],
        wave_kind: str = "generation",
        **_kwargs: Any,
    ) -> YoungsModulusBatchEvaluation:
        structure_list = [
            (int(structure_idx), tuple(candidates))
            for structure_idx, candidates in structures
        ]
        spec = make_cma_wave_evaluation_spec(
            dataset_dir=dataset_dir,
            structures=structure_list,
            wave_kind=str(wave_kind),
            scoring=scoring,
            replay_context=replay_context,
            num_directions=int(num_directions),
            direction_indices=direction_indices,
            max_envs_per_batch=int(max_envs_per_batch),
            seed=seed,
            include_excluded=bool(include_excluded),
            fail_fast=bool(fail_fast),
            action_dim=int(action_dim),
            multi_structure_batch=bool(multi_structure_batch),
            on_step=on_step,
        )
        batch = spawn_isolated_cma_wave_evaluation(
            spec,
            output_dir=output_dir,
            spawn_fn=spawn_fn,
            max_attempts=int(max_attempts),
        )
        if counter_callback is not None:
            counter_callback(batch, structure_list, str(wave_kind))
        return batch

    return evaluate_fn
