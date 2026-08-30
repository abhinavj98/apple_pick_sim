"""Subprocess worker: record one CMA snapshot MP4 in a fresh GL process."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from apple_pick_gym.batched_envs.cma_wave_evaluation import (
    CmaSnapshotVideoJob,
    CmaSnapshotVideoResult,
)
from apple_pick_gym.batched_examples.example_youngs_modulus_cmaes import (
    CmaSnapshotSample,
    record_cma_snapshot_video,
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: python -m apple_pick_gym.batched_envs.cma_snapshot_video_worker "
            "JOB.pkl RESULT.pkl",
            file=sys.stderr,
        )
        return 2
    job_path = Path(args[0])
    result_path = Path(args[1])
    with job_path.open("rb") as handle:
        job = pickle.load(handle)
    if not isinstance(job, CmaSnapshotVideoJob):
        print(f"expected CmaSnapshotVideoJob, got {type(job)!r}", file=sys.stderr)
        return 3
    sample = CmaSnapshotSample(
        structure_idx=int(job.structure_idx),
        generation_index=int(job.generation_index),
        candidate_index=int(job.candidate_index),
        log10_vector=tuple(float(v) for v in job.log10_vector),
        fitness=job.fitness,
    )
    path = record_cma_snapshot_video(
        sample,
        output_dir=Path(job.output_dir),
        replay_context=job.replay_context,
        scoring=job.scoring,
        dataset_dir=job.dataset_dir,
        num_directions=int(job.num_directions),
        direction_indices=job.direction_indices,
        max_envs_per_batch=int(job.max_envs_per_batch),
        seed=job.seed,
        include_excluded=bool(job.include_excluded),
        fail_fast=bool(job.fail_fast),
        action_dim=int(job.action_dim),
        show_pull_direction=bool(job.show_pull_direction),
        control_hz=float(job.control_hz),
    )
    frames = 1 if path.is_file() and path.stat().st_size > 0 else 0
    with result_path.open("wb") as handle:
        pickle.dump(
            CmaSnapshotVideoResult(path=path, frame_count=int(frames)),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
