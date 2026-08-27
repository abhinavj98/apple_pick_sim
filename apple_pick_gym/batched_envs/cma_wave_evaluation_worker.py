"""Subprocess worker entry point for isolated CMA evaluation waves."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from apple_pick_gym.batched_envs.cma_wave_evaluation import (
    CmaWaveEvaluationSpec,
    execute_cma_wave_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            f"usage: python -m apple_pick_gym.batched_envs.cma_wave_evaluation_worker "
            f"JOB.pkl RESULT.pkl",
            file=sys.stderr,
        )
        return 2
    job_path = Path(args[0])
    result_path = Path(args[1])
    with job_path.open("rb") as handle:
        spec = pickle.load(handle)
    if not isinstance(spec, CmaWaveEvaluationSpec):
        print(f"expected CmaWaveEvaluationSpec, got {type(spec)!r}", file=sys.stderr)
        return 3
    batch = execute_cma_wave_evaluation(spec)
    with result_path.open("wb") as handle:
        pickle.dump(batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
