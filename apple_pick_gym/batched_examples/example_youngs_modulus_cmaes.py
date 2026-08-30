"""Dataset-driven support-k_p + Young's-modulus CMA-ES fit entry point.

Fits a 3-vector ``(support_kp, E_spur, E_stem)`` — support joint k_p (shared
angular+linear; support zeta from dataset ``joint_damping_ratio``) is free while
spur/stem Young's modulus stay free;
primary E is fixed from ground truth. Runs one independent bounded pycma
optimizer per selected structure, advances active optimizers in synchronized
generation waves through fused structure x population x direction replay,
then explicitly scores each stopped distribution mean. Writes
``<output>/cmaes_report.json`` atomically and final-mean overlays at
``structure_XXX/youngs_modulus_overlay.html``.

Run from repo root::

    uv run python apple_pick_gym/batched_examples/example_youngs_modulus_cmaes.py \\
        --dataset /tmp/batched_sysid_dataset \\
        --output /tmp/youngs_cmaes

Edit ``CMA_SEARCH_PARAMS`` below to change optimizer search knobs (mean, sigma,
population, generations, bounds). ``--cma-seed`` and ``--max-generations``
override ``CMA_SEARCH_PARAMS["cma_seed"]`` and ``max_generations`` for
operational runs without editing the module defaults.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

import newton.examples
import newton.viewer

from apple_pick_gym.batched_examples.example_batched_sysid_mmd_grid import (
    parse_comma_separated_ints,
)
from apple_pick_gym.batched_examples import example_youngs_modulus_sys_id as _grid
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import list_usable_direction_indices
from apple_pick_sim.system_id.holdout_gates import (
    DIRECTION_SPLIT_SEED,
    choose_direction_split,
)
from apple_pick_gym.batched_envs.holdout_evaluation import run_holdout_evaluation
from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    CMA_OPTIMIZER_CHECKPOINT_FILENAME,
    CmaGenerationFailure,
    StructureCmaState,
    SupportKpYoungsCandidate,
    YoungsModulusBatchEvaluation,
    YoungsModulusCandidate,
    YoungsModulusScoringConfig,
    aggregate_fitted_youngs_modulus_stats,
    apply_cma_checkpoint_to_states,
    candidates_from_log10_vector,
    create_structure_cma_optimizer,
    derive_structure_cma_seeds,
    dump_cma_optimizer_checkpoint,
    evaluate_youngs_modulus_candidates,
    evaluate_youngs_modulus_structures,
    extract_support_kp_youngs_modulus_cma_bounds,
    fit_youngs_modulus_structures,
    gt_support_kp_youngs_candidate_from_structure,
    load_cma_optimizer_checkpoint,
    normalize_search_bounds_log10,
    resolve_initial_mean_log10,
    structure_cma_report_snapshot,
    to_strict_jsonable,
    validate_initial_sigma_log10,
    validate_max_sigma_log10,
)
from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
    SysIdReplayCancelled,
)
from apple_pick_gym.batched_envs.cma_wave_evaluation import (
    DEFAULT_WAVE_MAX_ATTEMPTS,
    CmaSnapshotVideoJob,
    build_cma_replay_context_from_cli,
    execute_cma_wave_evaluation,
    make_cma_wave_evaluation_spec,
    reuse_replicated_mujoco_for_cma,
    spawn_isolated_cma_snapshot_video,
    spawn_isolated_cma_wave_evaluation,
)
from apple_pick_gym.batched_envs.real_batched_replay_build import (
    bootstrap_joint_q_from_episode_metadata,
    check_action_semantics,
    control_hz_from_episode_metadata,
    dataset_declares_vic_pose,
    fruiting_base_pos_from_episode_metadata,
    make_real_replay_build_env_fn,
    real_replay_sim_config,
)
from apple_pick_gym.youngs_modulus_cmaes_viz import write_cmaes_visualization_bundle
from apple_pick_gym.youngs_modulus_overlay_viz import (
    overlay_episodes_from_replay_evaluation,
    write_youngs_modulus_overlay_html,
)
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
from robot_replay.gl_video_recorder import GlVideoRecorder

# Re-export grid helpers used by tests and shared replay setup.
SETTLE_GRAVITY_RAMP = _grid.SETTLE_GRAVITY_RAMP
SETTLE_QUIET_EVERY = _grid.SETTLE_QUIET_EVERY
SETTLE_SUBSTEPS = _grid.SETTLE_SUBSTEPS
build_sim_config = _grid.build_sim_config
_make_build_env_fn = _grid._make_build_env_fn
_resolve_structure_indices = _grid._resolve_structure_indices
_resolve_n_holds = _grid._resolve_n_holds
_resolve_n_directions = _grid._resolve_n_directions
_collection_control_hz = _grid._collection_control_hz
_settle_config_kwargs = _grid._settle_config_kwargs
_render_frame = _grid._render_frame
_positive_int = _grid._positive_int
require_gl_frame_capture = _grid.require_gl_frame_capture
make_grid_on_step = _grid.make_grid_on_step


# Shared cancel type from the multi-replay dependency layer.
ViewerCancelled = SysIdReplayCancelled

# Sole source of truth for CMA search knobs (edit here; not exposed on CLI).
# initial_mean_log10: start mean in log10 [support_kp, E_spur, E_stem], or
# "bounds_midpoint" to derive midpoints from the loaded fixture (spur/stem)
# plus the absolute support_kp safety box.
# search_bounds_log10: None = unbounded search; or
#   {"lower": [kp, s, t], "upper": [kp, s, t]} in log10.
# Support k_p: absolute safety box (not a per-structure DR quantity / fixture
# ε-band) — 100 .. 1e6 N/m or N*m/rad (log10 2-6). Spur/stem E: absolute
# 0.1-100 GPa (log10 8-11), same box as before. Init from the search box
# midpoint, never from ground truth. Sim-sim box is 0.1–100 GPa; real vic_pose
# uses spur/stem 10 MPa–100 GPa (log10 7–11) via _effective_search_bounds_log10.
_CMA_SEARCH_LOG10_LOWER = [2.0, 8.0, 8.0]  # support_kp 1e2, spur/stem 0.1 GPa
_CMA_SEARCH_LOG10_UPPER = [6.0, 11.0, 11.0]  # support_kp 1e6, spur/stem 100 GPa


_REAL_CMA_SEARCH_LOG10_LOWER = [2.0, 8.0, 6.0]  # support_kp 1e2, spur/stem 10 MPa
_REAL_CMA_SEARCH_LOG10_UPPER = [6.0, 11.0, 8.0]  # support_kp 1e6, spur/stem 100 GPa
_CMA_MEAN_LOG10 = [_CMA_SEARCH_LOG10_LOWER[i] + 0.5 * (_CMA_SEARCH_LOG10_UPPER[i] - _CMA_SEARCH_LOG10_LOWER[i]) for i in range(3)]
_REAL_CMA_MEAN_LOG10 = [
    _REAL_CMA_SEARCH_LOG10_LOWER[i]
    + 0.5 * (_REAL_CMA_SEARCH_LOG10_UPPER[i] - _REAL_CMA_SEARCH_LOG10_LOWER[i])
    for i in range(3)
]
CMA_SEARCH_PARAMS: dict[str, Any] = {
    "initial_mean_log10": list(_CMA_MEAN_LOG10),
    "initial_sigma_log10": 0.2,
    "max_sigma_log10": 0.5,
    "population_size": 20,
    "max_generations": 20,
    "cma_seed": 56,
    "search_bounds_log10": {
        "lower": _CMA_SEARCH_LOG10_LOWER,
        "upper": _CMA_SEARCH_LOG10_UPPER,
    },
}


def _effective_search_bounds_log10(
    mode: str,
    search: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Sim-sim uses CMA_SEARCH_PARAMS; vic_pose uses spur/stem 10 MPa–100 GPa."""
    if mode == "vic_pose":
        return (
            tuple(float(x) for x in _REAL_CMA_SEARCH_LOG10_LOWER),
            tuple(float(x) for x in _REAL_CMA_SEARCH_LOG10_UPPER),
        )
    raw = search.get("search_bounds_log10")
    return normalize_search_bounds_log10(raw)


def _effective_initial_mean_log10(
    mode: str,
    search: dict[str, Any],
    bounds: Any,
) -> list[float]:
    """Sim-sim uses CMA_SEARCH_PARAMS; vic_pose starts 1.5 decades softer in E."""
    raw = _REAL_CMA_MEAN_LOG10 if mode == "vic_pose" else search["initial_mean_log10"]
    return list(resolve_initial_mean_log10(raw, bounds))


def _require_ft_wrist_lpf_per_structure(
    dataset: Any,
    structure_indices: list[int],
    *,
    include_excluded: bool = False,
) -> None:
    """Real CMA scores convert-time ``ft_wrist_lpf``; refuse bags that omit it."""
    import numpy as np

    missing: list[int] = []
    for structure_idx in structure_indices:
        entries = [
            ep
            for ep in dataset.episode_entries()
            if int(ep.get("structure_idx", -1)) == int(structure_idx)
        ]
        if entries:
            direction_idxs = list_usable_direction_indices(
                dataset,
                int(structure_idx),
                include_excluded=bool(include_excluded),
            )
        else:
            collection = {}
            manifest = getattr(dataset, "manifest", None)
            if isinstance(manifest, dict):
                raw = manifest.get("collection")
                if isinstance(raw, dict):
                    collection = raw
            resolved = _resolve_n_directions(dataset, collection)
            num_directions = (
                int(resolved) if resolved is not None and int(resolved) >= 1 else 1
            )
            direction_idxs = list(range(num_directions))
        for direction_idx in direction_idxs:
            arrays = dataset.load_episode_obs_arrays(
                int(structure_idx), int(direction_idx)
            )
            lpf = arrays.get("ft_wrist_lpf") if isinstance(arrays, dict) else None
            if lpf is None or np.asarray(lpf).size == 0:
                missing.append(int(structure_idx))
                break
    if missing:
        raise SystemExit(
            "vic_pose CMA requires convert-time ft_wrist_lpf on each selected "
            f"structure; missing on {missing}. Re-run convert so the LPF "
            "column is written."
        )


def _resolve_holdout_direction_split(
    args: Any,
    dataset: BatchedSysIdDataset,
    structure_indices: list[int],
    *,
    include_excluded: bool,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Return (train, val) disk direction indices, or (None, None) if no holdout."""
    explicit_train = getattr(args, "direction_indices", None)
    explicit_val = getattr(args, "val_direction_indices", None)
    split_seed = getattr(args, "direction_split_seed", None)

    if (explicit_train is None) ^ (explicit_val is None):
        raise SystemExit(
            "--direction-indices and --val-direction-indices must be passed together"
        )

    holdout = split_seed is not None or explicit_train is not None
    if not holdout:
        return None, None

    structure_idx = int(structure_indices[0])
    disk_dirs = tuple(
        list_usable_direction_indices(
            dataset,
            structure_idx,
            include_excluded=bool(include_excluded),
        )
    )
    if len(disk_dirs) != 8:
        raise SystemExit(
            "holdout mode requires exactly 8 usable direction episodes on disk; "
            f"got {len(disk_dirs)}"
        )

    if explicit_train is not None:
        train = tuple(sorted(int(d) for d in explicit_train))
        val = tuple(sorted(int(d) for d in explicit_val))
        if not train or not val:
            raise SystemExit("train and val direction splits must be non-empty")
        if set(train) & set(val):
            raise SystemExit("train and val direction indices must be disjoint")
        disk_set = set(disk_dirs)
        if not set(train).issubset(disk_set) or not set(val).issubset(disk_set):
            raise SystemExit(
                "explicit direction indices must be a subset of usable disk directions"
            )
        return train, val

    train, val = choose_direction_split(disk_dirs, seed=int(split_seed))
    return train, val


def accumulate_cma_batch_counters(
    counters: dict[str, int],
    batch: YoungsModulusBatchEvaluation,
    *,
    structures: list[tuple[int, tuple[YoungsModulusCandidate, ...]]],
    wave_kind: str,
    num_directions: int,
) -> None:
    """Update logical/physical CMA counters for one evaluated wave.

    Batch ``physical_env_slots`` uses exact fused diagnostics totals when
    present; otherwise sums attributed per-structure slots (scalar-only).
    Per-structure slots prefer ``batch.physical_slots_by_structure`` (prepared
    usable directions, including scoring-failed-but-replayed), else
    ``len(candidates) * len(evaluation.direction_indices)``. Manifest/CLI
    ``num_directions`` is not used for physical attribution.
    """
    del num_directions  # kept for call-site compatibility; not for slot math
    final_mean = str(wave_kind) == "final_mean"
    cand_counts = {
        int(structure_idx): len(candidates)
        for structure_idx, candidates in structures
    }
    for structure_idx, candidates in structures:
        n = len(candidates)
        counters["replay_candidate_evaluations"] += n
        counters[f"replay_candidate_evaluations:{structure_idx}"] = (
            counters.get(f"replay_candidate_evaluations:{structure_idx}", 0) + n
        )
        if final_mean:
            counters["final_mean_evaluations"] += 1
            counters[f"final_mean_evaluations:{structure_idx}"] = (
                counters.get(f"final_mean_evaluations:{structure_idx}", 0) + 1
            )

    slots_by_structure = {
        int(idx): int(slots)
        for idx, slots in batch.physical_slots_by_structure.items()
    }
    if not slots_by_structure:
        for structure_idx, evaluation in batch.evaluations.items():
            n_cand = cand_counts.get(int(structure_idx), len(evaluation.scores))
            slots_by_structure[int(structure_idx)] = int(n_cand) * len(
                evaluation.direction_indices
            )

    diagnostics = batch.replay_diagnostics
    if diagnostics is not None:
        counters["physical_env_slots"] += int(diagnostics.flattened_envs)
        for structure_idx, planned in slots_by_structure.items():
            counters[f"physical_env_slots:{structure_idx}"] = (
                counters.get(f"physical_env_slots:{structure_idx}", 0) + int(planned)
            )
    elif slots_by_structure:
        for structure_idx, planned in slots_by_structure.items():
            planned_i = int(planned)
            counters["physical_env_slots"] += planned_i
            counters[f"physical_env_slots:{structure_idx}"] = (
                counters.get(f"physical_env_slots:{structure_idx}", 0) + planned_i
            )


def _resolve_ranges_path(
    args: Any,
    collection: dict[str, Any],
    *,
    cwd: Path | None = None,
) -> Path:
    """Resolve the authoritative ranges fixture for bounds and replay knobs.

    Prefer ``--ranges``, otherwise ``collection.ranges_path``. Relative paths
    resolve from the process CWD. Never falls back to an unrelated default
    fixture.
    """
    base = Path.cwd() if cwd is None else Path(cwd)
    raw = getattr(args, "ranges", None)
    if raw is None or str(raw).strip() == "":
        raw = collection.get("ranges_path")
    if raw is None or str(raw).strip() == "":
        raise SystemExit(
            "ranges fixture required: pass --ranges or set collection.ranges_path "
            "in the dataset manifest"
        )
    path = Path(str(raw))
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"ranges fixture not found: {path}")
    return path


def _clear_cma_owned_artifacts(
    output_dir: Path,
    *,
    structure_indices: list[int] | None = None,
) -> None:
    """Remove CMA-owned report/temp and selected-structure overlay targets."""
    report = output_dir / "cmaes_report.json"
    if report.exists():
        report.unlink()
    checkpoint = output_dir / CMA_OPTIMIZER_CHECKPOINT_FILENAME
    if checkpoint.exists():
        checkpoint.unlink()
    holdout_report = output_dir / "holdout_report.json"
    if holdout_report.exists():
        holdout_report.unlink()
    for path in output_dir.glob(".cmaes_report.json.*.tmp"):
        path.unlink(missing_ok=True)
    for path in output_dir.glob(".holdout_report.json.*.tmp"):
        path.unlink(missing_ok=True)
    videos_dir = output_dir / "videos"
    if videos_dir.is_dir():
        for video in videos_dir.glob("structure_*.mp4"):
            video.unlink(missing_ok=True)
        if not any(videos_dir.iterdir()):
            videos_dir.rmdir()
    if structure_indices is None:
        return
    for structure_idx in structure_indices:
        structure_dir = output_dir / f"structure_{int(structure_idx):03d}"
        overlay = structure_dir / "youngs_modulus_overlay.html"
        if overlay.exists():
            overlay.unlink()
        holdout_dir = structure_dir / "holdout"
        if holdout_dir.is_dir():
            for overlay_path in holdout_dir.glob("direction_*.html"):
                overlay_path.unlink(missing_ok=True)
            if not any(holdout_dir.iterdir()):
                holdout_dir.rmdir()


def _write_cmaes_report_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strict = to_strict_jsonable(payload)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(strict, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _gt_error_diagnostics(
    final_mean_log10: tuple[float, float, float] | None,
    gt: SupportKpYoungsCandidate | None,
) -> dict[str, Any] | None:
    """Support k_p error vs ``gt_support_kp_from_dataset``; spur/stem vs GT E."""
    if final_mean_log10 is None or gt is None:
        return None
    mean = candidates_from_log10_vector(final_mean_log10)
    log10_error: dict[str, float] = {}
    relative_error: dict[str, float] = {}
    for segment in ("support_kp", "spur", "stem"):
        mean_value = float(getattr(mean, segment))
        gt_value = float(getattr(gt, segment))
        mean_log10 = math.log10(mean_value)
        gt_log10 = math.log10(gt_value)
        log10_error[segment] = float(mean_log10 - gt_log10)
        relative_error[segment] = float(abs(mean_value - gt_value) / abs(gt_value))
    return {"log10_error": log10_error, "relative_error": relative_error}


def _build_cmaes_report_payload(
    states: dict[int, StructureCmaState],
    *,
    dataset: str,
    output: str,
    ranges_path: str,
    base_seed: int,
    initial_mean_log10: tuple[float, float, float] | list[float] | None,
    initial_sigma_log10: float,
    max_sigma_log10: float | None = None,
    max_generations: int,
    scoring: YoungsModulusScoringConfig,
    command_status: str,
    command_error: str | None = None,
    counter_totals: dict[str, int] | None = None,
    timing: dict[str, Any] | None = None,
    population_size: int | None = None,
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
    force_magnitude_weight: float = 0.0,
    isolated_eval_waves: bool = True,
    wave_max_attempts: int = DEFAULT_WAVE_MAX_ATTEMPTS,
) -> dict[str, Any]:
    counters = counter_totals or {}
    structures: dict[str, Any] = {}
    fitted_candidates: list[YoungsModulusCandidate] = []
    gt_candidates: list[YoungsModulusCandidate] = []
    failed_count = 0
    for structure_idx, state in states.items():
        snapshot = structure_cma_report_snapshot(
            state,
            base_seed=int(base_seed),
            initial_sigma_log10=float(initial_sigma_log10),
            replay_candidate_evaluations=int(
                counters.get(f"replay_candidate_evaluations:{structure_idx}", 0)
            ),
            final_mean_evaluations=int(
                counters.get(f"final_mean_evaluations:{structure_idx}", 0)
            ),
            physical_env_slots=int(
                counters.get(f"physical_env_slots:{structure_idx}", 0)
            ),
        )
        gt_diag = _gt_error_diagnostics(state.final_mean_log10, state.gt_candidate)
        if gt_diag is not None:
            snapshot["gt_diagnostics"] = gt_diag
        structures[str(int(structure_idx))] = snapshot
        if state.status == "fitted" and state.final_mean_log10 is not None:
            fitted_candidates.append(
                candidates_from_log10_vector(state.final_mean_log10)
            )
            if state.gt_candidate is not None:
                gt_candidates.append(state.gt_candidate)
        elif state.status == "failed":
            failed_count += 1

    aggregate = aggregate_fitted_youngs_modulus_stats(
        fitted_candidates=fitted_candidates,
        gt_candidates=gt_candidates,
        requested_count=len(states),
        failed_count=failed_count,
    )
    payload: dict[str, Any] = {
        "dataset": str(dataset),
        "output": str(output),
        "ranges_path": str(ranges_path),
        "cma": {
            "base_seed": int(base_seed),
            "initial_mean_log10": list(initial_mean_log10)
            if initial_mean_log10 is not None
            else None,
            "initial_sigma_log10": float(initial_sigma_log10),
            "max_sigma_log10": None
            if max_sigma_log10 is None
            else float(max_sigma_log10),
            "max_generations": int(max_generations),
            "population_size": population_size,
            "search_bounds_log10": None
            if search_bounds_log10 is None
            else {
                "lower": list(search_bounds_log10[0]),
                "upper": list(search_bounds_log10[1]),
            },
            "search_params_source": "CMA_SEARCH_PARAMS",
        },
        "scoring": {
            "hold_aggregation": scoring.hold_aggregation,
            "use_median": bool(scoring.use_median),
            "hold_id_onehot": bool(scoring.hold_id_onehot),
            "pool_directions": bool(scoring.pool_directions),
            "n_holds": scoring.n_holds,
            "n_directions": scoring.n_directions,
            "device": scoring.device,
            "include_delta": bool(scoring.include_delta),
            "categorical_weight": float(scoring.categorical_weight),
            "force_magnitude_weight": float(force_magnitude_weight),
            "isolated_eval_waves": bool(isolated_eval_waves),
            "wave_max_attempts": int(wave_max_attempts),
        },
        "command_status": str(command_status),
        "structures": structures,
        "aggregate": aggregate,
        "counters": {
            "replay_candidate_evaluations": int(
                counters.get("replay_candidate_evaluations", 0)
            ),
            "final_mean_evaluations": int(counters.get("final_mean_evaluations", 0)),
            "physical_env_slots": int(counters.get("physical_env_slots", 0)),
        },
        "timing": dict(timing or {}),
    }
    if command_error is not None:
        payload["command_error"] = str(command_error)
    return payload


def _write_final_mean_overlay(
    state: StructureCmaState,
    *,
    output_dir: Path,
) -> None:
    if state.final_evaluation is None:
        raise RuntimeError("fitted structure missing final_evaluation for overlay")
    structure_dir = output_dir / f"structure_{int(state.structure_idx):03d}"
    structure_dir.mkdir(parents=True, exist_ok=True)
    overlay_eps = overlay_episodes_from_replay_evaluation(
        state.final_evaluation,
        [0],
    )
    write_youngs_modulus_overlay_html(
        overlay_eps,
        structure_dir / "youngs_modulus_overlay.html",
        max_overlay_candidates=1,
        title=f"Young's modulus CMA overlay — structure {int(state.structure_idx)}",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value!r}")
    return parsed


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Directory containing batched_sysid_v1 manifest.json and episodes/.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory for cmaes_report.json and final-mean overlays.",
    )
    p.add_argument(
        "--structure-indices",
        type=parse_comma_separated_ints,
        default=None,
        help="Comma-separated structure indices (default: all in manifest).",
    )
    p.add_argument(
        "--ranges",
        type=str,
        default=None,
        help=(
            "Ranges fixture for CMA bounds and replay sim_build knobs "
            "(default: dataset collection.ranges_path)."
        ),
    )
    p.add_argument(
        "--cma-seed",
        type=int,
        default=None,
        help=(
            "Application CMA base seed (overrides CMA_SEARCH_PARAMS['cma_seed']; "
            "not passed directly to pycma)."
        ),
    )
    p.add_argument(
        "--max-generations",
        type=_positive_int,
        default=None,
        help=(
            "CMA generation cap (overrides CMA_SEARCH_PARAMS['max_generations'])."
        ),
    )
    p.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include manifest episodes marked excluded (debug only).",
    )
    p.add_argument(
        "--max-envs-per-batch",
        type=int,
        default=_grid.MAX_ENVS_PER_BATCH,
        help=(
            "Chunk candidates so chunk_size*num_directions <= this "
            f"(default {_grid.MAX_ENVS_PER_BATCH}; 0 = no chunking)."
        ),
    )
    p.add_argument(
        "--multi-structure-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replay compatible structures in one flattened GPU batch "
            "(disable for parity/debug baseline)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Replay RNG seed (default: manifest collection.seed).",
    )
    p.add_argument(
        "--controller-mode",
        choices=("vic", "vic_pose"),
        default=None,
        help="Replay controller mode (default: infer vic_pose from dataset, else vic).",
    )
    p.add_argument(
        "--direction-split-seed",
        nargs="?",
        type=int,
        const=DIRECTION_SPLIT_SEED,
        default=None,
        help=(
            "Holdout train/val split seed (default 17 when flag present without value). "
            "Requires eight usable direction episodes on disk."
        ),
    )
    p.add_argument(
        "--direction-indices",
        type=parse_comma_separated_ints,
        default=None,
        help="Explicit train direction indices (requires --val-direction-indices).",
    )
    p.add_argument(
        "--val-direction-indices",
        type=parse_comma_separated_ints,
        default=None,
        help="Explicit validation direction indices (requires --direction-indices).",
    )
    p.add_argument(
        "--use-median",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(deprecated) Use --hold-aggregation median instead.",
    )
    p.add_argument(
        "--hold-aggregation",
        choices=["median", "mean", "none"],
        default="none",
        help=(
            "Hold state aggregation for Sinkhorn scoring. Default: none "
            "(quasi-static level bags; use mean/median for legacy transition rows)."
        ),
    )
    p.add_argument(
        "--force-magnitude-weight",
        type=float,
        default=0.0,
        help=(
            "Weight λ on mean |log(sim‖F‖/real‖F‖)| added to aggregate Sinkhorn "
            "fitness (0 disables). Default: 0 (Sinkhorn-only; opt in for legacy runs)."
        ),
    )
    p.add_argument(
        "--include-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append the Δs half to each scored transition row. Default: off "
            "(level bags; pass --include-delta for legacy [s, Δs] rows)."
        ),
    )
    p.add_argument(
        "--categorical-weight",
        type=float,
        default=30.0,
        help=(
            "Reciprocal scale for hold/direction one-hot columns in Sinkhorn "
            "normalization (higher anchors per-hold/per-direction transport). "
            "Default: 30."
        ),
    )
    p.add_argument(
        "--hold-id-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append per-hold one-hot identity to Wasserstein transition features.",
    )
    p.add_argument(
        "--pool-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pool transition bags across directions for Sinkhorn scoring.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first structure error instead of recording it.",
    )
    p.add_argument(
        "--isolated-eval-waves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run each CMA evaluation wave in a fresh subprocess (default: on). "
            "Use --no-isolated-eval-waves to reuse USD/MuJoCo in-process "
            "(interactive viewer also forces this)."
        ),
    )
    p.add_argument(
        "--wave-max-attempts",
        type=_positive_int,
        default=DEFAULT_WAVE_MAX_ATTEMPTS,
        help=(
            "Total subprocess attempts per isolated evaluation wave before failing "
            f"(default: {DEFAULT_WAVE_MAX_ATTEMPTS})."
        ),
    )
    p.add_argument(
        "--enable-self-collision",
        action="store_true",
        help="Enable fruiting cable self-collisions during CMA replay (default: off).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from <output>/cma_optimizer_checkpoint.pkl when present. "
            "Allows a non-empty output directory without --overwrite."
        ),
    )
    p.add_argument(
        "--max-process-restarts",
        type=int,
        default=10,
        help=(
            "Auto-reexec the CLI with --resume after restartable failures "
            "(failed/global_error). 0 disables (default: 10)."
        ),
    )
    p.add_argument(
        "--cma-process-attempt",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--show-pull-direction",
        action="store_true",
        help="Draw cyan pull-direction arrows (requires --viewer gl).",
    )
    p.add_argument(
        "--snapshot-video-every",
        type=_nonnegative_int,
        default=0,
        help=(
            "Replay one random CMA sample to MP4 every N completed generations "
            "(0 disables; videos land in <output>/videos/). Each clip runs in "
            "a fresh headless GL subprocess so later generations keep recording."
        ),
    )
    p.add_argument("--settle-substeps", type=int, default=None)
    p.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=SETTLE_GRAVITY_RAMP,
    )
    p.add_argument("--settle-quiet-every", type=int, default=SETTLE_QUIET_EVERY)
    return p


class CmaSnapshotSample(NamedTuple):
    structure_idx: int
    generation_index: int
    candidate_index: int
    log10_vector: tuple[float, float, float]
    fitness: float | None


def should_record_cma_snapshot_video(
    interval: int,
    generation_index: int,
    last_recorded_generation: int | None,
) -> bool:
    """True when a snapshot is due for CMA ``generation_index`` (0-based)."""
    if int(interval) <= 0:
        return False
    generation = int(generation_index)
    if generation < 0:
        return False
    if generation % int(interval) != 0:
        return False
    if last_recorded_generation is not None and generation == int(last_recorded_generation):
        return False
    return True


def choose_random_cma_snapshot_sample(
    states: Mapping[int, Any],
    *,
    seed: int,
) -> CmaSnapshotSample | None:
    """Pick one asked sample from the latest completed generation, deterministically."""
    latest: list[tuple[int, Any]] = []
    latest_generation = -1
    for structure_idx, state in states.items():
        generations = getattr(state, "generations", None) or []
        if not generations:
            continue
        record = generations[-1]
        generation_index = int(getattr(record, "generation_index", -1))
        if generation_index > latest_generation:
            latest = [(int(structure_idx), record)]
            latest_generation = generation_index
        elif generation_index == latest_generation:
            latest.append((int(structure_idx), record))
    if not latest:
        return None
    rng = random.Random(int(seed) + latest_generation)
    structure_idx, record = rng.choice(sorted(latest, key=lambda item: item[0]))
    samples = tuple(tuple(float(v) for v in row) for row in record.ask_samples_log10)
    if not samples:
        return None
    candidate_index = rng.randrange(len(samples))
    fitness_values = getattr(record, "penalized_fitness", None) or ()
    fitness = None
    if candidate_index < len(fitness_values):
        value = fitness_values[candidate_index]
        if value is not None and math.isfinite(float(value)):
            fitness = float(value)
    return CmaSnapshotSample(
        structure_idx=int(structure_idx),
        generation_index=int(record.generation_index),
        candidate_index=int(candidate_index),
        log10_vector=tuple(samples[candidate_index]),
        fitness=fitness,
    )


def choose_random_snapshot_direction(
    direction_indices: tuple[int, ...] | None,
    *,
    seed: int,
    generation_index: int,
) -> int:
    """Pick one replay direction for the snapshot camera, deterministically."""
    dirs = tuple(int(d) for d in (direction_indices or ()))
    if not dirs:
        dirs = (0,)
    rng = random.Random(int(seed) + int(generation_index))
    return int(rng.choice(dirs))


def resolve_cma_snapshot_direction_indices(
    fit_direction_indices: tuple[int, ...] | None,
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    num_directions: int,
    include_excluded: bool,
) -> tuple[int, ...]:
    """Direction pool for snapshot replay: train split, else all usable disk dirs."""
    if fit_direction_indices is not None:
        return tuple(int(d) for d in fit_direction_indices)
    try:
        usable = tuple(
            int(d)
            for d in list_usable_direction_indices(
                dataset,
                int(structure_idx),
                include_excluded=bool(include_excluded),
            )
        )
        if usable:
            return usable
    except (OSError, ValueError, KeyError):
        pass
    n = max(1, int(num_directions))
    return tuple(range(n))


def load_snapshot_camera_to_base(
    dataset_dir: Path | str,
    structure_idx: int,
    direction_idx: int,
) -> object | None:
    """Episode ``camera_to_base_4x4`` for the chosen structure, if present."""
    try:
        dataset = BatchedSysIdDataset(dataset_dir)
        meta = dataset.load_episode_metadata(int(structure_idx), int(direction_idx))
    except (OSError, FileNotFoundError, ValueError, KeyError):
        return None
    if not isinstance(meta, dict):
        return None
    return meta.get("camera_to_base_4x4")


def cma_snapshot_video_path(
    output_dir: Path | str,
    sample: CmaSnapshotSample,
    *,
    direction_idx: int,
) -> Path:
    return (
        Path(output_dir)
        / "videos"
        / (
            f"structure_{int(sample.structure_idx):03d}"
            f"_gen_{int(sample.generation_index):03d}"
            f"_dir_{int(direction_idx):03d}"
            f"_sample_{int(sample.candidate_index):03d}.mp4"
        )
    )


def _gl_front_from_pitch_yaw(
    pitch_deg: float, yaw_deg: float
) -> tuple[float, float, float]:
    """Newton GL look direction for Z-up (matches ``Camera.get_front``)."""
    pitch = max(min(float(pitch_deg), 89.0), -89.0)
    yaw = float(yaw_deg)
    cp = math.cos(math.radians(pitch))
    fx = math.cos(math.radians(yaw)) * cp
    fy = math.sin(math.radians(yaw)) * cp
    fz = math.sin(math.radians(pitch))
    norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    return (fx / norm, fy / norm, fz / norm)


def _xyz_from_state_array(array: object) -> object:
    import numpy as np

    if array is None:
        return None
    data = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
    if data is None or np.asarray(data).size == 0:
        return None
    pts = np.asarray(data, dtype=np.float64)
    if pts.ndim < 2 or pts.shape[0] == 0:
        return None
    return pts[:, :3]


def _plant_points_from_env(env: object) -> object:
    import numpy as np

    sim = getattr(env, "_sim", None)
    scene = getattr(sim, "scene", None) if sim is not None else None
    cable = getattr(scene, "cable", None) if scene is not None else None
    state = getattr(cable, "state_0", None) if cable is not None else None
    if state is None:
        return None
    chunks = []
    for attr in ("body_q", "particle_q"):
        pts = _xyz_from_state_array(getattr(state, attr, None))
        if pts is not None:
            chunks.append(pts)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def frame_snapshot_camera_on_structure(
    viewer: object,
    env: object,
    *,
    camera_to_base_4x4: object | None = None,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    look_pitch_deg: float = -20.0,
    look_yaw_deg: float = 45.0,
    padding: float = 1.5,
    min_extent: float = 1.0,
) -> tuple[tuple[float, float, float], float, float] | None:
    """Place the GL camera on the chosen structure.

    Prefer episode ``camera_to_base_4x4`` (franka base, same frame as the
    fruiting plant). ``world_offset`` shifts that pose onto a batched copy.
    Falls back to a cable AABB fit when the parquet pose is missing.
    """
    from robot_replay.example_replay_real_batched import gl_camera_from_camera_to_base

    ox, oy, oz = (float(v) for v in world_offset)
    if camera_to_base_4x4 is not None:
        pose = gl_camera_from_camera_to_base(camera_to_base_4x4)
        if pose is not None:
            pos, pitch, yaw = pose
            pos = (pos[0] + ox, pos[1] + oy, pos[2] + oz)
            if hasattr(viewer, "set_camera"):
                viewer.set_camera(pos, pitch, yaw)
            return pos, pitch, yaw

    import numpy as np

    points = _plant_points_from_env(env)
    if points is None:
        return None
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    center = 0.5 * (lo + hi)
    extent = float(np.max(hi - lo))
    if not math.isfinite(extent) or extent < float(min_extent):
        extent = float(min_extent)
    camera = getattr(viewer, "camera", None)
    fov = float(getattr(camera, "fov", 45.0) or 45.0)
    fov = min(90.0, max(15.0, fov))
    half = math.tan(math.radians(fov) / 2.0)
    if half <= 1e-12:
        return None
    distance = extent / (2.0 * half) * float(padding)
    pitch = float(look_pitch_deg)
    yaw = float(look_yaw_deg)
    front = _gl_front_from_pitch_yaw(pitch, yaw)
    pos = (
        float(center[0] - front[0] * distance) + ox,
        float(center[1] - front[1] * distance) + oy,
        float(center[2] - front[2] * distance) + oz,
    )
    if hasattr(viewer, "set_camera"):
        viewer.set_camera(pos, pitch, yaw)
    return pos, pitch, yaw


def make_snapshot_on_step(
    viewer: object,
    *,
    control_hz: float,
    recorder: GlVideoRecorder | None,
    show_pull_direction: bool = False,
    camera_to_base_4x4: object | None = None,
    world_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    frame_camera: Callable[..., object] | None = None,
) -> Callable[..., bool]:
    """Render snapshot frames; frame the camera once from the structure pose."""
    viewer_state: dict[str, object] = {"model": None, "framed": False}
    if frame_camera is None:
        def frame_camera(viewer_obj: object, env_obj: object) -> object:
            return frame_snapshot_camera_on_structure(
                viewer_obj,
                env_obj,
                camera_to_base_4x4=camera_to_base_4x4,
                world_offset=world_offset,
            )

    def on_step(*, frame_idx: int, env: object) -> bool:
        if not _viewer_allows_snapshot_capture(viewer):
            return False
        sim = getattr(env, "_sim", None)
        if sim is None:
            return True
        scene = getattr(sim, "scene", None)
        if scene is None:
            return True
        active_model = scene.cable.model
        if viewer_state.get("model") is not active_model:
            viewer.set_model(active_model)
            if getattr(env, "num_envs", 1) > 1:
                viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
            if viewer_state.get("model") is None and hasattr(
                viewer, "hide_loading_splash"
            ):
                viewer.hide_loading_splash()
            viewer_state["model"] = active_model
        if not viewer_state["framed"]:
            frame_camera(viewer, env)
            viewer_state["framed"] = True
        hz = float(getattr(sim.config.runtime, "control_hz", control_hz))
        sim_time = float(frame_idx) / max(hz, 1e-9)
        _render_frame(
            viewer,
            env,
            sim_time,
            obs=getattr(env, "_last_obs", None),
            show_pull_direction=show_pull_direction,
        )
        if recorder is not None:
            if recorder.fps is None:
                recorder.set_fps(hz)
            recorder.capture(viewer)
        return True

    return on_step


def _reset_headless_gl_event_loop() -> None:
    """Clear pyglet's global exit flag before opening another headless GL viewer."""
    try:
        import pyglet
    except ImportError:
        return
    pyglet.app.event_loop.has_exit = False


def _viewer_allows_snapshot_capture(viewer: object) -> bool:
    """Headless snapshot replays ignore pyglet's stale ``has_exit`` flag."""
    renderer = getattr(viewer, "renderer", None)
    if bool(getattr(renderer, "headless", False)):
        return True
    if hasattr(viewer, "is_running"):
        return bool(viewer.is_running())
    return True


def _open_headless_gl_viewer() -> object:
    _reset_headless_gl_event_loop()
    return newton.viewer.ViewerGL(headless=True)


def record_cma_snapshot_video(
    sample: CmaSnapshotSample,
    *,
    output_dir: Path,
    replay_context: Any,
    scoring: YoungsModulusScoringConfig,
    dataset_dir: Path | str,
    num_directions: int,
    direction_indices: tuple[int, ...] | None,
    max_envs_per_batch: int,
    seed: int | None,
    include_excluded: bool,
    fail_fast: bool,
    action_dim: int,
    show_pull_direction: bool,
    control_hz: float,
    open_viewer: Any = _open_headless_gl_viewer,
) -> Path:
    """Replay one CMA sample under a headless GL viewer and write an MP4."""
    snapshot_seed = int(seed if seed is not None else 0)
    chosen_direction = choose_random_snapshot_direction(
        direction_indices,
        seed=snapshot_seed,
        generation_index=int(sample.generation_index),
    )
    path = cma_snapshot_video_path(
        output_dir, sample, direction_idx=chosen_direction
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = candidates_from_log10_vector(sample.log10_vector)
    viewer = open_viewer()
    recorder = GlVideoRecorder(path)
    try:
        require_gl_frame_capture(viewer)
        camera_to_base = load_snapshot_camera_to_base(
            dataset_dir,
            int(sample.structure_idx),
            chosen_direction,
        )
        on_step = make_snapshot_on_step(
            viewer,
            control_hz=float(control_hz),
            recorder=recorder,
            show_pull_direction=bool(show_pull_direction),
            camera_to_base_4x4=camera_to_base,
        )
        spec = make_cma_wave_evaluation_spec(
            dataset_dir=dataset_dir,
            structures=[(int(sample.structure_idx), (candidate,))],
            wave_kind="snapshot_video",
            scoring=scoring,
            replay_context=replay_context,
            num_directions=int(num_directions),
            direction_indices=(chosen_direction,),
            max_envs_per_batch=int(max_envs_per_batch),
            seed=seed,
            include_excluded=bool(include_excluded),
            fail_fast=bool(fail_fast),
            action_dim=int(action_dim),
            multi_structure_batch=True,
            on_step=on_step,
        )
        execute_cma_wave_evaluation(spec)
        if recorder.frame_count <= 0:
            raise RuntimeError(f"snapshot video wrote 0 frames ({path})")
    finally:
        recorder.close()
        if hasattr(viewer, "close"):
            viewer.close()
    return path


def cma_should_reexec(result: dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(result.get("exit_nonzero")):
        return False
    if str(result.get("command_status")) not in {"failed", "global_error"}:
        return False
    max_restarts = int(getattr(args, "max_process_restarts", 10))
    attempt = int(getattr(args, "cma_process_attempt", 0))
    return max_restarts > 0 and attempt < max_restarts


def _cma_entry_script() -> Path:
    return Path(__file__).resolve()


def _cma_repo_root() -> Path:
    return _cma_entry_script().parents[2]


def _cma_cli_argv_tail(argv: list[str]) -> list[str]:
    """Recover CMA CLI tokens from argv (wrapper scripts, ``python -c``, etc.)."""
    script = _cma_entry_script()
    script_name = script.name
    for index, arg in enumerate(argv):
        if arg == "-c":
            continue
        try:
            if arg.endswith(".py") and Path(arg).resolve() == script:
                return list(argv[index + 1 :])
        except OSError:
            pass
        if arg == script_name or arg.endswith(f"/{script_name}"):
            return list(argv[index + 1 :])
    for index, arg in enumerate(argv):
        if arg.startswith("--"):
            return list(argv[index:])
    return []


def cma_reexec_argv(argv: list[str], args: argparse.Namespace) -> list[str]:
    """Build argv for ``os.execv`` after a restartable CMA failure."""
    attempt = int(getattr(args, "cma_process_attempt", 0)) + 1
    script = str(_cma_entry_script())
    skip_next = False
    cli_tail: list[str] = []
    for arg in _cma_cli_argv_tail(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--overwrite":
            continue
        if arg == "--cma-process-attempt":
            skip_next = True
            continue
        if arg.startswith("--cma-process-attempt="):
            continue
        cli_tail.append(arg)
    if "--resume" not in cli_tail:
        cli_tail.append("--resume")
    cli_tail.extend(["--cma-process-attempt", str(attempt)])

    uv_executable = shutil.which("uv")
    if uv_executable is not None:
        return [
            uv_executable,
            "run",
            "--directory",
            str(_cma_repo_root()),
            "python",
            script,
            *cli_tail,
        ]
    return [sys.executable, script, *cli_tail]


def _run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    viewer: object,
) -> dict[str, Any]:
    output_dir = Path(args.output)
    resume = bool(getattr(args, "resume", False))
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not bool(args.overwrite)
        and not resume
    ):
        raise SystemExit(
            f"output directory {output_dir} is non-empty; pass --overwrite to continue"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CMA_OPTIMIZER_CHECKPOINT_FILENAME
    if resume and not checkpoint_path.is_file():
        raise SystemExit(
            f"--resume requested but checkpoint not found: {checkpoint_path}"
        )

    device = args.device
    if device == "cuda":
        device = "cuda:0"

    dataset = BatchedSysIdDataset(args.dataset)
    collection = dataset.manifest.get("collection", {})
    ranges_path = _resolve_ranges_path(args, collection)
    ranges = load_ranges(str(ranges_path))
    bounds = extract_support_kp_youngs_modulus_cma_bounds(ranges)
    search = CMA_SEARCH_PARAMS
    initial_mean = resolve_initial_mean_log10(search["initial_mean_log10"], bounds)
    initial_sigma = validate_initial_sigma_log10(float(search["initial_sigma_log10"]))
    max_sigma = validate_max_sigma_log10(search.get("max_sigma_log10"))
    if max_sigma is not None and max_sigma < initial_sigma:
        raise SystemExit("CMA_SEARCH_PARAMS['max_sigma_log10'] must be >= initial_sigma_log10")
    if getattr(args, "cma_seed", None) is not None:
        base_seed = int(args.cma_seed)
    else:
        base_seed = int(search["cma_seed"])
    if getattr(args, "max_generations", None) is not None:
        max_generations = int(args.max_generations)
    else:
        max_generations = int(search["max_generations"])
    if max_generations < 1:
        raise SystemExit("max_generations must be >= 1")
    population_size = search["population_size"]
    if population_size is not None:
        population_size = int(population_size)
        if population_size < 1:
            raise SystemExit("CMA_SEARCH_PARAMS['population_size'] must be >= 1")

    topology_seed = int(collection.get("topology_seed", 42))
    control_hz = _collection_control_hz(collection)
    num_directions = _resolve_n_directions(dataset, collection)
    if num_directions < 1:
        parser.error("manifest num_directions must be >= 1")

    structure_indices = _resolve_structure_indices(dataset, args.structure_indices)
    if not structure_indices:
        raise SystemExit("No structure indices to evaluate.")

    if bool(args.overwrite) and not resume:
        _clear_cma_owned_artifacts(
            output_dir, structure_indices=list(structure_indices)
        )

    replay_seed = args.seed
    if replay_seed is None and "seed" in collection:
        replay_seed = int(collection["seed"])

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
    if dataset_is_vic_pose and mode == "vic":
        raise SystemExit(
            "packed 19D vic_pose datasets must use --controller-mode vic_pose "
            "(or omit the flag), not twist vic"
        )
    initial_mean = _effective_initial_mean_log10(mode, search, bounds)

    train_direction_indices: tuple[int, ...] | None = None
    val_direction_indices: tuple[int, ...] | None = None
    try:
        train_direction_indices, val_direction_indices = _resolve_holdout_direction_split(
            args,
            dataset,
            list(structure_indices),
            include_excluded=bool(args.include_excluded),
        )
    except SystemExit:
        raise
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    fit_direction_indices = train_direction_indices

    action_dim = 19 if mode == "vic_pose" else 6
    if mode == "vic_pose":
        _require_ft_wrist_lpf_per_structure(dataset, structure_indices)

    try:
        search_bounds_log10 = _effective_search_bounds_log10(mode, search)
    except ValueError as exc:
        raise SystemExit(f"CMA_SEARCH_PARAMS['search_bounds_log10']: {exc}") from exc

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
    replay_context = build_cma_replay_context_from_cli(
        mode=mode,
        ranges_path=ranges_path,
        topology_seed=topology_seed if mode != "vic_pose" else int(
            collection.get("topology_seed", collection.get("seed", 0))
        ),
        control_hz=float(control_hz),
        device=device,
        settle_config=settle_config,
        post_grasp_settle_substeps=500,
        real_topology_seed=int(
            collection.get("topology_seed", collection.get("seed", 0))
        )
        if mode == "vic_pose"
        else None,
        fruiting_base_pos=fruiting_base_pos_from_episode_metadata(episode_meta)
        if mode == "vic_pose"
        else None,
        bootstrap_joint_q=bootstrap_joint_q_from_episode_metadata(episode_meta)
        if mode == "vic_pose"
        else None,
        episode_meta=episode_meta if mode == "vic_pose" else None,
        enable_self_collisions=bool(getattr(args, "enable_self_collision", False)),
    )
    scoring = YoungsModulusScoringConfig(
        use_median=args.use_median is True,
        hold_id_onehot=bool(args.hold_id_onehot),
        pool_directions=bool(args.pool_directions),
        n_holds=_resolve_n_holds(dataset, collection),
        n_directions=int(num_directions),
        device=device,
        hold_aggregation=getattr(args, "hold_aggregation", "none"),
        include_delta=bool(getattr(args, "include_delta", False)),
        categorical_weight=float(getattr(args, "categorical_weight", 30.0)),
    )

    derive_structure_cma_seeds(base_seed=base_seed, structure_indices=structure_indices)

    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    use_viewer = graphical or getattr(args, "viewer", None) != "null"
    if "PYTEST_CURRENT_TEST" in os.environ and not hasattr(args, "isolated_eval_waves"):
        use_isolated_eval_waves = False
    else:
        use_isolated_eval_waves = bool(getattr(args, "isolated_eval_waves", True))
    if graphical and use_isolated_eval_waves:
        raise SystemExit(
            "Interactive GL viewer cannot run with --isolated-eval-waves; "
            "pass --no-isolated-eval-waves."
        )
    if use_viewer and use_isolated_eval_waves:
        print(
            "Note: disabling isolated eval waves because interactive viewer is active.",
            file=sys.stderr,
        )
        use_isolated_eval_waves = False

    replay_context = _dc_replace(
        replay_context,
        reuse_replicated_mujoco=reuse_replicated_mujoco_for_cma(
            isolated_eval_waves=use_isolated_eval_waves
        ),
    )

    states: dict[int, StructureCmaState] = {}
    for structure_idx in structure_indices:
        es, effective_seed, _rng = create_structure_cma_optimizer(
            bounds,
            initial_mean_log10=initial_mean,
            initial_sigma_log10=initial_sigma,
            base_seed=base_seed,
            structure_idx=int(structure_idx),
            population_size=population_size,
            search_bounds_log10=search_bounds_log10,
            max_sigma_log10=max_sigma,
        )
        state = StructureCmaState(
            structure_idx=int(structure_idx),
            optimizer=es,
            bounds=bounds,
            effective_seed=int(effective_seed),
            population_size=int(es.popsize),
            search_bounds_log10=search_bounds_log10,
            max_sigma_log10=max_sigma,
        )
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
        states[int(structure_idx)] = state

    report_path = output_dir / "cmaes_report.json"
    counters: dict[str, int] = {
        "replay_candidate_evaluations": 0,
        "final_mean_evaluations": 0,
        "physical_env_slots": 0,
    }
    if resume:
        checkpoint = load_cma_optimizer_checkpoint(checkpoint_path)
        restored_counters = apply_cma_checkpoint_to_states(states, checkpoint)
        for key, value in restored_counters.items():
            counters[key] = int(value)

    timing: dict[str, Any] = {}
    command_started = time.perf_counter()
    command_status = "running"
    command_error: str | None = None
    exit_nonzero = False

    def write_report(*, status: str | None = None, error: str | None = None) -> None:
        nonlocal command_status, command_error
        if status is not None:
            command_status = status
        if error is not None:
            command_error = error
        timing["command_seconds"] = float(time.perf_counter() - command_started)
        payload = _build_cmaes_report_payload(
            states,
            dataset=str(args.dataset),
            output=str(output_dir),
            ranges_path=str(ranges_path),
            base_seed=base_seed,
            initial_mean_log10=initial_mean,
            initial_sigma_log10=initial_sigma,
            max_sigma_log10=max_sigma,
            max_generations=int(max_generations),
            scoring=scoring,
            command_status=command_status,
            command_error=command_error,
            counter_totals=counters,
            timing=timing,
            population_size=population_size,
            search_bounds_log10=search_bounds_log10,
            force_magnitude_weight=float(
                getattr(args, "force_magnitude_weight", 0.0)
            ),
            isolated_eval_waves=bool(use_isolated_eval_waves),
            wave_max_attempts=int(getattr(args, "wave_max_attempts", DEFAULT_WAVE_MAX_ATTEMPTS)),
        )
        _write_cmaes_report_atomic(report_path, payload)

    write_report(status="running")

    if bool(args.fail_fast) and any(state.status == "failed" for state in states.values()):
        failed = next(state for state in states.values() if state.status == "failed")
        message = (
            failed.failure.message if failed.failure is not None else "structure failed"
        )
        write_report(status="global_error", error=message)
        return {
            "dataset": str(args.dataset),
            "output": str(output_dir),
            "ranges_path": str(ranges_path),
            "structure_indices": structure_indices,
            "train_direction_indices": train_direction_indices,
            "val_direction_indices": val_direction_indices,
            "states": states,
            "exit_nonzero": True,
            "command_status": "global_error",
        }

    show_pull_direction = bool(args.show_pull_direction) and graphical
    frame_dt = 1.0 / float(control_hz)
    viewer_state: dict[str, object] = {"model": None}

    def on_step(*, frame_idx: int, env: object) -> bool:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            raise ViewerCancelled("viewer closed; cancelling CMA-ES fit")
        if not use_viewer:
            return True

        sim = getattr(env, "_sim", None)
        if sim is None:
            return True
        scene = getattr(sim, "scene", None)
        if scene is None:
            return True

        active_model = scene.cable.model
        if viewer_state.get("model") is not active_model:
            viewer.set_model(active_model)
            if graphical and getattr(env, "num_envs", 1) > 1:
                viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
            if viewer_state.get("model") is None and hasattr(
                viewer, "hide_loading_splash"
            ):
                viewer.hide_loading_splash()
            viewer_state["model"] = active_model

        hz = float(getattr(sim.config.runtime, "control_hz", control_hz))
        sim_time = float(frame_idx) / max(hz, 1e-9)
        obs = getattr(env, "_last_obs", None)
        _render_frame(
            viewer,
            env,
            sim_time,
            obs=obs,
            show_pull_direction=show_pull_direction,
        )
        if graphical:
            time.sleep(max(0.0, frame_dt))
        return True

    def _accumulate_wave_counters(
        batch: YoungsModulusBatchEvaluation,
        structure_list: list[tuple[int, tuple[Any, ...]]],
        wave_kind: str,
    ) -> None:
        accumulate_cma_batch_counters(
            counters,
            batch,
            structures=structure_list,
            wave_kind=str(wave_kind),
            num_directions=int(num_directions),
        )

    def evaluate_fn(
        *,
        structures,
        wave_kind: str = "generation",
        **_kwargs,
    ) -> YoungsModulusBatchEvaluation:
        structure_list = [
            (int(structure_idx), tuple(candidates))
            for structure_idx, candidates in structures
        ]
        spec = make_cma_wave_evaluation_spec(
            dataset_dir=args.dataset,
            structures=structure_list,
            wave_kind=str(wave_kind),
            scoring=scoring,
            replay_context=replay_context,
            num_directions=int(num_directions),
            direction_indices=fit_direction_indices,
            max_envs_per_batch=int(args.max_envs_per_batch),
            seed=replay_seed,
            include_excluded=bool(args.include_excluded),
            fail_fast=bool(args.fail_fast),
            action_dim=action_dim,
            multi_structure_batch=bool(getattr(args, "multi_structure_batch", True)),
            on_step=on_step if use_viewer else None,
        )
        if use_isolated_eval_waves:
            batch = spawn_isolated_cma_wave_evaluation(
                spec,
                output_dir=output_dir,
                max_attempts=int(
                    getattr(args, "wave_max_attempts", DEFAULT_WAVE_MAX_ATTEMPTS)
                ),
            )
        else:
            batch = execute_cma_wave_evaluation(spec)
        _accumulate_wave_counters(batch, structure_list, str(wave_kind))
        return batch

    snapshot_interval = int(getattr(args, "snapshot_video_every", 0) or 0)
    last_snapshot_generation: int | None = None
    snapshot_seed = int(base_seed if replay_seed is None else replay_seed)

    def on_progress(progress_states) -> None:
        nonlocal last_snapshot_generation
        write_report(status="running")
        dump_cma_optimizer_checkpoint(
            checkpoint_path,
            progress_states,
            counters=counters,
        )
        completed = max(
            (int(state.completed_generations) for state in progress_states.values()),
            default=0,
        )
        latest_generation = max(
            (
                int(state.generations[-1].generation_index)
                for state in progress_states.values()
                if getattr(state, "generations", None)
            ),
            default=-1,
        )
        if should_record_cma_snapshot_video(
            snapshot_interval, latest_generation, last_snapshot_generation
        ):
            sample = choose_random_cma_snapshot_sample(
                progress_states, seed=snapshot_seed
            )
            if sample is not None:
                try:
                    snapshot_direction_indices = resolve_cma_snapshot_direction_indices(
                        fit_direction_indices,
                        dataset=dataset,
                        structure_idx=int(sample.structure_idx),
                        num_directions=int(num_directions),
                        include_excluded=bool(args.include_excluded),
                    )
                    video_path = spawn_isolated_cma_snapshot_video(
                        CmaSnapshotVideoJob(
                            structure_idx=int(sample.structure_idx),
                            generation_index=int(sample.generation_index),
                            candidate_index=int(sample.candidate_index),
                            log10_vector=tuple(
                                float(v) for v in sample.log10_vector
                            ),
                            fitness=sample.fitness,
                            output_dir=output_dir,
                            replay_context=replay_context,
                            scoring=scoring,
                            dataset_dir=Path(args.dataset).resolve(),
                            num_directions=int(num_directions),
                            direction_indices=snapshot_direction_indices,
                            max_envs_per_batch=int(args.max_envs_per_batch),
                            seed=replay_seed,
                            include_excluded=bool(args.include_excluded),
                            fail_fast=bool(args.fail_fast),
                            action_dim=int(action_dim),
                            show_pull_direction=bool(show_pull_direction),
                            control_hz=float(control_hz),
                        ),
                        output_dir=output_dir,
                    )
                    last_snapshot_generation = int(sample.generation_index)
                    print(
                        "snapshot video "
                        f"gen={sample.generation_index} "
                        f"structure={sample.structure_idx} "
                        f"sample={sample.candidate_index} "
                        f"path={video_path}",
                        file=sys.stderr,
                    )
                except Exception as exc:
                    print(
                        f"warning: snapshot video failed at generation {latest_generation}: {exc}",
                        file=sys.stderr,
                    )
        if bool(args.fail_fast) and any(
            state.status == "failed" for state in progress_states.values()
        ):
            failed = next(
                state
                for state in progress_states.values()
                if state.status == "failed"
            )
            message = (
                failed.failure.message
                if failed.failure is not None
                else "structure failed"
            )
            raise CmaGenerationFailure(
                failed.failure.stage if failed.failure else "generation_evaluation",
                message,
            )

    try:
        fit_result = fit_youngs_modulus_structures(
            states,
            max_generations=int(max_generations),
            evaluate_fn=evaluate_fn,
            on_progress=on_progress,
            force_magnitude_weight=float(
                getattr(args, "force_magnitude_weight", 0.0)
            ),
        )
        timing.update(dict(fit_result.timing or {}))
        write_report(status="running")
    except ViewerCancelled as exc:
        exit_nonzero = True
        write_report(status="cancelled", error=str(exc))
        return {
            "dataset": str(args.dataset),
            "output": str(output_dir),
            "ranges_path": str(ranges_path),
            "structure_indices": structure_indices,
            "train_direction_indices": train_direction_indices,
            "val_direction_indices": val_direction_indices,
            "states": states,
            "exit_nonzero": True,
            "command_status": "cancelled",
        }
    except Exception as exc:
        exit_nonzero = True
        write_report(status="global_error", error=str(exc))
        return {
            "dataset": str(args.dataset),
            "output": str(output_dir),
            "ranges_path": str(ranges_path),
            "structure_indices": structure_indices,
            "train_direction_indices": train_direction_indices,
            "val_direction_indices": val_direction_indices,
            "states": states,
            "exit_nonzero": True,
            "command_status": "global_error",
        }

    for structure_idx, state in states.items():
        if state.status != "fitted":
            continue
        try:
            _write_final_mean_overlay(state, output_dir=output_dir)
        except Exception as exc:
            state.artifact_errors.append(str(exc))
        write_report(status="running")

    try:
        viz_paths = write_cmaes_visualization_bundle(report_path, output_dir)
        timing["visualization_paths"] = [str(path) for path in viz_paths]
    except Exception as exc:
        for state in states.values():
            if state.status == "fitted":
                state.artifact_errors.append(f"visualization: {exc}")
                break
        else:
            timing["visualization_error"] = str(exc)

    holdout_report_seed: int | None = None
    if val_direction_indices is not None and getattr(args, "direction_indices", None) is None:
        holdout_report_seed = int(getattr(args, "direction_split_seed"))

    fitted = [idx for idx, state in states.items() if state.status == "fitted"]
    if val_direction_indices is not None and fitted:
        for structure_idx in fitted:
            state = states[int(structure_idx)]
            if state.final_mean_log10 is None:
                continue

            def evaluate_val(
                log10_vector: list[float] | tuple[float, ...],
                val_dirs: tuple[int, ...],
                *,
                _structure_idx: int = int(structure_idx),
            ) -> YoungsModulusBatchEvaluation:
                candidate = candidates_from_log10_vector(tuple(log10_vector))
                return evaluate_youngs_modulus_structures(
                    dataset=dataset,
                    structures=[(_structure_idx, (candidate,))],
                    num_directions=int(num_directions),
                    build_env_fn=build_env_fn,
                    scoring=scoring,
                    max_envs_per_batch=int(args.max_envs_per_batch),
                    seed=replay_seed,
                    include_excluded=bool(args.include_excluded),
                    fail_fast=bool(args.fail_fast),
                    on_step=on_step,
                    replay_sim_config=replay_sim_config,
                    action_dim=action_dim,
                    direction_indices=val_dirs,
                )

            try:
                _report, gate_failures = run_holdout_evaluation(
                    output_dir=output_dir,
                    dataset=dataset,
                    structure_idx=int(structure_idx),
                    state=state,
                    train_direction_indices=train_direction_indices or (),
                    val_direction_indices=val_direction_indices,
                    direction_split_seed=holdout_report_seed,
                    baseline_log10=list(initial_mean),
                    fitted_log10=list(state.final_mean_log10),
                    num_directions=int(num_directions),
                    include_excluded=bool(args.include_excluded),
                    evaluate_val=evaluate_val,
                )
            except Exception as exc:
                exit_nonzero = True
                state.artifact_errors.append(f"holdout: {exc}")
                write_report(status="running")
                continue
            if gate_failures:
                exit_nonzero = True
                print(f"holdout gate failed: {gate_failures[0]}")
            write_report(status="running")

    fitted = [idx for idx, state in states.items() if state.status == "fitted"]
    failed = [idx for idx, state in states.items() if state.status == "failed"]
    if not fitted:
        exit_nonzero = True
        write_report(status="failed")
    else:
        write_report(status="completed")

    fit_seconds = timing.get("fit_seconds")
    command_seconds = timing.get("command_seconds")
    print(
        f"cmaes structures requested={len(states)} fitted={len(fitted)} "
        f"failed={len(failed)} ranges={ranges_path}"
        + (
            f" fit_seconds={float(fit_seconds):.2f}"
            if fit_seconds is not None
            else ""
        )
        + (
            f" command_seconds={float(command_seconds):.2f}"
            if command_seconds is not None
            else ""
        )
    )

    return {
        "dataset": str(args.dataset),
        "output": str(output_dir),
        "ranges_path": str(ranges_path),
        "structure_indices": structure_indices,
        "train_direction_indices": train_direction_indices,
        "val_direction_indices": val_direction_indices,
        "states": states,
        "fitted_structure_indices": fitted,
        "failed_structure_indices": failed,
        "exit_nonzero": bool(exit_nonzero),
        "command_status": command_status,
    }


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    try:
        result = _run(args, parser, viewer=viewer)
        if cma_should_reexec(result, args):
            reexec_argv = cma_reexec_argv(sys.argv, args)
            os.execv(reexec_argv[0], reexec_argv)
        if result.get("exit_nonzero"):
            raise SystemExit(1)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
