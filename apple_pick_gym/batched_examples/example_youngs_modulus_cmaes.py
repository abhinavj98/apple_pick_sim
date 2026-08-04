"""Dataset-driven support-k_p + Young's-modulus CMA-ES fit entry point.

Fits a 3-vector ``(support_kp, E_spur, E_stem)`` — support joint k_p (shared
angular+linear, zeta=1) is free while spur/stem Young's modulus stay free;
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
population, generations, bounds). ``--cma-seed`` overrides
``CMA_SEARCH_PARAMS["cma_seed"]`` so the multi-seed gate can vary optimizer RNG.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import newton.examples
import newton.viewer

from apple_pick_gym.batched_examples.example_batched_sysid_mmd_grid import (
    parse_comma_separated_ints,
)
from apple_pick_gym.batched_examples import example_youngs_modulus_sys_id as _grid
from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    CmaGenerationFailure,
    StructureCmaState,
    SupportKpYoungsCandidate,
    YoungsModulusBatchEvaluation,
    YoungsModulusCandidate,
    YoungsModulusScoringConfig,
    aggregate_fitted_youngs_modulus_stats,
    candidates_from_log10_vector,
    create_structure_cma_optimizer,
    derive_structure_cma_seeds,
    evaluate_youngs_modulus_candidates,
    evaluate_youngs_modulus_structures,
    extract_support_kp_youngs_modulus_cma_bounds,
    fit_youngs_modulus_structures,
    gt_support_kp_youngs_candidate_from_structure,
    normalize_search_bounds_log10,
    resolve_initial_mean_log10,
    structure_cma_report_snapshot,
    to_strict_jsonable,
    validate_initial_sigma_log10,
)
from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
    SysIdReplayCancelled,
)
from apple_pick_gym.youngs_modulus_cmaes_viz import write_cmaes_visualization_bundle
from apple_pick_gym.youngs_modulus_overlay_viz import (
    overlay_episodes_from_replay_evaluation,
    write_youngs_modulus_overlay_html,
)
from apple_pick_sim.fruiting_system import load_ranges
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

# Re-export grid helpers used by tests and shared replay setup.
SETTLE_GRAVITY_RAMP = _grid.SETTLE_GRAVITY_RAMP
SETTLE_QUIET_EVERY = _grid.SETTLE_QUIET_EVERY
build_sim_config = _grid.build_sim_config
_make_build_env_fn = _grid._make_build_env_fn
_resolve_structure_indices = _grid._resolve_structure_indices
_resolve_n_holds = _grid._resolve_n_holds
_resolve_n_directions = _grid._resolve_n_directions
_collection_control_hz = _grid._collection_control_hz
_settle_config_kwargs = _grid._settle_config_kwargs
_render_frame = _grid._render_frame
_positive_int = _grid._positive_int


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
# midpoint, never from ground truth.
_CMA_SEARCH_LOG10_LOWER = [2.0, 8.0, 8.0]  # support_kp 1e2, spur/stem 0.1 GPa
_CMA_SEARCH_LOG10_UPPER = [6.0, 11.0, 11.0]  # support_kp 1e6, spur/stem 100 GPa
_CMA_MEAN_LOG10 = [_CMA_SEARCH_LOG10_LOWER[i] + 0.5 * (_CMA_SEARCH_LOG10_UPPER[i] - _CMA_SEARCH_LOG10_LOWER[i]) for i in range(3)]
CMA_SEARCH_PARAMS: dict[str, Any] = {
    "initial_mean_log10": list(_CMA_MEAN_LOG10),
    "initial_sigma_log10": 1.0,
    "population_size": 15,
    "max_generations": 10,
    "cma_seed": 56,
    "search_bounds_log10": {
        "lower": _CMA_SEARCH_LOG10_LOWER,
        "upper": _CMA_SEARCH_LOG10_UPPER,
    },
}


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

    retried = len(batch.retried_structures)
    if retried:
        counters["scalar_retries"] += int(retried)
        for structure_idx in batch.retried_structures:
            counters[f"scalar_retries:{structure_idx}"] = (
                counters.get(f"scalar_retries:{structure_idx}", 0) + 1
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
    for path in output_dir.glob(".cmaes_report.json.*.tmp"):
        path.unlink(missing_ok=True)
    if structure_indices is None:
        return
    for structure_idx in structure_indices:
        overlay = (
            output_dir
            / f"structure_{int(structure_idx):03d}"
            / "youngs_modulus_overlay.html"
        )
        if overlay.exists():
            overlay.unlink()


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
    max_generations: int,
    scoring: YoungsModulusScoringConfig,
    command_status: str,
    command_error: str | None = None,
    counter_totals: dict[str, int] | None = None,
    timing: dict[str, Any] | None = None,
    population_size: int | None = None,
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
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
            scalar_retries=int(counters.get(f"scalar_retries:{structure_idx}", 0)),
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
            "use_median": bool(scoring.use_median),
            "hold_id_onehot": bool(scoring.hold_id_onehot),
            "pool_directions": bool(scoring.pool_directions),
            "n_holds": scoring.n_holds,
            "n_directions": scoring.n_directions,
            "device": scoring.device,
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
            "scalar_retries": int(counters.get("scalar_retries", 0)),
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
        "--use-median",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use full-hold median hold→hold features for Sinkhorn scoring.",
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
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    p.add_argument(
        "--show-pull-direction",
        action="store_true",
        help="Draw cyan pull-direction arrows (requires --viewer gl).",
    )
    p.add_argument("--settle-substeps", type=int, default=None)
    p.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=SETTLE_GRAVITY_RAMP,
    )
    p.add_argument("--settle-quiet-every", type=int, default=SETTLE_QUIET_EVERY)
    return p


def _run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    viewer: object,
) -> dict[str, Any]:
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise SystemExit(
            f"output directory {output_dir} is non-empty; pass --overwrite to continue"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if getattr(args, "cma_seed", None) is not None:
        base_seed = int(args.cma_seed)
    else:
        base_seed = int(search["cma_seed"])
    max_generations = int(search["max_generations"])
    if max_generations < 1:
        raise SystemExit("CMA_SEARCH_PARAMS['max_generations'] must be >= 1")
    population_size = search["population_size"]
    if population_size is not None:
        population_size = int(population_size)
        if population_size < 1:
            raise SystemExit("CMA_SEARCH_PARAMS['population_size'] must be >= 1")
    try:
        search_bounds_log10 = normalize_search_bounds_log10(
            search.get("search_bounds_log10")
        )
    except ValueError as exc:
        raise SystemExit(f"CMA_SEARCH_PARAMS['search_bounds_log10']: {exc}") from exc

    topology_seed = int(collection.get("topology_seed", 42))
    control_hz = _collection_control_hz(collection)
    num_directions = _resolve_n_directions(dataset, collection)
    if num_directions < 1:
        parser.error("manifest num_directions must be >= 1")

    structure_indices = _resolve_structure_indices(dataset, args.structure_indices)
    if not structure_indices:
        raise SystemExit("No structure indices to evaluate.")

    if bool(args.overwrite):
        _clear_cma_owned_artifacts(
            output_dir, structure_indices=list(structure_indices)
        )

    replay_seed = args.seed
    if replay_seed is None and "seed" in collection:
        replay_seed = int(collection["seed"])

    settle_config = _settle_config_kwargs(args=args)
    build_env_fn = _make_build_env_fn(
        ranges_path=str(ranges_path),
        topology_seed=topology_seed,
        control_hz=control_hz,
        device=device,
        settle_config=settle_config,
    )
    replay_sim_config = build_sim_config(num_envs=1, ranges=ranges, **settle_config)
    scoring = YoungsModulusScoringConfig(
        use_median=bool(args.use_median),
        hold_id_onehot=bool(args.hold_id_onehot),
        pool_directions=bool(args.pool_directions),
        n_holds=_resolve_n_holds(dataset, collection),
        n_directions=int(num_directions),
        device=device,
    )

    derive_structure_cma_seeds(base_seed=base_seed, structure_indices=structure_indices)

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
        )
        state = StructureCmaState(
            structure_idx=int(structure_idx),
            optimizer=es,
            bounds=bounds,
            effective_seed=int(effective_seed),
            population_size=int(es.popsize),
            search_bounds_log10=search_bounds_log10,
        )
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
        "scalar_retries": 0,
    }
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
            max_generations=int(max_generations),
            scoring=scoring,
            command_status=command_status,
            command_error=command_error,
            counter_totals=counters,
            timing=timing,
            population_size=population_size,
            search_bounds_log10=search_bounds_log10,
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
            "states": states,
            "exit_nonzero": True,
        }

    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    use_viewer = graphical or getattr(args, "viewer", None) != "null"
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
        if bool(getattr(args, "multi_structure_batch", True)):
            batch = evaluate_youngs_modulus_structures(
                dataset=dataset,
                structures=structure_list,
                num_directions=int(num_directions),
                build_env_fn=build_env_fn,
                scoring=scoring,
                max_envs_per_batch=int(args.max_envs_per_batch),
                seed=replay_seed,
                include_excluded=bool(args.include_excluded),
                fail_fast=bool(args.fail_fast),
                on_step=on_step,
                replay_sim_config=replay_sim_config,
            )
        else:
            evaluations: dict[int, Any] = {}
            errors: dict[int, str] = {}
            for structure_idx, candidates in structure_list:
                try:
                    evaluations[int(structure_idx)] = evaluate_youngs_modulus_candidates(
                        dataset=dataset,
                        structure_idx=int(structure_idx),
                        candidates=list(candidates),
                        num_directions=int(num_directions),
                        build_env_fn=build_env_fn,
                        scoring=scoring,
                        max_envs_per_batch=int(args.max_envs_per_batch),
                        seed=replay_seed,
                        include_excluded=bool(args.include_excluded),
                        on_step=on_step,
                        replay_sim_config=replay_sim_config,
                    )
                except ViewerCancelled:
                    raise
                except Exception as exc:
                    if bool(args.fail_fast):
                        raise
                    errors[int(structure_idx)] = str(exc)
            cand_by_idx = {
                int(structure_idx): candidates
                for structure_idx, candidates in structure_list
            }
            physical_slots_by_structure = {
                int(idx): len(cand_by_idx[int(idx)])
                * len(evaluation.direction_indices)
                for idx, evaluation in evaluations.items()
            }
            batch = YoungsModulusBatchEvaluation(
                evaluations=evaluations,
                errors=errors,
                replay_diagnostics=None,
                retried_structures=(),
                prepared_structures=len(evaluations),
                physical_slots_by_structure=physical_slots_by_structure,
            )
        accumulate_cma_batch_counters(
            counters,
            batch,
            structures=structure_list,
            wave_kind=str(wave_kind),
            num_directions=int(num_directions),
        )
        return batch

    def on_progress(progress_states) -> None:
        write_report(status="running")
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
            "states": states,
            "exit_nonzero": True,
        }
    except Exception as exc:
        exit_nonzero = True
        write_report(status="global_error", error=str(exc))
        return {
            "dataset": str(args.dataset),
            "output": str(output_dir),
            "ranges_path": str(ranges_path),
            "structure_indices": structure_indices,
            "states": states,
            "exit_nonzero": True,
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
        "states": states,
        "fitted_structure_indices": fitted,
        "failed_structure_indices": failed,
        "exit_nonzero": bool(exit_nonzero),
    }


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    try:
        result = _run(args, parser, viewer=viewer)
        if result.get("exit_nonzero"):
            raise SystemExit(1)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
