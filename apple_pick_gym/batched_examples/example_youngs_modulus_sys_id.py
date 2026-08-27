"""Dataset-driven Young's-modulus grid replay and ranking.

Replay recorded actions from an existing ``batched_sysid_v1`` dataset over a
Cartesian log10-E grid for each structure, score candidates with pooled hold-phase
Sinkhorn loss, and hand off structured results for reporting.

``ranking.json`` candidate fields (schema names unchanged):

- ``aggregate_sinkhorn``: pooled optimizer/ranking fitness across complete
  directions (direction one-hot features when pooling is enabled). Non-finite
  values serialize as JSON ``null``.
- ``per_direction_sinkhorn``: independently normalized diagnostic losses keyed
  by physical source direction ID strings (never the internal pooled bag key
  ``-1``).
- Missing/empty bags: candidate is unranked (``rank`` is ``null``),
  ``disqualified`` is true, and ``disqualification_reason`` names the cause
  (for example ``empty_transition_bag``).

Run from repo root::

    uv run python apple_pick_gym/batched_examples/example_youngs_modulus_sys_id.py \\
        --dataset /tmp/batched_sysid_dataset \\
        --output /tmp/youngs_rank \\
        --support-kp-values 1e3,1e4,1e5 --log10-e-spur 7.5 --log10-e-stem 7.0

    # Optional GL MP4 of the batched multi-world view (requires --viewer gl):
    #   --record-video /tmp/youngs_grid.mp4 --viewer gl

Candidates are ``support_kp x E_spur x E_stem``; primary \\(E\\) is fixed from
the structure's true/fixture params and is never a free grid axis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import newton.examples
import newton.viewer

from apple_pick_gym.batched_examples._youngs_e_grid_cli import (
    candidates_from_support_kp_grid_cli,
)
from apple_pick_gym.batched_examples.example_batched_sysid_mmd_grid import (
    parse_comma_separated_ints,
)
from apple_pick_gym.batched_envs.batched_sysid_cmaes import (
    YoungsModulusEvaluation,
    YoungsModulusScoringConfig,
    evaluate_youngs_modulus_candidates,
    evaluate_youngs_modulus_structures,
    gt_support_kp_youngs_candidate_from_structure,
    maybe_include_gt_candidate,
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
from apple_pick_gym.youngs_modulus_overlay_viz import (
    overlay_episodes_from_replay_evaluation,
    select_overlay_candidate_indices,
    write_youngs_modulus_overlay_html,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES,
    EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KD_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KP_OVERRIDES,
)
from apple_pick_sim.fruiting_system import (
    default_ranges_fixture_path,
    load_ranges,
    parse_sim_build,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
from apple_pick_sim.system_id.batched_replay_export import (
    ReplayCandidateSpec,
    export_replay_candidates_for_structure,
)
from robot_replay.gl_video_recorder import GlVideoRecorder

CONTROL_HZ = 30.0
SUB_DT = 1.0 / 1800.0
ENV_SPACING = (2.0, 2.0, 2.0)
SETTLE_SUBSTEPS = 5000
SETTLE_GRAVITY_RAMP = False
SETTLE_QUIET_EVERY: int | None = 300
MAX_ENVS_PER_BATCH = 25000
VIC_GAINS = ImpedanceGains(
    linear_k=200.0,
    linear_d=10.0,
    angular_k=10.0,
    angular_d=1.0,
)
JOINT_ANGULAR_KD_OVERRIDES = EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES
JOINT_LINEAR_KD_OVERRIDES = EXAMPLE_JOINT_LINEAR_KD_OVERRIDES
JOINT_ANGULAR_KP_OVERRIDES = EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES
JOINT_LINEAR_KP_OVERRIDES = EXAMPLE_JOINT_LINEAR_KP_OVERRIDES
DEFAULT_SUPPORT_KP_VALUES = "1e3,1e4,1e5"


def _json_float(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _json_log10(value: float) -> float | None:
    finite_value = _json_float(value)
    if finite_value is None or finite_value <= 0.0:
        return None
    return _json_float(math.log10(finite_value))


def _per_direction_sinkhorn_to_json(
    per_direction_sinkhorn: dict[int, float],
) -> dict[str, float | None]:
    """Serialize physical-direction diagnostic losses for ``ranking.json``.

    Keys are physical source direction IDs as decimal strings. Non-finite losses
    become ``None`` (JSON ``null``). An empty map stays ``{}``.
    """
    return {
        str(int(direction)): _json_float(loss)
        for direction, loss in per_direction_sinkhorn.items()
    }


def _candidate_to_json_row(score: Any) -> dict[str, Any]:
    """Serialize one candidate score row for ``ranking.json``.

    ``aggregate_sinkhorn`` is the pooled ranking fitness; non-finite values are
    ``null``. ``per_direction_sinkhorn`` keeps physical direction IDs only.
    Disqualified/empty candidates keep ``rank=null`` and the evaluator reason.
    Candidates are ``(support_kp, spur, stem)``; primary \\(E\\) is fixed and
    not part of the phenotype.
    """
    candidate = score.candidate
    return {
        "candidate_index": int(score.candidate_index),
        "support_kp": _json_float(candidate.support_kp),
        "youngs_modulus_pa": {
            "spur": _json_float(candidate.spur),
            "stem": _json_float(candidate.stem),
        },
        "log10_vector": [
            _json_log10(candidate.support_kp),
            _json_log10(candidate.spur),
            _json_log10(candidate.stem),
        ],
        "aggregate_sinkhorn": _json_float(score.aggregate_sinkhorn),
        "per_direction_sinkhorn": _per_direction_sinkhorn_to_json(
            score.per_direction_sinkhorn
        ),
        "rank": int(score.rank) if score.rank is not None else None,
        "is_gt": bool(score.is_gt),
        "instability_fraction": _json_float(score.instability_fraction),
        "disqualified": bool(score.disqualified),
        "disqualification_reason": score.disqualification_reason,
    }


def _winner_summary(evaluation: YoungsModulusEvaluation) -> dict[str, Any] | None:
    winner = next((score for score in evaluation.scores if score.rank == 1), None)
    if winner is None:
        return None
    gt = evaluation.gt_candidate
    log10_error: dict[str, float | None] = {}
    relative_error: dict[str, float | None] = {}
    for segment in ("support_kp", "spur", "stem"):
        winner_value = _json_float(getattr(winner.candidate, segment))
        gt_value = _json_float(getattr(gt, segment)) if gt is not None else None
        winner_log10 = _json_log10(getattr(winner.candidate, segment))
        gt_log10 = (
            _json_log10(getattr(gt, segment)) if gt is not None else None
        )
        log10_error[segment] = (
            _json_float(winner_log10 - gt_log10)
            if winner_log10 is not None and gt_log10 is not None
            else None
        )
        relative_error[segment] = (
            _json_float(abs(winner_value - gt_value) / abs(gt_value))
            if winner_value is not None and gt_value not in (None, 0.0)
            else None
        )
    return {
        "candidate_index": int(winner.candidate_index),
        "log10_error": log10_error,
        "relative_error": relative_error,
    }


def _structure_result_to_json(evaluation: YoungsModulusEvaluation) -> dict[str, Any]:
    gt = evaluation.gt_candidate
    gt_rank = next(
        (int(score.rank) for score in evaluation.scores if score.is_gt and score.rank is not None),
        None,
    )
    if gt is None:
        gt_support_kp = None
        gt_youngs = {"spur": None, "stem": None}
        gt_log10_vector: list[float | None] = [None, None, None]
    else:
        gt_support_kp = _json_float(gt.support_kp)
        gt_youngs = {
            "spur": _json_float(gt.spur),
            "stem": _json_float(gt.stem),
        }
        gt_log10_vector = [
            _json_log10(gt.support_kp),
            _json_log10(gt.spur),
            _json_log10(gt.stem),
        ]
    return {
        "structure_idx": int(evaluation.structure_idx),
        "gt_support_kp": gt_support_kp,
        "gt_youngs_modulus_pa": gt_youngs,
        "gt_log10_vector": gt_log10_vector,
        "fixed_secondary_e_pa": _json_float(evaluation.fixed_secondary_e_pa),
        "direction_indices": [int(d) for d in evaluation.direction_indices],
        "candidates": [_candidate_to_json_row(score) for score in evaluation.scores],
        "winner": _winner_summary(evaluation),
        "gt_rank": gt_rank,
        "overlay_error": None,
        "export_error": None,
    }


def _aggregate_ranking_report(
    structure_rows: list[dict[str, Any]],
    skipped_structures: list[dict[str, Any]],
    *,
    dataset: str,
    output: str,
    scoring: YoungsModulusScoringConfig,
) -> dict[str, Any]:
    gt_ranks = [
        int(row["gt_rank"])
        for row in structure_rows
        if row.get("gt_rank") is not None
    ]
    gt_rank_histogram: dict[str, int] = {}
    for rank in gt_ranks:
        key = str(int(rank))
        gt_rank_histogram[key] = gt_rank_histogram.get(key, 0) + 1

    winner_log10_error_mean: dict[str, float | None] = {
        "support_kp": None,
        "spur": None,
        "stem": None,
    }
    for segment in ("support_kp", "spur", "stem"):
        values = [
            float(row["winner"]["log10_error"][segment])
            for row in structure_rows
            if row.get("winner") is not None
            and row["winner"].get("log10_error", {}).get(segment) is not None
            and math.isfinite(float(row["winner"]["log10_error"][segment]))
        ]
        if values:
            winner_log10_error_mean[segment] = _json_float(sum(values) / len(values))

    return {
        "dataset": str(dataset),
        "output": str(output),
        "scoring": dataclasses.asdict(scoring),
        "structures": structure_rows,
        "skipped_structures": list(skipped_structures),
        "aggregate": {
            "n_structures": int(len(structure_rows) + len(skipped_structures)),
            "n_evaluated": int(len(structure_rows)),
            "n_skipped": int(len(skipped_structures)),
            "gt_rank_histogram": gt_rank_histogram,
            "winner_log10_error_mean": winner_log10_error_mean,
            "winner_support_kp_log10_error_mean": winner_log10_error_mean["support_kp"],
        },
    }


def _write_ranking_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _finalize_structure_outputs(
    *,
    dataset: BatchedSysIdDataset,
    evaluation: YoungsModulusEvaluation,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = _structure_result_to_json(evaluation)
    structure_dir = output_dir / f"structure_{int(evaluation.structure_idx):03d}"

    try:
        structure_dir.mkdir(parents=True, exist_ok=True)
        overlay_indices = select_overlay_candidate_indices(
            evaluation.scores,
            max_candidates=int(getattr(args, "max_overlay_candidates", 8)),
        )
        if overlay_indices:
            overlay_eps = overlay_episodes_from_replay_evaluation(
                evaluation,
                overlay_indices,
            )
            write_youngs_modulus_overlay_html(
                overlay_eps,
                structure_dir / "youngs_modulus_overlay.html",
                max_overlay_candidates=int(getattr(args, "max_overlay_candidates", 8)),
                title=(
                    f"Young's modulus overlay — structure "
                    f"{int(evaluation.structure_idx)}"
                ),
            )
        else:
            row["overlay_error"] = "no_eligible_candidates_for_overlay"
    except Exception as exc:
        row["overlay_error"] = str(exc)

    if bool(args.export_replays):
        try:
            specs_and_replays = []
            for score in evaluation.scores:
                cand_idx = int(score.candidate_index)
                specs_and_replays.append(
                    (
                        ReplayCandidateSpec(
                            candidate_index=cand_idx,
                            params=evaluation.applied_params[cand_idx],
                            stiffnesses={
                                "support_kp": float(score.candidate.support_kp),
                                "spur_e_pa": float(score.candidate.spur),
                                "stem_e_pa": float(score.candidate.stem),
                            },
                        ),
                        evaluation.replay_episodes[cand_idx],
                    )
                )
            export_replay_candidates_for_structure(
                output_dir,
                source_dataset=dataset,
                source_structure_idx=int(evaluation.structure_idx),
                specs_and_replays=specs_and_replays,
                source_direction_indices=evaluation.direction_indices,
                command_argv=sys.argv,
            )
        except Exception as exc:
            row["export_error"] = str(exc)

    return row


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _resolve_sim_build_knobs(ranges: dict) -> tuple[
    ImpedanceGains,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float | None,
]:
    sb = parse_sim_build(ranges)
    if sb is None:
        return (
            VIC_GAINS,
            dict(JOINT_ANGULAR_KD_OVERRIDES),
            dict(JOINT_LINEAR_KD_OVERRIDES),
            dict(JOINT_ANGULAR_KP_OVERRIDES),
            dict(JOINT_LINEAR_KP_OVERRIDES),
            None,
        )
    return (
        ImpedanceGains(
            linear_k=sb.vic_gains.linear_k,
            linear_d=sb.vic_gains.linear_d,
            angular_k=sb.vic_gains.angular_k,
            angular_d=sb.vic_gains.angular_d,
        ),
        dict(sb.joint_angular_kd_overrides),
        dict(sb.joint_linear_kd_overrides),
        dict(sb.joint_angular_kp_overrides),
        dict(sb.joint_linear_kp_overrides),
        sb.joint_damping_ratio,
    )


def build_sim_config(
    *,
    num_envs: int,
    settle_substeps: int | None = None,
    settle_gravity_ramp: bool | None = None,
    settle_quiet_every: int | None = None,
    device: str | None = None,
    ranges: dict | None = None,
    reuse_replicated_mujoco: bool = False,
) -> BatchedHeterogeneousCoupledSimConfig:
    if ranges is None:
        ranges = load_ranges(default_ranges_fixture_path())
    (
        vic_gains,
        joint_angular_kd,
        joint_linear_kd,
        joint_angular_kp,
        joint_linear_kp,
        joint_damping_ratio,
    ) = _resolve_sim_build_knobs(ranges)
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(num_envs))
    settle = SETTLE_SUBSTEPS if settle_substeps is None else int(settle_substeps)
    gravity_ramp = SETTLE_GRAVITY_RAMP if settle_gravity_ramp is None else bool(settle_gravity_ramp)
    quiet_every = SETTLE_QUIET_EVERY if settle_quiet_every is None else settle_quiet_every
    return dataclasses.replace(
        gym_cfg,
        runtime=dataclasses.replace(
            gym_cfg.runtime,
            control_hz=CONTROL_HZ,
            sub_dt=SUB_DT,
            env_spacing=ENV_SPACING,
            device=device,
        ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=settle,
            settle_gravity_ramp=gravity_ramp,
            settle_quiet_every=quiet_every,
        ),
        controller=dataclasses.replace(
            gym_cfg.controller,
            vic_gains=vic_gains,
        ),
        fruiting_system=dataclasses.replace(
            gym_cfg.fruiting_system,
            joint_angular_kd_overrides=joint_angular_kd,
            joint_linear_kd_overrides=joint_linear_kd,
            joint_angular_kp_overrides=joint_angular_kp,
            joint_linear_kp_overrides=joint_linear_kp,
            joint_damping_ratio=joint_damping_ratio,
        ),
        robot=dataclasses.replace(
            gym_cfg.robot,
            reuse_replicated_mujoco=bool(reuse_replicated_mujoco),
        ),
    )


def _collection_control_hz(collection: dict, *, default: float = CONTROL_HZ) -> float:
    if "control_hz" not in collection:
        return float(default)
    return float(collection["control_hz"])


def _resolve_structure_indices(dataset, requested: tuple[int, ...] | None) -> list[int]:
    if requested is not None:
        return list(requested)
    return list(range(len(dataset.structure_summaries())))


def _resolve_n_holds(dataset, collection: dict) -> int | None:
    if "n_holds" in collection:
        return int(collection["n_holds"])
    for ep in dataset.episode_entries():
        if ep.get("n_holds") is not None:
            return int(ep["n_holds"])
    move = collection.get("movement_per_step_m")
    total = collection.get("total_movement_m")
    if move is not None and total is not None:
        from apple_pick_sim.system_id import derive_n_steps

        return int(
            derive_n_steps(
                movement_per_step_m=float(move),
                total_movement_m=float(total),
            )
        )
    return None


def _resolve_n_directions(dataset, collection: dict) -> int:
    if "num_directions" in collection:
        return int(collection["num_directions"])
    max_dir = -1
    for ep in dataset.episode_entries():
        if ep.get("direction_idx") is not None:
            max_dir = max(max_dir, int(ep["direction_idx"]))
        elif ep.get("dir_idx") is not None:
            max_dir = max(max_dir, int(ep["dir_idx"]))
    if max_dir >= 0:
        return int(max_dir) + 1
    raise ValueError("cannot resolve num_directions from dataset manifest")


def _settle_config_kwargs(*, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "settle_substeps": args.settle_substeps,
        "settle_gravity_ramp": args.settle_gravity_ramp,
        "settle_quiet_every": args.settle_quiet_every,
    }


def _make_build_env_fn(
    *,
    ranges_path: str,
    topology_seed: int,
    control_hz: float,
    device: str | None = None,
    settle_config: dict | None = None,
    reuse_replicated_mujoco: bool = False,
):
    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv

    ranges = load_ranges(ranges_path)

    def build_env_fn(
        *,
        num_envs: int,
        per_env_params: list,
        max_episode_steps: int,
        gripper=None,
        per_env_grippers=None,
    ) -> ApplePickBatchedSysIdEnv:
        if gripper is not None and per_env_grippers is not None:
            raise ValueError(
                "scalar gripper and per_env_grippers cannot both be provided"
            )
        sim_config = build_sim_config(
            num_envs=num_envs,
            device=device,
            ranges=ranges,
            reuse_replicated_mujoco=reuse_replicated_mujoco,
            **(settle_config or {}),
        )
        sim_config = dataclasses.replace(
            sim_config,
            runtime=dataclasses.replace(sim_config.runtime, control_hz=float(control_hz)),
        )
        if gripper is not None:
            sim_config = dataclasses.replace(
                sim_config,
                robot=dataclasses.replace(sim_config.robot, gripper=gripper),
            )
        return ApplePickBatchedSysIdEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            ranges_path=ranges_path,
            topology_seed=int(topology_seed),
            use_settle_cache=False,
            per_env_params=per_env_params,
            per_env_grippers=per_env_grippers,
            control_hz=float(control_hz),
            sim_config=sim_config,
        )

    return build_env_fn


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
        help="Directory for ranking reports and overlays.",
    )
    p.add_argument(
        "--structure-indices",
        type=parse_comma_separated_ints,
        default=None,
        help="Comma-separated structure indices (default: all in manifest).",
    )
    p.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include manifest episodes marked excluded (debug only).",
    )
    p.add_argument(
        "--max-envs-per-batch",
        type=int,
        default=MAX_ENVS_PER_BATCH,
        help=(
            "Chunk candidates so chunk_size*num_directions <= this "
            f"(default {MAX_ENVS_PER_BATCH}; 0 = no chunking)."
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
        "--max-candidates",
        type=int,
        default=0,
        help="Refuse grids larger than this per structure (0 = no cap).",
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
        "--support-kp-values",
        type=str,
        default=None,
        help=(
            "Comma-separated physical support joint k_p grid values (shared "
            "angular+linear, N/m-/N*m/rad-like). Mutually exclusive with "
            f"--log10-support-kp; defaults to {DEFAULT_SUPPORT_KP_VALUES!r} "
            "when neither is given."
        ),
    )
    p.add_argument(
        "--log10-support-kp",
        type=str,
        default=None,
        help="Comma-separated log10(support k_p) grid values.",
    )
    p.add_argument("--log10-e-spur", type=str, default="7.5")
    p.add_argument("--log10-e-stem", type=str, default="7.0")
    p.add_argument(
        "--include-gt-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert the structure's exact GT E when absent from the grid.",
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
        "--export-replays",
        action="store_true",
        help="Export per-candidate replay mini-datasets.",
    )
    p.add_argument(
        "--max-overlay-candidates",
        type=_positive_int,
        default=8,
        help="Cap overlay candidates per structure.",
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
    p.add_argument(
        "--record-video",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write GL viewer frames to PATH.mp4 for the batched multi-world "
            "view (requires --viewer gl; --headless OK). FPS matches sim "
            "control_hz. Chunked candidate batches append into one file."
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


def require_gl_frame_capture(viewer: object) -> None:
    """Raise ``SystemExit`` unless ``viewer`` supports ``get_frame`` (ViewerGL)."""
    if not hasattr(viewer, "get_frame"):
        raise SystemExit(
            "--record-video requires a GL viewer with get_frame(); "
            "pass --viewer gl (optionally --headless)."
        )


def make_grid_on_step(
    viewer: object,
    *,
    control_hz: float,
    graphical: bool,
    use_viewer: bool,
    show_pull_direction: bool = False,
    recorder: GlVideoRecorder | None = None,
) -> Callable[..., bool]:
    """Render each replay frame; optionally capture GL frames into ``recorder``."""
    frame_dt = 1.0 / float(control_hz)
    viewer_state: dict[str, object] = {"model": None}

    def on_step(*, frame_idx: int, env: object) -> bool:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False
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
        if recorder is not None:
            if recorder.fps is None:
                recorder.set_fps(hz)
            recorder.capture(viewer)
        elif graphical:
            time.sleep(max(0.0, frame_dt))
        return True

    return on_step


def _render_frame(
    viewer: object,
    env: object,
    sim_time: float,
    *,
    obs: dict | None = None,
    show_pull_direction: bool = False,
) -> None:
    sim = env._sim
    scene = sim.scene
    if scene.last_vbd_contacts is not None:
        contacts = scene.last_vbd_contacts
    else:
        contacts = scene.cable.model.collide(
            scene.cable.state_0,
            collision_pipeline=scene.cable_collision_pipeline,
        )
    viewer.begin_frame(sim_time)
    viewer.log_state(scene.cable.state_0)
    viewer.log_contacts(contacts, scene.cable.state_0)
    if show_pull_direction and obs is not None:
        layout = sim.layout
        excitation = obs.get("excitation_direction")
        if layout is not None and excitation is not None:
            from apple_pick_sim.batched_viz import log_batched_movement_direction_arrows
            import numpy as np

            if hasattr(excitation, "detach"):
                directions = excitation.detach().cpu().numpy()
            else:
                directions = np.asarray(excitation, dtype=np.float64)
            log_batched_movement_direction_arrows(
                viewer,
                scene,
                layout,
                directions=directions,
                bufs=sim.obs_bufs,
            )
    viewer.end_frame()


def _candidates_for_structure(
    dataset: BatchedSysIdDataset,
    args: argparse.Namespace,
    structure_idx: int,
    *,
    parser: argparse.ArgumentParser,
    include_gt: bool | None = None,
) -> list:
    if include_gt is None:
        include_gt = bool(args.include_gt_candidate)
    support_kp_values = args.support_kp_values
    log10_support_kp = args.log10_support_kp
    if support_kp_values is None and log10_support_kp is None:
        support_kp_values = DEFAULT_SUPPORT_KP_VALUES
    candidates = candidates_from_support_kp_grid_cli(
        support_kp_values=support_kp_values,
        log10_support_kp=log10_support_kp,
        log10_e_spur=str(args.log10_e_spur),
        log10_e_stem=str(args.log10_e_stem),
    )
    if not candidates:
        parser.error("candidate grid is empty; provide at least one log10-E value per segment")
    if include_gt:
        gt = gt_support_kp_youngs_candidate_from_structure(dataset, int(structure_idx))
        candidates = maybe_include_gt_candidate(candidates, gt, include_gt=True)
    if int(args.max_candidates) > 0 and len(candidates) > int(args.max_candidates):
        parser.error(
            f"candidate grid has {len(candidates)} entries, exceeding "
            f"--max-candidates={int(args.max_candidates)}"
        )
    return candidates


def _run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    viewer: object,
    recorder: GlVideoRecorder | None = None,
) -> dict[str, Any]:
    device = args.device
    if device == "cuda":
        device = "cuda:0"

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.overwrite):
        raise SystemExit(
            f"output directory {output_dir} is non-empty; pass --overwrite to continue"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BatchedSysIdDataset(args.dataset)
    collection = dataset.manifest.get("collection", {})

    ranges_path = collection.get("ranges_path")
    if not ranges_path:
        ranges_path = str(default_ranges_fixture_path())
    ranges = load_ranges(ranges_path)
    topology_seed = int(collection.get("topology_seed", 42))
    control_hz = _collection_control_hz(collection)
    num_directions = _resolve_n_directions(dataset, collection)
    if num_directions < 1:
        parser.error("manifest num_directions must be >= 1")

    structure_indices = _resolve_structure_indices(dataset, args.structure_indices)
    if not structure_indices:
        raise SystemExit("No structure indices to evaluate.")

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
    action_dim = 19 if mode == "vic_pose" else 6

    replay_seed = args.seed
    if replay_seed is None and "seed" in collection:
        replay_seed = int(collection["seed"])

    settle_config = _settle_config_kwargs(args=args)
    if mode == "vic_pose":
        if bool(args.include_gt_candidate):
            print(
                "warning: --include-gt-candidate ignored for vic_pose_v1 "
                "(no sim-oracle GT)",
                file=sys.stderr,
            )
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
        include_gt = False
    else:
        build_env_fn = _make_build_env_fn(
            ranges_path=str(ranges_path),
            topology_seed=topology_seed,
            control_hz=control_hz,
            device=device,
            settle_config=settle_config,
        )
        replay_sim_config = build_sim_config(
            num_envs=1,
            ranges=ranges,
            **settle_config,
        )
        include_gt = bool(args.include_gt_candidate)
    scoring = YoungsModulusScoringConfig(
        use_median=bool(args.use_median),
        hold_id_onehot=bool(args.hold_id_onehot),
        pool_directions=bool(args.pool_directions),
        n_holds=_resolve_n_holds(dataset, collection),
        n_directions=int(num_directions),
        device=device,
    )

    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    use_viewer = graphical or getattr(args, "viewer", None) != "null"
    show_pull_direction = bool(args.show_pull_direction) and graphical
    on_step = make_grid_on_step(
        viewer,
        control_hz=control_hz,
        graphical=graphical,
        use_viewer=use_viewer,
        show_pull_direction=show_pull_direction,
        recorder=recorder,
    )

    candidates_by_structure: dict[int, list] = {}
    candidate_errors: dict[int, str] = {}
    for structure_idx in structure_indices:
        try:
            candidates_by_structure[int(structure_idx)] = _candidates_for_structure(
                dataset,
                args,
                int(structure_idx),
                parser=parser,
                include_gt=include_gt,
            )
        except Exception as exc:
            if bool(args.fail_fast):
                raise
            candidate_errors[int(structure_idx)] = str(exc)

    structure_results: list[dict[str, Any]] = []
    if bool(getattr(args, "multi_structure_batch", False)):
        requested = tuple(
            (structure_idx, candidates_by_structure[structure_idx])
            for structure_idx in structure_indices
            if structure_idx in candidates_by_structure
        )
        batch = evaluate_youngs_modulus_structures(
            dataset=dataset,
            structures=requested,
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
        )
        for structure_idx in structure_indices:
            evaluation = batch.evaluations.get(int(structure_idx))
            error = candidate_errors.get(
                int(structure_idx), batch.errors.get(int(structure_idx))
            )
            structure_results.append(
                {
                    "structure_idx": int(structure_idx),
                    "evaluation": evaluation,
                    "error": error,
                }
            )
        failed_count = sum(row["error"] is not None for row in structure_results)
        fused_count = max(
            0, int(batch.prepared_structures) - len(batch.retried_structures)
        )
        print(
            f"structures prepared={int(batch.prepared_structures)} "
            f"fused={fused_count} retried={len(batch.retried_structures)} "
            f"failed={failed_count}"
        )
        diagnostics = batch.replay_diagnostics
        if diagnostics is None:
            print("candidate_blocks=0 flattened_envs=0 chunks=0 chunk_envs=()")
            build_seconds = 0.0
            replay_seconds = 0.0
        else:
            print(
                f"candidate_blocks={int(diagnostics.candidate_blocks)} "
                f"flattened_envs={int(diagnostics.flattened_envs)} "
                f"chunks={len(diagnostics.chunk_env_counts)} "
                f"chunk_envs={diagnostics.chunk_env_counts}"
            )
            build_seconds = float(diagnostics.build_seconds)
            replay_seconds = float(diagnostics.replay_seconds)
        print(
            f"build_settle_seconds={build_seconds:.6f} "
            f"replay_seconds={replay_seconds:.6f} "
            f"scoring_seconds={float(batch.scoring_seconds):.6f} "
            f"total_seconds={float(batch.total_seconds):.6f}"
        )
    else:
        for structure_idx in structure_indices:
            if int(structure_idx) in candidate_errors:
                structure_results.append(
                    {
                        "structure_idx": int(structure_idx),
                        "evaluation": None,
                        "error": candidate_errors[int(structure_idx)],
                    }
                )
                continue
            candidates = candidates_by_structure[int(structure_idx)]
            try:
                evaluation = evaluate_youngs_modulus_candidates(
                    dataset=dataset,
                    structure_idx=int(structure_idx),
                    candidates=candidates,
                    num_directions=int(num_directions),
                    build_env_fn=build_env_fn,
                    scoring=scoring,
                    max_envs_per_batch=int(args.max_envs_per_batch),
                    seed=replay_seed,
                    include_excluded=bool(args.include_excluded),
                    on_step=on_step,
                    replay_sim_config=replay_sim_config,
                    action_dim=action_dim,
                )
                structure_results.append(
                    {
                        "structure_idx": int(structure_idx),
                        "evaluation": evaluation,
                        "error": None,
                    }
                )
                print(
                    f"structure {int(structure_idx)}: "
                    f"candidates={len(candidates)} directions={num_directions}"
                )
            except Exception as exc:
                if bool(args.fail_fast):
                    raise
                message = str(exc)
                structure_results.append(
                    {
                        "structure_idx": int(structure_idx),
                        "evaluation": None,
                        "error": message,
                    }
                )
                print(f"ERROR structure {int(structure_idx)}: {message}")

    structure_rows: list[dict[str, Any]] = []
    skipped_structures: list[dict[str, Any]] = []
    for row in structure_results:
        if row.get("error") is not None:
            skipped_structures.append(
                {
                    "structure_idx": int(row["structure_idx"]),
                    "error": str(row["error"]),
                }
            )
            continue
        evaluation = row.get("evaluation")
        assert evaluation is not None
        structure_rows.append(
            _finalize_structure_outputs(
                dataset=dataset,
                evaluation=evaluation,
                output_dir=output_dir,
                args=args,
            )
        )

    ranking_payload = _aggregate_ranking_report(
        structure_rows,
        skipped_structures,
        dataset=str(args.dataset),
        output=str(output_dir),
        scoring=scoring,
    )
    _write_ranking_json_atomic(output_dir / "ranking.json", ranking_payload)

    if recorder is not None:
        if recorder.frame_count <= 0:
            raise SystemExit(
                f"--record-video requested but wrote 0 frames ({recorder.path})"
            )
        print(
            f"recorded video frames={recorder.frame_count} path={recorder.path}",
            file=sys.stderr,
        )

    return {
        "dataset": str(args.dataset),
        "output": str(output_dir),
        "structure_indices": structure_indices,
        "num_directions": int(num_directions),
        "scoring": scoring,
        "structure_results": structure_results,
        "ranking": ranking_payload,
    }


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    recorder: GlVideoRecorder | None = None
    record_path = getattr(args, "record_video", None)
    if record_path is not None:
        require_gl_frame_capture(viewer)
        recorder = GlVideoRecorder(record_path)
    try:
        result = _run(args, parser, viewer=viewer, recorder=recorder)
        failures = [
            row for row in result["structure_results"] if row.get("error") is not None
        ]
        if failures and all(row.get("evaluation") is None for row in result["structure_results"]):
            raise SystemExit(1)
    finally:
        if recorder is not None:
            recorder.close()
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
