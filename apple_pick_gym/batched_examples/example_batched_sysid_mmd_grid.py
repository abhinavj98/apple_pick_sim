"""Parallel batched sys-ID grid replay and ranking validation.

Replay recorded actions across a bend-stiffness candidate grid for each structure
in a ``batched_sysid_v1`` dataset. Use ``--score-mse`` and/or ``--score-wasserstein``
for hold-phase ranking validation beside ``--replay-only`` summaries. Biased MMD
scoring remains available in the library (``evaluate_batched_mmd_grid``); the CLI
does not expose ``--score-mmd`` yet.

Build params default to privileged oracle ``true_params_for_structure``; pass
``--infer-params`` for obs-inferred digital-twin geometry (V.4.2.1). Init state
defaults to settle + metadata/obs init; ``--use-snapshot`` restores post-weld
snapshots (orthogonal to build params).

Run from the repository root::

    uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \\
      --dataset /tmp/batched_sysid_dataset \\
      --replay-only \\
      --primary-bend-stiffness-values 1e-4,2e-4 \\
      --secondary-bend-stiffness-values 1e-4 \\
      --spur-bend-stiffness-values 1e-4 \\
      --stem-bend-stiffness-values 1e-4,2e-4

Sim build (VIC gains, settle substeps, control_hz, …) is configured via module
constants in this file as fallbacks; when the ranges JSON includes optional
``sim_build``, those values win. Settle phase knobs also accept
``--settle-substeps``, ``--settle-gravity-ramp``, and ``--settle-quiet-every``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import numpy as np
import newton.examples

from apple_pick_gym.examples.run_system_identification import parse_positive_float_grid
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    DomainRandomizationConfig,
    EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES,
    EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KD_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KP_OVERRIDES,
    FruitingSystemConfig,
    MujocoConfig,
    ObsConfig,
    RobotConfig,
    RuntimeConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)
from apple_pick_sim.fruiting_system import default_ranges_fixture_path, load_ranges, parse_sim_build
from apple_pick_sim.fruiting_system.params import PLACEHOLDER_EE_MASS_KG, GripperProxyConfig
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_DEFAULT_ITERATIONS

# --- Sim build fallbacks (used when ranges omit ``sim_build``) ---
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
GRIPPER_PROXY = GripperProxyConfig(mass=PLACEHOLDER_EE_MASS_KG)

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")


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

_PLOT_METRIC_CHOICES = tuple(
    sorted(get_args(__import__("apple_pick_gym.grid_viz_plotly", fromlist=["Metric"]).Metric))
)


def _parse_plot_metrics(value: str) -> tuple[str, ...]:
    metrics = tuple(m.strip() for m in str(value).split(",") if m.strip())
    if not metrics:
        raise ValueError("--plot-metrics must contain at least one metric")
    allowed = set(_PLOT_METRIC_CHOICES)
    invalid = sorted({m for m in metrics if m not in allowed})
    if invalid:
        raise ValueError(
            f"invalid --plot-metrics entries: {', '.join(invalid)}; "
            f"allowed: {', '.join(_PLOT_METRIC_CHOICES)}"
        )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for m in metrics:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return tuple(out)


def parse_comma_separated_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of non-negative integers."""
    if value.strip() == "":
        raise argparse.ArgumentTypeError("expected at least one integer")
    parts = value.split(",")
    if any(part.strip() == "" for part in parts):
        raise argparse.ArgumentTypeError("empty index entries are not allowed")
    try:
        values = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("structure indices must be integers") from exc
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("structure indices must be >= 0")
    return values


def build_sim_config(
    *,
    num_envs: int,
    settle_substeps: int | None = None,
    settle_gravity_ramp: bool | None = None,
    settle_quiet_every: int | None = None,
    device: str | None = None,
    ranges: dict | None = None,
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
        runtime=dataclasses.replace(gym_cfg.runtime, control_hz=CONTROL_HZ, device=device),
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
    )


def chunk_candidates(
    candidates: Sequence[object],
    *,
    max_envs_per_batch: int,
    num_directions: int,
) -> list[list[object]]:
    """Split candidates so each chunk fits ``max_envs_per_batch`` parallel envs."""
    items = list(candidates)
    if not items:
        return []
    limit = int(max_envs_per_batch)
    directions = int(num_directions)
    if limit <= 0:
        return [items]
    max_chunk_size = limit // directions
    if max_chunk_size < 1:
        raise SystemExit(
            f"--max-envs-per-batch ({limit}) must be >= num_directions ({directions})"
        )
    return [items[i : i + max_chunk_size] for i in range(0, len(items), max_chunk_size)]


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Directory containing batched_sysid_v1 manifest.json and episodes/.",
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
        help="Include manifest episodes marked excluded (debug only; default skips them).",
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
        "--max-candidates",
        type=int,
        default=0,
        help="Cap evaluated stiffness candidates per structure (0 = full grid).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Replay RNG seed (default: manifest collection.seed).",
    )
    p.add_argument(
        "--replay-only",
        action="store_true",
        help="Replay recorded actions and print per-structure row-count summary.",
    )
    p.add_argument(
        "--score-mse",
        action="store_true",
        help="Compute per-candidate MSE vs recorded GT.",
    )
    p.add_argument(
        "--score-wasserstein",
        action="store_true",
        help=(
            "Compute per-candidate Sinkhorn divergence on hold transition bags "
            "(requires geomloss + torch; see docs/sysid-transition-features.md)."
        ),
    )
    p.add_argument(
        "--use-median",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use full-hold median hold→hold features for Wasserstein/MMD and paired "
            "per-hold median MSE (default: on). --no-use-median restores frame-wise bags/MSE."
        ),
    )
    p.add_argument(
        "--mse-hold-aggregation",
        type=str,
        choices=("mean", "median", "none"),
        default=None,
        help=(
            "Deprecated alias for --use-median / --no-use-median "
            "(mean|median → --use-median; none → --no-use-median)."
        ),
    )
    p.add_argument(
        "--mse-hold-latter-half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated no-op: full hold windows are always used (latter-half burn-in removed)."
        ),
    )
    p.add_argument(
        "--hold-id-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append per-hold one-hot identity to Wasserstein/MMD transition features "
            "(default: on; --no-hold-id-onehot to disable)."
        ),
    )
    p.add_argument(
        "--pool-directions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pool transition bags across directions for Sinkhorn (default: on; "
            "matches gate_pooled_dirs). Automatically appends a fixed-width "
            "dir_idx one-hot so pooled rows retain excitation identity. "
            "Use --no-pool-directions for per-direction bags."
        ),
    )
    p.add_argument(
        "--score-json-output",
        type=str,
        default=None,
        help="If set, write structured per-structure scoring summary JSON (for gate harness).",
    )
    p.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="If set, write Plotly HTML plots + JSON rows under this directory.",
    )
    p.add_argument(
        "--export-replay-dir",
        type=str,
        default=None,
        help=(
            "If set, write one batched_sysid_v1-compatible mini-dataset per candidate "
            "under this directory (structure_XXX/candidates/cYYY/)."
        ),
    )
    p.add_argument(
        "--export-skip-existing",
        action="store_true",
        help="Skip candidate export when cYYY/manifest.json already exists.",
    )
    p.add_argument(
        "--plot-metrics",
        type=_parse_plot_metrics,
        default=("err_pos_hold", "err_force_hold", "err_torque_hold"),
        help=(
            "Comma-separated metrics to plot: err_pos_all/hold, err_force_all/hold, "
            "err_torque_all/hold, err_woody_pos_all/hold."
        ),
    )
    p.add_argument(
        "--grid-values-are-gt-multipliers",
        action="store_true",
        help=(
            "Interpret --*-bend-stiffness-values as multipliers of the per-structure GT "
            "stiffness (builds a different absolute grid per structure). GT is still "
            "forced into the candidate list."
        ),
    )
    for segment in ROD_SEGMENTS:
        p.add_argument(
            f"--{segment}-bend-stiffness-values",
            dest=f"{segment}_bend_stiffness_values",
            type=parse_positive_float_grid,
            default=None,
            help=f"Comma-separated candidate values for {segment}.bend_stiffness.",
        )
    p.add_argument(
        "--use-snapshot",
        action="store_true",
        help=(
            "Restore post-weld initial_states/sXX_dYY.npz (settle_substeps=0, skip metadata "
            "init). Diagnostic only; requires collect with --save-snapshot. Independent of "
            "build-params (--infer-params)."
        ),
    )
    p.add_argument(
        "--infer-params",
        action="store_true",
        help=(
            "Build replay trees from obs-inferred digital-twin params instead of privileged "
            "true FruitingSystemParams (oracle default). Independent of --use-snapshot."
        ),
    )
    p.add_argument(
        "--settle-substeps",
        type=int,
        default=None,
        help=f"VBD substeps before runtime (default: {SETTLE_SUBSTEPS}).",
    )
    p.add_argument(
        "--settle-gravity-ramp",
        action=argparse.BooleanOptionalAction,
        default=SETTLE_GRAVITY_RAMP,
        help="Linear 0→−9.81 m/s² gravity ramp over all settle substeps (default: off).",
    )
    p.add_argument(
        "--settle-quiet-every",
        type=int,
        default=SETTLE_QUIET_EVERY,
        metavar="N",
        help=(
            "Zero all fruiting-system body twists every N VBD settle substeps "
            "(device-side; default: 300)."
        ),
    )
    return p


def _require_grid_values(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [
        f"--{segment}-bend-stiffness-values"
        for segment in ROD_SEGMENTS
        if getattr(args, f"{segment}_bend_stiffness_values") is None
    ]
    if missing:
        parser.error(
            "the following arguments are required when evaluating candidates: "
            + ", ".join(missing)
        )


def _structure_indices_from_args(dataset, args: argparse.Namespace) -> list[int]:
    if args.structure_indices is not None:
        return list(args.structure_indices)
    return list(range(len(dataset.structure_summaries())))


def _build_candidate_grid(args: argparse.Namespace):
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import iter_bend_stiffness_candidates

    return list(
        iter_bend_stiffness_candidates(
            primary_values=args.primary_bend_stiffness_values,
            secondary_values=args.secondary_bend_stiffness_values,
            spur_values=args.spur_bend_stiffness_values,
            stem_values=args.stem_bend_stiffness_values,
        )
    )


def _scaled_grid_from_gt(
    *,
    multipliers: tuple[float, ...],
    gt_value: float,
    eps: float = 1e-12,
) -> tuple[float, ...]:
    base = float(gt_value)
    if base <= 0.0:
        base = float(eps)
    return tuple(float(m) * base for m in multipliers)


def _candidates_for_structure(dataset, args: argparse.Namespace, structure_idx: int):
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
        ensure_gt_candidate_in_grid,
        gt_bend_stiffness_candidate_from_structure,
    )

    gt = gt_bend_stiffness_candidate_from_structure(dataset, int(structure_idx))
    if bool(getattr(args, "grid_values_are_gt_multipliers", False)):
        # Treat user-provided values as multipliers; build structure-specific absolute grids.
        primary_vals = _scaled_grid_from_gt(
            multipliers=tuple(args.primary_bend_stiffness_values),
            gt_value=float(gt.primary),
        )
        secondary_vals = _scaled_grid_from_gt(
            multipliers=tuple(args.secondary_bend_stiffness_values),
            gt_value=float(gt.secondary),
        )
        spur_vals = _scaled_grid_from_gt(
            multipliers=tuple(args.spur_bend_stiffness_values),
            gt_value=float(gt.spur),
        )
        stem_vals = _scaled_grid_from_gt(
            multipliers=tuple(args.stem_bend_stiffness_values),
            gt_value=float(gt.stem),
        )
        from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import iter_bend_stiffness_candidates

        candidates = list(
            iter_bend_stiffness_candidates(
                primary_values=primary_vals,
                secondary_values=secondary_vals,
                spur_values=spur_vals,
                stem_values=stem_vals,
            )
        )
    else:
        candidates = _build_candidate_grid(args)
    max_candidates = int(args.max_candidates)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    candidates = ensure_gt_candidate_in_grid(candidates, gt)
    return candidates


def _settle_config_kwargs(*, args: argparse.Namespace, use_snapshot: bool) -> dict:
    if use_snapshot:
        return {
            "settle_substeps": 0,
            "settle_gravity_ramp": False,
            "settle_quiet_every": None,
        }
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
    use_snapshot: bool = False,
    device: str | None = None,
    settle_config: dict | None = None,
):
    import dataclasses

    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv

    ranges = load_ranges(ranges_path)

    def build_env_fn(
        *,
        num_envs: int,
        per_env_params: list,
        max_episode_steps: int,
        gripper: GripperProxyConfig | None = None,
    ) -> ApplePickBatchedSysIdEnv:
        sim_config = build_sim_config(
            num_envs=num_envs,
            device=device,
            ranges=ranges,
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
            control_hz=float(control_hz),
            sim_config=sim_config,
        )

    return build_env_fn


def _collection_control_hz(collection: dict, *, default: float = CONTROL_HZ) -> float:
    if "control_hz" not in collection:
        return float(default)
    return float(collection["control_hz"])


def apply_deprecated_mse_cli_flags(args: argparse.Namespace) -> argparse.Namespace:
    """Map deprecated --mse-hold-* flags onto --use-median; latter-half is a no-op."""
    agg = getattr(args, "mse_hold_aggregation", None)
    if agg is not None:
        warnings.warn(
            "--mse-hold-aggregation is deprecated; use --use-median / --no-use-median",
            DeprecationWarning,
            stacklevel=2,
        )
        args.use_median = str(agg) != "none"
    latter = getattr(args, "mse_hold_latter_half", None)
    if latter is not None:
        warnings.warn(
            "--mse-hold-latter-half is deprecated and ignored (full holds are always used)",
            DeprecationWarning,
            stacklevel=2,
        )
    return args


def _resolve_n_holds(dataset, collection: dict) -> int | None:
    """Prefer collection/episode n_holds; fall back to movement geometry."""
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


def _resolve_n_directions(dataset, collection: dict) -> int | None:
    """Prefer collection num_directions; fall back to episode direction indices."""
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
    return None


def _replay_structure(
    *,
    dataset,
    structure_idx: int,
    candidates,
    num_directions: int,
    seed: int | None,
    max_envs_per_batch: int,
    build_env_fn,
    on_step,
    replay_sim_config,
    use_snapshot: bool = False,
    use_oracle_params: bool = True,
    direction_indices=None,
    include_excluded: bool = False,
):
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import replay_candidates_for_structure

    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=candidates,
        num_directions=int(num_directions),
        seed=seed,
        build_env_fn=build_env_fn,
        max_envs_per_batch=int(max_envs_per_batch),
        on_step=on_step,
        replay_sim_config=replay_sim_config,
        use_snapshot=bool(use_snapshot),
        use_oracle_params=bool(use_oracle_params),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    return collectors, len(candidates)


def _print_replay_summary(
    *,
    structure_idx: int,
    n_frames: int,
    num_candidates: int,
    num_directions: int,
    collectors,
) -> None:
    num_envs = num_candidates * num_directions
    row_counts = [collectors.n_rows(env_idx) for env_idx in range(num_envs)]
    print(
        f"structure {structure_idx}: n_frames={n_frames} "
        f"candidates={num_candidates} directions={num_directions} "
        f"collector_rows={row_counts}"
    )


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser, *, viewer: object) -> None:
    device = args.device
    if device == "cuda":
        device = "cuda:0"
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
        UNSTABLE_DISQUALIFY_THRESHOLD,
        direction_episodes_from_collectors,
        gt_bend_stiffness_candidate_from_structure,
        load_recorded_episodes_for_structure,
        replay_instability_fraction_all_frames,
        trajectory_mse,
        trajectory_paired_hold_median_mse,
        warn_recorded_gt_instability,
    )
    from apple_pick_gym.grid_viz_plotly import write_structure_bundle
    from apple_pick_gym.grid_viz_report import (
        summarize_across_structures,
        summarize_final_rank,
        summarize_structure,
    )
    from apple_pick_gym.grid_viz_table import build_grid_viz_rows, _mean_dict_all_directions

    _require_grid_values(parser, args)

    dataset = BatchedSysIdDataset(args.dataset)
    collection = dataset.manifest.get("collection", {})
    num_directions = int(collection.get("num_directions", 0))
    if num_directions < 1:
        raise SystemExit("manifest collection.num_directions must be >= 1")

    ranges_path = collection.get("ranges_path")
    if not ranges_path:
        ranges_path = str(default_ranges_fixture_path())
    ranges = load_ranges(ranges_path)
    topology_seed = int(collection.get("topology_seed", 42))
    control_hz = _collection_control_hz(collection)

    structure_indices = _structure_indices_from_args(dataset, args)
    if not structure_indices:
        raise SystemExit("No structure indices to evaluate.")

    build_env_fn = _make_build_env_fn(
        ranges_path=str(ranges_path),
        topology_seed=topology_seed,
        control_hz=control_hz,
        use_snapshot=bool(args.use_snapshot),
        device=device,
        settle_config=_settle_config_kwargs(args=args, use_snapshot=bool(args.use_snapshot)),
    )

    viewer_state: dict[str, object] = {"initialized": False}
    score_json_structures: list[dict] = []

    def _on_step(*, frame_idx: int, env: object) -> bool:
        sim = getattr(env, "_sim", None)
        if sim is None:
            return True
        scene = getattr(sim, "scene", None)
        if scene is None:
            return True

        if not viewer_state["initialized"]:
            if hasattr(viewer, "set_model") and hasattr(scene, "cable"):
                viewer.set_model(scene.cable.model)
            if hasattr(viewer, "set_world_offsets") and getattr(env, "num_envs", 1) > 1:
                viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
            if hasattr(viewer, "hide_loading_splash"):
                viewer.hide_loading_splash()
            viewer_state["initialized"] = True

        if not hasattr(viewer, "begin_frame"):
            return True

        hz = float(getattr(sim.config.runtime, "control_hz", CONTROL_HZ))
        sim_time = float(frame_idx) / max(hz, 1e-9)

        # Minimal render loop: log current state once per control step.
        viewer.begin_frame(sim_time)
        if hasattr(viewer, "log_state") and hasattr(scene, "cable"):
            viewer.log_state(scene.cable.state_0)
        viewer.end_frame()
        return True

    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import resolve_direction_indices

    for structure_idx in structure_indices:
        candidates = _candidates_for_structure(dataset, args, int(structure_idx))
        gt = gt_bend_stiffness_candidate_from_structure(dataset, int(structure_idx))
        try:
            usable_dirs = resolve_direction_indices(
                dataset,
                structure_idx=int(structure_idx),
                num_directions=int(num_directions),
                include_excluded=bool(args.include_excluded),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        structure_num_directions = len(usable_dirs)
        recorded_eps = load_recorded_episodes_for_structure(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=structure_num_directions,
            direction_indices=usable_dirs,
            include_excluded=bool(args.include_excluded),
        )
        gt_warn_messages = warn_recorded_gt_instability(
            structure_idx=int(structure_idx),
            recorded_eps=recorded_eps,
        )
        for msg in gt_warn_messages:
            print(f"WARNING: {msg}")
        collectors, num_candidates = _replay_structure(
            dataset=dataset,
            structure_idx=int(structure_idx),
            candidates=candidates,
            num_directions=structure_num_directions,
            seed=args.seed,
            max_envs_per_batch=int(args.max_envs_per_batch),
            build_env_fn=build_env_fn,
            on_step=_on_step,
            replay_sim_config=build_sim_config(
                num_envs=1,
                ranges=ranges,
                **_settle_config_kwargs(args=args, use_snapshot=bool(args.use_snapshot)),
            ),
            use_snapshot=bool(args.use_snapshot),
            use_oracle_params=not bool(args.infer_params),
            direction_indices=usable_dirs,
            include_excluded=bool(args.include_excluded),
        )
        if bool(args.replay_only):
            arrays = dataset.load_episode_obs_arrays(int(structure_idx), int(usable_dirs[0]))
            n_frames = int(arrays["action"].shape[0])
            _print_replay_summary(
                structure_idx=int(structure_idx),
                n_frames=n_frames,
                num_candidates=num_candidates,
                num_directions=structure_num_directions,
                collectors=collectors,
            )
            num_envs = num_candidates * structure_num_directions
            unstable_by_env = []
            for env_idx in range(num_envs):
                replay_arrays = collectors.to_arrays(env_idx)
                stable = np.asarray(replay_arrays.get("stable", np.ones(replay_arrays["action"].shape[0])), dtype=bool)
                unstable_by_env.append(int(np.count_nonzero(~stable)))
            from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
                per_candidate_unstable_counts,
            )

            cand_unstable = per_candidate_unstable_counts(
                unstable_by_env,
                num_candidates=num_candidates,
                num_directions=structure_num_directions,
            )
            print(
                f"structure {int(structure_idx)} stability: "
                f"unstable_frames_per_candidate={cand_unstable}"
            )
        if args.export_replay_dir is not None:
            from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
                base_params_for_replay,
            )
            from apple_pick_sim.system_id.batched_replay_export import (
                ReplayCandidateSpec,
                export_replay_candidates_for_structure,
            )

            base_params = base_params_for_replay(
                dataset,
                int(structure_idx),
                use_oracle_params=not bool(args.infer_params),
            )
            specs_and_replays = []
            for cand_idx, candidate in enumerate(candidates):
                replay_eps = direction_episodes_from_collectors(
                    collectors,
                    candidate_index=int(cand_idx),
                    num_directions=int(structure_num_directions),
                )
                applied = candidate.apply_to(base_params)
                specs_and_replays.append(
                    (
                        ReplayCandidateSpec(
                            candidate_index=int(cand_idx),
                            params=applied,
                            stiffnesses={
                                "primary": float(candidate.primary),
                                "secondary": float(candidate.secondary),
                                "spur": float(candidate.spur),
                                "stem": float(candidate.stem),
                            },
                        ),
                        replay_eps,
                    )
                )
            n_exported = export_replay_candidates_for_structure(
                args.export_replay_dir,
                source_dataset=dataset,
                source_structure_idx=int(structure_idx),
                specs_and_replays=specs_and_replays,
                command_argv=sys.argv,
                skip_existing=bool(args.export_skip_existing),
            )
            print(
                f"structure {int(structure_idx)}: exported {n_exported}/{num_candidates} "
                f"replay candidate datasets to {args.export_replay_dir}"
            )
        if bool(args.score_mse) or bool(args.score_wasserstein):
            if bool(args.score_wasserstein):
                try:
                    import geomloss  # noqa: F401
                except ImportError as exc:
                    raise SystemExit(
                        "geomloss is required for --score-wasserstein; "
                        "run: uv sync --extra gym --extra vic --extra dev"
                    ) from exc
                from apple_pick_sim.system_id.wasserstein import (
                    prepare_gt_wasserstein_context,
                    score_candidate_wasserstein,
                )
                from apple_pick_sim.system_id.wasserstein_ranking import (
                    sinkhorn_gt_preference,
                    sinkhorn_mse_spearman,
                )

                gt_wasserstein_context = prepare_gt_wasserstein_context(
                    recorded_eps,
                    use_median=bool(args.use_median),
                    hold_id_onehot=bool(args.hold_id_onehot),
                    n_holds=_resolve_n_holds(dataset, collection),
                    pool_directions=bool(args.pool_directions),
                    n_directions=_resolve_n_directions(dataset, collection),
                )

            from apple_pick_gym.grid_viz_metrics import bend_stiffness_values_match

            gt_candidate_index = next(
                (
                    cand_idx
                    for cand_idx, candidate in enumerate(candidates)
                    if bend_stiffness_values_match(candidate, gt)
                ),
                None,
            )
            use_median = bool(args.use_median)
            wasserstein_results = []
            disqualified_flags: list[bool] = []
            disqualify_reasons: list[list[str]] = []
            err_pos_hold: list[float] = []
            err_force_hold: list[float] = []
            err_torque_hold: list[float] = []

            if bool(args.score_mse):
                print(f"\n=== structure {int(structure_idx)}: MSE vs recorded GT ===")

            for cand_idx in range(num_candidates):
                candidate = candidates[cand_idx]
                replay_eps = direction_episodes_from_collectors(
                    collectors,
                    candidate_index=int(cand_idx),
                    num_directions=int(structure_num_directions),
                )
                direction_instability = [
                    replay_instability_fraction_all_frames(
                        replay=replay_eps[d],
                        recorded=recorded_eps[d],
                    )
                    for d in range(int(structure_num_directions))
                ]
                finite_instability = [
                    float(f) for f in direction_instability if np.isfinite(float(f))
                ]
                unstable_fraction_all = (
                    max(finite_instability) if finite_instability else float("nan")
                )
                reasons: list[str] = []
                disqualified = any(
                    np.isfinite(float(f)) and float(f) > float(UNSTABLE_DISQUALIFY_THRESHOLD)
                    for f in direction_instability
                )
                if disqualified:
                    reasons.append(
                        f"unstable_fraction>{UNSTABLE_DISQUALIFY_THRESHOLD}"
                    )
                disqualified_flags.append(bool(disqualified))

                per_dir = []
                if bool(args.score_mse):
                    for d in range(int(structure_num_directions)):
                        if use_median:
                            metrics = trajectory_paired_hold_median_mse(
                                replay=replay_eps[d],
                                recorded=recorded_eps[d],
                            )
                        else:
                            metrics = trajectory_mse(
                                replay=replay_eps[d],
                                recorded=recorded_eps[d],
                                skip_phase=-1,
                            )
                        per_dir.append(metrics)
                    if not per_dir:
                        raise SystemExit(
                            f"structure {structure_idx} candidate {cand_idx}: no direction metrics"
                        )

                    ft_force_rmse_mean = float(
                        sum(float(m["ft_force_rmse"]) for m in per_dir) / len(per_dir)
                    )
                    ft_torque_rmse_mean = float(
                        sum(float(m["ft_torque_rmse"]) for m in per_dir) / len(per_dir)
                    )
                    tcp_mean = float(
                        sum(float(m["tcp_pos_mse"]) for m in per_dir) / len(per_dir)
                    )
                    apple_mean = float(
                        sum(float(m["apple_pos_mse"]) for m in per_dir) / len(per_dir)
                    )
                    err_pos_hold.append(float(tcp_mean + apple_mean))
                    err_force_hold.append(ft_force_rmse_mean)
                    err_torque_hold.append(ft_torque_rmse_mean)

                if bool(args.score_wasserstein):
                    w_result = score_candidate_wasserstein(
                        candidate_index=int(cand_idx),
                        stiffnesses={
                            "primary": float(candidate.primary),
                            "secondary": float(candidate.secondary),
                            "spur": float(candidate.spur),
                            "stem": float(candidate.stem),
                        },
                        gt_context=gt_wasserstein_context,
                        replay_observations=replay_eps,
                        use_median=bool(args.use_median),
                        hold_id_onehot=bool(args.hold_id_onehot),
                        n_holds=_resolve_n_holds(dataset, collection),
                        pool_directions=bool(args.pool_directions),
                        n_directions=_resolve_n_directions(dataset, collection),
                    )
                    wasserstein_results.append(w_result)
                    if w_result.missing_directions:
                        disqualified_flags[-1] = True
                        reasons.append(
                            "missing_directions="
                            + ",".join(str(d) for d in w_result.missing_directions)
                        )
                disqualify_reasons.append(list(reasons))

                if bool(args.score_mse):
                    for d, metrics in enumerate(per_dir):
                        for key in (
                            "ft_force_rmse",
                            "ft_torque_rmse",
                            "tcp_pos_mse",
                            "apple_pos_mse",
                            "n_used_frames",
                        ):
                            val = float(metrics[key])
                            if not np.isfinite(val):
                                raise SystemExit(
                                    f"structure {structure_idx} candidate {cand_idx} "
                                    f"direction {d}: non-finite {key}={val}"
                                )
                        woody_by_seg = metrics.get("woody_pos_mse_by_segment", {})
                        if woody_by_seg:
                            for seg_name, seg_val in woody_by_seg.items():
                                val = float(seg_val)
                                if not np.isfinite(val):
                                    raise SystemExit(
                                        f"structure {structure_idx} candidate {cand_idx} "
                                        f"direction {d}: non-finite woody_pos_mse_by_segment"
                                        f"[{seg_name}]={val}"
                                    )
                    woody_mean = _mean_dict_all_directions(
                        "woody_pos_mse_by_segment",
                        per_dir,
                        expected_n=len(per_dir),
                    )
                    woody_mean_str = (
                        ", ".join(f"{name}={val:.6g}" for name, val in sorted(woody_mean.items()))
                        if woody_mean
                        else "{}"
                    )
                    used_min = int(min(float(m["n_used_frames"]) for m in per_dir))
                    disq_tag = (
                        f" DISQUALIFIED unstable_fraction_all={unstable_fraction_all:.3g}"
                        if disqualified
                        else ""
                    )
                    hold_mode = "paired_median" if use_median else "frame_mse"
                    print(
                        f"candidate {cand_idx}:{disq_tag} hold_mode={hold_mode} used_frames(min)={used_min} "
                        f"ft_force_rmse_N(mean)={ft_force_rmse_mean:.6g} "
                        f"ft_torque_rmse_Nm(mean)={ft_torque_rmse_mean:.6g} "
                        f"tcp_pos_mse(mean)={tcp_mean:.6g} "
                        f"apple_pos_mse(mean)={apple_mean:.6g} "
                        f"woody_pos_mse_by_segment(mean)={{{woody_mean_str}}}"
                    )

            if bool(args.score_wasserstein):
                print(
                    f"\n=== structure {int(structure_idx)}: Sinkhorn vs recorded GT ==="
                )
                for w_result, disqualified in zip(
                    wasserstein_results, disqualified_flags, strict=True
                ):
                    per_dir_str = ", ".join(
                        f"{direction}={value:.6g}"
                        for direction, value in sorted(
                            w_result.per_direction_sinkhorn.items()
                        )
                    )
                    low_sample = (
                        f" low_sample_dirs={list(w_result.low_sample_directions)}"
                        if w_result.low_sample_directions
                        else ""
                    )
                    missing_dirs = (
                        f" missing_dirs={list(w_result.missing_directions)}"
                        if w_result.missing_directions
                        else ""
                    )
                    disq_tag = " DISQUALIFIED" if disqualified else ""
                    print(
                        f"candidate {w_result.candidate_index}:{disq_tag} "
                        f"aggregate_sinkhorn={w_result.aggregate_sinkhorn:.6g} "
                        f"per_direction={{{per_dir_str}}}{low_sample}{missing_dirs}"
                    )
                if gt_candidate_index is None:
                    print("WARNING: GT candidate index not found in grid list.")
                else:
                    pref = sinkhorn_gt_preference(
                        results=wasserstein_results,
                        gt_candidate_index=int(gt_candidate_index),
                        disqualified=disqualified_flags,
                    )
                    print(
                        f"  sinkhorn_gt_rank={pref.gt_rank} best_is_gt={pref.best_is_gt} "
                        f"best_candidate={pref.best_candidate_index} "
                        f"disqualified={pref.n_disqualified}/{pref.n_candidates}"
                    )
                    gt_w = next(
                        (
                            r
                            for r in wasserstein_results
                            if int(r.candidate_index) == int(gt_candidate_index)
                        ),
                        None,
                    )
                    best_w = next(
                        (
                            r
                            for r in wasserstein_results
                            if pref.best_candidate_index is not None
                            and int(r.candidate_index) == int(pref.best_candidate_index)
                        ),
                        None,
                    )
                    gt_reason = (
                        disqualify_reasons[int(gt_candidate_index)]
                        if gt_candidate_index is not None
                        and int(gt_candidate_index) < len(disqualify_reasons)
                        else []
                    )
                    score_json_structures.append(
                        {
                            "structure_idx": int(structure_idx),
                            "gt_candidate_index": int(gt_candidate_index),
                            "gt_rank": pref.gt_rank,
                            "best_is_gt": pref.best_is_gt,
                            "best_candidate_index": pref.best_candidate_index,
                            "gt_disqualified": bool(pref.gt_disqualified),
                            "gt_disqualify_reasons": list(gt_reason),
                            "n_disqualified": int(pref.n_disqualified),
                            "n_candidates": int(pref.n_candidates),
                            "gt_aggregate_sinkhorn": (
                                None
                                if gt_w is None
                                else float(gt_w.aggregate_sinkhorn)
                            ),
                            "best_aggregate_sinkhorn": (
                                None
                                if best_w is None
                                else float(best_w.aggregate_sinkhorn)
                            ),
                            "per_candidate": [
                                {
                                    "candidate_index": int(r.candidate_index),
                                    "aggregate_sinkhorn": float(r.aggregate_sinkhorn),
                                    "disqualified": bool(disqualified_flags[i]),
                                    "disqualify_reasons": list(disqualify_reasons[i]),
                                    "stiffnesses": dict(r.stiffnesses),
                                    "n_transitions": dict(r.per_direction_n_transitions),
                                    "low_sample_directions": list(r.low_sample_directions),
                                    "err_pos_hold": (
                                        float(err_pos_hold[i])
                                        if i < len(err_pos_hold)
                                        else None
                                    ),
                                    "err_force_hold": (
                                        float(err_force_hold[i])
                                        if i < len(err_force_hold)
                                        else None
                                    ),
                                    "err_torque_hold": (
                                        float(err_torque_hold[i])
                                        if i < len(err_torque_hold)
                                        else None
                                    ),
                                }
                                for i, r in enumerate(wasserstein_results)
                            ],
                        }
                    )
                if bool(args.score_mse):
                    sinkhorn_values = [
                        float(r.aggregate_sinkhorn) for r in wasserstein_results
                    ]
                    for metric, mse_values in (
                        ("err_pos_hold", err_pos_hold),
                        ("err_force_hold", err_force_hold),
                        ("err_torque_hold", err_torque_hold),
                    ):
                        corr = sinkhorn_mse_spearman(
                            sinkhorn_values=sinkhorn_values,
                            mse_values=mse_values,
                            metric=metric,
                            disqualified=disqualified_flags,
                        )
                        print(
                            f"  spearman(sinkhorn,{metric})={corr.spearman:.3g}"
                        )

            if bool(args.score_mse) and bool(args.score_wasserstein):
                if gt_candidate_index is None:
                    print("WARNING: cannot compare GT ranks (GT candidate index not found).")
                else:
                    from apple_pick_gym.grid_viz_metrics import average_ranks

                    def _gt_rank(values: list[float], *, disqualified: list[bool]) -> float | None:
                        arr = np.asarray(values, dtype=np.float64).reshape(-1)
                        if arr.size != len(disqualified):
                            raise ValueError("rank inputs must have matching candidate counts")
                        masked = np.where(
                            np.asarray(disqualified, dtype=bool),
                            float("inf"),
                            arr,
                        )
                        ranks = average_ranks(masked)
                        rank = float(ranks[int(gt_candidate_index)])
                        return rank if np.isfinite(rank) else None

                    gt_rank_pos = _gt_rank(err_pos_hold, disqualified=disqualified_flags)
                    gt_rank_force = _gt_rank(err_force_hold, disqualified=disqualified_flags)
                    gt_rank_torque = _gt_rank(err_torque_hold, disqualified=disqualified_flags)
                    if gt_rank_pos is not None and gt_rank_force is not None:
                        gt_rank_combined = 0.5 * (float(gt_rank_pos) + float(gt_rank_force))
                    else:
                        gt_rank_combined = None

                    gt_rank_sinkhorn = None
                    if "pref" in locals():
                        gt_rank_sinkhorn = getattr(pref, "gt_rank", None)
                    print(
                        f"\n=== structure {int(structure_idx)}: GT rank summary (Sinkhorn vs MSE) ===\n"
                        f"  sinkhorn_gt_rank={gt_rank_sinkhorn} | "
                        f"mse_gt_rank_pos={gt_rank_pos} "
                        f"mse_gt_rank_force={gt_rank_force} "
                        f"mse_gt_rank_torque={gt_rank_torque} "
                        f"mse_gt_rank_pos_force_avg={gt_rank_combined}"
                    )

        if args.plot_output is not None:
            replay_eps_by_candidate = []
            for cand_idx in range(num_candidates):
                replay_eps_by_candidate.append(
                    direction_episodes_from_collectors(
                        collectors,
                        candidate_index=int(cand_idx),
                        num_directions=int(structure_num_directions),
                    )
                )
            rows = build_grid_viz_rows(
                structure_idx=int(structure_idx),
                candidates=list(candidates),
                gt_candidate=gt,
                recorded_eps=recorded_eps,
                replay_eps_by_candidate=replay_eps_by_candidate,
                hold_phase_value=1,
                pos_weights=(1.0, 1.0, 1.0),
                dist_keys=("primary", "spur", "stem"),
                hold_aggregation="median" if bool(args.use_median) else "none",
                hold_use_latter_half=False,
            )
            metrics = tuple(args.plot_metrics)
            rep = summarize_structure(
                structure_idx=int(structure_idx),
                rows=rows,
                metrics=metrics,  # type: ignore[arg-type]
            )
            print(f"\n=== structure {int(structure_idx)}: viz report ===")
            for s in rep.summaries:
                print(
                    f"  {s.metric}: best_is_gt={s.best_is_gt} gt_rank={s.gt_rank} "
                    f"spearman(dist,err)={s.spearman_dist_vs_err:.3g}"
                )
            final_rank = summarize_final_rank(rows)
            print(f"\n=== structure {int(structure_idx)}: final hold rank (pos+force) ===")
            print(
                f"  disqualified: {final_rank.n_disqualified}/{final_rank.n_candidates} "
                f"best_candidate={final_rank.best_candidate_index} "
                f"rank_combined={final_rank.best_rank_combined} "
                f"gt_rank={final_rank.gt_rank_combined} "
                f"best_is_gt={final_rank.best_is_gt}"
            )
            write_structure_bundle(
                output_dir=str(args.plot_output),
                structure_idx=int(structure_idx),
                rows=rows,
                metrics=metrics,  # type: ignore[arg-type]
                n_bins=10,
            )

    if args.score_json_output is not None:
        out = Path(str(args.score_json_output))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(args.dataset),
            "use_median": bool(args.use_median),
            "hold_id_onehot": bool(args.hold_id_onehot),
            "pool_directions": bool(args.pool_directions),
            "dir_id_onehot": bool(args.pool_directions),
            "n_directions": _resolve_n_directions(dataset, collection),
            "structures": score_json_structures,
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote score JSON: {out}")

    if (
        not bool(args.replay_only)
        and not bool(args.score_mse)
        and not bool(args.score_wasserstein)
        and args.plot_output is None
        and args.export_replay_dir is None
        and args.score_json_output is None
    ):
        print(
            "No scoring or export requested; re-run with --replay-only, --score-mse, "
            "--score-wasserstein, --plot-output, or --export-replay-dir."
        )


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    apply_deprecated_mse_cli_flags(args)
    try:
        _run(args, parser, viewer=viewer)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
