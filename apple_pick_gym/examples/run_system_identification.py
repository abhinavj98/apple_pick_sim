"""Diagnostic grid search over fruiting-system bend stiffnesses.

This script loads a sys-ID Parquet dataset, rebuilds replay episodes through
``ApplePickReplayEnv``, and replays recorded EE velocity actions for each
candidate set of rod bend stiffnesses.

Run from the repository root::

    uv run python apple_pick_gym/examples/run_system_identification.py \\
      --dataset /tmp/sysid_dataset --viewer null \\
      --primary-bend-stiffness-values 10,25,50 \\
      --secondary-bend-stiffness-values 10,25,50 \\
      --spur-bend-stiffness-values 10,25,50 \\
      --stem-bend-stiffness-values 10,25,50

Snapshot replay is opt-in via ``--use-snapshot`` and is intended only for
debugging against privileged sim-to-sim baselines.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from itertools import product
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm


def print(*args, **kwargs):
    """Route prints through ``tqdm.write`` so grid-search progress bars stay intact."""
    import builtins

    try:
        tqdm.write(" ".join(str(arg) for arg in args), file=kwargs.get("file", sys.stdout))
    except Exception:
        builtins.print(*args, **kwargs)

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")


class BendStiffnessCandidate(NamedTuple):
    """One grid point for segment bend stiffnesses."""

    primary: float
    secondary: float
    spur: float
    stem: float

    def to_overrides(self) -> dict[str, dict[str, float]]:
        return {
            "primary": {"bend_stiffness": float(self.primary)},
            "secondary": {"bend_stiffness": float(self.secondary)},
            "spur": {"bend_stiffness": float(self.spur)},
            "stem": {"bend_stiffness": float(self.stem)},
        }


class MmdDirectionContext(NamedTuple):
    """GT normalization and bandwidth for one direction."""

    gt_norm: np.ndarray
    stats: Any
    bandwidth: float


def parse_positive_float_grid(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of positive floats for one grid axis."""
    if value.strip() == "":
        raise argparse.ArgumentTypeError("expected at least one positive float")
    parts = value.split(",")
    if any(part.strip() == "" for part in parts):
        raise argparse.ArgumentTypeError("empty grid entries are not allowed")
    try:
        values = tuple(float(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid values must be floats") from exc
    if any((not math.isfinite(v)) or v <= 0.0 for v in values):
        raise argparse.ArgumentTypeError("grid values must be finite and positive")
    return values


def iter_bend_stiffness_candidates(
    *,
    primary_values: tuple[float, ...],
    secondary_values: tuple[float, ...],
    spur_values: tuple[float, ...],
    stem_values: tuple[float, ...],
):
    """Yield bend-stiffness grid candidates in Cartesian product order."""
    for primary, secondary, spur, stem in product(
        primary_values,
        secondary_values,
        spur_values,
        stem_values,
    ):
        yield BendStiffnessCandidate(
            primary=float(primary),
            secondary=float(secondary),
            spur=float(spur),
            stem=float(stem),
        )


def _make_parser() -> argparse.ArgumentParser:
    import newton.examples

    p = newton.examples.create_parser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Directory containing metadata.parquet and frames/",
    )
    p.add_argument(
        "--episode-id",
        action="append",
        default=None,
        help=(
            "Episode UUID to evaluate. Repeat to evaluate multiple episodes. "
            "Default: all episodes in metadata."
        ),
    )
    p.add_argument(
        "--list-episodes",
        action="store_true",
        help="Print episode ids and metadata, then exit.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Simulation reset seed (default: use seed stored in episode metadata).",
    )
    p.add_argument(
        "--fix-to-apple-warmup-substeps",
        type=int,
        default=1800,
        help="VBD settle substeps before welding.",
    )
    p.add_argument(
        "--no-robot-facing-weld",
        action="store_true",
        help="Disable robot-facing weld hemisphere (mismatches most collected episodes).",
    )
    p.add_argument(
        "--use-snapshot",
        action="store_true",
        help=(
            "Restore initial_states/*.npz privileged simulator state. "
            "Default is observable Parquet initialization."
        ),
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Cap evaluated stiffness candidates (0 = full grid).",
    )
    p.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Viewer refresh rate when --viewer gl.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step replay diagnostics and observation errors.",
    )
    p.add_argument(
        "--mmd-output",
        type=str,
        default=None,
        help="Directory for MMD grid CSV and ranked-loss PNG outputs.",
    )
    for segment in ROD_SEGMENTS:
        p.add_argument(
            f"--{segment}-bend-stiffness-values",
            dest=f"{segment}_bend_stiffness_values",
            type=parse_positive_float_grid,
            default=None,
            help=f"Comma-separated candidate values for {segment}.bend_stiffness.",
        )
    return p


def _candidate_label(candidate: BendStiffnessCandidate) -> str:
    return (
        f"primary={candidate.primary:g} "
        f"secondary={candidate.secondary:g} "
        f"spur={candidate.spur:g} "
        f"stem={candidate.stem:g}"
    )


def _candidate_stiffnesses(candidate: BendStiffnessCandidate) -> dict[str, float]:
    return {
        "primary": float(candidate.primary),
        "secondary": float(candidate.secondary),
        "spur": float(candidate.spur),
        "stem": float(candidate.stem),
    }


def _combine_transition_features(episodes: list[dict]) -> dict[tuple[float, float, float], np.ndarray]:
    from apple_pick_sim.system_id.mmd_features import build_transition_features_by_direction

    parts: dict[tuple[float, float, float], list[np.ndarray]] = {}
    for arrays in episodes:
        for direction, features in build_transition_features_by_direction(arrays).items():
            parts.setdefault(direction, []).append(features)
    return {
        direction: np.concatenate(chunks, axis=0)
        for direction, chunks in sorted(parts.items())
        if chunks
    }


def _prepare_gt_mmd_context(recorded_episodes: list[dict]) -> dict[tuple[float, float, float], MmdDirectionContext]:
    from apple_pick_sim.system_id.mmd import (
        apply_normalization,
        fit_gt_normalization,
        rbf_bandwidth_median,
    )

    gt_by_direction = _combine_transition_features(recorded_episodes)
    if not gt_by_direction:
        raise ValueError("No valid hold-only GT transition features were found.")

    context: dict[tuple[float, float, float], MmdDirectionContext] = {}
    for direction, gt_features in gt_by_direction.items():
        stats = fit_gt_normalization(gt_features)
        gt_norm = apply_normalization(gt_features, stats)
        bandwidth = rbf_bandwidth_median(gt_norm)
        context[direction] = MmdDirectionContext(
            gt_norm=gt_norm,
            stats=stats,
            bandwidth=bandwidth,
        )
    return context


def _compute_candidate_mmd_result(
    *,
    candidate_index: int,
    candidate: BendStiffnessCandidate,
    gt_context: dict[tuple[float, float, float], MmdDirectionContext],
    replay_observations: list[dict],
):
    from apple_pick_sim.system_id.mmd import apply_normalization, biased_mmd2
    from apple_pick_sim.system_id.mmd_results import MmdCandidateResult

    candidate_by_direction = _combine_transition_features(replay_observations)
    per_direction: dict[tuple[float, float, float], float] = {}
    for direction, context in gt_context.items():
        candidate_features = candidate_by_direction.get(direction)
        if candidate_features is None:
            print(f"  MMD direction {direction}: skipped (no valid candidate hold transitions)")
            continue
        if candidate_features.shape[1] != context.gt_norm.shape[1]:
            raise ValueError(
                "MMD feature dimension mismatch for direction "
                f"{direction}: gt={context.gt_norm.shape[1]} candidate={candidate_features.shape[1]}"
            )
        candidate_norm = apply_normalization(candidate_features, context.stats)
        per_direction[direction] = biased_mmd2(
            context.gt_norm,
            candidate_norm,
            context.bandwidth,
        )
    if not per_direction:
        raise ValueError("No candidate directions had valid hold-only MMD transitions.")
    aggregate = float(np.mean(list(per_direction.values())))
    return MmdCandidateResult(
        candidate_index=int(candidate_index),
        stiffnesses=_candidate_stiffnesses(candidate),
        aggregate_mmd2=aggregate,
        per_direction_mmd2=per_direction,
    )


def _print_mmd_ranking(results: list) -> None:
    from apple_pick_sim.system_id.mmd_results import rank_results

    print("\nMMD candidate ranking (lower is better):")
    for rank, result in enumerate(rank_results(results), start=1):
        direction_bits = " ".join(
            f"d=({direction[0]:+.3f},{direction[1]:+.3f},{direction[2]:+.3f})={loss:.6g}"
            for direction, loss in sorted(result.per_direction_mmd2.items())
        )
        print(
            f"  #{rank}: candidate {result.candidate_index} "
            f"aggregate={result.aggregate_mmd2:.6g} {direction_bits}"
        )


def _write_mmd_outputs(results: list, output_dir: str) -> None:
    from apple_pick_sim.system_id.mmd_results import (
        write_diagnostic_plots,
        write_results_csv,
    )

    csv_path = write_results_csv(results, output_dir)
    plot_paths = write_diagnostic_plots(results, output_dir)
    print(f"MMD results CSV: {csv_path}")
    for path in plot_paths:
        print(f"MMD diagnostic plot: {path}")


def _episode_ids_to_evaluate(dataset, requested: list[str] | None) -> list[str]:
    episode_ids = dataset.episode_ids()
    if not episode_ids:
        raise SystemExit(f"No episodes found in dataset: {dataset.dataset_dir}")
    if requested is None:
        return episode_ids
    missing = [episode_id for episode_id in requested if episode_id not in episode_ids]
    if missing:
        raise SystemExit(f"Episode id(s) not found: {', '.join(missing)}")
    return list(requested)


def _require_grid_values(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
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


def _record_episode_errors(
    *,
    env,
    recorded: dict,
    error_summary,
    observation_collector=None,
    n_frames: int,
    debug: bool,
    render_dt: float,
    viewer,
) -> int:
    from apple_pick_gym.examples.example_gym_replay import _compare_to_dataset, _fmt_force

    sim_time = 0.0
    step_dt = 1.0 / float(env._cfg.control_hz)
    steps = 0

    frame_range = range(n_frames)
    if (getattr(viewer, "name", None) == "null" or type(viewer).__name__ == "ViewerNull") and not debug:
        frame_range = tqdm(frame_range, desc="    Simulating frames", leave=False)

    for step_idx in frame_range:
        obs, _reward, terminated, truncated, info = env.step(env.action_space.sample())
        sim_time += step_dt
        steps += 1

        frame_idx = int(info.get("replay_frame_idx", step_idx))
        err = _compare_to_dataset(
            frame_idx=frame_idx,
            obs=obs,
            info=info,
            recorded=recorded,
        )
        if err is not None:
            error_summary.record(err)
        if observation_collector is not None:
            observation_collector.record(obs, frame_idx=frame_idx)

        if debug:
            ft = np.asarray(obs["ft_wrist"], dtype=np.float64)[:3]
            replay_action = np.asarray(info.get("replay_action", np.zeros(6)), dtype=np.float64)
            print(
                f"    step {step_idx + 1:4d}/{n_frames} frame={frame_idx} "
                f"|F|={np.linalg.norm(ft):6.2f} N F={_fmt_force(ft)} "
                f"cmd=({replay_action[0]:+.3f},{replay_action[1]:+.3f},"
                f"{replay_action[2]:+.3f}) m/s"
            )
            if err is not None:
                print(
                    f"      err |dF|={err.ft_force_n:.2f} N "
                    f"|dtau|={err.ft_torque_nm:.3f} N*m "
                    f"|dtcp|={err.tcp_pos_mm:.1f} mm"
                )

        scene = env._scene
        if scene is not None and viewer.is_running():
            if scene.last_vbd_contacts is not None:
                viz_contacts = scene.last_vbd_contacts
            else:
                viz_contacts = scene.cable.model.collide(
                    scene.cable.state_0,
                    collision_pipeline=scene.cable_collision_pipeline,
                )
            viewer.begin_frame(sim_time)
            viewer.log_state(scene.cable.state_0)
            viewer.log_contacts(viz_contacts, scene.cable.state_0)
            viewer.end_frame()

        if getattr(viewer, "name", None) != "null" and viewer.is_running():
            time.sleep(max(0.0, render_dt - step_dt))

        if terminated or truncated:
            break
    return steps


def _print_candidate_summary(candidate: BendStiffnessCandidate, summary) -> None:
    mean = summary._mean
    max_ = summary._max
    woody_start_mean = mean(summary.woody_start_mm)
    woody_end_mean = mean(summary.woody_end_mm)
    woody_start_max = max_(summary.woody_start_mm)
    woody_end_max = max_(summary.woody_end_mm)
    print(f"candidate {_candidate_label(candidate)}")
    print(f"  frames: {summary.n_steps}")
    print(f"  |dF| [N] mean/max:          {mean(summary.ft_force_n):.3f} / {max_(summary.ft_force_n):.3f}")
    print(f"  |dtau| [N*m] mean/max:      {mean(summary.ft_torque_nm):.3f} / {max_(summary.ft_torque_nm):.3f}")
    print(f"  tcp pos [mm] mean/max:      {mean(summary.tcp_pos_mm):.2f} / {max_(summary.tcp_pos_mm):.2f}")
    print(f"  tcp vel RMSE mean/max:      {mean(summary.tcp_vel_rmse):.4f} / {max_(summary.tcp_vel_rmse):.4f}")
    print(f"  apple pos [mm] mean/max:    {mean(summary.apple_pos_mm):.2f} / {max_(summary.apple_pos_mm):.2f}")
    print(
        f"  woody start [mm] mean/max:  "
        f"{woody_start_mean:.2f} / {woody_start_max:.2f}"
    )
    print(
        f"  woody end [mm] mean/max:    "
        f"{woody_end_mean:.2f} / {woody_end_max:.2f}"
    )


def _evaluate_candidate(
    *,
    args: argparse.Namespace,
    dataset,
    episode_ids: list[str],
    candidate: BendStiffnessCandidate,
    viewer,
):
    from apple_pick_gym.envs import ApplePickReplayEnv
    from apple_pick_gym.examples.example_gym_replay import ReplayErrorSummary
    from apple_pick_gym.examples.example_gym_replay_overrides import (
        apply_param_overrides,
        load_base_params_from_dataset,
    )
    from apple_pick_sim.system_id.mmd_features import ReplayObservationCollector

    summary = ReplayErrorSummary()
    replay_observations: list[dict] = []
    render_dt = 1.0 / float(args.hz)
    for episode_id in episode_ids:
        meta = dataset.load_episode_meta(episode_id)
        recorded = dataset.load_episode_obs_arrays(episode_id)
        n_frames = int(recorded["action"].shape[0])
        replay_seed = int(args.seed) if args.seed is not None else meta.get("seed")
        if replay_seed is None:
            replay_seed = 3

        base_params = load_base_params_from_dataset(dataset, episode_id)
        candidate_params = apply_param_overrides(base_params, candidate.to_overrides())

        env = ApplePickReplayEnv(
            render_mode=None,
            max_episode_steps=max(n_frames, 1),
            fix_to_apple=True,
            fix_to_apple_warmup_substeps=int(args.fix_to_apple_warmup_substeps),
            robot_facing_weld=not bool(args.no_robot_facing_weld),
            mujoco_solver_kwargs={"disable_contacts": True},
            device=args.device,
        )
        try:
            env.load_dataset(args.dataset, episode_id=episode_id)
            obs, info = env.reset(
                seed=int(replay_seed),
                options={
                    "use_snapshot": bool(args.use_snapshot),
                    "params": candidate_params,
                },
            )
            scene = env._scene
            if scene is None:
                raise RuntimeError("Env did not create a scene; did reset() succeed?")
            viewer.set_model(scene.cable.model)
            if hasattr(viewer, "hide_loading_splash"):
                viewer.hide_loading_splash()
            print(
                f"  episode {episode_id} frames={n_frames} "
                f"init={'snapshot' if info.get('initial_state_restored') else 'observation'}"
            )
            observation_collector = (
                ReplayObservationCollector(recorded)
                if getattr(args, "mmd_output", None)
                else None
            )
            _record_episode_errors(
                env=env,
                recorded=recorded,
                error_summary=summary,
                observation_collector=observation_collector,
                n_frames=n_frames,
                debug=bool(args.debug),
                render_dt=render_dt,
                viewer=viewer,
            )
            if observation_collector is not None and observation_collector.n_rows > 0:
                replay_observations.append(observation_collector.to_arrays())
        finally:
            env.close()
    return summary, replay_observations


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl).")

    parser = _make_parser()
    import newton.examples

    viewer, args = newton.examples.init(parser=parser)

    from apple_pick_gym.examples.example_gym_replay_overrides import (
        _print_episode_table,
        _warn_if_snapshot_mode,
    )
    from apple_pick_sim.system_id import TrajectoryDataset

    _warn_if_snapshot_mode(bool(args.use_snapshot))

    dataset = TrajectoryDataset(args.dataset)
    if args.list_episodes:
        _print_episode_table(dataset)
        return

    _require_grid_values(parser, args)
    episode_ids = _episode_ids_to_evaluate(dataset, args.episode_id)
    candidates = list(
        iter_bend_stiffness_candidates(
            primary_values=args.primary_bend_stiffness_values,
            secondary_values=args.secondary_bend_stiffness_values,
            spur_values=args.spur_bend_stiffness_values,
            stem_values=args.stem_bend_stiffness_values,
        )
    )
    max_candidates = int(args.max_candidates)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    print(f"Dataset: {dataset.dataset_dir}")
    print(f"Episodes: {len(episode_ids)}")
    print(f"Initialization: {'snapshot' if args.use_snapshot else 'observable parquet'}")

    gt_mmd_context = None
    mmd_results = []
    if args.mmd_output:
        print("MMD: preparing hold-only GT transition features")
        gt_mmd_context = _prepare_gt_mmd_context(
            [dataset.load_episode_obs_arrays(episode_id) for episode_id in episode_ids]
        )
        print(f"MMD: prepared {len(gt_mmd_context)} direction(s)")

    evaluated = 0
    for candidate in tqdm(candidates, desc="Grid Search Progress"):
        evaluated += 1
        print(f"\n[{evaluated}/{len(candidates)}] evaluating {_candidate_label(candidate)}")
        summary, replay_observations = _evaluate_candidate(
            args=args,
            dataset=dataset,
            episode_ids=episode_ids,
            candidate=candidate,
            viewer=viewer,
        )
        _print_candidate_summary(candidate, summary)
        if gt_mmd_context is not None:
            result = _compute_candidate_mmd_result(
                candidate_index=evaluated,
                candidate=candidate,
                gt_context=gt_mmd_context,
                replay_observations=replay_observations,
            )
            mmd_results.append(result)
            print(f"  aggregate biased MMD^2: {result.aggregate_mmd2:.6g}")

    if evaluated == 0:
        raise SystemExit("No stiffness candidates were generated.")
    if mmd_results:
        _print_mmd_ranking(mmd_results)
        _write_mmd_outputs(mmd_results, args.mmd_output)


if __name__ == "__main__":
    main()
