"""Replay a sysID dataset with explicit fruiting-parameter overrides.

This is the search-friendly replay entry point: it loads a recorded dataset,
rebuilds from observable Parquet metadata by default, applies CLI stiffness /
damping overrides to the candidate ``FruitingSystemParams``, and replays stored
EE velocity actions open-loop.

Run from the repository root::

    uv run python apple_pick_gym/examples/example_gym_replay_overrides.py \\
      --dataset /tmp/sysid_dataset --viewer null \\
      --stem-bend-stiffness 45.0 --stem-bend-damping 2.5

Snapshot replay is deliberately opt-in via ``--use-snapshot`` and emits a
warning because it restores privileged simulator state unavailable in real data.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")
OVERRIDE_FIELDS: tuple[str, ...] = (
    "bend_stiffness",
    "bend_damping",
    "stretch_stiffness",
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _flag_name(segment: str, field: str) -> str:
    return f"--{segment}-{field.replace('_', '-')}"


def _dest_name(segment: str, field: str) -> str:
    return f"{segment}_{field}"


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
        type=str,
        default=None,
        help="Episode UUID to replay (default: first episode in metadata).",
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
    for segment in ROD_SEGMENTS:
        for field in OVERRIDE_FIELDS:
            p.add_argument(
                _flag_name(segment, field),
                dest=_dest_name(segment, field),
                type=_positive_float,
                default=None,
                help=f"Override {segment}.{field} in the candidate replay model.",
            )
    return p


def _warn_if_snapshot_mode(use_snapshot: bool) -> None:
    if use_snapshot:
        warnings.warn(
            "--use-snapshot restores privileged simulator state; observable-data "
            "sys-ID workflows should leave it off.",
            RuntimeWarning,
            stacklevel=2,
        )


def _print_episode_table(dataset: Any) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset.dataset_dir / "metadata.parquet")
    print(f"Dataset: {dataset.dataset_dir}")
    print(f"Episodes: {table.num_rows}")
    for row_idx, episode_id in enumerate(table.column("episode_id").to_pylist()):
        excitation = table.column("excitation_type")[row_idx].as_py()
        n_frames_path = dataset.dataset_dir / "frames" / f"{episode_id}.parquet"
        n_frames = pq.read_table(n_frames_path).num_rows if n_frames_path.exists() else 0
        print(
            f"  {episode_id}  type={excitation!r}  frames={n_frames}  "
            f"seed={table.column('seed')[row_idx].as_py()}"
        )


def collect_param_overrides(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Return nested segment->field override values from parsed CLI args."""
    overrides: dict[str, dict[str, float]] = {}
    for segment in ROD_SEGMENTS:
        for field in OVERRIDE_FIELDS:
            value = getattr(args, _dest_name(segment, field), None)
            if value is not None:
                overrides.setdefault(segment, {})[field] = float(value)
    return overrides


def apply_param_overrides(params: Any, overrides: dict[str, dict[str, float]]) -> Any:
    """Return a copied ``FruitingSystemParams`` with selected rod scalar overrides."""
    from apple_pick_sim.fruiting_system.params import copy_fruiting_params

    out = copy_fruiting_params(params)
    for segment, fields in overrides.items():
        if segment not in ROD_SEGMENTS:
            raise ValueError(f"unknown rod segment {segment!r}")
        rod = getattr(out, segment)
        if rod is None:
            raise ValueError(f"Segment {segment!r} is disabled in params")
        unknown = set(fields) - set(OVERRIDE_FIELDS)
        if unknown:
            raise ValueError(f"unknown override fields for {segment!r}: {sorted(unknown)}")
        for field, value in fields.items():
            if float(value) <= 0.0:
                raise ValueError(f"{segment}.{field} must be positive")
        setattr(out, segment, dataclasses.replace(rod, **fields))
    return out


def load_base_params_from_dataset(dataset: Any, episode_id: str) -> Any:
    """Load candidate base params from observable metadata, preferring exact JSON if present."""
    from apple_pick_sim.fruiting_system import fruiting_params_from_json
    from apple_pick_sim.system_id.parquet_init import observation_reset_options_from_parquet

    meta = dataset.load_episode_meta(episode_id)
    serialized = meta.get("fruiting_system_params")
    if serialized is not None:
        return fruiting_params_from_json(str(serialized))

    options = observation_reset_options_from_parquet(dataset, episode_id)
    params = options.get("params")
    if params is None:
        raise ValueError(
            "dataset does not contain fruiting_system_params and observable metadata "
            "was insufficient to infer params"
        )
    return params


def _print_overrides(overrides: dict[str, dict[str, float]]) -> None:
    if not overrides:
        print("  param overrides: none")
        return
    print("  param overrides:")
    for segment in ROD_SEGMENTS:
        fields = overrides.get(segment)
        if not fields:
            continue
        for field in OVERRIDE_FIELDS:
            if field in fields:
                print(f"    {segment}.{field} = {fields[field]}")


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl).")

    parser = _make_parser()
    import newton.examples

    viewer, args = newton.examples.init(parser=parser)

    from apple_pick_gym.envs import ApplePickReplayEnv
    from apple_pick_gym.examples.example_gym_replay import (
        ReplayErrorSummary,
        _compare_to_dataset,
        _fmt_force,
        _print_vic_banner,
    )
    from apple_pick_sim.system_id import TrajectoryDataset

    _warn_if_snapshot_mode(bool(args.use_snapshot))

    dataset = TrajectoryDataset(args.dataset)
    if args.list_episodes:
        _print_episode_table(dataset)
        return

    episode_ids = dataset.episode_ids()
    if not episode_ids:
        raise SystemExit(f"No episodes found in dataset: {args.dataset}")

    episode_id = args.episode_id or episode_ids[0]
    meta = dataset.load_episode_meta(episode_id)
    recorded = dataset.load_episode_obs_arrays(episode_id)
    n_frames = int(recorded["action"].shape[0])
    replay_seed = int(args.seed) if args.seed is not None else meta.get("seed")
    if replay_seed is None:
        replay_seed = 3

    overrides = collect_param_overrides(args)
    base_params = load_base_params_from_dataset(dataset, episode_id)
    candidate_params = apply_param_overrides(base_params, overrides)

    print(f"Replaying episode {episode_id}")
    print(f"  excitation_type: {meta.get('excitation_type')}")
    print(f"  frames: {n_frames}")
    print(f"  control_hz: {meta.get('control_hz')}")
    print(f"  n_woody_parts: {meta.get('n_woody_parts')}")
    print(f"  reset seed: {replay_seed}")
    print(f"  initialization: {'snapshot' if args.use_snapshot else 'observable parquet'}")
    _print_overrides(overrides)
    if meta.get("params_fingerprint"):
        try:
            fp = json.loads(meta["params_fingerprint"])
            stem_k = fp.get("stem_bend_stiffness")
            if stem_k is not None:
                print(f"  recorded stem_bend_stiffness: {stem_k}")
        except json.JSONDecodeError:
            pass

    env = ApplePickReplayEnv(
        render_mode=None,
        max_episode_steps=max(n_frames, 1),
        fix_to_apple=True,
        fix_to_apple_warmup_substeps=int(args.fix_to_apple_warmup_substeps),
        robot_facing_weld=not bool(args.no_robot_facing_weld),
        mujoco_solver_kwargs={"disable_contacts": True},
    )
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

    if info.get("initial_state_restored"):
        print("  initial_state_restored: yes (loaded from initial_states/*.npz)")
    elif info.get("observation_init"):
        print("  initial_state_restored: no (observation-only parquet init)")
    else:
        print("  initial_state_restored: no (warmup-only reset)")

    _print_vic_banner(env)

    control_hz = float(meta.get("control_hz") or env._cfg.control_hz)
    sim_time = 0.0
    render_dt = 1.0 / float(args.hz)
    step_dt = 1.0 / control_hz

    viewer.set_model(scene.cable.model)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    error_summary = ReplayErrorSummary()
    wall_start = time.perf_counter()

    try:
        step_idx = 0
        while viewer.is_running():
            obs, _reward, terminated, truncated, info = env.step(env.action_space.sample())
            sim_time += step_dt
            step_idx += 1

            frame_idx = int(info.get("replay_frame_idx", step_idx - 1))
            err = _compare_to_dataset(
                frame_idx=frame_idx,
                obs=obs,
                info=info,
                recorded=recorded,
            )
            if err is not None:
                error_summary.record(err)

            scene = env._scene
            if scene is None:
                break

            ft = np.asarray(obs["ft_wrist"], dtype=np.float64)[:3]
            if args.debug:
                replay_action = np.asarray(info.get("replay_action", np.zeros(6)), dtype=np.float64)
                tcp_pos = np.asarray(obs["tcp_pos"], dtype=np.float64)
                cmd = replay_action[:3]
                print(
                    f"  step {step_idx:4d} frame={frame_idx}  |F|={np.linalg.norm(ft):6.2f} N  "
                    f"F={_fmt_force(ft)}  "
                    f"cmd=({cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f}) m/s  "
                    f"tcp_cm=({tcp_pos[0]*100:+.1f},{tcp_pos[1]*100:+.1f},{tcp_pos[2]*100:+.1f})"
                )
                if err is not None:
                    print(
                        f"           err  |dF|={err.ft_force_n:.2f} N  "
                        f"|dtau|={err.ft_torque_nm:.3f} N*m  "
                        f"|dtcp|={err.tcp_pos_mm:.1f} mm  "
                        f"action_rmse={err.action_rmse:.4f}"
                    )
            elif step_idx % max(1, n_frames // 10) == 0 or truncated:
                err_str = ""
                if err is not None:
                    err_str = (
                        f"  |dF|={err.ft_force_n:.1f} N  |dtcp|={err.tcp_pos_mm:.1f} mm"
                    )
                print(
                    f"  step {step_idx}/{n_frames}  |F|={np.linalg.norm(ft):.2f} N  "
                    f"sim={sim_time:.2f}s{err_str}"
                )

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

            if getattr(args, "viewer", None) != "null":
                time.sleep(max(0.0, render_dt - step_dt))

            if terminated or truncated:
                break

        print(
            f"\nReplay complete: {step_idx} steps in {time.perf_counter() - wall_start:.1f}s wall"
        )
        if error_summary.n_steps:
            error_summary.print_summary()
    finally:
        env.close()


if __name__ == "__main__":
    main()
