"""Replay a recorded sysID trajectory with ``ApplePickReplay-v0``.

Loads a Parquet dataset produced by ``example_gym_sysid.py --output <dir>`` and
replays stored EE velocity commands open-loop through the VIC dynamic arm.

Run from the repository root::

    uv sync --extra gym --extra vic --extra dev

Collect a dataset (if needed)::

    uv run python apple_pick_gym/examples/example_gym_sysid.py \\
      --viewer null --n-directions 1 --max-steps 200 \\
      --output /tmp/sysid_dataset

Replay headless (prints dataset vs live observation errors)::

    uv run python apple_pick_gym/examples/example_gym_replay.py \\
      --dataset /tmp/sysid_dataset --viewer null

List episodes in a dataset::

    uv run python apple_pick_gym/examples/example_gym_replay.py \\
      --dataset /tmp/sysid_dataset --list-episodes

Per-step error breakdown::

    uv run python apple_pick_gym/examples/example_gym_replay.py \\
      --dataset /tmp/sysid_dataset --debug --viewer null
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplayErrors:
    """Dataset vs live observation errors for one replay step."""

    frame_idx: int
    action_rmse: float
    ft_wrist_rmse: float
    ft_force_n: float
    ft_torque_nm: float
    tcp_pos_mm: float
    tcp_vel_rmse: float
    tcp_quat_rmse: float
    woody_start_mm: float
    woody_end_mm: float
    apple_pos_mm: float
    apple_quat_rmse: float
    robot_joint_q_rmse: float


@dataclass
class ReplayErrorSummary:
    n_steps: int = 0
    action_rmse: list[float] = field(default_factory=list)
    ft_wrist_rmse: list[float] = field(default_factory=list)
    ft_force_n: list[float] = field(default_factory=list)
    ft_torque_nm: list[float] = field(default_factory=list)
    tcp_pos_mm: list[float] = field(default_factory=list)
    tcp_vel_rmse: list[float] = field(default_factory=list)
    tcp_quat_rmse: list[float] = field(default_factory=list)
    woody_start_mm: list[float] = field(default_factory=list)
    woody_end_mm: list[float] = field(default_factory=list)
    apple_pos_mm: list[float] = field(default_factory=list)
    apple_quat_rmse: list[float] = field(default_factory=list)
    robot_joint_q_rmse: list[float] = field(default_factory=list)

    def record(self, err: ReplayErrors) -> None:
        self.n_steps += 1
        self.action_rmse.append(err.action_rmse)
        self.ft_wrist_rmse.append(err.ft_wrist_rmse)
        self.ft_force_n.append(err.ft_force_n)
        self.ft_torque_nm.append(err.ft_torque_nm)
        self.tcp_pos_mm.append(err.tcp_pos_mm)
        self.tcp_vel_rmse.append(err.tcp_vel_rmse)
        self.tcp_quat_rmse.append(err.tcp_quat_rmse)
        self.woody_start_mm.append(err.woody_start_mm)
        self.woody_end_mm.append(err.woody_end_mm)
        self.apple_pos_mm.append(err.apple_pos_mm)
        self.apple_quat_rmse.append(err.apple_quat_rmse)
        self.robot_joint_q_rmse.append(err.robot_joint_q_rmse)

    @staticmethod
    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    @staticmethod
    def _max(xs: list[float]) -> float:
        return float(np.max(xs)) if xs else 0.0

    def print_summary(self) -> None:
        print("\nDataset vs live observation errors (mean / max):")
        print(
            f"  action RMSE [m/s, rad/s]:     "
            f"{self._mean(self.action_rmse):.4f} / {self._max(self.action_rmse):.4f}"
        )
        print(
            f"  ft_wrist RMSE [N, N·m]:       "
            f"{self._mean(self.ft_wrist_rmse):.3f} / {self._max(self.ft_wrist_rmse):.3f}"
        )
        print(
            f"  |ΔF| [N]:                     "
            f"{self._mean(self.ft_force_n):.3f} / {self._max(self.ft_force_n):.3f}"
        )
        print(
            f"  |Δτ| [N·m]:                   "
            f"{self._mean(self.ft_torque_nm):.3f} / {self._max(self.ft_torque_nm):.3f}"
        )
        print(
            f"  |Δtcp_pos| [mm]:              "
            f"{self._mean(self.tcp_pos_mm):.2f} / {self._max(self.tcp_pos_mm):.2f}"
        )
        print(
            f"  tcp_vel RMSE [m/s, rad/s]:    "
            f"{self._mean(self.tcp_vel_rmse):.4f} / {self._max(self.tcp_vel_rmse):.4f}"
        )
        print(
            f"  woody_start |Δ| [mm]:         "
            f"{self._mean(self.woody_start_mm):.2f} / {self._max(self.woody_start_mm):.2f}"
        )
        print(
            f"  woody_end |Δ| [mm]:           "
            f"{self._mean(self.woody_end_mm):.2f} / {self._max(self.woody_end_mm):.2f}"
        )
        print(
            f"  |Δapple_pos| [mm]:            "
            f"{self._mean(self.apple_pos_mm):.2f} / {self._max(self.apple_pos_mm):.2f}"
        )
        print(
            f"  tcp_quat RMSE:                "
            f"{self._mean(self.tcp_quat_rmse):.4f} / {self._max(self.tcp_quat_rmse):.4f}"
        )
        print(
            f"  apple_quat RMSE:              "
            f"{self._mean(self.apple_quat_rmse):.4f} / {self._max(self.apple_quat_rmse):.4f}"
        )
        print(
            f"  robot_joint_q RMSE [rad]:     "
            f"{self._mean(self.robot_joint_q_rmse):.4f} / {self._max(self.robot_joint_q_rmse):.4f}"
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
        help="VBD settle substeps before welding (default 0; use 1800 to match collection).",
    )
    p.add_argument(
        "--no-robot-facing-weld",
        action="store_true",
        help="Disable robot-facing weld hemisphere (easier IK; mismatches collected episodes).",
    )
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Ignore initial_states/*.npz and initialize replay from observable Parquet data.",
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
    return p


def _print_episode_table(dataset) -> None:
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


def _fmt_force(force: np.ndarray) -> str:
    f = np.asarray(force, dtype=np.float64).reshape(3)
    return f"({f[0]:+.2f}, {f[1]:+.2f}, {f[2]:+.2f}) N"


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(d * d)))


def _norm_diff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(d))


def _recorded_rmse(
    obs: dict,
    recorded: dict[str, np.ndarray],
    key: str,
    frame_idx: int,
) -> float:
    rec = recorded.get(key)
    if key not in obs or rec is None or np.asarray(rec).size == 0:
        return 0.0
    return _rmse(np.asarray(obs[key], dtype=np.float32), np.asarray(rec[frame_idx], dtype=np.float32))


def _compare_to_dataset(
    *,
    frame_idx: int,
    obs: dict,
    info: dict,
    recorded: dict[str, np.ndarray],
) -> ReplayErrors | None:
    from apple_pick_sim.system_id.trajectory_store import stack_woody_pos_frame

    n_frames = int(recorded["action"].shape[0])
    if frame_idx < 0 or frame_idx >= n_frames:
        return None

    live_action = np.asarray(info.get("replay_action", np.zeros(6)), dtype=np.float32)
    rec_action = recorded["action"][frame_idx]
    live_ft = np.asarray(obs["ft_wrist"], dtype=np.float32)
    rec_ft = recorded["ft_wrist"][frame_idx]
    live_tcp_pos = np.asarray(obs["tcp_pos"], dtype=np.float32)
    rec_tcp_pos = recorded["tcp_pos"][frame_idx]
    live_tcp_vel = np.asarray(obs["tcp_velocity"], dtype=np.float32)
    rec_tcp_vel = recorded["tcp_velocity"][frame_idx]
    live_woody_start = np.asarray(obs["woody_start"], dtype=np.float32)
    junction_names = recorded.get("junction_names") or list(
        recorded["woody_part_start_pos"].keys()
    )
    rec_woody_start = stack_woody_pos_frame(
        recorded["woody_part_start_pos"], frame_idx, junction_names
    )
    live_woody_end = np.asarray(obs["woody_end"], dtype=np.float32)
    rec_woody_end_by_name = recorded.get("woody_part_end_pos")
    if isinstance(rec_woody_end_by_name, dict) and all(
        name in rec_woody_end_by_name for name in junction_names
    ):
        rec_woody_end = stack_woody_pos_frame(
            rec_woody_end_by_name, frame_idx, junction_names
        )
    else:
        # Bags no longer persist woody ends; skip the end-vs-live comparison.
        rec_woody_end = live_woody_end
    live_apple = np.asarray(obs["apple_pos"], dtype=np.float32)
    rec_apple = recorded["apple_pos"][frame_idx]

    return ReplayErrors(
        frame_idx=frame_idx,
        action_rmse=_rmse(live_action, rec_action),
        ft_wrist_rmse=_rmse(live_ft, rec_ft),
        ft_force_n=_norm_diff(live_ft[:3], rec_ft[:3]),
        ft_torque_nm=_norm_diff(live_ft[3:], rec_ft[3:]),
        tcp_pos_mm=1000.0 * _norm_diff(live_tcp_pos, rec_tcp_pos),
        tcp_vel_rmse=_rmse(live_tcp_vel, rec_tcp_vel),
        tcp_quat_rmse=_recorded_rmse(obs, recorded, "tcp_quat", frame_idx),
        woody_start_mm=1000.0 * _norm_diff(live_woody_start, rec_woody_start),
        woody_end_mm=1000.0 * _norm_diff(live_woody_end, rec_woody_end),
        apple_pos_mm=1000.0 * _norm_diff(live_apple, rec_apple),
        apple_quat_rmse=_recorded_rmse(obs, recorded, "apple_quat", frame_idx),
        robot_joint_q_rmse=_recorded_rmse(obs, recorded, "robot_joint_q", frame_idx),
    )


def _compare_reset_to_snapshot(obs: dict, snapshot: dict[str, np.ndarray]) -> ReplayErrors | None:
    if "obs_apple_pos" not in snapshot:
        return None
    return ReplayErrors(
        frame_idx=-1,
        action_rmse=0.0,
        ft_wrist_rmse=_rmse(obs["ft_wrist"], snapshot["obs_ft_wrist"]),
        ft_force_n=_norm_diff(obs["ft_wrist"][:3], snapshot["obs_ft_wrist"][:3]),
        ft_torque_nm=_norm_diff(obs["ft_wrist"][3:], snapshot["obs_ft_wrist"][3:]),
        tcp_pos_mm=1000.0 * _norm_diff(obs["tcp_pos"], snapshot["obs_tcp_pos"]),
        tcp_vel_rmse=_rmse(obs["tcp_velocity"], snapshot["obs_tcp_velocity"]),
        tcp_quat_rmse=0.0,
        woody_start_mm=1000.0 * _norm_diff(obs["woody_start"], snapshot["obs_woody_start"]),
        woody_end_mm=1000.0 * _norm_diff(obs["woody_end"], snapshot["obs_woody_end"]),
        apple_pos_mm=1000.0 * _norm_diff(obs["apple_pos"], snapshot["obs_apple_pos"]),
        apple_quat_rmse=0.0,
        robot_joint_q_rmse=0.0,
    )


def _print_vic_banner(env) -> None:
    from apple_pick_sim.robot import fr3_robot

    scene = env._scene
    controller = env._controller
    gains = getattr(scene, "vic_gains", None)
    joint_torques = bool(getattr(scene, "vic_use_joint_torques", False))
    print("Replay actuation: VIC dynamic arm")
    print(f"  controller: {type(controller).__name__}")
    print(f"  kinematic_mode: {getattr(scene, 'robot_kinematic_mode', None)}")
    print(f"  vic_use_joint_torques: {joint_torques}")
    if isinstance(controller, fr3_robot.Fr3EEImpedanceController) and gains is not None:
        print(
            f"  vic gains: K_lin={gains.linear_k:.0f} N/m  D_lin={gains.linear_d:.0f}  "
            f"K_ang={gains.angular_k:.0f}  D_ang={gains.angular_d:.0f}"
        )
    print("  action path: stored EE velocity -> update_fr3_ee_teleop -> VIC -> coupled_substep")


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl).")

    parser = _make_parser()
    import newton.examples

    viewer, args = newton.examples.init(parser=parser)

    from apple_pick_gym.envs import ApplePickReplayEnv
    from apple_pick_sim.system_id import TrajectoryDataset

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

    print(f"Replaying episode {episode_id}")
    print(f"  excitation_type: {meta.get('excitation_type')}")
    print(f"  frames: {n_frames}")
    print(f"  control_hz: {meta.get('control_hz')}")
    print(f"  n_woody_parts: {meta.get('n_woody_parts')}")
    print(f"  reset seed: {replay_seed}")
    if meta.get("params_fingerprint"):
        try:
            fp = json.loads(meta["params_fingerprint"])
            stem_k = fp.get("stem_bend_stiffness")
            if stem_k is not None:
                print(f"  stem_bend_stiffness: {stem_k}")
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
        options={"use_snapshot": not bool(args.no_snapshot)},
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

    saved_snapshot = dataset.load_initial_state(episode_id)
    reset_err = _compare_reset_to_snapshot(obs, saved_snapshot or {})
    if reset_err is not None:
        print(
            "  reset vs saved snapshot: "
            f"|ΔF|={reset_err.ft_force_n:.2f} N  "
            f"|Δtcp|={reset_err.tcp_pos_mm:.1f} mm  "
            f"|Δapple|={reset_err.apple_pos_mm:.1f} mm"
        )
    else:
        reset_err = _compare_to_dataset(
            frame_idx=0,
            obs=obs,
            info={"replay_action": np.zeros(6, dtype=np.float32)},
            recorded=recorded,
        )
        if reset_err is not None:
            print(
                "  reset vs dataset frame 0 (legacy, no snapshot obs): "
                f"|ΔF|={reset_err.ft_force_n:.2f} N  "
                f"|Δtcp|={reset_err.tcp_pos_mm:.1f} mm  "
                f"|Δapple|={reset_err.apple_pos_mm:.1f} mm"
            )

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
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, info = env.step(action)
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
                        f"           err  |ΔF|={err.ft_force_n:.2f} N  "
                        f"|Δτ|={err.ft_torque_nm:.3f} N·m  "
                        f"|Δtcp|={err.tcp_pos_mm:.1f} mm  "
                        f"action_rmse={err.action_rmse:.4f}"
                    )
            elif step_idx % max(1, n_frames // 10) == 0 or truncated:
                err_str = ""
                if err is not None:
                    err_str = (
                        f"  |ΔF|={err.ft_force_n:.1f} N  |Δtcp|={err.tcp_pos_mm:.1f} mm"
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
            if error_summary._mean(error_summary.action_rmse) > 1e-5:
                print(
                    "  note: action RMSE should be ~0; check dataset/env action layout if large."
                )
            if error_summary._mean(error_summary.ft_force_n) > 1.0:
                print(
                    "  note: large force/obs errors usually mean replay settings differ from "
                    "collection (seed, robot_facing_weld, warmup substeps, missing initial_states "
                    "snapshot) or params were changed."
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
