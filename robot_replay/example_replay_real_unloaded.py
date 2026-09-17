#!/usr/bin/env python3
"""Replay real ``vic_pose`` trajectories unloaded (arm only) and record TCP wrench.

Builds a ``mujoco_only`` FR3 scene (no VBD plant step, ``fix_to_apple=False`` so
no apple payload on TCP) and drives converted 19D ``vic_pose`` actions. Each
control frame records the world-frame **env-on-robot** wrench about TCP from EE
``body_parent_f`` (MuJoCo constraint / force-torque analog). Per-axis Kp/Kd from
the action are averaged into isotropic linear/angular gains (single-env VIC).

This is **not** gym ``ft_wrist`` (plant harvest). Do not subtract this series from
CMA candidate bags — see H3 ``docs/handbook-sysid-scoring.md``.

Example::

    uv run python robot_replay/example_replay_real_unloaded.py \\
      --dataset /path/to/converted_batched --out tmp/unloaded_tcp_wrench \\
      --direction-idx 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import warp as wp

from apple_pick_gym.batched_envs.real_batched_replay_build import (
    bootstrap_joint_q_from_episode_metadata,
    check_action_semantics,
    control_hz_from_episode_metadata,
    dataset_declares_vic_pose,
    fruiting_base_pos_from_episode_metadata,
)
from apple_pick_sim.coupled_fruiting.builders import build_coupled_fruiting_fr3
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    init_robot_mujoco_step_buffers,
)
from apple_pick_sim.coupled_fruiting.settle_then_weld import apply_open_loop_fr3_joint_q
from apple_pick_sim.fruiting_system.params import GripperProxyConfig, load_ranges, parse_sim_build
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.tcp_parent_wrench import tcp_world_wrench_from_scene
from apple_pick_sim.system_id import BatchedSysIdDataset
from apple_pick_sim.system_id.batched_trajectory_store import episode_filename

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_FIXTURE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)
_DEFAULT_OUT = Path("tmp/unloaded_tcp_wrench")
_CONTROL_HZ_FALLBACK = 30.0
_SUB_DT = 1.0 / 1800.0
_REAL_OSC_KP_NULL = 10.0
_REAL_OSC_KD_NULL = 15.0
_REAL_OSC_SEP_ORI = True


def substeps_for_control_hz(control_hz: float, *, sub_dt: float = _SUB_DT) -> int:
    """Match ``RuntimeConfig.substeps_per_step``."""
    hz = float(control_hz)
    if hz <= 0.0:
        raise ValueError(f"control_hz must be positive, got {hz}")
    return max(1, round(1.0 / (hz * float(sub_dt))))


def list_direction_indices(dataset: BatchedSysIdDataset, structure_idx: int) -> list[int]:
    """Sorted direction indices for one structure from the manifest."""
    dirs = sorted(
        {
            int(ep["direction_idx"])
            for ep in dataset.episode_entries()
            if int(ep.get("structure_idx", -1)) == int(structure_idx)
        }
    )
    if not dirs:
        raise ValueError(f"no episodes for structure_idx={structure_idx}")
    return dirs


def actions_from_episode_arrays(arrays: Mapping[str, Any], *, action_dim: int = 19) -> np.ndarray:
    """Return ``(T, action_dim)`` float32 actions from episode obs arrays."""
    raw = arrays.get("action")
    if raw is None:
        raise ValueError("episode arrays missing action column")
    act = np.asarray(raw, dtype=np.float32)
    if act.ndim != 2 or act.shape[1] < int(action_dim):
        raise ValueError(
            f"action must have shape (T, >= {action_dim}), got {act.shape}"
        )
    return np.ascontiguousarray(act[:, : int(action_dim)])


def write_unloaded_episode_parquet(
    path: Path,
    *,
    ft_tcp_world: np.ndarray,
    tcp_pos: np.ndarray,
    tcp_quat: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write one unloaded tare-baseline parquet with schema metadata.

    Columns:
    - ``ft_tcp_world`` / ``ft_wrist_tare`` — identical world env-on-robot wrench about TCP
    - ``tcp_pos`` / ``tcp_quat`` — TCP pose for CMA-style plots
    """
    ft = np.asarray(ft_tcp_world, dtype=np.float32).reshape(-1, 6)
    pos = np.asarray(tcp_pos, dtype=np.float32).reshape(-1, 3)
    quat = np.asarray(tcp_quat, dtype=np.float32).reshape(-1, 4)
    n = int(ft.shape[0])
    if pos.shape[0] != n or quat.shape[0] != n:
        raise ValueError(
            f"length mismatch: ft={n}, tcp_pos={pos.shape[0]}, tcp_quat={quat.shape[0]}"
        )
    ft_list = ft.tolist()
    table = pa.table(
        {
            "ft_tcp_world": pa.array(ft_list, type=pa.list_(pa.float32(), 6)),
            "ft_wrist_tare": pa.array(ft_list, type=pa.list_(pa.float32(), 6)),
            "tcp_pos": pa.array(pos.tolist(), type=pa.list_(pa.float32(), 3)),
            "tcp_quat": pa.array(quat.tolist(), type=pa.list_(pa.float32(), 4)),
            "step_idx": pa.array(list(range(n)), type=pa.int64()),
        }
    )
    meta = {
        "schema_version": "unloaded_tcp_wrench_v1",
        "ft_tcp_world": (
            "world-frame env-on-robot wrench about TCP from EE body_parent_f; "
            "unloaded tare baseline (not gym plant-harvest ft_wrist)"
        ),
        "ft_wrist_tare": "alias of ft_tcp_world for tare-baseline consumers",
    }
    if metadata:
        meta.update({str(k): str(v) if not isinstance(v, str) else v for k, v in metadata.items()})
    # Arrow schema metadata values must be bytes.
    schema = table.schema.with_metadata(
        {k.encode("utf-8"): str(v).encode("utf-8") for k, v in meta.items()}
    )
    table = table.cast(schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


_FORCE_LABELS = ("Fx", "Fy", "Fz")
_TORQUE_LABELS = ("Tx", "Ty", "Tz")
_POS_LABELS = ("x", "y", "z")
_COLORS_REAL = ("#2563eb", "#1d4ed8", "#1e40af")
_COLORS_TARE = ("#dc2626", "#b91c1c", "#991b1b")


def write_unloaded_force_plots(
    out_dir: Path,
    *,
    direction_idx: int,
    ft_tcp_world: np.ndarray,
    tcp_pos: np.ndarray,
    control_hz: float,
    title_prefix: str,
    real_ft_wrist: np.ndarray | None = None,
    real_tcp_pos: np.ndarray | None = None,
    write_html: bool = True,
    write_png: bool = True,
) -> list[Path]:
    """Write CMA-style per-direction force / torque / TCP plots for the tare baseline.

    When ``real_*`` arrays are provided (from the converted GT bag), overlays them as
    ``real`` vs unloaded ``tare`` with the same colors as ``cma_force_plots``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = np.asarray(ft_tcp_world, dtype=np.float64).reshape(-1, 6)
    pos = np.asarray(tcp_pos, dtype=np.float64).reshape(-1, 3)
    n = int(ft.shape[0])
    t = np.arange(n, dtype=np.float64) / max(float(control_hz), 1e-9)
    real_ft = None if real_ft_wrist is None else np.asarray(real_ft_wrist, dtype=np.float64).reshape(-1, 6)
    real_pos = None if real_tcp_pos is None else np.asarray(real_tcp_pos, dtype=np.float64).reshape(-1, 3)
    if real_ft is not None:
        n_r = min(n, int(real_ft.shape[0]))
        t_r = np.arange(n_r, dtype=np.float64) / max(float(control_hz), 1e-9)
        real_ft = real_ft[:n_r]
    else:
        t_r = t
    if real_pos is not None:
        n_rp = min(n, int(real_pos.shape[0]))
        t_rp = np.arange(n_rp, dtype=np.float64) / max(float(control_hz), 1e-9)
        real_pos = real_pos[:n_rp]
    else:
        t_rp = t

    kinds: list[tuple[str, tuple[str, ...], np.ndarray, np.ndarray | None, np.ndarray, str]] = [
        ("force", _FORCE_LABELS, ft[:, 0:3], None if real_ft is None else real_ft[:, 0:3], t_r, "N"),
        ("torque", _TORQUE_LABELS, ft[:, 3:6], None if real_ft is None else real_ft[:, 3:6], t_r, "N·m"),
        ("tcp", _POS_LABELS, pos, real_pos, t_rp, "m"),
    ]
    written: list[Path] = []
    d = int(direction_idx)

    for stem, labels, tare_y, real_y, t_real, unit in kinds:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, lab in enumerate(labels):
            ax = axes[i]
            if real_y is not None:
                ax.plot(
                    t_real,
                    real_y[:, i],
                    color=_COLORS_REAL[i],
                    lw=1.8,
                    label=f"real {lab}",
                )
            ax.plot(
                t,
                tare_y[:, i],
                color=_COLORS_TARE[i],
                lw=1.8,
                ls="--" if real_y is not None else "-",
                label=f"tare {lab}" if real_y is not None else lab,
            )
            ax.set_ylabel(f"{lab} [{unit}]")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"{title_prefix} d{d:02d} — {stem}")
        fig.tight_layout()
        png_path = out_dir / f"dir_{d:02d}_{stem}.png"
        if write_png:
            fig.savefig(png_path, dpi=150)
            written.append(png_path)
        plt.close(fig)

        if write_html:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError as exc:
                raise SystemExit(
                    "plotly is required for HTML plots; "
                    "uv sync --extra gym --extra vic --extra dev"
                ) from exc
            fig_h = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=list(labels),
                vertical_spacing=0.06,
            )
            for i, lab in enumerate(labels):
                if real_y is not None:
                    fig_h.add_trace(
                        go.Scatter(
                            x=t_real,
                            y=real_y[:, i],
                            mode="lines",
                            name=f"real {lab}",
                            line=dict(color=_COLORS_REAL[i], width=2),
                        ),
                        row=i + 1,
                        col=1,
                    )
                fig_h.add_trace(
                    go.Scatter(
                        x=t,
                        y=tare_y[:, i],
                        mode="lines",
                        name=f"tare {lab}" if real_y is not None else lab,
                        line=dict(
                            color=_COLORS_TARE[i],
                            width=2,
                            dash="dash" if real_y is not None else "solid",
                        ),
                        showlegend=True,
                    ),
                    row=i + 1,
                    col=1,
                )
                fig_h.update_yaxes(title_text=f"{lab} [{unit}]", row=i + 1, col=1)
            fig_h.update_layout(
                title=f"{title_prefix} d{d:02d} — {stem}",
                height=700,
                width=950,
                template="plotly_white",
            )
            fig_h.update_xaxes(title_text="time [s]", row=3, col=1)
            html_path = out_dir / f"dir_{d:02d}_{stem}.html"
            fig_h.write_html(str(html_path), include_plotlyjs=True)
            written.append(html_path)

    return written


def write_tcp_wrench_html(
    path: Path,
    *,
    ft_tcp_world: np.ndarray,
    control_hz: float,
    title: str,
    tcp_pos: np.ndarray | None = None,
) -> None:
    """Backward-compatible wrapper: write force+torque+TCP next to ``path``. """
    out_dir = Path(path).parent
    # Infer direction from ``dir_NN_*.html`` if possible.
    stem = Path(path).stem
    direction_idx = 0
    if stem.startswith("dir_"):
        try:
            direction_idx = int(stem.split("_")[1])
        except (IndexError, ValueError):
            direction_idx = 0
    pos = (
        np.asarray(tcp_pos, dtype=np.float32)
        if tcp_pos is not None
        else np.zeros((np.asarray(ft_tcp_world).reshape(-1, 6).shape[0], 3), dtype=np.float32)
    )
    write_unloaded_force_plots(
        out_dir,
        direction_idx=direction_idx,
        ft_tcp_world=ft_tcp_world,
        tcp_pos=pos,
        control_hz=control_hz,
        title_prefix=title,
        write_html=True,
        write_png=True,
    )


def _vic_gains_from_ranges(ranges: dict) -> ImpedanceGains:
    sb = parse_sim_build(ranges)
    if sb is None or sb.vic_gains is None:
        return ImpedanceGains(linear_k=200.0, linear_d=10.0, angular_k=10.0, angular_d=1.0)
    g = sb.vic_gains
    return ImpedanceGains(
        linear_k=float(g.linear_k),
        linear_d=float(g.linear_d),
        angular_k=float(g.angular_k),
        angular_d=float(g.angular_d),
    )


def build_unloaded_fr3_scene(
    ranges: dict,
    *,
    fruiting_base_pos: tuple[float, float, float],
    bootstrap_joint_q: Sequence[float],
    seed: int = 0,
    robot_gravity: bool = False,
) -> Any:
    """``mujoco_only`` FR3 with ``body_parent_f``, no apple payload, open-loop joints."""
    import newton

    scene = build_coupled_fruiting_fr3(
        ranges,
        int(seed),
        base_pos=fruiting_base_pos,
        robot_base_pos=(0.0, 0.0, 0.0),
        robot_base_from_proxy=False,
        mujoco_only=True,
        enable_self_collisions=False,
        mujoco_solver_kwargs=dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS),
        gripper_proxy=GripperProxyConfig(fix_to_apple=False),
        skip_ik_bootstrap=True,
        request_body_parent_f=True,
    )
    apply_open_loop_fr3_joint_q(scene, bootstrap_joint_q)
    init_robot_mujoco_step_buffers(scene)
    if robot_gravity:
        scene.robot_model.set_gravity((0.0, 0.0, -9.81))
        scene.mj_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    return scene


def configure_unloaded_vic_pose(
    scene: Any,
    *,
    ranges: dict,
) -> fr3_robot.Fr3EEImpedanceController:
    """Wire single-world ``vic_pose`` control (isotropic gains from action means)."""
    scene.robot_kinematic_mode = False
    scene.vic_use_joint_torques = True
    fr3_robot.init_mujoco_actuator_targets_from_model(
        scene.robot_model, scene.robot_control
    )
    fr3_robot.configure_vic_joint_torques_arm(
        scene.robot_model,
        scene.robot_state_0,
        scene.robot_control,
        scene.mj_solver,
        scene=scene,
        kp_null=_REAL_OSC_KP_NULL,
        kd_null=_REAL_OSC_KD_NULL,
        sep_ori=_REAL_OSC_SEP_ORI,
    )
    scene.vic_gains = _vic_gains_from_ranges(ranges)
    scene.vic_joint_torques_configured = True
    ctrl = fr3_robot.Fr3EEImpedanceController(
        tcp_body_index=int(scene.tcp_body_index),
        linear_speed=1.0,
        angular_speed=1.0,
    )
    ctrl.sync_target_from_state(scene.robot_state_0)
    scene.vic_controller = ctrl
    scene.vic_target_tf = ctrl.target_tf
    scene.vic_target_twist = fr3_robot.EEVelocity()
    return ctrl


def apply_vic_pose_action_single(
    ctrl: fr3_robot.Fr3EEImpedanceController,
    scene: Any,
    action_19: Any,
) -> None:
    """Set TCP target + isotropic gains from one 19D ``vic_pose`` action row.

    Layout: ``[pos(3), quat_wxyz(4), Kp(6), Kd(6)]``. Linear/angular gains are
    the means of the per-axis halves (single-env VIC is isotropic).
    """
    a = np.asarray(action_19, dtype=np.float64).reshape(19)
    pos = a[0:3]
    quat_wxyz = a[3:7]
    n = float(np.linalg.norm(quat_wxyz))
    if n < 1e-9:
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat_wxyz = quat_wxyz / n
    # Action contract is wxyz; Warp / body_q store xyzw.
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
    )
    kp = a[7:13]
    kd = a[13:19]
    ctrl.target_tf = wp.transform(
        wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
        wp.quat(
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
        ),
    )
    scene.vic_target_tf = ctrl.target_tf
    scene.vic_target_twist = fr3_robot.EEVelocity()
    scene.vic_gains = ImpedanceGains(
        linear_k=float(np.mean(kp[0:3])),
        linear_d=float(np.mean(kd[0:3])),
        angular_k=float(np.mean(kp[3:6])),
        angular_d=float(np.mean(kd[3:6])),
    )


def tcp_pose_from_scene(scene: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pos(3), quat_xyzw(4))`` for the TCP body."""
    tcp = int(scene.tcp_body_index)
    row = scene.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp]
    pos = np.asarray(row[:3], dtype=np.float32)
    # body_q stores Warp xyzw.
    quat_xyzw = np.asarray(row[3:7], dtype=np.float32)
    return pos, quat_xyzw


def replay_unloaded_direction(
    scene: Any,
    ctrl: fr3_robot.Fr3EEImpedanceController,
    actions: np.ndarray,
    *,
    control_hz: float,
    max_frames: int = 0,
    sub_dt: float = _SUB_DT,
) -> dict[str, np.ndarray]:
    """Drive pose actions and record TCP wrench / pose each control frame."""
    act = np.asarray(actions, dtype=np.float32)
    if act.ndim != 2 or act.shape[1] != 19:
        raise ValueError(f"actions must be (T, 19), got {act.shape}")
    n_frames = int(act.shape[0]) if int(max_frames) <= 0 else min(int(max_frames), int(act.shape[0]))
    n_sub = substeps_for_control_hz(control_hz, sub_dt=sub_dt)
    ft_rows: list[np.ndarray] = []
    pos_rows: list[np.ndarray] = []
    quat_rows: list[np.ndarray] = []

    for i in range(n_frames):
        apply_vic_pose_action_single(ctrl, scene, act[i])
        for _ in range(n_sub):
            scene.mujoco_substep(float(sub_dt))
        ft_rows.append(tcp_world_wrench_from_scene(scene).astype(np.float32))
        pos, quat = tcp_pose_from_scene(scene)
        pos_rows.append(pos)
        quat_rows.append(quat)

    return {
        "ft_tcp_world": np.stack(ft_rows, axis=0),
        "tcp_pos": np.stack(pos_rows, axis=0),
        "tcp_quat": np.stack(quat_rows, axis=0),
    }


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay converted real vic_pose trajectories unloaded (arm-only) and "
            "record world-frame env-on-robot TCP wrench from EE body_parent_f."
        )
    )
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Converted batched_sysid_v1 directory (manifest.json + episodes/).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT}).",
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Ranges fixture (default: dataset collection.ranges_path or variance fixture).",
    )
    p.add_argument("--structure-idx", type=int, default=0)
    p.add_argument(
        "--direction-idx",
        type=int,
        default=None,
        help="Replay one direction (default: all directions for the structure).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use <=0 for full episode (default).",
    )
    p.add_argument(
        "--robot-gravity",
        action="store_true",
        help="Enable Model A gravity (default off: ideal gravity compensation).",
    )
    p.add_argument(
        "--controller-mode",
        choices=["vic_pose"],
        default="vic_pose",
        help="Only vic_pose is supported for unloaded recording.",
    )
    p.add_argument(
        "--allow-wrench-as-twist",
        action="store_true",
        help="Unused for vic_pose; kept for check_action_semantics API parity.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip Plotly HTML force/torque/TCP plots.",
    )
    p.add_argument(
        "--no-png",
        action="store_true",
        help="Skip matplotlib PNG force/torque/TCP plots.",
    )
    return p


def run_unloaded_replay(args: argparse.Namespace) -> int:
    """CLI entry after argparse (testable)."""
    dataset = BatchedSysIdDataset(args.dataset)
    collection = dataset.manifest.get("collection", {})
    ranges_path = Path(
        args.fixture
        if args.fixture is not None
        else collection.get("ranges_path") or _DEFAULT_FIXTURE
    )
    if not ranges_path.is_file():
        raise SystemExit(f"ranges fixture not found: {ranges_path}")
    ranges = load_ranges(ranges_path)

    structure_idx = int(args.structure_idx)
    if args.direction_idx is None:
        direction_indices = list_direction_indices(dataset, structure_idx)
    else:
        direction_indices = [int(args.direction_idx)]

    out_root = Path(args.out)
    episodes_dir = out_root / "episodes"
    plots_dir = out_root / "plots"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    write_plots = (not args.no_html) or (not args.no_png)
    if write_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for direction_idx in direction_indices:
        try:
            episode_meta = dataset.load_episode_metadata(structure_idx, direction_idx)
            fruiting_base_pos = fruiting_base_pos_from_episode_metadata(episode_meta)
            bootstrap_joint_q = bootstrap_joint_q_from_episode_metadata(episode_meta)
            control_hz = control_hz_from_episode_metadata(
                episode_meta, collection=collection
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        check_action_semantics(
            controller_mode="vic_pose",
            collection=collection,
            episode_meta=episode_meta,
            allow_wrench_as_twist=bool(args.allow_wrench_as_twist),
        )
        if not dataset_declares_vic_pose(collection, episode_meta):
            raise SystemExit(
                "dataset must declare 19D vic_pose actions "
                "(action_layout=vic_pose_v1 or action_dim=19)"
            )

        arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
        actions = actions_from_episode_arrays(arrays, action_dim=19)

        print(
            f"unloaded s{structure_idx:02d}_d{direction_idx:02d} "
            f"frames={actions.shape[0]} control_hz={control_hz}",
            file=sys.stderr,
        )
        scene = build_unloaded_fr3_scene(
            ranges,
            fruiting_base_pos=fruiting_base_pos,
            bootstrap_joint_q=bootstrap_joint_q,
            seed=int(args.seed),
            robot_gravity=bool(args.robot_gravity),
        )
        ctrl = configure_unloaded_vic_pose(scene, ranges=ranges)
        recorded = replay_unloaded_direction(
            scene,
            ctrl,
            actions,
            control_hz=control_hz,
            max_frames=int(args.max_frames),
        )

        ep_name = Path(episode_filename(structure_idx, direction_idx)).name
        ep_path = episodes_dir / ep_name
        write_unloaded_episode_parquet(
            ep_path,
            ft_tcp_world=recorded["ft_tcp_world"],
            tcp_pos=recorded["tcp_pos"],
            tcp_quat=recorded["tcp_quat"],
            metadata={
                "structure_idx": structure_idx,
                "direction_idx": direction_idx,
                "control_hz": control_hz,
                "robot_gravity": bool(args.robot_gravity),
            },
        )
        if write_plots:
            real_ft = arrays.get("ft_wrist")
            real_tcp = arrays.get("tcp_pos")
            write_unloaded_force_plots(
                plots_dir,
                direction_idx=direction_idx,
                ft_tcp_world=recorded["ft_tcp_world"],
                tcp_pos=recorded["tcp_pos"],
                control_hz=control_hz,
                title_prefix=f"unloaded tare s{structure_idx:02d}",
                real_ft_wrist=real_ft,
                real_tcp_pos=real_tcp,
                write_html=not args.no_html,
                write_png=not args.no_png,
            )

        ft = recorded["ft_tcp_world"]
        summary.append(
            {
                "structure_idx": structure_idx,
                "direction_idx": direction_idx,
                "frames": int(ft.shape[0]),
                "episode": str(ep_path),
                "ft_rms": float(np.sqrt(np.mean(ft[:, :3] ** 2))),
                "tau_rms": float(np.sqrt(np.mean(ft[:, 3:] ** 2))),
            }
        )
        print(
            f"  wrote {ep_path} frames={ft.shape[0]} "
            f"ft_rms={summary[-1]['ft_rms']:.4g} tau_rms={summary[-1]['tau_rms']:.4g}",
            file=sys.stderr,
        )

    (out_root / "unloaded_report.json").write_text(
        json.dumps(
            {
                "dataset": str(Path(args.dataset).resolve()),
                "robot_gravity": bool(args.robot_gravity),
                "episodes": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"OK: wrote {len(summary)} unloaded episode(s) under {out_root}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    return run_unloaded_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
