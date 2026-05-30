"""Keyboard teleop: one FR3 drives N mega VBD plant columns (fd_ghost coupling).

Run from repository root::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_mega_coupled_keyboard.py --viewer gl

With ``--viewer gl``, focus the **simulation window** and use the same keys as
``example_coupled_fruiting.py --fr3-keyboard`` (I/K ±X, J/L ±Y, R/F ±Z, Z/X T/G U/O rotations).
Pass ``--terminal-keys`` to drive from the shell instead (q quits).

With **≥2** stiffness columns (default mega build), each frame is: keyboard teleop → sync all columns
from nominal → ``--fd-substeps`` × :meth:`MegaCoupledFruitingScene.coupled_substep` (same EE motion on
every column, different stiffness) → Jacobian / optional FIM → reset perturbed columns → viewer.

Single-column builds use a normal substep loop only.

``--fix-to-apple`` welds the proxy to the apple (stem-harvest coupling on the nominal column); use
``--fix-to-apple-warmup-substeps`` to settle freely before welding (same as ``example_gym_keyboard.py``).

``--global-frame-viz`` draws world RGB axes, per-column ``base_pos`` frames, and nominal apple/proxy
body frames (requires a viewer with ``log_lines`` / ``log_arrows``, e.g. ``--viewer gl``).

``--tcp-force-arrow`` draws harvested TCP force as a yellow arrow at the robot TCP (same viewer).
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path

import newton
import newton.examples
import numpy as np


def _default_ranges_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "fruiting_system_ranges_straight_rod_test.json"
    )


@contextmanager
def _raw_terminal_mode():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_nonblocking() -> str | None:
    if not sys.stdin.isatty():
        return None
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not r:
        return None
    return sys.stdin.read(1)


def _key_to_velocity(ch: str | None):
    from apple_pick_sim.robot import fr3_robot

    if ch is None:
        return fr3_robot.EEVelocity()
    c = ch.lower()
    if c == " ":
        return fr3_robot.EEVelocity()
    lin, ang = 0.2, 1.0
    table = {
        "i": fr3_robot.EEVelocity(linear=(+lin, 0.0, 0.0)),
        "k": fr3_robot.EEVelocity(linear=(-lin, 0.0, 0.0)),
        "j": fr3_robot.EEVelocity(linear=(0.0, +lin, 0.0)),
        "l": fr3_robot.EEVelocity(linear=(0.0, -lin, 0.0)),
        "r": fr3_robot.EEVelocity(linear=(0.0, 0.0, +lin)),
        "f": fr3_robot.EEVelocity(linear=(0.0, 0.0, -lin)),
        "z": fr3_robot.EEVelocity(angular=(+ang, 0.0, 0.0)),
        "x": fr3_robot.EEVelocity(angular=(-ang, 0.0, 0.0)),
        "t": fr3_robot.EEVelocity(angular=(0.0, +ang, 0.0)),
        "g": fr3_robot.EEVelocity(angular=(0.0, -ang, 0.0)),
        "u": fr3_robot.EEVelocity(angular=(0.0, 0.0, +ang)),
        "o": fr3_robot.EEVelocity(angular=(0.0, 0.0, -ang)),
    }
    return table.get(c, fr3_robot.EEVelocity())


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Fruiting ranges JSON (default: example_variance fixture).",
    )
    p.add_argument(
        "--stiffness-epsilon",
        type=float,
        default=0.02,
        help="FD column stiffness perturbation when building mega plant.",
    )
    p.add_argument(
        "--fd-substeps",
        type=int,
        default=None,
        help=(
            "Coupled substeps per frame for FD columns (default: SUBSTEPS_PER_FRAME, 30 at 30 Hz)."
        ),
    )
    p.add_argument(
        "--fd-print-interval",
        type=int,
        default=30,
        help="Print FD diagnostics every N frames when using multiple stiffness columns.",
    )
    p.add_argument(
        "--fd-fim",
        action="store_true",
        help="Compute per-frame FIM (sigma_inv = identity on default features).",
    )
    p.add_argument(
        "--fix-to-apple",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Weld gripper proxy to apple on every column (stem-harvest on nominal).",
    )
    p.add_argument(
        "--fix-to-apple-warmup-substeps",
        type=int,
        default=1800,
        help="VBD substeps to settle before welding (fix_to_apple only).",
    )
    p.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="Open MuJoCo passive viewer for the FR3 arm.",
    )
    p.add_argument(
        "--global-frame-viz",
        action="store_true",
        help="Draw world / column-base / nominal body RGB coordinate frames in the Newton viewer.",
    )
    p.add_argument(
        "--global-frame-length",
        type=float,
        default=0.35,
        help="Axis length [m] for --global-frame-viz (default 0.35).",
    )
    p.add_argument(
        "--tcp-force-arrow",
        action="store_true",
        help="Draw harvested TCP force as a yellow arrow at the robot TCP.",
    )
    p.add_argument(
        "--tcp-force-scale",
        type=float,
        default=0.02,
        help="Arrow length per newton [m/N] for --tcp-force-arrow (default 0.02).",
    )
    p.add_argument(
        "--tcp-force-arrow-gain",
        type=float,
        default=1.0,
        help="Dimensionless multiplier on --tcp-force-scale.",
    )
    p.add_argument(
        "--tcp-force-min-length",
        type=float,
        default=0.08,
        help="Minimum arrow length [m] (default 0.08).",
    )
    p.add_argument(
        "--tcp-force-max-length",
        type=float,
        default=1.5,
        help="Maximum arrow length [m]; 0 = no cap (default 1.5).",
    )
    p.add_argument(
        "--terminal-keys",
        action="store_true",
        help="Read teleop from this shell (default: ViewerGL window when available).",
    )
    return p


def _viewer_has_keyboard(viewer: object) -> bool:
    """True for ``ViewerGL`` (renderer polls keys); false for null/viser headless viewers."""
    if viewer is None or not hasattr(viewer, "is_key_down"):
        return False
    renderer = getattr(viewer, "renderer", None)
    return renderer is not None and hasattr(renderer, "update")


def _compute_coupled_fd_result(
    scene,
    epsilon: float,
    *,
    nominal_index: int,
    sigma_inv: np.ndarray | None,
    dt: float,
):
    """Features / FD Jacobian / FIM after coupled substeps (see :func:`extract_mega_fd_jacobian`)."""
    from apple_pick_sim.fruiting_system.mega_fd import extract_mega_fd_jacobian

    return extract_mega_fd_jacobian(
        scene.cable,
        epsilon,
        nominal_index=nominal_index,
        dt=dt,
        sigma_inv=sigma_inv,
    )


def _print_fd_state_errors(
    scene,
    result,
    *,
    frame_index: int,
    nominal_index: int,
) -> None:
    from apple_pick_sim.fruiting_system.mega_fd import default_mega_fd_features

    y_nom = result.features[nominal_index]
    n_inst = int(result.features.shape[0])
    nom_params = scene.cable.instance(nominal_index).params

    print(f"[fd] frame {frame_index} (nominal column {nominal_index}):")
    for col in range(n_inst):
        if col == nominal_index:
            continue

        # Resolve the perturbed segment name by comparing stiffness parameters
        pert_params = scene.cable.instance(col).params
        perturbed_seg = "unknown"
        for seg_name in ("primary", "secondary", "spur", "stem"):
            nom_seg = getattr(nom_params, seg_name, None)
            pert_seg = getattr(pert_params, seg_name, None)
            if nom_seg is not None and pert_seg is not None:
                if abs(nom_seg.bend_stiffness - pert_seg.bend_stiffness) > 1e-5:
                    perturbed_seg = seg_name
                    break

        feat_diff = result.features[col] - y_nom
        feat_err = float(np.linalg.norm(feat_diff))
        y_after = default_mega_fd_features(scene.cable, col)
        reset_err = float(np.linalg.norm(y_after - y_nom))
        jcol = col if col < nominal_index else col - 1
        j_norm = float(np.linalg.norm(result.jacobian[:, jcol]))

        # High-level feature group breakdown
        apple_norm = float(np.linalg.norm(feat_diff[:3]))
        proxy_norm = float(np.linalg.norm(feat_diff[3:6]))
        group_strs = [f"apple={apple_norm:.2e}", f"proxy={proxy_norm:.2e}"]

        feature_names = ["apple_x", "apple_y", "apple_z", "proxy_x", "proxy_y", "proxy_z"]
        if len(feat_diff) == 12:
            force_norm = float(np.linalg.norm(feat_diff[6:9]))
            torque_norm = float(np.linalg.norm(feat_diff[9:12]))
            group_strs += [f"force={force_norm:.2e}", f"torque={torque_norm:.2e}"]
            feature_names += ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]

        groups_str = ", ".join(group_strs)

        # Identify individual feature with the largest sensitivity (max absolute entry in Jacobian column)
        j_col_data = result.jacobian[:, jcol]
        max_idx = int(np.argmax(np.abs(j_col_data)))
        max_feat_name = feature_names[max_idx]
        max_feat_val = float(j_col_data[max_idx])

        print(
            f"  col {col} ({perturbed_seg}): |y-y_nom|={feat_err:.6e}  "
            f"post-reset |y-y_nom|={reset_err:.6e}  |J_col|={j_norm:.6e}\n"
            f"    [Groups] {groups_str}\n"
            f"    [Max Sens] {max_feat_name}={max_feat_val:+.3e}"
        )
    if result.fim_step is not None:
        print(f"  FIM trace={float(np.trace(result.fim_step)):.6e}")




def _print_nominal_apple_mass(scene) -> None:
    """Print apple ``model.body_mass`` for the nominal mega column."""
    from apple_pick_sim.fruiting_system.params import analytic_apple_mass_kg

    inst = scene.cable.instance(scene.nominal_index)
    if inst.apple_body is None:
        print("Apple mass: N/A (no apple body in nominal column).")
        return
    bid = inst.apple_body
    m_model = float(scene.cable.model.body_mass.numpy()[bid])
    p = inst.params
    m_phys = analytic_apple_mass_kg(p)
    if m_phys is not None:
        print(
            f"Apple mass: {m_model:.6f} kg "
            f"(r={p.apple_radius:.4f} m, rho={p.apple_density:.1f} kg/m³, col {scene.nominal_index}). "
            f"Quasi-static hold: expect TCP |F| ≈ {m_phys * 9.81:.2f} N (stem harvest, gain=1)."
        )
    else:
        print(f"Apple mass: {m_model:.6f} kg (nominal column {scene.nominal_index}).")


def _log_telemetry(viewer: object, scene, sub_dt: float) -> None:
    log = getattr(viewer, "log_scalar", None)
    if log is None:
        return

    from apple_pick_sim.coupling_force_debug import read_tcp_wrench
    from apple_pick_sim.fruiting_system.mega_fd import default_mega_fd_features

    w = read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index)
    log("Mega EE |F| [N]", float(np.linalg.norm(w[:3])), smoothing=3)

    for i in range(scene.cable.num_instances):
        feat = default_mega_fd_features(scene.cable, i)
        log(f"Col{i} proxy Z [m]", float(feat[-1]), smoothing=3)


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null (override with --viewer gl).")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    import apple_pick_sim.coupled_fruiting as cf
    import apple_pick_sim.fruiting_system as fs
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.sim_device import resolve_sim_device

    fr3_robot.enable_ik_bootstrap_warnings_for_examples()
    from apple_pick_sim.tests.conftest import SUBSTEPS_PER_FRAME, SUB_DT

    sim_device = resolve_sim_device(getattr(args, "device", None))
    print(f"Warp device: {sim_device}")

    ranges_path = args.json if args.json is not None else _default_ranges_path()
    ranges = fs.load_ranges(ranges_path)
    fix_to_apple = bool(getattr(args, "fix_to_apple", False))
    warmup = int(getattr(args, "fix_to_apple_warmup_substeps", 3000))
    build_kw = dict(
        stiffness_epsilon=float(args.stiffness_epsilon),
        enable_self_collisions=False,
        mujoco_solver_kwargs={"disable_contacts": True},
        device=sim_device,
    )
    gripper_free = fs.GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        fix_to_apple=False,
    )
    gripper_weld = fs.GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        fix_to_apple=True,
    )

    if fix_to_apple and warmup > 0:
        settled = cf.build_mega_coupled_fruiting_fr3(
            ranges, int(args.seed), gripper_proxy=gripper_free, **build_kw
        )
        cf.settle_vbd_substeps(settled, substeps=warmup, dt=SUB_DT)
        scene = cf.build_mega_coupled_fruiting_fr3(
            ranges,
            int(args.seed),
            gripper_proxy=gripper_weld,
            **build_kw,
        )
        cf.seed_mega_fix_to_apple_from_settled(
            welded_scene=scene, settled_scene=settled, quiet_apple_proxy=True
        )
    else:
        scene = cf.build_mega_coupled_fruiting_fr3(
            ranges,
            int(args.seed),
            gripper_proxy=gripper_weld if fix_to_apple else gripper_free,
            **build_kw,
        )

    controller = fr3_robot.Fr3EEDirectJointController(
        scene.robot_model, scene.tcp_body_index
    )
    controller.sync_target_from_state(scene.robot_state_0)

    viewer.set_model(scene.cable.model)
    frame_dt = 1.0 / float(args.hz)
    sim_time = 0.0
    fd_print_interval = max(1, int(args.fd_print_interval))
    fd_fim = bool(args.fd_fim)
    fd_substeps = (
        int(args.fd_substeps)
        if args.fd_substeps is not None
        else SUBSTEPS_PER_FRAME
    )
    if fd_substeps < 1:
        raise ValueError("--fd-substeps must be >= 1")
    mujoco_viewer = bool(args.mujoco_viewer)
    global_frame_viz = bool(args.global_frame_viz)
    global_frame_length = float(args.global_frame_length)
    if global_frame_length <= 0.0:
        raise ValueError("--global-frame-length must be positive")
    tcp_force_arrow = bool(args.tcp_force_arrow)
    tcp_force_scale = float(args.tcp_force_scale)
    tcp_force_gain = float(args.tcp_force_arrow_gain)
    tcp_force_min_len = float(args.tcp_force_min_length)
    tcp_force_max_len = float(args.tcp_force_max_length)
    if tcp_force_scale <= 0.0:
        raise ValueError("--tcp-force-scale must be positive")
    if tcp_force_gain <= 0.0:
        raise ValueError("--tcp-force-arrow-gain must be positive")
    if tcp_force_min_len < 0.0:
        raise ValueError("--tcp-force-min-length must be >= 0")
    if tcp_force_max_len < 0.0:
        raise ValueError("--tcp-force-max-length must be >= 0")
    if mujoco_viewer and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("Suppressing --mujoco-viewer (no DISPLAY/WAYLAND_DISPLAY).")
        mujoco_viewer = False

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    harvest = "stem-harvest" if fix_to_apple else "velocity-delta (nominal proxy)"
    n_cols = scene.cable.num_instances
    fd_mode = n_cols > 1
    fd_sigma_inv: np.ndarray | None = None
    if fd_mode and fd_fim:
        from apple_pick_sim.fruiting_system.mega_fd import default_mega_fd_features

        feat_dim = default_mega_fd_features(scene.cable, scene.nominal_index).size
        fd_sigma_inv = np.eye(feat_dim)
    if fd_mode:
        print(
            f"Mega coupled FD: {n_cols} stiffness columns, {fd_substeps} coupled substeps/frame, "
            f"epsilon={args.stiffness_epsilon}, fim={fd_fim}."
        )
    else:
        print(
            f"Mega coupled: {n_cols} column(s), {SUBSTEPS_PER_FRAME} substeps/frame "
            f"(single column — no FD sweep)."
        )
    print(
        f"fix_to_apple={fix_to_apple} ({harvest}), global_frame_viz={global_frame_viz}, "
        f"tcp_force_arrow={tcp_force_arrow}."
    )
    _print_nominal_apple_mass(scene)
    use_terminal_keys = bool(getattr(args, "terminal_keys", False))
    use_viewer_keys = _viewer_has_keyboard(viewer) and not use_terminal_keys
    if use_viewer_keys:
        fr3_robot.print_fr3_keyboard_bindings()
        print("Focus the simulation window to drive the arm (q in terminal still quits).")
    else:
        print(
            "Terminal keyboard (q to quit): i/k ±X, j/l ±Y, r/f ±Z, "
            "z/x t/g u/o rotations, space noop"
        )

    try:
        with _raw_terminal_mode():
            for _step in range(int(args.max_steps)):
                if not viewer.is_running():
                    break

                key = _read_key_nonblocking()
                if key is not None and key.lower() == "q":
                    break
                if use_viewer_keys:
                    fr3_robot.poll_viewer_events(viewer)
                    velocity = _key_to_velocity(key) if key is not None else None
                else:
                    velocity = _key_to_velocity(key)

                scene.apply_fr3_ee_teleop_direct(
                    frame_dt,
                    controller,
                    velocity=velocity,
                    viewer=viewer if use_viewer_keys else None,
                )

                if fd_mode:
                    from apple_pick_sim.fruiting_system.mega_fd import (
                        reset_perturbed_instances_to_nominal,
                    )

                    reset_perturbed_instances_to_nominal(
                        scene.cable, nominal_index=scene.nominal_index
                    )
                    for _ in range(fd_substeps):
                        scene.coupled_substep(SUB_DT)
                    fd_result = _compute_coupled_fd_result(
                        scene,
                        float(args.stiffness_epsilon),
                        nominal_index=scene.nominal_index,
                        sigma_inv=fd_sigma_inv,
                        dt=SUB_DT,
                    )
                    reset_perturbed_instances_to_nominal(
                        scene.cable, nominal_index=scene.nominal_index
                    )
                    if _step % fd_print_interval == 0:
                        _print_fd_state_errors(
                            scene,
                            fd_result,
                            frame_index=_step,
                            nominal_index=scene.nominal_index,
                        )
                else:
                    for _ in range(SUBSTEPS_PER_FRAME):
                        scene.coupled_substep(SUB_DT)

                sim_time += frame_dt

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
                if global_frame_viz:
                    from apple_pick_sim.global_frame_viz import log_mega_global_frames

                    log_mega_global_frames(
                        viewer,
                        scene.cable,
                        nominal_index=scene.nominal_index,
                        axis_length=global_frame_length,
                    )
                if tcp_force_arrow:
                    from apple_pick_sim.tcp_force_viz import log_coupled_scene_tcp_force

                    log_coupled_scene_tcp_force(
                        viewer,
                        scene,
                        scale_per_newton=tcp_force_scale,
                        gain=tcp_force_gain,
                        min_length=tcp_force_min_len,
                        max_length=tcp_force_max_len,
                    )
                _log_telemetry(viewer, scene, SUB_DT)
                viewer.end_frame()

                if mujoco_viewer:
                    fr3_robot.sync_mujoco_visual_state(
                        scene.mj_solver, scene.robot_model, scene.robot_state_0
                    )
                    scene.mj_solver.render_mujoco_viewer()

                time.sleep(max(0.0, frame_dt))
    finally:
        if scene.mj_solver is not None:
            scene.mj_solver.close_mujoco_viewer()


if __name__ == "__main__":
    main()
