"""Parser and smoke tests for example_batched_heterogeneous_coupled_sim (V.3.2)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES,
    EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KD_OVERRIDES,
    EXAMPLE_JOINT_LINEAR_KP_OVERRIDES,
    ObsConfig,
    RuntimeConfig,
)

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _EXAMPLES_DIR / "example_batched_heterogeneous_coupled_sim.py"

if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from example_batched_heterogeneous_coupled_sim import (  # noqa: E402
    JOINT_ANGULAR_KD_OVERRIDES,
    JOINT_ANGULAR_KP_OVERRIDES,
    JOINT_LINEAR_KD_OVERRIDES,
    JOINT_LINEAR_KP_OVERRIDES,
    _VIC_DEFAULT_ANGULAR_D,
    _VIC_DEFAULT_ANGULAR_K,
    _VIC_DEFAULT_LINEAR_D,
    _VIC_DEFAULT_LINEAR_K,
    _make_parser,
    _resolve_step_mode,
    _validate_tcp_force_viz_args,
    _viz_settings_from_args,
)

_PHYSICS_SUB_DT = 1.0 / 1800.0


def test_example_timing_matches_frame_dt():
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.defaults(),
        runtime=RuntimeConfig(
            num_envs=2,
            device="cpu",
            control_hz=30.0,
            sub_dt=_PHYSICS_SUB_DT,
        ),
    )
    assert cfg.runtime.sub_dt * cfg.runtime.substeps_per_step == pytest.approx(cfg.runtime.frame_dt)
    assert cfg.runtime.sub_dt == pytest.approx(_PHYSICS_SUB_DT)


def test_only_vbd_parser_flag():
    args = _make_parser().parse_args(["--only-vbd"])
    assert _resolve_step_mode(args) == "vbd"
    assert args.only_vbd is True
    assert args.only_mjc is False


def test_only_vbd_and_only_mjc_mutually_exclusive():
    import argparse

    with pytest.raises(SystemExit, match="mutually exclusive"):
        _resolve_step_mode(argparse.Namespace(only_vbd=True, only_mjc=True))


def test_only_mjc_rejected():
    import argparse

    with pytest.raises(SystemExit, match="--only-mjc"):
        _resolve_step_mode(argparse.Namespace(only_vbd=False, only_mjc=True))


def test_collision_parser_defaults():
    from example_batched_heterogeneous_coupled_sim import _config_from_args  # noqa: E402

    args = _make_parser().parse_args([])
    args._resolved_seed = 42
    cfg = _config_from_args(args)
    assert cfg.scene.enable_apple_woody_collisions is True
    assert cfg.scene.enable_proxy_woody_collisions is True

    args_off = _make_parser().parse_args(
        ["--no-apple-woody-collision", "--no-proxy-woody-collision"]
    )
    args_off._resolved_seed = 42
    cfg_off = _config_from_args(args_off)
    assert cfg_off.scene.enable_apple_woody_collisions is False
    assert cfg_off.scene.enable_proxy_woody_collisions is False


def test_config_from_args_matches_defaults_without_cli_overrides():
    from example_batched_heterogeneous_coupled_sim import _config_from_args  # noqa: E402

    args = _make_parser().parse_args([])
    cfg = _config_from_args(args)
    base = BatchedHeterogeneousCoupledSimConfig.defaults()

    assert cfg.runtime.num_envs == base.runtime.num_envs
    assert cfg.runtime.control_hz == base.runtime.control_hz
    assert cfg.runtime.env_spacing == base.runtime.env_spacing
    assert cfg.scene.settle_substeps == base.scene.settle_substeps
    assert cfg.robot.kind == base.robot.kind
    assert cfg.robot.fix_to_apple == base.robot.fix_to_apple
    assert cfg.robot.per_env_ik == base.robot.per_env_ik
    assert cfg.robot.ik_bootstrap_iterations == base.robot.ik_bootstrap_iterations
    assert cfg.controller.mode == base.controller.mode
    assert cfg.settle_diagnostics is not None
    assert cfg.obs is None
    assert cfg.domain_randomization.ranges_path == base.domain_randomization.ranges_path
    assert cfg.domain_randomization.topology_seed is None
    assert cfg.fruiting_system.joint_angular_kd_overrides == JOINT_ANGULAR_KD_OVERRIDES
    assert cfg.fruiting_system.joint_linear_kd_overrides == JOINT_LINEAR_KD_OVERRIDES
    assert cfg.fruiting_system.joint_angular_kp_overrides == JOINT_ANGULAR_KP_OVERRIDES
    assert cfg.fruiting_system.joint_linear_kp_overrides == JOINT_LINEAR_KP_OVERRIDES


def test_joint_kd_overrides_stay_in_module_constants():
    from example_batched_heterogeneous_coupled_sim import _config_from_args  # noqa: E402

    assert JOINT_ANGULAR_KD_OVERRIDES is EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES
    assert JOINT_LINEAR_KD_OVERRIDES is EXAMPLE_JOINT_LINEAR_KD_OVERRIDES
    assert JOINT_ANGULAR_KP_OVERRIDES is EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES
    assert JOINT_LINEAR_KP_OVERRIDES is EXAMPLE_JOINT_LINEAR_KP_OVERRIDES

    args = _make_parser().parse_args([])
    args._resolved_seed = 42
    cfg = _config_from_args(args)
    base = BatchedHeterogeneousCoupledSimConfig.defaults()
    assert cfg.fruiting_system == dataclasses.replace(
        base.fruiting_system,
        joint_angular_kd_overrides=JOINT_ANGULAR_KD_OVERRIDES,
        joint_linear_kd_overrides=JOINT_LINEAR_KD_OVERRIDES,
        joint_angular_kp_overrides=JOINT_ANGULAR_KP_OVERRIDES,
        joint_linear_kp_overrides=JOINT_LINEAR_KP_OVERRIDES,
    )
    args = _make_parser().parse_args([])
    assert args.vic_linear_k == _VIC_DEFAULT_LINEAR_K == 600.0
    assert args.vic_linear_d == _VIC_DEFAULT_LINEAR_D == 200.0
    assert args.vic_angular_k == _VIC_DEFAULT_ANGULAR_K == 20.0
    assert args.vic_angular_d == _VIC_DEFAULT_ANGULAR_D == 4.0


def test_show_settling_parser_defaults():
    args = _make_parser().parse_args([])
    assert args.show_settling is False

    args_on = _make_parser().parse_args(["--show-settling"])
    assert args_on.show_settling is True


def test_show_settling_disables_cache_and_forces_settle():
    args = _make_parser().parse_args(["--show-settling", "--viewer", "null"])
    graphical = False
    show_settling = graphical and bool(args.show_settling)
    use_settle_cache = bool(args.use_settle_cache) and not show_settling
    force_settle = bool(args.force_settle) or show_settling
    assert show_settling is False
    assert use_settle_cache is True
    assert force_settle is False

    args_gl = _make_parser().parse_args(["--show-settling", "--viewer", "gl"])
    show_settling_gl = True and bool(args_gl.show_settling)
    use_settle_cache_gl = bool(args_gl.use_settle_cache) and not show_settling_gl
    force_settle_gl = bool(args_gl.force_settle) or show_settling_gl
    assert show_settling_gl is True
    assert use_settle_cache_gl is False
    assert force_settle_gl is True


def test_tcp_force_viz_parser_defaults():
    args = _make_parser().parse_args([])
    assert args.tcp_force_arrow is False
    assert args.tcp_force_scale == 0.02
    assert args.tcp_force_arrow_gain == 1.0
    assert args.tcp_force_min_length == 0.08
    assert args.tcp_force_max_length == 1.5
    assert args.mark_endpoints is False


def test_tcp_force_viz_parser_accepts_flags():
    args = _make_parser().parse_args(
        [
            "--tcp-force-arrow",
            "--tcp-force-scale",
            "0.03",
            "--tcp-force-arrow-gain",
            "2.0",
            "--tcp-force-min-length",
            "0.1",
            "--tcp-force-max-length",
            "2.0",
            "--mark-endpoints",
        ]
    )
    viz = _viz_settings_from_args(args)
    assert viz.tcp_force_arrow is True
    assert viz.tcp_force_scale == 0.03
    assert viz.tcp_force_gain == 2.0
    assert viz.tcp_force_min_length == 0.1
    assert viz.tcp_force_max_length == 2.0
    assert viz.mark_endpoints is True


def test_tcp_force_scale_must_be_positive():
    import argparse

    args = argparse.Namespace(
        tcp_force_scale=0.0,
        tcp_force_arrow_gain=1.0,
        tcp_force_min_length=0.08,
        tcp_force_max_length=1.5,
    )
    with pytest.raises(ValueError, match="--tcp-force-scale"):
        _validate_tcp_force_viz_args(args)


def test_config_allocates_obs_only_when_viz_flags_set():
    from example_batched_heterogeneous_coupled_sim import _config_from_args  # noqa: E402

    args = _make_parser().parse_args([])
    args._resolved_seed = 42
    cfg_default = _config_from_args(args)
    assert cfg_default.obs is None

    args_viz = _make_parser().parse_args(["--mark-endpoints", "--tcp-force-arrow"])
    args_viz._resolved_seed = 42
    cfg_viz = _config_from_args(args_viz)
    assert cfg_viz.obs == ObsConfig(allocate_buffers=True, include_robot=True, include_forces=True)


@pytest.mark.slow
def test_example_headless_smoke_subprocess():
    cmd = [
        sys.executable,
        str(_EXAMPLE),
        "--viewer",
        "null",
        "--num-frames",
        "5",
        "--settle-substeps",
        "20",
        "--num-envs",
        "2",
        "--seed",
        "42",
        "--no-use-settle-cache",
        "--device",
        "cpu",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.slow
def test_example_headless_smoke_with_viz_flags_subprocess():
    cmd = [
        sys.executable,
        str(_EXAMPLE),
        "--viewer",
        "null",
        "--num-frames",
        "3",
        "--settle-substeps",
        "20",
        "--num-envs",
        "2",
        "--seed",
        "42",
        "--no-use-settle-cache",
        "--device",
        "cpu",
        "--mark-endpoints",
        "--tcp-force-arrow",
        "--status-every",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mark-endpoints" in proc.stdout.lower() or "Endpoint markers" in proc.stdout
