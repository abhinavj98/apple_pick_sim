"""CLI and config contract tests for the batched sys-ID collection example."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "batched_examples"
        / "example_batched_collect_sysid_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "example_batched_collect_sysid_under_test", path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collection_and_trajectory_cli_defaults(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    args = parser.parse_args(["--output", "/tmp/out"])

    assert args.num_structures == 1
    assert args.num_directions == 1
    assert args.max_steps == 0
    assert args.seed == 0
    assert args.topology_seed == 42
    assert args.overwrite is False
    assert args.debug is False
    assert args.show_pull_direction is False
    assert args.save_snapshot is False
    assert args.settle_substeps is None
    assert args.settle_gravity_ramp is False
    assert args.settle_quiet_every is None
    assert args.ranges_path is None
    assert args.movement_per_step_m == 0.02
    assert args.total_movement_m == 0.10
    assert args.move_speed_mps == 0.2
    assert args.hold_duration_s == 1.5
    assert args.skip_return is True


def test_trajectory_config_from_cli_args(monkeypatch):
    module = _load_example_module()

    import newton.examples

    monkeypatch.setattr(newton.examples, "create_parser", argparse.ArgumentParser)

    parser = module._make_parser()
    args = parser.parse_args(
        [
            "--output",
            "/tmp/out",
            "--movement-per-step-m",
            "0.01",
            "--total-movement-m",
            "0.05",
            "--move-speed-mps",
            "0.1",
            "--hold-duration-s",
            "0.5",
            "--no-skip-return",
        ]
    )

    traj = module.build_trajectory_config(args)
    assert traj.movement_per_step_m == 0.01
    assert traj.total_movement_m == 0.05
    assert traj.move_speed_mps == 0.1
    assert traj.hold_duration_s == 0.5
    assert traj.control_hz == module.CONTROL_HZ
    assert traj.skip_return is False


def test_trajectory_debug_formatters(monkeypatch):
    module = _load_example_module()
    from apple_pick_sim.system_id import QuasiStaticStepConfig

    config = QuasiStaticStepConfig(
        movement_per_step_m=0.005,
        total_movement_m=0.05,
        move_speed_mps=0.015,
        hold_duration_s=10.0 / 30.0,
        control_hz=30.0,
    )
    summary = module.summarize_trajectory_for_debug(config)
    assert "10 increments" in summary
    assert "move_out + 10 hold" in summary
    assert "@ 30 Hz" in summary

    assert "phase=move_out(0)" in module.format_trajectory_step_debug(
        step_idx=0,
        phase="move_out",
        sim_time=1.0 / 30.0,
        amplitude_m=0.005,
    )
    assert "phase=hold    (1)" in module.format_trajectory_step_debug(
        step_idx=11,
        phase="hold",
        sim_time=0.5,
        amplitude_m=0.005,
    )
    assert "phase=pre_weld(-1)" in module.format_trajectory_step_debug(
        step_idx=-1,
        phase="pre_weld",
        sim_time=0.0,
    )


def test_sim_config_stays_in_module_constants():
    module = _load_example_module()

    cfg = module.build_sim_config(num_envs=4)
    gym_cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=4)
    assert cfg == dataclasses.replace(
        gym_cfg,
        runtime=dataclasses.replace(gym_cfg.runtime, control_hz=module.CONTROL_HZ),
        scene=dataclasses.replace(
            gym_cfg.scene,
            settle_substeps=module.SETTLE_SUBSTEPS,
            settle_gravity_ramp=module.SETTLE_GRAVITY_RAMP,
            settle_quiet_every=module.SETTLE_QUIET_EVERY,
        ),
        controller=dataclasses.replace(
            gym_cfg.controller,
            vic_gains=module.VIC_GAINS,
        ),
        fruiting_system=dataclasses.replace(
            gym_cfg.fruiting_system,
            joint_angular_kd_overrides=module.JOINT_ANGULAR_KD_OVERRIDES,
            joint_linear_kd_overrides=module.JOINT_LINEAR_KD_OVERRIDES,
            joint_angular_kp_overrides=module.JOINT_ANGULAR_KP_OVERRIDES,
            joint_linear_kp_overrides=module.JOINT_LINEAR_KP_OVERRIDES,
        ),
    )


def test_build_sim_config_settle_quiet_every_override():
    module = _load_example_module()
    cfg = module.build_sim_config(num_envs=2, settle_quiet_every=150)
    assert cfg.scene.settle_quiet_every == 150


def test_build_sim_config_accepts_device_override():
    module = _load_example_module()
    cfg = module.build_sim_config(num_envs=2, device="cpu")
    assert cfg.runtime.device == "cpu"
    assert cfg.controller.mode == "vic"
