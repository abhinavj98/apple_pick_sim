"""Tests for batched heterogeneous coupled sim config dataclasses (V.3.1 prep)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    DomainRandomizationConfig,
    RuntimeConfig,
    SettleDiagnosticsConfig,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)
from apple_pick_sim.coupled_fruiting.settle_ke_decay import DEFAULT_KE_SAMPLE_EVERY
from apple_pick_sim.fruiting_system import (
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_DEFAULT_ITERATIONS

_TESTS_DIR = Path(__file__).resolve().parent
RANGES_FIXTURE = _TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
SUB_DT = 1.0 / 1800.0


def test_runtime_substeps_per_step_at_60hz():
    runtime = RuntimeConfig(control_hz=60.0, sub_dt=SUB_DT)
    assert runtime.substeps_per_step == 30
    assert runtime.frame_dt == pytest.approx(30 * SUB_DT)


def test_runtime_substeps_per_step_at_example_default_30hz():
    runtime = RuntimeConfig(control_hz=30.0, sub_dt=SUB_DT)
    assert runtime.substeps_per_step == 60
    assert runtime.frame_dt == pytest.approx(60 * SUB_DT)


def test_defaults_preset_constructs_and_validates():
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    cfg.validate()
    assert cfg.runtime.num_envs == 4
    assert cfg.runtime.env_spacing == (2.0, 2.0, 2.0)
    assert cfg.runtime.control_hz == 30.0
    assert cfg.runtime.sub_dt == pytest.approx(SUB_DT)
    assert cfg.domain_randomization.ranges_path == default_ranges_fixture_path()
    assert cfg.robot.fix_to_apple is True
    assert cfg.robot.kind == "fr3"
    assert cfg.robot.per_env_ik is True
    assert cfg.robot.ik_bootstrap_iterations == IK_BOOTSTRAP_DEFAULT_ITERATIONS
    assert cfg.scene.settle_substeps == 5000
    assert cfg.fruiting_system.stem_coupling_gain == DEFAULT_STEM_COUPLING_GAIN
    assert cfg.controller.mode == "direct"
    assert cfg.controller.linear_speed == pytest.approx(0.1)
    assert cfg.controller.ik_iterations == 128
    assert cfg.settle_diagnostics is not None
    assert cfg.settle_diagnostics.enabled is True
    assert cfg.obs is None


def test_gym_defaults_preset():
    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=1)
    cfg.validate()
    assert cfg.runtime.num_envs == 1
    assert cfg.settle_diagnostics is None
    assert cfg.obs is not None


def test_test_minimal_preset():
    cfg = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2)
    cfg.validate()
    assert cfg.runtime.num_envs == 2
    assert cfg.runtime.device == "cpu"
    assert cfg.domain_randomization.ranges_path == RANGES_FIXTURE
    assert cfg.scene.settle_substeps == 50


def test_validate_raises_on_per_env_params_length_mismatch():
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=1, num_envs=1)
    cfg = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2)
    cfg = dataclasses.replace(
        cfg,
        domain_randomization=dataclasses.replace(
            cfg.domain_randomization,
            per_env_params=params,
        ),
    )
    with pytest.raises(ValueError, match="per_env_params"):
        cfg.validate()


def test_validate_rejects_non_fr3_robot_kind():
    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=1)
    cfg = dataclasses.replace(
        cfg,
        robot=dataclasses.replace(cfg.robot, kind="placeholder"),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="robot.kind must be 'fr3'"):
        cfg.validate()


def test_validate_raises_vic_with_vbd_only():
    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=1)
    cfg = dataclasses.replace(
        cfg,
        robot=dataclasses.replace(cfg.robot, step_mode="vbd_only"),
        controller=dataclasses.replace(cfg.controller, mode="vic"),
    )
    with pytest.raises(ValueError, match="vic"):
        cfg.validate()


def test_validate_raises_vbd_only_with_non_direct_controller():
    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=1)
    cfg = dataclasses.replace(
        cfg,
        robot=dataclasses.replace(cfg.robot, step_mode="vbd_only"),
        controller=dataclasses.replace(cfg.controller, mode="ee"),
    )
    with pytest.raises(ValueError, match="vbd_only"):
        cfg.validate()


def test_configs_are_frozen():
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.runtime.num_envs = 8  # type: ignore[misc]


def test_resolve_device_honors_env_override(monkeypatch):
    monkeypatch.setenv("APPLE_PICK_SIM_DEVICE", "cpu")
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    assert cfg.resolve_device() == "cpu"


def test_vic_gains_defaults_match_heterogeneous_example():
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    g = cfg.controller.vic_gains
    assert isinstance(g, ImpedanceGains)
    assert g.linear_k == pytest.approx(600.0)
    assert g.linear_d == pytest.approx(200.0)
    assert g.angular_k == pytest.approx(20.0)
    assert g.angular_d == pytest.approx(4.0)


def test_fruiting_joint_kd_overrides_defaults():
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    kd = cfg.fruiting_system.joint_angular_kd_overrides
    assert kd["support"] == pytest.approx(1.0)
    assert kd["primary_spur"] == pytest.approx(1.0)
    assert kd["stem_apple"] == pytest.approx(5e-2)


def test_stem_caps_defaults():
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    assert cfg.fruiting_system.stem_force_cap_N == DEFAULT_STEM_FORCE_CAP_N
    assert cfg.fruiting_system.stem_torque_cap_Nm == DEFAULT_STEM_TORQUE_CAP_NM


def test_settle_diagnostics_optional_fields():
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.defaults(),
        settle_diagnostics=SettleDiagnosticsConfig(),
    )
    diag = cfg.settle_diagnostics
    assert diag is not None
    assert diag.ke_sample_every == DEFAULT_KE_SAMPLE_EVERY


def test_validate_warns_when_gripper_weld_true_but_robot_weld_false():
    cfg = BatchedHeterogeneousCoupledSimConfig.defaults()
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig

    cfg = dataclasses.replace(
        cfg,
        robot=dataclasses.replace(
            cfg.robot,
            fix_to_apple=False,
            gripper=GripperProxyConfig(mass=1.0, fix_to_apple=True, robot_facing_weld=True),
        ),
    )
    with pytest.warns(UserWarning, match="fix_to_apple"):
        cfg.validate()


def test_inject_mode_validate_passes_with_matching_params():
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(ranges, topology_seed=3, num_envs=2)
    cfg = BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2)
    cfg = dataclasses.replace(
        cfg,
        domain_randomization=dataclasses.replace(
            cfg.domain_randomization,
            per_env_params=params,
        ),
    )
    cfg.validate()
