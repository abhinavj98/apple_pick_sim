"""Tests for manifest sim-config serialization and replay parity warnings."""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    FruitingSystemConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system.params import PLACEHOLDER_EE_MASS_KG, GripperProxyConfig
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.system_id.manifest_sim_config import (
    sim_config_manifest_mismatches,
    sim_config_to_manifest_dict,
    warn_manifest_sim_config_mismatch,
)


def _sample_config(*, stem_apple_kd: float = 0.05) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2),
        scene=SceneSettleCollisionConfig(settle_substeps=5000),
        fruiting_system=FruitingSystemConfig(
            joint_angular_kd_overrides={
                "support": 1.0,
                "primary_spur": 1.0,
                "stem_apple": float(stem_apple_kd),
            },
            joint_linear_kd_overrides={
                "support": 2.0,
                "primary_spur": 2.0,
                "stem_apple": float(stem_apple_kd) * 2.0,
            },
            joint_angular_kp_overrides={"support": 2.0e4},
            joint_linear_kp_overrides={"support": 2.0e4},
        ),
        controller=ControllerConfig(
            mode="vic",
            linear_speed=1.0,
            angular_speed=1.0,
            ik_iterations=96,
            vic_gains=ImpedanceGains(
                linear_k=200.0,
                linear_d=10.0,
                angular_k=10.0,
                angular_d=1.0,
            ),
        ),
        robot=RobotConfig(
            fix_to_apple=True,
            gripper=GripperProxyConfig(mass=PLACEHOLDER_EE_MASS_KG),
        ),
    )


def test_sim_config_to_manifest_dict_includes_replay_relevant_fields():
    cfg = _sample_config(stem_apple_kd=1.0)
    recorded = sim_config_to_manifest_dict(
        cfg,
        applied_joint_kd_overrides={"stem_apple": 1.0},
        applied_joint_linear_kd_overrides={"stem_apple": 2.0},
        applied_joint_angular_kp_overrides={"support": 2.0e4},
        applied_joint_linear_kp_overrides={"support": 2.0e4},
    )

    assert recorded["settle_substeps"] == 5000
    assert recorded["joint_angular_kd_overrides"]["stem_apple"] == pytest.approx(1.0)
    assert recorded["joint_linear_kd_overrides"]["stem_apple"] == pytest.approx(2.0)
    assert recorded["joint_angular_kp_overrides"]["support"] == pytest.approx(2.0e4)
    assert recorded["joint_linear_kp_overrides"]["support"] == pytest.approx(2.0e4)
    assert recorded["joint_angular_kd_applied"]["stem_apple"] == pytest.approx(1.0)
    assert recorded["joint_linear_kd_applied"]["stem_apple"] == pytest.approx(2.0)
    assert recorded["joint_angular_kp_applied"]["support"] == pytest.approx(2.0e4)
    assert recorded["joint_linear_kp_applied"]["support"] == pytest.approx(2.0e4)
    assert recorded["joint_damping_ratio"] is None
    assert recorded["controller"]["mode"] == "vic"
    assert recorded["controller"]["vic_gains"]["linear_k"] == pytest.approx(200.0)
    assert recorded["robot"]["fix_to_apple"] is True
    assert recorded["robot"]["gripper_mass_kg"] == pytest.approx(PLACEHOLDER_EE_MASS_KG)
    assert "num_envs" not in recorded


def test_sim_config_to_manifest_dict_records_joint_damping_ratio():
    cfg = dataclasses.replace(
        _sample_config(),
        fruiting_system=dataclasses.replace(
            _sample_config().fruiting_system,
            joint_angular_kd_overrides={},
            joint_linear_kd_overrides={},
            joint_damping_ratio=10.0,
        ),
    )
    recorded = sim_config_to_manifest_dict(cfg)
    assert recorded["joint_damping_ratio"] == pytest.approx(10.0)
    assert recorded["joint_angular_kd_overrides"] == {}
    assert recorded["joint_linear_kd_overrides"] == {}


def test_sim_config_manifest_mismatches_detects_joint_damping_ratio_drift():
    recorded = sim_config_to_manifest_dict(
        dataclasses.replace(
            _sample_config(),
            fruiting_system=dataclasses.replace(
                _sample_config().fruiting_system,
                joint_angular_kd_overrides={},
                joint_linear_kd_overrides={},
                joint_damping_ratio=1.0,
            ),
        )
    )
    replay = dataclasses.replace(
        _sample_config(),
        fruiting_system=dataclasses.replace(
            _sample_config().fruiting_system,
            joint_angular_kd_overrides={},
            joint_linear_kd_overrides={},
            joint_damping_ratio=10.0,
        ),
    )
    mismatches = sim_config_manifest_mismatches(recorded, replay)
    assert any("joint_damping_ratio" in msg for msg in mismatches)

def test_sim_config_manifest_mismatches_detects_support_kp_drift():
    recorded = sim_config_to_manifest_dict(_sample_config())
    replay = dataclasses.replace(
        _sample_config(),
        fruiting_system=dataclasses.replace(
            _sample_config().fruiting_system,
            joint_angular_kp_overrides={"support": 1.0e4},
        ),
    )

    mismatches = sim_config_manifest_mismatches(recorded, replay)
    assert any("joint_angular_kp_overrides" in msg for msg in mismatches)


def test_sim_config_manifest_mismatches_detects_stem_apple_kd_drift():
    recorded = sim_config_to_manifest_dict(
        _sample_config(stem_apple_kd=1.0),
        applied_joint_kd_overrides={"stem_apple": 1.0},
    )
    replay = _sample_config(stem_apple_kd=0.05)

    mismatches = sim_config_manifest_mismatches(recorded, replay)
    assert any("stem_apple" in msg for msg in mismatches)


def test_sim_config_manifest_mismatches_empty_for_identical_config():
    cfg = _sample_config(stem_apple_kd=0.05)
    recorded = sim_config_to_manifest_dict(cfg, applied_joint_kd_overrides={"stem_apple": 0.05})
    assert sim_config_manifest_mismatches(recorded, cfg) == []


def test_sim_config_manifest_mismatches_empty_when_recorded_missing():
    replay = _sample_config()
    assert sim_config_manifest_mismatches(None, replay) == []
    assert sim_config_manifest_mismatches({}, replay) == []


def test_warn_manifest_sim_config_mismatch_emits_warnings():
    recorded = sim_config_to_manifest_dict(
        _sample_config(stem_apple_kd=1.0),
        applied_joint_kd_overrides={"stem_apple": 1.0},
    )
    replay = _sample_config(stem_apple_kd=0.05)
    captured: list[str] = []

    warn_manifest_sim_config_mismatch(
        {"collection": {"sim_config": recorded}},
        replay,
        warn=captured.append,
    )
    assert captured
    assert any("stem_apple" in msg for msg in captured)


def test_warn_manifest_sim_config_mismatch_silent_for_legacy_manifest():
    replay = _sample_config()
    captured: list[str] = []
    warn_manifest_sim_config_mismatch(
        {"collection": {}},
        replay,
        warn=captured.append,
    )
    assert captured == []
