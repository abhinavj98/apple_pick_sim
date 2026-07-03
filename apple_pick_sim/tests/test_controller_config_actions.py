"""Tests for ControllerConfig action validation (V.3.1 step B)."""

from __future__ import annotations

import dataclasses

import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
)


def _require_torch():
    pytest.importorskip("torch")
    import torch

    return torch


def test_expected_action_shape():
    ctrl = ControllerConfig(action_dim=6)
    assert ctrl.expected_action_shape(4) == (4, 6)


def test_validate_actions_wrong_shape():
    torch = _require_torch()
    ctrl = ControllerConfig(action_dim=6)
    bad = torch.zeros(4, 7, dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        ctrl.validate_actions(bad, num_envs=4, device="cpu", robot_step_mode="coupled")


def test_validate_actions_wrong_device():
    torch = _require_torch()
    ctrl = ControllerConfig(action_dim=6)
    actions = torch.zeros(2, 6, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="device"):
        ctrl.validate_actions(actions, num_envs=2, device="cuda:0", robot_step_mode="coupled")


def test_validate_actions_broadcast_action_dim():
    torch = _require_torch()
    ctrl = ControllerConfig(action_dim=6)
    actions = torch.ones(6, dtype=torch.float32, device="cpu")
    out = ctrl.validate_actions(actions, num_envs=3, device="cpu", robot_step_mode="coupled")
    assert out.shape == (3, 6)
    assert out.is_contiguous()
    assert torch.allclose(out[0], actions)
    assert torch.allclose(out[1], actions)


def test_validate_actions_vbd_only_rejects():
    torch = _require_torch()
    ctrl = ControllerConfig(action_dim=6)
    actions = torch.zeros(2, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="vbd_only"):
        ctrl.validate_actions(actions, num_envs=2, device="cpu", robot_step_mode="vbd_only")


def test_validate_actions_action_dim_in_config_validate():
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2),
        controller=ControllerConfig(action_dim=0),
    )
    with pytest.raises(ValueError, match="action_dim"):
        cfg.validate()


def test_validate_requires_action_buffer_for_fr3_coupled():
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2),
        controller=ControllerConfig(allocate_action_buffer=False),
    )
    with pytest.raises(ValueError, match="allocate_action_buffer"):
        cfg.validate()
