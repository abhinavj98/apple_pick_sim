"""Unit tests for EnvDisableController sticky soft-disable."""

from __future__ import annotations

import torch

from apple_pick_gym.batched_envs.env_disable_controller import EnvDisableController


def test_update_is_sticky_or():
    c = EnvDisableController(num_envs=3, device="cpu")
    c.update(torch.tensor([False, True, False]))
    c.update(torch.tensor([True, False, False]))
    assert c.disabled.tolist() == [True, True, False]


def test_apply_actions_zeros_disabled_rows_preserves_device_dtype():
    c = EnvDisableController(num_envs=2, device="cpu")
    c.update(torch.tensor([False, True]))
    actions = torch.ones(2, 6, dtype=torch.float32)
    out = c.apply_actions(actions)
    assert out.dtype == torch.float32
    assert out[0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert out[1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert actions[1, 0].item() == 1.0


def test_should_record_mask_is_not_disabled():
    c = EnvDisableController(num_envs=2, device="cpu")
    c.update(torch.tensor([True, False]))
    assert c.should_record_mask().tolist() == [False, True]


def test_initial_disabled_seeds_mask():
    init = torch.tensor([False, True, False])
    c = EnvDisableController(num_envs=3, device="cpu", initial_disabled=init)
    assert c.disabled.tolist() == [False, True, False]
    out = c.apply_actions(torch.ones(3, 6))
    assert out[1].abs().sum().item() == 0.0


def test_update_rejects_wrong_length():
    c = EnvDisableController(num_envs=2, device="cpu")
    try:
        c.update(torch.tensor([True, False, True]))
    except ValueError as exc:
        assert "length" in str(exc).lower() or "numel" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
