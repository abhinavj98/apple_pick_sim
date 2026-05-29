"""Tests for MuJoCo CPU vs Warp device selection."""

from __future__ import annotations

import pytest

from apple_pick_sim.sim_mujoco_device import resolve_mujoco_use_cpu


def test_resolve_mujoco_use_cpu_defaults():
    assert resolve_mujoco_use_cpu("cpu") is True
    assert resolve_mujoco_use_cpu("cuda:0") is False


def test_resolve_mujoco_use_cpu_explicit_override():
    assert resolve_mujoco_use_cpu("cuda:0", True) is True
    with pytest.raises(ValueError, match="CUDA"):
        resolve_mujoco_use_cpu("cpu", False)


def test_resolve_mujoco_use_cpu_none_follows_warp_device():
    assert resolve_mujoco_use_cpu("cpu", None) is True
    assert resolve_mujoco_use_cpu("cuda:0", None) is False
