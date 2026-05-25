"""Tests for default Warp device selection."""

from __future__ import annotations

import os

import pytest
import warp as wp

from apple_pick_sim.sim_device import default_sim_device, resolve_sim_device


def test_resolve_sim_device_explicit():
    assert resolve_sim_device("cpu") == "cpu"
    assert resolve_sim_device("cuda:0") == "cuda:0"


def test_resolve_sim_device_none_uses_default():
    assert resolve_sim_device(None) == default_sim_device()


def test_default_sim_device_matches_cuda_availability():
    expected = "cuda:0" if wp.is_cuda_available() else "cpu"
    assert default_sim_device() == expected


def test_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APPLE_PICK_SIM_DEVICE", "cpu")
    assert default_sim_device() == "cpu"
    monkeypatch.delenv("APPLE_PICK_SIM_DEVICE", raising=False)
    assert resolve_sim_device(None) == ("cuda:0" if wp.is_cuda_available() else "cpu")
