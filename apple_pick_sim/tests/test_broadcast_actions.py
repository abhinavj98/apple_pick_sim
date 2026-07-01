"""Tests for batched action broadcast helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import COUPLED_SCENE_KW
from apple_pick_sim.coupled_fruiting import build_batched_coupled_fruiting_placeholder
from apple_pick_sim.coupled_fruiting.broadcast_actions import broadcast_joint_q_from_world0
from apple_pick_sim.fruiting_system import load_ranges


def _two_env_placeholder_scene():
    ranges = load_ranges(
        "apple_pick_sim/fixtures/fruiting_system_ranges_example_variance_soft.json"
    )
    return build_batched_coupled_fruiting_placeholder(
        ranges,
        42,
        num_envs=2,
        device="cpu",
        **COUPLED_SCENE_KW,
    )


def test_broadcast_joint_q_warns_cpu_hot_path():
    """broadcast_joint_q_from_world0 must warn that it is not GPU-resident."""
    scene = _two_env_placeholder_scene()
    layout = scene.layout
    assert layout is not None

    with pytest.warns(UserWarning, match="not GPU-resident"):
        broadcast_joint_q_from_world0(scene, layout)
