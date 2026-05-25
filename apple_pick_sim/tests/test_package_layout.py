"""Package layout contract after shim removal and examples relocation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_ROOT = _REPO_ROOT / "apple_pick_sim"


def test_fr3_robot_imports_from_robot_package():
    from apple_pick_sim.robot import fr3_robot

    assert callable(fr3_robot.build_fr3_robot_model_from_usd)
    assert callable(fr3_robot.fr3_assets_available)


def test_proxy_coupling_imports_from_coupled_fruiting():
    from apple_pick_sim.coupled_fruiting import proxy_coupling as pc

    assert hasattr(pc, "ProxyBodyRegistry")
    assert hasattr(pc, "launch_mirror_robot_to_proxy")


def test_no_top_level_fr3_robot_shim_file():
    assert not (_PKG_ROOT / "fr3_robot.py").is_file()


def test_no_top_level_proxy_coupling_file():
    assert not (_PKG_ROOT / "proxy_coupling.py").is_file()


def test_fr3_robot_spec_points_at_robot_subpackage():
    spec = importlib.util.find_spec("apple_pick_sim.robot.fr3_robot")
    assert spec is not None
    assert spec.origin is not None
    assert "robot" in spec.origin and "fr3_robot" in spec.origin


def test_examples_coupled_fruiting_importable():
    from apple_pick_sim.examples import example_coupled_fruiting as ex

    assert hasattr(ex, "_make_parser")
