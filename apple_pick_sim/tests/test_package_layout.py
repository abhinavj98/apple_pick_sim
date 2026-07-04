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
    assert hasattr(pc, "launch_mirror_robot_to_proxy_offset")
    assert hasattr(pc.ProxyBodyRegistry, "from_repeated_robot")


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


def test_coupled_fruiting_public_runtime_api():
    from apple_pick_sim import coupled_fruiting as cf

    assert callable(cf.build_batched_heterogeneous_scene)
    assert callable(cf.BatchedHeterogeneousCoupledSim)
    assert callable(cf.settle_vbd_substeps)


def test_fruiting_system_coupled_api_not_on_scene_module():
    """Coupled cable API lives in ``fruiting_system.coupled``, not ``scene``."""
    import apple_pick_sim.fruiting_system.scene as scene_mod

    assert not hasattr(scene_mod, "generate_coupled_cable_scene")
    assert not hasattr(scene_mod, "geometry_fingerprint_coupled")


def test_sim_device_module_exports():
    from apple_pick_sim import sim_device as sd

    assert callable(sd.default_sim_device)
    assert callable(sd.resolve_sim_device)


def test_gym_package_separate_from_sim():
    spec = importlib.util.find_spec("apple_pick_gym")
    assert spec is not None
    spec_sim = importlib.util.find_spec("apple_pick_sim")
    assert spec_sim is not None
    assert spec.origin != spec_sim.origin
