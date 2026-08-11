"""FR3 asset paths and availability."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

TESTFR3_SCENE_USD = _REPO_ROOT / "assets" / "testfr3_resolved.usda"
_OMNI_FR3_ROOT = _REPO_ROOT / "assets" / "fr3" / "omniverse_fr3"
OMNIVERSE_FR3_USD = _OMNI_FR3_ROOT / "fr3.usd"
OMNIVERSE_FR3_SCHEMA = _OMNI_FR3_ROOT / "configuration" / "fr3_robot_schema.usd"

EE_MASS_KG = 1.5
EE_CYLINDER_RADIUS = 0.05
# 180 mm tool: 140 mm prior length + ~40 mm real-vs-sim tip gap (s02 FK calibration).
EE_CYLINDER_HALF_HEIGHT = 0.09
# TCP body +Z = tip-out (logged Franka / proxy). Geometry stays on ee −Z via fr3_joint8.
EE_TCP_ORIENT_WXYZ = (0.0, 1.0, 0.0, 0.0)  # RotX(180°)


def fr3_assets_available() -> bool:
    return (
        TESTFR3_SCENE_USD.is_file()
        and OMNIVERSE_FR3_USD.is_file()
        and OMNIVERSE_FR3_SCHEMA.is_file()
    )