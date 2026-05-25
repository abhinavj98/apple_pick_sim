"""FR3 asset paths and availability."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

TESTFR3_SCENE_USD = _REPO_ROOT / "assets" / "testfr3_resolved.usda"
_OMNI_FR3_ROOT = _REPO_ROOT / "assets" / "fr3" / "omniverse_fr3"
OMNIVERSE_FR3_USD = _OMNI_FR3_ROOT / "fr3.usd"
OMNIVERSE_FR3_SCHEMA = _OMNI_FR3_ROOT / "configuration" / "fr3_robot_schema.usd"

EE_MASS_KG = 1.5
EE_BOX_HALF_EXTENTS = (0.05, 0.05, 0.05)


def fr3_assets_available() -> bool:
    return (
        TESTFR3_SCENE_USD.is_file()
        and OMNIVERSE_FR3_USD.is_file()
        and OMNIVERSE_FR3_SCHEMA.is_file()
    )
