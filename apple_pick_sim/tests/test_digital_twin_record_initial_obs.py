"""Record initial digital-twin observation JSON from a built simulation scene.

Run the full suite::

    uv run --env-file pytest.env python -m pytest apple_pick_sim/tests/test_digital_twin_record_initial_obs.py -q

Regenerate the committed reusable fixture (overwrite)::

    WRITE_DIGITAL_TWIN_FIXTURES=1 uv run --env-file pytest.env python -m pytest \\
      apple_pick_sim/tests/test_digital_twin_record_initial_obs.py::test_regenerate_straight_rod_initial_obs_fixture -q

Then point ``example_digital_twin.py`` at::

    apple_pick_sim/fixtures/digital_twin_obs_straight_rod_initial.json
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.digital_twin import (
    build_digital_twin_scene,
    fruiting_tree_fixed_joints,
    load_digital_twin_obs,
    params_from_ranges_median,
    record_obs_from_scene,
    save_digital_twin_obs,
)
from apple_pick_sim.fruiting_system import (
    fixed_joint_anchors_world,
    generate_coupled_cable_scene,
    load_ranges,
    parse_fixture_args,
)
from apple_pick_sim.tests.conftest import FIXTURES_DIR, NO_SELF_COLLISION_KW, RANGES_FIXTURE

STRAIGHT_ROD_FIXTURE = RANGES_FIXTURE
INITIAL_OBS_FIXTURE = FIXTURES_DIR / "digital_twin_obs_straight_rod_initial.json"


def _build_reference_scene(*, device: str = "cpu"):
    """Straight-rod median scene (free proxy) used as the obs recording source."""
    ranges = load_ranges(STRAIGHT_ROD_FIXTURE)
    args = parse_fixture_args(ranges)
    base_pos = args.fruiting_base_pos or (0.0, 0.2, 1.3)
    params = params_from_ranges_median(ranges)
    scene = generate_coupled_cable_scene(
        ranges,
        seed=0,
        params=params,
        base_pos=base_pos,
        device=device,
        enable_self_collisions=False,
    )
    return args, base_pos, scene


def _record_straight_rod_initial_obs():
    args, base_pos, scene = _build_reference_scene()
    robot_base = args.robot_base_pos
    if robot_base is None:
        raise RuntimeError("straight-rod fixture has no robot_base_pos for weld_direction")
    return record_obs_from_scene(
        scene,
        fruiting_base_pos=base_pos,
        robot_base_pos=robot_base,
    )


def test_record_initial_obs_round_trip(tmp_path: Path):
    """Scene → recorded obs → twin scene matches junction anchors (<1 mm)."""
    args, base_pos, scene = _build_reference_scene()
    robot_base = args.robot_base_pos
    assert robot_base is not None

    obs = record_obs_from_scene(scene, fruiting_base_pos=base_pos, robot_base_pos=robot_base)
    out = tmp_path / "initial_obs.json"
    save_digital_twin_obs(obs, out)
    loaded = load_digital_twin_obs(out)

    assert loaded.rod_radii == {
        "primary": pytest.approx(float(scene.params.primary.radius)),
        "secondary": pytest.approx(float(scene.params.secondary.radius)),
        "spur": pytest.approx(float(scene.params.spur.radius)),
        "stem": pytest.approx(float(scene.params.stem.radius)),
    }

    twin = build_digital_twin_scene(
        loaded,
        STRAIGHT_ROD_FIXTURE,
        device="cpu",
        fix_to_apple=False,
        **NO_SELF_COLLISION_KW,
    )

    joints = fruiting_tree_fixed_joints(scene)
    ref_parent, ref_child = fixed_joint_anchors_world(
        scene.model, scene.state_0.body_q, joints
    )
    twin_parent, twin_child = fixed_joint_anchors_world(
        twin.model, twin.state_0.body_q, fruiting_tree_fixed_joints(twin)
    )
    np.testing.assert_allclose(twin_parent, ref_parent, atol=1e-3)
    np.testing.assert_allclose(twin_child, ref_child, atol=1e-3)


def test_straight_rod_initial_obs_fixture_on_disk():
    """Committed initial obs fixture loads and round-trips through the twin builder."""
    if not INITIAL_OBS_FIXTURE.is_file():
        pytest.skip(f"missing {INITIAL_OBS_FIXTURE.name}; regenerate with WRITE_DIGITAL_TWIN_FIXTURES=1")

    loaded = load_digital_twin_obs(INITIAL_OBS_FIXTURE)
    twin = build_digital_twin_scene(
        loaded,
        STRAIGHT_ROD_FIXTURE,
        device="cpu",
        fix_to_apple=False,
        **NO_SELF_COLLISION_KW,
    )
    assert loaded.rod_radii is not None
    assert loaded.rod_radii["stem"] > 0.0
    assert len(loaded.junction_names) >= 1
    assert twin.params.primary is not None


@pytest.mark.skipif(
    os.environ.get("WRITE_DIGITAL_TWIN_FIXTURES") != "1",
    reason="set WRITE_DIGITAL_TWIN_FIXTURES=1 to overwrite fixture JSON",
)
def test_regenerate_straight_rod_initial_obs_fixture():
    """Write ``digital_twin_obs_straight_rod_initial.json`` for reuse with example_digital_twin."""
    obs = _record_straight_rod_initial_obs()
    save_digital_twin_obs(obs, INITIAL_OBS_FIXTURE)
    reloaded = load_digital_twin_obs(INITIAL_OBS_FIXTURE)
    assert reloaded.junction_names == obs.junction_names
    np.testing.assert_allclose(reloaded.woody_part_start_pos, obs.woody_part_start_pos)
