"""Pytest configuration and shared FR3 coupled-scene helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
RANGES_FIXTURE = FIXTURES_DIR / "fruiting_system_ranges_straight_rod_test.json"

# Coupled cable scenes: default rod base near the FR3 at world origin (xy offset 0.5 m).
COUPLED_BASE_POS = (0.5, 0.5, 0.5)
NO_SELF_COLLISION_KW = {"enable_self_collisions": False}
COUPLED_SCENE_KW = {**NO_SELF_COLLISION_KW, "base_pos": COUPLED_BASE_POS}
DEFAULT_MJ_KW = {"disable_contacts": True}

FRAME_DT = 1.0 / 60.0
SUBSTEPS_PER_FRAME = 30
SUB_DT = FRAME_DT / SUBSTEPS_PER_FRAME
SIM_SUB_DT = SUB_DT  # example_coupled_fruiting default substep


def fr3_assets_available() -> bool:
    try:
        from apple_pick_sim.robot import fr3_robot

        return fr3_robot.fr3_assets_available()
    except Exception:
        return False


requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: long-horizon stability or optional benchmark-style tests",
    )


def build_coupled_fr3(cf, ranges, seed: int, **kwargs):
    """Build FR3 coupled scene with intra-chain self-collision disabled by default."""
    kwargs.setdefault("enable_self_collisions", False)
    return cf.build_coupled_fruiting_fr3(ranges, seed, **kwargs)


def build_vbd_only(cf, ranges, seed: int, **kwargs):
    kwargs.setdefault("enable_self_collisions", False)
    return cf.build_coupled_fruiting_fr3(ranges, seed, vbd_only=True, **kwargs)


def new_direct_controller(scene, fr3_robot):
    """Kinematic FR3 arm: direct ``joint_q`` writes (accurate for force tests)."""
    scene.robot_kinematic_mode = True
    ctrl = fr3_robot.Fr3EEDirectJointController(scene.robot_model, scene.tcp_body_index)
    ctrl.sync_target_from_state(scene.robot_state_0)
    return ctrl


def apply_direct_hold(scene, fr3_robot, ctrl, *, velocity=None) -> None:
    """One teleop frame with direct joints (zero velocity by default)."""
    scene.apply_fr3_ee_teleop_direct(
        FRAME_DT,
        ctrl,
        velocity=velocity if velocity is not None else fr3_robot.EEVelocity(),
    )


def run_coupled_substeps_direct_hold(
    scene,
    fr3_robot,
    n_substeps: int,
    *,
    sub_dt: float = SUB_DT,
    velocity=None,
    kinematic: bool = True,
) -> None:
    """Advance ``coupled_substep`` with periodic direct-joint hold (one frame / 30 substeps)."""
    if kinematic:
        scene.robot_kinematic_mode = True
        ctrl = new_direct_controller(scene, fr3_robot)
    else:
        ctrl = None
    for i in range(n_substeps):
        if kinematic and i % SUBSTEPS_PER_FRAME == 0:
            apply_direct_hold(scene, fr3_robot, ctrl, velocity=velocity)
        scene.coupled_substep(sub_dt)


def run_mujoco_substeps_direct_hold(
    scene,
    fr3_robot,
    n_substeps: int,
    *,
    sub_dt: float = SUB_DT,
    velocity=None,
) -> None:
    scene.robot_kinematic_mode = True
    ctrl = new_direct_controller(scene, fr3_robot)
    for i in range(n_substeps):
        if i % SUBSTEPS_PER_FRAME == 0:
            apply_direct_hold(scene, fr3_robot, ctrl, velocity=velocity)
        scene.mujoco_substep(sub_dt)
