"""Heterogeneous batched coupled fruiting: per-env params, uniform topology, vectorized VBD."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import COUPLED_SCENE_KW, RANGES_FIXTURE, SUB_DT, requires_fr3
from apple_pick_sim.fruiting_system import (
    GripperProxyConfig,
    load_ranges,
    sample_heterogeneous_params_list,
    sample_params,
)
from apple_pick_sim.fruiting_system.params import _fix_topology
from apple_pick_sim.coupled_fruiting import (
    build_heterogeneous_coupled_fruiting_fr3,
    seed_fix_to_apple_from_settled,
    settle_vbd_substeps,
)
from apple_pick_sim.coupled_fruiting.batched_build import build_heterogeneous_coupled_cable_scene
from apple_pick_sim.robot import fr3_robot

_NUM_ENVS = 2
_SETTLE_SUBSTEPS = 50


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


def _gripper_free() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        fix_to_apple=False,
        robot_facing_weld=False,
    )


def _gripper_welded() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
        fix_to_apple=True,
        robot_facing_weld=True,
    )


def test_fix_topology_preserves_continuous_params(ranges):
    topo = sample_params(ranges, seed=10)
    varied = sample_params(ranges, seed=11)
    fixed = _fix_topology(varied, topo)
    assert fixed.primary is not None and topo.primary is not None
    assert fixed.primary.num_segments == topo.primary.num_segments
    assert fixed.primary.bend_stiffness == varied.primary.bend_stiffness
    assert fixed.primary.direction == varied.primary.direction


def test_sample_heterogeneous_params_list_length(ranges):
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=5, num_envs=4)
    assert len(params_list) == 4
    segs = [p.primary.num_segments for p in params_list if p.primary is not None]
    assert len(set(segs)) == 1


def test_topology_uniform_across_envs(ranges):
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=7, num_envs=_NUM_ENVS)
    cable, _offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    starts = cable.model.body_world_start.numpy()
    assert len(starts) >= _NUM_ENVS + 1
    gap0 = int(starts[1] - starts[0])
    for w in range(_NUM_ENVS):
        assert int(starts[w + 1] - starts[w]) == gap0


def test_stiffness_differs_per_env(ranges):
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=8, num_envs=_NUM_ENVS)
    cable, _offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    jws = cable.model.joint_world_start.numpy()
    dof_per = int(jws[1] - jws[0])
    ke = cable.model.joint_target_ke.numpy()
    block0 = ke[:dof_per]
    block1 = ke[dof_per : 2 * dof_per]
    assert not np.allclose(block0, block1), "expected different stiffness per env"


def test_vbd_substep_does_not_crash(ranges):
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=9, num_envs=_NUM_ENVS)
    cable, _offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    from apple_pick_sim.coupled_fruiting import CoupledFruitingScene

    vbd_scene = CoupledFruitingScene(cable=cable, cable_collision_pipeline=None, vbd_only=True)
    vbd_scene.vbd_substep(SUB_DT)
    wp.synchronize()


def test_per_world_proxy_offsets_differ(ranges):
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=12, num_envs=_NUM_ENVS)
    _cable, offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_welded(),
    )
    assert offsets[0] is not None and offsets[1] is not None
    # Different sampled geometry should usually yield different grasp offsets
    if params_list[0].apple_radius != params_list[1].apple_radius or (
        params_list[0].stem
        and params_list[1].stem
        and params_list[0].stem.direction != params_list[1].stem.direction
    ):
        assert offsets[0] != offsets[1]


def _make_hetero_settle_then_weld(ranges, seed: int, *, settle_substeps: int = _SETTLE_SUBSTEPS):
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=seed, num_envs=_NUM_ENVS
    )
    build_kw = dict(
        ranges=ranges,
        params_list=params_list,
        device="cpu",
        env_spacing=(2.5, 2.5, 0.0),
        **COUPLED_SCENE_KW,
    )
    settled = build_heterogeneous_coupled_fruiting_fr3(
        vbd_only=True,
        gripper_proxy=_gripper_free(),
        **build_kw,
    )
    settle_vbd_substeps(settled, substeps=settle_substeps, dt=SUB_DT)
    welded = build_heterogeneous_coupled_fruiting_fr3(
        gripper_proxy=_gripper_welded(),
        skip_ik_bootstrap=True,
        defer_template_robot_bootstrap=True,
        **build_kw,
    )
    seed_fix_to_apple_from_settled(
        welded_scene=welded,
        settled_scene=settled,
        quiet_apple_proxy=True,
        per_env_ik=True,
        per_world_proxy_offsets=welded.per_world_proxy_offsets,
    )
    return welded, settled, params_list


@requires_fr3
@pytest.mark.slow
def test_per_env_ik_produces_different_joint_q(ranges):
    welded, _settled, _params = _make_hetero_settle_then_weld(ranges, seed=20)
    layout = welded.layout
    assert layout is not None
    jcs = welded.robot_model.joint_coord_world_start.numpy()
    coord_per = int(jcs[1] - jcs[0])
    jq = welded.robot_model.joint_q.numpy()
    row0 = jq[:coord_per]
    row1 = jq[coord_per : 2 * coord_per]
    assert not np.allclose(row0, row1, atol=1e-3)


SOFT_VARIANCE_RANGES = Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_example_variance_soft.json"


@requires_fr3
@pytest.mark.slow
def test_heterogeneous_batched_teleop_reduced_speed_step_converges():
    """One -Z teleop step at example keyboard speed (0.2 m/s @ 30 Hz) stays within IK tol."""
    soft_ranges = load_ranges(SOFT_VARIANCE_RANGES)
    welded, _settled, _params = _make_hetero_settle_then_weld(
        soft_ranges, seed=54, settle_substeps=200
    )
    ik_kw = fr3_robot.batched_ik_teleop_kwargs(welded)
    assert ik_kw
    ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
        welded.robot_model,
        linear_speed=0.2,
        angular_speed=1.0,
        ik_iterations=128,
        **ik_kw,
    )
    ctrl.sync_target_from_state(welded.robot_state_0)
    frame_dt = 1.0 / 30.0
    ctrl.run_ik_teleop_frame(
        frame_dt,
        welded.robot_state_0,
        velocity=fr3_robot.EEVelocity(linear=(0.0, 0.0, -0.2)),
    )
    for w, (pos_err, rot_err) in enumerate(
        ctrl.measure_ik_target_error_per_world(welded.robot_state_0)
    ):
        assert pos_err < fr3_robot.IK_TELEOP_POS_TOL_M, f"world {w} pos_err={pos_err}"
        assert rot_err < fr3_robot.IK_TELEOP_ROT_TOL_RAD, f"world {w} rot_err={rot_err}"


@requires_fr3
def test_robot_base_from_proxy_raises(ranges):
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=3, num_envs=_NUM_ENVS
    )
    with pytest.raises(ValueError, match="robot_base_from_proxy"):
        build_heterogeneous_coupled_fruiting_fr3(
            ranges,
            params_list,
            device="cpu",
            env_spacing=(2.5, 2.5, 0.0),
            robot_base_from_proxy=True,
            gripper_proxy=_gripper_free(),
            **COUPLED_SCENE_KW,
        )
