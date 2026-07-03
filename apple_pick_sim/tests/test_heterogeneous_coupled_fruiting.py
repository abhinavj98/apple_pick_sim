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
    set_fruiting_joint_angular_kd_batched,
    set_fruiting_joint_angular_kp_batched,
)
from apple_pick_sim.fruiting_system.build import FRUITING_VBD_RIGID_JOINT_ANGULAR_KD
from apple_pick_sim.fruiting_system.params import _fix_topology
from apple_pick_sim.coupled_fruiting import (
    build_heterogeneous_coupled_fruiting_fr3,
    seed_fix_to_apple_from_settled,
    quiet_all_cable_bodies,
    settle_vbd_substeps,
)
from apple_pick_sim.coupled_fruiting.batched_build import build_heterogeneous_coupled_cable_scene
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_POS_TOL_M

_NUM_ENVS = 2
_SETTLE_SUBSTEPS = 50


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


def _gripper_free() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=False,
        robot_facing_weld=False,
    )


def _gripper_welded() -> GripperProxyConfig:
    return GripperProxyConfig(
        mass=fr3_robot.EE_MASS_KG,
        fix_to_apple=True,
        robot_facing_weld=True,
    )


def _build_batched_cable_for_joint_kd(ranges):
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=3, num_envs=_NUM_ENVS
    )
    cable, _offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    return cable


def _joints_per_world(cable) -> int:
    jws = cable.model.joint_world_start.numpy()
    return int(jws[1] - jws[0])


def _angular_kd_at_joint(solver, global_joint_index: int) -> float:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd = solver.joint_penalty_kd.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(kd[c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR])


def _template_joint_by_label(fruiting_fixed_joints, label_substr: str) -> int:
    matches = [j for j, lab in fruiting_fixed_joints if label_substr in lab]
    assert len(matches) == 1, f"expected one joint for {label_substr!r}, got {matches}"
    return matches[0]


def _reference_batched_joint_angular_kd_numpy(
    solver,
    matched_by_key: dict[str, list[int]],
    label_kd: dict[str, float],
    *,
    num_envs: int,
    joints_per_world: int,
) -> np.ndarray:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    kd_np = solver.joint_penalty_kd.numpy().copy()
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR
    for w in range(num_envs):
        base = w * joints_per_world
        for key, template_indices in matched_by_key.items():
            kd_val = float(label_kd[key])
            for template_joint in template_indices:
                global_joint = base + int(template_joint)
                c0 = int(jc_start[global_joint])
                kd_np[c0 + ang_slot] = kd_val
    return kd_np


def test_set_fruiting_joint_angular_kd_batched_patches_all_envs(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable.model.world_count)
    joints_per_world = _joints_per_world(cable)
    j_primary = _template_joint_by_label(cable.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _template_joint_by_label(cable.fruiting_fixed_joints, "stem_apple")

    set_fruiting_joint_angular_kd_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        {"primary_secondary": 2.5, "stem_apple": 0.25},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    for w in range(num_envs):
        base = w * joints_per_world
        assert _angular_kd_at_joint(cable.solver, base + j_primary) == pytest.approx(2.5)
        assert _angular_kd_at_joint(cable.solver, base + j_stem_apple) == pytest.approx(0.25)


def test_set_fruiting_joint_angular_kd_batched_leaves_unmatched_joints_at_default(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable.model.world_count)
    joints_per_world = _joints_per_world(cable)
    j_spur_stem = _template_joint_by_label(cable.fruiting_fixed_joints, "spur_stem")

    set_fruiting_joint_angular_kd_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        {"primary_secondary": 3.0},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    for w in range(num_envs):
        global_joint = w * joints_per_world + j_spur_stem
        assert _angular_kd_at_joint(cable.solver, global_joint) == pytest.approx(
            FRUITING_VBD_RIGID_JOINT_ANGULAR_KD
        )


def test_set_fruiting_joint_angular_kd_batched_raises_on_unmatched_key(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    with pytest.raises(ValueError, match="nonexistent_key_xyz"):
        set_fruiting_joint_angular_kd_batched(
            cable.solver,
            cable.fruiting_fixed_joints,
            {"nonexistent_key_xyz": 1.0},
            num_envs=int(cable.model.world_count),
            joints_per_world=_joints_per_world(cable),
        )


def test_set_fruiting_joint_angular_kd_batched_raises_on_ambiguous_match(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    with pytest.raises(ValueError, match="ambiguous"):
        set_fruiting_joint_angular_kd_batched(
            cable.solver,
            cable.fruiting_fixed_joints,
            {"apple": 0.5, "stem_apple": 0.25},
            num_envs=int(cable.model.world_count),
            joints_per_world=_joints_per_world(cable),
        )


def test_set_fruiting_joint_angular_kd_batched_raises_on_negative_kd(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    with pytest.raises(ValueError, match="negative"):
        set_fruiting_joint_angular_kd_batched(
            cable.solver,
            cable.fruiting_fixed_joints,
            {"stem_apple": -0.1},
            num_envs=int(cable.model.world_count),
            joints_per_world=_joints_per_world(cable),
        )


def test_set_fruiting_joint_angular_kd_batched_matches_python_loop_reference(ranges):
    cable_a = _build_batched_cable_for_joint_kd(ranges)
    cable_b = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable_a.model.world_count)
    joints_per_world = _joints_per_world(cable_a)
    label_kd = {"primary_secondary": 2.0, "stem_apple": 0.2}
    template_matched = {
        "primary_secondary": [
            _template_joint_by_label(cable_a.fruiting_fixed_joints, "primary_secondary")
        ],
        "stem_apple": [
            _template_joint_by_label(cable_a.fruiting_fixed_joints, "stem_apple")
        ],
    }

    set_fruiting_joint_angular_kd_batched(
        cable_a.solver,
        cable_a.fruiting_fixed_joints,
        label_kd,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    expected = _reference_batched_joint_angular_kd_numpy(
        cable_b.solver,
        template_matched,
        label_kd,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    actual = cable_a.solver.joint_penalty_kd.numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def _run_batched_vbd_substeps(cable, *, num_substeps: int, sim_dt: float) -> None:
    from apple_pick_sim.coupled_fruiting import CoupledFruitingScene

    scene = CoupledFruitingScene(cable=cable, cable_collision_pipeline=None, vbd_only=True)
    for _ in range(num_substeps):
        scene.vbd_substep(sim_dt)
    wp.synchronize()


def test_set_fruiting_joint_angular_kd_batched_persists_through_solver_step(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable.model.world_count)
    joints_per_world = _joints_per_world(cable)
    j_stem_apple = _template_joint_by_label(cable.fruiting_fixed_joints, "stem_apple")

    set_fruiting_joint_angular_kd_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        {"stem_apple": 2.5},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    _run_batched_vbd_substeps(cable, num_substeps=8, sim_dt=SUB_DT)

    for w in range(num_envs):
        global_joint = w * joints_per_world + j_stem_apple
        assert _angular_kd_at_joint(cable.solver, global_joint) == pytest.approx(2.5)


def test_set_fruiting_joint_angular_kd_batched_changes_trajectory_after_steps(ranges):
    sim_dt = SUB_DT
    substeps = 120

    cable_default = _build_batched_cable_for_joint_kd(ranges)
    _run_batched_vbd_substeps(cable_default, num_substeps=substeps, sim_dt=sim_dt)
    q_default = cable_default.state_0.body_q.numpy().copy()

    cable_patched = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable_patched.model.world_count)
    joints_per_world = _joints_per_world(cable_patched)
    set_fruiting_joint_angular_kd_batched(
        cable_patched.solver,
        cable_patched.fruiting_fixed_joints,
        {"stem_apple": 50.0},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    _run_batched_vbd_substeps(cable_patched, num_substeps=substeps, sim_dt=sim_dt)
    q_patched = cable_patched.state_0.body_q.numpy().copy()

    assert not np.allclose(q_default, q_patched, rtol=0.0, atol=1.0e-4), (
        "batched patched stem_apple angular kd should change integrated trajectory"
    )


def _angular_kp_triple_at_joint(solver, global_joint_index: int) -> tuple[float, float, float]:
    import newton

    jc_start = solver.joint_constraint_start.numpy()
    k = solver.joint_penalty_k.numpy()
    k_min = solver.joint_penalty_k_min.numpy()
    k_max = solver.joint_penalty_k_max.numpy()
    c0 = int(jc_start[global_joint_index])
    slot = c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR
    return float(k[slot]), float(k_min[slot]), float(k_max[slot])


def _reference_batched_joint_angular_kp_numpy(
    solver,
    matched_by_key: dict[str, list[int]],
    label_kp: dict[str, float],
    *,
    num_envs: int,
    joints_per_world: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import newton

    from apple_pick_sim.fruiting_system.build import _patch_angular_k_constraint_slot

    jc_start = solver.joint_constraint_start.numpy()
    k_np = solver.joint_penalty_k.numpy().copy()
    k_min_np = solver.joint_penalty_k_min.numpy().copy()
    k_max_np = solver.joint_penalty_k_max.numpy().copy()
    ang_slot = newton.solvers.SolverVBD.JointSlot.ANGULAR
    for w in range(num_envs):
        base = w * joints_per_world
        for key, template_indices in matched_by_key.items():
            kp_val = float(label_kp[key])
            for template_joint in template_indices:
                global_joint = base + int(template_joint)
                c0 = int(jc_start[global_joint])
                _patch_angular_k_constraint_slot(
                    k_np, k_min_np, k_max_np, c0 + ang_slot, kp_val
                )
    return k_np, k_min_np, k_max_np


def test_set_fruiting_joint_angular_kp_batched_patches_all_envs(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable.model.world_count)
    joints_per_world = _joints_per_world(cable)
    j_primary = _template_joint_by_label(cable.fruiting_fixed_joints, "primary_secondary")
    j_stem_apple = _template_joint_by_label(cable.fruiting_fixed_joints, "stem_apple")

    set_fruiting_joint_angular_kp_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        {"primary_secondary": 2.0e5, "stem_apple": 5.0e4},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    for w in range(num_envs):
        base = w * joints_per_world
        k_primary, _, kmax_primary = _angular_kp_triple_at_joint(
            cable.solver, base + j_primary
        )
        k_stem, _, kmax_stem = _angular_kp_triple_at_joint(
            cable.solver, base + j_stem_apple
        )
        assert k_primary == pytest.approx(2.0e5)
        assert k_stem == pytest.approx(5.0e4)
        assert kmax_primary >= 2.0e5
        assert kmax_stem >= 5.0e4


def test_set_fruiting_joint_angular_kp_batched_leaves_unmatched_joints_at_default(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable.model.world_count)
    joints_per_world = _joints_per_world(cable)
    j_spur_stem = _template_joint_by_label(cable.fruiting_fixed_joints, "spur_stem")
    default_k, _, _ = _angular_kp_triple_at_joint(
        cable.solver, j_spur_stem
    )

    set_fruiting_joint_angular_kp_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        {"primary_secondary": 2.0e5},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    for w in range(num_envs):
        global_joint = w * joints_per_world + j_spur_stem
        k_spur, _, _ = _angular_kp_triple_at_joint(cable.solver, global_joint)
        assert k_spur == pytest.approx(default_k)


def test_set_fruiting_joint_angular_kp_batched_raises_on_unmatched_key(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    with pytest.raises(ValueError, match="nonexistent_key_xyz"):
        set_fruiting_joint_angular_kp_batched(
            cable.solver,
            cable.fruiting_fixed_joints,
            {"nonexistent_key_xyz": 1.0e5},
            num_envs=int(cable.model.world_count),
            joints_per_world=_joints_per_world(cable),
        )


def test_set_fruiting_joint_angular_kp_batched_raises_on_ambiguous_match(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    with pytest.raises(ValueError, match="ambiguous"):
        set_fruiting_joint_angular_kp_batched(
            cable.solver,
            cable.fruiting_fixed_joints,
            {"apple": 1.0e5, "stem_apple": 5.0e4},
            num_envs=int(cable.model.world_count),
            joints_per_world=_joints_per_world(cable),
        )


def test_set_fruiting_joint_angular_kp_batched_raises_on_negative_kp(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    with pytest.raises(ValueError, match="negative"):
        set_fruiting_joint_angular_kp_batched(
            cable.solver,
            cable.fruiting_fixed_joints,
            {"stem_apple": -1.0},
            num_envs=int(cable.model.world_count),
            joints_per_world=_joints_per_world(cable),
        )


def test_set_fruiting_joint_angular_kp_batched_matches_python_loop_reference(ranges):
    cable_a = _build_batched_cable_for_joint_kd(ranges)
    cable_b = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable_a.model.world_count)
    joints_per_world = _joints_per_world(cable_a)
    label_kp = {"primary_secondary": 2.0e5, "stem_apple": 5.0e4}
    template_matched = {
        "primary_secondary": [
            _template_joint_by_label(cable_a.fruiting_fixed_joints, "primary_secondary")
        ],
        "stem_apple": [
            _template_joint_by_label(cable_a.fruiting_fixed_joints, "stem_apple")
        ],
    }

    set_fruiting_joint_angular_kp_batched(
        cable_a.solver,
        cable_a.fruiting_fixed_joints,
        label_kp,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    expected_k, expected_k_min, expected_k_max = _reference_batched_joint_angular_kp_numpy(
        cable_b.solver,
        template_matched,
        label_kp,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    np.testing.assert_allclose(cable_a.solver.joint_penalty_k.numpy(), expected_k)
    np.testing.assert_allclose(cable_a.solver.joint_penalty_k_min.numpy(), expected_k_min)
    np.testing.assert_allclose(cable_a.solver.joint_penalty_k_max.numpy(), expected_k_max)


def test_set_fruiting_joint_angular_kp_batched_persists_through_solver_step(ranges):
    cable = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable.model.world_count)
    joints_per_world = _joints_per_world(cable)
    j_stem_apple = _template_joint_by_label(cable.fruiting_fixed_joints, "stem_apple")
    kp_set = 2.5e5
    substeps = 8

    set_fruiting_joint_angular_kp_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        {"stem_apple": kp_set},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    _run_batched_vbd_substeps(cable, num_substeps=substeps, sim_dt=SUB_DT)

    expected_k = kp_set * (cable.solver.rigid_avbd_gamma**substeps)
    for w in range(num_envs):
        global_joint = w * joints_per_world + j_stem_apple
        k, _, k_max = _angular_kp_triple_at_joint(cable.solver, global_joint)
        assert k == pytest.approx(expected_k)
        assert k_max >= kp_set


def test_set_fruiting_joint_angular_kp_batched_changes_trajectory_after_steps(ranges):
    sim_dt = SUB_DT
    substeps = 120

    cable_default = _build_batched_cable_for_joint_kd(ranges)
    _run_batched_vbd_substeps(cable_default, num_substeps=substeps, sim_dt=sim_dt)
    q_default = cable_default.state_0.body_q.numpy().copy()

    cable_patched = _build_batched_cable_for_joint_kd(ranges)
    num_envs = int(cable_patched.model.world_count)
    joints_per_world = _joints_per_world(cable_patched)
    set_fruiting_joint_angular_kp_batched(
        cable_patched.solver,
        cable_patched.fruiting_fixed_joints,
        {"stem_apple": 1.0e4},
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    _run_batched_vbd_substeps(cable_patched, num_substeps=substeps, sim_dt=sim_dt)
    q_patched = cable_patched.state_0.body_q.numpy().copy()

    assert not np.allclose(q_default, q_patched, rtol=0.0, atol=1.0e-4), (
        "batched patched stem_apple angular kp should change integrated trajectory"
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


def test_heterogeneous_build_syncs_body_q_prev_on_all_worlds(ranges):
    """Batched eval_fk must not leave stale body_q_prev on spur/stem/apple bodies."""
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=9, num_envs=_NUM_ENVS
    )
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
    bodies_per_world = int(starts[1] - starts[0])
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    bqp = cable.solver.body_q_prev.numpy().reshape(-1, 7)
    rel_branch = [*cable.spur_bodies, *cable.stem_bodies, int(cable.apple_body)]
    for w in range(_NUM_ENVS):
        offset = w * bodies_per_world
        for rel in rel_branch:
            idx = offset + int(rel)
            np.testing.assert_allclose(
                bqp[idx, :3],
                bq[idx, :3],
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"world {w} body {rel} body_q_prev stale after build",
            )


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
    quiet_all_cable_bodies(settled.cable)
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
def test_batched_ik_bootstrap_aligns_all_proxy_targets(ranges):
    """After per-env batched IK bootstrap, each TCP is within tolerance of its proxy."""
    welded, _settled, _params = _make_hetero_settle_then_weld(ranges, seed=54)
    layout = welded.layout
    assert layout is not None

    cable = welded.cable
    bq = cable.state_0.body_q.numpy().reshape(-1, 7)
    for w in range(layout.num_envs):
        proxy_idx = layout.proxy_body_indices[w]
        tcp_idx = layout.tcp_body_indices[w]
        proxy_pos = bq[proxy_idx, :3]
        tcp_pos = welded.robot_state_0.body_q.numpy().reshape(-1, 7)[tcp_idx, :3]
        pos_err = float(np.linalg.norm(tcp_pos - proxy_pos))
        assert pos_err < IK_BOOTSTRAP_POS_TOL_M, f"world {w} pos_err={pos_err}"


@requires_fr3
@pytest.mark.slow
def test_per_env_ik_produces_different_joint_q(ranges):
    welded, _settled, _params = _make_hetero_settle_then_weld(ranges, seed=54)
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


def test_batched_heterogeneous_example_timing_matches_frame_dt():
    """Physics substep count must advance exactly one frame_dt per simulate()."""
    import argparse
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        ExampleBatchedHeterogeneousCoupledFruiting,
    )

    viewer = MagicMock()
    args = argparse.Namespace(
        hz=30.0,
        json=None,
        seed=42,
        num_envs=2,
        env_spacing=[2.0, 2.0, 2.0],
        enable_self_collision=False,
        robot="placeholder",
        controller="direct",
        fr3_keyboard=False,
        fix_to_apple=False,
        settle_substeps=0,
        settle_gravity_ramp=True,
        inspect_settle=False,
        settle_report_brief=False,
        settle_max_speed=0.05,
        scripted_ee_vel=[0.05, 0.0, 0.0],
        demo_per_env_actions=False,
        status_every=0,
        print_robot_state=False,
        noisy_action=False,
        noisy_action_std=0.02,
        tcp_force_arrow=False,
        tcp_force_scale=0.02,
        tcp_force_arrow_gain=1.0,
        tcp_force_min_length=0.08,
        tcp_force_max_length=1.5,
        mark_endpoints=False,
        mujoco_viewer=False,
        vic_linear_k=600.0,
        vic_linear_d=200.0,
        vic_angular_k=20.0,
        vic_angular_d=4.0,
        device="cpu",
        only_vbd=False,
        only_mjc=False,
    )

    with pytest.warns(UserWarning, match="GPU parallelism is not fully utilized"):
        example = ExampleBatchedHeterogeneousCoupledFruiting(viewer, args)

    assert example.sim_dt * example.sim_substeps == pytest.approx(example.frame_dt)
    assert example.sim_dt == pytest.approx(1.0 / 1800.0)


def test_batched_heterogeneous_only_vbd_parser_flag():
    import sys
    from pathlib import Path

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        _make_parser,
        _resolve_step_mode,
    )

    args = _make_parser().parse_args(["--only-vbd"])
    assert _resolve_step_mode(args) == "vbd"
    assert args.only_vbd is True
    assert args.only_mjc is False


def test_batched_heterogeneous_only_vbd_and_only_mjc_mutually_exclusive():
    import sys
    from pathlib import Path

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import _resolve_step_mode  # noqa: E402

    import argparse

    with pytest.raises(SystemExit, match="mutually exclusive"):
        _resolve_step_mode(argparse.Namespace(only_vbd=True, only_mjc=True))


def test_print_per_env_params_includes_all_rod_stiffnesses(capsys):
    import sys
    from pathlib import Path

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import _print_per_env_params  # noqa: E402

    ranges = load_ranges(RANGES_FIXTURE)
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=0, num_envs=2)
    _print_per_env_params(params_list)

    out = capsys.readouterr().out
    for seg in ("primary", "spur", "stem"):
        assert f"{seg}: E=" in out
        assert f"{seg}: " in out and "k_bend=" in out and "k_stretch=" in out
    for p in params_list:
        for seg_name in ("primary", "spur", "stem"):
            rod = getattr(p, seg_name)
            assert rod is not None
            assert f"{rod.bend_stiffness:.4g}" in out
            assert f"{rod.stretch_stiffness:.4g}" in out


def test_batched_heterogeneous_only_mjc_rejected():
    import sys
    from pathlib import Path

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import _resolve_step_mode  # noqa: E402

    import argparse

    with pytest.raises(SystemExit, match="--only-mjc"):
        _resolve_step_mode(argparse.Namespace(only_vbd=False, only_mjc=True))


def test_batched_heterogeneous_only_vbd_builds_cable_only_scene():
    import argparse
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        ExampleBatchedHeterogeneousCoupledFruiting,
    )

    viewer = MagicMock()
    args = argparse.Namespace(
        hz=30.0,
        json=None,
        seed=42,
        num_envs=2,
        env_spacing=[2.0, 2.0, 2.0],
        enable_self_collision=False,
        robot="placeholder",
        controller="direct",
        fr3_keyboard=False,
        fix_to_apple=True,
        settle_substeps=0,
        inspect_settle=False,
        settle_report_brief=False,
        settle_max_speed=0.05,
        scripted_ee_vel=[0.05, 0.0, 0.0],
        demo_per_env_actions=False,
        status_every=0,
        print_robot_state=False,
        noisy_action=False,
        noisy_action_std=0.02,
        tcp_force_arrow=False,
        tcp_force_scale=0.02,
        tcp_force_arrow_gain=1.0,
        tcp_force_min_length=0.08,
        tcp_force_max_length=1.5,
        mark_endpoints=False,
        mujoco_viewer=False,
        vic_linear_k=600.0,
        vic_linear_d=200.0,
        vic_angular_k=20.0,
        vic_angular_d=4.0,
        device="cpu",
        only_vbd=True,
        only_mjc=False,
    )

    with pytest.warns(UserWarning, match="GPU parallelism is not fully utilized"):
        example = ExampleBatchedHeterogeneousCoupledFruiting(viewer, args)

    assert example.scene.vbd_only
    assert example.scene.robot_model is None
    assert example._ee_ctrl is None
    assert example.layout is not None
    assert example.layout.num_envs == 2

    vbd_calls = 0
    coupled_calls = 0
    original_vbd = example.scene.vbd_substep
    original_coupled = example.scene.coupled_substep

    def _track_vbd(dt, **kw):
        nonlocal vbd_calls
        vbd_calls += 1
        return original_vbd(dt, **kw)

    def _track_coupled(dt, **kw):
        nonlocal coupled_calls
        coupled_calls += 1
        return original_coupled(dt, **kw)

    example.scene.vbd_substep = _track_vbd
    example.scene.coupled_substep = _track_coupled
    example.simulate()
    assert vbd_calls == example.sim_substeps
    assert coupled_calls == 0


def test_defer_settle_to_viewer_only_without_fix_to_apple_and_gl():
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        _defer_settle_to_viewer,
    )

    import newton

    assert not _defer_settle_to_viewer(MagicMock(), fix_to_apple=True, settle_substeps=100)
    assert not _defer_settle_to_viewer(MagicMock(), fix_to_apple=False, settle_substeps=0)
    gl_viewer = MagicMock()
    gl_viewer.__class__ = newton.viewer.ViewerGL
    assert _defer_settle_to_viewer(gl_viewer, fix_to_apple=False, settle_substeps=100)


def test_batched_heterogeneous_only_vbd_runs_settle_with_gravity_ramp(capsys):
    """--only-vbd with settle_substeps>0 runs in-place settle and logs the ramp."""
    import argparse
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        ExampleBatchedHeterogeneousCoupledFruiting,
    )

    viewer = MagicMock()
    args = argparse.Namespace(
        hz=30.0,
        json=None,
        seed=42,
        num_envs=2,
        env_spacing=[2.0, 2.0, 2.0],
        enable_self_collision=False,
        robot="placeholder",
        controller="direct",
        fr3_keyboard=False,
        fix_to_apple=False,
        settle_substeps=8,
        settle_gravity_ramp=True,
        inspect_settle=False,
        settle_report_brief=True,
        settle_max_speed=0.05,
        settle_ke_decay=True,
        ke_sample_every=1,
        ke_analysis_tail_fraction=0.5,
        ke_min_peaks=3,
        ke_peak_decay_rtol=0.10,
        ke_peak_threshold_j=None,
        scripted_ee_vel=[0.05, 0.0, 0.0],
        demo_per_env_actions=False,
        status_every=0,
        print_robot_state=False,
        noisy_action=False,
        noisy_action_std=0.02,
        tcp_force_arrow=False,
        tcp_force_scale=0.02,
        tcp_force_arrow_gain=1.0,
        tcp_force_min_length=0.08,
        tcp_force_max_length=1.5,
        mark_endpoints=False,
        mujoco_viewer=False,
        vic_linear_k=600.0,
        vic_linear_d=200.0,
        vic_angular_k=20.0,
        vic_angular_d=4.0,
        device="cpu",
        only_vbd=True,
        only_mjc=False,
    )

    with pytest.warns(UserWarning, match="GPU parallelism is not fully utilized"):
        example = ExampleBatchedHeterogeneousCoupledFruiting(viewer, args)

    assert example._pending_settle_substeps == 0
    out = capsys.readouterr().out
    assert "VBD settle: 8 substeps" in out
    assert "gravity ramp 0 → −9.81 m/s²" in out
    assert "Post-settle KE decay" in out
    assert len(example._settle_ke_decay_reports) == 2
    g = example.scene.cable.model.gravity.numpy()
    g_z = float(g[2]) if g.ndim == 1 else float(g[0, 2])
    assert g_z == pytest.approx(-9.81, abs=1e-5)


def test_batched_heterogeneous_vic_defaults_match_parser():
    import sys
    from pathlib import Path

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        VIC_DEFAULT_ANGULAR_D,
        VIC_DEFAULT_ANGULAR_K,
        VIC_DEFAULT_LINEAR_D,
        VIC_DEFAULT_LINEAR_K,
        _make_parser,
    )

    args = _make_parser().parse_args([])
    assert args.vic_linear_k == VIC_DEFAULT_LINEAR_K == 600.0
    assert args.vic_linear_d == VIC_DEFAULT_LINEAR_D == 200.0
    assert args.vic_angular_k == VIC_DEFAULT_ANGULAR_K == 20.0
    assert args.vic_angular_d == VIC_DEFAULT_ANGULAR_D == 4.0


def test_placeholder_multienv_warns_at_init():
    """Placeholder robot must warn that simulate() is not fully GPU-resident."""
    import argparse
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    _EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))

    from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
        ExampleBatchedHeterogeneousCoupledFruiting,
    )

    viewer = MagicMock()
    args = argparse.Namespace(
        hz=30.0,
        json=None,
        seed=42,
        num_envs=2,
        env_spacing=[2.0, 2.0, 2.0],
        enable_self_collision=False,
        robot="placeholder",
        controller="direct",
        fr3_keyboard=False,
        fix_to_apple=False,
        settle_substeps=0,
        inspect_settle=False,
        settle_report_brief=False,
        settle_max_speed=0.05,
        scripted_ee_vel=[0.05, 0.0, 0.0],
        demo_per_env_actions=False,
        status_every=0,
        print_robot_state=False,
        noisy_action=False,
        noisy_action_std=0.02,
        tcp_force_arrow=False,
        tcp_force_scale=0.02,
        tcp_force_arrow_gain=1.0,
        tcp_force_min_length=0.08,
        tcp_force_max_length=1.5,
        mark_endpoints=False,
        mujoco_viewer=False,
        vic_linear_k=600.0,
        vic_linear_d=200.0,
        vic_angular_k=20.0,
        vic_angular_d=4.0,
        device="cpu",
        only_vbd=False,
        only_mjc=False,
    )

    with pytest.warns(UserWarning, match="GPU parallelism is not fully utilized"):
        ExampleBatchedHeterogeneousCoupledFruiting(viewer, args)
