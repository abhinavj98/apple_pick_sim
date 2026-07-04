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


def test_print_per_env_params_includes_all_rod_stiffnesses(capsys):
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import print_per_env_params

    ranges = load_ranges(RANGES_FIXTURE)
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=0, num_envs=2)
    print_per_env_params(params_list)

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


def test_batched_heterogeneous_only_vbd_builds_cable_only_scene(ranges):
    """vbd_only CoupledSim builds cable-only scene and steps VBD substeps only."""
    import dataclasses

    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
        BatchedHeterogeneousCoupledSimConfig,
        ObsConfig,
        RobotConfig,
        SceneSettleCollisionConfig,
    )
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
        BatchedHeterogeneousCoupledSim,
    )

    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=42, num_envs=_NUM_ENVS
    )
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(
            kind="placeholder",
            step_mode="vbd_only",
            fix_to_apple=False,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=0),
        obs=ObsConfig(allocate_buffers=False),
    )
    with pytest.warns(UserWarning, match="CPU host nudge"):
        sim = BatchedHeterogeneousCoupledSim(
            cfg, params_list, ranges, use_settle_cache=False
        )

    assert sim.scene.vbd_only
    assert sim.scene.robot_model is None
    assert sim.layout is not None
    assert sim.layout.num_envs == _NUM_ENVS

    vbd_calls = 0
    coupled_calls = 0
    original_vbd = sim.scene.vbd_substep
    original_coupled = sim.scene.coupled_substep

    def _track_vbd(dt, **kw):
        nonlocal vbd_calls
        vbd_calls += 1
        return original_vbd(dt, **kw)

    def _track_coupled(dt, **kw):
        nonlocal coupled_calls
        coupled_calls += 1
        return original_coupled(dt, **kw)

    sim.scene.vbd_substep = _track_vbd
    sim.scene.coupled_substep = _track_coupled
    sim.step(None)
    assert vbd_calls == cfg.runtime.substeps_per_step
    assert coupled_calls == 0


def test_batched_heterogeneous_only_vbd_runs_settle_with_gravity_ramp(ranges):
    """vbd_only with settle_substeps>0 and gravity ramp finishes at full gravity."""
    import dataclasses

    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
        BatchedHeterogeneousCoupledSimConfig,
        RobotConfig,
        SceneSettleCollisionConfig,
        SettleDiagnosticsConfig,
    )
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
        BatchedHeterogeneousCoupledSim,
    )

    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=42, num_envs=_NUM_ENVS
    )
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(
            kind="placeholder",
            step_mode="vbd_only",
            fix_to_apple=False,
        ),
        scene=SceneSettleCollisionConfig(
            settle_substeps=8,
            settle_gravity_ramp=True,
        ),
        settle_diagnostics=SettleDiagnosticsConfig(report_brief=True),
    )
    with pytest.warns(UserWarning, match="CPU host nudge"):
        sim = BatchedHeterogeneousCoupledSim(
            cfg, params_list, ranges, use_settle_cache=False
        )

    br = sim.build_result
    assert br.settle_stability_reports is not None
    assert len(br.settle_ke_decay_reports or ()) == _NUM_ENVS
    g = sim.scene.cable.model.gravity.numpy()
    g_z = float(g[2]) if g.ndim == 1 else float(g[0, 2])
    assert g_z == pytest.approx(-9.81, abs=1e-5)


def _shape_pairs_filtered(model, body_a: int, body_b: int) -> bool:
    shapes_a = model.body_shapes.get(body_a, [])
    shapes_b = model.body_shapes.get(body_b, [])
    if not shapes_a or not shapes_b:
        return False
    pairs = set(model.shape_collision_filter_pairs)
    for s1 in shapes_a:
        for s2 in shapes_b:
            if (s1, s2) in pairs or (s2, s1) in pairs:
                return True
    return False


def test_heterogeneous_apple_woody_collision_toggle(ranges):
    """``enable_apple_woody_collisions`` controls apple↔woody AVBD filter pairs."""
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=7, num_envs=_NUM_ENVS
    )
    base_kw = dict(
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    cable_on, _ = build_heterogeneous_coupled_cable_scene(
        params_list, enable_apple_woody_collisions=True, **base_kw
    )
    cable_off, _ = build_heterogeneous_coupled_cable_scene(
        params_list, enable_apple_woody_collisions=False, **base_kw
    )
    apple = cable_on.apple_body
    primary = cable_on.primary_bodies[0]
    assert apple is not None
    assert not _shape_pairs_filtered(cable_on.model, apple, primary)
    assert _shape_pairs_filtered(cable_off.model, apple, primary)


def test_heterogeneous_proxy_woody_collision_toggle(ranges):
    """``enable_proxy_woody_collisions`` controls proxy↔woody AVBD filter pairs."""
    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=9, num_envs=_NUM_ENVS
    )
    base_kw = dict(
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=_gripper_free(),
    )
    cable_on, _ = build_heterogeneous_coupled_cable_scene(
        params_list, enable_proxy_woody_collisions=True, **base_kw
    )
    cable_off, _ = build_heterogeneous_coupled_cable_scene(
        params_list, enable_proxy_woody_collisions=False, **base_kw
    )
    proxy = cable_on.gripper_proxy_body
    primary = cable_on.primary_bodies[0]
    assert not _shape_pairs_filtered(cable_on.model, proxy, primary)
    assert _shape_pairs_filtered(cable_off.model, proxy, primary)


def test_placeholder_multienv_warns_at_init(ranges):
    """Placeholder robot must warn that step() is not fully GPU-resident."""
    import dataclasses

    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
        BatchedHeterogeneousCoupledSimConfig,
        RobotConfig,
        SceneSettleCollisionConfig,
    )
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
        BatchedHeterogeneousCoupledSim,
    )

    params_list = sample_heterogeneous_params_list(
        ranges, topology_seed=42, num_envs=_NUM_ENVS
    )
    cfg = dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=_NUM_ENVS),
        robot=RobotConfig(
            kind="placeholder",
            step_mode="coupled",
            fix_to_apple=False,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=0),
    )
    with pytest.warns(UserWarning, match="CPU host nudge"):
        BatchedHeterogeneousCoupledSim(cfg, params_list, ranges, use_settle_cache=False)
