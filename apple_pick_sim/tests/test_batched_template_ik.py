"""Fast unit tests for template-model batched IK (no FR3 assets)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.robot.fr3_robot.batched_template_ik import BatchedTemplateIK
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity, integrate_tcp_target
from apple_pick_sim.robot.fr3_robot.placement import (
    IK_BOOTSTRAP_POS_TOL_M,
    IK_BOOTSTRAP_ROT_TOL_RAD,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity_batched import (
    Fr3BatchedEEVelocityController,
)


def _build_two_world_arm_pair(*, device: str = "cpu") -> tuple[newton.Model, newton.Model, BatchedEnvLayout]:
    """Single-world template + 2-world replicated 2-DOF revolute arm."""
    template_builder = newton.ModelBuilder()
    b1 = template_builder.add_link(mass=1.0)
    b2 = template_builder.add_link(mass=1.0)
    j1 = template_builder.add_joint_revolute(parent=-1, child=b1, axis=wp.vec3(0.0, 0.0, 1.0))
    j2 = template_builder.add_joint_revolute(parent=b1, child=b2, axis=wp.vec3(0.0, 0.0, 1.0))
    template_builder.add_articulation([j1, j2])
    template_builder.add_shape_box(body=b1, hx=0.05, hy=0.05, hz=0.05)
    template_builder.add_shape_box(body=b2, hx=0.05, hy=0.05, hz=0.05)
    template_model = template_builder.finalize(device=device)

    batched_builder = newton.ModelBuilder()
    batched_builder.replicate(template_builder, world_count=2, spacing=(1.0, 0.0, 0.0))
    batched_model = batched_builder.finalize(device=device)

    robot_bodies_per = int(batched_model.body_count // 2)
    layout = BatchedEnvLayout(
        num_envs=2,
        bodies_per_world=0,
        robot_bodies_per_world=robot_bodies_per,
        joints_per_world=0,
        joint_coord_count_per_world=int(template_model.joint_coord_count),
        joint_dof_count_per_world=int(template_model.joint_dof_count),
        template_tcp_body=1,
        template_proxy_body=0,
        template_apple_body=None,
        tcp_body_indices=(1, 1 + robot_bodies_per),
        proxy_body_indices=(0, 0),
        apple_body_indices=(-1, -1),
        env_spacing=(1.0, 0.0, 0.0),
    )
    return template_model, batched_model, layout


def test_ik_solver_dof_is_per_world_not_batched_total():
    import newton.ik as ik

    template_model, batched_model, layout = _build_two_world_arm_pair()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
    )
    assert ik_engine.n_coords == int(template_model.joint_coord_count)
    assert ik_engine.n_coords < int(batched_model.joint_coord_count)
    assert ik_engine.n_problems == layout.num_envs

    pos_obj = ik.IKObjectivePosition(
        link_index=1,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.zeros(1, dtype=wp.vec3, device=batched_model.device),
    )
    wrong = ik.IKSolver(
        model=batched_model,
        n_problems=1,
        objectives=[pos_obj],
    )
    assert wrong.n_coords == int(batched_model.joint_coord_count)
    assert wrong.n_coords > ik_engine.n_coords


def test_scatter_writes_all_world_slices():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
    )
    jq = batched_model.joint_q.numpy().copy()
    for w in range(layout.num_envs):
        sl = layout.joint_q_slice(w)
        jq[sl] = np.array([0.1 * (w + 1), 0.2 * (w + 1)], dtype=jq.dtype)
    batched_model.joint_q.assign(jq)
    state.joint_q.assign(jq)
    newton.eval_fk(batched_model, state.joint_q, state.joint_qd, state)

    ik_engine.seed_from_state(state)
    ik_engine.set_target_from_fk(state)
    ik_engine.step(iterations=8)
    ik_engine.scatter_to_model(state, eval_fk=False)

    out = batched_model.joint_q.numpy()
    for w in range(layout.num_envs):
        sl = layout.joint_q_slice(w)
        row = ik_engine.joint_q.numpy()[w]
        np.testing.assert_allclose(out[sl], row, atol=1e-5)


def test_same_velocity_all_worlds_equal_joint_q():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
    )
    jq = batched_model.joint_q.numpy().copy()
    for w in range(layout.num_envs):
        jq[layout.joint_q_slice(w)] = np.array([0.3, -0.2], dtype=jq.dtype)
    batched_model.joint_q.assign(jq)
    state.joint_q.assign(jq)
    newton.eval_fk(batched_model, state.joint_q, state.joint_qd, state)

    ik_engine.seed_from_state(state)
    ik_engine.set_target_from_fk(state)
    ik_engine.step(iterations=12)
    rows = ik_engine.joint_q.numpy()
    np.testing.assert_allclose(rows[0], rows[1], atol=1e-4)


def _scatter_to_model_loop_reference(
    ik_engine: BatchedTemplateIK,
    state: Any,
    *,
    eval_fk: bool,
) -> np.ndarray:
    """Reference loop implementation for parity checks."""
    full = ik_engine.sim_model.joint_q.numpy().copy()
    rows = ik_engine.joint_q.numpy()
    for w in range(ik_engine.n_problems):
        sl = ik_engine.layout.joint_q_slice(w)
        full[sl] = rows[w].astype(full.dtype, copy=False)
    jqd = np.zeros(int(ik_engine.sim_model.joint_dof_count), dtype=np.float32)
    ik_engine.sim_model.joint_q.assign(full)
    ik_engine.sim_model.joint_qd.assign(jqd)
    state.joint_q.assign(full)
    state.joint_qd.assign(jqd)
    if eval_fk:
        newton.eval_fk(ik_engine.sim_model, state.joint_q, state.joint_qd, state)
    return full


def _seed_from_state_loop_reference(ik_engine: BatchedTemplateIK) -> np.ndarray:
    full = ik_engine.sim_model.joint_q.numpy().reshape(-1)
    rows = np.zeros((ik_engine.n_problems, ik_engine.n_coords), dtype=np.float32)
    for w in range(ik_engine.n_problems):
        rows[w] = full[ik_engine.layout.joint_q_slice(w)].astype(np.float32, copy=False)
    return rows


def test_scatter_to_model_matches_loop():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
    )
    rows = np.array([[0.11, 0.22], [0.33, 0.44]], dtype=np.float32)
    ik_engine.joint_q.assign(rows)

    ik_engine.scatter_to_model(state, eval_fk=False)
    vectorized = batched_model.joint_q.numpy().copy()

    batched_model.joint_q.assign(np.zeros_like(batched_model.joint_q.numpy()))
    state.joint_q.assign(batched_model.joint_q.numpy())
    loop_ref = _scatter_to_model_loop_reference(ik_engine, state, eval_fk=False)

    np.testing.assert_allclose(vectorized, loop_ref, atol=1e-6)


def test_seed_from_state_matches_loop():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
    )
    jq = batched_model.joint_q.numpy().copy()
    for w in range(layout.num_envs):
        jq[layout.joint_q_slice(w)] = np.array([0.15 * (w + 1), -0.1 * w], dtype=jq.dtype)
    batched_model.joint_q.assign(jq)
    state.joint_q.assign(jq)

    ik_engine.seed_from_state(state)
    vectorized = ik_engine.joint_q.numpy()
    loop_ref = _seed_from_state_loop_reference(ik_engine)

    np.testing.assert_allclose(vectorized, loop_ref, atol=1e-6)


def test_advance_targets_batch_matches_sequential():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
    )
    jq = batched_model.joint_q.numpy().copy()
    for w in range(layout.num_envs):
        jq[layout.joint_q_slice(w)] = np.array([0.25 + 0.1 * w, -0.15], dtype=jq.dtype)
    batched_model.joint_q.assign(jq)
    state.joint_q.assign(jq)
    newton.eval_fk(batched_model, state.joint_q, state.joint_qd, state)

    pos_world = wp.zeros(layout.num_envs, dtype=wp.vec3, device=batched_model.device)
    rot_world = wp.zeros(layout.num_envs, dtype=wp.vec4, device=batched_model.device)
    ik_engine.gather_tcp_targets_from_state(state, pos_world, rot_world)

    dt = 1.0 / 60.0
    lin_vels = [
        wp.vec3(0.02, 0.01, 0.0),
        wp.vec3(0.0, 0.03, -0.01),
    ]
    ang_vels = [
        wp.vec3(0.0, 0.0, 0.5),
        wp.vec3(0.0, 0.2, 0.0),
    ]
    pos_seq = pos_world.numpy().copy()
    rot_seq = rot_world.numpy().copy()
    target_pos_seq = np.zeros((layout.num_envs, 3), dtype=np.float32)
    target_rot_seq = np.zeros((layout.num_envs, 4), dtype=np.float32)
    for w in range(layout.num_envs):
        tf = integrate_tcp_target(
            wp.transform(
                wp.vec3(float(pos_seq[w, 0]), float(pos_seq[w, 1]), float(pos_seq[w, 2])),
                wp.quat(
                    float(rot_seq[w, 0]),
                    float(rot_seq[w, 1]),
                    float(rot_seq[w, 2]),
                    float(rot_seq[w, 3]),
                ),
            ),
            linear_vel=lin_vels[w],
            angular_vel=ang_vels[w],
            dt=dt,
        )
        p = wp.transform_get_translation(tf)
        r = wp.transform_get_rotation(tf)
        pos_seq[w] = (float(p[0]), float(p[1]), float(p[2]))
        rot_seq[w] = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        ik_engine.set_target_world(w, tf)
        target_pos_seq[w] = ik_engine.target_positions.numpy()[w]
        target_rot_seq[w] = ik_engine.target_rotations.numpy()[w]

    pos_batch = wp.array(pos_world.numpy().copy(), dtype=wp.vec3, device=batched_model.device)
    rot_batch = wp.array(rot_world.numpy().copy(), dtype=wp.vec4, device=batched_model.device)
    lin_wp = wp.array(lin_vels, dtype=wp.vec3, device=batched_model.device)
    ang_wp = wp.array(ang_vels, dtype=wp.vec3, device=batched_model.device)
    ik_engine.advance_targets_batch(pos_batch, rot_batch, lin_wp, ang_wp, dt)

    np.testing.assert_allclose(
        ik_engine.target_positions.numpy(),
        target_pos_seq,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        ik_engine.target_rotations.numpy(),
        target_rot_seq,
        atol=1e-5,
    )
    np.testing.assert_allclose(pos_batch.numpy(), pos_seq, atol=1e-5)
    np.testing.assert_allclose(rot_batch.numpy(), rot_seq, atol=1e-5)


def _tcp_world_target_from_joint_q(
    template_model: newton.Model,
    layout: BatchedEnvLayout,
    world: int,
    joint_q: np.ndarray,
) -> wp.transform:
    jc = int(template_model.joint_coord_count)
    tpl_jq = template_model.joint_q.numpy().copy()
    tpl_jq[:jc] = np.asarray(joint_q, dtype=tpl_jq.dtype).reshape(-1)[:jc]
    tpl_jqd = np.zeros(int(template_model.joint_dof_count), dtype=np.float32)
    template_model.joint_q.assign(tpl_jq)
    template_model.joint_qd.assign(tpl_jqd)
    tpl_state = template_model.state()
    newton.eval_fk(template_model, template_model.joint_q, template_model.joint_qd, tpl_state)
    bq = tpl_state.body_q.numpy().reshape(-1, 7)[layout.template_tcp_body]
    ox, oy, oz = layout.world_origin(world)
    return wp.transform(
        wp.vec3(float(bq[0]) + ox, float(bq[1]) + oy, float(bq[2]) + oz),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )


def test_multi_seed_bootstrap_converges():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
        n_seeds=4,
        sampler="gauss",
    )
    jq_target = np.array([0.45, -0.35], dtype=np.float32)
    jq = batched_model.joint_q.numpy().copy()
    for w in range(layout.num_envs):
        jq[layout.joint_q_slice(w)] = jq_target
    batched_model.joint_q.assign(jq)
    state.joint_q.assign(jq)
    newton.eval_fk(batched_model, state.joint_q, state.joint_qd, state)

    ik_engine.seed_from_state(state)
    ik_engine.set_target_from_fk(state)
    ik_engine.step(iterations=32)

    targets_world = []
    for w in range(layout.num_envs):
        tpl = ik_engine.tcp_template_pose_from_model(w)
        targets_world.append(ik_engine.template_to_world(tpl, w))
    for w in range(layout.num_envs):
        pos_err, rot_err = ik_engine.pose_error(w, targets_world[w])
        assert pos_err < IK_BOOTSTRAP_POS_TOL_M, f"world {w} pos_err={pos_err}"
        assert rot_err < IK_BOOTSTRAP_ROT_TOL_RAD, f"world {w} rot_err={rot_err}"

    ik_engine.scatter_to_model(state, eval_fk=False)

    ik_engine.scatter_to_model(state, eval_fk=False)


def test_rotation_mismatch_penalized_by_multi_seed_cost():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    n_seeds = 4
    ik_engine = BatchedTemplateIK(
        ik_model=template_model,
        sim_model=batched_model,
        layout=layout,
        tcp_body_index=layout.template_tcp_body,
        n_seeds=n_seeds,
        sampler="roberts",
    )
    target_world = _tcp_world_target_from_joint_q(
        template_model, layout, 0, np.array([0.55, 0.25], dtype=np.float32)
    )
    ik_engine.set_target_world(0, target_world)

    upper = template_model.joint_limit_upper.numpy().reshape(-1)[: ik_engine.n_coords]
    seed_rows = np.zeros((layout.num_envs, ik_engine.n_coords), dtype=np.float32)
    seed_rows[0] = upper.astype(np.float32)
    ik_engine.joint_q.assign(seed_rows)
    ik_engine.step(iterations=64)

    costs = ik_engine.solver_costs_expanded().numpy().reshape(layout.num_envs, n_seeds)
    best_seed = int(np.argmin(costs[0]))
    pos_err, rot_err = ik_engine.pose_error(0, target_world)
    assert pos_err < IK_BOOTSTRAP_POS_TOL_M
    assert rot_err < IK_BOOTSTRAP_ROT_TOL_RAD

    jq_expanded = ik_engine.solver_joint_q_expanded().numpy().reshape(
        layout.num_envs, n_seeds, ik_engine.n_coords
    )
    best_cost = float(costs[0, best_seed])
    for seed_idx in range(n_seeds):
        if seed_idx == best_seed:
            continue
        alt_pos, alt_rot = ik_engine.pose_error_from_joint_coords(
            jq_expanded[0, seed_idx], 0, target_world
        )
        if alt_pos < pos_err and alt_rot > rot_err:
            assert best_cost < float(costs[0, seed_idx])
            break
    else:
        assert best_cost <= float(np.min(costs[0]))


def test_different_velocity_per_world_diverges_targets():
    template_model, batched_model, layout = _build_two_world_arm_pair()
    state = batched_model.state()
    ctrl = Fr3BatchedEEVelocityController(
        batched_model,
        layout,
        template_model,
        layout.template_tcp_body,
        velocity_for_world=lambda w: (
            EEVelocity(linear=(0.01, 0.0, 0.0)) if w == 0 else EEVelocity()
        ),
    )
    ctrl.sync_target_from_state(state)
    ctrl.advance_target(1.0 / 60.0, velocity=EEVelocity())
    t0 = wp.transform_get_translation(ctrl.target_tf[0])
    t1 = wp.transform_get_translation(ctrl.target_tf[1])
    assert float(t0[0]) != pytest.approx(float(t1[0]), abs=1e-6)
