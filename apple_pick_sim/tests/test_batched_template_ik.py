"""Fast unit tests for template-model batched IK (no FR3 assets)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.robot.fr3_robot.batched_template_ik import BatchedTemplateIK
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity
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
