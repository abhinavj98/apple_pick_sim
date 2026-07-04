"""Device-side broadcast kernels for batched robot joints and legacy cable seeding."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout


@wp.kernel(enable_backward=False)
def _broadcast_world0_slice_kernel(
    arr: wp.array(dtype=float),
    elems_per_world: int,
):
    w = wp.tid()
    if w == 0:
        return
    base = w * elems_per_world
    for i in range(elems_per_world):
        arr[base + i] = arr[i]


@wp.kernel(enable_backward=False)
def _scatter_template_row_kernel(
    batched: wp.array(dtype=float),
    tpl_row: wp.array(dtype=float),
    elems_per_world: int,
    tpl_count: int,
):
    w = wp.tid()
    dst_base = w * elems_per_world
    for i in range(tpl_count):
        batched[dst_base + i] = tpl_row[i]


@wp.kernel(enable_backward=False)
def _nudge_joint_coord_kernel(
    joint_q: wp.array(dtype=float),
    coord_index: int,
    delta: float,
):
    joint_q[coord_index] = joint_q[coord_index] + delta


@wp.kernel(enable_backward=False)
def _broadcast_settled_body_q_kernel(
    dst_body_q: wp.array(dtype=wp.transform),
    tpl_body_q: wp.array(dtype=wp.transform),
    tpl_body_count: int,
    bodies_per_world: int,
    world_offsets: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    w = tid // tpl_body_count
    local_b = tid % tpl_body_count
    tpl_tf = tpl_body_q[local_b]
    shift = world_offsets[w]
    p = wp.transform_get_translation(tpl_tf)
    dst_body_q[w * bodies_per_world + local_b] = wp.transform(
        p + shift,
        wp.transform_get_rotation(tpl_tf),
    )


@wp.kernel(enable_backward=False)
def _broadcast_settled_body_qd_kernel(
    dst_body_qd: wp.array(dtype=wp.spatial_vector),
    tpl_body_qd: wp.array(dtype=wp.spatial_vector),
    tpl_body_count: int,
    bodies_per_world: int,
):
    tid = wp.tid()
    w = tid // tpl_body_count
    local_b = tid % tpl_body_count
    dst_body_qd[w * bodies_per_world + local_b] = tpl_body_qd[local_b]


def _launch_broadcast_world0(arr: wp.array, elems_per_world: int, num_envs: int) -> None:
    if num_envs < 2 or elems_per_world <= 0:
        return
    wp.launch(
        _broadcast_world0_slice_kernel,
        dim=num_envs,
        inputs=[arr, int(elems_per_world)],
        device=arr.device,
    )


def broadcast_joint_q_from_world0_device(scene: Any, layout: BatchedEnvLayout) -> None:
    """Copy world-0 ``joint_q`` / ``joint_qd`` to every env on model and ``robot_state_0``."""
    if layout.num_envs < 2:
        return
    if scene.robot_model is None or scene.robot_state_0 is None:
        raise ValueError("broadcast_joint_q_from_world0 requires robot model and state")

    coord_per = int(layout.joint_coord_count_per_world)
    dof_per = int(layout.joint_dof_count_per_world)
    for target in (scene.robot_model, scene.robot_state_0):
        _launch_broadcast_world0(target.joint_q, coord_per, int(layout.num_envs))
        _launch_broadcast_world0(target.joint_qd, dof_per, int(layout.num_envs))

    newton.eval_fk(
        scene.robot_model,
        scene.robot_state_0.joint_q,
        scene.robot_state_0.joint_qd,
        scene.robot_state_0,
    )


def nudge_joint_q_coord_device(
    joint_q: wp.array,
    coord_index: int,
    delta: float,
) -> None:
    """Add ``delta`` to one scalar joint coordinate on device."""
    wp.launch(
        _nudge_joint_coord_kernel,
        dim=1,
        inputs=[joint_q, int(coord_index), float(delta)],
        device=joint_q.device,
    )


def broadcast_robot_state_from_template_device(
    template_model: Any,
    batched_model: Any,
) -> None:
    """Copy template robot ``joint_q`` / ``joint_qd`` into every world on device."""
    num_envs = int(batched_model.world_count)
    if num_envs < 1:
        return
    jcs = batched_model.joint_coord_world_start.numpy()
    coord_per = int(jcs[1] - jcs[0])
    dps = batched_model.joint_dof_world_start.numpy()
    dof_per = int(dps[1] - dps[0])
    tpl_jc = int(template_model.joint_coord_count)
    tpl_dof = int(template_model.joint_dof_count)
    tpl_jq = template_model.joint_q
    tpl_jqd = template_model.joint_qd
    wp.launch(
        _scatter_template_row_kernel,
        dim=num_envs,
        inputs=[batched_model.joint_q, tpl_jq, coord_per, tpl_jc],
        device=batched_model.device,
    )
    wp.launch(
        _scatter_template_row_kernel,
        dim=num_envs,
        inputs=[batched_model.joint_qd, tpl_jqd, dof_per, tpl_dof],
        device=batched_model.device,
    )


def broadcast_settled_cable_state_to_batched_worlds_device(
    settled_cable: Any,
    welded_cable: Any,
    layout: BatchedEnvLayout,
    env_spacing: Sequence[float],
) -> None:
    """Legacy device path: copy single-world settled cable into every replicated world."""
    tpl_bq = settled_cable.state_0.body_q
    tpl_bqd = settled_cable.state_0.body_qd
    tpl_n = int(tpl_bq.shape[0])
    bodies_per = int(layout.bodies_per_world)
    num_envs = int(layout.num_envs)
    world_offsets = newton.utils.compute_world_offsets(
        num_envs,
        tuple(float(v) for v in env_spacing),
        up_axis=newton.Axis.Z,
    )
    offsets_wp = wp.array(
        [wp.vec3(float(o[0]), float(o[1]), float(o[2])) for o in world_offsets],
        dtype=wp.vec3,
        device=tpl_bq.device,
    )
    dim = num_envs * tpl_n
    for state in (welded_cable.state_0, welded_cable.state_1):
        wp.launch(
            _broadcast_settled_body_q_kernel,
            dim=dim,
            inputs=[
                state.body_q,
                tpl_bq,
                tpl_n,
                bodies_per,
                offsets_wp,
            ],
            device=tpl_bq.device,
        )
        wp.launch(
            _broadcast_settled_body_qd_kernel,
            dim=dim,
            inputs=[state.body_qd, tpl_bqd, tpl_n, bodies_per],
            device=tpl_bqd.device,
        )
