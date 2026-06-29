"""GPU-native batched observation gather for coupled fruiting scenes.

After each simulation step, :func:`gather_batched_obs` fills pre-allocated Warp
buffers with endpoint positions and forces for every env in a batched layout.
Consumers (e.g. a future RL env) read ``wp.array`` views without a host round-trip.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.fruiting_system.coupled import CoupledCableScene
from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device


@wp.kernel(enable_backward=False)
def _gather_positions_kernel(
    body_q: wp.array(dtype=wp.transform),
    indices: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    idx = indices[i]
    if idx < 0:
        out[i, 0] = 0.0
        out[i, 1] = 0.0
        out[i, 2] = 0.0
        return
    t = body_q[idx]
    p = wp.transform_get_translation(t)
    out[i, 0] = p[0]
    out[i, 1] = p[1]
    out[i, 2] = p[2]


@wp.kernel(enable_backward=False)
def _gather_transforms_kernel(
    body_q: wp.array(dtype=wp.transform),
    indices: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    idx = indices[i]
    if idx < 0:
        for k in range(7):
            out[i, k] = 0.0
        out[i, 6] = 1.0
        return
    t = body_q[idx]
    p = wp.transform_get_translation(t)
    r = wp.transform_get_rotation(t)
    out[i, 0] = p[0]
    out[i, 1] = p[1]
    out[i, 2] = p[2]
    out[i, 3] = r[0]
    out[i, 4] = r[1]
    out[i, 5] = r[2]
    out[i, 6] = r[3]


@wp.kernel(enable_backward=False)
def _gather_spatial_vectors_kernel(
    arr: wp.array(dtype=wp.spatial_vector),
    indices: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    idx = indices[i]
    if idx < 0:
        for k in range(6):
            out[i, k] = 0.0
        return
    w = arr[idx]
    f = wp.spatial_top(w)
    tau = wp.spatial_bottom(w)
    out[i, 0] = f[0]
    out[i, 1] = f[1]
    out[i, 2] = f[2]
    out[i, 3] = tau[0]
    out[i, 4] = tau[1]
    out[i, 5] = tau[2]


@wp.kernel(enable_backward=False)
def _copy_vec3_rows_kernel(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    v = src[i]
    dst[i, 0] = v[0]
    dst[i, 1] = v[1]
    dst[i, 2] = v[2]


def launch_gather_positions(
    body_q: wp.array,
    indices: wp.array,
    out: wp.array,
) -> None:
    """Gather body COM positions into ``out`` with shape ``(len(indices), 3)``."""
    n = int(indices.shape[0])
    if n == 0:
        return
    wp.launch(
        _gather_positions_kernel,
        dim=n,
        inputs=[body_q, indices, out],
        device=out.device,
    )


def launch_gather_transforms(
    body_q: wp.array,
    indices: wp.array,
    out: wp.array,
) -> None:
    """Gather body transforms as ``[x,y,z,qx,qy,qz,qw]`` rows."""
    n = int(indices.shape[0])
    if n == 0:
        return
    wp.launch(
        _gather_transforms_kernel,
        dim=n,
        inputs=[body_q, indices, out],
        device=out.device,
    )


def launch_gather_spatial_vectors(
    arr: wp.array,
    indices: wp.array,
    out: wp.array,
) -> None:
    """Gather spatial wrench rows ``[Fx,Fy,Fz,τx,τy,τz]``."""
    n = int(indices.shape[0])
    if n == 0:
        return
    wp.launch(
        _gather_spatial_vectors_kernel,
        dim=n,
        inputs=[arr, indices, out],
        device=out.device,
    )


def _launch_copy_vec3_rows(src: wp.array, dst: wp.array) -> None:
    n = int(src.shape[0])
    if n == 0:
        return
    wp.launch(
        _copy_vec3_rows_kernel,
        dim=n,
        inputs=[src, dst],
        device=dst.device,
    )


def _upload_indices(values: list[int], device: str) -> wp.array:
    return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)


@dataclasses.dataclass
class BatchedObsBuffers:
    """Pre-allocated device buffers for batched endpoint observations."""

    num_envs: int
    num_junctions: int
    device: str

    apple_indices: wp.array
    proxy_indices: wp.array
    tcp_indices: wp.array
    woody_parent_indices: wp.array
    woody_child_indices: wp.array
    woody_joint_indices: wp.array

    apple_pos: wp.array
    proxy_pos: wp.array
    tcp_pose: wp.array
    tcp_force: wp.array
    tcp_coupling_force: wp.array
    woody_parent_pos: wp.array
    woody_child_pos: wp.array
    woody_force: wp.array
    woody_torque: wp.array


def make_batched_obs_buffers(
    layout: BatchedEnvLayout,
    cable: CoupledCableScene | Any,
    device: str,
) -> BatchedObsBuffers:
    """Allocate index and output buffers for :func:`gather_batched_obs`."""
    num_envs = int(layout.num_envs)
    joint_pairs = list(getattr(cable, "fruiting_fixed_joints", ()) or ())
    num_junctions = len(joint_pairs)
    n_j = num_envs * num_junctions

    model = cable.model
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()

    woody_parent_idx: list[int] = []
    woody_child_idx: list[int] = []
    woody_joint_idx: list[int] = []
    for w in range(num_envs):
        for j_idx, _label in joint_pairs:
            tpl_parent = int(joint_parent[int(j_idx)])
            tpl_child = int(joint_child[int(j_idx)])
            woody_parent_idx.append(layout.body_index(w, tpl_parent))
            woody_child_idx.append(layout.body_index(w, tpl_child))
            woody_joint_idx.append(layout.joint_index(w, int(j_idx)))

    return BatchedObsBuffers(
        num_envs=num_envs,
        num_junctions=num_junctions,
        device=device,
        apple_indices=_upload_indices(list(layout.apple_body_indices), device),
        proxy_indices=_upload_indices(list(layout.proxy_body_indices), device),
        tcp_indices=_upload_indices(list(layout.tcp_body_indices), device),
        woody_parent_indices=_upload_indices(woody_parent_idx, device),
        woody_child_indices=_upload_indices(woody_child_idx, device),
        woody_joint_indices=_upload_indices(woody_joint_idx, device),
        apple_pos=wp.zeros((num_envs, 3), dtype=wp.float32, device=device),
        proxy_pos=wp.zeros((num_envs, 3), dtype=wp.float32, device=device),
        tcp_pose=wp.zeros((num_envs, 7), dtype=wp.float32, device=device),
        tcp_force=wp.zeros((num_envs, 6), dtype=wp.float32, device=device),
        tcp_coupling_force=wp.zeros((num_envs, 6), dtype=wp.float32, device=device),
        woody_parent_pos=wp.zeros((n_j, 3), dtype=wp.float32, device=device),
        woody_child_pos=wp.zeros((n_j, 3), dtype=wp.float32, device=device),
        woody_force=wp.zeros((n_j, 3), dtype=wp.float32, device=device),
        woody_torque=wp.zeros((n_j, 3), dtype=wp.float32, device=device),
    )


def gather_batched_obs(
    bufs: BatchedObsBuffers,
    scene: Any,
    sim_dt: float,
) -> None:
    """Gather endpoint positions and forces into ``bufs`` (in-place, device-only)."""
    cable = scene.cable
    cable_bq = cable.state_0.body_q
    cable_bq_prev = cable.state_1.body_q
    robot_bq = scene.robot_state_0.body_q
    n_j = bufs.num_envs * bufs.num_junctions

    launch_gather_positions(cable_bq, bufs.apple_indices, bufs.apple_pos)
    launch_gather_positions(cable_bq, bufs.proxy_indices, bufs.proxy_pos)
    launch_gather_transforms(robot_bq, bufs.tcp_indices, bufs.tcp_pose)

    if scene.proxy_forces is not None:
        launch_gather_spatial_vectors(scene.proxy_forces, bufs.tcp_indices, bufs.tcp_force)
    if scene.coupling_forces_cache is not None:
        launch_gather_spatial_vectors(
            scene.coupling_forces_cache,
            bufs.tcp_indices,
            bufs.tcp_coupling_force,
        )

    if n_j > 0:
        launch_gather_positions(cable_bq, bufs.woody_parent_indices, bufs.woody_parent_pos)
        launch_gather_positions(cable_bq, bufs.woody_child_indices, bufs.woody_child_pos)
        out_f, out_t = gather_joint_wrench_child_com_device(
            cable.model,
            cable.solver,
            body_q=cable_bq,
            body_q_prev=cable_bq_prev,
            joint_indices=bufs.woody_joint_indices,
            dt=float(sim_dt),
        )
        _launch_copy_vec3_rows(out_f, bufs.woody_force)
        _launch_copy_vec3_rows(out_t, bufs.woody_torque)
