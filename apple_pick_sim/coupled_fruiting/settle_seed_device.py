"""Device-side settle→weld cable state seeding (proxy alignment, twist zero)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout

_ZERO_QD = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def offset7_to_transform(offset_7d: tuple | np.ndarray | Sequence[float]) -> wp.transform:
    off = np.asarray(offset_7d, dtype=np.float32).reshape(7)
    return wp.transform(
        wp.vec3(float(off[0]), float(off[1]), float(off[2])),
        wp.quat(float(off[3]), float(off[4]), float(off[5]), float(off[6])),
    )


def capture_body_q_numpy(body_q: wp.array) -> np.ndarray:
    """One host read of cable ``body_q`` for checkpoint persistence."""
    return (
        wp.to_torch(body_q)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 7)
        .astype(np.float32)
        .copy()
    )


def copy_cable_state_device(src_cable: Any, dst_cable: Any) -> None:
    """Copy ``state_0`` body poses/twists between cable scenes on device."""
    src_bq = src_cable.state_0.body_q
    src_bqd = src_cable.state_0.body_qd
    if wp.types.is_array(src_bq) and wp.types.is_array(src_bqd):
        wp.copy(dst_cable.state_0.body_q, src_bq)
        wp.copy(dst_cable.state_0.body_qd, src_bqd)
        return
    bq = np.asarray(src_bq.numpy() if hasattr(src_bq, "numpy") else src_bq, dtype=np.float32).reshape(-1, 7)
    bqd = np.asarray(
        src_bqd.numpy() if hasattr(src_bqd, "numpy") else src_bqd,
        dtype=np.float32,
    ).reshape(-1, 6)
    dst_cable.state_0.body_q.assign(bq)
    dst_cable.state_0.body_qd.assign(bqd)


@wp.kernel(enable_backward=False)
def _zero_body_qd_kernel(body_qd: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    body_qd[i] = _ZERO_QD


def zero_all_body_qd_device(body_qd: wp.array) -> None:
    n = int(body_qd.shape[0])
    if n <= 0:
        return
    wp.launch(_zero_body_qd_kernel, dim=n, inputs=[body_qd], device=body_qd.device)


@wp.kernel(enable_backward=False)
def _align_proxy_from_apple_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    apple_indices: wp.array(dtype=int),
    proxy_indices: wp.array(dtype=int),
    grasp_offsets: wp.array(dtype=wp.transform),
    quiet_apple_proxy: int,
):
    w = wp.tid()
    apple_idx = apple_indices[w]
    proxy_idx = proxy_indices[w]
    if apple_idx < 0 or proxy_idx < 0:
        return
    apple_tf = body_q[apple_idx]
    proxy_tf = wp.transform_multiply(apple_tf, grasp_offsets[w])
    body_q[proxy_idx] = proxy_tf
    if quiet_apple_proxy != 0:
        body_qd[apple_idx] = _ZERO_QD
        body_qd[proxy_idx] = _ZERO_QD


def _grasp_offsets_wp(
    layout: BatchedEnvLayout,
    per_world_proxy_offsets: tuple[tuple | None, ...] | None,
    default_offset: tuple | np.ndarray | None,
    device: Any,
) -> wp.array:
    if default_offset is None:
        raise ValueError("default_offset required to build grasp offset array")
    rows: list[wp.transform] = []
    for w in range(int(layout.num_envs)):
        off = default_offset
        if per_world_proxy_offsets is not None and w < len(per_world_proxy_offsets):
            candidate = per_world_proxy_offsets[w]
            if candidate is not None:
                off = candidate
        rows.append(offset7_to_transform(off))
    return wp.array(rows, dtype=wp.transform, device=device)


def align_batched_proxy_poses_device(
    cable: Any,
    layout: BatchedEnvLayout,
    *,
    per_world_proxy_offsets: tuple[tuple | None, ...] | None,
    default_offset: tuple | np.ndarray | None,
    quiet_apple_proxy: bool,
) -> None:
    """Write per-env proxy ``body_q`` from apple pose × grasp offset; optionally zero twists."""
    dev = cable.state_0.body_q.device
    apple_idx = wp.array(list(layout.apple_body_indices), dtype=int, device=dev)
    proxy_idx = wp.array(list(layout.proxy_body_indices), dtype=int, device=dev)
    grasp_offsets = _grasp_offsets_wp(
        layout,
        per_world_proxy_offsets,
        default_offset,
        dev,
    )
    num_envs = int(layout.num_envs)
    wp.launch(
        _align_proxy_from_apple_kernel,
        dim=num_envs,
        inputs=[
            cable.state_0.body_q,
            cable.state_0.body_qd,
            apple_idx,
            proxy_idx,
            grasp_offsets,
            1 if quiet_apple_proxy else 0,
        ],
        device=dev,
    )
    wp.copy(cable.state_1.body_q, cable.state_0.body_q)
    wp.copy(cable.state_1.body_qd, cable.state_0.body_qd)
