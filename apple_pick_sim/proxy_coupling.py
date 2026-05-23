"""M1 proxy coupling: mirror robot state onto VBD proxies and harvest reaction wrenches.

Implements the **staggered two-Model coupling** helpers from ``docs/ROADMAP.md``
([M1]): ``sync_proxy_state`` (pose/velocity mirror with double-integration guard) and
``harvest_proxy_wrenches_velocity_delta`` for **option 3** in the roadmap —
reconstruct net proxy wrench from the velocity jump across a VBD sub-step (no
``newton/`` API changes; direct read of VBD internal accumulators stays a future
optimization).

Spatial vectors are **world frame**, linear first / angular second [N, N·m].
"""

from __future__ import annotations

import dataclasses
from typing import Any

import warp as wp


@dataclasses.dataclass(frozen=True)
class ProxyBodyRegistry:
    """Maps robot ``Model`` body indices to proxy bodies on the cable ``Model``."""

    robot_to_proxy: tuple[tuple[int, int], ...]
    """Sorted ``(robot_body_id, proxy_body_id)`` pairs."""

    @classmethod
    def from_mapping(cls, mapping: dict[int, int]) -> ProxyBodyRegistry:
        return cls(robot_to_proxy=tuple(sorted(mapping.items())))

    @property
    def robot_body_ids(self) -> tuple[int, ...]:
        return tuple(r for r, _ in self.robot_to_proxy)

    @property
    def proxy_body_ids(self) -> tuple[int, ...]:
        return tuple(p for _, p in self.robot_to_proxy)

    def robot_ids_wp(self, device) -> wp.array:
        return wp.array(self.robot_body_ids, dtype=int, device=device)

    def proxy_ids_wp(self, device) -> wp.array:
        return wp.array(self.proxy_body_ids, dtype=int, device=device)


@wp.kernel
def sync_proxy_state(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    src_body_q: wp.array(dtype=wp.transform),
    src_body_qd: wp.array(dtype=wp.spatial_vector),
    dst_body_q: wp.array(dtype=wp.transform),
    dst_body_qd: wp.array(dtype=wp.spatial_vector),
    proxy_forces: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    dt: float,
):
    """Copy robot pose/vel to proxy bodies; remove lagged coupling + gravity from linear vel.

    Mirrors ``docs/ROADMAP.md`` coupling step 3 but fixes the linear gravity term to
    subtract ``gravity * dt`` (not ``inv_mass * gravity``).
    """
    i = wp.tid()
    rid = robot_ids[i]  # robot Model body (e.g. TCP)
    pid = proxy_ids[i]  # matching gripper proxy on cable Model

    # Kinematic lock: proxy pose follows robot (VBD will integrate from here).
    dst_body_q[pid] = src_body_q[rid]
    qd = src_body_qd[rid]

    # Lagged wrench harvested after the previous VBD substep; MuJoCo already applied it.
    f = proxy_forces[rid]
    inv_m = body_inv_mass[pid]
    # Linear impulse F*dt on proxy mass → Δv; subtract so VBD does not double-count coupling.
    delta_v_coupling = dt * inv_m * wp.spatial_top(f)

    r = wp.transform_get_rotation(dst_body_q[pid])
    # τ in body frame, I⁻¹, back to world: angular Δω from lagged torque over dt.
    tau_b = body_inv_inertia[pid] * wp.quat_rotate_inv(r, wp.spatial_bottom(f))
    delta_w_coupling = dt * wp.quat_rotate(r, tau_b)

    # World-frame twist for VBD: robot vel minus coupling undo; linear also pre-removes g*dt
    # (cable-model gravity; VBD applies gravity again during its step / harvest bookkeeping).
    v = wp.spatial_top(qd) - delta_v_coupling - gravity * dt
    w = wp.spatial_bottom(qd) - delta_w_coupling

    dst_body_qd[pid] = wp.spatial_vector(v, w)


@wp.kernel
def harvest_proxy_wrenches_velocity_delta_kernel(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    body_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_q_post: wp.array(dtype=wp.transform),
    qd_synced: wp.array(dtype=wp.spatial_vector),
    qd_post: wp.array(dtype=wp.spatial_vector),
    gravity: wp.vec3,
    dt: float,
    out_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Net spatial wrench on proxy (world frame, about COM) implied by the VBD velocity jump."""
    i = wp.tid()
    rid = robot_ids[i]  # robot slot (MuJoCo applies out_wrench[rid] next substep)
    pid = proxy_ids[i]

    m = body_mass[pid]
    if m < 1.0e-20:
        out_wrench[rid] = wp.spatial_vector()
        return

    # Pre/post VBD twist on proxy; qd_synced is the velocity set by sync_proxy_state.
    v0 = wp.spatial_top(qd_synced[pid])
    w0 = wp.spatial_bottom(qd_synced[pid])
    v1 = wp.spatial_top(qd_post[pid])
    w1 = wp.spatial_bottom(qd_post[pid])

    # F ≈ m*Δv/dt minus weight: pairs with sync's gravity*dt pre-removal (ROADMAP option 3).
    f_lin = m * (v1 - v0) / dt - m * gravity

    r = wp.transform_get_rotation(body_q_post[pid])
    domega = (w1 - w0) / dt
    # Net torque about COM: τ_world = R * (I_body * R⁻¹ * dω/dt).
    tau = wp.quat_rotate(r, body_inertia[pid] * wp.quat_rotate_inv(r, domega))

    out_wrench[rid] = wp.spatial_vector(f_lin, tau)


@wp.kernel
def _zero_wrench_slots_kernel(
    wrenches: wp.array(dtype=wp.spatial_vector), slot_indices: wp.array(dtype=int)
):
    tid = wp.tid()
    idx = slot_indices[tid]
    wrenches[idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def zero_robot_wrench_slots(wrenches: wp.array, robot_body_indices: wp.array) -> None:
    """Zero ``wrenches[i]`` for each ``i`` in ``robot_body_indices``."""
    dev = wrenches.device
    wp.launch(
        _zero_wrench_slots_kernel,
        dim=robot_body_indices.shape[0],
        inputs=[wrenches, robot_body_indices],
        device=dev,
    )


def launch_harvest_proxy_wrenches_velocity_delta(
    *,
    robot_ids: wp.array,
    proxy_ids: wp.array,
    model,
    body_q_post: wp.array,
    qd_synced: wp.array,
    qd_post: wp.array,
    gravity: wp.vec3,
    dt: float,
    out_robot_wrenches: wp.array,
    device=None,
) -> None:
    """Harvest using per-body inertia/mass from ``model``."""
    dev = device if device is not None else str(out_robot_wrenches.device)
    zero_robot_wrench_slots(out_robot_wrenches, robot_ids)
    wp.launch(
        harvest_proxy_wrenches_velocity_delta_kernel,
        dim=robot_ids.shape[0],
        inputs=[
            robot_ids,
            proxy_ids,
            model.body_mass,
            model.body_inertia,
            body_q_post,
            qd_synced,
            qd_post,
            gravity,
            dt,
            out_robot_wrenches,
        ],
        device=dev,
    )


def harvest_proxy_wrenches(
    vbd_solver,
    vbd_state_post,
    vbd_contacts,
    dt: float,
    *,
    registry: ProxyBodyRegistry,
    model,
    qd_synced: wp.array,
    gravity: wp.vec3,
    out_robot_wrenches: wp.array,
) -> wp.array:
    """Harvest lagged proxy wrenches (velocity-delta path; ``vbd_contacts`` reserved for Slice 2).

    Returns ``out_robot_wrenches`` (indexed by **robot** body id).
    """
    del vbd_solver, vbd_contacts  # unused until direct-accumulator readout lands
    launch_harvest_proxy_wrenches_velocity_delta(
        robot_ids=registry.robot_ids_wp(out_robot_wrenches.device),
        proxy_ids=registry.proxy_ids_wp(out_robot_wrenches.device),
        model=model,
        body_q_post=vbd_state_post.body_q,
        qd_synced=qd_synced,
        qd_post=vbd_state_post.body_qd,
        gravity=gravity,
        dt=dt,
        out_robot_wrenches=out_robot_wrenches,
    )
    return out_robot_wrenches


def harvest_stem_joint_wrench(
    *,
    cable_model,
    cable_solver,
    body_q_post: wp.array,
    body_q_prev: wp.array,
    dt: float,
    stem_apple_joint_index: int,
    tcp_body_index: int,
    out_robot_wrenches: wp.array,
    coupling_gain: float = 1.0,
    force_cap_N: float | None = None,
    torque_cap_Nm: float | None = None,
) -> None:
    """Write the stem-apple FIXED joint constraint wrench into ``out_robot_wrenches[tcp]``.

    Replaces velocity-delta harvest when the proxy and apple are co-teleported
    (``fix_to_apple=True``).  The stem joint force is the mechanical tension the tree
    exerts on the apple — equal in magnitude and direction to the load the apple+stem
    system imposes on the robot TCP through the rigid grasp.

    The wrench is evaluated at the **apple COM** (child body of the stem-apple joint)
    in the world frame.  Torque transport to the exact TCP COM is a future refinement.

    Args:
        cable_model: Finalized Newton ``Model`` for the cable scene.
        cable_solver: ``SolverVBD`` that produced the post-step state.
        body_q_post: Post-VBD body transforms (``cable.state_0.body_q`` after swap).
        body_q_prev: Pre-VBD body transforms (``cable.state_1.body_q`` after swap).
        dt: Substep size [s].
        stem_apple_joint_index: Index of the stem-to-apple FIXED joint.
        tcp_body_index: Robot body index where the result is written.
        out_robot_wrenches: Per-robot-body spatial wrench array (indexed by robot body id).
    """
    from apple_pick_sim.vbd_fixed_joint_wrenches import fixed_joint_wrenches_child_com_vbd

    # VBD constraint impulse on stem→apple FIXED joint (child = apple COM, world frame).
    records = fixed_joint_wrenches_child_com_vbd(
        cable_model,
        cable_solver,
        body_q=body_q_post,
        body_q_prev=body_q_prev,
        dt=dt,
        joint_pairs=[(stem_apple_joint_index, "_stem_apple")],
    )

    n = out_robot_wrenches.shape[0]
    wrenches = out_robot_wrenches.numpy().reshape(n, 6)
    wrenches[:] = 0.0  # only TCP slot is filled; other robot bodies stay zero this substep
    if records:
        rec = records[0]
        # Tree tension on apple ≈ load on TCP through rigid grasp (applied lagged next substep).
        wrenches[tcp_body_index, :3] = rec.force_world
        wrenches[tcp_body_index, 3:6] = rec.torque_at_child_com_world
    limit_stem_coupling_wrench(
        wrenches,
        tcp_body_index,
        coupling_gain=coupling_gain,
        force_cap_N=force_cap_N,
        torque_cap_Nm=torque_cap_Nm,
    )
    out_robot_wrenches.assign(wrenches.ravel())


def limit_stem_coupling_wrench(
    wrenches: Any,
    tcp_body_index: int,
    *,
    coupling_gain: float,
    force_cap_N: float | None,
    torque_cap_Nm: float | None,
) -> None:
    """Under-relax and clamp stem-harvest wrenches for explicit lagged MuJoCo feedback."""
    import numpy as np

    w = wrenches[tcp_body_index]
    # coupling_gain < 1 softens one-substep-lag explicit feedback; caps bound spike forces.
    f = np.asarray(w[:3], dtype=np.float64) * float(coupling_gain)
    tau = np.asarray(w[3:6], dtype=np.float64) * float(coupling_gain)
    if force_cap_N is not None and force_cap_N > 0.0:
        fn = float(np.linalg.norm(f))
        if fn > force_cap_N:
            f = f * (force_cap_N / fn)  # direction preserved, magnitude clipped
    if torque_cap_Nm is not None and torque_cap_Nm > 0.0:
        tn = float(np.linalg.norm(tau))
        if tn > torque_cap_Nm:
            tau = tau * (torque_cap_Nm / tn)
    wrenches[tcp_body_index, :3] = f.astype(np.float32)
    wrenches[tcp_body_index, 3:6] = tau.astype(np.float32)


def align_proxy_body_q_prev_for_vbd(
    cable_scene,
    proxy_body_ids: tuple[int, ...] | wp.array,
) -> None:
    """Align ``SolverVBD.body_q_prev`` with ``state.body_q`` on proxy bodies after kinematic sync.

    ``sync_proxy_state`` overwrites ``body_q`` / ``body_qd`` but leaves the solver's internal
    ``body_q_prev`` at the pre-sync pose. VBD finalizes twist as ``(body_q - body_q_prev) / dt``,
    so a few microns of pose mismatch at ``dt ≈ 1/600`` s appears as O(10 m/s) and ~mg spurious
    harvest wrenches that grow each substep.
    """
    if not proxy_body_ids:
        return
    ids = (
        tuple(int(i) for i in proxy_body_ids.numpy().tolist())
        if isinstance(proxy_body_ids, wp.array)
        else tuple(int(i) for i in proxy_body_ids)
    )
    bq = cable_scene.state_0.body_q.numpy().reshape(-1, 7)
    bqp = cable_scene.solver.body_q_prev.numpy().reshape(-1, 7).copy()
    for pid in ids:
        # Kinematic sync moved body_q but not body_q_prev → zero artificial Δq/dt at finalize.
        bqp[pid] = bq[pid]
    cable_scene.solver.body_q_prev.assign(bqp.ravel())


@wp.kernel
def sync_proxy_and_apple_state(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    src_body_q: wp.array(dtype=wp.transform),
    src_body_qd: wp.array(dtype=wp.spatial_vector),
    dst_body_q: wp.array(dtype=wp.transform),
    dst_body_qd: wp.array(dtype=wp.spatial_vector),
    proxy_forces: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    dt: float,
    apple_body_id: int,
    proxy_offset_in_apple: wp.vec3,
):
    """Teleport proxy (with double-integration correction) and apple from robot TCP.

    The apple transform is derived from the TCP pose by reversing the fixed grasp
    offset: ``apple_pos = tcp_pos - R_tcp * proxy_offset_in_apple``, keeping the
    apple orientation identical to the TCP orientation.  Both bodies receive the
    same corrected velocity so the VBD FIXED joint between them sees zero violation.

    When ``apple_body_id < 0`` the apple teleport is skipped (fallback to free-proxy
    behaviour).  This kernel replaces :func:`sync_proxy_state` when
    ``fix_to_apple=True``.
    """
    i = wp.tid()
    rid = robot_ids[i]  # robot TCP
    pid = proxy_ids[i]  # gripper proxy on cable Model

    tcp_q = src_body_q[rid]
    tcp_rot = wp.transform_get_rotation(tcp_q)
    tcp_pos = wp.transform_get_translation(tcp_q)

    # --- proxy: kinematic lock + double-integration guard (see sync_proxy_state) ---
    dst_body_q[pid] = tcp_q
    qd = src_body_qd[rid]

    f = proxy_forces[rid]  # lagged harvest; MuJoCo already integrated this on the robot
    inv_m = body_inv_mass[pid]
    delta_v_coupling = dt * inv_m * wp.spatial_top(f)

    tau_b = body_inv_inertia[pid] * wp.quat_rotate_inv(tcp_rot, wp.spatial_bottom(f))
    delta_w_coupling = dt * wp.quat_rotate(tcp_rot, tau_b)

    v_corr = wp.spatial_top(qd) - delta_v_coupling - gravity * dt
    w_corr = wp.spatial_bottom(qd) - delta_w_coupling
    dst_body_qd[pid] = wp.spatial_vector(v_corr, w_corr)

    # --- apple: rigid grasp offset from builder; same v_corr/w_corr keeps FIXED joint consistent ---
    if apple_body_id >= 0:
        # proxy_offset_in_apple: vector from apple COM to proxy in apple/body frame.
        apple_pos = tcp_pos - wp.quat_rotate(tcp_rot, proxy_offset_in_apple)
        dst_body_q[apple_body_id] = wp.transform(apple_pos, tcp_rot)
        dst_body_qd[apple_body_id] = wp.spatial_vector(v_corr, w_corr)


def launch_sync_proxy_and_apple_state(
    *,
    robot_ids: wp.array,
    proxy_ids: wp.array,
    src_body_q,
    src_body_qd,
    dst_body_q,
    dst_body_qd,
    proxy_forces,
    cable_model,
    gravity: wp.vec3,
    dt: float,
    apple_body_id: int,
    proxy_offset_in_apple: wp.vec3,
    device=None,
) -> None:
    """Teleport proxy + apple; ``body_inv_mass`` / ``body_inv_inertia`` from ``cable_model``."""
    wp.launch(
        sync_proxy_and_apple_state,
        dim=robot_ids.shape[0],
        inputs=[
            robot_ids,
            proxy_ids,
            src_body_q,
            src_body_qd,
            dst_body_q,
            dst_body_qd,
            proxy_forces,
            cable_model.body_inv_mass,
            cable_model.body_inv_inertia,
            gravity,
            dt,
            apple_body_id,
            proxy_offset_in_apple,
        ],
        device=device if device is not None else cable_model.device,
    )


def launch_sync_proxy_state(
    *,
    robot_ids: wp.array,
    proxy_ids: wp.array,
    src_body_q,
    src_body_qd,
    dst_body_q,
    dst_body_qd,
    proxy_forces,
    cable_model,
    gravity: wp.vec3,
    dt: float,
    device=None,
) -> None:
    """Convenience wrapper: ``body_inv_mass`` / ``body_inv_inertia`` from ``cable_model``."""
    wp.launch(
        sync_proxy_state,
        dim=robot_ids.shape[0],
        inputs=[
            robot_ids,
            proxy_ids,
            src_body_q,
            src_body_qd,
            dst_body_q,
            dst_body_qd,
            proxy_forces,
            cable_model.body_inv_mass,
            cable_model.body_inv_inertia,
            gravity,
            dt,
        ],
        device=device if device is not None else cable_model.device,
    )
