"""M1 proxy coupling: mirror robot state onto VBD proxies and harvest reaction wrenches.

Implements the **staggered two-Model coupling** helpers from ``docs/ROADMAP.md``
([M1]): ``mirror_robot_tcp_to_proxy_kernel`` (pose/velocity mirror with
double-integration guard) and velocity-delta harvest for **option 3** in the roadmap.

Spatial vectors are **world frame**, linear first / angular second [N, N·m].
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp


@dataclasses.dataclass(frozen=True)
class ProxyBodyRegistry:
    """Maps robot ``Model`` body indices to proxy bodies on the cable ``Model``."""

    robot_to_proxy: tuple[tuple[int, int], ...]
    """Sorted ``(robot_body_id, proxy_body_id)`` pairs."""
    _device_ids: dict[str, tuple[wp.array, wp.array]] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def from_mapping(cls, mapping: dict[int, int]) -> ProxyBodyRegistry:
        """Build a registry from ``{robot_body_id: proxy_body_id}`` pairs.

        Pairs are sorted by robot body id for deterministic kernel launch order.
        Used when a single robot TCP maps to one cable gripper proxy (standard
        ``build_coupled_fruiting_*`` scenes in :mod:`builders`).
        """
        return cls(robot_to_proxy=tuple(sorted(mapping.items())))

    @classmethod
    def from_repeated_robot(
        cls,
        robot_body_id: int,
        proxy_body_ids: Sequence[int],
    ) -> ProxyBodyRegistry:
        """One robot body mirrored onto many proxies (fd_ghost mega layout)."""
        rid = int(robot_body_id)
        return cls(robot_to_proxy=tuple((rid, int(p)) for p in proxy_body_ids))

    @property
    def robot_body_ids(self) -> tuple[int, ...]:
        """Ordered tuple of robot body IDs from ``robot_to_proxy``.

        Index ``i`` aligns with :attr:`proxy_body_ids` ``[i]`` for kernel row ``i``.
        Used by harvest and mirror launchers that need host-side id lists.
        """
        return tuple(r for r, _ in self.robot_to_proxy)

    @property
    def proxy_body_ids(self) -> tuple[int, ...]:
        """Ordered tuple of cable proxy body IDs from ``robot_to_proxy``.

        Index ``i`` aligns with :attr:`robot_body_ids` ``[i]`` for kernel row ``i``.
        Passed to :func:`align_proxy_body_q_prev_for_vbd` after kinematic TCP sync.
        """
        return tuple(p for _, p in self.robot_to_proxy)

    def ids_wp(self, device) -> tuple[wp.array, wp.array]:
        """Return cached ``(robot_ids, proxy_ids)`` arrays for ``device`` (built once)."""
        key = str(device)
        cached = self._device_ids.get(key)
        if cached is None:
            cached = (
                wp.array(self.robot_body_ids, dtype=int, device=device),
                wp.array(self.proxy_body_ids, dtype=int, device=device),
            )
            self._device_ids[key] = cached
        return cached

    def robot_ids_wp(self, device) -> wp.array:
        """Cached device array of robot body IDs (see :meth:`ids_wp`).

        Fed to mirror/harvest kernels as the ``robot_ids`` argument and to
        :func:`zero_robot_wrench_slots` before velocity-delta harvest.
        """
        return self.ids_wp(device)[0]

    def proxy_ids_wp(self, device) -> wp.array:
        """Cached device array of proxy body IDs (see :meth:`ids_wp`).

        Fed to mirror/harvest kernels as the ``proxy_ids`` argument; indexes
        into the cable ``Model`` ``body_q`` / ``body_qd`` buffers.
        """
        return self.ids_wp(device)[1]


@wp.func
def _corrected_proxy_twist_from_robot(
    qd_robot: wp.spatial_vector,
    proxy_force: wp.spatial_vector,
    body_inv_mass: float,
    body_inv_inertia: wp.mat33,
    body_rot: wp.quat,
    gravity: wp.vec3,
    dt: float,
) -> wp.spatial_vector:
    """Proxy twist after subtracting lagged coupling impulse and gravity pre-step."""
    inv_m = body_inv_mass
    delta_v_coupling = dt * inv_m * wp.spatial_top(proxy_force)
    tau_b = body_inv_inertia * wp.quat_rotate_inv(body_rot, wp.spatial_bottom(proxy_force))
    delta_w_coupling = dt * wp.quat_rotate(body_rot, tau_b)
    v = wp.spatial_top(qd_robot) - delta_v_coupling - gravity * dt
    w = wp.spatial_bottom(qd_robot) - delta_w_coupling
    return wp.spatial_vector(v, w)


@wp.kernel
def mirror_robot_tcp_to_proxy_kernel(
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

    Model A → proxy on Model B. Mirrors ``docs/ROADMAP.md`` coupling step 3.
    """
    i = wp.tid()
    rid = robot_ids[i]
    pid = proxy_ids[i]

    dst_body_q[pid] = src_body_q[rid]
    r = wp.transform_get_rotation(dst_body_q[pid])
    dst_body_qd[pid] = _corrected_proxy_twist_from_robot(
        src_body_qd[rid],
        proxy_forces[rid],
        body_inv_mass[pid],
        body_inv_inertia[pid],
        r,
        gravity,
        dt,
    )


@wp.kernel
def mirror_robot_tcp_to_proxy_offset_kernel(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    position_offsets: wp.array(dtype=wp.vec3),
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
    """Mirror robot TCP onto proxies with per-pair world-frame position offsets."""
    i = wp.tid()
    rid = robot_ids[i]
    pid = proxy_ids[i]
    delta = position_offsets[i]

    tcp_q = src_body_q[rid]
    tcp_rot = wp.transform_get_rotation(tcp_q)
    tcp_pos = wp.transform_get_translation(tcp_q)
    proxy_pos = tcp_pos + delta

    dst_body_q[pid] = wp.transform(proxy_pos, tcp_rot)
    dst_body_qd[pid] = _corrected_proxy_twist_from_robot(
        src_body_qd[rid],
        proxy_forces[rid],
        body_inv_mass[pid],
        body_inv_inertia[pid],
        tcp_rot,
        gravity,
        dt,
    )


@wp.kernel
def mirror_robot_tcp_to_proxy_offset_and_apple_kernel(
    robot_ids: wp.array(dtype=int),
    proxy_ids: wp.array(dtype=int),
    position_offsets: wp.array(dtype=wp.vec3),
    apple_body_ids: wp.array(dtype=int),
    proxy_offset_in_apple: wp.array(dtype=wp.transform),
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
    """Offset ghost mirror plus optional apple co-teleport from each proxy pose."""
    i = wp.tid()
    rid = robot_ids[i]
    pid = proxy_ids[i]
    delta = position_offsets[i]

    tcp_q = src_body_q[rid]
    tcp_rot = wp.transform_get_rotation(tcp_q)
    tcp_pos = wp.transform_get_translation(tcp_q)
    proxy_pos = tcp_pos + delta

    qd_corr = _corrected_proxy_twist_from_robot(
        src_body_qd[rid],
        proxy_forces[rid],
        body_inv_mass[pid],
        body_inv_inertia[pid],
        tcp_rot,
        gravity,
        dt,
    )
    dst_body_q[pid] = wp.transform(proxy_pos, tcp_rot)
    dst_body_qd[pid] = qd_corr

    aid = apple_body_ids[i]
    if aid >= 0:
        proxy_tf = wp.transform(proxy_pos, tcp_rot)
        apple_tf = wp.transform_multiply(
            proxy_tf, wp.transform_inverse(proxy_offset_in_apple[i])
        )
        dst_body_q[aid] = apple_tf

        r_local = wp.transform_get_translation(wp.transform_inverse(proxy_offset_in_apple[i]))
        r_world = wp.quat_rotate(tcp_rot, r_local)
        v_proxy = wp.spatial_top(qd_corr)
        w_proxy = wp.spatial_bottom(qd_corr)
        v_apple = v_proxy + wp.cross(w_proxy, r_world)
        dst_body_qd[aid] = wp.spatial_vector(v_apple, w_proxy)


@wp.kernel
def compute_proxy_reaction_wrench_kernel(
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
    rid = robot_ids[i]
    pid = proxy_ids[i]

    m = body_mass[pid]
    if m < 1.0e-20:
        out_wrench[rid] = wp.spatial_vector()
        return

    v0 = wp.spatial_top(qd_synced[pid])
    w0 = wp.spatial_bottom(qd_synced[pid])
    v1 = wp.spatial_top(qd_post[pid])
    w1 = wp.spatial_bottom(qd_post[pid])

    f_lin = m * (v1 - v0) / dt - m * gravity

    r = wp.transform_get_rotation(body_q_post[pid])
    domega = (w1 - w0) / dt
    tau = wp.quat_rotate(r, body_inertia[pid] * wp.quat_rotate_inv(r, domega))

    out_wrench[rid] = wp.spatial_vector(f_lin, tau)


@wp.kernel
def _zero_wrench_slots_kernel(
    wrenches: wp.array(dtype=wp.spatial_vector), slot_indices: wp.array(dtype=int)
):
    """Zero ``wrenches[slot_indices[tid]]`` in parallel.

    Clears only robot-indexed harvest slots before writing fresh reaction
    wrenches. Called from :func:`zero_robot_wrench_slots`, which
    :func:`launch_compute_proxy_reaction_wrench` uses at the start of each
    velocity-delta harvest substep.
    """
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


def launch_compute_proxy_reaction_wrench(
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
        compute_proxy_reaction_wrench_kernel,
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
    del vbd_solver, vbd_contacts
    launch_compute_proxy_reaction_wrench(
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


@wp.kernel
def _zero_all_wrenches_kernel(wrenches: wp.array(dtype=wp.spatial_vector)):
    """Zero every wrench slot in ``wrenches`` (one thread per slot).

    Unlike :func:`_zero_wrench_slots_kernel`, clears the full buffer. Launched
    at the start of :func:`harvest_stem_tension_for_tcp` before stem
    FIXED-joint gather when ``fix_to_apple`` uses stem harvest instead of
    velocity-delta.
    """
    wrenches[wp.tid()] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _limit_and_write_tcp_stem_wrench_kernel(
    wrenches: wp.array(dtype=wp.spatial_vector),
    tcp_index: int,
    force_raw: wp.array(dtype=wp.vec3),
    torque_raw: wp.array(dtype=wp.vec3),
    coupling_gain: float,
    force_cap_N: float,
    torque_cap_Nm: float,
    use_force_cap: int,
    use_torque_cap: int,
    use_explicit_apple_weight: int,
    apple_mass_kg: float,
    gravity: wp.vec3,
    robot_body_q: wp.array(dtype=wp.transform),
    cable_body_q: wp.array(dtype=wp.transform),
    apple_body_index: int,
    grasp_offset: wp.transform,
    use_grasp_offset: int,
):
    """Under-relax and clamp stem harvest; write spatial wrench at ``tcp_index``.

    Stem gather returns force/torque **on the apple (child)**; write that child-side
    wrench into TCP ``body_f`` (same sign as main; negating here destabilizes couple).
    """
    f_stem_at_com = force_raw[0]
    tau_stem_at_com = torque_raw[0]
    
    f_total_tcp = f_stem_at_com
    tau_total_tcp = tau_stem_at_com
    
    if apple_body_index >= 0:
        tcp_xf = robot_body_q[tcp_index]
        p_tcp = wp.transform_get_translation(tcp_xf)
        if use_grasp_offset != 0:
            apple_xf = wp.transform_multiply(tcp_xf, wp.transform_inverse(grasp_offset))
            p_apple = wp.transform_get_translation(apple_xf)
        else:
            p_apple = wp.transform_get_translation(cable_body_q[apple_body_index])
        
        r_tcp_to_apple_com = p_apple - p_tcp

        # Transfer stem force from apple COM to TCP (always; matches CPU harvest path).
        tau_total_tcp = tau_total_tcp + wp.cross(r_tcp_to_apple_com, f_stem_at_com)

        if use_explicit_apple_weight != 0 and apple_mass_kg > 0.0:
            g = gravity
            f_apple_weight = wp.vec3(
                -apple_mass_kg * g[0],
                -apple_mass_kg * g[1],
                -apple_mass_kg * g[2],
            )
            f_total_tcp = f_total_tcp + f_apple_weight
            tau_total_tcp = tau_total_tcp + wp.cross(r_tcp_to_apple_com, f_apple_weight)
            
    f_total_tcp = f_total_tcp * coupling_gain
    tau_total_tcp = tau_total_tcp * coupling_gain
    
    if use_force_cap != 0 and force_cap_N > 0.0:
        fn = wp.length(f_total_tcp)
        if fn > force_cap_N:
            f_total_tcp = f_total_tcp * (force_cap_N / fn)
    if use_torque_cap != 0 and torque_cap_Nm > 0.0:
        tn = wp.length(tau_total_tcp)
        if tn > torque_cap_Nm:
            tau_total_tcp = tau_total_tcp * (torque_cap_Nm / tn)
            
    wrenches[tcp_index] = wp.spatial_vector(
        f_total_tcp[0], f_total_tcp[1], f_total_tcp[2],
        tau_total_tcp[0], tau_total_tcp[1], tau_total_tcp[2],
    )


@wp.kernel
def _batched_limit_and_write_tcp_stem_wrench_kernel(
    wrenches: wp.array(dtype=wp.spatial_vector),
    tcp_indices: wp.array(dtype=int),
    force_raw: wp.array(dtype=wp.vec3),
    torque_raw: wp.array(dtype=wp.vec3),
    coupling_gain: float,
    force_cap_N: float,
    torque_cap_Nm: float,
    use_force_cap: int,
    use_torque_cap: int,
    use_explicit_apple_weight: wp.array(dtype=int),
    apple_mass_kg: wp.array(dtype=float),
    gravity: wp.vec3,
    robot_body_q: wp.array(dtype=wp.transform),
    cable_body_q: wp.array(dtype=wp.transform),
    apple_body_indices: wp.array(dtype=int),
    grasp_offsets: wp.array(dtype=wp.transform),
    use_grasp_offset: wp.array(dtype=int),
):
    """Under-relax and clamp batched stem harvest; write spatial wrench per TCP row.

    Same child-side sign convention as :func:`_limit_and_write_tcp_stem_wrench_kernel`.
    """
    i = wp.tid()
    tcp_index = tcp_indices[i]
    f_stem_at_com = force_raw[i]
    tau_stem_at_com = torque_raw[i]

    f_total_tcp = f_stem_at_com
    tau_total_tcp = tau_stem_at_com

    apple_body_index = apple_body_indices[i]
    if apple_body_index >= 0:
        tcp_xf = robot_body_q[tcp_index]
        p_tcp = wp.transform_get_translation(tcp_xf)
        if use_grasp_offset[i] != 0:
            apple_xf = wp.transform_multiply(tcp_xf, wp.transform_inverse(grasp_offsets[i]))
            p_apple = wp.transform_get_translation(apple_xf)
        else:
            p_apple = wp.transform_get_translation(cable_body_q[apple_body_index])

        r_tcp_to_apple_com = p_apple - p_tcp
        tau_total_tcp = tau_total_tcp + wp.cross(r_tcp_to_apple_com, f_stem_at_com)

        if use_explicit_apple_weight[i] != 0 and apple_mass_kg[i] > 0.0:
            g = gravity
            m = apple_mass_kg[i]
            f_apple_weight = wp.vec3(-m * g[0], -m * g[1], -m * g[2])
            f_total_tcp = f_total_tcp + f_apple_weight
            tau_total_tcp = tau_total_tcp + wp.cross(r_tcp_to_apple_com, f_apple_weight)

    f_total_tcp = f_total_tcp * coupling_gain
    tau_total_tcp = tau_total_tcp * coupling_gain

    if use_force_cap != 0 and force_cap_N > 0.0:
        fn = wp.length(f_total_tcp)
        if fn > force_cap_N:
            f_total_tcp = f_total_tcp * (force_cap_N / fn)
    if use_torque_cap != 0 and torque_cap_Nm > 0.0:
        tn = wp.length(tau_total_tcp)
        if tn > torque_cap_Nm:
            tau_total_tcp = tau_total_tcp * (torque_cap_Nm / tn)

    wrenches[tcp_index] = wp.spatial_vector(
        f_total_tcp[0],
        f_total_tcp[1],
        f_total_tcp[2],
        tau_total_tcp[0],
        tau_total_tcp[1],
        tau_total_tcp[2],
    )


def _grasp_offset_to_transform(offset: tuple | None) -> wp.transform:
    if offset is None:
        return wp.transform_identity()
    if len(offset) == 7:
        return wp.transform(
            wp.vec3(float(offset[0]), float(offset[1]), float(offset[2])),
            wp.quat(float(offset[3]), float(offset[4]), float(offset[5]), float(offset[6])),
        )
    return wp.transform(
        wp.vec3(float(offset[0]), float(offset[1]), float(offset[2])),
        wp.quat_identity(),
    )


def harvest_batched_stem_tension(
    *,
    stem_joint_indices_wp: wp.array,
    tcp_indices_wp: wp.array,
    apple_indices_wp: wp.array,
    grasp_offsets_wp: wp.array,
    apple_masses_wp: wp.array,
    use_grasp_offset_wp: wp.array,
    cable_model,
    cable_solver,
    body_q_post: wp.array,
    body_q_prev: wp.array,
    dt: float,
    out_robot_wrenches: wp.array,
    coupling_gain: float = 1.0,
    force_cap_N: float | None = None,
    torque_cap_Nm: float | None = None,
    explicit_apple_weight: bool = True,
    use_explicit_apple_weight_wp: wp.array | None = None,
    gravity: wp.vec3 | None = None,
    robot_body_q: wp.array | None = None,
    device: str | None = None,
    out_f: wp.array | None = None,
    out_t: wp.array | None = None,
) -> None:
    """Batched stem harvest for ``N`` envs: one gather + one write kernel launch.

    Prefer ``use_explicit_apple_weight_wp`` from
    :func:`prepare_batched_stem_harvest_arrays` to avoid per-call ``wp.full``.
    """
    from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device

    dev = device if device is not None else str(out_robot_wrenches.device)
    n = int(tcp_indices_wp.shape[0])
    if n == 0:
        return

    wp.launch(_zero_all_wrenches_kernel, dim=int(out_robot_wrenches.shape[0]), inputs=[out_robot_wrenches], device=dev)

    out_f, out_t = gather_joint_wrench_child_com_device(
        cable_model,
        cable_solver,
        body_q=body_q_post,
        body_q_prev=body_q_prev,
        joint_indices=stem_joint_indices_wp,
        dt=dt,
        control=cable_model.control(clone_variables=False),
        out_f=out_f,
        out_t=out_t,
    )
    g = gravity if gravity is not None else wp.vec3(0.0, 0.0, -9.81)
    robot_bq = robot_body_q if robot_body_q is not None else body_q_post
    if use_explicit_apple_weight_wp is not None:
        use_explicit_arr = use_explicit_apple_weight_wp
    else:
        use_explicit_arr = wp.full(n, 1 if explicit_apple_weight else 0, dtype=int, device=dev)

    f_cap = float(force_cap_N) if force_cap_N is not None else 0.0
    t_cap = float(torque_cap_Nm) if torque_cap_Nm is not None else 0.0
    wp.launch(
        _batched_limit_and_write_tcp_stem_wrench_kernel,
        dim=n,
        inputs=[
            out_robot_wrenches,
            tcp_indices_wp,
            out_f,
            out_t,
            float(coupling_gain),
            f_cap,
            t_cap,
            1 if force_cap_N is not None and force_cap_N > 0.0 else 0,
            1 if torque_cap_Nm is not None and torque_cap_Nm > 0.0 else 0,
            use_explicit_arr,
            apple_masses_wp,
            g,
            robot_bq,
            body_q_post,
            apple_indices_wp,
            grasp_offsets_wp,
            use_grasp_offset_wp,
        ],
        device=dev,
    )


def prepare_batched_stem_harvest_arrays(scene: Any, layout: Any) -> None:
    """Cache per-env stem harvest index/offset/mass arrays on ``scene`` (build-time)."""
    if layout is None or int(layout.num_envs) < 2:
        return
    tpl_stem = scene.stem_apple_joint_index
    if tpl_stem is None:
        return

    cable = scene.cable
    dev = str(cable.model.device)
    n = int(layout.num_envs)
    default_off = cable.gripper_proxy_offset_in_apple_frame
    per_offsets = getattr(scene, "per_world_proxy_offsets", None)

    stem_joints = [layout.joint_index(w, tpl_stem) for w in range(n)]
    grasp_list: list[wp.transform] = []
    use_grasp: list[int] = []
    for w in range(n):
        off = per_offsets[w] if per_offsets is not None and per_offsets[w] is not None else default_off
        if off is None:
            grasp_list.append(wp.transform_identity())
            use_grasp.append(0)
        else:
            grasp_list.append(_grasp_offset_to_transform(off))
            use_grasp.append(1)

    masses_np = cable.model.body_mass.numpy()
    apple_masses = [
        float(masses_np[int(layout.apple_body_indices[w])])
        if int(layout.apple_body_indices[w]) >= 0
        else 0.0
        for w in range(n)
    ]

    scene.stem_harvest_joint_indices_wp = wp.array(stem_joints, dtype=int, device=dev)
    scene.stem_harvest_tcp_indices_wp = wp.array(list(layout.tcp_body_indices), dtype=int, device=dev)
    scene.stem_harvest_apple_indices_wp = wp.array(list(layout.apple_body_indices), dtype=int, device=dev)
    scene.stem_harvest_grasp_offsets_wp = wp.array(grasp_list, dtype=wp.transform, device=dev)
    scene.stem_harvest_apple_masses_wp = wp.array(apple_masses, dtype=float, device=dev)
    scene.stem_harvest_use_grasp_offset_wp = wp.array(use_grasp, dtype=int, device=dev)
    scene.stem_harvest_wrench_f_scratch = wp.zeros(n, dtype=wp.vec3, device=dev)
    scene.stem_harvest_wrench_t_scratch = wp.zeros(n, dtype=wp.vec3, device=dev)
    explicit_on = 1 if bool(getattr(scene, "stem_harvest_explicit_apple_weight", False)) else 0
    scene.stem_harvest_use_explicit_wp = wp.full(n, explicit_on, dtype=int, device=dev)

    # Co-teleport arrays for welded multi-env mirror (reuse every substep).
    if (
        getattr(cable, "gripper_proxy_apple_joint", None) is not None
        and default_off is not None
    ):
        apple_ids, pos_off, grasp_off = welded_co_teleport_arrays_for_layout(
            layout,
            cable,
            device=dev,
            per_world_proxy_offsets=per_offsets,
        )
        scene.co_teleport_apple_ids_wp = apple_ids
        scene.co_teleport_pos_offsets_wp = pos_off
        scene.co_teleport_grasp_offsets_wp = grasp_off
    else:
        scene.co_teleport_apple_ids_wp = None
        scene.co_teleport_pos_offsets_wp = None
        scene.co_teleport_grasp_offsets_wp = None


def _harvest_stem_tension_for_tcp_cpu(
    *,
    cable_model,
    cable_solver,
    body_q_post: wp.array,
    body_q_prev: wp.array,
    dt: float,
    stem_apple_joint_index: int,
    tcp_body_index: int,
    out_robot_wrenches: wp.array,
    coupling_gain: float,
    force_cap_N: float | None,
    torque_cap_Nm: float | None,
    explicit_apple_weight: bool = True,
    apple_body_index: int | None = None,
    apple_mass_kg: float | None = None,
    gravity: wp.vec3 | None = None,
    robot_body_q: wp.array | None = None,
    grasp_offset_in_apple_frame: tuple | None = None,
) -> None:
    """CPU fallback for stem harvest: NumPy gather, lever-arm transfer, gain/caps.

    Mirrors :func:`harvest_stem_tension_for_tcp` on the host: gathers the
    stem–apple FIXED joint wrench, transfers force from apple COM to TCP,
    optionally adds explicit apple weight, then applies gain/caps via
    :func:`limit_stem_coupling_wrench`.

    Used by :mod:`tests.test_proxy_coupling` for CPU/GPU parity checks.
    Production hot path (``coupled_substep`` with ``fix_to_apple``) uses the
    device gather + :func:`_limit_and_write_tcp_stem_wrench_kernel` instead.
    """
    from apple_pick_sim.coupled_fruiting.explicit_load import (
        apple_mass_kg_from_model,
        explicit_apple_wrench_for_stem_harvest,
    )
    from apple_pick_sim.vbd_fixed_joint_wrenches import fixed_joint_wrenches_child_com_vbd

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
    wrenches[:] = 0.0
    if records:
        rec = records[0]
        f_stem_at_com = np.asarray(rec.force_world, dtype=np.float64)
        tau_stem_at_com = np.asarray(rec.torque_at_child_com_world, dtype=np.float64)
        
        f_total_tcp = f_stem_at_com.copy()
        tau_total_tcp = tau_stem_at_com.copy()
        
        if robot_body_q is not None and apple_body_index is not None:
            # Transfer the stem force from the apple COM to the TCP
            from apple_pick_sim.coupled_fruiting.explicit_load import (
                apple_com_from_tcp_grasp_offset,
                body_com_position_world,
                body_orientation_world,
            )
            
            p_tcp = body_com_position_world(robot_body_q, tcp_body_index)
            if grasp_offset_in_apple_frame is not None:
                tcp_rot = body_orientation_world(robot_body_q, tcp_body_index)
                p_apple = apple_com_from_tcp_grasp_offset(
                    p_tcp, tcp_rot, grasp_offset_in_apple_frame
                )
            else:
                p_apple = body_com_position_world(body_q_post, apple_body_index)
                
            r_tcp_to_apple_com = p_apple - p_tcp
            tau_total_tcp = tau_total_tcp + np.cross(r_tcp_to_apple_com, f_stem_at_com)
            
            if explicit_apple_weight:
                g = gravity if gravity is not None else wp.vec3(0.0, 0.0, -9.81)
                m = (
                    float(apple_mass_kg)
                    if apple_mass_kg is not None
                    else apple_mass_kg_from_model(cable_model, apple_body_index)
                )
                if m > 0.0:
                    from apple_pick_sim.coupled_fruiting.explicit_load import (
                        apple_explicit_wrench_about_tcp,
                    )
                    f_apple_weight, tau_apple_weight_at_tcp = apple_explicit_wrench_about_tcp(
                        m, g, p_tcp, apple_pos_world=p_apple
                    )
                    f_total_tcp = f_total_tcp + f_apple_weight
                    tau_total_tcp = tau_total_tcp + tau_apple_weight_at_tcp
        wrenches[tcp_body_index, :3] = f_total_tcp.astype(np.float32)
        wrenches[tcp_body_index, 3:6] = tau_total_tcp.astype(np.float32)
    limit_stem_coupling_wrench(
        wrenches,
        tcp_body_index,
        coupling_gain=coupling_gain,
        force_cap_N=force_cap_N,
        torque_cap_Nm=torque_cap_Nm,
    )
    out_robot_wrenches.assign(wrenches.reshape(-1, 6))


def harvest_stem_tension_for_tcp(
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
    explicit_apple_weight: bool = True,
    apple_body_index: int | None = None,
    apple_mass_kg: float | None = None,
    gravity: wp.vec3 | None = None,
    robot_body_q: wp.array | None = None,
    grasp_offset_in_apple_frame: tuple | None = None,
    clear_wrenches: bool = True,
) -> None:
    """Write the stem-apple FIXED joint constraint wrench into ``out_robot_wrenches[tcp]``.

    Replaces velocity-delta harvest when the proxy and apple are co-teleported
    (``fix_to_apple=True``). Runs gather + limit on device (no full-buffer host sync).

    When ``explicit_apple_weight`` is true, adds apple support force and offset torque
    about TCP (needs ``robot_body_q``; see ``explicit_load``). With
    ``grasp_offset_in_apple_frame``, the lever arm uses the kinematic grasp offset from
    the TCP orientation (same as ``fix_to_apple`` teleports).

    Pass ``apple_mass_kg`` from scene build so CUDA graph capture never syncs
    ``model.body_mass`` to the host each substep.

    When ``clear_wrenches`` is false, the caller must zero ``out_robot_wrenches``
    before the first call in a batched loop; subsequent calls accumulate.
    """
    from apple_pick_sim.coupled_fruiting.explicit_load import apple_mass_kg_from_model
    from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device

    dev = out_robot_wrenches.device
    n = int(out_robot_wrenches.shape[0])
    if clear_wrenches:
        wp.launch(_zero_all_wrenches_kernel, dim=n, inputs=[out_robot_wrenches], device=dev)

    out_f, out_t = gather_joint_wrench_child_com_device(
        cable_model,
        cable_solver,
        body_q=body_q_post,
        body_q_prev=body_q_prev,
        joint_indices=[stem_apple_joint_index],
        dt=dt,
        control=cable_model.control(clone_variables=False),
    )
    g = gravity if gravity is not None else wp.vec3(0.0, 0.0, -9.81)
    use_explicit = 0
    m_apple = 0.0
    apple_bid = -1
    grasp_off = wp.transform_identity()
    use_grasp_offset = 0
    robot_bq = body_q_post
    if robot_body_q is not None and apple_body_index is not None and int(apple_body_index) >= 0:
        apple_bid = int(apple_body_index)
        robot_bq = robot_body_q
        if grasp_offset_in_apple_frame is not None:
            go = grasp_offset_in_apple_frame
            if len(go) == 7:
                grasp_off = wp.transform(
                    wp.vec3(float(go[0]), float(go[1]), float(go[2])),
                    wp.quat(float(go[3]), float(go[4]), float(go[5]), float(go[6])),
                )
            else:
                grasp_off = wp.transform(
                    wp.vec3(float(go[0]), float(go[1]), float(go[2])),
                    wp.quat_identity(),
                )
            use_grasp_offset = 1
        if explicit_apple_weight:
            m_apple = (
                float(apple_mass_kg)
                if apple_mass_kg is not None
                else apple_mass_kg_from_model(cable_model, apple_body_index)
            )
            if m_apple > 0.0:
                use_explicit = 1
    f_cap = float(force_cap_N) if force_cap_N is not None else 0.0
    t_cap = float(torque_cap_Nm) if torque_cap_Nm is not None else 0.0
    wp.launch(
        _limit_and_write_tcp_stem_wrench_kernel,
        dim=1,
        inputs=[
            out_robot_wrenches,
            int(tcp_body_index),
            out_f,
            out_t,
            float(coupling_gain),
            f_cap,
            t_cap,
            1 if force_cap_N is not None and force_cap_N > 0.0 else 0,
            1 if torque_cap_Nm is not None and torque_cap_Nm > 0.0 else 0,
            use_explicit,
            float(m_apple),
            g,
            robot_bq,
            body_q_post,
            apple_bid,
            grasp_off,
            use_grasp_offset,
        ],
        device=dev,
    )


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
    f = np.asarray(w[:3], dtype=np.float64) * float(coupling_gain)
    tau = np.asarray(w[3:6], dtype=np.float64) * float(coupling_gain)
    if force_cap_N is not None and force_cap_N > 0.0:
        fn = float(np.linalg.norm(f))
        if fn > force_cap_N:
            f = f * (force_cap_N / fn)
    if torque_cap_Nm is not None and torque_cap_Nm > 0.0:
        tn = float(np.linalg.norm(tau))
        if tn > torque_cap_Nm:
            tau = tau * (torque_cap_Nm / tn)
    wrenches[tcp_body_index, :3] = f.astype(np.float32)
    wrenches[tcp_body_index, 3:6] = tau.astype(np.float32)


@wp.kernel
def _copy_body_state_kernel(
    body_ids: wp.array(dtype=int),
    src_body_q: wp.array(dtype=wp.transform),
    src_body_qd: wp.array(dtype=wp.spatial_vector),
    dst_body_q: wp.array(dtype=wp.transform),
    dst_body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Copy ``body_q`` / ``body_qd`` for each listed body ID from src to dst.

    Device-side bulk copy without host round-trip. Called from
    :func:`copy_cable_body_q_between_states`, which
    :func:`~apple_pick_sim.coupled_fruiting.scene._sync_single_proxy_after_mujoco`
    uses after ``fix_to_apple`` co-teleport to keep ``state_1`` aligned with
    prescribed proxy/apple poses for AVBD ``body_q_prev``.
    """
    i = wp.tid()
    bid = body_ids[i]
    dst_body_q[bid] = src_body_q[bid]
    dst_body_qd[bid] = src_body_qd[bid]


@wp.kernel
def _align_body_q_prev_kernel(
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
):
    """Copy ``body_q[bid]`` into ``body_q_prev[bid]`` for each listed body index."""
    i = wp.tid()
    bid = body_ids[i]
    body_q_prev[bid] = body_q[bid]


def sync_cable_body_q_prev_from_state(
    cable_scene,
    *,
    body_ids: tuple[int, ...] | wp.array | None = None,
) -> None:
    """Align ``SolverVBD.body_q_prev`` with ``state_0.body_q`` after ``eval_fk`` or kinematic sync.

    When ``body_ids`` is omitted, every cable body is updated. Call this after any
    ``newton.eval_fk`` that writes ``state_0.body_q`` but leaves ``body_q_prev`` at the
    solver's construction-time snapshot (avoids a spurious first-substep velocity spike).
    """
    sync_solver_body_q_prev_from_state(
        cable_scene,
        cable_scene.state_0.body_q,
        body_ids,
    )


def eval_fk_cable_state_0(cable_scene) -> None:
    """Run FK on ``state_0`` and refresh ``SolverVBD.body_q_prev`` to match."""
    import newton

    model = cable_scene.model
    newton.eval_fk(model, model.joint_q, model.joint_qd, cable_scene.state_0)
    sync_cable_body_q_prev_from_state(cable_scene)
    wp.synchronize()


def align_proxy_body_q_prev_for_vbd(
    cable_scene,
    proxy_body_ids: tuple[int, ...] | wp.array,
) -> None:
    """Align ``SolverVBD.body_q_prev`` with ``state_0.body_q`` on listed bodies after kinematic sync."""
    sync_cable_body_q_prev_from_state(cable_scene, body_ids=proxy_body_ids)


def sync_model_body_q_rest_from_state(cable_scene) -> None:
    """Copy ``state_0.body_q`` into ``model.body_q`` (VBD angular joint rest poses).

    SolverVBD passes ``model.body_q`` as ``body_q_rest`` when evaluating FIXED/D6
    angular residuals (``kappa``). After settle→weld seeding we rewrite cable
    ``state_0`` (and often align the proxy), but leave build-time ``model.body_q``
    untouched — that leaves a large rest-relative kappa on fruiting / weld FIXED
    joints and can yank the grasp on the first AVBD step.
    """
    wp.copy(cable_scene.model.body_q, cable_scene.state_0.body_q)


def sync_solver_body_q_prev_from_state(
    cable_scene,
    body_q_source,
    body_ids: tuple[int, ...] | wp.array | None = None,
) -> None:
    """Copy ``body_q_source[bid]`` into ``solver.body_q_prev[bid]`` for AVBD pre-step consistency."""
    if body_ids is None:
        body_ids = tuple(range(int(cable_scene.model.body_count)))
    if isinstance(body_ids, wp.array):
        ids_arr = body_ids
    else:
        if not body_ids:
            return
        dev = cable_scene.state_0.body_q.device
        ids_arr = wp.array(tuple(int(i) for i in body_ids), dtype=int, device=dev)
    dev = ids_arr.device
    wp.launch(
        _align_body_q_prev_kernel,
        dim=ids_arr.shape[0],
        inputs=[
            ids_arr,
            body_q_source,
            cable_scene.solver.body_q_prev,
        ],
        device=dev,
    )


def copy_cable_body_q_between_states(
    cable_scene,
    *,
    src_state,
    dst_state,
    body_ids: tuple[int, ...],
) -> None:
    """Copy ``body_q`` / ``body_qd`` for listed bodies between cable states (device-side)."""
    if not body_ids:
        return
    dev = cable_scene.state_0.body_q.device
    ids_arr = wp.array(tuple(int(i) for i in body_ids), dtype=int, device=dev)
    wp.launch(
        _copy_body_state_kernel,
        dim=ids_arr.shape[0],
        inputs=[
            ids_arr,
            src_state.body_q,
            src_state.body_qd,
            dst_state.body_q,
            dst_state.body_qd,
        ],
        device=dev,
    )


@wp.kernel
def mirror_robot_tcp_to_proxy_and_apple_kernel(
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
    proxy_offset_in_apple: wp.transform,
):
    """Teleport proxy (with double-integration correction) and apple from robot TCP."""
    i = wp.tid()
    rid = robot_ids[i]
    pid = proxy_ids[i]

    tcp_q = src_body_q[rid]
    tcp_rot = wp.transform_get_rotation(tcp_q)
    tcp_pos = wp.transform_get_translation(tcp_q)

    dst_body_q[pid] = tcp_q
    qd_corr = _corrected_proxy_twist_from_robot(
        src_body_qd[rid],
        proxy_forces[rid],
        body_inv_mass[pid],
        body_inv_inertia[pid],
        tcp_rot,
        gravity,
        dt,
    )
    dst_body_qd[pid] = qd_corr

    if apple_body_id >= 0:
        # X_apple = X_tcp * X_offset^{-1}
        tcp_tf = wp.transform(tcp_pos, tcp_rot)
        apple_tf = wp.transform_multiply(tcp_tf, wp.transform_inverse(proxy_offset_in_apple))
        dst_body_q[apple_body_id] = apple_tf

        # Lever arm for velocity: vector from proxy to apple in local frame, rotated to world
        r_proxy_to_apple_local = wp.transform_get_translation(wp.transform_inverse(proxy_offset_in_apple))
        r_proxy_to_apple_world = wp.quat_rotate(tcp_rot, r_proxy_to_apple_local)

        v_proxy = wp.spatial_top(qd_corr)
        w_proxy = wp.spatial_bottom(qd_corr)
        v_apple = v_proxy + wp.cross(w_proxy, r_proxy_to_apple_world)
        dst_body_qd[apple_body_id] = wp.spatial_vector(v_apple, w_proxy)


def launch_mirror_robot_to_proxy_and_apple(
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
    proxy_offset_in_apple: wp.transform,
    device=None,
) -> None:
    """Teleport proxy + apple; ``body_inv_mass`` / ``body_inv_inertia`` from ``cable_model``."""
    wp.launch(
        mirror_robot_tcp_to_proxy_and_apple_kernel,
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


def launch_mirror_robot_to_proxy(
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
        mirror_robot_tcp_to_proxy_kernel,
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


def launch_mirror_robot_to_proxy_offset(
    *,
    robot_ids: wp.array,
    proxy_ids: wp.array,
    position_offsets: wp.array,
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
    """Ghost mega sync: TCP pose + per-proxy world offset onto cable proxies."""
    wp.launch(
        mirror_robot_tcp_to_proxy_offset_kernel,
        dim=robot_ids.shape[0],
        inputs=[
            robot_ids,
            proxy_ids,
            position_offsets,
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


def launch_mirror_robot_to_proxy_offset_and_apple(
    *,
    robot_ids: wp.array,
    proxy_ids: wp.array,
    position_offsets: wp.array,
    apple_body_ids: wp.array,
    proxy_offset_in_apple: wp.array,
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
    """Ghost mega sync with welded apple co-teleport (offset mirror + apple per row)."""
    wp.launch(
        mirror_robot_tcp_to_proxy_offset_and_apple_kernel,
        dim=robot_ids.shape[0],
        inputs=[
            robot_ids,
            proxy_ids,
            position_offsets,
            apple_body_ids,
            proxy_offset_in_apple,
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


def welded_co_teleport_arrays_for_layout(
    layout: Any,
    cable: Any,
    *,
    device: str,
    per_world_proxy_offsets: tuple[tuple | None, ...] | None = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """Per-registry-row apple ids, zero TCP offsets, and grasp offsets for batched weld sync."""
    default_off = cable.gripper_proxy_offset_in_apple_frame
    if default_off is None and (
        per_world_proxy_offsets is None
        or all(o is None for o in per_world_proxy_offsets)
    ):
        raise ValueError("welded_co_teleport_arrays_for_layout requires grasp offset")
    n = int(layout.num_envs)
    grasp_list: list[wp.transform] = []
    for w in range(n):
        off = (
            per_world_proxy_offsets[w]
            if per_world_proxy_offsets is not None and per_world_proxy_offsets[w] is not None
            else default_off
        )
        grasp_list.append(_grasp_offset_to_transform(off))
    return (
        wp.array(list(layout.apple_body_indices), dtype=int, device=device),
        wp.zeros(n, dtype=wp.vec3, device=device),
        wp.array(grasp_list, dtype=wp.transform, device=device),
    )


def mega_welded_co_teleport_arrays_wp(
    mega: Any,
    *,
    device: str | None = None,
) -> tuple[wp.array, wp.array]:
    """Per-instance apple body ids (-1 if none) and grasp offsets for combined ghost sync."""
    dev = device if device is not None else str(mega.model.device)
    apple_ids: list[int] = []
    offsets: list[wp.transform] = []
    for inst in mega.instances:
        if inst.apple_body is None or inst.gripper_proxy_offset_in_apple_frame is None:
            apple_ids.append(-1)
            offsets.append(wp.transform())
            continue
        apple_ids.append(int(inst.apple_body))
        off = inst.gripper_proxy_offset_in_apple_frame
        offsets.append(
            wp.transform(
                wp.vec3(float(off[0]), float(off[1]), float(off[2])),
                wp.quat(float(off[3]), float(off[4]), float(off[5]), float(off[6])),
            )
        )
    return (
        wp.array(apple_ids, dtype=int, device=dev),
        wp.array(offsets, dtype=wp.transform, device=dev),
    )
