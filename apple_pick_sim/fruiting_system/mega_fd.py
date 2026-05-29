"""Finite-difference stepping and per-step reset for :class:`MegaCoupledCableScene`."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from apple_pick_sim.fruiting_system.mega import (
    FruitingInstanceLayout,
    MegaCoupledCableScene,
)


ApplyControlFn = Callable[[MegaCoupledCableScene, float], None]
ExtractFeaturesFn = Callable[[MegaCoupledCableScene, int], np.ndarray]


@dataclasses.dataclass(frozen=True)
class MegaFdStepResult:
    """One batched FD step over a mega plant."""

    features: np.ndarray
    """Shape ``(num_instances, feature_dim)``."""
    jacobian: np.ndarray
    """Shape ``(feature_dim, num_instances - 1)``; column ``i`` is ∂y/∂θ_i."""
    fim_step: np.ndarray | None = None
    """Shape ``(num_params, num_params)`` when ``sigma_inv`` was passed to :func:`mega_fd_step`."""


@dataclasses.dataclass(frozen=True)
class _InstanceIndexedSlices:
    """Contiguous index ranges for one mega instance in shared model/solver arrays."""

    joint_coord: slice
    joint_dof: slice
    joint_penalty: slice | None
    joint_rest_angle: slice | None


# SolverVBD per-body vec3 / mat33 / spatial buffers (indexed by global body id).
_SOLVER_BODY_VEC3 = (
    "body_forces",
    "body_torques",
)
_SOLVER_BODY_MAT33 = (
    "body_hessian_aa",
    "body_hessian_al",
    "body_hessian_ll",
)
_SOLVER_BODY_TRANSFORM = (
    "body_q_prev",
    "body_inertia_q",
)
# Per-joint vec3 AVBD state (indexed by global joint id).
_SOLVER_JOINT_VEC3 = (
    "joint_lambda_lin",
    "joint_lambda_ang",
    "joint_C0_lin",
    "joint_C0_ang",
    "joint_sigma_prev",
    "joint_kappa_prev",
    "joint_dkappa_prev",
    "joint_sigma_start",
    "joint_C_fric",
)


def instance_body_ids(inst: FruitingInstanceLayout) -> tuple[int, ...]:
    """Body indices for one mega instance (chain + gripper proxy)."""
    ids = list(inst.chain_bodies)
    if inst.gripper_proxy_body not in ids:
        ids.append(inst.gripper_proxy_body)
    return tuple(ids)


def _instance_body_pairs(
    src: FruitingInstanceLayout,
    dst: FruitingInstanceLayout,
) -> list[tuple[int, int]]:
    if len(src.chain_bodies) != len(dst.chain_bodies):
        raise ValueError(
            f"chain_bodies length mismatch: src={len(src.chain_bodies)} dst={len(dst.chain_bodies)}"
        )
    pairs = list(zip(src.chain_bodies, dst.chain_bodies, strict=True))
    pairs.append((src.gripper_proxy_body, dst.gripper_proxy_body))
    return pairs


def _instance_joint_pairs(
    src: FruitingInstanceLayout,
    dst: FruitingInstanceLayout,
) -> list[tuple[int, int]]:
    if len(src.joint_indices) != len(dst.joint_indices):
        raise ValueError("joint_indices length mismatch between instances")
    return list(zip(src.joint_indices, dst.joint_indices, strict=True))


def _offset_delta(
    src: FruitingInstanceLayout,
    dst: FruitingInstanceLayout,
) -> np.ndarray:
    return np.array(dst.base_pos, dtype=np.float64) - np.array(src.base_pos, dtype=np.float64)


def _instance_slices(
    mega: MegaCoupledCableScene,
    instance_index: int,
) -> _InstanceIndexedSlices:
    """Contiguous slices for one instance block in shared joint arrays."""
    model = mega.model
    n_inst = mega.num_instances
    inst = mega.instance(instance_index)
    n_joints_inst = len(inst.joint_indices)
    if model.joint_count % n_inst != 0:
        raise ValueError(
            f"mega model joint_count {model.joint_count} not divisible by {n_inst} instances"
        )
    joints_per_inst = model.joint_count // n_inst
    j0 = instance_index * joints_per_inst
    j1 = j0 + joints_per_inst

    jqs = model.joint_q_start.numpy()
    q_coord = slice(int(jqs[j0]), int(jqs[j1]) if j1 < model.joint_count else model.joint_coord_count)

    jds = model.joint_qd_start.numpy()
    q_dof = slice(int(jds[j0]), int(jds[j1]) if j1 < model.joint_count else model.joint_dof_count)

    solver = mega.solver
    pen_sl: slice | None = None
    rest_sl: slice | None = None
    if hasattr(solver, "joint_penalty_k") and solver.joint_penalty_k is not None:
        n_pen = int(solver.joint_penalty_k.shape[0])
        if n_pen % n_inst == 0:
            w = n_pen // n_inst
            pen_sl = slice(instance_index * w, (instance_index + 1) * w)
    if hasattr(solver, "joint_rest_angle") and solver.joint_rest_angle is not None:
        n_rest = int(solver.joint_rest_angle.shape[0])
        if n_rest % n_inst == 0:
            w = n_rest // n_inst
            rest_sl = slice(instance_index * w, (instance_index + 1) * w)

    if n_joints_inst != joints_per_inst:
        raise ValueError(
            f"instance {instance_index}: layout has {n_joints_inst} joints, "
            f"expected {joints_per_inst} from model layout"
        )
    return _InstanceIndexedSlices(
        joint_coord=q_coord,
        joint_dof=q_dof,
        joint_penalty=pen_sl,
        joint_rest_angle=rest_sl,
    )


def _apply_transform_offset_inplace(
    arr: np.ndarray,
    dst_indices: Sequence[int],
    src_indices: Sequence[int],
    delta: np.ndarray,
) -> None:
    for dst_bid, src_bid in zip(dst_indices, src_indices, strict=True):
        arr[dst_bid, :3] = arr[src_bid, :3] + delta
        arr[dst_bid, 3:7] = arr[src_bid, 3:7]


def _copy_paired_rows(
    dst: np.ndarray,
    src: np.ndarray,
    dst_indices: Sequence[int],
    src_indices: Sequence[int],
) -> None:
    for dst_i, src_i in zip(dst_indices, src_indices, strict=True):
        dst[dst_i] = src[src_i]


def _copy_block(dst: np.ndarray, src: np.ndarray, dst_sl: slice, src_sl: slice) -> None:
    dst[dst_sl] = np.asarray(src[src_sl], dtype=dst.dtype)


def _copy_mega_instance_state_arrays(
    mega: MegaCoupledCableScene,
    body_pairs: list[tuple[int, int]],
    joint_pairs: list[tuple[int, int]],
    delta: np.ndarray,
    src_sl: _InstanceIndexedSlices,
    dst_sl: _InstanceIndexedSlices,
) -> None:
    dst_body_ids = [d for _, d in body_pairs]
    src_body_ids = [s for s, _ in body_pairs]
    dst_joint_ids = [d for _, d in joint_pairs]
    src_joint_ids = [s for s, _ in joint_pairs]

    for state in (mega.state_0, mega.state_1):
        bq = state.body_q.numpy().reshape(-1, 7).copy()
        bqd = state.body_qd.numpy().reshape(-1, 6).copy()
        _apply_transform_offset_inplace(bq, dst_body_ids, src_body_ids, delta)
        _copy_paired_rows(bqd, bqd, dst_body_ids, src_body_ids)
        state.body_q.assign(bq.ravel())
        state.body_qd.assign(bqd.ravel())

        if state.body_f is not None:
            bf = state.body_f.numpy().reshape(-1, 6).copy()
            _copy_paired_rows(bf, bf, dst_body_ids, src_body_ids)
            state.body_f.assign(bf.ravel())

        if state.joint_q is not None:
            jq = state.joint_q.numpy().copy()
            _copy_block(jq, jq, dst_sl.joint_coord, src_sl.joint_coord)
            state.joint_q.assign(jq)

        if state.joint_qd is not None:
            jqd = state.joint_qd.numpy().copy()
            _copy_block(jqd, jqd, dst_sl.joint_dof, src_sl.joint_dof)
            state.joint_qd.assign(jqd)

    solver = mega.solver
    for name in _SOLVER_BODY_TRANSFORM:
        arr = getattr(solver, name, None)
        if arr is None:
            continue
        data = arr.numpy().reshape(-1, 7).copy()
        _apply_transform_offset_inplace(data, dst_body_ids, src_body_ids, delta)
        arr.assign(data.ravel())

    for name in _SOLVER_BODY_VEC3:
        arr = getattr(solver, name, None)
        if arr is None:
            continue
        data = arr.numpy().copy()
        _copy_paired_rows(data, data, dst_body_ids, src_body_ids)
        arr.assign(data)

    for name in _SOLVER_BODY_MAT33:
        arr = getattr(solver, name, None)
        if arr is None:
            continue
        data = arr.numpy().copy()
        _copy_paired_rows(data, data, dst_body_ids, src_body_ids)
        arr.assign(data)

    bf_int = getattr(solver, "_body_f_for_integration", None)
    if bf_int is not None:
        data = bf_int.numpy().reshape(-1, 6).copy()
        _copy_paired_rows(data, data, dst_body_ids, src_body_ids)
        bf_int.assign(data.ravel())

    for name in ("body_inv_mass_effective", "body_inv_inertia_effective"):
        arr = getattr(solver, name, None)
        if arr is None:
            continue
        data = arr.numpy().copy()
        _copy_paired_rows(data, data, dst_body_ids, src_body_ids)
        arr.assign(data)

    pre_bb = int(getattr(solver, "body_body_contact_buffer_pre_alloc", 0) or 0)
    if pre_bb > 0 and hasattr(solver, "body_body_contact_counts"):
        counts = solver.body_body_contact_counts.numpy().copy()
        indices = solver.body_body_contact_indices.numpy().copy()
        for dst_b, src_b in body_pairs:
            counts[dst_b] = counts[src_b]
            for slot in range(pre_bb):
                di = dst_b * pre_bb + slot
                si = src_b * pre_bb + slot
                indices[di] = indices[si]
        solver.body_body_contact_counts.assign(counts)
        solver.body_body_contact_indices.assign(indices)

    pre_bp = int(getattr(solver, "body_particle_contact_buffer_pre_alloc", 0) or 0)
    if pre_bp > 0 and hasattr(solver, "body_particle_contact_counts"):
        counts = solver.body_particle_contact_counts.numpy().copy()
        indices = solver.body_particle_contact_indices.numpy().copy()
        for dst_b, src_b in body_pairs:
            counts[dst_b] = counts[src_b]
            for slot in range(pre_bp):
                di = dst_b * pre_bp + slot
                si = src_b * pre_bp + slot
                indices[di] = indices[si]
        solver.body_particle_contact_counts.assign(counts)
        solver.body_particle_contact_indices.assign(indices)

    for name in _SOLVER_JOINT_VEC3:
        arr = getattr(solver, name, None)
        if arr is None:
            continue
        data = arr.numpy().copy()
        _copy_paired_rows(data, data, dst_joint_ids, src_joint_ids)
        arr.assign(data)

    # NOTE: joint_penalty_k, joint_penalty_k_min, joint_penalty_k_max, and
    # joint_penalty_kd are deliberately NOT copied.  These are the per-column
    # *model parameters* (cable bend/stretch stiffness & damping) baked into the
    # VBD solver at init from model.joint_target_ke/kd.  Overwriting them from
    # the nominal instance would erase the FD stiffness perturbation that makes
    # each mega column behave differently.
    #
    # joint_rest_angle is also a model-level quantity (rest-pose offset for
    # drive/limit comparison) and must not be copied between instances that
    # share the same topology.


def _invalidate_solver_contact_warm_start(solver: Any) -> None:
    """Drop global rigid-contact history so the next step does not warm-start from stale manifolds."""
    for name in (
        "_prev_contact_lambda",
        "_prev_contact_stick_flag",
        "_prev_contact_penalty_k",
        "_prev_contact_point0",
        "_prev_contact_point1",
        "_prev_contact_normal",
    ):
        if hasattr(solver, name):
            setattr(solver, name, None)


def copy_mega_instance_state(
    mega: MegaCoupledCableScene,
    src_index: int,
    dst_index: int,
) -> None:
    """Copy full plant + VBD internal state from ``src_index`` onto ``dst_index``.

    Copies observable state (``body_q``, ``body_qd``, ``body_f``, ``joint_q``, ``joint_qd``)
    and solver buffers (``body_q_prev``, AVBD joint/body accumulators, per-body contact CSR
    slots, joint friction state, penalty stiffness blocks). World-frame poses use the instance
    ``base_pos`` offset; joint-coordinate blocks are copied contiguously per instance.
    """
    if src_index == dst_index:
        return
    src_inst = mega.instance(src_index)
    dst_inst = mega.instance(dst_index)
    delta = _offset_delta(src_inst, dst_inst)
    body_pairs = _instance_body_pairs(src_inst, dst_inst)
    joint_pairs = _instance_joint_pairs(src_inst, dst_inst)
    src_sl = _instance_slices(mega, src_index)
    dst_sl = _instance_slices(mega, dst_index)
    _copy_mega_instance_state_arrays(
        mega,
        body_pairs,
        joint_pairs,
        delta,
        src_sl,
        dst_sl,
    )


def reset_perturbed_instances_to_nominal(
    mega: MegaCoupledCableScene,
    *,
    nominal_index: int = 0,
) -> None:
    """Reset every instance except ``nominal_index`` from the nominal full simulator state."""
    for k in range(mega.num_instances):
        if k != nominal_index:
            copy_mega_instance_state(mega, nominal_index, k)
    _invalidate_solver_contact_warm_start(mega.solver)


def sync_all_instances_from_nominal(
    mega: MegaCoupledCableScene,
    *,
    nominal_index: int = 0,
) -> None:
    """Alias for :func:`reset_perturbed_instances_to_nominal` (build-time FD init)."""
    reset_perturbed_instances_to_nominal(mega, nominal_index=nominal_index)


def default_mega_fd_features(
    mega: MegaCoupledCableScene,
    instance_index: int,
    *,
    dt: float = 1.0 / 1800.0,
) -> np.ndarray:
    """Instance-local apple and proxy positions (minus ``base_pos``).

    When ``fix_to_apple`` is set, appends stem–apple joint wrench (world frame at child COM).
    Uses ``state_1.body_q`` as pre-step poses (same convention as coupled stem harvest).
    """
    inst = mega.instance(instance_index)
    base = np.array(inst.base_pos, dtype=np.float64)
    bq = mega.state_0.body_q.numpy().reshape(-1, 7)
    parts: list[np.ndarray] = []
    if inst.apple_body is not None:
        parts.append(bq[inst.apple_body, :3] - base)
    parts.append(bq[inst.gripper_proxy_body, :3] - base)

    # Append fixed-joint reaction forces/torques when fix_to_apple is True.
    # When welded/pinned, kinematics are identical across all columns, making position features insensitive.
    # Reaction force/torque is the physically meaningful quantity to measure stiffness variations.
    if mega.gripper_proxy_config.fix_to_apple:
        from apple_pick_sim.vbd_fixed_joint_wrenches import fixed_joint_wrenches_child_com_vbd

        jchild = mega.model.joint_child.numpy()
        stem_apple_joint_index = None
        for j_idx, _label in inst.fruiting_fixed_joints:
            if int(jchild[j_idx]) == inst.apple_body:
                stem_apple_joint_index = j_idx
                break

        wrench = np.zeros(6, dtype=np.float64)
        if stem_apple_joint_index is not None:
            records = fixed_joint_wrenches_child_com_vbd(
                mega.model,
                mega.solver,
                body_q=mega.state_0.body_q,
                body_q_prev=mega.state_1.body_q,
                dt=dt,
                joint_pairs=[(stem_apple_joint_index, "stem_apple")],
            )
            if records:
                rec = records[0]
                wrench = np.concatenate([rec.force_world, rec.torque_at_child_com_world]).astype(
                    np.float64
                )
        parts.append(wrench)

    return np.concatenate(parts, dtype=np.float64)


def extract_mega_fd_jacobian(
    mega: MegaCoupledCableScene,
    epsilon: float,
    *,
    nominal_index: int = 0,
    extract_features: ExtractFeaturesFn = default_mega_fd_features,
    dt: float = 1.0 / 1800.0,
    sigma_inv: np.ndarray | None = None,
) -> MegaFdStepResult:
    """Form ``features`` and forward-difference ``jacobian`` from the current mega state."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    n = mega.num_instances
    if n < 1:
        raise ValueError("mega scene must have at least one instance")

    def _features(inst_index: int) -> np.ndarray:
        if extract_features is default_mega_fd_features:
            return default_mega_fd_features(mega, inst_index, dt=dt)
        return extract_features(mega, inst_index)

    feat_list = [_features(i) for i in range(n)]
    features = np.stack(feat_list, axis=0)
    y0 = features[nominal_index]
    jacobian = np.stack(
        [(features[i] - y0) / epsilon for i in range(n) if i != nominal_index],
        axis=1,
    )
    fim_step: np.ndarray | None = None
    if sigma_inv is not None:
        fim_step = jacobian.T @ sigma_inv @ jacobian
    return MegaFdStepResult(features=features, jacobian=jacobian, fim_step=fim_step)



def mega_vbd_substep(
    mega: MegaCoupledCableScene,
    dt: float,
    *,
    collision_pipeline: Any | None = None,
) -> Any:
    """One VBD substep on the shared mega model (all instances)."""
    mega.state_0.clear_forces()
    if collision_pipeline is None:
        contacts = mega.model.collide(mega.state_0)
    else:
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(mega.state_0, contacts)
    mega.solver.step(
        mega.state_0,
        mega.state_1,
        mega.control,
        contacts,
        dt,
    )
    mega.state_0, mega.state_1 = mega.state_1, mega.state_0
    return contacts


def _noop_control(_mega: MegaCoupledCableScene, _dt: float) -> None:
    return None


def mega_fd_step(
    mega: MegaCoupledCableScene,
    epsilon: float,
    *,
    apply_control: ApplyControlFn = _noop_control,
    extract_features: ExtractFeaturesFn = default_mega_fd_features,
    dt: float,
    collision_pipeline: Any | None = None,
    nominal_index: int = 0,
    sigma_inv: np.ndarray | None = None,
) -> MegaFdStepResult:
    """Apply control, batched substep, FD Jacobian, reset perturbed instances."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if mega.num_instances < 1:
        raise ValueError("mega scene must have at least one instance")

    apply_control(mega, dt)
    mega_vbd_substep(mega, dt, collision_pipeline=collision_pipeline)

    result = extract_mega_fd_jacobian(
        mega,
        epsilon,
        nominal_index=nominal_index,
        extract_features=extract_features,
        dt=dt,
        sigma_inv=sigma_inv,
    )
    reset_perturbed_instances_to_nominal(mega, nominal_index=nominal_index)
    return result


def _assign_wp_array_if_present(dst_arr: Any, src_arr: Any) -> None:
    if dst_arr is not None and src_arr is not None:
        dst_arr.assign(src_arr.numpy())


def _copy_coupled_full_state(scene_nom: Any, scene_pert: Any) -> None:
    """Copy nominal coupled plant state onto perturbed (same ``base_pos``, no offset)."""
    n = scene_nom.model.body_count
    nom_bq = scene_nom.state_0.body_q.numpy().reshape(-1, 7)
    nom_bqd = scene_nom.state_0.body_qd.numpy().reshape(-1, 6)
    nom_bf = None
    if scene_nom.state_0.body_f is not None:
        nom_bf = scene_nom.state_0.body_f.numpy().reshape(-1, 6)

    for state in (scene_pert.state_0, scene_pert.state_1):
        bq = state.body_q.numpy().reshape(-1, 7).copy()
        bqd = state.body_qd.numpy().reshape(-1, 6).copy()
        bq[:n] = nom_bq[:n]
        bqd[:n] = nom_bqd[:n]
        state.body_q.assign(bq.ravel())
        state.body_qd.assign(bqd.ravel())
        if state.body_f is not None and nom_bf is not None:
            bf = state.body_f.numpy().reshape(-1, 6).copy()
            bf[:n] = nom_bf[:n]
            state.body_f.assign(bf.ravel())
        if state.joint_q is not None and scene_nom.state_0.joint_q is not None:
            state.joint_q.assign(scene_nom.state_0.joint_q.numpy())
        if state.joint_qd is not None and scene_nom.state_0.joint_qd is not None:
            state.joint_qd.assign(scene_nom.state_0.joint_qd.numpy())

    body_ids = list(range(n))
    body_pairs = list(zip(body_ids, body_ids, strict=True))
    joint_ids = list(range(scene_nom.model.joint_count))
    joint_pairs = list(zip(joint_ids, joint_ids, strict=True))
    sl = _InstanceIndexedSlices(
        joint_coord=slice(0, scene_nom.model.joint_coord_count),
        joint_dof=slice(0, scene_nom.model.joint_dof_count),
        joint_penalty=None,
        joint_rest_angle=None,
    )
    pen_k = getattr(scene_nom.solver, "joint_penalty_k", None)
    if pen_k is not None:
        sl = dataclasses.replace(sl, joint_penalty=slice(0, int(pen_k.shape[0])))
    rest = getattr(scene_nom.solver, "joint_rest_angle", None)
    if rest is not None:
        sl = dataclasses.replace(sl, joint_rest_angle=slice(0, int(rest.shape[0])))

    class _ScratchMega:
        """Minimal stand-in so :func:`_copy_mega_instance_state_arrays` can run on one solver."""

        def __init__(self, scene: Any) -> None:
            self.state_0 = scene.state_0
            self.state_1 = scene.state_1
            self.model = scene.model
            self.solver = scene.solver

    _copy_mega_instance_state_arrays(
        _ScratchMega(scene_pert),  # type: ignore[arg-type]
        body_pairs,
        joint_pairs,
        np.zeros(3),
        sl,
        sl,
    )
    for name in _SOLVER_BODY_TRANSFORM + _SOLVER_BODY_VEC3 + _SOLVER_BODY_MAT33 + _SOLVER_JOINT_VEC3:
        _assign_wp_array_if_present(getattr(scene_pert.solver, name, None), getattr(scene_nom.solver, name, None))
    _assign_wp_array_if_present(
        getattr(scene_pert.solver, "_body_f_for_integration", None),
        getattr(scene_nom.solver, "_body_f_for_integration", None),
    )
    # NOTE: joint_penalty_k/k_min/k_max/kd and joint_rest_angle are excluded —
    # they encode per-scene model parameters, not per-step integration state.
    # Copying them would erase the stiffness perturbation in the perturbed scene.
    pre_bb = int(getattr(scene_nom.solver, "body_body_contact_buffer_pre_alloc", 0) or 0)
    if pre_bb > 0:
        _assign_wp_array_if_present(
            scene_pert.solver.body_body_contact_counts,
            scene_nom.solver.body_body_contact_counts,
        )
        _assign_wp_array_if_present(
            scene_pert.solver.body_body_contact_indices,
            scene_nom.solver.body_body_contact_indices,
        )
    pre_bp = int(getattr(scene_nom.solver, "body_particle_contact_buffer_pre_alloc", 0) or 0)
    if pre_bp > 0:
        _assign_wp_array_if_present(
            scene_pert.solver.body_particle_contact_counts,
            scene_nom.solver.body_particle_contact_counts,
        )
        _assign_wp_array_if_present(
            scene_pert.solver.body_particle_contact_indices,
            scene_nom.solver.body_particle_contact_indices,
        )
    _invalidate_solver_contact_warm_start(scene_pert.solver)


def copy_coupled_scene_from_nominal(
    scene_nom: Any,
    scene_pert: Any,
) -> None:
    """Align perturbed standalone cable scene to nominal (full state + VBD buffers)."""
    _copy_coupled_full_state(scene_nom, scene_pert)


def instance_body_ids_from_coupled(scene: Any) -> tuple[int, ...]:
    """Body indices for a :class:`CoupledCableScene` (chain + proxy)."""
    ids = (
        list(scene.primary_bodies)
        + list(scene.secondary_bodies)
        + list(scene.spur_bodies)
        + list(scene.stem_bodies)
    )
    if scene.apple_body is not None:
        ids.append(scene.apple_body)
    if scene.gripper_proxy_body not in ids:
        ids.append(scene.gripper_proxy_body)
    return tuple(ids)


def coupled_vbd_substep(scene: Any, dt: float, *, collision_pipeline: Any | None = None) -> None:
    """One VBD substep on a standalone :class:`CoupledCableScene`."""
    scene.state_0.clear_forces()
    if collision_pipeline is None:
        contacts = scene.model.collide(scene.state_0)
    else:
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(scene.state_0, contacts)
    scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, dt)
    scene.state_0, scene.state_1 = scene.state_1, scene.state_0
