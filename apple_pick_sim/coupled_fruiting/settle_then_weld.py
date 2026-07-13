"""Utilities for "settle freely, then weld" initialization (fix_to_apple quiet start).

Newton/VBD model topology is fixed after :class:`newton.ModelBuilder` is finalized, so
the proxy↔apple FIXED joint (``joint_apple_gripper_proxy``) cannot be toggled on/off at
runtime. This module provides a robust two-build workflow:

1) Build a free-apple scene (``fix_to_apple=False``) and run VBD substeps to settle.
   Optionally use a linear gravity ramp via :func:`settle_vbd_substeps` (``gravity_ramp=True``).
2) Build a welded scene (``fix_to_apple=True``) and seed its cable state from the
   settled configuration so the welded constraint starts near zero violation.
3) Re-run FR3 IK bootstrap at the scene's fixed robot base (from fixture
   ``robot_base_pos`` or builder placement). Raise if the settled proxy is unreachable.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from apple_pick_sim.coupled_fruiting.proxy_coupling import (
    align_proxy_body_q_prev_for_vbd,
    sync_model_body_q_rest_from_state,
    sync_solver_body_q_prev_from_state,
)
from apple_pick_sim.coupled_fruiting.batched_build import (
    broadcast_settled_cable_state_to_batched_worlds,
)
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.broadcast_actions import broadcast_joint_q_from_world0
from apple_pick_sim.coupled_fruiting.scene import init_robot_mujoco_step_buffers
from apple_pick_sim.coupled_fruiting.settle_seed_device import (
    align_batched_proxy_poses_device,
    copy_cable_state_device,
    zero_all_body_qd_device,
)


def _proxy_world_pose_from_apple(
    apple_body_q7: np.ndarray,
    offset_7d: tuple | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame proxy (position, quaternion) from apple pose and 7D grasp offset.

    Returns ``(pos_f32[3], quat_f32[4])``.
    """
    apple_tf = wp.transform(
        wp.vec3(float(apple_body_q7[0]), float(apple_body_q7[1]), float(apple_body_q7[2])),
        wp.quat(float(apple_body_q7[3]), float(apple_body_q7[4]), float(apple_body_q7[5]), float(apple_body_q7[6])),
    )
    offset_tf = wp.transform(
        wp.vec3(float(offset_7d[0]), float(offset_7d[1]), float(offset_7d[2])),
        wp.quat(float(offset_7d[3]), float(offset_7d[4]), float(offset_7d[5]), float(offset_7d[6])),
    )
    # X_proxy = X_apple * X_offset
    proxy_tf = wp.transform_multiply(apple_tf, offset_tf)
    proxy_pos = wp.transform_get_translation(proxy_tf)
    proxy_rot = wp.transform_get_rotation(proxy_tf)
    pos = np.array([proxy_pos[0], proxy_pos[1], proxy_pos[2]], dtype=np.float32)
    quat = np.array([proxy_rot[0], proxy_rot[1], proxy_rot[2], proxy_rot[3]], dtype=np.float32)
    return pos, quat


def should_quiet_cable_bodies_at_settle_substep(
    completed_substep: int,
    quiet_every: int | None,
) -> bool:
    """True when periodic settle quiet should run after substep ``completed_substep`` (1-based)."""
    if quiet_every is None:
        return False
    every = int(quiet_every)
    if every <= 0:
        return False
    return int(completed_substep) % every == 0


def warn_settle_quiet_every_alignment(
    settle_substeps: int,
    quiet_every: int | None,
) -> int | None:
    """Print ``settle_substeps % quiet_every``; warn when remainder is 0.

    When remainder is 0, the periodic quiet lands on the final settle substep and
    zeros ``body_qd`` before post-settle residual-speed checks.

    Returns the remainder, or ``None`` when quieting is disabled.
    """
    if quiet_every is None:
        return None
    every = int(quiet_every)
    n = int(settle_substeps)
    if every <= 0 or n <= 0:
        return None
    remainder = n % every
    print(
        f"settle quiet alignment: settle_substeps={n} quiet_every={every} "
        f"remainder={remainder}",
        flush=True,
    )
    if remainder == 0:
        warnings.warn(
            f"settle_substeps ({n}) % settle_quiet_every ({every}) == 0 "
            f"(remainder={remainder}); periodic quiet lands on the final settle "
            f"substep, zeroing velocities before the stability residual-speed check.",
            UserWarning,
            stacklevel=2,
        )
    return remainder


def quiet_all_cable_bodies(cable: Any) -> None:
    """Zero all cable body twists and align VBD ``body_q_prev`` with ``state_0.body_q``.

    Call after settle (and after any settle stability diagnostics that need residual
    ``body_qd``) so the plant starts kinematically static without stale warm-start
    velocity from AVBD integration.
    """
    body_count = int(cable.model.body_count)
    if body_count <= 0:
        return
    zero_all_body_qd_device(cable.state_0.body_qd)
    zero_all_body_qd_device(cable.state_1.body_qd)
    sync_solver_body_q_prev_from_state(
        cable,
        cable.state_0.body_q,
        tuple(range(body_count)),
    )
    wp.synchronize()


def settle_gravity_z_for_substep(
    step_index: int,
    substeps: int,
    *,
    target_z: float = -9.81,
) -> float:
    """Linear gravity ramp: substep ``i`` uses ``target_z * (i + 1) / N``."""
    n = int(substeps)
    if n <= 0:
        return float(target_z)
    return float(target_z) * (int(step_index) + 1) / n


def _apply_settle_gravity(scene: Any, g_xyz: tuple[float, float, float]) -> None:
    scene.cable.model.set_gravity(g_xyz)
    if hasattr(scene, "gravity_vec"):
        scene.gravity_vec = wp.vec3(float(g_xyz[0]), float(g_xyz[1]), float(g_xyz[2]))


def settle_vbd_substeps(
    scene: Any,
    *,
    substeps: int,
    dt: float,
    gravity_ramp: bool = False,
    gravity_target: tuple[float, float, float] = (0.0, 0.0, -9.81),
    quiet_every: int | None = None,
) -> None:
    """Advance ``scene`` by ``substeps`` VBD-only substeps.

    When ``gravity_ramp`` is True, gravity on the cable model ramps linearly
    from zero to ``gravity_target`` over all substeps before each ``vbd_substep`` call.
    Full target magnitude applies only on the final substep; increase ``substeps`` if
    post-settle stability reports still show residual motion.

    Args:
        scene: A :class:`~apple_pick_sim.coupled_fruiting.scene.CoupledFruitingScene`
            or any object exposing ``vbd_substep(dt)`` and ``cable.model``.
        substeps: Number of VBD substeps to advance.
        dt: Step size [s] per VBD substep.
        gravity_ramp: If True, linearly ramp cable gravity over substeps; if False,
            leave build-time gravity unchanged (instant full g throughout).
        gravity_target: World-frame gravity vector at the end of the ramp (default −9.81 z).
        quiet_every: When set, zero all cable body twists every this many settle substeps
            (device-side via :func:`quiet_all_cable_bodies`); ``None`` disables.
    """
    n = int(substeps)
    if n <= 0:
        return
    h = float(dt)
    for i in range(n):
        apply_settle_gravity_for_substep(
            scene,
            i,
            n,
            gravity_ramp=gravity_ramp,
            gravity_target=gravity_target,
        )
        scene.vbd_substep(h)
        if should_quiet_cable_bodies_at_settle_substep(i + 1, quiet_every):
            quiet_all_cable_bodies(scene.cable)


def apply_settle_gravity_for_substep(
    scene: Any,
    step_index: int,
    substeps: int,
    *,
    gravity_ramp: bool,
    gravity_target: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> None:
    """Set cable gravity for one settle substep (no-op when ``gravity_ramp`` is False)."""
    if not gravity_ramp:
        return
    target_z = float(gravity_target[2])
    gz = settle_gravity_z_for_substep(step_index, substeps, target_z=target_z)
    _apply_settle_gravity(
        scene,
        (float(gravity_target[0]), float(gravity_target[1]), gz),
    )


def _nominal_cable_view(scene: Any) -> Any:
    """Return a single-instance cable view when the scene exposes one, else the full cable.

    When ``cable`` supports ``as_single_instance_coupled`` (duck-typed), picks
    ``scene.nominal_index`` (default 0) so settle/seed logic operates on one
    fruiting instance without iterating all ghosts.

    Used by :func:`seed_fix_to_apple_from_settled` when copying settled poses
    into the welded scene during ``fix_to_apple`` quiet-start initialization.
    """
    cable = scene.cable
    if hasattr(cable, "as_single_instance_coupled"):
        idx = int(getattr(scene, "nominal_index", 0))
        return cable.as_single_instance_coupled(idx)
    return cable


def _bootstrap_tcp_at_fixed_origin(
    scene: Any,
    *,
    ik_iterations: int = 96,
) -> None:
    """Align TCP to the seeded cable proxy using the scene's fixed FR3 base placement.

    For batched scenes (world_count > 1), IK is solved on the single-world template
    model stored in ``scene.ik_template_robot_model`` to avoid Newton's IK FK building
    an incorrect kinematic chain from multi-world joint coordinates.  The solved
    world-0 joint_q is then written into the batched model and broadcast to all worlds.
    """
    if scene.robot_model is None or scene.robot_state_0 is None or scene.mj_solver is None:
        return

    import newton

    from apple_pick_sim.coupled_fruiting.bootstrap import bootstrap_articulated_tcp_from_proxy
    from apple_pick_sim.coupled_fruiting.bootstrap import bootstrap_tcp_joint_from_proxy
    from apple_pick_sim.robot import fr3_robot
    from apple_pick_sim.robot.fr3_robot.placement import IKBootstrapConvergenceError

    cable = _nominal_cable_view(scene)
    root = np.asarray(getattr(scene, "fr3_root_world_pos", (0.0, 0.0, 0.0)), dtype=np.float64)
    root_xyz = (float(root[0]), float(root[1]), float(root[2]))

    layout = getattr(scene, "layout", None)
    tpl_robot = getattr(scene, "ik_template_robot_model", None)
    use_template_ik = (
        tpl_robot is not None
        and layout is not None
        and int(scene.robot_model.world_count) > 1
    )

    if use_template_ik:
        tpl_state = tpl_robot.state()
        tpl_tcp = int(layout.template_tcp_body)
        bootstrap_fn = (
            bootstrap_tcp_joint_from_proxy
            if int(tpl_robot.body_count) <= 2
            else bootstrap_articulated_tcp_from_proxy
        )
        try:
            bootstrap_fn(
                cable,
                tpl_robot,
                tpl_tcp,
                tpl_state,
                **(
                    {"ik_iterations": ik_iterations, "raise_on_failure": True}
                    if bootstrap_fn is bootstrap_articulated_tcp_from_proxy
                    else {}
                ),
            )
        except IKBootstrapConvergenceError as exc:
            raise IKBootstrapConvergenceError(
                "Settled gripper proxy is unreachable from the specified FR3 base at "
                f"({root_xyz[0]:.3f}, {root_xyz[1]:.3f}, {root_xyz[2]:.3f}): {exc}"
            ) from exc
        # Copy solved world-0 joint_q into the batched model and broadcast.
        tpl_jq = tpl_robot.joint_q.numpy().copy()
        coord_per = int(tpl_jq.shape[0])
        batched_jq = scene.robot_model.joint_q.numpy().copy()
        batched_jq[:coord_per] = tpl_jq
        scene.robot_model.joint_q.assign(batched_jq)
        scene.robot_state_0.joint_q.assign(batched_jq)
        newton.eval_fk(
            scene.robot_model,
            scene.robot_model.joint_q,
            scene.robot_model.joint_qd,
            scene.robot_state_0,
        )
        broadcast_joint_q_from_world0(scene, layout)
    else:
        try:
            bootstrap_articulated_tcp_from_proxy(
                cable,
                scene.robot_model,
                scene.tcp_body_index,
                scene.robot_state_0,
                ik_iterations=ik_iterations,
                raise_on_failure=True,
            )
        except IKBootstrapConvergenceError as exc:
            raise IKBootstrapConvergenceError(
                "Settled gripper proxy is unreachable from the specified FR3 base at "
                f"({root_xyz[0]:.3f}, {root_xyz[1]:.3f}, {root_xyz[2]:.3f}): {exc}"
            ) from exc

    init_robot_mujoco_step_buffers(scene)
    fr3_robot.init_mujoco_actuator_targets_from_model(
        scene.robot_model, scene.robot_control
    )
    if scene.proxy_forces is not None:
        scene.proxy_forces.zero_()
    if scene.coupling_forces_cache is not None:
        scene.coupling_forces_cache.zero_()


def _proxy_targets_world_from_cable(
    cable_state: Any,
    layout: BatchedEnvLayout,
) -> list[wp.transform]:
    """Build per-env world-frame proxy targets from cable ``body_q`` (single host sync)."""
    proxy_bq = cable_state.body_q.numpy().reshape(-1, 7)
    return [
        wp.transform(
            wp.vec3(
                float(proxy_bq[layout.proxy_body_indices[w], 0]),
                float(proxy_bq[layout.proxy_body_indices[w], 1]),
                float(proxy_bq[layout.proxy_body_indices[w], 2]),
            ),
            wp.quat(
                float(proxy_bq[layout.proxy_body_indices[w], 3]),
                float(proxy_bq[layout.proxy_body_indices[w], 4]),
                float(proxy_bq[layout.proxy_body_indices[w], 5]),
                float(proxy_bq[layout.proxy_body_indices[w], 6]),
            ),
        )
        for w in range(int(layout.num_envs))
    ]


def _bootstrap_tcp_per_env(
    scene: Any,
    layout: BatchedEnvLayout,
    *,
    ik_iterations: int | None = None,
    n_seeds: int | None = None,
) -> list[tuple[float, float, bool]]:
    """Align each world's TCP to its own settled proxy via batched template IK."""
    from apple_pick_sim.robot.fr3_robot.placement import (
        IK_BOOTSTRAP_DEFAULT_ITERATIONS,
        IK_BOOTSTRAP_DEFAULT_MAX_SEEDS,
        IK_BOOTSTRAP_POS_TOL_M,
        IK_BOOTSTRAP_ROT_TOL_RAD,
        warn_if_ik_bootstrap_not_converged,
    )

    if ik_iterations is None:
        ik_iterations = IK_BOOTSTRAP_DEFAULT_ITERATIONS
    if n_seeds is None:
        n_seeds = IK_BOOTSTRAP_DEFAULT_MAX_SEEDS
    tpl_robot = getattr(scene, "ik_template_robot_model", None)
    if tpl_robot is None or scene.robot_model is None:
        return

    from apple_pick_sim.robot.fr3_robot.batched_template_ik import BatchedTemplateIK

    ik_solver = BatchedTemplateIK(
        tpl_robot,
        scene.robot_model,
        layout,
        layout.template_tcp_body,
        n_seeds=int(n_seeds),
        sampler="roberts",
    )
    dev = tpl_robot.device
    proxy_indices_wp = wp.array(list(layout.proxy_body_indices), dtype=int, device=dev)

    ik_solver.gather_targets_from_proxy_state(scene.cable.state_0, proxy_indices_wp)
    ik_solver.step(int(ik_iterations))

    targets_world = _proxy_targets_world_from_cable(scene.cable.state_0, layout)
    results: list[tuple[float, float, bool]] = []
    for w, (pos_err, rot_err) in enumerate(ik_solver.pose_errors_per_world(targets_world)):
        inside = pos_err < IK_BOOTSTRAP_POS_TOL_M and rot_err < IK_BOOTSTRAP_ROT_TOL_RAD
        results.append((pos_err, rot_err, inside))
        target_pos = wp.transform_get_translation(targets_world[w])
        warn_if_ik_bootstrap_not_converged(
            pos_err,
            rot_err,
            target_pos=(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
        )

    ik_solver.scatter_to_model(scene.robot_state_0)
    scene.settle_ik_envelope_results = results
    return results


def seed_fix_to_apple_from_settled_body_q(
    *,
    welded_scene: Any,
    settled_body_q: np.ndarray,
    quiet_apple_proxy: bool = True,
    per_env_ik: bool = False,
    per_world_proxy_offsets: tuple[tuple | None, ...] | None = None,
    ik_bootstrap_iterations: int | None = None,
) -> None:
    """Seed welded scene cable poses from settled ``body_q`` (checkpoint path)."""
    bq = np.asarray(settled_body_q, dtype=np.float32).reshape(-1, 7)
    bqd = np.zeros((bq.shape[0], 6), dtype=np.float32)

    class _ArrayView:
        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def numpy(self) -> np.ndarray:
            return self._arr

    settled_view = type(
        "_SettledScene",
        (),
        {
            "cable": type(
                "_SettledCable",
                (),
                {
                    "model": welded_scene.cable.model,
                    "state_0": type(
                        "_State",
                        (),
                        {"body_q": _ArrayView(bq), "body_qd": _ArrayView(bqd)},
                    )(),
                },
            )()
        },
    )()
    seed_fix_to_apple_from_settled(
        welded_scene=welded_scene,
        settled_scene=settled_view,
        quiet_apple_proxy=quiet_apple_proxy,
        per_env_ik=per_env_ik,
        per_world_proxy_offsets=per_world_proxy_offsets,
        ik_bootstrap_iterations=ik_bootstrap_iterations,
    )


def seed_fix_to_apple_from_settled(
    *,
    welded_scene: Any,
    settled_scene: Any,
    quiet_apple_proxy: bool = True,
    per_env_ik: bool = False,
    per_world_proxy_offsets: tuple[tuple | None, ...] | None = None,
    ik_bootstrap_iterations: int | None = None,
) -> None:
    """Seed a welded (``fix_to_apple=True``) scene from a settled free-apple scene.

    This copies the *entire* cable model state (body poses + twists) from
    ``settled_scene`` into ``welded_scene`` and then enforces that the proxy starts
    at the configured grasp offset from the apple so the proxy↔apple fixed joint has
    minimal initial violation.

    When ``settled_scene`` has the same world count as ``welded_scene`` (``N > 1``),
    each world is copied directly (world *i* settled → world *i* welded). When the
    settled scene has a single world and the welded scene has multiple, the legacy
    single-world broadcast path applies spacing offsets per env.

    Args:
        welded_scene: A coupled scene built with ``GripperProxyConfig(fix_to_apple=True)``.
        settled_scene: A coupled scene built with ``GripperProxyConfig(fix_to_apple=False)``
            that has already been advanced to a quasi-static configuration.
        quiet_apple_proxy: If True, zero the apple and proxy twists after seeding.
    """
    cable_w = welded_scene.cable
    cable_s = settled_scene.cable

    layout: BatchedEnvLayout | None = getattr(welded_scene, "layout", None)
    settled_worlds = int(cable_s.model.world_count)
    env_spacing = getattr(welded_scene, "env_spacing", None)
    if (
        layout is not None
        and settled_worlds == 1
        and layout.num_envs > 1
        and env_spacing is not None
    ):
        broadcast_settled_cable_state_to_batched_worlds(
            cable_s, cable_w, layout, env_spacing
        )
    else:
        copy_cable_state_device(cable_s, cable_w)

    apple = cable_w.apple_body
    proxy = cable_w.gripper_proxy_body
    default_offset = cable_w.gripper_proxy_offset_in_apple_frame
    if apple is None or default_offset is None:
        return

    if layout is not None and layout.num_envs >= 1:
        align_batched_proxy_poses_device(
            cable_w,
            layout,
            per_world_proxy_offsets=per_world_proxy_offsets,
            default_offset=default_offset,
            quiet_apple_proxy=quiet_apple_proxy,
        )
    else:
        bq_w = cable_w.state_0.body_q.numpy().reshape(-1, 7).copy()
        bqd_w = cable_w.state_0.body_qd.numpy().reshape(-1, 6).copy()
        proxy_pos, proxy_quat = _proxy_world_pose_from_apple(bq_w[apple], default_offset)
        bq_w[proxy, :3] = proxy_pos
        bq_w[proxy, 3:] = proxy_quat
        if quiet_apple_proxy:
            bqd_w[apple] = 0.0
            bqd_w[proxy] = 0.0
        cable_w.state_0.body_q.assign(bq_w.reshape(-1, 7))
        cable_w.state_0.body_qd.assign(bqd_w.reshape(-1, 6))
        cable_w.state_1.body_q.assign(bq_w.reshape(-1, 7))
        cable_w.state_1.body_qd.assign(bqd_w.reshape(-1, 6))

    # Do not call eval_fk() here: the welded and settled models do not share joint-space
    # coordinates (free proxy vs fixed proxy↔apple), so FK from welded joint_q would
    # overwrite the seeded settled body poses.
    # Align VBD's warm-start/previous-pose buffers so the first step does not see
    # a mixed settled/unsettled state and inject an artificial stem impulse.
    body_count = int(cable_w.model.body_count)
    align_proxy_body_q_prev_for_vbd(cable_w, tuple(range(body_count)))
    # VBD angular joints use model.body_q as rest; keep it in sync with the seeded
    # poses (else FIXED kappa is measured against pre-settle build geometry).
    sync_model_body_q_rest_from_state(cable_w)

    wp.synchronize()
    if per_env_ik and layout is not None and layout.num_envs > 1:
        _bootstrap_tcp_per_env(
            welded_scene,
            layout,
            ik_iterations=ik_bootstrap_iterations,
        )
        from apple_pick_sim.coupled_fruiting.proxy_coupling import prepare_batched_stem_harvest_arrays

        prepare_batched_stem_harvest_arrays(welded_scene, layout)
    else:
        from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_DEFAULT_ITERATIONS

        bootstrap_iters = (
            int(ik_bootstrap_iterations)
            if ik_bootstrap_iterations is not None
            else IK_BOOTSTRAP_DEFAULT_ITERATIONS
        )
        _bootstrap_tcp_at_fixed_origin(
            welded_scene, ik_iterations=bootstrap_iters
        )
        if layout is not None:
            broadcast_joint_q_from_world0(welded_scene, layout)
        return

    from apple_pick_sim.robot import fr3_robot

    init_robot_mujoco_step_buffers(welded_scene)
    fr3_robot.init_mujoco_actuator_targets_from_model(
        welded_scene.robot_model, welded_scene.robot_control
    )
    if welded_scene.proxy_forces is not None:
        welded_scene.proxy_forces.zero_()
    if welded_scene.coupling_forces_cache is not None:
        welded_scene.coupling_forces_cache.zero_()
