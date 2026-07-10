"""Template-model IK for replicated robot worlds (Newton cube-stacking pattern)."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.robot.fr3_robot.placement import tcp_ik_target_pose_errors


def _quat_to_ik_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


@wp.func
def _vec4_to_quat(v: wp.vec4) -> wp.quat:
    return wp.quat(v[0], v[1], v[2], v[3])


@wp.func
def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


@wp.kernel(enable_backward=False)
def _gather_proxy_targets_from_cable_kernel(
    cable_body_q: wp.array(dtype=wp.transform),
    proxy_indices: wp.array(dtype=int),
    target_positions: wp.array(dtype=wp.vec3),
    target_rotations: wp.array(dtype=wp.vec4),
):
    """Gather per-env proxy poses from cable ``body_q`` into template-frame IK targets."""
    i = wp.tid()
    tf = cable_body_q[proxy_indices[i]]
    body_pos = wp.transform_get_translation(tf)
    body_rot = wp.transform_get_rotation(tf)
    target_positions[i] = body_pos
    target_rotations[i] = wp.vec4(body_rot[0], body_rot[1], body_rot[2], body_rot[3])


@wp.kernel(enable_backward=False)
def _gather_tcp_world_poses_kernel(
    body_q: wp.array(dtype=wp.transform),
    tcp_indices: wp.array(dtype=int),
    out_pos_world: wp.array(dtype=wp.vec3),
    out_rot_world: wp.array(dtype=wp.vec4),
    target_positions: wp.array(dtype=wp.vec3),
    target_rotations: wp.array(dtype=wp.vec4),
):
    """Gather TCP poses from batched FK; world frame equals template frame (co-located)."""
    i = wp.tid()
    tcp_idx = tcp_indices[i]
    tf = body_q[tcp_idx]
    body_pos = wp.transform_get_translation(tf)
    body_rot = wp.transform_get_rotation(tf)
    rot_v4 = wp.vec4(body_rot[0], body_rot[1], body_rot[2], body_rot[3])
    out_pos_world[i] = body_pos
    out_rot_world[i] = rot_v4
    target_positions[i] = body_pos
    target_rotations[i] = rot_v4


@wp.kernel(enable_backward=False)
def _integrate_tcp_targets_kernel(
    pos_world: wp.array(dtype=wp.vec3),
    rot_world: wp.array(dtype=wp.vec4),
    lin_vels: wp.array(dtype=wp.vec3),
    ang_vels: wp.array(dtype=wp.vec3),
    dt: float,
    target_positions: wp.array(dtype=wp.vec3),
    target_rotations: wp.array(dtype=wp.vec4),
):
    """Integrate co-located world-frame TCP targets and write IK objectives."""
    i = wp.tid()
    pos = pos_world[i]
    rot = _vec4_to_quat(rot_world[i])
    lin = lin_vels[i]
    ang = ang_vels[i]
    pos_new = pos + lin * dt
    ang_mag = wp.length(ang)
    if ang_mag > 1.0e-12:
        delta_rot = wp.quat_from_axis_angle(ang / ang_mag, ang_mag * dt)
        rot_new = wp.normalize(_quat_mul(delta_rot, rot))
    else:
        rot_new = rot
    rot_v4 = wp.vec4(rot_new[0], rot_new[1], rot_new[2], rot_new[3])
    pos_world[i] = pos_new
    rot_world[i] = rot_v4
    target_positions[i] = pos_new
    target_rotations[i] = rot_v4


class BatchedTemplateIK:
    """Run ``IKSolver`` on a single-world template with ``n_problems=num_envs``."""

    def __init__(
        self,
        ik_model: newton.Model,
        sim_model: newton.Model,
        layout: BatchedEnvLayout,
        tcp_body_index: int,
        *,
        joint_limit_weight: float = 10.0,
        lambda_initial: float = 0.1,
        n_seeds: int = 1,
        sampler: str = "none",
        rng_seed: int = 12345,
    ) -> None:
        import newton.ik as ik

        self.ik_model = ik_model
        self.sim_model = sim_model
        self.layout = layout
        self.tcp_body_index = int(tcp_body_index)
        self.device = ik_model.device
        self.n_problems = int(layout.num_envs)
        self.n_coords = int(ik_model.joint_coord_count)
        self.n_dofs = int(ik_model.joint_dof_count)
        self.n_seeds = int(n_seeds)

        dev = self.device
        n = self.n_problems
        self.target_positions = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.target_rotations = wp.zeros(n, dtype=wp.vec4, device=dev)

        self._tcp_body_indices_wp = wp.array(
            list(layout.tcp_body_indices),
            dtype=int,
            device=dev,
        )

        self._pos_obj = ik.IKObjectivePosition(
            link_index=self.tcp_body_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=self.target_positions,
        )
        self._rot_obj = ik.IKObjectiveRotation(
            link_index=self.tcp_body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.target_rotations,
        )
        self._limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=ik_model.joint_limit_lower,
            joint_limit_upper=ik_model.joint_limit_upper,
            weight=joint_limit_weight,
        )
        self.joint_q = wp.zeros((n, self.n_coords), dtype=wp.float32, device=dev)
        self._solver = ik.IKSolver(
            model=ik_model,
            n_problems=n,
            objectives=[self._pos_obj, self._rot_obj, self._limits],
            lambda_initial=lambda_initial,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            n_seeds=self.n_seeds,
            sampler=sampler,
            rng_seed=rng_seed,
        )

    def template_to_world(self, tf_template: wp.transform, world: int) -> wp.transform:
        """Express a template-frame TCP pose in world coordinates (co-located)."""
        del world
        return tf_template

    def world_to_template(self, tf_world: wp.transform, world: int) -> wp.transform:
        """Express a world-frame TCP pose in the single-world template frame (co-located)."""
        del world
        return tf_world

    def tcp_pose_from_joint_coords(self, joint_coords: np.ndarray) -> wp.transform:
        """FK on the template model for one world's joint coordinates."""
        tpl_jq = self.ik_model.joint_q.numpy().copy()
        tpl_jq[: self.n_coords] = np.asarray(joint_coords, dtype=tpl_jq.dtype).reshape(-1)[
            : self.n_coords
        ]
        tpl_jqd = np.zeros(int(self.ik_model.joint_dof_count), dtype=np.float32)
        self.ik_model.joint_q.assign(tpl_jq.astype(self.ik_model.joint_q.dtype))
        self.ik_model.joint_qd.assign(tpl_jqd)
        tpl_state = self.ik_model.state()
        newton.eval_fk(
            self.ik_model,
            self.ik_model.joint_q,
            self.ik_model.joint_qd,
            tpl_state,
        )
        bq = tpl_state.body_q.numpy().reshape(-1, 7)[self.tcp_body_index]
        return wp.transform(
            wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
            wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
        )

    def tcp_template_pose_from_model(self, world: int) -> wp.transform:
        """Template-frame TCP FK from the batched model's ``joint_q`` for ``world``."""
        full = self.sim_model.joint_q.numpy().reshape(-1)
        w_jq = full[self.layout.joint_q_slice(world)]
        return self.tcp_pose_from_joint_coords(w_jq)

    def sim_tcp_pose_from_model(self, world: int) -> wp.transform:
        """World-frame TCP pose from ``sim_model.joint_q`` (co-located)."""
        return self.tcp_template_pose_from_model(world)

    def seed_from_state(self, state: Any) -> None:
        """Copy each world's ``joint_q`` slice from the batched model into IK rows."""
        del state
        sim_jq_2d = self.sim_model.joint_q.reshape((self.n_problems, self.n_coords))
        wp.copy(self.joint_q, sim_jq_2d)

    def gather_targets_from_proxy_state(
        self,
        cable_state: Any,
        proxy_indices_wp: wp.array,
    ) -> None:
        """Set IK targets from per-env cable proxy poses (template frame)."""
        wp.launch(
            _gather_proxy_targets_from_cable_kernel,
            dim=self.n_problems,
            inputs=[
                cable_state.body_q,
                proxy_indices_wp,
                self.target_positions,
                self.target_rotations,
            ],
            device=self.device,
        )

    def gather_tcp_targets_from_state(
        self,
        state: Any,
        out_pos_world: wp.array,
        out_rot_world: wp.array,
    ) -> None:
        """Batch-gather TCP poses from ``state.body_q`` into world buffers and IK targets."""
        wp.launch(
            _gather_tcp_world_poses_kernel,
            dim=self.n_problems,
            inputs=[
                state.body_q,
                self._tcp_body_indices_wp,
                out_pos_world,
                out_rot_world,
                self.target_positions,
                self.target_rotations,
            ],
            device=self.device,
        )

    def advance_targets_batch(
        self,
        pos_world: wp.array,
        rot_world: wp.array,
        lin_vels_wp: wp.array,
        ang_vels_wp: wp.array,
        dt: float,
    ) -> None:
        """Integrate per-env twists and upload template-frame targets to the IK objectives."""
        wp.launch(
            _integrate_tcp_targets_kernel,
            dim=self.n_problems,
            inputs=[
                pos_world,
                rot_world,
                lin_vels_wp,
                ang_vels_wp,
                float(dt),
                self.target_positions,
                self.target_rotations,
            ],
            device=self.device,
        )

    def set_target(self, world: int, target_tf: wp.transform) -> None:
        """Set IK target in template frame."""
        pos = wp.transform_get_translation(target_tf)
        rot = wp.transform_get_rotation(target_tf)
        self._pos_obj.set_target_position(world, pos)
        self._rot_obj.set_target_rotation(world, _quat_to_ik_vec4(rot))

    def set_target_world(self, world: int, target_tf_world: wp.transform) -> None:
        """Set IK target from a world-frame TCP pose."""
        self.set_target(world, self.world_to_template(target_tf_world, world))

    def set_target_from_fk(self, state: Any) -> None:
        """Anchor IK targets to each world's TCP pose from ``state.body_q``."""
        pos_world = wp.zeros(self.n_problems, dtype=wp.vec3, device=self.device)
        rot_world = wp.zeros(self.n_problems, dtype=wp.vec4, device=self.device)
        self.gather_tcp_targets_from_state(state, pos_world, rot_world)

    def step(self, iterations: int) -> None:
        self._solver.step(self.joint_q, self.joint_q, iterations=iterations)

    def solver_costs_expanded(self) -> wp.array:
        """Per-seed objective costs from the most recent :meth:`step` call."""
        return self._solver.costs

    def solver_joint_q_expanded(self) -> wp.array:
        """Expanded joint coordinates for all sampled seeds."""
        return self._solver.joint_q

    def pose_error_from_joint_coords(
        self,
        joint_coords: np.ndarray,
        world: int,
        target_tf_world: wp.transform,
    ) -> tuple[float, float]:
        """TCP pose error for ``joint_coords`` vs a world-frame target."""
        target_tpl = self.world_to_template(target_tf_world, world)
        tpl_state = self.ik_model.state()
        return tcp_ik_target_pose_errors(
            self.ik_model,
            tpl_state,
            tcp_body_index=self.tcp_body_index,
            target_tf=target_tpl,
            joint_q=joint_coords,
        )

    def scatter_to_model(self, state: Any, *, eval_fk: bool = True) -> None:
        """Write IK rows into batched ``sim_model`` / ``state`` joint arrays."""
        sim_jq_2d = self.sim_model.joint_q.reshape((self.n_problems, self.n_coords))
        wp.copy(sim_jq_2d, self.joint_q)
        state_jq_2d = state.joint_q.reshape((self.n_problems, self.n_coords))
        wp.copy(state_jq_2d, self.joint_q)
        jqd = np.zeros(int(self.sim_model.joint_dof_count), dtype=np.float32)
        self.sim_model.joint_qd.assign(jqd)
        state.joint_qd.assign(jqd)
        if eval_fk:
            newton.eval_fk(self.sim_model, state.joint_q, state.joint_qd, state)

    def pose_error(self, world: int, target_tf_world: wp.transform) -> tuple[float, float]:
        """TCP pose error for one problem row vs a world-frame ``target_tf_world``."""
        target_tpl = self.world_to_template(target_tf_world, world)
        tpl_state = self.ik_model.state()
        jq = self.joint_q.numpy()[world]
        return tcp_ik_target_pose_errors(
            self.ik_model,
            tpl_state,
            tcp_body_index=self.tcp_body_index,
            target_tf=target_tpl,
            joint_q=jq,
        )

    def max_pose_error(self, targets: list[wp.transform]) -> tuple[float, float]:
        max_pos = 0.0
        max_rot = 0.0
        for w, tf in enumerate(targets):
            pos_err, rot_err = self.pose_error(w, tf)
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
        return max_pos, max_rot

    def pose_errors_per_world(self, targets: list[wp.transform]) -> list[tuple[float, float]]:
        return [self.pose_error(w, targets[w]) for w in range(len(targets))]
