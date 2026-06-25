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

        dev = self.device
        n = self.n_problems
        self.target_positions = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.target_rotations = wp.zeros(n, dtype=wp.vec4, device=dev)

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
        )

    def _world_origin_offset(self, world: int) -> tuple[float, float, float]:
        return self.layout.world_origin(world)

    def template_to_world(self, tf_template: wp.transform, world: int) -> wp.transform:
        """Express a template-frame TCP pose in world coordinates."""
        ox, oy, oz = self._world_origin_offset(world)
        p = wp.transform_get_translation(tf_template)
        return wp.transform(
            wp.vec3(float(p[0]) + ox, float(p[1]) + oy, float(p[2]) + oz),
            wp.transform_get_rotation(tf_template),
        )

    def world_to_template(self, tf_world: wp.transform, world: int) -> wp.transform:
        """Express a world-frame TCP pose in the single-world template frame."""
        ox, oy, oz = self._world_origin_offset(world)
        p = wp.transform_get_translation(tf_world)
        return wp.transform(
            wp.vec3(float(p[0]) - ox, float(p[1]) - oy, float(p[2]) - oz),
            wp.transform_get_rotation(tf_world),
        )

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
        """World-frame TCP pose from ``sim_model.joint_q`` (adds replicate origin offset)."""
        tpl = self.tcp_template_pose_from_model(world)
        ox, oy, oz = self._world_origin_offset(world)
        p = wp.transform_get_translation(tpl)
        return wp.transform(
            wp.vec3(float(p[0]) + ox, float(p[1]) + oy, float(p[2]) + oz),
            wp.transform_get_rotation(tpl),
        )

    def seed_from_state(self, state: Any) -> None:
        """Copy each world's ``joint_q`` slice from the batched model into IK rows."""
        del state
        full = self.sim_model.joint_q.numpy().reshape(-1)
        rows = np.zeros((self.n_problems, self.n_coords), dtype=np.float32)
        for w in range(self.n_problems):
            rows[w] = full[self.layout.joint_q_slice(w)].astype(np.float32, copy=False)
        self.joint_q.assign(rows)

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
        """Anchor IK targets to each world's TCP pose from ``sim_model.joint_q``."""
        del state
        for w in range(self.n_problems):
            self.set_target_world(w, self.sim_tcp_pose_from_model(w))

    def step(self, iterations: int) -> None:
        self._solver.step(self.joint_q, self.joint_q, iterations=iterations)

    def scatter_to_model(self, state: Any, *, eval_fk: bool = True) -> np.ndarray:
        """Write IK rows into batched ``sim_model`` / ``state`` joint arrays."""
        full = self.sim_model.joint_q.numpy().copy()
        rows = self.joint_q.numpy()
        for w in range(self.n_problems):
            sl = self.layout.joint_q_slice(w)
            full[sl] = rows[w].astype(full.dtype, copy=False)
        jqd = np.zeros(int(self.sim_model.joint_dof_count), dtype=np.float32)
        self.sim_model.joint_q.assign(full)
        self.sim_model.joint_qd.assign(jqd)
        state.joint_q.assign(full)
        state.joint_qd.assign(jqd)
        if eval_fk:
            newton.eval_fk(self.sim_model, state.joint_q, state.joint_qd, state)
        return full

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
