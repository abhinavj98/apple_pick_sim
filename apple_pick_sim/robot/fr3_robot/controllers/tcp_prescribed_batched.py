"""Batched kinematic FR3 that follows a prescribed TCP pose (arm-free plant replay).

``controller.mode="tcp_pose"``: each frame's ``(N, 7)`` action ``[pos, quat_wxyz]`` is an
absolute world TCP target for the *end* of the frame. The arm is kinematic
(``scene.robot_kinematic_mode``): per frame, batched IK on the template solves the
end-of-frame joint coordinates (warm-started from the current ones); per substep,
:meth:`write_substep` linearly interpolates ``joint_q`` and holds the matching constant
``joint_qd``, so the kinematic FK prefix mirrors a smoothly moving TCP onto the gripper
proxy instead of a once-per-frame teleport.

No arm dynamics, gains, friction, or task-space inertia enter the plant's boundary
condition — that is the point (see ``apple_pick_sim.system_id.tcp_prescription``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity_batched import (
    Fr3BatchedEEVelocityController,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity
from apple_pick_sim.robot.fr3_robot.setup import sync_mujoco_visual_state

TCP_PRESCRIBED_IK_ITERATIONS = 32
"""Warm-started per-frame IK iterations (frame-to-frame TCP motion is sub-mm)."""


@wp.kernel(enable_backward=False)
def _interp_joint_coords_kernel(
    q_start: wp.array2d(dtype=wp.float32),
    q_end: wp.array2d(dtype=wp.float32),
    alpha: float,
    inv_frame_dt: float,
    model_q: wp.array2d(dtype=wp.float32),
    state_q: wp.array2d(dtype=wp.float32),
    model_qd: wp.array2d(dtype=wp.float32),
    state_qd: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    q0 = q_start[i, j]
    dq = q_end[i, j] - q0
    q = q0 + alpha * dq
    qd = dq * inv_frame_dt
    model_q[i, j] = q
    state_q[i, j] = q
    model_qd[i, j] = qd
    state_qd[i, j] = qd


@wp.kernel(enable_backward=False)
def _tcp_target_residual_kernel(
    body_q: wp.array(dtype=wp.transform),
    tcp_indices: wp.array(dtype=int),
    target_pos: wp.array(dtype=wp.vec3),
    target_rot: wp.array(dtype=wp.vec4),
    out: wp.array2d(dtype=wp.float32),
):
    """Per-env ``[|Δp| (m), rotation angle (rad)]`` of the realized TCP vs its target."""
    i = wp.tid()
    tf = body_q[tcp_indices[i]]
    out[i, 0] = wp.length(wp.transform_get_translation(tf) - target_pos[i])
    r = wp.transform_get_rotation(tf)
    t = target_rot[i]
    d = wp.abs(r[0] * t[0] + r[1] * t[1] + r[2] * t[2] + r[3] * t[3])
    out[i, 1] = 2.0 * wp.acos(wp.min(d, 1.0))


class Fr3BatchedTcpPrescribedController(Fr3BatchedEEVelocityController):
    """Kinematic per-env TCP pose follower: per-frame IK, per-substep interpolation."""

    def __init__(
        self,
        robot_model: newton.Model,
        layout: BatchedEnvLayout,
        ik_robot_model: newton.Model,
        ik_tcp_body_index: int,
        *,
        velocity_for_world: Callable[[int], EEVelocity] | None = None,
        linear_speed: float = 1.0,
        angular_speed: float = 1.0,
        ik_iterations: int = TCP_PRESCRIBED_IK_ITERATIONS,
        joint_limit_weight: float = 10.0,
    ) -> None:
        super().__init__(
            robot_model,
            layout,
            ik_robot_model,
            ik_tcp_body_index,
            velocity_for_world=velocity_for_world,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            ik_iterations=ik_iterations,
            joint_limit_weight=joint_limit_weight,
        )
        n = int(layout.num_envs)
        n_coords = int(self._ik.n_coords)
        if int(self._ik.n_dofs) != n_coords:
            raise ValueError(
                "tcp_pose interpolation assumes joint_coord_count == joint_dof_count per "
                f"world, got coords={n_coords} dofs={self._ik.n_dofs}"
            )
        dev = robot_model.device
        self._n_coords = n_coords
        self._q_start = wp.zeros((n, n_coords), dtype=wp.float32, device=dev)
        self._q_end = wp.zeros((n, n_coords), dtype=wp.float32, device=dev)
        self._inv_frame_dt = 0.0
        self.ik_residual = wp.zeros((n, 2), dtype=wp.float32, device=dev)
        """Per-env ``[pos_err_m, rot_err_rad]`` after the last completed frame."""

    def _model_q_2d(self) -> wp.array:
        return self.robot_model.joint_q.reshape((self.layout.num_envs, self._n_coords))

    def _model_qd_2d(self) -> wp.array:
        return self.robot_model.joint_qd.reshape((self.layout.num_envs, self._n_coords))

    def set_pose_targets(self, actions) -> None:
        """Upload ``(N, 7)`` ``[pos, quat_wxyz]`` rows as world TCP targets (IK + residual)."""
        import torch

        n = int(self.layout.num_envs)
        a = actions.to(device=wp.device_to_torch(self.robot_model.device), dtype=torch.float32)
        if a.shape != (n, 7):
            raise ValueError(f"tcp_pose action must have shape ({n}, 7), got {tuple(a.shape)}")
        pos = a[:, 0:3].contiguous()
        quat_wxyz = a[:, 3:7]
        norm = torch.linalg.norm(quat_wxyz, dim=1, keepdim=True).clamp_min(1.0e-9)
        quat_xyzw = (quat_wxyz / norm)[:, [1, 2, 3, 0]].contiguous()
        pos_wp = wp.from_torch(pos, dtype=wp.vec3)
        rot_wp = wp.from_torch(quat_xyzw, dtype=wp.vec4)
        # Co-located worlds: world frame == IK template frame.
        wp.copy(self._target_pos_wp, pos_wp)
        wp.copy(self._target_rot_wp, rot_wp)
        wp.copy(self._ik.target_positions, pos_wp)
        wp.copy(self._ik.target_rotations, rot_wp)

    def begin_frame(self, state: Any, actions, frame_dt: float) -> None:
        """Solve end-of-frame joint coordinates for this frame's TCP targets."""
        if frame_dt <= 0.0:
            raise ValueError(f"frame_dt must be positive, got {frame_dt}")
        del state
        wp.copy(self._q_start, self._model_q_2d())
        self.set_pose_targets(actions)
        self._ik.seed_from_state(None)
        self._ik.step(self.ik_iterations)
        wp.copy(self._q_end, self._ik.joint_q)
        self._inv_frame_dt = 1.0 / float(frame_dt)

    def hold_frame(self) -> None:
        """Hold the current joint coordinates for the next frame (zero velocity)."""
        q = self._model_q_2d()
        wp.copy(self._q_start, q)
        wp.copy(self._q_end, q)

    def write_substep(self, state: Any, alpha: float) -> None:
        """Write interpolated ``joint_q`` (and constant ``joint_qd``) for one substep."""
        n = int(self.layout.num_envs)
        wp.launch(
            _interp_joint_coords_kernel,
            dim=(n, self._n_coords),
            inputs=[
                self._q_start,
                self._q_end,
                float(alpha),
                float(self._inv_frame_dt),
                self._model_q_2d(),
                state.joint_q.reshape((n, self._n_coords)),
                self._model_qd_2d(),
                state.joint_qd.reshape((n, self._n_coords)),
            ],
            device=self.robot_model.device,
        )

    def hold_still(self, state: Any) -> None:
        """Zero ``joint_qd`` (model + state) so a held kinematic arm mirrors no velocity."""
        self.robot_model.joint_qd.zero_()
        state.joint_qd.zero_()

    def measure_residual(self, state: Any) -> wp.array:
        """Fill :attr:`ik_residual` from ``state.body_q`` (call after the frame's last substep)."""
        wp.launch(
            _tcp_target_residual_kernel,
            dim=int(self.layout.num_envs),
            inputs=[
                state.body_q,
                self._ik._tcp_body_indices_wp,
                self._target_pos_wp,
                self._target_rot_wp,
                self.ik_residual,
            ],
            device=self.robot_model.device,
        )
        return self.ik_residual

    def sync_visual(self, mj_solver: Any, state: Any) -> None:
        if mj_solver is not None:
            sync_mujoco_visual_state(mj_solver, self.robot_model, state)
