"""Batched multi-env FR3 variable-impedance teleop (target integration only, no IK)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.robot.fr3_robot.batched_template_ik import BatchedTemplateIK
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import (
    EEVelocity,
    _KeyViewer,
    read_keyboard_ee_velocity,
)


class Fr3BatchedEEImpedanceController:
    """Per-env TCP target integration for batched VIC (joint-torque path)."""

    def __init__(
        self,
        robot_model: newton.Model,
        layout: BatchedEnvLayout,
        ik_robot_model: newton.Model,
        ik_tcp_body_index: int,
        *,
        velocity_for_world: Callable[[int], EEVelocity] | None = None,
        linear_speed: float = 0.2,
        angular_speed: float = 1.0,
    ) -> None:
        self.robot_model = robot_model
        self.layout = layout
        self.tcp_body_index = int(layout.tcp_body_indices[0])
        self._velocity_for_world = velocity_for_world
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        n = layout.num_envs
        dev = robot_model.device
        self.target_tf = [
            wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
            for _ in range(n)
        ]
        self._target_pos_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._target_rot_wp = wp.zeros(n, dtype=wp.vec4, device=dev)
        self._lin_vels_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._ang_vels_wp = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._ik = BatchedTemplateIK(
            ik_model=ik_robot_model,
            sim_model=robot_model,
            layout=layout,
            tcp_body_index=ik_tcp_body_index,
        )

    def _velocity_for(self, world: int, velocity: EEVelocity | None) -> EEVelocity:
        if self._velocity_for_world is not None:
            return self._velocity_for_world(world)
        if velocity is not None:
            return velocity
        return EEVelocity()

    def _sync_target_tf_from_device(self) -> None:
        pos_np = self._target_pos_wp.numpy()
        rot_np = self._target_rot_wp.numpy()
        for w in range(self.layout.num_envs):
            p = pos_np[w]
            r = rot_np[w]
            self.target_tf[w] = wp.transform(
                wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                wp.quat(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
            )

    def sync_target_from_state(self, state: Any) -> None:
        self._ik.gather_tcp_targets_from_state(
            state, self._target_pos_wp, self._target_rot_wp
        )
        self._sync_target_tf_from_device()

    def advance_target(
        self,
        dt: float,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
        lock_angular: bool = False,
    ) -> EEVelocity:
        if velocity is None and self._velocity_for_world is None:
            velocity = read_keyboard_ee_velocity(
                viewer,
                linear_speed=self.linear_speed,
                angular_speed=self.angular_speed,
                poll_events=poll_events,
            )
        lin_np = np.zeros((self.layout.num_envs, 3), dtype=np.float32)
        ang_np = np.zeros((self.layout.num_envs, 3), dtype=np.float32)
        for w in range(self.layout.num_envs):
            v = self._velocity_for(w, velocity)
            if lock_angular:
                v = EEVelocity(linear=v.linear, angular=(0.0, 0.0, 0.0))
            lin_np[w] = v.linear
            ang_np[w] = v.angular
        self._lin_vels_wp.assign(lin_np)
        self._ang_vels_wp.assign(ang_np)
        self._ik.advance_targets_batch(
            self._target_pos_wp,
            self._target_rot_wp,
            self._lin_vels_wp,
            self._ang_vels_wp,
            dt,
        )
        self._sync_target_tf_from_device()
        return velocity if velocity is not None else self._velocity_for(0, None)

    def command_velocity_for_world(
        self, world: int, *, fallback: EEVelocity | None = None
    ) -> EEVelocity:
        return self._velocity_for(world, fallback)

    def tcp_world_pose(self, world: int) -> wp.transform:
        return self._ik.sim_tcp_pose_from_model(world)

    def stage_targets_to_scene(self, scene: Any) -> None:
        """Wire device target buffers into ``scene`` for batched VIC substeps."""
        scene.vic_target_positions_wp = self._target_pos_wp
        scene.vic_target_rotations_wp = self._target_rot_wp

    def run_coupled_teleop_frame(
        self,
        state: Any,
        control: Any,
        mj_solver: Any,
        dt: float,
        *,
        viewer: _KeyViewer | None = None,
        velocity: EEVelocity | None = None,
    ) -> EEVelocity:
        del control, mj_solver
        velocity = self.advance_target(
            dt,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        return velocity
