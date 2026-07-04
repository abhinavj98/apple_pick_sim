"""Batched direct joint-write FR3 controller (kinematic testing)."""

from __future__ import annotations

from typing import Any

from newton.solvers import SolverMuJoCo

from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity_batched import (
    Fr3BatchedEEVelocityController,
)
from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity, _KeyViewer
from apple_pick_sim.robot.fr3_robot.setup import (
    init_mujoco_actuator_targets_from_model,
    sync_mujoco_visual_state,
)


class Fr3BatchedEEDirectJointController(Fr3BatchedEEVelocityController):
    """Kinematic batched teleop: IK on template, direct ``joint_q`` write per world."""

    def apply_direct_joints(
        self,
        state: Any,
        control: Any | None = None,
        *,
        mj_solver: SolverMuJoCo | None = None,
    ) -> None:
        self.apply_to_model_and_state(state)
        if control is not None:
            init_mujoco_actuator_targets_from_model(self.robot_model, control)
        if mj_solver is not None:
            sync_mujoco_visual_state(mj_solver, self.robot_model, state)

    def run_coupled_teleop_frame_from_actions(
        self,
        state: Any,
        control: Any,
        mj_solver: Any,
        dt: float,
        actions,
        *,
        lock_angular: bool = False,
    ) -> EEVelocity:
        velocity = self.run_ik_teleop_frame_from_actions(
            dt,
            state,
            actions,
            lock_angular=lock_angular,
        )
        self.apply_direct_joints(state, control, mj_solver=mj_solver)
        return velocity

    def run_ik_teleop_frame_from_actions(
        self,
        dt: float,
        state: Any,
        actions,
        *,
        lock_angular: bool = False,
    ) -> EEVelocity:
        self.sync_target_from_state(state)
        self.advance_target_from_actions(dt, actions, lock_angular=lock_angular)
        self.solve_ik(state)
        return EEVelocity()

    def run_coupled_teleop_frame(
        self,
        state: Any,
        control: Any,
        mj_solver: SolverMuJoCo,
        dt: float,
        *,
        viewer: _KeyViewer | None = None,
        velocity: EEVelocity | None = None,
    ) -> EEVelocity:
        velocity = self.run_ik_teleop_frame(
            dt,
            state,
            velocity=velocity,
            viewer=viewer,
            poll_events=True,
        )
        self.apply_direct_joints(state, control, mj_solver=mj_solver)
        return velocity
