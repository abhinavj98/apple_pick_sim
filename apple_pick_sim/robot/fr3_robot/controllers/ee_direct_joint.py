"""Direct joint-write FR3 controller (kinematic testing)."""

from __future__ import annotations

from typing import Any

from newton.solvers import SolverMuJoCo

from apple_pick_sim.robot.fr3_robot.controllers.ee_velocity import Fr3EEVelocityController
from apple_pick_sim.robot.fr3_robot.setup import (
    init_mujoco_actuator_targets_from_model,
    sync_mujoco_visual_state,
)

class Fr3EEDirectJointController(Fr3EEVelocityController):
    """Testing controller: TCP velocity + IK, then **direct** ``joint_q`` write (kinematic arm).

    Pair with :meth:`~apple_pick_sim.coupled_fruiting.CoupledFruitingScene.update_fr3_ee_teleop_direct`
    and ``robot_kinematic_mode=True`` so ``SolverMuJoCo`` does not re-integrate the arm between
    substeps. Coupled VBD + proxy sync still run.
    """

    def apply_direct_joints(
        self,
        state: Any,
        control: Any | None = None,
        *,
        mj_solver: SolverMuJoCo | None = None,
    ) -> None:
        """Write IK ``joint_q`` into model/state, zero ``joint_qd``, refresh FK and MuJoCo buffers."""
        self.apply_to_model_and_state(state)
        if control is not None:
            init_mujoco_actuator_targets_from_model(self.robot_model, control)
        if mj_solver is not None:
            sync_mujoco_visual_state(mj_solver, self.robot_model, state)
