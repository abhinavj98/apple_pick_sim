"""M2.1 Gymnasium Env wrapping the M1 coupled sim (FR3, direct-joint control).

This environment is a thin adapter over public `apple_pick_sim` APIs:
- Build: `apple_pick_sim.coupled_fruiting.build_coupled_fruiting_fr3`
- Control: `CoupledFruitingScene.update_fr3_ee_teleop_direct` with `Fr3EEDirectJointController`
- Step: control_hz-derived substeps × `CoupledFruitingScene.coupled_substep(SUB_DT)`

Observation space is re-declared on each ``reset()`` with the actual woody-part count ``N``
from ``measure_fruiting_forces`` (topology can vary with sampled segment counts). ``N`` is
also exposed in ``info["n_woody_parts"]``.

No Gymnasium imports are allowed in `apple_pick_sim/`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from apple_pick_gym.envs.apple_pick_base_env import ApplePickBaseEnv


class ApplePickCoupledEnv(ApplePickBaseEnv):
    """Gymnasium env for the coupled FR3 + fruiting-system simulation (M2.1).

    Observations (all ``float32``):

    - ``woody_part_start_pos``: ``(N*3,)`` parent-side fixed-joint anchor positions [m]
    - ``woody_part_end_pos``: ``(N*3,)`` child-side fixed-joint anchor positions [m]
    - ``woody_part_force``: ``(N*6,)`` fixed-joint wrenches ``[F(3), tau(3)]`` [N, N·m]
      (index order matches :attr:`~apple_pick_gym.envs.apple_pick_base_env.ApplePickBaseEnv.junction_names`)
    - ``apple_pos``: ``(3,)`` apple body world position [m]
    - ``tcp_force``: ``(6,)`` harvested TCP coupling wrench [N, N·m]
    - ``tcp_velocity``: ``(6,)`` TCP spatial velocity ``[v(3), omega(3)]`` [m/s, rad/s]

    Action contract: single keyboard-style command per step (``Discrete(13)``).
    """

    @staticmethod
    def _observation_space_for(n_woody: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "woody_part_start_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 3,), dtype=np.float32
                ),
                "woody_part_end_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 3,), dtype=np.float32
                ),
                "woody_part_force": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 6,), dtype=np.float32
                ),
                "apple_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "tcp_force": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "tcp_velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            }
        )

    def _setup_action_space(self) -> None:
        self.action_space = spaces.Discrete(13)
        self.observation_space = self._observation_space_for(self._cfg.max_woody_parts)

    def _setup_observation_space(self) -> None:
        self.observation_space = self._observation_space_for(self._n_woody_parts)

    def _make_obs(self) -> dict[str, Any]:
        _, _, sub_dt = self._timing_constants()
        start_pos, end_pos = self._woody_start_end_pos()
        return {
            "woody_part_start_pos": start_pos,
            "woody_part_end_pos": end_pos,
            "woody_part_force": self._woody_part_forces(sub_dt),
            "apple_pos": self._apple_pos(),
            "tcp_force": self._end_effector_wrench(),
            "tcp_velocity": self._tcp_velocity(),
        }

    def _action_to_command(self, action: int):
        from apple_pick_sim.robot import fr3_robot

        a = int(action)
        if a < 0 or a >= 13:
            raise ValueError(f"Invalid action {a}; expected 0..12")

        lin = 0.2
        ang = 1.0

        if a == 12:
            return fr3_robot.EEVelocity()

        if a == 0:
            return fr3_robot.EEVelocity(linear=(+lin, 0.0, 0.0))
        if a == 1:
            return fr3_robot.EEVelocity(linear=(-lin, 0.0, 0.0))
        if a == 2:
            return fr3_robot.EEVelocity(linear=(0.0, +lin, 0.0))
        if a == 3:
            return fr3_robot.EEVelocity(linear=(0.0, -lin, 0.0))
        if a == 4:
            return fr3_robot.EEVelocity(linear=(0.0, 0.0, +lin))
        if a == 5:
            return fr3_robot.EEVelocity(linear=(0.0, 0.0, -lin))
        if a == 6:
            return fr3_robot.EEVelocity(angular=(+ang, 0.0, 0.0))
        if a == 7:
            return fr3_robot.EEVelocity(angular=(-ang, 0.0, 0.0))
        if a == 8:
            return fr3_robot.EEVelocity(angular=(0.0, +ang, 0.0))
        if a == 9:
            return fr3_robot.EEVelocity(angular=(0.0, -ang, 0.0))
        if a == 10:
            return fr3_robot.EEVelocity(angular=(0.0, 0.0, +ang))
        if a == 11:
            return fr3_robot.EEVelocity(angular=(0.0, 0.0, -ang))

        raise RuntimeError("Unreachable action mapping")

    def _action_to_velocity(self, action: int):
        """Backward-compatible alias used by parity tests and keyboard examples."""
        return self._action_to_command(action)

    def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> float:
        del obs, info
        return 0.0

    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> bool:
        del obs, info
        return False
