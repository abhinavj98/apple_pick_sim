"""Open-loop trajectory replay env for sim-to-real calibration rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from apple_pick_gym.envs.apple_pick_base_env import ApplePickBaseEnv


class ApplePickReplayEnv(ApplePickBaseEnv):
    """Replay pre-recorded EE velocity trajectories in the coupled sim.

    Observations (all ``float32``):

    - ``ft_wrist``: ``(6,)`` TCP coupling wrench ``[F(3), tau(3)]`` [N, N·m]
    - ``woody_start``: ``(N*3,)`` proximal joint-anchor positions [m]
    - ``woody_end``: ``(N*3,)`` distal child-body COM positions [m]
    - ``tcp_velocity``: ``(6,)`` TCP spatial velocity ``[v(3), omega(3)]`` [m/s, rad/s]

    Action contract: continuous EE velocity command per step (``Box(6)``).
    Use ``reset(options={'params': FruitingSystemParams(...)})`` to inject tuned params.
    """

    @staticmethod
    def _observation_space_for(n_woody: int) -> spaces.Dict:
        return spaces.Dict(
            {
                "ft_wrist": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "woody_start": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 3,), dtype=np.float32
                ),
                "woody_end": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n_woody * 3,), dtype=np.float32
                ),
                "tcp_velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
            }
        )

    def _setup_action_space(self) -> None:
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.observation_space = self._observation_space_for(self._cfg.max_woody_parts)

    def _setup_observation_space(self) -> None:
        self.observation_space = self._observation_space_for(self._n_woody_parts)

    def _make_obs(self) -> dict[str, Any]:
        start_pos, end_pos = self._woody_start_end_pos()
        return {
            "ft_wrist": self._end_effector_wrench(),
            "woody_start": start_pos,
            "woody_end": end_pos,
            "tcp_velocity": self._tcp_velocity(),
        }

    def _action_to_command(self, action):
        from apple_pick_sim.robot import fr3_robot

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {arr.shape}")
        return fr3_robot.EEVelocity(
            linear=(float(arr[0]), float(arr[1]), float(arr[2])),
            angular=(float(arr[3]), float(arr[4]), float(arr[5])),
        )

    def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> float:
        del obs, info
        return 0.0

    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> bool:
        del obs, info
        return False
