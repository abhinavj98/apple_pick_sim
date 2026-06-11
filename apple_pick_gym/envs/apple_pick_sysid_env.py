"""Gymnasium env for system-ID excitation (§2.1 quasi-static stiffness mapping)."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from apple_pick_gym.envs.apple_pick_vic_env import ApplePickVicEnv
from apple_pick_sim.system_id.excitation_state import ExcitationContext


_EXCITATION_TYPE_TO_INT: dict[str, int] = {
    "quasi_static": 0,
    "translational_chirp": 1,
    "torsional": 2,
}


class ApplePickSysIdEnv(ApplePickVicEnv):
    """VIC env with continuous EE velocity actions and excitation metadata obs.

    Extends :class:`ApplePickVicEnv` for post-grasp quasi-static stepping.
    ``tcp_pos`` is the **actual** TCP body position (not the VIC target) so
    stiffness estimates from ``ft_wrist`` displacement are unbiased by compliance.
    """

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        enable_self_collisions: bool = False,
        fix_to_apple: bool = True,
        fix_to_apple_warmup_substeps: int = 1800,
        max_woody_parts: int = 64,
        mujoco_solver_kwargs: dict[str, Any] | None = None,
        control_hz: float = 60.0,
        vic_linear_k: float = 800.0,
        vic_linear_d: float = 80.0,
        vic_angular_k: float = 40.0,
        vic_angular_d: float = 4.0,
        vic_use_joint_torques: bool = True,
        max_tcp_force_n: float = 30.0,
        max_linear_vel: float = 0.2,
        max_angular_vel: float = 1.0,
    ) -> None:
        self._max_tcp_force_n = float(max_tcp_force_n)
        self._max_linear_vel = float(max_linear_vel)
        self._max_angular_vel = float(max_angular_vel)
        self._excitation_context = ExcitationContext(
            type="quasi_static",
            f_inst=0.0,
            direction=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            enable_self_collisions=enable_self_collisions,
            fix_to_apple=fix_to_apple,
            fix_to_apple_warmup_substeps=fix_to_apple_warmup_substeps,
            max_woody_parts=max_woody_parts,
            mujoco_solver_kwargs=mujoco_solver_kwargs,
            control_hz=control_hz,
            vic_linear_k=vic_linear_k,
            vic_linear_d=vic_linear_d,
            vic_angular_k=vic_angular_k,
            vic_angular_d=vic_angular_d,
            vic_use_joint_torques=vic_use_joint_torques,
        )

    def set_excitation_context(self, ctx: ExcitationContext) -> None:
        self._excitation_context = ctx

    def _setup_action_space(self) -> None:
        lin = self._max_linear_vel
        ang = self._max_angular_vel
        self.action_space = spaces.Box(
            low=np.array([-lin, -lin, -lin, -ang, -ang, -ang], dtype=np.float32),
            high=np.array([lin, lin, lin, ang, ang, ang], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._observation_space_for(self._cfg.max_woody_parts)

    @staticmethod
    def _observation_space_for(n_woody: int) -> spaces.Dict:
        base = ApplePickVicEnv._observation_space_for(n_woody)
        return spaces.Dict(
            {
                **dict(base.spaces),
                "excitation_type": spaces.Discrete(3),
                "excitation_f_inst": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(), dtype=np.float32
                ),
                "excitation_direction": spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "tcp_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )

    def _tcp_pos(self) -> np.ndarray:
        assert self._scene is not None
        tcp = int(self._scene.tcp_body_index)
        bq = self._scene.robot_state_0.body_q.numpy().reshape(-1, 7)
        return np.asarray(bq[tcp, :3], dtype=np.float32)

    def _excitation_obs(self) -> dict[str, Any]:
        ctx = self._excitation_context
        type_int = _EXCITATION_TYPE_TO_INT.get(ctx.type)
        if type_int is None:
            raise ValueError(f"Unknown excitation type {ctx.type!r}")
        return {
            "excitation_type": int(type_int),
            "excitation_f_inst": np.float32(ctx.f_inst),
            "excitation_direction": np.asarray(ctx.direction, dtype=np.float32),
        }

    def _make_obs(self) -> dict[str, Any]:
        obs = super()._make_obs()
        obs.update(self._excitation_obs())
        obs["tcp_pos"] = self._tcp_pos()
        return obs

    def _action_to_command(self, action):
        from apple_pick_sim.robot import fr3_robot

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (6,):
            raise ValueError(f"Expected action shape (6,), got {arr.shape}")
        lin = np.clip(arr[:3], -self._max_linear_vel, self._max_linear_vel)
        ang = np.clip(arr[3:], -self._max_angular_vel, self._max_angular_vel)
        return fr3_robot.EEVelocity(
            linear=(float(lin[0]), float(lin[1]), float(lin[2])),
            angular=(float(ang[0]), float(ang[1]), float(ang[2])),
        )

    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> bool:
        del info
        force = np.asarray(obs["ft_wrist"][:3], dtype=np.float64)
        return float(np.linalg.norm(force)) > self._max_tcp_force_n

    @staticmethod
    def log_movement_direction_arrow(
        viewer,
        obs: dict[str, Any],
        *,
        scene: Any | None = None,
        linear_velocity: tuple[float, float, float] | np.ndarray | None = None,
        length_m: float = 0.4,
        velocity_threshold: float = 1e-6,
    ) -> None:
        """Draw commanded EE linear velocity at the TCP (bright cyan arrow).

        Uses ``linear_velocity`` when provided (preferred). Falls back to
        ``excitation_direction`` in ``obs`` when velocity is omitted.
        """
        from apple_pick_sim.tcp_force_viz import log_direction_arrow, tcp_origin_world

        name = "/gym/movement_direction"
        if scene is None:
            log_direction_arrow(viewer, name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), length_m=length_m)
            return

        if linear_velocity is not None:
            direction = np.asarray(linear_velocity, dtype=np.float64).reshape(3)
        else:
            direction = obs.get("excitation_direction")
            if direction is None:
                log_direction_arrow(
                    viewer, name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), length_m=length_m
                )
                return
            direction = np.asarray(direction, dtype=np.float64).reshape(3)

        if float(np.linalg.norm(direction)) < velocity_threshold:
            log_direction_arrow(viewer, name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), length_m=length_m)
            return

        origin = tcp_origin_world(scene)
        log_direction_arrow(viewer, name, origin, direction, length_m=length_m)
