"""M2.1 Gymnasium Env wrapping the M1 coupled sim (FR3, direct-joint control).

This environment is a thin adapter over public `apple_pick_sim` APIs:
- Build: `apple_pick_sim.coupled_fruiting.build_coupled_fruiting_fr3`
- Control: `CoupledFruitingScene.apply_fr3_ee_teleop_direct` with `Fr3EEDirectJointController`
- Step: `SUBSTEPS_PER_FRAME` × `CoupledFruitingScene.coupled_substep(SUB_DT)`

No Gymnasium imports are allowed in `apple_pick_sim/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "apple_pick_gym requires gymnasium to be installed. "
        "Install via the Newton uv environment (e.g. newton[dev])."
    ) from e


@dataclass
class _EnvConfig:
    max_episode_steps: int
    enable_self_collisions: bool
    mujoco_solver_kwargs: dict[str, Any]
    fix_to_apple: bool


class ApplePickCoupledEnv(gym.Env):
    """Gymnasium env for the coupled FR3 + fruiting-system simulation (M2.1).

    Observation contract is a placeholder `Dict` until sensors land; action contract is
    a single keyboard-style command per step (Discrete(13)).
    """

    metadata = {"render_modes": [None], "render_fps": 60}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        enable_self_collisions: bool = False,
        fix_to_apple: bool = False,
        mujoco_solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if render_mode not in (None, "none"):
            raise ValueError("Only headless operation is supported in M2.1 (render_mode=None).")

        self._cfg = _EnvConfig(
            max_episode_steps=int(max_episode_steps),
            enable_self_collisions=bool(enable_self_collisions),
            mujoco_solver_kwargs=dict(mujoco_solver_kwargs or {"disable_contacts": True}),
            fix_to_apple=bool(fix_to_apple),
        )

        # M2.1a placeholder observation: Dict with dummy values + schema versioning.
        self.observation_space = spaces.Dict(
            {
                "dummy": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "schema_version": spaces.Discrete(2),
            }
        )

        # M2.1b action contract: single keyboard-style command per step.
        self.action_space = spaces.Discrete(13)

        self._step_count = 0
        self._scene = None
        self._controller = None

    # --- Helpers ---

    def _fixture_ranges_path(self) -> Path:
        # `apple_pick_sim` may be a namespace package in this repo layout (no `__file__`).
        # Use importlib.resources to locate packaged fixture data robustly.
        import importlib.resources as resources

        return (
            resources.files("apple_pick_sim")
            / "fixtures"
            / "fruiting_system_ranges_example_variance.json"
        )

    def _make_obs(self) -> dict[str, Any]:
        return {"dummy": np.zeros((1,), dtype=np.float32), "schema_version": 1}

    def _end_effector_wrench(self) -> np.ndarray:
        from apple_pick_sim.coupling_force_debug import read_tcp_wrench

        assert self._scene is not None and self._scene.proxy_forces is not None
        return read_tcp_wrench(self._scene.proxy_forces, self._scene.tcp_body_index).astype(
            np.float32
        )

    def _fruiting_link_forces(self, sub_dt: float) -> dict[str, dict[str, Any]]:
        import apple_pick_sim.fruiting_system as fs

        assert self._scene is not None
        cable = self._scene.cable
        measured = fs.measure_fruiting_forces(
            cable,
            cable.state_0.body_q,
            cable.state_1.body_q,
            dt=float(sub_dt),
        )
        out: dict[str, dict[str, Any]] = {}
        for rec in measured["fixed_joints"]:
            key = rec.label.removeprefix("joint_") or rec.label
            out[key] = {
                "joint_index": int(rec.joint_index),
                "child_body": int(rec.child_body),
                "force_world": np.asarray(rec.force_world, dtype=np.float32),
                "torque_at_child_com_world": np.asarray(
                    rec.torque_at_child_com_world, dtype=np.float32
                ),
            }
        return out

    def _make_info(self) -> dict[str, Any]:
        import apple_pick_sim.fruiting_system as fs

        assert self._scene is not None
        _, _, sub_dt = self._timing_constants()
        return {
            "step_count": int(self._step_count),
            "params_fingerprint": fs.params_fingerprint(self._scene.cable.params),
            "end_effector_wrench": self._end_effector_wrench(),
            "fruiting_link_forces": self._fruiting_link_forces(sub_dt),
        }

    def _timing_constants(self) -> tuple[float, int, float]:
        # Keep the env contract aligned with existing direct-path tests.
        from apple_pick_sim.tests.conftest import FRAME_DT, SUBSTEPS_PER_FRAME, SUB_DT

        return float(FRAME_DT), int(SUBSTEPS_PER_FRAME), float(SUB_DT)

    def _action_to_velocity(self, action: int):
        from apple_pick_sim.robot import fr3_robot

        a = int(action)
        if a < 0 or a >= 13:
            raise ValueError(f"Invalid action {a}; expected 0..12")

        # Keep the same magnitudes as the FR3 controller defaults (linear_speed=0.2, angular_speed=1.0).
        lin = 0.2
        ang = 1.0

        if a == 12:
            return fr3_robot.EEVelocity()

        # 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z, 6:+rotX, 7:-rotX, 8:+rotY, 9:-rotY, 10:+rotZ, 11:-rotZ
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

    # --- Gymnasium API ---

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        import apple_pick_sim.coupled_fruiting as cf
        import apple_pick_sim.fruiting_system as fs
        from apple_pick_sim.robot import fr3_robot

        # Deterministic build from seed; default to 0 if unspecified.
        scene_seed = int(0 if seed is None else seed)
        ranges = fs.load_ranges(self._fixture_ranges_path())

        self._scene = cf.build_coupled_fruiting_fr3(
            ranges,
            scene_seed,
            enable_self_collisions=self._cfg.enable_self_collisions,
            mujoco_solver_kwargs=self._cfg.mujoco_solver_kwargs,
            gripper_proxy=fs.GripperProxyConfig(
                mass=fr3_robot.EE_MASS_KG,
                box_half_extents=fr3_robot.EE_BOX_HALF_EXTENTS,
                fix_to_apple=self._cfg.fix_to_apple,
            ),
        )

        # Kinematic robot mode for stable, deterministic direct-joint control.
        self._scene.robot_kinematic_mode = True

        self._controller = fr3_robot.Fr3EEDirectJointController(
            self._scene.robot_model, self._scene.tcp_body_index
        )
        self._controller.sync_target_from_state(self._scene.robot_state_0)

        self._step_count = 0
        return self._make_obs(), self._make_info()

    def step(self, action):
        if self._scene is None or self._controller is None:
            raise RuntimeError("Environment must be reset() before step().")

        frame_dt, substeps_per_frame, sub_dt = self._timing_constants()
        vel = self._action_to_velocity(int(action))

        # Apply one teleop frame (updates desired IK target + writes joint_q directly).
        self._scene.apply_fr3_ee_teleop_direct(frame_dt, self._controller, velocity=vel)

        # Integrate coupled sim for one control frame.
        for _ in range(substeps_per_frame):
            self._scene.coupled_substep(sub_dt)

        self._step_count += 1

        obs = self._make_obs()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self._cfg.max_episode_steps
        info = self._make_info()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._scene = None
        self._controller = None

