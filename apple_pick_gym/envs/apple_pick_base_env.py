"""Abstract Gymnasium base for coupled FR3 + fruiting-system environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
        "Install from the repo root (e.g. uv sync --extra gym)."
    ) from e


@dataclass
class _EnvConfig:
    max_episode_steps: int
    enable_self_collisions: bool
    mujoco_solver_kwargs: dict[str, Any]
    fix_to_apple: bool
    fix_to_apple_warmup_substeps: int
    max_woody_parts: int
    control_hz: float


class ApplePickBaseEnv(gym.Env, ABC):
    """Shared scene scaffold for coupled FR3 + fruiting-system Gymnasium envs.

    Concrete subclasses implement action/observation spaces, action decoding,
    reward, and termination. Scene build, timing, sensor helpers, and optional
    ``FruitingSystemParams`` injection live here.

    After ``reset()``, :attr:`junction_names` and :meth:`junction_forces_dict` expose
    per-junction branch wrenches keyed by labels such as ``"stem_apple"``.
    """

    metadata = {"render_modes": [None], "render_fps": 60}
    SUB_DT: float = 1.0 / 1800.0

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        enable_self_collisions: bool = False,
        fix_to_apple: bool = False,
        fix_to_apple_warmup_substeps: int = 1800,
        max_woody_parts: int = 64,
        mujoco_solver_kwargs: dict[str, Any] | None = None,
        control_hz: float = 60.0,
        device: str | None = None,
    ) -> None:
        if render_mode not in (None, "none"):
            raise ValueError("Only headless operation is supported (render_mode=None).")

        self._cfg = _EnvConfig(
            max_episode_steps=int(max_episode_steps),
            enable_self_collisions=bool(enable_self_collisions),
            mujoco_solver_kwargs=dict(mujoco_solver_kwargs or {"disable_contacts": True}),
            fix_to_apple=bool(fix_to_apple),
            fix_to_apple_warmup_substeps=int(fix_to_apple_warmup_substeps),
            max_woody_parts=int(max_woody_parts),
            control_hz=float(control_hz),
        )

        self._step_count = 0
        self._scene = None
        self._controller = None
        self._n_woody_parts = 0
        self._pending_weld_direction: tuple[float, float, float] | None = None
        self._pending_weld_reference_pos: tuple[float, float, float] | None = None
        self._pending_weld_reference_quat: tuple[float, float, float, float] | None = None
        self._reset_fruiting_base_pos: tuple[float, float, float] | None = None
        self._reset_robot_base_pos: tuple[float, float, float] | None = None
        self._device = device

        self._setup_action_space()

    # --- Abstract hooks ---

    @abstractmethod
    def _setup_action_space(self) -> None:
        """Assign ``self.action_space``."""

    @abstractmethod
    def _setup_observation_space(self) -> None:
        """Assign ``self.observation_space`` after scene build in ``reset()``."""

    @abstractmethod
    def _action_to_command(self, action: Any):
        """Decode a Gymnasium action into an FR3 EE velocity command."""

    @abstractmethod
    def _make_obs(self) -> dict[str, Any]:
        """Build an observation matching ``self.observation_space``."""

    @abstractmethod
    def compute_reward(self, obs: dict[str, Any], info: dict[str, Any]) -> float:
        """Return scalar reward for the current step."""

    @abstractmethod
    def compute_terminated(self, obs: dict[str, Any], info: dict[str, Any]) -> bool:
        """Return whether the episode should terminate."""

    # --- Timing / scene plumbing ---

    def _timing_constants(self) -> tuple[float, int, float]:
        substeps_per_step = max(1, round(1.0 / (self._cfg.control_hz * self.SUB_DT)))
        frame_dt = substeps_per_step * self.SUB_DT
        return float(frame_dt), int(substeps_per_step), float(self.SUB_DT)

    def _fixture_ranges_path(self) -> Path:
        from apple_pick_sim.fruiting_system import default_ranges_fixture_path

        return default_ranges_fixture_path()

    @staticmethod
    def _coerce_optional_xyz(value: Any, *, field: str) -> tuple[float, float, float] | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError(f"{field} must be a length-3 position")
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    def _coupled_build_kwargs(self, *, params: Any | None = None) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "enable_self_collisions": self._cfg.enable_self_collisions,
            "mujoco_solver_kwargs": self._cfg.mujoco_solver_kwargs,
            "ik_bootstrap_iterations": 256,
        }
        if self._device is not None:
            kw["device"] = self._device
        if self._reset_fruiting_base_pos is not None:
            kw["base_pos"] = self._reset_fruiting_base_pos
        if params is not None:
            kw["params"] = params
        if self._reset_robot_base_pos is not None:
            kw["robot_base_pos"] = self._reset_robot_base_pos
        if self._cfg.fix_to_apple:
            kw["robot_base_from_proxy"] = False
        else:
            kw["robot_base_from_proxy"] = True
        return kw

    # --- Sensor helpers ---

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

    def _measure_fixed_joints(self, sub_dt: float):
        import apple_pick_sim.fruiting_system as fs

        assert self._scene is not None
        cable = self._scene.cable
        return fs.measure_fruiting_forces(
            cable,
            cable.state_0.body_q,
            cable.state_1.body_q,
            dt=float(sub_dt),
        )["fixed_joints"]

    @staticmethod
    def _placeholder_junction_names(n_woody: int) -> list[str]:
        return [f"joint_{i}" for i in range(int(n_woody))]

    @staticmethod
    def _woody_pos_obs_space(junction_names: list[str]) -> spaces.Dict:
        return spaces.Dict(
            {
                name: spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
                for name in junction_names
            }
        )

    def _woody_pos_dict(
        self, flat_start: np.ndarray, flat_end: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Split flat ``(N*3,)`` anchor arrays into per-junction ``(3,)`` dicts."""
        names = self.junction_names
        start = np.asarray(flat_start, dtype=np.float32).reshape(-1, 3)
        end = np.asarray(flat_end, dtype=np.float32).reshape(-1, 3)
        return (
            {name: start[i].copy() for i, name in enumerate(names)},
            {name: end[i].copy() for i, name in enumerate(names)},
        )

    def _woody_start_end_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame fixed-joint anchors for each inter-segment FIXED joint.

        Returns flat ``(N*3,)`` arrays: parent-side anchor xyz, then child-side anchor xyz,
        in the same order as ``fruiting_fixed_joints`` / ``woody_part_force``.
        """
        import apple_pick_sim.fruiting_system as fs

        assert self._scene is not None
        cable = self._scene.cable
        return fs.fixed_joint_anchors_world(
            cable.model,
            cable.state_0.body_q,
            cable.fruiting_fixed_joints,
        )

    def _woody_part_forces(self, sub_dt: float) -> np.ndarray:
        measured = self._measure_fixed_joints(sub_dt)
        if not measured:
            return np.zeros((0,), dtype=np.float32)
        parts: list[np.ndarray] = []
        for rec in measured:
            parts.append(np.asarray(rec.force_world, dtype=np.float32))
            parts.append(np.asarray(rec.torque_at_child_com_world, dtype=np.float32))
        return np.concatenate(parts, dtype=np.float32)

    @property
    def junction_names(self) -> list[str]:
        """Ordered junction labels matching ``woody_part_force`` (post-``reset()``).

        Each name is the ``fruiting_fixed_joints`` label with the ``joint_`` prefix
        removed (e.g. ``"stem_apple"`` for the stem→apple junction). Index ``i``
        in this list corresponds to ``woody_part_force[i*6:(i+1)*6]``.
        """
        if self._scene is None:
            raise RuntimeError("Call reset() before accessing junction_names.")
        return [
            label.removeprefix("joint_")
            for _, label in self._scene.cable.fruiting_fixed_joints
        ]

    def junction_forces_dict(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Map junction labels to 6-vectors ``[Fx, Fy, Fz, τx, τy, τz]`` from ``obs``.

        Keys match :attr:`junction_names`. Values are ``float32`` arrays in world
        frame (force on the child body, torque about child COM), same convention
        as ``woody_part_force`` / ``info["fruiting_link_forces"]``.
        """
        names = self.junction_names
        flat = np.asarray(obs["woody_part_force"], dtype=np.float32)
        return {
            name: flat[i * 6 : (i + 1) * 6]
            for i, name in enumerate(names)
        }

    def _apple_pos(self) -> np.ndarray:
        assert self._scene is not None
        apple = self._scene.cable.apple_body
        if apple is None:
            return np.zeros((3,), dtype=np.float32)
        bq = self._scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        return np.asarray(bq[int(apple), :3], dtype=np.float32)

    def _apple_quat(self) -> np.ndarray:
        assert self._scene is not None
        apple = self._scene.cable.apple_body
        if apple is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        bq = self._scene.cable.state_0.body_q.numpy().reshape(-1, 7)
        return np.asarray(bq[int(apple), 3:7], dtype=np.float32)

    def _rod_radii(self) -> dict[str, float]:
        assert self._scene is not None
        params = self._scene.cable.params
        out: dict[str, float] = {}
        for name in ("primary", "secondary", "spur", "stem"):
            rod = getattr(params, name, None)
            if rod is not None:
                out[name] = float(rod.radius)
        return out

    def _tcp_velocity(self) -> np.ndarray:
        assert self._scene is not None
        tcp = int(self._scene.tcp_body_index)
        bqd = self._scene.robot_state_0.body_qd.numpy().reshape(-1, 6)
        return np.asarray(bqd[tcp], dtype=np.float32)

    def _make_info(self) -> dict[str, Any]:
        import apple_pick_sim.fruiting_system as fs

        assert self._scene is not None
        _, _, sub_dt = self._timing_constants()
        return {
            "obs_schema": "v3",
            "step_count": int(self._step_count),
            "n_woody_parts": int(self._n_woody_parts),
            "params_fingerprint": fs.params_fingerprint(self._scene.cable.params),
            "end_effector_wrench": self._end_effector_wrench(),
            "fruiting_link_forces": self._fruiting_link_forces(sub_dt),
        }

    def _set_n_woody_parts(self) -> None:
        assert self._scene is not None
        n = len(self._scene.cable.fruiting_fixed_joints)
        if n > self._cfg.max_woody_parts:
            raise ValueError(
                f"Scene has {n} woody fixed joints but max_woody_parts={self._cfg.max_woody_parts}"
            )
        self._n_woody_parts = n

    def _make_gripper_proxy_config(self, *, fix_to_apple: bool):
        """Return gripper proxy config for scene build (subclasses may extend)."""
        import apple_pick_sim.fruiting_system as fs
        from apple_pick_sim.robot import fr3_robot

        return fs.GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            fix_to_apple=fix_to_apple,
        )

    def _weld_direction_before_fix_to_apple_build(self, probe_scene: Any) -> tuple[float, float, float] | None:
        """Return explicit weld direction from a settled or vbd-only probe scene."""
        del probe_scene
        return None

    def _build_vbd_only_probe(
        self,
        ranges: dict,
        scene_seed: int,
        *,
        injected_params: Any | None = None,
    ) -> Any:
        """Lightweight cable-only scene for apple pose before welded rebuild."""
        import apple_pick_sim.coupled_fruiting as cf

        return cf.build_coupled_fruiting_fr3(
            ranges,
            scene_seed,
            vbd_only=True,
            **self._coupled_build_kwargs(params=injected_params),
            gripper_proxy=self._make_gripper_proxy_config(fix_to_apple=False),
        )

    def _build_scene(
        self,
        ranges: dict,
        scene_seed: int,
        *,
        injected_params: Any | None = None,
    ) -> None:
        import apple_pick_sim.coupled_fruiting as cf

        build_kw = self._coupled_build_kwargs(params=injected_params)

        if self._cfg.fix_to_apple and self._cfg.fix_to_apple_warmup_substeps > 0:
            settled = self._build_vbd_only_probe(
                ranges, scene_seed, injected_params=injected_params
            )
            _frame_dt, _substeps_per_step, sub_dt = self._timing_constants()
            cf.settle_vbd_substeps(
                settled, substeps=self._cfg.fix_to_apple_warmup_substeps, dt=sub_dt
            )
            self._pending_weld_direction = self._weld_direction_before_fix_to_apple_build(
                settled
            )
            self._scene = cf.build_coupled_fruiting_fr3(
                ranges,
                scene_seed,
                **build_kw,
                skip_ik_bootstrap=True,
                gripper_proxy=self._make_gripper_proxy_config(fix_to_apple=True),
            )
            self._pending_weld_direction = None
            self._pending_weld_reference_pos = None
            self._pending_weld_reference_quat = None
            cf.seed_fix_to_apple_from_settled(
                welded_scene=self._scene,
                settled_scene=settled,
                quiet_apple_proxy=True,
            )
        else:
            if self._cfg.fix_to_apple:
                probe = self._build_vbd_only_probe(
                    ranges, scene_seed, injected_params=injected_params
                )
                self._pending_weld_direction = self._weld_direction_before_fix_to_apple_build(
                    probe
                )
            self._scene = cf.build_coupled_fruiting_fr3(
                ranges,
                scene_seed,
                **build_kw,
                gripper_proxy=self._make_gripper_proxy_config(
                    fix_to_apple=self._cfg.fix_to_apple
                ),
            )
            self._pending_weld_direction = None
            self._pending_weld_reference_pos = None
            self._pending_weld_reference_quat = None

        self._finalize_scene()

    def _create_controller(self):
        from apple_pick_sim.robot import fr3_robot

        return fr3_robot.Fr3EEDirectJointController(
            self._scene.robot_model, self._scene.tcp_body_index
        )

    def _finalize_scene(self) -> None:
        self._scene.robot_kinematic_mode = True
        self._controller = self._create_controller()
        self._controller.sync_target_from_state(self._scene.robot_state_0)

    def _update_teleop(self, frame_dt: float, vel) -> None:
        self._scene.update_fr3_ee_teleop_direct(frame_dt, self._controller, velocity=vel)

    # --- Gymnasium API ---

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        import apple_pick_sim.fruiting_system as fs

        scene_seed = int(0 if seed is None else seed)
        ranges_path = options.get("ranges_path")
        if ranges_path is not None:
            ranges = fs.load_ranges(Path(ranges_path))
        else:
            ranges = fs.load_ranges(self._fixture_ranges_path())
        fixture_args = fs.parse_fixture_args(ranges)
        self._reset_fruiting_base_pos = self._coerce_optional_xyz(
            options.get("fruiting_base_pos", fixture_args.fruiting_base_pos),
            field="fruiting_base_pos",
        )
        self._reset_robot_base_pos = self._coerce_optional_xyz(
            options.get("robot_base_pos", fixture_args.robot_base_pos),
            field="robot_base_pos",
        )

        injected_params = options.get("params")
        self._build_scene(ranges, scene_seed, injected_params=injected_params)

        self._step_count = 0
        self._set_n_woody_parts()
        self._setup_observation_space()
        return self._make_obs(), self._make_info()

    def step(self, action):
        if self._scene is None or self._controller is None:
            raise RuntimeError("Environment must be reset() before step().")

        frame_dt, substeps_per_step, sub_dt = self._timing_constants()
        vel = self._action_to_command(action)

        self._update_teleop(frame_dt, vel)

        for _ in range(substeps_per_step):
            self._scene.coupled_substep(sub_dt)

        self._step_count += 1

        obs = self._make_obs()
        info = self._make_info()
        reward = self.compute_reward(obs, info)
        terminated = self.compute_terminated(obs, info)
        truncated = self._step_count >= self._cfg.max_episode_steps
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._scene = None
        self._controller = None
