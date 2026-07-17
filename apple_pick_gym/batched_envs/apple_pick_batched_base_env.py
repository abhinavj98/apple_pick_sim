"""Abstract SKRL-native batched Gymnasium base over BatchedHeterogeneousCoupledSim."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:  # pragma: no cover
    raise ImportError(
        "apple_pick_gym requires gymnasium. Install from repo root (uv sync --extra gym)."
    ) from e

import torch

from apple_pick_gym.batched_envs.obs_torch import obs_dict_from_bufs

if TYPE_CHECKING:
    from apple_pick_sim.fruiting_system import GripperProxyConfig


class ApplePickBatchedBaseEnv(gym.Env, ABC):
    """Batched GPU coupled fruiting env for SKRL (``num_envs`` + device + torch batch dim)."""

    metadata = {"render_modes": [None], "render_fps": 60}

    def __init__(
        self,
        *,
        num_envs: int = 1,
        render_mode: str | None = None,
        max_episode_steps: int = 240,
        max_woody_parts: int = 64,
        device: str | None = None,
        sim_config: Any | None = None,
        ranges_path: Path | str | None = None,
        topology_seed: int = 42,
        use_settle_cache: bool = False,
        per_env_params: Sequence[Any] | None = None,
        per_env_grippers: Sequence[GripperProxyConfig] | None = None,
    ) -> None:
        if render_mode not in (None, "none"):
            raise ValueError("Only headless operation is supported (render_mode=None).")
        if int(num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        from apple_pick_sim.coupled_fruiting import (
            BatchedHeterogeneousCoupledSim,
            BatchedHeterogeneousCoupledSimConfig,
        )
        from apple_pick_sim.fruiting_system import (
            default_ranges_fixture_path,
            load_ranges,
            sample_heterogeneous_params_list,
        )
        from apple_pick_sim.sim_device import resolve_sim_device

        self._max_episode_steps = int(max_episode_steps)
        self._max_woody_parts = int(max_woody_parts)
        self._step_count = 0
        self._ranges_path = Path(ranges_path) if ranges_path is not None else default_ranges_fixture_path()
        self._topology_seed = int(topology_seed)

        cfg = sim_config or BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(num_envs))
        if device is not None:
            import dataclasses

            cfg = dataclasses.replace(
                cfg, runtime=dataclasses.replace(cfg.runtime, device=device)
            )
        cfg.validate()

        ranges = load_ranges(self._ranges_path)
        if per_env_params is not None:
            params_list = list(per_env_params)
            if len(params_list) != int(num_envs):
                raise ValueError(
                    f"per_env_params length ({len(params_list)}) must match "
                    f"num_envs ({num_envs})"
                )
        else:
            params_list = sample_heterogeneous_params_list(
                ranges,
                topology_seed=self._topology_seed,
                num_envs=int(num_envs),
            )

        self._sim = BatchedHeterogeneousCoupledSim(
            cfg,
            params_list,
            ranges,
            per_env_grippers=per_env_grippers,
            use_settle_cache=use_settle_cache,
        )
        self._sim.capture_episode_snapshot()

        resolved = resolve_sim_device(cfg.runtime.device)
        self.num_envs = int(num_envs)
        self.device = torch.device(resolved)

        self._junction_names = self._compute_junction_names()
        if len(self._junction_names) > self._max_woody_parts:
            raise ValueError(
                f"Scene has {len(self._junction_names)} woody joints but "
                f"max_woody_parts={self._max_woody_parts}"
            )

        lin = float(cfg.controller.linear_speed)
        ang = float(cfg.controller.angular_speed)
        self.action_space = spaces.Box(
            low=np.array([-lin, -lin, -lin, -ang, -ang, -ang], dtype=np.float32),
            high=np.array([lin, lin, lin, ang, ang, ang], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = self._observation_space_for(self._junction_names)

    @staticmethod
    def _compute_junction_names_from_cable(cable: Any) -> list[str]:
        from apple_pick_sim.digital_twin.record import fruiting_tree_fixed_joints

        return [
            label.removeprefix("joint_")
            for _, label in fruiting_tree_fixed_joints(cable)
        ]

    def _compute_junction_names(self) -> list[str]:
        return self._compute_junction_names_from_cable(self._sim.scene.cable)

    @property
    def junction_names(self) -> list[str]:
        return list(self._junction_names)

    @staticmethod
    def _woody_part_info_space(junction_names: list[str]) -> spaces.Dict:
        return spaces.Dict(
            {
                name: spaces.Dict(
                    {
                        "anchors_pos": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                        "anchor_force": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                        ),
                    }
                )
                for name in junction_names
            }
        )

    @classmethod
    def _observation_space_for(cls, junction_names: list[str]) -> spaces.Dict:
        return spaces.Dict(
            {
                "woody_part_info": cls._woody_part_info_space(junction_names),
                "apple_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "tcp_force": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "tcp_velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
                "ft_wrist": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )

    def _gather_obs(self) -> dict[str, Any]:
        self._sim.gather_obs()
        bufs = self._sim.obs_bufs
        if bufs is None:
            raise RuntimeError("sim observation buffers not allocated")
        return obs_dict_from_bufs(bufs, self._junction_names, self.device)

    def _make_info(self) -> dict[str, Any]:
        import apple_pick_sim.fruiting_system as fs

        return {
            "obs_schema": "v3",
            "obs_layout": "batched_vic",
            "step_count": int(self._step_count),
            "n_woody_parts": len(self._junction_names),
            "params_fingerprint": fs.params_fingerprint(self._sim.scene.cable.params),
        }

    def _actions_tensor(self, action: Any) -> torch.Tensor:
        cfg = self._sim.config
        return cfg.controller.validate_actions(
            action,
            num_envs=self.num_envs,
            device=str(self.device),
            robot_step_mode=cfg.robot.step_mode,
        )

    @abstractmethod
    def compute_reward(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> torch.Tensor:
        """Return shape ``(num_envs, 1)``."""

    @abstractmethod
    def compute_terminated(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> torch.Tensor:
        """Return shape ``(num_envs, 1)`` bool."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self._sim.restore_episode_snapshot()
        self._step_count = 0
        obs = self._gather_obs()
        info = self._make_info()
        return obs, info

    def step(self, action):
        actions = self._actions_tensor(action)
        self._sim.step(actions)
        self._step_count += 1

        obs = self._gather_obs()
        info = self._make_info()
        reward = self.compute_reward(obs, info)
        terminated = self.compute_terminated(obs, info)
        truncated = torch.full(
            (self.num_envs, 1),
            float(self._step_count >= self._max_episode_steps),
            dtype=torch.bool,
            device=self.device,
        )
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._sim = None  # type: ignore[assignment]
