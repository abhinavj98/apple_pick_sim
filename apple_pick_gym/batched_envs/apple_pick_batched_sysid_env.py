"""Batched sys-ID gym env for parallel quasi-static data collection (V.4.2)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import warp as wp

from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv
from apple_pick_gym.batched_envs.batched_sysid_world_info import per_world_sysid_reset_info
from apple_pick_gym.batched_envs.obs_torch import sysid_numpy_obs_from_batched
from apple_pick_sim.system_id.excitation_state import ExcitationContext

if TYPE_CHECKING:
    from apple_pick_sim.fruiting_system import GripperProxyConfig

_EXCITATION_TYPE_TO_INT: dict[str, int] = {
    "quasi_static": 0,
    "translational_chirp": 1,
    "torsional": 2,
}


class ApplePickBatchedSysIdEnv(ApplePickBatchedBaseEnv):
    """Batched fix-to-apple VIC env with sys-ID observation fields and metadata."""

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
        control_hz: float | None = None,
    ) -> None:
        import dataclasses

        from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
            BatchedHeterogeneousCoupledSimConfig,
            ControllerConfig,
            ObsConfig,
            RobotConfig,
            RuntimeConfig,
        )

        n = int(num_envs)
        cfg = sim_config or BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=n)
        # Prefer explicit control_hz; otherwise keep sim_config.runtime.control_hz
        # (do not silently overwrite with a hardcoded 60 default).
        hz = float(cfg.runtime.control_hz if control_hz is None else control_hz)
        cfg = dataclasses.replace(
            cfg,
            runtime=dataclasses.replace(cfg.runtime, control_hz=hz),
            controller=dataclasses.replace(
                cfg.controller,
                linear_speed=1.0,
                angular_speed=1.0,
            ),
            obs=ObsConfig(allocate_buffers=True) if cfg.obs is None else cfg.obs,
        )
        if not cfg.robot.fix_to_apple:
            cfg = dataclasses.replace(
                cfg,
                robot=dataclasses.replace(cfg.robot, fix_to_apple=True, force_batched_layout=True),
            )

        self._control_hz = hz
        self._last_obs: dict[str, Any] | None = None
        self._per_env_reset_info: list[dict[str, Any]] = []

        super().__init__(
            num_envs=n,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_woody_parts=max_woody_parts,
            device=device,
            sim_config=cfg,
            ranges_path=ranges_path,
            topology_seed=topology_seed,
            use_settle_cache=use_settle_cache,
            per_env_params=per_env_params,
            per_env_grippers=per_env_grippers,
        )

        dev = self.device
        self._excitation_type = torch.zeros(n, dtype=torch.long, device=dev)
        self._excitation_f_inst = torch.zeros(n, dtype=torch.float32, device=dev)
        default_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=dev)
        self._excitation_direction = default_dir.unsqueeze(0).expand(n, -1).clone()

    def set_excitation_context(self, env_idx: int, ctx: ExcitationContext) -> None:
        type_int = _EXCITATION_TYPE_TO_INT.get(ctx.type)
        if type_int is None:
            raise ValueError(f"Unknown excitation type {ctx.type!r}")
        i = int(env_idx)
        self._excitation_type[i] = int(type_int)
        self._excitation_f_inst[i] = float(ctx.f_inst)
        self._excitation_direction[i] = torch.as_tensor(
            ctx.direction, dtype=torch.float32, device=self.device
        )

    def set_excitation_contexts(self, contexts: Sequence[ExcitationContext]) -> None:
        if len(contexts) != self.num_envs:
            raise ValueError(
                f"contexts length ({len(contexts)}) must match num_envs ({self.num_envs})"
            )
        for i, ctx in enumerate(contexts):
            self.set_excitation_context(i, ctx)

    def sysid_numpy_obs(self, env_idx: int) -> dict[str, Any]:
        if self._last_obs is None:
            raise RuntimeError("call reset() or step() before sysid_numpy_obs()")
        return sysid_numpy_obs_from_batched(
            self._last_obs,
            self.junction_names,
            int(env_idx),
        )

    def pre_weld_sysid_obs(self, env_idx: int) -> dict[str, Any] | None:
        """Settled tree observation captured before grasp weld (for geometry rebuild)."""
        build_result = getattr(self._sim, "build_result", None)
        if build_result is None:
            return None
        rows = getattr(build_result, "pre_weld_tree_obs", None)
        if rows is None:
            return None
        i = int(env_idx)
        if i < 0 or i >= len(rows):
            return None
        return dict(rows[i])

    def per_env_reset_info(self, env_idx: int) -> dict[str, Any]:
        return dict(self._per_env_reset_info[int(env_idx)])

    def compute_reward(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> torch.Tensor:
        del obs, info
        return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

    def compute_terminated(
        self, obs: dict[str, Any], info: dict[str, Any]
    ) -> torch.Tensor:
        del obs, info
        return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)

    def _apple_quat_tensor(self) -> torch.Tensor:
        layout = self._sim.layout
        if layout is None:
            raise RuntimeError("batched scene missing layout")
        cable = self._sim.scene.cable
        bq = cable.state_0.body_q.numpy().reshape(-1, 7)
        quats = np.stack(
            [bq[int(layout.apple_body_indices[w]), 3:7] for w in range(self.num_envs)],
            axis=0,
        )
        return torch.as_tensor(quats, dtype=torch.float32, device=self.device)

    def _gather_obs(self) -> dict[str, Any]:
        obs = super()._gather_obs()
        bufs = self._sim.obs_bufs
        if bufs is None:
            raise RuntimeError("sim observation buffers not allocated")

        tcp_pose = wp.to_torch(bufs.tcp_pose).to(device=self.device, dtype=torch.float32)
        obs["tcp_pos"] = tcp_pose[:, :3]
        obs["tcp_quat"] = tcp_pose[:, 3:7]
        obs["apple_quat"] = self._apple_quat_tensor()
        obs["robot_joint_q"] = wp.to_torch(bufs.joint_q).to(
            device=self.device, dtype=torch.float32
        )
        obs["raw_ft_wrist"] = obs["ft_wrist"].clone()
        obs["excitation_type"] = self._excitation_type
        obs["excitation_f_inst"] = self._excitation_f_inst
        obs["excitation_direction"] = self._excitation_direction
        self._last_obs = obs
        return obs

    def _make_info(self) -> dict[str, Any]:
        info = super()._make_info()
        info["obs_layout"] = "batched_sysid"
        info["control_hz"] = float(self._control_hz)
        info["per_env"] = list(self._per_env_reset_info)
        if self._per_env_reset_info:
            info["weld_direction"] = [row["weld_direction"] for row in self._per_env_reset_info]
            info["robot_base_pos"] = [row["robot_base_pos"] for row in self._per_env_reset_info]
            info["params_fingerprint"] = [
                row["params_fingerprint"] for row in self._per_env_reset_info
            ]
        return info

    def _refresh_per_env_reset_info(self) -> None:
        scene = self._sim.scene
        layout = self._sim.layout
        if layout is None:
            raise RuntimeError("batched scene missing layout")
        params_list = self._sim._per_env_params
        self._per_env_reset_info = [
            per_world_sysid_reset_info(scene, layout, w, params_list[w])
            for w in range(self.num_envs)
        ]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._refresh_per_env_reset_info()
        info = self._make_info()
        return obs, info
