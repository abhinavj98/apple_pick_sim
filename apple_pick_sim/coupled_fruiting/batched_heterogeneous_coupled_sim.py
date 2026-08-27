"""Batched heterogeneous coupled fruiting runtime (V.3.1 step B)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from apple_pick_sim.batched_obs import gather_batched_obs, make_batched_obs_buffers
from apple_pick_sim.coupled_fruiting.episode_state_snapshot import EpisodeStateSnapshot
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import (
    BatchedHeterogeneousBuildResult,
    build_batched_heterogeneous_scene,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout
from apple_pick_sim.coupled_fruiting.scene import CoupledFruitingScene
from apple_pick_sim.coupled_fruiting.settled_checkpoint import (
    SettledCheckpoint,
    settle_cache_path_for,
)
from apple_pick_sim.fruiting_system import FruitingSystemParams, GripperProxyConfig
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot.controllers.batched_action_twists import clip_action_tensor


class BatchedHeterogeneousCoupledSim:
    """Runtime parent for batched heterogeneous coupled fruiting."""

    def __init__(
        self,
        config: BatchedHeterogeneousCoupledSimConfig,
        per_env_params: Sequence[FruitingSystemParams],
        ranges: dict,
        *,
        per_env_grippers: Sequence[GripperProxyConfig] | None = None,
        viewer: Any | None = None,
        use_settle_cache: bool = False,
        force_settle: bool = False,
        settle_cache_dir: Path | str | None = None,
    ) -> None:
        config.validate()
        self._config = config
        self._per_env_params = tuple(per_env_params)
        self._per_env_grippers = (
            tuple(per_env_grippers) if per_env_grippers is not None else None
        )
        self._ranges = ranges
        self._device = config.resolve_device()
        self._sim_time = 0.0
        if self._per_env_grippers is not None and use_settle_cache:
            raise ValueError("per_env_grippers require use_settle_cache=False")

        cache_path = settle_cache_path_for(
            config,
            ranges,
            self._per_env_params,
            cache_dir=settle_cache_dir,
        )
        self._settle_cache_path = cache_path

        loaded_checkpoint: SettledCheckpoint | None = None
        build_viewer = viewer
        if (
            use_settle_cache
            and not force_settle
            and cache_path is not None
            and cache_path.is_file()
        ):
            loaded_checkpoint = SettledCheckpoint.load(cache_path)
            loaded_checkpoint.validate_against(
                config=config,
                ranges=ranges,
                per_env_params=self._per_env_params,
            )
            build_viewer = None

        self._settled_checkpoint = loaded_checkpoint
        build_result = build_batched_heterogeneous_scene(
            config,
            self._per_env_params,
            ranges,
            per_env_grippers=self._per_env_grippers,
            viewer=build_viewer,
            settled_checkpoint=loaded_checkpoint,
        )
        self._build_result = build_result
        self._scene = build_result.scene
        self._layout = self._scene.layout
        if self._layout is None:
            raise RuntimeError("batched scene missing layout")

        if (
            loaded_checkpoint is None
            and cache_path is not None
            and build_result.settled_body_q is not None
        ):
            checkpoint = SettledCheckpoint.from_build_context(
                body_q=build_result.settled_body_q,
                config=config,
                ranges=ranges,
                per_env_params=self._per_env_params,
            )
            checkpoint.save(cache_path)
            self._settled_checkpoint = checkpoint

        self._ee_ctrl: Any | None = None
        if self._scene.robot_model is not None and config.robot.step_mode != "vbd_only":
            self._ee_ctrl = self._configure_fr3_controller(config.controller.mode)

        self._action_buffer = None
        if config.controller.allocate_action_buffer:
            import torch

            n, d = config.controller.expected_action_shape(config.runtime.num_envs)
            self._action_buffer = torch.zeros(
                n, d, dtype=torch.float32, device=self._device
            )

        self._obs_bufs = None
        obs_cfg = config.obs
        if obs_cfg is not None and obs_cfg.allocate_buffers:
            self._obs_bufs = make_batched_obs_buffers(
                self._layout,
                self._scene.cable,
                self._device,
            )

        self._episode_snapshot: EpisodeStateSnapshot | None = None

    @property
    def obs_bufs(self):
        """Allocated when ``config.obs.allocate_buffers`` is True."""
        return self._obs_bufs

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BatchedHeterogeneousCoupledSim:
        """Alias for :meth:`__init__`."""
        return cls(*args, **kwargs)

    @property
    def config(self) -> BatchedHeterogeneousCoupledSimConfig:
        return self._config

    @property
    def scene(self) -> CoupledFruitingScene:
        return self._scene

    @property
    def layout(self) -> BatchedEnvLayout | None:
        return self._layout

    @property
    def per_env_params(self) -> tuple[FruitingSystemParams, ...]:
        return self._per_env_params

    @property
    def ranges(self) -> dict:
        return self._ranges

    @property
    def build_result(self) -> BatchedHeterogeneousBuildResult:
        return self._build_result

    @property
    def settled_checkpoint(self) -> SettledCheckpoint | None:
        return self._settled_checkpoint

    @property
    def settle_cache_path(self) -> Path | None:
        return self._settle_cache_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def frame_dt(self) -> float:
        return self._config.runtime.frame_dt

    @property
    def sub_dt(self) -> float:
        return float(self._config.runtime.sub_dt)

    @property
    def num_envs(self) -> int:
        return int(self._config.runtime.num_envs)

    @property
    def sim_time(self) -> float:
        return self._sim_time

    def _configure_fr3_controller(self, mode: str):
        ik_kw = fr3_robot.batched_ik_teleop_kwargs(self._scene)
        if not ik_kw:
            raise RuntimeError("batched FR3 scene missing template IK layout")

        velocity_for_world = self._velocity_for_world

        if mode == "ee":
            self._scene.robot_kinematic_mode = False
            ctrl = fr3_robot.Fr3BatchedEEVelocityController(
                self._scene.robot_model,
                linear_speed=self._config.controller.linear_speed,
                angular_speed=self._config.controller.angular_speed,
                ik_iterations=self._config.controller.ik_iterations,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
        elif mode in ("vic", "vic_pose"):
            ctrl = self._configure_fr3_vic(ik_kw, velocity_for_world)
        else:
            self._scene.robot_kinematic_mode = True
            ctrl = fr3_robot.Fr3BatchedEEDirectJointController(
                self._scene.robot_model,
                linear_speed=self._config.controller.linear_speed,
                angular_speed=self._config.controller.angular_speed,
                ik_iterations=self._config.controller.ik_iterations,
                velocity_for_world=velocity_for_world,
                **ik_kw,
            )
        ctrl.sync_target_from_state(self._scene.robot_state_0)
        return ctrl

    def _configure_fr3_vic(self, ik_kw: dict, velocity_for_world):
        from apple_pick_sim.coupled_fruiting.vic_joint_torques import _require_torch

        _require_torch()
        self._scene.robot_kinematic_mode = False
        fr3_robot.init_mujoco_actuator_targets_from_model(
            self._scene.robot_model, self._scene.robot_control
        )
        self._scene.vic_use_joint_torques = True
        vic = fr3_robot.Fr3BatchedEEImpedanceController(
            self._scene.robot_model,
            linear_speed=self._config.controller.linear_speed,
            angular_speed=self._config.controller.angular_speed,
            velocity_for_world=velocity_for_world,
            **ik_kw,
        )
        self._scene.vic_controller = vic
        g = self._config.controller.vic_gains
        self._scene.vic_gains = fr3_robot.ImpedanceGains(
            linear_k=float(g.linear_k),
            linear_d=float(g.linear_d),
            angular_k=float(g.angular_k),
            angular_d=float(g.angular_d),
        )
        fr3_robot.configure_vic_joint_torques_arm_batched(
            self._scene.robot_model,
            self._scene.robot_state_0,
            self._scene.robot_control,
            self._scene.mj_solver,
            scene=self._scene,
            layout=self._scene.layout,
        )
        self._scene.vic_joint_torques_configured = True
        vic.sync_target_from_state(self._scene.robot_state_0)
        vic.stage_targets_to_scene(self._scene)
        self._scene.vic_target_twist = fr3_robot.EEVelocity()
        return vic

    def _velocity_for_world(self, world: int) -> fr3_robot.EEVelocity:
        import torch

        if self._action_buffer is None:
            return fr3_robot.EEVelocity()
        row = self._action_buffer[world]
        linear = tuple(float(v) for v in row[:3].detach().cpu().tolist())
        angular = tuple(float(v) for v in row[3:6].detach().cpu().tolist())
        return fr3_robot.EEVelocity(linear=linear, angular=angular)

    def _clip_actions(self, actions):
        if self._config.controller.mode == "vic_pose":
            return actions
        return clip_action_tensor(
            actions,
            linear_speed=float(self._config.controller.linear_speed),
            angular_speed=float(self._config.controller.angular_speed),
        )

    def _run_fr3_teleop_from_actions(self) -> None:
        cfg = self._config
        assert self._ee_ctrl is not None
        if cfg.controller.mode == "vic_pose":
            assert isinstance(self._ee_ctrl, fr3_robot.Fr3BatchedEEImpedanceController)
            velocity = self._ee_ctrl.run_coupled_teleop_frame_from_pose_actions(
                self._scene.robot_state_0,
                self._scene.robot_control,
                self._scene.mj_solver,
                self.frame_dt,
                self._action_buffer,
            )
            self._ee_ctrl.stage_targets_to_scene(self._scene)
            self._ee_ctrl.stage_pose_gains_to_scene(self._scene)
            self._scene.vic_target_twist = velocity
            return
        velocity = self._ee_ctrl.run_coupled_teleop_frame_from_actions(
            self._scene.robot_state_0,
            self._scene.robot_control,
            self._scene.mj_solver,
            self.frame_dt,
            self._action_buffer,
        )
        if getattr(self._scene, "vic_controller", None) is not None:
            if isinstance(self._ee_ctrl, fr3_robot.Fr3BatchedEEImpedanceController):
                self._ee_ctrl.stage_targets_to_scene(self._scene)
                self._scene.vic_target_twist = velocity

    def step(self, actions=None) -> None:
        """Advance one control frame."""
        cfg = self._config
        if cfg.robot.step_mode == "vbd_only":
            if actions is not None:
                raise ValueError(
                    "robot_step_mode='vbd_only' does not accept actions; pass actions=None"
                )
            for _ in range(cfg.runtime.substeps_per_step):
                self._scene.vbd_substep(self.sub_dt)
            self._sim_time += self.frame_dt
            return

        import torch

        if actions is None:
            n, d = cfg.controller.expected_action_shape(cfg.runtime.num_envs)
            actions = torch.zeros(n, d, dtype=torch.float32, device=self._device)
        else:
            actions = cfg.controller.validate_actions(
                actions,
                num_envs=cfg.runtime.num_envs,
                device=self._device,
                robot_step_mode=cfg.robot.step_mode,
            )

        actions = self._clip_actions(actions)
        if self._action_buffer is not None:
            self._action_buffer.copy_(actions)

        if self._ee_ctrl is not None:
            if self._action_buffer is not None:
                self._run_fr3_teleop_from_actions()
            else:
                velocity = self._velocity_for_world(0)
                if cfg.controller.mode == "direct":
                    self._scene.update_fr3_ee_teleop_direct(
                        self.frame_dt,
                        self._ee_ctrl,
                        velocity=velocity,
                    )
                else:
                    self._scene.update_fr3_ee_teleop(
                        self.frame_dt,
                        self._ee_ctrl,
                        velocity=velocity,
                    )

        for _ in range(cfg.runtime.substeps_per_step):
            self._scene.coupled_substep(self.sub_dt)

        self._sim_time += self.frame_dt

    def gather_obs(self) -> dict[str, Any]:
        """Fill obs buffers and return a snapshot dict."""
        if self._obs_bufs is None:
            raise RuntimeError(
                "observation buffers not allocated; set config.obs.allocate_buffers=True"
            )
        obs_cfg = self._config.obs
        include_robot = True if obs_cfg is None else obs_cfg.include_robot
        include_forces = True if obs_cfg is None else obs_cfg.include_forces
        gather_batched_obs(
            self._obs_bufs,
            self._scene,
            self.sub_dt,
            include_robot=include_robot,
            include_forces=include_forces,
        )
        bufs = self._obs_bufs
        out: dict[str, Any] = {
            "apple_pos": bufs.apple_pos,
            "apple_pose": bufs.apple_pose,
            "proxy_pos": bufs.proxy_pos,
        }
        if include_robot:
            out.update(
                {
                    "tcp_pose": bufs.tcp_pose,
                    "tcp_velocity": bufs.tcp_velocity,
                    "joint_q": bufs.joint_q,
                    "joint_qd": bufs.joint_qd,
                }
            )
        if include_forces:
            out.update(
                {
                    "tcp_force": bufs.tcp_force,
                    "tcp_coupling_force": bufs.tcp_coupling_force,
                    "woody_parent_pos": bufs.woody_parent_pos,
                    "woody_child_pos": bufs.woody_child_pos,
                    "woody_force": bufs.woody_force,
                    "woody_torque": bufs.woody_torque,
                }
            )
        return out

    @property
    def episode_snapshot(self) -> EpisodeStateSnapshot | None:
        return self._episode_snapshot

    def capture_episode_snapshot(self) -> EpisodeStateSnapshot:
        """Capture post-weld episode baseline for cheap ``restore_episode_snapshot()``."""
        self._episode_snapshot = EpisodeStateSnapshot.capture(self)
        return self._episode_snapshot

    def restore_episode_snapshot(self) -> None:
        """Restore physics to the last captured episode baseline."""
        if self._episode_snapshot is None:
            raise RuntimeError("no episode snapshot; call capture_episode_snapshot() first")
        self._episode_snapshot.restore(self)

