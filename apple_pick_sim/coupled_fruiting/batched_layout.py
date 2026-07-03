"""Per-world index layout for homogeneous batched coupled scenes."""

from __future__ import annotations

import dataclasses

import numpy as np
import warp as wp

import newton

from apple_pick_sim.fruiting_system import CoupledCableScene


@dataclasses.dataclass(frozen=True)
class BatchedEnvLayout:
    """Maps template (world-0) body/joint indices to global batched model indices."""

    num_envs: int
    bodies_per_world: int
    robot_bodies_per_world: int
    joints_per_world: int
    joint_coord_count_per_world: int
    joint_dof_count_per_world: int
    template_tcp_body: int
    template_proxy_body: int
    template_apple_body: int | None
    tcp_body_indices: tuple[int, ...]
    proxy_body_indices: tuple[int, ...]
    apple_body_indices: tuple[int, ...]
    env_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def world_origin(self, world: int) -> tuple[float, float, float]:
        """World-frame translation of replicated env ``world`` (template is env 0)."""
        offsets = newton.utils.compute_world_offsets(
            self.num_envs, self.env_spacing, up_axis=newton.Axis.Z
        )
        o = np.asarray(offsets[int(world)], dtype=np.float64).reshape(3)
        return float(o[0]), float(o[1]), float(o[2])

    @classmethod
    def from_template_scene(
        cls,
        template_cable: CoupledCableScene,
        cable_model: newton.Model,
        robot_model: newton.Model,
        *,
        template_tcp_body: int,
        env_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> BatchedEnvLayout:
        """Build layout from a single-world template cable scene and replicated models."""
        num_envs = int(cable_model.world_count)
        if num_envs < 1:
            raise ValueError("batched layout requires world_count >= 1")
        if int(robot_model.world_count) != num_envs:
            raise ValueError(
                f"cable world_count {num_envs} != robot world_count {robot_model.world_count}"
            )
        bodies_per = _per_world_count(cable_model.body_world_start, num_envs)
        robot_bodies_per = _per_world_count(robot_model.body_world_start, num_envs)
        joints_per = _per_world_count(cable_model.joint_world_start, num_envs)
        coord_per = _per_world_count(robot_model.joint_coord_world_start, num_envs)
        dof_per = _per_world_count(robot_model.joint_dof_world_start, num_envs)

        tpl_proxy = int(template_cable.gripper_proxy_body)
        tpl_apple = template_cable.apple_body
        tcp_indices = tuple(
            _global_body_index(w, template_tcp_body, robot_bodies_per) for w in range(num_envs)
        )
        proxy_indices = tuple(
            _global_body_index(w, tpl_proxy, bodies_per) for w in range(num_envs)
        )
        if tpl_apple is not None:
            apple_indices = tuple(
                _global_body_index(w, int(tpl_apple), bodies_per) for w in range(num_envs)
            )
        else:
            apple_indices = tuple(-1 for _ in range(num_envs))

        return cls(
            num_envs=num_envs,
            bodies_per_world=bodies_per,
            robot_bodies_per_world=robot_bodies_per,
            joints_per_world=joints_per,
            joint_coord_count_per_world=coord_per,
            joint_dof_count_per_world=dof_per,
            template_tcp_body=int(template_tcp_body),
            template_proxy_body=tpl_proxy,
            template_apple_body=tpl_apple,
            tcp_body_indices=tcp_indices,
            proxy_body_indices=proxy_indices,
            apple_body_indices=apple_indices,
            env_spacing=tuple(float(v) for v in env_spacing),
        )

    @classmethod
    def from_cable_only(
        cls,
        template_cable: CoupledCableScene,
        cable_model: newton.Model,
        *,
        env_spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> BatchedEnvLayout:
        """Build layout for batched cable-only (``vbd_only``) scenes without a robot model."""
        num_envs = int(cable_model.world_count)
        if num_envs < 1:
            raise ValueError("batched layout requires world_count >= 1")
        bodies_per = _per_world_count(cable_model.body_world_start, num_envs)
        joints_per = _per_world_count(cable_model.joint_world_start, num_envs)
        tpl_proxy = int(template_cable.gripper_proxy_body)
        tpl_apple = template_cable.apple_body
        proxy_indices = tuple(
            _global_body_index(w, tpl_proxy, bodies_per) for w in range(num_envs)
        )
        if tpl_apple is not None:
            apple_indices = tuple(
                _global_body_index(w, int(tpl_apple), bodies_per) for w in range(num_envs)
            )
        else:
            apple_indices = tuple(-1 for _ in range(num_envs))
        return cls(
            num_envs=num_envs,
            bodies_per_world=bodies_per,
            robot_bodies_per_world=0,
            joints_per_world=joints_per,
            joint_coord_count_per_world=0,
            joint_dof_count_per_world=0,
            template_tcp_body=-1,
            template_proxy_body=tpl_proxy,
            template_apple_body=tpl_apple,
            tcp_body_indices=tuple(-1 for _ in range(num_envs)),
            proxy_body_indices=proxy_indices,
            apple_body_indices=apple_indices,
            env_spacing=tuple(float(v) for v in env_spacing),
        )

    def body_index(self, world: int, template_body: int) -> int:
        """Global cable body index for ``template_body`` in ``world``."""
        return _global_body_index(world, template_body, self.bodies_per_world)

    def robot_body_index(self, world: int, template_body: int) -> int:
        """Global robot body index for ``template_body`` in ``world``."""
        return _global_body_index(world, template_body, self.robot_bodies_per_world)

    def joint_q_slice(self, world: int) -> slice:
        """Slice into flat ``joint_q`` for one world."""
        start = int(world) * self.joint_coord_count_per_world
        end = start + self.joint_coord_count_per_world
        return slice(start, end)

    def joint_qd_slice(self, world: int) -> slice:
        """Slice into flat ``joint_qd`` for one world."""
        start = int(world) * self.joint_dof_count_per_world
        end = start + self.joint_dof_count_per_world
        return slice(start, end)

    def joint_index(self, world: int, template_joint: int) -> int:
        """Global cable joint index for a template-world joint in ``world``."""
        return int(world) * self.joints_per_world + int(template_joint)


def _per_world_count(world_start: wp.array | None, num_envs: int) -> int:
    if world_start is None:
        raise ValueError("model missing world_start array")
    arr = world_start.numpy()
    return int(arr[1] - arr[0])


def _global_body_index(world: int, template_body: int, bodies_per_world: int) -> int:
    return int(world) * int(bodies_per_world) + int(template_body)
