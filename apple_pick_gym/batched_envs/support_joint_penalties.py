"""Per-env support joint kp/kd applicator for batched sys-ID (gym-side wrapper)."""

from __future__ import annotations

from collections.abc import Sequence

from apple_pick_sim.coupled_fruiting.scene import CoupledFruitingScene
from apple_pick_sim.fruiting_system import (
    set_fruiting_joint_angular_kd_batched,
    set_fruiting_joint_angular_kp_batched,
    set_fruiting_joint_linear_kd_batched,
    set_fruiting_joint_linear_kp_batched,
)
from apple_pick_sim.fruiting_system.joint_kd_scaling import joint_kd_from_damping_ratio

SUPPORT_JOINT_ZETA: float = 1.0


def _validate_support_kp_per_env(
    support_kp_per_env: Sequence[float],
    *,
    num_envs: int,
) -> tuple[float, ...]:
    if len(support_kp_per_env) != int(num_envs):
        raise ValueError(
            f"support_kp_per_env length ({len(support_kp_per_env)}) must match "
            f"num_envs ({num_envs})"
        )
    validated: list[float] = []
    for idx, raw in enumerate(support_kp_per_env):
        kp = float(raw)
        if kp <= 0.0:
            raise ValueError(
                f"support_kp_per_env[{idx}] must be positive, got {raw!r}"
            )
        validated.append(kp)
    return tuple(validated)


def apply_per_env_support_joint_penalties(
    scene: CoupledFruitingScene,
    support_kp_per_env: Sequence[float],
    *,
    num_envs: int,
    joints_per_world: int,
    zeta: float = SUPPORT_JOINT_ZETA,
) -> None:
    """Set per-env support angular/linear kp and critical-damping kd (ζ via ``zeta``).

    Non-support roles retain their build-time penalty values.
    """
    kp_per_env = _validate_support_kp_per_env(support_kp_per_env, num_envs=num_envs)
    layout = scene.layout
    if layout is None:
        raise ValueError("scene.layout is required for per-env support joint penalties")

    cable = scene.cable
    per_env_ang_kp = [{"support": kp} for kp in kp_per_env]
    per_env_lin_kp = [{"support": kp} for kp in kp_per_env]

    set_fruiting_joint_angular_kp_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        label_kp_per_env=per_env_ang_kp,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    set_fruiting_joint_linear_kp_batched(
        cable.solver,
        cable.fruiting_fixed_joints,
        label_kp_per_env=per_env_lin_kp,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )

    model = cable.model
    body_mass = model.body_mass.numpy()
    body_inertia = model.body_inertia.numpy()
    joint_child = model.joint_child.numpy()
    bodies_per_world = int(layout.bodies_per_world)
    joints = list(cable.fruiting_fixed_joints)

    per_env_ang_kd: list[dict[str, float]] = []
    per_env_lin_kd: list[dict[str, float]] = []
    for w, kp in enumerate(kp_per_env):
        ang_kd, lin_kd = joint_kd_from_damping_ratio(
            zeta=zeta,
            roles=("support",),
            fruiting_fixed_joints=joints,
            body_mass=body_mass,
            body_inertia=body_inertia,
            joint_child=joint_child,
            angular_kp_by_role={"support": kp},
            linear_kp_by_role={"support": kp},
            body_offset=int(w) * bodies_per_world,
        )
        per_env_ang_kd.append(ang_kd)
        per_env_lin_kd.append(lin_kd)

    set_fruiting_joint_angular_kd_batched(
        cable.solver,
        joints,
        label_kd_per_env=per_env_ang_kd,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
    set_fruiting_joint_linear_kd_batched(
        cable.solver,
        joints,
        label_kd_per_env=per_env_lin_kd,
        num_envs=num_envs,
        joints_per_world=joints_per_world,
    )
