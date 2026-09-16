"""Per-env support joint kp/kd applicator for batched sys-ID (gym-side wrapper)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from apple_pick_sim.coupled_fruiting.scene import CoupledFruitingScene
from apple_pick_sim.fruiting_system import (
    set_fruiting_joint_angular_kd_batched,
    set_fruiting_joint_angular_kp_batched,
    set_fruiting_joint_linear_kd_batched,
    set_fruiting_joint_linear_kp_batched,
)
from apple_pick_sim.fruiting_system.joint_kd_scaling import (
    SUPPORT_ANGULAR_KP_LENGTH_FACTOR,
    joint_kd_from_damping_ratio,
    support_angular_kp_from_linear,
)

# Used only when dataset ``collection.sim_config.joint_damping_ratio`` is absent.
# Prefer dataset ζ so replay support kd matches collect-time weld damping.
SUPPORT_JOINT_ZETA_FALLBACK: float = 1.0
# Back-compat alias (was hardcoded 1.0; keep equal to the variance fixture ζ).
SUPPORT_JOINT_ZETA: float = SUPPORT_JOINT_ZETA_FALLBACK


def support_joint_zeta_from_dataset(dataset: Any) -> float:
    """Return support-joint ζ for candidate apply (not a free sys-ID parameter).

    Reads ``manifest['collection']['sim_config']['joint_damping_ratio']`` so
    replay support ``kd`` matches the damping used when the dataset was
    collected. Falls back to :data:`SUPPORT_JOINT_ZETA_FALLBACK` when missing.
    """
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(manifest, dict):
        return float(SUPPORT_JOINT_ZETA_FALLBACK)
    collection = manifest.get("collection", {})
    if not isinstance(collection, dict):
        return float(SUPPORT_JOINT_ZETA_FALLBACK)
    sim_config = collection.get("sim_config", {})
    if not isinstance(sim_config, dict):
        return float(SUPPORT_JOINT_ZETA_FALLBACK)
    raw = sim_config.get("joint_damping_ratio", None)
    if raw is None:
        return float(SUPPORT_JOINT_ZETA_FALLBACK)
    zeta = float(raw)
    if not math.isfinite(zeta) or zeta < 0.0:
        path = getattr(dataset, "dataset_dir", "<unknown>")
        raise ValueError(
            f"joint_damping_ratio must be finite and >= 0 "
            f"(dataset={path!s}, got {raw!r})"
        )
    return zeta


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


def _validate_dowel_length_per_env(
    dowel_length_m_per_env: Sequence[float],
    *,
    num_envs: int,
) -> tuple[float, ...]:
    if len(dowel_length_m_per_env) != int(num_envs):
        raise ValueError(
            f"dowel_length_m_per_env length ({len(dowel_length_m_per_env)}) must "
            f"match num_envs ({num_envs})"
        )
    validated: list[float] = []
    for raw in dowel_length_m_per_env:
        support_angular_kp_from_linear(1.0, raw)
        validated.append(float(raw))
    return tuple(validated)


def apply_per_env_support_joint_penalties(
    scene: CoupledFruitingScene,
    support_kp_per_env: Sequence[float],
    *,
    num_envs: int,
    joints_per_world: int,
    dowel_length_m_per_env: Sequence[float],
    zeta: float = SUPPORT_JOINT_ZETA_FALLBACK,
    zeta_per_env: Sequence[float] | None = None,
) -> None:
    """Set per-env support angular/linear kp and kd (ζ via ``zeta`` / ``zeta_per_env``).

    ``support_kp_per_env`` is linear (N/m). Angular kp is
    ``(3/4) * L**2 * k_lin`` with ``L`` from
    ``dowel_length_m_per_env`` (primary length). Non-support roles retain
    build-time penalty values. Callers should pass
    ``zeta=support_joint_zeta_from_dataset(dataset)`` for collect/replay parity,
    or ``zeta_per_env`` when CMA searches support-joint ζ per candidate.
    """
    kp_per_env = _validate_support_kp_per_env(support_kp_per_env, num_envs=num_envs)
    lengths = _validate_dowel_length_per_env(
        dowel_length_m_per_env, num_envs=num_envs
    )
    if zeta_per_env is not None:
        if len(zeta_per_env) != int(num_envs):
            raise ValueError(
                f"zeta_per_env length ({len(zeta_per_env)}) must match "
                f"num_envs ({num_envs})"
            )
        zetas = tuple(float(z) for z in zeta_per_env)
        for idx, z in enumerate(zetas):
            if not math.isfinite(z) or z < 0.0:
                raise ValueError(
                    f"zeta_per_env[{idx}] must be finite and >= 0, got {z!r}"
                )
    else:
        zetas = tuple(float(zeta) for _ in range(int(num_envs)))
    layout = scene.layout
    if layout is None:
        raise ValueError("scene.layout is required for per-env support joint penalties")

    cable = scene.cable
    per_env_ang_kp = [
        {"support": support_angular_kp_from_linear(kp, length)}
        for kp, length in zip(kp_per_env, lengths, strict=True)
    ]
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
    for w, (kp, length, env_zeta) in enumerate(
        zip(kp_per_env, lengths, zetas, strict=True)
    ):
        ang_kd, lin_kd = joint_kd_from_damping_ratio(
            zeta=env_zeta,
            roles=("support",),
            fruiting_fixed_joints=joints,
            body_mass=body_mass,
            body_inertia=body_inertia,
            joint_child=joint_child,
            angular_kp_by_role={
                "support": support_angular_kp_from_linear(kp, length)
            },
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


def apply_per_env_support_roll_penalties(
    scene: CoupledFruitingScene,
    support_roll_kp_per_env: Sequence[float],
    *,
    num_envs: int,
    joints_per_world: int,
    zeta: float = SUPPORT_JOINT_ZETA_FALLBACK,
    zeta_per_env: Sequence[float] | None = None,
) -> None:
    """Set per-env T-roll support kp/kd (dual-write target_ke + penalty slot).

    ``support_roll_kp_per_env`` is revolute drive stiffness (N·m/rad). Callers
    should pass ``zeta=support_joint_zeta_from_dataset(dataset)`` for collect
    / replay parity, or ``zeta_per_env`` when CMA searches support-joint ζ.
    """
    from apple_pick_sim.fruiting_system import set_fruiting_joint_roll_kp_batched
    from apple_pick_sim.fruiting_system.build import _roll_kd_overrides_from_damping_ratio

    if len(support_roll_kp_per_env) != int(num_envs):
        raise ValueError(
            f"support_roll_kp_per_env length ({len(support_roll_kp_per_env)}) "
            f"must match num_envs ({num_envs})"
        )
    layout = scene.layout
    if layout is None:
        raise ValueError("scene.layout is required for per-env support roll penalties")

    if zeta_per_env is not None:
        if len(zeta_per_env) != int(num_envs):
            raise ValueError(
                f"zeta_per_env length ({len(zeta_per_env)}) must match "
                f"num_envs ({num_envs})"
            )
        zetas = tuple(float(z) for z in zeta_per_env)
        for idx, z in enumerate(zetas):
            if not math.isfinite(z) or z < 0.0:
                raise ValueError(
                    f"zeta_per_env[{idx}] must be finite and >= 0, got {z!r}"
                )
    else:
        zetas = tuple(float(zeta) for _ in range(int(num_envs)))

    cable = scene.cable
    per_env_kp = [{"support": float(kp)} for kp in support_roll_kp_per_env]
    for idx, kp in enumerate(support_roll_kp_per_env):
        if float(kp) < 0.0:
            raise ValueError(
                f"support_roll_kp_per_env[{idx}] must be >= 0, got {kp!r}"
            )

    model = cable.model
    joint_child = model.joint_child.numpy()
    body_inertia = model.body_inertia.numpy()
    bodies_per_world = int(layout.bodies_per_world)
    per_env_kd = [
        _roll_kd_overrides_from_damping_ratio(
            cable.fruiting_fixed_joints,
            env_kp,
            zeta=float(env_zeta),
            joint_child=joint_child,
            body_inertia=body_inertia,
            body_offset=int(w) * bodies_per_world,
        )
        for w, (env_kp, env_zeta) in enumerate(zip(per_env_kp, zetas, strict=True))
    ]
    set_fruiting_joint_roll_kp_batched(
        cable.solver,
        cable.model,
        cable.fruiting_fixed_joints,
        label_kp_per_env=per_env_kp,
        label_kd_per_env=per_env_kd,
        num_envs=int(num_envs),
        joints_per_world=int(joints_per_world),
    )
