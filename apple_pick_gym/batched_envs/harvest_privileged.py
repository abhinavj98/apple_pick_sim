"""Privileged (critic-only) ground truth for the harvest env's domain randomization.

The asymmetric actor-critic gives the critic what the actor has to infer from
interaction history: every DR draw that shapes this episode's dynamics.

- **Plant** (baked per env at build, fixed for the run): spur/stem flexural and
  axial moduli, damping ratios, primary rod density.
- **Support joints** (baked per env at build): kp, roll_kp, zeta.
- **Arm, build-time** (per env, fixed for the run): link mass/inertia and EE
  payload scales.
- **Arm, per reset**: joint armature, Coulomb friction, viscous damping.
- **Geometry** (:func:`build_plant_geometry`): spur/stem length and radius, apple
  radius and density, and the grasp's weld (pull) direction.

Moduli and support stiffnesses span decades (up to ~1e10 Pa) and CMA searches
them in log space, so they enter as ``log10``; everything else is raw and is
standardized downstream by the trainer's running scaler.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from apple_pick_gym.batched_envs.harvest_obs import _PRIVILEGED_FIELDS

LOG10_PRIVILEGED_FIELDS: tuple[str, ...] = (
    "spur_flexural_modulus_pa",
    "stem_flexural_modulus_pa",
    "spur_youngs_modulus_pa",
    "stem_youngs_modulus_pa",
    "support_kp",
    "support_roll_kp",
)

PLANT_GEOMETRY_FIELDS: tuple[tuple[str, int], ...] = (
    ("spur_length_m", 1),
    ("spur_radius_m", 1),
    ("stem_length_m", 1),
    ("stem_radius_m", 1),
    ("apple_radius_m", 1),
    ("apple_density", 1),
    ("weld_direction", 3),
)


def _col(values: Sequence[float] | np.ndarray, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32).reshape(-1, 1), device=device)


def _nominal_joint_arrays(n: int) -> dict[str, np.ndarray]:
    from apple_pick_sim.robot.fr3_robot import setup as fr3_setup

    def tile(v):
        return np.tile(np.asarray(v, dtype=np.float32).reshape(1, 7), (n, 1))

    return {
        "armature": tile(fr3_setup.FR3_REFLECTED_MOTOR_INERTIA_KGM2),
        "friction": tile(fr3_setup.FR3_DEFAULT_JOINT_FRICTION),
        "joint_damping": tile(fr3_setup.FR3_DEFAULT_VIC_JOINT_DAMPING),
    }


def build_privileged_fields(
    per_env_params: Sequence[Any],
    support_sample: Any,
    arm_build_sample: Any,
    arm_joint_sample: Any | None,
    *,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """``_PRIVILEGED_FIELDS``-ordered dict of ``(N, width)`` float32 tensors.

    ``arm_joint_sample`` is the per-reset joint-dynamics draw (``None`` before the
    first reset: nominal values).
    """
    n = len(per_env_params)
    for name, arr in (
        ("support_sample.kp", support_sample.kp),
        ("arm_build_sample.link_mass_scale", arm_build_sample.link_mass_scale),
    ):
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} envs, expected {n}")
    if arm_joint_sample is None:
        joint = _nominal_joint_arrays(n)
    else:
        joint = {
            "armature": np.asarray(arm_joint_sample.armature, dtype=np.float32),
            "friction": np.asarray(arm_joint_sample.friction, dtype=np.float32),
            "joint_damping": np.asarray(arm_joint_sample.joint_damping, dtype=np.float32),
        }
        if joint["armature"].shape != (n, 7):
            raise ValueError(f"arm_joint_sample.armature shape {joint['armature'].shape}, expected ({n}, 7)")

    raw: dict[str, Any] = {
        "spur_flexural_modulus_pa": [p.spur.flexural_modulus_pa for p in per_env_params],
        "stem_flexural_modulus_pa": [p.stem.flexural_modulus_pa for p in per_env_params],
        "spur_youngs_modulus_pa": [p.spur.youngs_modulus_pa for p in per_env_params],
        "stem_youngs_modulus_pa": [p.stem.youngs_modulus_pa for p in per_env_params],
        "spur_damping_ratio": [p.spur.damping_ratio for p in per_env_params],
        "stem_damping_ratio": [p.stem.damping_ratio for p in per_env_params],
        "primary_density": [p.primary.density for p in per_env_params],
        "support_kp": support_sample.kp,
        "support_roll_kp": support_sample.roll_kp,
        "support_zeta": support_sample.zeta,
        "arm_link_mass_scale": arm_build_sample.link_mass_scale,
        "arm_link_inertia_scale": arm_build_sample.link_inertia_scale,
        "arm_ee_mass_scale": arm_build_sample.ee_mass_scale,
        "arm_ee_inertia_scale": arm_build_sample.ee_inertia_scale,
    }
    out: dict[str, torch.Tensor] = {}
    for name, width in _PRIVILEGED_FIELDS:
        if name == "arm_armature":
            out[name] = torch.as_tensor(joint["armature"], device=device)
        elif name == "arm_friction":
            out[name] = torch.as_tensor(joint["friction"], device=device)
        elif name == "arm_joint_damping":
            out[name] = torch.as_tensor(joint["joint_damping"], device=device)
        else:
            vals = np.asarray(raw[name], dtype=np.float64)
            if name in LOG10_PRIVILEGED_FIELDS:
                vals = np.log10(np.maximum(vals, 1e-30))
            out[name] = _col(vals, device)
        assert out[name].shape == (n, width), name
    return out


def build_plant_geometry(
    per_env_params: Sequence[Any],
    weld_directions: Sequence[Sequence[float]],
    *,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """``PLANT_GEOMETRY_FIELDS``-ordered dict: rod/apple geometry and the grasp's weld axis."""
    if len(weld_directions) != len(per_env_params):
        raise ValueError("weld_directions and per_env_params differ in length")

    def _apple(p, attr):
        v = getattr(p, attr)
        return float("nan") if v is None else float(v)

    out = {
        "spur_length_m": _col([p.spur.length for p in per_env_params], device),
        "spur_radius_m": _col([p.spur.radius for p in per_env_params], device),
        "stem_length_m": _col([p.stem.length for p in per_env_params], device),
        "stem_radius_m": _col([p.stem.radius for p in per_env_params], device),
        "apple_radius_m": _col([_apple(p, "apple_radius") for p in per_env_params], device),
        "apple_density": _col([_apple(p, "apple_density") for p in per_env_params], device),
        "weld_direction": torch.as_tensor(np.asarray(weld_directions, dtype=np.float32), device=device),
    }
    for name, t in out.items():
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"plant geometry field {name!r} is not finite (unset apple params?)")
    return out
