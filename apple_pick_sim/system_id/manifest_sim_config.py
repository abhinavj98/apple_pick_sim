"""Serialize and compare replay-relevant sim build settings for batched sys-ID manifests."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)

_FLOAT_ATOL = 1e-9


def _float_close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= _FLOAT_ATOL


def _dict_float_close(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    return all(_float_close(left[k], right[k]) for k in left)


def sim_config_to_manifest_dict(
    config: BatchedHeterogeneousCoupledSimConfig,
    *,
    applied_joint_kd_overrides: Mapping[str, float] | None = None,
    applied_joint_linear_kd_overrides: Mapping[str, float] | None = None,
    applied_joint_angular_kp_overrides: Mapping[str, float] | None = None,
    applied_joint_linear_kp_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Serialize replay-relevant fields from the effective batched sim config."""
    fs_cfg = config.fruiting_system
    ctrl = config.controller
    scene = config.scene
    robot = config.robot
    gains = ctrl.vic_gains

    out: dict[str, Any] = {
        "sub_dt": float(config.runtime.sub_dt),
        "settle_substeps": int(scene.settle_substeps),
        "settle_gravity_ramp": bool(scene.settle_gravity_ramp),
        "settle_max_speed_m_s": float(scene.settle_max_speed_m_s),
        "enable_self_collisions": bool(scene.enable_self_collisions),
        "enable_apple_woody_collisions": bool(scene.enable_apple_woody_collisions),
        "enable_proxy_woody_collisions": bool(scene.enable_proxy_woody_collisions),
        "stem_coupling_gain": float(fs_cfg.stem_coupling_gain),
        "stem_force_cap_N": (
            None if fs_cfg.stem_force_cap_N is None else float(fs_cfg.stem_force_cap_N)
        ),
        "stem_torque_cap_Nm": (
            None if fs_cfg.stem_torque_cap_Nm is None else float(fs_cfg.stem_torque_cap_Nm)
        ),
        "joint_angular_kd_overrides": {
            str(k): float(v) for k, v in sorted(fs_cfg.joint_angular_kd_overrides.items())
        },
        "joint_linear_kd_overrides": {
            str(k): float(v) for k, v in sorted(fs_cfg.joint_linear_kd_overrides.items())
        },
        "joint_angular_kp_overrides": {
            str(k): float(v) for k, v in sorted(fs_cfg.joint_angular_kp_overrides.items())
        },
        "joint_linear_kp_overrides": {
            str(k): float(v) for k, v in sorted(fs_cfg.joint_linear_kp_overrides.items())
        },
        "controller": {
            "mode": str(ctrl.mode),
            "linear_speed": float(ctrl.linear_speed),
            "angular_speed": float(ctrl.angular_speed),
            "ik_iterations": int(ctrl.ik_iterations),
            "vic_gains": {
                "linear_k": float(gains.linear_k),
                "linear_d": float(gains.linear_d),
                "angular_k": float(gains.angular_k),
                "angular_d": float(gains.angular_d),
            },
        },
        "robot": {
            "fix_to_apple": bool(robot.fix_to_apple),
            "gripper_mass_kg": float(robot.gripper.mass),
        },
    }
    if applied_joint_kd_overrides is not None:
        out["joint_angular_kd_applied"] = {
            str(k): float(v) for k, v in sorted(applied_joint_kd_overrides.items())
        }
    if applied_joint_linear_kd_overrides is not None:
        out["joint_linear_kd_applied"] = {
            str(k): float(v) for k, v in sorted(applied_joint_linear_kd_overrides.items())
        }
    if applied_joint_angular_kp_overrides is not None:
        out["joint_angular_kp_applied"] = {
            str(k): float(v) for k, v in sorted(applied_joint_angular_kp_overrides.items())
        }
    if applied_joint_linear_kp_overrides is not None:
        out["joint_linear_kp_applied"] = {
            str(k): float(v) for k, v in sorted(applied_joint_linear_kp_overrides.items())
        }
    return out


def _append_float_mismatch(
    mismatches: list[str],
    *,
    path: str,
    recorded: Any,
    replay: float,
) -> None:
    if recorded is None:
        mismatches.append(f"{path}: missing in manifest sim_config (replay={replay!r})")
        return
    if not _float_close(recorded, replay):
        mismatches.append(
            f"{path}: manifest={float(recorded)!r} replay={float(replay)!r}"
        )


def _append_bool_mismatch(
    mismatches: list[str],
    *,
    path: str,
    recorded: Any,
    replay: bool,
) -> None:
    if recorded is None:
        mismatches.append(f"{path}: missing in manifest sim_config (replay={replay!r})")
        return
    if bool(recorded) != bool(replay):
        mismatches.append(
            f"{path}: manifest={bool(recorded)!r} replay={bool(replay)!r}"
        )


def sim_config_manifest_mismatches(
    recorded: Mapping[str, Any] | None,
    replay_config: BatchedHeterogeneousCoupledSimConfig,
) -> list[str]:
    """Return human-readable mismatch messages; empty when recorded is absent or matches."""
    if not recorded:
        return []

    replay = sim_config_to_manifest_dict(replay_config)
    mismatches: list[str] = []

    for key in (
        "sub_dt",
        "settle_substeps",
        "settle_gravity_ramp",
        "settle_max_speed_m_s",
        "enable_self_collisions",
        "enable_apple_woody_collisions",
        "enable_proxy_woody_collisions",
        "stem_coupling_gain",
        "stem_force_cap_N",
        "stem_torque_cap_Nm",
    ):
        rec_val = recorded.get(key)
        rep_val = replay[key]
        if isinstance(rep_val, bool):
            _append_bool_mismatch(mismatches, path=key, recorded=rec_val, replay=rep_val)
        elif isinstance(rep_val, int) and not isinstance(rep_val, bool):
            if rec_val is None or int(rec_val) != int(rep_val):
                mismatches.append(
                    f"{key}: manifest={rec_val!r} replay={int(rep_val)!r}"
                )
        else:
            _append_float_mismatch(mismatches, path=key, recorded=rec_val, replay=float(rep_val))

    rec_kd = recorded.get("joint_angular_kd_overrides")
    rep_kd = replay["joint_angular_kd_overrides"]
    if rec_kd is None:
        mismatches.append(
            "joint_angular_kd_overrides: missing in manifest sim_config"
        )
    elif not _dict_float_close(
        {str(k): float(v) for k, v in rec_kd.items()},
        rep_kd,
    ):
        mismatches.append(
            "joint_angular_kd_overrides: "
            f"manifest={dict(rec_kd)!r} replay={rep_kd!r}"
        )

    rec_lin_kd = recorded.get("joint_linear_kd_overrides")
    if rec_lin_kd is not None:
        rep_lin_kd = replay["joint_linear_kd_overrides"]
        if not _dict_float_close(
            {str(k): float(v) for k, v in rec_lin_kd.items()},
            rep_lin_kd,
        ):
            mismatches.append(
                "joint_linear_kd_overrides: "
                f"manifest={dict(rec_lin_kd)!r} replay={rep_lin_kd!r}"
            )

    rec_ang_kp = recorded.get("joint_angular_kp_overrides")
    if rec_ang_kp is not None:
        rep_ang_kp = replay["joint_angular_kp_overrides"]
        if not _dict_float_close(
            {str(k): float(v) for k, v in rec_ang_kp.items()},
            rep_ang_kp,
        ):
            mismatches.append(
                "joint_angular_kp_overrides: "
                f"manifest={dict(rec_ang_kp)!r} replay={rep_ang_kp!r}"
            )

    rec_lin_kp = recorded.get("joint_linear_kp_overrides")
    if rec_lin_kp is not None:
        rep_lin_kp = replay["joint_linear_kp_overrides"]
        if not _dict_float_close(
            {str(k): float(v) for k, v in rec_lin_kp.items()},
            rep_lin_kp,
        ):
            mismatches.append(
                "joint_linear_kp_overrides: "
                f"manifest={dict(rec_lin_kp)!r} replay={rep_lin_kp!r}"
            )

    rec_ctrl = recorded.get("controller") or {}
    rep_ctrl = replay["controller"]
    for key in ("mode", "linear_speed", "angular_speed", "ik_iterations"):
        path = f"controller.{key}"
        rec_val = rec_ctrl.get(key)
        rep_val = rep_ctrl[key]
        if key == "mode":
            if rec_val is None or str(rec_val) != str(rep_val):
                mismatches.append(
                    f"{path}: manifest={rec_val!r} replay={rep_val!r}"
                )
        elif key == "ik_iterations":
            if rec_val is None or int(rec_val) != int(rep_val):
                mismatches.append(
                    f"{path}: manifest={rec_val!r} replay={int(rep_val)!r}"
                )
        else:
            _append_float_mismatch(
                mismatches,
                path=path,
                recorded=rec_val,
                replay=float(rep_val),
            )

    rec_gains = rec_ctrl.get("vic_gains") or {}
    rep_gains = rep_ctrl["vic_gains"]
    for key in ("linear_k", "linear_d", "angular_k", "angular_d"):
        path = f"controller.vic_gains.{key}"
        _append_float_mismatch(
            mismatches,
            path=path,
            recorded=rec_gains.get(key),
            replay=float(rep_gains[key]),
        )

    rec_robot = recorded.get("robot") or {}
    rep_robot = replay["robot"]
    _append_bool_mismatch(
        mismatches,
        path="robot.fix_to_apple",
        recorded=rec_robot.get("fix_to_apple"),
        replay=bool(rep_robot["fix_to_apple"]),
    )
    _append_float_mismatch(
        mismatches,
        path="robot.gripper_mass_kg",
        recorded=rec_robot.get("gripper_mass_kg"),
        replay=float(rep_robot["gripper_mass_kg"]),
    )

    return mismatches


def warn_manifest_sim_config_mismatch(
    dataset_or_manifest: Any,
    replay_config: BatchedHeterogeneousCoupledSimConfig,
    *,
    warn: Callable[[str], Any] = warnings.warn,
) -> list[str]:
    """Emit warnings when replay sim config differs from manifest collection.sim_config."""
    if hasattr(dataset_or_manifest, "manifest"):
        manifest = dataset_or_manifest.manifest
    else:
        manifest = dict(dataset_or_manifest)

    recorded = manifest.get("collection", {}).get("sim_config")
    mismatches = sim_config_manifest_mismatches(recorded, replay_config)
    for message in mismatches:
        warn(f"manifest sim_config mismatch: {message}")
    return mismatches
