"""Per-env Newton / MuJoCo robot diagnostics for batched coupled scenes."""

from __future__ import annotations

from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout

_TEMPLATE_ROOT_BODY = 0


def _fmt_vec(values: np.ndarray, *, precision: int = 4) -> str:
    parts = [f"{float(v):+.{precision}f}" for v in np.asarray(values).reshape(-1)]
    return "[" + ", ".join(parts) + "]"


def _mujoco_qpos_for_world(mj_solver: Any, world: int) -> np.ndarray | None:
    """Return MuJoCo ``qpos`` for one replicated world, or ``None`` if unavailable."""
    mjw = getattr(mj_solver, "mjw_data", None)
    if mjw is not None and getattr(mjw, "qpos", None) is not None:
        qpos = mjw.qpos.numpy()
        if qpos.ndim == 2:
            return np.asarray(qpos[int(world)], dtype=np.float64).reshape(-1)
        if int(world) == 0:
            return np.asarray(qpos, dtype=np.float64).reshape(-1)
        return None
    mj_data = getattr(mj_solver, "mj_data", None)
    if mj_data is None:
        return None
    if int(world) == 0:
        return np.asarray(mj_data.qpos, dtype=np.float64).reshape(-1)
    return None


def batched_robot_diagnostics(
    scene: Any,
    layout: BatchedEnvLayout,
    world: int,
) -> dict[str, Any]:
    """Collect Newton and MuJoCo robot state for one batched env."""
    model = scene.robot_model
    state = scene.robot_state_0
    if model is None or state is None:
        return {}

    root_idx = layout.robot_body_index(world, _TEMPLATE_ROOT_BODY)
    body_q = state.body_q.numpy().reshape(-1, 7)[root_idx]
    base_pos = np.asarray(body_q[:3], dtype=np.float64)
    base_quat = np.asarray(body_q[3:7], dtype=np.float64)

    jq_slice = layout.joint_q_slice(world)
    jqd_slice = layout.joint_qd_slice(world)
    newton_joint_q = np.asarray(model.joint_q.numpy()[jq_slice], dtype=np.float64)
    state_joint_q = np.asarray(state.joint_q.numpy()[jq_slice], dtype=np.float64)
    newton_joint_qd = np.asarray(state.joint_qd.numpy()[jqd_slice], dtype=np.float64)

    mj_qpos: np.ndarray | None = None
    mj_solver = getattr(scene, "mj_solver", None)
    if mj_solver is not None:
        mj_qpos = _mujoco_qpos_for_world(mj_solver, world)

    target_pos: np.ndarray | None = None
    control = getattr(scene, "robot_control", None)
    if control is not None and getattr(control, "joint_target_pos", None) is not None:
        target_pos = np.asarray(
            control.joint_target_pos.numpy()[jqd_slice],
            dtype=np.float64,
        )

    jq_mj_delta: float | None = None
    if mj_qpos is not None and mj_qpos.size == newton_joint_q.size:
        jq_mj_delta = float(np.max(np.abs(newton_joint_q - mj_qpos)))

    model_state_delta = float(np.max(np.abs(newton_joint_q - state_joint_q)))

    return {
        "world": int(world),
        "base_pos": base_pos,
        "base_quat": base_quat,
        "newton_joint_q": newton_joint_q,
        "state_joint_q": state_joint_q,
        "newton_joint_qd": newton_joint_qd,
        "mujoco_qpos": mj_qpos,
        "joint_target_pos": target_pos,
        "model_state_joint_q_max_abs": model_state_delta,
        "newton_mujoco_qpos_max_abs": jq_mj_delta,
    }


def format_batched_robot_status_line(
    diag: dict[str, Any],
    *,
    joint_precision: int = 4,
) -> str:
    """One-line summary for an env."""
    w = int(diag["world"])
    base = diag["base_pos"]
    parts = [
        f"env{w}: base=({base[0]:.{joint_precision}f}, {base[1]:.{joint_precision}f}, "
        f"{base[2]:.{joint_precision}f})",
        f"newton_q={_fmt_vec(diag['newton_joint_q'], precision=joint_precision)}",
    ]
    mj_qpos = diag.get("mujoco_qpos")
    if mj_qpos is not None:
        parts.append(f"mj_qpos={_fmt_vec(mj_qpos, precision=joint_precision)}")
    tgt = diag.get("joint_target_pos")
    if tgt is not None:
        parts.append(f"mj_tgt={_fmt_vec(tgt, precision=joint_precision)}")
    deltas: list[str] = []
    if diag.get("model_state_joint_q_max_abs") is not None:
        deltas.append(f"|model-state_q|={diag['model_state_joint_q_max_abs']:.2e}")
    if diag.get("newton_mujoco_qpos_max_abs") is not None:
        deltas.append(f"|newton-mj_q|={diag['newton_mujoco_qpos_max_abs']:.2e}")
    if deltas:
        parts.append(" ".join(deltas))
    return "  " + "  ".join(parts)


def print_batched_robot_status(scene: Any, layout: BatchedEnvLayout, *, prefix: str = "") -> None:
    """Print base pose and joint angles for every replicated robot world."""
    head = f"{prefix}batched robot state ({layout.num_envs} envs):"
    print(head, flush=True)
    for w in range(layout.num_envs):
        diag = batched_robot_diagnostics(scene, layout, w)
        if not diag:
            print(f"  env{w}: (no robot state)", flush=True)
            continue
        print(format_batched_robot_status_line(diag), flush=True)
