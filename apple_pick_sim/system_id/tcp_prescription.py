"""Recorded TCP trajectory → prescribed ``tcp_pose`` action rows (arm-free replay).

Plant sys-ID with ``replay_mode="tcp_prescribed"`` drives the gripper proxy along the
recorded TCP motion instead of replaying the recorded ``vic_pose`` targets through the
sim arm. The plant only interacts with the arm through the grasp, so given the
recorded interface motion its response depends on the plant alone; arm-model error
(Ki, joint friction, task-space inertia) cannot leak into the fitted plant.

The prescription applies the recording's world-frame motion *relative to its
pre-motion reference pose* onto the sim's own post-reset TCP pose, so the sim's
settled grasp is never yanked at t = 0:

    p[k] = p_sim0 + (p_rec[j(k)] - p_ref)
    R[k] = R_rec[j(k)] R_ref^T R_sim0

The replay loop records obs[k] *before* stepping action[k]
(``replay_control_horizon_record_before_step``), so action[k] must reach the pose the
recording shows one control period later:

* converted real bags record *before* stepping (frame 0 is the grasp):
  ``j(k) = k + 1`` (last row holds) and the reference defaults to ``rec[0]``;
* sim-sim collection records *after* stepping (row k is the state after action k,
  the reset pose is not a row): ``j(k) = k``, and the reference must be the
  episode's ``initial_tcp_pos`` / ``initial_tcp_quat`` metadata.

A wrong reference is not benign: the plant is stiff near the TCP, so a ~3 mm offset
(one sim-sim frame of motion at 0.2 m/s) shows up as several newtons.

Quaternions in: recorded/sim ``tcp_quat`` is xyzw (Warp transforms, converter
``pose_4x4_to_pos_quat``). Quaternion out: wxyz, matching the ``vic_pose`` action
contract.
"""

from __future__ import annotations

import numpy as np

TCP_POSE_ACTION_DIM = 7
"""``[pos(3), quat_wxyz(4)]`` absolute world TCP target at the end of the frame."""


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` for xyzw quaternions (broadcasts over leading dims)."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def _normalized_xyzw(q: np.ndarray, *, name: str) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm < 1e-9):
        raise ValueError(f"{name} contains zero-length or non-finite quaternions")
    return q / norm


def prescribed_tcp_actions(
    tcp_pos: np.ndarray,
    tcp_quat_xyzw: np.ndarray,
    *,
    sim_start_pos: np.ndarray,
    sim_start_quat_xyzw: np.ndarray,
    reference_pos: np.ndarray | None = None,
    reference_quat_xyzw: np.ndarray | None = None,
    recorded_after_step: bool = False,
) -> np.ndarray:
    """Return ``(n_frames, 7)`` ``[pos, quat_wxyz]`` world TCP targets, one per replay step.

    ``reference_*`` default to the first recorded frame; see the module docstring for
    ``recorded_after_step``.
    """
    pos = np.asarray(tcp_pos, dtype=np.float64)
    quat = np.asarray(tcp_quat_xyzw, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"tcp_pos must be (n_frames, 3), got {pos.shape}")
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"tcp_quat_xyzw must be (n_frames, 4), got {quat.shape}")
    if pos.shape[0] != quat.shape[0]:
        raise ValueError(
            f"tcp_pos and tcp_quat_xyzw frame counts differ: {pos.shape[0]} vs {quat.shape[0]}"
        )
    n = int(pos.shape[0])
    if n < 1:
        raise ValueError("need at least one recorded frame")
    p_sim0 = np.asarray(sim_start_pos, dtype=np.float64).reshape(3)
    q_sim0 = _normalized_xyzw(
        np.asarray(sim_start_quat_xyzw, dtype=np.float64).reshape(4), name="sim_start_quat_xyzw"
    )
    quat = _normalized_xyzw(quat, name="tcp_quat_xyzw")
    p_ref = pos[0] if reference_pos is None else np.asarray(reference_pos, np.float64).reshape(3)
    q_ref = (
        quat[0]
        if reference_quat_xyzw is None
        else _normalized_xyzw(
            np.asarray(reference_quat_xyzw, np.float64).reshape(4), name="reference_quat_xyzw"
        )
    )

    src = np.arange(n) if recorded_after_step else np.minimum(np.arange(n) + 1, n - 1)
    out_pos = p_sim0 + (pos[src] - p_ref)
    world_delta = _quat_mul_xyzw(quat[src], _quat_conj_xyzw(q_ref))
    out_quat = _normalized_xyzw(_quat_mul_xyzw(world_delta, q_sim0), name="prescribed quat")

    actions = np.empty((n, TCP_POSE_ACTION_DIM), dtype=np.float64)
    actions[:, :3] = out_pos
    actions[:, 3:] = out_quat[:, [3, 0, 1, 2]]
    return actions
