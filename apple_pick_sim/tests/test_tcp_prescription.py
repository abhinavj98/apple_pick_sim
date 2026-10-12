"""Tests for recorded-TCP → prescribed ``tcp_pose`` action rows."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from apple_pick_sim.system_id.tcp_prescription import (
    TCP_POSE_ACTION_DIM,
    prescribed_tcp_actions,
)

_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def _quat_wxyz_to_rot(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q_wxyz)[[1, 2, 3, 0]])


def test_static_recording_holds_sim_start_pose():
    n = 4
    rec_pos = np.tile([0.4, 0.1, 0.3], (n, 1))
    rec_quat = np.tile(_IDENTITY_XYZW, (n, 1))
    sim_pos = np.array([0.5, 0.2, 0.25])
    sim_quat = Rotation.from_euler("x", 30, degrees=True).as_quat()

    actions = prescribed_tcp_actions(
        rec_pos, rec_quat, sim_start_pos=sim_pos, sim_start_quat_xyzw=sim_quat
    )

    assert actions.shape == (n, TCP_POSE_ACTION_DIM)
    np.testing.assert_allclose(actions[:, :3], np.tile(sim_pos, (n, 1)), atol=1e-12)
    for row in actions:
        assert (_quat_wxyz_to_rot(row[3:]) * Rotation.from_quat(sim_quat).inv()).magnitude() < 1e-9


def test_action_k_targets_recorded_frame_k_plus_one_and_last_frame_holds():
    """Replay records obs[k] before stepping action[k], so action[k] → recorded frame k+1."""
    rec_pos = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.03, 0.0, 0.0]])
    rec_quat = np.tile(_IDENTITY_XYZW, (3, 1))
    sim_pos = np.array([1.0, 2.0, 3.0])

    actions = prescribed_tcp_actions(
        rec_pos, rec_quat, sim_start_pos=sim_pos, sim_start_quat_xyzw=_IDENTITY_XYZW
    )

    np.testing.assert_allclose(
        actions[:, :3],
        [[1.01, 2.0, 3.0], [1.03, 2.0, 3.0], [1.03, 2.0, 3.0]],
        atol=1e-12,
    )


def test_world_frame_rotation_delta_composes_onto_sim_start():
    """R[k] = R_rec[k+1] R_rec[0]^T R_sim0 (world-frame increment, sim's own start)."""
    rec_r0 = Rotation.from_euler("y", 20, degrees=True)
    step = Rotation.from_euler("z", 7, degrees=True)
    rec_quat = np.stack([rec_r0.as_quat(), (step * rec_r0).as_quat()])
    rec_pos = np.zeros((2, 3))
    sim_r0 = Rotation.from_euler("x", -15, degrees=True)

    actions = prescribed_tcp_actions(
        rec_pos, rec_quat, sim_start_pos=np.zeros(3), sim_start_quat_xyzw=sim_r0.as_quat()
    )

    expected = step * sim_r0
    got = _quat_wxyz_to_rot(actions[0, 3:])
    assert (got * expected.inv()).magnitude() < 1e-9
    np.testing.assert_allclose(np.linalg.norm(actions[:, 3:], axis=1), 1.0, atol=1e-12)


def test_translation_delta_is_not_rotated_by_orientation_offset():
    """Position deltas are world-frame: a sim/real orientation offset must not rotate them."""
    rec_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
    rec_quat = np.tile(_IDENTITY_XYZW, (2, 1))
    sim_quat = Rotation.from_euler("z", 90, degrees=True).as_quat()

    actions = prescribed_tcp_actions(
        rec_pos, rec_quat, sim_start_pos=np.zeros(3), sim_start_quat_xyzw=sim_quat
    )

    np.testing.assert_allclose(actions[0, :3], [0.0, 0.02, 0.0], atol=1e-12)


def test_recorded_after_step_targets_same_frame_from_explicit_reference():
    """Sim-sim rows are recorded after stepping: action[k] → rec[k], reference = reset pose."""
    rec_pos = np.array([[0.01, 0.0, 0.0], [0.03, 0.0, 0.0]])
    rec_quat = np.tile(_IDENTITY_XYZW, (2, 1))

    actions = prescribed_tcp_actions(
        rec_pos,
        rec_quat,
        sim_start_pos=np.array([1.0, 0.0, 0.0]),
        sim_start_quat_xyzw=_IDENTITY_XYZW,
        reference_pos=np.zeros(3),
        reference_quat_xyzw=_IDENTITY_XYZW,
        recorded_after_step=True,
    )

    np.testing.assert_allclose(actions[:, 0], [1.01, 1.03], atol=1e-12)


def test_reference_defaults_to_first_recorded_frame():
    rec_pos = np.array([[0.5, 0.0, 0.0], [0.52, 0.0, 0.0]])
    rec_quat = np.tile(_IDENTITY_XYZW, (2, 1))
    kwargs = dict(sim_start_pos=np.zeros(3), sim_start_quat_xyzw=_IDENTITY_XYZW)

    default = prescribed_tcp_actions(rec_pos, rec_quat, **kwargs)
    explicit = prescribed_tcp_actions(
        rec_pos, rec_quat, reference_pos=rec_pos[0], reference_quat_xyzw=rec_quat[0], **kwargs
    )

    np.testing.assert_allclose(default, explicit)


def test_rejects_mismatched_or_empty_inputs():
    with pytest.raises(ValueError, match="tcp_pos"):
        prescribed_tcp_actions(
            np.zeros((3, 2)),
            np.tile(_IDENTITY_XYZW, (3, 1)),
            sim_start_pos=np.zeros(3),
            sim_start_quat_xyzw=_IDENTITY_XYZW,
        )
    with pytest.raises(ValueError, match="frame"):
        prescribed_tcp_actions(
            np.zeros((3, 3)),
            np.tile(_IDENTITY_XYZW, (2, 1)),
            sim_start_pos=np.zeros(3),
            sim_start_quat_xyzw=_IDENTITY_XYZW,
        )
    with pytest.raises(ValueError, match="at least one"):
        prescribed_tcp_actions(
            np.zeros((0, 3)),
            np.zeros((0, 4)),
            sim_start_pos=np.zeros(3),
            sim_start_quat_xyzw=_IDENTITY_XYZW,
        )
