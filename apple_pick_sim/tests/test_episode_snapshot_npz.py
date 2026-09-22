"""EpisodeStateSnapshot <-> npz: persist the settled episode baseline across processes."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.episode_state_snapshot import (
    EpisodeStateSnapshot,
    assign_snapshot_arrays,
    load_snapshot_npz,
    save_snapshot_npz,
    snapshot_arrays,
)


def _snap(scale: float = 1.0) -> EpisodeStateSnapshot:
    def a(shape, dtype=wp.float32):
        return wp.array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * scale, dtype=dtype, device="cpu")

    return EpisodeStateSnapshot(
        robot_body_q=wp.array(np.full((3, 7), scale, np.float32), dtype=wp.transform, device="cpu"),
        robot_body_qd=wp.array(np.full((3, 6), scale, np.float32), dtype=wp.spatial_vector, device="cpu"),
        robot_joint_q=a((9,)),
        robot_joint_qd=a((9,)),
        model_joint_q=a((9,)),
        model_joint_qd=a((9,)),
        cable_body_q_0=wp.array(np.full((4, 7), scale, np.float32), dtype=wp.transform, device="cpu"),
        cable_body_qd_0=wp.array(np.full((4, 6), scale, np.float32), dtype=wp.spatial_vector, device="cpu"),
        cable_body_q_1=wp.array(np.full((4, 7), scale, np.float32), dtype=wp.transform, device="cpu"),
        cable_body_qd_1=wp.array(np.full((4, 6), scale, np.float32), dtype=wp.spatial_vector, device="cpu"),
        vic_target_pos=wp.array(np.full((2, 3), scale, np.float32), dtype=wp.vec3, device="cpu"),
        joint_penalty_k=a((5,)),
    )


def test_npz_round_trip_restores_every_present_array_in_place(tmp_path):
    src, dst = _snap(2.0), _snap(0.0)
    path = tmp_path / "snap.npz"
    save_snapshot_npz(src, path, metadata={"world_ids": ["a", "b"], "note": "x"})
    arrays, meta = load_snapshot_npz(path)
    assert meta == {"world_ids": ["a", "b"], "note": "x"}
    assert set(arrays) == set(snapshot_arrays(src))
    assign_snapshot_arrays(dst, arrays)
    for name, arr in snapshot_arrays(dst).items():
        np.testing.assert_array_equal(arr, snapshot_arrays(src)[name], err_msg=name)
    assert dst.vic_target_rot is None  # absent fields stay absent


def test_assign_rejects_shape_mismatch(tmp_path):
    src = _snap(1.0)
    arrays = snapshot_arrays(src)
    arrays["cable_body_q_0"] = np.zeros((5, 7), np.float32)
    with pytest.raises(ValueError, match="cable_body_q_0"):
        assign_snapshot_arrays(_snap(0.0), arrays)


def test_assign_rejects_missing_required_array():
    arrays = snapshot_arrays(_snap(1.0))
    del arrays["robot_joint_q"]
    with pytest.raises(ValueError, match="robot_joint_q"):
        assign_snapshot_arrays(_snap(0.0), arrays)
