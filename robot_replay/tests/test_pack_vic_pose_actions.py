"""Tests for robot_replay/pack_vic_pose_actions.py (6D twist -> 19D pose+gains action)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robot_replay.pack_vic_pose_actions import pack_vic_pose_actions


def _write_minimal_dataset(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "episodes").mkdir(parents=True)
    (src / "manifest.json").write_text(
        json.dumps({"schema_version": "batched_sysid_v1", "collection": {"seed": 0}})
    )

    n = 3
    target_pose = np.tile(np.eye(4, dtype=np.float32).reshape(-1), (n, 1))
    target_pose[:, 3] = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # x translation per row varies below
    for i in range(n):
        target_pose[i, 3] = 0.1 + 0.01 * i
        target_pose[i, 7] = 0.2
        target_pose[i, 11] = 0.3

    table = pa.table(
        {
            "action": pa.array(
                [[float(j) for j in range(6)] for _ in range(n)],
                type=pa.list_(pa.float32(), 6),
            ),
            "target_pose_4x4": pa.array(
                target_pose.tolist(), type=pa.list_(pa.float32(), 16)
            ),
            "step_idx": pa.array(list(range(n)), type=pa.int64()),
        }
    )
    pq.write_table(table, src / "episodes" / "s00_d00.parquet")
    return src


def _write_tcp_pos_quat_dataset(tmp_path: Path) -> Path:
    src = tmp_path / "src_pos_quat"
    (src / "episodes").mkdir(parents=True)
    (src / "manifest.json").write_text(
        json.dumps({"schema_version": "batched_sysid_v1", "collection": {"seed": 0}})
    )

    n = 2
    tcp_pos = [[0.1, 0.2, 0.3], [0.11, 0.21, 0.31]]
    tcp_quat_xyzw = [
        [0.0, 0.0, 0.0, 1.0],
        [0.1, 0.2, 0.3, 0.4],
    ]
    table = pa.table(
        {
            "action": pa.array(
                [[float(j) for j in range(6)] for _ in range(n)],
                type=pa.list_(pa.float32(), 6),
            ),
            "tcp_pos": pa.array(tcp_pos, type=pa.list_(pa.float32(), 3)),
            "tcp_quat": pa.array(tcp_quat_xyzw, type=pa.list_(pa.float32(), 4)),
            "step_idx": pa.array(list(range(n)), type=pa.int64()),
        }
    )
    pq.write_table(table, src / "episodes" / "s00_d00.parquet")
    return src


def test_pack_vic_pose_actions_writes_19_wide_action(tmp_path):
    src = _write_minimal_dataset(tmp_path)
    dst = tmp_path / "dst"
    stats = pack_vic_pose_actions(
        src, dst, kp=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), kd=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    )
    assert stats["episodes"] == 1
    assert stats["frames"] == 3

    out = pq.read_table(dst / "episodes" / "s00_d00.parquet")
    actions = np.stack([out.column("action")[i].as_py() for i in range(out.num_rows)])
    assert actions.shape == (3, 19)
    np.testing.assert_allclose(actions[0, 0:3], [0.1, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(actions[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(actions[0, 7:13], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(actions[0, 13:19], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)

    manifest = json.loads((dst / "manifest.json").read_text())
    assert manifest["collection"]["action_dim"] == 19
    assert manifest["collection"]["action_layout"] == "vic_pose_v1"


def test_pack_vic_pose_actions_from_tcp_pos_quat(tmp_path):
    src = _write_tcp_pos_quat_dataset(tmp_path)
    dst = tmp_path / "dst_pos_quat"
    kp = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    kd = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    stats = pack_vic_pose_actions(src, dst, kp=kp, kd=kd)
    assert stats["episodes"] == 1
    assert stats["frames"] == 2

    out = pq.read_table(dst / "episodes" / "s00_d00.parquet")
    actions = np.stack([out.column("action")[i].as_py() for i in range(out.num_rows)])
    assert actions.shape == (2, 19)
    np.testing.assert_allclose(actions[0, 0:3], [0.1, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(actions[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(actions[1, 0:3], [0.11, 0.21, 0.31], atol=1e-5)
    np.testing.assert_allclose(actions[1, 3:7], [0.4, 0.1, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(actions[0, 7:13], kp, atol=1e-6)
    np.testing.assert_allclose(actions[0, 13:19], kd, atol=1e-6)

    manifest = json.loads((dst / "manifest.json").read_text())
    assert manifest["collection"]["action_dim"] == 19
    assert manifest["collection"]["action_layout"] == "vic_pose_v1"


def test_pack_vic_pose_actions_refuses_existing_dst_without_overwrite(tmp_path):
    src = _write_minimal_dataset(tmp_path)
    dst = tmp_path / "dst"
    pack_vic_pose_actions(src, dst, kp=(1.0,) * 6, kd=(1.0,) * 6)
    with pytest.raises(FileExistsError):
        pack_vic_pose_actions(src, dst, kp=(1.0,) * 6, kd=(1.0,) * 6)
