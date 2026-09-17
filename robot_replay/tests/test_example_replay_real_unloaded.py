"""CLI and helper tests for robot_replay/example_replay_real_unloaded.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_replay.example_replay_real_unloaded import (
    actions_from_episode_arrays,
    list_direction_indices,
    main,
    substeps_for_control_hz,
    write_tcp_wrench_html,
    write_unloaded_episode_parquet,
    write_unloaded_force_plots,
)


def test_substeps_for_control_hz_matches_runtime_config():
    assert substeps_for_control_hz(30.0) == 60
    assert substeps_for_control_hz(60.0) == 30
    with pytest.raises(ValueError, match="positive"):
        substeps_for_control_hz(0.0)


def test_actions_from_episode_arrays_slices_19d():
    arrays = {"action": np.arange(38, dtype=np.float32).reshape(2, 19)}
    got = actions_from_episode_arrays(arrays, action_dim=19)
    assert got.shape == (2, 19)
    np.testing.assert_array_equal(got[0], np.arange(19, dtype=np.float32))


def test_actions_from_episode_arrays_rejects_short_width():
    with pytest.raises(ValueError, match="action must have shape"):
        actions_from_episode_arrays({"action": np.zeros((3, 6), dtype=np.float32)})


def test_write_unloaded_episode_parquet_writes_ft_tcp_world(tmp_path: Path):
    ft = np.zeros((1, 6), dtype=np.float32)
    ft[0, 2] = -1.1 * 9.81
    pos = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    quat = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    path = tmp_path / "episodes" / "s00_d00.parquet"
    write_unloaded_episode_parquet(
        path,
        ft_tcp_world=ft,
        tcp_pos=pos,
        tcp_quat=quat,
        metadata={"direction_idx": 0},
    )
    table = pq.read_table(path)
    assert table.num_rows == 1
    assert "ft_tcp_world" in table.column_names
    assert "ft_wrist_tare" in table.column_names
    row = np.asarray(table.column("ft_tcp_world")[0].as_py(), dtype=np.float32)
    tare = np.asarray(table.column("ft_wrist_tare")[0].as_py(), dtype=np.float32)
    np.testing.assert_allclose(row[2], -1.1 * 9.81, atol=1e-5)
    np.testing.assert_array_equal(row, tare)
    meta = pq.read_metadata(path).schema.to_arrow_schema().metadata
    assert meta[b"schema_version"] == b"unloaded_tcp_wrench_v1"


def test_write_unloaded_force_plots_writes_force_torque_tcp(tmp_path: Path):
    pytest.importorskip("plotly")
    ft = np.zeros((4, 6), dtype=np.float32)
    ft[:, 0] = np.linspace(0.0, 1.0, 4)
    pos = np.linspace(0.0, 0.1, 12, dtype=np.float32).reshape(4, 3)
    real_ft = ft + 0.5
    real_pos = pos + 0.01
    written = write_unloaded_force_plots(
        tmp_path,
        direction_idx=3,
        ft_tcp_world=ft,
        tcp_pos=pos,
        control_hz=30.0,
        title_prefix="tare test",
        real_ft_wrist=real_ft,
        real_tcp_pos=real_pos,
        write_html=True,
        write_png=True,
    )
    names = {p.name for p in written}
    assert "dir_03_force.html" in names
    assert "dir_03_torque.html" in names
    assert "dir_03_tcp.html" in names
    assert "dir_03_force.png" in names
    html = (tmp_path / "dir_03_force.html").read_text(encoding="utf-8")
    assert "Fx" in html
    assert "tare" in html.lower() or "real" in html.lower()


def test_write_tcp_wrench_html(tmp_path: Path):
    pytest.importorskip("plotly")
    ft = np.zeros((4, 6), dtype=np.float32)
    ft[:, 0] = np.linspace(0.0, 1.0, 4)
    out = tmp_path / "dir_00_tcp.html"
    write_tcp_wrench_html(out, ft_tcp_world=ft, control_hz=30.0, title="test")
    assert (tmp_path / "dir_00_force.html").is_file()
    assert (tmp_path / "dir_00_torque.html").is_file()
    assert (tmp_path / "dir_00_tcp.html").is_file()


def test_apply_vic_pose_action_single_sets_target_and_mean_gains():
    import warp as wp

    from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import (
        Fr3EEImpedanceController,
    )
    from robot_replay.example_replay_real_unloaded import apply_vic_pose_action_single

    class _Scene:
        vic_target_tf = None
        vic_target_twist = None
        vic_gains = None

    ctrl = Fr3EEImpedanceController(tcp_body_index=0)
    scene = _Scene()
    action = np.zeros(19, dtype=np.float64)
    action[0:3] = [0.1, 0.2, 0.3]
    action[3:7] = [1.0, 0.0, 0.0, 0.0]  # wxyz identity
    action[7:13] = [100.0, 200.0, 300.0, 10.0, 20.0, 30.0]
    action[13:19] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    apply_vic_pose_action_single(ctrl, scene, action)
    p = wp.transform_get_translation(ctrl.target_tf)
    np.testing.assert_allclose([p[0], p[1], p[2]], [0.1, 0.2, 0.3], atol=1e-6)
    assert scene.vic_gains.linear_k == pytest.approx(200.0)
    assert scene.vic_gains.angular_k == pytest.approx(20.0)
    assert scene.vic_gains.linear_d == pytest.approx(2.0)
    assert scene.vic_gains.angular_d == pytest.approx(0.2)


def test_cli_requires_dataset():
    with pytest.raises(SystemExit):
        main([])


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_list_direction_indices_from_manifest(tmp_path: Path):
    from apple_pick_sim.system_id import BatchedSysIdDataset

    root = tmp_path / "ds"
    (root / "episodes").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "batched_sysid_v1",
                "collection": {},
                "episodes": [
                    {"structure_idx": 0, "direction_idx": 2, "filename": "episodes/s00_d02.parquet"},
                    {"structure_idx": 0, "direction_idx": 0, "filename": "episodes/s00_d00.parquet"},
                    {"structure_idx": 1, "direction_idx": 0, "filename": "episodes/s01_d00.parquet"},
                ],
            }
        ),
        encoding="utf-8",
    )
    ds = BatchedSysIdDataset(root)
    assert list_direction_indices(ds, 0) == [0, 2]


def test_run_unloaded_replay_writes_expected_column(tmp_path: Path, monkeypatch):
    """Stub the heavy sim path; ensure CLI writes ft_tcp_world shape (1, 6)."""
    from apple_pick_sim.system_id import BatchedSysIdDataset
    import robot_replay.example_replay_real_unloaded as mod

    root = tmp_path / "ds"
    (root / "episodes").mkdir(parents=True)
    # Minimal episode parquet with 19D action + metadata.
    import pyarrow as pa

    action = [0.0] * 19
    action[3] = 1.0  # quat w
    table = pa.table(
        {
            "action": pa.array([action], type=pa.list_(pa.float32(), 19)),
            "step_idx": pa.array([0], type=pa.int64()),
            "phase": pa.array([0], type=pa.int8()),
            "excitation_type": pa.array([0], type=pa.int8()),
            "excitation_direction": pa.array(
                [[1.0, 0.0, 0.0]], type=pa.list_(pa.float32(), 3)
            ),
            "ft_wrist": pa.array([[0.0] * 6], type=pa.list_(pa.float32(), 6)),
            "tcp_velocity": pa.array([[0.0] * 6], type=pa.list_(pa.float32(), 6)),
            "tcp_pos": pa.array([[0.1, 0.2, 0.3]], type=pa.list_(pa.float32(), 3)),
            "apple_pos": pa.array([[0.1, 0.2, 0.3]], type=pa.list_(pa.float32(), 3)),
            "tcp_quat": pa.array([[0.0, 0.0, 0.0, 1.0]], type=pa.list_(pa.float32(), 4)),
            "apple_quat": pa.array([[0.0, 0.0, 0.0, 1.0]], type=pa.list_(pa.float32(), 4)),
            "robot_joint_q": pa.array([[0.0] * 7], type=pa.list_(pa.float32(), 7)),
        }
    )
    meta = {
        b"fruiting_base_pos": b"[0.2, 0.2, 0.5]",
        b"initial_robot_joint_q": b"[0, -0.7, 0, -2.0, 0, 1.5, 0.7]",
        b"control_hz": b"30.0",
        b"action_layout": b'"vic_pose_v1"',
        b"action_dim": b"19",
    }
    schema = table.schema.with_metadata(meta)
    pq.write_table(table.cast(schema), root / "episodes" / "s00_d00.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "batched_sysid_v1",
                "collection": {
                    "action_layout": "vic_pose_v1",
                    "action_dim": 19,
                    "control_hz": 30.0,
                    "ranges_path": str(
                        _ROOT
                        / "apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json"
                    ),
                },
                "episodes": [
                    {
                        "structure_idx": 0,
                        "direction_idx": 0,
                        "filename": "episodes/s00_d00.parquet",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    sentinel = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32)

    def _fake_build(*_a, **_k):
        return object()

    def _fake_configure(scene, *, ranges):
        del scene, ranges
        return object()

    def _fake_replay(scene, ctrl, actions, *, control_hz, max_frames=0, sub_dt=1.0 / 1800.0):
        del scene, ctrl, control_hz, max_frames, sub_dt
        assert np.asarray(actions).shape == (1, 19)
        return {
            "ft_tcp_world": sentinel.copy(),
            "tcp_pos": np.zeros((1, 3), dtype=np.float32),
            "tcp_quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        }

    monkeypatch.setattr(mod, "build_unloaded_fr3_scene", _fake_build)
    monkeypatch.setattr(mod, "configure_unloaded_vic_pose", _fake_configure)
    monkeypatch.setattr(mod, "replay_unloaded_direction", _fake_replay)

    out = tmp_path / "out"
    rc = main(
        [
            "--dataset",
            str(root),
            "--out",
            str(out),
            "--direction-idx",
            "0",
            "--no-html",
        ]
    )
    assert rc == 0
    ep = out / "episodes" / "s00_d00.parquet"
    assert ep.is_file()
    table = pq.read_table(ep)
    ft = np.asarray(table.column("ft_tcp_world")[0].as_py(), dtype=np.float32)
    assert ft.shape == (6,)
    np.testing.assert_allclose(ft, sentinel[0])
    report = json.loads((out / "unloaded_report.json").read_text(encoding="utf-8"))
    assert report["episodes"][0]["frames"] == 1
    # Ensure dataset path still resolves.
    assert BatchedSysIdDataset(root).episode_entries()
