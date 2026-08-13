"""Tests for batched sys-ID MMD grid and recorded-action tensor helpers."""

from __future__ import annotations

import dataclasses
import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.tests.conftest import RANGES_FIXTURE


def _sample_params(seed: int = 0) -> FruitingSystemParams:
    ranges = fs.load_ranges(RANGES_FIXTURE)
    return fs.sample_params(ranges, seed=seed)


def test_bend_stiffness_candidates_are_cartesian_product_in_grid_order():
    candidates = list(
        grid.iter_bend_stiffness_candidates(
            primary_values=(1.0, 2.0),
            secondary_values=(10.0,),
            spur_values=(100.0,),
            stem_values=(1000.0, 2000.0),
        )
    )

    assert candidates == [
        grid.BendStiffnessCandidate(
            primary=1.0,
            secondary=10.0,
            spur=100.0,
            stem=1000.0,
        ),
        grid.BendStiffnessCandidate(
            primary=1.0,
            secondary=10.0,
            spur=100.0,
            stem=2000.0,
        ),
        grid.BendStiffnessCandidate(
            primary=2.0,
            secondary=10.0,
            spur=100.0,
            stem=1000.0,
        ),
        grid.BendStiffnessCandidate(
            primary=2.0,
            secondary=10.0,
            spur=100.0,
            stem=2000.0,
        ),
    ]


def test_bend_stiffness_candidate_maps_to_segment_override_dict():
    candidate = grid.BendStiffnessCandidate(
        primary=1.0,
        secondary=2.0,
        spur=3.0,
        stem=4.0,
    )

    assert candidate.to_overrides() == {
        "primary": {"bend_stiffness": 1.0},
        "secondary": {"bend_stiffness": 2.0},
        "spur": {"bend_stiffness": 3.0},
        "stem": {"bend_stiffness": 4.0},
    }


def test_bend_stiffness_candidate_apply_to_sets_enabled_segments():
    base = _sample_params(seed=0)
    candidate = grid.BendStiffnessCandidate(
        primary=11.0,
        secondary=22.0,
        spur=33.0,
        stem=44.0,
    )

    out = candidate.apply_to(base)

    assert out.primary is not None
    assert out.primary.bend_stiffness == pytest.approx(11.0)
    assert out.secondary is not None
    assert out.secondary.bend_stiffness == pytest.approx(22.0)
    assert out.spur is not None
    assert out.spur.bend_stiffness == pytest.approx(33.0)
    assert out.stem is not None
    assert out.stem.bend_stiffness == pytest.approx(44.0)
    assert base.primary is not None
    assert base.primary.bend_stiffness != pytest.approx(11.0)
    for segment, target in (
        ("primary", 11.0),
        ("secondary", 22.0),
        ("spur", 33.0),
        ("stem", 44.0),
    ):
        base_rod = getattr(base, segment)
        out_rod = getattr(out, segment)
        assert base_rod is not None and out_rod is not None
        assert out_rod.damping_ratio == pytest.approx(base_rod.damping_ratio)
        ratio = math.sqrt(target / base_rod.bend_stiffness)
        assert out_rod.bend_damping == pytest.approx(base_rod.bend_damping * ratio)


def test_bend_stiffness_candidate_apply_to_skips_disabled_segments():
    base = _sample_params(seed=0)
    base = dataclasses.replace(fs.copy_fruiting_params(base), spur=None)
    candidate = grid.BendStiffnessCandidate(
        primary=11.0,
        secondary=22.0,
        spur=33.0,
        stem=44.0,
    )

    out = candidate.apply_to(base)

    assert out.spur is None
    assert out.stem is not None
    assert out.stem.bend_stiffness == pytest.approx(44.0)


def test_ensure_gt_candidate_in_grid_unchanged_when_present():
    gt = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)
    candidates = [
        grid.BendStiffnessCandidate(10.0, 20.0, 30.0, 40.0),
        gt,
    ]

    out = grid.ensure_gt_candidate_in_grid(candidates, gt)

    assert out == candidates
    assert out is not candidates


def test_ensure_gt_candidate_in_grid_appends_when_missing():
    gt = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)
    candidates = [
        grid.BendStiffnessCandidate(10.0, 20.0, 30.0, 40.0),
        grid.BendStiffnessCandidate(11.0, 21.0, 31.0, 41.0),
    ]

    out = grid.ensure_gt_candidate_in_grid(candidates, gt)

    assert out == candidates + [gt]
    assert out is not candidates


def test_ensure_gt_candidate_in_grid_matches_within_tolerance():
    gt = grid.BendStiffnessCandidate(1.0, 0.0, 3.0, 4.0)
    near_gt = grid.BendStiffnessCandidate(1.0, 1e-12, 3.0, 4.0)
    candidates = [
        grid.BendStiffnessCandidate(10.0, 20.0, 30.0, 40.0),
        near_gt,
    ]

    out = grid.ensure_gt_candidate_in_grid(candidates, gt)

    assert out == candidates
    assert out is not candidates


def test_ensure_gt_candidate_in_grid_returns_singleton_when_empty():
    gt = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)

    assert grid.ensure_gt_candidate_in_grid([], gt) == [gt]


def test_gt_bend_stiffness_candidate_from_structure_reads_true_stiffness(monkeypatch):
    base = _sample_params(seed=1)
    assert base.primary is not None
    assert base.secondary is not None
    assert base.spur is not None
    assert base.stem is not None
    expected = grid.BendStiffnessCandidate(
        primary=base.primary.bend_stiffness,
        secondary=base.secondary.bend_stiffness,
        spur=base.spur.bend_stiffness,
        stem=base.stem.bend_stiffness,
    )

    dataset = MagicMock()
    monkeypatch.setattr(
        grid,
        "true_params_for_structure",
        lambda _dataset, _idx: base,
    )

    got = grid.gt_bend_stiffness_candidate_from_structure(dataset, structure_idx=0)

    assert got == expected


def test_base_params_for_replay_defaults_to_oracle(monkeypatch):
    oracle = _sample_params(seed=1)
    inferred = _sample_params(seed=2)
    dataset = MagicMock()
    monkeypatch.setattr(grid, "true_params_for_structure", lambda *_a, **_k: oracle)
    monkeypatch.setattr(grid, "infer_base_params_for_structure", lambda *_a, **_k: inferred)

    assert grid.base_params_for_replay(dataset, 0) is oracle
    assert grid.base_params_for_replay(dataset, 0, use_oracle_params=True) is oracle
    assert grid.base_params_for_replay(dataset, 0, use_oracle_params=False) is inferred


def _episode_with_pre_weld(*, n_trajectory_frames: int, direction_idx: int = 0) -> dict:
    """Synthetic episode: one pre_weld row + ``n_trajectory_frames`` real steps."""
    n_total = n_trajectory_frames + 1
    action = np.zeros((n_total, 6), dtype=np.float32)
    action[0] = 0.0
    action[1:, 0] = np.arange(n_trajectory_frames, dtype=np.float32) + float(direction_idx)
    return {
        "step_idx": np.concatenate(
            [np.array([-1], dtype=np.int32), np.arange(n_trajectory_frames, dtype=np.int32)]
        ),
        "phase": np.concatenate(
            [np.array([-1], dtype=np.int8), np.zeros(n_trajectory_frames, dtype=np.int8)]
        ),
        "action": action,
        "ft_wrist": np.zeros((n_total, 6), dtype=np.float32),
    }


def _make_mock_dataset(*, num_directions: int, n_frames: int) -> MagicMock:
    dataset = MagicMock()

    def load_episode_obs_arrays(structure_idx: int, direction_idx: int) -> dict:
        del structure_idx
        action = np.full(
            (n_frames, 6),
            fill_value=float(direction_idx) + 0.1,
            dtype=np.float32,
        )
        action[:, 0] = np.arange(n_frames, dtype=np.float32) + direction_idx
        return {"action": action}

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays
    return dataset


def _make_mock_dataset_with_pre_weld(*, num_directions: int, n_trajectory_frames: int) -> MagicMock:
    dataset = MagicMock()

    def load_episode_obs_arrays(structure_idx: int, direction_idx: int) -> dict:
        del structure_idx
        return _episode_with_pre_weld(
            n_trajectory_frames=n_trajectory_frames,
            direction_idx=direction_idx,
        )

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays
    return dataset


def test_strip_pre_weld_rows_removes_leading_pre_weld_snapshot():
    raw = _episode_with_pre_weld(n_trajectory_frames=4, direction_idx=2)
    stripped = grid.strip_pre_weld_rows(raw)

    assert stripped["action"].shape == (4, 6)
    assert stripped["step_idx"].tolist() == [0, 1, 2, 3]
    assert stripped["phase"].tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(stripped["action"][:, 0], np.arange(4, dtype=np.float32) + 2.0)


def test_build_recorded_actions_tensor_excludes_pre_weld_row():
    num_directions = 2
    num_candidates = 2
    n_trajectory_frames = 5
    dataset = _make_mock_dataset_with_pre_weld(
        num_directions=num_directions,
        n_trajectory_frames=n_trajectory_frames,
    )

    tensor = grid.build_recorded_actions_tensor(
        dataset,
        structure_idx=0,
        num_directions=num_directions,
        num_candidates=num_candidates,
    )

    assert tensor.shape == (num_candidates * num_directions, n_trajectory_frames, 6)
    assert int(tensor[0, 0, 0]) == 0
    assert int(tensor[1, 0, 0]) == 1


def test_load_recorded_episodes_for_structure_excludes_pre_weld_row():
    dataset = _make_mock_dataset_with_pre_weld(num_directions=2, n_trajectory_frames=3)

    episodes = grid.load_recorded_episodes_for_structure(
        dataset,
        structure_idx=0,
        num_directions=2,
    )

    assert len(episodes) == 2
    for direction_idx, episode in enumerate(episodes):
        assert episode["action"].shape == (3, 6)
        assert int(episode["step_idx"][0]) == 0
        np.testing.assert_array_equal(
            episode["dir_idx"],
            np.full(3, direction_idx, dtype=np.int32),
        )


def test_recorded_metadata_by_env_excludes_pre_weld_row():
    dataset = _make_mock_dataset_with_pre_weld(num_directions=2, n_trajectory_frames=3)

    got = grid.recorded_metadata_by_env(
        dataset,
        structure_idx=0,
        num_directions=2,
        num_candidates=2,
    )

    assert len(got) == 4
    for recorded in got:
        assert recorded["action"].shape == (3, 6)
        assert int(recorded["step_idx"][0]) == 0


def test_trajectory_mse_frame_zero_matches_first_real_trajectory_step():
    recorded = _episode_with_pre_weld(n_trajectory_frames=4)
    recorded["phase"][1:] = np.array([1, 1, 1, 1], dtype=np.int8)
    recorded["ft_wrist"] = np.tile(np.arange(6, dtype=np.float32), (5, 1))
    recorded["tcp_pos"] = np.tile(np.arange(3, dtype=np.float32), (5, 1))
    recorded["apple_pos"] = recorded["tcp_pos"].copy()

    recorded = grid.strip_pre_weld_rows(recorded)
    replay = {
        "ft_wrist": recorded["ft_wrist"].copy(),
        "tcp_pos": recorded["tcp_pos"].copy(),
        "apple_pos": recorded["apple_pos"].copy(),
    }

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)

    assert out["n_frames"] == 4.0
    assert out["n_used_frames"] == 4.0
    assert out["ft_wrist_mse"] == pytest.approx(0.0)


def test_build_recorded_actions_tensor_shape_and_candidate_broadcast():
    num_directions = 3
    num_candidates = 2
    n_frames = 5
    dataset = _make_mock_dataset(num_directions=num_directions, n_frames=n_frames)

    tensor = grid.build_recorded_actions_tensor(
        dataset,
        structure_idx=0,
        num_directions=num_directions,
        num_candidates=num_candidates,
    )

    assert tensor.shape == (num_candidates * num_directions, n_frames, 6)
    assert tensor.dtype == np.float32

    for candidate_idx in range(num_candidates):
        for direction_idx in range(num_directions):
            env_idx = candidate_idx * num_directions + direction_idx
            expected = dataset.load_episode_obs_arrays(0, direction_idx)["action"]
            np.testing.assert_array_equal(tensor[env_idx], expected)

    for direction_idx in range(num_directions):
        env_a = 0 * num_directions + direction_idx
        env_b = 1 * num_directions + direction_idx
        np.testing.assert_array_equal(tensor[env_a], tensor[env_b])


def test_build_recorded_actions_tensor_rejects_mismatched_frame_counts():
    dataset = MagicMock()

    def load_episode_obs_arrays(structure_idx: int, direction_idx: int) -> dict:
        del structure_idx
        n_frames = 4 if direction_idx == 0 else 5
        return {"action": np.zeros((n_frames, 6), dtype=np.float32)}

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays

    with pytest.raises(ValueError, match="n_frames"):
        grid.build_recorded_actions_tensor(
            dataset,
            structure_idx=0,
            num_directions=2,
            num_candidates=1,
        )


def test_actions_tensor_from_recorded_frame_shape_and_device():
    recorded = np.arange(24, dtype=np.float32).reshape(2, 2, 6)
    device = torch.device("cpu")

    out = grid.actions_tensor_from_recorded_frame(
        recorded,
        frame_idx=1,
        device=device,
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 6)
    assert out.device.type == "cpu"
    assert out.dtype == torch.float32
    np.testing.assert_array_equal(out.numpy(), recorded[:, 1, :])


def test_replay_youngs_modulus_candidates_preserves_sparse_direction_ids(monkeypatch):
    from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes

    base = _sample_params(seed=0)
    candidate_0 = cmaes.YoungsModulusCandidate(primary=1.0e8, spur=2.0e7, stem=3.0e7)
    candidate_1 = cmaes.YoungsModulusCandidate(primary=2.0e8, spur=3.0e7, stem=4.0e7)
    n_trajectory_frames = 2
    build_calls: list[dict] = []

    def mock_build_env_fn(**kwargs):
        build_calls.append(kwargs)
        num_envs = int(kwargs["num_envs"])
        env = MagicMock()
        env.device = torch.device("cpu")

        def _batched_obs() -> dict:
            woody_part_info: dict[str, dict[str, torch.Tensor]] = {}
            for name in ("joint_a", "joint_b"):
                woody_part_info[name] = {
                    "anchors_pos": torch.zeros(num_envs, 6, dtype=torch.float32),
                    "anchor_force": torch.zeros(num_envs, 6, dtype=torch.float32),
                }
            return {
                "woody_part_info": woody_part_info,
                "apple_pos": torch.zeros(num_envs, 3, dtype=torch.float32),
                "tcp_force": torch.zeros(num_envs, 6, dtype=torch.float32),
                "tcp_velocity": torch.zeros(num_envs, 6, dtype=torch.float32),
                "ft_wrist": torch.zeros(num_envs, 6, dtype=torch.float32),
                "raw_ft_wrist": torch.zeros(num_envs, 6, dtype=torch.float32),
                "tcp_pos": torch.zeros(num_envs, 3, dtype=torch.float32),
                "tcp_quat": torch.zeros(num_envs, 4, dtype=torch.float32),
                "apple_quat": torch.zeros(num_envs, 4, dtype=torch.float32),
                "robot_joint_q": torch.zeros(num_envs, 7, dtype=torch.float32),
                "excitation_type": torch.zeros(num_envs, dtype=torch.long),
                "excitation_f_inst": torch.zeros(num_envs, dtype=torch.float32),
                "excitation_direction": torch.zeros(num_envs, 3, dtype=torch.float32),
            }

        env._last_obs = _batched_obs()
        env.reset = MagicMock()
        env.step = MagicMock(side_effect=lambda _actions: setattr(env, "_last_obs", _batched_obs()))
        env.close = MagicMock()
        env.sysid_numpy_obs = MagicMock(
            return_value=_sysid_numpy_obs_for_frame(
                frame_idx=0,
                junction_names=["joint_a", "joint_b"],
            )
        )
        return env

    dataset = MagicMock()
    dataset.manifest = {"collection": {"action_layout": "vic_pose_v1", "action_dim": 19}}

    def load_episode_obs_arrays(structure_idx: int, direction_idx: int) -> dict:
        del structure_idx
        return _recorded_arrays_for_replay(
            n_frames=n_trajectory_frames,
            direction_idx=direction_idx,
        )

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays
    dataset.load_episode_metadata.return_value = {
        "junction_names": ["joint_a", "joint_b"],
        "pull_direction": [1.0, 0.0, 0.0],
    }

    monkeypatch.setattr(grid, "base_params_for_replay", lambda *_a, **_k: base)
    monkeypatch.setattr(grid, "initialize_batched_env_from_dataset", lambda *_a, **_k: None)
    monkeypatch.setattr(
        grid,
        "ik_bootstrap_unstable_mask",
        lambda _env, num_envs: torch.zeros(int(num_envs), dtype=torch.bool),
    )
    sim_gripper = MagicMock(name="sim_gripper")
    real_gripper = MagicMock(name="real_gripper")
    monkeypatch.setattr(
        grid,
        "gripper_proxy_from_episode_metadata",
        lambda _meta: sim_gripper,
    )
    monkeypatch.setattr(
        grid,
        "gripper_proxy_for_real_batched_replay",
        lambda _meta: real_gripper,
    )

    collectors = grid.replay_batched_sysid_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=[candidate_0, candidate_1],
        num_directions=2,
        seed=0,
        build_env_fn=mock_build_env_fn,
        direction_indices=[0, 2],
    )

    assert build_calls[0]["gripper"] is real_gripper
    assert build_calls[0]["per_env_params"] == [
        candidate_0.apply_to(base),
        candidate_0.apply_to(base),
        candidate_1.apply_to(base),
        candidate_1.apply_to(base),
    ]
    assert collectors.to_arrays(0)["dir_idx"][0] == 0
    assert collectors.to_arrays(1)["dir_idx"][0] == 2


def test_replay_candidates_for_structure_threads_on_step(monkeypatch):
    calls: list[tuple[int, int]] = []
    forwarded_action_dims: list[int] = []

    def fake_replay_batched_sysid_structure(
        *, candidates, on_step=None, action_dim=6, **_kwargs
    ):
        assert on_step is not None
        for i in range(3):
            assert on_step(frame_idx=i, env=MagicMock())
        calls.append((len(candidates), 1))
        forwarded_action_dims.append(int(action_dim))
        recorded_by_env = [
            _recorded_arrays_for_replay(n_frames=4, direction_idx=0),
        ]
        return grid.BatchedSysIdReplayCollectors(num_envs=1, recorded_by_env=recorded_by_env)

    monkeypatch.setattr(grid, "chunk_candidates", lambda candidates, **_kwargs: [list(candidates)])
    monkeypatch.setattr(grid, "replay_batched_sysid_structure", fake_replay_batched_sysid_structure)

    dataset = MagicMock()

    def on_step(*, frame_idx: int, env) -> bool:
        del env
        return frame_idx < 10

    out = grid.replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=[grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)],
        num_directions=1,
        seed=None,
        build_env_fn=MagicMock(),
        max_envs_per_batch=0,
        on_step=on_step,
        action_dim=19,
    )

    assert isinstance(out, grid.BatchedSysIdReplayCollectors)
    assert calls, "expected replay_batched_sysid_structure to be called"
    assert forwarded_action_dims == [19]


def _recorded_arrays_for_replay(*, n_frames: int, direction_idx: int = 0) -> dict:
    junction_names = ["joint_a", "joint_b"]
    base = np.arange(n_frames, dtype=np.float32).reshape(n_frames, 1)
    return {
        "action": np.full((n_frames, 6), float(direction_idx), dtype=np.float32),
        "phase": np.full(n_frames, direction_idx, dtype=np.int8),
        "dir_idx": np.full(n_frames, direction_idx, dtype=np.int32),
        "excitation_type": np.zeros(n_frames, dtype=np.int8),
        "excitation_direction": np.tile(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (n_frames, 1),
        ),
        "junction_names": junction_names,
    }


def _sysid_numpy_obs_for_frame(*, frame_idx: int, junction_names: list[str]) -> dict:
    return {
        "ft_wrist": np.full(6, 100.0 + frame_idx, dtype=np.float32),
        "tcp_velocity": np.full(6, 200.0 + frame_idx, dtype=np.float32),
        "tcp_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32) + frame_idx,
        "apple_pos": np.array([4.0, 5.0, 6.0], dtype=np.float32) + frame_idx,
        "woody_part_start_pos": {
            name: np.array([10.0, 11.0, 12.0], dtype=np.float32) + frame_idx
            for name in junction_names
        },
        "woody_part_end_pos": {
            name: np.array([20.0, 21.0, 22.0], dtype=np.float32) + frame_idx
            for name in junction_names
        },
    }


def test_recorded_metadata_by_env_broadcasts_direction_across_candidates():
    num_directions = 2
    num_candidates = 3
    dataset = MagicMock()
    direction_arrays = {
        0: _recorded_arrays_for_replay(n_frames=4, direction_idx=0),
        1: _recorded_arrays_for_replay(n_frames=4, direction_idx=1),
    }

    def load_episode_obs_arrays(structure_idx: int, direction_idx: int) -> dict:
        del structure_idx
        return direction_arrays[direction_idx]

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays

    got = grid.recorded_metadata_by_env(
        dataset,
        structure_idx=7,
        num_directions=num_directions,
        num_candidates=num_candidates,
    )

    assert len(got) == num_candidates * num_directions
    for env_idx, recorded in enumerate(got):
        direction_idx = env_idx % num_directions
        expected = direction_arrays[direction_idx]
        assert recorded["action"] is expected["action"]
        np.testing.assert_array_equal(recorded["dir_idx"], expected["dir_idx"])
        dataset.load_episode_obs_arrays.assert_any_call(7, direction_idx)


def _batched_torch_obs_for_replay(*, num_envs: int, frame_idx: int, junction_names: list[str]) -> dict:
    woody_part_info: dict[str, dict[str, torch.Tensor]] = {}
    for name in junction_names:
        anchors = torch.zeros(num_envs, 6, dtype=torch.float32)
        for env_idx in range(num_envs):
            anchors[env_idx] = torch.tensor(
                [10.0 + frame_idx, 11.0 + frame_idx, 12.0 + frame_idx,
                 20.0 + frame_idx, 21.0 + frame_idx, 22.0 + frame_idx],
                dtype=torch.float32,
            ) + float(env_idx)
        woody_part_info[name] = {
            "anchors_pos": anchors,
            "anchor_force": torch.zeros(num_envs, 6, dtype=torch.float32),
        }

    def _vec(cols: int, base: float) -> torch.Tensor:
        return torch.full((num_envs, cols), base + float(frame_idx), dtype=torch.float32)

    return {
        "woody_part_info": woody_part_info,
        "apple_pos": _vec(3, 4.0),
        "tcp_force": _vec(6, 0.0),
        "tcp_velocity": _vec(6, 200.0),
        "ft_wrist": _vec(6, 100.0),
        "raw_ft_wrist": _vec(6, 0.0),
        "tcp_pos": _vec(3, 1.0),
        "tcp_quat": _vec(4, 0.0),
        "apple_quat": _vec(4, 0.0),
        "robot_joint_q": _vec(7, 0.0),
        "excitation_type": torch.zeros(num_envs, dtype=torch.long),
        "excitation_f_inst": torch.zeros(num_envs, dtype=torch.float32),
        "excitation_direction": torch.zeros(num_envs, 3, dtype=torch.float32),
    }


def test_batched_sysid_replay_collectors_record_all_envs_step_matches_record_step():
    from apple_pick_gym.batched_envs.obs_torch import sysid_numpy_obs_from_batched

    num_envs = 2
    n_frames = 3
    recorded_by_env = [
        _recorded_arrays_for_replay(n_frames=n_frames, direction_idx=env_idx)
        for env_idx in range(num_envs)
    ]
    per_env_collectors = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)
    batched_collectors = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

    junction_names = recorded_by_env[0]["junction_names"]
    env = MagicMock()

    for frame_idx in range(2):
        batched_obs = _batched_torch_obs_for_replay(
            num_envs=num_envs,
            frame_idx=frame_idx,
            junction_names=junction_names,
        )
        env._last_obs = batched_obs
        batched_collectors.record_all_envs_step(env, frame_idx=frame_idx)

        for env_idx in range(num_envs):
            env.sysid_numpy_obs.return_value = sysid_numpy_obs_from_batched(
                batched_obs,
                junction_names,
                env_idx=env_idx,
            )
            per_env_collectors.record_step(env, env_idx=env_idx, frame_idx=frame_idx)

    for env_idx in range(num_envs):
        batched_arrays = batched_collectors.to_arrays(env_idx)
        per_env_arrays = per_env_collectors.to_arrays(env_idx)
        for key in ("action", "ft_wrist", "tcp_pos", "apple_pos", "phase", "dir_idx"):
            np.testing.assert_allclose(batched_arrays[key], per_env_arrays[key])
        for name in junction_names:
            np.testing.assert_allclose(
                batched_arrays["woody_part_start_pos"][name],
                per_env_arrays["woody_part_start_pos"][name],
            )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batched_sysid_replay_collectors_record_step_accepts_torch_unstable_mask(device):
    from apple_pick_gym.batched_envs.obs_torch import sysid_numpy_obs_from_batched

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    num_envs = 2
    n_frames = 3
    recorded_by_env = [
        _recorded_arrays_for_replay(n_frames=n_frames, direction_idx=env_idx)
        for env_idx in range(num_envs)
    ]
    per_env_collectors = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)
    batched_collectors = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

    junction_names = recorded_by_env[0]["junction_names"]
    env = MagicMock()
    unstable = torch.tensor([False, True], dtype=torch.bool, device=device)

    batched_obs = _batched_torch_obs_for_replay(
        num_envs=num_envs,
        frame_idx=0,
        junction_names=junction_names,
    )
    env._last_obs = batched_obs
    batched_collectors.record_all_envs_step(env, frame_idx=0, unstable=unstable)

    for env_idx in range(num_envs):
        env.sysid_numpy_obs.return_value = sysid_numpy_obs_from_batched(
            batched_obs,
            junction_names,
            env_idx=env_idx,
        )
        per_env_collectors.record_step(
            env, env_idx=env_idx, frame_idx=0, unstable=unstable
        )

    assert batched_collectors.to_arrays(0)["stable"].tolist() == [True]
    assert batched_collectors.to_arrays(1)["stable"].tolist() == [False]
    assert per_env_collectors.to_arrays(0)["stable"].tolist() == [True]
    assert per_env_collectors.to_arrays(1)["stable"].tolist() == [False]


def test_batched_sysid_replay_collectors_record_step_and_to_arrays():
    num_envs = 2
    n_frames = 3
    recorded_by_env = [
        _recorded_arrays_for_replay(n_frames=n_frames, direction_idx=env_idx)
        for env_idx in range(num_envs)
    ]
    collectors = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

    env = MagicMock()
    junction_names = recorded_by_env[0]["junction_names"]

    for frame_idx in range(2):
        for env_idx in range(num_envs):
            env.sysid_numpy_obs.return_value = _sysid_numpy_obs_for_frame(
                frame_idx=frame_idx,
                junction_names=junction_names,
            )
            collectors.record_step(env, env_idx=env_idx, frame_idx=frame_idx)

    assert collectors.n_rows(0) == 2
    assert collectors.n_rows(1) == 2

    arrays = collectors.to_arrays(0)
    assert arrays["action"].shape == (2, 6)
    assert arrays["ft_wrist"].shape == (2, 6)
    assert arrays["tcp_pos"].shape == (2, 3)
    assert arrays["apple_pos"].shape == (2, 3)
    assert set(arrays["woody_part_start_pos"]) == set(junction_names)
    assert arrays["woody_part_start_pos"]["joint_a"].shape == (2, 3)
    np.testing.assert_allclose(arrays["ft_wrist"][1], 101.0)


def test_batched_sysid_replay_collectors_merge_concatenates_per_env():
    num_envs = 2
    n_frames = 4
    recorded_by_env = [
        _recorded_arrays_for_replay(n_frames=n_frames, direction_idx=env_idx)
        for env_idx in range(num_envs)
    ]
    left = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)
    right = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

    env = MagicMock()
    junction_names = recorded_by_env[0]["junction_names"]

    for frame_idx in (0, 1):
        for env_idx in range(num_envs):
            env.sysid_numpy_obs.return_value = _sysid_numpy_obs_for_frame(
                frame_idx=frame_idx,
                junction_names=junction_names,
            )
            left.record_step(env, env_idx=env_idx, frame_idx=frame_idx)

    for frame_idx in (2, 3):
        for env_idx in range(num_envs):
            env.sysid_numpy_obs.return_value = _sysid_numpy_obs_for_frame(
                frame_idx=frame_idx,
                junction_names=junction_names,
            )
            right.record_step(env, env_idx=env_idx, frame_idx=frame_idx)

    merged = left.merge(right)

    assert merged.n_rows(0) == 4
    assert merged.n_rows(1) == 4

    arrays = merged.to_arrays(0)
    assert arrays["action"].shape == (4, 6)
    assert arrays["ft_wrist"].shape == (4, 6)
    np.testing.assert_allclose(arrays["ft_wrist"][0], 100.0)
    np.testing.assert_allclose(arrays["ft_wrist"][3], 103.0)


def test_batched_sysid_replay_collectors_concat_envs_appends_env_slots():
    num_envs = 2
    n_frames = 3
    recorded_by_env = [
        _recorded_arrays_for_replay(n_frames=n_frames, direction_idx=env_idx)
        for env_idx in range(num_envs)
    ]
    left = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)
    right = grid.BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

    combined = left.concat_envs(right)

    # Record one frame in each env slot so to_arrays() is defined.
    env = MagicMock()
    junction_names = recorded_by_env[0]["junction_names"]
    for env_idx in range(4):
        env.sysid_numpy_obs.return_value = _sysid_numpy_obs_for_frame(
            frame_idx=0,
            junction_names=junction_names,
        )
        combined.record_step(env, env_idx=env_idx, frame_idx=0)

    assert combined.n_rows(0) == 1
    assert combined.n_rows(3) == 1
    assert combined.to_arrays(0)["junction_names"] == recorded_by_env[0]["junction_names"]
    assert combined.to_arrays(3)["junction_names"] == recorded_by_env[1]["junction_names"]


def _arrays_for_steps(*, steps: int, junction_names: list[str] | None = None, shift: float = 0.0) -> dict:
    """Synthetic hold-phase episode arrays (same contract as test_mmd_features)."""
    junction_names = junction_names or ["joint_b", "joint_a"]
    base = np.arange(steps, dtype=np.float32).reshape(steps, 1) + float(shift)
    woody_start = {
        "joint_a": np.hstack([base + 100.0, base + 101.0, base + 102.0]).astype(np.float32),
        "joint_b": np.hstack([base + 200.0, base + 201.0, base + 202.0]).astype(np.float32),
    }
    woody_end = {
        "joint_a": np.hstack([base + 300.0, base + 301.0, base + 302.0]).astype(np.float32),
        "joint_b": np.hstack([base + 400.0, base + 401.0, base + 402.0]).astype(np.float32),
    }
    return {
        "ft_wrist": np.hstack([base + i for i in range(6)]).astype(np.float32),
        "tcp_velocity": np.hstack([base + 10.0 + i for i in range(6)]).astype(np.float32),
        "action": np.hstack([base + 20.0 + i for i in range(6)]).astype(np.float32),
        "tcp_pos": np.hstack([base + 30.0 + i for i in range(3)]).astype(np.float32),
        "apple_pos": np.hstack([base + 40.0 + i for i in range(3)]).astype(np.float32),
        "woody_part_start_pos": woody_start,
        "woody_part_end_pos": woody_end,
        "excitation_direction": np.tile(
            np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (steps, 1)
        ),
        "phase": np.ones(steps, dtype=np.int8),
        "excitation_type": np.zeros(steps, dtype=np.int8),
        "dir_idx": np.zeros(steps, dtype=np.int32),
        "junction_names": junction_names,
    }


def test_load_recorded_episodes_for_structure_synthesizes_dir_idx():
    num_directions = 2
    n_frames = 5
    dataset = MagicMock()

    def load_episode_obs_arrays(structure_idx: int, direction_idx: int) -> dict:
        del structure_idx
        return {
            "action": np.zeros((n_frames, 6), dtype=np.float32),
            "phase": np.zeros(n_frames, dtype=np.int8),
        }

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays

    episodes = grid.load_recorded_episodes_for_structure(
        dataset,
        structure_idx=3,
        num_directions=num_directions,
    )

    assert len(episodes) == num_directions
    for direction_idx, episode in enumerate(episodes):
        np.testing.assert_array_equal(
            episode["dir_idx"],
            np.full(n_frames, direction_idx, dtype=np.int32),
        )


def test_prepare_gt_mmd_context_from_synthetic_arrays():
    episodes = [_arrays_for_steps(steps=8)]

    context = grid.prepare_gt_mmd_context(episodes)

    assert len(context) == 1
    direction = 0
    assert direction in context
    entry = context[direction]
    assert isinstance(entry, grid.MmdDirectionContext)
    assert entry.gt_norm.ndim == 2
    assert entry.gt_norm.shape[0] >= 1
    assert entry.bandwidth > 0.0


def test_score_candidate_mmd_identical_features_near_zero():
    episodes = [_arrays_for_steps(steps=8)]
    gt_context = grid.prepare_gt_mmd_context(episodes)
    candidate = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)

    result = grid.score_candidate_mmd(
        candidate_index=0,
        candidate=candidate,
        gt_context=gt_context,
        replay_observations=episodes,
    )

    assert result.aggregate_mmd2 == pytest.approx(0.0, abs=1e-6)
    assert result.candidate_index == 0
    assert result.stiffnesses == {
        "primary": 1.0,
        "secondary": 2.0,
        "spur": 3.0,
        "stem": 4.0,
    }


def test_score_candidate_mmd_shifted_features_higher():
    episodes = [_arrays_for_steps(steps=8)]
    gt_context = grid.prepare_gt_mmd_context(episodes)
    candidate = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)

    identical = grid.score_candidate_mmd(
        candidate_index=0,
        candidate=candidate,
        gt_context=gt_context,
        replay_observations=episodes,
    )
    shifted = grid.score_candidate_mmd(
        candidate_index=1,
        candidate=candidate,
        gt_context=gt_context,
        replay_observations=[_arrays_for_steps(steps=8, shift=10.0)],
    )

    assert identical.aggregate_mmd2 < shifted.aggregate_mmd2
    assert identical.missing_directions == ()
    assert shifted.missing_directions == ()


def test_score_candidate_mmd_reports_missing_directions():
    ep0 = _arrays_for_steps(steps=8)
    ep1 = _arrays_for_steps(steps=8)
    ep1["dir_idx"] = np.ones(8, dtype=np.int32)
    ep1["excitation_direction"] = np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1)
    )
    gt_context = grid.prepare_gt_mmd_context([ep0, ep1])
    candidate = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)

    result = grid.score_candidate_mmd(
        candidate_index=0,
        candidate=candidate,
        gt_context=gt_context,
        replay_observations=[ep0],
    )

    assert 0 in result.per_direction_mmd2
    assert result.missing_directions == (1,)


def test_trajectory_mse_empty_alignment_returns_nan_metrics():
    recorded = _arrays_for_steps(steps=0)
    replay = _arrays_for_steps(steps=0)
    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)
    assert out["n_frames"] == 0.0
    assert out["n_used_frames"] == 0.0
    assert np.isnan(out["ft_wrist_mse"])
    assert np.isnan(out["tcp_pos_mse"])
    assert np.isnan(out["apple_pos_mse"])
    assert np.isnan(out["ft_force_rmse"])
    assert np.isnan(out["ft_torque_rmse"])


def test_trajectory_mse_skips_pre_weld_phase_minus_one():
    recorded = _arrays_for_steps(steps=4, shift=0.0)
    replay = _arrays_for_steps(steps=4, shift=0.0)

    recorded["step_idx"] = np.array([-1, 0, 1, 2], dtype=np.int32)
    recorded["phase"] = np.array([-1, 1, 1, 1], dtype=np.int8)
    replay["phase"] = np.array([-1, 1, 1, 1], dtype=np.int8)
    recorded["ft_wrist"][0] += 999.0
    recorded["tcp_pos"][0] += 999.0
    recorded["apple_pos"][0] += 999.0

    recorded = grid.strip_pre_weld_rows(recorded)
    replay = grid.strip_pre_weld_rows(replay)

    out = grid.trajectory_mse(
        replay=replay,
        recorded=recorded,
        skip_phase=-1,
    )

    assert out["n_used_frames"] == 3
    assert out["ft_wrist_mse"] == pytest.approx(0.0)


def test_trajectory_mse_rejects_leading_pre_weld_without_strip():
    recorded = _arrays_for_steps(steps=5, shift=0.0)
    recorded["phase"] = np.array([-1, 1, 1, 1, 1], dtype=np.int8)
    replay = {
        "ft_wrist": recorded["ft_wrist"].copy(),
        "tcp_pos": recorded["tcp_pos"].copy(),
        "apple_pos": recorded["apple_pos"].copy(),
    }

    with pytest.raises(ValueError, match="pre_weld"):
        grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)


def test_trajectory_mse_matches_after_pre_weld_strip():
    recorded = _arrays_for_steps(steps=5, shift=0.0)
    recorded["step_idx"] = np.array([-1, 0, 1, 2, 3], dtype=np.int32)
    recorded["phase"] = np.array([-1, 1, 1, 1, 1], dtype=np.int8)

    replay = {
        "ft_wrist": recorded["ft_wrist"][1:].copy(),
        "tcp_pos": recorded["tcp_pos"][1:].copy(),
        "apple_pos": recorded["apple_pos"][1:].copy(),
        "phase": np.ones(4, dtype=np.int8),
    }
    recorded = grid.strip_pre_weld_rows(recorded)

    out = grid.trajectory_mse(
        replay=replay,
        recorded=recorded,
        skip_phase=-1,
    )

    assert out["n_frames"] == 4
    assert out["n_used_frames"] == 4
    assert out["ft_wrist_mse"] == pytest.approx(0.0)


def test_trajectory_hold_aggregated_mse_ignores_move_frames():
    recorded = _arrays_for_steps(steps=6, shift=0.0)
    replay = _arrays_for_steps(steps=6, shift=0.0)
    recorded["step_idx"] = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32)
    recorded["phase"] = np.array([-1, 0, 0, 1, 1, 1], dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    replay["ft_wrist"][:2] += 50.0
    replay["tcp_pos"][:2] += 0.5
    replay["apple_pos"][:2] += 0.5

    recorded = grid.strip_pre_weld_rows(recorded)
    replay = grid.strip_pre_weld_rows(replay)
    replay["ft_wrist"][:2] += 50.0
    replay["tcp_pos"][:2] += 0.5
    replay["apple_pos"][:2] += 0.5

    out = grid.trajectory_hold_aggregated_mse(
        replay=replay,
        recorded=recorded,
        aggregation="mean",
        use_latter_half=False,
    )

    assert out["n_used_frames"] == 3
    assert out["ft_wrist_mse"] == pytest.approx(0.0)
    assert out["tcp_pos_mse"] == pytest.approx(0.0)
    assert out["apple_pos_mse"] == pytest.approx(0.0)


def test_trajectory_hold_aggregated_mse_mean_detects_hold_offset():
    recorded = _arrays_for_steps(steps=4, shift=0.0)
    replay = _arrays_for_steps(steps=4, shift=0.0)
    recorded["phase"] = np.array([1, 1, 1, 1], dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    replay["ft_wrist"][:, :3] += 2.0
    replay["tcp_pos"] += 0.1

    out = grid.trajectory_hold_aggregated_mse(
        replay=replay,
        recorded=recorded,
        aggregation="mean",
        use_latter_half=False,
    )

    assert out["n_used_frames"] == 4
    assert out["ft_force_rmse"] == pytest.approx(2.0)
    assert out["tcp_pos_mse"] == pytest.approx(0.01, rel=0, abs=1e-5)


def test_trajectory_hold_aggregated_mse_median_ignores_single_outlier():
    recorded = _arrays_for_steps(steps=5, shift=0.0)
    replay = _arrays_for_steps(steps=5, shift=0.0)
    recorded["phase"] = np.ones(5, dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    const_ft = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32)
    recorded["ft_wrist"] = np.tile(const_ft, (5, 1))
    replay["ft_wrist"] = recorded["ft_wrist"].copy()
    replay["ft_wrist"][0, :3] += 100.0

    mean_out = grid.trajectory_hold_aggregated_mse(
        replay=replay,
        recorded=recorded,
        aggregation="mean",
        use_latter_half=False,
    )
    median_out = grid.trajectory_hold_aggregated_mse(
        replay=replay,
        recorded=recorded,
        aggregation="median",
        use_latter_half=False,
    )

    assert mean_out["ft_force_rmse"] > median_out["ft_force_rmse"]
    assert median_out["ft_force_rmse"] == pytest.approx(0.0, abs=1e-6)


def test_trajectory_mse_includes_all_frames_when_skip_phase_none():
    recorded = _arrays_for_steps(steps=2, shift=0.0)
    replay = _arrays_for_steps(steps=2, shift=0.0)

    recorded["step_idx"] = np.array([0, 1], dtype=np.int32)
    recorded["phase"] = np.array([1, 1], dtype=np.int8)
    replay["ft_wrist"][0] += 10.0

    out = grid.trajectory_mse(
        replay=replay,
        recorded=recorded,
        skip_phase=None,
    )

    assert out["n_used_frames"] == 2
    assert out["ft_wrist_mse"] > 0.0
    assert out["ft_force_rmse"] > 0.0


def test_trajectory_mse_reports_woody_pos_mse_by_segment():
    recorded = _arrays_for_steps(steps=4, shift=0.0)
    replay = _arrays_for_steps(steps=4, shift=0.0)

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)

    assert set(out["woody_pos_mse_by_segment"]) == set(recorded["junction_names"])
    for name in recorded["junction_names"]:
        assert out["woody_pos_mse_by_segment"][name] == pytest.approx(0.0)


def test_trajectory_mse_woody_pos_mse_by_segment_detects_shift():
    recorded = _arrays_for_steps(steps=3, shift=0.0)
    replay = _arrays_for_steps(steps=3, shift=1.0)

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)

    for name in recorded["junction_names"]:
        assert out["woody_pos_mse_by_segment"][name] > 0.0


def test_trajectory_mse_without_junction_names_returns_empty_woody_dict():
    recorded = _arrays_for_steps(steps=2, shift=0.0)
    replay = {
        "ft_wrist": recorded["ft_wrist"].copy(),
        "tcp_pos": recorded["tcp_pos"].copy(),
        "apple_pos": recorded["apple_pos"].copy(),
    }
    del recorded["junction_names"]
    del recorded["woody_part_start_pos"]
    del recorded["woody_part_end_pos"]

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)

    assert out["woody_pos_mse_by_segment"] == {}


def test_trajectory_mse_without_replay_woody_returns_empty_woody_dict():
    recorded = _arrays_for_steps(steps=2, shift=0.0)
    replay = {
        "ft_wrist": recorded["ft_wrist"].copy(),
        "tcp_pos": recorded["tcp_pos"].copy(),
        "apple_pos": recorded["apple_pos"].copy(),
    }

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)

    assert out["woody_pos_mse_by_segment"] == {}


def test_trajectory_hold_aggregated_mse_reports_woody_pos_mse_by_segment():
    recorded = _arrays_for_steps(steps=6, shift=0.0)
    replay = _arrays_for_steps(steps=6, shift=0.0)
    recorded["phase"] = np.array([0, 0, 1, 1, 1, 1], dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()

    out = grid.trajectory_hold_aggregated_mse(
        replay=replay,
        recorded=recorded,
        aggregation="mean",
        use_latter_half=False,
    )

    assert set(out["woody_pos_mse_by_segment"]) == set(recorded["junction_names"])
    for name in recorded["junction_names"]:
        assert out["woody_pos_mse_by_segment"][name] == pytest.approx(0.0)


def test_trajectory_mse_drops_gt_unstable_rows():
    recorded = _arrays_for_steps(steps=4, shift=0.0)
    replay = _arrays_for_steps(steps=4, shift=0.0)
    recorded["phase"] = np.ones(4, dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    recorded["stable"] = np.array([True, False, True, True], dtype=bool)
    replay["stable"] = np.ones(4, dtype=bool)
    replay["ft_wrist"][1, :3] += 500.0

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)
    assert out["n_used_frames"] == 3.0
    assert out["ft_force_rmse"] == pytest.approx(0.0, abs=1e-6)


def test_trajectory_mse_excludes_replay_unstable_frames():
    recorded = _arrays_for_steps(steps=3, shift=0.0)
    replay = _arrays_for_steps(steps=3, shift=0.0)
    recorded["phase"] = np.ones(3, dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    recorded["stable"] = np.ones(3, dtype=bool)
    replay["stable"] = np.array([True, False, True], dtype=bool)
    replay["ft_wrist"][1, 0] = 999.0

    out = grid.trajectory_mse(replay=replay, recorded=recorded, skip_phase=-1)
    assert out["n_used_frames"] == 2.0
    assert out["ft_force_rmse"] == pytest.approx(0.0, abs=1e-6)


def test_trajectory_paired_hold_median_mse_woody_is_mean_of_per_hold():
    """Woody must average per-hold median MSEs (not one median over all holds)."""
    recorded = _arrays_for_steps(steps=10, junction_names=["joint_a"], shift=0.0)
    replay = _arrays_for_steps(steps=10, junction_names=["joint_a"], shift=0.0)
    recorded["phase"] = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    recorded["dir_idx"] = np.zeros(10, dtype=np.int32)
    replay["dir_idx"] = recorded["dir_idx"].copy()
    # Hold 2 only: shift woody endpoints on replay frames 5,6,7.
    for key in ("woody_part_start_pos", "woody_part_end_pos"):
        replay[key]["joint_a"][5:8] = recorded[key]["joint_a"][5:8] + 2.0

    out = grid.trajectory_paired_hold_median_mse(replay=replay, recorded=recorded)
    # Hold1 MSE=0; hold2 median shift=2 on start(3)+end(3)=6 dims → mse = 4.0
    assert out["woody_pos_mse_by_segment"]["joint_a"] == pytest.approx(2.0)

    # Flat-bag median over all hold frames differs from mean of per-hold MSEs.
    flat = grid.woody_segment_pos_mse_hold_aggregated(
        replay=replay,
        recorded=recorded,
        junction_names=["joint_a"],
        n=10,
        hold_idx=np.array([1, 2, 3, 5, 6, 7], dtype=np.int64),
        aggregation="median",
    )
    assert flat["joint_a"] != pytest.approx(out["woody_pos_mse_by_segment"]["joint_a"])


def test_trajectory_hold_aggregated_mse_excludes_replay_unstable_hold_frames():
    recorded = _arrays_for_steps(steps=4, shift=0.0)
    replay = _arrays_for_steps(steps=4, shift=0.0)
    recorded["phase"] = np.ones(4, dtype=np.int8)
    replay["phase"] = recorded["phase"].copy()
    recorded["stable"] = np.ones(4, dtype=bool)
    replay["stable"] = np.array([True, False, True, True], dtype=bool)
    replay["ft_wrist"][1, 0] = 999.0

    out = grid.trajectory_hold_aggregated_mse(
        replay=replay,
        recorded=recorded,
        aggregation="mean",
        use_latter_half=False,
    )
    assert out["n_used_frames"] == 3.0
    assert out["ft_force_rmse"] == pytest.approx(0.0, abs=1e-6)


def test_replay_instability_fraction_all_frames_counts_replay_unstable():
    recorded = _arrays_for_steps(steps=10, shift=0.0)
    replay = _arrays_for_steps(steps=10, shift=0.0)
    recorded["stable"] = np.ones(10, dtype=bool)
    replay["stable"] = np.ones(10, dtype=bool)
    replay["stable"][2] = False
    replay["stable"][3] = False
    assert grid.replay_instability_fraction_all_frames(replay=replay, recorded=recorded) == pytest.approx(
        0.2
    )


def test_recorded_instability_fraction_all_frames_ignores_pre_weld():
    recorded = _arrays_for_steps(steps=4, shift=0.0)
    recorded["phase"] = np.array([-1, 0, 0, 0], dtype=np.int8)
    recorded["stable"] = np.array([False, True, False, True], dtype=bool)
    assert grid.recorded_instability_fraction_all_frames(recorded) == pytest.approx(1.0 / 3.0)


def test_warn_recorded_gt_instability_emits_warning():
    recorded = _arrays_for_steps(steps=10, shift=0.0)
    recorded["stable"] = np.ones(10, dtype=bool)
    recorded["stable"][:3] = False

    with pytest.warns(UserWarning, match="recorded GT dataset"):
        msgs = grid.warn_recorded_gt_instability(structure_idx=0, recorded_eps=[recorded])
    assert len(msgs) == 1


def test_concat_replay_arrays_propagates_stable():
    left = {
        "action": np.zeros((2, 6), dtype=np.float32),
        "ft_wrist": np.zeros((2, 6), dtype=np.float32),
        "tcp_velocity": np.zeros((2, 6), dtype=np.float32),
        "tcp_pos": np.zeros((2, 3), dtype=np.float32),
        "apple_pos": np.zeros((2, 3), dtype=np.float32),
        "phase": np.zeros(2, dtype=np.int8),
        "dir_idx": np.zeros(2, dtype=np.int32),
        "excitation_type": np.zeros(2, dtype=np.int8),
        "excitation_direction": np.zeros((2, 3), dtype=np.float32),
        "stable": np.array([True, False], dtype=bool),
        "woody_part_start_pos": {"j": np.zeros((2, 3), dtype=np.float32)},
        "woody_part_end_pos": {"j": np.zeros((2, 3), dtype=np.float32)},
        "junction_names": ["j"],
    }
    right = dict(left)
    right["stable"] = np.array([False, True], dtype=bool)
    merged = grid._concat_replay_arrays(left, right)
    assert merged["stable"].tolist() == [True, False, False, True]


def test_list_usable_direction_indices_skips_excluded():
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
    from unittest.mock import MagicMock

    dataset = MagicMock()
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 0, "excluded": False},
        {"structure_idx": 0, "direction_idx": 1, "excluded": True},
        {"structure_idx": 0, "direction_idx": 2, "excluded": False},
        {"structure_idx": 1, "direction_idx": 0, "excluded": False},
    ]
    assert grid.list_usable_direction_indices(dataset, 0) == [0, 2]
    assert grid.list_usable_direction_indices(dataset, 0, include_excluded=True) == [0, 1, 2]


def test_list_usable_direction_indices_raises_when_all_excluded():
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
    from unittest.mock import MagicMock
    import pytest

    dataset = MagicMock()
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 0, "excluded": True},
        {"structure_idx": 0, "direction_idx": 1, "excluded": True},
    ]
    with pytest.raises(ValueError, match="no usable directions"):
        grid.list_usable_direction_indices(dataset, 0)


def test_per_candidate_unstable_counts_groups_by_usable_directions():
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid

    # 3 candidates x 2 usable directions; per-env unstable frame counts.
    unstable_by_env = [1, 2, 10, 20, 100, 200]

    got = grid.per_candidate_unstable_counts(
        unstable_by_env, num_candidates=3, num_directions=2
    )

    assert got == [3, 30, 300]


def test_per_candidate_unstable_counts_rejects_length_mismatch():
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
    import pytest

    # 3 candidates x 2 directions requires 6 entries, not 5.
    with pytest.raises(ValueError, match="num_candidates"):
        grid.per_candidate_unstable_counts(
            [1, 2, 3, 4, 5], num_candidates=3, num_directions=2
        )


def test_load_recorded_episodes_skips_excluded_directions():
    from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid
    from unittest.mock import MagicMock
    import numpy as np

    dataset = MagicMock()
    dataset.episode_entries.return_value = [
        {"structure_idx": 0, "direction_idx": 0, "excluded": False},
        {"structure_idx": 0, "direction_idx": 1, "excluded": True},
    ]

    def _load(s, d):
        return {
            "action": np.zeros((2, 6), dtype=np.float32),
            "phase": np.zeros(2, dtype=np.int8),
            "step_idx": np.arange(2, dtype=np.int32),
        }

    dataset.load_episode_obs_arrays.side_effect = _load
    eps = grid.load_recorded_episodes_for_structure(
        dataset, structure_idx=0, num_directions=2
    )
    assert len(eps) == 1
    assert int(eps[0]["dir_idx"][0]) == 0

