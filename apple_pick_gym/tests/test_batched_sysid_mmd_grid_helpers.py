"""Tests for batched sys-ID MMD grid and recorded-action tensor helpers."""

from __future__ import annotations

import dataclasses
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


def test_ensure_gt_candidate_in_grid_replaces_last_when_missing():
    gt = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)
    candidates = [
        grid.BendStiffnessCandidate(10.0, 20.0, 30.0, 40.0),
        grid.BendStiffnessCandidate(11.0, 21.0, 31.0, 41.0),
    ]

    out = grid.ensure_gt_candidate_in_grid(candidates, gt)

    assert out[:-1] == candidates[:-1]
    assert out[-1] == gt
    assert out is not candidates


def test_ensure_gt_candidate_in_grid_returns_singleton_when_empty():
    gt = grid.BendStiffnessCandidate(1.0, 2.0, 3.0, 4.0)

    assert grid.ensure_gt_candidate_in_grid([], gt) == [gt]


def test_gt_bend_stiffness_candidate_from_structure_reads_inferred_stiffness(monkeypatch):
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
        "infer_base_params_for_structure",
        lambda _dataset, _idx: base,
    )

    got = grid.gt_bend_stiffness_candidate_from_structure(dataset, structure_idx=0)

    assert got == expected


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
