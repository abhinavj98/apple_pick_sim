"""Tests for prepare/evaluate direction subset wiring in batched CMA-ES."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.tests.conftest import RANGES_FIXTURE
from apple_pick_gym.batched_envs import batched_sysid_cmaes as cmaes
from apple_pick_gym.batched_envs import batched_sysid_mmd_grid as grid


def _base_primary_spur_stem_with_secondary(seed: int = 0) -> fs.FruitingSystemParams:
    ranges = fs.load_ranges(RANGES_FIXTURE)
    params = fs.sample_params(ranges, seed=seed)
    assert params.primary is not None and params.spur is not None and params.stem is not None
    assert params.secondary is not None
    return params


def _median_hold_episode(*, direction_idx: int, n_holds: int = 2) -> dict:
    """Episode with median holds and disk ``dir_idx`` for Wasserstein feature build."""
    chunks: list[np.ndarray] = []
    for _ in range(n_holds):
        chunks.append(np.zeros(1, dtype=np.int8))
        chunks.append(np.ones(3, dtype=np.int8))
    chunks.append(np.zeros(1, dtype=np.int8))
    phase = np.concatenate(chunks)
    n_frames = int(phase.size)
    junction_names = ["primary_spur", "spur_stem"]
    base = np.arange(n_frames, dtype=np.float32).reshape(n_frames, 1)
    woody_start = {
        "primary_spur": np.hstack([base + 100.0, base + 101.0, base + 102.0]).astype(
            np.float32
        ),
        "spur_stem": np.hstack([base + 200.0, base + 201.0, base + 202.0]).astype(
            np.float32
        ),
    }
    return {
        "action": np.zeros((n_frames, 6), dtype=np.float32),
        "phase": phase,
        "dir_idx": np.full(n_frames, int(direction_idx), dtype=np.int32),
        "excitation_type": np.zeros(n_frames, dtype=np.int8),
        "excitation_direction": np.tile(
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            (n_frames, 1),
        ),
        "junction_names": junction_names,
        "ft_wrist": np.zeros((n_frames, 6), dtype=np.float32),
        "tcp_velocity": np.zeros((n_frames, 6), dtype=np.float32),
        "tcp_pos": np.zeros((n_frames, 3), dtype=np.float32),
        "apple_pos": np.zeros((n_frames, 3), dtype=np.float32),
        "woody_part_start_pos": woody_start,
        "stable": np.ones(n_frames, dtype=bool),
    }


def _make_eight_dir_dataset(
    *,
    structure_idx: int = 0,
    gt_params: fs.FruitingSystemParams,
    vic_pose: bool = False,
) -> MagicMock:
    dataset = MagicMock()
    collection: dict[str, object] = {
        "num_directions": 8,
        "action_dim": 19 if vic_pose else 6,
        "seed": 0,
    }
    if vic_pose:
        collection["action_layout"] = "vic_pose_v1"
    dataset.manifest = {"collection": collection}
    dataset.episode_entries.return_value = [
        {"structure_idx": structure_idx, "direction_idx": d} for d in range(8)
    ]
    obs_calls: list[tuple[int, int]] = []

    def load_episode_obs_arrays(s_idx: int, d_idx: int) -> dict:
        obs_calls.append((int(s_idx), int(d_idx)))
        return _median_hold_episode(direction_idx=int(d_idx))

    dataset.load_episode_obs_arrays.side_effect = load_episode_obs_arrays
    dataset._obs_calls = obs_calls

    def load_episode_metadata(s_idx: int, d_idx: int) -> dict:
        return {
            "structure_idx": int(s_idx),
            "direction_idx": int(d_idx),
            "meta_tag": f"s{s_idx}-d{d_idx}",
        }

    dataset.load_episode_metadata.side_effect = load_episode_metadata
    return dataset


def _default_scoring(*, n_directions: int = 8) -> cmaes.YoungsModulusScoringConfig:
    return cmaes.YoungsModulusScoringConfig(
        use_median=True,
        hold_id_onehot=True,
        pool_directions=True,
        n_holds=4,
        n_directions=n_directions,
        device="cpu",
    )


def _default_candidate(gt_params: fs.FruitingSystemParams) -> cmaes.YoungsModulusCandidate:
    return cmaes.youngs_modulus_candidate_from_params(gt_params)


class _StubReplayCollectors:
    """Minimal collector stub: ``direction_episodes_from_collectors`` indexing only."""

    def __init__(self, episodes: list[dict]) -> None:
        self._episodes = episodes

    def to_arrays(self, env_idx: int) -> dict:
        return self._episodes[int(env_idx)]


@pytest.fixture
def gt_params() -> fs.FruitingSystemParams:
    return _base_primary_spur_stem_with_secondary()


def _patch_prepare_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gt_params: fs.FruitingSystemParams,
) -> None:
    monkeypatch.setattr(cmaes, "true_params_for_structure", lambda _ds, _idx: gt_params)
    monkeypatch.setattr(
        cmaes,
        "gripper_proxy_from_episode_metadata",
        lambda _meta: fs.GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        cmaes,
        "gripper_proxy_for_real_batched_replay",
        lambda _meta: fs.GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(0.0, 0.0, 1.0),
        ),
    )


def test_prepare_uses_only_requested_directions(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(_default_candidate(gt_params),),
        num_directions=8,
        scoring=_default_scoring(),
        direction_indices=(0, 1),
    )

    loaded_dirs = {d for _s, d in dataset._obs_calls}
    assert loaded_dirs == {0, 1}
    assert 5 not in loaded_dirs
    assert prepared.direction_indices == (0, 1)


def test_prepare_onehot_width_stays_collection_num_directions(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params)
    selected = (2, 4, 5, 6, 7)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(_default_candidate(gt_params),),
        num_directions=8,
        scoring=_default_scoring(n_directions=8),
        direction_indices=selected,
    )

    assert prepared.scoring_n_directions == 8
    assert set(prepared.gt_context.expected_directions) == set(selected)
    assert 5 in prepared.gt_context.per_direction
    base_dim = prepared.gt_context.per_direction[5].gt_norm.shape[1]
    assert prepared.gt_context.pooled.gt_norm.shape[1] == base_dim + 8


def test_prepare_attaches_meta_by_direction_for_selection(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params, vic_pose=True)
    selected = (2, 4, 5)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(_default_candidate(gt_params),),
        num_directions=8,
        scoring=_default_scoring(),
        direction_indices=selected,
    )

    meta = prepared.replay_request.meta_by_direction
    assert meta is not None
    assert set(meta) == set(selected)
    for direction_idx in selected:
        assert meta[direction_idx]["direction_idx"] == direction_idx
        assert meta[direction_idx]["meta_tag"] == f"s0-d{direction_idx}"


def test_prepare_omits_meta_by_direction_for_sim_sim(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(_default_candidate(gt_params),),
        num_directions=8,
        scoring=_default_scoring(),
        direction_indices=(2, 4, 5),
    )

    assert prepared.replay_request.meta_by_direction is None


def test_prepare_defaults_to_all_usable_directions(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(_default_candidate(gt_params),),
        num_directions=8,
        scoring=_default_scoring(),
    )

    loaded_dirs = {d for _s, d in dataset._obs_calls}
    assert loaded_dirs == set(range(8))
    assert prepared.direction_indices == tuple(range(8))


def test_prepare_rejects_direction_not_on_disk(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params)

    with pytest.raises(ValueError, match=r"\b99\b"):
        cmaes.prepare_youngs_modulus_structure(
            dataset=dataset,
            structure_idx=0,
            candidates=(_default_candidate(gt_params),),
            num_directions=8,
            scoring=_default_scoring(),
            direction_indices=(0, 99),
        )


def test_collector_local_index_zips_to_disk_ids(
    monkeypatch: pytest.MonkeyPatch,
    gt_params: fs.FruitingSystemParams,
):
    _patch_prepare_side_effects(monkeypatch, gt_params=gt_params)
    dataset = _make_eight_dir_dataset(structure_idx=0, gt_params=gt_params)
    selected = (2, 4, 5, 6, 7)

    prepared = cmaes.prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=0,
        candidates=(_default_candidate(gt_params),),
        num_directions=8,
        scoring=_default_scoring(),
        direction_indices=selected,
    )

    collectors = _StubReplayCollectors(
        [
            _median_hold_episode(direction_idx=local_idx)
            for local_idx in range(len(selected))
        ]
    )
    local_eps = grid.direction_episodes_from_collectors(
        collectors,
        candidate_index=0,
        num_directions=len(selected),
    )
    assert len(local_eps) == len(selected)
    disk_ids = tuple(prepared.direction_indices)
    assert disk_ids == selected
    for local_idx, (disk_id, ep) in enumerate(zip(disk_ids, local_eps, strict=True)):
        assert int(ep["dir_idx"][0]) == local_idx
        assert disk_id == selected[local_idx]
