"""Tests for settled checkpoint disk I/O (V.3.1 step B)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.coupled_fruiting.settled_checkpoint import (
    SettledCheckpoint,
    build_cache_key,
    resolve_settle_cache_dir,
    settle_cache_path_for,
)
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list

RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "fruiting_system_ranges_straight_rod_test.json"
)


@pytest.fixture
def ranges():
    return load_ranges(RANGES_FIXTURE)


@pytest.fixture
def per_env_params(ranges):
    return sample_heterogeneous_params_list(ranges, topology_seed=11, num_envs=2)


@pytest.fixture
def config():
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=2).domain_randomization,
            topology_seed=11,
        ),
    )


def _sample_checkpoint(config, ranges, per_env_params) -> SettledCheckpoint:
    body_q = np.random.randn(20, 7).astype(np.float32)
    return SettledCheckpoint.from_build_context(
        body_q=body_q,
        config=config,
        ranges=ranges,
        per_env_params=per_env_params,
    )


def test_cache_key_includes_settle_quiet_every(config, ranges, per_env_params):
    key_off = build_cache_key(config, ranges, per_env_params)
    quiet_cfg = dataclasses.replace(
        config,
        scene=dataclasses.replace(config.scene, settle_quiet_every=100),
    )
    key_on = build_cache_key(quiet_cfg, ranges, per_env_params)
    assert key_off != key_on
    assert "quiet_every=100" in key_on
    assert "quiet_every=300" in key_off


def test_save_load_roundtrip(tmp_path, config, ranges, per_env_params):
    ckpt = _sample_checkpoint(config, ranges, per_env_params)
    path = tmp_path / "test.npz"
    ckpt.save(path)
    loaded = SettledCheckpoint.load(path)
    np.testing.assert_array_equal(loaded.body_q, ckpt.body_q)
    assert loaded.metadata["schema_version"] == ckpt.metadata["schema_version"]
    assert loaded.metadata["cache_key"] == ckpt.metadata["cache_key"]


def test_validate_rejects_mismatch(tmp_path, config, ranges, per_env_params):
    ckpt = _sample_checkpoint(config, ranges, per_env_params)
    path = tmp_path / "test.npz"
    ckpt.save(path)
    loaded = SettledCheckpoint.load(path)
    bad_cfg = dataclasses.replace(
        config,
        scene=dataclasses.replace(config.scene, settle_substeps=config.scene.settle_substeps + 1),
    )
    with pytest.raises(ValueError, match="settle_substeps"):
        loaded.validate_against(config=bad_cfg, ranges=ranges, per_env_params=per_env_params)


def test_validate_rejects_params_mismatch(tmp_path, config, ranges, per_env_params):
    ckpt = _sample_checkpoint(config, ranges, per_env_params)
    path = tmp_path / "test.npz"
    ckpt.save(path)
    loaded = SettledCheckpoint.load(path)
    other_params = sample_heterogeneous_params_list(ranges, topology_seed=99, num_envs=2)
    with pytest.raises(ValueError, match="per_env_params"):
        loaded.validate_against(config=config, ranges=ranges, per_env_params=other_params)


def test_settle_cache_path_deterministic(config, ranges, per_env_params, tmp_path):
    p1 = settle_cache_path_for(
        config, ranges, per_env_params, cache_dir=tmp_path
    )
    p2 = settle_cache_path_for(
        config, ranges, per_env_params, cache_dir=tmp_path
    )
    assert p1 == p2
    assert p1.parent == tmp_path


def test_resolve_settle_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("APPLE_PICK_SIM_SETTLE_CACHE_DIR", str(tmp_path / "custom"))
    assert resolve_settle_cache_dir(None) == tmp_path / "custom"
