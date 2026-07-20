"""Tests for post-weld episode snapshot capture/restore on batched coupled sim."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.coupled_fruiting.episode_state_snapshot import EpisodeStateSnapshot
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list

from conftest import requires_fr3

RANGES_FIXTURE = _TESTS_DIR.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
_NUM_ENVS = 2


def _require_torch():
    pytest.importorskip("torch")
    import torch

    return torch


def _vic_gym_config(*, num_envs: int = _NUM_ENVS) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            # Unfixed TCP so VIC actions produce a measurable state change for
            # capture/restore round-trips (fix_to_apple can keep TCP stationary).
            fix_to_apple=False,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=8),
        controller=ControllerConfig(mode="vic"),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).domain_randomization,
            topology_seed=21,
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


def _tcp_positions(sim: BatchedHeterogeneousCoupledSim) -> np.ndarray:
    layout = sim.layout
    assert layout is not None
    bq = sim.scene.robot_state_0.body_q.numpy().reshape(-1, 7)
    rows = []
    for tcp_idx in layout.tcp_body_indices:
        rows.append(bq[int(tcp_idx), :3])
    return np.stack(rows, axis=0)


@requires_fr3
def test_capture_restore_round_trip_after_steps():
    torch = _require_torch()
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(
        ranges, topology_seed=21, num_envs=_NUM_ENVS
    )
    sim = BatchedHeterogeneousCoupledSim(
        _vic_gym_config(),
        params,
        ranges,
        use_settle_cache=False,
    )
    baseline_tcp = _tcp_positions(sim)
    snapshot = EpisodeStateSnapshot.capture(sim)

    actions = torch.zeros((_NUM_ENVS, 6), dtype=torch.float32, device=sim.device)
    actions[0, 0] = 0.05
    for _ in range(4):
        sim.step(actions)

    moved_tcp = _tcp_positions(sim)
    assert not np.allclose(moved_tcp, baseline_tcp, atol=1e-5)

    snapshot.restore(sim)
    restored_tcp = _tcp_positions(sim)
    np.testing.assert_allclose(restored_tcp, baseline_tcp, rtol=1e-6, atol=1e-5)


@requires_fr3
def test_sim_wrapper_capture_restore():
    torch = _require_torch()
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(
        ranges, topology_seed=21, num_envs=_NUM_ENVS
    )
    sim = BatchedHeterogeneousCoupledSim(
        _vic_gym_config(),
        params,
        ranges,
        use_settle_cache=False,
    )
    sim.capture_episode_snapshot()
    baseline_tcp = _tcp_positions(sim)

    actions = torch.zeros((_NUM_ENVS, 6), dtype=torch.float32, device=sim.device)
    actions[0, 0] = 0.05
    for _ in range(4):
        sim.step(actions)
    assert not np.allclose(_tcp_positions(sim), baseline_tcp, atol=1e-5)

    sim.restore_episode_snapshot()
    np.testing.assert_allclose(_tcp_positions(sim), baseline_tcp, rtol=1e-6, atol=1e-5)


@requires_fr3
def test_restore_without_capture_raises():
    ranges = load_ranges(RANGES_FIXTURE)
    params = sample_heterogeneous_params_list(
        ranges, topology_seed=21, num_envs=_NUM_ENVS
    )
    sim = BatchedHeterogeneousCoupledSim(
        _vic_gym_config(),
        params,
        ranges,
        use_settle_cache=False,
    )
    with pytest.raises(RuntimeError, match="snapshot"):
        sim.restore_episode_snapshot()
