"""HarvestMetricsLogger: metric keys, per-env traces, episode outcomes (pure torch, no sim)."""

from __future__ import annotations

import math

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
from apple_pick_gym.batched_envs.harvest_logging import HarvestMetricsLogger
from apple_pick_gym.batched_envs.harvest_reward import (
    HarvestRewardConfig,
    compute_dense_reward,
    compute_dense_reward_terms,
    weight_dense_reward_terms,
)

_JUNCTIONS = ["primary_spur", "spur_stem", "stem_apple", "support"]
_N = 5


def _obs_info(*, success=None, safety_junction=None, safety_wrist=None, frozen=None):
    n = _N
    z = torch.zeros(n, dtype=torch.bool)
    obs = {
        "tcp_pos": torch.zeros(n, 3),
        "tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1),  # xyzw identity
        "tcp_velocity": torch.zeros(n, 6),
        "ft_wrist": torch.full((n, 6), 2.0),
    }
    target = torch.zeros(n, 7)
    target[:, 3] = 1.0  # wxyz identity
    info = {
        "woody_part_force": {j: torch.full((n, 6), float(k + 1)) for k, j in enumerate(_JUNCTIONS)},
        "target_junction_force": torch.full((n, 6), 3.0),
        "ft_wrist": torch.full((n, 6), 1.0),
        "apple_pos": torch.zeros(n, 3),
        "target_pose": target,
        "reward_terms": {
            "raw": {k: torch.full((n,), 1.0) for k in ("progress", "pullout", "collateral")},
            "weighted": {k: torch.full((n,), 0.5) for k in ("progress", "pullout", "collateral")},
            "dense": torch.full((n,), 1.5),
            "terminal": torch.zeros(n),
            "total": torch.full((n,), 1.5),
        },
        "episode": {
            "success_this_step": z.clone() if success is None else success,
            "success_achieved": z.clone() if success is None else success,
            "success_streak": torch.zeros(n, dtype=torch.int64),
            "safety_junction": z.clone() if safety_junction is None else safety_junction,
            "safety_wrist": z.clone() if safety_wrist is None else safety_wrist,
            "frozen": z.clone() if frozen is None else frozen,
        },
    }
    return obs, info


def _logger(**kw):
    return HarvestMetricsLogger(num_envs=_N, action_bounds=HarvestActionBounds(), **kw)


def _actions():
    b = HarvestActionBounds()
    a = torch.zeros(_N, 13)
    a[:, 6:9] = b.k_lin_max
    a[:, 9:12] = b.k_ang_max
    a[:, 12] = b.zeta_max
    return a


def _step(logger, obs, info):
    z = torch.zeros(_N, 1, dtype=torch.bool)
    return logger.step_metrics(obs, info, _actions(), z, z)


def test_metrics_cover_reward_terms_junctions_wrist_and_action():
    obs, info = _obs_info()
    lg = _logger()
    lg.on_reset(info)
    m = _step(lg, obs, info)
    for name in ("progress", "pullout", "collateral"):
        assert f"reward/raw/{name}/mean" in m
        assert f"reward/weighted/{name}/mean" in m
    for j in _JUNCTIONS:
        assert f"force/junction/{j}/F/max" in m
        assert f"force/junction/{j}/T/mean" in m
    assert "wrist/raw/F/mean" in m and "wrist/obs/F/mean" in m
    assert "action/K_lin/mean" in m and "action/D_lin/mean" in m
    assert "tracking/pos_err_m/mean" in m
    assert all(math.isfinite(v) for v in m.values())


def test_wrist_obs_minus_raw_and_junction_values():
    obs, info = _obs_info()
    lg = _logger()
    lg.on_reset(info)
    m = _step(lg, obs, info)
    # obs=2, raw=1 on every axis -> |diff[:3]| = sqrt(3)
    assert m["wrist/obs_minus_raw/F/mean"] == pytest.approx(math.sqrt(3), rel=1e-5)
    # junction k has wrench filled with (k+1): |F| = (k+1)*sqrt(3)
    assert m["force/junction/stem_apple/F/mean"] == pytest.approx(3 * math.sqrt(3), rel=1e-5)


def test_per_env_traces_only_for_traced_envs():
    obs, info = _obs_info()
    lg = _logger(trace_env_ids=[1, 3])
    lg.on_reset(info)
    m = _step(lg, obs, info)
    assert "env1/force/target/F" in m and "env3/wrist/raw/Fz" in m
    assert not any(k.startswith(("env0/", "env2/", "env4/")) for k in m)


def test_default_traces_first_k_envs():
    lg = _logger(trace_envs=2)
    assert lg.trace_env_ids == [0, 1]


def test_trace_env_id_out_of_range_rejected():
    with pytest.raises(ValueError):
        _logger(trace_env_ids=[_N])


def test_tracking_error_uses_correct_quat_order():
    obs, info = _obs_info()
    # tcp rotated 90deg about z (xyzw), target identity (wxyz) -> rot_err = pi/2
    s = math.sqrt(0.5)
    obs["tcp_quat"] = torch.tensor([[0.0, 0.0, s, s]]).repeat(_N, 1)
    lg = _logger()
    lg.on_reset(info)
    m = _step(lg, obs, info)
    assert m["tracking/rot_err_rad/mean"] == pytest.approx(math.pi / 2, abs=1e-4)


def test_episode_outcomes_success_safety_truncated():
    lg = _logger()
    obs, info = _obs_info()
    lg.on_reset(info)
    succ = torch.tensor([True, False, False, False, False])
    safe = torch.tensor([False, True, False, False, False])
    obs, info = _obs_info(success=succ, safety_junction=safe)
    _step(lg, obs, info)
    # second step: nothing new; env 2..4 end via truncation
    obs, info = _obs_info()
    z = torch.zeros(_N, 1, dtype=torch.bool)
    lg.step_metrics(obs, info, _actions(), z, torch.ones(_N, 1, dtype=torch.bool))
    s = lg.on_reset(info)
    assert s["episode/success_rate"] == pytest.approx(1 / _N)
    assert s["episode/safety_rate"] == pytest.approx(1 / _N)
    assert s["episode/truncated_rate"] == pytest.approx(3 / _N)
    assert s["episode/steps_to_success_mean"] == 1.0


def test_safety_takes_precedence_over_success_like_the_env():
    lg = _logger()
    obs, info = _obs_info()
    lg.on_reset(info)
    both = torch.tensor([True, False, False, False, False])
    obs, info = _obs_info(success=both, safety_wrist=both)
    _step(lg, obs, info)
    s = lg.flush_summary()
    assert s["episode/safety_rate"] == pytest.approx(1 / _N)
    assert s["episode/success_rate"] == 0.0


def test_return_stops_accumulating_after_env_finishes():
    lg = _logger()
    obs, info = _obs_info()
    lg.on_reset(info)
    safe = torch.tensor([True, False, False, False, False])
    obs, info = _obs_info(safety_junction=safe)
    _step(lg, obs, info)  # env0 ends this step (counts 1.5)
    obs, info = _obs_info()
    _step(lg, obs, info)  # env0 already finished -> no more return
    s = lg.flush_summary()
    assert s["episode/return_min"] == pytest.approx(1.5)
    assert s["episode/return_max"] == pytest.approx(3.0)


def test_reward_terms_recombine_to_dense_reward():
    n = 4
    cfg = HarvestRewardConfig()
    obs = {"tcp_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)}
    info = {
        "target_junction_force": torch.tensor([[3.0, 0, 0, 0, 0, 0]]).repeat(n, 1),
        "ft_wrist": torch.tensor([[0.0, 0.0, 4.0, 0, 0, 0]]).repeat(n, 1),
        "woody_part_force": {
            "spur_stem": torch.ones(n, 6),
            "support": torch.full((n, 6), 2.0),
        },
    }
    terms = compute_dense_reward_terms(obs, info, target_junction_name="spur_stem", cfg=cfg)
    w = weight_dense_reward_terms(terms, cfg)
    total = (w["progress"] + w["pullout"] + w["collateral"]).unsqueeze(-1)
    torch.testing.assert_close(
        total, compute_dense_reward(obs, info, target_junction_name="spur_stem", cfg=cfg)
    )
    assert torch.all(w["pullout"] <= 0) and torch.all(w["collateral"] <= 0)
