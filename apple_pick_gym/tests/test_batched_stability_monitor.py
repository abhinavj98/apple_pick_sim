"""Tests for vectorized batched stability monitor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest
import torch

from apple_pick_gym.batched_envs.batched_stability_monitor import (
    BatchedStabilityMonitor,
    StabilityThresholds,
    ik_bootstrap_unstable_mask,
)

_KNOWN_OBS_KEYS = frozenset(
    {"ft_wrist", "tcp_velocity", "apple_pos", "woody_part_info"}
)


def _nominal_obs(*, num_envs: int = 3) -> dict[str, Any]:
    """Hand-built batched obs dict with nominal (stable) values."""
    n = int(num_envs)
    return {
        "ft_wrist": torch.zeros(n, 6, dtype=torch.float32),
        "tcp_velocity": torch.zeros(n, 6, dtype=torch.float32),
        "apple_pos": torch.zeros(n, 3, dtype=torch.float32),
        "woody_part_info": {
            "stem_apple": {
                "anchors_pos": torch.zeros(n, 6, dtype=torch.float32),
                "anchor_force": torch.zeros(n, 6, dtype=torch.float32),
            },
        },
    }


def _monitor(*, num_envs: int = 3, **kwargs) -> BatchedStabilityMonitor:
    return BatchedStabilityMonitor(
        num_envs,
        known_obs_keys=set(_KNOWN_OBS_KEYS),
        **kwargs,
    )


def test_nominal_obs_is_stable():
    obs = _nominal_obs(num_envs=3)
    report = _monitor(num_envs=3).check(obs, step_idx=0)
    assert not bool(report.unstable.any())
    assert report.reasons == [[], [], []]


def test_nan_in_ft_wrist_flags_only_that_env():
    obs = _nominal_obs(num_envs=3)
    obs["ft_wrist"] = obs["ft_wrist"].clone()
    obs["ft_wrist"][1, 0] = float("nan")
    report = _monitor(num_envs=3).check(obs, step_idx=0)
    assert report.unstable.tolist() == [False, True, False]
    assert "nan_or_inf:ft_wrist" in report.reasons[1]


def test_inf_in_tcp_velocity_flags_with_reason():
    obs = _nominal_obs(num_envs=2)
    obs["tcp_velocity"] = obs["tcp_velocity"].clone()
    obs["tcp_velocity"][0, 2] = float("inf")
    report = _monitor(num_envs=2).check(obs, step_idx=0)
    assert report.unstable.tolist() == [True, False]
    assert "nan_or_inf:tcp_velocity" in report.reasons[0]


def test_default_stability_thresholds_match_scene_wrench_caps_and_speed_bounds():
    """Defaults track stem wrench caps (scene) and loosened speed bounds."""
    from apple_pick_sim.coupled_fruiting.scene import (
        DEFAULT_STEM_FORCE_CAP_N,
        DEFAULT_STEM_TORQUE_CAP_NM,
    )

    t = StabilityThresholds()
    assert t.max_force_n == pytest.approx(DEFAULT_STEM_FORCE_CAP_N)
    assert t.max_torque_nm == pytest.approx(DEFAULT_STEM_TORQUE_CAP_NM)
    assert DEFAULT_STEM_FORCE_CAP_N == pytest.approx(50.0)
    assert DEFAULT_STEM_TORQUE_CAP_NM == pytest.approx(20.0)
    assert t.max_tcp_speed_mps == pytest.approx(5.0)
    assert t.max_apple_speed_mps == pytest.approx(5.0)


def test_force_cap_exceeded():
    obs = _nominal_obs(num_envs=2)
    obs["ft_wrist"] = obs["ft_wrist"].clone()
    obs["ft_wrist"][0, 0] = 2500.0
    report = _monitor(num_envs=2).check(obs, step_idx=0)
    assert report.unstable.tolist() == [True, False]
    assert "force_cap_exceeded" in report.reasons[0]


def test_torque_cap_exceeded():
    obs = _nominal_obs(num_envs=2)
    obs["ft_wrist"] = obs["ft_wrist"].clone()
    obs["ft_wrist"][1, 3] = 600.0
    report = _monitor(num_envs=2).check(obs, step_idx=0)
    assert report.unstable.tolist() == [False, True]
    assert "torque_cap_exceeded" in report.reasons[1]


def test_tcp_speed_exceeded():
    obs = _nominal_obs(num_envs=2)
    obs["tcp_velocity"] = obs["tcp_velocity"].clone()
    obs["tcp_velocity"][0, :3] = 15.0
    report = _monitor(num_envs=2).check(obs, step_idx=0)
    assert report.unstable.tolist() == [True, False]
    assert "tcp_speed_exceeded" in report.reasons[0]


def test_vectorization_multiple_envs_different_reasons():
    obs = _nominal_obs(num_envs=5)
    obs["ft_wrist"] = obs["ft_wrist"].clone()
    obs["tcp_velocity"] = obs["tcp_velocity"].clone()
    obs["ft_wrist"][1, 0] = 2500.0
    obs["tcp_velocity"][3, 0] = 15.0
    report = _monitor(num_envs=5).check(obs, step_idx=0)
    assert report.unstable.tolist() == [False, True, False, True, False]
    assert "force_cap_exceeded" in report.reasons[1]
    assert "tcp_speed_exceeded" in report.reasons[3]
    assert report.reasons[0] == []
    assert report.reasons[2] == []
    assert report.reasons[4] == []


@dataclass
class _FakePlugin:
    name: str = "fake_plugin"
    required_obs_keys: frozenset[str] = frozenset({"tcp_velocity"})

    def check(self, obs: Mapping[str, Any], *, step_idx: int):
        from apple_pick_gym.batched_envs.batched_stability_monitor import PluginCheckResult

        n = int(obs["tcp_velocity"].shape[0])
        unstable = torch.zeros(n, dtype=torch.bool)
        unstable[2] = True
        reasons: list[str | None] = [None] * n
        reasons[2] = "fake_reason"
        return PluginCheckResult(unstable=unstable, reasons=reasons)


def test_plugin_merging_flags_env_two():
    obs = _nominal_obs(num_envs=4)
    report = _monitor(num_envs=4, plugins=[_FakePlugin()]).check(obs, step_idx=0)
    assert report.unstable.tolist() == [False, False, True, False]
    assert report.reasons[2] == ["fake_reason"]


def test_fail_fast_construction_missing_plugin_obs_keys():
    with pytest.raises(ValueError, match="phase"):
        BatchedStabilityMonitor(
            2,
            known_obs_keys={"ft_wrist", "tcp_velocity"},
            plugins=[_PluginRequiringPhase()],
        )


@dataclass
class _PluginRequiringPhase:
    name: str = "hold_quality"
    required_obs_keys: frozenset[str] = frozenset({"phase"})

    def check(self, obs: Mapping[str, Any], *, step_idx: int):
        raise NotImplementedError


def test_threshold_override_flags_lower_force():
    obs = _nominal_obs(num_envs=1)
    obs["ft_wrist"] = obs["ft_wrist"].clone()
    obs["ft_wrist"][0, 0] = 15.0
    monitor = _monitor(
        num_envs=1,
        thresholds=StabilityThresholds(max_force_n=10.0),
    )
    report = monitor.check(obs, step_idx=0)
    assert bool(report.unstable[0])
    assert "force_cap_exceeded" in report.reasons[0]


def test_initial_unstable_sticky_across_checks():
    obs = _nominal_obs(num_envs=3)
    initial = torch.tensor([False, True, False], dtype=torch.bool)
    monitor = _monitor(
        num_envs=3,
        initial_unstable=initial,
        initial_reason="ik_bootstrap_not_converged",
    )
    report0 = monitor.check(obs, step_idx=0)
    report1 = monitor.check(obs, step_idx=1)
    assert report0.unstable.tolist() == [False, True, False]
    assert report1.unstable.tolist() == [False, True, False]
    assert "ik_bootstrap_not_converged" in report0.reasons[1]
    assert "ik_bootstrap_not_converged" in report1.reasons[1]
    assert report0.reasons[0] == []
    assert report0.reasons[2] == []


def test_initial_unstable_none_default_unchanged():
    obs = _nominal_obs(num_envs=2)
    report = _monitor(num_envs=2).check(obs, step_idx=0)
    assert not bool(report.unstable.any())
    assert report.reasons == [[], []]


def test_ik_bootstrap_unstable_mask_from_build_result():
    class _BuildResult:
        ik_envelope_results = (
            (0.01, 0.01, True),
            (0.10, 0.10, False),
            (0.02, 0.02, True),
        )

    class _Sim:
        build_result = _BuildResult()

    class _Env:
        _sim = _Sim()

    mask = ik_bootstrap_unstable_mask(_Env(), num_envs=3)
    assert mask.tolist() == [False, True, False]


def test_ik_bootstrap_unstable_mask_missing_build_result():
    class _Env:
        pass

    mask = ik_bootstrap_unstable_mask(_Env(), num_envs=2)
    assert mask.tolist() == [False, False]


def test_apple_speed_exceeded_on_second_check_only():
    obs0 = _nominal_obs(num_envs=1)
    obs0["apple_pos"] = torch.zeros(1, 3, dtype=torch.float32)
    monitor = _monitor(num_envs=1, thresholds=StabilityThresholds(max_apple_speed_mps=1.0))

    report0 = monitor.check(obs0, step_idx=0)
    assert not bool(report0.unstable.any())

    obs1 = _nominal_obs(num_envs=1)
    obs1["apple_pos"] = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)
    report1 = monitor.check(obs1, step_idx=1)
    assert bool(report1.unstable[0])
    assert "apple_speed_exceeded" in report1.reasons[0]


def test_hard_blowup_mask_ignores_force_cap_keeps_nan():
    import torch
    from apple_pick_gym.batched_envs.batched_stability_monitor import (
        BatchedStabilityReport,
        hard_blowup_mask,
    )

    report = BatchedStabilityReport(
        step_idx=0,
        unstable=torch.tensor([True, True, False]),
        reasons=[
            ["force_cap_exceeded"],
            ["nan_or_inf:ft_wrist"],
            [],
        ],
    )
    assert hard_blowup_mask(report).tolist() == [False, True, False]
