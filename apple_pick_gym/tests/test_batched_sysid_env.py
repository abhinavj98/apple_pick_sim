"""Tests for ApplePickBatchedSysIdEnv (V.4.2 collection env)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv
from apple_pick_gym.batched_envs.apple_pick_batched_base_env import ApplePickBatchedBaseEnv
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    ObsConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.system_id.excitation_state import ExcitationContext
from apple_pick_sim.tests.conftest import RANGES_FIXTURE, fr3_assets_available

_SEED = 42


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed",
)

requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)


def _test_sim_config(*, num_envs: int) -> BatchedHeterogeneousCoupledSimConfig:
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            skip_ik_bootstrap=False,
            defer_template_robot_bootstrap=False,
            force_batched_layout=True,
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=8),
        controller=ControllerConfig(mode="vic", linear_speed=1.0, angular_speed=1.0),
        domain_randomization=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).domain_randomization,
            topology_seed=_SEED,
        ),
        obs=ObsConfig(allocate_buffers=True),
    )


def _make_env(*, num_envs: int = 1) -> ApplePickBatchedSysIdEnv:
    return ApplePickBatchedSysIdEnv(
        num_envs=num_envs,
        max_episode_steps=10,
        ranges_path=RANGES_FIXTURE,
        topology_seed=_SEED,
        use_settle_cache=False,
        sim_config=_test_sim_config(num_envs=num_envs),
    )


_SYSID_OBS_KEYS = frozenset(
    {
        "woody_part_start_pos",
        "woody_part_end_pos",
        "woody_part_force",
        "apple_pos",
        "apple_quat",
        "tcp_pos",
        "tcp_quat",
        "tcp_velocity",
        "ft_wrist",
        "raw_ft_wrist",
        "robot_joint_q",
        "excitation_type",
        "excitation_f_inst",
        "excitation_direction",
    }
)


def test_batched_sysid_env_forwards_per_env_grippers(monkeypatch):
    per_env_grippers = (object(), object())
    captured: dict[str, object] = {}

    def _fake_base_init(self, **kwargs):
        captured.update(kwargs)
        self.num_envs = int(kwargs["num_envs"])
        self.device = torch.device("cpu")

    monkeypatch.setattr(ApplePickBatchedBaseEnv, "__init__", _fake_base_init)

    ApplePickBatchedSysIdEnv(
        num_envs=2,
        per_env_grippers=per_env_grippers,
        use_settle_cache=False,
    )

    assert captured["per_env_grippers"] is per_env_grippers


def test_batched_sysid_env_preserves_sim_config_control_hz(monkeypatch):
    """Env must not overwrite sim_config.runtime.control_hz with default 60."""
    captured: dict[str, object] = {}

    def _fake_base_init(self, **kwargs):
        captured.update(kwargs)
        self.num_envs = int(kwargs["num_envs"])
        self.device = torch.device("cpu")

    monkeypatch.setattr(ApplePickBatchedBaseEnv, "__init__", _fake_base_init)

    base = _test_sim_config(num_envs=1)
    cfg = dataclasses.replace(
        base,
        runtime=dataclasses.replace(base.runtime, control_hz=15.0),
    )
    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        sim_config=cfg,
        use_settle_cache=False,
    )
    assert env._control_hz == pytest.approx(15.0)
    assert captured["sim_config"].runtime.control_hz == pytest.approx(15.0)


def test_batched_sysid_env_explicit_control_hz_overrides_sim_config(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_base_init(self, **kwargs):
        captured.update(kwargs)
        self.num_envs = int(kwargs["num_envs"])
        self.device = torch.device("cpu")

    monkeypatch.setattr(ApplePickBatchedBaseEnv, "__init__", _fake_base_init)

    base = _test_sim_config(num_envs=1)
    cfg = dataclasses.replace(
        base,
        runtime=dataclasses.replace(base.runtime, control_hz=15.0),
    )
    env = ApplePickBatchedSysIdEnv(
        num_envs=1,
        sim_config=cfg,
        control_hz=30.0,
        use_settle_cache=False,
    )
    assert env._control_hz == pytest.approx(30.0)
    assert captured["sim_config"].runtime.control_hz == pytest.approx(30.0)


@gymnasium_available
@requires_fr3
def test_batched_sysid_obs_shapes_and_sysid_numpy_export():
    env = _make_env(num_envs=2)
    try:
        obs, info = env.reset(seed=_SEED)
        assert info["obs_layout"] == "batched_sysid"
        assert obs["tcp_pos"].shape == (2, 3)
        assert obs["tcp_quat"].shape == (2, 4)
        assert obs["apple_quat"].shape == (2, 4)
        assert obs["robot_joint_q"].shape[0] == 2
        assert obs["excitation_direction"].shape == (2, 3)
        assert len(info["per_env"]) == 2
        assert len(info["weld_direction"]) == 2

        exported = env.sysid_numpy_obs(1)
        assert _SYSID_OBS_KEYS <= frozenset(exported)
        assert exported["ft_wrist"].shape == (6,)
        assert exported["excitation_direction"].shape == (3,)
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_batched_sysid_excitation_context_round_trip():
    env = _make_env(num_envs=2)
    try:
        env.reset(seed=_SEED)
        direction = np.array([0.2, -0.3, 0.9], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        ctx = ExcitationContext(type="quasi_static", f_inst=0.0, direction=direction)
        env.set_excitation_context(1, ctx)
        obs = env._gather_obs()
        np.testing.assert_allclose(
            obs["excitation_direction"][1].detach().cpu().numpy(),
            direction.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )
        assert int(obs["excitation_type"][1].item()) == 0
        exported = env.sysid_numpy_obs(1)
        np.testing.assert_allclose(
            exported["excitation_direction"],
            direction.astype(np.float32),
            rtol=0,
            atol=1e-6,
        )
    finally:
        env.close()


@gymnasium_available
@requires_fr3
def test_batched_sysid_step_preserves_export_contract():
    env = _make_env(num_envs=1)
    try:
        env.reset(seed=_SEED)
        actions = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
        actions[0, 0] = 0.05
        obs, *_rest = env.step(actions)
        exported = env.sysid_numpy_obs(0)
        assert _SYSID_OBS_KEYS <= frozenset(exported)
        assert np.isfinite(exported["ft_wrist"]).all()
        assert obs["tcp_pos"].shape == (1, 3)
    finally:
        env.close()


@gymnasium_available
def test_batched_base_env_close_drops_sim_and_reclaims_device_memory(monkeypatch):
    """close() must drop the sim AND force GC + Warp sync (naive ``_sim = None`` leaks)."""
    import apple_pick_gym.batched_envs.apple_pick_batched_base_env as base_mod

    gc_calls: list[int] = []
    sync_calls: list[int] = []
    monkeypatch.setattr(base_mod.gc, "collect", lambda: gc_calls.append(1) or 0)
    monkeypatch.setattr(base_mod.wp, "synchronize", lambda: sync_calls.append(1))

    class _Holder:
        def __init__(self) -> None:
            self._sim = object()

    env = _Holder()
    ApplePickBatchedBaseEnv.close(env)
    assert env._sim is None
    assert gc_calls == [1]
    assert sync_calls == [1]


@gymnasium_available
def test_batched_base_env_close_releases_obs_aliases_before_dropping_sim(monkeypatch):
    """Torch obs from wp.to_torch aliases Warp storage; release before freeing ``_sim``."""
    import apple_pick_gym.batched_envs.apple_pick_batched_base_env as base_mod

    order: list[str] = []
    monkeypatch.setattr(base_mod.gc, "collect", lambda: order.append("gc") or 0)
    monkeypatch.setattr(base_mod.wp, "synchronize", lambda: order.append("sync"))

    class _Holder:
        def __init__(self) -> None:
            self._sim = object()
            self._last_obs: dict[str, object] | None = {"ft_wrist": object()}

        def _release_observation_aliases(self) -> None:
            order.append("release")
            assert self._sim is not None
            self._last_obs = None

    env = _Holder()
    ApplePickBatchedBaseEnv.close(env)
    assert env._sim is None
    assert env._last_obs is None
    assert order == ["sync", "release", "gc"]


@gymnasium_available
def test_batched_base_env_release_observation_aliases_clears_last_obs():
    """The base default must drop obs aliases; an empty body silently pins Warp buffers."""

    class _Holder:
        def __init__(self) -> None:
            self._last_obs: dict[str, object] | None = {"ft_wrist": object()}

    env = _Holder()
    ApplePickBatchedBaseEnv._release_observation_aliases(env)
    assert env._last_obs is None


@gymnasium_available
def test_batched_base_env_close_synchronizes_before_dropping_sim(monkeypatch):
    """Queued Warp work must finish before ``_sim`` arrays are collected."""
    import apple_pick_gym.batched_envs.apple_pick_batched_base_env as base_mod

    order: list[str] = []
    monkeypatch.setattr(
        base_mod.wp,
        "synchronize",
        lambda: order.append("sync") or order.append(f"sim={env._sim is not None}"),
    )
    monkeypatch.setattr(base_mod.gc, "collect", lambda: order.append("gc") or 0)

    class _Holder:
        def __init__(self) -> None:
            self._sim = object()

    env = _Holder()
    ApplePickBatchedBaseEnv.close(env)
    assert env._sim is None
    assert order[0] == "sync"
    assert order[1] == "sim=True"
    assert order[-1] == "gc"


@gymnasium_available
def test_batched_base_env_close_does_not_call_clear_kernel_cache(monkeypatch):
    """close() must not call wp.clear_kernel_cache (too slow; does not fix CMA leak)."""
    import apple_pick_gym.batched_envs.apple_pick_batched_base_env as base_mod

    def _forbidden_clear_kernel_cache() -> None:
        raise AssertionError("close() must not call wp.clear_kernel_cache")

    monkeypatch.setattr(base_mod.wp, "clear_kernel_cache", _forbidden_clear_kernel_cache)
    monkeypatch.setattr(base_mod.gc, "collect", lambda: 0)
    monkeypatch.setattr(base_mod.wp, "synchronize", lambda: None)

    class _Holder:
        def __init__(self) -> None:
            self._sim = object()

    env = _Holder()
    ApplePickBatchedBaseEnv.close(env)
    assert env._sim is None


def test_batched_sysid_env_release_observation_aliases_clears_last_obs():
    env = ApplePickBatchedSysIdEnv.__new__(ApplePickBatchedSysIdEnv)
    env._last_obs = {"ft_wrist": object()}
    ApplePickBatchedSysIdEnv._release_observation_aliases(env)
    assert env._last_obs is None


def test_sysid_gather_obs_reads_apple_quat_from_apple_pose_not_body_q_numpy(monkeypatch):
    import warp as wp

    numpy_calls: list[str] = []

    class _ForbiddenBodyQ:
        def numpy(self):
            numpy_calls.append("body_q")
            raise AssertionError("apple quat must not download fused body_q")

    apple_pose = wp.zeros((2, 7), dtype=wp.float32, device="cpu")
    pose_np = np.zeros((2, 7), dtype=np.float32)
    pose_np[1] = [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, 0.9]
    apple_pose.assign(pose_np)
    tcp_pose = wp.zeros((2, 7), dtype=wp.float32, device="cpu")
    joint_q = wp.zeros((2, 7), dtype=wp.float32, device="cpu")

    env = ApplePickBatchedSysIdEnv.__new__(ApplePickBatchedSysIdEnv)
    env.device = torch.device("cpu")
    env.num_envs = 2
    env._excitation_type = torch.zeros(2, dtype=torch.long)
    env._excitation_f_inst = torch.zeros(2, dtype=torch.float32)
    env._excitation_direction = torch.zeros(2, 3, dtype=torch.float32)
    env._sim = SimpleNamespace(
        obs_bufs=SimpleNamespace(
            apple_pose=apple_pose,
            tcp_pose=tcp_pose,
            joint_q=joint_q,
        ),
        scene=SimpleNamespace(
            cable=SimpleNamespace(state_0=SimpleNamespace(body_q=_ForbiddenBodyQ()))
        ),
    )
    base_obs = {
        "apple_pos": torch.zeros(2, 3),
        "ft_wrist": torch.zeros(2, 6),
        "tcp_force": torch.zeros(2, 6),
        "tcp_velocity": torch.zeros(2, 6),
        "woody_part_info": {},
    }
    monkeypatch.setattr(
        ApplePickBatchedBaseEnv,
        "_gather_obs",
        lambda self: dict(base_obs),
    )

    obs = ApplePickBatchedSysIdEnv._gather_obs(env)
    assert numpy_calls == []
    torch.testing.assert_close(
        obs["apple_quat"][1],
        torch.tensor([0.1, 0.2, 0.3, 0.9], dtype=torch.float32),
    )


@gymnasium_available
@requires_fr3
@pytest.mark.slow
def test_batched_sysid_env_close_clears_last_obs_and_tracks_host_heap():
    """After close, drop wp.to_torch obs aliases and record host/wp growth across rebuilds."""
    import gc
    import warp as wp

    def _vm_rss_mb() -> float:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        return -1.0

    def _wp_array_count() -> int:
        count = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, wp.array):
                    count += 1
            except Exception:
                continue
        return count

    gc.collect()
    wp.synchronize()
    rss_series: list[float] = []
    wp_series: list[int] = []

    for cycle in range(5):
        env = _make_env(num_envs=2)
        env.reset(seed=_SEED + cycle)
        actions = torch.zeros((2, 6), dtype=torch.float32, device=env.device)
        env.step(actions)
        assert env._last_obs is not None
        env.close()
        assert env._last_obs is None
        del env
        gc.collect()
        wp.synchronize()
        rss_series.append(_vm_rss_mb())
        wp_series.append(_wp_array_count())

    # First rebuild pays one-time Warp/MuJoCo module cost; later cycles must not
    # compound via retained ``_last_obs`` aliases. Allow residual Warp Var growth
    # (upstream codegen cache) but keep host RSS from runaway doubling.
    later_rss = rss_series[2:]
    assert max(later_rss) < later_rss[0] * 2.0 + 256.0
    later_wp = wp_series[2:]
    per_cycle = [later_wp[i] - later_wp[i - 1] for i in range(1, len(later_wp))]
    assert per_cycle, "expected multiple post-warmup rebuild samples"
    assert max(per_cycle) < 5000, (
        f"wp.array growth per rebuild too large: {per_cycle} (series={wp_series})"
    )
