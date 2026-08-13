"""Unit tests for shared real-replay env build helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apple_pick_sim.fruiting_system.params import load_ranges

_VARIANCE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)


def test_dataset_declares_vic_pose_from_layout_or_dim():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        dataset_declares_vic_pose,
    )

    assert dataset_declares_vic_pose({"action_layout": "vic_pose_v1"}) is True
    assert dataset_declares_vic_pose({"action_dim": 19}) is True
    assert dataset_declares_vic_pose({"action_dim": 6}) is False


def test_check_action_semantics_refuses_wrench_as_twist_on_vic_pose():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        check_action_semantics,
    )

    with pytest.raises(SystemExit, match="legacy 6D"):
        check_action_semantics(
            controller_mode="vic",
            collection={"action_dim": 19, "action_layout": "vic_pose_v1"},
            episode_meta={},
            allow_wrench_as_twist=True,
        )


def test_allow_wrench_as_twist_rejects_pose_packed_dataset():
    """The escape hatch is legacy-6D only; 19D vic_pose datasets must fail fast."""
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        check_action_semantics,
    )

    collection = {"action_dim": 19, "action_layout": "vic_pose_v1"}
    with pytest.raises(SystemExit, match="legacy 6D"):
        check_action_semantics(
            controller_mode="vic",
            collection=collection,
            episode_meta={"action_compatible_with_vic_twist": False},
            allow_wrench_as_twist=True,
        )


def test_action_semantics_refuses_wrench_marked_dataset_under_vic():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        check_action_semantics,
    )

    with pytest.raises(SystemExit, match="vic_pose"):
        check_action_semantics(
            controller_mode="vic",
            collection={"action_dim": 6},
            episode_meta={"action_compatible_with_vic_twist": False},
            allow_wrench_as_twist=False,
        )


def test_action_semantics_allows_legacy_6d_hatch_and_vic_pose_mode():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        check_action_semantics,
    )

    check_action_semantics(
        controller_mode="vic",
        collection={"action_dim": 6},
        episode_meta={"action_compatible_with_vic_twist": False},
        allow_wrench_as_twist=True,
    )
    check_action_semantics(
        controller_mode="vic_pose",
        collection={"action_dim": 19, "action_layout": "vic_pose_v1"},
        episode_meta={"action_compatible_with_vic_twist": False},
        allow_wrench_as_twist=False,
    )


def test_make_real_replay_build_env_fn_honors_per_env_grippers(monkeypatch):
    """Factory must pass fused per_env_grippers into ApplePickBatchedSysIdEnv, not delete them."""
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig

    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        make_real_replay_build_env_fn,
    )

    captured = {}

    class _FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._sim = SimpleNamespace(scene=SimpleNamespace(cable=None, layout=None))

    monkeypatch.setattr(
        "apple_pick_gym.batched_envs.real_batched_replay_build.ApplePickBatchedSysIdEnv",
        _FakeEnv,
    )
    g0 = GripperProxyConfig()
    g1 = GripperProxyConfig()
    meta = {
        "control_hz": 15.0,
        "fruiting_base_pos": [0.0, 0.5, 0.95],
        "initial_robot_joint_q": [0.0] * 7,
        "initial_apple_pos": [0.1, 0.2, 0.3],
        "initial_apple_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_tcp_pos": [0.1, 0.15, 0.3],
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
    }
    fn = make_real_replay_build_env_fn(
        ranges_path=_VARIANCE,
        ranges=load_ranges(_VARIANCE),
        topology_seed=0,
        fruiting_base_pos=(0.0, 0.5, 0.95),
        episode_meta=meta,
        bootstrap_joint_q=(0.0,) * 7,
        controller_mode="vic_pose",
        control_hz=15.0,
    )
    fn(num_envs=2, per_env_params=[None, None], max_episode_steps=4, per_env_grippers=[g0, g1])
    assert captured["per_env_grippers"] == [g0, g1]


def test_bootstrap_joint_q_from_episode_metadata():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        bootstrap_joint_q_from_episode_metadata,
    )

    meta = {"initial_robot_joint_q": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    q = bootstrap_joint_q_from_episode_metadata(meta)
    assert q == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    with pytest.raises(ValueError, match="initial_robot_joint_q"):
        bootstrap_joint_q_from_episode_metadata({})


def test_control_hz_from_episode_metadata_prefers_episode_then_collection():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        control_hz_from_episode_metadata,
    )

    assert control_hz_from_episode_metadata({"control_hz": 15.0}) == pytest.approx(15.0)
    assert control_hz_from_episode_metadata(
        {}, collection={"control_hz": 12.5}
    ) == pytest.approx(12.5)
    assert control_hz_from_episode_metadata(
        {"control_hz": 15.0}, collection={"control_hz": 12.5}
    ) == pytest.approx(15.0)
    with pytest.raises(ValueError, match="control_hz"):
        control_hz_from_episode_metadata({})
    with pytest.raises(ValueError, match="control_hz"):
        control_hz_from_episode_metadata({"control_hz": 0.0})


def test_fruiting_base_pos_from_episode_metadata():
    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        fruiting_base_pos_from_episode_metadata,
    )

    meta = {"fruiting_base_pos": [0.1, 0.2, 0.3]}
    assert fruiting_base_pos_from_episode_metadata(meta) == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="fruiting_base_pos"):
        fruiting_base_pos_from_episode_metadata({})


def test_real_replay_sim_config_applies_vic_pose_and_control_hz():
    if not _VARIANCE.is_file():
        pytest.skip(f"missing {_VARIANCE}")

    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        real_replay_sim_config,
    )

    ranges = load_ranges(_VARIANCE)
    q = (0.1, 0.2, 0.3, -1.0, 0.0, 1.5, -0.5)
    cfg = real_replay_sim_config(
        num_envs=1,
        topology_seed=0,
        fruiting_base_pos=(0.117, 0.787, 0.577),
        ranges=ranges,
        bootstrap_joint_q=q,
        controller_mode="vic_pose",
        control_hz=15.0,
    )
    assert cfg.controller.mode == "vic_pose"
    assert cfg.controller.action_dim == 19
    assert cfg.robot.per_env_ik is False
    assert cfg.robot.bootstrap_joint_q == q
    assert cfg.scene.fruiting_base_pos == (0.117, 0.787, 0.577)
    assert cfg.scene.post_grasp_settle_substeps == 500
    assert cfg.runtime.control_hz == pytest.approx(15.0)
