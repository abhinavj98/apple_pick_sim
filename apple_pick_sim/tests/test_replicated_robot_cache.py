"""Unit tests for in-process FR3 USD / SolverMuJoCo reuse."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apple_pick_sim.coupled_fruiting.replicated_robot_cache import (
    CachedReplicatedRobot,
    ReplicatedRobotMuJoCoCache,
    acquire_replicated_fr3_robot,
    clear_process_replicated_robot_cache,
    make_replicated_robot_cache_key,
    process_replicated_robot_cache,
)


def test_cache_get_or_create_calls_factory_once_for_same_key():
    cache = ReplicatedRobotMuJoCoCache()
    factory_calls = []

    def factory():
        factory_calls.append(1)
        entry = SimpleNamespace(restore_calls=[])
        entry.restore_rest_pose = lambda: entry.restore_calls.append(1)
        return entry

    key = ("cpu", 4)
    first = cache.get_or_create(key, factory)
    second = cache.get_or_create(key, factory)
    assert first is second
    assert factory_calls == [1]
    assert first.restore_calls == [1]
    assert cache.misses == 1
    assert cache.hits == 1


def test_cache_misses_on_distinct_keys():
    cache = ReplicatedRobotMuJoCoCache()
    n = {"calls": 0}

    def factory():
        n["calls"] += 1
        return SimpleNamespace(n=n["calls"], restore_rest_pose=lambda: None)

    a = cache.get_or_create(("cpu", 2), factory)
    b = cache.get_or_create(("cpu", 8), factory)
    assert a is not b
    assert n["calls"] == 2


def test_acquire_skips_cache_when_reuse_disabled():
    cache = ReplicatedRobotMuJoCoCache()
    n = {"calls": 0}

    def factory():
        n["calls"] += 1
        return SimpleNamespace(n=n["calls"])

    first = acquire_replicated_fr3_robot(
        reuse=False, key=("cpu", 4), factory=factory, cache=cache
    )
    second = acquire_replicated_fr3_robot(
        reuse=False, key=("cpu", 4), factory=factory, cache=cache
    )
    assert first is not second
    assert n["calls"] == 2
    assert cache.hits == 0
    assert cache.misses == 0


def test_acquire_uses_process_cache_when_reuse_enabled():
    clear_process_replicated_robot_cache()
    n = {"calls": 0}

    def factory():
        n["calls"] += 1
        return SimpleNamespace(n=n["calls"], restore_rest_pose=lambda: None)

    key = make_replicated_robot_cache_key(
        num_envs=4,
        device="cpu",
        usd_path=None,
        add_apple_payload=True,
        robot_base_pos=(0.0, 0.0, 0.0),
        mujoco_kwargs={"separate_worlds": True, "use_mujoco_cpu": True},
    )
    first = acquire_replicated_fr3_robot(reuse=True, key=key, factory=factory)
    second = acquire_replicated_fr3_robot(reuse=True, key=key, factory=factory)
    assert first is second
    assert n["calls"] == 1
    assert process_replicated_robot_cache().hits == 1
    clear_process_replicated_robot_cache()


def test_cached_robot_restore_rest_pose_assigns_and_broadcasts(monkeypatch):
    assigned: list[tuple[str, np.ndarray]] = []

    class _Arr:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float32)

        def numpy(self):
            return self._values.copy()

        def assign(self, values):
            assigned.append((id(self), np.asarray(values)))
            self._values = np.asarray(values, dtype=np.float32)

    template = SimpleNamespace(joint_q=_Arr([1.0, 2.0]), joint_qd=_Arr([0.0, 0.0]))
    batched = SimpleNamespace(joint_q=_Arr([9.0, 9.0]), joint_qd=_Arr([1.0, 1.0]))
    broadcasts: list[tuple[object, object]] = []
    monkeypatch.setattr(
        "apple_pick_sim.coupled_fruiting.batched_build._broadcast_robot_state_from_template",
        lambda tpl, batched_model: broadcasts.append((tpl, batched_model)),
    )
    entry = CachedReplicatedRobot(
        robot_model=batched,
        template_tcp=3,
        mj_solver=object(),
        template_model=template,
        rest_joint_q=np.array([0.1, 0.2], dtype=np.float32),
        rest_joint_qd=np.array([0.0, 0.0], dtype=np.float32),
    )
    entry.restore_rest_pose()
    assert broadcasts == [(template, batched)]
    assert assigned[0][1].tolist() == pytest.approx([0.1, 0.2])


def test_build_fr3_robot_model_from_usd_skips_solver_when_requested(monkeypatch):
    from apple_pick_sim.robot.fr3_robot import setup as setup_mod

    class FakeModel:
        body_label = ["base", "arm/ee", "arm/ee/tcp"]

        def set_gravity(self, _g):
            return None

    class FakeBuilder:
        def finalize(self, device=None):
            del device
            return FakeModel()

    solver_calls: list[int] = []

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            solver_calls.append(1)

    monkeypatch.setattr(setup_mod, "fr3_assets_available", lambda: True)
    monkeypatch.setattr(
        setup_mod, "build_fr3_robot_builder", lambda **_k: (FakeBuilder(), 2)
    )
    monkeypatch.setattr(setup_mod, "SolverMuJoCo", FakeSolver)
    monkeypatch.setattr(setup_mod, "resolve_sim_device", lambda d: d or "cpu")
    monkeypatch.setattr(setup_mod, "resolve_tcp_body_index", lambda _m: 2)

    _model, tcp, solver = setup_mod.build_fr3_robot_model_from_usd(
        device="cpu",
        create_solver=False,
    )
    assert tcp == 2
    assert solver is None
    assert solver_calls == []
