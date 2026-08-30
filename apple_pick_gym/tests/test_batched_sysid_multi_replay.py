"""Typed planning and fused execution tests for multi-structure sys-ID replay."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs import batched_sysid_multi_replay as multi
from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import GripperProxyConfig
from apple_pick_sim.tests.conftest import RANGES_FIXTURE


@dataclasses.dataclass(frozen=True)
class _Candidate:
    marker: float
    support_kp: float | None = None

    def apply_to(self, base: fs.FruitingSystemParams) -> fs.FruitingSystemParams:
        return dataclasses.replace(base, apple_density=float(self.marker))


def _params(seed: int = 0) -> fs.FruitingSystemParams:
    return fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=seed)


def _recorded(
    structure_idx: int,
    direction_idx: int,
    *,
    frames: int = 3,
    action_dim: int = 6,
    junction_names: tuple[str, ...] = ("support", "primary_spur", "spur_stem"),
) -> dict[str, Any]:
    sentinel = 10000 * structure_idx + direction_idx
    return {
        "action": np.full((frames, action_dim), sentinel, dtype=np.float32),
        "junction_names": list(junction_names),
        "recorded_sentinel": sentinel,
    }


def _request(
    structure_idx: int,
    *,
    candidates: tuple[_Candidate, ...] = (_Candidate(10.0),),
    directions: tuple[int, ...] = (0, 2),
    params: fs.FruitingSystemParams | None = None,
    gripper: GripperProxyConfig | None = None,
    frames: int = 3,
    action_dim: int = 6,
    junction_names: tuple[str, ...] = ("support", "primary_spur", "spur_stem"),
    meta_by_direction: dict[int, dict[str, Any]] | None = None,
) -> multi.ReplayStructureRequest:
    kwargs: dict[str, Any] = {
        "structure_idx": structure_idx,
        "candidates": candidates,
        "direction_indices": directions,
        "base_params": _params(structure_idx) if params is None else params,
        "recorded_by_direction": {
            direction_idx: _recorded(
                structure_idx,
                direction_idx,
                frames=frames,
                action_dim=action_dim,
                junction_names=junction_names,
            )
            for direction_idx in directions
        },
        "gripper": GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        )
        if gripper is None
        else gripper,
    }
    if meta_by_direction is not None:
        kwargs["meta_by_direction"] = meta_by_direction
    return multi.ReplayStructureRequest(**kwargs)


def _real_dir_meta(*, apple_pos: tuple[float, float, float]) -> dict[str, Any]:
    tcp_pos = (apple_pos[0], apple_pos[1] - 0.05, apple_pos[2])
    return {
        "weld_direction": [0.0, -1.0, 0.0],
        "initial_apple_pos": list(apple_pos),
        "initial_apple_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_tcp_pos": list(tcp_pos),
        "initial_tcp_quat": [0.0, 0.0, 0.0, 1.0],
        "weld_reference_pos": list(apple_pos),
        "weld_reference_quat": [0.0, 0.0, 0.0, 1.0],
        "initial_robot_joint_q": [0.1, 0.2, 0.3, -1.0, 0.0, 1.5, -0.5],
    }


def test_build_replay_candidate_blocks_preserves_original_stable_identity_and_payloads():
    gripper_4 = GripperProxyConfig(
        fix_to_apple=True,
        weld_direction=(1.0, 0.0, 0.0),
        weld_reference_pos=(4.0, 0.0, 0.0),
    )
    gripper_1 = dataclasses.replace(
        gripper_4,
        weld_direction=(0.0, 1.0, 0.0),
        weld_reference_pos=(1.0, 0.0, 0.0),
    )
    params_4 = _params(4)
    params_1 = dataclasses.replace(
        _params(1),
        topology=params_4.topology,
        primary=dataclasses.replace(
            _params(1).primary,
            num_segments=params_4.primary.num_segments,
        ),
        secondary=dataclasses.replace(
            _params(1).secondary,
            num_segments=params_4.secondary.num_segments,
        ),
        spur=dataclasses.replace(
            _params(1).spur,
            num_segments=params_4.spur.num_segments,
        ),
        stem=dataclasses.replace(
            _params(1).stem,
            num_segments=params_4.stem.num_segments,
        ),
    )
    requests = (
        _request(
            4,
            candidates=(_Candidate(40.0), _Candidate(41.0)),
            params=params_4,
            gripper=gripper_4,
        ),
        _request(
            1,
            candidates=(_Candidate(10.0),),
            params=params_1,
            gripper=gripper_1,
        ),
    )

    blocks = multi.build_replay_candidate_blocks(requests)

    assert [[slot.key for slot in block.slots] for block in blocks] == [
        [
            multi.ReplaySlotKey(
                structure_idx=4,
                local_candidate_idx=0,
                direction_idx=0,
            ),
            multi.ReplaySlotKey(
                structure_idx=4,
                local_candidate_idx=0,
                direction_idx=2,
            ),
        ],
        [
            multi.ReplaySlotKey(
                structure_idx=4,
                local_candidate_idx=1,
                direction_idx=0,
            ),
            multi.ReplaySlotKey(
                structure_idx=4,
                local_candidate_idx=1,
                direction_idx=2,
            ),
        ],
        [
            multi.ReplaySlotKey(
                structure_idx=1,
                local_candidate_idx=0,
                direction_idx=0,
            ),
            multi.ReplaySlotKey(
                structure_idx=1,
                local_candidate_idx=0,
                direction_idx=2,
            ),
        ],
    ]
    assert len({slot.key for block in blocks for slot in block.slots}) == 6
    for block in blocks:
        request = next(r for r in requests if r.structure_idx == block.structure_idx)
        for slot in block.slots:
            assert slot.params.apple_density == request.candidates[
                slot.key.local_candidate_idx
            ].marker
            assert slot.recorded is request.recorded_by_direction[slot.key.direction_idx]
            assert slot.source == multi.ReplayEpisodeSource(
                structure_idx=slot.key.structure_idx,
                direction_idx=slot.key.direction_idx,
            )
            assert slot.gripper is request.gripper


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda r: dataclasses.replace(
                r,
                base_params=dataclasses.replace(
                    r.base_params,
                    topology=(
                        "t_junction"
                        if r.base_params.topology == "linear_chain"
                        else "linear_chain"
                    ),
                ),
            ),
            "topology",
        ),
        (lambda r: dataclasses.replace(r, base_params=dataclasses.replace(r.base_params, secondary=None)), "enabled rods"),
        (
            lambda r: dataclasses.replace(
                r,
                base_params=dataclasses.replace(
                    r.base_params,
                    primary=dataclasses.replace(
                        r.base_params.primary,
                        num_segments=r.base_params.primary.num_segments + 1,
                    ),
                ),
            ),
            "segment counts",
        ),
        (
            lambda r: dataclasses.replace(
                r,
                recorded_by_direction={
                    d: {**ep, "junction_names": list(reversed(ep["junction_names"]))}
                    for d, ep in r.recorded_by_direction.items()
                },
            ),
            "junction names",
        ),
        (
            lambda r: dataclasses.replace(
                r,
                recorded_by_direction={
                    d: _recorded(r.structure_idx, d, frames=4)
                    for d in r.direction_indices
                },
            ),
            "frame count",
        ),
        (lambda r: dataclasses.replace(r, direction_indices=(0, 3), recorded_by_direction={0: _recorded(r.structure_idx, 0), 3: _recorded(r.structure_idx, 3)}), "direction"),
        (lambda r: dataclasses.replace(r, gripper=dataclasses.replace(r.gripper, mass=r.gripper.mass + 1.0)), "gripper"),
        (lambda r: dataclasses.replace(r, gripper=dataclasses.replace(r.gripper, cylinder_radius=r.gripper.cylinder_radius + 0.01)), "gripper"),
    ],
)
def test_build_replay_candidate_blocks_distinguishes_valid_fusion_incompatibility(
    change,
    match,
):
    first = _request(4)
    second = change(_request(1, params=first.base_params))

    with pytest.raises(multi.ReplayFusionIncompatible, match=match):
        multi.build_replay_candidate_blocks((first, second))


def test_build_replay_candidate_blocks_accepts_different_recorded_weld_poses():
    first = _request(4)
    second = _request(
        1,
        params=first.base_params,
        gripper=dataclasses.replace(
            first.gripper,
            weld_direction=(0.0, 1.0, 0.0),
            weld_reference_pos=(9.0, 8.0, 7.0),
            weld_reference_quat=(0.0, 0.0, 1.0, 0.0),
            weld_reference_stem_dir=(0.0, 0.0, 1.0),
        ),
    )

    assert len(multi.build_replay_candidate_blocks((first, second))) == 2


def test_build_replay_candidate_blocks_accepts_19d_actions_and_rejects_mixed_widths():
    first = _request(4, action_dim=19)
    compatible = _request(1, params=first.base_params, action_dim=19)

    assert len(multi.build_replay_candidate_blocks((first, compatible))) == 2

    mixed_width = _request(1, params=first.base_params, action_dim=6)
    with pytest.raises(multi.ReplayFusionIncompatible, match="action width"):
        multi.build_replay_candidate_blocks((first, mixed_width))


@pytest.mark.parametrize(
    ("requests", "match"),
    [
        ((_request(4), _request(4)), "duplicate structure_idx"),
        ((dataclasses.replace(_request(4), candidates=()),), "candidates"),
        (
            (
                dataclasses.replace(
                    _request(4),
                    direction_indices=(0, 0),
                ),
            ),
            "duplicate direction",
        ),
        (
            (
                dataclasses.replace(
                    _request(4),
                    recorded_by_direction={0: _recorded(4, 0)},
                ),
            ),
            "exactly cover",
        ),
        (
            (
                dataclasses.replace(
                    _request(4),
                    recorded_by_direction={
                        0: {**_recorded(4, 0), "action": np.zeros((3, 5))},
                        2: _recorded(4, 2),
                    },
                ),
            ),
            "action shape",
        ),
    ],
)
def test_build_replay_candidate_blocks_rejects_malformed_requests(requests, match):
    with pytest.raises(ValueError, match=match) as exc_info:
        multi.build_replay_candidate_blocks(requests)
    assert not isinstance(exc_info.value, multi.ReplayFusionIncompatible)


def test_chunk_replay_candidate_blocks_is_deterministic_and_atomic():
    blocks = multi.build_replay_candidate_blocks(
        (
            _request(
                4,
                candidates=(_Candidate(40.0), _Candidate(41.0)),
            ),
            _request(1, params=_request(4).base_params),
        )
    )

    chunks = multi.chunk_replay_candidate_blocks(blocks, max_envs_per_batch=4)

    assert [[len(block.slots) for block in chunk] for chunk in chunks] == [[2, 2], [2]]
    # Unlimited env budget still keeps distinct structures in separate physical
    # chunks — cross-structure heterogeneous batches can collapse trajectories.
    unlimited = multi.chunk_replay_candidate_blocks(blocks, max_envs_per_batch=0)
    assert [[block.structure_idx for block in chunk] for chunk in unlimited] == [
        [4, 4],
        [1],
    ]
    assert tuple(block for chunk in unlimited for block in chunk) == blocks
    assert tuple(block for chunk in chunks for block in chunk) == blocks
    with pytest.raises(ValueError, match="must fit one candidate direction block"):
        multi.chunk_replay_candidate_blocks(blocks, max_envs_per_batch=1)


def test_chunk_replay_candidate_blocks_never_mixes_structures_even_under_cap():
    # One candidate each (2 dirs → 2 envs). Cap of 4 would fit both structures
    # together, but chunks must stay structure-homogeneous.
    blocks = multi.build_replay_candidate_blocks(
        (
            _request(4, candidates=(_Candidate(40.0),)),
            _request(1, params=_request(4).base_params, candidates=(_Candidate(10.0),)),
        )
    )
    chunks = multi.chunk_replay_candidate_blocks(blocks, max_envs_per_batch=4)
    assert [[block.structure_idx for block in chunk] for chunk in chunks] == [
        [4],
        [1],
    ]
    assert [[len(block.slots) for block in chunk] for chunk in chunks] == [[2], [2]]


class _FakeEnv:
    def __init__(self, params, grippers, *, fail_step: bool = False):
        self.params = list(params)
        self.grippers = list(grippers)
        self.num_envs = len(self.params)
        self.device = torch.device("cpu")
        self._last_obs = {
            "ft_wrist": torch.zeros((self.num_envs, 6)),
            "tcp_velocity": torch.zeros((self.num_envs, 6)),
            "apple_pos": torch.zeros((self.num_envs, 3)),
        }
        self._sim = SimpleNamespace(build_result=None)
        self.fail_step = fail_step
        self.closed = False
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed: int):
        self.seed = seed

    def step(self, actions):
        if self.fail_step:
            raise RuntimeError("synthetic step failure")
        self.actions.append(np.asarray(actions))

    def close(self):
        self.closed = True


@pytest.fixture
def _fake_replay_runtime(monkeypatch):
    built: list[_FakeEnv] = []
    initialized_sources: list[tuple[multi.ReplayEpisodeSource, ...]] = []
    collector_recorded: list[list[dict[str, Any]]] = []

    class FakeMonitor:
        def __init__(self, num_envs, **_kwargs):
            self.num_envs = num_envs

        def check(self, _obs, *, step_idx):
            return SimpleNamespace(
                step_idx=step_idx,
                unstable=torch.zeros(self.num_envs, dtype=torch.bool),
                reasons=[[] for _ in range(self.num_envs)],
            )

    class FakeDisable:
        def __init__(self, num_envs, **_kwargs):
            self.num_envs = num_envs

        def apply_actions(self, actions):
            return actions

        def should_record_mask(self):
            return torch.ones(self.num_envs, dtype=torch.bool)

        def update(self, _mask):
            return None

    class FakeCollectors:
        def __init__(self, num_envs, recorded_by_env):
            assert num_envs == len(recorded_by_env)
            self.recorded = list(recorded_by_env)
            self.env = None
            collector_recorded.append(self.recorded)

        def record_all_envs_step(self, env, **_kwargs):
            self.env = env

        def to_arrays(self, env_idx):
            assert self.env is not None
            marker = int(self.env.params[env_idx].apple_density)
            direction_idx = int(self.recorded[env_idx]["recorded_sentinel"]) % 10000
            sentinel = 1000 * marker + direction_idx
            return {"ft_wrist": np.asarray([[sentinel]], dtype=np.float32)}

    monkeypatch.setattr(multi, "BatchedStabilityMonitor", FakeMonitor, raising=False)
    monkeypatch.setattr(multi, "EnvDisableController", FakeDisable, raising=False)
    monkeypatch.setattr(multi, "BatchedSysIdReplayCollectors", FakeCollectors, raising=False)
    monkeypatch.setattr(
        multi,
        "ik_bootstrap_unstable_mask",
        lambda _env, n: torch.zeros(n, dtype=torch.bool),
        raising=False,
    )
    monkeypatch.setattr(
        multi,
        "hard_blowup_mask",
        lambda report: torch.zeros_like(report.unstable),
        raising=False,
    )
    monkeypatch.setattr(
        multi,
        "actions_tensor_from_recorded_frame",
        lambda recorded_actions, *, frame_idx, device: torch.as_tensor(
            recorded_actions[:, frame_idx, :],
            device=device,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        multi,
        "initialize_batched_env_from_episode_sources",
        lambda _env, _dataset, sources: initialized_sources.append(tuple(sources)),
        raising=False,
    )
    return SimpleNamespace(
        built=built,
        initialized_sources=initialized_sources,
        collector_recorded=collector_recorded,
    )


def test_replay_multi_structure_routes_forced_chunks_by_stable_key(
    _fake_replay_runtime,
):
    params = _params(4)
    blocks = multi.build_replay_candidate_blocks(
        (
            _request(
                4,
                params=params,
                candidates=(_Candidate(40.0), _Candidate(41.0)),
            ),
            _request(1, params=params, candidates=(_Candidate(10.0),)),
        )
    )
    reordered = (blocks[1], blocks[2], blocks[0])
    build_calls: list[dict[str, Any]] = []

    def build_env_fn(**kwargs):
        build_calls.append(kwargs)
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    outcome = multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 99}}),
        blocks=reordered,
        build_env_fn=build_env_fn,
        max_envs_per_batch=2,
        seed=7,
    )

    assert outcome.replay_by_key[
        multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=0, direction_idx=0)
    ]["ft_wrist"][0, 0] == 40000
    assert outcome.replay_by_key[
        multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=1, direction_idx=2)
    ]["ft_wrist"][0, 0] == 41002
    assert outcome.replay_by_key[
        multi.ReplaySlotKey(structure_idx=1, local_candidate_idx=0, direction_idx=2)
    ]["ft_wrist"][0, 0] == 10002
    planned_keys = {slot.key for block in blocks for slot in block.slots}
    assert set(outcome.replay_by_key) == planned_keys
    assert outcome.failed_structures == {}
    assert outcome.diagnostics.candidate_blocks == 3
    assert outcome.diagnostics.flattened_envs == 6
    assert outcome.diagnostics.chunk_env_counts == (2, 2, 2)
    assert outcome.diagnostics.failed_chunk_indices == ()
    assert all(env.closed for env in _fake_replay_runtime.built)

    flattened_slots = [tuple(block.slots) for block in reordered]
    for call, slots, sources, recorded in zip(
        build_calls,
        flattened_slots,
        _fake_replay_runtime.initialized_sources,
        _fake_replay_runtime.collector_recorded,
        strict=True,
    ):
        assert call["num_envs"] == 2
        assert call["per_env_params"] == [slot.params for slot in slots]
        assert call["per_env_grippers"] == [slot.gripper for slot in slots]
        assert call["max_episode_steps"] == 3
        assert sources == tuple(slot.source for slot in slots)
        assert recorded == [slot.recorded for slot in slots]
        np.testing.assert_array_equal(
            _fake_replay_runtime.built[build_calls.index(call)].actions[0],
            np.stack([slot.recorded["action"][0] for slot in slots]),
        )


def test_replay_multi_structure_prunes_only_failed_structures_and_discards_partials(
    _fake_replay_runtime,
):
    params = _params(4)
    blocks = multi.build_replay_candidate_blocks(
        (
            _request(
                4,
                params=params,
                candidates=(
                    _Candidate(40.0),
                    _Candidate(41.0),
                    _Candidate(42.0),
                ),
            ),
            _request(1, params=params, candidates=(_Candidate(10.0),)),
        )
    )
    build_count = 0

    def build_env_fn(**kwargs):
        nonlocal build_count
        if build_count == 1:
            build_count += 1
            raise RuntimeError("synthetic failure")
        build_count += 1
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    outcome = multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=2,
        fail_fast=False,
    )

    detail = outcome.failed_structures[4]
    assert detail.startswith("chunk 1: synthetic failure")
    assert "Traceback (most recent call last)" in detail
    assert "RuntimeError: synthetic failure" in detail
    assert all(key.structure_idx != 4 for key in outcome.replay_by_key)
    assert multi.ReplaySlotKey(1, 0, 0) in outcome.replay_by_key
    assert outcome.diagnostics.failed_chunk_indices == (1,)
    assert build_count == 3


def test_replay_multi_structure_synchronizes_before_close_on_failure(
    monkeypatch,
    _fake_replay_runtime,
):
    """Failure path must drain Warp before tearing down the env."""
    order: list[str] = []
    monkeypatch.setattr(
        multi,
        "_synchronize_device",
        lambda: order.append("sync"),
        raising=False,
    )
    blocks = multi.build_replay_candidate_blocks((_request(4),))

    class _FailingEnv(_FakeEnv):
        def close(self):
            order.append("close")
            super().close()

        def step(self, actions):
            raise RuntimeError("synthetic step failure")

    def build_env_fn(**kwargs):
        env = _FailingEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    outcome = multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        fail_fast=False,
    )
    assert 4 in outcome.failed_structures
    assert "Traceback" in outcome.failed_structures[4]
    assert order == ["sync", "sync", "close"]


def test_replay_multi_structure_fail_fast_reraises_original_exception(
    _fake_replay_runtime,
):
    params = _params(4)
    blocks = multi.build_replay_candidate_blocks(
        (_request(4, params=params), _request(1, params=params))
    )
    build_count = 0

    def build_env_fn(**kwargs):
        nonlocal build_count
        build_count += 1
        if build_count == 2:
            raise RuntimeError("synthetic failure")
        return _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])

    with pytest.raises(RuntimeError, match="synthetic failure"):
        multi.replay_multi_structure_candidate_blocks(
            dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
            blocks=blocks,
            build_env_fn=build_env_fn,
            max_envs_per_batch=2,
            fail_fast=True,
        )


def test_replay_multi_structure_propagates_sysid_replay_cancelled(
    _fake_replay_runtime,
):
    params = _params(4)
    blocks = multi.build_replay_candidate_blocks(
        (_request(4, params=params), _request(1, params=params))
    )

    def build_env_fn(**kwargs):
        return _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])

    def on_step(*, frame_idx: int, env) -> bool:
        raise multi.SysIdReplayCancelled("viewer closed")

    with pytest.raises(multi.SysIdReplayCancelled, match="viewer closed"):
        multi.replay_multi_structure_candidate_blocks(
            dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
            blocks=blocks,
            build_env_fn=build_env_fn,
            max_envs_per_batch=0,
            fail_fast=False,
            on_step=on_step,
        )


def test_replay_multi_structure_synchronizes_before_stopping_gpu_timers(
    monkeypatch,
    _fake_replay_runtime,
):
    synchronizations: list[str] = []
    monkeypatch.setattr(
        multi,
        "_synchronize_device",
        lambda: synchronizations.append("sync"),
        raising=False,
    )
    blocks = multi.build_replay_candidate_blocks((_request(4),))

    def build_env_fn(**kwargs):
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
    )

    assert synchronizations == ["sync", "sync", "sync"]


def test_build_replay_candidate_blocks_copies_support_kp_from_candidate():
    requests = (
        _request(
            4,
            candidates=(
                _Candidate(40.0, support_kp=1.0e3),
                _Candidate(41.0, support_kp=2.0e4),
            ),
            directions=(0,),
        ),
    )

    blocks = multi.build_replay_candidate_blocks(requests)

    assert [block.slots[0].support_kp for block in blocks] == [1.0e3, 2.0e4]
    assert all(slot.support_kp is None for block in blocks for slot in block.slots[1:])


def test_build_replay_candidate_blocks_leaves_support_kp_none_without_candidate_attr():
    blocks = multi.build_replay_candidate_blocks((_request(4, candidates=(_Candidate(40.0),)),))

    assert all(slot.support_kp is None for block in blocks for slot in block.slots)


def test_replay_multi_structure_applies_support_kp_before_reset(
    monkeypatch,
    _fake_replay_runtime,
):
    apply_calls: list[dict[str, Any]] = []
    event_order: list[str] = []

    def fake_apply(scene, support_kp_per_env, *, num_envs, joints_per_world, zeta):
        apply_calls.append(
            {
                "scene": scene,
                "support_kp_per_env": tuple(support_kp_per_env),
                "num_envs": num_envs,
                "joints_per_world": joints_per_world,
                "zeta": zeta,
            }
        )
        event_order.append("apply")

    monkeypatch.setattr(
        multi,
        "apply_per_env_support_joint_penalties",
        fake_apply,
        raising=False,
    )

    class _SupportKpFakeEnv(_FakeEnv):
        def __init__(self, params, grippers):
            super().__init__(params, grippers)
            self._sim = SimpleNamespace(
                scene=SimpleNamespace(label="scene"),
                layout=SimpleNamespace(num_envs=self.num_envs, joints_per_world=7),
            )

        def reset(self, *, seed: int):
            event_order.append("reset")
            super().reset(seed=seed)

    params = _params(4)
    blocks = multi.build_replay_candidate_blocks(
        (
            _request(
                4,
                params=params,
                candidates=(
                    _Candidate(40.0, support_kp=1.0e3),
                    _Candidate(41.0, support_kp=2.0e4),
                ),
                directions=(0,),
            ),
        )
    )

    def build_env_fn(**kwargs):
        env = _SupportKpFakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(
            manifest={
                "collection": {
                    "seed": 7,
                    "sim_config": {"joint_damping_ratio": 0.5},
                }
            }
        ),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    assert len(apply_calls) == 1
    assert apply_calls[0]["support_kp_per_env"] == (1.0e3, 2.0e4)
    assert apply_calls[0]["num_envs"] == 2
    assert apply_calls[0]["joints_per_world"] == 7
    assert apply_calls[0]["scene"].label == "scene"
    assert apply_calls[0]["zeta"] == pytest.approx(0.5)
    assert event_order == ["apply", "reset"]


def test_replay_multi_structure_skips_support_kp_apply_when_unset(
    monkeypatch,
    _fake_replay_runtime,
):
    apply_calls: list[tuple[float, ...]] = []

    monkeypatch.setattr(
        multi,
        "apply_per_env_support_joint_penalties",
        lambda _scene, kp, **_kwargs: apply_calls.append(tuple(kp)),
        raising=False,
    )

    blocks = multi.build_replay_candidate_blocks((_request(4, directions=(0,)),))

    def build_env_fn(**kwargs):
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
    )

    assert apply_calls == []


def test_replay_multi_structure_support_kp_sets_solver_arrays_per_env(
    _fake_replay_runtime,
):
    import sys
    from pathlib import Path

    import newton

    _SIM_TESTS_DIR = (
        Path(__file__).resolve().parent.parent.parent / "apple_pick_sim" / "tests"
    )
    if str(_SIM_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SIM_TESTS_DIR))

    from apple_pick_sim.tests.conftest import COUPLED_SCENE_KW  # noqa: E402
    from apple_pick_sim.coupled_fruiting import CoupledFruitingScene  # noqa: E402
    from apple_pick_sim.coupled_fruiting.batched_layout import BatchedEnvLayout  # noqa: E402
    from apple_pick_sim.coupled_fruiting.batched_build import (  # noqa: E402
        build_heterogeneous_coupled_cable_scene,
    )
    from apple_pick_sim.fruiting_system import (  # noqa: E402
        GripperProxyConfig,
        load_ranges,
        sample_heterogeneous_params_list,
    )
    from apple_pick_sim.robot import fr3_robot  # noqa: E402

    fixture = (
        Path(__file__).resolve().parent.parent.parent
        / "apple_pick_sim"
        / "fixtures"
        / "fruiting_system_ranges_real_world_proxy_variance.json"
    )
    ranges = load_ranges(fixture)
    params_list = sample_heterogeneous_params_list(ranges, topology_seed=7, num_envs=2)
    cable, _offsets = build_heterogeneous_coupled_cable_scene(
        params_list,
        env_spacing=(2.5, 2.5, 0.0),
        device="cpu",
        enable_self_collisions=False,
        base_pos=COUPLED_SCENE_KW["base_pos"],
        robot_base_pos=COUPLED_SCENE_KW["robot_base_pos"],
        gripper_proxy=GripperProxyConfig(
            mass=fr3_robot.EE_MASS_KG,
            fix_to_apple=False,
            robot_facing_weld=False,
        ),
    )
    layout = BatchedEnvLayout.from_cable_only(cable, cable.model)
    scene = CoupledFruitingScene(
        cable=cable,
        cable_collision_pipeline=None,
        vbd_only=True,
        layout=layout,
    )
    joints_per_world = int(layout.joints_per_world)
    support_labels = [
        j for j, lab in cable.fruiting_fixed_joints if "primary_support_left" in lab
    ]
    assert len(support_labels) == 1
    j_support = support_labels[0]

    def _angular_kp_at_joint(solver, global_joint_index: int) -> float:
        jc_start = solver.joint_constraint_start.numpy()
        k = solver.joint_penalty_k.numpy()
        c0 = int(jc_start[global_joint_index])
        return float(k[c0 + newton.solvers.SolverVBD.JointSlot.ANGULAR])

    class _SceneReplayEnv(_FakeEnv):
        def __init__(self, params, grippers):
            super().__init__(params, grippers)
            self._sim = SimpleNamespace(scene=scene, layout=layout)
            self.kp_before_reset: list[float] | None = None

        def reset(self, *, seed: int):
            solver = scene.cable.solver
            self.kp_before_reset = [
                _angular_kp_at_joint(solver, w * joints_per_world + j_support)
                for w in range(self.num_envs)
            ]
            super().reset(seed=seed)

    params = _params(4)
    blocks = multi.build_replay_candidate_blocks(
        (
            _request(
                4,
                params=params,
                candidates=(
                    _Candidate(40.0, support_kp=1.0e3),
                    _Candidate(41.0, support_kp=2.0e4),
                ),
                directions=(0,),
            ),
        )
    )

    def build_env_fn(**kwargs):
        env = _SceneReplayEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    env = _fake_replay_runtime.built[0]
    assert env.kp_before_reset is not None
    assert env.kp_before_reset[0] == pytest.approx(1.0e3)
    assert env.kp_before_reset[1] == pytest.approx(2.0e4)


def test_real_build_env_fn_advertises_per_env_meta():
    from pathlib import Path

    from apple_pick_gym.batched_envs.real_batched_replay_build import (
        make_real_replay_build_env_fn,
    )
    from apple_pick_sim.fruiting_system.params import load_ranges

    ranges_path = Path(
        "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
    )
    fn = make_real_replay_build_env_fn(
        ranges_path=ranges_path,
        ranges=load_ranges(ranges_path),
        topology_seed=0,
        fruiting_base_pos=(0.0, 0.5, 0.95),
        episode_meta=_real_dir_meta(apple_pos=(0.1, 0.2, 0.3)),
    )
    assert getattr(fn, "wants_per_env_meta", False) is True


def test_two_direction_batch_gets_distinct_weld_poses(_fake_replay_runtime):
    meta_d0 = _real_dir_meta(apple_pos=(1.0, 0.0, 0.0))
    meta_d1 = _real_dir_meta(apple_pos=(2.0, 0.0, 0.0))
    request = _request(
        4,
        candidates=(_Candidate(40.0),),
        directions=(0, 1),
        meta_by_direction={0: meta_d0, 1: meta_d1},
    )
    blocks = multi.build_replay_candidate_blocks((request,))
    slots = blocks[0].slots
    assert slots[0].episode_meta != slots[1].episode_meta
    assert slots[0].gripper.weld_reference_pos != slots[1].gripper.weld_reference_pos
    d0_weld = tuple(float(x) for x in meta_d0["weld_reference_pos"])
    assert slots[0].gripper.weld_reference_pos == pytest.approx(d0_weld)
    assert slots[1].gripper.weld_reference_pos != d0_weld

    build_calls: list[dict[str, Any]] = []

    def build_env_fn(**kwargs):
        build_calls.append(kwargs)
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    build_env_fn.wants_per_env_meta = True
    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    assert build_calls[0]["per_env_episode_meta"] == [meta_d0, meta_d1]


def test_slots_without_meta_do_not_pass_per_env_meta(_fake_replay_runtime):
    request = _request(4, candidates=(_Candidate(40.0),), directions=(0, 1))
    blocks = multi.build_replay_candidate_blocks((request,))
    assert all(slot.gripper is request.gripper for slot in blocks[0].slots)

    build_calls: list[dict[str, Any]] = []

    def build_env_fn(**kwargs):
        build_calls.append(kwargs)
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    build_env_fn.wants_per_env_meta = True
    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    assert "per_env_episode_meta" not in build_calls[0]
    assert all(slot.gripper == request.gripper for slot in blocks[0].slots)


_PADDED_FRAME_SENTINEL = 8888.0


def _unequal_length_request(
    structure_idx: int = 4, *, action_dim: int = 6
) -> multi.ReplayStructureRequest:
    action0 = np.stack(
        [np.full(action_dim, 10.0 + float(i), dtype=np.float32) for i in range(5)]
    )
    action1 = np.stack(
        [np.full(action_dim, 20.0 + float(i), dtype=np.float32) for i in range(3)]
    )
    rec0 = _recorded(structure_idx, 0, frames=5, action_dim=action_dim)
    rec1 = _recorded(structure_idx, 1, frames=3, action_dim=action_dim)
    rec0["action"] = action0
    rec1["action"] = action1
    return dataclasses.replace(
        _request(
            structure_idx,
            candidates=(_Candidate(40.0),),
            directions=(0, 1),
            frames=5,
            action_dim=action_dim,
        ),
        recorded_by_direction={0: rec0, 1: rec1},
    )


def _patch_horizon_collectors(monkeypatch, *, pad_sentinel: float) -> None:
    class HorizonCollectors:
        def __init__(self, num_envs, recorded_by_env):
            assert num_envs == len(recorded_by_env)
            self.recorded = list(recorded_by_env)
            self.env = None
            self._n_steps = 0

        def record_all_envs_step(self, env, *, frame_idx, **_kwargs):
            self.env = env
            self._n_steps = int(frame_idx) + 1

        def to_arrays(self, env_idx):
            n_replayed = int(self._n_steps)
            n_true = int(np.asarray(self.recorded[env_idx]["action"]).shape[0])
            ft = np.arange(n_replayed, dtype=np.float32).reshape(n_replayed, 1)
            if n_true < n_replayed:
                ft[n_true:] = pad_sentinel
            return {"ft_wrist": ft.copy(), "tcp_pos": ft.copy()}

    monkeypatch.setattr(
        multi, "BatchedSysIdReplayCollectors", HorizonCollectors, raising=False
    )


def test_drive_tensor_pads_short_directions_with_last_action(
    monkeypatch,
    _fake_replay_runtime,
):
    _patch_horizon_collectors(monkeypatch, pad_sentinel=_PADDED_FRAME_SENTINEL)
    request = _unequal_length_request()
    blocks = multi.build_replay_candidate_blocks((request,))
    last_logged = np.asarray(request.recorded_by_direction[1]["action"][-1])

    def build_env_fn(**kwargs):
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    env = _fake_replay_runtime.built[0]
    drive = np.stack([np.asarray(step) for step in env.actions], axis=1)
    assert drive.shape == (2, 5, 6)
    np.testing.assert_array_equal(drive[0], request.recorded_by_direction[0]["action"])
    np.testing.assert_array_equal(drive[1, :3], request.recorded_by_direction[1]["action"])
    np.testing.assert_array_equal(drive[1, 3], last_logged)
    np.testing.assert_array_equal(drive[1, 4], last_logged)
    assert not np.allclose(drive[1, 3:], 0.0)


def test_replay_arrays_truncate_to_recorded_length(monkeypatch, _fake_replay_runtime):
    _patch_horizon_collectors(monkeypatch, pad_sentinel=_PADDED_FRAME_SENTINEL)
    request = _unequal_length_request()
    blocks = multi.build_replay_candidate_blocks((request,))

    def build_env_fn(**kwargs):
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    outcome = multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    key0 = multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=0, direction_idx=0)
    key1 = multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=0, direction_idx=1)
    arrays0 = outcome.replay_by_key[key0]
    arrays1 = outcome.replay_by_key[key1]
    assert arrays0["ft_wrist"].shape[0] == 5
    assert arrays1["ft_wrist"].shape[0] == 3
    assert arrays1["tcp_pos"].shape[0] == 3
    assert int(np.asarray(request.recorded_by_direction[1]["action"]).shape[0]) == 3


def test_padded_frames_absent_from_features(monkeypatch, _fake_replay_runtime):
    _patch_horizon_collectors(monkeypatch, pad_sentinel=_PADDED_FRAME_SENTINEL)
    request = _unequal_length_request()
    blocks = multi.build_replay_candidate_blocks((request,))

    def build_env_fn(**kwargs):
        env = _FakeEnv(kwargs["per_env_params"], kwargs["per_env_grippers"])
        _fake_replay_runtime.built.append(env)
        return env

    outcome = multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
    )

    key1 = multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=0, direction_idx=1)
    arrays1 = outcome.replay_by_key[key1]
    stacked = np.concatenate(
        [np.asarray(arrays1["ft_wrist"]).reshape(-1), np.asarray(arrays1["tcp_pos"]).reshape(-1)]
    )
    assert np.all(stacked != _PADDED_FRAME_SENTINEL)
    assert arrays1["ft_wrist"].shape[0] == 3


def _recorded_for_real_collector(
    structure_idx: int,
    direction_idx: int,
    *,
    frames: int,
    action_dim: int = 6,
    junction_names: tuple[str, ...] = ("support", "primary_spur", "spur_stem"),
) -> dict[str, Any]:
    recorded = _recorded(
        structure_idx,
        direction_idx,
        frames=frames,
        action_dim=action_dim,
        junction_names=junction_names,
    )
    recorded["phase"] = np.zeros(frames, dtype=np.int8)
    recorded["dir_idx"] = np.full(frames, direction_idx, dtype=np.int32)
    recorded["excitation_type"] = np.zeros(frames, dtype=np.int8)
    recorded["excitation_direction"] = np.tile(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        (frames, 1),
    )
    return recorded


def _batched_replay_obs(
    *,
    num_envs: int,
    frame_idx: int,
    junction_names: list[str],
    recorded_n_frames: tuple[int, ...],
    pad_sentinel: float,
) -> dict[str, Any]:
    woody_part_info: dict[str, dict[str, torch.Tensor]] = {}
    for name in junction_names:
        anchors = torch.zeros(num_envs, 6, dtype=torch.float32)
        for env_idx in range(num_envs):
            anchors[env_idx] = torch.tensor(
                [
                    10.0 + frame_idx,
                    11.0 + frame_idx,
                    12.0 + frame_idx,
                    20.0 + frame_idx,
                    21.0 + frame_idx,
                    22.0 + frame_idx,
                ],
                dtype=torch.float32,
            ) + float(env_idx)
        woody_part_info[name] = {
            "anchors_pos": anchors,
            "anchor_force": torch.zeros(num_envs, 6, dtype=torch.float32),
        }

    ft_wrist = torch.zeros(num_envs, 6, dtype=torch.float32)
    for env_idx in range(num_envs):
        if frame_idx >= int(recorded_n_frames[env_idx]):
            ft_wrist[env_idx] = pad_sentinel
        else:
            ft_wrist[env_idx] = 100.0 + float(frame_idx) + float(env_idx)

    return {
        "woody_part_info": woody_part_info,
        "apple_pos": torch.full((num_envs, 3), 4.0 + float(frame_idx), dtype=torch.float32),
        "tcp_force": torch.zeros(num_envs, 6, dtype=torch.float32),
        "tcp_velocity": torch.full((num_envs, 6), 200.0 + float(frame_idx), dtype=torch.float32),
        "ft_wrist": ft_wrist,
        "raw_ft_wrist": torch.zeros(num_envs, 6, dtype=torch.float32),
        "tcp_pos": torch.full((num_envs, 3), 1.0 + float(frame_idx), dtype=torch.float32),
        "tcp_quat": torch.zeros(num_envs, 4, dtype=torch.float32),
        "apple_quat": torch.zeros(num_envs, 4, dtype=torch.float32),
        "robot_joint_q": torch.zeros(num_envs, 7, dtype=torch.float32),
        "excitation_type": torch.zeros(num_envs, dtype=torch.long),
        "excitation_f_inst": torch.zeros(num_envs, dtype=torch.float32),
        "excitation_direction": torch.zeros(num_envs, 3, dtype=torch.float32),
    }


class _ObsFakeEnv:
    def __init__(
        self,
        params,
        grippers,
        *,
        recorded_n_frames: tuple[int, ...],
        pad_sentinel: float,
    ):
        self.params = list(params)
        self.grippers = list(grippers)
        self.num_envs = len(self.params)
        self.device = torch.device("cpu")
        self.recorded_n_frames = tuple(int(n) for n in recorded_n_frames)
        self.pad_sentinel = float(pad_sentinel)
        self.junction_names = list(
            _recorded_for_real_collector(0, 0, frames=1)["junction_names"]
        )
        self._sim = SimpleNamespace(build_result=None)
        self.closed = False
        self.actions: list[np.ndarray] = []
        self._last_obs = _batched_replay_obs(
            num_envs=self.num_envs,
            frame_idx=0,
            junction_names=self.junction_names,
            recorded_n_frames=self.recorded_n_frames,
            pad_sentinel=self.pad_sentinel,
        )

    def reset(self, *, seed: int):
        self.seed = seed

    def step(self, actions):
        self.actions.append(np.asarray(actions))
        frame_idx = len(self.actions) - 1
        self._last_obs = _batched_replay_obs(
            num_envs=self.num_envs,
            frame_idx=frame_idx,
            junction_names=self.junction_names,
            recorded_n_frames=self.recorded_n_frames,
            pad_sentinel=self.pad_sentinel,
        )

    def close(self):
        self.closed = True


def _patch_real_collector_replay_runtime(monkeypatch) -> None:
    class FakeMonitor:
        def __init__(self, num_envs, **_kwargs):
            self.num_envs = num_envs

        def check(self, _obs, *, step_idx):
            return SimpleNamespace(
                step_idx=step_idx,
                unstable=torch.zeros(self.num_envs, dtype=torch.bool),
                reasons=[[] for _ in range(self.num_envs)],
            )

    class FakeDisable:
        def __init__(self, num_envs, **_kwargs):
            self.num_envs = num_envs

        def apply_actions(self, actions):
            return actions

        def should_record_mask(self):
            return torch.ones(self.num_envs, dtype=torch.bool)

        def update(self, _mask):
            return None

    monkeypatch.setattr(multi, "BatchedStabilityMonitor", FakeMonitor, raising=False)
    monkeypatch.setattr(multi, "EnvDisableController", FakeDisable, raising=False)
    monkeypatch.setattr(
        multi,
        "ik_bootstrap_unstable_mask",
        lambda _env, n: torch.zeros(n, dtype=torch.bool),
        raising=False,
    )
    monkeypatch.setattr(
        multi,
        "hard_blowup_mask",
        lambda report: torch.zeros_like(report.unstable),
        raising=False,
    )
    monkeypatch.setattr(
        multi,
        "actions_tensor_from_recorded_frame",
        lambda recorded_actions, *, frame_idx, device: torch.as_tensor(
            recorded_actions[:, frame_idx, :],
            device=device,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        multi,
        "initialize_batched_env_from_episode_sources",
        lambda _env, _dataset, _sources: None,
        raising=False,
    )


def test_unequal_length_replay_gates_real_collector_record_mask(monkeypatch):
    _patch_real_collector_replay_runtime(monkeypatch)
    action0 = np.stack(
        [np.full(6, 10.0 + float(i), dtype=np.float32) for i in range(5)]
    )
    action1 = np.stack(
        [np.full(6, 20.0 + float(i), dtype=np.float32) for i in range(3)]
    )
    rec0 = _recorded_for_real_collector(4, 0, frames=5)
    rec1 = _recorded_for_real_collector(4, 1, frames=3)
    rec0["action"] = action0
    rec1["action"] = action1
    request = dataclasses.replace(
        _request(
            4,
            candidates=(_Candidate(40.0),),
            directions=(0, 1),
            frames=5,
        ),
        recorded_by_direction={0: rec0, 1: rec1},
    )
    blocks = multi.build_replay_candidate_blocks((request,))
    built: list[_ObsFakeEnv] = []

    def build_env_fn(**kwargs):
        env = _ObsFakeEnv(
            kwargs["per_env_params"],
            kwargs["per_env_grippers"],
            recorded_n_frames=(5, 3),
            pad_sentinel=_PADDED_FRAME_SENTINEL,
        )
        built.append(env)
        return env

    outcome = multi.replay_multi_structure_candidate_blocks(
        dataset=SimpleNamespace(manifest={"collection": {"seed": 7}}),
        blocks=blocks,
        build_env_fn=build_env_fn,
        max_envs_per_batch=0,
        fail_fast=True,
    )

    key0 = multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=0, direction_idx=0)
    key1 = multi.ReplaySlotKey(structure_idx=4, local_candidate_idx=0, direction_idx=1)
    arrays0 = outcome.replay_by_key[key0]
    arrays1 = outcome.replay_by_key[key1]
    assert arrays0["ft_wrist"].shape[0] == 5
    assert arrays1["ft_wrist"].shape[0] == 3
    stacked = np.concatenate(
        [np.asarray(arrays1["ft_wrist"]).reshape(-1), np.asarray(arrays1["tcp_pos"]).reshape(-1)]
    )
    assert np.all(stacked != _PADDED_FRAME_SENTINEL)
    assert len(built) == 1
    assert len(built[0].actions) == 5

