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

    def apply_to(self, base: fs.FruitingSystemParams) -> fs.FruitingSystemParams:
        return dataclasses.replace(base, apple_density=float(self.marker))


def _params(seed: int = 0) -> fs.FruitingSystemParams:
    return fs.sample_params(fs.load_ranges(RANGES_FIXTURE), seed=seed)


def _recorded(
    structure_idx: int,
    direction_idx: int,
    *,
    frames: int = 3,
    junction_names: tuple[str, ...] = ("support", "primary_spur", "spur_stem"),
) -> dict[str, Any]:
    sentinel = 10000 * structure_idx + direction_idx
    return {
        "action": np.full((frames, 6), sentinel, dtype=np.float32),
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
    junction_names: tuple[str, ...] = ("support", "primary_spur", "spur_stem"),
) -> multi.ReplayStructureRequest:
    return multi.ReplayStructureRequest(
        structure_idx=structure_idx,
        candidates=candidates,
        direction_indices=directions,
        base_params=_params(structure_idx) if params is None else params,
        recorded_by_direction={
            direction_idx: _recorded(
                structure_idx,
                direction_idx,
                frames=frames,
                junction_names=junction_names,
            )
            for direction_idx in directions
        },
        gripper=GripperProxyConfig(
            fix_to_apple=True,
            weld_direction=(1.0, 0.0, 0.0),
        )
        if gripper is None
        else gripper,
    )


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

    assert outcome.failed_structures == {4: "chunk 1: synthetic failure"}
    assert all(key.structure_idx != 4 for key in outcome.replay_by_key)
    assert multi.ReplaySlotKey(1, 0, 0) in outcome.replay_by_key
    assert outcome.diagnostics.failed_chunk_indices == (1,)
    assert build_count == 3


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

    assert synchronizations == ["sync", "sync"]

