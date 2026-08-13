"""Typed planning and fused execution for multi-structure sys-ID replay."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Protocol

import numpy as np

from apple_pick_gym.batched_envs.batched_stability_monitor import (
    BatchedStabilityMonitor,
    hard_blowup_mask,
    ik_bootstrap_unstable_mask,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    BatchedSysIdReplayCollectors,
    actions_tensor_from_recorded_frame,
)
from apple_pick_gym.batched_envs.support_joint_penalties import (
    apply_per_env_support_joint_penalties,
    support_joint_zeta_from_dataset,
)
from apple_pick_gym.batched_envs.env_disable_controller import EnvDisableController
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
)
from apple_pick_sim.system_id import (
    ReplayEpisodeSource,
    initialize_batched_env_from_episode_sources,
)


class ReplayFusionIncompatible(ValueError):
    """Valid replay requests cannot share one heterogeneous Newton model."""


class SysIdReplayCancelled(RuntimeError):
    """Replay aborted by an external cancel signal (e.g. viewer closed).

    Must propagate through fused and scalar replay paths without being treated
    as a structure-local failure or triggering scalar retries.
    """


class SysIdReplayCandidate(Protocol):
    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams: ...


@dataclasses.dataclass(frozen=True, order=True)
class ReplaySlotKey:
    structure_idx: int
    local_candidate_idx: int
    direction_idx: int


@dataclasses.dataclass(frozen=True)
class ReplayStructureRequest:
    structure_idx: int
    candidates: tuple[SysIdReplayCandidate, ...]
    direction_indices: tuple[int, ...]
    base_params: FruitingSystemParams
    recorded_by_direction: Mapping[int, dict[str, Any]]
    gripper: GripperProxyConfig


@dataclasses.dataclass(frozen=True)
class ReplaySlot:
    key: ReplaySlotKey
    params: FruitingSystemParams
    recorded: dict[str, Any]
    source: ReplayEpisodeSource
    gripper: GripperProxyConfig
    support_kp: float | None = None


@dataclasses.dataclass(frozen=True)
class ReplayCandidateBlock:
    structure_idx: int
    local_candidate_idx: int
    slots: tuple[ReplaySlot, ...]


@dataclasses.dataclass(frozen=True)
class MultiStructureReplayDiagnostics:
    candidate_blocks: int
    flattened_envs: int
    chunk_env_counts: tuple[int, ...]
    failed_chunk_indices: tuple[int, ...]
    build_seconds: float
    replay_seconds: float


@dataclasses.dataclass
class MultiStructureReplayOutcome:
    replay_by_key: dict[ReplaySlotKey, dict[str, Any]]
    failed_structures: dict[int, str]
    diagnostics: MultiStructureReplayDiagnostics


@dataclasses.dataclass(frozen=True)
class _ReplayCompatibilitySignature:
    topology: str
    enabled_rods: tuple[str, ...]
    segment_counts: tuple[int, ...]
    apple_present: bool
    junction_names: tuple[str, ...]
    frame_count: int
    action_width: int
    direction_indices: tuple[int, ...]
    gripper: tuple[Any, ...]


_ROD_NAMES = ("primary", "secondary", "spur", "stem")


def _synchronize_device() -> None:
    """Complete queued Warp work before recording GPU-sensitive timings."""
    import warp as wp

    wp.synchronize()


def _validate_request(request: ReplayStructureRequest) -> _ReplayCompatibilitySignature:
    if not request.candidates:
        raise ValueError(
            f"candidates must be non-empty for structure {int(request.structure_idx)}"
        )
    directions = tuple(int(direction_idx) for direction_idx in request.direction_indices)
    if not directions:
        raise ValueError(
            f"direction_indices must be non-empty for structure {int(request.structure_idx)}"
        )
    if len(set(directions)) != len(directions):
        raise ValueError(
            f"duplicate direction IDs for structure {int(request.structure_idx)}"
        )
    recorded_keys = {int(key) for key in request.recorded_by_direction}
    if recorded_keys != set(directions):
        raise ValueError(
            "recorded_by_direction keys must exactly cover direction_indices "
            f"for structure {int(request.structure_idx)}"
        )

    frame_count: int | None = None
    action_width: int | None = None
    junction_names: tuple[str, ...] | None = None
    for direction_idx in directions:
        recorded = request.recorded_by_direction[direction_idx]
        action = np.asarray(recorded.get("action"))
        if action.ndim != 2 or action.shape[1] not in (6, 19):
            raise ValueError(
                "expected action shape (T, 6) or (T, 19) for "
                f"structure {int(request.structure_idx)} direction {direction_idx}, "
                f"got {action.shape!r}"
            )
        episode_action_width = int(action.shape[1])
        if action_width is None:
            action_width = episode_action_width
        elif episode_action_width != action_width:
            raise ValueError(
                "all direction episodes must have the same action width for "
                f"structure {int(request.structure_idx)}"
            )
        episode_frames = int(action.shape[0])
        if frame_count is None:
            frame_count = episode_frames
        elif episode_frames != frame_count:
            raise ValueError(
                "all direction episodes must have the same frame count for "
                f"structure {int(request.structure_idx)}"
            )
        episode_junctions = tuple(str(name) for name in recorded.get("junction_names", ()))
        if junction_names is None:
            junction_names = episode_junctions
        elif episode_junctions != junction_names:
            raise ValueError(
                "all direction episodes must have the same ordered junction names for "
                f"structure {int(request.structure_idx)}"
            )

    params = request.base_params
    enabled_rods = tuple(name for name in _ROD_NAMES if getattr(params, name) is not None)
    segment_counts = tuple(
        int(getattr(params, name).num_segments) for name in enabled_rods
    )
    gripper = request.gripper
    gripper_signature = (
        float(gripper.mass),
        str(gripper.shape),
        float(gripper.cylinder_radius),
        float(gripper.cylinder_half_height),
        tuple(float(value) for value in gripper.box_half_extents),
        str(gripper.label),
        bool(gripper.fix_to_apple),
    )
    assert frame_count is not None and action_width is not None and junction_names is not None
    return _ReplayCompatibilitySignature(
        topology=str(params.topology),
        enabled_rods=enabled_rods,
        segment_counts=segment_counts,
        apple_present=params.apple_radius is not None and params.apple_density is not None,
        junction_names=junction_names,
        frame_count=frame_count,
        action_width=action_width,
        direction_indices=directions,
        gripper=gripper_signature,
    )


def _raise_if_incompatible(
    baseline: _ReplayCompatibilitySignature,
    candidate: _ReplayCompatibilitySignature,
) -> None:
    labels = (
        ("topology", baseline.topology, candidate.topology),
        ("enabled rods", baseline.enabled_rods, candidate.enabled_rods),
        ("segment counts", baseline.segment_counts, candidate.segment_counts),
        ("apple presence", baseline.apple_present, candidate.apple_present),
        ("junction names", baseline.junction_names, candidate.junction_names),
        ("frame count", baseline.frame_count, candidate.frame_count),
        ("action width", baseline.action_width, candidate.action_width),
        ("direction layout", baseline.direction_indices, candidate.direction_indices),
        ("gripper configuration", baseline.gripper, candidate.gripper),
    )
    for label, expected, actual in labels:
        if actual != expected:
            raise ReplayFusionIncompatible(
                f"replay fusion {label} mismatch: expected {expected!r}, got {actual!r}"
            )


def build_replay_candidate_blocks(
    requests: Sequence[ReplayStructureRequest],
) -> tuple[ReplayCandidateBlock, ...]:
    """Validate and flatten requests in request/candidate/direction order."""
    request_list = tuple(requests)
    structure_indices = tuple(int(request.structure_idx) for request in request_list)
    if len(set(structure_indices)) != len(structure_indices):
        raise ValueError("duplicate structure_idx requests are not allowed")

    signatures = tuple(_validate_request(request) for request in request_list)
    if signatures:
        for signature in signatures[1:]:
            _raise_if_incompatible(signatures[0], signature)

    blocks: list[ReplayCandidateBlock] = []
    seen_keys: set[ReplaySlotKey] = set()
    for request in request_list:
        structure_idx = int(request.structure_idx)
        directions = tuple(int(direction_idx) for direction_idx in request.direction_indices)
        for local_candidate_idx, candidate in enumerate(request.candidates):
            params = candidate.apply_to(request.base_params)
            support_kp = getattr(candidate, "support_kp", None)
            slots = tuple(
                ReplaySlot(
                    key=ReplaySlotKey(
                        structure_idx=structure_idx,
                        local_candidate_idx=local_candidate_idx,
                        direction_idx=direction_idx,
                    ),
                    params=params,
                    recorded=request.recorded_by_direction[direction_idx],
                    source=ReplayEpisodeSource(
                        structure_idx=structure_idx,
                        direction_idx=direction_idx,
                    ),
                    gripper=request.gripper,
                    support_kp=float(support_kp)
                    if support_kp is not None
                    else None,
                )
                for direction_idx in directions
            )
            block = ReplayCandidateBlock(
                structure_idx=structure_idx,
                local_candidate_idx=local_candidate_idx,
                slots=slots,
            )
            assert all(slot.key.structure_idx == block.structure_idx for slot in slots)
            assert all(
                slot.key.local_candidate_idx == block.local_candidate_idx for slot in slots
            )
            assert tuple(slot.key.direction_idx for slot in slots) == directions
            for slot in slots:
                if slot.key in seen_keys:
                    raise RuntimeError(f"duplicate replay slot key: {slot.key}")
                seen_keys.add(slot.key)
                assert slot.source == ReplayEpisodeSource(
                    structure_idx=slot.key.structure_idx,
                    direction_idx=slot.key.direction_idx,
                )
            blocks.append(block)
    return tuple(blocks)


def chunk_replay_candidate_blocks(
    blocks: Sequence[ReplayCandidateBlock],
    *,
    max_envs_per_batch: int,
) -> tuple[tuple[ReplayCandidateBlock, ...], ...]:
    """Greedily chunk complete candidate-direction blocks without splitting.

    Physical chunks never mix distinct ``structure_idx`` values. Cross-structure
    heterogeneous batches can collapse per-direction trajectories, so even an
    unlimited ``max_envs_per_batch`` keeps one structure per chunk.
    """
    items = tuple(blocks)
    if not items:
        return ()
    limit = int(max_envs_per_batch)
    if any(limit > 0 and len(block.slots) > limit for block in items):
        raise ValueError(
            f"max_envs_per_batch ({limit}) must fit one candidate direction block"
        )
    chunks: list[tuple[ReplayCandidateBlock, ...]] = []
    current: list[ReplayCandidateBlock] = []
    current_envs = 0
    current_structure: int | None = None
    for block in items:
        block_envs = len(block.slots)
        structure_idx = int(block.structure_idx)
        structure_break = current_structure is not None and structure_idx != current_structure
        capacity_break = (
            limit > 0 and current and current_envs + block_envs > limit
        )
        if structure_break or capacity_break:
            chunks.append(tuple(current))
            current = []
            current_envs = 0
            current_structure = None
        current.append(block)
        current_envs += block_envs
        current_structure = structure_idx
    if current:
        chunks.append(tuple(current))
    return tuple(chunks)


def _replay_seed(dataset: Any, seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    collection = dataset.manifest.get("collection", {})
    if "seed" not in collection:
        raise ValueError("replay seed is None and manifest.collection.seed is missing")
    return int(collection["seed"])


def _validated_planned_keys(
    blocks: tuple[ReplayCandidateBlock, ...],
) -> set[ReplaySlotKey]:
    keys: set[ReplaySlotKey] = set()
    for block in blocks:
        if not block.slots:
            raise ValueError("replay candidate blocks must contain at least one slot")
        for slot in block.slots:
            if slot.key.structure_idx != int(block.structure_idx):
                raise ValueError("slot structure key does not match candidate block")
            if slot.key.local_candidate_idx != int(block.local_candidate_idx):
                raise ValueError("slot candidate key does not match candidate block")
            expected_source = ReplayEpisodeSource(
                structure_idx=slot.key.structure_idx,
                direction_idx=slot.key.direction_idx,
            )
            if slot.source != expected_source:
                raise ValueError("slot source does not match its stable replay key")
            if slot.key in keys:
                raise ValueError(f"duplicate planned replay key: {slot.key}")
            keys.add(slot.key)
    return keys


def replay_multi_structure_candidate_blocks(
    *,
    dataset: Any,
    blocks: Sequence[ReplayCandidateBlock],
    build_env_fn: Callable[..., Any],
    max_envs_per_batch: int = 0,
    seed: int | None = None,
    fail_fast: bool = False,
    on_step: Callable[..., bool] | None = None,
) -> MultiStructureReplayOutcome:
    """Execute fused physical chunks and route every result by stable source identity."""
    block_list = tuple(blocks)
    planned_keys = _validated_planned_keys(block_list)
    chunks = chunk_replay_candidate_blocks(
        block_list,
        max_envs_per_batch=int(max_envs_per_batch),
    )
    replay_seed = _replay_seed(dataset, seed)
    replay_by_key: dict[ReplaySlotKey, dict[str, Any]] = {}
    failed_structures: dict[int, str] = {}
    failed_chunk_indices: list[int] = []
    chunk_env_counts: list[int] = []
    build_seconds = 0.0
    replay_seconds = 0.0

    for chunk_idx, chunk in enumerate(chunks):
        surviving_blocks = tuple(
            block
            for block in chunk
            if int(block.structure_idx) not in failed_structures
        )
        if not surviving_blocks:
            continue
        slots = tuple(slot for block in surviving_blocks for slot in block.slots)
        chunk_env_counts.append(len(slots))
        recorded_actions = np.stack(
            [
                np.asarray(slot.recorded["action"], dtype=np.float32)
                for slot in slots
            ],
            axis=0,
        )
        env = None
        try:
            build_started = time.perf_counter()
            env = build_env_fn(
                num_envs=len(slots),
                per_env_params=[slot.params for slot in slots],
                per_env_grippers=[slot.gripper for slot in slots],
                max_episode_steps=int(recorded_actions.shape[1]),
            )
            _synchronize_device()
            build_seconds += time.perf_counter() - build_started

            if any(slot.support_kp is not None for slot in slots):
                apply_per_env_support_joint_penalties(
                    env._sim.scene,
                    [slot.support_kp for slot in slots],
                    num_envs=env._sim.layout.num_envs,
                    joints_per_world=env._sim.layout.joints_per_world,
                    zeta=support_joint_zeta_from_dataset(dataset),
                )

            replay_started = time.perf_counter()
            env.reset(seed=replay_seed)
            initialize_batched_env_from_episode_sources(
                env,
                dataset,
                tuple(slot.source for slot in slots),
            )
            initial_unstable = ik_bootstrap_unstable_mask(env, len(slots))
            last_obs = getattr(env, "_last_obs", None)
            if last_obs is None:
                raise RuntimeError("env._last_obs missing after reset")
            monitor = BatchedStabilityMonitor(
                len(slots),
                known_obs_keys=set(last_obs.keys()),
                initial_unstable=initial_unstable,
            )
            disable_ctrl = EnvDisableController(
                len(slots),
                device=env.device,
                initial_disabled=initial_unstable,
            )
            collectors = BatchedSysIdReplayCollectors(
                len(slots),
                [slot.recorded for slot in slots],
            )
            for frame_idx in range(int(recorded_actions.shape[1])):
                actions = actions_tensor_from_recorded_frame(
                    recorded_actions,
                    frame_idx=frame_idx,
                    device=env.device,
                )
                env.step(disable_ctrl.apply_actions(actions))
                if on_step is not None and not bool(
                    on_step(frame_idx=frame_idx, env=env)
                ):
                    break
                last_obs = getattr(env, "_last_obs", None)
                if last_obs is None:
                    raise RuntimeError("env._last_obs missing after step")
                step_report = monitor.check(last_obs, step_idx=frame_idx)
                collectors.record_all_envs_step(
                    env,
                    frame_idx=frame_idx,
                    unstable=step_report.unstable,
                    record_mask=disable_ctrl.should_record_mask(),
                )
                disable_ctrl.update(hard_blowup_mask(step_report))

            for env_idx, slot in enumerate(slots):
                if slot.key in replay_by_key:
                    raise RuntimeError(f"duplicate replay result key: {slot.key}")
                replay_by_key[slot.key] = collectors.to_arrays(env_idx)
            _synchronize_device()
            replay_seconds += time.perf_counter() - replay_started
        except SysIdReplayCancelled:
            raise
        except Exception as exc:
            if fail_fast:
                raise
            failed_chunk_indices.append(chunk_idx)
            failed_now = {slot.key.structure_idx for slot in slots}
            for structure_idx in failed_now:
                failed_structures.setdefault(
                    structure_idx,
                    f"chunk {chunk_idx}: {exc}",
                )
            replay_by_key = {
                key: arrays
                for key, arrays in replay_by_key.items()
                if key.structure_idx not in failed_now
            }
        finally:
            if env is not None:
                env.close()

    surviving_planned_keys = {
        key for key in planned_keys if key.structure_idx not in failed_structures
    }
    actual_keys = set(replay_by_key)
    if actual_keys != surviving_planned_keys:
        missing = sorted(surviving_planned_keys - actual_keys)
        unexpected = sorted(actual_keys - surviving_planned_keys)
        raise RuntimeError(
            "replay key conservation failed: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    return MultiStructureReplayOutcome(
        replay_by_key=replay_by_key,
        failed_structures=failed_structures,
        diagnostics=MultiStructureReplayDiagnostics(
            candidate_blocks=len(block_list),
            flattened_envs=sum(len(block.slots) for block in block_list),
            chunk_env_counts=tuple(chunk_env_counts),
            failed_chunk_indices=tuple(failed_chunk_indices),
            build_seconds=float(build_seconds),
            replay_seconds=float(replay_seconds),
        ),
    )
