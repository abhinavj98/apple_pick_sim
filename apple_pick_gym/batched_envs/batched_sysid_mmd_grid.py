"""Bend-stiffness grid and recorded-action tensor helpers for batched sys-ID MMD."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Protocol

from apple_pick_gym.batched_envs.batched_sysid_collect import broadcast_structure_params
from apple_pick_gym.batched_envs.batched_stability_monitor import (
    BatchedStabilityMonitor,
    hard_blowup_mask,
    ik_bootstrap_unstable_mask,
)
from apple_pick_gym.batched_envs.env_disable_controller import EnvDisableController
from apple_pick_gym.grid_viz_metrics import (
    bend_stiffness_values_match,
    woody_segment_pos_mse_hold_aggregated,
    woody_segment_pos_mse_masked,
)
from apple_pick_sim.system_id.batched_digital_twin_init import (
    gripper_proxy_from_episode_metadata,
    infer_base_params_for_structure,
    initialize_batched_env_from_dataset,
    true_params_for_structure,
)
from apple_pick_sim.system_id.batched_hold_quasi_static import hold_metric_frame_indices
from apple_pick_sim.system_id.batched_trajectory_store import (
    BatchedSysIdDataset,
    PRE_WELD_STEP_IDX,
)
from apple_pick_sim.system_id.mmd import apply_normalization, biased_mmd2, fit_gt_normalization, rbf_bandwidth_median
from apple_pick_sim.system_id.mmd_features import (
    ReplayObservationCollector,
    combine_transition_features,
    replay_obs_dict_from_sysid_numpy,
)
from apple_pick_sim.system_id.manifest_sim_config import warn_manifest_sim_config_mismatch
from apple_pick_sim.system_id.mmd_results import MmdCandidateResult, write_results_csv

import numpy as np
import torch

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")

UNSTABLE_DISQUALIFY_THRESHOLD = 0.10


def _stable_mask_arrays(
    *,
    recorded: Mapping[str, Any],
    replay: Mapping[str, Any],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    gt_stable = np.asarray(
        recorded.get("stable", np.ones(n, dtype=bool)),
        dtype=bool,
    ).reshape(-1)[:n]
    replay_stable = np.asarray(
        replay.get("stable", np.ones(n, dtype=bool)),
        dtype=bool,
    ).reshape(-1)[:n]
    return gt_stable, replay_stable


def _unstable_mask_as_bool_numpy(unstable: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(unstable, torch.Tensor):
        return unstable.detach().cpu().numpy().astype(bool).reshape(-1)
    return np.asarray(unstable, dtype=bool).reshape(-1)


def _scored_frame_mask(
    arrays: Mapping[str, Any],
    n: int,
    *,
    skip_phase: int | None = -1,
) -> np.ndarray:
    if skip_phase is None:
        return np.ones(int(n), dtype=bool)
    phase = np.asarray(arrays["phase"], dtype=np.int64).reshape(-1)[: int(n)]
    return phase != int(skip_phase)


def recorded_instability_fraction_all_frames(
    recorded: Mapping[str, Any],
    *,
    skip_phase: int | None = -1,
) -> float:
    """Recorded-collection unstable fraction among scored frames (``phase != skip_phase``)."""
    n = int(np.asarray(recorded["ft_wrist"]).reshape(-1, 6).shape[0])
    scored = _scored_frame_mask(recorded, n, skip_phase=skip_phase)
    stable = np.asarray(
        recorded.get("stable", np.ones(n, dtype=bool)),
        dtype=bool,
    ).reshape(-1)[:n]
    denom = int(np.count_nonzero(scored))
    if denom == 0:
        return float("nan")
    return float(np.count_nonzero(scored & ~stable)) / float(denom)


def replay_instability_fraction_all_frames(
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
    *,
    skip_phase: int | None = -1,
) -> float:
    """Replay-unstable fraction among GT-stable scored frames."""
    n = _aligned_frame_count(replay, recorded)
    if n <= 0:
        return float("nan")
    scored = _scored_frame_mask(recorded, n, skip_phase=skip_phase)
    gt_stable, replay_stable = _stable_mask_arrays(recorded=recorded, replay=replay, n=n)
    denom_mask = scored & gt_stable
    denom = int(np.count_nonzero(denom_mask))
    if denom == 0:
        return float("nan")
    return float(np.count_nonzero(denom_mask & ~replay_stable)) / float(denom)


def warn_recorded_gt_instability(
    *,
    structure_idx: int,
    recorded_eps: Sequence[Mapping[str, Any]],
    threshold: float = UNSTABLE_DISQUALIFY_THRESHOLD,
) -> list[str]:
    """Emit warnings when recorded GT episodes exceed the instability threshold."""
    messages: list[str] = []
    for direction_idx, recorded in enumerate(recorded_eps):
        frac = recorded_instability_fraction_all_frames(recorded)
        if not np.isfinite(frac) or float(frac) <= float(threshold):
            continue
        msg = (
            f"structure {int(structure_idx)} direction {int(direction_idx)}: recorded GT dataset "
            f"has {float(frac):.1%} unstable frames (>{float(threshold):.0%}); "
            "replay scoring may be unreliable"
        )
        warnings.warn(msg, stacklevel=2)
        messages.append(msg)
    return messages


def _masked_rmse(
    rep: np.ndarray,
    rec: np.ndarray,
    *,
    keep: np.ndarray,
) -> float:
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    if not bool(np.any(keep)):
        return float("nan")
    diff = np.asarray(rep, dtype=np.float64) - np.asarray(rec, dtype=np.float64)
    row_vals = np.sqrt(np.mean(diff * diff, axis=-1))
    return float(np.mean(row_vals[keep]))


def _masked_mse(
    rep: np.ndarray,
    rec: np.ndarray,
    *,
    keep: np.ndarray,
) -> float:
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    if not bool(np.any(keep)):
        return float("nan")
    diff = np.asarray(rep, dtype=np.float64) - np.asarray(rec, dtype=np.float64)
    if diff.ndim == 1:
        row_vals = diff * diff
    else:
        row_vals = np.mean(diff * diff, axis=-1)
    return float(np.mean(row_vals[keep]))


class MmdDirectionContext(NamedTuple):
    """GT normalization and bandwidth for one excitation direction."""

    gt_norm: np.ndarray
    stats: Any
    bandwidth: float


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(diff * diff))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def strip_pre_weld_rows(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Drop leading pre-weld snapshot rows from per-frame episode arrays.

    Collection records ``step_idx=-1`` / ``phase=-1`` as a settled-tree snapshot that
    is not replayed via ``env.step``. Grid replay/scoring must exclude that row so
    ``recorded[i]`` aligns with replay step ``i``.
    """
    out = dict(arrays)

    def _strip_with_keep(keep: np.ndarray) -> dict[str, Any]:
        stripped = dict(out)
        for key, value in list(stripped.items()):
            if key == "junction_names":
                continue
            if isinstance(value, dict):
                stripped[key] = {
                    sub_key: np.asarray(sub_val)[keep]
                    for sub_key, sub_val in value.items()
                }
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == keep.shape[0]:
                stripped[key] = arr[keep]
        return stripped

    if "step_idx" in out:
        step_idx = np.asarray(out["step_idx"], dtype=np.int32).reshape(-1)
        if step_idx.size >= 1 and int(step_idx[0]) == int(PRE_WELD_STEP_IDX):
            keep = step_idx != int(PRE_WELD_STEP_IDX)
            return _strip_with_keep(keep)

    if "phase" in out:
        phase = np.asarray(out["phase"], dtype=np.int64).reshape(-1)
        if phase.size >= 1 and int(phase[0]) == -1:
            keep = phase != -1
            return _strip_with_keep(keep)

    return out


def _require_no_leading_pre_weld(recorded: Mapping[str, Any]) -> None:
    """Fail fast if replay/scoring inputs still contain a pre-weld row."""
    if "step_idx" in recorded:
        step_idx = np.asarray(recorded["step_idx"], dtype=np.int32).reshape(-1)
        if step_idx.size >= 1 and int(step_idx[0]) == int(PRE_WELD_STEP_IDX):
            raise ValueError(
                "recorded episode still contains a pre_weld row; call strip_pre_weld_rows first"
            )
    if "phase" in recorded:
        phase = np.asarray(recorded["phase"], dtype=np.int64).reshape(-1)
        if phase.size >= 1 and int(phase[0]) == -1:
            raise ValueError(
                "recorded episode still contains a pre_weld row; call strip_pre_weld_rows first"
            )


def _aligned_frame_count(
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
) -> int:
    """Return the number of frame-aligned rows shared by replay and recorded arrays."""
    _require_no_leading_pre_weld(recorded)
    rep_n = int(np.asarray(replay["ft_wrist"]).reshape(-1, 6).shape[0])
    rec_n = int(np.asarray(recorded["ft_wrist"]).reshape(-1, 6).shape[0])
    return min(rep_n, rec_n)


def _aggregate_rows(values: np.ndarray, *, aggregation: Literal["mean", "median"]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] == 0:
        raise ValueError("cannot aggregate an empty hold window")
    if aggregation == "mean":
        return np.mean(arr, axis=0)
    if aggregation == "median":
        return np.median(arr, axis=0)
    raise ValueError(f"unsupported hold aggregation: {aggregation!r}")


def trajectory_hold_aggregated_mse(
    *,
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
    aggregation: Literal["mean", "median"] = "mean",
    use_latter_half: bool = True,
    skip_phase: int | None = -1,
) -> dict[str, float]:
    """Compare replay vs recorded using one hold aggregate per signal (mean or median).

  Uses the latter half of each contiguous hold segment by default (burn-in discard).
    """

    def _as_2d_full(source: Mapping[str, Any], key: str, cols: int) -> np.ndarray:
        return np.asarray(source[key], dtype=np.float64).reshape(-1, cols)

    alignment = _aligned_frame_count(replay, recorded)
    junction_names = list(recorded.get("junction_names", []))
    if alignment <= 0:
        return {
            "n_frames": 0.0,
            "n_used_frames": 0.0,
            "ft_wrist_mse": float("nan"),
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": {},
        }

    n = alignment

    ft_rep_full = _as_2d_full(replay, "ft_wrist", 6)[:n]
    tcp_rep_full = _as_2d_full(replay, "tcp_pos", 3)[:n]
    apple_rep_full = _as_2d_full(replay, "apple_pos", 3)[:n]

    ft_rec_full = _as_2d_full(recorded, "ft_wrist", 6)[:n]
    tcp_rec_full = _as_2d_full(recorded, "tcp_pos", 3)[:n]
    apple_rec_full = _as_2d_full(recorded, "apple_pos", 3)[:n]

    recorded_slice: dict[str, Any] = {
        "phase": np.asarray(recorded["phase"], dtype=np.int8).reshape(-1)[:n],
    }
    if "amplitude_m" in recorded:
        recorded_slice["amplitude_m"] = np.asarray(recorded["amplitude_m"], dtype=np.float64).reshape(
            -1
        )[:n]
    if "dir_idx" in recorded:
        recorded_slice["dir_idx"] = np.asarray(recorded["dir_idx"], dtype=np.int32).reshape(-1)[:n]

    hold_idx = hold_metric_frame_indices(recorded_slice, use_latter_half=use_latter_half)
    gt_stable, replay_stable = _stable_mask_arrays(recorded=recorded, replay=replay, n=n)
    hold_idx = hold_idx[gt_stable[hold_idx] & replay_stable[hold_idx]]
    used = int(hold_idx.size)
    if used == 0:
        return {
            "n_frames": float(n),
            "n_used_frames": 0.0,
            "ft_wrist_mse": float("nan"),
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": woody_segment_pos_mse_hold_aggregated(
                replay=replay,
                recorded=recorded,
                junction_names=junction_names,
                n=n,
                hold_idx=hold_idx,
                aggregation=aggregation,
            ),
        }

    ft_rep = _aggregate_rows(ft_rep_full[hold_idx], aggregation=aggregation)
    tcp_rep = _aggregate_rows(tcp_rep_full[hold_idx], aggregation=aggregation)
    apple_rep = _aggregate_rows(apple_rep_full[hold_idx], aggregation=aggregation)
    ft_rec = _aggregate_rows(ft_rec_full[hold_idx], aggregation=aggregation)
    tcp_rec = _aggregate_rows(tcp_rec_full[hold_idx], aggregation=aggregation)
    apple_rec = _aggregate_rows(apple_rec_full[hold_idx], aggregation=aggregation)

    return {
        "n_frames": float(n),
        "n_used_frames": float(used),
        "ft_wrist_mse": _mse(ft_rep.reshape(1, -1), ft_rec.reshape(1, -1)),
        "ft_force_rmse": _rmse(ft_rep[:3].reshape(1, -1), ft_rec[:3].reshape(1, -1)),
        "ft_torque_rmse": _rmse(ft_rep[3:].reshape(1, -1), ft_rec[3:].reshape(1, -1)),
        "tcp_pos_mse": _mse(tcp_rep.reshape(1, -1), tcp_rec.reshape(1, -1)),
        "apple_pos_mse": _mse(apple_rep.reshape(1, -1), apple_rec.reshape(1, -1)),
        "woody_pos_mse_by_segment": woody_segment_pos_mse_hold_aggregated(
            replay=replay,
            recorded=recorded,
            junction_names=junction_names,
            n=n,
            hold_idx=hold_idx,
            aggregation=aggregation,
        ),
    }


def trajectory_mse(
    *,
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
    skip_phase: int | None = -1,
) -> dict[str, float]:
    """Compute simple frame-aligned MSE metrics between replay and recorded arrays.

    By default skips frames whose recorded ``phase`` equals -1 (``pre_weld``).
    """

    def _as_2d_full(source: Mapping[str, Any], key: str, cols: int) -> np.ndarray:
        return np.asarray(source[key], dtype=np.float64).reshape(-1, cols)

    ft_rep_full = _as_2d_full(replay, "ft_wrist", 6)
    tcp_rep_full = _as_2d_full(replay, "tcp_pos", 3)
    apple_rep_full = _as_2d_full(replay, "apple_pos", 3)

    ft_rec_full = _as_2d_full(recorded, "ft_wrist", 6)
    tcp_rec_full = _as_2d_full(recorded, "tcp_pos", 3)
    apple_rec_full = _as_2d_full(recorded, "apple_pos", 3)

    n = _aligned_frame_count(replay, recorded)
    junction_names = list(recorded.get("junction_names", []))
    if n == 0:
        return {
            "n_frames": 0.0,
            "n_used_frames": 0.0,
            "ft_wrist_mse": float("nan"),
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": {},
        }

    ft_rep = ft_rep_full[:n]
    tcp_rep = tcp_rep_full[:n]
    apple_rep = apple_rep_full[:n]
    ft_rec = ft_rec_full[:n]
    tcp_rec = tcp_rec_full[:n]
    apple_rec = apple_rec_full[:n]

    if skip_phase is None:
        mask = np.ones(n, dtype=bool)
    else:
        phase = np.asarray(recorded["phase"], dtype=np.int64).reshape(-1)[:n]
        mask = phase != int(skip_phase)

    gt_stable, replay_stable = _stable_mask_arrays(recorded=recorded, replay=replay, n=n)
    keep = np.asarray(mask, dtype=bool) & gt_stable & replay_stable

    used = int(np.count_nonzero(keep))
    if used == 0:
        return {
            "n_frames": float(n),
            "n_used_frames": 0.0,
            "ft_wrist_mse": float("nan"),
            "ft_force_rmse": float("nan"),
            "ft_torque_rmse": float("nan"),
            "tcp_pos_mse": float("nan"),
            "apple_pos_mse": float("nan"),
            "woody_pos_mse_by_segment": woody_segment_pos_mse_masked(
                replay=replay,
                recorded=recorded,
                junction_names=junction_names,
                n=n,
                mask=keep,
            ),
        }

    woody_mse: dict[str, float] = {}
    if junction_names and _has_woody_for_scoring(replay, recorded, junction_names):
        for name in junction_names:
            start_rep = np.asarray(replay["woody_part_start_pos"][name], dtype=np.float64).reshape(-1, 3)[:n]
            end_rep = np.asarray(replay["woody_part_end_pos"][name], dtype=np.float64).reshape(-1, 3)[:n]
            start_rec = np.asarray(recorded["woody_part_start_pos"][name], dtype=np.float64).reshape(-1, 3)[:n]
            end_rec = np.asarray(recorded["woody_part_end_pos"][name], dtype=np.float64).reshape(-1, 3)[:n]
            rep = np.concatenate([start_rep, end_rep], axis=1)
            rec = np.concatenate([start_rec, end_rec], axis=1)
            woody_mse[name] = _masked_mse(
                rep,
                rec,
                keep=keep,
            )

    return {
        "n_frames": float(n),
        "n_used_frames": float(used),
        # NOTE: ft_wrist mixes units (N, N·m). Keep the legacy metric for debugging, but
        # prefer separate force/torque RMSE for interpretable diagnostics.
        "ft_wrist_mse": _masked_mse(
            ft_rep,
            ft_rec,
            keep=keep,
        ),
        "ft_force_rmse": _masked_rmse(
            ft_rep[:, :3],
            ft_rec[:, :3],
            keep=keep,
        ),
        "ft_torque_rmse": _masked_rmse(
            ft_rep[:, 3:],
            ft_rec[:, 3:],
            keep=keep,
        ),
        "tcp_pos_mse": _masked_mse(
            tcp_rep,
            tcp_rec,
            keep=keep,
        ),
        "apple_pos_mse": _masked_mse(
            apple_rep,
            apple_rec,
            keep=keep,
        ),
        "woody_pos_mse_by_segment": woody_mse,
    }


def _has_woody_for_scoring(
    replay: Mapping[str, Any],
    recorded: Mapping[str, Any],
    junction_names: list[str],
) -> bool:
    for key in ("woody_part_start_pos", "woody_part_end_pos"):
        for source in (replay, recorded):
            woody = source.get(key)
            if not isinstance(woody, dict):
                return False
            for name in junction_names:
                if name not in woody:
                    return False
    return bool(junction_names)


def prepare_gt_mmd_context(
    recorded_episodes: list[dict],
) -> dict[int, MmdDirectionContext]:
    """Fit per-direction GT normalization and RBF bandwidth from recorded episodes."""
    gt_by_direction = combine_transition_features(recorded_episodes)
    if not gt_by_direction:
        raise ValueError("No valid hold-only GT transition features were found.")

    context: dict[int, MmdDirectionContext] = {}
    for direction, gt_features in gt_by_direction.items():
        stats = fit_gt_normalization(gt_features)
        gt_norm = apply_normalization(gt_features, stats)
        bandwidth = rbf_bandwidth_median(gt_norm)
        context[int(direction)] = MmdDirectionContext(
            gt_norm=gt_norm,
            stats=stats,
            bandwidth=bandwidth,
        )
    return context


def _candidate_stiffnesses(candidate: BendStiffnessCandidate) -> dict[str, float]:
    return {
        "primary": float(candidate.primary),
        "secondary": float(candidate.secondary),
        "spur": float(candidate.spur),
        "stem": float(candidate.stem),
    }


def score_candidate_mmd(
    *,
    candidate_index: int,
    candidate: BendStiffnessCandidate,
    gt_context: dict[int, MmdDirectionContext],
    replay_observations: list[dict],
) -> MmdCandidateResult:
    """Score one replayed candidate against precomputed GT MMD context."""
    candidate_by_direction = combine_transition_features(replay_observations)
    missing = sorted(set(gt_context) - set(candidate_by_direction))
    missing_directions = tuple(int(d) for d in missing)
    per_direction: dict[int, float] = {}
    for direction, context in gt_context.items():
        candidate_features = candidate_by_direction.get(int(direction))
        if candidate_features is None:
            continue
        if candidate_features.shape[1] != context.gt_norm.shape[1]:
            raise ValueError(
                "MMD feature dimension mismatch for direction "
                f"{direction}: gt={context.gt_norm.shape[1]} "
                f"candidate={candidate_features.shape[1]}"
            )
        candidate_norm = apply_normalization(candidate_features, context.stats)
        per_direction[int(direction)] = biased_mmd2(
            context.gt_norm,
            candidate_norm,
            context.bandwidth,
        )
    if not per_direction:
        raise ValueError("No candidate directions had valid hold-only MMD transitions.")
    aggregate = float(np.mean(list(per_direction.values())))
    return MmdCandidateResult(
        candidate_index=int(candidate_index),
        stiffnesses=_candidate_stiffnesses(candidate),
        aggregate_mmd2=aggregate,
        per_direction_mmd2=per_direction,
        missing_directions=missing_directions,
    )


def _resolve_replay_seed(dataset: BatchedSysIdDataset, seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    collection = dataset.manifest.get("collection", {})
    if "seed" not in collection:
        raise ValueError("replay seed is None and manifest.collection.seed is missing")
    return int(collection["seed"])



def episode_is_excluded(entry: dict) -> bool:
    """Return True when a manifest episode row is marked excluded."""
    return bool(entry.get("excluded", False))


def per_candidate_unstable_counts(
    unstable_by_env: list[int],
    *,
    num_candidates: int,
    num_directions: int,
) -> list[int]:
    """Sum per-env unstable-frame counts into one total per candidate.

    Envs are laid out as ``env_idx = candidate * num_directions + direction``,
    where ``num_directions`` is the number of *usable* directions for the
    structure (excluded directions are not replayed). ``unstable_by_env`` must
    therefore have exactly ``num_candidates * num_directions`` entries.
    """
    expected = int(num_candidates) * int(num_directions)
    if len(unstable_by_env) != expected:
        raise ValueError(
            f"unstable_by_env length {len(unstable_by_env)} != "
            f"num_candidates*num_directions ({num_candidates}*{num_directions}={expected})"
        )
    return [
        sum(
            unstable_by_env[c * int(num_directions) + d]
            for d in range(int(num_directions))
        )
        for c in range(int(num_candidates))
    ]


def list_usable_direction_indices(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    *,
    include_excluded: bool = False,
) -> list[int]:
    """Ascending direction indices for one structure, optionally keeping excluded."""
    idxs: list[int] = []
    n_excluded = 0
    for ep in dataset.episode_entries():
        if int(ep.get("structure_idx", -1)) != int(structure_idx):
            continue
        d = int(ep["direction_idx"])
        if episode_is_excluded(ep):
            n_excluded += 1
            if not include_excluded:
                continue
        idxs.append(d)
    idxs = sorted(set(idxs))
    if not idxs:
        raise ValueError(
            f"structure {int(structure_idx)} has no usable directions "
            f"(excluded={n_excluded})"
        )
    return idxs


def resolve_direction_indices(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
) -> list[int]:
    """Resolve disk direction indices for load/replay (dense local layout)."""
    if direction_indices is not None:
        dirs = [int(d) for d in direction_indices]
        if not dirs:
            raise ValueError("direction_indices must be non-empty")
        return dirs
    entries = [
        ep
        for ep in dataset.episode_entries()
        if int(ep.get("structure_idx", -1)) == int(structure_idx)
    ]
    if not entries:
        # Legacy mocks / catalogs without per-structure episode rows.
        return list(range(int(num_directions)))
    return list_usable_direction_indices(
        dataset,
        int(structure_idx),
        include_excluded=bool(include_excluded),
    )


def load_recorded_episodes_for_structure(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
) -> list[dict]:
    """Load recorded observation arrays for each usable direction of one structure."""
    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    out: list[dict] = []
    for direction_idx in dirs:
        recorded = strip_pre_weld_rows(
            dict(dataset.load_episode_obs_arrays(int(structure_idx), int(direction_idx)))
        )
        n_frames = int(np.asarray(recorded["action"]).shape[0])
        if "dir_idx" not in recorded:
            recorded["dir_idx"] = np.full(n_frames, int(direction_idx), dtype=np.int32)
        out.append(recorded)
    return out


def direction_episodes_from_collectors(
    collectors: BatchedSysIdReplayCollectors,
    *,
    candidate_index: int,
    num_directions: int,
) -> list[dict]:
    """Gather per-direction replay arrays for one candidate from merged collectors."""
    d = int(num_directions)
    c = int(candidate_index)
    return [collectors.to_arrays(c * d + direction_idx) for direction_idx in range(d)]


def chunk_candidates(
    candidates: Sequence[BendStiffnessCandidate],
    *,
    max_envs_per_batch: int,
    num_directions: int,
) -> list[list[BendStiffnessCandidate]]:
    """Split candidates so each chunk fits ``max_envs_per_batch`` parallel envs."""
    items = list(candidates)
    if not items:
        return []
    limit = int(max_envs_per_batch)
    directions = int(num_directions)
    if limit <= 0:
        return [items]
    max_chunk_size = limit // directions
    if max_chunk_size < 1:
        raise ValueError(
            f"max_envs_per_batch ({limit}) must be >= num_directions ({directions})"
        )
    return [items[i : i + max_chunk_size] for i in range(0, len(items), max_chunk_size)]


def replay_candidates_for_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[BendStiffnessCandidate],
    num_directions: int,
    seed: int | None = None,
    build_env_fn: Callable[..., Any],
    max_envs_per_batch: int = 0,
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
    use_snapshot: bool = False,
    use_oracle_params: bool = True,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
) -> BatchedSysIdReplayCollectors:
    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    d = len(dirs)
    chunks = chunk_candidates(
        candidates,
        max_envs_per_batch=int(max_envs_per_batch),
        num_directions=d,
    )
    if not chunks:
        raise ValueError(f"No stiffness candidates for structure {structure_idx}")

    merged: BatchedSysIdReplayCollectors | None = None
    for chunk in chunks:
        collectors = replay_batched_sysid_structure(
            dataset=dataset,
            structure_idx=int(structure_idx),
            candidates=chunk,
            num_directions=d,
            seed=seed,
            build_env_fn=build_env_fn,
            on_step=on_step,
            replay_sim_config=replay_sim_config,
            use_snapshot=bool(use_snapshot),
            use_oracle_params=bool(use_oracle_params),
            direction_indices=dirs,
            include_excluded=bool(include_excluded),
        )
        merged = collectors if merged is None else merged.concat_envs(collectors)
    assert merged is not None
    return merged


def evaluate_batched_mmd_grid(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[BendStiffnessCandidate],
    num_directions: int,
    seed: int | None = None,
    build_env_fn: Callable[..., Any],
    output_dir: Path | str | None = None,
) -> list[MmdCandidateResult]:
    """Score bend-stiffness candidates for one structure against recorded GT episodes."""
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must be non-empty")

    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
    )
    d = len(dirs)
    gt_context = prepare_gt_mmd_context(
        load_recorded_episodes_for_structure(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=d,
            direction_indices=dirs,
        )
    )
    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=candidate_list,
        num_directions=d,
        seed=seed,
        build_env_fn=build_env_fn,
        direction_indices=dirs,
    )

    results: list[MmdCandidateResult] = []
    for candidate_index, candidate in enumerate(candidate_list):
        replay_observations = direction_episodes_from_collectors(
            collectors,
            candidate_index=candidate_index,
            num_directions=d,
        )
        results.append(
            score_candidate_mmd(
                candidate_index=candidate_index,
                candidate=candidate,
                gt_context=gt_context,
                replay_observations=replay_observations,
            )
        )

    if output_dir is not None:
        write_results_csv(results, output_dir)
    return results


class _ReplayCollectorLike(Protocol):
    @property
    def n_rows(self) -> int: ...

    def to_arrays(self) -> dict[str, Any]: ...


class _FrozenReplayCollector:
    """Replay collector backed by pre-built arrays (used after merge)."""

    def __init__(self, arrays: dict[str, Any]) -> None:
        self._arrays = arrays

    @property
    def n_rows(self) -> int:
        return int(self._arrays["action"].shape[0])

    def to_arrays(self) -> dict[str, Any]:
        return self._arrays


def _concat_replay_arrays(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    junction_names = list(left["junction_names"])
    if junction_names != list(right["junction_names"]):
        raise ValueError("junction_names mismatch between replay collectors")

    def _cat_1d_or_2d(key: str) -> np.ndarray:
        a = np.asarray(left[key])
        b = np.asarray(right[key])
        if a.shape[0] == 0:
            return b
        if b.shape[0] == 0:
            return a
        return np.concatenate([a, b], axis=0)

    def _cat_bool_1d(key: str) -> np.ndarray:
        def _one_side(arrays: dict[str, Any]) -> np.ndarray:
            if key not in arrays:
                n = int(np.asarray(arrays["action"]).shape[0])
                return np.ones(n, dtype=bool)
            return np.asarray(arrays[key], dtype=bool).reshape(-1)

        a = _one_side(left)
        b = _one_side(right)
        if a.shape[0] == 0:
            return b
        if b.shape[0] == 0:
            return a
        return np.concatenate([a, b], axis=0)

    return {
        "action": _cat_1d_or_2d("action").astype(np.float32, copy=False),
        "ft_wrist": _cat_1d_or_2d("ft_wrist").astype(np.float32, copy=False),
        "tcp_velocity": _cat_1d_or_2d("tcp_velocity").astype(np.float32, copy=False),
        "tcp_pos": _cat_1d_or_2d("tcp_pos").astype(np.float32, copy=False),
        "apple_pos": _cat_1d_or_2d("apple_pos").astype(np.float32, copy=False),
        "phase": _cat_1d_or_2d("phase").astype(np.int8, copy=False),
        "dir_idx": _cat_1d_or_2d("dir_idx").astype(np.int32, copy=False),
        "excitation_type": _cat_1d_or_2d("excitation_type").astype(np.int8, copy=False),
        "excitation_direction": _cat_1d_or_2d("excitation_direction").astype(
            np.float32, copy=False
        ),
        "stable": _cat_bool_1d("stable"),
        "woody_part_start_pos": {
            name: _cat_1d_or_2d_for_woody(left["woody_part_start_pos"][name], right["woody_part_start_pos"][name])
            for name in junction_names
        },
        "woody_part_end_pos": {
            name: _cat_1d_or_2d_for_woody(left["woody_part_end_pos"][name], right["woody_part_end_pos"][name])
            for name in junction_names
        },
        "junction_names": junction_names,
    }


def _cat_1d_or_2d_for_woody(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape[0] == 0:
        return b_arr.astype(np.float32, copy=False)
    if b_arr.shape[0] == 0:
        return a_arr.astype(np.float32, copy=False)
    return np.concatenate([a_arr, b_arr], axis=0).astype(np.float32, copy=False)


class BatchedSysIdReplayCollectors:
    """One ReplayObservationCollector per parallel env for replay feature accumulation."""

    def __init__(
        self,
        num_envs: int,
        recorded_by_env: Sequence[Mapping[str, Any]],
    ) -> None:
        recorded = list(recorded_by_env)
        n = int(num_envs)
        if len(recorded) != n:
            raise ValueError(
                f"recorded_by_env length ({len(recorded)}) must match num_envs ({n})"
            )
        self._recorded_by_env = recorded
        self._collectors: list[_ReplayCollectorLike] = [
            ReplayObservationCollector(item) for item in recorded
        ]

    def record_step(
        self,
        env: Any,
        *,
        env_idx: int,
        frame_idx: int,
        unstable: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        idx = int(env_idx)
        recorded = self._recorded_by_env[idx]
        obs = env.sysid_numpy_obs(idx)
        adapted = replay_obs_dict_from_sysid_numpy(
            obs,
            junction_names=list(recorded["junction_names"]),
        )
        collector = self._collectors[idx]
        if not isinstance(collector, ReplayObservationCollector):
            raise RuntimeError("cannot record_step on merged replay collectors")
        stable = True
        if unstable is not None:
            unstable_arr = _unstable_mask_as_bool_numpy(unstable)
            stable = not bool(unstable_arr[int(env_idx)])
        collector.record(adapted, frame_idx=int(frame_idx), stable=stable)

    def record_all_envs_step(
        self,
        env: Any,
        *,
        frame_idx: int,
        unstable: torch.Tensor | np.ndarray | None = None,
        record_mask: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Record one replay frame for every env with a single batched GPU download."""
        from apple_pick_gym.batched_envs.obs_torch import (
            download_batched_replay_obs_numpy,
            replay_obs_dict_from_batched_numpy_row,
        )

        last_obs = getattr(env, "_last_obs", None)
        if last_obs is None:
            raise RuntimeError("call reset() or step() before record_all_envs_step()")

        num_envs = len(self._collectors)
        if num_envs == 0:
            return

        junction_names = [str(name) for name in self._recorded_by_env[0]["junction_names"]]
        batched = download_batched_replay_obs_numpy(last_obs, junction_names)
        record_mask_np = None
        if record_mask is not None:
            record_mask_np = np.asarray(record_mask.detach().cpu() if hasattr(record_mask, "detach") else record_mask, dtype=bool).reshape(-1)

        for env_idx in range(num_envs):
            if record_mask_np is not None and not bool(record_mask_np[int(env_idx)]):
                continue
            recorded = self._recorded_by_env[env_idx]
            env_junction_names = [str(name) for name in recorded["junction_names"]]
            if env_junction_names != junction_names:
                raise ValueError(
                    "record_all_envs_step requires identical junction_names across envs"
                )
            collector = self._collectors[env_idx]
            if not isinstance(collector, ReplayObservationCollector):
                raise RuntimeError("cannot record_all_envs_step on merged replay collectors")
            adapted = replay_obs_dict_from_batched_numpy_row(batched, env_idx=env_idx)
            stable = True
            if unstable is not None:
                unstable_arr = _unstable_mask_as_bool_numpy(unstable)
                stable = not bool(unstable_arr[int(env_idx)])
            collector.record(adapted, frame_idx=int(frame_idx), stable=stable)

    def to_arrays(self, env_idx: int) -> dict[str, Any]:
        return self._collectors[int(env_idx)].to_arrays()

    def n_rows(self, env_idx: int) -> int:
        return int(self._collectors[int(env_idx)].n_rows)

    def merge(self, other: BatchedSysIdReplayCollectors) -> BatchedSysIdReplayCollectors:
        """Concatenate rows per env_idx for chunking across replay batches."""
        if len(self._collectors) != len(other._collectors):
            raise ValueError("cannot merge replay collectors with different num_envs")
        merged = BatchedSysIdReplayCollectors.__new__(BatchedSysIdReplayCollectors)
        merged._recorded_by_env = list(self._recorded_by_env)
        merged._collectors = []
        for left, right in zip(self._collectors, other._collectors, strict=True):
            arrays = _concat_replay_arrays(left.to_arrays(), right.to_arrays())
            merged._collectors.append(_FrozenReplayCollector(arrays))
        return merged

    def concat_envs(self, other: BatchedSysIdReplayCollectors) -> BatchedSysIdReplayCollectors:
        """Append env slots (used for candidate chunking)."""
        combined = BatchedSysIdReplayCollectors.__new__(BatchedSysIdReplayCollectors)
        combined._recorded_by_env = list(self._recorded_by_env) + list(other._recorded_by_env)
        combined._collectors = list(self._collectors) + list(other._collectors)
        return combined


def recorded_metadata_by_env(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
) -> list[dict[str, Any]]:
    """Load recorded obs arrays per env_idx for ReplayObservationCollector init."""
    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    d = len(dirs)
    num_envs = int(num_candidates) * d
    out: list[dict[str, Any]] = []
    for env_idx in range(num_envs):
        direction_idx = int(dirs[int(env_idx) % d])
        recorded = strip_pre_weld_rows(
            dict(dataset.load_episode_obs_arrays(int(structure_idx), direction_idx))
        )
        n_frames = int(np.asarray(recorded["action"]).shape[0])
        if "dir_idx" not in recorded:
            recorded["dir_idx"] = np.full(n_frames, direction_idx, dtype=np.int32)
        out.append(recorded)
    return out


class BendStiffnessCandidate(NamedTuple):
    """One grid point for segment bend stiffnesses."""

    primary: float
    secondary: float
    spur: float
    stem: float

    def to_overrides(self) -> dict[str, dict[str, float]]:
        return {
            "primary": {"bend_stiffness": float(self.primary)},
            "secondary": {"bend_stiffness": float(self.secondary)},
            "spur": {"bend_stiffness": float(self.spur)},
            "stem": {"bend_stiffness": float(self.stem)},
        }

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        out = base
        for segment, value in (
            ("primary", self.primary),
            ("secondary", self.secondary),
            ("spur", self.spur),
            ("stem", self.stem),
        ):
            if getattr(base, segment) is not None:
                out = fs.set_rod_bend_stiffness(out, segment, float(value))
        return out


def iter_bend_stiffness_candidates(
    *,
    primary_values: tuple[float, ...],
    secondary_values: tuple[float, ...],
    spur_values: tuple[float, ...],
    stem_values: tuple[float, ...],
):
    """Yield bend-stiffness grid candidates in Cartesian product order."""
    for primary, secondary, spur, stem in product(
        primary_values,
        secondary_values,
        spur_values,
        stem_values,
    ):
        yield BendStiffnessCandidate(
            primary=float(primary),
            secondary=float(secondary),
            spur=float(spur),
            stem=float(stem),
        )


def base_params_for_replay(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    *,
    use_oracle_params: bool = True,
):
    """Select structure build params: oracle (default) or obs-inferred digital twin."""
    if use_oracle_params:
        return true_params_for_structure(dataset, int(structure_idx))
    return infer_base_params_for_structure(dataset, int(structure_idx))


def gt_bend_stiffness_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> BendStiffnessCandidate:
    """Build the GT bend-stiffness candidate from recorded structure params."""
    params = true_params_for_structure(dataset, structure_idx)
    values: dict[str, float] = {}
    for segment in ROD_SEGMENTS:
        rod = getattr(params, segment)
        # Disabled segments use 0.0; BendStiffnessCandidate.apply_to skips None rods.
        values[segment] = 0.0 if rod is None else float(rod.bend_stiffness)
    return BendStiffnessCandidate(
        primary=values["primary"],
        secondary=values["secondary"],
        spur=values["spur"],
        stem=values["stem"],
    )


def ensure_gt_candidate_in_grid(
    candidates: list[BendStiffnessCandidate],
    gt: BendStiffnessCandidate,
) -> list[BendStiffnessCandidate]:
    """Ensure ``gt`` appears in the candidate list (within float tolerance), appending if missing."""
    for candidate in candidates:
        if bend_stiffness_values_match(candidate, gt):
            return list(candidates)
    if not candidates:
        return [gt]
    return [*candidates, gt]


def build_recorded_actions_tensor(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
) -> np.ndarray:
    """Stack recorded EE actions for all candidate/direction env slots."""
    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    direction_actions: list[np.ndarray] = []
    n_frames: int | None = None
    for direction_idx in dirs:
        arrays = strip_pre_weld_rows(
            dataset.load_episode_obs_arrays(structure_idx, int(direction_idx))
        )
        action = np.asarray(arrays["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 6:
            raise ValueError(
                f"expected action shape (n_frames, 6), got {action.shape!r}"
            )
        if n_frames is None:
            n_frames = int(action.shape[0])
        elif int(action.shape[0]) != n_frames:
            raise ValueError("all direction episodes must have same n_frames")
        direction_actions.append(action)

    if n_frames is None:
        raise ValueError("num_directions must be positive")

    d = len(dirs)
    num_envs = int(num_candidates) * d
    out = np.empty((num_envs, n_frames, 6), dtype=np.float32)
    for candidate_idx in range(num_candidates):
        for local_dir, _direction_idx in enumerate(dirs):
            env_idx = candidate_idx * d + local_dir
            out[env_idx] = direction_actions[local_dir]
    return out


def actions_tensor_from_recorded_frame(
    recorded_actions: np.ndarray,
    *,
    frame_idx: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return one recorded action frame for every env on ``device``."""
    frame = np.asarray(recorded_actions[:, frame_idx, :], dtype=np.float32)
    return torch.as_tensor(frame, device=device, dtype=torch.float32)


def replay_batched_sysid_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[BendStiffnessCandidate],
    num_directions: int,
    seed: int | None = None,
    build_env_fn: Callable[..., Any],
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
    use_snapshot: bool = False,
    use_oracle_params: bool = True,
    direction_indices: Sequence[int] | None = None,
    include_excluded: bool = False,
) -> BatchedSysIdReplayCollectors:
    """Replay recorded actions for one structure across bend-stiffness candidates."""
    num_candidates = len(candidates)
    if num_candidates < 1:
        raise ValueError("candidates must be non-empty")
    dirs = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        direction_indices=direction_indices,
        include_excluded=bool(include_excluded),
    )
    d = len(dirs)
    if d < 1:
        raise ValueError("num_directions must be >= 1")

    num_envs = num_candidates * d
    base_params = base_params_for_replay(
        dataset,
        int(structure_idx),
        use_oracle_params=bool(use_oracle_params),
    )
    per_env_params = broadcast_structure_params(
        [c.apply_to(base_params) for c in candidates],
        d,
    )
    recorded_actions = build_recorded_actions_tensor(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=d,
        num_candidates=num_candidates,
        direction_indices=dirs,
        include_excluded=bool(include_excluded),
    )
    n_frames = int(recorded_actions.shape[1])
    replay_seed = _resolve_replay_seed(dataset, seed)
    structure_meta = dataset.load_episode_metadata(int(structure_idx), int(dirs[0]))
    replay_gripper = gripper_proxy_from_episode_metadata(structure_meta)

    env = build_env_fn(
        num_envs=num_envs,
        per_env_params=per_env_params,
        max_episode_steps=n_frames,
        gripper=replay_gripper,
    )
    try:
        # Compare against the effective sim config used by the environment
        # (the sys-ID env may override controller clip limits, allocate buffers, etc.).
        if replay_sim_config is not None:
            warn_manifest_sim_config_mismatch(dataset, env._sim.config)
        if use_snapshot:
            from apple_pick_sim.system_id.batched_episode_snapshot_io import (
                load_and_restore_episode_snapshots,
            )

            load_and_restore_episode_snapshots(
                env,
                dataset,
                structure_idx=int(structure_idx),
                num_directions=d,
                direction_indices=dirs,
            )
        else:
            env.reset(seed=replay_seed)
            initialize_batched_env_from_dataset(
                env,
                dataset,
                structure_idx=int(structure_idx),
                num_directions=d,
                direction_indices=dirs,
            )
        initial_unstable = ik_bootstrap_unstable_mask(env, num_envs)
        last_obs = getattr(env, "_last_obs", None)
        if last_obs is None:
            raise RuntimeError("env._last_obs missing after reset")
        monitor = BatchedStabilityMonitor(
            num_envs,
            known_obs_keys=set(last_obs.keys()),
            initial_unstable=initial_unstable,
        )
        disable_ctrl = EnvDisableController(
            num_envs,
            device=env.device,
            initial_disabled=initial_unstable,
        )
        recorded_by_env = recorded_metadata_by_env(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=d,
            num_candidates=num_candidates,
            direction_indices=dirs,
            include_excluded=bool(include_excluded),
        )
        collectors = BatchedSysIdReplayCollectors(num_envs, recorded_by_env)

        for frame_idx in range(n_frames):
            actions = actions_tensor_from_recorded_frame(
                recorded_actions,
                frame_idx=frame_idx,
                device=env.device,
            )
            actions = disable_ctrl.apply_actions(actions)
            env.step(actions)
            if on_step is not None:
                keep_going = bool(on_step(frame_idx=frame_idx, env=env))
                if not keep_going:
                    break
            last_obs = getattr(env, "_last_obs", None)
            if last_obs is None:
                raise RuntimeError("env._last_obs missing after step")
            step_report = monitor.check(last_obs, step_idx=int(frame_idx))
            collectors.record_all_envs_step(
                env,
                frame_idx=frame_idx,
                unstable=step_report.unstable,
                record_mask=disable_ctrl.should_record_mask(),
            )
            disable_ctrl.update(hard_blowup_mask(step_report))
    finally:
        env.close()

    return collectors
