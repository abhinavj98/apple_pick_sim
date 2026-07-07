"""Bend-stiffness grid and recorded-action tensor helpers for batched sys-ID MMD."""

from __future__ import annotations

from itertools import product
from typing import NamedTuple

import numpy as np
import torch

from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.batched_digital_twin_init import infer_base_params_for_structure
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")


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


def gt_bend_stiffness_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> BendStiffnessCandidate:
    """Build the GT bend-stiffness candidate from inferred structure params."""
    params = infer_base_params_for_structure(dataset, structure_idx)
    values: dict[str, float] = {}
    for segment in ROD_SEGMENTS:
        rod = getattr(params, segment)
        if rod is None:
            raise ValueError(f"Segment {segment!r} is missing in inferred params")
        values[segment] = float(rod.bend_stiffness)
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
    """Ensure ``gt`` appears in the candidate list, replacing the last entry if needed."""
    for candidate in candidates:
        if (
            candidate.primary == gt.primary
            and candidate.secondary == gt.secondary
            and candidate.spur == gt.spur
            and candidate.stem == gt.stem
        ):
            return list(candidates)
    if not candidates:
        return [gt]
    updated = list(candidates)
    updated[-1] = gt
    return updated


def build_recorded_actions_tensor(
    dataset: BatchedSysIdDataset,
    *,
    structure_idx: int,
    num_directions: int,
    num_candidates: int,
) -> np.ndarray:
    """Stack recorded EE actions for all candidate/direction env slots."""
    direction_actions: list[np.ndarray] = []
    n_frames: int | None = None
    for direction_idx in range(num_directions):
        arrays = dataset.load_episode_obs_arrays(structure_idx, direction_idx)
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

    num_envs = int(num_candidates) * int(num_directions)
    out = np.empty((num_envs, n_frames, 6), dtype=np.float32)
    for candidate_idx in range(num_candidates):
        for direction_idx in range(num_directions):
            env_idx = candidate_idx * num_directions + direction_idx
            out[env_idx] = direction_actions[direction_idx]
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
