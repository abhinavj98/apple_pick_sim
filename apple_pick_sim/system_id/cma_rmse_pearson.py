"""Per-feature RMSE and Pearson for CMA holdout fitted replays.

Defaults: full-episode window, fitted side, phase-corrected (best-lag) alignment.

Features
--------
Force/torque (prefer ``match_metrics`` RMSE when present; Pearson from NPZ)::

    fx, fy, fz, force_combined
    tx, ty, tz, torque_combined

Pose / kinematics (from STATE_VECTOR NPZ; pull axis from dataset manifest)::

    tcp_disp_along_pull   — signed (tcp_pos - tcp_pos[0]) · pull_hat
    tcp_orientation       — tcp_rotvec (T, 3); combined magnitude phase metrics

Woody junctions (STATE_VECTOR woody_part_start_pos; displacement peak mode)::

    woody_primary_spur, woody_spur_stem
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from apple_pick_sim.system_id.holdout_gates import signed_parallel_series
from apple_pick_sim.system_id.match_metrics import (
    best_cross_corr_lag,
    scalar_match_stats,
    vector3_match_stats,
)

_RUN_DIR_RE = re.compile(
    r"^cma_s(?P<structure>\d+)_val\d+_seed(?P<seed>\d+)$"
)

DEFAULT_FEATURES: tuple[str, ...] = (
    "fx",
    "fy",
    "fz",
    "force_combined",
    "tx",
    "ty",
    "tz",
    "torque_combined",
    "tcp_disp_along_pull",
    "tcp_orientation",
    "woody_primary_spur",
    "woody_spur_stem",
)

# STATE_VECTOR column slices (see mmd_features.build_state_matrix).
_FORCE_SLICE = slice(0, 3)
_TORQUE_SLICE = slice(3, 6)
_TCP_POS_SLICE = slice(12, 15)
_TCP_ROTVEC_SLICE = slice(15, 18)
_WOODY_PRIMARY_SPUR_SLICE = slice(18, 21)
_WOODY_SPUR_STEM_SLICE = slice(21, 24)

_SCALAR_FORCE = {"fx": 0, "fy": 1, "fz": 2}
_SCALAR_TORQUE = {"tx": 0, "ty": 1, "tz": 2}

# match_metrics uses displacement peak mode for woody junctions.
_FEATURE_PEAK_MODE: dict[str, str] = {
    "woody_primary_spur": "displacement",
    "woody_spur_stem": "displacement",
}

# match_metrics.json path under directions[d][window][fitted]
_MATCH_METRICS_PATH: dict[str, tuple[str, str]] = {
    "fx": ("force", "fx"),
    "fy": ("force", "fy"),
    "fz": ("force", "fz"),
    "force_combined": ("force", "combined"),
    "tx": ("torque", "tx"),
    "ty": ("torque", "ty"),
    "tz": ("torque", "tz"),
    "torque_combined": ("torque", "combined"),
    "woody_primary_spur": ("woody", "primary_spur"),
    "woody_spur_stem": ("woody", "spur_stem"),
}


@dataclass(frozen=True)
class FeatureMatch:
    feature: str
    rmse: float
    pearson: float


@dataclass(frozen=True)
class DirectionFeatureMatch:
    direction: int
    n_frames: int
    features: tuple[FeatureMatch, ...]


@dataclass(frozen=True)
class RunFeatureMatch:
    structure: int
    seed: int
    run_path: Path
    per_direction: tuple[DirectionFeatureMatch, ...]


@dataclass(frozen=True)
class DirectionForceMatch:
    direction: int
    rmse: float
    pearson: float
    n_frames: int


@dataclass(frozen=True)
class RunForceMatch:
    structure: int
    seed: int
    run_path: Path
    per_direction: tuple[DirectionForceMatch, ...]

    @property
    def mean_rmse(self) -> float:
        vals = [d.rmse for d in self.per_direction if math.isfinite(d.rmse)]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    @property
    def mean_pearson(self) -> float:
        vals = [d.pearson for d in self.per_direction if math.isfinite(d.pearson)]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))


def _aligned_pair(real: np.ndarray, sim: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(real, dtype=np.float64).reshape(-1)
    sim = np.asarray(sim, dtype=np.float64).reshape(-1)
    n = min(int(real.size), int(sim.size))
    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    lag = int(lag)
    if lag >= n or -lag >= n:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if lag >= 0:
        return real[: n - lag], sim[lag:n]
    lead = -lag
    return real[lead:n], sim[: n - lead]


def _pearson(real: np.ndarray, sim: np.ndarray) -> float:
    if real.size < 2 or sim.size < 2 or real.size != sim.size:
        return float("nan")
    if float(np.std(real)) == 0.0 or float(np.std(sim)) == 0.0:
        return 1.0 if np.allclose(real, sim) else 0.0
    return float(np.corrcoef(real, sim)[0, 1])


def scalar_rmse_and_pearson(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    max_lag: int = 30,
) -> tuple[float, float]:
    """Phase-corrected RMSE and Pearson for one 1D series pair."""
    stats = scalar_match_stats(real, sim, max_lag=max_lag)
    rmse = float(stats["phase_corrected_rmse"])
    lag_raw = float(stats["cross_corr_lag"])
    lag = (
        int(lag_raw)
        if math.isfinite(lag_raw)
        else best_cross_corr_lag(
            np.asarray(real, dtype=np.float64).reshape(-1),
            np.asarray(sim, dtype=np.float64).reshape(-1),
            max_lag=max_lag,
        )
    )
    aligned_r, aligned_s = _aligned_pair(real, sim, lag)
    return rmse, _pearson(aligned_r, aligned_s)


def vector3_rmse_and_pearson(
    real_vec: np.ndarray,
    sim_vec: np.ndarray,
    *,
    max_lag: int = 30,
    peak_mode: str = "norm",
) -> tuple[float, float]:
    """Phase-corrected (T, 3) RMSE and Pearson on peak-mode magnitude series."""
    stats = vector3_match_stats(
        real_vec, sim_vec, max_lag=max_lag, peak_mode=peak_mode
    )
    rmse = float(stats["phase_corrected_rmse"])
    real = np.asarray(real_vec, dtype=np.float64)
    sim = np.asarray(sim_vec, dtype=np.float64)
    if real.ndim == 1:
        real = real.reshape(-1, 3)
    if sim.ndim == 1:
        sim = sim.reshape(-1, 3)
    n = min(int(real.shape[0]), int(sim.shape[0]))
    if n <= 0:
        return rmse, float("nan")
    real = real[:n]
    sim = sim[:n]
    if peak_mode == "displacement":
        real_mag = np.linalg.norm(real - real[0:1], axis=1)
        sim_mag = np.linalg.norm(sim - sim[0:1], axis=1)
    else:
        real_mag = np.linalg.norm(real, axis=1)
        sim_mag = np.linalg.norm(sim, axis=1)
    lag_raw = float(stats["cross_corr_lag"])
    lag = (
        int(lag_raw)
        if math.isfinite(lag_raw)
        else best_cross_corr_lag(real_mag, sim_mag, max_lag=max_lag)
    )
    aligned_r, aligned_s = _aligned_pair(real_mag, sim_mag, lag)
    return rmse, _pearson(aligned_r, aligned_s)


def force_rmse_and_pearson(
    real_force: np.ndarray,
    sim_force: np.ndarray,
    *,
    max_lag: int = 30,
) -> tuple[float, float]:
    """Phase-corrected vector-magnitude RMSE and Pearson (aligned magnitudes)."""
    return vector3_rmse_and_pearson(
        real_force, sim_force, max_lag=max_lag, peak_mode="norm"
    )


def feature_rmse_and_pearson(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    max_lag: int = 30,
    peak_mode: str = "norm",
) -> tuple[float, float]:
    """Dispatch scalar vs (T, 3) combined phase-corrected RMSE / Pearson."""
    real_arr = np.asarray(real, dtype=np.float64)
    sim_arr = np.asarray(sim, dtype=np.float64)
    if real_arr.ndim == 1 or (real_arr.ndim == 2 and real_arr.shape[1] == 1):
        return scalar_rmse_and_pearson(real_arr.reshape(-1), sim_arr.reshape(-1), max_lag=max_lag)
    if real_arr.ndim == 2 and real_arr.shape[1] == 3:
        return vector3_rmse_and_pearson(
            real_arr, sim_arr, max_lag=max_lag, peak_mode=peak_mode
        )
    raise ValueError(f"unsupported series shape {real_arr.shape}")


def extract_feature_series(
    real_state: np.ndarray,
    sim_state: np.ndarray,
    feature: str,
    *,
    pull_direction: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract real/sim series for one named feature from STATE_VECTOR rows."""
    real = np.asarray(real_state, dtype=np.float64)
    sim = np.asarray(sim_state, dtype=np.float64)
    if real.ndim != 2 or sim.ndim != 2:
        raise ValueError("real_state and sim_state must be 2D")
    n = min(int(real.shape[0]), int(sim.shape[0]))
    real = real[:n]
    sim = sim[:n]
    name = str(feature)

    if name in _SCALAR_FORCE:
        idx = _SCALAR_FORCE[name]
        return real[:, idx], sim[:, idx]
    if name in _SCALAR_TORQUE:
        idx = _SCALAR_TORQUE[name]
        return real[:, 3 + idx], sim[:, 3 + idx]
    if name == "force_combined":
        return real[:, _FORCE_SLICE], sim[:, _FORCE_SLICE]
    if name == "torque_combined":
        return real[:, _TORQUE_SLICE], sim[:, _TORQUE_SLICE]
    if name == "tcp_orientation":
        return real[:, _TCP_ROTVEC_SLICE], sim[:, _TCP_ROTVEC_SLICE]
    if name == "tcp_disp_along_pull":
        if pull_direction is None:
            raise ValueError("pull_direction is required for tcp_disp_along_pull")
        real_pos = real[:, _TCP_POS_SLICE]
        sim_pos = sim[:, _TCP_POS_SLICE]
        real_along = signed_parallel_series(real_pos, pull_direction)
        sim_along = signed_parallel_series(sim_pos, pull_direction)
        return real_along - real_along[0], sim_along - sim_along[0]
    if name == "woody_primary_spur":
        return real[:, _WOODY_PRIMARY_SPUR_SLICE], sim[:, _WOODY_PRIMARY_SPUR_SLICE]
    if name == "woody_spur_stem":
        return real[:, _WOODY_SPUR_STEM_SLICE], sim[:, _WOODY_SPUR_STEM_SLICE]
    raise ValueError(f"unknown feature {feature!r}")


def _mask_states_from_npz(
    payload: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    real_state = np.asarray(payload["real_state"], dtype=np.float64)
    sim_state = np.asarray(payload["sim_state"], dtype=np.float64)
    stable_real = np.asarray(
        payload.get("stable_real", np.ones(real_state.shape[0], dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    stable_sim = np.asarray(
        payload.get("stable_sim", np.ones(sim_state.shape[0], dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    real_m = real_state[stable_real]
    sim_m = sim_state[stable_sim]
    n = min(int(real_m.shape[0]), int(sim_m.shape[0]))
    return real_m[:n], sim_m[:n]


def _mask_force_from_npz(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Extract masked force (T, 3) real/sim from a holdout fitted npz payload."""
    real_m, sim_m = _mask_states_from_npz(payload)
    return real_m[:, _FORCE_SLICE], sim_m[:, _FORCE_SLICE]


def _rmse_from_match_metrics(
    report: dict,
    direction: int,
    feature: str,
    *,
    window: str = "full",
) -> float:
    path = _MATCH_METRICS_PATH.get(feature)
    if path is None:
        return float("nan")
    group, key = path
    try:
        block = report["directions"][str(int(direction))][window]["fitted"][group][key]
    except KeyError:
        return float("nan")
    value = block.get("phase_corrected_rmse")
    if value is None:
        return float("nan")
    number = float(value)
    return number if math.isfinite(number) else float("nan")


def resolve_dataset_root(run_dir: Path, *, repo_root: Path | None = None) -> Path | None:
    """Resolve ``cmaes_report.json`` dataset path relative to run/repo."""
    report_path = run_dir / "cmaes_report.json"
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text())
    raw = payload.get("dataset")
    if not raw:
        return None
    candidate = Path(str(raw))
    if candidate.is_absolute() and candidate.is_dir():
        return candidate
    for base in (run_dir, run_dir.parent, repo_root or Path.cwd()):
        if base is None:
            continue
        resolved = (base / candidate).resolve()
        if resolved.is_dir():
            return resolved
    return None


def load_pull_direction(
    dataset_root: Path | str,
    direction: int,
    *,
    structure_idx: int = 0,
) -> np.ndarray:
    """Load unit-ish pull_direction for one direction from a batched dataset manifest."""
    manifest_path = Path(dataset_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for episode in manifest.get("episodes", []):
        if int(episode.get("direction_idx", -1)) != int(direction):
            continue
        if int(episode.get("structure_idx", 0)) != int(structure_idx):
            continue
        if bool(episode.get("excluded", False)):
            continue
        pull = np.asarray(episode["pull_direction"], dtype=np.float64).reshape(3)
        return pull
    raise KeyError(
        f"no episode for direction={direction} structure_idx={structure_idx} in {manifest_path}"
    )


def load_run_feature_rmse_pearson(
    run_dir: Path | str,
    *,
    features: Sequence[str] = DEFAULT_FEATURES,
    window: str = "full",
    structure_subdir: str = "structure_000",
    dataset_root: Path | str | None = None,
    repo_root: Path | None = None,
) -> RunFeatureMatch:
    """Score each holdout val direction for the requested features."""
    run_path = Path(run_dir)
    match = _RUN_DIR_RE.match(run_path.name)
    report_path = run_path / "match_metrics.json"
    report = json.loads(report_path.read_text())
    if match is not None:
        structure = int(match.group("structure"))
        seed = int(match.group("seed"))
    else:
        structure = int(str(report.get("tree", "s0")).lstrip("sS") or 0)
        seed = int(report.get("cma_seed", -1))

    val_dirs = [int(d) for d in report.get("val_direction_indices", [])]
    if not val_dirs:
        val_dirs = sorted(int(d) for d in report.get("directions", {}))

    feature_names = tuple(str(f) for f in features)
    needs_pull = "tcp_disp_along_pull" in feature_names
    ds_root: Path | None
    if dataset_root is not None:
        ds_root = Path(dataset_root)
    else:
        ds_root = resolve_dataset_root(run_path, repo_root=repo_root)
    if needs_pull and ds_root is None:
        raise FileNotFoundError(
            f"dataset root required for tcp_disp_along_pull but missing for {run_path}"
        )

    fitted_dir = run_path / structure_subdir / "holdout" / "fitted"
    per_dir: list[DirectionFeatureMatch] = []
    for direction in val_dirs:
        npz_path = fitted_dir / f"dir_{direction:02d}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        real_m, sim_m = _mask_states_from_npz(arrays)
        pull = None
        if needs_pull:
            assert ds_root is not None
            pull = load_pull_direction(ds_root, direction, structure_idx=0)

        feature_rows: list[FeatureMatch] = []
        for feature in feature_names:
            real_s, sim_s = extract_feature_series(
                real_m, sim_m, feature, pull_direction=pull
            )
            peak_mode = _FEATURE_PEAK_MODE.get(feature, "norm")
            rmse_npz, pearson = feature_rmse_and_pearson(
                real_s, sim_s, peak_mode=peak_mode
            )
            rmse_json = _rmse_from_match_metrics(
                report, direction, feature, window=window
            )
            rmse = rmse_json if math.isfinite(rmse_json) else rmse_npz
            feature_rows.append(
                FeatureMatch(
                    feature=feature,
                    rmse=float(rmse),
                    pearson=float(pearson),
                )
            )
        per_dir.append(
            DirectionFeatureMatch(
                direction=int(direction),
                n_frames=int(min(real_m.shape[0], sim_m.shape[0])),
                features=tuple(feature_rows),
            )
        )
    return RunFeatureMatch(
        structure=structure,
        seed=seed,
        run_path=run_path,
        per_direction=tuple(per_dir),
    )


def load_run_force_rmse_pearson(
    run_dir: Path | str,
    *,
    window: str = "full",
    structure_subdir: str = "structure_000",
) -> RunForceMatch:
    """Backward-compatible force-combined-only loader."""
    row = load_run_feature_rmse_pearson(
        run_dir,
        features=("force_combined",),
        window=window,
        structure_subdir=structure_subdir,
        # force_combined does not need dataset / pull
        dataset_root=None,
    )
    return RunForceMatch(
        structure=row.structure,
        seed=row.seed,
        run_path=row.run_path,
        per_direction=tuple(
            DirectionForceMatch(
                direction=d.direction,
                rmse=d.features[0].rmse,
                pearson=d.features[0].pearson,
                n_frames=d.n_frames,
            )
            for d in row.per_direction
        ),
    )


def discover_run_dirs(
    root: Path | str,
    *,
    structures: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
) -> list[Path]:
    root_path = Path(root)
    struct_filter = None if structures is None else {int(s) for s in structures}
    seed_filter = None if seeds is None else {int(s) for s in seeds}
    found: list[Path] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if match is None:
            continue
        structure = int(match.group("structure"))
        seed = int(match.group("seed"))
        if struct_filter is not None and structure not in struct_filter:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        if (child / "match_metrics.json").is_file():
            found.append(child)
    return found


def summarize_feature_runs(
    root: Path | str,
    *,
    structures: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    features: Sequence[str] = DEFAULT_FEATURES,
    window: str = "full",
    repo_root: Path | None = None,
) -> list[RunFeatureMatch]:
    """Score all matching CMA runs under ``root`` for the requested features."""
    root_path = Path(root)
    return [
        load_run_feature_rmse_pearson(
            run_dir,
            features=features,
            window=window,
            repo_root=repo_root or root_path.parent,
        )
        for run_dir in discover_run_dirs(root, structures=structures, seeds=seeds)
    ]


def summarize_runs(
    root: Path | str,
    *,
    structures: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    window: str = "full",
) -> list[RunForceMatch]:
    """Score all matching CMA runs (force_combined only)."""
    return [
        load_run_force_rmse_pearson(run_dir, window=window)
        for run_dir in discover_run_dirs(root, structures=structures, seeds=seeds)
    ]
