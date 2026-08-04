"""CMA-ES / Young's-modulus candidate helpers for batched sys-ID (V.5.2).

Owns material candidate maps, bounded pycma construction, synchronized
generation waves, and fit reporting for the separate CMA-ES entry point.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple

import cma
import numpy as np

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.fruiting_system import params as fs
from apple_pick_sim.fruiting_system.params import FruitingSystemParams
from apple_pick_sim.system_id.batched_digital_twin_init import (
    gripper_proxy_from_episode_metadata,
    true_params_for_structure,
)
from apple_pick_sim.system_id.wasserstein import (
    WassersteinScoringContext,
    prepare_gt_wasserstein_scoring_context,
    score_candidate_wasserstein_complete,
)
from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
    UNSTABLE_DISQUALIFY_THRESHOLD,
    direction_episodes_from_collectors,
    load_recorded_episodes_for_structure,
    replay_candidates_for_structure,
    replay_instability_fraction_all_frames,
    resolve_direction_indices,
)
from apple_pick_gym.batched_envs.batched_sysid_multi_replay import (
    MultiStructureReplayDiagnostics,
    ReplayFusionIncompatible,
    ReplaySlotKey,
    ReplayStructureRequest,
    SysIdReplayCancelled,
    build_replay_candidate_blocks,
    replay_multi_structure_candidate_blocks,
)

if TYPE_CHECKING:
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

DEFAULT_INITIAL_SIGMA_LOG10 = 1.0
_UINT32_MOD = 2**32


class YoungsModulusCandidate(NamedTuple):
    """One material candidate: Young's modulus (Pa) for primary, spur, stem."""

    primary: float
    spur: float
    stem: float

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        """Return a copy with candidate ``E`` re-derived into VBD knobs.

        Only ``primary``, ``spur``, and ``stem`` are updated when present on
        ``base``. ``secondary`` (and any other fields) are left unchanged.
        Geometry and ``damping_ratio`` are frozen; axial stretch overrides on
        the base rod are preserved when they differ from beam theory.
        """
        out = base
        for segment, value in (
            ("primary", self.primary),
            ("spur", self.spur),
            ("stem", self.stem),
        ):
            if getattr(base, segment) is not None:
                out = fs.set_rod_youngs_modulus(out, segment, float(value))
        return out

    def short_label(self) -> str:
        """Compact legend label."""
        return (
            f"log10=({math.log10(self.primary):.2f},"
            f"{math.log10(self.spur):.2f},"
            f"{math.log10(self.stem):.2f})"
        )


def iter_youngs_modulus_candidates(
    *,
    primary_values: Sequence[float],
    spur_values: Sequence[float],
    stem_values: Sequence[float],
) -> Iterable[YoungsModulusCandidate]:
    """Yield Young's-modulus grid candidates in Cartesian product order."""
    for primary, spur, stem in product(primary_values, spur_values, stem_values):
        yield YoungsModulusCandidate(
            primary=float(primary),
            spur=float(spur),
            stem=float(stem),
        )


def candidates_from_log10_e(
    log10_e: Sequence[float],
) -> YoungsModulusCandidate:
    """Map ``log10([E_primary, E_spur, E_stem])`` to a physical candidate."""
    if len(log10_e) != 3:
        raise ValueError(f"log10_e must have length 3, got {len(log10_e)}")
    return YoungsModulusCandidate(
        primary=10.0 ** float(log10_e[0]),
        spur=10.0 ** float(log10_e[1]),
        stem=10.0 ** float(log10_e[2]),
    )


def log10_e_from_params(params: FruitingSystemParams) -> tuple[float, float, float]:
    """Extract ``log10(E)`` for primary, spur, stem (hard error if missing)."""
    if params.primary is None or params.spur is None or params.stem is None:
        raise ValueError(
            "params must include primary, spur, and stem rods for log10_e_from_params"
        )
    return (
        math.log10(float(params.primary.youngs_modulus_pa)),
        math.log10(float(params.spur.youngs_modulus_pa)),
        math.log10(float(params.stem.youngs_modulus_pa)),
    )


def youngs_modulus_candidate_from_params(
    params: FruitingSystemParams,
) -> YoungsModulusCandidate:
    """Build a candidate from absolute ``E`` on primary/spur/stem."""
    if params.primary is None or params.spur is None or params.stem is None:
        raise ValueError(
            "params must include primary, spur, and stem rods"
        )
    return YoungsModulusCandidate(
        primary=float(params.primary.youngs_modulus_pa),
        spur=float(params.spur.youngs_modulus_pa),
        stem=float(params.stem.youngs_modulus_pa),
    )


def _candidate_stiffness_diagnostics(candidate: Any) -> dict[str, float]:
    """Diagnostic-only stiffness map recorded on Wasserstein scoring results.

    Not consumed by the Sinkhorn math itself; only ``SupportKpYoungsCandidate``
    (support_kp, spur, stem) and ``YoungsModulusCandidate`` (primary, spur,
    stem) are expected here.
    """
    support_kp = getattr(candidate, "support_kp", None)
    if support_kp is not None:
        return {
            "support_kp": float(support_kp),
            "spur_e_pa": float(candidate.spur),
            "stem_e_pa": float(candidate.stem),
        }
    return {
        "primary_e_pa": float(candidate.primary),
        "spur_e_pa": float(candidate.spur),
        "stem_e_pa": float(candidate.stem),
    }


def gt_youngs_modulus_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> YoungsModulusCandidate:
    return youngs_modulus_candidate_from_params(
        true_params_for_structure(dataset, int(structure_idx))
    )


def youngs_modulus_values_match(
    left: YoungsModulusCandidate,
    right: YoungsModulusCandidate,
    *,
    log10_atol: float = 1e-9,
) -> bool:
    return all(
        math.isclose(math.log10(a), math.log10(b), rel_tol=0.0, abs_tol=log10_atol)
        for a, b in zip(left, right, strict=True)
    )


def maybe_include_gt_candidate(
    candidates: Sequence[YoungsModulusCandidate],
    gt: YoungsModulusCandidate,
    *,
    include_gt: bool,
) -> list[YoungsModulusCandidate]:
    items = list(candidates)
    if not include_gt or any(youngs_modulus_values_match(item, gt) for item in items):
        return items
    return [*items, gt]


class SupportKpYoungsCandidate(NamedTuple):
    """One sys-ID candidate: support joint k_p plus spur/stem Young's modulus (Pa)."""

    support_kp: float
    spur: float
    stem: float

    def apply_to(self, base: FruitingSystemParams) -> FruitingSystemParams:
        """Return a copy with spur/stem ``E`` re-derived into VBD knobs.

        Primary (and secondary) material is left unchanged. ``support_kp`` is
        not applied here — fused replay patches support joints per env.
        """
        out = base
        for segment, value in (
            ("spur", self.spur),
            ("stem", self.stem),
        ):
            if getattr(base, segment) is not None:
                out = fs.set_rod_youngs_modulus(out, segment, float(value))
        return out

    def short_label(self) -> str:
        """Compact legend label."""
        return (
            f"log10=({math.log10(self.support_kp):.2f},"
            f"{math.log10(self.spur):.2f},"
            f"{math.log10(self.stem):.2f})"
        )


def iter_support_kp_youngs_candidates(
    *,
    support_kp_values: Sequence[float],
    spur_values: Sequence[float],
    stem_values: Sequence[float],
) -> Iterable[SupportKpYoungsCandidate]:
    """Yield support-k_p x spur-E x stem-E grid candidates (Cartesian order)."""
    for support_kp, spur, stem in product(support_kp_values, spur_values, stem_values):
        yield SupportKpYoungsCandidate(
            support_kp=float(support_kp),
            spur=float(spur),
            stem=float(stem),
        )


def candidates_from_log10_vector(
    log10_vector: Sequence[float],
) -> SupportKpYoungsCandidate:
    """Map ``log10([k_p_support, E_spur, E_stem])`` to a physical candidate."""
    if len(log10_vector) != 3:
        raise ValueError(
            f"log10_vector must have length 3, got {len(log10_vector)}"
        )
    return SupportKpYoungsCandidate(
        support_kp=10.0 ** float(log10_vector[0]),
        spur=10.0 ** float(log10_vector[1]),
        stem=10.0 ** float(log10_vector[2]),
    )


def log10_vector_from_candidate(
    candidate: SupportKpYoungsCandidate,
) -> tuple[float, float, float]:
    """Extract ``log10([k_p_support, E_spur, E_stem])``."""
    return (
        math.log10(float(candidate.support_kp)),
        math.log10(float(candidate.spur)),
        math.log10(float(candidate.stem)),
    )


def gt_support_kp_from_dataset(dataset: BatchedSysIdDataset) -> float:
    """Read dataset-level GT support k_p from ``manifest['collection']['sim_config']``.

  ``joint_angular_kp_overrides``/``joint_linear_kp_overrides`` are a single
  build-time config for the whole collection run, not per-structure.
  """
    dataset_path = str(dataset.dataset_dir)
    sim_config = dataset.manifest.get("collection", {}).get("sim_config")
    if sim_config is None:
        raise ValueError(
            f"manifest collection.sim_config missing in dataset {dataset_path}"
        )

    angular = sim_config.get("joint_angular_kp_overrides")
    linear = sim_config.get("joint_linear_kp_overrides")
    if not isinstance(angular, Mapping) or "support" not in angular:
        raise ValueError(
            f"joint_angular_kp_overrides['support'] missing in dataset {dataset_path}"
        )
    if not isinstance(linear, Mapping) or "support" not in linear:
        raise ValueError(
            f"joint_linear_kp_overrides['support'] missing in dataset {dataset_path}"
        )

    ang_kp = float(angular["support"])
    lin_kp = float(linear["support"])
    if not math.isclose(ang_kp, lin_kp, rel_tol=1e-9):
        raise ValueError(
            f"support joint_angular_kp_overrides ({ang_kp}) disagrees with "
            f"joint_linear_kp_overrides ({lin_kp}) in dataset {dataset_path}"
        )
    return ang_kp


def gt_support_kp_youngs_candidate_from_structure(
    dataset: BatchedSysIdDataset,
    structure_idx: int,
) -> SupportKpYoungsCandidate:
    params = true_params_for_structure(dataset, int(structure_idx))
    if params.spur is None or params.stem is None:
        raise ValueError(
            "params must include spur and stem rods for "
            "gt_support_kp_youngs_candidate_from_structure"
        )
    return SupportKpYoungsCandidate(
        support_kp=gt_support_kp_from_dataset(dataset),
        spur=float(params.spur.youngs_modulus_pa),
        stem=float(params.stem.youngs_modulus_pa),
    )


@dataclass(frozen=True)
class SegmentYoungsModulusBounds:
    physical_min_pa: float
    physical_max_pa: float
    log10_min: float
    log10_max: float
    log10_midpoint: float


@dataclass(frozen=True)
class YoungsModulusCmaBounds:
    primary: SegmentYoungsModulusBounds
    spur: SegmentYoungsModulusBounds
    stem: SegmentYoungsModulusBounds

    @property
    def log10_lower(self) -> tuple[float, float, float]:
        return (
            self.primary.log10_min,
            self.spur.log10_min,
            self.stem.log10_min,
        )

    @property
    def log10_upper(self) -> tuple[float, float, float]:
        return (
            self.primary.log10_max,
            self.spur.log10_max,
            self.stem.log10_max,
        )

    @property
    def log10_midpoint(self) -> tuple[float, float, float]:
        return (
            self.primary.log10_midpoint,
            self.spur.log10_midpoint,
            self.stem.log10_midpoint,
        )


def _require_positive_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _segment_youngs_bounds(ranges: Mapping[str, Any], segment: str) -> SegmentYoungsModulusBounds:
    segment_payload = ranges.get(segment)
    if not isinstance(segment_payload, Mapping):
        raise ValueError(f"{segment} youngs_modulus_pa bounds are missing")
    youngs = segment_payload.get("youngs_modulus_pa")
    if not isinstance(youngs, Mapping):
        raise ValueError(f"{segment} youngs_modulus_pa bounds are missing")
    if "min" not in youngs or "max" not in youngs:
        raise ValueError(f"{segment} youngs_modulus_pa bounds are missing")
    if youngs.get("min") is None or youngs.get("max") is None:
        raise ValueError(f"{segment} youngs_modulus_pa bounds are missing")
    physical_min = _require_positive_finite_number(
        youngs["min"], field=f"{segment}.youngs_modulus_pa.min"
    )
    physical_max = _require_positive_finite_number(
        youngs["max"], field=f"{segment}.youngs_modulus_pa.max"
    )
    if physical_min >= physical_max:
        raise ValueError(
            f"{segment} youngs_modulus_pa bounds must satisfy min < max"
        )
    log10_min = math.log10(physical_min)
    log10_max = math.log10(physical_max)
    return SegmentYoungsModulusBounds(
        physical_min_pa=physical_min,
        physical_max_pa=physical_max,
        log10_min=log10_min,
        log10_max=log10_max,
        log10_midpoint=0.5 * (log10_min + log10_max),
    )


def extract_youngs_modulus_cma_bounds(
    ranges: Mapping[str, Any],
) -> YoungsModulusCmaBounds:
    """Extract immutable primary/spur/stem Young's bounds from a ranges fixture."""
    return YoungsModulusCmaBounds(
        primary=_segment_youngs_bounds(ranges, "primary"),
        spur=_segment_youngs_bounds(ranges, "spur"),
        stem=_segment_youngs_bounds(ranges, "stem"),
    )


def validate_initial_sigma_log10(sigma: float) -> float:
    """Reject non-positive or non-finite initial sigma (log10 decades)."""
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise ValueError("initial sigma must be numeric")
    value = float(sigma)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("initial sigma must be positive and finite")
    return value


def resolve_initial_mean_log10(
    spec: Any,
    bounds: YoungsModulusCmaBounds,
) -> tuple[float, float, float]:
    """Resolve CMA start mean in log10-E coordinates.

    ``\"bounds_midpoint\"`` (or ``None``) uses fixture midpoints. Otherwise
    ``spec`` must be a length-3 finite numeric sequence (fixture box is not a
    search constraint unless ``search_bounds_log10`` is configured).
    """
    if spec is None or spec == "bounds_midpoint":
        return bounds.log10_midpoint
    try:
        values = tuple(float(v) for v in spec)
    except TypeError as exc:
        raise ValueError(
            "initial_mean_log10 must be 'bounds_midpoint' or a length-3 sequence"
        ) from exc
    if len(values) != 3:
        raise ValueError(
            f"initial_mean_log10 must have length 3, got {len(values)}"
        )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("initial_mean_log10 must be finite")
    return values  # type: ignore[return-value]


def normalize_search_bounds_log10(
    spec: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Parse CMA search box; ``None`` means unbounded (no pycma clipping).

    Accepted forms:
    - ``None`` → unbounded
    - ``{"lower": [p,s,t], "upper": [p,s,t]}`` in log10-E
    """
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError(
            "search_bounds_log10 must be None or a mapping with lower/upper"
        )
    if "lower" not in spec or "upper" not in spec:
        raise ValueError("search_bounds_log10 requires 'lower' and 'upper'")
    try:
        lower = tuple(float(v) for v in spec["lower"])
        upper = tuple(float(v) for v in spec["upper"])
    except TypeError as exc:
        raise ValueError(
            "search_bounds_log10 lower/upper must be length-3 sequences"
        ) from exc
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("search_bounds_log10 lower/upper must have length 3")
    if not all(math.isfinite(v) for v in (*lower, *upper)):
        raise ValueError("search_bounds_log10 lower/upper must be finite")
    for lo, hi in zip(lower, upper, strict=True):
        if lo >= hi:
            raise ValueError("search_bounds_log10 requires lower < upper per axis")
    return lower, upper  # type: ignore[return-value]


def search_bounds_report_payload(
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None,
) -> dict[str, Any] | None:
    """JSON fragment for active CMA search bounds; ``None`` when unbounded."""
    if search_bounds_log10 is None:
        return None
    lower, upper = search_bounds_log10
    return {
        "log10_lower": list(lower),
        "log10_upper": list(upper),
        "physical_min_pa": [float(10.0**v) for v in lower],
        "physical_max_pa": [float(10.0**v) for v in upper],
    }


def derive_structure_cma_seed(base_seed: int, structure_idx: int) -> int:
    """Derive a stable positive 32-bit seed for one structure optimizer."""
    sequence = np.random.SeedSequence(
        [int(base_seed) % _UINT32_MOD, int(structure_idx) % _UINT32_MOD]
    )
    seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
    return 1 if seed == 0 else seed


def derive_structure_cma_seeds(
    base_seed: int,
    structure_indices: Sequence[int],
) -> dict[int, int]:
    """Map selected structure indices to distinct effective CMA seeds."""
    seeds: dict[int, int] = {}
    seen: dict[int, int] = {}
    for structure_idx in structure_indices:
        idx = int(structure_idx)
        effective = derive_structure_cma_seed(int(base_seed), idx)
        if effective in seen:
            raise ValueError(
                "effective CMA seed collision between selected structures "
                f"{seen[effective]} and {idx}"
            )
        seen[effective] = idx
        seeds[idx] = effective
    return seeds


def make_pycma_randn(generator: np.random.Generator) -> Callable[..., Any]:
    """Adapt ``Generator.standard_normal`` to pycma's ``randn(*size)`` calling style."""

    def randn(*size: int) -> Any:
        if not size:
            return float(generator.standard_normal())
        if len(size) == 1:
            return generator.standard_normal(size=int(size[0]))
        return generator.standard_normal(size=tuple(int(dim) for dim in size))

    return randn


def build_pycma_options(
    *,
    randn: Callable[..., Any],
    population_size: int | None = None,
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
) -> dict[str, Any]:
    """Build pycma options; omit bounds when ``search_bounds_log10`` is None."""
    options: dict[str, Any] = {
        "randn": randn,
        "seed": np.nan,
        "verbose": -9,
    }
    if search_bounds_log10 is not None:
        lower, upper = search_bounds_log10
        options["bounds"] = [list(lower), list(upper)]
    if population_size is not None:
        options["popsize"] = int(population_size)
    return options


def create_structure_cma_optimizer(
    bounds: YoungsModulusCmaBounds,
    *,
    initial_mean_log10: Sequence[float] | None = None,
    initial_sigma_log10: float = DEFAULT_INITIAL_SIGMA_LOG10,
    base_seed: int,
    structure_idx: int,
    population_size: int | None = None,
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
) -> tuple[cma.CMAEvolutionStrategy, int, np.random.Generator]:
    """Construct one pycma optimizer (bounded only when search bounds are set)."""
    sigma = validate_initial_sigma_log10(initial_sigma_log10)
    mean = resolve_initial_mean_log10(
        "bounds_midpoint" if initial_mean_log10 is None else initial_mean_log10,
        bounds,
    )
    effective_seed = derive_structure_cma_seed(int(base_seed), int(structure_idx))
    rng = np.random.default_rng(effective_seed)
    options = build_pycma_options(
        randn=make_pycma_randn(rng),
        population_size=population_size,
        search_bounds_log10=search_bounds_log10,
    )
    es = cma.CMAEvolutionStrategy(
        list(mean),
        float(sigma),
        options,
    )
    return es, effective_seed, rng


@dataclass(frozen=True)
class YoungsModulusScoringConfig:
    use_median: bool = True
    hold_id_onehot: bool = True
    pool_directions: bool = True
    n_holds: int | None = None
    n_directions: int | None = None
    device: str | None = None


@dataclass(frozen=True)
class YoungsModulusCandidateScore:
    candidate_index: int
    candidate: YoungsModulusCandidate
    aggregate_sinkhorn: float
    per_direction_sinkhorn: dict[int, float]
    instability_fraction: float
    disqualified: bool
    disqualification_reason: str | None
    rank: int | None
    is_gt: bool


@dataclass
class YoungsModulusEvaluation:
    structure_idx: int
    gt_candidate: YoungsModulusCandidate
    fixed_secondary_e_pa: float | None
    direction_indices: tuple[int, ...]
    scores: list[YoungsModulusCandidateScore]
    replay_episodes: list[list[dict[str, Any]]]
    applied_params: list[FruitingSystemParams]


@dataclass(frozen=True)
class PreparedYoungsModulusStructure:
    """Structure-local immutable inputs shared by scalar and fused replay."""

    replay_request: ReplayStructureRequest
    candidates: tuple[YoungsModulusCandidate, ...]
    gt_candidate: YoungsModulusCandidate
    fixed_secondary_e_pa: float | None
    direction_indices: tuple[int, ...]
    recorded_episodes: tuple[dict[str, Any], ...]
    gt_context: WassersteinScoringContext
    scoring_n_directions: int


@dataclass
class YoungsModulusBatchEvaluation:
    evaluations: dict[int, YoungsModulusEvaluation]
    errors: dict[int, str]
    replay_diagnostics: MultiStructureReplayDiagnostics | None
    retried_structures: tuple[int, ...]
    prepared_structures: int = 0
    scoring_seconds: float = 0.0
    total_seconds: float = 0.0
    # Exact candidate×usable-direction slots per prepared structure (includes
    # scoring-failed-but-replayed). Empty when the caller did not prepare.
    physical_slots_by_structure: dict[int, int] = field(default_factory=dict)


def prepare_youngs_modulus_structure(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[YoungsModulusCandidate],
    num_directions: int,
    scoring: YoungsModulusScoringConfig,
    include_excluded: bool = False,
) -> PreparedYoungsModulusStructure:
    """Load and prepare one structure without running physical replay."""
    candidate_list = tuple(candidates)
    if not candidate_list:
        raise ValueError("candidates must be non-empty")
    resolved = resolve_direction_indices(
        dataset,
        structure_idx=int(structure_idx),
        num_directions=int(num_directions),
        include_excluded=bool(include_excluded),
    )
    direction_indices = tuple(int(direction_idx) for direction_idx in resolved)
    recorded = tuple(
        load_recorded_episodes_for_structure(
            dataset,
            structure_idx=int(structure_idx),
            num_directions=len(direction_indices),
            direction_indices=direction_indices,
            include_excluded=bool(include_excluded),
        )
    )
    if len(recorded) != len(direction_indices):
        raise RuntimeError(
            "recorded episode count does not match resolved direction count: "
            f"{len(recorded)} != {len(direction_indices)}"
        )
    scoring_n_directions = (
        int(num_directions)
        if scoring.n_directions is None
        else int(scoring.n_directions)
    )
    gt_context = prepare_gt_wasserstein_scoring_context(
        recorded,
        use_median=bool(scoring.use_median),
        hold_id_onehot=bool(scoring.hold_id_onehot),
        n_holds=scoring.n_holds,
        n_directions=scoring_n_directions,
    )
    base_params = true_params_for_structure(dataset, int(structure_idx))
    if isinstance(candidate_list[0], SupportKpYoungsCandidate):
        gt_candidate = gt_support_kp_youngs_candidate_from_structure(
            dataset, int(structure_idx)
        )
    else:
        gt_candidate = youngs_modulus_candidate_from_params(base_params)
    fixed_secondary_e_pa = (
        None
        if base_params.secondary is None
        else float(base_params.secondary.youngs_modulus_pa)
    )
    first_direction_idx = direction_indices[0]
    gripper = gripper_proxy_from_episode_metadata(
        dataset.load_episode_metadata(int(structure_idx), first_direction_idx)
    )
    recorded_by_direction = dict(zip(direction_indices, recorded, strict=True))
    return PreparedYoungsModulusStructure(
        replay_request=ReplayStructureRequest(
            structure_idx=int(structure_idx),
            candidates=candidate_list,
            direction_indices=direction_indices,
            base_params=base_params,
            recorded_by_direction=recorded_by_direction,
            gripper=gripper,
        ),
        candidates=candidate_list,
        gt_candidate=gt_candidate,
        fixed_secondary_e_pa=fixed_secondary_e_pa,
        direction_indices=direction_indices,
        recorded_episodes=recorded,
        gt_context=gt_context,
        scoring_n_directions=scoring_n_directions,
    )


def score_prepared_youngs_modulus_structure(
    prepared: PreparedYoungsModulusStructure,
    *,
    replay_by_key: dict[ReplaySlotKey, dict[str, Any]],
    scoring: YoungsModulusScoringConfig,
) -> YoungsModulusEvaluation:
    """Score routed replay using original structure/candidate/direction identity."""
    scoring_n_directions = int(prepared.scoring_n_directions)
    provisional: list[YoungsModulusCandidateScore] = []
    replay_episodes: list[list[dict[str, Any]]] = []

    for local_candidate_idx, candidate in enumerate(prepared.candidates):
        keys = tuple(
            ReplaySlotKey(
                structure_idx=prepared.replay_request.structure_idx,
                local_candidate_idx=local_candidate_idx,
                direction_idx=direction_idx,
            )
            for direction_idx in prepared.direction_indices
        )
        for key, direction_idx in zip(keys, prepared.direction_indices, strict=True):
            if (
                key.structure_idx != prepared.replay_request.structure_idx
                or key.local_candidate_idx != local_candidate_idx
                or key.direction_idx != direction_idx
            ):
                raise RuntimeError(f"invalid routed replay key: {key}")
        replay_eps = [replay_by_key[key] for key in keys]
        replay_episodes.append(replay_eps)
        direction_instability = [
            replay_instability_fraction_all_frames(replay=replay, recorded=recorded)
            for replay, recorded in zip(
                replay_eps, prepared.recorded_episodes, strict=True
            )
        ]
        finite_instability = [
            float(fraction)
            for fraction in direction_instability
            if math.isfinite(float(fraction))
        ]
        instability_fraction = (
            max(finite_instability) if finite_instability else float("nan")
        )
        disqualified = any(
            math.isfinite(float(fraction))
            and float(fraction) > float(UNSTABLE_DISQUALIFY_THRESHOLD)
            for fraction in direction_instability
        )
        disqualification_reason = "replay_instability" if disqualified else None

        w_result = score_candidate_wasserstein_complete(
            candidate_index=local_candidate_idx,
            stiffnesses=_candidate_stiffness_diagnostics(candidate),
            gt_context=prepared.gt_context,
            replay_observations=replay_eps,
            device=scoring.device,
            use_median=bool(scoring.use_median),
            hold_id_onehot=bool(scoring.hold_id_onehot),
            n_holds=scoring.n_holds,
            pool_directions=bool(scoring.pool_directions),
            n_directions=scoring_n_directions,
        )
        if int(w_result.candidate_index) != local_candidate_idx:
            raise RuntimeError(
                "Wasserstein scorer candidate index mismatch: "
                f"expected {local_candidate_idx}, got {w_result.candidate_index}"
            )
        if w_result.missing_directions:
            disqualified = True
            if disqualification_reason is None:
                expected = set(prepared.gt_context.expected_directions)
                missing = {int(direction) for direction in w_result.missing_directions}
                disqualification_reason = (
                    "empty_transition_bag"
                    if expected and missing == expected
                    else "missing_directions"
                )
        aggregate_sinkhorn = float(w_result.aggregate_sinkhorn)
        if not math.isfinite(aggregate_sinkhorn):
            disqualified = True
            if disqualification_reason is None:
                disqualification_reason = "non_finite_sinkhorn"
        provisional.append(
            YoungsModulusCandidateScore(
                candidate_index=local_candidate_idx,
                candidate=candidate,
                aggregate_sinkhorn=aggregate_sinkhorn,
                per_direction_sinkhorn=dict(w_result.per_direction_sinkhorn),
                instability_fraction=float(instability_fraction),
                disqualified=bool(disqualified),
                disqualification_reason=disqualification_reason,
                rank=None,
                is_gt=youngs_modulus_values_match(candidate, prepared.gt_candidate),
            )
        )

    eligible = [
        score
        for score in provisional
        if not score.disqualified and math.isfinite(score.aggregate_sinkhorn)
    ]
    ordered = sorted(
        eligible,
        key=lambda score: (score.aggregate_sinkhorn, score.candidate_index),
    )
    rank_by_index = {
        score.candidate_index: rank for rank, score in enumerate(ordered, start=1)
    }
    scores = [
        YoungsModulusCandidateScore(
            candidate_index=score.candidate_index,
            candidate=score.candidate,
            aggregate_sinkhorn=score.aggregate_sinkhorn,
            per_direction_sinkhorn=dict(score.per_direction_sinkhorn),
            instability_fraction=score.instability_fraction,
            disqualified=score.disqualified,
            disqualification_reason=score.disqualification_reason,
            rank=rank_by_index.get(score.candidate_index),
            is_gt=score.is_gt,
        )
        for score in provisional
    ]
    base_params = prepared.replay_request.base_params
    return YoungsModulusEvaluation(
        structure_idx=prepared.replay_request.structure_idx,
        gt_candidate=prepared.gt_candidate,
        fixed_secondary_e_pa=prepared.fixed_secondary_e_pa,
        direction_indices=prepared.direction_indices,
        scores=scores,
        replay_episodes=replay_episodes,
        applied_params=[
            candidate.apply_to(base_params) for candidate in prepared.candidates
        ],
    )


def evaluate_youngs_modulus_candidates(
    *,
    dataset: BatchedSysIdDataset,
    structure_idx: int,
    candidates: Sequence[YoungsModulusCandidate],
    num_directions: int,
    build_env_fn: Callable[..., Any],
    scoring: YoungsModulusScoringConfig,
    max_envs_per_batch: int = 0,
    seed: int | None = None,
    include_excluded: bool = False,
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
) -> YoungsModulusEvaluation:
    """Compatibility wrapper using scalar per-structure physical replay."""
    prepared = prepare_youngs_modulus_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=candidates,
        num_directions=int(num_directions),
        scoring=scoring,
        include_excluded=bool(include_excluded),
    )
    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=prepared.candidates,
        num_directions=len(prepared.direction_indices),
        direction_indices=prepared.direction_indices,
        seed=seed,
        build_env_fn=build_env_fn,
        max_envs_per_batch=int(max_envs_per_batch),
        on_step=on_step,
        replay_sim_config=replay_sim_config,
        use_oracle_params=True,
        include_excluded=bool(include_excluded),
    )
    replay_by_key: dict[ReplaySlotKey, dict[str, Any]] = {}
    for candidate_index in range(len(prepared.candidates)):
        replay_eps = direction_episodes_from_collectors(
            collectors,
            candidate_index=candidate_index,
            num_directions=len(prepared.direction_indices),
        )
        for direction_idx, replay in zip(
            prepared.direction_indices, replay_eps, strict=True
        ):
            replay_by_key[
                ReplaySlotKey(
                    structure_idx=int(structure_idx),
                    local_candidate_idx=candidate_index,
                    direction_idx=direction_idx,
                )
            ] = replay
    return score_prepared_youngs_modulus_structure(
        prepared,
        replay_by_key=replay_by_key,
        scoring=scoring,
    )


def evaluate_youngs_modulus_structures(
    *,
    dataset: BatchedSysIdDataset,
    structures: Sequence[tuple[int, Sequence[YoungsModulusCandidate]]],
    num_directions: int,
    build_env_fn: Callable[..., Any],
    scoring: YoungsModulusScoringConfig,
    max_envs_per_batch: int = 0,
    seed: int | None = None,
    include_excluded: bool = False,
    fail_fast: bool = False,
    on_step: Callable[..., bool] | None = None,
    replay_sim_config: BatchedHeterogeneousCoupledSimConfig | None = None,
) -> YoungsModulusBatchEvaluation:
    """Prepare independently, replay compatibly in fused chunks, and score."""
    total_started = time.perf_counter()
    prepared_by_idx: dict[int, PreparedYoungsModulusStructure] = {}
    errors: dict[int, str] = {}
    structures_by_idx: dict[int, tuple[YoungsModulusCandidate, ...]] = {}
    for structure_idx, candidates in structures:
        idx = int(structure_idx)
        if idx in structures_by_idx:
            raise ValueError(f"duplicate structure_idx request: {idx}")
        structures_by_idx[idx] = tuple(candidates)
        try:
            prepared_by_idx[idx] = prepare_youngs_modulus_structure(
                dataset=dataset,
                structure_idx=idx,
                candidates=structures_by_idx[idx],
                num_directions=int(num_directions),
                scoring=scoring,
                include_excluded=bool(include_excluded),
            )
        except SysIdReplayCancelled:
            raise
        except Exception as exc:
            if fail_fast:
                raise
            errors[idx] = str(exc)

    evaluations: dict[int, YoungsModulusEvaluation] = {}
    retried: list[int] = []
    replay_diagnostics: MultiStructureReplayDiagnostics | None = None
    scoring_seconds = 0.0

    def scalar_retry(structure_idx: int) -> None:
        if structure_idx in retried:
            return
        retried.append(structure_idx)
        try:
            evaluations[structure_idx] = evaluate_youngs_modulus_candidates(
                dataset=dataset,
                structure_idx=structure_idx,
                candidates=structures_by_idx[structure_idx],
                num_directions=int(num_directions),
                build_env_fn=build_env_fn,
                scoring=scoring,
                max_envs_per_batch=int(max_envs_per_batch),
                seed=seed,
                include_excluded=bool(include_excluded),
                on_step=on_step,
                replay_sim_config=replay_sim_config,
            )
        except SysIdReplayCancelled:
            raise
        except Exception as exc:
            if fail_fast:
                raise
            errors[structure_idx] = str(exc)

    physical_slots_by_structure = {
        int(idx): len(prepared.candidates) * len(prepared.direction_indices)
        for idx, prepared in prepared_by_idx.items()
    }

    prepared_items = tuple(prepared_by_idx.values())
    if prepared_items:
        try:
            blocks = build_replay_candidate_blocks(
                tuple(item.replay_request for item in prepared_items)
            )
        except ReplayFusionIncompatible:
            for structure_idx in prepared_by_idx:
                scalar_retry(structure_idx)
        else:
            outcome = replay_multi_structure_candidate_blocks(
                dataset=dataset,
                blocks=blocks,
                build_env_fn=build_env_fn,
                max_envs_per_batch=int(max_envs_per_batch),
                seed=seed,
                fail_fast=bool(fail_fast),
                on_step=on_step,
            )
            replay_diagnostics = outcome.diagnostics
            for structure_idx, prepared in prepared_by_idx.items():
                if structure_idx in outcome.failed_structures:
                    scalar_retry(structure_idx)
                    continue
                scoring_started = time.perf_counter()
                try:
                    evaluations[structure_idx] = (
                        score_prepared_youngs_modulus_structure(
                            prepared,
                            replay_by_key=outcome.replay_by_key,
                            scoring=scoring,
                        )
                    )
                except SysIdReplayCancelled:
                    raise
                except Exception as exc:
                    if fail_fast:
                        raise
                    errors[structure_idx] = str(exc)
                finally:
                    scoring_seconds += time.perf_counter() - scoring_started

    ordered_evaluations = {
        structure_idx: evaluations[structure_idx]
        for structure_idx in structures_by_idx
        if structure_idx in evaluations
    }
    ordered_errors = {
        structure_idx: errors[structure_idx]
        for structure_idx in structures_by_idx
        if structure_idx in errors and structure_idx not in evaluations
    }
    return YoungsModulusBatchEvaluation(
        evaluations=ordered_evaluations,
        errors=ordered_errors,
        replay_diagnostics=replay_diagnostics,
        retried_structures=tuple(retried),
        prepared_structures=len(prepared_by_idx),
        scoring_seconds=float(scoring_seconds),
        total_seconds=float(time.perf_counter() - total_started),
        physical_slots_by_structure=physical_slots_by_structure,
    )


class CmaGenerationFailure(Exception):
    """Structure-local CMA generation failure with a machine-readable stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = str(stage)
        self.message = str(message)


@dataclass(frozen=True)
class CmaDistributionSnapshot:
    mean_log10: tuple[float, float, float]
    sigma: float
    covariance: dict[str, Any] | None = None


@dataclass(frozen=True)
class CmaGenerationRecord:
    generation_index: int
    structure_idx: int
    ask_samples_log10: tuple[tuple[float, float, float], ...]
    candidates: tuple[YoungsModulusCandidate, ...]
    raw_scores: tuple[YoungsModulusCandidateScore, ...]
    penalized_fitness: tuple[float, ...]
    penalty_metadata: tuple[dict[str, Any], ...]
    ask_distribution: CmaDistributionSnapshot
    post_tell_distribution: CmaDistributionSnapshot
    wave_seconds: float | None = None


@dataclass
class StructureCmaState:
    structure_idx: int
    optimizer: Any
    bounds: YoungsModulusCmaBounds
    effective_seed: int
    population_size: int
    status: str = "active"
    completed_generations: int = 0
    optimizer_samples_told: int = 0
    generations: list[CmaGenerationRecord] = field(default_factory=list)
    failure: CmaGenerationFailure | None = None
    stop_kind: str | None = None
    stop_conditions: dict[str, Any] = field(default_factory=dict)
    final_mean_log10: tuple[float, float, float] | None = None
    best_sample_log10: tuple[float, float, float] | None = None
    best_sample_fitness: float | None = None
    final_evaluation: YoungsModulusEvaluation | None = None
    gt_candidate: YoungsModulusCandidate | None = None
    artifact_errors: list[str] = field(default_factory=list)
    # Active CMA box; None means unbounded (report bounds JSON null).
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        None
    )


@dataclass(frozen=True)
class CmaGenerationWaveResult:
    records: dict[int, CmaGenerationRecord]
    failures: dict[int, CmaGenerationFailure]
    batch_evaluation: YoungsModulusBatchEvaluation | None


def _optimizer_mean_log10(optimizer: Any) -> tuple[float, float, float]:
    """Bounded phenotype mean in log10-E (``result.xfavorite``, not genotype ``mean``).

    With BoundTransform, pycma's internal ``es.mean`` can sit slightly outside the
    physical box; ``result.xfavorite`` is the mapped phenotype center that ask()
    samples and the fitted estimate already use.
    """
    return snapshot_xfavorite_log10(optimizer)


def _optimizer_sigma(optimizer: Any) -> float:
    return float(getattr(optimizer, "sigma", float("nan")))


def snapshot_optimizer_distribution(optimizer: Any) -> CmaDistributionSnapshot:
    return CmaDistributionSnapshot(
        mean_log10=_optimizer_mean_log10(optimizer),
        sigma=_optimizer_sigma(optimizer),
        covariance=optimizer_covariance_diagnostics(optimizer),
    )


def _validate_ask_population(
    samples: Sequence[Any],
    *,
    population_size: int,
    bounds: YoungsModulusCmaBounds,
    search_bounds_log10: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
) -> tuple[tuple[float, float, float], ...]:
    del bounds  # fixture ranges are not the search box unless search_bounds_log10 is set
    if len(samples) != int(population_size):
        raise CmaGenerationFailure(
            "generation_evaluation",
            f"ask population size {len(samples)} != {population_size}",
        )
    parsed: list[tuple[float, float, float]] = []
    for sample in samples:
        if len(sample) != 3:
            raise CmaGenerationFailure(
                "generation_evaluation",
                f"ask sample must have length 3, got {len(sample)}",
            )
        values = tuple(float(v) for v in sample)
        if not all(math.isfinite(v) for v in values):
            raise CmaGenerationFailure(
                "generation_evaluation",
                "ask sample contains non-finite values",
            )
        if search_bounds_log10 is not None:
            lower, upper = search_bounds_log10
            for value, lo, hi in zip(values, lower, upper, strict=True):
                if value < lo - 1e-9 or value > hi + 1e-9:
                    raise CmaGenerationFailure(
                        "generation_evaluation",
                        "ask sample outside search bounds",
                    )
        parsed.append(values)  # type: ignore[arg-type]
    return tuple(parsed)


def _require_complete_candidate_indices(
    scores: Sequence[YoungsModulusCandidateScore],
    *,
    population_size: int,
) -> None:
    indices = [int(score.candidate_index) for score in scores]
    expected = list(range(int(population_size)))
    if sorted(indices) != expected or len(indices) != len(expected):
        raise CmaGenerationFailure(
            "generation_evaluation",
            f"candidate indices must be exactly 0..{population_size - 1}, got {indices}",
        )


def generation_score_summary(
    penalty_metadata: Sequence[Mapping[str, Any]],
    *,
    penalized_fitness: Sequence[float],
) -> dict[str, Any]:
    """Summarize eligible and penalized Sinkhorn scores for one generation.

    Sample variance/std use ``ddof=1`` and are JSON ``null`` when fewer than two
    values are available in that category.
    """

    def _stats(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return None, None, None
        mean = float(sum(finite) / len(finite))
        if len(finite) < 2:
            return mean, None, None
        var = float(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))
        return mean, var, float(math.sqrt(var))

    eligible = [
        float(meta["raw_aggregate_sinkhorn"])
        for meta in penalty_metadata
        if (not bool(meta.get("penalized")))
        and meta.get("raw_aggregate_sinkhorn") is not None
        and math.isfinite(float(meta["raw_aggregate_sinkhorn"]))
    ]
    eligible_mean, eligible_var, eligible_std = _stats(eligible)
    penalized_mean, penalized_var, penalized_std = _stats(penalized_fitness)
    n_penalized = sum(1 for meta in penalty_metadata if bool(meta.get("penalized")))
    return {
        "n_eligible": int(len(eligible)),
        "n_penalized": int(n_penalized),
        "eligible_mean": eligible_mean,
        "eligible_variance": eligible_var,
        "eligible_std": eligible_std,
        "best_eligible": float(min(eligible)) if eligible else None,
        "penalized_mean": penalized_mean,
        "penalized_variance": penalized_var,
        "penalized_std": penalized_std,
    }


def penalize_youngs_modulus_scores(
    scores: Sequence[YoungsModulusCandidateScore],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Replace invalid scores with worst_finite + max(1, abs(worst_finite))."""
    ordered = sorted(scores, key=lambda score: int(score.candidate_index))
    _require_complete_candidate_indices(ordered, population_size=len(ordered))
    eligible = [
        float(score.aggregate_sinkhorn)
        for score in ordered
        if (not score.disqualified) and math.isfinite(float(score.aggregate_sinkhorn))
    ]
    if not eligible:
        raise CmaGenerationFailure(
            "all_invalid",
            "no eligible finite scores in generation",
        )
    worst_finite = max(eligible)
    penalty = worst_finite + max(1.0, abs(worst_finite))
    if not math.isfinite(penalty):
        raise CmaGenerationFailure(
            "penalty",
            "penalty overflowed to a non-finite value",
        )
    fitness: list[float] = []
    metadata: list[dict[str, Any]] = []
    for score in ordered:
        raw = float(score.aggregate_sinkhorn)
        invalid = bool(score.disqualified) or (not math.isfinite(raw))
        value = penalty if invalid else raw
        fitness.append(float(value))
        metadata.append(
            {
                "candidate_index": int(score.candidate_index),
                "penalized": bool(invalid),
                "raw_aggregate_sinkhorn": raw,
                "fitness": float(value),
                "disqualification_reason": score.disqualification_reason,
            }
        )
    return fitness, metadata


# After this many additional ask()/evaluate cycles still all-DQ, fall back to a
# flat-penalty tell so CMA can move/shrink instead of killing the structure.
DEFAULT_ALL_INVALID_REASKS = 3
ALL_INVALID_FLAT_PENALTY = 1.0e12


def flat_penalize_all_invalid_scores(
    scores: Sequence[YoungsModulusCandidateScore],
    *,
    flat_penalty: float = ALL_INVALID_FLAT_PENALTY,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Assign a uniform huge fitness when no eligible scores exist.

    Used only after re-asks are exhausted so ``tell()`` still updates the
    distribution away from an unstable region.
    """
    ordered = sorted(scores, key=lambda score: int(score.candidate_index))
    _require_complete_candidate_indices(ordered, population_size=len(ordered))
    penalty = float(flat_penalty)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise CmaGenerationFailure(
            "penalty",
            f"flat penalty must be a positive finite value, got {penalty}",
        )
    fitness: list[float] = []
    metadata: list[dict[str, Any]] = []
    for score in ordered:
        raw = float(score.aggregate_sinkhorn)
        fitness.append(penalty)
        metadata.append(
            {
                "candidate_index": int(score.candidate_index),
                "penalized": True,
                "raw_aggregate_sinkhorn": raw,
                "fitness": penalty,
                "disqualification_reason": score.disqualification_reason,
                "flat_penalty_tell": True,
            }
        )
    return fitness, metadata


def _annotate_penalty_metadata(
    metadata: list[dict[str, Any]],
    *,
    all_invalid_reasks: int,
) -> list[dict[str, Any]]:
    if int(all_invalid_reasks) <= 0:
        return metadata
    out: list[dict[str, Any]] = []
    for meta in metadata:
        annotated = dict(meta)
        annotated["all_invalid_reasks"] = int(all_invalid_reasks)
        out.append(annotated)
    return out


def _update_best_sample(
    state: StructureCmaState,
    *,
    samples_log10: Sequence[tuple[float, float, float]],
    fitness: Sequence[float],
) -> None:
    for sample, value in zip(samples_log10, fitness, strict=True):
        if not math.isfinite(float(value)):
            continue
        if state.best_sample_fitness is None or float(value) < float(
            state.best_sample_fitness
        ):
            state.best_sample_fitness = float(value)
            state.best_sample_log10 = tuple(float(v) for v in sample)  # type: ignore[assignment]


def run_cma_generation_wave(
    active_states: Mapping[int, StructureCmaState],
    *,
    evaluate_fn: Callable[..., YoungsModulusBatchEvaluation],
    generation_index: int,
    all_invalid_reasks: int = DEFAULT_ALL_INVALID_REASKS,
) -> CmaGenerationWaveResult:
    """Ask active optimizers, evaluate fused, then tell independently.

    If a structure's population is entirely disqualified (`all_invalid`):
    1. re-``ask()`` / re-evaluate up to ``all_invalid_reasks`` more times, then
    2. if still all-DQ, ``tell()`` with :data:`ALL_INVALID_FLAT_PENALTY` so CMA
       can move/shrink instead of failing the structure.

    Pass ``all_invalid_reasks=0`` to keep the legacy fail-without-tell behavior.
    """
    if not active_states:
        return CmaGenerationWaveResult(records={}, failures={}, batch_evaluation=None)

    reask_budget = max(0, int(all_invalid_reasks))
    ordered_indices = list(active_states.keys())
    ask_samples: dict[int, Any] = {}
    ask_log10: dict[int, tuple[tuple[float, float, float], ...]] = {}
    ask_distributions: dict[int, CmaDistributionSnapshot] = {}
    structure_candidates: list[tuple[int, tuple[YoungsModulusCandidate, ...]]] = []
    failures: dict[int, CmaGenerationFailure] = {}
    records: dict[int, CmaGenerationRecord] = {}
    reask_counts: dict[int, int] = {idx: 0 for idx in ordered_indices}
    wave_seconds = 0.0
    last_batch: YoungsModulusBatchEvaluation | None = None

    def _ask_structure(structure_idx: int) -> tuple[int, tuple[YoungsModulusCandidate, ...]]:
        state = active_states[structure_idx]
        ask_distributions[structure_idx] = snapshot_optimizer_distribution(
            state.optimizer
        )
        samples = state.optimizer.ask()
        ask_samples[structure_idx] = samples
        parsed = _validate_ask_population(
            samples,
            population_size=state.population_size,
            bounds=state.bounds,
            search_bounds_log10=state.search_bounds_log10,
        )
        ask_log10[structure_idx] = parsed
        candidates = tuple(candidates_from_log10_e(row) for row in parsed)
        return (int(structure_idx), candidates)

    def _commit_tell(
        structure_idx: int,
        *,
        ordered_scores: tuple[YoungsModulusCandidateScore, ...],
        fitness: list[float],
        metadata: list[dict[str, Any]],
        update_best: bool,
    ) -> None:
        state = active_states[structure_idx]
        samples = ask_samples[structure_idx]
        state.optimizer.tell(samples, fitness)
        state.completed_generations += 1
        state.optimizer_samples_told += len(fitness)
        if update_best:
            _update_best_sample(
                state,
                samples_log10=ask_log10[structure_idx],
                fitness=fitness,
            )
        annotated = _annotate_penalty_metadata(
            metadata,
            all_invalid_reasks=reask_counts.get(structure_idx, 0),
        )
        record = CmaGenerationRecord(
            generation_index=int(generation_index),
            structure_idx=int(structure_idx),
            ask_samples_log10=ask_log10[structure_idx],
            candidates=tuple(score.candidate for score in ordered_scores),
            raw_scores=ordered_scores,
            penalized_fitness=tuple(fitness),
            penalty_metadata=tuple(annotated),
            ask_distribution=ask_distributions[structure_idx],
            post_tell_distribution=snapshot_optimizer_distribution(state.optimizer),
            wave_seconds=wave_seconds,
        )
        state.generations.append(record)
        records[structure_idx] = record

    def _ordered_scores_from_evaluation(
        structure_idx: int,
        batch: YoungsModulusBatchEvaluation,
    ) -> tuple[YoungsModulusCandidateScore, ...]:
        state = active_states[structure_idx]
        if structure_idx in batch.errors:
            raise CmaGenerationFailure(
                "generation_evaluation",
                str(batch.errors[structure_idx]),
            )
        if structure_idx not in batch.evaluations:
            raise CmaGenerationFailure(
                "generation_evaluation",
                "missing evaluation for structure",
            )
        evaluation = batch.evaluations[structure_idx]
        _require_complete_candidate_indices(
            evaluation.scores,
            population_size=state.population_size,
        )
        score_by_index = {
            int(score.candidate_index): score for score in evaluation.scores
        }
        return tuple(score_by_index[i] for i in range(state.population_size))

    for structure_idx in ordered_indices:
        state = active_states[structure_idx]
        try:
            structure_candidates.append(_ask_structure(structure_idx))
        except CmaGenerationFailure as exc:
            state.failure = exc
            state.status = "failed"
            failures[structure_idx] = exc

    if not structure_candidates:
        return CmaGenerationWaveResult(
            records={},
            failures=failures,
            batch_evaluation=None,
        )

    pending = {int(idx) for idx, _ in structure_candidates}
    batch_started = time.perf_counter()
    last_batch = evaluate_fn(structures=structure_candidates, wave_kind="generation")
    wave_seconds = float(time.perf_counter() - batch_started)

    while pending:
        still_pending: set[int] = set()
        for structure_idx in sorted(pending):
            state = active_states[structure_idx]
            try:
                ordered_scores = _ordered_scores_from_evaluation(
                    structure_idx, last_batch
                )
                try:
                    fitness, metadata = penalize_youngs_modulus_scores(ordered_scores)
                except CmaGenerationFailure as exc:
                    if exc.stage != "all_invalid":
                        raise
                    if reask_counts[structure_idx] < reask_budget:
                        reask_counts[structure_idx] += 1
                        still_pending.add(structure_idx)
                        continue
                    if reask_budget <= 0 and reask_counts[structure_idx] == 0:
                        # Legacy: no re-asks configured → fail without tell.
                        raise
                    fitness, metadata = flat_penalize_all_invalid_scores(ordered_scores)
                    _commit_tell(
                        structure_idx,
                        ordered_scores=ordered_scores,
                        fitness=fitness,
                        metadata=metadata,
                        update_best=False,
                    )
                    continue
                _commit_tell(
                    structure_idx,
                    ordered_scores=ordered_scores,
                    fitness=fitness,
                    metadata=metadata,
                    update_best=True,
                )
            except CmaGenerationFailure as exc:
                state.failure = exc
                state.status = "failed"
                failures[structure_idx] = exc

        if not still_pending:
            break

        retry_candidates: list[tuple[int, tuple[YoungsModulusCandidate, ...]]] = []
        for structure_idx in sorted(still_pending):
            state = active_states[structure_idx]
            try:
                retry_candidates.append(_ask_structure(structure_idx))
            except CmaGenerationFailure as exc:
                state.failure = exc
                state.status = "failed"
                failures[structure_idx] = exc
                still_pending.discard(structure_idx)

        if not retry_candidates:
            break

        batch_started = time.perf_counter()
        last_batch = evaluate_fn(structures=retry_candidates, wave_kind="generation")
        wave_seconds += float(time.perf_counter() - batch_started)
        pending = {int(idx) for idx, _ in retry_candidates}

    return CmaGenerationWaveResult(
        records=records,
        failures=failures,
        batch_evaluation=last_batch,
    )


@dataclass(frozen=True)
class YoungsModulusCmaFitResult:
    states: dict[int, StructureCmaState]
    fitted_structure_indices: tuple[int, ...]
    failed_structure_indices: tuple[int, ...]
    generation_waves: int
    final_mean_batch: YoungsModulusBatchEvaluation | None
    timing: dict[str, Any] = field(default_factory=dict)


def snapshot_xfavorite_log10(optimizer: Any) -> tuple[float, float, float]:
    """Snapshot pycma's bounded phenotype mean (``result.xfavorite``)."""
    result = getattr(optimizer, "result", None)
    favorite = getattr(result, "xfavorite", None)
    if favorite is None:
        raise ValueError("optimizer.result.xfavorite is unavailable")
    values = tuple(float(v) for v in favorite)
    if len(values) != 3:
        raise ValueError(f"xfavorite must have length 3, got {len(values)}")
    return values  # type: ignore[return-value]


def optimizer_covariance_diagnostics(optimizer: Any) -> dict[str, Any]:
    """Report C, sigma, scaling, phenotype std, and effective unbounded covariance."""
    C = np.asarray(optimizer.C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"optimizer.C must be square 2-d, got shape {C.shape}")
    n = int(C.shape[0])
    sigma = float(optimizer.sigma)
    scaling = np.asarray(optimizer.sigma_vec.scaling, dtype=float).reshape(-1)
    if scaling.size == 1:
        scaling = np.full(n, float(scaling.item()), dtype=float)
    elif scaling.size != n:
        raise ValueError(
            f"sigma_vec.scaling length {scaling.size} does not match C dim {n}"
        )
    scale_diag = np.diag(scaling)
    effective = (sigma**2) * (scale_diag @ C @ scale_diag)
    phenotype_std = tuple(
        float(v) for v in (sigma * scaling * np.sqrt(np.diag(C)))
    )
    return {
        "C": C.tolist(),
        "sigma": sigma,
        "sigma_vec_scaling": scaling.tolist(),
        "phenotype_std": phenotype_std,
        "effective_unbounded_covariance": effective.tolist(),
    }


def _stop_mapping(optimizer: Any) -> dict[str, Any]:
    stop = optimizer.stop()
    if stop is None:
        return {}
    if isinstance(stop, dict):
        return dict(stop)
    return {"stop": stop}


def _apply_stop_transition(
    state: StructureCmaState,
    *,
    max_generations: int,
) -> bool:
    """Return True when the structure newly transitions to stopped_pending_final."""
    if state.status != "active":
        return False
    stop_map = _stop_mapping(state.optimizer)
    hit_cap = int(state.completed_generations) >= int(max_generations)
    hit_pycma = bool(stop_map)
    if not hit_cap and not hit_pycma:
        return False
    if hit_cap and hit_pycma:
        state.stop_kind = "both"
    elif hit_cap:
        state.stop_kind = "generation_cap"
    else:
        state.stop_kind = "pycma"
    state.stop_conditions = stop_map
    state.final_mean_log10 = snapshot_xfavorite_log10(state.optimizer)
    state.status = "stopped_pending_final_evaluation"
    return True


def _evaluate_final_means(
    pending_states: Mapping[int, StructureCmaState],
    *,
    evaluate_fn: Callable[..., YoungsModulusBatchEvaluation],
) -> YoungsModulusBatchEvaluation | None:
    if not pending_states:
        return None
    structures: list[tuple[int, tuple[YoungsModulusCandidate, ...]]] = []
    for structure_idx, state in pending_states.items():
        if state.final_mean_log10 is None:
            state.status = "failed"
            state.failure = CmaGenerationFailure(
                "final_mean",
                "missing final mean snapshot",
            )
            continue
        candidate = candidates_from_log10_e(state.final_mean_log10)
        structures.append((int(structure_idx), (candidate,)))
    if not structures:
        return None
    batch = evaluate_fn(structures=structures, wave_kind="final_mean")
    for structure_idx, _cands in structures:
        state = pending_states[structure_idx]
        try:
            if structure_idx in batch.errors:
                raise CmaGenerationFailure(
                    "final_mean",
                    str(batch.errors[structure_idx]),
                )
            if structure_idx not in batch.evaluations:
                raise CmaGenerationFailure(
                    "final_mean",
                    "missing final-mean evaluation",
                )
            evaluation = batch.evaluations[structure_idx]
            if len(evaluation.scores) != 1 or int(evaluation.scores[0].candidate_index) != 0:
                raise CmaGenerationFailure(
                    "final_mean",
                    "final-mean evaluation must use candidate index 0",
                )
            score = evaluation.scores[0]
            if score.disqualified or not math.isfinite(float(score.aggregate_sinkhorn)):
                raise CmaGenerationFailure(
                    "final_mean",
                    score.disqualification_reason or "final mean invalid",
                )
            state.final_evaluation = evaluation
            state.status = "fitted"
        except CmaGenerationFailure as exc:
            state.failure = exc
            state.status = "failed"
    return batch


def fit_youngs_modulus_structures(
    states: Mapping[int, StructureCmaState],
    *,
    max_generations: int,
    evaluate_fn: Callable[..., YoungsModulusBatchEvaluation],
    on_progress: Callable[[Mapping[int, StructureCmaState]], None] | None = None,
) -> YoungsModulusCmaFitResult:
    """Coordinate synchronized waves until active optimizers stop, then score means."""
    if int(max_generations) <= 0:
        raise ValueError("max_generations must be positive")
    ordered = {int(idx): states[idx] for idx in states}
    for state in ordered.values():
        _apply_stop_transition(state, max_generations=max_generations)
    if on_progress is not None:
        on_progress(ordered)

    fit_started = time.perf_counter()
    wave_timings: list[dict[str, Any]] = []
    generation_waves = 0
    while True:
        active = {
            idx: state
            for idx, state in ordered.items()
            if state.status == "active"
        }
        if not active:
            break
        wave = run_cma_generation_wave(
            active,
            evaluate_fn=evaluate_fn,
            generation_index=generation_waves,
        )
        if wave.records:
            seconds = float(next(iter(wave.records.values())).wave_seconds or 0.0)
        else:
            seconds = 0.0
        wave_timings.append(
            {
                "wave_index": int(generation_waves),
                "wave_kind": "generation",
                "seconds": seconds,
                "active_structure_indices": [int(idx) for idx in active],
            }
        )
        generation_waves += 1
        for idx, state in active.items():
            if idx in wave.failures:
                continue
            _apply_stop_transition(state, max_generations=max_generations)
        if on_progress is not None:
            on_progress(ordered)

    pending = {
        idx: state
        for idx, state in ordered.items()
        if state.status == "stopped_pending_final_evaluation"
    }
    final_started = time.perf_counter()
    final_batch = _evaluate_final_means(pending, evaluate_fn=evaluate_fn)
    if pending:
        wave_timings.append(
            {
                "wave_index": int(generation_waves),
                "wave_kind": "final_mean",
                "seconds": float(time.perf_counter() - final_started),
                "active_structure_indices": [int(idx) for idx in pending],
            }
        )
    if on_progress is not None:
        on_progress(ordered)

    fitted = tuple(
        idx for idx, state in ordered.items() if state.status == "fitted"
    )
    failed = tuple(
        idx for idx, state in ordered.items() if state.status == "failed"
    )
    return YoungsModulusCmaFitResult(
        states=dict(ordered),
        fitted_structure_indices=fitted,
        failed_structure_indices=failed,
        generation_waves=generation_waves,
        final_mean_batch=final_batch,
        timing={
            "fit_seconds": float(time.perf_counter() - fit_started),
            "waves": wave_timings,
        },
    )


def to_strict_jsonable(value: Any) -> Any:
    """Recursively convert values to strict-JSON-safe Python builtins."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON value: {number}")
        return number
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return to_strict_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): to_strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_strict_jsonable(item) for item in value]
    if isinstance(value, YoungsModulusCandidate):
        return {
            "primary": to_strict_jsonable(value.primary),
            "spur": to_strict_jsonable(value.spur),
            "stem": to_strict_jsonable(value.stem),
        }
    raise TypeError(f"value is not strict-JSON serializable: {type(value)!r}")


def _candidate_to_e_list(candidate: YoungsModulusCandidate) -> list[float]:
    return [float(candidate.primary), float(candidate.spur), float(candidate.stem)]


def _candidate_to_log10_list(candidate: YoungsModulusCandidate) -> list[float]:
    return [
        math.log10(float(candidate.primary)),
        math.log10(float(candidate.spur)),
        math.log10(float(candidate.stem)),
    ]


def evaluated_history_extrema(
    generations: Sequence[CmaGenerationRecord],
) -> dict[str, list[float] | None]:
    """Component-wise extrema of samples actually submitted in CMA populations."""
    samples = [
        sample
        for generation in generations
        for sample in generation.ask_samples_log10
    ]
    if not samples:
        return {
            "min_log10_e": None,
            "max_log10_e": None,
            "min_e_pa": None,
            "max_e_pa": None,
        }

    log10_samples = np.asarray(samples, dtype=float)
    min_log10 = log10_samples.min(axis=0)
    max_log10 = log10_samples.max(axis=0)
    return {
        "min_log10_e": min_log10.tolist(),
        "max_log10_e": max_log10.tolist(),
        "min_e_pa": np.power(10.0, min_log10).tolist(),
        "max_e_pa": np.power(10.0, max_log10).tolist(),
    }


def structure_cma_report_snapshot(
    state: StructureCmaState,
    *,
    base_seed: int,
    initial_sigma_log10: float,
    replay_candidate_evaluations: int = 0,
    final_mean_evaluations: int = 0,
    physical_env_slots: int = 0,
    scalar_retries: int = 0,
) -> dict[str, Any]:
    """Build a per-structure CMA report fragment."""
    try:
        covariance = optimizer_covariance_diagnostics(state.optimizer)
    except (AttributeError, TypeError, ValueError):
        covariance = None

    generations = []
    for record in state.generations:
        generations.append(
            {
                "generation_index": int(record.generation_index),
                "ask_samples_log10": [list(row) for row in record.ask_samples_log10],
                "penalized_fitness": list(record.penalized_fitness),
                "penalty_metadata": list(record.penalty_metadata),
                "score_summary": generation_score_summary(
                    record.penalty_metadata,
                    penalized_fitness=record.penalized_fitness,
                ),
                "wave_seconds": (
                    None
                    if record.wave_seconds is None
                    else float(record.wave_seconds)
                ),
                "raw_scores": [
                    {
                        "candidate_index": int(score.candidate_index),
                        "aggregate_sinkhorn": float(score.aggregate_sinkhorn),
                        "disqualified": bool(score.disqualified),
                        "disqualification_reason": score.disqualification_reason,
                        "per_direction_sinkhorn": {
                            str(k): float(v)
                            for k, v in score.per_direction_sinkhorn.items()
                        },
                    }
                    for score in record.raw_scores
                ],
                "ask_distribution": {
                    "mean_log10": list(record.ask_distribution.mean_log10),
                    "sigma": float(record.ask_distribution.sigma),
                    "covariance": record.ask_distribution.covariance,
                },
                "post_tell_distribution": {
                    "mean_log10": list(record.post_tell_distribution.mean_log10),
                    "sigma": float(record.post_tell_distribution.sigma),
                    "covariance": record.post_tell_distribution.covariance,
                },
            }
        )

    final_mean = None
    if state.final_mean_log10 is not None:
        mean_candidate = candidates_from_log10_e(state.final_mean_log10)
        final_mean = {
            "log10_e": list(state.final_mean_log10),
            "e_pa": _candidate_to_e_list(mean_candidate),
        }
        if state.final_evaluation is not None and state.final_evaluation.scores:
            score = state.final_evaluation.scores[0]
            final_mean["aggregate_sinkhorn"] = float(score.aggregate_sinkhorn)
            final_mean["per_direction_sinkhorn"] = {
                str(k): float(v) for k, v in score.per_direction_sinkhorn.items()
            }

    best_sample = None
    if state.best_sample_log10 is not None:
        best_candidate = candidates_from_log10_e(state.best_sample_log10)
        best_sample = {
            "log10_e": list(state.best_sample_log10),
            "e_pa": _candidate_to_e_list(best_candidate),
            "fitness": state.best_sample_fitness,
        }

    gt = None
    if state.gt_candidate is not None:
        gt = {
            "e_pa": _candidate_to_e_list(state.gt_candidate),
            "log10_e": _candidate_to_log10_list(state.gt_candidate),
        }

    failure = None
    if state.failure is not None:
        failure = {
            "stage": state.failure.stage,
            "message": state.failure.message,
        }

    return {
        "structure_idx": int(state.structure_idx),
        "status": str(state.status),
        "base_seed": int(base_seed),
        "effective_seed": int(state.effective_seed),
        "initial_sigma_log10": float(initial_sigma_log10),
        "population_size": int(state.population_size),
        "bounds": search_bounds_report_payload(state.search_bounds_log10),
        "completed_generations": int(state.completed_generations),
        "optimizer_samples_told": int(state.optimizer_samples_told),
        "replay_candidate_evaluations": int(replay_candidate_evaluations),
        "final_mean_evaluations": int(final_mean_evaluations),
        "physical_env_slots": int(physical_env_slots),
        "scalar_retries": int(scalar_retries),
        "stop_kind": state.stop_kind,
        "stop_conditions": dict(state.stop_conditions),
        "generations": generations,
        "final_mean": final_mean,
        "best_sample": best_sample,
        "gt": gt,
        "evaluated_history_extrema": evaluated_history_extrema(state.generations),
        "covariance": covariance,
        "failure": failure,
        "artifact_errors": list(state.artifact_errors),
    }


def aggregate_fitted_youngs_modulus_stats(
    *,
    fitted_candidates: Sequence[YoungsModulusCandidate],
    gt_candidates: Sequence[YoungsModulusCandidate],
    requested_count: int,
    failed_count: int,
) -> dict[str, Any]:
    """Component-wise cross-structure aggregate statistics for fitted means."""
    fitted_count = len(fitted_candidates)
    if fitted_count == 0:
        return {
            "requested_structures": int(requested_count),
            "fitted_structures": 0,
            "failed_structures": int(failed_count),
            "mean_log10_e": None,
            "geometric_mean_e_pa": None,
            "mean_e_pa": None,
            "min_e_pa": None,
            "max_e_pa": None,
            "mean_gt_e_pa": None,
            "sample_cov_log10_e": None,
            "sample_std_log10_e": None,
        }

    e_matrix = np.asarray(
        [_candidate_to_e_list(candidate) for candidate in fitted_candidates],
        dtype=float,
    )
    log_matrix = np.log10(e_matrix)
    mean_log10 = log_matrix.mean(axis=0)
    geometric_mean = 10.0 ** mean_log10
    mean_e = e_matrix.mean(axis=0)
    min_e = e_matrix.min(axis=0)
    max_e = e_matrix.max(axis=0)

    mean_gt = None
    if gt_candidates:
        gt_matrix = np.asarray(
            [_candidate_to_e_list(candidate) for candidate in gt_candidates],
            dtype=float,
        )
        mean_gt = gt_matrix.mean(axis=0).tolist()

    if fitted_count < 2:
        cov = None
        std = None
    else:
        cov = np.cov(log_matrix, rowvar=False, ddof=1).tolist()
        std = np.std(log_matrix, axis=0, ddof=1).tolist()

    return {
        "requested_structures": int(requested_count),
        "fitted_structures": int(fitted_count),
        "failed_structures": int(failed_count),
        "mean_log10_e": mean_log10.tolist(),
        "geometric_mean_e_pa": geometric_mean.tolist(),
        "mean_e_pa": mean_e.tolist(),
        "min_e_pa": min_e.tolist(),
        "max_e_pa": max_e.tolist(),
        "mean_gt_e_pa": mean_gt,
        "sample_cov_log10_e": cov,
        "sample_std_log10_e": std,
    }
