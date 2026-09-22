"""Screened harvest worlds: one ``WorldSpec`` per world, persisted as JSONL.

A world is everything baked into a harvest env at build time -- the exact
fruiting-system params, the grasp (weld direction), the support-joint DR sample
and the build-time arm DR scales (link mass/inertia, EE payload). Arm joint
dynamics are deliberately *not* part of a world: they resample on every
``reset()``.

The RL pipeline screens many candidate worlds for build/settle/interaction
stability (``example_screen_harvest_worlds.py``), keeps a diverse stable set
here, and trains on that set: ``world_specs_to_env_kwargs`` rebuilds exactly
those worlds in ``ApplePickVicHarvestEnv``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

WORLD_SPEC_SCHEMA = "harvest_world_spec_v1"

_ARM_BUILD_SCALE_FIELDS = (
    ("link_mass_scale", "arm_link_mass_scale"),
    ("link_inertia_scale", "arm_link_inertia_scale"),
    ("ee_mass_scale", "arm_ee_mass_scale"),
    ("ee_inertia_scale", "arm_ee_inertia_scale"),
)


@dataclasses.dataclass(frozen=True)
class WorldSpec:
    """One harvest world: plant + grasp + build-time DR, plus its screening record."""

    world_id: str
    params_json: str  # fruiting_params_to_json (fruiting_system_params_v3)
    weld_direction: tuple[float, float, float]
    support_kp: float
    support_roll_kp: float
    support_zeta: float
    arm_link_mass_scale: float
    arm_link_inertia_scale: float
    arm_ee_mass_scale: float
    arm_ee_inertia_scale: float
    screening: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        d = tuple(float(x) for x in self.weld_direction)
        if len(d) != 3 or not math.isclose(math.sqrt(sum(x * x for x in d)), 1.0, abs_tol=1e-4):
            raise ValueError(f"weld_direction must be a unit 3-vector, got {self.weld_direction!r}")
        object.__setattr__(self, "weld_direction", d)

    @property
    def passed(self) -> bool:
        return bool(self.screening.get("passed", False))

    def to_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["weld_direction"] = list(self.weld_direction)
        row["schema"] = WORLD_SPEC_SCHEMA
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> WorldSpec:
        row = dict(row)
        schema = row.pop("schema", WORLD_SPEC_SCHEMA)
        if schema != WORLD_SPEC_SCHEMA:
            raise ValueError(f"unsupported world spec schema {schema!r}")
        row["weld_direction"] = tuple(row["weld_direction"])
        return cls(**row)


def save_world_set(path: Path | str, specs: Sequence[WorldSpec], *, append: bool = False) -> None:
    """Write ``specs`` as JSONL (one world per line); ``append`` accumulates batches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for spec in specs:
            fh.write(json.dumps(spec.to_row(), sort_keys=True) + "\n")


def load_world_set(path: Path | str, *, passed_only: bool = False) -> list[WorldSpec]:
    specs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            specs.append(WorldSpec.from_row(json.loads(line)))
    return [s for s in specs if s.passed] if passed_only else specs


def world_specs_to_env_kwargs(specs: Sequence[WorldSpec]) -> dict[str, Any]:
    """``ApplePickVicHarvestEnv`` kwargs that rebuild exactly these worlds, in order."""
    from apple_pick_gym.batched_envs.support_joint_dr import SupportJointDRSample
    from apple_pick_sim.fruiting_system import GripperProxyConfig, PLACEHOLDER_EE_MASS_KG
    from apple_pick_sim.fruiting_system.params import fruiting_params_from_json

    if not specs:
        raise ValueError("world set is empty")
    return {
        "num_envs": len(specs),
        "per_env_params": [fruiting_params_from_json(s.params_json) for s in specs],
        "per_env_grippers": [
            GripperProxyConfig(
                mass=PLACEHOLDER_EE_MASS_KG,
                fix_to_apple=True,
                dynamic_apple=True,
                robot_facing_weld=False,
                weld_direction=s.weld_direction,
            )
            for s in specs
        ],
        "support_dr_sample": SupportJointDRSample(
            kp=np.array([s.support_kp for s in specs], dtype=np.float32),
            roll_kp=np.array([s.support_roll_kp for s in specs], dtype=np.float32),
            zeta=np.array([s.support_zeta for s in specs], dtype=np.float32),
        ),
        "arm_build_dr_scales": {
            sample_key: np.array([getattr(s, spec_key) for s in specs], dtype=np.float32)
            for sample_key, spec_key in _ARM_BUILD_SCALE_FIELDS
        },
    }


_ROD_KNOBS = ("length", "radius", "density", "youngs_modulus_pa", "flexural_modulus_pa", "damping_ratio")


def world_spec_knobs(spec: WorldSpec) -> dict[str, float]:
    """Flat named randomized quantities of one world (for coverage / bias reports)."""
    from apple_pick_sim.fruiting_system.params import fruiting_params_from_json

    p = fruiting_params_from_json(spec.params_json)
    knobs: dict[str, float] = {}
    for rod_name in ("primary", "spur", "stem"):
        rod = getattr(p, rod_name, None)
        if rod is None:
            continue
        for k in _ROD_KNOBS:
            knobs[f"{rod_name}.{k}"] = float(getattr(rod, k))
        knobs[f"{rod_name}.elevation_deg"] = math.degrees(math.asin(max(-1.0, min(1.0, rod.direction[2]))))
    if p.apple_radius is not None:
        knobs["apple_radius"] = float(p.apple_radius)
    if p.apple_density is not None:
        knobs["apple_density"] = float(p.apple_density)
    for f in ("support_kp", "support_roll_kp", "support_zeta") + tuple(s for _, s in _ARM_BUILD_SCALE_FIELDS):
        knobs[f] = float(getattr(spec, f))
    knobs["weld_polar_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, -spec.weld_direction[2]))))
    return knobs


def world_set_coverage(specs: Sequence[WorldSpec]) -> list[dict[str, Any]]:
    """Per knob: candidate vs accepted range, and rejection rate per candidate tercile.

    A rejection rate concentrated in one tercile means screening biases the
    accepted set away from that end of the knob's range.
    """
    if not specs:
        return []
    table = [world_spec_knobs(s) for s in specs]
    accepted = np.array([s.passed for s in specs], dtype=bool)
    rows = []
    for knob in table[0]:
        v = np.array([t[knob] for t in table], dtype=np.float64)
        if np.ptp(v) <= 1e-9 * max(1.0, float(np.abs(v).max())):
            continue  # pinned knob: nothing to cover
        edges = np.quantile(v, [1 / 3, 2 / 3])
        tercile = np.searchsorted(edges, v, side="left")
        rates = []
        for b in range(3):
            m = tercile == b
            rates.append(float((~accepted[m]).mean()) if m.any() else float("nan"))
        acc = v[accepted]
        rows.append(
            {
                "knob": knob,
                "n_candidates": int(v.size),
                "n_accepted": int(accepted.sum()),
                "candidate_min": float(v.min()),
                "candidate_max": float(v.max()),
                "accepted_min": float(acc.min()) if acc.size else float("nan"),
                "accepted_max": float(acc.max()) if acc.size else float("nan"),
                "reject_rate_by_tercile": rates,
            }
        )
    return rows
