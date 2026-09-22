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
