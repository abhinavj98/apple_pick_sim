"""Load and save digital-twin observation files (``digital_twin_v2`` schema)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "digital_twin_v2"
SUPPORTED_SCHEMA_VERSIONS = {"digital_twin_v1", SCHEMA_VERSION}


@dataclasses.dataclass(frozen=True)
class DigitalTwinObs:
    """Partial real-world observations used to seed a quasi-static digital twin.

    Junction labels match gym :attr:`~apple_pick_gym.envs.apple_pick_base_env.ApplePickBaseEnv.junction_names`
    (``joint_`` prefix removed). Flat position arrays follow the gym observation contract.
    """

    fruiting_base_pos: tuple[float, float, float]
    weld_direction: tuple[float, float, float]
    junction_names: list[str]
    woody_part_start_pos: np.ndarray
    woody_part_end_pos: np.ndarray
    apple_radius: float | None = None
    rod_radii: dict[str, float] | None = None

    def __post_init__(self) -> None:
        start = np.asarray(self.woody_part_start_pos, dtype=np.float32).reshape(-1)
        end = np.asarray(self.woody_part_end_pos, dtype=np.float32).reshape(-1)
        if start.size % 3 != 0 or end.size % 3 != 0:
            raise ValueError("woody_part_*_pos lengths must be multiples of 3")
        n = len(self.junction_names)
        if start.size != n * 3 or end.size != n * 3:
            raise ValueError(
                f"junction_names has {n} entries but woody_part arrays have "
                f"{start.size // 3} / {end.size // 3} positions"
            )
        object.__setattr__(self, "woody_part_start_pos", start)
        object.__setattr__(self, "woody_part_end_pos", end)
        if self.rod_radii is not None:
            object.__setattr__(
                self,
                "rod_radii",
                {str(name): float(radius) for name, radius in self.rod_radii.items()},
            )
        _validate_unit_vector(self.weld_direction, field="weld_direction")


def _validate_unit_vector(vec: tuple[float, float, float], *, field: str) -> None:
    n = math.sqrt(sum(float(v) ** 2 for v in vec))
    if n < 1e-9:
        raise ValueError(f"{field} must be a non-zero vector")
    if abs(n - 1.0) > 1e-3:
        raise ValueError(f"{field} must be unit length (got norm {n:.6f})")


def _coerce_xyz(raw: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{field} must be [x, y, z]")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def load_digital_twin_obs(path: str | Path) -> DigitalTwinObs:
    """Load a supported digital-twin observation JSON file."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("observation file must be a JSON object")
    schema = data.get("schema", SCHEMA_VERSION)
    if schema not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported schema {schema!r}; expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}"
        )

    junction_names = data.get("junction_names")
    if not isinstance(junction_names, list) or not junction_names:
        raise ValueError("junction_names must be a non-empty list of strings")
    if not all(isinstance(n, str) for n in junction_names):
        raise ValueError("junction_names entries must be strings")

    apple_radius = data.get("apple_radius")
    if apple_radius is not None:
        apple_radius = float(apple_radius)
    rod_radii = data.get("rod_radii")
    if rod_radii is not None:
        if not isinstance(rod_radii, dict):
            raise ValueError("rod_radii must be an object mapping rod names to radii")
        rod_radii = {str(k): float(v) for k, v in rod_radii.items()}

    return DigitalTwinObs(
        fruiting_base_pos=_coerce_xyz(data["fruiting_base_pos"], field="fruiting_base_pos"),
        weld_direction=_coerce_xyz(data["weld_direction"], field="weld_direction"),
        junction_names=list(junction_names),
        woody_part_start_pos=np.asarray(data["woody_part_start_pos"], dtype=np.float32),
        woody_part_end_pos=np.asarray(data["woody_part_end_pos"], dtype=np.float32),
        apple_radius=apple_radius,
        rod_radii=rod_radii,
    )


def save_digital_twin_obs(obs: DigitalTwinObs, path: str | Path) -> None:
    """Write a ``digital_twin_v2`` observation JSON file."""
    payload = {
        "schema": SCHEMA_VERSION,
        "fruiting_base_pos": list(obs.fruiting_base_pos),
        "weld_direction": list(obs.weld_direction),
        "junction_names": list(obs.junction_names),
        "woody_part_start_pos": obs.woody_part_start_pos.reshape(-1).tolist(),
        "woody_part_end_pos": obs.woody_part_end_pos.reshape(-1).tolist(),
    }
    if obs.apple_radius is not None:
        payload["apple_radius"] = float(obs.apple_radius)
    if obs.rod_radii is not None:
        payload["rod_radii"] = {str(k): float(v) for k, v in obs.rod_radii.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
