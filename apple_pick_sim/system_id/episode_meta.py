"""Episode-level metadata for sysID trajectory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeMeta:
    """One row in ``metadata.parquet`` for a recorded sysID episode."""

    episode_id: str
    weld_direction: tuple[float, float, float]
    excitation_type: str
    n_woody_parts: int
    params_fingerprint: str
    control_hz: float
    junction_names: list[str] | None = None
    timestamp: str | None = None
    seed: int | None = None
    n_directions: int | None = None
    initial_tcp_pos: tuple[float, float, float] | None = None
    fixture_path: str | None = None
    movement_per_step_m: float | None = None
    total_movement_m: float | None = None
    hold_duration_s: float | None = None
    move_speed_mps: float | None = None
    skip_return: bool | None = None

    def to_row(self) -> dict[str, Any]:
        """Return a flat dict suitable for PyArrow metadata table rows."""
        return {
            "episode_id": self.episode_id,
            "weld_direction": list(self.weld_direction),
            "excitation_type": self.excitation_type,
            "n_woody_parts": int(self.n_woody_parts),
            "junction_names": (
                None if self.junction_names is None else list(self.junction_names)
            ),
            "params_fingerprint": self.params_fingerprint,
            "control_hz": float(self.control_hz),
            "timestamp": self.timestamp,
            "seed": self.seed,
            "n_directions": self.n_directions,
            "initial_tcp_pos": (
                None if self.initial_tcp_pos is None else list(self.initial_tcp_pos)
            ),
            "fixture_path": self.fixture_path,
            "movement_per_step_m": self.movement_per_step_m,
            "total_movement_m": self.total_movement_m,
            "hold_duration_s": self.hold_duration_s,
            "move_speed_mps": self.move_speed_mps,
            "skip_return": self.skip_return,
        }
