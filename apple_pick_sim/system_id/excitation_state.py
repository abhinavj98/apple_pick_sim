"""Excitation context shared between trajectory runners and Gym envs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExcitationContext:
    """Metadata describing the current system-ID excitation."""

    type: str  # "quasi_static" | "translational_chirp" | "torsional"
    f_inst: float
    direction: np.ndarray  # (3,) unit direction in world frame

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            raise ValueError("direction must be non-zero")
        object.__setattr__(self, "direction", (d / norm).astype(np.float64))
