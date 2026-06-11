"""Quasi-static stepped EE trajectory for stiffness mapping (§2.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity


@dataclass(frozen=True)
class QuasiStaticStepConfig:
    step_size_m: float = 0.02
    n_steps: int = 5
    hold_duration_s: float = 1.5
    move_speed_mps: float = 0.05
    control_hz: float = 60.0


class QuasiStaticTrajectory:
    """Stepped push-hold-return sequence along multiple world directions."""

    def __init__(
        self,
        directions: np.ndarray,
        config: QuasiStaticStepConfig | None = None,
    ) -> None:
        dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        if dirs.shape[0] == 0:
            raise ValueError("directions must be non-empty")
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("all directions must be non-zero")
        self._directions = (dirs / norms).astype(np.float64)
        self._config = config or QuasiStaticStepConfig()
        self._dir_index = 0
        self._phase: str | None = None
        self._amplitude_m = 0.0

    @property
    def current_direction(self) -> np.ndarray:
        return self._directions[self._dir_index].copy()

    @property
    def current_amplitude_m(self) -> float:
        return float(self._amplitude_m)

    def _move_frame_count(self) -> int:
        cfg = self._config
        duration = cfg.step_size_m / cfg.move_speed_mps
        return max(1, int(math.ceil(duration * cfg.control_hz)))

    def _return_frame_count(self) -> int:
        cfg = self._config
        total_dist = cfg.n_steps * cfg.step_size_m
        duration = total_dist / cfg.move_speed_mps
        return max(1, int(math.ceil(duration * cfg.control_hz)))

    def _hold_frame_count(self) -> int:
        cfg = self._config
        return max(0, int(math.ceil(cfg.hold_duration_s * cfg.control_hz)))

    def iter_frames(self) -> Iterator[tuple[str, EEVelocity]]:
        """Yield ``(phase, EEVelocity)`` frames for each direction cycle."""
        cfg = self._config
        move_frames = self._move_frame_count()
        return_frames = self._return_frame_count()
        hold_frames = self._hold_frame_count()
        total_out_frames = cfg.n_steps * move_frames
        out_step_delta = (cfg.n_steps * cfg.step_size_m) / total_out_frames
        ret_step_delta = (cfg.n_steps * cfg.step_size_m) / return_frames

        for dir_idx, direction in enumerate(self._directions):
            self._dir_index = dir_idx
            self._amplitude_m = 0.0

            self._phase = "move_out"
            out_vel = EEVelocity(linear=tuple(direction * cfg.move_speed_mps))
            for _ in range(total_out_frames):
                self._amplitude_m += out_step_delta
                yield self._phase, out_vel

            self._phase = "hold"
            hold_vel = EEVelocity()
            for _ in range(hold_frames):
                yield self._phase, hold_vel

            self._phase = "return"
            ret_vel = EEVelocity(linear=tuple(-direction * cfg.move_speed_mps))
            for _ in range(return_frames):
                self._amplitude_m = max(0.0, self._amplitude_m - ret_step_delta)
                yield self._phase, ret_vel

            self._amplitude_m = 0.0

        self._phase = None
