"""Quasi-static stepped EE trajectory for stiffness mapping (§2.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from apple_pick_sim.robot.fr3_robot.controllers.keyboard import EEVelocity


@dataclass(frozen=True)
class QuasiStaticStepConfig:
    """Per-increment fast move + hold quasi-static push parameters."""

    movement_per_step_m: float = 0.05
    total_movement_m: float = 0.10
    hold_duration_s: float = 1.5
    move_speed_mps: float = 0.2
    control_hz: float = 60.0
    skip_return: bool = True

    def __post_init__(self) -> None:
        if self.movement_per_step_m <= 0.0:
            raise ValueError("movement_per_step_m must be positive")
        if self.total_movement_m <= 0.0:
            raise ValueError("total_movement_m must be positive")
        derive_n_steps(
            movement_per_step_m=self.movement_per_step_m,
            total_movement_m=self.total_movement_m,
        )


def derive_n_steps(*, movement_per_step_m: float, total_movement_m: float) -> int:
    """Return the number of move-hold increments for a quasi-static push."""
    if movement_per_step_m <= 0.0:
        raise ValueError("movement_per_step_m must be positive")
    if total_movement_m <= 0.0:
        raise ValueError("total_movement_m must be positive")
    n_steps = max(1, round(total_movement_m / movement_per_step_m))
    if abs(n_steps * movement_per_step_m - total_movement_m) > 1e-6:
        raise ValueError(
            "total_movement_m must be an integer multiple of movement_per_step_m "
            f"(got {total_movement_m} / {movement_per_step_m})"
        )
    return n_steps


def estimate_trajectory_frames(config: QuasiStaticStepConfig, n_directions: int) -> int:
    """Estimate env steps for a full multi-direction quasi-static trajectory."""
    n_steps = derive_n_steps(
        movement_per_step_m=config.movement_per_step_m,
        total_movement_m=config.total_movement_m,
    )
    move_frames = max(
        1,
        int(math.ceil(config.movement_per_step_m / config.move_speed_mps * config.control_hz)),
    )
    return_frames = max(
        1,
        int(math.ceil(config.total_movement_m / config.move_speed_mps * config.control_hz)),
    )
    hold_frames = max(0, int(math.ceil(config.hold_duration_s * config.control_hz)))
    per_dir = n_steps * (move_frames + hold_frames)
    if not config.skip_return:
        per_dir += return_frames
    return int(n_directions) * per_dir


class QuasiStaticTrajectory:
    """Fast move + hold push sequence along multiple world directions."""

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
        self._step_index = 0
        self._phase: str | None = None
        self._amplitude_m = 0.0

    @property
    def n_steps(self) -> int:
        cfg = self._config
        return derive_n_steps(
            movement_per_step_m=cfg.movement_per_step_m,
            total_movement_m=cfg.total_movement_m,
        )

    @property
    def current_direction(self) -> np.ndarray:
        return self._directions[self._dir_index].copy()

    @property
    def current_step_index(self) -> int:
        return int(self._step_index)

    @property
    def current_amplitude_m(self) -> float:
        return float(self._amplitude_m)

    def _move_frame_count(self) -> int:
        cfg = self._config
        duration = cfg.movement_per_step_m / cfg.move_speed_mps
        return max(1, int(math.ceil(duration * cfg.control_hz)))

    def _return_frame_count(self) -> int:
        cfg = self._config
        duration = cfg.total_movement_m / cfg.move_speed_mps
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
        step_delta = cfg.movement_per_step_m / move_frames
        ret_step_delta = cfg.total_movement_m / return_frames
        n_steps = self.n_steps

        for dir_idx, direction in enumerate(self._directions):
            self._dir_index = dir_idx
            self._amplitude_m = 0.0

            out_vel = EEVelocity(linear=tuple(direction * cfg.move_speed_mps))
            for step_idx in range(n_steps):
                self._step_index = step_idx
                self._phase = "move_out"
                for _ in range(move_frames):
                    self._amplitude_m += step_delta
                    yield self._phase, out_vel

                self._phase = "hold"
                self._amplitude_m = (step_idx + 1) * cfg.movement_per_step_m
                hold_vel = EEVelocity()
                for _ in range(hold_frames):
                    yield self._phase, hold_vel

            if not cfg.skip_return:
                self._phase = "return"
                ret_vel = EEVelocity(linear=tuple(-direction * cfg.move_speed_mps))
                for _ in range(return_frames):
                    self._amplitude_m = max(0.0, self._amplitude_m - ret_step_delta)
                    yield self._phase, ret_vel

            self._amplitude_m = 0.0

        self._phase = None
        self._step_index = 0
