"""Debug readouts for staggered VBD ↔ MuJoCo proxy wrench transfer.

Records the **lagged wrench applied to the robot** (copy of ``proxy_forces`` at the
start of the MuJoCo substep) and the **fresh harvest** from the VBD proxy after the
cable step. Plot via :meth:`CouplingForceDebugRecorder.log_to_viewer` (Newton
``ViewerGL`` / ``ViewerViser`` ``log_scalar`` time series).

With a quiescent placeholder TCP, both traces should stay near **zero** (not ~mg
growing each substep). Large |F| usually means ``body_q_prev`` was not aligned after
``sync_proxy_state`` (see :func:`~apple_pick_sim.coupled_fruiting.proxy_coupling.align_proxy_body_q_prev_for_vbd`)
or the proxy is contact-pinned while the robot moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp


def wrench_magnitudes(wrench_6: np.ndarray) -> tuple[float, float]:
    """Return ``(|F|, |τ|)`` for a spatial wrench ``[fx,fy,fz,tx,ty,tz]`` (world frame)."""
    w = np.asarray(wrench_6, dtype=np.float64).reshape(6)
    return float(np.linalg.norm(w[:3])), float(np.linalg.norm(w[3:]))


def read_tcp_wrench(
    wrenches: wp.array | np.ndarray,
    tcp_body_index: int,
) -> np.ndarray:
    """Read one TCP spatial wrench from a ``body_count`` array of spatial vectors."""
    arr = wrenches.numpy() if isinstance(wrenches, wp.array) else np.asarray(wrenches)
    flat = arr.reshape(-1, 6)
    return flat[tcp_body_index].astype(np.float64, copy=False)


@dataclass
class CouplingForceDebugRecorder:
    """Last substep snapshots for the TCP coupling wrenches (world frame, about COM)."""

    applied_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    harvested_wrench: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    applied_force_mag: float = 0.0
    applied_torque_mag: float = 0.0
    harvested_force_mag: float = 0.0
    harvested_torque_mag: float = 0.0

    def record_applied(self, wrench_6: np.ndarray | wp.array) -> None:
        """Wrench about to be written into ``robot_state_0.body_f`` (lagged harvest)."""
        self._store(wrench_6, target="applied")

    def record_harvested(self, wrench_6: np.ndarray | wp.array) -> None:
        """Wrench from ``harvest_proxy_wrenches`` after the VBD substep."""
        self._store(wrench_6, target="harvested")

    def record_applied_from_scene(self, scene: Any) -> None:
        """Read lagged wrench from ``scene.coupling_forces_cache`` at ``scene.tcp_body_index``."""
        if scene.coupling_forces_cache is None:
            return
        self.record_applied(read_tcp_wrench(scene.coupling_forces_cache, scene.tcp_body_index))

    def record_harvested_from_scene(self, scene: Any) -> None:
        """Read fresh harvest from ``scene.proxy_forces`` at ``scene.tcp_body_index``."""
        if scene.proxy_forces is None:
            return
        self.record_harvested(read_tcp_wrench(scene.proxy_forces, scene.tcp_body_index))

    def _store(self, wrench_6: np.ndarray | wp.array, *, target: str) -> None:
        w = np.asarray(wrench_6, dtype=np.float64).reshape(6)
        fmag, tmag = wrench_magnitudes(w)
        if target == "applied":
            self.applied_wrench = w
            self.applied_force_mag = fmag
            self.applied_torque_mag = tmag
        else:
            self.harvested_wrench = w
            self.harvested_force_mag = fmag
            self.harvested_torque_mag = tmag

    def log_to_viewer(self, viewer: Any, *, smoothing: int = 3) -> None:
        """Push scalar time series for both coupling channels (requires ``log_scalar``)."""
        log = getattr(viewer, "log_scalar", None)
        if log is None:
            return

        def _log(label: str, value: float) -> None:
            log(label, value, smoothing=smoothing)

        _log("Coupling → MuJoCo |F| [N]", self.applied_force_mag)
        _log("Coupling → MuJoCo |τ| [N·m]", self.applied_torque_mag)
        _log("Coupling ← VBD harvest |F| [N]", self.harvested_force_mag)
        _log("Coupling ← VBD harvest |τ| [N·m]", self.harvested_torque_mag)

        for axis, idx in zip("xyz", range(3), strict=True):
            _log(f"Coupling → MuJoCo F{axis} [N]", float(self.applied_wrench[idx]))
            _log(f"Coupling ← VBD harvest F{axis} [N]", float(self.harvested_wrench[idx]))
