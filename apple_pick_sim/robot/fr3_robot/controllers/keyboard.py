"""Keyboard teleop helpers for FR3 TCP velocity control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import warp as wp

class _KeyViewer(Protocol):
    def is_key_down(self, key: str) -> bool: ...


@dataclass(frozen=True)
class EEVelocity:
    """TCP twist in the **world** frame (m/s and rad/s)."""

    linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def linear_vec(self) -> wp.vec3:
        return wp.vec3(*self.linear)

    @property
    def angular_vec(self) -> wp.vec3:
        return wp.vec3(*self.angular)

    def is_zero(self, tol: float = 1e-9) -> bool:
        return all(abs(v) < tol for v in (*self.linear, *self.angular))


def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    """Hamilton product ``a * b`` (warp quats are ``x, y, z, w``)."""
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def integrate_tcp_target(
    target: wp.transform,
    *,
    linear_vel: wp.vec3,
    angular_vel: wp.vec3,
    dt: float,
) -> wp.transform:
    """Integrate a rigid TCP target pose by ``dt`` using a constant world-frame twist."""
    pos = wp.transform_get_translation(target)
    rot = wp.transform_get_rotation(target)
    pos_new = pos + linear_vel * dt
    w = angular_vel
    ang_mag = float(wp.length(w))
    if ang_mag > 1e-12:
        delta_rot = wp.quat_from_axis_angle(w / ang_mag, ang_mag * dt)
        rot_new = wp.normalize(_quat_mul(delta_rot, rot))
    else:
        rot_new = rot
    return wp.transform(pos_new, rot_new)


# (key, action) — must stay in sync with :func:`read_keyboard_ee_velocity` axis pairs.
FR3_KEYBOARD_BINDINGS: tuple[tuple[str, str], ...] = (
    ("i", "translate TCP +world X"),
    ("k", "translate TCP -world X"),
    ("j", "translate TCP +world Y"),
    ("l", "translate TCP -world Y"),
    ("r", "translate TCP +world Z"),
    ("f", "translate TCP -world Z"),
    ("z", "rotate TCP +world X"),
    ("x", "rotate TCP -world X"),
    ("t", "rotate TCP +world Y"),
    ("g", "rotate TCP -world Y"),
    ("u", "rotate TCP +world Z"),
    ("o", "rotate TCP -world Z"),
)


def print_fr3_keyboard_bindings(*, stream: Any | None = None) -> None:
    """Print FR3 TCP teleop key map (``ViewerGL``; focus the simulation window)."""
    import sys

    out = sys.stdout if stream is None else stream
    print("FR3 keyboard teleop — focus the viewer window:", file=out)
    for key, action in FR3_KEYBOARD_BINDINGS:
        print(f"  {key}: {action}", file=out)
    print("  (W/A/S/D/Q/E move the camera, not the arm.)", file=out)


def _keyboard_axis(viewer: _KeyViewer, neg_key: str, pos_key: str) -> float:
    val = 0.0
    if viewer.is_key_down(neg_key):
        val -= 1.0
    if viewer.is_key_down(pos_key):
        val += 1.0
    return val


def poll_viewer_events(viewer: object | None) -> None:
    """Process pending GL window events so the next :func:`is_key_down` query is current.

    Newton's ``ViewerGL`` only polls the keyboard during ``end_frame()``; call this at the
    start of each simulation step when the main loop runs ``step()`` before ``render()``.
    """
    if viewer is None:
        return
    renderer = getattr(viewer, "renderer", None)
    if renderer is not None and hasattr(renderer, "update"):
        renderer.update()


def read_keyboard_ee_velocity(
    viewer: _KeyViewer | None,
    *,
    linear_speed: float = 0.2,
    angular_speed: float = 1.0,
    poll_events: bool = True,
) -> EEVelocity:
    """Read a world-frame TCP twist from the Newton ``ViewerGL`` keyboard (window must have focus).

    Requires ``ViewerGL`` (``is_key_down`` is a no-op on ``ViewerNull`` / ``ViewerViser``).

    Layout (avoids ``ViewerGL`` camera keys **W/A/S/D/Q/E**):

    - **I / K** — world ±X
    - **J / L** — world ±Y
    - **R / F** — world ±Z
    - **U / O** — rotate about world Z
    - **T / G** — rotate about world Y
    - **Z / X** — rotate about world X
    """
    if viewer is None or not hasattr(viewer, "is_key_down"):
        return EEVelocity()
    if poll_events:
        poll_viewer_events(viewer)
    lin = (
        _keyboard_axis(viewer, "k", "i") * linear_speed,
        _keyboard_axis(viewer, "l", "j") * linear_speed,
        _keyboard_axis(viewer, "f", "r") * linear_speed,
    )
    ang = (
        _keyboard_axis(viewer, "x", "z") * angular_speed,
        _keyboard_axis(viewer, "g", "t") * angular_speed,
        _keyboard_axis(viewer, "o", "u") * angular_speed,
    )
    return EEVelocity(linear=lin, angular=ang)

