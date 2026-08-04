"""Collision geometry helpers for the VBD gripper proxy and MuJoCo TCP stand-in."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import newton
import warp as wp

if TYPE_CHECKING:
    from apple_pick_sim.fruiting_system.params import GripperProxyConfig

# Real-world bench tool (docs/real-world-proxy.md): 50 mm radius, 140 mm length.
GRIPPER_PROXY_CYLINDER_RADIUS = 0.05
GRIPPER_PROXY_CYLINDER_HALF_HEIGHT = 0.07


def gripper_proxy_clearance(config: GripperProxyConfig) -> float:
    """Distance from apple-surface contact to the proxy body origin along approach.

    For the default TCP-at-tip cylinder, the body origin sits on the apple surface
    (distal tip) and the tool bulk extends toward the flange along local −Z
    (tip-out +Z toward the fruit); clearance is zero.

    For a centered box proxy, the body origin sits one half-extent outside the surface.
    """
    if config.shape == "cylinder":
        return 0.0
    hx, hy, hz = config.box_half_extents
    return max(hx, hy, hz)


def gripper_proxy_cylinder_tcp_xform(config: GripperProxyConfig) -> wp.transform:
    """Cylinder with distal tip at the TCP / body origin; bulk extends toward −Z (flange).

    Matches the USD / recorded-TCP contract: TCP is the face farthest from the
    flange, local +Z is tip-out (toward the apple when grasping).
    """
    hh = float(config.cylinder_half_height)
    return wp.transform(wp.vec3(0.0, 0.0, -hh), wp.quat_identity())


def add_gripper_proxy_collision_shape(
    builder: newton.ModelBuilder,
    body: int,
    config: GripperProxyConfig,
    *,
    shape_cfg: newton.ShapeConfig,
) -> None:
    """Add box or cylinder collision geometry matching ``config``."""
    if config.shape == "cylinder":
        builder.add_shape_cylinder(
            body=body,
            xform=gripper_proxy_cylinder_tcp_xform(config),
            radius=config.cylinder_radius,
            half_height=config.cylinder_half_height,
            cfg=shape_cfg,
        )
        return

    hx, hy, hz = config.box_half_extents
    builder.add_shape_box(body=body, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)


def gripper_proxy_shape_type(config: GripperProxyConfig) -> Literal["box", "cylinder"]:
    return config.shape
