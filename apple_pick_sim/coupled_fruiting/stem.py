"""Stem–apple joint discovery for unified-sync harvest path."""

from __future__ import annotations

from apple_pick_sim.fruiting_system import CoupledCableScene


def _find_stem_apple_joint(cable: CoupledCableScene) -> int | None:
    """Return the joint index of the stem-to-apple FIXED joint, or ``None``."""
    if cable.apple_body is None:
        return None
    jchild = cable.model.joint_child.numpy()
    for j_idx, _label in cable.fruiting_fixed_joints:
        if int(jchild[j_idx]) == cable.apple_body:
            return j_idx
    return None
