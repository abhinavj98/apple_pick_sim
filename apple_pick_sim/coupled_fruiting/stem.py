"""Stem–apple joint discovery for unified-sync harvest path."""

from __future__ import annotations

from apple_pick_sim.fruiting_system import CoupledCableScene


def _find_stem_apple_joint(cable: CoupledCableScene) -> int | None:
    """Return the joint index of the stem-to-apple FIXED joint, or ``None``.

    Skips ``gripper_proxy_apple_joint`` when ``fix_to_apple`` adds a second FIXED
    joint with the same child body.
    """
    if cable.apple_body is None:
        return None
    skip: set[int] = set()
    proxy_j = getattr(cable, "gripper_proxy_apple_joint", None)
    if proxy_j is not None:
        skip.add(int(proxy_j))
    jchild = cable.model.joint_child.numpy()
    for j_idx, _label in cable.fruiting_fixed_joints:
        if int(j_idx) in skip:
            continue
        if int(jchild[j_idx]) == cable.apple_body:
            return j_idx
    return None
