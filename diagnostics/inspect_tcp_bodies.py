"""Diagnostic: print all body names, TCP body used for IK, and current tcp_pose.

Run from the repo root::

    uv run python diagnostics/inspect_tcp_bodies.py

Prints:
  1. All Newton body labels (index → name)
  2. Which body index resolve_tcp_body_index() picks (IK target)
  3. Which body index resolve_ee_body_index()  picks (EE link)
  4. MuJoCo body names visible to the solver (mj_model.body_names)
  5. Which MuJoCo body name maps to the Newton tcp_body_index
  6. The post-FK tcp_pose (position + quaternion) stored in state.body_q
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from apple_pick_sim.robot import fr3_robot


def main() -> None:
    print("=" * 60)
    print("Building FR3 model from USD …")
    model, tcp_idx, mj_solver = fr3_robot.build_fr3_robot_model_from_usd()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # ── 1. All Newton body labels ───────────────────────────────────────────
    labels = list(model.body_label)
    print(f"\n{'─' * 60}")
    print(f"Newton bodies ({len(labels)} total):")
    for i, lbl in enumerate(labels):
        print(f"  [{i:3d}]  {lbl}")

    # ── 2. TCP body chosen by resolve_tcp_body_index ────────────────────────
    print(f"\n{'─' * 60}")
    tcp_label = labels[tcp_idx] if 0 <= tcp_idx < len(labels) else "<out of range>"
    print(f"resolve_tcp_body_index() → index {tcp_idx}  label: {tcp_label!r}")

    # ── 3. EE body chosen by resolve_ee_body_index ──────────────────────────
    try:
        ee_idx = fr3_robot.resolve_ee_body_index(model)
        ee_label = labels[ee_idx] if 0 <= ee_idx < len(labels) else "<out of range>"
    except ValueError as e:
        ee_idx = -1
        ee_label = f"<error: {e}>"
    print(f"resolve_ee_body_index()  → index {ee_idx}  label: {ee_label!r}")

    are_same = tcp_idx == ee_idx
    print(f"\n  ⚠  TCP == EE body?  {'YES — no separate /tcp child found; ee body IS the tcp'  if are_same else 'NO — separate tcp child exists'}")

    # ── 4. MuJoCo body names ────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    mj_model = mj_solver.mj_model
    try:
        # mujoco body names: body 0 is the world body, skip it
        mj_body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
        print(f"MuJoCo bodies ({mj_model.nbody} total, including world body at index 0):")
        for i, name in enumerate(mj_body_names):
            print(f"  [{i:3d}]  {name}")
    except Exception as e:
        print(f"  (could not read mj_model body names: {e})")
        mj_body_names = []

    # ── 5. Find the MuJoCo body that corresponds to the Newton tcp_body_index
    print(f"\n{'─' * 60}")
    print(f"Newton tcp label: {tcp_label!r}")
    # Newton body_label paths look like "Robot/fr3/ee" or "Robot/fr3/ee/tcp".
    # MuJoCo body names are typically the last path component.
    tcp_short = tcp_label.split("/")[-1] if "/" in tcp_label else tcp_label
    matched_mj = [(i, n) for i, n in enumerate(mj_body_names) if n == tcp_short or n == tcp_label]
    if matched_mj:
        for mi, mn in matched_mj:
            print(f"  MuJoCo body for IK: index {mi}  name: {mn!r}")
    else:
        print(f"  No exact MuJoCo body match for {tcp_short!r} / {tcp_label!r}")
        # Fuzzy: show any that contain 'tcp' or 'ee'
        fuzzy = [(i, n) for i, n in enumerate(mj_body_names) if "tcp" in n.lower() or "ee" in n.lower()]
        if fuzzy:
            print("  Fuzzy matches (contain 'tcp' or 'ee'):")
            for mi, mn in fuzzy:
                print(f"    [{mi}]  {mn!r}")

    # ── 6. tcp_pose from state.body_q ───────────────────────────────────────
    print(f"\n{'─' * 60}")
    body_q = state.body_q.numpy().reshape(-1, 7)
    tcp_q7 = body_q[tcp_idx]
    pos = tcp_q7[:3]
    quat = tcp_q7[3:]  # Newton quaternion is (x, y, z, w) in body_q
    print("tcp_pose stored in state.body_q (post-FK, zero-command pose):")
    print(f"  position  : x={pos[0]:.6f}  y={pos[1]:.6f}  z={pos[2]:.6f}  [m]")
    print(f"  quaternion: x={quat[0]:.6f}  y={quat[1]:.6f}  z={quat[2]:.6f}  w={quat[3]:.6f}")
    print(f"  body_q row: {tcp_q7.tolist()}")

    if are_same:
        print(
            "\n  NOTE: The body labelled '/ee' is being used as the TCP.\n"
            "  There is NO separate '/ee/tcp' child body in the model.\n"
            "  This means IK, velocity control, and tcp_pos observations\n"
            "  all refer to the /ee body frame."
        )
    else:
        ee_q7 = body_q[ee_idx]
        ee_pos = ee_q7[:3]
        offset = np.linalg.norm(pos - ee_pos)
        print(f"\n  EE body position : x={ee_pos[0]:.6f}  y={ee_pos[1]:.6f}  z={ee_pos[2]:.6f}")
        print(f"  TCP vs EE offset : {offset*1000:.3f} mm")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
