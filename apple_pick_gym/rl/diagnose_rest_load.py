"""Where does the spur-stem rest torque come from? Statics of the hanging stem + apple at reset.

The maintainer's question (2026-09-24): the target junction already carries 16-45% of tau_max at
rest. Is that physical, or a bug?

At reset the system is static, so the free body *below* the spur-stem junction (every stem
segment plus the apple) is in equilibrium under:

- the junction wrench ``W_J``: the envelope's stem-elastic wrench (D1), plus the rigid readout
  for comparison;
- gravity on the subtree, ``W_g``: computed here from body masses and COMs;
- the gripper, through the apple-proxy weld, ``W_grip``: read from the raw wrist wrench.

So ``W_J + W_g + W_grip ~ 0`` about the junction anchor, and the numbers split the rest torque
into a gravity part (a stem bent by the apple's weight: physical) and a grip part (a grasp or
settle that pushes the apple off its hanging pose: an artifact to fix). The wrist wrench's sign
convention is not assumed; both signs are reported with their balance residuals.

Also reported: stem angle from vertical, the apple's horizontal lever about the junction, and
the VIC hold-target offset at reset (a nonzero offset means the controller pushes at rest).

    uv run python -m apple_pick_gym.rl.diagnose_rest_load --config apple_pick_gym/rl/configs/sim_smoke_gpu.json \\
        --out runs/diag/rest_load.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from apple_pick_gym.rl.config import TrainConfig


def descendants(root: int, joint_parent: np.ndarray, joint_child: np.ndarray) -> list[int]:
    """Bodies reachable from ``root`` by following joints parent -> child (``root`` included)."""
    out, stack = {int(root)}, [int(root)]
    while stack:
        b = stack.pop()
        for c in joint_child[joint_parent == b]:
            c = int(c)
            if c >= 0 and c not in out:
                out.add(c)
                stack.append(c)
    return sorted(out)


def gravity_wrench_about(
    com: torch.Tensor, mass: torch.Tensor, point: torch.Tensor, *, g: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gravity force and moment about ``point`` of ``(N, B)`` point masses at ``(N, B, 3)`` COMs.

    Padded bodies take mass 0. Returns ``(N, 3)`` force and ``(N, 3)`` moment.
    """
    gv = torch.as_tensor(g, dtype=com.dtype, device=com.device)
    f_b = mass.unsqueeze(-1) * gv  # (N, B, 3)
    r = com - point.unsqueeze(1)
    return f_b.sum(1), torch.cross(r, f_b, dim=-1).sum(1)


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.float()
    return {k: float(x.quantile(q)) for k, q in (("p10", 0.1), ("median", 0.5), ("p90", 0.9), ("max", 1.0))}


def _hold_action(env, k_lin: float = 150.0, k_ang: float = 10.0, zeta: float = 1.0) -> torch.Tensor:
    """Env-unit 13-D action: no target change, moderate stiffness."""
    a = torch.zeros(env.num_envs, 13, device=env.device)
    a[:, 6:9], a[:, 9:12], a[:, 12] = k_lin, k_ang, zeta
    return a


def _weld_wrench_on_apple_at(env, point: torch.Tensor) -> torch.Tensor | None:
    """``(N, 6)`` wrench the gripper applies to the apple through the proxy-apple weld, about ``point``.

    Read from the weld joint itself (wrench on its child, the proxy, at the child COM), negated
    for the apple and moved to ``point``. ``None`` if the scene has no batched weld indices.
    """
    import warp as wp

    from apple_pick_gym.batched_envs.harvest_reward import quat_rotate_vector
    from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device

    sim = env._sim
    scene = sim.scene
    joints = getattr(scene, "weld_harvest_joint_indices_wp", None)
    if joints is None:
        return None
    cable = scene.cable
    f, t = gather_joint_wrench_child_com_device(
        cable.model, cable.solver, body_q=cable.state_0.body_q, body_q_prev=cable.state_1.body_q,
        joint_indices=joints, dt=float(sim.sub_dt),
    )
    dev = env.device
    force = -wp.to_torch(f).to(device=dev, dtype=torch.float32)
    torque_com = -wp.to_torch(t).to(device=dev, dtype=torch.float32)
    child = torch.as_tensor(cable.model.joint_child.numpy()[joints.numpy()].astype(np.int64), device=dev)
    body_q = wp.to_torch(cable.state_0.body_q).to(device=dev, dtype=torch.float32)
    body_com = wp.to_torch(cable.model.body_com).to(device=dev, dtype=torch.float32)
    q = body_q[child]
    com = q[:, :3] + quat_rotate_vector(q[:, 3:7], body_com[child])
    return torch.cat([force, torque_com + torch.cross(com - point, force, dim=-1)], dim=-1)


def diagnose(env, *, hold_steps: int = 0) -> dict:
    """Reset ``env`` (``ApplePickVicHarvestEnv``), optionally hold ``hold_steps`` steps, and decompose
    the target junction's rest wrench."""
    import warp as wp

    from apple_pick_gym.batched_envs.harvest_reward import quat_rotate_vector

    obs, info = env.reset()
    for _ in range(hold_steps):
        obs, _r, _te, _tr, info = env.step(_hold_action(env))
    n, dev = env.num_envs, env.device
    cable = env._sim.scene.cable
    model = cable.model
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    labels = list(getattr(model, "body_label", []) or [])
    roots = env._target_child_body.cpu().numpy()
    # The gripper proxy is welded to the apple but held by the arm: its weight is not hanging load.
    # It reaches the apple through the weld, whose reaction is the wrist wrench below.
    is_proxy = lambda b: b < len(labels) and "proxy" in labels[b]
    subtrees = [[b for b in descendants(int(r), parent, child) if not is_proxy(b)] for r in roots]
    width = max(len(s) for s in subtrees)
    idx = torch.zeros(n, width, dtype=torch.long)
    live = torch.zeros(n, width)
    for w, s in enumerate(subtrees):
        idx[w, : len(s)] = torch.as_tensor(s)
        live[w, : len(s)] = 1.0
    idx, live = idx.to(dev), live.to(dev)

    body_q = wp.to_torch(cable.state_0.body_q).to(device=dev, dtype=torch.float32)
    body_com = wp.to_torch(model.body_com).to(device=dev, dtype=torch.float32)
    body_mass = wp.to_torch(model.body_mass).to(device=dev, dtype=torch.float32)
    q = body_q[idx]  # (N, B, 7)
    com = q[..., :3] + quat_rotate_vector(q[..., 3:7], body_com[idx])
    mass = body_mass[idx] * live
    g = model.gravity.numpy().reshape(-1, 3)[0] if model.gravity is not None else np.array([0.0, 0.0, -9.81])
    j = info["woody_part_info"][env.target_junction_name]["anchors_pos"][:, 3:6] if "woody_part_info" in info else (
        env._last_full_obs["woody_part_info"][env.target_junction_name]["anchors_pos"][:, 3:6]
    )
    f_g, t_g = gravity_wrench_about(com, mass, j, g=tuple(float(v) for v in g))

    w_j = info["target_junction_wrench"]
    w_ro = info["junction_readout_wrench"]
    ft = info["ft_wrist"]  # raw weld-reaction wrench at the TCP (world frame)
    tcp = obs["tcp_pos"]
    t_grip_j = ft[:, 3:] + torch.cross(tcp - j, ft[:, :3], dim=-1)

    weld = _weld_wrench_on_apple_at(env, j)
    res = {}
    if weld is not None:
        rf = w_j[:, :3] + f_g + weld[:, :3]
        rt = w_j[:, 3:] + t_g + weld[:, 3:]
        res["weld_joint"] = {
            "force_residual_n": _stats(torch.linalg.norm(rf, dim=-1)),
            "torque_residual_nm": _stats(torch.linalg.norm(rt, dim=-1)),
        }
    for s in (+1.0, -1.0):
        rf = w_j[:, :3] + f_g + s * ft[:, :3]
        rt = w_j[:, 3:] + t_g + s * t_grip_j
        res[f"{'+' if s > 0 else '-'}ft"] = {
            "force_residual_n": _stats(torch.linalg.norm(rf, dim=-1)),
            "torque_residual_nm": _stats(torch.linalg.norm(rt, dim=-1)),
        }

    axis = info.get("target_junction_axis")
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    stem_deg = torch.rad2deg(torch.arccos(torch.clamp((axis * down).sum(-1), -1.0, 1.0))) if axis is not None else None
    apple = info.get("apple_pos")
    lever = torch.linalg.norm((apple - j)[:, :2], dim=-1) if apple is not None else None
    tgt = getattr(env, "_target_pose", None)
    hold_off = torch.linalg.norm(tgt[:, :3] - tcp, dim=-1) if tgt is not None else None

    valid = ~env.invalid_env_mask
    v = lambda x: x[valid]
    norm = lambda x: torch.linalg.norm(x, dim=-1)
    tau_max = float(env._reward_cfg.detach.tau_max_nm)
    out = {
        "num_envs": n,
        "valid_envs": int(valid.sum()),
        "subtree_bodies_env0": [labels[b] if b < len(labels) else str(b) for b in subtrees[0]],
        "subtree_body_mass_env0_kg": [float(body_mass[b]) for b in subtrees[0]],
        "subtree_mass_kg": _stats(v(mass.sum(1))),
        "gravity": [float(x) for x in g],
        "junction_torque_stem_nm": _stats(v(norm(w_j[:, 3:]))),
        "junction_torque_readout_nm": _stats(v(norm(w_ro[:, 3:]))),
        "junction_force_n": _stats(v(norm(w_j[:, :3]))),
        "rest_torque_fraction_of_tau_max": _stats(v(norm(w_j[:, 3:])) / tau_max),
        "gravity_torque_about_junction_nm": _stats(v(norm(t_g))),
        "gravity_force_n": _stats(v(norm(f_g))),
        "grip_torque_about_junction_nm": _stats(v(norm(t_grip_j))),
        "grip_force_n": _stats(v(norm(ft[:, :3]))),
        "cos_junction_vs_minus_gravity_torque": _stats(
            v(torch.nn.functional.cosine_similarity(w_j[:, 3:], -t_g, dim=-1))
        ),
        "hold_steps": hold_steps,
        "balance_residuals": res,
    }
    if weld is not None:
        out["weld_force_on_apple_n"] = _stats(v(norm(weld[:, :3])))
        out["weld_torque_about_junction_nm"] = _stats(v(norm(weld[:, 3:])))
    if stem_deg is not None:
        out["stem_angle_from_down_deg"] = _stats(v(stem_deg))
    if lever is not None:
        out["apple_horizontal_lever_m"] = _stats(v(lever))
    if hold_off is not None:
        out["vic_hold_target_offset_m"] = _stats(v(hold_off))
    return out


def main(argv: list[str] | None = None) -> int:
    from apple_pick_gym.rl.trainer import build_env

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="TrainConfig JSON (env section is used)")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--hold-steps", type=int, default=0, help="hold steps before measuring (0 = at reset)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    cfg = TrainConfig.load_json(a.config)
    env_cfg = cfg.env if a.num_envs is None else dataclasses.replace(cfg.env, num_envs=a.num_envs)
    env = build_env(env_cfg, seed=a.seed)
    try:
        res = diagnose(env, hold_steps=a.hold_steps)
    finally:
        env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    for k, val in res.items():
        print(f"{k}: {json.dumps(val)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
