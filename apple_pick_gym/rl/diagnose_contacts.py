"""Does the fruit press on the tree? Apple/stem contact forces during a pick.

The bend-detach policy (D11) loads the wrist ~2x the spur-stem junction force. With 0.2 N of weld
force at rest, the difference most likely goes through contact: the apple pressed against the
branch. Whether that strategy is realistic (fruit bruising) depends on this contact.

For one synchronized episode of a checkpoint or baseline (same CLI as ``eval_vic_harvest``), each
step reads the plant solver's rigid contact forces (``SolverVBD.collect_rigid_contact_forces`` on
the contacts the step used). Per world, it sums the net contact force

- between the fruit (apple + stem segments) and the woody tree (``fruit_woody``);
- between the fruit and the gripper proxy (``fruit_proxy``).

It reports their episode peaks next to the peak wrist force, over all valid envs and over
successful envs. The robot's own contacts (MuJoCo) are disabled in this scene, so the gripper
proxy is the only robot-side contact.

    uv run python -m apple_pick_gym.rl.diagnose_contacts --checkpoint runs/.../ckpt_000001600 \\
        --out runs/diag/contacts.json
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

OTHER, FRUIT, PROXY = 0, 1, 2


def classify_bodies(labels: Sequence[str]) -> np.ndarray:
    """Body category per label: ``FRUIT`` (apple, stem segments), ``PROXY`` (gripper), else ``OTHER``."""
    out = np.full(len(labels), OTHER, dtype=np.int32)
    for i, name in enumerate(labels):
        low = str(name).lower()
        if "proxy" in low:
            out[i] = PROXY
        elif "apple" in low or low.startswith("stem"):
            out[i] = FRUIT
    return out


def per_env_contact_forces(
    body0: np.ndarray,
    body1: np.ndarray,
    force_on_body1: np.ndarray,
    count: int,
    category: np.ndarray,
    body_world: np.ndarray,
    *,
    num_envs: int,
) -> dict[str, np.ndarray]:
    """Per env: |net contact force| fruit<->woody and fruit<->proxy, and the fruit<->woody contact count.

    Only the first ``count`` contacts are active. The force on the fruit side is what is summed, so
    contacts are oriented to act on the fruit body before summing.
    """
    n = int(count)
    b0, b1, f = body0[:n], body1[:n], force_on_body1[:n]
    ok = (b0 >= 0) & (b1 >= 0)
    b0, b1, f = b0[ok], b1[ok], f[ok]
    c0, c1 = category[b0], category[b1]
    # force on the fruit body: body1 fruit -> +f ; body0 fruit -> -f
    fruit1 = c1 == FRUIT
    fruit0 = (c0 == FRUIT) & ~fruit1
    f_fruit = np.where(fruit1[:, None], f, -f)
    other = np.where(fruit1, c0, c1)
    env = body_world[np.where(fruit1, b1, b0)]
    out = {}
    for key, cls in (("fruit_woody", OTHER), ("fruit_proxy", PROXY)):
        sel = (fruit1 | fruit0) & (other == cls)
        net = np.zeros((num_envs, 3))
        np.add.at(net, env[sel], f_fruit[sel])
        out[f"{key}_n"] = np.linalg.norm(net, axis=-1)
        if key == "fruit_woody":
            out["fruit_woody_count"] = np.bincount(env[sel], minlength=num_envs)
    return out


def _stats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "p90": float(np.quantile(x, 0.9)), "max": float(x.max())}


def diagnose(wrapper, policy) -> dict:
    import warp as wp

    env = wrapper._env
    sim = env._sim
    cable = sim.scene.cable
    model, solver = cable.model, cable.solver
    labels = list(getattr(model, "body_label", []) or [])
    category = classify_bodies(labels)
    body_world = model.body_world.numpy() if model.body_world is not None else np.zeros(len(labels), np.int32)
    n = wrapper.num_envs

    obs, _ = wrapper.reset()
    state = wrapper.state()
    policy.reset(wrapper)
    peak = {k: np.zeros(n) for k in ("fruit_woody_n", "fruit_proxy_n", "wrist_n")}
    max_count = np.zeros(n, dtype=np.int64)
    info = {}
    for _ in range(env.max_episode_steps):
        obs, _r, term, trunc, info = wrapper.step(policy.act(wrapper, obs, state))
        state = wrapper.state()
        if hasattr(policy, "on_step"):
            policy.on_step(term | trunc)
        contacts = getattr(sim.scene, "last_vbd_contacts", None)
        if contacts is None:
            raise RuntimeError("scene.last_vbd_contacts is None: the step stored no plant contacts")
        if contacts.rigid_contact_force is None:  # extended attribute: allocate it for the readout
            contacts.rigid_contact_force = wp.zeros(contacts.rigid_contact_max, dtype=wp.vec3, device=solver.device)
        b0, b1, _p0, _p1, f, cnt = solver.collect_rigid_contact_forces(
            cable.state_0.body_q, cable.state_1.body_q, contacts, float(sim.sub_dt)
        )
        live = ~info["episode"]["frozen"].cpu().numpy() | term.flatten().cpu().numpy()
        res = per_env_contact_forces(
            b0.numpy(), b1.numpy(), f.numpy(), int(cnt.numpy()[0]), category, body_world, num_envs=n
        )
        for k in ("fruit_woody_n", "fruit_proxy_n"):
            peak[k] = np.where(live, np.maximum(peak[k], res[k]), peak[k])
        wrist = torch.linalg.norm(info["ft_wrist"][:, :3], dim=-1).cpu().numpy()
        peak["wrist_n"] = np.where(live, np.maximum(peak["wrist_n"], wrist), peak["wrist_n"])
        max_count = np.where(live, np.maximum(max_count, res["fruit_woody_count"]), max_count)
        if bool(trunc.all()):
            break
    valid = ~env.invalid_env_mask.cpu().numpy()
    ep = info["episode"]
    won = valid & ep["success_achieved"].cpu().numpy() & ~(ep["safety_junction"] | ep["safety_wrist"]).cpu().numpy()
    out = {"num_envs": n, "valid_envs": int(valid.sum()), "success_envs": int(won.sum())}
    for name, mask in (("all", valid), ("success", won)):
        out[name] = {k: _stats(v[mask]) for k, v in peak.items()}
        out[name]["fruit_woody_contacts_max"] = _stats(max_count[mask].astype(float))
        out[name]["envs_with_fruit_woody_contact"] = float((max_count[mask] > 0).mean()) if mask.any() else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    from apple_pick_gym.rl.eval_vic_harvest import build_eval, build_parser

    p = build_parser()
    p.description = __doc__
    args = p.parse_args(argv)
    _cfg, wrapper, policy, label = build_eval(args)
    try:
        res = diagnose(wrapper, policy)
    finally:
        wrapper.close()
    res["policy"] = label
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
