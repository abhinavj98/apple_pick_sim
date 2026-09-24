"""Record raw spur-stem wrenches with detachment disabled, then replay detach rules offline.

Why: on GPU the junction torque carries a solver-noise floor comparable to ``tau_max``, so which
envs "detach" depends on noise. The rule has to be chosen on data, and without paying for a GPU
run per variant. This tool rolls each policy for one episode with the envelope set out of reach,
so nobody freezes on success and every env's full load history is recorded. It then evaluates a
grid of rules offline:

- envelope: ``total`` ((F/f_max)^2 + (tau/tau_max)^2), ``split`` (torsion and bending get their
  own limits), ``force_only`` ((F/f_max)^2);
- filter on the wrench before the envelope: ``raw`` or a causal EMA at 10 / 3 / 1 Hz (60 Hz control);
- success streak: 3 / 10 / 30 consecutive steps at index >= 1;
- a ``tau_max`` grid for the total envelope (``--tau-max-grid``): which limit stops a do-nothing or
  random policy from "detaching" on noise while a real pull still detaches.

Rows report the would-detach rate over valid envs and the median steps to detach. The noise
block gives percentiles of |F|, |tau|, step-to-step |d tau|, torsion and bending over live
steps. On the real env it also reports the torque read without the AVBD penalty-damping term
(``kd * dC/dt``), which tests whether the noise comes from that velocity term.

Open-loop baselines only (zero / random / scripted): a frozen env would change what a closed-loop
policy does next, so the replay is exact only for policies that ignore observations.

Example (GPU)::

    uv run python -m apple_pick_gym.rl.detach_sweep --config apple_pick_gym/rl/configs/sim_wiring_gpu.json \\
        --policies zero random scripted_pull scripted_twist_pull --seed 12345 --out runs/diag/sweep.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from apple_pick_gym.batched_envs.harvest_detach import (
    DetachEnvelopeConfig,
    detach_index,
    junction_wrench_at_anchor,
    split_torque,
)
from apple_pick_gym.rl.config import TrainConfig

_FILTERS = {"raw": None, "ema10hz": 10.0, "ema3hz": 3.0, "ema1hz": 1.0}
_STREAKS = (3, 10, 30)


def ema(x: torch.Tensor, *, alpha: float) -> torch.Tensor:
    """Causal EMA over the leading (time) axis, initialized at the first sample."""
    out = torch.empty_like(x)
    acc = x[0]
    for t in range(x.shape[0]):
        acc = x[t] if t == 0 else alpha * x[t] + (1.0 - alpha) * acc
        out[t] = acc
    return out


def first_detach(index: torch.Tensor, *, streak: int) -> torch.Tensor:
    """``(T, N)`` index -> ``(N,)`` step at which ``index >= 1`` completes ``streak`` consecutive steps, else -1."""
    t_len, n = index.shape
    run = torch.zeros(n, dtype=torch.long)
    first = torch.full((n,), -1, dtype=torch.long)
    for t in range(t_len):
        run = torch.where(index[t] >= 1.0, run + 1, torch.zeros_like(run))
        first = torch.where((first < 0) & (run >= streak), torch.full_like(first, t), first)
    return first


def _undamped_target_wrench(env) -> torch.Tensor | None:
    """Real env only: spur-stem wrench (anchor frame) read without the AVBD penalty-damping term."""
    sim = getattr(env, "_sim", None)
    if sim is None:
        return None
    import warp as wp

    from apple_pick_sim.vbd_fixed_joint_wrenches import gather_joint_wrench_child_com_device

    cable = sim.scene.cable
    bufs = sim.obs_bufs
    joints = bufs.woody_joint_indices.numpy().reshape(env.num_envs, -1)[:, env._target_junction_idx]
    f, t = gather_joint_wrench_child_com_device(
        cable.model,
        cable.solver,
        body_q=cable.state_0.body_q,
        body_q_prev=cable.state_1.body_q,
        joint_indices=np.ascontiguousarray(joints, dtype=np.int32),
        dt=float(sim.sub_dt),
        include_penalty_damping=False,
    )
    w = torch.cat([wp.to_torch(f), wp.to_torch(t)], dim=-1).to(env.device, torch.float32)
    anchor = env._last_full_obs["woody_part_info"][env._target_junction_name]["anchors_pos"][:, 3:6]
    return junction_wrench_at_anchor(w, child_anchor=anchor, child_com=env._target_child_com_world())


def _pct(x: torch.Tensor, q: float) -> float:
    return float(x.float().quantile(q)) if x.numel() else float("nan")


def record(wrapper, policy) -> dict:
    env = wrapper._env
    obs, _ = wrapper.reset()
    state = wrapper.state()
    policy.reset(wrapper)
    rec = {"W": [], "AX": [], "WU": [], "FROZ": [], "APPLE": []}
    for _ in range(env.max_episode_steps - 1):
        obs, _r, _term, _trunc, info = wrapper.step(policy.act(wrapper, obs, state))
        state = wrapper.state()
        rec["W"].append(info["target_junction_wrench"].detach().cpu())
        axis = info.get("target_junction_axis")
        rec["AX"].append(None if axis is None else axis.detach().cpu())
        wu = _undamped_target_wrench(env)
        rec["WU"].append(None if wu is None else wu.detach().cpu())
        rec["FROZ"].append(info["episode"]["frozen"].detach().cpu())
        apple = info.get("apple_pos")
        rec["APPLE"].append(None if apple is None else apple.detach().cpu())
    return {
        "APPLE": None if rec["APPLE"][0] is None else torch.stack(rec["APPLE"]),
        "W": torch.stack(rec["W"]),
        "AX": None if rec["AX"][0] is None else torch.stack(rec["AX"]),
        "WU": None if rec["WU"][0] is None else torch.stack(rec["WU"]),
        "FROZ": torch.stack(rec["FROZ"]),
        "valid": (~env.invalid_env_mask).cpu(),
    }


def evaluate(
    rec: dict,
    *,
    f_max: float,
    tau_max: float,
    torsion_max: float,
    bending_max: float,
    control_hz: float,
    tau_max_grid: tuple[float, ...] = (),
) -> dict:
    W, AX, valid = rec["W"], rec["AX"], rec["valid"]
    t_len, n, _ = W.shape
    envelopes = {
        "total": DetachEnvelopeConfig(f_max_n=f_max, tau_max_nm=tau_max),
        "force_only": None,
    }
    for v in tau_max_grid:
        envelopes[f"total_tau{v:g}"] = DetachEnvelopeConfig(f_max_n=f_max, tau_max_nm=float(v))
    if AX is not None:
        envelopes["split"] = DetachEnvelopeConfig(
            f_max_n=f_max, torque_mode="split", torsion_max_nm=torsion_max, bending_max_nm=bending_max
        )
    rules = {}
    for fname, fc in _FILTERS.items():
        Wf = W if fc is None else ema(W, alpha=1.0 - math.exp(-2.0 * math.pi * fc / control_hz))
        for ename, env_cfg in envelopes.items():
            if ename == "force_only":
                idx = (torch.linalg.norm(Wf[..., :3], dim=-1) / f_max) ** 2
            else:
                idx = torch.stack(
                    [detach_index(Wf[t], env_cfg, stem_axis=None if AX is None else AX[t]) for t in range(t_len)]
                )
            for s in _STREAKS:
                first = first_detach(idx, streak=s)
                won = (first >= 0) & valid
                rules[f"{ename}|{fname}|streak{s}"] = {
                    "success": float(won.sum()) / max(1, int(valid.sum())),
                    "steps_median": float(first[won].float().median()) if won.any() else float("nan"),
                }
    live = ~rec["FROZ"] & valid.unsqueeze(0)
    both = live[1:] & live[:-1]
    tau = torch.linalg.norm(W[..., 3:], dim=-1)
    noise = {
        "force_p50": _pct(torch.linalg.norm(W[..., :3], dim=-1)[live], 0.5),
        "force_p99": _pct(torch.linalg.norm(W[..., :3], dim=-1)[live], 0.99),
        "tau_p50": _pct(tau[live], 0.5),
        "tau_p99": _pct(tau[live], 0.99),
        "dtau_p99": _pct((tau[1:] - tau[:-1]).abs()[both], 0.99),
    }
    if AX is not None:
        tors, bend = split_torque(W[..., 3:].reshape(-1, 3), AX.reshape(-1, 3))
        tors, bend = tors.reshape(t_len, n), bend.reshape(t_len, n)
        noise.update(torsion_p99=_pct(tors[live], 0.99), bending_p99=_pct(bend[live], 0.99))
    else:
        noise.update(torsion_p99=float("nan"), bending_p99=float("nan"))
    if rec.get("APPLE") is not None:
        # does the torque move without the apple moving? (readout noise vs real load change)
        d_apple = torch.linalg.norm(rec["APPLE"][1:] - rec["APPLE"][:-1], dim=-1)[both]
        noise.update(apple_step_mm_p50=1000 * _pct(d_apple, 0.5), apple_step_mm_p99=1000 * _pct(d_apple, 0.99))
        quiet = both & (torch.linalg.norm(rec["APPLE"][1:] - rec["APPLE"][:-1], dim=-1) < 1e-4)
        noise.update(dtau_p99_when_apple_still=_pct((tau[1:] - tau[:-1]).abs()[quiet], 0.99), still_steps=int(quiet.sum()))
    if rec["WU"] is not None:
        tu = torch.linalg.norm(rec["WU"][..., 3:], dim=-1)
        noise.update(
            undamped_tau_p99=_pct(tu[live], 0.99),
            undamped_dtau_p99=_pct((tu[1:] - tu[:-1]).abs()[both], 0.99),
        )
    return {"rules": rules, "noise": noise, "recorded_frozen_fraction": float(rec["FROZ"][-1][valid].float().mean())}


def main(argv: list[str] | None = None) -> int:
    from apple_pick_gym.rl.baselines import BASELINES, RandomPolicy
    from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper
    from apple_pick_gym.rl.trainer import build_env

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--policies", nargs="+", default=["zero", "random", "scripted_pull", "scripted_twist_pull"], choices=sorted(BASELINES))
    p.add_argument("--env", choices=("sim", "surrogate"))
    p.add_argument("--num-envs", type=int)
    p.add_argument("--max-episode-steps", type=int)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--f-max", type=float, default=20.0)
    p.add_argument("--tau-max", type=float, default=0.05)
    p.add_argument("--torsion-max", type=float, default=0.05)
    p.add_argument("--bending-max", type=float, default=0.9)
    p.add_argument("--tau-max-grid", type=float, nargs="*", default=[0.05, 0.1, 0.15, 0.2, 0.3],
                   help="extra total-envelope torque limits to replay (N*m)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = TrainConfig.load_json(args.config) if args.config else TrainConfig()
    over = {k: v for k, v in {"kind": args.env, "num_envs": args.num_envs, "max_episode_steps": args.max_episode_steps, "device": args.device}.items() if v is not None}
    # detachment out of reach while recording (safety caps still apply)
    over.update(torque_mode="total", f_max_n=1e9, tau_max_nm=1e9)
    env_cfg = dataclasses.replace(cfg.env, **over)
    wrapper = HarvestSkrlWrapper(build_env(env_cfg, seed=args.seed))
    report = {"env": dataclasses.asdict(env_cfg), "limits": {"f_max": args.f_max, "tau_max": args.tau_max, "torsion_max": args.torsion_max, "bending_max": args.bending_max}, "policies": {}}
    try:
        for name in args.policies:
            pol = RandomPolicy(seed=args.seed) if name == "random" else BASELINES[name]()
            rep = evaluate(
                record(wrapper, pol),
                f_max=args.f_max, tau_max=args.tau_max, torsion_max=args.torsion_max, bending_max=args.bending_max,
                control_hz=60.0,
                tau_max_grid=tuple(args.tau_max_grid),
            )
            report["policies"][name] = rep
            r, nz = rep["rules"], rep["noise"]
            keys = ["total|raw|streak3"] + [f"total_tau{v:g}|raw|streak3" for v in args.tau_max_grid] + [
                "total|ema3hz|streak10", "split|ema3hz|streak10", "force_only|raw|streak3"]
            print(
                f"{name}: " + " ".join(f"{k}={r[k]['success']:.2f}" for k in keys if k in r)
                + f" | tau p99 {nz['tau_p99']:.4f} dtau p99 {nz['dtau_p99']:.4f}"
                + (f" apple step p99 {nz['apple_step_mm_p99']:.3f} mm, dtau p99 with apple still {nz['dtau_p99_when_apple_still']:.4f} ({nz['still_steps']} steps)" if "apple_step_mm_p99" in nz else "")
                + (f" undamped tau p99 {nz['undamped_tau_p99']:.4f} dtau p99 {nz['undamped_dtau_p99']:.4f}" if "undamped_tau_p99" in nz else "")
            )
    finally:
        wrapper.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
