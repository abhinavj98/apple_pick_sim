"""How do envs cross the detach envelope? Force vs torque share, spikes, rest load -- per policy.

Answers "why does policy X detach?" (e.g. a random policy reaching high success): for each
policy it rolls one synchronized episode and reports

- success / safety rate over valid envs and steps to detach;
- the rest torque at reset (how close the junction already is to ``tau_max``);
- at the detach step: |F|, |tau| at the joint anchor and at the child COM, and each half's
  share of the envelope ``(F/F_max)^2`` vs ``(tau/tau_max)^2``;
- over live steps: |F|, |tau|, detach-index percentiles, the step-to-step torque change
  (spiky if its p99 approaches ``tau_max``) and the lengths of runs with index >= 1
  (``transient_crossings``: the index went back under 1 while still live -- spikes; a spike
  lasting ``success_streak_steps`` counts as a detach).

Example (GPU)::

    uv run python -m apple_pick_gym.rl.diagnose_detach --config apple_pick_gym/rl/configs/sim_wiring_gpu.json \\
        --policies zero random scripted_pull --out runs/diag_detach.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

from apple_pick_gym.rl.config import TrainConfig


def _q(x: torch.Tensor, q: float) -> float:
    return float(x.float().quantile(q)) if x.numel() else float("nan")


def diagnose(wrapper, policy) -> dict:
    env = wrapper._env
    det = env._reward_cfg.detach
    obs, info0 = wrapper.reset()
    state = wrapper.state()
    policy.reset(wrapper)
    n = wrapper.num_envs
    valid = (~env.invalid_env_mask).cpu()
    tau0 = torch.linalg.norm(info0["target_junction_wrench"][:, 3:], dim=-1).cpu()
    rec = {k: [] for k in ("F", "TA", "TC", "IDX", "TERM", "SUCC", "SAFE", "FROZ")}
    for _ in range(env.max_episode_steps - 1):
        obs, _r, term, _trunc, info = wrapper.step(policy.act(wrapper, obs, state))
        state = wrapper.state()
        tw, raw, ep = info["target_junction_wrench"], info["target_junction_force"], info["episode"]
        rec["F"].append(torch.linalg.norm(tw[:, :3], dim=-1))
        rec["TA"].append(torch.linalg.norm(tw[:, 3:], dim=-1))
        rec["TC"].append(torch.linalg.norm(raw[:, 3:], dim=-1))
        rec["IDX"].append(info["detach_index"])
        rec["TERM"].append(term.flatten())
        rec["SUCC"].append(ep["success_achieved"])
        rec["SAFE"].append(ep["safety_junction"] | ep["safety_wrist"])
        rec["FROZ"].append(ep["frozen"])
    R = {k: torch.stack(v).cpu() for k, v in rec.items()}  # (T, N)
    ar = torch.arange(n)
    ended = R["TERM"].any(0)
    first = torch.where(ended, R["TERM"].float().argmax(0), torch.full((n,), -1))
    k = first.clamp(min=0)
    won = ended & R["SUCC"][k, ar] & ~R["SAFE"][k, ar] & valid
    failed = ended & R["SAFE"][k, ar] & valid
    fs = (R["F"][k, ar] / det.f_max_n) ** 2
    ts = (R["TA"][k, ar] / det.tau_max_nm) ** 2
    live = ~R["FROZ"] & valid
    both = live[1:] & live[:-1]
    dtau = (R["TA"][1:] - R["TA"][:-1]).abs()[both]
    over = (R["IDX"] >= 1.0) & live
    runs, transient = [], 0
    T = over.shape[0]
    for e in range(n):
        if not valid[e]:
            continue
        t = 0
        while t < T:
            if over[t, e]:
                t0 = t
                while t < T and over[t, e]:
                    t += 1
                runs.append(t - t0)
                # transient: the index fell back under 1 while the env was still live (a spike)
                if t < T and live[t, e]:
                    transient += 1
            else:
                t += 1
    nv = max(1, int(valid.sum()))
    return {
        "valid_envs": int(valid.sum()),
        "success_rate": float(won.sum()) / nv,
        "safety_rate": float(failed.sum()) / nv,
        "steps_to_detach_median": float(first[won].float().median()) if won.any() else float("nan"),
        "rest_tau_median": _q(tau0[valid], 0.5),
        "rest_tau_p90": _q(tau0[valid], 0.9),
        "force_at_detach_median": _q(R["F"][k, ar][won], 0.5),
        "tau_anchor_at_detach_median": _q(R["TA"][k, ar][won], 0.5),
        "tau_com_at_detach_median": _q(R["TC"][k, ar][won], 0.5),
        "force_share_at_detach_median": _q(fs[won], 0.5),
        "torque_share_at_detach_median": _q(ts[won], 0.5),
        "torque_dominated_fraction": float((ts > fs)[won].float().mean()) if won.any() else float("nan"),
        "live_force_median": _q(R["F"][live], 0.5),
        "live_force_p99": _q(R["F"][live], 0.99),
        "live_tau_median": _q(R["TA"][live], 0.5),
        "live_tau_p99": _q(R["TA"][live], 0.99),
        "live_index_p99": _q(R["IDX"][live], 0.99),
        "dtau_median": _q(dtau, 0.5),
        "dtau_p99": _q(dtau, 0.99),
        "runs_over_envelope": len(runs),
        "run_length_median": float(np.median(runs)) if runs else 0.0,
        "transient_crossings": transient,
        "transient_crossings_per_env": transient / nv,
        "tau_max_nm": det.tau_max_nm,
        "f_max_n": det.f_max_n,
    }


def main(argv: list[str] | None = None) -> int:
    from apple_pick_gym.rl.baselines import BASELINES, RandomPolicy
    from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper
    from apple_pick_gym.rl.trainer import build_env

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="TrainConfig JSON for the env (default: TrainConfig defaults)")
    p.add_argument("--policies", nargs="+", default=["zero", "random", "scripted_pull"], choices=sorted(BASELINES))
    p.add_argument("--env", choices=("sim", "surrogate"))
    p.add_argument("--num-envs", type=int)
    p.add_argument("--max-episode-steps", type=int)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = TrainConfig.load_json(args.config) if args.config else TrainConfig()
    over = {k: v for k, v in {"kind": args.env, "num_envs": args.num_envs, "max_episode_steps": args.max_episode_steps, "device": args.device}.items() if v is not None}
    cfg = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, **over))
    wrapper = HarvestSkrlWrapper(build_env(cfg.env, seed=args.seed))
    report = {"env": dataclasses.asdict(cfg.env), "policies": {}}
    try:
        for name in args.policies:
            pol = RandomPolicy(seed=args.seed) if name == "random" else BASELINES[name]()
            report["policies"][name] = diagnose(wrapper, pol)
            r = report["policies"][name]
            print(
                f"{name}: success {r['success_rate']:.3f} safety {r['safety_rate']:.3f} | rest tau med {r['rest_tau_median']:.4f} | "
                f"at detach F {r['force_at_detach_median']:.2f} N tau {r['tau_anchor_at_detach_median']:.4f} "
                f"(share F {r['force_share_at_detach_median']:.2f} / tau {r['torque_share_at_detach_median']:.2f}) | "
                f"dtau p99 {r['dtau_p99']:.4f} | crossings {r['runs_over_envelope']} (transient {r['transient_crossings']})"
            )
    finally:
        wrapper.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
