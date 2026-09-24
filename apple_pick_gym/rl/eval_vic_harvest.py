"""Evaluate a trained checkpoint or a baseline on a harvest env for K episodes.

The learned policy runs deterministically (the Gaussian mean) with its LSTM state carried
through each episode and reset at episode ends. Baselines: ``zero``, ``random``,
``scripted_pull``, ``scripted_twist_pull`` (:mod:`apple_pick_gym.rl.baselines`).

Examples::

    uv run python -m apple_pick_gym.rl.eval_vic_harvest --checkpoint runs/vic_harvest/exp1/checkpoints/ckpt_000100000 \\
        --episodes 3 --out runs/vic_harvest/exp1/eval_train.json
    uv run python -m apple_pick_gym.rl.eval_vic_harvest --baseline scripted_pull --episodes 3 --out baselines/pull.json

Writes ``metrics.json``: success / safety rate over valid envs, return, steps to success,
peak detach index / target / collateral / wrist force, reward-term sums, impedance usage
(per episode and averaged). Env settings come from the checkpoint's config (or the
defaults for a baseline) and can be overridden with the same flags as training.
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


class RecurrentPolicyRunner:
    """Deterministic rollout of a skrl agent's LSTM policy with its own hidden state."""

    name = "checkpoint"

    def __init__(self, agent) -> None:
        self.agent = agent
        agent.enable_models_training_mode(False)
        spec = agent.policy.get_specification()["rnn"]
        self._sizes = spec["sizes"]
        self._rnn = None

    def reset(self, wrapper) -> None:
        self._rnn = [torch.zeros(s, device=wrapper.device) for s in self._sizes]

    @torch.no_grad()
    def act(self, wrapper, obs, state) -> torch.Tensor:
        a = self.agent
        inputs = {
            "observations": a._observation_preprocessor(obs),
            "states": a._state_preprocessor(state),
            "rnn": self._rnn,
        }
        _, out = a.policy.act(inputs, role="policy")
        self._rnn = out["rnn"]
        return out["mean_actions"].clamp(-1.0, 1.0)

    def on_step(self, done: torch.Tensor) -> None:
        idx = done.flatten().nonzero(as_tuple=True)[0]
        if idx.numel():
            for h in self._rnn:
                h[:, idx] = 0.0


def evaluate(wrapper, policy, *, episodes: int) -> dict:
    """Roll ``policy`` for ``episodes`` full (synchronized) episodes; aggregate the wrapper's stats."""
    obs, _ = wrapper.reset()
    state = wrapper.state()
    policy.reset(wrapper)
    per_episode = []
    while len(per_episode) < episodes:
        actions = policy.act(wrapper, obs, state)
        obs, _r, terminated, truncated, info = wrapper.step(actions)
        state = wrapper.state()
        if hasattr(policy, "on_step"):
            policy.on_step(terminated | truncated)
        log = info.get("log", {})
        if "Episode / success rate" in log:
            per_episode.append({k: float(v) for k, v in log.items() if k.startswith("Episode /")})
            if not hasattr(policy, "on_step"):  # the recurrent runner resets its state in on_step
                policy.reset(wrapper)
    keys = {
        "success_rate": "Episode / success rate",
        "safety_rate": "Episode / safety rate",
        "invalid_fraction": "Episode / invalid fraction",
        "return_mean": "Episode / return (mean)",
        "steps_to_success_mean": "Episode / steps to success (mean)",
        "peak_detach_index_mean": "Episode / peak detach index (mean)",
        "peak_target_force_n_mean": "Episode / peak target force N (mean)",
        "peak_collateral_n_mean": "Episode / peak collateral N (mean)",
        "peak_collateral_n_success_mean": "Episode / peak collateral N, successful (mean)",
        "peak_wrist_force_n_mean": "Episode / peak wrist force N (mean)",
        "reward_progress_sum": "Episode / reward progress (sum)",
        "reward_pullout_sum": "Episode / reward pullout (sum)",
        "reward_wrist_sum": "Episode / reward wrist (sum)",
        "reward_collateral_sum": "Episode / reward collateral (sum)",
        "reward_slack_sum": "Episode / reward slack (sum)",
        "reward_terminal_sum": "Episode / reward terminal (sum)",
        "k_lin_mean": "Episode / K_lin used (mean)",
        "k_ang_mean": "Episode / K_ang used (mean)",
        "zeta_mean": "Episode / zeta used (mean)",
        "peak_tcp_speed_mps_mean": "Episode / peak TCP speed m/s (mean)",
        "mean_tcp_speed_mps_mean": "Episode / mean TCP speed m/s (mean)",
    }
    out = {k: float(np.mean([e[v] for e in per_episode])) for k, v in keys.items()}
    # [D6] success-conditioned collateral: weight each batch episode by its successful valid envs
    won = [(e["Episode / success rate"] * (1.0 - e["Episode / invalid fraction"]), e[keys["peak_collateral_n_success_mean"]]) for e in per_episode]
    won = [(w, c) for w, c in won if w > 0.0 and np.isfinite(c)]
    out["peak_collateral_n_success_mean"] = (
        float(sum(w * c for w, c in won) / sum(w for w, _ in won)) if won else float("nan")
    )
    for k in per_episode[0]:
        if k.startswith("Episode / peak force ") and k.endswith(" N (mean)"):
            name = k[len("Episode / peak force ") : -len(" N (mean)")]
            out[f"peak_force_{name}_n_mean"] = float(np.mean([e[k] for e in per_episode]))
    for label, key in (
        ("force N", "detach_force_n_mean"),
        ("torque N*m", "detach_torque_nm_mean"),
        ("torsion N*m", "detach_torsion_nm_mean"),
        ("bending N*m", "detach_bending_nm_mean"),
        ("force share", "detach_force_share_mean"),
        ("torque share", "detach_torque_share_mean"),
    ):
        vals = [e[f"Episode / detach {label} (mean)"] for e in per_episode if np.isfinite(e.get(f"Episode / detach {label} (mean)", np.nan))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    for cap in ("target force", "target torque", "wrist force", "wrist torque"):
        out[f"safety_{cap.replace(' ', '_')}_frac"] = float(np.mean([e.get(f"Episode / safety {cap} (frac)", 0.0) for e in per_episode]))
    out["peak_wrist_torque_nm_mean"] = float(np.mean([e["Episode / peak wrist torque N*m (mean)"] for e in per_episode]))
    out["peak_target_torque_nm_mean"] = float(np.mean([e["Episode / peak target torque N*m (mean)"] for e in per_episode]))
    out["episodes"] = per_episode
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint directory (ckpt_<timestep>)")
    src.add_argument("--baseline", choices=("zero", "random", "scripted_pull", "scripted_twist_pull"))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", required=True, help="metrics JSON path")
    p.add_argument("--config", help="TrainConfig JSON for a baseline's env (default: TrainConfig defaults)")
    p.add_argument("--env", choices=("sim", "surrogate"))
    p.add_argument("--world-set")
    p.add_argument("--snapshot", help="'none' to hold-settle")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--max-episode-steps", type=int)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=12345, help="eval DR seed (differs from training by default)")
    return p


def main(argv: list[str] | None = None) -> int:
    from apple_pick_gym.rl.baselines import BASELINES, RandomPolicy
    from apple_pick_gym.rl.checkpoint import load_checkpoint
    from apple_pick_gym.rl.skrl_wrapper import HarvestSkrlWrapper
    from apple_pick_gym.rl.trainer import build_agent, build_env

    args = build_parser().parse_args(argv)
    if args.checkpoint:
        meta = json.loads((Path(args.checkpoint) / "meta.json").read_text())
        cfg = TrainConfig.from_dict(meta["config"])
    else:
        cfg = TrainConfig.load_json(args.config) if args.config else TrainConfig()
    env_over = {
        "kind": args.env,
        "world_set": args.world_set,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "device": args.device,
    }
    env_over = {k: v for k, v in env_over.items() if v is not None}
    if args.snapshot is not None:
        env_over["snapshot"] = None if args.snapshot == "none" else args.snapshot
    cfg = dataclasses.replace(cfg, env=dataclasses.replace(cfg.env, **env_over), seed=args.seed)

    torch.manual_seed(args.seed)
    wrapper = HarvestSkrlWrapper(build_env(cfg.env, seed=args.seed))
    try:
        if args.checkpoint:
            agent = build_agent(wrapper, cfg, run_dir=Path(args.out).parent / "_eval_agent")
            load_checkpoint(args.checkpoint, agent, wrapper, cfg)
            policy = RecurrentPolicyRunner(agent)
            label = f"checkpoint:{args.checkpoint}"
        else:
            policy = RandomPolicy(seed=args.seed) if args.baseline == "random" else BASELINES[args.baseline]()
            label = f"baseline:{args.baseline}"
        metrics = evaluate(wrapper, policy, episodes=args.episodes)
    finally:
        wrapper.close()
    metrics["policy"] = label
    metrics["env"] = dataclasses.asdict(cfg.env)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"{label}: success {metrics['success_rate']:.3f}, safety {metrics['safety_rate']:.3f}, "
        f"return {metrics['return_mean']:.2f}, peak collateral {metrics['peak_collateral_n_mean']:.2f} N -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
