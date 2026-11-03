"""Score the no-oracle "tilt, then pull" scripted policy on a harvest world set.

The default is the baseline: tip the gripper UP by 30 deg about its sideways axis (in place, at the TCP), then pull
straight BACK along the tilted tool axis (in line with the TCP) at a constant speed, with fixed stiff gains
(k_lin 400, k_ang 40, the env's angular cap). It reads only gravity in the tool frame -- no stem or junction pose -- and
never the F/T sensor (:class:`apple_pick_gym.rl.baselines.ScriptedStaticRotatePullPolicy`).

    uv run python -m apple_pick_gym.rl.run_tilt_pull --config apple_pick_gym/rl/configs/<tool-frame train config>.json \\
        --world-set apple_pick_gym/world_sets/snapshots/harvest_worlds_ell_test/shard_00.jsonl \\
        --out runs/scripted/tilt_up_pull_back.json

``--pull down`` pulls along gravity instead (the variant that scores far higher); ``--tip down`` tilts the other way.
Writes the usual eval ``<out>.json`` and ``<out>.per_env.npz`` (one episode per world, success = the env's 20-step detach
streak, safety cap as in the config). Needs a config whose env block uses ``action_frame = "tool"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from apple_pick_gym.rl.baselines import ScriptedStaticRotatePullPolicy

K_ANG_CAP = 40.0  # HarvestActionBounds.k_ang_max: anything above is silently clipped by the env


def _k_ang(text: str) -> float:
    v = float(text)
    if not 0.0 < v <= K_ANG_CAP:
        raise argparse.ArgumentTypeError(f"k_ang must be in (0, {K_ANG_CAP:g}]: the env clips anything above its cap")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="TrainConfig JSON for the env (must use action_frame='tool')")
    p.add_argument("--world-set", required=True, help="shard_00.jsonl of a screened world set")
    p.add_argument("--snapshot", help="default: shard_00_snapshot.npz next to the world set")
    p.add_argument("--out", required=True, help="metrics JSON path (per-env record goes next to it as .per_env.npz)")
    p.add_argument("--num-envs", type=int, default=1000, help="must equal the world-set size when a snapshot is used")
    p.add_argument("--max-episode-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--trace", action="store_true", help="also write <out>.trace.npz: per-step series of the episode (as eval_vic_harvest --trace)")
    p.add_argument("--tilt-deg", type=float, default=30.0)
    p.add_argument("--tip", choices=("up", "down"), default="up", help="up: the grip axis turns up; down: it turns down")
    p.add_argument("--pull", choices=("back", "down", "world-back"), default="back",
                   help="back: along the tilted tool axis (in line with the TCP); down: along gravity; "
                        "world-back: horizontally away from the plant (does not tilt with the gripper)")
    p.add_argument("--rate-mm", type=float, default=1.05, help="pull speed in mm per control step (1.05 = 0.063 m/s at 60 Hz)")
    p.add_argument("--k-lin", type=float, default=400.0)
    p.add_argument("--k-ang", type=_k_ang, default=K_ANG_CAP, help=f"angular stiffness, N*m/rad (env cap {K_ANG_CAP:g})")
    return p


def build_policy(args: argparse.Namespace) -> ScriptedStaticRotatePullPolicy:
    return ScriptedStaticRotatePullPolicy(
        rotate_deg=args.tilt_deg,
        axis="sideways",
        rotate_sign=-1.0 if args.tip == "up" else 1.0,
        pull_dir={"back": "tool", "down": "down", "world-back": "world_back"}[args.pull],
        rate_m_per_step=args.rate_mm / 1000.0,
        k_lin=args.k_lin,
        k_ang=args.k_ang,
    )


def snapshot_for(world_set: str, snapshot: str | None) -> str:
    return snapshot if snapshot else str(Path(world_set).with_name("shard_00_snapshot.npz"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    policy = build_policy(a)

    from apple_pick_gym.rl import eval_vic_harvest as ev  # heavy imports (torch, sim) only when actually running

    ev_args = ev.build_parser().parse_args([
        "--baseline", "scripted_static_rotate_pull", "--config", a.config, "--world-set", a.world_set,
        "--snapshot", snapshot_for(a.world_set, a.snapshot), "--num-envs", str(a.num_envs),
        "--max-episode-steps", str(a.max_episode_steps), "--episodes", "1", "--seed", str(a.seed), "--out", a.out,
    ])
    cfg, wrapper, _, _ = ev.build_eval(ev_args)
    trace: dict | None = {} if a.trace else None
    try:
        metrics = ev.evaluate(wrapper, policy, episodes=1, trace=trace)
    finally:
        wrapper.close()
    per_env = metrics.pop("per_env")
    metrics["policy"] = (
        f"tilt {a.tip} {a.tilt_deg:g} deg, pull {a.pull}, k_lin {a.k_lin:g}, k_ang {a.k_ang:g}, {a.rate_mm:g} mm/step (no oracle)"
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    if trace:
        np.savez_compressed(out.with_suffix(".trace.npz"), **{k: np.stack(v) for k, v in trace.items()})
    if per_env:
        np.savez_compressed(
            out.with_suffix(".per_env.npz"),
            world_set=np.array(str(a.world_set)),
            **{f"ep{i}/{k}": v for i, e in enumerate(per_env) for k, v in e.items()},
        )
    print(
        f"{metrics['policy']}: success {metrics['success_rate']:.3f}, safety {metrics['safety_rate']:.3f}, "
        f"invalid {metrics['invalid_fraction']:.3f} -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
