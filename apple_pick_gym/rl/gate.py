"""[D2] The Task 11 exit gate for a learned harvest policy, from ``eval_vic_harvest`` metrics JSONs.

A straight scripted pull detaches by loading the whole serial chain (measured on GPU: 19 N at the
spur-stem junction, 45 N peak collateral), so "beat scripted pull on success" alone would reward
that. The gate therefore requires *all* of:

- ``success``: success rate >= the scripted pull's;
- ``safety``: safety-violation rate <= the scripted pull's;
- ``collateral``: peak collateral load <= ``collateral_ratio`` (default 0.5) x the scripted pull's;
- ``beats_random``: success rate >= the random policy's (guards against an envelope a random
  policy already trips, as the total-moment envelope did).

Evaluate every policy on the same held-out snapshot and seed.

    uv run python -m apple_pick_gym.rl.gate --policy eval/policy.json --scripted-pull eval/pull.json \\
        --random eval/random.json --out eval/gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def evaluate_gate(policy: dict, *, scripted_pull: dict, random: dict | None = None, collateral_ratio: float = 0.5) -> dict:
    crit = {
        "success": (policy["success_rate"], ">=", scripted_pull["success_rate"]),
        "safety": (policy["safety_rate"], "<=", scripted_pull["safety_rate"]),
        "collateral": (policy["peak_collateral_n_mean"], "<=", collateral_ratio * scripted_pull["peak_collateral_n_mean"]),
    }
    if random is not None:
        crit["beats_random"] = (policy["success_rate"], ">=", random["success_rate"])
    out = {
        name: {"value": v, "op": op, "threshold": t, "passed": bool(v >= t if op == ">=" else v <= t)}
        for name, (v, op, t) in crit.items()
    }
    return {"passed": all(c["passed"] for c in out.values()), "criteria": out, "collateral_ratio": collateral_ratio}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--scripted-pull", required=True)
    p.add_argument("--random")
    p.add_argument("--collateral-ratio", type=float, default=0.5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    load = lambda f: json.loads(Path(f).read_text())
    res = evaluate_gate(
        load(a.policy), scripted_pull=load(a.scripted_pull), random=load(a.random) if a.random else None,
        collateral_ratio=a.collateral_ratio,
    )
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    for name, c in res["criteria"].items():
        print(f"{name:13s} {'PASS' if c['passed'] else 'FAIL'}  {c['value']:.3f} {c['op']} {c['threshold']:.3f}")
    print("GATE", "PASSED" if res["passed"] else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
