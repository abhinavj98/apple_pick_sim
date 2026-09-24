"""[D2] The Task 11 exit gate for a learned harvest policy, from ``eval_vic_harvest`` metrics JSONs.

A straight scripted pull detaches by loading the whole serial chain (measured on GPU: 19 N at the
spur-stem junction, 45 N peak collateral), so "beat scripted pull on success" alone would reward
that. The gate therefore requires *all* of:

- ``success``: success rate >= the scripted pull's;
- ``safety``: safety-violation rate <= the scripted pull's;
- ``collateral``: peak collateral load <= ``collateral_ratio`` (default 0.5) x the scripted pull's;
- ``beats_random``: success rate >= the random policy's (guards against an envelope a random
  policy already trips, as the total-moment envelope did);
- ``collateral_vs_random`` [D2a]: peak collateral strictly < the random policy's (so random itself fails). On GPU random peaked at
  15.2 N vs the scripted pull's 45 N, so random alone passed the 0.5x clause; a learned policy must
  not load the tree more than random flailing does.

The random clauses apply only when a random baseline is given.

[D6] The collateral clauses compare ``peak_collateral_n_success_mean`` (load per *successful* pick)
when every metrics JSON has it, else ``peak_collateral_n_mean``. The all-episode mean is diluted by
failed episodes that never pulled, which flatters a policy that often fails (random). A baseline with
no successful pick sets no collateral bar; a policy with none fails its collateral clauses.

Evaluate every policy on the same held-out snapshot and seed.

    uv run python -m apple_pick_gym.rl.gate --policy eval/policy.json --scripted-pull eval/pull.json \\
        --random eval/random.json --out eval/gate.json
"""

from __future__ import annotations

import argparse
import math
import json
import sys
from pathlib import Path


_OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t, "<": lambda v, t: v < t}
_COLL_ALL, _COLL_SUCCESS = "peak_collateral_n_mean", "peak_collateral_n_success_mean"


def _num(x) -> float:
    return float("nan") if x is None else float(x)


def _passed(v: float, op: str, t: float) -> bool:
    # a NaN threshold is a baseline with no successful pick: it sets no bar. A NaN value is a
    # policy with no successful pick: it fails.
    if math.isnan(v):
        return False
    if math.isnan(t):
        return True
    return bool(_OPS[op](v, t))


def evaluate_gate(policy: dict, *, scripted_pull: dict, random: dict | None = None, collateral_ratio: float = 0.5) -> dict:
    # [D6] compare load on the tree per successful pick when every metrics JSON reports it
    reports = [policy, scripted_pull] + ([random] if random is not None else [])
    coll = _COLL_SUCCESS if all(_COLL_SUCCESS in m for m in reports) else _COLL_ALL
    crit = {
        "success": (policy["success_rate"], ">=", scripted_pull["success_rate"]),
        "safety": (policy["safety_rate"], "<=", scripted_pull["safety_rate"]),
        "collateral": (_num(policy[coll]), "<=", collateral_ratio * _num(scripted_pull[coll])),
    }
    if random is not None:
        crit["beats_random"] = (policy["success_rate"], ">=", random["success_rate"])
        crit["collateral_vs_random"] = (_num(policy[coll]), "<", _num(random[coll]))
    out = {
        name: {"value": v, "op": op, "threshold": t, "passed": _passed(float(v), op, float(t))}
        for name, (v, op, t) in crit.items()
    }
    return {
        "passed": all(c["passed"] for c in out.values()),
        "criteria": out,
        "collateral_ratio": collateral_ratio,
        "collateral_metric": coll,
    }


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
        print(f"{name:20s} {'PASS' if c['passed'] else 'FAIL'}  {c['value']:.3f} {c['op']} {c['threshold']:.3f}")
    print("GATE", "PASSED" if res["passed"] else "FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
