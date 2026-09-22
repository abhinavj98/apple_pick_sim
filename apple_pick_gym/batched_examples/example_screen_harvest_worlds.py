"""Screen harvest worlds for stability; build the curated world set RL trains on.

One batch per process (never rebuild envs in-process -- see
``docs/in-process-rebuild-heap-corruption.md``).

**Pass 1 -- sample** fresh candidate worlds from the default ranges fixture and
screen them (every candidate is written, with its pass/fail record)::

    uv run python apple_pick_gym/batched_examples/example_screen_harvest_worlds.py \\
        sample --num-envs 128 --seed 0 --out tmp/harvest_worlds/pass1.jsonl

**Pass 2 -- rescreen** the pass-1 survivors in a different batch composition
(stability was observed to vary between builds with identical parameters); a
world is accepted only if it passes both::

    uv run python apple_pick_gym/batched_examples/example_screen_harvest_worlds.py \\
        rescreen --from-set tmp/harvest_worlds/pass1.jsonl --shuffle-seed 0 \\
        --chunk-size 128 --chunk-index 0 --out tmp/harvest_worlds/pass2.jsonl

The accepted set is ``load_world_set(pass2, passed_only=True)``.

**Report** coverage of any screened files (candidate vs accepted range and
rejection rate per tercile of every randomized knob -- catches screening that
quietly biases the set toward easy plants)::

    uv run python apple_pick_gym/batched_examples/example_screen_harvest_worlds.py \\
        report tmp/harvest_worlds/pass1_s*.jsonl --out-md tmp/harvest_worlds/pass1_coverage.md
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

_CFG_FIELDS = (
    "hold_steps",
    "pull_rest_steps",
    "pull_ramp_steps",
    "pull_hold_steps",
    "pull_settle_steps",
    "num_pull_episodes",
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("sets", nargs="+", help="Screened JSONL files (a world's latest record decides).")
    rp.add_argument("--out-md", default=None)
    for name in ("sample", "rescreen"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", required=True, help="JSONL to append screened worlds to.")
        sp.add_argument("--device", default=None)
        from apple_pick_gym.batched_envs.harvest_world_screening import ScreeningConfig

        defaults = ScreeningConfig()
        for field in _CFG_FIELDS:
            sp.add_argument(f"--{field.replace('_', '-')}", type=int, default=getattr(defaults, field))
        if name == "sample":
            sp.add_argument("--num-envs", type=int, required=True)
            sp.add_argument("--seed", type=int, required=True, help="Topology + DR seed; world ids are s<seed>_e<i>.")
        else:
            sp.add_argument("--from-set", required=True, help="Pass-1 JSONL; only its passed worlds are rescreened.")
            sp.add_argument("--shuffle-seed", type=int, required=True)
            sp.add_argument("--chunk-size", type=int, required=True)
            sp.add_argument("--chunk-index", type=int, required=True)
    return p


def _screen(env: Any, cfg: Any) -> tuple[np.ndarray, list[list[str]], dict[str, np.ndarray]]:
    from apple_pick_gym.batched_envs.harvest_world_screening import evaluate_screening, run_screening

    metrics = run_screening(env, cfg)
    passed, reasons = evaluate_screening(metrics, cfg)
    return passed, reasons, metrics


def _record(i: int, passed: np.ndarray, reasons: list[list[str]], metrics: dict[str, np.ndarray]) -> dict[str, Any]:
    rec: dict[str, Any] = {"passed": bool(passed[i]), "reasons": reasons[i]}
    for k, v in metrics.items():
        rec[k] = bool(v[i]) if v.dtype == bool else float(v[i])
    return rec


def _report(paths: Sequence[str], out_md: str | None) -> str:
    from pathlib import Path

    from apple_pick_gym.batched_envs.world_set import load_world_set, world_set_coverage

    by_id = {}
    for path in paths:
        for spec in load_world_set(path):
            by_id[spec.world_id] = spec
    specs = list(by_id.values())
    rows = world_set_coverage(specs)
    n_acc = sum(s.passed for s in specs)
    lines = [
        f"{n_acc}/{len(specs)} worlds accepted",
        "",
        "| knob | candidate range | accepted range | reject rate low / mid / high tercile |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        rates = " / ".join(f"{x:.0%}" for x in r["reject_rate_by_tercile"])
        lines.append(
            f"| {r['knob']} | {r['candidate_min']:.4g} .. {r['candidate_max']:.4g} "
            f"| {r['accepted_min']:.4g} .. {r['accepted_max']:.4g} | {rates} |"
        )
    text = "\n".join(lines)
    print(text)
    if out_md:
        Path(out_md).write_text(text + "\n")
    return text


def main(argv: Sequence[str] | None = None) -> None:
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.harvest_world_screening import (
        ScreeningConfig,
        screening_episode_config,
    )
    from apple_pick_gym.batched_envs.world_set import (
        load_world_set,
        save_world_set,
        world_specs_to_env_kwargs,
    )

    args = make_parser().parse_args(argv)
    if args.mode == "report":
        _report(args.sets, args.out_md)
        return
    cfg = ScreeningConfig(**{f: getattr(args, f) for f in _CFG_FIELDS})
    common = dict(device=args.device, episode_config=screening_episode_config(cfg), max_episode_steps=10**6)
    t0 = time.time()

    if args.mode == "sample":
        env = ApplePickVicHarvestEnv(
            num_envs=args.num_envs, topology_seed=args.seed, dr_seed=args.seed, **common
        )
        tag, prior = "pass1", None
        try:
            specs = env.export_world_specs(prefix=f"s{args.seed}")
            passed, reasons, metrics = _screen(env, cfg)
        finally:
            env.close()
    else:
        pool = load_world_set(args.from_set, passed_only=True)
        order = np.random.default_rng(args.shuffle_seed).permutation(len(pool))
        chunk = [pool[j] for j in order[args.chunk_index * args.chunk_size : (args.chunk_index + 1) * args.chunk_size]]
        if not chunk:
            print(f"chunk {args.chunk_index} is empty ({len(pool)} passed worlds)")
            return
        env = ApplePickVicHarvestEnv(**world_specs_to_env_kwargs(chunk), **common)
        tag, prior = "pass2", chunk
        try:
            specs = chunk
            passed, reasons, metrics = _screen(env, cfg)
        finally:
            env.close()

    out = []
    for i, spec in enumerate(specs):
        rec = _record(i, passed, reasons, metrics)
        screening = dict(spec.screening) if prior is not None else {}
        screening[tag] = rec
        screening["passed"] = all(screening[k]["passed"] for k in ("pass1", "pass2") if k in screening)
        out.append(dataclasses.replace(spec, screening=screening))
    save_world_set(args.out, out, append=True)

    n_pass = int(sum(s.screening["passed"] for s in out))
    print(f"{args.mode}: {n_pass}/{len(out)} passed ({time.time() - t0:.0f} s) -> {args.out}")
    for s in out:
        if not s.screening[tag]["passed"]:
            print(f"  REJECT {s.world_id}: {', '.join(s.screening[tag]['reasons'])}")


if __name__ == "__main__":
    main()
