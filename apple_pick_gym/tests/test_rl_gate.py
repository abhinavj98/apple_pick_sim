"""[D2] Task 11 gate: success, safety and collateral vs the scripted-pull (and random) baselines."""

from __future__ import annotations

import json

from apple_pick_gym.rl import gate


def _m(success, safety, coll):
    return {"success_rate": success, "safety_rate": safety, "peak_collateral_n_mean": coll}


def test_passes_only_when_every_criterion_holds():
    pull, rnd = _m(0.9, 0.02, 45.0), _m(0.5, 0.0, 15.0)
    ok = gate.evaluate_gate(_m(0.95, 0.01, 10.0), scripted_pull=pull, random=rnd)
    assert ok["passed"] and all(c["passed"] for c in ok["criteria"].values())
    lots_of_collateral = gate.evaluate_gate(_m(0.95, 0.01, 30.0), scripted_pull=pull, random=rnd)
    assert not lots_of_collateral["passed"] and not lots_of_collateral["criteria"]["collateral"]["passed"]
    unsafe = gate.evaluate_gate(_m(0.95, 0.05, 10.0), scripted_pull=pull, random=rnd)
    assert not unsafe["criteria"]["safety"]["passed"]
    worse_than_random = gate.evaluate_gate(_m(0.4, 0.0, 5.0), scripted_pull=_m(0.3, 0.0, 45.0), random=rnd)
    assert not worse_than_random["criteria"]["beats_random"]["passed"]


def test_cli_reads_eval_jsons(tmp_path):
    for name, m in {"p": _m(0.95, 0.0, 10.0), "s": _m(0.9, 0.0, 45.0), "r": _m(0.2, 0.0, 15.0)}.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(m))
    out = tmp_path / "gate.json"
    rc = gate.main(["--policy", str(tmp_path / "p.json"), "--scripted-pull", str(tmp_path / "s.json"),
                    "--random", str(tmp_path / "r.json"), "--out", str(out)])
    assert rc == 0 and json.loads(out.read_text())["passed"]
