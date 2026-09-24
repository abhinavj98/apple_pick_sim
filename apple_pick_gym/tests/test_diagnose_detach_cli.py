"""diagnose_detach: how envs cross the detach envelope (force vs torque, spikes) per policy."""

from __future__ import annotations

import json

from apple_pick_gym.rl import diagnose_detach as dd

_ENV = ["--env", "surrogate", "--num-envs", "16", "--max-episode-steps", "120", "--device", "cpu"]


def test_reports_force_torque_breakdown_per_policy(tmp_path):
    out = tmp_path / "diag.json"
    assert dd.main([*_ENV, "--policies", "zero", "scripted_pull", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert set(rep["policies"]) == {"zero", "scripted_pull"}
    z, p = rep["policies"]["zero"], rep["policies"]["scripted_pull"]
    for key in ("success_rate", "rest_tau_median", "live_tau_p99", "dtau_p99", "transient_crossings",
                "live_torsion_p99", "live_bending_p99", "torsion_at_detach_median", "bending_at_detach_median"):
        assert key in z, key
    assert z["success_rate"] == 0.0
    assert p["success_rate"] > 0.5
    # the scripted pull detaches through force on the surrogate
    assert p["force_share_at_detach_median"] > p["torque_share_at_detach_median"]


def test_envelope_variant_flags_reach_the_env(tmp_path):
    out = tmp_path / "split.json"
    assert dd.main([*_ENV, "--policies", "zero", "--torque-mode", "split", "--bending-max", "0.5", "--out", str(out)]) == 0
    env = json.loads(out.read_text())["env"]
    assert env["torque_mode"] == "split" and env["bending_max_nm"] == 0.5
