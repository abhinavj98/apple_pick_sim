"""eval_vic_harvest per-step trace (--trace): same keys and layout as the torque-reward session's traces."""

from __future__ import annotations

import numpy as np
import torch

from apple_pick_gym.rl import eval_vic_harvest as ev


def _info(t: float, n: int = 3):
    return {
        "stem_tension": torch.full((n,), t),
        "detach_index": torch.full((n,), 2 * t),
        "detach_force": torch.full((n,), 3 * t),
        "theta_b_deg": torch.full((n,), 4 * t),
        "tcp_advance_m": torch.full((n,), 5 * t),
        "target_junction_wrench": torch.full((n, 6), 6 * t),
        "ft_wrist": torch.full((n, 6), 7 * t),
        "episode": {"frozen": torch.tensor([False, True, False]), "success_this_step": torch.tensor([False, False, True])},
    }


def test_trace_keys_match_the_shared_format():
    assert ev._TRACE_KEYS == ("stem_tension", "detach_index", "detach_force", "theta_b_deg", "tcp_advance_m", "target_junction_wrench", "ft_wrist")


def test_record_trace_appends_cpu_numpy_series_and_the_freeze_flags():
    trace: dict = {}
    for t in (1.0, 2.0):
        ev._record_trace(trace, _info(t))
    out = {k: np.stack(v) for k, v in trace.items()}
    assert out["stem_tension"].shape == (2, 3) and out["target_junction_wrench"].shape == (2, 3, 6)
    np.testing.assert_allclose(out["ft_wrist"][1], 14.0)
    assert out["frozen"].dtype == bool and out["frozen"][0].tolist() == [False, True, False]
    assert out["success_this_step"][1].tolist() == [False, False, True]


def test_record_trace_skips_keys_the_env_did_not_publish():
    info = _info(1.0)
    del info["tcp_advance_m"]
    trace: dict = {}
    ev._record_trace(trace, info)
    assert "tcp_advance_m" not in trace and "stem_tension" in trace
