"""Smoke tests for settle→weld stability sweep diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from apple_pick_sim.tests.conftest import RANGES_FIXTURE, requires_fr3


def _load_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "diagnostics"
        / "sweep_settle_weld_stability.py"
    )
    name = "apple_pick_sim.diagnostics.sweep_settle_weld_stability"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@requires_fr3
@pytest.mark.slow
def test_settle_weld_sweep_smoke_fr3_cpu():
    mod = _load_module()
    config = mod.SettleWeldSweepConfig(
        seed=42,
        num_envs=2,
        settle_substeps_list=(5, 10),
        post_weld_frames=1,
        sim_substeps=15,
        ranges_path=RANGES_FIXTURE,
        device="cpu",
        robot="fr3",
        settle_gravity_ramp=False,
        verbose=False,
    )
    results = mod.run_settle_weld_sweep(config)
    assert len(results) == 2
    for trial in results:
        assert trial.settle_substeps in (5, 10)
        assert len(trial.post_settle_reports) == 2
        assert len(trial.post_weld_reports) == 2
        assert len(trial.post_hold_reports) == 2
        assert 0.0 <= trial.post_settle_stable_rate <= 1.0
        assert 0.0 <= trial.post_hold_stable_rate <= 1.0
