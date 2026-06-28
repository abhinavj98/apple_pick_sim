"""Smoke tests for batched heterogeneous coupled-fruiting profiler."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from apple_pick_sim.tests.conftest import RANGES_FIXTURE, requires_fr3

_EXPECTED_PHASES = (
    "build_settled",
    "settle",
    "build_welded",
    "ik_bootstrap",
    "warmup",
    "step_bench",
)


def _load_benchmark_module():
    path = (
        Path(__file__).resolve().parent.parent
        / "diagnostics"
        / "benchmark_batched_heterogeneous.py"
    )
    spec = importlib.util.spec_from_file_location(
        "apple_pick_sim.diagnostics.benchmark_batched_heterogeneous",
        path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires_fr3
@pytest.mark.slow
def test_run_profile_smoke_fr3_cpu():
    mod = _load_benchmark_module()
    report = mod.run_profile(
        ranges_path=RANGES_FIXTURE,
        seed=42,
        num_envs=2,
        robot="fr3",
        device="cpu",
        settle_substeps=5,
        sim_substeps=15,
        warmup_frames=1,
        bench_frames=2,
        write_json=False,
    )

    assert report["num_envs"] == 2
    assert report["seed"] == 42
    assert report["device"] == "cpu"
    assert report["robot"] == "fr3"

    phases = report["phases"]
    assert set(phases) == set(_EXPECTED_PHASES)
    for name in _EXPECTED_PHASES:
        phase = phases[name]
        assert phase["calls"] == 1
        assert phase["total_ms"] >= 0.0

    step = phases["step_bench"]
    assert step["ms_per_frame"] > 0.0
    assert step["ms_per_substep"] > 0.0
    assert step["fps"] > 0.0

    encoded = json.dumps(report)
    roundtrip = json.loads(encoded)
    assert roundtrip["phases"]["step_bench"]["fps"] == pytest.approx(step["fps"])
