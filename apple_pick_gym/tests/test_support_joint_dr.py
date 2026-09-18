"""Per-env support-joint domain randomization: sampling + wiring to solver arrays.

Fills CMA search-box knobs 8-10 (support k_p, support roll k_p, support joint
zeta) via the new ``sim_build.support_dr`` schema block. The application side
(``apply_per_env_support_joint_penalties`` / ``apply_per_env_support_roll_penalties``)
already exists and is exercised here, not reimplemented.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import newton
import numpy as np
import pytest

from apple_pick_gym.batched_envs.support_joint_dr import (
    SupportJointDRSample,
    apply_support_joint_dr,
    sample_support_joint_dr,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_build import (
    build_batched_heterogeneous_scene,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list
from apple_pick_sim.fruiting_system.params import RangeF, SupportJointDRRanges
from apple_pick_sim.tests.conftest import fr3_assets_available

# T-junction topology (needed: a "support" joint role) -- the straight-rod
# test fixture apple_pick_sim.tests.conftest.RANGES_FIXTURE has no support joint.
_T_JUNCTION_RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "apple_pick_sim"
    / "fixtures"
    / "fruiting_system_ranges_real_world_proxy_variance.json"
)

requires_fr3 = pytest.mark.skipif(
    not fr3_assets_available(),
    reason="Requires bundled assets/fr3 and usd-core",
)

_NUM_ENVS = 2


def _good_ranges() -> SupportJointDRRanges:
    return SupportJointDRRanges(
        kp=RangeF(min=100.0, max=1_000_000.0),
        roll_kp=RangeF(min=0.075, max=7.5),
        zeta=RangeF(min=0.05, max=0.95),
    )


def _vbd_only_config(num_envs: int) -> BatchedHeterogeneousCoupledSimConfig:
    """CPU-friendly, plant-only config -- mirrors apple_pick_sim's own test helper."""
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        robot=RobotConfig(kind="fr3", step_mode="vbd_only", fix_to_apple=False),
        scene=SceneSettleCollisionConfig(settle_substeps=0),
    )


def _linear_kp_at_joint(solver, global_joint_index: int) -> float:
    jc_start = solver.joint_constraint_start.numpy()
    k = solver.joint_penalty_k.numpy()
    c0 = int(jc_start[global_joint_index])
    return float(k[c0 + newton.solvers.SolverVBD.JointSlot.LINEAR])


class TestSampleSupportJointDR:
    """Pure numpy -- no sim required."""

    def test_shapes(self):
        sample = sample_support_joint_dr(_good_ranges(), num_envs=5, rng=np.random.default_rng(0))
        assert sample.kp.shape == (5,)
        assert sample.roll_kp.shape == (5,)
        assert sample.zeta.shape == (5,)

    def test_values_within_ranges(self):
        ranges = _good_ranges()
        sample = sample_support_joint_dr(ranges, num_envs=200, rng=np.random.default_rng(1))
        assert np.all(sample.kp >= ranges.kp.min - 1e-6)
        assert np.all(sample.kp <= ranges.kp.max + 1e-6)
        assert np.all(sample.roll_kp >= ranges.roll_kp.min - 1e-6)
        assert np.all(sample.roll_kp <= ranges.roll_kp.max + 1e-6)
        assert np.all(sample.zeta >= ranges.zeta.min - 1e-6)
        assert np.all(sample.zeta <= ranges.zeta.max + 1e-6)

    def test_kp_and_roll_kp_are_log_uniform_matching_cma_encoding(self):
        """kp/roll_kp span multiple decades and are searched on a log10 scale by
        CMA -- a linear-uniform sample would waste most draws in the top decade.
        zeta is CMA's own linear [0,1]-style dimension and stays linear-uniform."""
        ranges = _good_ranges()
        sample = sample_support_joint_dr(ranges, num_envs=2000, rng=np.random.default_rng(2))
        log_kp = np.log10(sample.kp)
        # A log-uniform sample over 4 decades should have roughly equal mass in
        # the bottom half vs top half of the log range (linear-uniform would not).
        mid = 0.5 * (np.log10(ranges.kp.min) + np.log10(ranges.kp.max))
        frac_below_mid = float(np.mean(log_kp < mid))
        assert 0.35 < frac_below_mid < 0.65, f"kp is not log-uniform: frac_below_mid={frac_below_mid}"

        log_roll = np.log10(sample.roll_kp)
        mid_roll = 0.5 * (np.log10(ranges.roll_kp.min) + np.log10(ranges.roll_kp.max))
        frac_below_mid_roll = float(np.mean(log_roll < mid_roll))
        assert 0.35 < frac_below_mid_roll < 0.65, (
            f"roll_kp is not log-uniform: frac_below_mid={frac_below_mid_roll}"
        )

    def test_reproducible_for_fixed_seed(self):
        ranges = _good_ranges()
        a = sample_support_joint_dr(ranges, num_envs=10, rng=np.random.default_rng(42))
        b = sample_support_joint_dr(ranges, num_envs=10, rng=np.random.default_rng(42))
        np.testing.assert_allclose(a.kp, b.kp)
        np.testing.assert_allclose(a.roll_kp, b.roll_kp)
        np.testing.assert_allclose(a.zeta, b.zeta)

    def test_envs_get_distinct_values(self):
        sample = sample_support_joint_dr(
            _good_ranges(), num_envs=20, rng=np.random.default_rng(3)
        )
        assert len(np.unique(sample.kp.round(4))) > 1
        assert len(np.unique(sample.roll_kp.round(4))) > 1
        assert len(np.unique(sample.zeta.round(4))) > 1


class TestApplySupportJointDR:
    """Integration: build a small VBD-only scene, apply DR, read back solver arrays."""

    @requires_fr3
    def test_per_env_support_kp_reaches_solver_arrays_and_topology_stays_uniform(self):
        ranges = load_ranges(_T_JUNCTION_RANGES_FIXTURE)
        params = sample_heterogeneous_params_list(ranges, topology_seed=7, num_envs=_NUM_ENVS)
        cfg = _vbd_only_config(_NUM_ENVS)
        result = build_batched_heterogeneous_scene(cfg, params, ranges)
        scene = result.scene

        # (d) segment topology stays identical across envs -- sample_heterogeneous_params_list
        # already guarantees this (verified here, not re-derived).
        assert (
            result.per_env_params[0].primary.num_segments
            == result.per_env_params[1].primary.num_segments
        )

        dr_sample = SupportJointDRSample(
            kp=np.array([500.0, 5000.0]),
            roll_kp=np.array([0.2, 2.0]),
            zeta=np.array([0.2, 0.8]),
        )
        apply_support_joint_dr(
            scene,
            dr_sample,
            num_envs=_NUM_ENVS,
            joints_per_world=scene.layout.joints_per_world,
            per_env_params=result.per_env_params,
        )

        j_support = next(
            j for j, lab in scene.cable.fruiting_fixed_joints if "primary_support_left" in lab
        )
        joints_per_world = int(scene.layout.joints_per_world)
        kp0 = _linear_kp_at_joint(scene.cable.solver, j_support)
        kp1 = _linear_kp_at_joint(scene.cable.solver, j_support + joints_per_world)
        assert kp0 == pytest.approx(500.0, rel=1e-4)
        assert kp1 == pytest.approx(5000.0, rel=1e-4)
        assert kp0 != kp1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
