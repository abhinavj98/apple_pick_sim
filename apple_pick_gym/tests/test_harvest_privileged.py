"""Privileged (critic-only) ground truth: plant DR, support DR, arm DR, grasp geometry.

Pure numpy/torch against fake DR samples -- no sim build.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from apple_pick_gym.batched_envs.harvest_obs import _PRIVILEGED_FIELDS
from apple_pick_gym.batched_envs.harvest_privileged import (
    LOG10_PRIVILEGED_FIELDS,
    PLANT_GEOMETRY_FIELDS,
    build_plant_geometry,
    build_privileged_fields,
)
from apple_pick_gym.batched_envs.support_joint_dr import SupportJointDRSample
from apple_pick_gym.batched_envs.world_set import load_world_set
from apple_pick_sim.fruiting_system.params import fruiting_params_from_json
from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
    ArmDomainRandomizationRanges,
    sample_arm_domain_randomization,
)

N = 3


@pytest.fixture(scope="module")
def world():
    from pathlib import Path

    specs = load_world_set(Path("apple_pick_gym/world_sets/harvest_worlds_v1.jsonl"))[:N]
    params = [fruiting_params_from_json(s.params_json) for s in specs]
    support = SupportJointDRSample(
        kp=np.array([s.support_kp for s in specs], dtype=np.float32),
        roll_kp=np.array([s.support_roll_kp for s in specs], dtype=np.float32),
        zeta=np.array([s.support_zeta for s in specs], dtype=np.float32),
    )
    rng = np.random.default_rng(0)
    arm_build = sample_arm_domain_randomization(ArmDomainRandomizationRanges(), num_envs=N, rng=rng)
    arm_joint = sample_arm_domain_randomization(ArmDomainRandomizationRanges(), num_envs=N, rng=rng)
    welds = [s.weld_direction for s in specs]
    return params, support, arm_build, arm_joint, welds


def test_fields_match_the_documented_critic_layout(world):
    params, support, arm_build, arm_joint, _ = world
    out = build_privileged_fields(params, support, arm_build, arm_joint, device="cpu")
    assert list(out) == [name for name, _ in _PRIVILEGED_FIELDS]
    for name, width in _PRIVILEGED_FIELDS:
        assert out[name].shape == (N, width), name
        assert out[name].dtype == torch.float32
        assert bool(torch.isfinite(out[name]).all()), name


def test_moduli_and_support_stiffness_are_log10(world):
    params, support, arm_build, arm_joint, _ = world
    out = build_privileged_fields(params, support, arm_build, arm_joint, device="cpu")
    assert set(LOG10_PRIVILEGED_FIELDS) == {
        "spur_flexural_modulus_pa",
        "stem_flexural_modulus_pa",
        "spur_youngs_modulus_pa",
        "stem_youngs_modulus_pa",
        "support_kp",
        "support_roll_kp",
    }
    for i, p in enumerate(params):
        assert float(out["stem_youngs_modulus_pa"][i, 0]) == pytest.approx(math.log10(p.stem.youngs_modulus_pa), rel=1e-6)
        assert float(out["spur_flexural_modulus_pa"][i, 0]) == pytest.approx(math.log10(p.spur.flexural_modulus_pa), rel=1e-6)
        assert float(out["support_kp"][i, 0]) == pytest.approx(math.log10(float(support.kp[i])), rel=1e-6)
        assert float(out["stem_damping_ratio"][i, 0]) == pytest.approx(p.stem.damping_ratio, rel=1e-6)
        assert float(out["primary_density"][i, 0]) == pytest.approx(p.primary.density, rel=1e-6)
    # no decade-spanning raw SI values survive (raw moduli are up to ~1e10); the largest
    # remaining input is primary density (~1e3 kg/m^3), which the running scaler handles
    for name, _ in _PRIVILEGED_FIELDS:
        assert float(out[name].abs().max()) < 1e4, name


def test_arm_joint_dr_is_the_per_reset_sample_and_build_scales_are_the_build_sample(world):
    params, support, arm_build, arm_joint, _ = world
    out = build_privileged_fields(params, support, arm_build, arm_joint, device="cpu")
    torch.testing.assert_close(out["arm_armature"], torch.as_tensor(arm_joint.armature))
    torch.testing.assert_close(out["arm_friction"], torch.as_tensor(arm_joint.friction))
    torch.testing.assert_close(out["arm_joint_damping"], torch.as_tensor(arm_joint.joint_damping))
    torch.testing.assert_close(out["arm_link_mass_scale"][:, 0], torch.as_tensor(arm_build.link_mass_scale))
    torch.testing.assert_close(out["arm_ee_inertia_scale"][:, 0], torch.as_tensor(arm_build.ee_inertia_scale))


def test_missing_arm_joint_sample_uses_nominal_scale_one(world):
    params, support, arm_build, _, _ = world
    out = build_privileged_fields(params, support, arm_build, None, device="cpu")
    from apple_pick_sim.robot.fr3_robot import setup as fr3_setup

    nominal = torch.as_tensor(np.asarray(fr3_setup.FR3_DEFAULT_JOINT_FRICTION, dtype=np.float32))
    torch.testing.assert_close(out["arm_friction"], nominal.expand(N, 7))


def test_env_count_mismatch_raises(world):
    params, support, arm_build, arm_joint, _ = world
    short = dataclasses.replace(support, kp=support.kp[:2])
    with pytest.raises(ValueError):
        build_privileged_fields(params, short, arm_build, arm_joint, device="cpu")


def test_plant_geometry_and_grasp_axis(world):
    params, _, _, _, welds = world
    geo = build_plant_geometry(params, welds, device="cpu")
    assert list(geo) == [name for name, _ in PLANT_GEOMETRY_FIELDS]
    for name, width in PLANT_GEOMETRY_FIELDS:
        assert geo[name].shape == (N, width)
    torch.testing.assert_close(geo["weld_direction"], torch.tensor(welds, dtype=torch.float32))
    assert float(geo["stem_length_m"][0, 0]) == pytest.approx(params[0].stem.length, rel=1e-6)
    assert float(geo["apple_radius_m"][0, 0]) == pytest.approx(params[0].apple_radius, rel=1e-6)
