"""Per-env FR3 arm domain randomization: joint dynamics, link mass/inertia, EE payload."""

from __future__ import annotations

import unittest

import numpy as np
import newton

from apple_pick_sim.coupled_fruiting.batched_build import build_replicated_robot_model
from apple_pick_sim.robot import fr3_robot
from apple_pick_sim.robot.fr3_robot import setup as fr3_setup
from apple_pick_sim.robot.fr3_robot.arm_domain_randomization import (
    ArmDomainRandomizationRanges,
    apply_ee_payload_dr,
    apply_joint_dynamics_dr,
    apply_link_mass_inertia_dr,
    sample_arm_domain_randomization,
)

_N_ARM_DOF = 7


def _usd_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return fr3_robot.fr3_assets_available()


def _build_small_robot(num_envs: int):
    tpl_model, tpl_tcp, _ = fr3_robot.build_fr3_robot_model_from_usd(
        device="cpu", create_solver=False
    )

    def _factory():
        return fr3_robot.build_fr3_robot_builder()

    model, tcp, solver = build_replicated_robot_model(
        tpl_model,
        tpl_tcp,
        num_envs=num_envs,
        env_spacing=(2.0, 2.0, 2.0),
        device="cpu",
        template_builder_factory=_factory,
        mujoco_solver_kwargs={"use_mujoco_cpu": True, "disable_contacts": True},
    )
    return model, tcp, solver


class TestSampleArmDomainRandomization(unittest.TestCase):
    """Pure numpy sampling logic -- no sim required."""

    def test_shapes(self):
        ranges = ArmDomainRandomizationRanges()
        sample = sample_arm_domain_randomization(ranges, num_envs=5, rng=np.random.default_rng(0))
        self.assertEqual(sample.armature.shape, (5, _N_ARM_DOF))
        self.assertEqual(sample.friction.shape, (5, _N_ARM_DOF))
        self.assertEqual(sample.joint_damping.shape, (5, _N_ARM_DOF))
        self.assertEqual(sample.link_mass_scale.shape, (5,))
        self.assertEqual(sample.link_inertia_scale.shape, (5,))
        self.assertEqual(sample.ee_mass_scale.shape, (5,))
        self.assertEqual(sample.ee_inertia_scale.shape, (5,))

    def test_scales_within_configured_ranges(self):
        ranges = ArmDomainRandomizationRanges(
            armature_scale=(0.5, 1.5),
            link_mass_scale=(0.8, 1.2),
            ee_payload_mass_scale=(0.95, 1.05),
        )
        nominal_armature = np.array([1.0] * _N_ARM_DOF)
        sample = sample_arm_domain_randomization(
            ranges,
            num_envs=200,
            rng=np.random.default_rng(1),
            nominal_armature=nominal_armature,
        )
        ratio = sample.armature / nominal_armature[None, :]
        self.assertTrue(np.all(ratio >= 0.5 - 1e-6))
        self.assertTrue(np.all(ratio <= 1.5 + 1e-6))
        self.assertTrue(np.all(sample.link_mass_scale >= 0.8 - 1e-6))
        self.assertTrue(np.all(sample.link_mass_scale <= 1.2 + 1e-6))
        self.assertTrue(np.all(sample.ee_mass_scale >= 0.95 - 1e-6))
        self.assertTrue(np.all(sample.ee_mass_scale <= 1.05 + 1e-6))

    def test_same_seed_reproducible_different_seed_differs(self):
        ranges = ArmDomainRandomizationRanges()
        a = sample_arm_domain_randomization(ranges, num_envs=10, rng=np.random.default_rng(42))
        b = sample_arm_domain_randomization(ranges, num_envs=10, rng=np.random.default_rng(42))
        c = sample_arm_domain_randomization(ranges, num_envs=10, rng=np.random.default_rng(7))
        np.testing.assert_allclose(a.armature, b.armature)
        self.assertFalse(np.allclose(a.armature, c.armature))

    def test_envs_get_distinct_values_not_all_identical(self):
        ranges = ArmDomainRandomizationRanges()
        sample = sample_arm_domain_randomization(
            ranges, num_envs=20, rng=np.random.default_rng(3)
        )
        self.assertGreater(len(np.unique(sample.link_mass_scale.round(6))), 1)


@unittest.skipUnless(_usd_available(), "Requires usd-core and bundled assets/fr3")
class TestApplyJointDynamicsDR(unittest.TestCase):
    """Per-world joint dynamics: Newton model array + MuJoCo sync via notify."""

    def test_setup_setter_accepts_per_world_2d_array_and_preserves_existing_scalar_behavior(self):
        model, _tcp, solver = _build_small_robot(3)
        # Existing (unchanged) scalar/vector broadcast-to-every-world behavior,
        # matching how configure_vic_joint_torques_arm_batched always calls
        # these setters on a multi-world model (explicit dofs_per_world).
        fr3_setup._set_fr3_joint_armature(
            model, np.full(_N_ARM_DOF, 2.5), dofs_per_world=_N_ARM_DOF
        )
        arr = model.joint_armature.numpy().reshape(3, _N_ARM_DOF)
        np.testing.assert_allclose(arr, np.full((3, _N_ARM_DOF), 2.5))

        # New per-world 2D array behavior.
        per_world = np.array([[1.0] * 7, [2.0] * 7, [3.0] * 7])
        fr3_setup._set_fr3_joint_armature(model, per_world, dofs_per_world=_N_ARM_DOF)
        arr2 = model.joint_armature.numpy().reshape(3, _N_ARM_DOF)
        np.testing.assert_allclose(arr2, per_world)

    def test_apply_joint_dynamics_dr_syncs_distinct_per_world_mujoco_arrays(self):
        model, _tcp, solver = _build_small_robot(3)
        ranges = ArmDomainRandomizationRanges()
        sample = sample_arm_domain_randomization(
            ranges, num_envs=3, rng=np.random.default_rng(0)
        )
        apply_joint_dynamics_dr(model, solver, sample, dofs_per_world=_N_ARM_DOF)

        mjw_armature = solver.mjw_model.dof_armature.numpy()
        self.assertEqual(mjw_armature.shape, (3, _N_ARM_DOF))
        np.testing.assert_allclose(mjw_armature, sample.armature, atol=1e-5)
        # Worlds must actually differ (not a silent broadcast of world 0).
        self.assertFalse(np.allclose(mjw_armature[0], mjw_armature[1]))

        mjw_friction = solver.mjw_model.dof_frictionloss.numpy()
        np.testing.assert_allclose(mjw_friction, sample.friction, atol=1e-5)

        mjw_damping = solver.mjw_model.dof_damping.numpy()
        np.testing.assert_allclose(mjw_damping, sample.joint_damping, atol=1e-5)

    def test_apply_joint_dynamics_dr_is_reassignable(self):
        """Arm DR can be re-randomized at any time (e.g. per-episode on reset)."""
        model, _tcp, solver = _build_small_robot(3)
        ranges = ArmDomainRandomizationRanges()
        first = sample_arm_domain_randomization(ranges, num_envs=3, rng=np.random.default_rng(0))
        apply_joint_dynamics_dr(model, solver, first, dofs_per_world=_N_ARM_DOF)
        second = sample_arm_domain_randomization(ranges, num_envs=3, rng=np.random.default_rng(99))
        apply_joint_dynamics_dr(model, solver, second, dofs_per_world=_N_ARM_DOF)
        mjw_armature = solver.mjw_model.dof_armature.numpy()
        np.testing.assert_allclose(mjw_armature, second.armature, atol=1e-5)
        self.assertFalse(np.allclose(mjw_armature, first.armature))


@unittest.skipUnless(_usd_available(), "Requires usd-core and bundled assets/fr3")
class TestApplyLinkAndEePayloadDR(unittest.TestCase):
    def test_apply_link_mass_inertia_dr_scales_distinct_per_world(self):
        model, _tcp, solver = _build_small_robot(3)
        mass_before = model.body_mass.numpy().copy()

        ranges = ArmDomainRandomizationRanges(link_mass_scale=(0.5, 2.0))
        sample = sample_arm_domain_randomization(
            ranges, num_envs=3, rng=np.random.default_rng(0)
        )
        robot_bodies_per_world = int(model.body_count) // 3
        apply_link_mass_inertia_dr(
            model, solver, sample, robot_bodies_per_world=robot_bodies_per_world, num_envs=3
        )

        mass_after = model.body_mass.numpy()
        self.assertFalse(np.allclose(mass_before, mass_after))
        # At least two worlds must end up with distinct total arm mass.
        # (robot_bodies_per_world=None path resolves per-world tiling internally.)
        n_bodies = mass_after.shape[0]
        per_world_body_count = n_bodies // 3
        world_masses = [
            mass_after[w * per_world_body_count : (w + 1) * per_world_body_count].sum()
            for w in range(3)
        ]
        self.assertGreater(len(set(round(m, 4) for m in world_masses)), 1)

    def test_apply_ee_payload_dr_scales_ee_body_per_world(self):
        model, _tcp, solver = _build_small_robot(3)
        mass_before = model.body_mass.numpy().copy()

        ranges = ArmDomainRandomizationRanges(ee_payload_mass_scale=(0.5, 2.0))
        sample = sample_arm_domain_randomization(
            ranges, num_envs=3, rng=np.random.default_rng(0)
        )
        robot_bodies_per_world = int(model.body_count) // 3
        apply_ee_payload_dr(
            model, solver, sample, robot_bodies_per_world=robot_bodies_per_world, num_envs=3
        )

        mass_after = model.body_mass.numpy()
        self.assertFalse(np.allclose(mass_before, mass_after))


if __name__ == "__main__":
    unittest.main()
