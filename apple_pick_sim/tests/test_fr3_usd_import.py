"""FR3 USD import smoke tests (M1 Slice 2)."""

from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import newton
from newton.solvers import SolverMuJoCo

from apple_pick_sim.robot import fr3_robot


def _usd_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return fr3_robot.fr3_assets_available()


@unittest.skipUnless(_usd_available(), "Requires usd-core and bundled assets/fr3")
class TestFr3UsdImport(unittest.TestCase):
    def test_import_body_and_joint_counts(self):
        model, tcp_idx, _solver = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        self.assertGreater(model.body_count, 2)
        self.assertGreater(model.joint_count, 2)
        self.assertGreaterEqual(tcp_idx, 0)
        self.assertLess(tcp_idx, model.body_count)

    def test_resolve_tcp_body_index_unique(self):
        model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        self.assertEqual(tcp_idx, fr3_robot.resolve_tcp_body_index(model))

    def test_tcp_label_suffix(self):
        model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        lbl = model.body_label[tcp_idx]
        leaf = lbl.split("/")[-1]
        self.assertIn(leaf, ("tcp", "ee"), lbl)

    def test_ee_body_from_testfr3_scene(self):
        model, _tcp, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        ee_idx = fr3_robot.resolve_ee_body_index(model)
        self.assertGreaterEqual(ee_idx, 0)
        self.assertIn("ee", model.body_label[ee_idx])

    def test_mujoco_solver_constructs(self):
        _model, _tcp, solver = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        self.assertIsInstance(solver, SolverMuJoCo)

    def test_mujoco_solver_constructs_without_deprecation_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            _model, _tcp, solver = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        self.assertIsInstance(solver, SolverMuJoCo)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            deprecations,
            [],
            msg="; ".join(str(w.message) for w in deprecations),
        )

    def test_joint_coord_and_dof_counts_positive(self):
        model, _tcp, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        n_coord = int(model.joint_coord_count)
        n_dof = int(model.joint_dof_count)
        self.assertGreaterEqual(n_coord, n_dof)
        self.assertGreater(n_dof, 6)

    def test_tcp_body_mass_is_positive_finite(self):
        import numpy as np

        model, tcp_idx, _ = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        m = float(model.body_mass.numpy()[tcp_idx])
        self.assertTrue(np.isfinite(m) and m > 0.0)

    def test_sync_robot_gravity_zeros_mujoco_opt_gravity(self):
        import numpy as np

        model, _tcp, solver = fr3_robot.build_fr3_robot_model_from_usd(device="cpu")
        fr3_robot.sync_robot_gravity_to_mujoco(model, solver)
        g = solver.mj_model.opt.gravity
        np.testing.assert_allclose(np.asarray(g).reshape(3), 0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
