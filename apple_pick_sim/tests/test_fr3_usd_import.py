"""FR3 USD import smoke tests (M1 Slice 2)."""

from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
