"""Per-world MuJoCo constraint/contact arena sizing must not scale with num_envs.

``njmax``/``nconmax`` passed to ``newton.solvers.SolverMuJoCo`` are **per-world**
sizes: mujoco_warp allocates constraint arrays as ``(nworld, njmax)`` and
derives the global contact pool itself as ``nconmax * nworld`` internally
(``mujoco_warp._src.io.put_data`` / ``_resolve_batch_size``). Multiplying the
per-world size by ``num_envs`` before passing it in applies the world count
twice, making allocation ``O(N^2)``. See
``docs/superpowers/plans/2026-09-17-rl-vic-harvest-policy.md`` Task 0a.
"""

from __future__ import annotations

import unittest

import newton
import warp as wp

from apple_pick_sim.coupled_fruiting.batched_build import build_replicated_robot_model
from apple_pick_sim.robot import fr3_robot


def _usd_available() -> bool:
    try:
        import pxr  # noqa: F401
    except ImportError:
        return False
    return fr3_robot.fr3_assets_available()


def _build_at(num_envs: int) -> newton.solvers.SolverMuJoCo:
    tpl_model, tpl_tcp, _ = fr3_robot.build_fr3_robot_model_from_usd(
        device="cpu", create_solver=False
    )

    def _factory() -> tuple[newton.ModelBuilder, int]:
        return fr3_robot.build_fr3_robot_builder()

    _model, _tcp, solver = build_replicated_robot_model(
        tpl_model,
        tpl_tcp,
        num_envs=num_envs,
        env_spacing=(2.0, 2.0, 2.0),
        device="cpu",
        template_builder_factory=_factory,
        mujoco_solver_kwargs={"use_mujoco_cpu": True, "disable_contacts": False},
    )
    return solver


@unittest.skipUnless(_usd_available(), "Requires usd-core and bundled assets/fr3")
class TestMujocoConstraintArenaSizing(unittest.TestCase):
    """Regression guard: fails if ``* num_envs`` is reintroduced on njmax/nconmax."""

    def test_per_world_njmax_is_constant_across_num_envs(self):
        njmax_by_n = {}
        for n in (2, 8, 32):
            solver = _build_at(n)
            njmax_by_n[n] = int(solver.mjw_data.njmax)

        self.assertEqual(
            len(set(njmax_by_n.values())),
            1,
            f"njmax must not scale with num_envs, got {njmax_by_n}",
        )

    def test_per_world_nconmax_is_constant_across_num_envs(self):
        # naconmax is mujoco_warp's derived GLOBAL contact pool
        # (nconmax_per_world * nworld); it legitimately scales with num_envs.
        # What must stay constant is the PER-WORLD nconmax we pass in, which
        # we recover as naconmax / nworld.
        per_world_nconmax_by_n = {}
        for n in (2, 8, 32):
            solver = _build_at(n)
            naconmax = int(solver.mjw_data.naconmax)
            self.assertEqual(
                naconmax % n,
                0,
                f"naconmax ({naconmax}) should be an exact multiple of nworld ({n})",
            )
            per_world_nconmax_by_n[n] = naconmax // n

        self.assertEqual(
            len(set(per_world_nconmax_by_n.values())),
            1,
            f"per-world nconmax must not scale with num_envs, got {per_world_nconmax_by_n}",
        )

    def test_njmax_does_not_grow_quadratically_with_num_envs(self):
        # The historical bug (njmax = 80 * num_envs) makes njmax scale
        # linearly with N by itself, on top of mujoco_warp's own per-world
        # allocation -- i.e. total constraint-array footprint (nworld *
        # njmax) grows as O(N^2). Assert the *global* footprint is linear in
        # N, not quadratic, as a second, harder-to-fake regression signal.
        footprints = {}
        for n in (4, 32):
            solver = _build_at(n)
            njmax = int(solver.mjw_data.njmax)
            footprints[n] = n * njmax  # total (nworld, njmax) array size

        ratio_n = 32 / 4
        ratio_footprint = footprints[32] / footprints[4]
        self.assertAlmostEqual(
            ratio_footprint,
            ratio_n,
            delta=0.5,
            msg=(
                f"constraint-array footprint should scale linearly with "
                f"num_envs (expected ~{ratio_n}x), got {ratio_footprint}x: "
                f"{footprints}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
