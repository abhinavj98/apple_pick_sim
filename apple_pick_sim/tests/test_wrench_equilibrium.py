"""Equilibrium validation tests for SolverVBD fixed-joint wrench readouts.

WHY THESE TESTS EXIST
---------------------
``fixed_joint_wrenches_child_com_vbd`` (backed by
``SolverVBD.gather_joint_wrench_child_com``) reads the same joint
force/torque pair that the AVBD kernel assembles for the child body during
``solve_rigid_body``.  These tests confirm, via independent analytic
arguments from Newton's first and third laws, that:

  (a) the numbers are physically correct (not just finite),
  (b) the reference frame is world-frame Z-up, and
  (c) the sign convention is "force on child from joint" (positive Z =
      upward reaction when joint supports a hanging body).

TWO INDEPENDENT ANALYTIC CHECKS
--------------------------------
1. **Single-body free-body balance (cleanest, zero subtlety)**

   The apple body has exactly ONE joint on it (``joint_stem_apple``,
   type FIXED).  Its only other external load is gravity.  At
   quasi-static rest (|v| ≈ 0, |a| ≈ 0):

       ΣF = 0  →  F_joint[Z] + (−m·g) = 0
                  F_joint[Z] = model.body_mass[apple] · g   [N, positive = up]

       Στ_COM = 0  →  τ_joint_at_COM = 0                   [N·m, about apple COM]

   The zero-torque identity holds because gravity acts at the COM (no
   gravitational moment about COM) and the joint is the only other
   source of wrench.  For a straight vertical chain (r ∥ F) the
   lever-arm term vanishes and the angular-constraint contribution
   must also be zero.

2. **Subtree cut theorem (valid for every FIXED joint in the serial chain)**

   For a serial chain {primary → secondary → spur → stem → apple} at
   quasi-static rest with no ground contacts in the subtree:

       F_joint_j[Z] = M_subtree_j · g

   where M_subtree_j is the sum of ``model.body_mass`` of every body
   that lies strictly below the cut at joint j.

   Proof: treat the subtree as a free-body.  Internal forces (cable joints
   within each rod, the FIXED stem→apple joint) are Newton's-3rd-law pairs
   and cancel.  External forces are gravity on each subtree body plus the
   single constraint wrench from joint j on its child.  ΣF = 0 gives the
   identity.

NOTE ON APPLE BODY MASS
-----------------------
``model.body_mass[apple_body]`` is the mass the AVBD solver uses in
``integrate_rigid_body``.  The fruiting-system builder passes
``m = (4/3)πr³ρ`` to ``add_link`` and adds the collision sphere with
``ShapeConfig(density=0.0)`` so the shape does **not** add a second mass
from the default ``~1000`` kg/m³ shape density.  ``model.body_mass[apple]``
therefore matches the analytic sphere mass.  See docs/WRENCH_READOUT.md
section 6.

NOTE ON JOINT DAMPING
---------------------
Inter-segment FIXED joints use a **small** default ``rigid_joint_linear_kd`` /
``rigid_joint_angular_kd`` in :func:`apple_pick_sim.fruiting_system.make_fruiting_solver_vbd`
(see ``FRUITING_VBD_RIGID_JOINT_LINEAR_KD`` / ``FRUITING_VBD_RIGID_JOINT_ANGULAR_KD`` in ``fruiting_system/build.py``): enough to damp
oscillations within the settling horizon without the huge effective stiffness that a
large ``kd`` produces together with ``rigid_joint_linear_k_start ~ 1e8``.

FRAME & SIGN SUMMARY (see also docs/WRENCH_READOUT.md)
-------------------------------------------------------
* All components are in **world frame** (Y-forward, Z-up, X-right).
* ``force_world``                   : linear force on the child body [N].
* ``torque_at_child_com_world``     : torque about the child's COM [N·m].
* For a hanging body (gravity along −Z), force_world[Z] is **positive**.

TEST STRUCTURE
--------------
* ``test_apple_joint_force_at_equilibrium``          – F_joint[Z] ≈ m*g
* ``test_apple_joint_torque_vanishes_at_equilibrium``– |τ| ≈ 0
* ``test_joint_force_z_is_upward_for_hanging_apple`` – sign / frame check
* ``test_subtree_weight_theorem_straight_chain``     – all 4 joints
* ``test_subtree_forces_are_cumulative_down_the_chain``– monotonicity
* ``test_apple_force_proportional_to_apple_mass``   – linearity / scaling
* ``test_apple_body_mass_matches_analytic_sphere`` — no double-counted shape mass
* ``test_fruiting_ranges_fixture_chain_nearly_vertical`` — default JSON stays ≈ −Z
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G = 9.81  # m/s²  — matches model.set_gravity((0, 0, -G)) in _build_scene

# Default range file: nearly vertical chain for demos and sanity checks.
_RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
)

# Base height: the anchor of the first segment.  Must be high enough that no
# subtree body can touch the ground (z = 0) after the chain settles.
_BASE_Z = 4.0


# ---------------------------------------------------------------------------
# Scene-building helpers
# ---------------------------------------------------------------------------


def _import_fs():
    """Lazy import to allow test collection before the package is installed."""
    import apple_pick_sim.fruiting_system as fs

    return fs


def _make_minimal_scene(apple_radius: float, apple_density: float, device: str = "cpu"):
    """Minimal scene: 2-segment vertical stem + apple only.

    The chain hangs straight down from ``_BASE_Z``.  Starting configuration
    already approximates mechanical equilibrium; the solver primarily needs
    to build up its Lagrange multipliers (AVBD dual variables).
    Returns the scene built by ``_build_scene`` (default fruiting solver includes light rigid-joint kd).
    """
    fs = _import_fs()
    params = fs.FruitingSystemParams(
        primary=None,
        secondary=None,
        spur=None,
        stem=fs.RodParams(
            num_segments=2,
            length=0.20,
            radius=0.005,
            bend_stiffness=100.0,
            bend_damping=5.0,
            stretch_stiffness=500.0,
            density=200.0,
            direction=(0.0, 0.0, -1.0),
        ),
        apple_radius=apple_radius,
        apple_density=apple_density,
        topology=fs.TOPOLOGY_LINEAR_CHAIN,
    )
    scene = fs._build_scene(
        params,
        base_pos=(0.0, 0.0, _BASE_Z),
        device=device,
        enable_self_collisions=False,
    )
    return scene


def _make_full_chain_scene(device: str = "cpu"):
    """Full primary → secondary → spur → stem → apple scene.

    All segments point straight down so the chain starts near equilibrium.
    Masses are deterministic (no random sampling) for exact analytic expectations.
    """
    fs = _import_fs()
    params = fs.FruitingSystemParams(
        primary=fs.RodParams(
            num_segments=2,
            length=0.15,
            radius=0.008,
            bend_stiffness=300.0,
            bend_damping=5.0,
            stretch_stiffness=800.0,
            density=300.0,
            direction=(0.0, 0.0, -1.0),
        ),
        secondary=fs.RodParams(
            num_segments=2,
            length=0.12,
            radius=0.006,
            bend_stiffness=100.0,
            bend_damping=5.0,
            stretch_stiffness=500.0,
            density=200.0,
            direction=(0.0, 0.0, -1.0),
        ),
        spur=fs.RodParams(
            num_segments=2,
            length=0.08,
            radius=0.004,
            bend_stiffness=60.0,
            bend_damping=5.0,
            stretch_stiffness=300.0,
            density=150.0,
            direction=(0.0, 0.0, -1.0),
        ),
        stem=fs.RodParams(
            num_segments=2,
            length=0.10,
            radius=0.003,
            bend_stiffness=40.0,
            bend_damping=5.0,
            stretch_stiffness=200.0,
            density=100.0,
            direction=(0.0, 0.0, -1.0),
        ),
        apple_radius=0.040,
        apple_density=850.0,
        topology=fs.TOPOLOGY_LINEAR_CHAIN,
    )
    scene = fs._build_scene(
        params,
        base_pos=(0.0, 0.0, _BASE_Z),
        device=device,
        enable_self_collisions=False,
    )
    return scene


# ---------------------------------------------------------------------------
# Settling helper
# ---------------------------------------------------------------------------


def _settle(scene, *, num_frames: int = 220, substeps: int = 10, fps: float = 60.0):
    """Advance the scene to quasi-static equilibrium.

    The chain starts (approximately) at its equilibrium configuration; the
    AVBD dual variables (Lagrange multipliers) need a few seconds of
    simulation time to converge.  Light rigid-joint ``kd`` from
    ``fruiting_system.make_fruiting_solver_vbd`` damps oscillation
    around equilibrium slowly enough to avoid distorting ``F_z``.

    Uses :func:`apple_pick_sim.fruiting_system.example_collision_pipeline` so
    collision detection matches ``example_fruiting_system.py``.

    Returns:
        Tuple ``(q_prev_np, sim_dt)`` ready for
        :func:`~apple_pick_sim.fruiting_system.fixed_joint_wrenches_child_com_vbd`.
        ``q_prev_np`` is the snapshot taken *before* the final substep so the
        wrench API receives a consistent ``(body_q_prev → body_q, dt)`` pair.
    """
    fs = _import_fs()
    pipe = fs.example_collision_pipeline(scene.model)
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / substeps
    last_frame = num_frames - 1
    last_sub = substeps - 1
    q_prev_np = None

    for f in range(num_frames):
        for s in range(substeps):
            scene.state_0.clear_forces()
            contacts = scene.model.collide(scene.state_0, collision_pipeline=pipe)
            if f == last_frame and s == last_sub:
                q_prev_np = scene.state_0.body_q.numpy().copy()
            scene.solver.step(
                scene.state_0, scene.state_1, scene.control, contacts, sim_dt
            )
            scene.state_0, scene.state_1 = scene.state_1, scene.state_0

    return q_prev_np, sim_dt


def _settle_joint_fz_last_frame_mean(
    scene,
    joint_labels: tuple[str, ...],
    *,
    num_frames: int,
    substeps: int,
    fps: float = 60.0,
) -> tuple[dict[str, float], float]:
    """Like :func:`_settle`, but returns mean ``force_world[Z]`` per joint over the **last frame's** substeps.

    Multi-segment chains still vibrate slightly in VBD even with light joint
    damping; averaging the readout over one macro-step damps the snapshot
    noise without changing the static balance the subtree theorem targets.
    """
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / substeps
    last_frame = num_frames - 1
    fs = _import_fs()
    pipe = fs.example_collision_pipeline(scene.model)
    accum: dict[str, list[float]] = {lab: [] for lab in joint_labels}

    for f in range(num_frames):
        for s in range(substeps):
            scene.state_0.clear_forces()
            contacts = scene.model.collide(scene.state_0, collision_pipeline=pipe)
            q_prev_np = scene.state_0.body_q.numpy().copy()
            scene.solver.step(
                scene.state_0, scene.state_1, scene.control, contacts, sim_dt
            )
            scene.state_0, scene.state_1 = scene.state_1, scene.state_0
            if f == last_frame:
                wrenches = fs.fixed_joint_wrenches_child_com_vbd(
                    scene.model,
                    scene.solver,
                    body_q=scene.state_0.body_q.numpy(),
                    body_q_prev=q_prev_np,
                    dt=sim_dt,
                    joint_pairs=list(scene.fruiting_fixed_joints),
                )
                by_label = {w.label: w for w in wrenches}
                for lab in joint_labels:
                    accum[lab].append(float(by_label[lab].force_world[2]))

    mean_fz = {lab: float(np.mean(v)) for lab, v in accum.items()}
    return mean_fz, sim_dt


def _apple_mass_from_model(scene) -> float:
    """Return ``model.body_mass[apple_body]`` — the mass the AVBD solver uses."""
    return float(scene.model.body_mass.numpy()[scene.apple_body])


def _subtree_mass(scene, joint_label: str) -> float:
    """Total ``model.body_mass`` of every body strictly below the given FIXED joint.

    For the serial chain primary → secondary → spur → stem → apple:

        joint_stem_apple       : {apple}
        joint_spur_stem        : {stem_bodies} + {apple}
        joint_secondary_spur   : {spur_bodies} + {stem_bodies} + {apple}
        joint_primary_secondary: {secondary_bodies} + {spur_bodies} +
                                  {stem_bodies} + {apple}

    ``primary_bodies[0]`` has mass=0 (pin anchor) and is above every cut.
    All other primary bodies are also above every cut (they are ABOVE
    joint_primary_secondary, not below it).
    """
    body_mass = scene.model.body_mass.numpy()
    apple = [scene.apple_body] if scene.apple_body is not None else []

    subtree_map = {
        "joint_stem_apple": apple,
        "joint_spur_stem": list(scene.stem_bodies) + apple,
        "joint_secondary_spur": list(scene.spur_bodies) + list(scene.stem_bodies) + apple,
        "joint_primary_secondary": (
            list(scene.secondary_bodies)
            + list(scene.spur_bodies)
            + list(scene.stem_bodies)
            + apple
        ),
    }
    return float(sum(body_mass[b] for b in subtree_map[joint_label]))


def _get_wrenches_by_label(scene, q_prev_np, sim_dt) -> dict:
    """Run gather_joint_wrench and return ``{label: FixedJointWrenchRecord}``."""
    fs = _import_fs()
    wrenches = fs.fixed_joint_wrenches_child_com_vbd(
        scene.model,
        scene.solver,
        body_q=scene.state_0.body_q.numpy(),
        body_q_prev=q_prev_np,
        dt=sim_dt,
        joint_pairs=list(scene.fruiting_fixed_joints),
    )
    return {w.label: w for w in wrenches}


def _check_no_ground_contact(scene, label: str):
    """Assert every dynamic subtree body is above z = 0.1 m (test-validity guard).

    If any body has touched the ground, contact forces pollute the wrench
    balance and the test assertion is no longer meaningful.
    """
    body_q = scene.state_0.body_q.to("cpu").numpy()
    body_mass = scene.model.body_mass.numpy()
    apple = [scene.apple_body] if scene.apple_body is not None else []
    all_dynamic = (
        [b for b in scene.stem_bodies if body_mass[b] > 0]
        + [b for b in scene.spur_bodies if body_mass[b] > 0]
        + [b for b in scene.secondary_bodies if body_mass[b] > 0]
        + apple
    )
    for b in all_dynamic:
        z = float(body_q[b, 2])
        assert z > 0.1, (
            f"Body {b} z={z:.3f} m ≤ 0.1 m — ground contact would "
            f"invalidate the {label!r} test assertion."
        )


# ===========================================================================
# Tests
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Single-body free-body balance on the apple
# ---------------------------------------------------------------------------


def test_apple_joint_force_at_equilibrium():
    """F_joint_stem_apple[Z] ≈ model.body_mass[apple] * g  (±5 %).

    Physical argument
    -----------------
    The apple body has exactly ONE constraint: ``joint_stem_apple`` (type
    FIXED, child = apple_body).  Its only external load is gravity
    F_grav = (0, 0, −m·g).  At quasi-static rest, Newton's 1st law requires:

        ΣF_Z = 0
        F_joint[Z] − m·g = 0
        F_joint[Z] = model.body_mass[apple] · G

    This assertion is *independent of the solver internals*: it follows
    solely from F = ma with a ≈ 0 and knowledge of which bodies and joints
    act on the apple.

    ``model.body_mass[apple]`` matches ``(4/3)πr³ρ`` because the apple
    collision sphere is added with ``density=0`` (mass comes only from
    ``add_link``).
    """
    scene = _make_minimal_scene(apple_radius=0.040, apple_density=850.0)
    m_apple = _apple_mass_from_model(scene)

    q_prev, sim_dt = _settle(scene, num_frames=220, substeps=10)
    _check_no_ground_contact(scene, "test_apple_joint_force_at_equilibrium")

    w_by_label = _get_wrenches_by_label(scene, q_prev, sim_dt)
    w = w_by_label["joint_stem_apple"]

    expected_Fz = m_apple * G
    np.testing.assert_allclose(
        w.force_world[2],
        expected_Fz,
        rtol=0.05,
        err_msg=(
            f"joint_stem_apple force_world[Z]={w.force_world[2]:.4f} N "
            f"should equal model_mass*g={expected_Fz:.4f} N (±5 %)"
        ),
    )


def test_apple_joint_torque_vanishes_at_equilibrium():
    """|τ_joint_stem_apple at apple COM| ≈ 0 at quasi-static rest.

    Physical argument
    -----------------
    Gravity acts at the apple's COM (zero moment arm), so it produces no
    torque about the COM.  The FIXED joint is the only other load, and at
    quasi-static rest Στ_COM = I·α ≈ 0 requires:

        |torque_at_child_com_world| ≈ 0

    The returned torque is the *sum* of:
    * linear lever-arm: ``cross(r_anchor – COM, F_lin)``
    * angular-constraint contribution: T_ang

    For a straight vertical chain (r ∥ F) the lever-arm term is identically
    zero.  The angular term must also vanish because no external torque is
    applied to the apple.  Together both terms must sum to zero at equilibrium
    regardless of the joint anchor geometry.
    """
    scene = _make_minimal_scene(apple_radius=0.040, apple_density=850.0)
    m_apple = _apple_mass_from_model(scene)

    q_prev, sim_dt = _settle(scene, num_frames=220, substeps=10)
    _check_no_ground_contact(scene, "test_apple_joint_torque_vanishes_at_equilibrium")

    w_by_label = _get_wrenches_by_label(scene, q_prev, sim_dt)
    w = w_by_label["joint_stem_apple"]

    torque_mag = float(np.linalg.norm(w.torque_at_child_com_world))
    # Threshold: 2 % of the scale m·g·r (apple radius is the relevant length scale)
    threshold = 0.02 * m_apple * G * 0.040
    assert torque_mag < threshold, (
        f"Torque magnitude {torque_mag:.4e} N·m exceeds threshold "
        f"{threshold:.4e} N·m (2 % of m·g·r) — system has not reached "
        f"rotational equilibrium"
    )


# ---------------------------------------------------------------------------
# 2. Frame and sign convention
# ---------------------------------------------------------------------------


def test_joint_force_z_is_upward_for_hanging_apple():
    """force_world[Z] > 0 and dominates when the chain hangs along −Z.

    This test makes the frame convention and sign explicit:

    Frame
    -----
    World frame is Z-up.  Gravity is (0, 0, −9.81).  ``force_world`` is
    expressed in this frame.

    Sign
    ----
    ``force_world`` is the force that the joint exerts **on the child body**
    (i.e. the apple).  For a hanging apple, the joint must push UP on the
    apple to balance gravity, so ``force_world[Z] > 0``.

    Derivation from the kernel (``rigid_vbd_kernels.py``, ~line 522)::

        f_attachment = k·C + λ + damping_term
        force_child  = −f_attachment      # 'is_parent = False' branch

    At equilibrium, ``f_attachment ≈ −λ`` (with C ≈ 0, damping ≈ 0).
    The Lagrange multiplier λ for the apple joint converges to the value
    that satisfies the child free-body balance, giving ``force_child[Z] > 0``.

    Direction check
    ---------------
    For a perfectly vertical chain there are no horizontal loads, so the
    Z component must constitute at least 90 % of the total force magnitude.
    """
    scene = _make_minimal_scene(apple_radius=0.040, apple_density=850.0)

    q_prev, sim_dt = _settle(scene, num_frames=220, substeps=10)

    w_by_label = _get_wrenches_by_label(scene, q_prev, sim_dt)
    w = w_by_label["joint_stem_apple"]
    fx = float(w.force_world[0])
    fy = float(w.force_world[1])
    fz = float(w.force_world[2])
    fmag = float(np.linalg.norm(w.force_world))

    # Force on child (apple) must be upward (+Z)
    assert fz > 0, (
        f"force_world[Z]={fz:.4f} N should be positive (upward) "
        f"for a hanging apple"
    )
    # Dominant Z component for a vertical chain
    assert fz > 0.90 * fmag, (
        f"Z component {fz:.4f} N should be > 90 % of total magnitude "
        f"{fmag:.4f} N for a straight vertical chain"
    )
    # Negligible horizontal components
    assert abs(fx) < 0.05 * fmag, (
        f"|Fx|={abs(fx):.4e} N should be < 5 % of magnitude {fmag:.4e} N"
    )
    assert abs(fy) < 0.05 * fmag, (
        f"|Fy|={abs(fy):.4e} N should be < 5 % of magnitude {fmag:.4e} N"
    )


# ---------------------------------------------------------------------------
# 3. Subtree cut theorem
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_subtree_weight_theorem_straight_chain():
    """F_joint_j[Z] ≈ M_subtree_j · g for every FIXED joint (±20 %).

    Physical argument (subtree cut theorem)
    ----------------------------------------
    For each FIXED joint j in the serial chain, treat every body below
    the cut as a free-body system S_j.  Forces on S_j:

    * Gravity on each body i ∈ S_j  : (0, 0, −m_i · g)
    * Constraint wrench from joint j on child of j  : (0, 0, +F_Z_j)
    * Internal forces (cable joints between rod segments, the
      stem→apple FIXED joint) : Newton's-3rd-law pairs, cancel in the sum

    At quasi-static rest (ΣF = 0):

        F_joint_j[Z] = M_subtree_j · g       M_subtree_j = Σ m_i for i ∈ S_j

    The masses are read from ``model.body_mass`` (the same values the AVBD
    solver uses for gravity integration), so any systematic error in the
    builder's mass computation would cancel between the expectation and the
    readout.  The test therefore probes the *wrench API* itself.

    The full-chain scene uses the same soft cable parameters as before this
    file's stiffening experiment; ``force_world[Z]`` is averaged over the last
    frame's substeps.  Flexible VBD cables plus an off-COM apple anchor can
    bias ``F_z`` away from the ideal scalar subtree identity, so the relative
    tolerance here is intentionally wider than for the minimal stem+apple tests.

    Expected subtrees (this test's fixed params):
        joint_stem_apple       : {apple}
        joint_spur_stem        : {stem_bodies} + {apple}
        joint_secondary_spur   : {spur_bodies} + {stem_bodies} + {apple}
        joint_primary_secondary: {secondary_bodies} + {spur_bodies} +
                                  {stem_bodies} + {apple}
    """
    scene = _make_full_chain_scene()

    mean_fz, _sim_dt = _settle_joint_fz_last_frame_mean(
        scene,
        (
            "joint_stem_apple",
            "joint_spur_stem",
            "joint_secondary_spur",
            "joint_primary_secondary",
        ),
        num_frames=420,
        substeps=10,
    )
    _check_no_ground_contact(scene, "test_subtree_weight_theorem_straight_chain")

    for label in [
        "joint_stem_apple",
        "joint_spur_stem",
        "joint_secondary_spur",
        "joint_primary_secondary",
    ]:
        fz = mean_fz[label]
        expected_Fz = _subtree_mass(scene, label) * G
        np.testing.assert_allclose(
            fz,
            expected_Fz,
            rtol=0.20,
            err_msg=(
                f"{label}: mean force_world[Z]={fz:.4f} N "
                f"should equal subtree_mass·g={expected_Fz:.4f} N (±20 %)"
            ),
        )


@pytest.mark.slow
def test_subtree_forces_are_cumulative_down_the_chain():
    """Deeper joints carry strictly more force than shallower joints.

    Corollary of the subtree theorem: each successive subtree strictly
    includes more bodies with positive mass, so the forces must be ordered:

        F[joint_stem_apple][Z]
        < F[joint_spur_stem][Z]
        < F[joint_secondary_spur][Z]
        < F[joint_primary_secondary][Z]
    """
    scene = _make_full_chain_scene()
    mean_fz, _sim_dt = _settle_joint_fz_last_frame_mean(
        scene,
        (
            "joint_stem_apple",
            "joint_spur_stem",
            "joint_secondary_spur",
            "joint_primary_secondary",
        ),
        num_frames=420,
        substeps=10,
    )

    fz = [
        mean_fz[label]
        for label in [
            "joint_stem_apple",
            "joint_spur_stem",
            "joint_secondary_spur",
            "joint_primary_secondary",
        ]
    ]
    for i in range(len(fz) - 1):
        assert fz[i] < fz[i + 1], (
            f"Expected F[{i}] < F[{i+1}]: {fz[i]:.4f} N < {fz[i+1]:.4f} N"
        )


# ---------------------------------------------------------------------------
# 4. Linearity / mass scaling
# ---------------------------------------------------------------------------


def test_apple_force_proportional_to_apple_mass():
    """Force ratio between two scenes equals model.body_mass ratio (±5 %).

    Because the subtree below joint_stem_apple is exactly the apple body:

        F_joint[Z] = model.body_mass[apple] · g

    which is linear in model.body_mass[apple].  We test this by building
    two scenes with different apple radii (density fixed) and verifying
    that the force ratio matches the model mass ratio.
    """
    r_small, r_large = 0.030, 0.045
    density = 850.0

    scene_s = _make_minimal_scene(apple_radius=r_small, apple_density=density)
    scene_l = _make_minimal_scene(apple_radius=r_large, apple_density=density)

    q_prev_s, dt_s = _settle(scene_s, num_frames=220, substeps=10)
    q_prev_l, dt_l = _settle(scene_l, num_frames=220, substeps=10)

    _check_no_ground_contact(scene_s, "small-apple")
    _check_no_ground_contact(scene_l, "large-apple")

    ws = _get_wrenches_by_label(scene_s, q_prev_s, dt_s)["joint_stem_apple"]
    wl = _get_wrenches_by_label(scene_l, q_prev_l, dt_l)["joint_stem_apple"]

    m_s = _apple_mass_from_model(scene_s)
    m_l = _apple_mass_from_model(scene_l)

    expected_ratio = m_l / m_s
    actual_ratio = float(wl.force_world[2]) / float(ws.force_world[2])

    np.testing.assert_allclose(
        actual_ratio,
        expected_ratio,
        rtol=0.05,
        err_msg=(
            f"Force ratio {actual_ratio:.4f} should match "
            f"model-mass ratio {expected_ratio:.4f} (linearity, ±5 %)"
        ),
    )


# ---------------------------------------------------------------------------
# 5. Apple mass matches analytic sphere (no shape double-counting)
# ---------------------------------------------------------------------------


def test_apple_body_mass_matches_analytic_sphere():
    """``model.body_mass[apple]`` equals (4/3)πr³ρ; joint F_z matches m·g.

    The apple collision sphere must not add a second mass via the builder
    default shape density.  At equilibrium, ``joint_stem_apple`` supports the
    full apple weight.
    """
    r, rho = 0.040, 850.0
    scene = _make_minimal_scene(apple_radius=r, apple_density=rho)

    model_mass = _apple_mass_from_model(scene)
    analytic_mass = (4.0 / 3.0) * math.pi * r**3 * rho

    np.testing.assert_allclose(
        model_mass,
        analytic_mass,
        rtol=1.0e-5,
        atol=1.0e-9,
        err_msg="Apple sphere must use density=0 so link mass is not doubled",
    )

    q_prev, sim_dt = _settle(scene, num_frames=220, substeps=10)
    _check_no_ground_contact(scene, "test_apple_body_mass_matches_analytic_sphere")

    w_by_label = _get_wrenches_by_label(scene, q_prev, sim_dt)
    fz = float(w_by_label["joint_stem_apple"].force_world[2])

    np.testing.assert_allclose(
        fz,
        analytic_mass * G,
        rtol=0.05,
        err_msg=(
            f"force_world[Z]={fz:.4f} N should match analytic_mass*g="
            f"{analytic_mass * G:.4f} N (±5 %)"
        ),
    )


def test_fruiting_ranges_fixture_chain_nearly_vertical():
    """Sampled primary with ``elevation_deg=-90`` stays within a few degrees of −Z."""
    import copy

    fs = _import_fs()
    ranges = copy.deepcopy(fs.load_ranges(_RANGES_FIXTURE))
    ranges["primary"]["elevation_deg"] = {"min": -90.0, "max": -90.0}
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cos_lim = math.cos(math.radians(4.0))
    for seed in range(12):
        p = fs.sample_params(ranges, seed=seed)
        for seg_name in ("primary", "secondary", "spur", "stem"):
            seg = getattr(p, seg_name)
            if seg is None:
                continue
            d = np.array(seg.direction, dtype=np.float64)
            n = float(np.linalg.norm(d))
            assert n > 1.0e-9
            d /= n
            # ``down`` is world −Z; parallel same-way segments have d·down ≈ +1.
            c = float(np.dot(d, down))
            assert c >= cos_lim, (
                f"seed={seed} {seg_name}: direction {seg.direction} "
                f"only {math.degrees(math.acos(np.clip(c, -1.0, 1.0))):.2f}° from -Z "
                f"(need ≤ 4°)"
            )


def _make_t_junction_scene(device: str = "cpu"):
    """Deterministic T topology: stiff horizontal primary, vertical spur→stem→apple."""
    fs = _import_fs()
    params = fs.FruitingSystemParams(
        primary=fs.RodParams(
            num_segments=2,
            length=0.20,
            radius=0.008,
            bend_stiffness=3000.0,
            bend_damping=10.0,
            stretch_stiffness=1.0e8,
            density=300.0,
            direction=(1.0, 0.0, 0.0),
        ),
        secondary=None,
        spur=fs.RodParams(
            num_segments=2,
            length=0.08,
            radius=0.004,
            bend_stiffness=600.0,
            bend_damping=5.0,
            stretch_stiffness=1.0e7,
            density=150.0,
            direction=(0.0, 0.0, -1.0),
        ),
        stem=fs.RodParams(
            num_segments=2,
            length=0.06,
            radius=0.003,
            bend_stiffness=100.0,
            bend_damping=5.0,
            stretch_stiffness=1.0e6,
            density=200.0,
            direction=(0.0, 0.0, -1.0),
        ),
        apple_radius=0.04,
        apple_density=700.0,
        topology=fs.TOPOLOGY_T_JUNCTION,
        spur_attach_fraction=0.5,
    )
    return fs._build_scene(
        params,
        base_pos=(0.0, 0.0, _BASE_Z),
        device=device,
        enable_self_collisions=False,
    )


def test_t_junction_stem_apple_at_equilibrium():
    """T branch below mid-span: ``joint_stem_apple`` Fz ≈ m_apple·g at quasi-static rest."""
    scene = _make_t_junction_scene()
    expected_mass = _apple_mass_from_model(scene)
    mean_fz, _sim_dt = _settle_joint_fz_last_frame_mean(
        scene,
        joint_labels=["joint_stem_apple"],
        num_frames=220,
        substeps=10,
    )
    np.testing.assert_allclose(
        mean_fz["joint_stem_apple"],
        expected_mass * G,
        rtol=0.08,
        err_msg=(
            f"joint_stem_apple Fz={mean_fz['joint_stem_apple']:.4f} N "
            f"expected ≈ {expected_mass * G:.4f} N"
        ),
    )
