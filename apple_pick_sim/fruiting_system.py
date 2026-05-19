"""Variational fruiting-system generator for the apple-picking simulation (P0).

Entry point
-----------
>>> from apple_pick_sim.fruiting_system import load_ranges, generate_scene, geometry_fingerprint
>>> ranges = load_ranges("apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json")
>>> scene = generate_scene(ranges, seed=42)
>>> fp = geometry_fingerprint(scene)

The generator builds a **primary → secondary → spur → stem → apple** chain from a
JSON range file and a deterministic **seed**, using the same Newton
``ModelBuilder`` rod/capsule + ``SolverVBD`` pattern as ``example_apple_stem.py``.
Fixtures: tests use ``apple_pick_sim/fixtures/fruiting_system_ranges_straight_rod_test.json``;
:mod:`example_fruiting_system` defaults to
``fruiting_system_ranges_example_variance.json``.

Any segment whose range entry is JSON ``null`` (Python ``None``) is omitted; the
remaining pieces connect in order with the first rod pinned at ``base_pos``. Use
``omit=...`` on :func:`sample_params` / :func:`generate_scene` to force segments off
without editing JSON; omission is applied **during** sampling so directions stay
consistent when skipping intermediate rods.

Fixed-joint wrenches: :func:`iter_fruiting_fixed_joint_indices` and
:func:`fixed_joint_wrenches_child_com_vbd` with ``joint_pairs`` from
:attr:`FruitingSystemScene.fruiting_fixed_joints`, or :func:`measure_fruiting_forces`.
See ``apple_pick_sim/vbd_fixed_joint_wrenches.py`` and ``docs/WRENCH_READOUT.md``.

M1 coupled cable: :func:`generate_coupled_cable_scene` builds the P0 fruiting tree on one
``Model`` plus a collision-equipped **gripper proxy** body for staggered robot coupling
(see ``docs/ROADMAP.md`` [M1]).

Staggered VBD ↔ MuJoCo force transfer
---------------------------------------
The cable tree and the robot arm live on **two separate** ``Model`` instances with
**different solvers** (``SolverVBD`` vs ``SolverMuJoCo``). They cannot share joints, so
interaction is bridged by a **gripper proxy** rigid body on the cable model that mirrors
the robot TCP pose. The proxy has collision geometry so VBD can resolve EE–apple contact
while the robot side is integrated by MuJoCo.

Per simulation substep the **staggered, one-step-lag** protocol (implemented in
``apple_pick_sim/coupled_fruiting.py``; kernels in ``apple_pick_sim/proxy_coupling.py``)
exchanges wrenches through the proxy pair registered via
:meth:`CoupledCableScene.proxy_registry`:

1. **MuJoCo ← VBD (lagged):** Apply the wrench harvested at the *previous* substep
   (``proxy_forces[robot_tcp]``) into ``robot_state.body_f[tcp]``. MuJoCo integrates
   the robot with this external load plus any user EE wrench.
2. **MuJoCo step:** Advance the robot ``Model`` by ``dt``.
3. **Kinematic sync (robot → proxy):** Copy TCP pose/velocity onto the cable-model
   proxy body. Subtract the *same* lagged wrench and a gravity term from the proxy's
   linear/angular velocity so VBD does not **double-integrate** forces MuJoCo already
   applied (``sync_proxy_state`` in ``proxy_coupling.py``).
4. **VBD step:** Collide and integrate the cable model (rods, apple, proxy contacts).
5. **VBD → MuJoCo (harvest for next step):** Infer the net spatial wrench the cable
   exerted on the proxy from the proxy velocity jump across the VBD step
   (``harvest_proxy_wrenches``; velocity-delta option 3 in the roadmap). Store in
   ``proxy_forces`` for step 1 of the *next* substep.

Spatial wrenches are **world frame**, linear then angular ``[N, N·m]``. The one-step lag
means coupling is explicit (not a fixed-point iteration within a step); stability depends
on timestep and proxy mass/stiffness. P0 fixed-joint readouts on the cable tree
(:func:`measure_fruiting_forces`) are independent of this proxy wrench channel.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.solvers

from apple_pick_sim.vbd_fixed_joint_wrenches import (
    FixedJointWrenchRecord,
    fixed_joint_wrenches_child_com_vbd,
    iter_fixed_joint_indices,
)


# Small rigid-joint linear/angular damping so inter-segment FIXED joints settle
# (matches equilibrium tests; avoids undamped oscillation in viewer FJ plots).
FRUITING_VBD_RIGID_JOINT_KD = 5.0e-4


def _pin_body_vbd_prescribed(builder: newton.ModelBuilder, body_id: int) -> None:
    """Make a body non-integrated by VBD (``inv_mass == 0``) while staying articulation-valid.

    Pose/velocity are overwritten each MuJoCo substep by staggered coupling sync.
    ``BodyFlags.KINEMATIC`` is not set because Newton only allows kinematic bodies on
    world-root joints; the apple remains a dynamic-flag child of the stem.
    """
    builder.body_mass[body_id] = 0.0
    builder.body_inv_mass[body_id] = 0.0
    builder.body_inv_inertia[body_id] = wp.mat33(0.0)


def make_fruiting_solver_vbd(model: newton.Model, **overrides: Any) -> newton.solvers.SolverVBD:
    """Construct the default :class:`~newton.solvers.SolverVBD` used for fruiting scenes.

    Extra keyword arguments override defaults (e.g. ``iterations=80``).
    """
    kwargs: dict[str, Any] = {
        "iterations": 50,
        "friction_epsilon": 0.1,
        "rigid_contact_k_start": 1.0e4,
        "rigid_joint_linear_k_start": 1.0e8,
        "rigid_joint_angular_k_start": 1.0e6,
        "rigid_joint_linear_kd": FRUITING_VBD_RIGID_JOINT_KD,
        "rigid_joint_angular_kd": FRUITING_VBD_RIGID_JOINT_KD,
    }
    kwargs.update(overrides)
    return newton.solvers.SolverVBD(model, **kwargs)


def example_collision_pipeline(
    model: newton.Model,
    args: Any | None = None,
    *,
    broad_phase: str | None = None,
    **kwargs: Any,
) -> Any:
    """Same collision pipeline factory as Newton examples (explicit broad-phase default).

    Pass the result to :func:`run_rollout` or :meth:`newton.Model.collide` so headless
    rollouts match ``example_fruiting_system.py`` (which uses
    ``newton.examples.create_collision_pipeline``).
    """
    import newton.examples

    return newton.examples.create_collision_pipeline(
        model, args=args, broad_phase=broad_phase, **kwargs
    )


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RodParams:
    """Sampled parameters for a single rod segment in the fruiting chain."""

    num_segments: int
    length: float
    radius: float
    bend_stiffness: float
    bend_damping: float
    stretch_stiffness: float
    density: float
    direction: tuple[float, float, float]  # unit vector in world space


@dataclasses.dataclass
class FruitingSystemParams:
    """All sampled parameters for a single fruiting-system instance.

    ``None`` on a rod field or on apple scalars means that piece is disabled
    (not built). At least one rod segment must be enabled.

    To turn off segments from code while keeping RNG and downstream directions
    consistent with JSON ``null``, use :func:`sample_params` ``omit=...`` rather
    than setting rod fields to ``None`` after a full sample (the latter can
    leave spur/stem directions wrong if an *intermediate* rod is removed).
    """

    primary: RodParams | None
    secondary: RodParams | None
    spur: RodParams | None
    stem: RodParams | None
    apple_radius: float | None
    apple_density: float | None


# ---------------------------------------------------------------------------
# Scene result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FruitingSystemScene:
    """Built Newton scene for one fruiting-system instance."""

    model: newton.Model
    state_0: Any
    state_1: Any
    control: Any
    solver: newton.solvers.SolverVBD
    params: FruitingSystemParams

    # Body indices per segment (length == params.<seg>.num_segments when present)
    primary_bodies: list[int]
    secondary_bodies: list[int]
    spur_bodies: list[int]
    stem_bodies: list[int]
    apple_body: int | None

    # Explicit (joint_index, label) for inter-rod and rod–apple FIXED joints (sorted by index)
    fruiting_fixed_joints: tuple[tuple[int, str], ...]
    # Cable / bend-stretch joints from each ``add_rod`` (articulation order, not including link joints)
    cable_joint_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class GripperProxyConfig:
    """Gripper proxy rigid body added on the cable ``Model`` for M1 coupling.

    The proxy is the VBD-side stand-in for the robot TCP: its pose tracks the MuJoCo
    body each substep, and contact/joint reactions on the proxy are harvested as wrenches
    fed back into ``robot_state.body_f`` on the next substep (see module docstring).
    Mass and box half-extents should match the robot TCP link built in
    ``coupled_fruiting.build_placeholder_tcp_robot_model`` so velocity-delta harvest
    uses consistent inertia.
    """

    mass: float = 5.0
    box_half_extents: tuple[float, float, float] = (0.05, 0.05, 0.05)
    label: str = "gripper_proxy"
    fix_to_apple: bool = True
    """If ``True``, weld the proxy to the apple with a FIXED joint at the exterior pole.

    Use with :func:`generate_coupled_cable_scene` and the coupled builders (default is ``False``).
    Pass ``GripperProxyConfig(fix_to_apple=True)`` for stem-harvest / apple co-teleport regression tests.
    """


@dataclasses.dataclass
class CoupledCableScene:
    """P0 fruiting cable ``Model`` plus gripper proxy body(ies) for two-``Model`` M1.

    Shares the same fields as :class:`FruitingSystemScene` so :func:`run_rollout` and
    :func:`measure_fruiting_forces` work unchanged. Pair with ``apple_pick_sim/coupled_fruiting.py``
    for the MuJoCo robot ``Model`` and staggered coupling loop (M1 Slice 2b).

    **Force path:** ``gripper_proxy_body`` is the VBD body that receives mirrored TCP
    state and participates in cable-side collision. :meth:`proxy_registry` maps the robot
    TCP index → ``gripper_proxy_body`` for ``sync_proxy_state`` / ``harvest_proxy_wrenches``.
    When ``gripper_proxy_apple_joint`` is set (``fix_to_apple``), the proxy is rigidly
    attached to the apple—a regression baseline without free proxy dynamics.
    """

    model: newton.Model
    state_0: Any
    state_1: Any
    control: Any
    solver: newton.solvers.SolverVBD
    params: FruitingSystemParams

    primary_bodies: list[int]
    secondary_bodies: list[int]
    spur_bodies: list[int]
    stem_bodies: list[int]
    apple_body: int | None

    fruiting_fixed_joints: tuple[tuple[int, str], ...]
    cable_joint_indices: tuple[int, ...]

    gripper_proxy_body: int
    gripper_proxy_config: GripperProxyConfig
    gripper_proxy_apple_joint: int | None = None
    """Apple-body-frame vector from apple COM to proxy COM when ``gripper_proxy_apple_joint`` is set."""
    gripper_proxy_offset_in_apple_frame: tuple[float, float, float] | None = None

    def proxy_registry(self, robot_body_id: int):
        """Map robot TCP body id → cable ``gripper_proxy_body`` for staggered wrench I/O.

        Used by :meth:`apple_pick_sim.coupled_fruiting.CoupledFruitingScene.coupled_substep`:
        harvest writes ``proxy_forces[robot_body_id]``; sync reads the same slot when
        undoing the lagged coupling velocity on the proxy.
        """
        from apple_pick_sim.proxy_coupling import ProxyBodyRegistry

        return ProxyBodyRegistry.from_mapping({robot_body_id: self.gripper_proxy_body})


@dataclasses.dataclass
class _FruitingChainArtifacts:
    """Mutable builder state for the fruiting rod chain before ``finalize``."""

    cable_joint_indices: list[int]
    fruiting_fixed_joints: list[tuple[int, str]]
    seg_bodies: dict[str, list[int]]
    apple_body: int | None
    all_joints: list[int]
    chain_bodies: list[int]
    proxy_placement_origin: wp.vec3
    proxy_placement_dir: wp.vec3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ranges(path: str | Path) -> dict:
    """Load a fruiting-system range JSON file.

    Args:
        path: Path to the JSON range file.

    Returns:
        Dict with keys ``primary``, ``secondary``, ``spur``, ``stem``, ``apple``.
        A rod or ``apple`` entry may be JSON ``null`` (``None`` in Python) to omit that
        piece from sampling and scene construction (at least one rod must remain).
    """
    with open(path) as f:
        data = json.load(f)
    _validate_ranges(data)
    return data


def _coerce_omit(omit: Collection[str] | None) -> frozenset[str]:
    """Validate ``omit`` keys for :func:`sample_params` / :func:`generate_scene`."""
    if omit is None:
        return frozenset()
    allowed = frozenset({"primary", "secondary", "spur", "stem", "apple"})
    o = frozenset(omit)
    extra = o - allowed
    if extra:
        raise ValueError(
            f"omit contains unknown keys: {sorted(extra)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return o


def sample_params(
    ranges: dict,
    seed: int,
    *,
    omit: Collection[str] | None = None,
) -> FruitingSystemParams:
    """Sample fruiting-system parameters from ``ranges`` deterministically via ``seed``.

    Skips sampling for any rod segment (or apple) whose range entry is ``None``.
    Names in ``omit`` (e.g. ``{"secondary", "apple"}``) force that piece off even
    when the range entry is present—useful for toggling topology from code without
    editing JSON. Omission is applied **during** sampling so ``parent_dir`` for
    spur/stem matches omitting intermediate rods (same as setting that range to
    ``null``).

    When both **primary** and **secondary** are enabled, enforces
    ``primary.bend_stiffness >= secondary.bend_stiffness``.

    Args:
        ranges: Range dict as returned by :func:`load_ranges`.
        seed: Integer seed for the RNG.
        omit: Optional set of segment names to force to ``None`` in the result.

    Returns:
        A :class:`FruitingSystemParams` instance.

    Raises:
        ValueError: If ``omit`` contains unknown keys, or no rod segment remains enabled.
    """
    rng = np.random.default_rng(seed)
    omit_set = _coerce_omit(omit)

    def _s(seg_ranges: dict, key: str) -> float:
        return float(rng.uniform(seg_ranges[key]["min"], seg_ranges[key]["max"]))

    def _si(seg_ranges: dict, key: str) -> int:
        return int(rng.integers(seg_ranges[key]["min"], seg_ranges[key]["max"] + 1))

    # Parent direction for lateral segments: last built rod, or +X when primary is off.
    parent_dir: tuple[float, float, float] = (1.0, 0.0, 0.0)

    primary: RodParams | None = None
    pr = ranges.get("primary")
    if pr is not None and "primary" not in omit_set:
        primary_az = _s(pr, "azimuth_deg")
        primary_el = _s(pr, "elevation_deg")
        primary_dir = _direction_from_angles(primary_az, primary_el)
        primary_bend = _s(pr, "bend_stiffness")
        primary = RodParams(
            num_segments=max(2, _si(pr, "num_segments")),
            length=_s(pr, "length"),
            radius=_s(pr, "radius"),
            bend_stiffness=primary_bend,
            bend_damping=_s(pr, "bend_damping"),
            stretch_stiffness=_s(pr, "stretch_stiffness"),
            density=_s(pr, "density"),
            direction=primary_dir,
        )
        parent_dir = primary.direction

    secondary: RodParams | None = None
    sr = ranges.get("secondary")
    if sr is not None and "secondary" not in omit_set:
        if primary is not None:
            secondary_bend_max = min(sr["bend_stiffness"]["max"], primary.bend_stiffness)
            secondary_bend_min = min(sr["bend_stiffness"]["min"], secondary_bend_max)
            secondary_bend = float(rng.uniform(secondary_bend_min, secondary_bend_max))
        else:
            secondary_bend = float(
                rng.uniform(sr["bend_stiffness"]["min"], sr["bend_stiffness"]["max"])
            )
        secondary_el_delta = _s(sr, "elevation_delta_deg")
        secondary_lat_delta = _s(sr, "lateral_delta_deg")
        secondary_dir = _deflect_direction(parent_dir, secondary_el_delta, secondary_lat_delta)
        secondary = RodParams(
            num_segments=max(2, _si(sr, "num_segments")),
            length=_s(sr, "length"),
            radius=_s(sr, "radius"),
            bend_stiffness=secondary_bend,
            bend_damping=_s(sr, "bend_damping"),
            stretch_stiffness=_s(sr, "stretch_stiffness"),
            density=_s(sr, "density"),
            direction=secondary_dir,
        )
        parent_dir = secondary.direction

    spur: RodParams | None = None
    spr = ranges.get("spur")
    if spr is not None and "spur" not in omit_set:
        spur_el_delta = _s(spr, "elevation_delta_deg")
        spur_lat_delta = _s(spr, "lateral_delta_deg")
        spur_dir = _deflect_direction(parent_dir, spur_el_delta, spur_lat_delta)
        spur = RodParams(
            num_segments=max(2, _si(spr, "num_segments")),
            length=_s(spr, "length"),
            radius=_s(spr, "radius"),
            bend_stiffness=_s(spr, "bend_stiffness"),
            bend_damping=_s(spr, "bend_damping"),
            stretch_stiffness=_s(spr, "stretch_stiffness"),
            density=_s(spr, "density"),
            direction=spur_dir,
        )
        parent_dir = spur.direction

    stem: RodParams | None = None
    stem_r = ranges.get("stem")
    if stem_r is not None and "stem" not in omit_set:
        stem_el_delta = _s(stem_r, "elevation_delta_deg")
        stem_lat_delta = _s(stem_r, "lateral_delta_deg")
        stem_dir = _deflect_direction(parent_dir, stem_el_delta, stem_lat_delta)
        stem = RodParams(
            num_segments=max(2, _si(stem_r, "num_segments")),
            length=_s(stem_r, "length"),
            radius=_s(stem_r, "radius"),
            bend_stiffness=_s(stem_r, "bend_stiffness"),
            bend_damping=_s(stem_r, "bend_damping"),
            stretch_stiffness=_s(stem_r, "stretch_stiffness"),
            density=_s(stem_r, "density"),
            direction=stem_dir,
        )

    apple_radius: float | None = None
    apple_density: float | None = None
    ar = ranges.get("apple")
    if ar is not None and "apple" not in omit_set:
        apple_radius = _s(ar, "radius")
        apple_density = _s(ar, "density")

    if not any((primary, secondary, spur, stem)):
        raise ValueError(
            "At least one rod segment must be enabled (check ranges and omit)."
        )

    return FruitingSystemParams(
        primary=primary,
        secondary=secondary,
        spur=spur,
        stem=stem,
        apple_radius=apple_radius,
        apple_density=apple_density,
    )


def params_fingerprint(params: FruitingSystemParams) -> dict:
    """Return a dict of scalar summaries from sampled params (no Newton model needed).

    This is cheaper than building the full scene and useful for quick determinism checks.
    Fields for disabled segments are ``None``.
    """
    p, s, sp, st = params.primary, params.secondary, params.spur, params.stem
    return {
        "primary_num_segments": None if p is None else p.num_segments,
        "primary_length": None if p is None else round(p.length, 9),
        "primary_radius": None if p is None else round(p.radius, 9),
        "primary_bend_stiffness": None if p is None else round(p.bend_stiffness, 6),
        "secondary_num_segments": None if s is None else s.num_segments,
        "secondary_length": None if s is None else round(s.length, 9),
        "secondary_radius": None if s is None else round(s.radius, 9),
        "secondary_bend_stiffness": None if s is None else round(s.bend_stiffness, 6),
        "spur_num_segments": None if sp is None else sp.num_segments,
        "spur_length": None if sp is None else round(sp.length, 9),
        "stem_num_segments": None if st is None else st.num_segments,
        "stem_length": None if st is None else round(st.length, 9),
        "apple_radius": None if params.apple_radius is None else round(params.apple_radius, 9),
        "apple_density": None if params.apple_density is None else round(params.apple_density, 6),
        "primary_dir_x": None if p is None else round(p.direction[0], 6),
        "secondary_dir_x": None if s is None else round(s.direction[0], 6),
        "spur_dir_z": None if sp is None else round(sp.direction[2], 6),
        "stem_dir_z": None if st is None else round(st.direction[2], 6),
    }


def generate_scene(
    ranges: dict,
    seed: int,
    *,
    base_pos: tuple[float, float, float] = (0.0, 0.0, 3.0),
    device: str | None = None,
    omit: Collection[str] | None = None,
    enable_self_collisions: bool = True,
) -> FruitingSystemScene:
    """Generate a Newton scene for a fruiting system from ``ranges`` and ``seed``.

    Args:
        ranges: Range dict as returned by :func:`load_ranges`.
        seed: Deterministic integer seed.
        base_pos: World-space position of the first rod segment's base (pinned).
        device: Warp device string (e.g. ``"cpu"``, ``"cuda:0"``). Defaults to
            ``"cpu"`` for deterministic headless use.
        omit: Forwarded to :func:`sample_params` to force segments off without editing JSON.
        enable_self_collisions: If ``False``, register shape collision filter pairs between
            every pair of distinct chain bodies (primary through apple), so the articulation
            does not self-collide; ground contact is unchanged. If ``True`` (default), only
            Newton joint parent/child filters apply, so non-adjacent chain links may collide.

    Returns:
        A :class:`FruitingSystemScene` ready to simulate.
    """
    if device is None:
        device = "cpu"

    params = sample_params(ranges, seed, omit=omit)
    return _build_scene(
        params,
        base_pos=base_pos,
        device=device,
        enable_self_collisions=enable_self_collisions,
    )


def generate_coupled_cable_scene(
    ranges: dict,
    seed: int,
    *,
    base_pos: tuple[float, float, float] = (0.0, 0.0, 3.0),
    device: str | None = None,
    omit: Collection[str] | None = None,
    enable_self_collisions: bool = True,
    gripper_proxy: GripperProxyConfig | None = None,
) -> CoupledCableScene:
    """Build the VBD cable ``Model``: P0 fruiting tree + collision-equipped gripper proxy.

    This is **Model B** in the two-``Model`` stack: ``SolverVBD`` integrates rods,
    apple, and the gripper proxy. The robot arm (**Model A**, ``SolverMuJoCo``) is built
    separately in ``coupled_fruiting.build_coupled_fruiting_placeholder``; coupling forces
    never cross ``Model`` boundaries directly—they flow through the proxy registry and
    per-substep ``proxy_forces`` buffer (see module docstring).

    The proxy body is a free rigid body (or FIXED to the apple when
    ``gripper_proxy.fix_to_apple``) placed near the apple for EE–apple contact during
    staggered ``SolverMuJoCo`` + ``SolverVBD`` coupling. P0 body indices and fixed-joint
    metadata match :func:`generate_scene` for the same ``(ranges, seed)``.

    Args:
        ranges: Range dict as returned by :func:`load_ranges`.
        seed: Deterministic integer seed.
        base_pos: World-space position of the first rod segment's base (pinned).
        device: Warp device string. Defaults to ``"cpu"``.
        omit: Forwarded to :func:`sample_params`.
        enable_self_collisions: Same semantics as :func:`generate_scene` (proxy is excluded
            from intra-chain filter pairs so it can contact the apple).
        gripper_proxy: Proxy mass/shape/placement options; defaults to :class:`GripperProxyConfig`.

    Returns:
        A :class:`CoupledCableScene` ready for VBD-only rollouts or robot coupling.
    """
    if device is None:
        device = "cpu"

    params = sample_params(ranges, seed, omit=omit)
    proxy_cfg = gripper_proxy if gripper_proxy is not None else GripperProxyConfig()
    return _build_coupled_cable_scene(
        params,
        base_pos=base_pos,
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=proxy_cfg,
    )


def geometry_fingerprint(scene: FruitingSystemScene) -> dict:
    """Return a dict of scalar geometry summaries extracted from a built scene.

    The same ``(ranges, seed)`` always produces an identical fingerprint.
    Different seeds (with broad enough ranges) produce distinct fingerprints.
    Omitted or empty segments yield ``None`` for the corresponding position keys.
    """
    fp = params_fingerprint(scene.params)

    # Augment with body-count facts extracted from the built model
    fp["total_body_count"] = scene.model.body_count
    fp["primary_body_count"] = len(scene.primary_bodies)
    fp["secondary_body_count"] = len(scene.secondary_bodies)
    fp["spur_body_count"] = len(scene.spur_bodies)
    fp["stem_body_count"] = len(scene.stem_bodies)

    # Key world-space positions from the initial state
    body_q = scene.state_0.body_q.to("cpu").numpy()

    def _pos(body_idx: int) -> tuple[float, float, float]:
        p = body_q[body_idx]
        return (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5))

    if scene.primary_bodies:
        fp["primary_base_pos"] = _pos(scene.primary_bodies[0])
        fp["primary_tip_pos"] = _pos(scene.primary_bodies[-1])
    else:
        fp["primary_base_pos"] = None
        fp["primary_tip_pos"] = None
    fp["secondary_tip_pos"] = (
        _pos(scene.secondary_bodies[-1]) if scene.secondary_bodies else None
    )
    fp["spur_tip_pos"] = _pos(scene.spur_bodies[-1]) if scene.spur_bodies else None
    fp["stem_tip_pos"] = _pos(scene.stem_bodies[-1]) if scene.stem_bodies else None
    fp["apple_pos"] = _pos(scene.apple_body) if scene.apple_body is not None else None

    return fp


def geometry_fingerprint_coupled(scene: CoupledCableScene) -> dict:
    """Like :func:`geometry_fingerprint` plus gripper-proxy body index and position."""
    fp = geometry_fingerprint(scene)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    p = body_q[scene.gripper_proxy_body]
    fp["gripper_proxy_body"] = scene.gripper_proxy_body
    fp["gripper_proxy_pos"] = (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5))
    return fp


def run_rollout(
    scene: FruitingSystemScene,
    num_steps: int = 20,
    sim_substeps: int = 10,
    fps: float = 60.0,
    *,
    collision_pipeline: Any | None = None,
) -> None:
    """Advance a scene's simulation in-place for ``num_steps`` frames.

    Args:
        scene: Scene returned by :func:`generate_scene`.
        num_steps: Number of rendered frames to simulate.
        sim_substeps: Sub-steps per frame.
        fps: Frames per second (determines per-frame dt).
        collision_pipeline: Optional pipeline from :func:`example_collision_pipeline`
            (same as ``example_fruiting_system``). If ``None``, uses the model's default
            pipeline (first :meth:`~newton.Model.collide` initializes it).
    """
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / sim_substeps

    for _ in range(num_steps):
        for _ in range(sim_substeps):
            scene.state_0.clear_forces()
            if collision_pipeline is None:
                contacts = scene.model.collide(scene.state_0)
            else:
                contacts = scene.model.collide(
                    scene.state_0, collision_pipeline=collision_pipeline
                )
            scene.solver.step(scene.state_0, scene.state_1, scene.control, contacts, sim_dt)
            scene.state_0, scene.state_1 = scene.state_1, scene.state_0


def iter_fruiting_fixed_joint_indices(scene: FruitingSystemScene) -> list[tuple[int, str]]:
    """Return ``(joint_index, label)`` for FIXED joints recorded on this scene."""
    return list(scene.fruiting_fixed_joints)


def measure_fruiting_forces(
    scene: FruitingSystemScene,
    body_q: Any,
    body_q_prev: Any,
    dt: float,
    *,
    control: newton.Control | None = None,
) -> dict[str, Any]:
    """Return fixed-joint wrench records plus cable joint index metadata.

    Cable **scalar forces** are not computed here; penalty-style cable metrics follow
    the ``example_apple_stem.py`` pattern (joint displacement × stiffness) when needed.
    """
    fixed = fixed_joint_wrenches_child_com_vbd(
        scene.model,
        scene.solver,
        body_q=body_q,
        body_q_prev=body_q_prev,
        dt=dt,
        control=control,
        joint_pairs=list(scene.fruiting_fixed_joints),
    )
    return {
        "fixed_joints": fixed,
        "cable_joint_indices": scene.cable_joint_indices,
    }


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _apply_all_chain_collision_filters(
    builder: newton.ModelBuilder,
    chain_body_indices: list[int],
) -> None:
    """Add shape collision filter pairs for every distinct pair of chain bodies."""
    bodies = chain_body_indices
    for i, body1 in enumerate(bodies):
        for body2 in bodies[i + 1 :]:
            for shape1 in builder.body_shapes.get(body1, []):
                for shape2 in builder.body_shapes.get(body2, []):
                    builder.add_shape_collision_filter_pair(shape1, shape2)



def _new_fruiting_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e2
    builder.default_shape_cfg.kd = 1.0e1
    builder.default_shape_cfg.mu = 0.8
    return builder


def _build_fruiting_chain_into_builder(
    builder: newton.ModelBuilder,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
) -> _FruitingChainArtifacts:
    """Populate rod chain + apple on ``builder`` (no articulation / finalize yet)."""
    if not any((params.primary, params.secondary, params.spur, params.stem)):
        raise ValueError(
            "At least one rod segment (primary, secondary, spur, or stem) must be non-None."
        )

    cable_joint_indices: list[int] = []
    fruiting_fixed_joints: list[tuple[int, str]] = []

    origin = wp.vec3(*base_pos)

    rod_specs: list[tuple[str, RodParams]] = [
        (n, rp)
        for n, rp in (
            ("primary", params.primary),
            ("secondary", params.secondary),
            ("spur", params.spur),
            ("stem", params.stem),
        )
        if rp is not None
    ]

    seg_bodies: dict[str, list[int]] = {
        "primary": [],
        "secondary": [],
        "spur": [],
        "stem": [],
    }

    all_joints: list[int] = []
    prev_bodies: list[int] | None = None
    prev_name: str | None = None
    prev_rod: RodParams | None = None
    prev_points: list[wp.vec3] | None = None
    prev_quats: list[wp.quat] | None = None

    for name, rod in rod_specs:
        start = origin if prev_bodies is None else prev_points[-1]
        points, quats = _make_rod_geometry(
            start, rod.direction, rod.length, rod.num_segments
        )
        rod_cfg = builder.ShapeConfig(density=rod.density)
        bodies, joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=rod.radius,
            cfg=rod_cfg,
            bend_stiffness=rod.bend_stiffness,
            bend_damping=rod.bend_damping,
            stretch_stiffness=rod.stretch_stiffness,
            stretch_damping=0.0,
            wrap_in_articulation=False,
            label=name,
        )
        all_joints.extend(joints)
        cable_joint_indices.extend(joints)

        if prev_bodies is None:
            builder.body_mass[bodies[0]] = 0.0
            builder.body_inv_mass[bodies[0]] = 0.0
        else:
            assert prev_rod is not None and prev_name is not None
            parent_seg_len = prev_rod.length / prev_rod.num_segments
            j_link = _connect_rod_tip_to_base(
                builder,
                parent_body=prev_bodies[-1],
                parent_seg_length=parent_seg_len,
                child_body=bodies[0],
                key=f"joint_{prev_name}_{name}",
            )
            all_joints.append(j_link)
            fruiting_fixed_joints.append((j_link, f"joint_{prev_name}_{name}"))

        seg_bodies[name] = bodies
        prev_bodies = bodies
        prev_name = name
        prev_rod = rod
        prev_points = points
        prev_quats = quats

    assert prev_bodies is not None and prev_rod is not None
    assert prev_points is not None and prev_quats is not None
    last_name = prev_name
    assert last_name is not None
    tip_seg_len = prev_rod.length / prev_rod.num_segments

    stem_tip_pt = prev_points[-1]
    stem_base_pt = prev_points[-2]
    last_seg_dir = wp.normalize(stem_tip_pt - stem_base_pt)
    proxy_placement_origin = stem_tip_pt
    proxy_placement_dir = last_seg_dir

    apple_body: int | None = None
    if params.apple_radius is not None and params.apple_density is not None:
        apple_pos = stem_tip_pt + last_seg_dir * params.apple_radius
        proxy_placement_origin = apple_pos
        apple_mass = (4.0 / 3.0) * math.pi * params.apple_radius**3 * params.apple_density
        apple_quat = wp.quat_identity()
        apple_body = builder.add_link(
            xform=wp.transform(apple_pos, apple_quat),
            mass=apple_mass,
            label="apple",
        )
        apple_shape_cfg = builder.default_shape_cfg.copy()
        apple_shape_cfg.density = 0.0
        builder.add_shape_sphere(
            body=apple_body, radius=params.apple_radius, cfg=apple_shape_cfg
        )
        j_st2apple = _connect_rod_tip_to_apple(
            builder,
            stem_tip_body=prev_bodies[-1],
            stem_tip_quat=prev_quats[-1],
            stem_seg_length=tip_seg_len,
            segment_dir_world=last_seg_dir,
            apple_radius=params.apple_radius,
            apple_body=apple_body,
            apple_quat=apple_quat,
            key=f"joint_{last_name}_apple",
        )
        all_joints.append(j_st2apple)
        fruiting_fixed_joints.append((j_st2apple, f"joint_{last_name}_apple"))

    chain_bodies: list[int] = (
        seg_bodies["primary"]
        + seg_bodies["secondary"]
        + seg_bodies["spur"]
        + seg_bodies["stem"]
    )
    if apple_body is not None:
        chain_bodies.append(apple_body)
    
    return _FruitingChainArtifacts(
        cable_joint_indices=cable_joint_indices,
        fruiting_fixed_joints=fruiting_fixed_joints,
        seg_bodies=seg_bodies,
        apple_body=apple_body,
        all_joints=all_joints,
        chain_bodies=chain_bodies,
        proxy_placement_origin=proxy_placement_origin,
        proxy_placement_dir=proxy_placement_dir,
    )


def _finalize_fruiting_builder(
    builder: newton.ModelBuilder,
    artifacts: _FruitingChainArtifacts,
    *,
    device: str,
    enable_self_collisions: bool,
) -> newton.Model:
    all_joints = sorted(artifacts.all_joints)
    builder.add_articulation(all_joints)
    if not enable_self_collisions:
        _apply_all_chain_collision_filters(builder, artifacts.chain_bodies)
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _add_gripper_proxy(
    builder: newton.ModelBuilder,
    artifacts: _FruitingChainArtifacts,
    config: GripperProxyConfig,
    *,
    apple_radius: float | None,
) -> tuple[int, int | None, tuple[float, float, float] | None]:
    """Add gripper proxy link, shape, and joint(s).

    Returns ``(body_id, apple_fixed_joint, proxy_offset_in_apple_frame)``.

    Default: ``add_joint_free`` so the proxy is a dynamic body whose pose/velocity are
    overwritten each MuJoCo substep by ``sync_proxy_state`` (not integrated from cable
    gravity alone). With ``fix_to_apple``, a FIXED joint welds the proxy to the apple
    at the exterior pole (same placement as the free proxy, via ``parent_xform`` /
    ``child_xform``). The apple and proxy use ``inv_mass == 0`` so VBD does not integrate
    them; staggered coupling teleports their poses from the robot TCP each substep while
    the stem supplies the harvested wrench.
    """
    hx, hy, hz = config.box_half_extents
    clearance = max(hx, hy, hz)

    if apple_radius is not None:
        proxy_pos = artifacts.proxy_placement_origin + artifacts.proxy_placement_dir * (
            apple_radius + clearance
        )
    elif config.fix_to_apple:
        if artifacts.apple_body is None:
            raise ValueError("fix_to_apple requires an apple body in the scene")
        proxy_pos = artifacts.proxy_placement_origin
    else:
        proxy_pos = artifacts.proxy_placement_origin + artifacts.proxy_placement_dir * clearance
  
    proxy_body = builder.add_link(
        xform=wp.transform(proxy_pos, wp.quat_identity()),
        mass=config.mass,
        label=config.label,
    )
    proxy_shape_cfg = builder.default_shape_cfg.copy()
    proxy_shape_cfg.density = 0.0
    builder.add_shape_box(
        body=proxy_body,
        hx=hx,
        hy=hy,
        hz=hz,
        cfg=proxy_shape_cfg,
    )

    apple_fixed_joint: int | None = None
    proxy_offset_in_apple_frame: tuple[float, float, float] | None = None
    if config.fix_to_apple:
        assert artifacts.apple_body is not None
        if apple_radius is not None:
            d = artifacts.proxy_placement_dir
            surface_on_apple = d * apple_radius
            proxy_face_toward_apple = d * (-clearance)
            parent_xform = wp.transform(surface_on_apple, wp.quat_identity())
            child_xform = wp.transform(proxy_face_toward_apple, wp.quat_identity())
            off = d * (apple_radius + clearance)
            proxy_offset_in_apple_frame = (float(off[0]), float(off[1]), float(off[2]))
        else:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
            child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        apple_fixed_joint = builder.add_joint_fixed(
            parent=artifacts.apple_body,
            child=proxy_body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            label="joint_apple_gripper_proxy",
        )
        artifacts.all_joints.append(apple_fixed_joint)
        artifacts.fruiting_fixed_joints.append(
            (apple_fixed_joint, "joint_apple_gripper_proxy")
        )
        _pin_body_vbd_prescribed(builder, artifacts.apple_body)
        _pin_body_vbd_prescribed(builder, proxy_body)
    else:
        j_free = builder.add_joint_free(parent=-1, child=proxy_body)
        artifacts.all_joints.append(j_free)

    return proxy_body, apple_fixed_joint, proxy_offset_in_apple_frame


def _scene_states_from_model(model: newton.Model) -> tuple[Any, Any, Any, newton.solvers.SolverVBD]:
    solver = make_fruiting_solver_vbd(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    return state_0, state_1, control, solver


def _build_scene(
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
    device: str,
    *,
    enable_self_collisions: bool = True,
) -> FruitingSystemScene:
    """Build a Newton ModelBuilder scene from sampled params."""
    builder = _new_fruiting_builder()
    artifacts = _build_fruiting_chain_into_builder(builder, params, base_pos)
    model = _finalize_fruiting_builder(
        builder, artifacts, device=device, enable_self_collisions=enable_self_collisions
    )
    artifacts.fruiting_fixed_joints.sort(key=lambda p: p[0])
    state_0, state_1, control, solver = _scene_states_from_model(model)

    return FruitingSystemScene(
        model=model,
        state_0=state_0,
        state_1=state_1,
        control=control,
        solver=solver,
        params=params,
        primary_bodies=artifacts.seg_bodies["primary"],
        secondary_bodies=artifacts.seg_bodies["secondary"],
        spur_bodies=artifacts.seg_bodies["spur"],
        stem_bodies=artifacts.seg_bodies["stem"],
        apple_body=artifacts.apple_body,
        fruiting_fixed_joints=tuple(artifacts.fruiting_fixed_joints),
        cable_joint_indices=tuple(artifacts.cable_joint_indices),
    )


def _build_coupled_cable_scene(
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
    device: str,
    *,
    enable_self_collisions: bool,
    gripper_proxy: GripperProxyConfig,
) -> CoupledCableScene:
    builder = _new_fruiting_builder()
    artifacts = _build_fruiting_chain_into_builder(builder, params, base_pos)
    proxy_body, proxy_apple_joint, proxy_offset_in_apple = _add_gripper_proxy(
        builder,
        artifacts,
        gripper_proxy,
        apple_radius=params.apple_radius,
    )
    model = _finalize_fruiting_builder(
        builder, artifacts, device=device, enable_self_collisions=enable_self_collisions
    )
    artifacts.fruiting_fixed_joints.sort(key=lambda p: p[0])
    state_0, state_1, control, solver = _scene_states_from_model(model)

    return CoupledCableScene(
        model=model,
        state_0=state_0,
        state_1=state_1,
        control=control,
        solver=solver,
        params=params,
        primary_bodies=artifacts.seg_bodies["primary"],
        secondary_bodies=artifacts.seg_bodies["secondary"],
        spur_bodies=artifacts.seg_bodies["spur"],
        stem_bodies=artifacts.seg_bodies["stem"],
        apple_body=artifacts.apple_body,
        fruiting_fixed_joints=tuple(artifacts.fruiting_fixed_joints),
        cable_joint_indices=tuple(artifacts.cable_joint_indices),
        gripper_proxy_body=proxy_body,
        gripper_proxy_config=gripper_proxy,
        gripper_proxy_apple_joint=proxy_apple_joint,
        gripper_proxy_offset_in_apple_frame=proxy_offset_in_apple,
    )


def _make_rod_geometry(
    start: wp.vec3,
    direction: tuple[float, float, float],
    length: float,
    num_segments: int,
) -> tuple[list[wp.vec3], list[wp.quat]]:
    """Build straight-rod positions and per-segment quaternions."""
    d = wp.normalize(wp.vec3(*direction))
    points = [start + d * (length * i / num_segments) for i in range(num_segments + 1)]
    quats = newton.utils.create_parallel_transport_cable_quaternions(points)
    return points, quats


def _connect_rod_tip_to_base(
    builder: newton.ModelBuilder,
    parent_body: int,
    parent_seg_length: float,
    child_body: int,
    key: str,
) -> int:
    """Add a fixed joint connecting the tip of one rod to the base of the next.

    In rod convention, the tip of a segment body is at local (0, 0, seg_length).
    The base of the next segment body is at local (0, 0, 0).
    """
    return builder.add_joint_fixed(
        parent=parent_body,
        child=child_body,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, parent_seg_length), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        label=key,
    )


def _connect_rod_tip_to_apple(
    builder: newton.ModelBuilder,
    stem_tip_body: int,
    stem_tip_quat: wp.quat,
    stem_seg_length: float,
    segment_dir_world: wp.vec3,
    apple_radius: float,
    apple_body: int,
    apple_quat: wp.quat,
    key: str,
) -> int:
    """Add a fixed joint attaching the apple to the distal end of the last rod segment.

    The parent anchor is the capsule tip in the stem-tip body's local frame
    (same construction as ``example_apple_stem.py``). The child anchor is offset
    from the apple COM toward the stem by one radius so the stem-side pole of
    the sphere meets the rod tip instead of burying the COM at the junction.
    The child offset is expressed in the apple body's local frame (rotated by
    ``apple_quat`` from world).
    """
    segment_vector_world = segment_dir_world * stem_seg_length
    parent_local_anchor = wp.quat_rotate(
        wp.quat_inverse(stem_tip_quat), segment_vector_world
    )
    stem_toward_apple_world = segment_dir_world * (-apple_radius)
    child_local_anchor = wp.quat_rotate(wp.quat_inverse(apple_quat), stem_toward_apple_world)
    return builder.add_joint_fixed(
        parent=stem_tip_body,
        child=apple_body,
        parent_xform=wp.transform(parent_local_anchor, wp.quat_identity()),
        child_xform=wp.transform(child_local_anchor, wp.quat_identity()),
        label=key,
    )


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------


def _direction_from_angles(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Convert azimuth + elevation angles (degrees) to a unit direction vector."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_el = math.cos(el)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el))


def _deflect_direction(
    parent_dir: tuple[float, float, float],
    elevation_delta_deg: float,
    lateral_delta_deg: float,
) -> tuple[float, float, float]:
    """Deflect a parent direction by elevation and lateral angle deltas.

    The parent direction's azimuth is shifted by ``lateral_delta_deg`` and
    its elevation is increased by ``elevation_delta_deg``, clamped to [-90, 90] deg.
    """
    dx, dy, dz = float(parent_dir[0]), float(parent_dir[1]), float(parent_dir[2])
    az = math.atan2(dy, dx) + math.radians(lateral_delta_deg)
    el_parent = math.asin(max(-1.0, min(1.0, dz)))
    el_new = el_parent + math.radians(elevation_delta_deg)
    el_new = max(-math.pi / 2.0, min(math.pi / 2.0, el_new))
    cos_el = math.cos(el_new)
    return (cos_el * math.cos(az), cos_el * math.sin(az), math.sin(el_new))


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_ranges(data: dict) -> None:
    """Raise ValueError if the range dict is missing required keys or has invalid bounds."""
    required_segments = ("primary", "secondary", "spur", "stem", "apple")
    for seg in required_segments:
        if seg not in data:
            raise ValueError(f"Missing segment '{seg}' in range file")

    rod_required = (
        "num_segments",
        "length",
        "radius",
        "bend_stiffness",
        "bend_damping",
        "stretch_stiffness",
        "density",
    )
    for seg in ("primary", "secondary", "spur", "stem"):
        seg_data = data[seg]
        if seg_data is None:
            continue
        if not isinstance(seg_data, dict):
            raise ValueError(f"Segment '{seg}' must be a JSON object or null")
        for key in rod_required:
            if key not in seg_data:
                raise ValueError(f"Missing key '{key}' in segment '{seg}'")
            rng = seg_data[key]
            if "min" not in rng or "max" not in rng:
                raise ValueError(f"Range {seg}.{key} must have 'min' and 'max'")
            if rng["min"] > rng["max"]:
                raise ValueError(
                    f"Range {seg}.{key}: min ({rng['min']}) > max ({rng['max']})"
                )

        if seg == "primary":
            for key in ("azimuth_deg", "elevation_deg"):
                if key not in seg_data:
                    raise ValueError(f"Missing key '{key}' in segment 'primary'")
                rng = seg_data[key]
                if "min" not in rng or "max" not in rng:
                    raise ValueError(f"Range primary.{key} must have 'min' and 'max'")
                if rng["min"] > rng["max"]:
                    raise ValueError(
                        f"Range primary.{key}: min ({rng['min']}) > max ({rng['max']})"
                    )
        else:
            for key in ("elevation_delta_deg", "lateral_delta_deg"):
                if key not in seg_data:
                    raise ValueError(f"Missing key '{key}' in segment '{seg}'")
                rng = seg_data[key]
                if "min" not in rng or "max" not in rng:
                    raise ValueError(f"Range {seg}.{key} must have 'min' and 'max'")
                if rng["min"] > rng["max"]:
                    raise ValueError(
                        f"Range {seg}.{key}: min ({rng['min']}) > max ({rng['max']})"
                    )

    apple = data["apple"]
    if apple is not None:
        if not isinstance(apple, dict):
            raise ValueError("Segment 'apple' must be a JSON object or null")
        for key in ("radius", "density"):
            if key not in apple:
                raise ValueError(f"Missing key '{key}' in apple")
            rng = apple[key]
            if "min" not in rng or "max" not in rng:
                raise ValueError(f"Range apple.{key} must have 'min' and 'max'")
            if rng["min"] > rng["max"]:
                raise ValueError(
                    f"Range apple.{key}: min ({rng['min']}) > max ({rng['max']})"
                )

    any_rod = any(data.get(s) is not None for s in ("primary", "secondary", "spur", "stem"))
    if not any_rod:
        raise ValueError(
            "At least one rod segment (primary, secondary, spur, or stem) must be non-null in the range file"
        )
