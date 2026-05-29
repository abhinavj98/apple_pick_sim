"""P0 fruiting scene types, generation, rollout, and force readouts."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from typing import Any

import newton

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.fruiting_system.build import (
    _build_fruiting_chain_into_builder,
    _finalize_fruiting_builder,
    _new_fruiting_builder,
    _scene_states_from_model,
    make_fruiting_solver_vbd,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    RodParams,
    load_ranges,
    params_fingerprint,
    sample_params,
)
from apple_pick_sim.vbd_fixed_joint_wrenches import (
    fixed_joint_wrenches_child_com_vbd,
    iter_fixed_joint_indices,
)

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
            :func:`~apple_pick_sim.sim_device.default_sim_device` (``cuda:0`` when CUDA is available).
        omit: Forwarded to :func:`sample_params` to force segments off without editing JSON.
        enable_self_collisions: If ``False``, register shape collision filter pairs between
            every pair of distinct chain bodies (primary through apple), so the articulation
            does not self-collide; ground contact is unchanged. If ``True`` (default), only
            Newton joint parent/child filters apply, so non-adjacent chain links may collide.

    Returns:
        A :class:`FruitingSystemScene` ready to simulate.
    """
    device = resolve_sim_device(device)

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
    base_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
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
        device: Warp device string. Defaults to :func:`~apple_pick_sim.sim_device.default_sim_device`.
        omit: Forwarded to :func:`sample_params`.
        enable_self_collisions: Same semantics as :func:`generate_scene` (proxy is excluded
            from intra-chain filter pairs so it can contact the apple).
        gripper_proxy: Proxy mass/shape/placement options; defaults to :class:`GripperProxyConfig`.

    Returns:
        A :class:`CoupledCableScene` ready for VBD-only rollouts or robot coupling.
    """
    device = resolve_sim_device(device)

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
