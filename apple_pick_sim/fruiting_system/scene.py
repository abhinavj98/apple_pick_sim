"""P0 fruiting scene types, generation, rollout, and force readouts."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import warp as wp

import newton

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.fruiting_system.build import (
    _attach_t_junction_world_supports,
    _build_fruiting_chain_into_builder,
    _finalize_fruiting_builder,
    _new_fruiting_builder,
    _scene_states_from_model,
    make_fruiting_solver_vbd,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    RodParams,
    TOPOLOGY_T_JUNCTION,
    load_ranges,
    params_fingerprint,
    resolve_fruiting_base_pos,
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
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Collection[str] | None = None,
    enable_self_collisions: bool = False,
) -> FruitingSystemScene:
    """Generate a Newton scene for a fruiting system from ``ranges`` and ``seed``.

    Args:
        ranges: Range dict as returned by :func:`load_ranges`.
        seed: Deterministic integer seed.
        base_pos: World-space position of the first rod segment's base (pinned).
        device: Warp device string (e.g. ``"cpu"``, ``"cuda:0"``). Defaults to
            :func:`~apple_pick_sim.sim_device.default_sim_device` (``cuda:0`` when CUDA is available).
        omit: Forwarded to :func:`sample_params` to force segments off without editing JSON.
        enable_self_collisions: If ``False`` (default), register shape collision filter pairs
            so woody segments (primary/secondary/spur) and the stem do not collide with each
            other or with the apple; apple↔woody contacts remain enabled. Ground contact is
            unchanged. If ``True``, only Newton joint parent/child filters apply, so
            non-adjacent chain links may collide.

    Returns:
        A :class:`FruitingSystemScene` ready to simulate.
    """
    device = resolve_sim_device(device)

    params = sample_params(ranges, seed, omit=omit)
    return _build_scene(
        params,
        base_pos=resolve_fruiting_base_pos(ranges, (0.0, 0.0, 3.0), override=base_pos),
        device=device,
        enable_self_collisions=enable_self_collisions,
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


def _transform_from_body_q_row(row: np.ndarray) -> wp.transform:
    return wp.transform(
        wp.vec3(float(row[0]), float(row[1]), float(row[2])),
        wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )


def fixed_joint_anchors_world(
    model: newton.Model,
    body_q: Any,
    joint_pairs: Sequence[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame parent/child anchor positions for fixed joints.

    Returns two flat ``(N*3,)`` ``float32`` arrays in ``joint_pairs`` order.
    Parent-side anchors map to proximal obs; child-side anchors map to distal obs.
    """
    if hasattr(body_q, "numpy"):
        bq_np = body_q.numpy().reshape(-1, 7)
    else:
        bq_np = np.asarray(body_q, dtype=np.float64).reshape(-1, 7)

    jparent = model.joint_parent.numpy()
    jchild = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()

    parent_anchors: list[np.ndarray] = []
    child_anchors: list[np.ndarray] = []
    for ji, _label in joint_pairs:
        parent = int(jparent[ji])
        child = int(jchild[ji])
        X_bp = _transform_from_body_q_row(Xp[ji])
        X_bc = _transform_from_body_q_row(Xc[ji])
        if parent >= 0:
            X_wp = wp.mul(_transform_from_body_q_row(bq_np[parent]), X_bp)
        else:
            X_wp = X_bp
        X_wc = wp.mul(_transform_from_body_q_row(bq_np[child]), X_bc)
        parent_anchors.append(
            np.asarray(wp.transform_get_translation(X_wp), dtype=np.float32)
        )
        child_anchors.append(
            np.asarray(wp.transform_get_translation(X_wc), dtype=np.float32)
        )

    if not parent_anchors:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return (
        np.concatenate(parent_anchors, dtype=np.float32),
        np.concatenate(child_anchors, dtype=np.float32),
    )


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
    enable_self_collisions: bool = False,
) -> FruitingSystemScene:
    """Build a Newton ModelBuilder scene from sampled params."""
    builder = _new_fruiting_builder()
    artifacts = _build_fruiting_chain_into_builder(builder, params, base_pos)
    if params.topology == TOPOLOGY_T_JUNCTION:
        _attach_t_junction_world_supports(builder, artifacts)
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
