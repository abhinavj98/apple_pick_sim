"""M1 coupled cable scene (P0 tree + gripper proxy on Model B)."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from typing import Any

import warp as wp

import newton

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.fruiting_system.build import (
    _FruitingChainArtifacts,
    _add_gripper_proxy,
    _build_fruiting_chain_into_builder,
    _finalize_fruiting_builder,
    _new_fruiting_builder,
    _scene_states_from_model,
)
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    resolve_fruiting_base_pos,
    sample_params,
)
from apple_pick_sim.fruiting_system.scene import (
    FruitingSystemScene,
    _build_scene,
    geometry_fingerprint,
)

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
    gripper_proxy_offset_in_apple_frame: tuple[float, float, float, float, float, float, float] | None = None
    gripper_proxy_vis_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame clearance offset from proxy COM to the apple surface (= approach_dir × clearance).

    Stored at build time for viewer/debug use; not applied automatically during coupling.
    """

    def proxy_registry(self, robot_body_id: int):
        """Map robot TCP body id → cable ``gripper_proxy_body`` for staggered wrench I/O.

        Used by :meth:`apple_pick_sim.coupled_fruiting.CoupledFruitingScene.coupled_substep`:
        harvest writes ``proxy_forces[robot_body_id]``; sync reads the same slot when
        undoing the lagged coupling velocity on the proxy.
        """
        from apple_pick_sim.coupled_fruiting.proxy_coupling import ProxyBodyRegistry

        return ProxyBodyRegistry.from_mapping({robot_body_id: self.gripper_proxy_body})


@dataclasses.dataclass
class CoupledCablePopulateResult:
    """Builder-populated cable template before ``finalize``."""

    artifacts: _FruitingChainArtifacts
    proxy_body: int
    proxy_apple_joint: int | None
    proxy_offset_in_apple: tuple[float, float, float, float, float, float, float] | None
    proxy_free_joint: int | None
    vis_offset: tuple[float, float, float]


def _populate_coupled_cable_builder(
    builder: newton.ModelBuilder,
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
    *,
    gripper_proxy: GripperProxyConfig,
    robot_base_pos: tuple[float, float, float] | None = None,
) -> CoupledCablePopulateResult:
    """Add fruiting chain + gripper proxy to ``builder`` (no ``finalize``)."""
    artifacts = _build_fruiting_chain_into_builder(builder, params, base_pos)
    proxy_body, proxy_apple_joint, proxy_offset_in_apple, proxy_free_joint, vis_offset = (
        _add_gripper_proxy(
            builder,
            artifacts,
            gripper_proxy,
            apple_radius=params.apple_radius,
            robot_base_pos=robot_base_pos,
        )
    )
    return CoupledCablePopulateResult(
        artifacts=artifacts,
        proxy_body=proxy_body,
        proxy_apple_joint=proxy_apple_joint,
        proxy_offset_in_apple=proxy_offset_in_apple,
        proxy_free_joint=proxy_free_joint,
        vis_offset=(
            float(vis_offset[0]),
            float(vis_offset[1]),
            float(vis_offset[2]),
        ),
    )


def generate_coupled_cable_scene(
    ranges: dict,
    seed: int,
    *,
    params: FruitingSystemParams | None = None,
    base_pos: tuple[float, float, float] | None = None,
    device: str | None = None,
    omit: Collection[str] | None = None,
    enable_self_collisions: bool = False,
    gripper_proxy: GripperProxyConfig | None = None,
    robot_base_pos: tuple[float, float, float] | None = None,
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
        params: Pre-sampled parameters; when ``None``, :func:`sample_params` runs from
            ``(ranges, seed, omit)``.
        omit: Forwarded to :func:`sample_params` when ``params`` is ``None``.
        enable_self_collisions: Same semantics as :func:`generate_scene` (proxy is excluded
            from intra-chain filter pairs so it can contact the apple).
        gripper_proxy: Proxy mass/shape/placement options; defaults to :class:`GripperProxyConfig`.
        robot_base_pos: World translation of the robot base; required when
            ``gripper_proxy.robot_facing_weld`` is ``True``.

    Returns:
        A :class:`CoupledCableScene` ready for VBD-only rollouts or robot coupling.
    """
    device = resolve_sim_device(device)

    resolved_params = params if params is not None else sample_params(ranges, seed, omit=omit)
    proxy_cfg = gripper_proxy if gripper_proxy is not None else GripperProxyConfig()
    return _build_coupled_cable_scene(
        resolved_params,
        base_pos=resolve_fruiting_base_pos(ranges, (0.5, 0.5, 0.5), override=base_pos),
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=proxy_cfg,
        robot_base_pos=robot_base_pos,
    )

def geometry_fingerprint_coupled(scene: CoupledCableScene) -> dict:
    """Like :func:`geometry_fingerprint` plus gripper-proxy body index and position."""
    fp = geometry_fingerprint(scene)
    body_q = scene.state_0.body_q.to("cpu").numpy()
    p = body_q[scene.gripper_proxy_body]
    fp["gripper_proxy_body"] = scene.gripper_proxy_body
    fp["gripper_proxy_pos"] = (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5))
    return fp

def _align_coupled_scene_chain_from_reference(
    coupled: CoupledCableScene,
    *,
    base_pos: tuple[float, float, float],
    device: str,
    enable_self_collisions: bool,
) -> None:
    """Match P0 chain ``body_q`` / ``joint_q`` on the coupled model (proxy DOFs unchanged).

    A world-root FREE proxy in the same articulation as the fruiting tree skews Newton FK at
    finalize; this copies the reference P0 kinematics for shared bodies before coupling.
    """
    from apple_pick_sim.coupled_fruiting.proxy_coupling import align_proxy_body_q_prev_for_vbd

    ref = _build_scene(
        coupled.params,
        base_pos,
        device,
        enable_self_collisions=enable_self_collisions,
    )
    n = ref.model.body_count
    ref_bq = ref.state_0.body_q.numpy().reshape(-1, 7)
    ref_bqd = ref.state_0.body_qd.numpy().reshape(-1, 6)
    for state in (coupled.state_0, coupled.state_1):
        bq = state.body_q.numpy().reshape(-1, 7).copy()
        bqd = state.body_qd.numpy().reshape(-1, 6).copy()
        bq[:n] = ref_bq[:n]
        bqd[:n] = ref_bqd[:n]
        state.body_q.assign(bq.ravel())
        state.body_qd.assign(bqd.ravel())

    jc = ref.model.joint_coord_count
    jq = coupled.model.joint_q.numpy().copy()
    jqd = coupled.model.joint_qd.numpy().copy()
    jq[:jc] = ref.model.joint_q.numpy()[:jc]
    jqd[:jc] = ref.model.joint_qd.numpy()[:jc]
    coupled.model.joint_q.assign(jq)
    coupled.model.joint_qd.assign(jqd)

    # Do not run full-model eval_fk here: the world-root proxy articulation skews the tree.
    align_proxy_body_q_prev_for_vbd(coupled, tuple(range(n)))


def _build_coupled_cable_scene(
    params: FruitingSystemParams,
    base_pos: tuple[float, float, float],
    device: str,
    *,
    enable_self_collisions: bool,
    gripper_proxy: GripperProxyConfig,
    robot_base_pos: tuple[float, float, float] | None = None,
) -> CoupledCableScene:
    builder = _new_fruiting_builder()
    populated = _populate_coupled_cable_builder(
        builder,
        params,
        base_pos,
        gripper_proxy=gripper_proxy,
        robot_base_pos=robot_base_pos,
    )
    artifacts = populated.artifacts
    proxy_body = populated.proxy_body
    proxy_apple_joint = populated.proxy_apple_joint
    proxy_offset_in_apple = populated.proxy_offset_in_apple
    proxy_free_joint = populated.proxy_free_joint
    vis_offset = populated.vis_offset
    model = _finalize_fruiting_builder(
        builder,
        artifacts,
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy_joint=proxy_free_joint,
    )
    if gripper_proxy.fix_to_apple:
        from apple_pick_sim.fruiting_system.build import prescribe_body_vbd_on_model

        assert artifacts.apple_body is not None
        prescribe_body_vbd_on_model(model, artifacts.apple_body, proxy_body)
    artifacts.fruiting_fixed_joints.sort(key=lambda p: p[0])
    state_0, state_1, control, solver = _scene_states_from_model(model)

    scene = CoupledCableScene(
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
        gripper_proxy_vis_offset=(
            float(vis_offset[0]), float(vis_offset[1]), float(vis_offset[2])
        ),
    )
    _align_coupled_scene_chain_from_reference(
        scene,
        base_pos=base_pos,
        device=device,
        enable_self_collisions=enable_self_collisions,
    )
    return scene
