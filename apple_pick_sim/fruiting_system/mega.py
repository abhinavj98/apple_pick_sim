"""Multi-instance VBD cable model (mega plant) for batched FD / FIM."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence

import newton

from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.fruiting_system.build import (
    _add_gripper_proxy,
    _apply_all_chain_collision_filters,
    _apply_collision_filters_between_chain_groups,
    _build_fruiting_chain_into_builder,
    _new_fruiting_builder,
    _scene_states_from_model,
)
from apple_pick_sim.fruiting_system.scene import _build_scene
from apple_pick_sim.fruiting_system.coupled import CoupledCableScene
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    fd_stiffness_param_columns,
    load_ranges,
    resolve_fruiting_base_pos,
    sample_params,
)


@dataclasses.dataclass(frozen=True)
class FruitingInstanceLayout:
    """One fruiting tree + gripper proxy inside a :class:`MegaCoupledCableScene`."""

    index: int
    params: FruitingSystemParams
    base_pos: tuple[float, float, float]
    primary_bodies: tuple[int, ...]
    secondary_bodies: tuple[int, ...]
    spur_bodies: tuple[int, ...]
    stem_bodies: tuple[int, ...]
    apple_body: int | None
    chain_bodies: tuple[int, ...]
    fruiting_fixed_joints: tuple[tuple[int, str], ...]
    cable_joint_indices: tuple[int, ...]
    joint_indices: tuple[int, ...]
    """All joints for this instance (sorted), including proxy / fixed joints."""
    gripper_proxy_body: int
    gripper_proxy_apple_joint: int | None
    gripper_proxy_offset_in_apple_frame: tuple[float, float, float] | None


@dataclasses.dataclass
class MegaCoupledCableScene:
    """Several coupled-ready fruiting instances in one VBD ``Model`` (mega plant).

    Each instance has its own rod chain, apple, and gripper proxy at a spatial offset.
    Inter-instance rigid collisions are filtered off. One MuJoCo arm can drive all proxies
    in ``fd_ghost`` mode (coupling layer not included here — plant model only).

    Build via :meth:`build` or :func:`generate_mega_coupled_cable_scene`.
    """

    model: newton.Model
    state_0: object
    state_1: object
    control: object
    solver: newton.solvers.SolverVBD
    instances: tuple[FruitingInstanceLayout, ...]
    gripper_proxy_config: GripperProxyConfig

    @property
    def num_instances(self) -> int:
        return len(self.instances)

    def instance(self, index: int) -> FruitingInstanceLayout:
        return self.instances[index]

    def all_gripper_proxy_body_ids(self) -> tuple[int, ...]:
        """Gripper proxy body index per instance (column order)."""
        return tuple(inst.gripper_proxy_body for inst in self.instances)

    def proxy_registry(self, instance_index: int, robot_body_id: int):
        """Map robot TCP → gripper proxy for one instance (staggered coupling)."""
        from apple_pick_sim.coupled_fruiting.proxy_coupling import ProxyBodyRegistry

        inst = self.instances[instance_index]
        return ProxyBodyRegistry.from_mapping({robot_body_id: inst.gripper_proxy_body})

    def as_single_instance_coupled(self, index: int) -> CoupledCableScene:
        """View one instance as a :class:`CoupledCableScene` sharing this mega ``Model``."""
        inst = self.instances[index]
        return CoupledCableScene(
            model=self.model,
            state_0=self.state_0,
            state_1=self.state_1,
            control=self.control,
            solver=self.solver,
            params=inst.params,
            primary_bodies=list(inst.primary_bodies),
            secondary_bodies=list(inst.secondary_bodies),
            spur_bodies=list(inst.spur_bodies),
            stem_bodies=list(inst.stem_bodies),
            apple_body=inst.apple_body,
            fruiting_fixed_joints=inst.fruiting_fixed_joints,
            cable_joint_indices=inst.cable_joint_indices,
            gripper_proxy_body=inst.gripper_proxy_body,
            gripper_proxy_config=self.gripper_proxy_config,
            gripper_proxy_apple_joint=inst.gripper_proxy_apple_joint,
            gripper_proxy_offset_in_apple_frame=inst.gripper_proxy_offset_in_apple_frame,
        )

    @classmethod
    def build(
        cls,
        params_list: Sequence[FruitingSystemParams],
        *,
        base_pos: tuple[float, float, float] = (0.5, 0.5, 1.5),
        instance_spacing: tuple[float, float, float] = (0.0, 1.5, 0.0),
        device: str | None = None,
        enable_self_collisions: bool = False,
        gripper_proxy: GripperProxyConfig | None = None,
    ) -> MegaCoupledCableScene:
        """Assemble ``len(params_list)`` fruiting instances in one cable ``Model``."""
        if not params_list:
            raise ValueError("params_list must contain at least one FruitingSystemParams")
        device = resolve_sim_device(device)
        proxy_cfg = gripper_proxy if gripper_proxy is not None else GripperProxyConfig()

        builder = _new_fruiting_builder()
        instance_chain_groups: list[list[int]] = []
        instance_joint_lists: list[list[int]] = []
        instance_proxy_free_joints: list[int | None] = []
        layouts: list[FruitingInstanceLayout] = []

        for i, params in enumerate(params_list):
            offset = (
                base_pos[0] + i * instance_spacing[0],
                base_pos[1] + i * instance_spacing[1],
                base_pos[2] + i * instance_spacing[2],
            )
            artifacts = _build_fruiting_chain_into_builder(builder, params, offset)
            (
                proxy_body,
                proxy_apple_joint,
                proxy_offset_in_apple,
                proxy_free_joint,
            ) = _add_gripper_proxy(
                builder,
                artifacts,
                proxy_cfg,
                apple_radius=params.apple_radius,
            )
            artifacts.fruiting_fixed_joints.sort(key=lambda p: p[0])
            instance_joint_lists.append(list(artifacts.all_joints))
            instance_chain_groups.append(list(artifacts.chain_bodies))
            instance_proxy_free_joints.append(proxy_free_joint)

            layouts.append(
                FruitingInstanceLayout(
                    index=i,
                    params=params,
                    base_pos=offset,
                    primary_bodies=tuple(artifacts.seg_bodies["primary"]),
                    secondary_bodies=tuple(artifacts.seg_bodies["secondary"]),
                    spur_bodies=tuple(artifacts.seg_bodies["spur"]),
                    stem_bodies=tuple(artifacts.seg_bodies["stem"]),
                    apple_body=artifacts.apple_body,
                    chain_bodies=tuple(artifacts.chain_bodies),
                    fruiting_fixed_joints=tuple(artifacts.fruiting_fixed_joints),
                    cable_joint_indices=tuple(artifacts.cable_joint_indices),
                    joint_indices=tuple(sorted(artifacts.all_joints)),
                    gripper_proxy_body=proxy_body,
                    gripper_proxy_apple_joint=proxy_apple_joint,
                    gripper_proxy_offset_in_apple_frame=proxy_offset_in_apple,
                )
            )

        for group in instance_chain_groups:
            if not enable_self_collisions:
                _apply_all_chain_collision_filters(builder, group)
        for i in range(len(instance_chain_groups)):
            for j in range(i + 1, len(instance_chain_groups)):
                _apply_collision_filters_between_chain_groups(
                    builder,
                    instance_chain_groups[i],
                    instance_chain_groups[j],
                )

        for inst_joints, proxy_free_joint in zip(
            instance_joint_lists, instance_proxy_free_joints
        ):
            joint_list = sorted(inst_joints)
            if proxy_free_joint is not None:
                chain_joints = [j for j in joint_list if j != proxy_free_joint]
                builder.add_articulation(chain_joints)
                builder.add_articulation([proxy_free_joint])
            else:
                builder.add_articulation(joint_list)

        builder.add_ground_plane()
        builder.color()
        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, -9.81))
        if proxy_cfg.fix_to_apple:
            from apple_pick_sim.fruiting_system.build import prescribe_body_vbd_on_model

            prescribed: list[int] = []
            for inst in layouts:
                if inst.apple_body is not None:
                    prescribed.append(inst.apple_body)
                prescribed.append(inst.gripper_proxy_body)
            prescribe_body_vbd_on_model(model, *prescribed)
        state_0, state_1, control, solver = _scene_states_from_model(model)
        scene = cls(
            model=model,
            state_0=state_0,
            state_1=state_1,
            control=control,
            solver=solver,
            instances=tuple(layouts),
            gripper_proxy_config=proxy_cfg,
        )
        for inst in layouts:
            _align_mega_instance_bodies(
                scene,
                inst,
                device=device,
                enable_self_collisions=enable_self_collisions,
            )
        if len(layouts) > 1:
            from apple_pick_sim.fruiting_system.mega_fd import sync_all_instances_from_nominal

            sync_all_instances_from_nominal(scene)
        return scene


def _align_mega_instance_bodies(
    mega: MegaCoupledCableScene,
    inst: FruitingInstanceLayout,
    *,
    device: str,
    enable_self_collisions: bool,
) -> None:
    """Copy reference P0 ``body_q`` onto this instance's chain bodies (per-offset)."""
    from apple_pick_sim.coupled_fruiting.proxy_coupling import align_proxy_body_q_prev_for_vbd

    ref = _build_scene(
        inst.params,
        inst.base_pos,
        device,
        enable_self_collisions=enable_self_collisions,
    )
    n = ref.model.body_count
    chain_ids = list(inst.chain_bodies)
    if len(chain_ids) < n:
        raise ValueError(
            f"Instance {inst.index}: expected at least {n} chain bodies, got {len(chain_ids)}"
        )
    ref_bq = ref.state_0.body_q.numpy().reshape(-1, 7)
    ref_bqd = ref.state_0.body_qd.numpy().reshape(-1, 6)
    for state in (mega.state_0, mega.state_1):
        bq = state.body_q.numpy().reshape(-1, 7).copy()
        bqd = state.body_qd.numpy().reshape(-1, 6).copy()
        for k in range(n):
            bid = chain_ids[k]
            bq[bid] = ref_bq[k]
            bqd[bid] = ref_bqd[k]
        state.body_q.assign(bq.ravel())
        state.body_qd.assign(bqd.ravel())
    coupled_view = mega.as_single_instance_coupled(inst.index)
    align_proxy_body_q_prev_for_vbd(coupled_view, tuple(chain_ids[:n]))


def generate_mega_coupled_cable_scene(
    ranges: dict,
    seed: int,
    params_list: Sequence[FruitingSystemParams] | None = None,
    *,
    stiffness_epsilon: float | None = None,
    base_pos: tuple[float, float, float] | None = None,
    instance_spacing: tuple[float, float, float] = (0.0, 1.5, 0.0),
    device: str | None = None,
    omit: Collection[str] | None = None,
    enable_self_collisions: bool = True,
    gripper_proxy: GripperProxyConfig | None = None,
) -> MegaCoupledCableScene:
    """Build a mega plant from explicit params and/or FD stiffness columns.

    If ``params_list`` is omitted, samples ``nominal = sample_params(ranges, seed)``.
    When ``stiffness_epsilon`` is set, instances are
    ``fd_stiffness_param_columns(nominal, stiffness_epsilon)`` (nominal + perturbed rods).
    When ``params_list`` is provided, ``stiffness_epsilon`` is ignored.
    """
    if params_list is None:
        nominal = sample_params(ranges, seed, omit=omit)
        if stiffness_epsilon is not None:
            params_list = fd_stiffness_param_columns(nominal, stiffness_epsilon)
        else:
            params_list = [nominal]
    return MegaCoupledCableScene.build(
        params_list,
        base_pos=resolve_fruiting_base_pos(ranges, (0.5, 0.5, 1.5), override=base_pos),
        instance_spacing=instance_spacing,
        device=device,
        enable_self_collisions=enable_self_collisions,
        gripper_proxy=gripper_proxy,
    )


def load_ranges_and_build_mega(
    range_json: str | Path,
    seed: int,
    *,
    stiffness_epsilon: float,
    **kwargs: object,
) -> MegaCoupledCableScene:
    """Convenience: load JSON ranges and build FD stiffness columns."""
    from pathlib import Path as _Path

    ranges = load_ranges(_Path(range_json) if isinstance(range_json, str) else range_json)
    return generate_mega_coupled_cable_scene(
        ranges,
        seed,
        stiffness_epsilon=stiffness_epsilon,
        **kwargs,  # type: ignore[arg-type]
    )
