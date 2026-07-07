"""Parallel batched sys-ID MMD grid replay (phase 2, pre-scoring).

Replay recorded actions across a bend-stiffness candidate grid for each structure
in a ``batched_sysid_v1`` dataset. MMD scoring is not implemented yet; use
``--replay-only`` to validate replay row counts.

Run from the repository root::

    uv run python apple_pick_gym/batched_examples/example_batched_sysid_mmd_grid.py \\
      --dataset /tmp/batched_sysid_dataset \\
      --replay-only \\
      --primary-bend-stiffness-values 1e-4,2e-4 \\
      --secondary-bend-stiffness-values 1e-4 \\
      --spur-bend-stiffness-values 1e-4 \\
      --stem-bend-stiffness-values 1e-4,2e-4

Sim build (VIC gains, settle substeps, control_hz, …) is configured via module
constants in this file, not CLI flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import newton.examples

from apple_pick_gym.examples.run_system_identification import parse_positive_float_grid
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    ControllerConfig,
    DomainRandomizationConfig,
    FruitingSystemConfig,
    MujocoConfig,
    ObsConfig,
    RobotConfig,
    RuntimeConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)
from apple_pick_sim.fruiting_system.params import PLACEHOLDER_EE_MASS_KG, GripperProxyConfig
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_DEFAULT_ITERATIONS

# --- Sim build (edit here; not exposed on CLI) ---
CONTROL_HZ = 30.0
SUB_DT = 1.0 / 1800.0
ENV_SPACING = (2.0, 2.0, 2.0)
SETTLE_SUBSTEPS = 5000
VIC_GAINS = ImpedanceGains(
    linear_k=6000.0,
    linear_d=0.0,
    angular_k=2000.0,
    angular_d=0.0,
)
JOINT_ANGULAR_KD_OVERRIDES = {
    "support": 1.0,
    "primary_spur": 1.0,
    "stem_apple": 5e-2,
}
GRIPPER_PROXY = GripperProxyConfig(mass=PLACEHOLDER_EE_MASS_KG)

ROD_SEGMENTS: tuple[str, ...] = ("primary", "secondary", "spur", "stem")


def parse_comma_separated_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of non-negative integers."""
    if value.strip() == "":
        raise argparse.ArgumentTypeError("expected at least one integer")
    parts = value.split(",")
    if any(part.strip() == "" for part in parts):
        raise argparse.ArgumentTypeError("empty index entries are not allowed")
    try:
        values = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("structure indices must be integers") from exc
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("structure indices must be >= 0")
    return values


def build_sim_config(*, num_envs: int) -> BatchedHeterogeneousCoupledSimConfig:
    return BatchedHeterogeneousCoupledSimConfig(
        runtime=RuntimeConfig(
            num_envs=int(num_envs),
            env_spacing=ENV_SPACING,
            device=None,
            control_hz=CONTROL_HZ,
            sub_dt=SUB_DT,
        ),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,
            gripper=GRIPPER_PROXY,
            robot_base_pos=None,
            per_env_ik=True,
            ik_bootstrap_iterations=IK_BOOTSTRAP_DEFAULT_ITERATIONS,
            skip_ik_bootstrap=True,
            defer_template_robot_bootstrap=True,
            force_batched_layout=True,
        ),
        scene=SceneSettleCollisionConfig(
            fruiting_base_pos=None,
            settle_substeps=SETTLE_SUBSTEPS,
            settle_gravity_ramp=False,
            settle_max_speed_m_s=0.05,
            enable_self_collisions=False,
            enable_apple_woody_collisions=True,
            enable_proxy_woody_collisions=True,
        ),
        domain_randomization=DomainRandomizationConfig(
            ranges_path=None,
            topology_seed=None,
            per_env_params=None,
        ),
        fruiting_system=FruitingSystemConfig(
            stem_coupling_gain=DEFAULT_STEM_COUPLING_GAIN,
            stem_force_cap_N=DEFAULT_STEM_FORCE_CAP_N,
            stem_torque_cap_Nm=DEFAULT_STEM_TORQUE_CAP_NM,
            stem_harvest_explicit_apple_weight=False,
            joint_angular_kd_overrides=dict(JOINT_ANGULAR_KD_OVERRIDES),
        ),
        controller=ControllerConfig(
            mode="vic",
            action_dim=6,
            linear_speed=0.1,
            angular_speed=0.1,
            ik_iterations=128,
            vic_gains=VIC_GAINS,
            allocate_action_buffer=True,
        ),
        mujoco=MujocoConfig(
            solver_kwargs=dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS),
            use_cpu=None,
        ),
        settle_diagnostics=None,
        obs=ObsConfig(
            allocate_buffers=True,
            include_robot=True,
            include_forces=True,
        ),
    )


def chunk_candidates(
    candidates: Sequence[object],
    *,
    max_envs_per_batch: int,
    num_directions: int,
) -> list[list[object]]:
    """Split candidates so each chunk fits ``max_envs_per_batch`` parallel envs."""
    items = list(candidates)
    if not items:
        return []
    limit = int(max_envs_per_batch)
    directions = int(num_directions)
    if limit <= 0:
        return [items]
    max_chunk_size = limit // directions
    if max_chunk_size < 1:
        raise SystemExit(
            f"--max-envs-per-batch ({limit}) must be >= num_directions ({directions})"
        )
    return [items[i : i + max_chunk_size] for i in range(0, len(items), max_chunk_size)]


def _make_parser() -> argparse.ArgumentParser:
    p = newton.examples.create_parser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Directory containing batched_sysid_v1 manifest.json and episodes/.",
    )
    p.add_argument(
        "--structure-indices",
        type=parse_comma_separated_ints,
        default=None,
        help="Comma-separated structure indices (default: all in manifest).",
    )
    p.add_argument(
        "--max-envs-per-batch",
        type=int,
        default=0,
        help="Chunk candidates so chunk_size*num_directions <= this (0 = no chunking).",
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Cap evaluated stiffness candidates per structure (0 = full grid).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Replay RNG seed (default: manifest collection.seed).",
    )
    p.add_argument(
        "--replay-only",
        action="store_true",
        help="Replay recorded actions and print per-structure row-count summary.",
    )
    for segment in ROD_SEGMENTS:
        p.add_argument(
            f"--{segment}-bend-stiffness-values",
            dest=f"{segment}_bend_stiffness_values",
            type=parse_positive_float_grid,
            default=None,
            help=f"Comma-separated candidate values for {segment}.bend_stiffness.",
        )
    return p


def _require_grid_values(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [
        f"--{segment}-bend-stiffness-values"
        for segment in ROD_SEGMENTS
        if getattr(args, f"{segment}_bend_stiffness_values") is None
    ]
    if missing:
        parser.error(
            "the following arguments are required when evaluating candidates: "
            + ", ".join(missing)
        )


def _structure_indices_from_args(dataset, args: argparse.Namespace) -> list[int]:
    if args.structure_indices is not None:
        return list(args.structure_indices)
    return list(range(len(dataset.structure_summaries())))


def _build_candidate_grid(args: argparse.Namespace):
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import iter_bend_stiffness_candidates

    return list(
        iter_bend_stiffness_candidates(
            primary_values=args.primary_bend_stiffness_values,
            secondary_values=args.secondary_bend_stiffness_values,
            spur_values=args.spur_bend_stiffness_values,
            stem_values=args.stem_bend_stiffness_values,
        )
    )


def _candidates_for_structure(dataset, args: argparse.Namespace, structure_idx: int):
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
        ensure_gt_candidate_in_grid,
        gt_bend_stiffness_candidate_from_structure,
    )

    candidates = _build_candidate_grid(args)
    gt = gt_bend_stiffness_candidate_from_structure(dataset, int(structure_idx))
    candidates = ensure_gt_candidate_in_grid(candidates, gt)
    max_candidates = int(args.max_candidates)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    return candidates


def _make_build_env_fn(*, ranges_path: str, topology_seed: int):
    import dataclasses

    from apple_pick_gym.batched_envs import ApplePickBatchedSysIdEnv

    def build_env_fn(
        *,
        num_envs: int,
        per_env_params: list,
        max_episode_steps: int,
        gripper: GripperProxyConfig | None = None,
    ) -> ApplePickBatchedSysIdEnv:
        sim_config = build_sim_config(num_envs=num_envs)
        if gripper is not None:
            sim_config = dataclasses.replace(
                sim_config,
                robot=dataclasses.replace(sim_config.robot, gripper=gripper),
            )
        return ApplePickBatchedSysIdEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            ranges_path=ranges_path,
            topology_seed=int(topology_seed),
            use_settle_cache=False,
            per_env_params=per_env_params,
            control_hz=CONTROL_HZ,
            sim_config=sim_config,
        )

    return build_env_fn


def _replay_structure(
    *,
    dataset,
    structure_idx: int,
    candidates,
    num_directions: int,
    seed: int | None,
    max_envs_per_batch: int,
    build_env_fn,
):
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import replay_candidates_for_structure

    collectors = replay_candidates_for_structure(
        dataset=dataset,
        structure_idx=int(structure_idx),
        candidates=candidates,
        num_directions=int(num_directions),
        seed=seed,
        build_env_fn=build_env_fn,
        max_envs_per_batch=int(max_envs_per_batch),
    )
    return collectors, len(candidates)


def _print_replay_summary(
    *,
    structure_idx: int,
    n_frames: int,
    num_candidates: int,
    num_directions: int,
    collectors,
) -> None:
    num_envs = num_candidates * num_directions
    row_counts = [collectors.n_rows(env_idx) for env_idx in range(num_envs)]
    print(
        f"structure {structure_idx}: n_frames={n_frames} "
        f"candidates={num_candidates} directions={num_directions} "
        f"collector_rows={row_counts}"
    )


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

    _require_grid_values(parser, args)

    dataset = BatchedSysIdDataset(args.dataset)
    collection = dataset.manifest.get("collection", {})
    num_directions = int(collection.get("num_directions", 0))
    if num_directions < 1:
        raise SystemExit("manifest collection.num_directions must be >= 1")

    ranges_path = collection.get("ranges_path")
    if not ranges_path:
        from apple_pick_sim.fruiting_system import default_ranges_fixture_path

        ranges_path = str(default_ranges_fixture_path())
    topology_seed = int(collection.get("topology_seed", 42))

    structure_indices = _structure_indices_from_args(dataset, args)
    if not structure_indices:
        raise SystemExit("No structure indices to evaluate.")

    build_env_fn = _make_build_env_fn(
        ranges_path=str(ranges_path),
        topology_seed=topology_seed,
    )

    for structure_idx in structure_indices:
        candidates = _candidates_for_structure(dataset, args, int(structure_idx))
        collectors, num_candidates = _replay_structure(
            dataset=dataset,
            structure_idx=int(structure_idx),
            candidates=candidates,
            num_directions=num_directions,
            seed=args.seed,
            max_envs_per_batch=int(args.max_envs_per_batch),
            build_env_fn=build_env_fn,
        )
        if bool(args.replay_only):
            arrays = dataset.load_episode_obs_arrays(int(structure_idx), 0)
            n_frames = int(arrays["action"].shape[0])
            _print_replay_summary(
                structure_idx=int(structure_idx),
                n_frames=n_frames,
                num_candidates=num_candidates,
                num_directions=num_directions,
                collectors=collectors,
            )

    if not bool(args.replay_only):
        print("MMD scoring is not implemented yet; re-run with --replay-only for summaries.")


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "1"])

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)
    try:
        _run(args, parser)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
