"""Minimal visual inspector for ``BatchedHeterogeneousCoupledSim`` (V.3.1 step B).

Builds a short-settle batched scene via the config-driven runtime (not the legacy
monolith example), optionally animates VBD settle during init, then steps coupled
simulation with a slow world-0 nudge.

Run from the repository root::

    uv run python apple_pick_sim/examples/inspect_batched_heterogeneous_coupled_sim.py \\
      --viewer gl --num-envs 2 --seed 21 --settle-substeps 50

Headless / CI smoke (no DISPLAY)::

    uv run python apple_pick_sim/examples/inspect_batched_heterogeneous_coupled_sim.py \\
      --viewer null --num-frames 5 --settle-substeps 50
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

import newton
import newton.examples
import torch

from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
    DomainRandomizationConfig,
    RobotConfig,
    SceneSettleCollisionConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.fruiting_system import load_ranges, sample_heterogeneous_params_list
from apple_pick_sim.sim_device import resolve_sim_device

_RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_real_world_proxy_variance.json"
)
_DEFAULT_SETTLE_SUBSTEPS = 50
_NUDGE_VX = 0.02  # m/s on world-0 linear-x action


def _make_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=2, help="Batch size.")
    parser.add_argument("--seed", type=int, default=21, help="Topology + DR seed.")
    parser.add_argument(
        "--settle-substeps",
        type=int,
        default=_DEFAULT_SETTLE_SUBSTEPS,
        help=f"VBD substeps before weld (default: {_DEFAULT_SETTLE_SUBSTEPS}).",
    )
    return parser


def _make_config(args) -> BatchedHeterogeneousCoupledSimConfig:
    num_envs = int(args.num_envs)
    return dataclasses.replace(
        BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs),
        runtime=dataclasses.replace(
            BatchedHeterogeneousCoupledSimConfig.test_minimal(num_envs=num_envs).runtime,
            device=resolve_sim_device(getattr(args, "device", None)),
        ),
        robot=RobotConfig(
            kind="fr3",
            step_mode="coupled",
            fix_to_apple=True,  # inline settle on final scene (settle-then-weld needs FR3)
        ),
        scene=SceneSettleCollisionConfig(settle_substeps=int(args.settle_substeps)),
        domain_randomization=DomainRandomizationConfig(
            ranges_path=_RANGES_FIXTURE,
            topology_seed=int(args.seed),
        ),
        settle_diagnostics=None,
        obs=None,
    )


def _print_scene_summary(sim: BatchedHeterogeneousCoupledSim) -> None:
    layout = sim.layout
    if layout is None:
        print("Scene summary: layout missing")
        return
    spacing = sim.config.runtime.env_spacing
    print(f"Scene summary: num_envs={layout.num_envs} env_spacing={spacing}")
    body_q = sim.scene.cable.state_0.body_q.numpy().reshape(-1, 7)
    for w, apple_idx in enumerate(layout.apple_body_indices):
        if apple_idx < 0:
            continue
        x, y, z = (float(body_q[apple_idx, i]) for i in range(3))
        print(f"  env{w} apple pos=({x:.4f}, {y:.4f}, {z:.4f}) m")


def _setup_viewer(viewer, sim: BatchedHeterogeneousCoupledSim) -> bool:
    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    viewer.set_model(sim.scene.cable.model)
    if graphical and sim.num_envs > 1:
        viewer.set_world_offsets(tuple(sim.config.runtime.env_spacing))
    return graphical


def _render_frame(viewer, sim: BatchedHeterogeneousCoupledSim, sim_time: float) -> None:
    scene = sim.scene
    contacts = scene.cable.model.collide(
        scene.cable.state_0,
        collision_pipeline=scene.cable_collision_pipeline,
    )
    viewer.begin_frame(sim_time)
    viewer.log_state(scene.cable.state_0)
    viewer.log_contacts(contacts, scene.cable.state_0)
    viewer.end_frame()


def _zero_actions(sim: BatchedHeterogeneousCoupledSim) -> torch.Tensor:
    n, d = sim.config.controller.expected_action_shape(sim.num_envs)
    actions = torch.zeros(n, d, dtype=torch.float32, device=sim.device)
    actions[0, 0] = _NUDGE_VX
    return actions


def main() -> None:
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "5"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 5.")

    parser = _make_parser()
    viewer, args = newton.examples.init(parser=parser)

    cfg = _make_config(args)
    ranges = load_ranges(_RANGES_FIXTURE)
    per_env_params = sample_heterogeneous_params_list(
        ranges, topology_seed=int(args.seed), num_envs=cfg.runtime.num_envs
    )

    graphical = isinstance(viewer, newton.viewer.ViewerGL)
    build_viewer = viewer if graphical else None

    print(
        f"Building BatchedHeterogeneousCoupledSim: "
        f"num_envs={cfg.runtime.num_envs} settle_substeps={cfg.scene.settle_substeps} "
        f"seed={args.seed} device={cfg.resolve_device()}",
        flush=True,
    )
    sim = BatchedHeterogeneousCoupledSim(
        cfg,
        per_env_params,
        ranges,
        viewer=build_viewer,
        use_settle_cache=False,
    )

    _setup_viewer(viewer, sim)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    _print_scene_summary(sim)

    actions = _zero_actions(sim)
    print("Starting coupled step loop…", flush=True)
    while viewer.is_running():
        sim.step(actions)
        _render_frame(viewer, sim, sim.sim_time)
        if graphical:
            time.sleep(max(0.0, sim.frame_dt))

    print(f"Done (sim_time={sim.sim_time:.3f}s).", flush=True)


if __name__ == "__main__":
    main()
