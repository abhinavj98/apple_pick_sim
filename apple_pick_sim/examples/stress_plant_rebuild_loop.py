"""Stress-test fused VBD plant rebuilds or replay reset/init loops.

Modes:

* ``rebuild`` (default): build ``BatchedHeterogeneousCoupledSim`` → teardown → repeat.
* ``replay-reset``: build once, then loop ``reset()`` → episode init → optional replay steps.
* ``rebuild-replay``: CMA-like waves — rebuild → ``--resets-per-wave`` reset loops → repeat.

Replay steps copy ``_last_obs`` to CPU numpy (CMA collector path) and check
``apple_quat`` vs cable ``body_q`` and ``ft_wrist`` vs ``coupling_forces_cache``.

Run from the repository root::

    uv run python apple_pick_sim/examples/stress_plant_rebuild_loop.py \\
      --num-envs 100 --cycles 20 --reuse-replicated-mujoco

CMA-like hybrid (10 rebuild waves × 100 resets each)::

    uv run python apple_pick_sim/examples/stress_plant_rebuild_loop.py \\
      --mode rebuild-replay --dataset tmp/real_batched_s09_k_frame \\
      --num-envs 100 --cycles 10 --resets-per-wave 100 --reuse-replicated-mujoco \\
      --post-grasp-settle-substeps 500
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np
import warp as wp

from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import strip_pre_weld_rows
from apple_pick_gym.batched_envs.obs_torch import (
    _torch_to_numpy_f32_copy,
    download_batched_replay_obs_numpy,
)
from apple_pick_gym.batched_envs.real_batched_replay_build import (
    bootstrap_joint_q_from_episode_metadata,
    control_hz_from_episode_metadata,
    dataset_declares_vic_pose,
    fruiting_base_pos_from_episode_metadata,
    make_real_replay_build_env_fn,
)
from apple_pick_gym.batched_envs.support_joint_penalties import (
    apply_per_env_support_joint_penalties,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import (
    BatchedHeterogeneousCoupledSimConfig,
)
from apple_pick_sim.coupled_fruiting.batched_heterogeneous_coupled_sim import (
    BatchedHeterogeneousCoupledSim,
)
from apple_pick_sim.coupled_fruiting.replicated_robot_cache import (
    clear_process_replicated_robot_cache,
    process_replicated_robot_cache,
)
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    default_ranges_fixture_path,
    load_ranges,
    sample_heterogeneous_params_list,
    set_rod_youngs_modulus,
)
from apple_pick_sim.sim_device import resolve_sim_device
from apple_pick_sim.system_id import ReplayEpisodeSource
from apple_pick_sim.system_id.batched_digital_twin_init import (
    gripper_proxy_for_real_batched_replay,
    initialize_batched_env_from_episode_sources,
    true_params_for_structure,
)
from apple_pick_sim.system_id.batched_trajectory_store import BatchedSysIdDataset

# Real-replay CMA search box (log10): support_kp, spur E, stem E.
DEFAULT_LOG10_LOWER = (2.0, 7.0, 7.0)
DEFAULT_LOG10_UPPER = (6.0, 10, 10)


@dataclass(frozen=True)
class StressCandidate:
    support_kp: float
    spur_pa: float
    stem_pa: float


@dataclass(frozen=True)
class CycleResult:
    cycle: int
    build_s: float
    rss_mib: float
    hwm_mib: float
    mujoco_hits: int
    mujoco_misses: int
    support_kp_min: float
    support_kp_max: float
    spur_pa_min: float
    spur_pa_max: float
    stem_pa_min: float
    stem_pa_max: float


@dataclass(frozen=True)
class ReplayResetCycleResult:
    wave: int
    reset_idx: int
    reset_s: float
    init_s: float
    step_s: float
    total_s: float
    rss_mib: float
    hwm_mib: float
    replay_steps: int


@dataclass(frozen=True)
class HybridBuildResult:
    wave: int
    build_s: float
    rss_mib: float
    hwm_mib: float
    mujoco_hits: int
    mujoco_misses: int
    support_kp_min: float
    support_kp_max: float
    spur_pa_min: float
    spur_pa_max: float
    stem_pa_min: float
    stem_pa_max: float


@dataclass
class ReplayResetSetup:
    env: Any
    dataset: BatchedSysIdDataset
    sources: tuple[ReplayEpisodeSource, ...]
    recorded_actions: np.ndarray | None


def read_rss_mib() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return -1.0


def read_hwm_mib() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    return -1.0


def build_stress_config(
    *,
    num_envs: int,
    device: str | None,
    settle_substeps: int,
    post_grasp_settle_substeps: int,
    reuse_mujoco: bool,
    topology_seed: int,
    ranges_path: Path | str | None = None,
) -> BatchedHeterogeneousCoupledSimConfig:
    cfg = BatchedHeterogeneousCoupledSimConfig.gym_defaults(num_envs=int(num_envs))
    path = default_ranges_fixture_path() if ranges_path is None else Path(ranges_path)
    return dataclasses.replace(
        cfg,
        runtime=dataclasses.replace(cfg.runtime, device=device),
        scene=dataclasses.replace(
            cfg.scene,
            settle_substeps=int(settle_substeps),
            settle_gravity_ramp=False,
            settle_quiet_every=50,
            post_grasp_settle_substeps=int(post_grasp_settle_substeps),
        ),
        robot=dataclasses.replace(
            cfg.robot,
            fix_to_apple=True,
            force_batched_layout=True,
            reuse_replicated_mujoco=bool(reuse_mujoco),
        ),
        domain_randomization=dataclasses.replace(
            cfg.domain_randomization,
            ranges_path=path,
            topology_seed=int(topology_seed),
        ),
    )


def parse_log10_bounds(
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("log10 bounds must have length 3")
    lo = tuple(float(v) for v in lower)
    hi = tuple(float(v) for v in upper)
    for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
        if a > b:
            raise ValueError(f"log10 bound pair {i} has lower > upper: {a} > {b}")
    return lo, hi


def sample_random_candidates(
    *,
    rng: random.Random,
    num_envs: int,
    log10_lower: tuple[float, float, float],
    log10_upper: tuple[float, float, float],
) -> list[StressCandidate]:
    out: list[StressCandidate] = []
    for _ in range(int(num_envs)):
        log10 = [rng.uniform(lo, hi) for lo, hi in zip(log10_lower, log10_upper, strict=True)]
        out.append(
            StressCandidate(
                support_kp=10.0 ** log10[0],
                spur_pa=10.0 ** log10[1],
                stem_pa=10.0 ** log10[2],
            )
        )
    return out


def make_params_list_from_candidates(
    *,
    base_params: FruitingSystemParams,
    candidates: Sequence[StressCandidate],
) -> tuple[list[FruitingSystemParams], list[float]]:
    params_list: list[FruitingSystemParams] = []
    support_kps: list[float] = []
    for candidate in candidates:
        params = set_rod_youngs_modulus(base_params, "spur", candidate.spur_pa)
        params = set_rod_youngs_modulus(params, "stem", candidate.stem_pa)
        params_list.append(params)
        support_kps.append(float(candidate.support_kp))
    return params_list, support_kps


def _candidate_ranges(candidates: Sequence[StressCandidate]) -> dict[str, float]:
    support = [c.support_kp for c in candidates]
    spur = [c.spur_pa for c in candidates]
    stem = [c.stem_pa for c in candidates]
    return {
        "support_kp_min": min(support),
        "support_kp_max": max(support),
        "spur_pa_min": min(spur),
        "spur_pa_max": max(spur),
        "stem_pa_min": min(stem),
        "stem_pa_max": max(stem),
    }


def resolve_replay_episode_sources(
    *,
    structure_idx: int,
    num_envs: int,
    direction_indices: Sequence[int] | None,
) -> tuple[ReplayEpisodeSource, ...]:
    if int(num_envs) < 1:
        raise ValueError("num_envs must be >= 1")
    if direction_indices is None:
        dirs = list(range(int(num_envs)))
    else:
        dirs = [int(d) for d in direction_indices]
        if not dirs:
            raise ValueError("direction_indices must be non-empty when provided")
    return tuple(
        ReplayEpisodeSource(int(structure_idx), int(dirs[env_idx % len(dirs)]))
        for env_idx in range(int(num_envs))
    )


def clone_step_obs_cpu(env: Any) -> dict[str, np.ndarray]:
    """CPU copy of CMA collector fields plus ``apple_quat`` (no torch/Warp alias)."""
    last_obs = getattr(env, "_last_obs", None)
    if last_obs is None:
        raise RuntimeError("clone_step_obs_cpu requires env._last_obs after reset/step")
    cloned = download_batched_replay_obs_numpy(last_obs, list(env.junction_names))
    cloned["apple_quat"] = _torch_to_numpy_f32_copy(last_obs["apple_quat"])
    return cloned


def assert_cloned_obs_matches_sim(
    env: Any,
    cloned: dict[str, np.ndarray],
    *,
    atol: float = 1.0e-5,
) -> None:
    """Check cloned ``apple_quat`` / ``ft_wrist`` against live sim buffers."""
    layout = env._sim.layout
    scene = env._sim.scene
    if layout is None:
        raise RuntimeError("batched scene missing layout")
    body_q = np.asarray(scene.cable.state_0.body_q.numpy(), dtype=np.float32).reshape(-1, 7)
    cache = scene.coupling_forces_cache
    if cache is None:
        raise RuntimeError("coupling_forces_cache missing for ft_wrist check")
    cache_np = np.asarray(cache.numpy(), dtype=np.float32).reshape(-1, 6)
    apple_quat = np.asarray(cloned["apple_quat"], dtype=np.float32)
    ft_wrist = np.asarray(cloned["ft_wrist"], dtype=np.float32)
    n = int(env.num_envs)
    expected_quat = np.stack(
        [body_q[int(layout.apple_body_indices[w]), 3:7] for w in range(n)],
        axis=0,
    )
    expected_ft = np.stack(
        [cache_np[int(layout.tcp_body_indices[w])] for w in range(n)],
        axis=0,
    )
    np.testing.assert_allclose(
        apple_quat,
        expected_quat,
        atol=atol,
        rtol=0.0,
        err_msg="apple_quat does not match cable body_q[apple, 3:7]",
    )
    np.testing.assert_allclose(
        ft_wrist,
        expected_ft,
        atol=atol,
        rtol=0.0,
        err_msg="ft_wrist does not match coupling_forces_cache[tcp]",
    )


def stack_recorded_actions_for_sources(
    dataset: BatchedSysIdDataset,
    sources: Sequence[ReplayEpisodeSource],
    *,
    action_dim: int,
) -> np.ndarray:
    per_env_actions: list[np.ndarray] = []
    for source in sources:
        arrays = strip_pre_weld_rows(
            dataset.load_episode_obs_arrays(source.structure_idx, source.direction_idx)
        )
        action = np.asarray(arrays["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != int(action_dim):
            raise ValueError(
                f"expected action shape (n_frames, {action_dim}), got {action.shape!r}"
            )
        per_env_actions.append(action)
    t_max = max(int(action.shape[0]) for action in per_env_actions)
    out = np.empty((len(sources), t_max, int(action_dim)), dtype=np.float32)
    for env_idx, action in enumerate(per_env_actions):
        n_frames = int(action.shape[0])
        out[env_idx, :n_frames] = action
        if n_frames < t_max:
            out[env_idx, n_frames:] = action[-1]
    return out


def teardown_sim(sim: BatchedHeterogeneousCoupledSim | None) -> None:
    del sim
    gc.collect()
    wp.synchronize()


def teardown_env(env: Any | None) -> None:
    if env is not None:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    gc.collect()
    wp.synchronize()


def run_rebuild_cycle(
    *,
    config: BatchedHeterogeneousCoupledSimConfig,
    ranges: dict,
    cycle: int,
    base_params: FruitingSystemParams,
    rng: random.Random,
    log10_lower: tuple[float, float, float],
    log10_upper: tuple[float, float, float],
    apply_support_kp: bool,
) -> CycleResult:
    candidates = sample_random_candidates(
        rng=rng,
        num_envs=config.runtime.num_envs,
        log10_lower=log10_lower,
        log10_upper=log10_upper,
    )
    params_list, support_kps = make_params_list_from_candidates(
        base_params=base_params,
        candidates=candidates,
    )
    t0 = time.perf_counter()
    sim = BatchedHeterogeneousCoupledSim(
        config,
        params_list,
        ranges,
        use_settle_cache=False,
    )
    if apply_support_kp:
        apply_per_env_support_joint_penalties(
            sim.scene,
            support_kps,
            num_envs=sim.layout.num_envs,
            joints_per_world=sim.layout.joints_per_world,
        )
    build_s = time.perf_counter() - t0
    cache = process_replicated_robot_cache()
    cand_ranges = _candidate_ranges(candidates)
    result = CycleResult(
        cycle=int(cycle),
        build_s=float(build_s),
        rss_mib=read_rss_mib(),
        hwm_mib=read_hwm_mib(),
        mujoco_hits=int(cache.hits),
        mujoco_misses=int(cache.misses),
        **cand_ranges,
    )
    teardown_sim(sim)
    return result


def format_cycle_result(result: CycleResult) -> str:
    return (
        f"cycle={result.cycle} build_s={result.build_s:.2f} "
        f"rss_mib={result.rss_mib:.1f} hwm_mib={result.hwm_mib:.1f} "
        f"mujoco_hits={result.mujoco_hits} mujoco_misses={result.mujoco_misses} "
        f"support_kp=[{result.support_kp_min:.3e},{result.support_kp_max:.3e}] "
        f"spur_E=[{result.spur_pa_min:.3e},{result.spur_pa_max:.3e}] "
        f"stem_E=[{result.stem_pa_min:.3e},{result.stem_pa_max:.3e}]"
    )


def run_rebuild_cycles(
    *,
    config: BatchedHeterogeneousCoupledSimConfig,
    ranges: dict,
    cycles: int,
    params_seed: int = 0,
    log10_lower: tuple[float, float, float] = DEFAULT_LOG10_LOWER,
    log10_upper: tuple[float, float, float] = DEFAULT_LOG10_UPPER,
    apply_support_kp: bool = True,
    base_params: FruitingSystemParams | None = None,
    jsonl: TextIO | None = None,
) -> list[CycleResult]:
    if base_params is None:
        base_params = sample_heterogeneous_params_list(
            ranges,
            topology_seed=config.domain_randomization.topology_seed or 0,
            num_envs=1,
        )[0]
    results: list[CycleResult] = []
    for cycle in range(int(cycles)):
        rng = random.Random(int(params_seed) + int(cycle))
        result = run_rebuild_cycle(
            config=config,
            ranges=ranges,
            cycle=cycle,
            base_params=base_params,
            rng=rng,
            log10_lower=log10_lower,
            log10_upper=log10_upper,
            apply_support_kp=apply_support_kp,
        )
        results.append(result)
        line = format_cycle_result(result)
        print(line, flush=True)
        if jsonl is not None:
            jsonl.write(json.dumps(dataclasses.asdict(result)) + "\n")
            jsonl.flush()
    return results


def build_replay_reset_setup(
    *,
    dataset: BatchedSysIdDataset,
    num_envs: int,
    structure_idx: int,
    direction_indices: Sequence[int] | None,
    settle_substeps: int,
    post_grasp_settle_substeps: int,
    topology_seed: int,
    reuse_mujoco: bool,
    params_seed: int,
    log10_lower: tuple[float, float, float],
    log10_upper: tuple[float, float, float],
    apply_support_kp: bool,
    replay_steps: int,
) -> tuple[ReplayResetSetup, float]:
    """Build one replay env; return setup and wall build seconds."""
    collection = dataset.manifest.get("collection", {})
    if not dataset_declares_vic_pose(collection):
        raise ValueError("replay-reset mode requires a vic_pose dataset")

    sources = resolve_replay_episode_sources(
        structure_idx=int(structure_idx),
        num_envs=int(num_envs),
        direction_indices=direction_indices,
    )
    first_meta = dataset.load_episode_metadata(
        sources[0].structure_idx,
        sources[0].direction_idx,
    )
    per_env_meta = [
        dataset.load_episode_metadata(source.structure_idx, source.direction_idx)
        for source in sources
    ]
    ranges_path = Path(str(collection.get("ranges_path") or first_meta.get("ranges_path")))
    ranges = load_ranges(ranges_path)
    real_topology_seed = int(collection.get("topology_seed", topology_seed))
    control_hz = control_hz_from_episode_metadata(first_meta, collection=collection)
    fruiting_base_pos = fruiting_base_pos_from_episode_metadata(first_meta)
    bootstrap_joint_q = bootstrap_joint_q_from_episode_metadata(first_meta)

    build_env_fn = make_real_replay_build_env_fn(
        ranges_path=ranges_path,
        ranges=ranges,
        topology_seed=real_topology_seed,
        fruiting_base_pos=fruiting_base_pos,
        episode_meta=first_meta,
        settle_substeps=int(settle_substeps),
        settle_quiet_every=50,
        settle_gravity_ramp=False,
        post_grasp_settle_substeps=int(post_grasp_settle_substeps),
        bootstrap_joint_q=bootstrap_joint_q,
        controller_mode="vic_pose",
        control_hz=float(control_hz),
        reuse_replicated_mujoco=bool(reuse_mujoco),
    )

    base_params = true_params_for_structure(dataset, int(structure_idx))
    rng = random.Random(int(params_seed))
    candidates = sample_random_candidates(
        rng=rng,
        num_envs=int(num_envs),
        log10_lower=log10_lower,
        log10_upper=log10_upper,
    )
    params_list, support_kps = make_params_list_from_candidates(
        base_params=base_params,
        candidates=candidates,
    )
    grippers = [
        gripper_proxy_for_real_batched_replay(meta) for meta in per_env_meta
    ]
    max_episode_steps = int(collection.get("max_steps") or 240)

    t0 = time.perf_counter()
    env = build_env_fn(
        num_envs=int(num_envs),
        per_env_params=params_list,
        max_episode_steps=max_episode_steps,
        per_env_grippers=grippers,
        per_env_episode_meta=per_env_meta,
    )
    if apply_support_kp:
        apply_per_env_support_joint_penalties(
            env._sim.scene,
            support_kps,
            num_envs=env._sim.layout.num_envs,
            joints_per_world=env._sim.layout.joints_per_world,
        )
    build_s = time.perf_counter() - t0

    action_dim = int(collection.get("action_dim") or 19)
    recorded_actions = None
    if int(replay_steps) > 0:
        recorded_actions = stack_recorded_actions_for_sources(
            dataset,
            sources,
            action_dim=action_dim,
        )

    return (
        ReplayResetSetup(
            env=env,
            dataset=dataset,
            sources=sources,
            recorded_actions=recorded_actions,
        ),
        float(build_s),
    )


def run_replay_reset_cycle(
    *,
    setup: ReplayResetSetup,
    wave: int,
    reset_idx: int,
    replay_seed: int,
    replay_steps: int,
) -> ReplayResetCycleResult:
    """One reset → episode init → optional replay steps on a live env."""
    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
        actions_tensor_from_recorded_frame,
    )

    env = setup.env
    seed = int(replay_seed) + int(wave) * 10_000 + int(reset_idx)
    t0 = time.perf_counter()
    env.reset(seed=seed)
    reset_s = time.perf_counter() - t0
    print(
        f"wave={wave} reset={reset_idx} reset_ok reset_s={reset_s:.3f}",
        flush=True,
    )

    t1 = time.perf_counter()
    initialize_batched_env_from_episode_sources(env, setup.dataset, setup.sources)
    init_s = time.perf_counter() - t1
    print(
        f"wave={wave} reset={reset_idx} init_ok init_s={init_s:.3f}",
        flush=True,
    )

    step_s = 0.0
    steps = int(replay_steps)
    if steps > 0:
        if setup.recorded_actions is None:
            raise RuntimeError("recorded_actions missing despite replay_steps > 0")
        t2 = time.perf_counter()
        n_frames = int(setup.recorded_actions.shape[1])
        for frame_idx in range(min(steps, n_frames)):
            actions = actions_tensor_from_recorded_frame(
                setup.recorded_actions,
                frame_idx=frame_idx,
                device=env.device,
            )
            t_frame = time.perf_counter()
            env.step(actions)
            print(
                f"wave={wave} reset={reset_idx} step_frame={frame_idx} "
                f"step_ok step_s={time.perf_counter() - t_frame:.3f}",
                flush=True,
            )
            t_collect = time.perf_counter()
            cloned = clone_step_obs_cpu(env)
            assert_cloned_obs_matches_sim(env, cloned)
            print(
                f"wave={wave} reset={reset_idx} step_frame={frame_idx} "
                f"collect_ok collect_s={time.perf_counter() - t_collect:.3f}",
                flush=True,
            )
        step_s = time.perf_counter() - t2

    total_s = reset_s + init_s + step_s
    return ReplayResetCycleResult(
        wave=int(wave),
        reset_idx=int(reset_idx),
        reset_s=float(reset_s),
        init_s=float(init_s),
        step_s=float(step_s),
        total_s=float(total_s),
        rss_mib=read_rss_mib(),
        hwm_mib=read_hwm_mib(),
        replay_steps=steps,
    )


def format_replay_reset_cycle_result(result: ReplayResetCycleResult) -> str:
    return (
        f"wave={result.wave} reset={result.reset_idx} "
        f"reset_s={result.reset_s:.3f} init_s={result.init_s:.3f} "
        f"step_s={result.step_s:.3f} total_s={result.total_s:.3f} "
        f"replay_steps={result.replay_steps} "
        f"rss_mib={result.rss_mib:.1f} hwm_mib={result.hwm_mib:.1f}"
    )


def format_hybrid_build_result(result: HybridBuildResult) -> str:
    return (
        f"wave={result.wave} build_s={result.build_s:.2f} "
        f"rss_mib={result.rss_mib:.1f} hwm_mib={result.hwm_mib:.1f} "
        f"mujoco_hits={result.mujoco_hits} mujoco_misses={result.mujoco_misses} "
        f"support_kp=[{result.support_kp_min:.3e},{result.support_kp_max:.3e}] "
        f"spur_E=[{result.spur_pa_min:.3e},{result.spur_pa_max:.3e}] "
        f"stem_E=[{result.stem_pa_min:.3e},{result.stem_pa_max:.3e}]"
    )


def run_replay_reset_cycles(
    *,
    setup: ReplayResetSetup,
    cycles: int,
    wave: int = 0,
    replay_seed: int = 0,
    replay_steps: int = 0,
    jsonl: TextIO | None = None,
) -> list[ReplayResetCycleResult]:
    results: list[ReplayResetCycleResult] = []
    for reset_idx in range(int(cycles)):
        result = run_replay_reset_cycle(
            setup=setup,
            wave=int(wave),
            reset_idx=int(reset_idx),
            replay_seed=int(replay_seed),
            replay_steps=int(replay_steps),
        )
        results.append(result)
        line = format_replay_reset_cycle_result(result)
        print(line, flush=True)
        if jsonl is not None:
            payload = {"kind": "reset", **dataclasses.asdict(result)}
            jsonl.write(json.dumps(payload) + "\n")
            jsonl.flush()
    return results


def run_rebuild_replay_waves(
    *,
    dataset: BatchedSysIdDataset,
    num_envs: int,
    waves: int,
    resets_per_wave: int,
    structure_idx: int,
    direction_indices: Sequence[int] | None,
    settle_substeps: int,
    post_grasp_settle_substeps: int,
    topology_seed: int,
    reuse_mujoco: bool,
    params_seed: int,
    log10_lower: tuple[float, float, float],
    log10_upper: tuple[float, float, float],
    apply_support_kp: bool,
    replay_steps: int,
    replay_seed: int,
    clear_robot_cache_each_wave: bool,
    jsonl: TextIO | None = None,
) -> None:
    """Rebuild plant, run reset loops, teardown — repeat for each wave."""
    setup: ReplayResetSetup | None = None
    try:
        for wave in range(int(waves)):
            if clear_robot_cache_each_wave and wave > 0:
                clear_process_replicated_robot_cache()
            if setup is not None:
                teardown_env(setup.env)
                setup = None

            setup, build_s = build_replay_reset_setup(
                dataset=dataset,
                num_envs=int(num_envs),
                structure_idx=int(structure_idx),
                direction_indices=direction_indices,
                settle_substeps=int(settle_substeps),
                post_grasp_settle_substeps=int(post_grasp_settle_substeps),
                topology_seed=int(topology_seed),
                reuse_mujoco=bool(reuse_mujoco),
                params_seed=int(params_seed) + int(wave),
                log10_lower=log10_lower,
                log10_upper=log10_upper,
                apply_support_kp=apply_support_kp,
                replay_steps=int(replay_steps),
            )
            cache = process_replicated_robot_cache()
            rng = random.Random(int(params_seed) + int(wave))
            candidates = sample_random_candidates(
                rng=rng,
                num_envs=int(num_envs),
                log10_lower=log10_lower,
                log10_upper=log10_upper,
            )
            build_result = HybridBuildResult(
                wave=int(wave),
                build_s=float(build_s),
                rss_mib=read_rss_mib(),
                hwm_mib=read_hwm_mib(),
                mujoco_hits=int(cache.hits),
                mujoco_misses=int(cache.misses),
                **_candidate_ranges(candidates),
            )
            line = format_hybrid_build_result(build_result)
            print(line, flush=True)
            if jsonl is not None:
                payload = {"kind": "build", **dataclasses.asdict(build_result)}
                jsonl.write(json.dumps(payload) + "\n")
                jsonl.flush()

            run_replay_reset_cycles(
                setup=setup,
                cycles=int(resets_per_wave),
                wave=int(wave),
                replay_seed=int(replay_seed),
                replay_steps=int(replay_steps),
                jsonl=jsonl,
            )
    finally:
        if setup is not None:
            teardown_env(setup.env)


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")
    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")
    if args.settle_substeps < 0:
        raise SystemExit("--settle-substeps must be >= 0")
    if args.replay_steps < 0:
        raise SystemExit("--replay-steps must be >= 0")
    if args.post_grasp_settle_substeps < 0:
        raise SystemExit("--post-grasp-settle-substeps must be >= 0")
    if args.resets_per_wave < 1:
        raise SystemExit("--resets-per-wave must be >= 1")
    if args.mode in ("replay-reset", "rebuild-replay") and args.dataset is None:
        raise SystemExit(f"--mode {args.mode} requires --dataset")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("rebuild", "replay-reset", "rebuild-replay"),
        default="rebuild",
        help=(
            "rebuild: plant teardown loop; replay-reset: reset/init/steps on one env; "
            "rebuild-replay: rebuild then --resets-per-wave reset loops per --cycles wave."
        ),
    )
    p.add_argument("--num-envs", type=int, default=100)
    p.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Rebuild cycles (rebuild mode) or waves (rebuild-replay mode).",
    )
    p.add_argument(
        "--resets-per-wave",
        type=int,
        default=100,
        help="Reset/init loops after each rebuild (rebuild-replay mode only).",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--settle-substeps", type=int, default=5000)
    p.add_argument(
        "--post-grasp-settle-substeps",
        type=int,
        default=500,
        help="Welded post-grasp VBD settle after FR3 bootstrap (CMA default: 500; 0 = skip).",
    )
    p.add_argument("--topology-seed", type=int, default=42)
    p.add_argument("--ranges-path", type=Path, default=None)
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Batched sys-ID dataset directory (required for replay-reset).",
    )
    p.add_argument(
        "--structure-idx",
        type=int,
        default=0,
        help="Structure index for replay-reset episode sources.",
    )
    p.add_argument(
        "--direction-indices",
        type=int,
        nargs="+",
        default=None,
        help="Physical direction indices per env (cycles when num_envs > len).",
    )
    p.add_argument(
        "--replay-steps",
        type=int,
        default=0,
        help="Recorded-action env.step count after reset/init (0 = skip stepping).",
    )
    p.add_argument(
        "--replay-seed",
        type=int,
        default=0,
        help="Base seed passed to env.reset(seed=...).",
    )
    p.add_argument(
        "--reuse-replicated-mujoco",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache USD/MuJoCo FR3 across rebuilds (default: on).",
    )
    p.add_argument(
        "--clear-robot-cache-each-cycle",
        action="store_true",
        help="Cold-build FR3 every rebuild cycle/wave.",
    )
    p.add_argument(
        "--params-seed",
        type=int,
        default=0,
        help="Seed for per-cycle random CMA-bounds candidate draws.",
    )
    p.add_argument(
        "--log10-lower",
        type=float,
        nargs=3,
        default=list(DEFAULT_LOG10_LOWER),
        metavar=("LOG10_KP", "LOG10_SPUR", "LOG10_STEM"),
        help="Lower log10 search bounds (support_kp, spur E, stem E).",
    )
    p.add_argument(
        "--log10-upper",
        type=float,
        nargs=3,
        default=list(DEFAULT_LOG10_UPPER),
        metavar=("LOG10_KP", "LOG10_SPUR", "LOG10_STEM"),
        help="Upper log10 search bounds (support_kp, spur E, stem E).",
    )
    p.add_argument(
        "--no-apply-support-kp",
        action="store_true",
        help="Skip post-build support joint k_p patch (CMA applies this per env).",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Append one JSON object per cycle to this file.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable()
    args = make_parser().parse_args(argv)
    validate_cli_args(args)

    device = resolve_sim_device(args.device)
    log10_lower, log10_upper = parse_log10_bounds(args.log10_lower, args.log10_upper)

    jsonl_handle: TextIO | None = None
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = args.jsonl.open("a", encoding="utf-8")

    print(
        f"stress_plant_rebuild_loop mode={args.mode} num_envs={args.num_envs} "
        f"cycles={args.cycles} resets_per_wave={args.resets_per_wave} "
        f"device={device} settle_substeps={args.settle_substeps} "
        f"post_grasp_settle_substeps={args.post_grasp_settle_substeps} "
        f"reuse_mujoco={args.reuse_replicated_mujoco} params_seed={args.params_seed} "
        f"replay_steps={args.replay_steps} log10_lower={log10_lower} "
        f"log10_upper={log10_upper}",
        flush=True,
    )

    try:
        if args.mode in ("replay-reset", "rebuild-replay"):
            dataset = BatchedSysIdDataset(str(args.dataset))
            if args.mode == "rebuild-replay":
                run_rebuild_replay_waves(
                    dataset=dataset,
                    num_envs=args.num_envs,
                    waves=args.cycles,
                    resets_per_wave=args.resets_per_wave,
                    structure_idx=args.structure_idx,
                    direction_indices=args.direction_indices,
                    settle_substeps=args.settle_substeps,
                    post_grasp_settle_substeps=args.post_grasp_settle_substeps,
                    topology_seed=args.topology_seed,
                    reuse_mujoco=bool(args.reuse_replicated_mujoco),
                    params_seed=args.params_seed,
                    log10_lower=log10_lower,
                    log10_upper=log10_upper,
                    apply_support_kp=not args.no_apply_support_kp,
                    replay_steps=args.replay_steps,
                    replay_seed=args.replay_seed,
                    clear_robot_cache_each_wave=args.clear_robot_cache_each_cycle,
                    jsonl=jsonl_handle,
                )
                return 0

            setup, build_s = build_replay_reset_setup(
                dataset=dataset,
                num_envs=args.num_envs,
                structure_idx=args.structure_idx,
                direction_indices=args.direction_indices,
                settle_substeps=args.settle_substeps,
                post_grasp_settle_substeps=args.post_grasp_settle_substeps,
                topology_seed=args.topology_seed,
                reuse_mujoco=bool(args.reuse_replicated_mujoco),
                params_seed=args.params_seed,
                log10_lower=log10_lower,
                log10_upper=log10_upper,
                apply_support_kp=not args.no_apply_support_kp,
                replay_steps=args.replay_steps,
            )
            print(
                f"build_s={build_s:.2f} rss_mib={read_rss_mib():.1f} "
                f"hwm_mib={read_hwm_mib():.1f} sources={len(setup.sources)}",
                flush=True,
            )
            run_replay_reset_cycles(
                setup=setup,
                cycles=args.cycles,
                replay_seed=args.replay_seed,
                replay_steps=args.replay_steps,
                jsonl=jsonl_handle,
            )
            teardown_env(setup.env)
            return 0

        config = build_stress_config(
            num_envs=args.num_envs,
            device=device,
            settle_substeps=args.settle_substeps,
            post_grasp_settle_substeps=args.post_grasp_settle_substeps,
            reuse_mujoco=bool(args.reuse_replicated_mujoco),
            topology_seed=args.topology_seed,
            ranges_path=args.ranges_path,
        )
        ranges = load_ranges(config.domain_randomization.ranges_path)
        base_params = sample_heterogeneous_params_list(
            ranges,
            topology_seed=config.domain_randomization.topology_seed or 0,
            num_envs=1,
        )[0]
        for cycle in range(int(args.cycles)):
            if args.clear_robot_cache_each_cycle and cycle > 0:
                clear_process_replicated_robot_cache()
            rng = random.Random(int(args.params_seed) + cycle)
            result = run_rebuild_cycle(
                config=config,
                ranges=ranges,
                cycle=cycle,
                base_params=base_params,
                rng=rng,
                log10_lower=log10_lower,
                log10_upper=log10_upper,
                apply_support_kp=not args.no_apply_support_kp,
            )
            line = format_cycle_result(result)
            print(line, flush=True)
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(dataclasses.asdict(result)) + "\n")
                jsonl_handle.flush()
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
