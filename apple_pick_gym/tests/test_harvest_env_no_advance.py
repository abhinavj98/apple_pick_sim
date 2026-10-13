"""[D17m] The no-advance wall and deviation collateral on a real build (one build, one test).

With ``max_advance_m`` set, the VIC target never passes the plane through the grasp TCP normal to
the grip axis at reset (TCP +z, into the apple and the tree); pulling back is untouched. The env
publishes the TCP's own advance (``info["tcp_advance_m"]``, compliance can carry it past the
plane) and the rest force vectors for ``collateral_mode="deviation"``.

Runs on CUDA when available (the arm integrates); on CPU it checks the wiring only. Run on its own
(one build per process): ``pytest apple_pick_gym/tests/test_harvest_env_no_advance.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_WORLDS = Path(__file__).resolve().parent.parent / "world_sets" / "harvest_worlds_v1.jsonl"


def _push_in(env, axis, steps):
    """Push the target 2 mm/step along the reset grip axis (world frame, into the apple); last info.

    World frame on purpose: a tool +z push turns with the TCP as it tilts and stops pointing along the
    reset axis (measured: the target stalls at ~15 mm). The wall does not depend on the action frame.
    """
    a = torch.zeros(env.num_envs, 13, device=env.device)
    a[:, :3] = 0.002 * axis
    a[:, 6:9], a[:, 9:12], a[:, 12] = 300.0, 20.0, 1.0
    info = None
    for _ in range(steps):
        _obs, _r, _t, _tr, info = env.step(a)
    return info


def test_no_advance_wall_and_rest_force_vectors():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds, grip_axis
    from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig
    from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig, compute_collateral_penalty
    from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs

    cuda = torch.cuda.is_available()
    n, steps = 2, 30
    specs = load_world_set(_WORLDS)[:n]
    bounds = HarvestActionBounds(
        linear_delta_m=0.002, max_target_pos_offset_m=0.15, max_target_rot_offset_rad=0.5, k_lin_max=500.0
    )
    warn = contextlib.nullcontext() if cuda else pytest.warns(UserWarning, match="does not integrate")
    with warn:
        env = ApplePickVicHarvestEnv(
            device="cuda:0" if cuda else "cpu",
            max_episode_steps=steps + 5,
            action_bounds=bounds,
            action_frame="world",
            # detach and the safety trip out of reach (as in v2): either would freeze the env and hold
            # its target (pushing in tripped the 40 N wrist cap at step 23 on world 0)
            reward_config=HarvestRewardConfig(
                detach=DetachEnvelopeConfig(f_max_n=1e6, tau_max_nm=1e6), collateral_mode="deviation"
            ),
            episode_config=EpisodeConfig(safety_force_cap_n=1e12, safety_torque_cap_nm=1e12),
            **world_specs_to_env_kwargs(specs),
        )
    try:
        obs0, info = env.reset()
        base = info["collateral_baseline_force"]
        assert set(base) == set(info["collateral_baseline_norm"])
        at_rest = compute_collateral_penalty(
            info["woody_part_force"], target_junction_name=env.target_junction_name, baseline_force=base, mode="deviation"
        )
        torch.testing.assert_close(at_rest, torch.zeros(n, device=at_rest.device), atol=1e-4, rtol=0)
        torch.testing.assert_close(info["tcp_advance_m"], torch.zeros(n, device=env.device), atol=1e-6, rtol=0)
        # obs["tcp_pos"] / ["tcp_quat"] are views of the sim's buffers: copy the start pose
        start, axis = obs0["tcp_pos"].clone(), grip_axis(obs0["tcp_quat"]).clone()

        def target_advance(info):
            return ((info["target_pose"][:, :3] - start) * axis).sum(-1)

        off = _push_in(env, axis, steps)  # wall off: the target goes in
        adv_off, tcp_off = target_advance(off), off["tcp_advance_m"]
        assert not bool(off["episode"]["frozen"].any())
        assert float(adv_off.min()) > 0.03, adv_off

        env._action_bounds = dataclasses.replace(bounds, max_advance_m=0.0)  # same build, wall on
        obs0, _ = env.reset()
        start, axis = obs0["tcp_pos"].clone(), grip_axis(obs0["tcp_quat"]).clone()
        on = _push_in(env, axis, steps)
        adv_on, tcp_on = target_advance(on), on["tcp_advance_m"]
        assert float(adv_on.max()) <= 1e-5, adv_on
        print(f"\ncuda={cuda} TCP advance mm, wall off {1000 * tcp_off} / on {1000 * tcp_on}")
        if cuda:
            assert float(tcp_off.min()) > 0.01, tcp_off  # the arm follows the target in
            # the wall holds the TCP at the plane (GPU, 2 v1 worlds, 30 steps: 41 / 46 mm off, 0.2 / 0.03 mm on)
            assert float(tcp_on.max()) < 0.002, tcp_on
    finally:
        env.close()
