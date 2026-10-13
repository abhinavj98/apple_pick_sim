"""[D17n] Stem-only bending angle on a real build (one build, one test).

With ``bend_frame="spur"`` the rest stem chord is stored in the spur tip body's frame (the spur-stem
junction's parent) and carried by its rotation, so ``theta_b`` reads how far the stem has bent
relative to its spur; rigid rotation of the spur / plant no longer counts. The world-frame angle is
published alongside as ``info["theta_b_world_deg"]``. ``bend_pull_theta_max_deg`` holds ``F_d`` flat
past the fitted range.

Runs on CUDA when available (the arm integrates); on CPU it checks the wiring only. Run on its own
(one build per process): ``pytest apple_pick_gym/tests/test_harvest_env_stem_bend_frame.py``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_WORLDS = Path(__file__).resolve().parent.parent / "world_sets" / "harvest_worlds_v1.jsonl"


def test_spur_frame_bend_angle_at_rest_and_under_a_sideways_drag():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
    from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig, bend_pull_detach_force
    from apple_pick_gym.batched_envs.harvest_episode import EpisodeConfig
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig
    from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs

    cuda = torch.cuda.is_available()
    n, steps = 2, 40
    specs = load_world_set(_WORLDS)[:n]
    # F_d scaled out of reach so nothing detaches / freezes during the drag
    det = DetachEnvelopeConfig(
        criterion="bend_pull", bend_frame="spur", bend_pull_theta_max_deg=30.0, detach_scale_range=(100.0, 100.0)
    )
    warn = contextlib.nullcontext() if cuda else pytest.warns(UserWarning, match="does not integrate")
    with warn:
        env = ApplePickVicHarvestEnv(
            device="cuda:0" if cuda else "cpu",
            max_episode_steps=steps + 5,
            action_bounds=HarvestActionBounds(linear_delta_m=0.002, k_lin_max=500.0),
            action_frame="world",
            reward_config=HarvestRewardConfig(detach=det),
            episode_config=EpisodeConfig(safety_force_cap_n=1e12, safety_torque_cap_nm=1e12),
            **world_specs_to_env_kwargs(specs),
        )
    try:
        assert not torch.equal(env._target_parent_body, env._target_child_body)
        _obs, info = env.reset()
        zero = torch.zeros(n, device=env.device)
        torch.testing.assert_close(info["theta_b_deg"], zero, atol=1e-2, rtol=0)
        torch.testing.assert_close(info["theta_b_world_deg"], zero, atol=1e-2, rtol=0)
        torch.testing.assert_close(info["detach_force"], bend_pull_detach_force(info["theta_b_deg"], info["detach_scale"], det))

        a = torch.zeros(n, 13, device=env.device)
        a[:, 0] = 0.002  # drag the apple sideways (world +x), 2 mm/step
        a[:, 6:9], a[:, 9:12], a[:, 12] = 500.0, 20.0, 1.0
        for _ in range(steps):
            _obs, _r, _t, _tr, info = env.step(a)
        th, th_w = info["theta_b_deg"], info["theta_b_world_deg"]
        assert bool(torch.isfinite(th).all() and torch.isfinite(th_w).all())
        assert not bool(info["episode"]["frozen"].any())
        torch.testing.assert_close(info["detach_force"], bend_pull_detach_force(th, info["detach_scale"], det))
        print(f"\ncuda={cuda} after {steps} steps: theta_b stem-only {th.tolist()} deg, world {th_w.tolist()} deg")
        if cuda:
            assert float(th_w.min()) > 1.0, th_w  # the drag moves the stem chord
            assert not torch.allclose(th, th_w, atol=0.1), (th, th_w)  # the spur turns too
    finally:
        env.close()
