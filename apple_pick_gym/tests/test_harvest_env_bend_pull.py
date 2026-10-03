"""[D17] The bend-pull detach criterion on a real build (one build, one test).

At rest the stem hangs straight from its settled chord, so ``theta_b = 0`` and the stem is in
tension under the apple's weight: ``stem_tension`` is positive and carries almost all of the
stem-root force. ``detach_index`` / ``detach_utilization`` are ``tension / F_d(theta_b)`` with the
episode's ``detach_scale``, which is re-drawn on every reset.

CPU build (wiring only; the arm does not integrate on CPU). Run on its own (one build per
process): ``pytest apple_pick_gym/tests/test_harvest_env_bend_pull.py``. The GPU check of the same
quantities under a scripted pull (16 worlds, 2026-10-02): rest tension 3.6 N (u0 0.14), rising to
~27 N, every world detached by step 60.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

_WORLDS = Path(__file__).resolve().parent.parent / "world_sets" / "harvest_worlds_v1.jsonl"


def test_bend_pull_state_at_rest_and_scale_redraw():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig, bend_pull_detach_force
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig
    from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs

    n = 2
    specs = load_world_set(_WORLDS)[:n]
    det = DetachEnvelopeConfig(criterion="bend_pull", detach_scale_range=(0.85, 1.15))
    with pytest.warns(UserWarning, match="does not integrate"):
        env = ApplePickVicHarvestEnv(
            device="cpu",
            max_episode_steps=12,
            reward_config=HarvestRewardConfig(detach=det),
            **world_specs_to_env_kwargs(specs),
        )
    try:
        _obs, info = env.reset()
        for key in ("theta_b_deg", "stem_tension", "detach_force", "detach_scale"):
            assert info[key].shape == (n,), key
        torch.testing.assert_close(info["theta_b_deg"], torch.zeros(n), atol=1e-3, rtol=0)
        tension = info["stem_tension"]
        force = torch.linalg.norm(info["target_junction_wrench"][:, :3], dim=-1)
        assert bool((tension > 0.5).all()), tension  # hanging apple: the stem is pulled, not pushed
        assert bool((tension >= 0.8 * force).all()), (tension, force)  # mostly axial at rest
        s = info["detach_scale"]
        assert bool(((s >= 0.85) & (s <= 1.15)).all())
        f_d = bend_pull_detach_force(info["theta_b_deg"], s, det)
        torch.testing.assert_close(info["detach_force"], f_d)
        torch.testing.assert_close(info["detach_utilization"], tension / f_d)
        torch.testing.assert_close(info["detach_index"], info["detach_utilization"])
        assert float(info["detach_utilization"].max()) < 0.5  # rest is far from detaching

        _obs, info2 = env.reset()
        assert not torch.equal(info2["detach_scale"], s)  # re-drawn per episode
    finally:
        env.close()
