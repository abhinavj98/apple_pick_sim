"""ApplePickVicHarvestEnv's RL-facing contract on a real build (one build, one test).

Checks what the skrl wrapper and critic rely on:

- ``info["target_junction_wrench"]`` is the anchor-frame wrench and
  ``info["detach_index"]`` is the envelope index computed from it;
- ``info["collateral_baseline_norm"]`` is recorded at reset for every non-target junction;
- ``terminated`` fires exactly once per env (the freeze edge), then the env stays frozen
  with zero reward -- forced here with a 1 N envelope the rest load already exceeds;
- DR: arm joint dynamics and the F/T sensor bias resample on reset, and
  ``privileged_fields()`` follows the per-reset arm sample; plant/support/geometry are
  per env and fixed.

CPU build: the arm does not integrate on CPU (see the env's warning), which does not
matter here -- everything asserted is wiring and bookkeeping. Run on its own
(one build per process): ``pytest apple_pick_gym/tests/test_harvest_env_rl_contract.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

_WORLDS = Path(__file__).resolve().parent.parent / "world_sets" / "harvest_worlds_v1.jsonl"


def test_rl_contract_and_domain_randomization():
    from apple_pick_gym.batched_envs.apple_pick_vic_harvest_env import ApplePickVicHarvestEnv
    from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
    from apple_pick_gym.batched_envs.harvest_detach import DetachEnvelopeConfig, detach_index
    from apple_pick_gym.batched_envs.harvest_obs import _PRIVILEGED_FIELDS
    from apple_pick_gym.batched_envs.harvest_reward import HarvestRewardConfig
    from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig
    from apple_pick_gym.batched_envs.world_set import load_world_set, world_specs_to_env_kwargs

    n = 2
    specs = load_world_set(_WORLDS)[:n]
    with pytest.warns(UserWarning, match="does not integrate"):
        env = ApplePickVicHarvestEnv(
            device="cpu",
            max_episode_steps=12,
            reward_config=HarvestRewardConfig(detach=DetachEnvelopeConfig(f_max_n=1.0, tau_max_nm=0.05)),
            action_bounds=HarvestActionBounds(max_target_pos_offset_m=0.05, max_target_rot_offset_rad=0.3),
            ft_sensor_config=FtSensorConfig.rl_training(),
            **world_specs_to_env_kwargs(specs),
        )
    try:
        obs, info = env.reset()
        tj = env.TARGET_JUNCTION_NAME
        # --- anchor-frame wrench + detach index
        w = info["target_junction_wrench"]
        assert w.shape == (n, 6)
        torch.testing.assert_close(w[:, :3], info["target_junction_force"][:, :3])
        torch.testing.assert_close(info["detach_index"], detach_index(w, env._reward_cfg.detach))
        # --- collateral rest baseline
        base = info["collateral_baseline_norm"]
        assert set(base) == set(env._junction_names) - {tj}
        for name, v in base.items():
            torch.testing.assert_close(v, torch.linalg.norm(info["woody_part_force"][name][:, :3], dim=-1))

        # --- DR: privileged fields, geometry, per-reset arm DR and sensor bias
        priv0 = env.privileged_fields()
        assert list(priv0) == [k for k, _ in _PRIVILEGED_FIELDS]
        geo = env.plant_geometry()
        torch.testing.assert_close(geo["weld_direction"], torch.tensor([s.weld_direction for s in specs]))
        assert len(set(env._sim.config.fruiting_system.support_kp_per_env)) == n
        arm0 = env._last_arm_dr_sample.friction.copy()
        bias0 = env._ft_sensor._bias.clone()
        assert float(bias0.abs().sum()) > 0.0  # sensor DR is on
        torch.testing.assert_close(priv0["arm_friction"], torch.as_tensor(arm0))

        # --- terminated fires once: the 1 N envelope is exceeded by the rest load
        action = torch.zeros(n, 13)
        action[:, 6:9] = 100.0
        action[:, 9:12] = 10.0
        action[:, 12] = 1.0
        terms, rewards, frozen = [], [], []
        for _ in range(12):
            obs, r, term, trunc, info = env.step(action)
            terms.append(term.flatten().clone())
            rewards.append(r.flatten().clone())
            frozen.append(info["episode"]["frozen"].clone())
        terms_t = torch.stack(terms)
        assert terms_t.sum(0).tolist() == [1, 1]
        edge_step = int(terms_t[:, 0].nonzero()[0])
        assert edge_step == env._episode_cfg.success_streak_steps - 1
        assert bool(torch.stack(frozen)[edge_step:].all())
        assert float(torch.stack(rewards)[edge_step + 1 :].abs().max()) == 0.0
        assert bool(trunc.all())  # synchronized time limit

        # --- reset resamples arm joint DR and the sensor bias; privileged follows
        env.reset()
        assert not np.array_equal(arm0, env._last_arm_dr_sample.friction)
        assert not torch.equal(bias0, env._ft_sensor._bias)
        priv1 = env.privileged_fields()
        torch.testing.assert_close(priv1["arm_friction"], torch.as_tensor(env._last_arm_dr_sample.friction))
        for k in ("stem_youngs_modulus_pa", "support_kp", "arm_link_mass_scale"):
            torch.testing.assert_close(priv1[k], priv0[k])  # build-time DR is fixed
    finally:
        env.close()
