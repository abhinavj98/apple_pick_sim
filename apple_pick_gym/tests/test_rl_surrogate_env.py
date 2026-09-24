"""SurrogateHarvestEnv: the real harvest env's RL contract on an analytic spring plant."""

from __future__ import annotations

import math
import time

import pytest
import torch

from apple_pick_gym.batched_envs.harvest_action import HarvestActionBounds
from apple_pick_gym.batched_envs.harvest_obs import _PRIVILEGED_FIELDS, actor_obs_layout, flatten_actor_obs
from apple_pick_gym.batched_envs.harvest_privileged import PLANT_GEOMETRY_FIELDS
from apple_pick_gym.batched_envs.sensor_realism import FtSensorConfig
from apple_pick_gym.rl.surrogate_env import JUNCTION_NAMES, SurrogateHarvestEnv

N = 8


def _env(**kw):
    kw.setdefault("ft_sensor_config", FtSensorConfig())  # noise-free unless a test wants DR
    return SurrogateHarvestEnv(num_envs=N, max_episode_steps=kw.pop("steps", 60), seed=kw.pop("seed", 0), **kw)


def _action(dp=None, drot=None, k_lin=300.0, k_ang=30.0, zeta=1.0, n=N):
    a = torch.zeros(n, 13)
    if dp is not None:
        a[:, :3] = dp
    if drot is not None:
        a[:, 3:6] = drot
    a[:, 6:9], a[:, 9:12], a[:, 12] = k_lin, k_ang, zeta
    return a


def _run(env, fn, steps):
    out = []
    for t in range(steps):
        out.append(env.step(fn(t)))
    return out


def test_obs_and_info_follow_the_real_env_contract():
    env = _env()
    obs, info = env.reset()
    assert flatten_actor_obs(obs).shape == (N, actor_obs_layout().total_width)
    assert set(info["woody_part_force"]) == set(JUNCTION_NAMES) == set(env.junction_names)
    for key in ("target_junction_wrench", "target_junction_force", "detach_index", "ft_wrist", "invalid_env", "collateral_baseline_norm"):
        assert key in info, key
    obs, r, term, trunc, info = env.step(_action())
    assert r.shape == term.shape == trunc.shape == (N, 1)
    for key in ("frozen", "success_achieved", "safety_junction", "safety_wrist", "success_streak", "terminated_edge"):
        assert info["episode"][key].shape == (N,), key
    assert set(info["reward_terms"]["raw"]) == {"progress", "pullout", "collateral"}
    assert env.target_junction_name == "spur_stem"


def test_holding_stays_at_rest_below_the_envelope():
    env = _env()
    env.reset()
    out = _run(env, lambda t: _action(), 30)
    info = out[-1][-1]
    assert float(info["detach_index"].max()) < 0.3
    assert not bool(torch.stack([o[2] for o in out]).any())


def test_pull_along_weld_loads_the_stem_and_the_chain_push_does_not():
    env = _env()
    env.reset()
    pull = _run(env, lambda t: _action(dp=env.weld * 0.003), 25)[-1][-1]
    env.reset()
    push = _run(env, lambda t: _action(dp=-env.weld * 0.003), 25)[-1][-1]
    f_pull = torch.linalg.norm(pull["target_junction_wrench"][:, :3], dim=-1)
    f_push = torch.linalg.norm(push["target_junction_wrench"][:, :3], dim=-1)
    assert bool((f_pull > f_push + 2.0).all())
    assert bool((pull["reward_terms"]["raw"]["collateral"] > push["reward_terms"]["raw"]["collateral"]).all())


def test_twist_and_pull_detaches_with_less_collateral_than_a_straight_pull():
    """The task's intended structure: torque at the spur-stem junction is cheap in collateral."""
    bounds = HarvestActionBounds(max_target_pos_offset_m=0.12, max_target_rot_offset_rad=1.2)

    def run(twist: bool):
        env = _env(action_bounds=bounds, steps=200)
        env.reset()
        peak_coll = torch.zeros(N)
        done_at = torch.full((N,), -1)
        for t in range(200):
            drot = env.weld * (0.02 if twist else 0.0)
            dp = env.weld * (0.0015 if twist else 0.003)
            _, _, term, _, info = env.step(_action(dp=dp, drot=drot, k_ang=40.0))
            live = ~info["episode"]["frozen"] | term.flatten()
            peak_coll = torch.where(live, torch.maximum(peak_coll, info["reward_terms"]["raw"]["collateral"]), peak_coll)
            done_at = torch.where(term.flatten() & (done_at < 0), torch.full_like(done_at, t), done_at)
        return info, peak_coll, done_at

    _, pull_coll, pull_done = run(twist=False)
    _, twist_coll, twist_done = run(twist=True)
    # every env detaches (terminates, not by a safety violation -- those caps are not hit here)
    assert bool((pull_done >= 0).all()) and bool((twist_done >= 0).all())
    assert float(twist_coll.mean()) < 0.7 * float(pull_coll.mean())


def test_terminated_once_freeze_and_synchronized_truncation():
    env = _env(action_bounds=HarvestActionBounds(max_target_pos_offset_m=0.1), steps=80)
    env.reset()
    out = _run(env, lambda t: _action(dp=env.weld * 0.004), 80)
    terms = torch.stack([o[2].flatten() for o in out])
    assert terms.sum(0).tolist() == [1] * N
    last = out[-1]
    assert bool(last[-1]["episode"]["frozen"].all())
    assert float(last[1].abs().max()) == 0.0
    assert bool(last[3].all()) and not bool(out[-2][3].any())


def test_plant_dr_is_per_env_and_fixed_arm_and_sensor_dr_resample_per_reset():
    env = _env(ft_sensor_config=FtSensorConfig.rl_training())
    env.reset()
    priv0, geo0 = env.privileged_fields(), env.plant_geometry()
    assert list(priv0) == [k for k, _ in _PRIVILEGED_FIELDS]
    assert list(geo0) == [k for k, _ in PLANT_GEOMETRY_FIELDS]
    for k, w in _PRIVILEGED_FIELDS:
        assert priv0[k].shape == (N, w), k
    assert len(set(priv0["stem_youngs_modulus_pa"].flatten().tolist())) == N  # per-env plant
    bias0, arm0 = env._ft_sensor._bias.clone(), priv0["arm_friction"].clone()
    env.reset()
    priv1 = env.privileged_fields()
    torch.testing.assert_close(priv1["stem_youngs_modulus_pa"], priv0["stem_youngs_modulus_pa"])
    torch.testing.assert_close(env.plant_geometry()["weld_direction"], geo0["weld_direction"])
    assert not torch.equal(priv1["arm_friction"], arm0)
    assert not torch.equal(env._ft_sensor._bias, bias0)


def test_same_seed_same_worlds_different_seed_different_worlds():
    a, b, c = _env(seed=3), _env(seed=3), _env(seed=4)
    torch.testing.assert_close(a.k_pull, b.k_pull)
    assert not torch.allclose(a.k_pull, c.k_pull)


def test_invalid_envs_are_frozen_from_reset_with_zero_reward():
    env = _env(invalid_fraction=0.25)
    _, info = env.reset()
    inv = info["invalid_env"]
    assert int(inv.sum()) == 2
    _, r, term, _, info = env.step(_action(dp=env.weld * 0.004))
    assert bool(info["episode"]["frozen"][inv].all())
    assert float(r[inv].abs().max()) == 0.0 and not bool(term[inv].any())


def test_leash_bounds_the_target_offset():
    env = _env(action_bounds=HarvestActionBounds(max_target_pos_offset_m=0.05))
    env.reset()
    _run(env, lambda t: _action(dp=torch.tensor([0.02, 0.0, 0.0]).expand(N, 3), k_lin=20.0), 30)
    # the leash holds when the target is commanded; the TCP then moves within the step
    offset = torch.linalg.norm(env._target[:, :3] - env._x, dim=-1)
    assert float(offset.max()) <= 0.05 + 1e-3


def test_is_fast_enough_for_cpu_training():
    env = SurrogateHarvestEnv(num_envs=256, max_episode_steps=100)
    env.reset()
    a = _action(n=256)
    t0 = time.perf_counter()
    for _ in range(100):
        env.step(a)
    rate = 256 * 100 / (time.perf_counter() - t0)
    assert rate > 5_000, f"{rate:.0f} env-steps/s"
    assert math.isfinite(rate)
