"""Named junction force access on ApplePickBaseEnv subclasses."""

from __future__ import annotations

import numpy as np
import pytest


def _maybe_import_gymnasium():
    try:
        import gymnasium as gym  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed (expected to be provided by newton[dev])",
)


@gymnasium_available
def test_junction_names_before_reset():
    from apple_pick_gym.envs import ApplePickCoupledEnv

    env = ApplePickCoupledEnv()
    with pytest.raises(RuntimeError, match="reset"):
        _ = env.junction_names


@gymnasium_available
def test_junction_names_after_reset():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    _, info = env.reset(seed=0)
    unwrapped = env.unwrapped

    names = unwrapped.junction_names
    assert isinstance(names, list)
    assert len(names) == int(info["n_woody_parts"])
    assert "primary_secondary" in names
    assert "secondary_spur" in names
    assert "spur_stem" in names
    assert "stem_apple" in names
    assert names == list(info["fruiting_link_forces"].keys())
    env.close()


@gymnasium_available
def test_junction_forces_dict_shape():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    obs, _ = env.reset(seed=0)
    unwrapped = env.unwrapped

    jf = unwrapped.junction_forces_dict(obs)
    assert isinstance(jf, dict)
    assert set(jf.keys()) == set(unwrapped.junction_names)
    for name, wrench in jf.items():
        assert isinstance(name, str)
        assert wrench.shape == (6,)
        assert wrench.dtype == np.float32
    env.close()


@gymnasium_available
def test_junction_forces_dict_matches_flat_obs():
    import gymnasium as gym
    from apple_pick_sim.tests.conftest import fr3_assets_available

    if not fr3_assets_available():
        pytest.skip("Requires bundled assets/fr3 and usd-core")

    env = gym.make("ApplePickCoupled-v0", render_mode=None, max_episode_steps=2)
    obs, _ = env.reset(seed=0)
    unwrapped = env.unwrapped

    jf = unwrapped.junction_forces_dict(obs)
    flat = np.asarray(obs["woody_part_force"], dtype=np.float32)
    names = unwrapped.junction_names

    for i, name in enumerate(names):
        np.testing.assert_array_equal(jf[name], flat[i * 6 : (i + 1) * 6])

    assert "stem_apple" in jf
    np.testing.assert_array_equal(jf["stem_apple"], flat[-6:])
    env.close()
