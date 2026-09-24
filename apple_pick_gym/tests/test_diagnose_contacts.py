"""Contact-force bookkeeping for ``rl/diagnose_contacts``: which body pairs press on which."""

from __future__ import annotations

import numpy as np

from apple_pick_gym.rl.diagnose_contacts import FRUIT, OTHER, PROXY, classify_bodies, per_env_contact_forces


def test_classify_bodies_by_label():
    labels = ["primary_edge_body_0", "spur_edge_body_2", "stem_edge_body_0", "apple", "gripper_proxy"]
    np.testing.assert_array_equal(classify_bodies(labels), [OTHER, OTHER, FRUIT, FRUIT, PROXY])


def test_per_env_net_contact_force_by_pair_class():
    cat = np.array([OTHER, FRUIT, PROXY, OTHER, FRUIT, PROXY])
    world = np.array([0, 0, 0, 1, 1, 1])
    # contacts: (body0, body1, force on body1)
    body0 = np.array([0, 1, 1, 3, 0, -1])
    body1 = np.array([1, 0, 2, 4, 0, -1])
    force = np.array([[3.0, 0, 0], [0, 4.0, 0], [0, 0, 2.0], [1.0, 0, 0], [9.0, 0, 0], [7.0, 0, 0]])
    count = 5  # the 6th slot is past the active count
    out = per_env_contact_forces(body0, body1, force, count, cat, world, num_envs=2)
    # env 0: fruit-woody contacts (3,0,0) and (0,4,0) -> net |(3,4,0)| = 5 ; fruit-proxy |(0,0,2)| = 2
    np.testing.assert_allclose(out["fruit_woody_n"], [5.0, 1.0])
    np.testing.assert_allclose(out["fruit_proxy_n"], [2.0, 0.0])
    np.testing.assert_array_equal(out["fruit_woody_count"], [2, 1])
    # contact 4 is woody-woody (0,0): not counted
