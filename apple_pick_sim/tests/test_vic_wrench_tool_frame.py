"""[D17] Tool-frame anisotropic VIC: per-axis stiffness / damping along the TCP's own axes.

``compute_vic_spatial_wrench_aniso`` scales the world-frame pose error component-wise, so "stiff
along the grip axis, soft sideways" is not expressible for a tilted grasp. The tool-frame variant
applies ``K_world = R diag(k) R^T`` (and the same for D), with ``R`` the TCP orientation. With an
identity TCP rotation the two must agree exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from apple_pick_sim.coupled_fruiting.vic_wrench import (
    compute_vic_spatial_wrench_aniso,
    compute_vic_spatial_wrench_aniso_tool,
)


@wp.kernel
def _wrench_kernel(
    tcp: wp.array(dtype=wp.transform),
    qd: wp.array(dtype=wp.spatial_vector),
    target: wp.array(dtype=wp.transform),
    kp_lin: wp.array(dtype=wp.vec3),
    kp_ang: wp.array(dtype=wp.vec3),
    kd_lin: wp.array(dtype=wp.vec3),
    kd_ang: wp.array(dtype=wp.vec3),
    tool_frame: int,
    out: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    if tool_frame != 0:
        out[i] = compute_vic_spatial_wrench_aniso_tool(tcp[i], qd[i], target[i], kp_lin[i], kp_ang[i], kd_lin[i], kd_ang[i])
    else:
        out[i] = compute_vic_spatial_wrench_aniso(tcp[i], qd[i], target[i], kp_lin[i], kp_ang[i], kd_lin[i], kd_ang[i])


def _quat_from_rotvec(r: np.ndarray) -> np.ndarray:
    """xyzw quaternion of a rotation vector."""
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = r / a
    return np.concatenate([ax * np.sin(a / 2), [np.cos(a / 2)]])


def _rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _run(tcp_q, tcp_p, tgt_p, v, w, kp_lin, kp_ang, kd_lin, kd_ang, tool_frame):
    n = len(tcp_q)
    dev = "cpu"
    arr = lambda a, dt: wp.array(np.asarray(a, dtype=np.float32), dtype=dt, device=dev)
    tcp = wp.array([wp.transform(wp.vec3(*tcp_p[i]), wp.quat(*tcp_q[i])) for i in range(n)], dtype=wp.transform, device=dev)
    # target orientation = TCP orientation: the angular error is driven only by the velocity term
    tgt = wp.array([wp.transform(wp.vec3(*tgt_p[i]), wp.quat(*tcp_q[i])) for i in range(n)], dtype=wp.transform, device=dev)
    qd = wp.array([wp.spatial_vector(*v[i], *w[i]) for i in range(n)], dtype=wp.spatial_vector, device=dev)
    out = wp.zeros(n, dtype=wp.spatial_vector, device=dev)
    wp.launch(
        _wrench_kernel,
        dim=n,
        inputs=[tcp, qd, tgt, arr(kp_lin, wp.vec3), arr(kp_ang, wp.vec3), arr(kd_lin, wp.vec3), arr(kd_ang, wp.vec3), int(tool_frame), out],
        device=dev,
    )
    return out.numpy()


def test_tool_frame_equals_world_frame_for_an_identity_tcp_rotation():
    rng = np.random.default_rng(0)
    n = 6
    q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    p = rng.normal(size=(n, 3))
    args = (q, p, p + rng.normal(scale=0.05, size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, 3)),
            rng.uniform(20, 500, (n, 3)), rng.uniform(2, 40, (n, 3)), rng.uniform(5, 40, (n, 3)), rng.uniform(1, 9, (n, 3)))
    np.testing.assert_allclose(_run(*args, tool_frame=1), _run(*args, tool_frame=0), rtol=1e-5, atol=1e-5)


def test_stiffness_follows_the_tcp_axes_for_a_quarter_turn():
    # TCP rotated +90 deg about world z: tool x = world y. Stiff only along tool x.
    q = np.array([_quat_from_rotvec(np.array([0.0, 0.0, np.pi / 2]))] * 2)
    p = np.zeros((2, 3))
    tgt = np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.0]])  # world y (= tool x), world x (= tool -y)
    z = np.zeros((2, 3))
    k = np.array([[1000.0, 0.0, 0.0]] * 2)
    out = _run(q, p, tgt, z, z, k, z, z, z, tool_frame=1)
    np.testing.assert_allclose(out[0, :3], [0.0, 100.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(out[1, :3], [0.0, 0.0, 0.0], atol=1e-3)


def test_tool_frame_wrench_is_r_diag_k_rt_for_a_random_rotation():
    rng = np.random.default_rng(1)
    n = 8
    q = np.array([_quat_from_rotvec(rng.normal(size=3)) for _ in range(n)])
    p = rng.normal(size=(n, 3))
    e = rng.normal(scale=0.05, size=(n, 3))
    v, w = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    kpl, kpa = rng.uniform(20, 500, (n, 3)), rng.uniform(2, 40, (n, 3))
    kdl, kda = rng.uniform(5, 40, (n, 3)), rng.uniform(1, 9, (n, 3))
    out = _run(q, p, p + e, v, w, kpl, kpa, kdl, kda, tool_frame=1)
    for i in range(n):
        R = _rotmat(q[i])
        f = R @ np.diag(kpl[i]) @ R.T @ e[i] - R @ np.diag(kdl[i]) @ R.T @ v[i]
        t = -(R @ np.diag(kda[i]) @ R.T @ w[i])  # zero orientation error
        np.testing.assert_allclose(out[i, :3], f, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[i, 3:], t, rtol=1e-4, atol=1e-4)


def test_controller_config_defaults_to_world_frame_gains():
    from apple_pick_sim.coupled_fruiting.batched_heterogeneous_config import ControllerConfig

    assert ControllerConfig().vic_gain_frame == "world"  # sys-ID replay matches the rig's base-frame gains
    with pytest.raises(ValueError):
        ControllerConfig(vic_gain_frame="body")
