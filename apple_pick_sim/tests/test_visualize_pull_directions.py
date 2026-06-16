"""Tests for Fibonacci pull-direction visualization (gym env code path)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apple_pick_sim.system_id.fibonacci_hemisphere import sample_robot_facing_pull_directions
from apple_pick_sim.system_id.pull_direction_viz import collect_pull_direction_geometry
from apple_pick_sim.tests.conftest import COUPLED_ROBOT_BASE_POS, fr3_assets_available


def _maybe_import_gymnasium() -> bool:
    try:
        import gymnasium  # noqa: F401

        return True
    except Exception:
        return False


gymnasium_available = pytest.mark.skipif(
    not _maybe_import_gymnasium(),
    reason="gymnasium not installed (expected to be provided by newton[dev])",
)


def _make_sysid_env(*, fix_to_apple: bool = True, n_weld_hemisphere_samples: int = 10):
    from apple_pick_gym.envs import ApplePickSysIdEnv

    return ApplePickSysIdEnv(
        render_mode=None,
        max_episode_steps=2,
        fix_to_apple=fix_to_apple,
        fix_to_apple_warmup_substeps=0,
        n_weld_hemisphere_samples=n_weld_hemisphere_samples,
        mujoco_solver_kwargs={"disable_contacts": True},
    )


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_directions_forward_hemisphere():
    env = _make_sysid_env()
    try:
        obs, _ = env.reset(seed=0)
        geom = collect_pull_direction_geometry(env, obs, n_directions=10)
        dots = geom.pull_directions @ geom.robot_dir
        cos_min = float(np.cos(geom.max_pull_polar_angle_rad))
        assert np.all(dots >= cos_min - 1e-9)
        assert geom.min_pull_robot_dot >= cos_min - 1e-9
    finally:
        env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_weld_is_robot_facing():
    env = _make_sysid_env()
    try:
        obs, _ = env.reset(seed=0)
        geom = collect_pull_direction_geometry(env, obs, n_directions=10)
        assert geom.weld_dir is not None
        assert geom.weld_robot_dot is not None
        assert geom.weld_robot_dot >= 0.0
        assert float(np.dot(geom.weld_dir, geom.robot_dir)) >= 0.0
    finally:
        env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_proxy_on_robot_facing_hemisphere():
    env = _make_sysid_env()
    try:
        obs, _ = env.reset(seed=0)
        geom = collect_pull_direction_geometry(env, obs, n_directions=10)
        assert geom.proxy_robot_dot is not None
        assert geom.proxy_robot_dot >= 0.0
    finally:
        env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_weld_varies_across_resets():
    env = _make_sysid_env(n_weld_hemisphere_samples=8)
    try:
        welds: list[np.ndarray] = []
        for _ in range(8):
            _, info = env.reset()
            weld = np.asarray(info["weld_direction"], dtype=np.float64).reshape(3)
            welds.append(weld)
        unique = {tuple(np.round(w, 6)) for w in welds}
        assert len(unique) > 1
    finally:
        env.close()


def test_viz_cli_default_90_degree_hemisphere():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "apple_pick_gym" / "examples" / "visualize_pull_directions.py"
    spec = importlib.util.spec_from_file_location("visualize_pull_directions", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._make_parser().parse_args([])
    assert args.max_polar_angle_deg == 90.0


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_pull_directions_match_collection_helper():
    env = _make_sysid_env()
    try:
        obs, _ = env.reset(seed=0)
        geom = collect_pull_direction_geometry(env, obs, n_directions=10)
        apple_pos = np.asarray(obs["apple_pos"], dtype=np.float64)
        robot_vec = np.asarray(COUPLED_ROBOT_BASE_POS, dtype=np.float64) - apple_pos
        expected = sample_robot_facing_pull_directions(
            10,
            geom.physical_stem_dir,
            robot_vec,
            max_polar_angle=geom.max_pull_polar_angle_rad,
        )
        np.testing.assert_allclose(geom.pull_directions, expected, rtol=1e-9, atol=1e-9)
    finally:
        env.close()


@gymnasium_available
@pytest.mark.skipif(not fr3_assets_available(), reason="Requires bundled assets/fr3 and usd-core")
def test_output_png_created(tmp_path: Path):
    out = tmp_path / "pull_dirs.png"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "apple_pick_gym" / "examples" / "visualize_pull_directions.py"
    cmd = [
        sys.executable,
        str(script),
        "--seed",
        "0",
        "--n-directions",
        "6",
        "--n-resets",
        "3",
        "--output",
        str(out),
        "--fix-to-apple-warmup-substeps",
        "0",
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert out.exists()
    assert out.stat().st_size > 0
