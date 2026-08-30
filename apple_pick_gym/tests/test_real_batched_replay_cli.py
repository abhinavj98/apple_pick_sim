"""CLI / viewer on_step unit tests for real batched replay (no GPU window)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPLAY = (
    Path(__file__).resolve().parents[2]
    / "robot_replay"
    / "example_replay_real_batched.py"
)
_VARIANCE = Path(
    "apple_pick_sim/fixtures/fruiting_system_ranges_real_world_proxy_variance.json"
)
_REAL_SRC = Path("robot_replay/s02-d00_action.parquet")


def _load_replay():
    spec = importlib.util.spec_from_file_location("example_replay_real_batched", _REPLAY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parser_accepts_viewer_gl_and_null():
    mod = _load_replay()
    p = mod._make_parser()
    gl = p.parse_args(["--dataset", "/tmp/ds", "--viewer", "gl"])
    assert gl.viewer == "gl"
    null = p.parse_args(["--dataset", "/tmp/ds", "--viewer", "null"])
    assert null.viewer == "null"


def test_parser_accepts_allow_wrench_as_twist():
    mod = _load_replay()
    p = mod._make_parser()
    args = p.parse_args(["--dataset", "/tmp/ds", "--allow-wrench-as-twist"])
    assert args.allow_wrench_as_twist is True
    default = p.parse_args(["--dataset", "/tmp/ds"])
    assert default.allow_wrench_as_twist is False


def test_allow_wrench_as_twist_rejects_pose_packed_dataset():
    """The escape hatch is legacy-6D only; 19D vic_pose datasets must fail fast."""
    mod = _load_replay()
    collection = {"action_dim": 19, "action_layout": "vic_pose_v1"}
    with pytest.raises(SystemExit, match="legacy 6D"):
        mod.check_action_semantics(
            controller_mode="vic",
            collection=collection,
            episode_meta={"action_compatible_with_vic_twist": False},
            allow_wrench_as_twist=True,
        )


def test_action_semantics_refuses_wrench_marked_dataset_under_vic():
    mod = _load_replay()
    with pytest.raises(SystemExit, match="vic_pose"):
        mod.check_action_semantics(
            controller_mode="vic",
            collection={"action_dim": 6},
            episode_meta={"action_compatible_with_vic_twist": False},
            allow_wrench_as_twist=False,
        )


def test_action_semantics_allows_legacy_6d_hatch_and_vic_pose_mode():
    mod = _load_replay()
    mod.check_action_semantics(
        controller_mode="vic",
        collection={"action_dim": 6},
        episode_meta={"action_compatible_with_vic_twist": False},
        allow_wrench_as_twist=True,
    )
    mod.check_action_semantics(
        controller_mode="vic_pose",
        collection={"action_dim": 19, "action_layout": "vic_pose_v1"},
        episode_meta={"action_compatible_with_vic_twist": False},
        allow_wrench_as_twist=False,
    )


def test_parser_settle_defaults_match_pre_grasp_settle_viewer():
    """Defaults align with example_view_pre_grasp_settle (2000 / quiet 100 / post 500)."""
    mod = _load_replay()
    p = mod._make_parser()
    args = p.parse_args(["--dataset", "/tmp/ds", "--viewer", "null"])
    assert args.settle_substeps == 2000
    assert args.settle_quiet_every == 100
    assert args.settle_gravity_ramp is False
    assert args.post_grasp_settle_substeps == 500


def test_sim_config_applies_settle_quiet_post_grasp_and_substeps():
    from apple_pick_sim.fruiting_system.params import load_ranges

    mod = _load_replay()
    ranges = load_ranges(_VARIANCE)
    cfg = mod._test_sim_config(
        num_envs=1,
        topology_seed=0,
        fruiting_base_pos=(0.117, 0.787, 0.577),
        ranges=ranges,
        settle_substeps=2000,
        settle_quiet_every=100,
        settle_gravity_ramp=False,
        post_grasp_settle_substeps=500,
    )
    assert cfg.scene.settle_substeps == 2000
    assert cfg.scene.settle_quiet_every == 100
    assert cfg.scene.settle_gravity_ramp is False
    assert cfg.scene.post_grasp_settle_substeps == 500


def test_sim_config_uses_gym_defaults_with_fixture_sim_build_not_cpu_test_minimal():
    """Batched settle needs fixture support joints + CUDA-capable device.

    ``test_minimal`` forces CPU and empty joint overrides; with catalog primary
    length that explodes settle (~20 m proxy drift) while the plant-only settle
    viewer (CUDA, single-env) stays stable on the same parquet.
    """
    from apple_pick_sim.fruiting_system.params import load_ranges, parse_sim_build

    if not _VARIANCE.is_file():
        pytest.skip(f"missing {_VARIANCE}")

    mod = _load_replay()
    ranges = load_ranges(_VARIANCE)
    sb = parse_sim_build(ranges)
    assert sb is not None
    cfg = mod._test_sim_config(
        num_envs=1,
        topology_seed=0,
        fruiting_base_pos=(0.117, 0.787, 0.577),
        ranges=ranges,
    )
    assert cfg.runtime.device is None
    assert cfg.fruiting_system.joint_angular_kp_overrides == sb.joint_angular_kp_overrides
    assert cfg.fruiting_system.joint_linear_kp_overrides == sb.joint_linear_kp_overrides
    assert cfg.fruiting_system.joint_damping_ratio == pytest.approx(sb.joint_damping_ratio)
    assert cfg.robot.fix_to_apple is True
    assert cfg.controller.mode == "vic_pose"
    assert cfg.controller.action_dim == 19


def test_sim_config_wires_open_loop_bootstrap_joint_q_from_episode():
    from apple_pick_sim.fruiting_system.params import load_ranges

    if not _VARIANCE.is_file():
        pytest.skip(f"missing {_VARIANCE}")

    mod = _load_replay()
    ranges = load_ranges(_VARIANCE)
    q = (0.1, 0.2, 0.3, -1.0, 0.0, 1.5, -0.5)
    cfg = mod._test_sim_config(
        num_envs=1,
        topology_seed=0,
        fruiting_base_pos=(0.117, 0.787, 0.577),
        ranges=ranges,
        bootstrap_joint_q=q,
    )
    assert cfg.robot.bootstrap_joint_q == q


def test_bootstrap_joint_q_from_episode_metadata():
    mod = _load_replay()
    meta = {"initial_robot_joint_q": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    q = mod.bootstrap_joint_q_from_episode_metadata(meta)
    assert q == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    with pytest.raises(ValueError, match="initial_robot_joint_q"):
        mod.bootstrap_joint_q_from_episode_metadata({})


def test_control_hz_from_episode_metadata_prefers_episode_then_collection():
    mod = _load_replay()
    assert mod.control_hz_from_episode_metadata({"control_hz": 15.0}) == pytest.approx(15.0)
    assert mod.control_hz_from_episode_metadata(
        {}, collection={"control_hz": 12.5}
    ) == pytest.approx(12.5)
    assert mod.control_hz_from_episode_metadata(
        {"control_hz": 15.0}, collection={"control_hz": 12.5}
    ) == pytest.approx(15.0)
    with pytest.raises(ValueError, match="control_hz"):
        mod.control_hz_from_episode_metadata({})
    with pytest.raises(ValueError, match="control_hz"):
        mod.control_hz_from_episode_metadata({"control_hz": 0.0})


def test_sim_config_applies_recorded_control_hz():
    from apple_pick_sim.fruiting_system.params import load_ranges

    if not _VARIANCE.is_file():
        pytest.skip(f"missing {_VARIANCE}")

    mod = _load_replay()
    ranges = load_ranges(_VARIANCE)
    cfg = mod._test_sim_config(
        num_envs=1,
        topology_seed=0,
        fruiting_base_pos=(0.117, 0.787, 0.577),
        ranges=ranges,
        control_hz=15.0,
    )
    assert cfg.runtime.control_hz == pytest.approx(15.0)


def test_parser_print_woody_forces_default_off():
    mod = _load_replay()
    p = mod._make_parser()
    args = p.parse_args(["--dataset", "/tmp/ds"])
    assert args.print_woody_forces == 0


def test_format_woody_force_lines_lists_force_and_norm_per_junction():
    mod = _load_replay()
    lines = mod.format_woody_force_lines(
        {
            "primary_spur": [1.0, 0.0, 0.0],
            "stem_apple": [0.0, 3.0, 4.0],
        },
        frame_idx=7,
    )
    assert lines[0] == "woody_forces frame=7"
    assert "primary_spur F=[1.000, 0.000, 0.000] |F|=1.000" in lines[1]
    assert "stem_apple F=[0.000, 3.000, 4.000] |F|=5.000" in lines[2]


def test_woody_forces_from_last_obs_env0():
    mod = _load_replay()
    import torch

    last_obs = {
        "woody_part_info": {
            "spur_stem": {
                "anchor_force": torch.tensor(
                    [[1.0, 2.0, 3.0, 9.0, 9.0, 9.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                )
            }
        }
    }
    forces = mod.woody_forces_from_last_obs(last_obs, ["spur_stem"], env_idx=0)
    assert forces == {"spur_stem": (1.0, 2.0, 3.0)}


def test_parser_accepts_record_video():
    mod = _load_replay()
    p = mod._make_parser()
    args = p.parse_args(
        ["--dataset", "/tmp/ds", "--record-video", "/tmp/replay.mp4"]
    )
    assert args.record_video == Path("/tmp/replay.mp4")
    default = p.parse_args(["--dataset", "/tmp/ds"])
    assert default.record_video is None


def test_require_gl_frame_capture_rejects_null_viewer():
    mod = _load_replay()
    with pytest.raises(SystemExit, match="--viewer gl"):
        mod.require_gl_frame_capture(SimpleNamespace())


def test_make_replay_on_step_renders_and_stops_at_max_frames():
    mod = _load_replay()
    calls: list = []

    class Viewer:
        def set_model(self, model):
            calls.append(("set_model", model))

        def hide_loading_splash(self):
            calls.append("splash")

        def begin_frame(self, t):
            calls.append(("begin", t))

        def log_state(self, state):
            calls.append(("log", state))

        def end_frame(self):
            calls.append("end")

        def is_running(self):
            return True

    cable = SimpleNamespace(model="MODEL", state_0="STATE")
    scene = SimpleNamespace(cable=cable)
    sim = SimpleNamespace(
        scene=scene, config=SimpleNamespace(runtime=SimpleNamespace(control_hz=10.0))
    )
    env = SimpleNamespace(_sim=sim, num_envs=1)

    on_step = mod.make_replay_on_step(Viewer(), max_frames=2, control_hz_fallback=30.0)
    assert on_step(frame_idx=0, env=env) is True
    assert on_step(frame_idx=1, env=env) is False
    assert ("set_model", "MODEL") in calls
    assert ("begin", 0.0) in calls
    assert ("log", "STATE") in calls
    assert "end" in calls


def test_make_replay_on_step_captures_video_after_end_frame():
    mod = _load_replay()
    order: list[str] = []

    class Viewer:
        def set_model(self, model):
            pass

        def hide_loading_splash(self):
            pass

        def begin_frame(self, t):
            order.append("begin")

        def log_state(self, state):
            order.append("log")

        def end_frame(self):
            order.append("end")

        def get_frame(self):
            order.append("get_frame")
            return object()

        def is_running(self):
            return True

    class Recorder:
        def __init__(self):
            self.fps = None
            self.captures = 0

        def set_fps(self, fps: float) -> None:
            self.fps = float(fps)

        def capture(self, viewer) -> None:
            order.append("capture")
            viewer.get_frame()
            self.captures += 1

    cable = SimpleNamespace(model="MODEL", state_0="STATE")
    scene = SimpleNamespace(cable=cable)
    sim = SimpleNamespace(
        scene=scene, config=SimpleNamespace(runtime=SimpleNamespace(control_hz=10.0))
    )
    env = SimpleNamespace(_sim=sim, num_envs=1)
    recorder = Recorder()
    on_step = mod.make_replay_on_step(
        Viewer(), max_frames=0, recorder=recorder
    )
    assert on_step(frame_idx=0, env=env) is True
    assert recorder.captures == 1
    assert recorder.fps == pytest.approx(10.0)
    assert order == ["begin", "log", "end", "capture", "get_frame"]


def test_gl_camera_from_camera_to_base_looks_along_plus_z():
    mod = _load_replay()
    # Optical +Z = (1, 0, 0) → look +X in Z-up → pitch=0, yaw=0.
    T = [
        [0.0, 0.0, 1.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [-1.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pos, pitch, yaw = mod.gl_camera_from_camera_to_base(T)
    assert pos == pytest.approx((0.1, 0.2, 0.3))
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)


def test_make_replay_on_step_sets_camera_from_extrinsic_once():
    mod = _load_replay()
    calls: list = []

    class Viewer:
        def set_model(self, model):
            calls.append(("set_model", model))

        def hide_loading_splash(self):
            pass

        def set_camera(self, pos, pitch, yaw):
            calls.append(("set_camera", (float(pos[0]), float(pos[1]), float(pos[2])), float(pitch), float(yaw)))

        def begin_frame(self, t):
            pass

        def log_state(self, state):
            pass

        def end_frame(self):
            pass

        def is_running(self):
            return True

    cable = SimpleNamespace(model="MODEL", state_0="STATE")
    scene = SimpleNamespace(cable=cable)
    sim = SimpleNamespace(
        scene=scene, config=SimpleNamespace(runtime=SimpleNamespace(control_hz=10.0))
    )
    env = SimpleNamespace(_sim=sim, num_envs=1)
    # Identity rotation: optical +Z = world +Z → pitch=90.
    T = [
        [1.0, 0.0, 0.0, -0.3],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.4],
        [0.0, 0.0, 0.0, 1.0],
    ]
    on_step = mod.make_replay_on_step(
        Viewer(), max_frames=0, camera_to_base_4x4=T
    )
    assert on_step(frame_idx=0, env=env) is True
    assert on_step(frame_idx=1, env=env) is True
    cam_calls = [c for c in calls if c[0] == "set_camera"]
    assert len(cam_calls) == 1
    assert cam_calls[0][1] == pytest.approx((-0.3, 0.5, 0.4))
    assert cam_calls[0][2] == pytest.approx(90.0, abs=1e-6)


def test_make_replay_on_step_prints_woody_forces_every_n_frames(capsys):
    mod = _load_replay()
    import torch

    class Viewer:
        def is_running(self):
            return True

    last_obs = {
        "woody_part_info": {
            "primary_spur": {
                "anchor_force": torch.tensor(
                    [[3.0, 0.0, 4.0, 0.0, 0.0, 0.0]], dtype=torch.float32
                )
            }
        }
    }
    env = SimpleNamespace(
        _sim=None,
        num_envs=1,
        junction_names=["primary_spur"],
        _last_obs=last_obs,
    )
    on_step = mod.make_replay_on_step(
        Viewer(), max_frames=0, print_woody_forces_every=2
    )
    assert on_step(frame_idx=0, env=env) is True
    assert on_step(frame_idx=1, env=env) is True
    assert on_step(frame_idx=2, env=env) is True
    out = capsys.readouterr().out
    assert out.count("woody_forces frame=") == 2
    assert "primary_spur F=[3.000, 0.000, 4.000] |F|=5.000" in out
    assert "frame=1" not in out


def test_replay_rebuilds_from_episode_metadata_not_fixture_clamps(tmp_path, monkeypatch):
    """Convert→replay must use episode fruiting_base_pos + catalog primary length.

    Fixture C6 overrides put the plant at the wrong place relative to the settle
    viewer (example_view_pre_grasp_settle) which rebuilds from native params.
    """
    if not _REAL_SRC.is_file():
        pytest.skip(f"missing {_REAL_SRC}")
    if not _VARIANCE.is_file():
        pytest.skip(f"missing {_VARIANCE}")

    from apple_pick_gym.batched_envs.batched_sysid_mmd_grid import (
        true_params_for_structure,
    )
    from apple_pick_sim.fruiting_system.params import (
        load_ranges,
        parse_fixture_args,
    )
    from apple_pick_sim.system_id import BatchedSysIdDataset
    from apple_pick_sim.system_id.real_to_batched_sysid import (
        export_real_episode_to_batched_dataset,
        range_midpoint,
    )

    out = tmp_path / "real_batched"
    export_real_episode_to_batched_dataset(
        _REAL_SRC,
        fixture_path=_VARIANCE,
        output_dir=out,
        overwrite=True,
    )
    dataset = BatchedSysIdDataset(out)
    meta = dataset.load_episode_metadata(0, 0)
    episode_base = tuple(float(x) for x in meta["fruiting_base_pos"])
    ranges = load_ranges(_VARIANCE)
    fixture_base = parse_fixture_args(ranges).fruiting_base_pos
    assert fixture_base is not None
    assert episode_base != tuple(float(x) for x in fixture_base)

    mod = _load_replay()
    resolved = mod.fruiting_base_pos_from_episode_metadata(meta)
    assert resolved == episode_base

    oracle = true_params_for_structure(dataset, 0)
    assert oracle.primary is not None
    catalog_len = float(oracle.primary.length)
    clamp_len = float(range_midpoint(ranges["primary"]["length"]))
    assert catalog_len != pytest.approx(clamp_len)

    captured: dict = {}

    def _fake_env(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ApplePickBatchedSysIdEnv")

    monkeypatch.setattr(
        "apple_pick_gym.batched_envs.real_batched_replay_build.ApplePickBatchedSysIdEnv",
        _fake_env,
    )
    build = mod._build_env_fn(
        ranges_path=_VARIANCE,
        ranges=ranges,
        topology_seed=0,
        fruiting_base_pos=resolved,
        episode_meta=meta,
    )
    build(
        num_envs=1,
        per_env_params=[oracle],
        max_episode_steps=8,
    )
    built = captured["per_env_params"][0]
    assert float(built.primary.length) == pytest.approx(catalog_len)
    assert captured["sim_config"].scene.fruiting_base_pos == episode_base
