"""Keyboard teleop for the bundled FR3 + EE using TCP velocity + IK.

Focus the Newton viewer window, then drive the tool center point (``tcp``) with:

- **I / K** — world ±X
- **J / L** — world ±Y
- **R / F** — world ±Z (not W/S — those move the camera)
- **U / O** — rotate about world Z
- **T / G** — rotate about world Y
- **Z / X** — rotate about world X

Run from the repository root::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fr3_keyboard.py

Headless smoke::

    PYTHONPATH=$(pwd) uv run --directory newton python ../apple_pick_sim/examples/example_fr3_keyboard.py --viewer null --num-frames 120
"""

from __future__ import annotations

import os
import sys

import newton
import newton.examples
import warp as wp

from apple_pick_sim.robot import fr3_robot


class ExampleFr3Keyboard:
    def __init__(self, viewer, args=None) -> None:
        del args
        if not fr3_robot.fr3_assets_available():
            raise FileNotFoundError(
                "FR3 assets missing; see assets/fr3/README.md and docs/fr3-usd-import-implementation.md"
            )

        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.model, self.tcp_idx, _solver = fr3_robot.build_fr3_robot_model_from_usd()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.ee_ctrl = fr3_robot.Fr3EEVelocityController(
            self.model,
            self.tcp_idx,
            linear_speed=0.2,
            angular_speed=1.0,
            ik_iterations=24,
        )
        self.ee_ctrl.sync_target_from_state(self.state)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.2, -1.2, 0.9), pitch=-25.0, yaw=45.0)

        if not hasattr(self.viewer, "is_key_down"):
            print(
                "Warning: viewer has no keyboard API (use --viewer gl). "
                "I/K/J/L/R/F and U/O/T/G/;/' will be ignored.",
                file=sys.stderr,
            )

    def step(self) -> None:
        # Host-side input + target integration (must not run inside CUDA graph capture).
        self.ee_ctrl.advance_target(self.frame_dt, viewer=self.viewer)
        self.ee_ctrl.solve_ik()
        self.ee_ctrl.apply_to_model_and_state(self.state)
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        bq = self.state.body_q.numpy().reshape(-1, 7)[self.tcp_idx]
        assert float(bq[2]) > -0.5, f"TCP fell unexpectedly: z={bq[2]}"


if __name__ == "__main__":
    if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            sys.argv.extend(["--viewer", "null", "--num-frames", "120"])
            print("No DISPLAY/WAYLAND_DISPLAY: using --viewer null --num-frames 120.")

    viewer, args = newton.examples.init()
    example = ExampleFr3Keyboard(viewer, args)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    print("FR3 keyboard teleop — focus the viewer window (I/K/J/L/R/F, U/O/T/G, Z/X).")
    for _ in range(10):
        example.step()

    while viewer.is_running():
        example.step()
        example.render()

    example.test_final()
