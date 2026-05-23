"""FR3 + custom end-effector import for M1 ``robot_model`` (``SolverMuJoCo``).

Loads the **project Isaac scene** (``assets/testfr3_resolved.usda``) — the authoring export
paired with ``assets/testfr3.usd`` — which references the bundled arm under
``assets/fr3/omniverse_fr3/``. Newton has no ``import_articulation``; use ``ModelBuilder.add_usd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from newton.usd import SchemaResolverMjc, SchemaResolverNewton

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical sim scene (local ``fr3`` ref + EE / tcp prims authored like ``testfr3.usd``).
TESTFR3_SCENE_USD = _REPO_ROOT / "assets" / "testfr3_resolved.usda"

# Bundled Omniverse FR3 subtree (referenced from ``testfr3_resolved.usda`` under ``./fr3/…``).
_OMNI_FR3_ROOT = _REPO_ROOT / "assets" / "fr3" / "omniverse_fr3"
OMNIVERSE_FR3_USD = _OMNI_FR3_ROOT / "fr3.usd"
OMNIVERSE_FR3_SCHEMA = _OMNI_FR3_ROOT / "configuration" / "fr3_robot_schema.usd"

# Default gripper-proxy matching (collision cylinder in scene is separate; proxy box for VBD coupling).
EE_MASS_KG = 1.5
EE_BOX_HALF_EXTENTS = (0.05, 0.05, 0.05)


def fr3_assets_available() -> bool:
    """True when the composed scene file and Omniverse subtree exist."""
    return (
        TESTFR3_SCENE_USD.is_file()
        and OMNIVERSE_FR3_USD.is_file()
        and OMNIVERSE_FR3_SCHEMA.is_file()
    )


def resolve_tcp_body_index(model: newton.Model) -> int:
    """Return the unique body index for the tcp link. Heuristic: find ``ee``, then see if a direct child ``tcp`` exists."""
    labels = list(model.body_label)

    # First, try to find "ee"
    ee_hits = [
        i
        for i, lbl in enumerate(labels)
        if lbl.endswith("/ee") or lbl.split("/")[-1] == "ee"
    ]
    if len(ee_hits) != 1:
        raise ValueError(f"ambiguous or missing ee in body_label ({len(ee_hits)} hits): {labels}")
    ee_index = ee_hits[0]
    ee_label = labels[ee_index]

    # Now, look for a "tcp" child: "<...>/ee/tcp" or like that
    tcp_hits = [
        i
        for i, lbl in enumerate(labels)
        if lbl.endswith("/tcp") or lbl.split("/")[-1] == "tcp"
        if lbl.startswith(ee_label)
    ]
    if len(tcp_hits) == 1:
        return tcp_hits[0]
    elif len(tcp_hits) == 0:
        # fall back to returning the ee index itself (if no tcp)
        return ee_index
    else:
        raise ValueError(f"ambiguous tcp underneath ee in body_label ({len(tcp_hits)} hits): {labels}")

def sync_robot_gravity_to_mujoco(robot_model: newton.Model, mj_solver: SolverMuJoCo) -> None:
    """Zero Model A gravity and push it into the embedded MuJoCo ``mj_model``.

    Cable VBD (Model B) keeps its own ``cable.model.gravity`` and ``CoupledFruitingScene.gravity_vec``.
    After ``set_gravity``, ``notify_model_changed`` is required so ``mj_model.opt.gravity`` updates.
    """
    robot_model.set_gravity((0.0, 0.0, 0.0))
    mj_solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)


def resolve_ee_body_index(model: newton.Model) -> int:
    """Return the unique body index for the custom end-effector link ``ee``."""
    labels = list(model.body_label)
    for needle in ("/ee", "ee"):
        hits = [
            i
            for i, lbl in enumerate(labels)
            if lbl.endswith(needle) or lbl.split("/")[-1] == needle
        ]
        if len(hits) == 1:
            return hits[0]
    raise ValueError(f"ambiguous or missing ee in body_label: {labels}")


def build_fr3_robot_model_from_usd(
    *,
    device: str = "cpu",
    usd_path: Path | str | None = None,
    root_xform: wp.transform | None = None,
    add_ground_plane: bool = True,
    mujoco_solver_kwargs: dict[str, Any] | None = None,
) -> tuple[newton.Model, int, SolverMuJoCo]:
    """Build FR3 + Isaac-exported EE/tcp from USD for ``SolverMuJoCo``.

    Default USD is [`assets/testfr3_resolved.usda`] (paired with [`assets/testfr3.usd`] in Omni).

    **Fixed joints** from Isaac (including EE welds) are preserved --- pass
    ``collapse_fixed_joints=False`` implicitly so ``ee`` / ``tcp`` rigid bodies remain.

    To import a **patched binary** [`assets/testfr3.usd`] instead, rewrite its ``fr3``
    payload reference to `./fr3/omniverse_fr3/fr3.usd` so it resolves offline (see
    [`assets/fr3/README.md`]), fix EE joints like ``resolved`` if Newton reports a joint
    cycle, then pass ``usd_path=``.

    Returns ``(model, tcp_body_index, mj_solver)``.
    """
    if not fr3_assets_available():
        raise FileNotFoundError(
            f"Bundled FR3 scene or Omniverse subtree missing; see {TESTFR3_SCENE_USD} and assets/fr3/README.md"
        )

    path = Path(usd_path) if usd_path is not None else TESTFR3_SCENE_USD
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
    SolverMuJoCo.register_custom_attributes(builder)

    usd_kw: dict[str, Any] = {
        "floating": False,
        # EE / tcp are separate rigid bodies welded with FIXED joints --- must stay explicit.
        "collapse_fixed_joints": False,
        "enable_self_collisions": False,
        "schema_resolvers": [SchemaResolverMjc(), SchemaResolverNewton()],
    }
    if root_xform is not None:
        usd_kw["xform"] = root_xform
        usd_kw["override_root_xform"] = True
    builder.add_usd(str(path), **usd_kw)

    if add_ground_plane:
        builder.add_ground_plane()

    model = builder.finalize(device=device)
    # Model A: zero gravity for teleop/PD hold (cable VBD keeps -9.81 on its own model).
    model.set_gravity((0.0, 0.0, 0.0))

    tcp_idx = resolve_tcp_body_index(model)

    mj_kw: dict[str, Any] = {
        "solver": "newton",
        "integrator": "implicitfast",
        "cone": "elliptic",
        "iterations": 20,
        "ls_iterations": 10,
        "ls_parallel": True,
        "impratio": 1000.0,
        "use_mujoco_contacts": False,
        "use_mujoco_cpu": True,
        "disable_contacts": False,
    }
    if mujoco_solver_kwargs:
        mj_kw.update(mujoco_solver_kwargs)

    solver = SolverMuJoCo(
        model,
        njmax=200,
        nconmax=200,
        **mj_kw,
    )
    return model, tcp_idx, solver


def placement_xform_for_proxy(
    proxy_body_q7: Any,
    *,
    vertical_reach_m: float = 0.85,
) -> wp.transform:
    """World transform to park the FR3 root so ``tcp`` can reach a high gripper proxy."""
    import numpy as np

    p = np.asarray(proxy_body_q7, dtype=np.float64).reshape(7)
    base_z = max(0.0, float(p[2]) - vertical_reach_m)
    return wp.transform(wp.vec3(float(p[0]), float(p[1]), base_z), wp.quat_identity())


def bootstrap_tcp_ik_from_proxy(
    cable_scene: Any,
    robot_model: newton.Model,
    tcp_body_index: int,
    robot_state_0: Any,
    *,
    ik_iterations: int = 48,
) -> None:
    """Place the arm so ``tcp`` matches the cable gripper proxy pose (position + orientation)."""
    import newton.ik as ik

    proxy_body = cable_scene.gripper_proxy_body
    bq = cable_scene.state_0.body_q.numpy().reshape(-1, 7)[proxy_body]
    target_tf = wp.transform(
        wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
        wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
    )
    target_pos = wp.transform_get_translation(target_tf)
    target_rot = wp.transform_get_rotation(target_tf)
    dev = robot_model.device

    pos_obj = ik.IKObjectivePosition(
        link_index=tcp_body_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([target_pos], dtype=wp.vec3, device=dev),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=tcp_body_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array(
            [wp.vec4(target_rot[0], target_rot[1], target_rot[2], target_rot[3])],
            dtype=wp.vec4,
            device=dev,
        ),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=robot_model.joint_limit_lower,
        joint_limit_upper=robot_model.joint_limit_upper,
        weight=10.0,
    )

    joint_q = robot_model.joint_q.reshape((1, int(robot_model.joint_coord_count)))
    solver = ik.IKSolver(
        model=robot_model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    solver.step(joint_q, joint_q, iterations=ik_iterations)

    jq = joint_q.numpy().reshape(-1).astype(robot_model.joint_q.dtype)
    jqd = np_zeros_like_joint_qd(robot_model)

    robot_model.joint_q.assign(jq)
    robot_model.joint_qd.assign(jqd)
    robot_state_0.joint_q.assign(jq)
    robot_state_0.joint_qd.assign(jqd)
    newton.eval_fk(robot_model, robot_model.joint_q, robot_model.joint_qd, robot_state_0)


def np_zeros_like_joint_qd(robot_model: newton.Model):
    import numpy as np

    return np.zeros(int(robot_model.joint_dof_count), dtype=np.float32)


def sync_mujoco_visual_state(
    mj_solver: SolverMuJoCo,
    robot_model: newton.Model,
    state: Any,
) -> None:
    """Push Newton ``joint_q`` into MuJoCo and run forward kinematics for the passive viewer.

    ``_update_mjc_data`` alone updates ``mj_data.qpos`` but not body poses; ``mj_forward`` (CPU)
    or ``mjw_data`` sync (GPU) is required before :meth:`~newton.solvers.SolverMuJoCo.render_mujoco_viewer`.
    """
    mj_solver._update_mjc_data(mj_solver.mj_data, robot_model, state)
    if mj_solver.use_mujoco_cpu:
        mj_solver._mujoco.mj_forward(mj_solver.mj_model, mj_solver.mj_data)
    else:
        with wp.ScopedDevice(robot_model.device):
            mj_solver._update_mjc_data(mj_solver.mjw_data, robot_model, state)


def init_mujoco_actuator_targets_from_model(
    robot_model: newton.Model,
    control: Any,
) -> None:
    """Align MuJoCo position actuators with the model's current ``joint_q`` (post-bootstrap)."""
    control.joint_target_pos.assign(robot_model.joint_q.numpy())
    control.joint_target_vel.assign(np_zeros_like_joint_qd(robot_model))


def sync_mujoco_actuator_targets_from_joint_q(
    robot_model: newton.Model,
    state: Any,
    control: Any,
    target_joint_q: Any,
    *,
    frame_dt: float,
    command_velocity: EEVelocity | None = None,
) -> None:
    """Write IK joint targets into ``control`` for ``SolverMuJoCo`` position actuators.

    MuJoCo integrates the arm toward ``joint_target_pos`` each ``mj_solver.step``; do not
    teleport ``state.joint_q`` when using this path.

    When ``command_velocity`` is zero (keyboard idle), ``joint_target_vel`` is set to zero
    so PD does not chase a perpetual ``(q_tgt - q_cur) / frame_dt`` feedforward.
    """
    import numpy as np

    n_dof = int(robot_model.joint_dof_count)
    q_tgt = np.asarray(target_joint_q, dtype=np.float32).reshape(-1)[:n_dof]
    q_cur = state.joint_q.numpy().reshape(-1).astype(np.float32)[:n_dof]
    if command_velocity is not None and command_velocity.is_zero():
        qd_tgt = np.zeros(n_dof, dtype=np.float32)
        if float(np.linalg.norm(q_tgt[:n_dof] - q_cur[:n_dof])) < 0.02:
            q_tgt = q_cur.copy()
    elif frame_dt > 1e-9:
        qd_tgt = (q_tgt - q_cur) / float(frame_dt)
    else:
        qd_tgt = np.zeros(n_dof, dtype=np.float32)
    control.joint_target_pos.assign(q_tgt)
    control.joint_target_vel.assign(qd_tgt.astype(np.float32))


# ---------------------------------------------------------------------------
# End-effector velocity teleop (world-frame twist → integrated TCP target → IK)
# ---------------------------------------------------------------------------


class _KeyViewer(Protocol):
    def is_key_down(self, key: str) -> bool: ...


@dataclass(frozen=True)
class EEVelocity:
    """TCP twist in the **world** frame (m/s and rad/s)."""

    linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def linear_vec(self) -> wp.vec3:
        return wp.vec3(*self.linear)

    @property
    def angular_vec(self) -> wp.vec3:
        return wp.vec3(*self.angular)

    def is_zero(self, tol: float = 1e-9) -> bool:
        return all(abs(v) < tol for v in (*self.linear, *self.angular))


def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    """Hamilton product ``a * b`` (warp quats are ``x, y, z, w``)."""
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def integrate_tcp_target(
    target: wp.transform,
    *,
    linear_vel: wp.vec3,
    angular_vel: wp.vec3,
    dt: float,
) -> wp.transform:
    """Integrate a rigid TCP target pose by ``dt`` using a constant world-frame twist."""
    pos = wp.transform_get_translation(target)
    rot = wp.transform_get_rotation(target)
    pos_new = pos + linear_vel * dt
    w = angular_vel
    ang_mag = float(wp.length(w))
    if ang_mag > 1e-12:
        delta_rot = wp.quat_from_axis_angle(w / ang_mag, ang_mag * dt)
        rot_new = wp.normalize(_quat_mul(delta_rot, rot))
    else:
        rot_new = rot
    return wp.transform(pos_new, rot_new)


# (key, action) — must stay in sync with :func:`read_keyboard_ee_velocity` axis pairs.
FR3_KEYBOARD_BINDINGS: tuple[tuple[str, str], ...] = (
    ("i", "translate TCP +world X"),
    ("k", "translate TCP -world X"),
    ("j", "translate TCP +world Y"),
    ("l", "translate TCP -world Y"),
    ("r", "translate TCP +world Z"),
    ("f", "translate TCP -world Z"),
    ("z", "rotate TCP +world Z"),
    ("x", "rotate TCP -world Z"),
    ("t", "rotate TCP +world Y"),
    ("g", "rotate TCP -world Y"),
    ("u", "rotate TCP +world X"),
    ("o", "rotate TCP -world X"),
)


def print_fr3_keyboard_bindings(*, stream: Any | None = None) -> None:
    """Print FR3 TCP teleop key map (``ViewerGL``; focus the simulation window)."""
    import sys

    out = sys.stdout if stream is None else stream
    print("FR3 keyboard teleop — focus the viewer window:", file=out)
    for key, action in FR3_KEYBOARD_BINDINGS:
        print(f"  {key}: {action}", file=out)
    print("  (W/A/S/D/Q/E move the camera, not the arm.)", file=out)


def _keyboard_axis(viewer: _KeyViewer, neg_key: str, pos_key: str) -> float:
    val = 0.0
    if viewer.is_key_down(neg_key):
        val -= 1.0
    if viewer.is_key_down(pos_key):
        val += 1.0
    return val


def poll_viewer_events(viewer: object | None) -> None:
    """Process pending GL window events so the next :func:`is_key_down` query is current.

    Newton's ``ViewerGL`` only polls the keyboard during ``end_frame()``; call this at the
    start of each simulation step when the main loop runs ``step()`` before ``render()``.
    """
    if viewer is None:
        return
    renderer = getattr(viewer, "renderer", None)
    if renderer is not None and hasattr(renderer, "update"):
        renderer.update()


def read_keyboard_ee_velocity(
    viewer: _KeyViewer | None,
    *,
    linear_speed: float = 0.2,
    angular_speed: float = 1.0,
    poll_events: bool = True,
) -> EEVelocity:
    """Read a world-frame TCP twist from the Newton ``ViewerGL`` keyboard (window must have focus).

    Requires ``ViewerGL`` (``is_key_down`` is a no-op on ``ViewerNull`` / ``ViewerViser``).

    Layout (avoids ``ViewerGL`` camera keys **W/A/S/D/Q/E**):

    - **I / K** — world ±X
    - **J / L** — world ±Y
    - **R / F** — world ±Z
    - **U / O** — rotate about world Z
    - **T / G** — rotate about world Y
    - **Z / X** — rotate about world X
    """
    if viewer is None or not hasattr(viewer, "is_key_down"):
        return EEVelocity()
    if poll_events:
        poll_viewer_events(viewer)
    lin = (
        _keyboard_axis(viewer, "k", "i") * linear_speed,
        _keyboard_axis(viewer, "l", "j") * linear_speed,
        _keyboard_axis(viewer, "f", "r") * linear_speed,
    )
    ang = (
        _keyboard_axis(viewer, "x", "z") * angular_speed,
        _keyboard_axis(viewer, "g", "t") * angular_speed,
        _keyboard_axis(viewer, "o", "u") * angular_speed,
    )
    return EEVelocity(linear=lin, angular=ang)


def _quat_to_ik_vec4(q: wp.quat) -> wp.vec4:
    return wp.vec4(q[0], q[1], q[2], q[3])


def _make_tcp_ik_solver(
    robot_model: newton.Model,
    tcp_body_index: int,
    target_tf: wp.transform,
    *,
    joint_limit_weight: float = 10.0,
    lambda_initial: float = 0.1,
):
    """Build position + rotation + joint-limit IK objectives for ``tcp``."""
    import newton.ik as ik

    dev = robot_model.device
    target_pos = wp.transform_get_translation(target_tf)
    target_rot = wp.transform_get_rotation(target_tf)
    pos_obj = ik.IKObjectivePosition(
        link_index=tcp_body_index,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([target_pos], dtype=wp.vec3, device=dev),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=tcp_body_index,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([_quat_to_ik_vec4(target_rot)], dtype=wp.vec4, device=dev),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=robot_model.joint_limit_lower,
        joint_limit_upper=robot_model.joint_limit_upper,
        weight=joint_limit_weight,
    )
    joint_q = robot_model.joint_q.reshape((1, int(robot_model.joint_coord_count)))
    solver = ik.IKSolver(
        model=robot_model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, limits],
        lambda_initial=lambda_initial,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    return pos_obj, rot_obj, joint_q, solver


class Fr3EEVelocityController:
    """Integrate a TCP velocity command, solve IK, and write ``joint_q`` on the robot model."""

    def __init__(
        self,
        robot_model: newton.Model,
        tcp_body_index: int,
        *,
        linear_speed: float = 0.2,
        angular_speed: float = 1.0,
        ik_iterations: int = 24,
        joint_limit_weight: float = 10.0,
    ) -> None:
        self.robot_model = robot_model
        self.tcp_body_index = tcp_body_index
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.ik_iterations = ik_iterations
        self.target_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        self._pos_obj, self._rot_obj, self.joint_q, self._ik_solver = _make_tcp_ik_solver(
            robot_model,
            tcp_body_index,
            self.target_tf,
            joint_limit_weight=joint_limit_weight,
        )

    def sync_target_from_state(self, state: Any) -> None:
        """Set the integrated TCP target to the current FK pose of ``tcp``."""
        bq = state.body_q.numpy().reshape(-1, 7)[self.tcp_body_index]
        self.target_tf = wp.transform(
            wp.vec3(float(bq[0]), float(bq[1]), float(bq[2])),
            wp.quat(float(bq[3]), float(bq[4]), float(bq[5]), float(bq[6])),
        )
        self._push_target_to_ik()

    def _push_target_to_ik(self) -> None:
        pos = wp.transform_get_translation(self.target_tf)
        q = wp.transform_get_rotation(self.target_tf)
        self._pos_obj.set_target_position(0, pos)
        self._rot_obj.set_target_rotation(0, _quat_to_ik_vec4(q))

    def seed_ik_from_state(self, state: Any) -> None:
        """Copy simulated ``state.joint_q`` into the IK seed before :meth:`solve_ik`."""
        jq = state.joint_q.numpy().reshape(1, int(self.robot_model.joint_coord_count))
        self.joint_q.assign(jq.astype(self.robot_model.joint_q.dtype))

    def advance_target(
        self,
        dt: float,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
    ) -> EEVelocity:
        """Integrate the TCP target twist on the host (safe to call outside a CUDA graph capture)."""
        if velocity is None:
            velocity = read_keyboard_ee_velocity(
                viewer,
                linear_speed=self.linear_speed,
                angular_speed=self.angular_speed,
                poll_events=poll_events,
            )
        self.target_tf = integrate_tcp_target(
            self.target_tf,
            linear_vel=velocity.linear_vec,
            angular_vel=velocity.angular_vec,
            dt=dt,
        )
        self._push_target_to_ik()
        return velocity

    def run_ik_teleop_frame(
        self,
        dt: float,
        state: Any,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
    ) -> EEVelocity:
        """Integrate TCP target, optionally sync on idle, and solve IK from ``state``."""
        velocity = self.advance_target(
            dt, velocity=velocity, viewer=viewer, poll_events=poll_events
        )
        if velocity.is_zero():
            self.sync_target_from_state(state)
        self.solve_ik(state)
        return velocity

    def solve_ik(self, state: Any | None = None) -> None:
        """Run the IK solver for the current target (may be CUDA-graph captured).

        Pass ``state`` (or call :meth:`seed_ik_from_state` first) so the seed matches the
        simulated arm rather than a stale ``robot_model.joint_q``.
        """
        if state is not None:
            self.seed_ik_from_state(state)
        self._ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)

    def step(
        self,
        dt: float,
        *,
        velocity: EEVelocity | None = None,
        viewer: _KeyViewer | None = None,
        poll_events: bool = True,
    ) -> EEVelocity:
        """Advance the TCP target, run IK, and leave the solution in ``joint_q``."""
        velocity = self.advance_target(
            dt, velocity=velocity, viewer=viewer, poll_events=poll_events
        )
        self.solve_ik()
        return velocity

    def apply_to_model_and_state(self, state: Any) -> None:
        """Copy the latest IK ``joint_q`` into the model and ``state``, then refresh FK.

        Kinematic teleop only (e.g. ``example_fr3_keyboard`` without MuJoCo stepping).
        For the coupled stack use :meth:`apply_ik_to_mujoco_control` instead.
        """
        import numpy as np

        jq = self.joint_q.numpy().reshape(-1).astype(self.robot_model.joint_q.dtype)
        jqd = np.zeros(int(self.robot_model.joint_dof_count), dtype=np.float32)
        self.robot_model.joint_q.assign(jq)
        self.robot_model.joint_qd.assign(jqd)
        state.joint_q.assign(jq)
        state.joint_qd.assign(jqd)
        newton.eval_fk(self.robot_model, self.robot_model.joint_q, self.robot_model.joint_qd, state)

    def apply_ik_to_mujoco_control(
        self,
        state: Any,
        control: Any,
        frame_dt: float,
        *,
        command_velocity: EEVelocity | None = None,
    ) -> None:
        """Push the latest IK solution to MuJoCo PD actuators (``joint_target_pos`` / ``vel``)."""
        sync_mujoco_actuator_targets_from_joint_q(
            self.robot_model,
            state,
            control,
            self.joint_q.numpy().reshape(-1),
            frame_dt=frame_dt,
            command_velocity=command_velocity,
        )


class Fr3EEDirectJointController(Fr3EEVelocityController):
    """Testing controller: TCP velocity + IK, then **direct** ``joint_q`` write (kinematic arm).

    Pair with :meth:`~apple_pick_sim.coupled_fruiting.CoupledFruitingScene.apply_fr3_ee_teleop_direct`
    and ``robot_kinematic_mode=True`` so ``SolverMuJoCo`` does not re-integrate the arm between
    substeps. Coupled VBD + proxy sync still run.
    """

    def apply_direct_joints(
        self,
        state: Any,
        control: Any | None = None,
        *,
        mj_solver: SolverMuJoCo | None = None,
    ) -> None:
        """Write IK ``joint_q`` into model/state, zero ``joint_qd``, refresh FK and MuJoCo buffers."""
        self.apply_to_model_and_state(state)
        if control is not None:
            init_mujoco_actuator_targets_from_model(self.robot_model, control)
        if mj_solver is not None:
            sync_mujoco_visual_state(mj_solver, self.robot_model, state)
