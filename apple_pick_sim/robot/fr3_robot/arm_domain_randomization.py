"""Per-env FR3 arm domain randomization: joint dynamics, link mass/inertia, EE payload.

Applied AFTER the batched robot model is built (nominal values are identical
across worlds at that point) and after any one-shot VIC setup
(``configure_vic_joint_torques_arm_batched``). Link-mass/inertia and EE-payload
scaling read the model's CURRENT per-world value as the baseline to scale from,
so this is safe to call exactly once right after build.

Unlike plant domain randomization (baked into the fruiting-system build and
fixed for the run), arm DR is a resample of live ``newton.Model`` arrays plus
one ``notify_model_changed`` call, so it is cheap to call again from a gym
``reset()`` path to re-randomize the arm **per episode**. Callers doing that
should keep the ``np.random.Generator`` across resets (or reseed
deterministically) rather than create a fresh one each time.

**Controller gains are sampled here but not yet wired per-env.**
:func:`sample_arm_domain_randomization` produces ``kp_null``/``kd_null``
arrays for completeness (per the design spec's "controller gains" axis), but
``apple_pick_sim.coupled_fruiting.vic_joint_torques_batched``'s null-space law
currently takes ``kp_null``/``kd_null`` as batch-uniform Python floats
(see ``allocate_vic_joint_torque_buffers_batched``). The underlying torch math
would broadcast a per-env ``(N, 1)`` tensor correctly with no other changes
(``u_null = kd_null * (-joint_vel) + kp_null * dist`` on ``(N, 7)`` tensors),
but threading that through ``scene.vic_jt_kp_null`` / ``scene.vic_jt_kd_null``
and their consumers is a separate, contained follow-up -- out of scope for
this module, which only writes ``newton.Model`` arrays.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import newton
from newton.solvers import SolverMuJoCo

from apple_pick_sim.robot.fr3_robot import setup as fr3_setup
from apple_pick_sim.robot.fr3_robot.fr3_v21_props import (
    _set_body_inertial_full,
    _tile_body_indices,
    load_fr3_v21_inertials,
    resolve_fr3_link_body_index,
)

_N_ARM_DOF = 7
_DEFAULT_KP_NULL = 10.0
_DEFAULT_KD_NULL = 6.3246


@dataclasses.dataclass(frozen=True)
class ArmDomainRandomizationRanges:
    """Multiplicative ``(min, max)`` scale ranges applied to nominal per-env arm properties.

    Defaults are a starting design choice pending RL tuning, not derived from
    data. ``ee_payload_*`` ranges are kept tight relative to the others
    because the ``/fr3/ee`` mass/COM/inertia were deliberately calibrated in
    the real-sim feature-alignment work (Slice 0); randomizing them widely
    would undo that calibration rather than model genuine arm-to-arm variance.
    """

    armature_scale: tuple[float, float] = (0.7, 1.3)
    friction_scale: tuple[float, float] = (0.5, 1.5)
    joint_damping_scale: tuple[float, float] = (0.7, 1.3)
    link_mass_scale: tuple[float, float] = (0.9, 1.1)
    link_inertia_scale: tuple[float, float] = (0.9, 1.1)
    ee_payload_mass_scale: tuple[float, float] = (0.95, 1.05)
    ee_payload_inertia_scale: tuple[float, float] = (0.95, 1.05)
    kp_null_scale: tuple[float, float] = (0.8, 1.2)
    kd_null_scale: tuple[float, float] = (0.8, 1.2)


@dataclasses.dataclass(frozen=True)
class ArmDomainRandomizationSample:
    """Per-env sampled arm DR: ``(N, 7)`` DOF arrays plus ``(N,)`` scalar scale factors."""

    armature: np.ndarray  # (N, 7)
    friction: np.ndarray  # (N, 7)
    joint_damping: np.ndarray  # (N, 7)
    link_mass_scale: np.ndarray  # (N,)
    link_inertia_scale: np.ndarray  # (N,)
    ee_mass_scale: np.ndarray  # (N,)
    ee_inertia_scale: np.ndarray  # (N,)
    kp_null: np.ndarray  # (N,) -- sampled only; see module docstring
    kd_null: np.ndarray  # (N,) -- sampled only; see module docstring


def sample_arm_domain_randomization(
    ranges: ArmDomainRandomizationRanges,
    *,
    num_envs: int,
    rng: np.random.Generator,
    nominal_armature: np.ndarray | tuple[float, ...] = fr3_setup.FR3_REFLECTED_MOTOR_INERTIA_KGM2,
    nominal_friction: np.ndarray | tuple[float, ...] = fr3_setup.FR3_DEFAULT_JOINT_FRICTION,
    nominal_joint_damping: np.ndarray | tuple[float, ...] = fr3_setup.FR3_DEFAULT_VIC_JOINT_DAMPING,
    nominal_kp_null: float = _DEFAULT_KP_NULL,
    nominal_kd_null: float = _DEFAULT_KD_NULL,
) -> ArmDomainRandomizationSample:
    """Draw one uniform sample per env for every arm DR axis in ``ranges``."""
    n = int(num_envs)

    def _u(bounds: tuple[float, float], size) -> np.ndarray:
        lo, hi = bounds
        return rng.uniform(lo, hi, size=size).astype(np.float32)

    nominal_arm = np.asarray(nominal_armature, dtype=np.float32).reshape(1, _N_ARM_DOF)
    nominal_fric = np.asarray(nominal_friction, dtype=np.float32).reshape(1, _N_ARM_DOF)
    nominal_damp = np.asarray(nominal_joint_damping, dtype=np.float32).reshape(1, _N_ARM_DOF)

    return ArmDomainRandomizationSample(
        armature=nominal_arm * _u(ranges.armature_scale, (n, 1)),
        friction=nominal_fric * _u(ranges.friction_scale, (n, 1)),
        joint_damping=nominal_damp * _u(ranges.joint_damping_scale, (n, 1)),
        link_mass_scale=_u(ranges.link_mass_scale, n),
        link_inertia_scale=_u(ranges.link_inertia_scale, n),
        ee_mass_scale=_u(ranges.ee_payload_mass_scale, n),
        ee_inertia_scale=_u(ranges.ee_payload_inertia_scale, n),
        kp_null=float(nominal_kp_null) * _u(ranges.kp_null_scale, n),
        kd_null=float(nominal_kd_null) * _u(ranges.kd_null_scale, n),
    )


def apply_joint_dynamics_dr(
    robot_model: newton.Model,
    mj_solver: SolverMuJoCo,
    sample: ArmDomainRandomizationSample,
    *,
    dofs_per_world: int,
) -> None:
    """Write per-env armature/friction/damping into the batched Model and sync once.

    ``sample.armature``/``friction``/``joint_damping`` are each ``(N, 7)`` --
    the per-world 2D form the generalized ``setup.py`` setters accept.
    """
    fr3_setup._set_fr3_joint_armature(
        robot_model, sample.armature, dofs_per_world=int(dofs_per_world)
    )
    fr3_setup._set_fr3_joint_friction(
        robot_model, sample.friction, dofs_per_world=int(dofs_per_world)
    )
    fr3_setup._set_vic_passive_joint_damping(
        robot_model, sample.joint_damping, dofs_per_world=int(dofs_per_world)
    )
    mj_solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)


def _scale_body_inertial(model: newton.Model, body_index: int, mass_scale: float, inertia_scale: float) -> None:
    """Multiply the body's current mass/inertia by the given scale factors; COM unchanged."""
    idx = int(body_index)
    mass = float(model.body_mass.numpy()[idx]) * float(mass_scale)
    inertia = model.body_inertia.numpy()[idx] * float(inertia_scale)
    com = tuple(model.body_com.numpy()[idx].tolist())
    _set_body_inertial_full(model, idx, mass_kg=mass, com_m=com, inertia=inertia)


def apply_link_mass_inertia_dr(
    robot_model: newton.Model,
    mj_solver: SolverMuJoCo,
    sample: ArmDomainRandomizationSample,
    *,
    robot_bodies_per_world: int,
    num_envs: int,
) -> None:
    """Scale every FR3 arm link body per env by that env's mass/inertia factor, then sync.

    All 7 arm links in a given env are scaled by the SAME per-env factor
    (``sample.link_mass_scale[w]`` / ``sample.link_inertia_scale[w]``) --
    modeling "this env's arm mass estimate is systematically off by X%"
    rather than randomizing each link independently.
    """
    links = load_fr3_v21_inertials()
    template_indices = {
        link.link_num: resolve_fr3_link_body_index(robot_model, link.link_num) for link in links
    }
    tiled = _tile_body_indices(
        template_indices,
        robot_bodies_per_world=int(robot_bodies_per_world),
        num_envs=int(num_envs),
    )
    for indices in tiled.values():
        for w, idx in enumerate(indices):
            _scale_body_inertial(
                robot_model, idx, sample.link_mass_scale[w], sample.link_inertia_scale[w]
            )
    mj_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)


def _resolve_ee_body_indices_per_world(
    robot_model: newton.Model, *, robot_bodies_per_world: int, num_envs: int
) -> list[int]:
    """Resolve the ``ee`` payload body index for every world (mirrors ``resolve_ee_body_index``)."""
    labels = list(robot_model.body_label)
    hits = [i for i, lbl in enumerate(labels) if lbl.split("/")[-1] == "ee"]
    if not hits:
        raise ValueError("no 'ee' body found in body_label")
    template_idx = min(hits)
    tiled = _tile_body_indices(
        {0: template_idx},
        robot_bodies_per_world=int(robot_bodies_per_world),
        num_envs=int(num_envs),
    )
    return tiled[0]


def apply_ee_payload_dr(
    robot_model: newton.Model,
    mj_solver: SolverMuJoCo,
    sample: ArmDomainRandomizationSample,
    *,
    robot_bodies_per_world: int,
    num_envs: int,
) -> None:
    """Scale the ``ee`` payload body per env (tight band -- sys-ID calibrated), then sync."""
    ee_indices = _resolve_ee_body_indices_per_world(
        robot_model, robot_bodies_per_world=int(robot_bodies_per_world), num_envs=int(num_envs)
    )
    for w, idx in enumerate(ee_indices):
        _scale_body_inertial(robot_model, idx, sample.ee_mass_scale[w], sample.ee_inertia_scale[w])
    mj_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
