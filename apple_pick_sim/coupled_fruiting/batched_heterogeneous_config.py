"""Frozen configuration dataclasses for batched heterogeneous coupled sim (V.3.1).

Composes build-time settings for ``BatchedHeterogeneousCoupledSim``.
Viewer, argparse, and keyboard teleop do not belong here.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Literal

from apple_pick_sim.coupled_fruiting.scene import (
    DEFAULT_FR3_MUJOCO_SOLVER_KWARGS,
    DEFAULT_STEM_COUPLING_GAIN,
    DEFAULT_STEM_FORCE_CAP_N,
    DEFAULT_STEM_TORQUE_CAP_NM,
)
from apple_pick_sim.coupled_fruiting.settle_ke_decay import (
    DEFAULT_KE_SAMPLE_EVERY,
    SettleKeAnalysisConfig,
)
from apple_pick_sim.fruiting_system.build import (
    FRUITING_VBD_RIGID_JOINT_ANGULAR_KD,
    FRUITING_VBD_RIGID_JOINT_LINEAR_KD,
)
from apple_pick_sim.fruiting_system import default_ranges_fixture_path
from apple_pick_sim.fruiting_system.params import (
    FruitingSystemParams,
    GripperProxyConfig,
    PLACEHOLDER_EE_MASS_KG,
)
from apple_pick_sim.robot.fr3_robot.controllers.ee_impedance import ImpedanceGains
from apple_pick_sim.robot.fr3_robot.placement import IK_BOOTSTRAP_DEFAULT_ITERATIONS
from apple_pick_sim.sim_device import resolve_sim_device

# Per-role FIXED-joint kd overrides (see docs/damping-tuning.md §3).
# Defaults mirror ``make_fruiting_solver_vbd`` → Newton ``SolverVBD`` rigid joint kd.
_JOINT_KD_OVERRIDE_ROLES = ("support", "primary_spur", "spur_stem", "stem_apple")

_DEFAULT_JOINT_ANGULAR_KD_OVERRIDES: dict[str, float] = {
    role: FRUITING_VBD_RIGID_JOINT_ANGULAR_KD for role in _JOINT_KD_OVERRIDE_ROLES
}

_DEFAULT_JOINT_LINEAR_KD_OVERRIDES: dict[str, float] = {
    role: FRUITING_VBD_RIGID_JOINT_LINEAR_KD for role in _JOINT_KD_OVERRIDE_ROLES
}

# Tuned overrides shared by heterogeneous + sys-ID batched examples (see docs/damping-tuning.md).
EXAMPLE_JOINT_ANGULAR_KD_OVERRIDES: dict[str, float] = {
    "support": 0.3,
    "primary_spur": 0.3,
    "spur_stem": 0.3,
    "stem_apple": 0.3,
}
EXAMPLE_JOINT_LINEAR_KD_OVERRIDES: dict[str, float] = {
    "support": 0.3,
    "primary_spur": 0.3,
    "spur_stem": 0.3,
    "stem_apple": 0.3,
}
EXAMPLE_JOINT_ANGULAR_KP_OVERRIDES: dict[str, float] = {"support": 2000.0}
EXAMPLE_JOINT_LINEAR_KP_OVERRIDES: dict[str, float] = {"support": 2000.0}

# Heterogeneous / sys-ID example VIC defaults (match variance fixture sim_build).
_VIC_DEFAULT_LINEAR_K = 200.0
_VIC_DEFAULT_LINEAR_D = 10.0
_VIC_DEFAULT_ANGULAR_K = 10.0
_VIC_DEFAULT_ANGULAR_D = 1.0

RobotKind = Literal["fr3"]
StepMode = Literal["coupled", "vbd_only"]
ControllerMode = Literal["direct", "ee", "vic"]

_TESTS_RANGES_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "fruiting_system_ranges_straight_rod_test.json"
)


def _default_gripper_proxy_config() -> GripperProxyConfig:
    """Gripper shape/mass only; weld flags come from :attr:`RobotConfig.fix_to_apple` at build."""
    return GripperProxyConfig(
        mass=PLACEHOLDER_EE_MASS_KG,
        fix_to_apple=False,
        robot_facing_weld=False,
    )


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    """Batch topology, device, and control-step timing."""

    num_envs: int = 4
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 2.0)
    device: str | None = None
    control_hz: float = 60.0
    sub_dt: float = 1.0 / 1800.0

    @property
    def substeps_per_step(self) -> int:
        return max(1, round(1.0 / (self.control_hz * self.sub_dt)))

    @property
    def frame_dt(self) -> float:
        return float(self.substeps_per_step) * float(self.sub_dt)


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    """Manipulator, gripper proxy, and weld/coupling mode."""

    kind: RobotKind = "fr3"
    step_mode: StepMode = "coupled"
    fix_to_apple: bool = True
    gripper: GripperProxyConfig = dataclasses.field(default_factory=_default_gripper_proxy_config)
    robot_base_pos: tuple[float, float, float] | None = None
    per_env_ik: bool = True
    ik_bootstrap_iterations: int = IK_BOOTSTRAP_DEFAULT_ITERATIONS
    skip_ik_bootstrap: bool = True
    defer_template_robot_bootstrap: bool = True
    force_batched_layout: bool = False


@dataclasses.dataclass(frozen=True)
class SceneSettleCollisionConfig:
    """Fruiting placement, VBD settle, and AVBD collision policy."""

    fruiting_base_pos: tuple[float, float, float] | None = None
    settle_substeps: int = 5000
    settle_gravity_ramp: bool = False
    settle_quiet_every: int | None = None
    settle_max_speed_m_s: float = 0.05
    enable_self_collisions: bool = False
    enable_apple_woody_collisions: bool = True
    enable_proxy_woody_collisions: bool = True


@dataclasses.dataclass(frozen=True)
class DomainRandomizationConfig:
    """How per-env :class:`~apple_pick_sim.fruiting_system.FruitingSystemParams` are chosen."""

    ranges_path: Path | None = None
    topology_seed: int | None = None
    per_env_params: tuple[FruitingSystemParams, ...] | None = None

    @property
    def inject_mode(self) -> bool:
        return self.per_env_params is not None

    def resolved_ranges_path(self) -> Path:
        if self.ranges_path is not None:
            return Path(self.ranges_path)
        return default_ranges_fixture_path()


@dataclasses.dataclass(frozen=True)
class FruitingSystemConfig:
    """Fruiting physics knobs applied at build (not DR sampling policy)."""

    stem_coupling_gain: float = DEFAULT_STEM_COUPLING_GAIN
    stem_force_cap_N: float | None = DEFAULT_STEM_FORCE_CAP_N
    stem_torque_cap_Nm: float | None = DEFAULT_STEM_TORQUE_CAP_NM
    stem_harvest_explicit_apple_weight: bool = False
    joint_angular_kd_overrides: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(_DEFAULT_JOINT_ANGULAR_KD_OVERRIDES)
    )
    joint_linear_kd_overrides: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(_DEFAULT_JOINT_LINEAR_KD_OVERRIDES)
    )
    joint_angular_kp_overrides: dict[str, float] = dataclasses.field(default_factory=dict)
    joint_linear_kp_overrides: dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ControllerConfig:
    """Frame-rate robot control (direct / ee / vic)."""

    mode: ControllerMode = "vic"
    action_dim: int = 6
    linear_speed: float = 1.0
    angular_speed: float = 1.0
    ik_iterations: int = 128
    vic_gains: ImpedanceGains = dataclasses.field(
        default_factory=lambda: ImpedanceGains(
            linear_k=_VIC_DEFAULT_LINEAR_K,
            linear_d=_VIC_DEFAULT_LINEAR_D,
            angular_k=_VIC_DEFAULT_ANGULAR_K,
            angular_d=_VIC_DEFAULT_ANGULAR_D,
        )
    )
    allocate_action_buffer: bool = True

    def expected_action_shape(self, num_envs: int) -> tuple[int, int]:
        """Return ``(num_envs, action_dim)``."""
        return (int(num_envs), int(self.action_dim))

    def validate_actions(
        self,
        actions,
        *,
        num_envs: int,
        device: str,
        robot_step_mode: StepMode,
    ):
        """Validate action tensor; does not clip speeds."""
        if robot_step_mode == "vbd_only":
            raise ValueError(
                "robot_step_mode='vbd_only' does not accept actions; pass actions=None to step()"
            )

        import torch

        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"actions must be torch.Tensor, got {type(actions).__name__}")
        if actions.dtype != torch.float32:
            raise ValueError(f"actions must be float32, got {actions.dtype}")
        if str(actions.device) != str(device):
            raise ValueError(
                f"actions device {actions.device!s} does not match sim device {device!r}"
            )

        action_dim = int(self.action_dim)
        n = int(num_envs)
        if actions.ndim == 1:
            if actions.shape != (action_dim,):
                raise ValueError(
                    f"actions broadcast shape must be ({action_dim},), got {tuple(actions.shape)}"
                )
            actions = actions.unsqueeze(0).expand(n, action_dim)
        elif actions.shape != (n, action_dim):
            raise ValueError(
                f"actions shape must be ({n}, {action_dim}) or ({action_dim},), "
                f"got {tuple(actions.shape)}"
            )
        return actions.contiguous()


@dataclasses.dataclass(frozen=True)
class MujocoConfig:
    """MuJoCo solver options for the coupled robot model."""

    solver_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_FR3_MUJOCO_SOLVER_KWARGS)
    )
    use_cpu: bool | None = None


@dataclasses.dataclass(frozen=True)
class SettleDiagnosticsConfig:
    """Optional settle KE/stability reporting (examples and CI; off in gym)."""

    enabled: bool = True
    ke_analysis: SettleKeAnalysisConfig = dataclasses.field(default_factory=SettleKeAnalysisConfig)
    ke_sample_every: int = DEFAULT_KE_SAMPLE_EVERY
    report_brief: bool = False


@dataclasses.dataclass(frozen=True)
class ObsConfig:
    """GPU batched observation buffer allocation (``gather_batched_obs``)."""

    allocate_buffers: bool = True
    include_robot: bool = True
    include_forces: bool = True


@dataclasses.dataclass(frozen=True)
class BatchedHeterogeneousCoupledSimConfig:
    """Root config for batched heterogeneous coupled fruiting sim build and stepping."""

    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
    robot: RobotConfig = dataclasses.field(default_factory=RobotConfig)
    scene: SceneSettleCollisionConfig = dataclasses.field(default_factory=SceneSettleCollisionConfig)
    domain_randomization: DomainRandomizationConfig = dataclasses.field(
        default_factory=DomainRandomizationConfig
    )
    fruiting_system: FruitingSystemConfig = dataclasses.field(default_factory=FruitingSystemConfig)
    controller: ControllerConfig = dataclasses.field(default_factory=ControllerConfig)
    mujoco: MujocoConfig = dataclasses.field(default_factory=MujocoConfig)
    settle_diagnostics: SettleDiagnosticsConfig | None = None
    obs: ObsConfig | None = dataclasses.field(default_factory=ObsConfig)

    def resolve_device(self) -> str:
        return resolve_sim_device(self.runtime.device)

    def validate(self) -> None:
        """Cross-field validation; call before build."""
        if self.runtime.num_envs < 1:
            raise ValueError(f"runtime.num_envs must be >= 1, got {self.runtime.num_envs}")
        if self.runtime.control_hz <= 0.0:
            raise ValueError(f"runtime.control_hz must be positive, got {self.runtime.control_hz}")
        if self.runtime.sub_dt <= 0.0:
            raise ValueError(f"runtime.sub_dt must be positive, got {self.runtime.sub_dt}")
        if self.scene.settle_substeps < 0:
            raise ValueError(
                f"scene.settle_substeps must be >= 0, got {self.scene.settle_substeps}"
            )
        if self.scene.settle_quiet_every is not None and int(self.scene.settle_quiet_every) <= 0:
            raise ValueError(
                "scene.settle_quiet_every must be positive when set, "
                f"got {self.scene.settle_quiet_every}"
            )

        injected = self.domain_randomization.per_env_params
        if injected is not None and len(injected) != self.runtime.num_envs:
            raise ValueError(
                f"domain_randomization.per_env_params length ({len(injected)}) "
                f"must match runtime.num_envs ({self.runtime.num_envs})"
            )

        if self.robot.kind != "fr3":
            raise ValueError(f"robot.kind must be 'fr3', got {self.robot.kind!r}")

        if self.controller.mode == "vic":
            if self.robot.step_mode != "coupled":
                raise ValueError("controller.mode='vic' requires robot.step_mode='coupled'")

        if self.robot.step_mode == "vbd_only" and self.controller.mode != "direct":
            raise ValueError(
                "robot.step_mode='vbd_only' requires controller.mode='direct' "
                f"(got {self.controller.mode!r})"
            )

        if self.robot.step_mode != "vbd_only" and self.controller.action_dim < 1:
            raise ValueError(
                f"controller.action_dim must be >= 1 for step_mode={self.robot.step_mode!r}, "
                f"got {self.controller.action_dim}"
            )

        if (
            self.robot.kind == "fr3"
            and self.robot.step_mode == "coupled"
            and not self.controller.allocate_action_buffer
        ):
            raise ValueError(
                "robot.kind='fr3' with step_mode='coupled' requires "
                "controller.allocate_action_buffer=True so per-env actions reach teleop"
            )

        if self.robot.gripper.fix_to_apple and not self.robot.fix_to_apple:
            warnings.warn(
                "robot.gripper.fix_to_apple=True but robot.fix_to_apple=False; "
                "build applies robot.fix_to_apple to the gripper proxy at weld time",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def defaults(cls) -> BatchedHeterogeneousCoupledSimConfig:
        """Heterogeneous example physics: 4 envs, variance fixture, example settle/collision."""
        return cls(
            runtime=RuntimeConfig(control_hz=30.0),
            domain_randomization=DomainRandomizationConfig(
                ranges_path=default_ranges_fixture_path(),
            ),
            settle_diagnostics=SettleDiagnosticsConfig(),
            obs=None,
        )

    @classmethod
    def gym_defaults(cls, *, num_envs: int = 1) -> BatchedHeterogeneousCoupledSimConfig:
        """Gym-oriented preset: VIC coupled FR3, fix-to-apple, obs on, diagnostics off."""
        return cls(
            runtime=RuntimeConfig(num_envs=int(num_envs)),
            robot=RobotConfig(fix_to_apple=True, force_batched_layout=True),
            controller=ControllerConfig(mode="vic"),
            settle_diagnostics=None,
            obs=ObsConfig(),
        )

    @classmethod
    def test_minimal(cls, *, num_envs: int = 2) -> BatchedHeterogeneousCoupledSimConfig:
        """CPU-friendly tests: short settle, straight-rod fixture, kinematic direct control."""
        return cls(
            runtime=RuntimeConfig(num_envs=int(num_envs), device="cpu"),
            scene=SceneSettleCollisionConfig(settle_substeps=50),
            domain_randomization=DomainRandomizationConfig(ranges_path=_TESTS_RANGES_FIXTURE),
            controller=ControllerConfig(mode="direct"),
            settle_diagnostics=None,
        )
