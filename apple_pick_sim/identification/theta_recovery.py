"""1D primary bend-stiffness recovery via Gauss–Newton and mega-plant FD columns."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np

from apple_pick_sim.coupled_fruiting.settle_then_weld import (
    seed_fix_to_apple_from_settled,
)
from apple_pick_sim.fruiting_system import (
    FruitingSystemParams,
    GripperProxyConfig,
    MegaCoupledCableScene,
    example_collision_pipeline,
    perturb_rod_stiffness,
    set_rod_bend_stiffness,
)
from apple_pick_sim.fruiting_system.mega_fd import (
    default_mega_fd_features,
    mega_fd_step,
    mega_vbd_substep,
    reset_perturbed_instances_to_nominal,
)

DEFAULT_DAMPING = 1e-6
DEFAULT_MIN_J_NORM = 0.05
WRENCH_FEATURE_SLICE = slice(6, 12)


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    """Observation slice and settle-then-weld settings for recovery."""

    fix_to_apple: bool
    feature_slice: slice
    warmup_substeps: int

    @classmethod
    def from_fix_to_apple(
        cls,
        fix_to_apple: bool,
        *,
        warmup_substeps: int | None = None,
    ) -> FeatureConfig:
        if fix_to_apple:
            return cls(
                fix_to_apple=True,
                feature_slice=WRENCH_FEATURE_SLICE,
                warmup_substeps=int(warmup_substeps if warmup_substeps is not None else 300),
            )
        return cls(
            fix_to_apple=False,
            feature_slice=slice(None),
            warmup_substeps=0,
        )


@dataclasses.dataclass(frozen=True)
class ThetaRecoveryResult:
    """Gauss–Newton recovery trace for one primary bend-stiffness target."""

    k_star: float
    y_star: np.ndarray
    k_hist: tuple[float, ...]
    loss_hist: tuple[float, ...]
    k_final: float
    rel_err: float
    feature_cfg: FeatureConfig


def primary_bend_bounds(ranges: dict[str, Any]) -> tuple[float, float]:
    """Inclusive ``(min, max)`` for ``primary.bend_stiffness`` from a ranges JSON dict."""
    block = ranges["primary"]["bend_stiffness"]
    return float(block["min"]), float(block["max"])


def _default_fd_kw(
    *,
    base_pos: tuple[float, float, float],
    instance_spacing: tuple[float, float, float],
    no_self_collision_kw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "base_pos": base_pos,
        "instance_spacing": instance_spacing,
        **no_self_collision_kw,
    }


def build_fd_mega(
    k: float,
    base_params: FruitingSystemParams,
    epsilon: float,
    *,
    gripper_proxy: GripperProxyConfig,
    fd_kw: dict[str, Any],
) -> MegaCoupledCableScene:
    """Two-column mega plant at ``k`` and ``k + epsilon`` on primary bend."""
    pk = set_rod_bend_stiffness(base_params, "primary", k)
    ppert = perturb_rod_stiffness(pk, "primary", bend_delta=epsilon)
    return MegaCoupledCableScene.build(
        [pk, ppert],
        gripper_proxy=gripper_proxy,
        **fd_kw,
    )


def _settle_free_mega(
    params: FruitingSystemParams,
    *,
    fd_kw: dict[str, Any],
    warmup_substeps: int,
    dt: float,
) -> tuple[MegaCoupledCableScene, Any]:
    """Free-proxy mega (one instance) advanced through ``warmup_substeps``."""
    gripper = GripperProxyConfig(fix_to_apple=False)
    mega = MegaCoupledCableScene.build([params], gripper_proxy=gripper, **fd_kw)
    pipe = example_collision_pipeline(mega.model)
    for _ in range(int(warmup_substeps)):
        mega_vbd_substep(mega, dt, collision_pipeline=pipe)
    return mega, pipe


def _seed_welded_mega_from_settled(
    welded: MegaCoupledCableScene,
    settled: MegaCoupledCableScene,
) -> None:
    """Copy settled free-apple state onto welded instance 0 and sync FD columns."""
    welded_view = welded.as_single_instance_coupled(0)
    settled_view = settled.as_single_instance_coupled(0)
    seed_fix_to_apple_from_settled(
        welded_scene=SimpleNamespace(
            cable=welded_view,
            robot_model=None,
            robot_state_0=None,
            mj_solver=None,
        ),
        settled_scene=SimpleNamespace(cable=settled_view),
        quiet_apple_proxy=True,
    )
    reset_perturbed_instances_to_nominal(welded)


def rollout_features(
    mega: MegaCoupledCableScene,
    n_substeps: int,
    dt: float,
    collision_pipeline: Any,
    feature_cfg: FeatureConfig,
) -> np.ndarray:
    """Advance ``mega`` and return sliced observation features for instance 0."""
    for _ in range(int(n_substeps)):
        mega_vbd_substep(mega, dt, collision_pipeline=collision_pipeline)
    full = default_mega_fd_features(mega, 0, dt=dt)
    return full[feature_cfg.feature_slice].astype(np.float64, copy=False)


def fd_jacobian_column(
    mega: MegaCoupledCableScene,
    epsilon: float,
    dt: float,
    collision_pipeline: Any,
    feature_cfg: FeatureConfig,
) -> np.ndarray:
    """One batched FD substep from the current state; return ∂y/∂k column."""
    reset_perturbed_instances_to_nominal(mega)
    result = mega_fd_step(
        mega,
        epsilon,
        dt=dt,
        collision_pipeline=collision_pipeline,
    )
    return result.jacobian[feature_cfg.feature_slice, 0].astype(np.float64, copy=False)


def evaluate_at_k(
    k: float,
    base_params: FruitingSystemParams,
    epsilon: float,
    *,
    n_substeps: int,
    dt: float,
    fd_kw: dict[str, Any],
    feature_cfg: FeatureConfig,
    settled_reference: MegaCoupledCableScene | None = None,
) -> tuple[np.ndarray, np.ndarray, MegaCoupledCableScene, Any]:
    """Roll out at ``k``, form ``y`` and forward-difference column ``J``."""
    pk = set_rod_bend_stiffness(base_params, "primary", k)

    if feature_cfg.fix_to_apple:
        if settled_reference is None:
            settled, _ = _settle_free_mega(
                pk,
                fd_kw=fd_kw,
                warmup_substeps=feature_cfg.warmup_substeps,
                dt=dt,
            )
        else:
            settled = settled_reference
        gripper = GripperProxyConfig(fix_to_apple=True)
        mega = build_fd_mega(
            k,
            base_params,
            epsilon,
            gripper_proxy=gripper,
            fd_kw=fd_kw,
        )
        _seed_welded_mega_from_settled(mega, settled)
    else:
        gripper = GripperProxyConfig(fix_to_apple=False)
        mega = build_fd_mega(
            k,
            base_params,
            epsilon,
            gripper_proxy=gripper,
            fd_kw=fd_kw,
        )

    pipe = example_collision_pipeline(mega.model)
    y = rollout_features(mega, n_substeps, dt, pipe, feature_cfg)
    j_col = fd_jacobian_column(mega, epsilon, dt, pipe, feature_cfg)
    return y, j_col, mega, pipe


def gauss_newton_step_1d(
    k: float,
    y: np.ndarray,
    y_star: np.ndarray,
    j_col: np.ndarray,
    *,
    k_min: float,
    k_max: float,
    damping: float = DEFAULT_DAMPING,
) -> float:
    """Scalar Gauss–Newton proposal for primary ``bend_stiffness``."""
    residual = y - y_star
    jtj = float(j_col @ j_col)
    if jtj + damping <= 0.0:
        return float(k)
    delta_k = -float(j_col @ residual) / (jtj + damping)
    return float(np.clip(k + delta_k, k_min, k_max))


def gauss_newton_step_1d_backtracking(
    k: float,
    y: np.ndarray,
    y_star: np.ndarray,
    j_col: np.ndarray,
    *,
    k_min: float,
    k_max: float,
    evaluate_y: Any,
    damping: float = DEFAULT_DAMPING,
    alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
) -> tuple[float, np.ndarray]:
    """Gauss–Newton with backtracking on loss ``||y(k)-y*||``."""
    loss0 = _loss(y, y_star)
    jtj = float(j_col @ j_col)
    if jtj + damping <= 0.0:
        return float(k), y
    delta_k = -float(j_col @ (y - y_star)) / (jtj + damping)
    best_k = float(k)
    best_y = y
    best_loss = loss0
    for alpha in alphas:
        k_try = float(np.clip(k + alpha * delta_k, k_min, k_max))
        if k_try == k:
            continue
        y_try = evaluate_y(k_try)
        loss_try = _loss(y_try, y_star)
        if loss_try < best_loss:
            best_k = k_try
            best_y = y_try
            best_loss = loss_try
            break
    return best_k, best_y


def _loss(y: np.ndarray, y_star: np.ndarray) -> float:
    r = y - y_star
    return float(np.linalg.norm(r))


def compute_y_star(
    k_star: float,
    base_params: FruitingSystemParams,
    *,
    n_substeps: int,
    dt: float,
    fd_kw: dict[str, Any],
    feature_cfg: FeatureConfig,
) -> np.ndarray:
    """Target features at ``k_star``."""
    pk = set_rod_bend_stiffness(base_params, "primary", k_star)

    if feature_cfg.fix_to_apple:
        settled, pipe = _settle_free_mega(
            pk,
            fd_kw=fd_kw,
            warmup_substeps=feature_cfg.warmup_substeps,
            dt=dt,
        )
        gripper = GripperProxyConfig(fix_to_apple=True)
        mega = MegaCoupledCableScene.build([pk], gripper_proxy=gripper, **fd_kw)
        _seed_welded_mega_from_settled(mega, settled)
    else:
        gripper = GripperProxyConfig(fix_to_apple=False)
        mega = MegaCoupledCableScene.build([pk], gripper_proxy=gripper, **fd_kw)
        pipe = example_collision_pipeline(mega.model)

    return rollout_features(mega, n_substeps, dt, pipe, feature_cfg)


def recover_primary_bend_stiffness(
    base_params: FruitingSystemParams,
    ranges: dict[str, Any],
    *,
    k_star: float | None = None,
    k0: float | None = None,
    k0_scale: float = 0.7,
    epsilon: float = 0.02,
    n_substeps: int = 90,
    dt: float = 1.0 / 1800.0,
    max_iter: int = 10,
    loss_tol: float = 1e-8,
    fd_kw: dict[str, Any] | None = None,
    feature_cfg: FeatureConfig | None = None,
    min_j_norm: float = DEFAULT_MIN_J_NORM,
    base_pos: tuple[float, float, float] = (0.5, 0.5, 0.5),
    instance_spacing: tuple[float, float, float] = (0.0, 1.5, 0.0),
    no_self_collision_kw: dict[str, Any] | None = None,
) -> ThetaRecoveryResult:
    """Recover ``primary.bend_stiffness`` from ``y_star`` via Gauss–Newton + FD."""
    if fd_kw is None:
        fd_kw = _default_fd_kw(
            base_pos=base_pos,
            instance_spacing=instance_spacing,
            no_self_collision_kw=no_self_collision_kw or {"enable_self_collisions": False},
        )
    if feature_cfg is None:
        feature_cfg = FeatureConfig.from_fix_to_apple(False)

    if k_star is None:
        assert base_params.primary is not None
        k_star = float(base_params.primary.bend_stiffness)
    if k0 is None:
        k0 = float(k0_scale * k_star)

    k_min, k_max = primary_bend_bounds(ranges)
    y_star = compute_y_star(
        k_star,
        base_params,
        n_substeps=n_substeps,
        dt=dt,
        fd_kw=fd_kw,
        feature_cfg=feature_cfg,
    )

    k = float(k0)
    k_hist: list[float] = [k]
    loss_hist: list[float] = []

    def _y_at_k(k_val: float) -> np.ndarray:
        y_val, _, _, _ = evaluate_at_k(
            k_val,
            base_params,
            epsilon,
            n_substeps=n_substeps,
            dt=dt,
            fd_kw=fd_kw,
            feature_cfg=feature_cfg,
        )
        return y_val

    for _ in range(int(max_iter)):
        y, j_col, _, _ = evaluate_at_k(
            k,
            base_params,
            epsilon,
            n_substeps=n_substeps,
            dt=dt,
            fd_kw=fd_kw,
            feature_cfg=feature_cfg,
        )
        if feature_cfg.fix_to_apple and float(np.linalg.norm(j_col)) < min_j_norm:
            raise ValueError(
                f"welded FD Jacobian norm {float(np.linalg.norm(j_col)):.6e} "
                f"< min_j_norm={min_j_norm}; wrench features may be insensitive"
            )
        loss = _loss(y, y_star)
        loss_hist.append(loss)
        if len(loss_hist) >= 2 and abs(loss_hist[-2] - loss_hist[-1]) < loss_tol:
            break
        k, y = gauss_newton_step_1d_backtracking(
            k,
            y,
            y_star,
            j_col,
            k_min=k_min,
            k_max=k_max,
            evaluate_y=_y_at_k,
        )
        k_hist.append(k)

    k_final = k_hist[-1]
    rel_err = abs(k_final - k_star) / max(abs(k_star), 1e-12)
    return ThetaRecoveryResult(
        k_star=k_star,
        y_star=y_star,
        k_hist=tuple(k_hist),
        loss_hist=tuple(loss_hist),
        k_final=k_final,
        rel_err=rel_err,
        feature_cfg=feature_cfg,
    )


def brute_force_grid_loss(
    base_params: FruitingSystemParams,
    y_star: np.ndarray,
    k_star: float,
    *,
    k_grid: np.ndarray,
    epsilon: float,
    n_substeps: int,
    dt: float,
    fd_kw: dict[str, Any],
    feature_cfg: FeatureConfig,
) -> tuple[float, float]:
    """Return ``(k_best, loss_min)`` over a 1D grid of ``k`` values."""
    best_k = float(k_grid[0])
    best_loss = float("inf")
    for k in k_grid:
        y, _, _, _ = evaluate_at_k(
            float(k),
            base_params,
            epsilon,
            n_substeps=n_substeps,
            dt=dt,
            fd_kw=fd_kw,
            feature_cfg=feature_cfg,
        )
        loss = _loss(y, y_star)
        if loss < best_loss:
            best_loss = loss
            best_k = float(k)
    return best_k, best_loss
