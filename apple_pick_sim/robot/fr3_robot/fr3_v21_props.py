"""Official FR3 v2.1 inertial and dynamics properties from vendored YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import newton
from newton.solvers import SolverMuJoCo

_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "franka_fr3v2_1"
INERTIALS_YAML = _FIXTURES_DIR / "inertials.yaml"
DYNAMICS_YAML = _FIXTURES_DIR / "dynamics.yaml"

_NUM_ARM_LINKS = 8  # link0 … link7


@dataclass(frozen=True)
class Fr3LinkInertial:
    link_num: int
    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_kgm2: np.ndarray  # 3×3 symmetric, body frame


@dataclass(frozen=True)
class Fr3V21Dynamics:
    reflected_motor_inertia_kgm2: tuple[float, ...]
    mu_viscous: float


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def _inertia_matrix_from_yaml(entry: dict[str, Any]) -> np.ndarray:
    inertia = entry["inertia"]
    return np.array(
        [
            [inertia["xx"], inertia["xy"], inertia["xz"]],
            [inertia["xy"], inertia["yy"], inertia["yz"]],
            [inertia["xz"], inertia["yz"], inertia["zz"]],
        ],
        dtype=np.float64,
    )


def _parse_xyz(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = value.split()
        if len(parts) != 3:
            raise ValueError(f"expected xyz string with 3 tokens, got {value!r}")
        return tuple(float(p) for p in parts)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(v) for v in value)
    raise TypeError(f"unsupported xyz value type: {type(value)!r}")


def parse_fr3_v21_inertials(data: dict[str, Any] | None = None) -> tuple[Fr3LinkInertial, ...]:
    """Parse link0–7 mass/COM/inertia from ``inertials.yaml``."""
    raw = data if data is not None else _load_yaml(INERTIALS_YAML)
    links: list[Fr3LinkInertial] = []
    for n in range(_NUM_ARM_LINKS):
        key = f"link{n}"
        entry = raw[key]
        com = _parse_xyz(entry["origin"]["xyz"])
        links.append(
            Fr3LinkInertial(
                link_num=n,
                mass_kg=float(entry["mass"]),
                com_m=com,
                inertia_kgm2=_inertia_matrix_from_yaml(entry),
            )
        )
    return tuple(links)


def parse_fr3_v21_dynamics(data: dict[str, Any] | None = None) -> Fr3V21Dynamics:
    """Parse reflected motor inertia and ``mu_viscous`` from ``dynamics.yaml``."""
    raw = data if data is not None else _load_yaml(DYNAMICS_YAML)
    armature: list[float] = []
    mu_values: list[float] = []
    for j in range(1, 8):
        dyn = raw[f"joint{j}"]["dynamic"]
        motor = float(dyn["motor_inertia"])
        gear = float(dyn["gear_ratio"])
        armature.append(motor * gear * gear)
        mu_values.append(float(dyn["mu_viscous"]))
    if len(set(mu_values)) != 1:
        raise ValueError(f"expected uniform mu_viscous across joints, got {mu_values}")
    return Fr3V21Dynamics(
        reflected_motor_inertia_kgm2=tuple(armature),
        mu_viscous=mu_values[0],
    )


@lru_cache(maxsize=1)
def load_fr3_v21_dynamics() -> Fr3V21Dynamics:
    return parse_fr3_v21_dynamics()


@lru_cache(maxsize=1)
def load_fr3_v21_inertials() -> tuple[Fr3LinkInertial, ...]:
    return parse_fr3_v21_inertials()


def resolve_fr3_link_body_index(
    model: newton.Model,
    link_num: int,
    *,
    template_only: bool = True,
) -> int:
    """Return Newton body index for ``fr3_link{link_num}`` (world-0 / template)."""
    needle = f"fr3_link{int(link_num)}"
    hits = [
        i
        for i, lbl in enumerate(model.body_label)
        if lbl.split("/")[-1] == needle or lbl.endswith(f"/{needle}")
    ]
    if not hits:
        raise ValueError(f"ambiguous or missing {needle} in body_label (0 hits)")
    if template_only:
        return int(min(hits))
    if len(hits) != 1:
        raise ValueError(f"ambiguous or missing {needle} in body_label ({len(hits)} hits)")
    return int(hits[0])


def _tile_body_indices(
    template_indices: dict[int, int],
    *,
    robot_bodies_per_world: int | None,
    num_envs: int | None,
) -> dict[int, list[int]]:
    """Map link_num → global body indices (one per env when batched)."""
    if robot_bodies_per_world is None or num_envs is None or int(num_envs) <= 1:
        return {n: [idx] for n, idx in template_indices.items()}
    stride = int(robot_bodies_per_world)
    n_env = int(num_envs)
    out: dict[int, list[int]] = {}
    for link_num, tpl_idx in template_indices.items():
        local = int(tpl_idx) % stride
        out[link_num] = [w * stride + local for w in range(n_env)]
    return out


def _set_body_inertial_full(
    model: newton.Model,
    body_index: int,
    *,
    mass_kg: float,
    com_m: tuple[float, float, float],
    inertia: np.ndarray,
) -> None:
    idx = int(body_index)
    m = float(mass_kg)
    com = np.asarray(com_m, dtype=np.float32).reshape(3)
    I = np.asarray(inertia, dtype=np.float32).reshape(3, 3)

    mass_np = model.body_mass.numpy().copy()
    inv_mass_np = model.body_inv_mass.numpy().copy()
    inertia_np = model.body_inertia.numpy().copy()
    inv_inertia_np = model.body_inv_inertia.numpy().copy()
    com_np = model.body_com.numpy().copy()

    mass_np[idx] = m
    inv_mass_np[idx] = (1.0 / m) if m > 0.0 else 0.0
    inertia_np[idx] = I
    if m > 0.0 and float(np.linalg.norm(I)) > 0.0:
        inv_inertia_np[idx] = np.linalg.inv(I).astype(np.float32)
    else:
        inv_inertia_np[idx] = np.zeros((3, 3), dtype=np.float32)
    com_np[idx] = com

    model.body_mass.assign(mass_np)
    model.body_inv_mass.assign(inv_mass_np)
    model.body_inertia.assign(inertia_np)
    model.body_inv_inertia.assign(inv_inertia_np)
    model.body_com.assign(com_np)


def apply_fr3_v21_link_inertials(
    robot_model: newton.Model,
    *,
    robot_bodies_per_world: int | None = None,
    num_envs: int | None = None,
) -> None:
    """Overwrite link0–7 mass/COM/inertia from official YAML (tiles per robot world)."""
    links = load_fr3_v21_inertials()
    template_indices = {link.link_num: resolve_fr3_link_body_index(robot_model, link.link_num) for link in links}
    tiled = _tile_body_indices(
        template_indices,
        robot_bodies_per_world=robot_bodies_per_world,
        num_envs=num_envs,
    )
    for link in links:
        for body_idx in tiled[link.link_num]:
            _set_body_inertial_full(
                robot_model,
                body_idx,
                mass_kg=link.mass_kg,
                com_m=link.com_m,
                inertia=link.inertia_kgm2,
            )


def apply_fr3_v21_arm_properties(
    robot_model: newton.Model,
    mj_solver: SolverMuJoCo,
    *,
    robot_bodies_per_world: int | None = None,
    num_envs: int | None = None,
) -> None:
    """Apply link inertials and notify MuJoCo of body inertial updates."""
    apply_fr3_v21_link_inertials(
        robot_model,
        robot_bodies_per_world=robot_bodies_per_world,
        num_envs=num_envs,
    )
    mj_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
