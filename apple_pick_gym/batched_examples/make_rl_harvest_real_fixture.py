"""Build an RL harvest ranges fixture whose materials AND geometry come from real sys-ID data.

Materials: CMA final means over the fitted ``tmp/cma_s*_val03_seed*`` runs (log10 mu +- k sigma for
moduli and support kp/roll_kp -- the CMA search encoding; linear for zetas and primary density).
Spur/stem moduli are written with ``"scale": "log"`` so they are sampled log-uniform (support kp /
roll_kp already are, in ``support_joint_dr``). Geometry: per-structure real params from the converted
``tmp/real_batched_sNN`` datasets of the same structures, linear mu +- k sigma or (``--geom-mode
minmax``) their min..max; ``args.fruiting_base_pos`` is the real mean.

    uv run python apple_pick_gym/batched_examples/make_rl_harvest_real_fixture.py \\
        --sysid-root ../apple_pick_sim-dynamic-apple \\
        --base-fixture apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_variance.json \\
        --k-geom 0.5 --k-mat 1.0 --robot-base-pos 0 0.2 0 \\
        --out apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_real_g05_m1.json

    # "true" fixture: sys-ID 95% (mu +- 1.96 sigma), log-uniform moduli, real geometry min..max;
    # its spur / stem / apple / primary densities were then pinned to their means (min == max) by hand
    uv run python apple_pick_gym/batched_examples/make_rl_harvest_real_fixture.py \\
        --sysid-root ../apple_pick_sim-dynamic-apple \\
        --base-fixture apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_variance.json \\
        --k-mat 1.96 --geom-mode minmax --robot-base-pos 0 0.2 0 \\
        --out apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_true.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PHENO = [
    "support_kp", "E_flex_spur", "E_flex_stem", "E_youngs_spur", "E_youngs_stem",
    "support_roll_kp", "spur_damping_ratio", "stem_damping_ratio", "support_joint_zeta",
    "primary_density",
]
LOG_KNOBS = {"support_kp", "E_flex_spur", "E_flex_stem", "E_youngs_spur", "E_youngs_stem", "support_roll_kp"}


def rng_lin(x, k, lo_clip=None, hi_clip=None):
    x = np.asarray(x, float)
    m, s = x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)
    lo, hi = m - k * s, m + k * s
    if lo_clip is not None:
        lo = max(lo, lo_clip)
    if hi_clip is not None:
        hi = min(hi, hi_clip)
    return {"min": float(lo), "max": float(hi)}


def rng_log(x, k):
    lx = np.log10(np.asarray(x, float))
    m, s = lx.mean(), lx.std(ddof=1)
    return {"min": float(10 ** (m - k * s)), "max": float(10 ** (m + k * s))}


def rng_minmax(x, lo_clip=None, hi_clip=None):
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    if lo_clip is not None:
        lo = max(lo, lo_clip)
    if hi_clip is not None:
        hi = min(hi, hi_clip)
    return {"min": lo, "max": hi}


def log_scaled(r):
    """Mark a range for log-uniform sampling (``params.sample_params`` reads ``scale``)."""
    return {**r, "scale": "log"}


def build_fixture(final, structs, rows, bases, base_fixture, *, k_mat, k_geom, stem_angles, geom_mode, robot_base_pos):
    """The ranges fixture from CMA final means ``final`` (runs x PHENO) and the real structures' ``rows``.

    Spur / stem moduli are log10 mu +- k_mat sigma and marked ``"scale": "log"`` (log-uniform sampling).
    Support kp / roll_kp are log10 mu +- k_mat sigma too, but carry no ``scale`` key: ``support_joint_dr``
    always samples them log-uniform. Zetas and primary density are linear mu +- k_mat sigma. Geometry is
    linear mu +- k_geom sigma (``geom_mode="sigma"``) or the real min..max (``"minmax"``).
    """
    final = np.asarray(final, float)
    mat = {}
    for i, name in enumerate(PHENO):
        col_i = final[:, i]
        if name in LOG_KNOBS:
            mat[name] = rng_log(col_i, k_mat)
        elif name == "primary_density":
            mat[name] = rng_lin(col_i, k_mat, lo_clip=1.0)
        else:
            mat[name] = rng_lin(col_i, k_mat, lo_clip=0.0)

    def col(path):
        out = []
        for r in rows:
            v = r
            for k in path:
                v = v[k]
            out.append(v)
        return out

    if geom_mode == "minmax":
        geom = lambda x, lo_clip=None, hi_clip=None: rng_minmax(x, lo_clip, hi_clip)  # noqa: E731
        geom_desc = "real min..max"
    elif geom_mode == "sigma":
        geom = lambda x, lo_clip=None, hi_clip=None: rng_lin(x, k_geom, lo_clip, hi_clip)  # noqa: E731
        geom_desc = f"mu +- {k_geom} sigma"
    else:
        raise ValueError(f"geom_mode must be 'sigma' or 'minmax', got {geom_mode!r}")

    stem_dirs = np.asarray(col(("stem", "direction")), float)
    stem_dirs = stem_dirs / np.linalg.norm(stem_dirs, axis=1, keepdims=True)
    stem_el = np.degrees(np.arcsin(stem_dirs[:, 2])) + 90.0
    stem_lat = np.degrees(np.arctan2(stem_dirs[:, 1], stem_dirs[:, 0])) - 90.0

    out = copy.deepcopy(base_fixture)
    out["_comment"] = (
        f"RL harvest DR fixture generated from real sys-ID data ({len(final)} CMA runs over structures "
        f"{sorted(structs)}). Materials: CMA final-mean mu +- {k_mat} sigma (log10 and log-uniform sampling "
        f"for spur/stem moduli and support kp/roll_kp; linear for zetas and primary density). Geometry: "
        f"per-structure real params, {geom_desc}, linear (stem angles: {stem_angles}). Invariant quantities "
        f"are pinned. robot_base_pos: {out['args'].get('robot_base_pos')}."
    )
    out["args"]["fruiting_base_pos"] = [float(x) for x in np.mean(bases, axis=0)]
    if robot_base_pos is not None:
        out["args"]["robot_base_pos"] = [float(x) for x in robot_base_pos]

    pr = out["primary"]
    pr["length"] = geom(col(("primary", "length")))
    pr["radius"] = geom(col(("primary", "radius")))
    pr["density"] = mat["primary_density"]

    sp = out["spur"]
    sp["length"] = geom(col(("spur", "length")), lo_clip=1e-3)
    sp["radius"] = geom(col(("spur", "radius")), lo_clip=5e-4)
    sp["density"] = geom(col(("spur", "density")), lo_clip=1.0)
    sp["elevation_delta_deg"] = {"min": -90.0, "max": -90.0}
    sp["lateral_delta_deg"] = {"min": 90.0, "max": 90.0}
    sp["youngs_modulus_pa"] = log_scaled(mat["E_youngs_spur"])
    sp["flexural_modulus_pa"] = log_scaled(mat["E_flex_spur"])
    sp["damping_ratio"] = mat["spur_damping_ratio"]

    st = out["stem"]
    st["length"] = geom(col(("stem", "length")), lo_clip=1e-3)
    st["radius"] = geom(col(("stem", "radius")), lo_clip=2e-4)
    st["density"] = geom(col(("stem", "density")), lo_clip=1.0)
    if stem_angles == "uniform":
        st["elevation_delta_deg"] = {"min": float(stem_el.min()), "max": float(stem_el.max())}
        st["lateral_delta_deg"] = {"min": float(stem_lat.min()), "max": float(stem_lat.max())}
    else:
        st["elevation_delta_deg"] = rng_lin(stem_el, k_geom, lo_clip=0.0, hi_clip=90.0)
        st["lateral_delta_deg"] = rng_lin(stem_lat, k_geom)
    st["youngs_modulus_pa"] = log_scaled(mat["E_youngs_stem"])
    st["flexural_modulus_pa"] = log_scaled(mat["E_flex_stem"])
    st["damping_ratio"] = mat["stem_damping_ratio"]

    out["apple"]["radius"] = geom([r["apple_radius"] for r in rows], lo_clip=0.01)
    out["apple"]["density"] = geom([r["apple_density"] for r in rows], lo_clip=1.0)

    sdr = out["sim_build"]["support_dr"]
    sdr["kp"] = mat["support_kp"]
    sdr["roll_kp"] = mat["support_roll_kp"]
    sdr["zeta"] = mat["support_joint_zeta"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sysid-root", required=True, help="Checkout holding tmp/cma_s* and tmp/real_batched_s*.")
    p.add_argument("--base-fixture", required=True)
    p.add_argument("--k-geom", type=float, default=0.5)
    p.add_argument("--k-mat", type=float, default=1.0)
    p.add_argument(
        "--stem-angles",
        choices=("uniform", "sigma"),
        default="uniform",
        help="uniform: stem elevation/lateral deltas span the observed real min..max (the real "
        "stems form separate clusters, so mu +- k sigma would only sample between them); "
        "sigma: mu +- k-geom sigma like the other geometry.",
    )
    p.add_argument(
        "--geom-mode",
        choices=("sigma", "minmax"),
        default="sigma",
        help="sigma: geometry (lengths, radii, spur/stem/apple density, apple radius) is mu +- k-geom "
        "sigma over the real structures; minmax: their observed min..max. Sampled linearly either way.",
    )
    p.add_argument(
        "--robot-base-pos",
        type=float,
        nargs=3,
        default=None,
        help="Override args.robot_base_pos (FR3 root, world) -- e.g. move the robot toward the plant.",
    )
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.sysid_root)

    # -- materials from CMA final means ---------------------------------------------------
    final, structs = [], set()
    for d in sorted(glob.glob(str(root / "tmp/cma_s*_val03_seed*"))):
        rp = Path(d) / "cmaes_report.json"
        if not rp.exists():
            continue
        s = json.loads(rp.read_text())["structures"]["0"]
        if s.get("status") != "fitted":
            continue
        final.append(s["final_mean"]["e_pa"])
        structs.add(re.search(r"cma_s(\d+)_", d).group(1))
    final = np.asarray(final)

    # -- geometry from the same structures' converted real datasets -----------------------
    rows, bases = [], []
    for s in sorted(structs):
        f = sorted(glob.glob(str(root / f"tmp/real_batched_s{s}/episodes/*.parquet")))[0]
        md = {k.decode(): v.decode() for k, v in pq.read_schema(f).metadata.items()}
        fs = json.loads(json.loads(md["fruiting_system_params"]))
        rows.append(fs)
        bases.append(json.loads(md["fruiting_base_pos"]))

    out = build_fixture(
        final,
        structs,
        rows,
        bases,
        json.loads(Path(a.base_fixture).read_text()),
        k_mat=a.k_mat,
        k_geom=a.k_geom,
        stem_angles=a.stem_angles,
        geom_mode=a.geom_mode,
        robot_base_pos=a.robot_base_pos,
    )
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    show = {k: out[k] for k in ("args", "primary", "spur", "stem", "apple")}
    show["support_dr"] = out["sim_build"]["support_dr"]
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
