"""Build an RL harvest ranges fixture whose materials AND geometry come from real sys-ID data.

Materials: CMA final means over the fitted ``tmp/cma_s*_val03_seed*`` runs (log10 mu +- k sigma for
moduli and support kp/roll_kp -- the CMA search encoding; linear for zetas and primary density).
Geometry: per-structure real params from the converted ``tmp/real_batched_sNN`` datasets of the
same structures, linear mu +- k sigma; ``args.fruiting_base_pos`` is the real mean.

    uv run python apple_pick_gym/batched_examples/make_rl_harvest_real_fixture.py \\
        --sysid-root ../apple_pick_sim-dynamic-apple \\
        --base-fixture apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_variance.json \\
        --k-geom 0.5 --k-mat 1.0 --robot-base-pos 0 0.2 0 \\
        --out apple_pick_sim/fixtures/fruiting_system_ranges_rl_harvest_real_g05_m1.json
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
    mat = {}
    for i, name in enumerate(PHENO):
        col = final[:, i]
        if name in LOG_KNOBS:
            mat[name] = rng_log(col, a.k_mat)
        elif name == "primary_density":
            mat[name] = rng_lin(col, a.k_mat, lo_clip=1.0)
        else:
            mat[name] = rng_lin(col, a.k_mat, lo_clip=0.0)

    # -- geometry from the same structures' converted real datasets -----------------------
    rows, bases = [], []
    for s in sorted(structs):
        f = sorted(glob.glob(str(root / f"tmp/real_batched_s{s}/episodes/*.parquet")))[0]
        md = {k.decode(): v.decode() for k, v in pq.read_schema(f).metadata.items()}
        fs = json.loads(json.loads(md["fruiting_system_params"]))
        rows.append(fs)
        bases.append(json.loads(md["fruiting_base_pos"]))

    def col(path):
        out = []
        for r in rows:
            v = r
            for k in path:
                v = v[k]
            out.append(v)
        return out

    stem_dirs = np.asarray(col(("stem", "direction")))
    stem_el = np.degrees(np.arcsin(stem_dirs[:, 2])) + 90.0
    stem_lat = np.degrees(np.arctan2(stem_dirs[:, 1], stem_dirs[:, 0])) - 90.0
    kg = a.k_geom

    fx = json.loads(Path(a.base_fixture).read_text())
    out = copy.deepcopy(fx)
    out["_comment"] = (
        f"RL harvest DR fixture generated from real sys-ID data ({len(final)} CMA runs over structures "
        f"{sorted(structs)}). Materials: CMA final-mean mu +- {a.k_mat} sigma (log10 for moduli and "
        f"support kp/roll_kp, linear for zetas and primary density). Geometry: per-structure real "
        f"params, mu +- {kg} sigma (stem angles: {a.stem_angles}). Invariant quantities are pinned. "
        f"robot_base_pos: {out['args'].get('robot_base_pos')}."
    )
    out["args"]["fruiting_base_pos"] = [float(x) for x in np.mean(bases, axis=0)]
    if a.robot_base_pos is not None:
        out["args"]["robot_base_pos"] = [float(x) for x in a.robot_base_pos]

    pr = out["primary"]
    pr["length"] = rng_lin(col(("primary", "length")), kg)
    pr["radius"] = rng_lin(col(("primary", "radius")), kg)
    pr["density"] = mat["primary_density"]

    sp = out["spur"]
    sp["length"] = rng_lin(col(("spur", "length")), kg, lo_clip=1e-3)
    sp["radius"] = rng_lin(col(("spur", "radius")), kg, lo_clip=5e-4)
    sp["density"] = rng_lin(col(("spur", "density")), kg, lo_clip=1.0)
    sp["elevation_delta_deg"] = {"min": -90.0, "max": -90.0}
    sp["lateral_delta_deg"] = {"min": 90.0, "max": 90.0}
    sp["youngs_modulus_pa"] = mat["E_youngs_spur"]
    sp["flexural_modulus_pa"] = mat["E_flex_spur"]
    sp["damping_ratio"] = mat["spur_damping_ratio"]

    st = out["stem"]
    st["length"] = rng_lin(col(("stem", "length")), kg, lo_clip=1e-3)
    st["radius"] = rng_lin(col(("stem", "radius")), kg, lo_clip=2e-4)
    st["density"] = rng_lin(col(("stem", "density")), kg, lo_clip=1.0)
    if a.stem_angles == "uniform":
        st["elevation_delta_deg"] = {"min": float(stem_el.min()), "max": float(stem_el.max())}
        st["lateral_delta_deg"] = {"min": float(stem_lat.min()), "max": float(stem_lat.max())}
    else:
        st["elevation_delta_deg"] = rng_lin(stem_el, kg, lo_clip=0.0, hi_clip=90.0)
        st["lateral_delta_deg"] = rng_lin(stem_lat, kg)
    st["youngs_modulus_pa"] = mat["E_youngs_stem"]
    st["flexural_modulus_pa"] = mat["E_flex_stem"]
    st["damping_ratio"] = mat["stem_damping_ratio"]

    out["apple"]["radius"] = rng_lin([r["apple_radius"] for r in rows], kg, lo_clip=0.01)
    out["apple"]["density"] = rng_lin([r["apple_density"] for r in rows], kg, lo_clip=1.0)

    sdr = out["sim_build"]["support_dr"]
    sdr["kp"] = mat["support_kp"]
    sdr["roll_kp"] = mat["support_roll_kp"]
    sdr["zeta"] = mat["support_joint_zeta"]

    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    show = {k: out[k] for k in ("args", "primary", "spur", "stem", "apple")}
    show["support_dr"] = sdr
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
