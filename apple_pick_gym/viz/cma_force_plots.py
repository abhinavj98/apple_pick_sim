"""Real vs sim force plots from persisted CMA generation bags."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT

_HOLD = int(PHASE_TO_INT["hold"])
_FORCE_LABELS = ("Fx", "Fy", "Fz")
_TORQUE_LABELS = ("Tx", "Ty", "Tz")
_COLORS_REAL = ("#2563eb", "#1d4ed8", "#1e40af")
_COLORS_SIM = ("#dc2626", "#b91c1c", "#991b1b")
_DEFAULT_SIM_LPF_HZ = 5.0
_FORCE_NORM_KEY = "per_direction_mean_hold_force_norm_n"
_TORQUE_NORM_KEY = "per_direction_mean_hold_torque_norm_nm"


def list_persisted_generations(structure_dir: Path) -> list[int]:
    """Return sorted generation indices that have a ``best`` (or any) role bag."""
    gens_root = Path(structure_dir) / "generations"
    if not gens_root.is_dir():
        return []
    out: list[int] = []
    for path in gens_root.glob("gen_*"):
        try:
            out.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def pull_directions_from_manifest(manifest_path: Path) -> dict[int, tuple[float, float, float]]:
    """Map ``direction_idx`` to ``pull_direction`` XYZ from a dataset manifest."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    episodes = payload.get("episodes") or []
    out: dict[int, tuple[float, float, float]] = {}
    for ep in episodes:
        if not isinstance(ep, Mapping):
            continue
        if "direction_idx" not in ep or "pull_direction" not in ep:
            continue
        vec = np.asarray(ep["pull_direction"], dtype=np.float64).reshape(-1)
        if vec.size < 3:
            continue
        out[int(ep["direction_idx"])] = (float(vec[0]), float(vec[1]), float(vec[2]))
    return out


def _fmt_pull(pull: Mapping[int, tuple[float, float, float]], direction: int) -> str:
    vec = pull.get(int(direction))
    if vec is None:
        return ""
    return f"[{vec[0]:+.2f}, {vec[1]:+.2f}, {vec[2]:+.2f}]"


def _hold_spans(time: np.ndarray, phase: np.ndarray) -> list[tuple[float, float]]:
    t = np.asarray(time, dtype=np.float64)
    p = np.asarray(phase, dtype=np.int8)
    spans: list[tuple[float, float]] = []
    i = 0
    n = int(t.size)
    while i < n:
        if int(p[i]) != _HOLD:
            i += 1
            continue
        j = i
        while j < n and int(p[j]) == _HOLD:
            j += 1
        spans.append((float(t[i]), float(t[j - 1])))
        i = j
    return spans


def _n_directions_in_role(role_dir: Path) -> int:
    return len(sorted(role_dir.glob("dir_*.npz")))


def _sample_hz_from_time(time: np.ndarray) -> float | None:
    t = np.asarray(time, dtype=np.float64).reshape(-1)
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def lowpass_sim_force(
    values: np.ndarray,
    time: np.ndarray,
    *,
    cutoff_hz: float | None,
) -> np.ndarray:
    """Zero-phase Butterworth on sim force; no-op when ``cutoff_hz`` is None/<=0."""
    x = np.asarray(values, dtype=np.float64)
    if cutoff_hz is None or float(cutoff_hz) <= 0.0:
        return x
    source_hz = _sample_hz_from_time(time)
    if source_hz is None:
        return x
    from apple_pick_sim.system_id.real_to_batched_sysid import zero_phase_lowpass

    return zero_phase_lowpass(x, source_hz=source_hz, cutoff_hz=float(cutoff_hz))


def _mean_hold_norm(
    meta: Mapping[str, Any],
    key: str,
    direction: int,
    rs: np.ndarray,
    ss: np.ndarray,
    phase_r: np.ndarray,
    *,
    col0: int,
) -> tuple[float, float]:
    """Return (real, sim) mean-hold vector-norm from metadata, else from the bag."""
    block = meta.get(key)
    entry = block.get(str(direction)) if isinstance(block, Mapping) else None
    if isinstance(entry, Mapping) and "real" in entry and "sim" in entry:
        return float(entry["real"]), float(entry["sim"])
    hold = np.asarray(phase_r, dtype=np.int8) == _HOLD
    if not np.any(hold):
        hold = np.ones(int(rs.shape[0]), dtype=bool)
    real = float(np.linalg.norm(rs[hold, col0 : col0 + 3], axis=1).mean())
    sim = float(np.linalg.norm(ss[hold, col0 : col0 + 3], axis=1).mean())
    return real, sim


def _pooled_sim_real_ratio(norms: Any) -> float:
    if not isinstance(norms, Mapping) or not norms:
        return float("nan")
    dirs = sorted(int(k) for k in norms)
    real = sum(float(norms[str(d)]["real"]) for d in dirs)
    sim = sum(float(norms[str(d)]["sim"]) for d in dirs)
    return sim / real if real > 1e-9 else float("nan")


def _load_dir_arrays(
    role_dir: Path,
    direction: int,
    *,
    sim_lpf_hz: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(Path(role_dir) / f"dir_{direction:02d}.npz")
    tr = np.asarray(z["sim_time_real"], dtype=np.float64)
    ts = np.asarray(z["sim_time_sim"], dtype=np.float64)
    rs = np.asarray(z["real_state"], dtype=np.float64)
    ss = np.asarray(z["sim_state"], dtype=np.float64)
    phase_r = np.asarray(z["phase_real"], dtype=np.int8)
    ss = lowpass_sim_force(ss, ts, cutoff_hz=sim_lpf_hz)
    return tr, ts, rs, ss, phase_r


_WRENCH_KINDS: tuple[tuple[str, tuple[str, str, str], int, str, str, str, str], ...] = (
    (
        "force",
        _FORCE_LABELS,
        0,
        "N",
        _FORCE_NORM_KEY,
        "|F|",
        "force",
    ),
    (
        "torque",
        _TORQUE_LABELS,
        3,
        "N·m",
        _TORQUE_NORM_KEY,
        "|τ|",
        "torque",
    ),
)


def write_generation_force_plots(
    role_dir: Path,
    out_dir: Path,
    *,
    gen: int,
    run_name: str,
    pull: Mapping[int, tuple[float, float, float]] | None = None,
    n_directions: int | None = None,
    write_html: bool = True,
    sim_lpf_hz: float | None = _DEFAULT_SIM_LPF_HZ,
) -> dict[str, Any]:
    """Write per-direction force and torque plots for one persisted role."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    role_dir = Path(role_dir)
    meta = json.loads((role_dir / "metadata.json").read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pull = dict(pull or {})
    n_dirs = int(n_directions) if n_directions is not None else _n_directions_in_role(role_dir)
    lpf_note = f" sim LPF {float(sim_lpf_hz):.0f} Hz" if sim_lpf_hz and float(sim_lpf_hz) > 0 else ""

    for stem, labels, col0, unit, norm_key, stacked_name, kind_word in _WRENCH_KINDS:
        for d in range(n_dirs):
            tr, ts, rs, ss, phase_r = _load_dir_arrays(role_dir, d, sim_lpf_hz=sim_lpf_hz)
            real_n, sim_n = _mean_hold_norm(meta, norm_key, d, rs, ss, phase_r, col0=col0)
            ratio = sim_n / real_n if real_n > 1e-9 else float("nan")
            pull_txt = _fmt_pull(pull, d)
            title_pull = f" pull {pull_txt}" if pull_txt else ""

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(
                f"gen {gen} best d{d}{title_pull}{lpf_note}  |  mean-hold {stacked_name}: "
                f"real={real_n:.2f}{unit} sim={sim_n:.2f}{unit} ({ratio:.2f}x)",
                fontsize=11,
            )
            for i, ax in enumerate(axes):
                ax.plot(
                    tr,
                    rs[:, col0 + i],
                    color=_COLORS_REAL[i],
                    lw=1.8,
                    label=f"real {labels[i]}",
                )
                ax.plot(
                    ts,
                    ss[:, col0 + i],
                    color=_COLORS_SIM[i],
                    lw=1.8,
                    ls="--",
                    label=f"sim {labels[i]}",
                )
                ax.set_ylabel(f"{labels[i]} [{unit}]")
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8, ncol=2)
            for t0, t1 in _hold_spans(tr, phase_r):
                for ax in axes:
                    ax.axvspan(t0, t1, color="#2563eb", alpha=0.06, lw=0)
            axes[-1].set_xlabel("time [s]")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(out_dir / f"dir_{d:02d}_{stem}.png", dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(n_dirs, 1, figsize=(10, max(3.0, 2.5 * n_dirs)), sharex=True)
        if n_dirs == 1:
            axes = [axes]
        for d in range(n_dirs):
            tr, ts, rs, ss, _phase_r = _load_dir_arrays(role_dir, d, sim_lpf_hz=sim_lpf_hz)
            ax = axes[d]
            ax.plot(
                tr,
                np.linalg.norm(rs[:, col0 : col0 + 3], axis=1),
                color="#2563eb",
                lw=1.5,
                label=f"real {stacked_name}",
            )
            ax.plot(
                ts,
                np.linalg.norm(ss[:, col0 : col0 + 3], axis=1),
                color="#dc2626",
                lw=1.5,
                ls="--",
                label=f"sim {stacked_name}",
            )
            ax.set_ylabel(f"{stacked_name} [{unit}]")
            pull_txt = _fmt_pull(pull, d)
            ax.set_title(f"d{d} {pull_txt}".rstrip(), fontsize=9, loc="left")
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(
            f"Real vs sim {kind_word} — gen {gen} best (Sinkhorn={meta['aggregate_sinkhorn']:.1f}, "
            f"cand {meta['candidate_index']}){lpf_note}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_dir / f"all_directions_{stem}.png", dpi=150)
        plt.close(fig)

        if write_html:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                for d in range(n_dirs):
                    tr, ts, rs, ss, phase_r = _load_dir_arrays(
                        role_dir, d, sim_lpf_hz=sim_lpf_hz
                    )
                    real_n, sim_n = _mean_hold_norm(
                        meta, norm_key, d, rs, ss, phase_r, col0=col0
                    )
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=list(labels),
                        vertical_spacing=0.06,
                    )
                    for i, lab in enumerate(labels):
                        fig.add_trace(
                            go.Scatter(
                                x=tr,
                                y=rs[:, col0 + i],
                                mode="lines",
                                name=f"real {lab}",
                                line=dict(color=_COLORS_REAL[i], width=2),
                            ),
                            row=i + 1,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=ts,
                                y=ss[:, col0 + i],
                                mode="lines",
                                name=f"sim {lab}",
                                line=dict(color=_COLORS_SIM[i], width=2, dash="dash"),
                                showlegend=False,
                            ),
                            row=i + 1,
                            col=1,
                        )
                        fig.update_yaxes(title_text=f"{lab} [{unit}]", row=i + 1, col=1)
                    pull_txt = _fmt_pull(pull, d)
                    fig.update_layout(
                        title=(
                            f"gen {gen} d{d}"
                            + (f" pull {pull_txt}" if pull_txt else "")
                            + (
                                f" — mean-hold {stacked_name} "
                                f"real={real_n:.2f}{unit} sim={sim_n:.2f}{unit}"
                            )
                            + lpf_note
                        ),
                        height=700,
                        width=950,
                        template="plotly_white",
                    )
                    fig.update_xaxes(title_text="time [s]", row=3, col=1)
                    fig.write_html(
                        str(out_dir / f"dir_{d:02d}_{stem}.html"), include_plotlyjs=True
                    )
            except ImportError:
                pass

    (out_dir / "README.md").write_text(
        (
            "# Force and torque time series — real vs sim\n\n"
            f"Run: `{run_name}`, gen {gen} best "
            f"(Sinkhorn={meta['aggregate_sinkhorn']:.2f}, cand {meta['candidate_index']})"
            f"{lpf_note}\n\n"
            "**PNG (open in editor):**\n"
            "- [all_directions_force.png](all_directions_force.png)\n"
            "- [all_directions_torque.png](all_directions_torque.png)\n"
            "- dir_00_force.png / dir_00_torque.png …\n\n"
            "**HTML (offline, no CDN):** dir_XX_force.html, dir_XX_torque.html\n"
        ),
        encoding="utf-8",
    )
    return meta


def write_run_force_plots(
    run_dir: Path,
    *,
    structure_idx: int = 0,
    role: str = "best",
    manifest_path: Path | None = None,
    n_directions: int | None = None,
    write_html: bool = True,
    sim_lpf_hz: float | None = _DEFAULT_SIM_LPF_HZ,
) -> Path:
    """Write ``structure_XXX/force_plots/gen_YY/`` for every persisted generation."""
    run_dir = Path(run_dir)
    struct = run_dir / f"structure_{int(structure_idx):03d}"
    plots_root = struct / "force_plots"
    gens = list_persisted_generations(struct)
    if not gens:
        raise FileNotFoundError(f"no persisted generations under {struct / 'generations'}")

    pull: dict[int, tuple[float, float, float]] = {}
    manifest = manifest_path
    if manifest is None:
        sibling = run_dir / "manifest.json"
        if sibling.is_file():
            manifest = sibling
    if manifest is not None:
        pull = pull_directions_from_manifest(manifest)

    index_lines = [
        "# Force and torque plots — all generations",
        "",
        f"Run: `{run_dir.name}`",
        "",
    ]
    if sim_lpf_hz and float(sim_lpf_hz) > 0:
        index_lines.append(f"Sim traces: {float(sim_lpf_hz):.0f} Hz zero-phase LPF (plot-only).")
        index_lines.append("")
    for gen in gens:
        role_dir = struct / f"generations/gen_{gen:02d}" / role
        if not (role_dir / "metadata.json").is_file():
            continue
        meta = write_generation_force_plots(
            role_dir,
            plots_root / f"gen_{gen:02d}",
            gen=gen,
            run_name=run_dir.name,
            pull=pull,
            n_directions=n_directions,
            write_html=write_html,
            sim_lpf_hz=sim_lpf_hz,
        )
        f_ratio = _pooled_sim_real_ratio(meta.get(_FORCE_NORM_KEY))
        t_ratio = _pooled_sim_real_ratio(meta.get(_TORQUE_NORM_KEY))
        tau_txt = f", |τ| sim/real={t_ratio:.2f}x" if t_ratio == t_ratio else ""
        index_lines.append(
            f"- [gen {gen:02d} force](gen_{gen:02d}/all_directions_force.png) · "
            f"[torque](gen_{gen:02d}/all_directions_torque.png) — "
            f"Sinkhorn={meta['aggregate_sinkhorn']:.1f}, |F| sim/real={f_ratio:.2f}x"
            f"{tau_txt}"
        )
    plots_root.mkdir(parents=True, exist_ok=True)
    (plots_root / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return plots_root


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot real vs sim Fx/Fy/Fz and Tx/Ty/Tz from persisted CMA generation bags."
    )
    p.add_argument(
        "--run",
        type=Path,
        required=True,
        help="CMA output directory (contains structure_XXX/generations).",
    )
    p.add_argument("--structure-idx", type=int, default=0)
    p.add_argument("--role", default="best", help="Persisted role folder (default: best).")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Dataset manifest.json for pull-direction titles.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip Plotly HTML (PNG + README only).",
    )
    p.add_argument(
        "--sim-lpf-hz",
        type=float,
        default=_DEFAULT_SIM_LPF_HZ,
        help="Zero-phase Butterworth cutoff on sim wrench traces (default: 5). 0 disables.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    write_run_force_plots(
        args.run,
        structure_idx=int(args.structure_idx),
        role=str(args.role),
        manifest_path=args.manifest,
        write_html=not bool(args.no_html),
        sim_lpf_hz=float(args.sim_lpf_hz),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
