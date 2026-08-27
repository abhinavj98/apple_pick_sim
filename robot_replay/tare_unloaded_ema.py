#!/usr/bin/env python3
"""Subtract unloaded EMA F/T from loaded EMA F/T and plot the residual.

Loaded compiled bags store EMA-filtered F/T in both ``ft_wrist`` (already
raw-minus-unloaded-raw) and ``ft_wrist_raw``. The unloaded replay already
stores comm-loop EMA in ``ft_wrist``. This tool index-matches that column
and writes:

- ``ft_wrist_ema_baseline`` — matched unloaded EMA
- ``ft_wrist_ema_tared`` — loaded ``ft_wrist_raw`` minus that EMA

Example::

    uv run python robot_replay/tare_unloaded_ema.py \\
      --input robot_replay/new_data/s09/s09-d00.parquet \\
      --plot tmp/s09-d00_ema_tare.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_UNLOADED_EMA_COLUMN = "ft_wrist"
_LOADED_EMA_COLUMN = "ft_wrist_raw"


def _as_int_col(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.int32)


def _as_wrench_col(table: pa.Table, name: str) -> np.ndarray:
    return np.stack(
        [np.asarray(row, dtype=np.float64).reshape(6) for row in table.column(name).to_pylist()]
    )


def _data_rows(table: pa.Table) -> pa.Table:
    if "row_kind" not in table.column_names:
        return table
    kinds = [str(value) if value is not None else "data" for value in table.column("row_kind").to_pylist()]
    keep = [i for i, kind in enumerate(kinds) if kind != "metadata"]
    return table.take(keep)


def _match_baseline_frames(
    source_ft: np.ndarray,
    target_frame_count: int,
    *,
    max_relative_difference: float = 0.10,
) -> np.ndarray:
    source_ft = np.asarray(source_ft, dtype=np.float64)
    if source_ft.ndim != 2 or source_ft.shape[1] != 6 or len(source_ft) == 0:
        raise ValueError("Baseline wrench data must be a non-empty array of shape (n, 6)")
    if target_frame_count < 1:
        raise ValueError("Regular recording must contain at least one frame")

    relative_difference = abs(len(source_ft) - target_frame_count) / max(
        len(source_ft), target_frame_count
    )
    if relative_difference > max_relative_difference:
        raise ValueError(
            f"Baseline and regular frame counts differ by more than {max_relative_difference}: "
            f"baseline={len(source_ft)}, regular={target_frame_count}"
        )

    matched = source_ft[:target_frame_count]
    if len(matched) < target_frame_count:
        matched = np.pad(
            matched,
            ((0, target_frame_count - len(matched)), (0, 0)),
            mode="edge",
        )
    return matched


def match_baseline_wrench(
    loaded_hold: np.ndarray,
    loaded_phase: np.ndarray,
    loaded_step: np.ndarray,
    baseline_hold: np.ndarray,
    baseline_phase: np.ndarray,
    baseline_step: np.ndarray,
    baseline_ft: np.ndarray,
    *,
    max_relative_difference: float = 0.10,
) -> np.ndarray:
    """Copy compile's per-(hold, phase) index match onto loaded frames."""
    del loaded_step  # sort key lives on the baseline side; loaded rows stay in file order
    loaded_hold = np.asarray(loaded_hold, dtype=np.int32)
    loaded_phase = np.asarray(loaded_phase, dtype=np.int32)
    baseline_hold = np.asarray(baseline_hold, dtype=np.int32)
    baseline_phase = np.asarray(baseline_phase, dtype=np.int32)
    baseline_step = np.asarray(baseline_step, dtype=np.int32)
    baseline_ft = np.asarray(baseline_ft, dtype=np.float64)
    if baseline_ft.shape != (len(baseline_hold), 6):
        raise ValueError(f"baseline_ft shape {baseline_ft.shape} != ({len(baseline_hold)}, 6)")

    matched = np.zeros((len(loaded_hold), 6), dtype=np.float64)
    filled = np.zeros(len(loaded_hold), dtype=bool)
    loaded_idx = np.arange(len(loaded_hold))
    for hold_idx in np.unique(loaded_hold):
        phases = []
        for phase in loaded_phase[loaded_hold == hold_idx]:
            if int(phase) not in phases:
                phases.append(int(phase))
        for phase in phases:
            current_idx = loaded_idx[(loaded_hold == hold_idx) & (loaded_phase == phase)]
            source_mask = (baseline_hold == hold_idx) & (baseline_phase == phase)
            source_idx = np.where(source_mask)[0]
            if source_idx.size == 0:
                raise ValueError(
                    f"Baseline has no rows for hold_index={int(hold_idx)}, phase={int(phase)}"
                )
            order = np.argsort(baseline_step[source_idx], kind="stable")
            source_ft = baseline_ft[source_idx[order]]
            matched[current_idx] = _match_baseline_frames(
                source_ft,
                len(current_idx),
                max_relative_difference=max_relative_difference,
            )
            filled[current_idx] = True
    if not filled.all():
        missing = np.where(~filled)[0][:5]
        raise ValueError(f"Unmatched loaded frames, first indices={missing.tolist()}")
    return matched


def tared_ft_from_loaded_and_baseline(
    *,
    loaded_raw: np.ndarray,
    loaded_hold: np.ndarray,
    loaded_phase: np.ndarray,
    loaded_step: np.ndarray,
    baseline_ema: np.ndarray,
    baseline_raw: np.ndarray,
    baseline_hold: np.ndarray,
    baseline_phase: np.ndarray,
    baseline_step: np.ndarray,
    max_relative_difference: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(matched_unloaded_ema, loaded_ema - matched_unloaded_ema)``."""
    del baseline_raw  # intentionally unused: tare must not subtract unfiltered raw
    ema_baseline = match_baseline_wrench(
        loaded_hold,
        loaded_phase,
        loaded_step,
        baseline_hold,
        baseline_phase,
        baseline_step,
        baseline_ema,
        max_relative_difference=max_relative_difference,
    )
    loaded_raw = np.asarray(loaded_raw, dtype=np.float64)
    ema_tared = loaded_raw - ema_baseline
    return ema_baseline, ema_tared


def _read_metadata(table: pa.Table) -> dict[str, Any]:
    raw = table.schema.metadata or {}
    payload = raw.get(b"dataset_metadata")
    if payload is None:
        return {}
    return json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)


def resolve_baseline_path(episode_path: Path, metadata: dict[str, Any]) -> Path:
    recorded = str((metadata.get("dynamic_baseline") or {}).get("baseline_path") or "")
    if recorded:
        local = episode_path.parent / Path(recorded).name
        if local.is_file():
            return local
        remote = Path(recorded)
        if remote.is_file():
            return remote
    theta = metadata.get("theta_rad", (metadata.get("dump") or {}).get("theta_rad"))
    phi = metadata.get("phi_rad", (metadata.get("dump") or {}).get("phi_rad"))
    kp = (
        ((metadata.get("dump") or {}).get("robot_info") or {}).get("kp")
        or ((metadata.get("dump") or {}).get("controller_gains") or {}).get("task_prop_gains")
    )
    candidates = sorted(episode_path.parent.glob("*baseline_robot.parquet"))
    if theta is not None and phi is not None:
        token = f"theta{float(theta):.2f}_phi{float(phi):.2f}"
        typed = [path for path in candidates if token in path.name]
        if len(typed) == 1:
            return typed[0]
        if kp is not None and not isinstance(kp, list):
            kp_token = f"kp{int(round(float(kp)))}"
            typed_kp = [path for path in typed if kp_token in path.name]
            if len(typed_kp) == 1:
                return typed_kp[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not resolve unloaded baseline next to {episode_path}. "
        "Pass --baseline PATH."
    )


def add_ema_tare_columns(
    episode_path: Path | str,
    baseline_path: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    episode_path = Path(episode_path)
    baseline_path = Path(baseline_path)
    output_path = Path(output_path) if output_path is not None else episode_path

    loaded = pq.read_table(episode_path)
    baseline = _data_rows(pq.read_table(baseline_path))
    if _LOADED_EMA_COLUMN not in loaded.column_names:
        raise ValueError(f"{episode_path} is missing {_LOADED_EMA_COLUMN}")
    if _UNLOADED_EMA_COLUMN not in baseline.column_names:
        raise ValueError(f"{baseline_path} is missing {_UNLOADED_EMA_COLUMN}")

    ema_baseline, ema_tared = tared_ft_from_loaded_and_baseline(
        loaded_raw=_as_wrench_col(loaded, _LOADED_EMA_COLUMN),
        loaded_hold=_as_int_col(loaded, "hold_index"),
        loaded_phase=_as_int_col(loaded, "phase"),
        loaded_step=_as_int_col(loaded, "hold_step_idx"),
        baseline_ema=_as_wrench_col(baseline, _UNLOADED_EMA_COLUMN),
        baseline_raw=_as_wrench_col(
            baseline,
            "ft_wrist_raw" if "ft_wrist_raw" in baseline.column_names else _UNLOADED_EMA_COLUMN,
        ),
        baseline_hold=_as_int_col(baseline, "hold_index"),
        baseline_phase=_as_int_col(baseline, "phase"),
        baseline_step=_as_int_col(baseline, "hold_step_idx"),
    )

    arrays = {name: loaded.column(name) for name in loaded.column_names}
    wrench_type = pa.list_(pa.float32(), 6)
    arrays["ft_wrist_ema_baseline"] = pa.array(ema_baseline.astype(np.float32).tolist(), type=wrench_type)
    arrays["ft_wrist_ema_tared"] = pa.array(ema_tared.astype(np.float32).tolist(), type=wrench_type)
    out_table = pa.table(arrays)

    metadata = _read_metadata(loaded)
    metadata["ema_tare"] = {
        "applied": True,
        "loaded_column": _LOADED_EMA_COLUMN,
        "unloaded_column": _UNLOADED_EMA_COLUMN,
        "baseline_path": str(baseline_path.resolve()),
        "method": "per-(hold_index, phase) frame-index match; loaded EMA minus unloaded EMA",
        "tared_column": "ft_wrist_ema_tared",
        "matched_unloaded_column": "ft_wrist_ema_baseline",
    }
    existing = dict(loaded.schema.metadata or {})
    existing[b"dataset_metadata"] = json.dumps(metadata).encode("utf-8")
    out_table = out_table.replace_schema_metadata(existing)
    pq.write_table(out_table, output_path)
    return output_path


def write_tare_comparison_html(episode_path: Path | str, plot_path: Path | str) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    episode_path = Path(episode_path)
    plot_path = Path(plot_path)
    table = pq.read_table(episode_path)
    required = (
        "timestamp",
        "phase_name",
        "ft_wrist",
        "ft_wrist_raw",
        "ft_wrist_baseline",
        "ft_wrist_ema_baseline",
        "ft_wrist_ema_tared",
    )
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"{episode_path} missing columns {missing}")

    t = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    t = t - t[0]
    names = [str(value) for value in table.column("phase_name").to_pylist()]
    current = _as_wrench_col(table, "ft_wrist")
    loaded = _as_wrench_col(table, "ft_wrist_raw")
    raw_base = _as_wrench_col(table, "ft_wrist_baseline")
    ema_base = _as_wrench_col(table, "ft_wrist_ema_baseline")
    ema_tared = _as_wrench_col(table, "ft_wrist_ema_tared")

    labels = ("Fx", "Fy", "Fz")
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(*labels, "‖F‖ (N)"),
    )
    series = (
        ("loaded EMA (ft_wrist_raw)", loaded, "rgba(80,80,80,0.7)"),
        ("unloaded raw (current baseline)", raw_base, "rgba(214,39,40,0.45)"),
        ("unloaded EMA (new baseline)", ema_base, "rgba(255,127,14,0.9)"),
        ("current tare (EMA−raw)", current, "rgba(148,103,189,0.9)"),
        ("EMA tare (EMA−EMA)", ema_tared, "rgba(31,119,180,1.0)"),
    )
    for axis, label in enumerate(labels):
        for name, data, color in series:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=data[:, axis],
                    name=name,
                    legendgroup=name,
                    showlegend=axis == 0,
                    line=dict(color=color, width=1.4 if "EMA tare" in name else 1.0),
                ),
                row=axis + 1,
                col=1,
            )
            fig.update_yaxes(title_text=f"{label} (N)", row=axis + 1, col=1)
    for name, data, color in series:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=np.linalg.norm(data[:, :3], axis=1),
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=color, width=1.4 if "EMA tare" in name else 1.0),
            ),
            row=4,
            col=1,
        )
    fig.update_yaxes(title_text="‖F‖ (N)", row=4, col=1)

    pull = np.array([name == "pull" for name in names], dtype=bool)
    if pull.any():
        starts = np.where(np.r_[pull[0], pull[1:] & ~pull[:-1]])[0]
        ends = np.where(np.r_[~pull[1:] & pull[:-1], pull[-1]])[0]
        for start, end in zip(starts, ends):
            fig.add_vrect(
                x0=float(t[start]),
                x1=float(t[end]),
                fillcolor="rgba(120,180,255,0.12)",
                line_width=0,
                layer="below",
            )
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_layout(
        title=f"EMA tare vs raw tare — {episode_path.name} (blue bands = pull)",
        height=1100,
        legend=dict(orientation="h", y=1.04),
        hovermode="x unified",
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(plot_path), include_plotlyjs="cdn")
    return plot_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Compiled unified episode parquet")
    parser.add_argument("--baseline", type=Path, default=None, help="Unloaded *_baseline_robot.parquet")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the episode with new columns (default: overwrite --input)",
    )
    parser.add_argument("--plot", type=Path, default=None, help="Plotly HTML comparison path")
    args = parser.parse_args(argv)

    loaded = pq.read_table(args.input)
    metadata = _read_metadata(loaded)
    baseline_path = args.baseline or resolve_baseline_path(args.input, metadata)
    output_path = add_ema_tare_columns(args.input, baseline_path, output_path=args.output)
    print(f"Wrote EMA tare columns to {output_path}")
    print(f"Unloaded EMA source {baseline_path}")
    if args.plot is not None:
        plot_path = write_tare_comparison_html(output_path, args.plot)
        print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
