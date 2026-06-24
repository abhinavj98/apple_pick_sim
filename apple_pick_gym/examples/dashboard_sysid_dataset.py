"""Local Dash dashboard for sys-ID trajectory datasets.

Run from the repository root::

    uv run python apple_pick_gym/examples/dashboard_sysid_dataset.py \
      --dataset /tmp/apple_pick_sysid_gt
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np

from apple_pick_sim.system_id import TrajectoryDataset
from apple_pick_sim.system_id.dashboard_data import (
    PHASE_NAMES,
    build_frame_mask,
    hold_summaries,
    phase_names_for_values,
    woody_endpoint_series,
)


ALL_DIRECTIONS = "__all__"
PHASE_COLORS = {
    "move_out": "#1f77b4",
    "hold": "#2ca02c",
    "return": "#ff7f0e",
}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a local sys-ID dataset dashboard.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Directory containing metadata.parquet and frames/*.parquet.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash bind host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash bind port.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the dashboard URL in the default browser before serving.",
    )
    return parser


def _episode_options(dataset: TrajectoryDataset) -> list[dict[str, str]]:
    return [{"label": episode_id, "value": episode_id} for episode_id in dataset.episode_ids()]


def _direction_options(arrays: dict[str, Any]) -> list[dict[str, str | int]]:
    directions = sorted({int(value) for value in np.asarray(arrays["dir_idx"]).reshape(-1)})
    return [{"label": "All directions", "value": ALL_DIRECTIONS}] + [
        {"label": f"Direction {direction}", "value": direction}
        for direction in directions
    ]


def _selected_direction(value: str | int | None) -> int | None:
    if value in (None, ALL_DIRECTIONS):
        return None
    return int(value)


def _time_axis(arrays: dict[str, Any]) -> np.ndarray:
    if "sim_time" in arrays:
        sim_time = np.asarray(arrays["sim_time"], dtype=np.float64).reshape(-1)
        if sim_time.size:
            return sim_time
    if "step_idx" in arrays:
        return np.asarray(arrays["step_idx"], dtype=np.float64).reshape(-1)
    n_frames = int(np.asarray(arrays["phase"]).reshape(-1).shape[0])
    return np.arange(n_frames, dtype=np.float64)


def _marker_colors(phases: np.ndarray) -> list[str]:
    names = phase_names_for_values(phases)
    return [PHASE_COLORS.get(name, "#7f7f7f") for name in names]


def _has_series(arrays: dict[str, Any], name: str) -> bool:
    return name in arrays and np.asarray(arrays[name]).reshape(-1).size > 0


def _set_3d_bounds(fig, arrays: list[np.ndarray]) -> None:
    points = [arr.reshape(-1, 3) for arr in arrays if arr.size > 0]
    if not points:
        return
    stacked = np.concatenate(points, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    if span < 1e-6:
        span = 1.0
    half = 0.55 * span
    fig.update_layout(
        scene={
            "xaxis": {"range": [center[0] - half, center[0] + half], "title": "x [m]"},
            "yaxis": {"range": [center[1] - half, center[1] + half], "title": "y [m]"},
            "zaxis": {"range": [center[2] - half, center[2] + half], "title": "z [m]"},
            "aspectmode": "cube",
        }
    )


def _trajectory_figure(arrays: dict[str, Any], mask: np.ndarray):
    import plotly.graph_objects as go

    tcp = np.asarray(arrays["tcp_pos"], dtype=np.float64).reshape(-1, 3)[mask]
    apple = np.asarray(arrays["apple_pos"], dtype=np.float64).reshape(-1, 3)[mask]
    phases = np.asarray(arrays["phase"]).reshape(-1)[mask]
    colors = _marker_colors(phases)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=tcp[:, 0],
            y=tcp[:, 1],
            z=tcp[:, 2],
            mode="lines+markers",
            marker={"size": 3, "color": colors},
            line={"color": "#1f77b4", "width": 4},
            name="TCP",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=apple[:, 0],
            y=apple[:, 1],
            z=apple[:, 2],
            mode="lines+markers",
            marker={"size": 3, "color": "#ff7f0e"},
            line={"color": "#ff7f0e", "width": 3},
            name="Apple",
        )
    )
    endpoint_arrays = []
    for endpoint in woody_endpoint_series(arrays, mask):
        endpoint_arrays.append(endpoint.xyz)
        fig.add_trace(
            go.Scatter3d(
                x=endpoint.xyz[:, 0],
                y=endpoint.xyz[:, 1],
                z=endpoint.xyz[:, 2],
                mode="lines",
                line={"width": 2},
                name=endpoint.name,
                opacity=0.65,
            )
        )
    fig.update_layout(
        title="3D TCP, apple, and woody endpoint trajectories",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h"},
    )
    _set_3d_bounds(fig, [tcp, apple] + endpoint_arrays)
    return fig


def _motion_figure(arrays: dict[str, Any], mask: np.ndarray):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    t = _time_axis(arrays)[mask]
    action = np.asarray(arrays["action"], dtype=np.float64).reshape(-1, 6)[mask]
    tcp_velocity = np.asarray(arrays["tcp_velocity"], dtype=np.float64).reshape(-1, 6)[mask]
    has_amplitude = _has_series(arrays, "amplitude_m")
    rows = 3 if has_amplitude else 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            ["Amplitude [m]"] if has_amplitude else []
        ) + ["Action linear velocity [m/s]", "TCP linear velocity [m/s]"],
    )
    row = 1
    if has_amplitude:
        fig.add_trace(
            go.Scatter(x=t, y=np.asarray(arrays["amplitude_m"])[mask], name="amplitude"),
            row=row,
            col=1,
        )
        row += 1
    for axis, label in enumerate(("vx", "vy", "vz")):
        fig.add_trace(go.Scatter(x=t, y=action[:, axis], name=f"action {label}"), row=row, col=1)
    row += 1
    for axis, label in enumerate(("vx", "vy", "vz")):
        fig.add_trace(
            go.Scatter(x=t, y=tcp_velocity[:, axis], name=f"tcp {label}"),
            row=row,
            col=1,
        )
    fig.update_layout(title="Motion command and TCP velocity", height=650)
    fig.update_xaxes(title_text="time [s] or step")
    return fig


def _wrench_figure(arrays: dict[str, Any], mask: np.ndarray):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    t = _time_axis(arrays)[mask]
    wrench = np.asarray(arrays["ft_wrist"], dtype=np.float64).reshape(-1, 6)[mask]
    raw = (
        np.asarray(arrays["raw_ft_wrist"], dtype=np.float64).reshape(-1, 6)[mask]
        if "raw_ft_wrist" in arrays
        else None
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["TCP force [N]", "TCP torque [N*m]"],
    )
    for axis, label in enumerate(("Fx", "Fy", "Fz")):
        fig.add_trace(go.Scatter(x=t, y=wrench[:, axis], name=label), row=1, col=1)
        if raw is not None and not np.allclose(raw[:, axis], wrench[:, axis]):
            fig.add_trace(
                go.Scatter(x=t, y=raw[:, axis], name=f"raw {label}", line={"dash": "dash"}),
                row=1,
                col=1,
            )
    for axis, label in zip(range(3, 6), ("Tx", "Ty", "Tz"), strict=True):
        fig.add_trace(go.Scatter(x=t, y=wrench[:, axis], name=label), row=2, col=1)
    fig.update_layout(title="Wrist wrench", height=550)
    fig.update_xaxes(title_text="time [s] or step")
    return fig


def _initial_tcp_pos(meta: dict[str, Any], arrays: dict[str, Any]) -> np.ndarray:
    value = meta.get("initial_tcp_pos")
    if value is None:
        return np.asarray(arrays["tcp_pos"], dtype=np.float64).reshape(-1, 3)[0]
    return np.asarray(value, dtype=np.float64).reshape(3)


def _hold_table_rows(
    arrays: dict[str, Any],
    meta: dict[str, Any],
    *,
    direction: int | None,
) -> list[dict[str, float | int | str]]:
    rows = []
    for summary in hold_summaries(arrays, initial_tcp_pos=_initial_tcp_pos(meta, arrays)):
        if direction is not None and summary.direction != direction:
            continue
        rows.append(
            {
                "direction": summary.direction,
                "amplitude_m": summary.amplitude_m,
                "n_frames": summary.n_frames,
                "mean_force_n": float(summary.force_norm_n),
                "tcp_displacement_m": float(summary.tcp_displacement_m),
                "stiffness_n_per_m": (
                    ""
                    if summary.stiffness_n_per_m is None
                    else float(summary.stiffness_n_per_m)
                ),
                "force_xyz_n": (
                    f"({summary.mean_force_n[0]:+.3f}, "
                    f"{summary.mean_force_n[1]:+.3f}, {summary.mean_force_n[2]:+.3f})"
                ),
            }
        )
    return rows


def create_dashboard_app(dataset_dir: str | Path):
    from dash import Dash, Input, Output, dash_table, dcc, html

    dataset = TrajectoryDataset(dataset_dir)
    episode_options = _episode_options(dataset)
    if not episode_options:
        raise ValueError(f"No episodes found in dataset: {dataset.dataset_dir}")
    first_episode = episode_options[0]["value"]
    first_arrays = dataset.load_episode_obs_arrays(first_episode)

    app = Dash(__name__)
    app.title = "Apple Pick Sys-ID Dataset Dashboard"
    app.layout = html.Div(
        [
            html.H1("Apple Pick Sys-ID Dataset Dashboard"),
            html.Div(
                [
                    html.Label("Episode"),
                    dcc.Dropdown(
                        id="episode",
                        options=episode_options,
                        value=first_episode,
                        clearable=False,
                    ),
                    html.Label("Direction"),
                    dcc.Dropdown(
                        id="direction",
                        options=_direction_options(first_arrays),
                        value=ALL_DIRECTIONS,
                        clearable=False,
                    ),
                    html.Label("Phases"),
                    dcc.Checklist(
                        id="phases",
                        options=[
                            {"label": name, "value": name}
                            for _, name in sorted(PHASE_NAMES.items())
                        ],
                        value=list(PHASE_NAMES.values()),
                        inline=True,
                    ),
                ],
                style={"display": "grid", "gap": "0.5rem", "maxWidth": "900px"},
            ),
            dcc.Graph(id="trajectory-3d"),
            dcc.Graph(id="motion-timeseries"),
            dcc.Graph(id="wrench-timeseries"),
            html.H2("Hold Summary"),
            dash_table.DataTable(
                id="hold-summary",
                page_size=12,
                sort_action="native",
                style_table={"overflowX": "auto"},
            ),
        ],
        style={"fontFamily": "sans-serif", "margin": "1rem"},
    )

    @app.callback(
        Output("direction", "options"),
        Output("direction", "value"),
        Input("episode", "value"),
    )
    def _update_direction_options(episode_id: str):
        arrays = dataset.load_episode_obs_arrays(episode_id)
        return _direction_options(arrays), ALL_DIRECTIONS

    @app.callback(
        Output("trajectory-3d", "figure"),
        Output("motion-timeseries", "figure"),
        Output("wrench-timeseries", "figure"),
        Output("hold-summary", "data"),
        Output("hold-summary", "columns"),
        Input("episode", "value"),
        Input("direction", "value"),
        Input("phases", "value"),
    )
    def _update_figures(episode_id: str, direction_value: str | int, phases: list[str]):
        arrays = dataset.load_episode_obs_arrays(episode_id)
        meta = dataset.load_episode_meta(episode_id)
        direction = _selected_direction(direction_value)
        mask = build_frame_mask(arrays, direction=direction, phases=phases)
        if not np.any(mask):
            mask = np.zeros_like(mask, dtype=bool)
        rows = _hold_table_rows(arrays, meta, direction=direction)
        columns = [{"name": name, "id": name} for name in rows[0].keys()] if rows else []
        return (
            _trajectory_figure(arrays, mask),
            _motion_figure(arrays, mask),
            _wrench_figure(arrays, mask),
            rows,
            columns,
        )

    return app


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    app = create_dashboard_app(args.dataset)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving sys-ID dataset dashboard at {url}")
    if bool(args.open_browser):
        webbrowser.open(url)
    app.run(host=args.host, port=int(args.port), debug=bool(args.debug))
    return 0


if __name__ == "__main__":
    sys.exit(main())
