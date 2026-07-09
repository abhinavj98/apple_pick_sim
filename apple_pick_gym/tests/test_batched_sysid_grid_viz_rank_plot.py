from __future__ import annotations

import numpy as np
import pytest

from apple_pick_gym.grid_viz_plotly import (
    compute_dual_metric_ranks,
    compute_pareto_front_mask,
    load_grid_viz_rows_from_json,
    make_3d_rank_scatter,
    make_3d_scatter,
)
from apple_pick_gym.grid_viz_table import GridVizRow, assign_hold_combined_ranks


def _row(candidate_index: int, *, pos: float, force: float, gt: bool = False, disqualified: bool = False) -> GridVizRow:
    return GridVizRow(
        structure_idx=0,
        candidate_index=candidate_index,
        gt_flag=gt,
        primary=10.0,
        secondary=1e-12,
        spur=1e-3,
        stem=1e-4,
        dist_log_gt=0.0 if gt else 0.5,
        n_frames_all=10.0,
        err_pos_all=pos,
        err_force_all=force,
        err_torque_all=1.0,
        n_frames_hold=10.0,
        err_pos_hold=pos,
        err_force_hold=force,
        err_torque_hold=1.0,
        woody_pos_mse_all={},
        woody_pos_mse_hold={},
        err_woody_pos_all=float("nan"),
        err_woody_pos_hold=float("nan"),
        n_directions_all=1.0,
        n_directions_hold=1.0,
        unstable_fraction_all=0.15 if disqualified else 0.0,
        disqualified=bool(disqualified),
        rank_pos_hold=float("nan"),
        rank_force_hold=float("nan"),
        rank_combined=float("inf") if disqualified else float("nan"),
    )


def test_compute_dual_metric_ranks_orders_by_each_metric():
    rows = [
        _row(0, pos=0.3, force=30.0),
        _row(1, pos=0.1, force=20.0),
        _row(2, pos=0.2, force=10.0),
    ]
    ranks = compute_dual_metric_ranks(rows)

    assert ranks["rank_pos"].tolist() == [3.0, 1.0, 2.0]
    assert ranks["rank_force"].tolist() == [3.0, 2.0, 1.0]
    assert ranks["rank_combined"].tolist() == [3.0, 1.5, 1.5]


def test_load_grid_viz_rows_from_json_round_trip(tmp_path):
    rows = [_row(0, pos=0.1, force=10.0, gt=True), _row(1, pos=0.2, force=20.0)]
    path = tmp_path / "structure_000_rows.json"
    path.write_text("\n".join([__import__("json").dumps(r.to_dict(), sort_keys=True) for r in rows]) + "\n")

    loaded = load_grid_viz_rows_from_json(path)
    assert len(loaded) == 2
    assert loaded[0].candidate_index == 0
    assert loaded[0].gt_flag is True
    assert loaded[1].err_force_hold == 20.0


def test_assign_hold_combined_ranks_excludes_disqualified():
    rows = [
        _row(0, pos=0.3, force=30.0, gt=False, disqualified=True),
        _row(1, pos=0.1, force=20.0, gt=True),
        _row(2, pos=0.2, force=10.0, gt=False),
    ]
    ranked = assign_hold_combined_ranks(rows)
    assert ranked[0].disqualified is True
    assert ranked[0].rank_combined == float("inf")
    assert ranked[1].rank_combined == pytest.approx(1.5)
    assert ranked[2].rank_combined == pytest.approx(1.5)


def test_make_3d_rank_scatter_includes_disqualified_trace():
    rows = assign_hold_combined_ranks(
        [
            _row(0, pos=0.3, force=30.0, gt=False, disqualified=True),
            _row(1, pos=0.1, force=20.0, gt=True),
            _row(2, pos=0.2, force=10.0, gt=False),
        ]
    )
    fig = make_3d_rank_scatter(rows=rows, title="t")
    names = [trace.name for trace in fig.data]
    assert "disqualified" in names
    assert "candidates" in names
    assert "GT" in names


def test_make_3d_scatter_gt_shares_color_scale():
    rows = assign_hold_combined_ranks(
        [
            _row(0, pos=0.3, force=30.0, gt=False),
            _row(1, pos=0.1, force=20.0, gt=True),
            _row(2, pos=0.2, force=10.0, gt=False),
        ]
    )
    fig = make_3d_scatter(rows=rows, metric="err_pos_hold", title="t")
    assert len(fig.data) == 2
    candidates = fig.data[0]
    gt = fig.data[1]
    assert candidates.name == "candidates"
    assert gt.name == "GT"
    assert candidates.marker.cmin == gt.marker.cmin
    assert candidates.marker.cmax == gt.marker.cmax
    assert candidates.marker.showscale is True
    assert gt.marker.showscale is False


def test_compute_pareto_front_mask_minimization_nondominated_set():
    # Three points in 2D objective space:
    # - A dominates B, C trades off.
    rows = [
        _row(0, pos=1.0, force=1.0),  # A
        _row(1, pos=2.0, force=2.0),  # B (dominated by A)
        _row(2, pos=0.5, force=3.0),  # C (tradeoff)
    ]
    mask = compute_pareto_front_mask(rows, metrics=("err_pos_hold", "err_force_hold"))
    assert mask.tolist() == [True, False, True]
