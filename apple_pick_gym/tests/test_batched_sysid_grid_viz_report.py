from __future__ import annotations

import pytest

from apple_pick_gym.grid_viz_table import GridVizRow


def _row(*, cand: int, gt: bool, dist: float, err: float, disqualified: bool = False) -> GridVizRow:
    return GridVizRow(
        structure_idx=0,
        candidate_index=cand,
        gt_flag=gt,
        primary=1.0,
        secondary=0.0,
        spur=1.0,
        stem=1.0,
        dist_log_gt=float(dist),
        n_frames_all=10.0,
        err_pos_all=float(err),
        err_force_all=float(err),
        err_torque_all=float(err),
        n_frames_hold=5.0,
        err_pos_hold=float(err),
        err_force_hold=float(err),
        err_torque_hold=float(err),
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


def test_summarize_final_rank_ignores_disqualified_best():
    from apple_pick_gym.grid_viz_report import summarize_final_rank
    from apple_pick_gym.grid_viz_table import assign_hold_combined_ranks

    rows = assign_hold_combined_ranks(
        [
            _row(cand=0, gt=False, dist=2.0, err=0.05, disqualified=True),
            _row(cand=1, gt=True, dist=0.0, err=0.0),
            _row(cand=2, gt=False, dist=1.0, err=0.2),
        ]
    )
    summary = summarize_final_rank(rows)
    assert summary.n_disqualified == 1
    assert summary.best_candidate_index == 1
    assert summary.best_is_gt is True


def test_structure_report_ranks_gt_and_computes_spearman():
    from apple_pick_gym.grid_viz_report import summarize_structure

    rows = [
        _row(cand=0, gt=False, dist=2.0, err=4.0),
        _row(cand=1, gt=True, dist=0.0, err=0.0),
        _row(cand=2, gt=False, dist=1.0, err=1.0),
    ]
    rep = summarize_structure(
        structure_idx=0,
        rows=rows,
        metrics=("err_pos_hold",),
    )

    s = rep.summaries[0]
    assert s.best_is_gt is True
    assert s.gt_rank == 1
    assert s.spearman_dist_vs_err == pytest.approx(1.0)


def test_across_structures_report_aggregates():
    from apple_pick_gym.grid_viz_report import summarize_across_structures, summarize_structure

    rep0 = summarize_structure(
        structure_idx=0,
        rows=[_row(cand=0, gt=True, dist=0.0, err=0.0), _row(cand=1, gt=False, dist=1.0, err=1.0)],
        metrics=("err_pos_hold",),
    )
    rep1 = summarize_structure(
        structure_idx=1,
        rows=[_row(cand=0, gt=True, dist=0.0, err=0.0), _row(cand=1, gt=False, dist=1.0, err=2.0)],
        metrics=("err_pos_hold",),
    )
    across = summarize_across_structures(structure_reports=[rep0, rep1])
    m = across.metrics["err_pos_hold"]
    assert m["n_structures"] == 2
    assert m["p_best_is_gt"] == pytest.approx(1.0)

