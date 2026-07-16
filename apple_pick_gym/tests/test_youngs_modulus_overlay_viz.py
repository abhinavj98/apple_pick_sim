"""Unit tests for multi-E Young’s-modulus overlay Plotly hygiene."""

from __future__ import annotations

import numpy as np
import pytest

from apple_pick_sim.system_id.trajectory_store import PHASE_TO_INT


def _synthetic_episode(
    *,
    structure_idx: int,
    direction_idx: int,
    log10_e: tuple[float, float, float],
    pull: tuple[float, float, float],
    excluded: bool = False,
    n: int = 20,
):
    from apple_pick_gym.youngs_modulus_overlay_viz import OverlayEpisode

    t = np.linspace(0.0, 2.0, n, dtype=np.float64)
    phase = np.full(n, PHASE_TO_INT["move_out"], dtype=np.int8)
    phase[n // 2 :] = PHASE_TO_INT["hold"]
    pull_arr = np.asarray(pull, dtype=np.float64)
    pull_arr = pull_arr / np.linalg.norm(pull_arr)
    scale = 10.0 ** (log10_e[0] - 8.0)
    tcp0 = np.array([0.5, 0.0, 0.4], dtype=np.float64)
    tcp = tcp0 + np.outer(t, pull_arr) * (0.05 / scale)
    ft = np.zeros((n, 6), dtype=np.float64)
    ft[:, :3] = pull_arr * (scale * (1.0 + t[:, None]))
    ft[:, 3:] = 0.1 * ft[:, :3]
    return OverlayEpisode(
        structure_idx=structure_idx,
        direction_idx=direction_idx,
        candidate_label=f"log10={log10_e}",
        log10_e=log10_e,
        sim_time=t,
        phase=phase,
        ft_wrist=ft,
        tcp_pos=tcp,
        pull_direction=pull_arr,
        excluded=excluded,
    )


def test_overlay_facets_by_direction_and_caps_series():
    from apple_pick_gym.youngs_modulus_overlay_viz import make_youngs_modulus_overlay_figure

    eps = [
        _synthetic_episode(
            structure_idx=0,
            direction_idx=0,
            log10_e=(8.0, 7.5, 7.0),
            pull=(1.0, 0.0, 0.0),
        ),
        _synthetic_episode(
            structure_idx=1,
            direction_idx=0,
            log10_e=(8.5, 7.5, 7.0),
            pull=(1.0, 0.0, 0.0),
        ),
        _synthetic_episode(
            structure_idx=0,
            direction_idx=1,
            log10_e=(8.0, 7.5, 7.0),
            pull=(0.0, 1.0, 0.0),
        ),
        _synthetic_episode(
            structure_idx=1,
            direction_idx=1,
            log10_e=(8.5, 7.5, 7.0),
            pull=(0.0, 1.0, 0.0),
        ),
    ]
    fig = make_youngs_modulus_overlay_figure(eps, max_overlay_candidates=8)
    # 2 direction rows + 1 move-vs-pull row; 3 cols
    rows, cols = fig._get_subplot_rows_columns()
    assert list(rows) == [1, 2, 3]
    assert list(cols) == [1, 2, 3]
    # Default mode adds F/T/dTCP traces (3 panels × 2 candidates × 2 dirs) + move scatters
    # plus phase shapes — component dump must not appear in default mode.
    names = [getattr(tr, "name", "") or "" for tr in fig.data]
    assert not any("Fx" in n or "Tx" in n for n in names)
    # Shape count: phase bands present
    assert fig.layout.shapes is not None and len(fig.layout.shapes) > 0


def test_overlay_refuses_when_over_candidate_cap():
    from apple_pick_gym.youngs_modulus_overlay_viz import make_youngs_modulus_overlay_figure

    eps = [
        _synthetic_episode(
            structure_idx=i,
            direction_idx=0,
            log10_e=(8.0 + 0.1 * i, 7.5, 7.0),
            pull=(1.0, 0.0, 0.0),
        )
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="max_overlay_candidates"):
        make_youngs_modulus_overlay_figure(eps, max_overlay_candidates=3)


def test_overlay_omits_excluded_episodes():
    from apple_pick_gym.youngs_modulus_overlay_viz import make_youngs_modulus_overlay_figure

    eps = [
        _synthetic_episode(
            structure_idx=0,
            direction_idx=0,
            log10_e=(8.0, 7.5, 7.0),
            pull=(1.0, 0.0, 0.0),
        ),
        _synthetic_episode(
            structure_idx=1,
            direction_idx=0,
            log10_e=(8.5, 7.5, 7.0),
            pull=(1.0, 0.0, 0.0),
            excluded=True,
        ),
    ]
    fig = make_youngs_modulus_overlay_figure(eps, max_overlay_candidates=8)
    # Only one candidate's traces on time panels (excluded omitted).
    legend_names = {tr.name for tr in fig.data if tr.name and "log10" in tr.name}
    # Each candidate appears once in legend (legendgroup); excluded should not.
    assert any("8.0" in n for n in legend_names)
    assert not any("8.5" in n for n in legend_names)


def test_write_overlay_html(tmp_path):
    from apple_pick_gym.youngs_modulus_overlay_viz import write_youngs_modulus_overlay_html

    eps = [
        _synthetic_episode(
            structure_idx=0,
            direction_idx=0,
            log10_e=(8.0, 7.5, 7.0),
            pull=(1.0, 0.0, 0.0),
        )
    ]
    out = write_youngs_modulus_overlay_html(eps, tmp_path / "overlay.html")
    assert out.is_file()
    assert out.stat().st_size > 100


def test_dataset_adapter_strips_pre_weld_frame():
    from apple_pick_gym.youngs_modulus_overlay_viz import (
        overlay_episodes_from_batched_dataset,
    )

    class FakeDataset:
        manifest = {
            "collection": {"num_structures": 1, "num_directions": 1},
        }

        @staticmethod
        def structure_summaries():
            return [{"structure_idx": 0}]

        @staticmethod
        def episode_entries():
            return [
                {
                    "structure_idx": 0,
                    "direction_idx": 0,
                    "excluded": False,
                }
            ]

        @staticmethod
        def load_episode_metadata(structure_idx, direction_idx):
            del structure_idx, direction_idx
            return {"pull_direction": [1.0, 0.0, 0.0]}

        @staticmethod
        def load_episode_obs_arrays(structure_idx, direction_idx):
            del structure_idx, direction_idx
            return {
                "step_idx": np.array([-1, 0, 1], dtype=np.int32),
                "sim_time": np.array([0.0, 0.1, 0.2]),
                "phase": np.array(
                    [
                        PHASE_TO_INT["pre_weld"],
                        PHASE_TO_INT["move_out"],
                        PHASE_TO_INT["hold"],
                    ],
                    dtype=np.int8,
                ),
                "ft_wrist": np.zeros((3, 6), dtype=np.float32),
                "tcp_pos": np.array(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                    dtype=np.float32,
                ),
            }

    episodes = overlay_episodes_from_batched_dataset(
        FakeDataset(),
        candidate_labels=["candidate"],
        candidate_log10_e=[(8.0, 7.5, 7.0)],
    )

    assert len(episodes) == 1
    assert episodes[0].phase.tolist() == [
        PHASE_TO_INT["move_out"],
        PHASE_TO_INT["hold"],
    ]
    assert episodes[0].tcp_pos[:, 0].tolist() == pytest.approx([1.0, 1.1])
