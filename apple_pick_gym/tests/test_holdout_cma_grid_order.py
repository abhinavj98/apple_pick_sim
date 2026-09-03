"""Job ordering for the holdout CMA grid supervisor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SCRIPT = REPO / "scripts" / "run_holdout_cma_grid.py"


def _load_grid_module():
    spec = importlib.util.spec_from_file_location("run_holdout_cma_grid", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def grid():
    return _load_grid_module()


def test_by_tree_keeps_same_structure_seeds_together(grid):
    jobs = grid.build_jobs(["s02", "s03", "s04"], [56, 57])
    ordered = grid.order_jobs(jobs, order="by_tree")
    assert [(j.tree, j.seed) for j in ordered] == [
        ("s02", 56),
        ("s02", 57),
        ("s03", 56),
        ("s03", 57),
        ("s04", 56),
        ("s04", 57),
    ]


def test_interleaved_puts_different_trees_first(grid):
    jobs = grid.build_jobs(["s02", "s03", "s04"], [56, 57])
    ordered = grid.order_jobs(jobs, order="interleaved")
    assert [(j.tree, j.seed) for j in ordered] == [
        ("s02", 56),
        ("s03", 56),
        ("s04", 56),
        ("s02", 57),
        ("s03", 57),
        ("s04", 57),
    ]
    # First concurrent wave (3) is three different structures.
    assert {j.tree for j in ordered[:3]} == {"s02", "s03", "s04"}


def test_shuffle_is_deterministic_with_seed(grid):
    jobs = grid.build_jobs(["s02", "s03", "s04", "s06"], [56, 57, 58])
    a = grid.order_jobs(jobs, order="shuffle", shuffle_seed=7)
    b = grid.order_jobs(jobs, order="shuffle", shuffle_seed=7)
    c = grid.order_jobs(jobs, order="shuffle", shuffle_seed=8)
    assert [(j.tree, j.seed) for j in a] == [(j.tree, j.seed) for j in b]
    assert [(j.tree, j.seed) for j in a] != [(j.tree, j.seed) for j in c]
    assert sorted((j.tree, j.seed) for j in a) == sorted((j.tree, j.seed) for j in jobs)
