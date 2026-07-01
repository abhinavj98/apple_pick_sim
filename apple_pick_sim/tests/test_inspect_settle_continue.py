"""SPACE-to-continue detection for --inspect-settle settled-scene preview."""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from example_batched_heterogeneous_coupled_fruiting import (  # noqa: E402
    _settle_inspect_continue_requested,
)


class _MockViewer:
    def __init__(self, *, space_down: bool = False, paused: bool = False) -> None:
        self._space_down = space_down
        self._paused = paused

    def is_key_down(self, key: str) -> bool:
        return key == "space" and self._space_down

    def is_paused(self) -> bool:
        return self._paused


def test_continue_when_space_held():
    viewer = _MockViewer(space_down=True)
    assert _settle_inspect_continue_requested(viewer, graphical=True, paused_before=False)


def test_continue_on_space_tap_via_pause_toggle():
    viewer = _MockViewer(paused=True)
    assert _settle_inspect_continue_requested(viewer, graphical=True, paused_before=False)


def test_no_continue_for_literal_space_key_string():
    viewer = _MockViewer()

    class _LiteralSpaceViewer(_MockViewer):
        def is_key_down(self, key: str) -> bool:
            return key == " "

    assert not _settle_inspect_continue_requested(
        _LiteralSpaceViewer(), graphical=True, paused_before=False
    )


def test_no_continue_when_not_graphical():
    viewer = _MockViewer(space_down=True)
    assert not _settle_inspect_continue_requested(viewer, graphical=False, paused_before=False)
