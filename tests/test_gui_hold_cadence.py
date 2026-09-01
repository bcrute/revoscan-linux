"""Whether the frame spacing a capture actually used is the one it was asked for.

:mod:`scanner3d.cadence` decides the spacing and ``tests/test_cadence.py``
proves it decides correctly; :func:`~scanner3d.v4l2_stream.stream_frames`
selects the deliveries and ``tests/test_v4l2_stream.py`` proves it selects the
right ones. Neither of those can catch the seam between them - a controller
whose answers reach nobody, a capture that builds one and then hands the stream
the plain integer anyway - and that seam is the whole subject here.

So everything below drives the real capture path: the real ``_start_capture``,
its real callbacks, and the real dequeue loop against the scripted fake driver.
Nothing here opens a device or builds a widget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from scanner3d.v4l2_stream import StreamResult
from tests.fake_v4l2 import Delivery, stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    alignments,
    capture_dir,
    clouds,
    deprojected,
    live,
    models,
    start_capture,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "deprojected",
    "live",
    "models",
    "stream_kernel",
    "stream_node",
]


def frames(count: int) -> tuple[Delivery, ...]:
    return tuple(whole_frame(0x40 + index, buffer_index=index % 4) for index in range(count))


def _result(window: CaptureWindow) -> StreamResult:
    assert isinstance(window.result, StreamResult)
    return window.result


def test_a_capture_that_was_not_asked_to_hold_the_spacing_runs_exactly_as_it_did(
    stream_kernel, capture_dir: Path, deprojected: list[int], live
) -> None:
    """The default, and the one that has to be untouched.

    The control is off unless the operator ticks it, so the ordinary capture
    hands the stream the same plain integer it always did - and its record says
    one spacing, held, because that is what happened.
    """
    stream_kernel(frames(9))
    live()
    window = CaptureWindow(frames=9, decimate=3)

    start_capture(window)

    result = _result(window)
    assert result.kept == 3
    assert result.decimate == 3
    assert result.decimate_lowest == result.decimate_highest == 3
    assert result.decimate_varied is False
    assert result.decimate_refusals == 0
    assert "Frame spacing" not in window.report


def test_a_held_capture_that_never_leaves_the_band_keeps_the_operators_own_spacing(
    stream_kernel, capture_dir: Path, deprojected: list[int], live
) -> None:
    """Holding is not moving.

    A controller that drifted off the setting on a capture with nothing wrong
    with it would be worse than no controller: the operator would get a folder
    spaced differently from the number they typed, for no reason either of them
    could point at.
    """
    stream_kernel(frames(9))
    live()
    window = CaptureWindow(frames=9, decimate=3, hold_cadence=True)

    start_capture(window)

    result = _result(window)
    assert result.decimate_lowest == result.decimate_highest == 3
    assert result.decimate_varied is False
    assert "held at 1 in 3 for the whole capture" in window.report


def test_a_refused_frame_tightens_the_spacing_and_the_capture_says_so(
    stream_kernel, capture_dir: Path, deprojected: list[int], live
) -> None:
    """The seam, end to end, on the signal that matters most.

    A frame the tracker could not place carries no travel, so a controller fed
    only accepted frames would learn nothing from the one event that says "too
    fast" before the streak that ends a capture. Here frame 1 fails, the spacing
    halves from the operator's 3 to the finest there is, and every later
    delivery is kept rather than two in every three being discarded unexamined.
    """
    stream_kernel(frames(9))
    live(failed_alignments={1})
    window = CaptureWindow(frames=9, decimate=3, hold_cadence=True)

    start_capture(window)

    result = _result(window)
    assert result.decimate_lowest == 1
    assert result.decimate_highest == 3
    assert result.decimate_varied is True
    # Deliveries 0 and 3 at the original spacing, then every one after the
    # refusal on the frame kept at 3.
    assert result.kept > 3
    assert "Frame spacing moved" in window.report
    assert "frames the tracker could not place" in window.report


def test_the_same_refusal_moves_nothing_when_the_operator_did_not_ask(
    stream_kernel, capture_dir: Path, deprojected: list[int], live
) -> None:
    """The companion, and the one that proves the test above measured the control.

    Identical frames and an identical failure, with the box unticked. A capture
    that tightened anyway would be a capture ignoring a setting rather than one
    honouring a different one.
    """
    stream_kernel(frames(9))
    live(failed_alignments={1})
    window = CaptureWindow(frames=9, decimate=3)

    start_capture(window)

    result = _result(window)
    assert result.kept == 3
    assert result.decimate_lowest == result.decimate_highest == 3
