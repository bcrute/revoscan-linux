"""Whether a capture in progress remains visibly alive, and how the model refreshes.

The owner's 2026-08-26 report was that "What has been scanned so far" updated
once or twice and then appeared to stop, on a sweep that offline replay later
proved had aligned all 81 of its frames successfully. Three things made a
healthy capture look dead:

- The only per-frame feedback in that panel was the 3D picture itself, which on
  a slowly turning object changes subtly enough to read as frozen.
- The refresh was scheduled on exact SAVED-frame multiples, so a single frame
  that failed to align at frame 12 or 24 lost that refresh entirely and the
  model sat still for the next two dozen frames.
- Alignment and coverage failures were swallowed whole, so nothing anywhere
  recorded which of those had happened.

So the schedule now counts SUCCESSFUL ALIGNMENTS, the panel carries a cheap
integer heartbeat that advances on every saved frame, and both failure paths are
logged. This drives all of that through the real capture path: the real
``_start_capture``, its real ``on_frame``, the real dequeue loop against the
scripted fake driver, and the real GUI slots that write the panel text.

Deprojection, alignment and the coverage build are stubbed - not to avoid the
work, but because the questions here are about SCHEDULING and REPORTING, and a
test that cannot decide which frame fails to align cannot ask them at all. Their
own correctness lives in ``test_icp_plane.py`` and ``test_coverage.py``.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.gui import COVERAGE_EVERY, MIN_COVERAGE_CLOUDS, capture_progress_text
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    alignments,
    capture_dir,
    clouds,
    live,
    models,
    start_capture,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "live",
    "models",
    "stream_kernel",
    "stream_node",
]


def frames(count: int) -> tuple[object, ...]:
    """``count`` whole frames, cycling the four buffers a real driver grants."""
    return tuple(whole_frame(1 + index % 250, buffer_index=index % 4) for index in range(count))


def _notes(window: CaptureWindow) -> list[str]:
    """What each model snapshot said about itself, in the order produced."""
    return [
        cast(gui.ModelSnapshot, snapshot).note for (snapshot,) in window.worker.coverage.emitted
    ]


def _saved(window: CaptureWindow) -> list[int]:
    return [cast(int, count) for (count,) in window.worker.saved.emitted]


# -- the heartbeat ---------------------------------------------------------


def test_the_heartbeat_advances_on_every_saved_frame(
    stream_kernel, capture_dir: Path, live
) -> None:
    """One cheap signal per frame written, whatever the model is doing.

    Five frames in, the model has been rebuilt once - at the third, the first
    point there is one worth drawing. The heartbeat has fired five times. That
    difference is the whole point: it is what distinguishes a capture that is
    between refreshes from one that has stopped.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5)
    live()

    start_capture(window)

    assert _saved(window) == [1, 2, 3, 4, 5]
    assert len(window.worker.coverage.emitted) == 1
    assert len(window.frames_on_disk()) == 5


def test_the_panel_says_how_many_frames_are_saved_and_when_the_model_last_changed(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Two numbers, kept apart, because they move at different rates.

    Read literally, the sentence has to survive being read by somebody deciding
    whether to keep turning the turntable.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5)
    live()

    start_capture(window)

    assert window._capture_progress.text == (
        "Capture ended: 5 frames saved. The model below was last rebuilt after 3 of "
        "them (1 rebuild)."
    )
    assert window._coverage_summary.text.startswith("preview after 3 aligned frames of 3 saved -")


def test_the_panel_admits_when_there_is_no_model_yet(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Two frames is not a model, and saying so beats an empty view.

    ``MIN_COVERAGE_CLOUDS`` observations is the floor for the coverage grid's
    "needs more scanning" judgement to mean anything, so below it the panel says
    what it is waiting for rather than drawing something misleading.
    """
    stream_kernel(frames(2))
    window = CaptureWindow(frames=2)
    live()

    start_capture(window)

    assert window.worker.coverage.emitted == []
    assert window._capture_progress.text == (
        "Capture ended: 2 frames saved. Too few of them aligned to build a model "
        "here; Reconstruct still can."
    )


def test_the_progress_sentence_is_written_from_its_counts_alone() -> None:
    """The sentence itself, in the four states it has, without a capture.

    Singular and plural both, because "1 frames saved" is the kind of thing that
    quietly tells a reader nobody looked at this.
    """
    assert capture_progress_text(0, None, 0) == "No frames saved yet."
    assert capture_progress_text(0, None, 0, running=False) == "No frames were saved."
    assert capture_progress_text(1, None, 0) == (
        "Capturing: 1 frame saved. The model below appears once enough of them have aligned."
    )
    assert capture_progress_text(40, 36, 3) == (
        "Capturing: 40 frames saved. The model below was last rebuilt after 36 of "
        "them (3 rebuilds)."
    )


# -- cancellation skips the expensive work ---------------------------------


def test_a_cancellation_seen_before_the_live_work_skips_all_of_it(
    stream_kernel, capture_dir: Path, clouds: list[int], live
) -> None:
    """Stop is visible to the callback, so the callback stops doing the optional part.

    The frame still reaches disk - it arrived, and it is a whole one - and the
    buffer still goes back to the driver. Everything after that is preview work
    for a capture that is ending.

    Cancelled from the saved-frame signal, which fires inside ``on_frame`` after
    the write and before the first optional stage: the callback's own "is this
    still wanted" boundary, reached with a dequeued buffer in hand. A flag set
    before the capture starts would be a different test entirely - the job now
    refuses that one before it ever opens the node.
    """
    kernel = stream_kernel(frames(2))
    window = CaptureWindow(frames=2)
    aligned, built = live()
    window.worker.saved.connect(lambda _saved_frames: window.worker.cancel())

    start_capture(window)

    assert len(window.frames_on_disk()) == 1
    assert _saved(window) == [1]
    # Not deprojected, not aligned, no model built - and nothing escaped the
    # callback, so the stream reached its own clean stop.
    assert clouds == []
    assert aligned == []
    assert built == []
    assert window.stopped == 1
    assert kernel.requeued == [0]


def test_a_cancellation_seen_during_deprojection_skips_alignment_and_the_model(
    stream_kernel, capture_dir: Path, monkeypatch, alignments, models
) -> None:
    """The check is per stage, not once at the top.

    A Stop pressed while a frame is being deprojected should not still pay for
    the alignment after it. Cancelling from inside the deprojection stub is how
    that moment is expressed offline.
    """
    aligned, built = alignments(set()), models(set())
    window = CaptureWindow(frames=4)

    def load(path: Path, _target_points: int) -> tuple[tuple[float, float, float], ...]:
        window.worker.cancelled = True
        return ((0.0, 0.0, 100.0),)

    monkeypatch.setattr(gui, "_load_cloud", load)
    stream_kernel(frames(4))

    start_capture(window)

    assert _saved(window) == [1]
    assert aligned == []
    assert built == []
    assert window.stopped == 1


# -- the model refresh schedule --------------------------------------------


def test_the_model_appears_as_soon_as_there_is_one_and_then_periodically(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Early once, then on a fixed alignment interval.

    Waiting a full interval for the first one means the panel says nothing for
    the opening dozen frames of every scan, which is exactly when an operator is
    deciding whether the capture is working at all.
    """
    total = MIN_COVERAGE_CLOUDS + COVERAGE_EVERY
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total)
    _aligned, built = live()

    start_capture(window)

    assert len(built) == 2
    assert _notes(window) == [
        f"preview after {MIN_COVERAGE_CLOUDS} aligned frames of {MIN_COVERAGE_CLOUDS} saved -",
        f"preview after {total} aligned frames of {total} saved -",
    ]


def test_one_frame_that_will_not_align_no_longer_costs_a_whole_refresh(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The 2026-08-26 stall, made impossible.

    The alignment that would have produced the second refresh fails. Under the
    old saved-frame schedule that refresh was simply lost and the model held
    still until the next multiple came round. Counting alignments instead, the
    next frame that DOES align is still due one, and gets it.
    """
    total = MIN_COVERAGE_CLOUDS + COVERAGE_EVERY + 1
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total)
    # Attempt 12 is the alignment of saved frame 13 - the one that would have
    # carried the count to the interval.
    aligned, built = live(failed_alignments={COVERAGE_EVERY})

    start_capture(window)

    assert len(aligned) == total - 1
    assert len(built) == 2
    # One fewer aligned cloud than frames saved, and the refresh still happened -
    # one frame later than it would have, rather than a dozen.
    assert _notes(window)[1] == (f"preview after {total - 1} aligned frames of {total} saved -")


def test_the_refresh_indices_across_a_long_capture_are_exactly_the_schedule(
    stream_kernel, capture_dir: Path, live
) -> None:
    """Every refresh of a thirty-frame sweep, named, including two failures.

    The owner's 15:50:53 run measured what the old interval cost: eight models
    over the whole sweep, at 3, 15, 27, 39, 51, 63, 75 and 87 aligned frames -
    every one of them applied, none superseded, none dropped, and each one built
    in between 0.008 s and 0.175 s. Nothing was struggling. The picture simply
    changed once every 8-9 seconds, which over four minutes is a view the
    operator cannot tell from a stalled one.

    So the interval is four alignments, and this states the whole resulting
    series rather than a property of it. Two frames are deliberately unhealthy:
    one alignment fails, and one model build fails. Neither is allowed to cost a
    refresh - the first slips the schedule by one frame because the schedule
    counts ALIGNMENTS, and the second is retried on the very next one.
    """
    total = 30
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total)
    # Attempt 9 aligns saved frame 10; build attempt 3 is the refresh due at 11
    # aligned frames.
    aligned, built = live(failed_alignments={9}, failed_models={3})

    start_capture(window)

    assert len(aligned) == total - 1
    assert len(built) == 8
    # Aligned frames at each SUCCESSFUL refresh: three to start, then the series
    # the interval produces. The failed alignment at frame 10 slips every later
    # entry by one saved frame without costing a refresh, because the schedule
    # counts alignments; the failed build at 11 aligned frames costs nothing at
    # all, because the mark only moves on success.
    series = [
        cast(gui.ModelSnapshot, snapshot).aligned for (snapshot,) in window.worker.coverage.emitted
    ]
    assert series == [3, 7, 12, 16, 20, 24, 28]
    # Four alignments apart everywhere except across the failed build, where the
    # retry landed on the very next alignment rather than waiting for the next
    # multiple - which is five, and is the failure mode being bounded, not an
    # interval of five.
    assert [later - earlier for earlier, later in pairwise(series)] == [
        COVERAGE_EVERY,
        COVERAGE_EVERY + 1,
        COVERAGE_EVERY,
        COVERAGE_EVERY,
        COVERAGE_EVERY,
        COVERAGE_EVERY,
    ]
    assert len(window.frames_on_disk()) == total


def test_the_first_model_of_a_capture_frames_the_view_and_the_rest_do_not(
    stream_kernel, capture_dir: Path, live
) -> None:
    """One fit at the start of a capture; every later refresh leaves it alone.

    Re-centring and re-zooming on every snapshot is what made a growing model
    look like a still one: the object was normalised back to the same apparent
    size each time, and whatever the operator had panned or zoomed to was thrown
    away with it. The first model of a capture has no such pose to preserve -
    the previous one described a different object - so it is the boundary.
    """
    total = MIN_COVERAGE_CLOUDS + 2 * COVERAGE_EVERY
    stream_kernel(frames(total))
    window = CaptureWindow(frames=total)
    live()

    start_capture(window)

    assert window._cloud.fits == [True, False, False]


def test_a_first_model_that_fails_to_build_leaves_the_fit_to_its_successor(
    stream_kernel, capture_dir: Path, live
) -> None:
    """The boundary belongs to the first model DRAWN, not the first attempted.

    A build that raised put nothing on the screen, so the framing it would have
    carried has not happened yet and still has to. Marking the boundary spent on
    an attempt would leave the first visible model of a capture drawn through
    the previous capture's camera.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5)
    _aligned, built = live(failed_models={1})

    start_capture(window)

    assert len(built) == 2
    assert window._cloud.fits == [True]


def test_a_failed_model_build_is_retried_on_the_very_next_alignment(
    stream_kernel, capture_dir: Path, live
) -> None:
    """A model that fails to build once must not be a model that never builds.

    The mark of "last refreshed" only moves on success, so the frame after a
    failure is still due one. Advancing it regardless would have made a single
    transient failure permanent for the next dozen frames.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5)
    _aligned, built = live(failed_models={1})

    start_capture(window)

    assert len(built) == 2
    assert _notes(window) == ["preview after 4 aligned frames of 4 saved -"]
    assert window._cloud.clouds != []


def test_both_live_failures_are_recorded_rather_than_swallowed(
    stream_kernel, capture_dir: Path, live, caplog
) -> None:
    """Why the model did not refresh has to be answerable afterwards.

    Both paths deliberately keep capturing - the raw frames are already safe on
    disk and Reconstruct does the real job later - but "keep going" and "say
    nothing" are separable, and only the second one left the owner guessing.
    """
    stream_kernel(frames(5))
    window = CaptureWindow(frames=5)
    live(failed_alignments={2}, failed_models={1})

    with caplog.at_level(logging.WARNING, logger="scanner3d.gui"):
        start_capture(window)

    messages = [record.getMessage() for record in caplog.records]
    assert any("live alignment failed" in message for message in messages)
    assert any("live model failed" in message for message in messages)
    assert any("synthetic alignment failure" in message for message in messages)
    assert any("synthetic coverage failure" in message for message in messages)


def test_a_stage_that_fails_on_every_frame_does_not_flood_the_log(
    stream_kernel, capture_dir: Path, live, caplog
) -> None:
    """A log nobody can read afterwards is the same as no log.

    At the measured 4.6 fps a per-frame warning is thousands of identical lines
    in a file whose only purpose is being read after the fact, so the first few
    carry the reason and the rest are summarised.

    Asked of the MODEL stage rather than the alignment stage, which is the one
    that can still fail on every frame: three consecutive alignment failures now
    end the capture as a tracking loss (see ``test_gui_tracking_loss.py``), so
    alignment can no longer produce a flood for this to throttle. The throttle
    itself is one shared helper and is unchanged.
    """
    stream_kernel(frames(8))
    window = CaptureWindow(frames=8)
    live(failed_models=set(range(1, 9)))

    with caplog.at_level(logging.WARNING, logger="scanner3d.gui"):
        start_capture(window)

    warnings = [record for record in caplog.records if "live model failed" in record.getMessage()]
    assert len(warnings) == gui.FAILURES_REPORTED
    assert window.worker.coverage.emitted == []
    # And the capture itself was untouched by any of it.
    assert len(window.frames_on_disk()) == 8
