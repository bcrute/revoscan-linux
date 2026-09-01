"""The depth panel, built by the real window, asked what it shows about a frame.

Why this one needs Qt
----------------------
Almost nothing in this suite does - :mod:`tests.gui_harness` exists so that
capture logic can be asked without an event loop. But the report this work
answers was about what the panel LOOKED like: on 2026-08-28 the owner read a
depth preview that had gone almost entirely black as "the preview has stopped
showing the object", when the frames really had stopped returning depth. So the
question here is a question about widgets - what is on the screen beside the
picture, how loud it is, and whether it goes away when the frame comes back -
and it is asked of the real :meth:`~scanner3d.gui.ScannerWindow._build_depth_panel`
and the real :meth:`~scanner3d.gui.ScannerWindow._show_frame`, on the offscreen
platform :mod:`tests.qt_offscreen` describes.

What is deliberately NOT here
------------------------------
The panel stand-in below carries exactly the attributes the depth path touches
and nothing else, in the way :mod:`tests.gui_harness` does: a
:meth:`~scanner3d.gui.ScannerWindow._show_frame` that reached for the frame
counter, a worker, the model panel or a capture directory would raise
``AttributeError`` here rather than pass quietly. That is this file's evidence
that the return-health line changed nothing about saved-frame cadence,
``LatestOnly`` accounting, tracking decisions or model publication; the
behaviour of all four is asked, unchanged, by their own neighbours
(``test_gui_preview_cadence.py``, ``test_gui_live_model.py``,
``test_gui_tracking_loss.py``).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

from scanner3d import gui
from scanner3d.gui import ScannerWindow
from scanner3d.return_health import (
    COLLAPSING,
    NO_RETURNS,
    ReturnHealthTracker,
)
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH
from scanner3d.viewer import FrameStats
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    PreviewWindow,
    PreviewWorker,
    capture_dir,
    deprojected,
    previewing,
    start_capture,
)
from tests.qt_offscreen import qt_app

__all__ = [
    "capture_dir",
    "deprojected",
    "previewing",
    "qt_app",
    "stream_kernel",
    "stream_node",
]


def make_frame(returns: int, *, code: int = 5000, one_sided: bool = False) -> bytes:
    """A whole frame with exactly ``returns`` returning pixels.

    ``one_sided`` puts them all in the left half, which is the shape the
    physical capture ended in; otherwise they run along the top rows.
    """
    samples = [0] * (FRAME_WIDTH * FRAME_HEIGHT)
    written = 0
    for row in range(FRAME_HEIGHT):
        for column in range(FRAME_WIDTH // 2 if one_sided else FRAME_WIDTH):
            if written == returns:
                break
            samples[row * FRAME_WIDTH + column] = code
            written += 1
        if written == returns:
            break
    assert written == returns
    return struct.pack(f"<{len(samples)}H", *samples)


ALL_ZERO_FRAME = make_frame(0)
#: A frame with plenty of geometry in it: nothing to warn about.
HEALTHY_FRAME = make_frame(FRAME_WIDTH * FRAME_HEIGHT // 3)


class DepthPanel:
    """The real depth panel and the real slots that write it, and nothing else.

    Every method below is :class:`~scanner3d.gui.ScannerWindow`'s own, called
    unbound against this object exactly as :mod:`tests.gui_harness` calls the
    capture path's. The attribute list is what the depth path is allowed to
    touch; anything else it reached for would raise ``AttributeError``.
    """

    #: Named by the real builder, not by this class.
    _view: QLabel
    _return_label: QLabel
    _return_advice: QLabel
    _frame_label: QLabel

    def __init__(self) -> None:
        self._return_health = ReturnHealthTracker()
        self._return_verdict: str | None = None
        #: The run identity the real window keeps, so the run boundaries and
        #: the stale-frame guard are asked of the real methods here too.
        self._depth_run = 0
        self.panel: QWidget = ScannerWindow._build_depth_panel(cast(ScannerWindow, self))

    def _show_frame(self, raw: bytes) -> None:
        """The REAL one, reached the way a worker connection reaches it."""
        ScannerWindow._show_frame(cast(ScannerWindow, self), raw)

    def show_frame(self, raw: bytes) -> None:
        self._show_frame(raw)

    def _show_return_health(self, stats: FrameStats) -> None:
        """The REAL one, reached the way the real window reaches it."""
        ScannerWindow._show_return_health(cast(ScannerWindow, self), stats)

    def _reset_return_health(self) -> None:
        """The REAL one, reached the way the run boundaries reach it."""
        ScannerWindow._reset_return_health(cast(ScannerWindow, self))

    def reset(self) -> None:
        self._reset_return_health()

    def begin_run(self) -> None:
        ScannerWindow._begin_depth_run(cast(ScannerWindow, self))

    def end_run(self) -> None:
        ScannerWindow._end_depth_run(cast(ScannerWindow, self))

    def frame_slot(self):
        """A depth slot bound to the CURRENT run, as a worker connection is."""
        return ScannerWindow._frame_slot(cast(ScannerWindow, self), self._depth_run)


@pytest.fixture
def panel(qt_app: QApplication) -> DepthPanel:
    del qt_app  # the application must exist before a widget is built
    return DepthPanel()


def test_a_frame_with_no_returns_puts_a_loud_warning_beside_the_picture(panel: DepthPanel):
    panel.show_frame(ALL_ZERO_FRAME)
    assert panel._return_label.text() == "NO DEPTH SIGNAL — 0.0% returned"
    assert panel._return_label.styleSheet() == gui.LOUD_RETURN_STYLE
    assert panel._return_advice.text() == (
        "Stop moving. Turn back toward the last good view and aim the surface more squarely."
    )
    assert "returned no depth" in panel._return_advice.toolTip()
    # ...and the ordinary numeric line is still there, unchanged.
    assert "returns 0 of 256000" in panel._frame_label.text()


def test_the_warning_names_where_the_returns_actually_are(panel: DepthPanel):
    """One-sided collapse, named as one: the physical capture's own shape."""
    panel.show_frame(make_frame(2000, one_sided=True))
    text = panel._return_label.toolTip()
    assert "right" in text
    assert "columns 0-" in text


def test_a_sparse_frame_is_a_warning_and_a_healthy_one_is_silent(panel: DepthPanel):
    panel.show_frame(make_frame(1000))
    assert "DEPTH SIGNAL TOO LOW" in panel._return_label.text()

    panel.show_frame(HEALTHY_FRAME)
    assert panel._return_label.text() == ""
    assert panel._return_advice.text() == ""
    assert panel._return_label.styleSheet() == gui.QUIET_RETURN_STYLE


def test_a_recovered_frame_visibly_clears_the_warning(panel: DepthPanel):
    for _ in range(4):
        panel.show_frame(HEALTHY_FRAME)
    # A tenth of what the last four frames returned, and still well clear
    # of the severe floor, so the verdict is about the FALL rather than
    # about how little is left.
    panel.show_frame(make_frame(FRAME_WIDTH * FRAME_HEIGHT // 10))
    assert COLLAPSING.upper() in panel._return_label.text()
    assert panel._return_label.styleSheet() == gui.LOUD_RETURN_STYLE

    panel.show_frame(HEALTHY_FRAME)
    assert panel._return_label.text() == ""
    assert panel._return_label.styleSheet() == gui.QUIET_RETURN_STYLE


def test_the_warning_is_a_label_and_never_a_modal_dialog(panel: DepthPanel, monkeypatch):
    """It must not take a click, or the focus, from someone holding the scanner."""

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("the depth panel opened a dialog")

    monkeypatch.setattr(QMessageBox, "critical", refuse)
    monkeypatch.setattr(QMessageBox, "warning", refuse)
    monkeypatch.setattr(QMessageBox, "information", refuse)
    panel.show_frame(ALL_ZERO_FRAME)
    assert "NO DEPTH SIGNAL" in panel._return_label.text()


def test_the_picture_and_the_sentence_describe_the_same_frame(panel: DepthPanel):
    """The pixmap really is redrawn on the frame the warning is about."""
    panel.show_frame(HEALTHY_FRAME)
    first = panel._view.pixmap().cacheKey()
    panel.show_frame(ALL_ZERO_FRAME)
    assert panel._view.pixmap().cacheKey() != first
    assert "NO DEPTH SIGNAL" in panel._return_label.text()


def test_the_depth_path_reaches_for_nothing_but_the_depth_panel(panel: DepthPanel):
    """No counter, worker, capture directory or model panel is touched.

    The stand-in has none of them, so anything the slot reached for would raise
    ``AttributeError`` here.
    """
    for frame in (HEALTHY_FRAME, ALL_ZERO_FRAME, HEALTHY_FRAME):
        panel.show_frame(frame)
    assert not hasattr(panel, "_saved_frames")
    assert not hasattr(panel, "_worker")
    assert not hasattr(panel, "_cloud")


def test_a_reset_clears_both_the_warning_and_the_history(panel: DepthPanel):
    panel.show_frame(ALL_ZERO_FRAME)
    assert panel._return_health.frames == 1
    panel.reset()
    assert panel._return_health.frames == 0
    assert panel._return_label.text() == ""
    assert panel._return_advice.text() == ""
    assert panel._return_label.styleSheet() == gui.QUIET_RETURN_STYLE


def test_the_log_records_transitions_rather_than_a_line_per_frame(panel: DepthPanel, caplog):
    """Ten frames a second of "still empty" would bury the moment it changed."""
    with caplog.at_level("INFO", logger=gui.LOG.name):
        for _ in range(6):
            panel.show_frame(ALL_ZERO_FRAME)
        for _ in range(6):
            panel.show_frame(HEALTHY_FRAME)
    lines = [record for record in caplog.records if "return health" in record.getMessage()]
    assert len(lines) == 2
    assert NO_RETURNS in lines[0].getMessage()


# --- run boundaries: one history per preview and per capture ------------------


def test_starting_a_preview_forgets_the_previous_runs_return_history(previewing):
    """A preview must not open under the last run's collapse.

    Driven through the real :meth:`~scanner3d.gui.ScannerWindow._start_preview`
    against :mod:`tests.gui_harness`'s stand-in, which carries the real
    tracker: a start that dropped the reset would leave those three frames in
    the window and judge this run's first frame against them.
    """
    window = PreviewWindow()
    for _ in range(3):
        window._return_health.update(0.02)
    window._return_label.setText("NO RETURNS: from the last run")
    assert window._return_health.frames == 3

    previewing(window)

    assert window._return_health.frames == 0
    assert window._return_health.baseline is None
    assert window._return_label.text == ""


def test_starting_a_capture_forgets_the_previous_runs_return_history(
    stream_kernel, capture_dir: Path, deprojected: list[int]
):
    del capture_dir, deprojected  # named to bring the capture path's own fixtures in
    stream_kernel((whole_frame(0xA1),))
    window = CaptureWindow(frames=1)
    for _ in range(3):
        window._return_health.update(0.02)
    window._return_label.setText("NO RETURNS: from the last run")

    start_capture(window)

    assert window._return_health.frames == 0
    assert window._return_health.baseline is None
    assert window._return_label.text == ""


# --- run ENDS, and frames that arrive after one ------------------------------
#
# Round 1 reset the history where a run began. That left two gaps the round-1
# review found: a warning raised by a preview stayed on screen after the
# preview had stopped, and a depth frame already queued by that preview could
# be delivered after the capture replacing it had started - redrawing the new
# run's panel and becoming the first entry in its baseline. The frames below
# are injected explicitly; nothing here waits for, or races, a thread.


def test_ending_a_run_clears_the_warning_it_raised(panel: DepthPanel):
    """A stopped stream's warning must not sit there looking current."""
    panel.show_frame(ALL_ZERO_FRAME)
    assert "NO DEPTH SIGNAL" in panel._return_label.text()

    panel.end_run()

    assert panel._return_label.text() == ""
    assert panel._return_advice.text() == ""
    assert panel._return_label.styleSheet() == gui.QUIET_RETURN_STYLE
    assert panel._return_health.frames == 0


def test_a_frame_queued_by_an_ended_run_neither_redraws_nor_advances_health(
    panel: DepthPanel,
):
    """The stale delivery itself, injected rather than hoped for."""
    stale_slot = panel.frame_slot()
    stale_slot(HEALTHY_FRAME)
    assert panel._return_health.frames == 1
    drawn = panel._view.pixmap().cacheKey()

    panel.end_run()
    stale_slot(ALL_ZERO_FRAME)

    assert panel._view.pixmap().cacheKey() == drawn
    assert panel._return_health.frames == 0
    assert panel._return_label.text() == ""


def test_a_stale_preview_frame_cannot_become_the_next_runs_first_baseline(
    panel: DepthPanel,
):
    """The exact crossing: preview stops, capture starts, the old frame lands."""
    stale_slot = panel.frame_slot()
    panel.end_run()  # the preview stopped
    panel.begin_run()  # the capture started
    current_slot = panel.frame_slot()

    stale_slot(ALL_ZERO_FRAME)
    assert panel._return_health.frames == 0
    assert panel._return_label.text() == ""

    # The capture's own frames still arrive normally.
    current_slot(HEALTHY_FRAME)
    assert panel._return_health.frames == 1


def test_stopping_a_preview_clears_its_warning_and_history(previewing):
    """Manual stop: the toolbar's Preview untick, and what a capture does first."""
    window = PreviewWindow()
    previewing(window)
    window._return_health.update(0.02)
    window._return_label.setText("NO RETURNS: from the run that just stopped")

    ScannerWindow._stop_preview(cast(ScannerWindow, window))

    assert window._return_health.frames == 0
    assert window._return_label.text == ""


def test_a_preview_that_ends_on_its_own_or_fails_clears_its_warning():
    """The ceiling and the failure, which are not the operator's Stop."""
    for end in ("ended", "failed"):
        window = PreviewWindow()
        window._return_health.update(0.02)
        window._return_label.setText("NO RETURNS: from a stream that is gone")
        if end == "ended":
            window._preview_ended("the stream ended")
        else:
            window._preview_failed("the device went away")
        assert window._return_health.frames == 0, end
        assert window._return_label.text == "", end


def test_a_preview_completion_delivered_after_a_new_run_started_is_ignored(previewing):
    """The exact crossing for the TERMINAL signals, not the frame ones.

    Qt queues ``finished_ok`` just as it queues ``preview``, so a preview that
    hit its ceiling in the instant before Stop has its completion delivered
    after the capture that replaced it has begun. Unguarded, that stale
    callback ends the live run: it moves the run identity on - which then makes
    the capture's own frames stale and drops them - clears the capture's
    warning, and unticks Preview against a stream it does not belong to.
    """
    window = PreviewWindow()
    previewing(window)
    worker = cast(PreviewWorker, window._preview_worker)
    # The preview above ran and ended; this is the capture, or the next
    # preview, that starts after it.
    window._begin_depth_run()
    run = window._depth_run
    current_slot = window._frame_slot(run)
    window._return_health.update(0.4)
    window._return_label.setText("the live run's own warning")
    window._preview_action.setChecked(True)
    endings = len(window.ended)
    window.frames.clear()

    worker.finished_ok.emit("a completion queued by the preview that already stopped")

    # The live run is untouched: identity, history, warning and the button.
    assert window._depth_run == run
    assert len(window.ended) == endings
    assert window._return_health.frames == 1
    assert window._return_label.text == "the live run's own warning"
    assert window._preview_action.isChecked() is True

    # ...and its frame slot is still live, which is the failure this guards.
    current_slot(HEALTHY_FRAME)
    assert window.frames == [HEALTHY_FRAME]


def test_a_preview_failure_delivered_after_a_new_run_started_is_ignored(previewing):
    """The same crossing for ``failed``, which also carries a message.

    The news is not thrown away - it is logged - but a device that went away on
    the run BEFORE this one does not end this one, empty its warning or put a
    failure dialog's text into the live run's report.
    """
    window = PreviewWindow()
    previewing(window)
    worker = cast(PreviewWorker, window._preview_worker)
    window._begin_depth_run()
    run = window._depth_run
    current_slot = window._frame_slot(run)
    window._return_health.update(0.4)
    window._return_label.setText("the live run's own warning")
    window._preview_action.setChecked(True)
    window.frames.clear()

    worker.failed.emit("Traceback (most recent call last):\nOSError: the device went away")

    assert window.failures == []
    assert window._depth_run == run
    assert window._return_health.frames == 1
    assert window._return_label.text == "the live run's own warning"
    assert window._preview_action.isChecked() is True
    assert "Preview failed" not in window.report

    current_slot(HEALTHY_FRAME)
    assert window.frames == [HEALTHY_FRAME]


def test_a_preview_ending_in_its_own_run_still_ends_it(previewing):
    """The guard must not deafen the ordinary case: this is its positive control."""
    window = PreviewWindow()
    previewing(window)

    # The fixture's worker runs its job synchronously, so the preview's own
    # completion was delivered inside its own run, above.
    assert window.ended == ["the stream ended"]
    assert window._preview_action.isChecked() is False
    assert window._return_health.frames == 0
    assert window._return_label.text == ""


def test_a_frame_emitted_after_a_preview_stopped_never_reaches_the_window(previewing):
    """The worker's own channel, emitted after the stop, delivered to nothing."""
    window = PreviewWindow()
    previewing(window)
    worker = cast(PreviewWorker, window._preview_worker)
    assert window.frames == [b"a depth frame"]

    ScannerWindow._stop_preview(cast(ScannerWindow, window))
    window.frames.clear()
    worker.preview.emit(b"a frame the stopped preview had already produced")

    assert window.frames == []
    assert window._return_health.frames == 0


def test_a_capture_drops_a_frame_belonging_to_the_preview_it_replaced(
    stream_kernel, capture_dir: Path, deprojected: list[int]
):
    del capture_dir, deprojected  # named to bring the capture path's own fixtures in
    stream_kernel((whole_frame(0xA1),))
    window = CaptureWindow(frames=1)
    # The slot the preview running before this capture was connected through.
    stale_slot = window._frame_slot(window._depth_run)

    start_capture(window)
    window.depth_frames.clear()
    stale_slot(ALL_ZERO_FRAME)

    assert window.depth_frames == []


def test_a_capture_that_ends_clears_its_return_warning(
    stream_kernel, capture_dir: Path, deprojected: list[int]
):
    """Completion, the operator's Stop and a failure all end the run."""
    del capture_dir, deprojected
    stream_kernel((whole_frame(0xA1),))
    window = CaptureWindow(frames=1)
    start_capture(window)
    window._return_health.update(0.02)
    window._return_label.setText("SEVERELY SPARSE: from the capture that just ended")

    ScannerWindow._capture_done(cast(ScannerWindow, window), None)

    assert window._return_health.frames == 0
    assert window._return_label.text == ""
