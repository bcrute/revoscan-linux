"""What pressing Stop during a capture actually does.

The owner's 2026-08-26 report was that Stop appeared ineffective. The log said
otherwise and both were right: recording ended at 12:48:14 and the preview
stream came straight back up, projector lit, frames flowing, until the window
was closed twenty-five seconds later. From outside the process that is
indistinguishable from a button that does nothing.

Two separate faults produced it, and they are tested separately here:

- A caller stop returns from :func:`~scanner3d.v4l2_stream.stream_frames`
  NORMALLY, with ``stopped_early`` set. The GUI took that return as a finished
  capture, ran its completion path and resumed the preview it had interrupted.
- The interrupted-preview record was read but never cleared, so the intent
  outlived the capture that set it.

Since 2026-09-05 no ending restarts the preview: a capture that finished on its
own used to put it back, and from the bench that read as the scanner never
shutting off. The record of an interrupted preview now serves one sentence in
the capture's ending, and nothing else.

Two boundaries either side of the stream are tested here as well, because both
decide the same question - whether a press counts - and both get it wrong in a
way that is invisible from the report alone: a Stop landing during the wait for
the device must cost no buffer and no file, and a Stop landing after the stream
has already returned normally must not relabel a finished capture as one the
operator ended.

Everything below drives the real capture path against the scripted fake driver:
the real ``_start_capture``, its real ``on_frame``, the real dequeue loop and
whichever real terminal path the job ends in. The stop itself is the real
:meth:`~scanner3d.gui.ScannerWindow._stop`, fired from a slot on the saved-frame
signal - which is as close as an offline test gets to a hand on the button while
frames are arriving.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES
from scanner3d.v4l2_stream import CALLER_STOP_REASON, StreamResult
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import CaptureWindow, capture_dir, deprojected, start_capture

__all__ = ["capture_dir", "deprojected", "stream_kernel", "stream_node"]


def _stop_after(window: CaptureWindow, frames: int) -> None:
    """Press Stop once, the moment the ``frames``-th frame reaches disk."""

    def pressed(saved: int) -> None:
        if saved == frames:
            window._stop()

    window.worker.saved.connect(pressed)


def _three_frames() -> tuple[object, ...]:
    return (
        whole_frame(0xA1, buffer_index=0),
        whole_frame(0xA2, buffer_index=1),
        whole_frame(0xA3, buffer_index=2),
    )


def test_stop_ends_the_capture_stopped_rather_than_completed(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The stopped outcome, not the completion one.

    ``completed`` empty is the load-bearing assertion: before this change the
    stopped run went through ``_capture_done``, which is what printed a normal
    completion and put the preview back.
    """
    kernel = stream_kernel(_three_frames())
    window = CaptureWindow(frames=3)
    _stop_after(window, 2)

    start_capture(window)

    assert window.completed == []
    assert window.stopped == 1
    assert "Stopped. 2 frames were already written" in window.report
    assert "Captured" not in window.report
    assert "Press Reconstruct" not in window.report
    assert window.busy == [True, False]
    # The run really did end early rather than merely being relabelled: the
    # third scripted delivery was never taken.
    assert len(kernel.dequeued) == 2


def test_stop_leaves_every_frame_already_written_whole_and_reconstructable(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Stopping abandons the rest of the sweep, not the part already captured."""
    stream_kernel(_three_frames())
    window = CaptureWindow(frames=3)
    _stop_after(window, 2)

    start_capture(window)

    written = window.frames_on_disk()
    assert [path.name for path in written] == ["frame_0000.raw", "frame_0001.raw"]
    assert [path.stat().st_size for path in written] == [EXPECTED_FRAME_BYTES] * 2
    assert [path.read_bytes()[0] for path in written] == [0xA1, 0xA2]
    assert "safe to reconstruct" in window.report


def test_stop_does_not_put_the_preview_back_up(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The whole reported symptom, asserted directly.

    Preview was running before the capture, so the interrupted-preview intent is
    set. The capture is stopped. Nothing may re-tick that action, and the report
    has to say so - a live view that silently does not come back is its own small
    mystery.
    """
    stream_kernel(_three_frames())
    window = CaptureWindow(frames=3, previewing=True)
    _stop_after(window, 1)

    start_capture(window)

    # One setChecked, the False that released the node for the capture.
    assert window._preview_action.checks == [False]
    assert window._preview_action.checked is False
    assert window._preview_interrupted is False
    assert "Preview was NOT restarted" in window.report


def test_the_stream_hands_the_buffer_back_before_the_stop_takes_effect(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Stopping must not cost the driver a buffer.

    This is why the GUI stops through the flag and converts the result
    afterwards rather than raising out of ``on_frame``: an exception there skips
    the ``VIDIOC_QBUF`` for the buffer it was handed, and this hardware answers a
    buffer still owned by userspace at teardown by re-enumerating.
    """
    kernel = stream_kernel(_three_frames())
    window = CaptureWindow(frames=3)
    _stop_after(window, 2)

    start_capture(window)

    # Both dequeued buffers went back, including the one being held when Stop
    # was pressed.
    assert kernel.requeued == [0, 1]
    assert len(kernel.requeued) == len(kernel.dequeued)


def test_a_capture_that_finishes_on_its_own_leaves_the_preview_off_like_stop(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """A completed scan leaves the scanner idle and says so.

    The preview used to come back here, projector lit, and from the bench
    that read as the scanner never shutting off after a capture ran its
    course (2026-09-05). The one setChecked is the False that released the
    node for the capture.
    """
    stream_kernel((whole_frame(0xB1), whole_frame(0xB2, buffer_index=1)))
    window = CaptureWindow(frames=2, previewing=True)

    start_capture(window)

    assert window._preview_action.checks == [False]
    assert window._preview_action.checked is False
    assert window._preview_interrupted is False
    assert isinstance(window.result, StreamResult)
    assert window.result.stopped_early is False
    assert "Captured 2 frames" in window.report
    assert "Preview was NOT restarted" in window.report


def test_a_completed_capture_that_was_not_previewing_does_not_start_one(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Resuming means resuming. It never means starting something new."""
    stream_kernel((whole_frame(0xB3),))
    window = CaptureWindow(frames=1, previewing=False)

    start_capture(window)

    assert window._preview_action.checks == [False]
    assert window._preview_action.checked is False


def test_the_resume_intent_is_cleared_by_a_completion(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Read once, and gone. Left set it belongs to no capture in particular."""
    stream_kernel((whole_frame(0xB4),))
    window = CaptureWindow(frames=1, previewing=True)

    start_capture(window)

    assert window._preview_interrupted is False
    # A second terminal path, arriving later, finds nothing to act on.
    window._preview_action.checks.clear()
    window._capture_done(window.result)
    assert window._preview_action.checks == []


def test_the_resume_intent_is_cleared_by_a_failure(monkeypatch) -> None:
    """A failed capture is not the moment to reopen the device.

    Nor is the job after it: before this change the intent survived the failure
    and the next terminal path found it still set.
    """
    monkeypatch.setattr(gui, "QMessageBox", SimpleNamespace(critical=lambda *_a: None))
    window = CaptureWindow(frames=1, previewing=True)
    window._preview_interrupted = True

    window._fail("Traceback...\nSomethingBroke: no")

    assert window._preview_interrupted is False
    assert window._preview_action.checks == []
    assert "FAILED" in window.report


def test_the_resume_intent_is_cleared_by_a_stop() -> None:
    """Consumed on the stopped path too, and deliberately not acted on."""
    window = CaptureWindow(frames=1, previewing=True)
    window._preview_interrupted = True

    window._was_stopped()

    assert window._preview_interrupted is False
    assert window._preview_action.checks == []


def test_pressing_stop_twice_asks_the_job_once(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """A second press cannot make the safe boundary arrive sooner.

    Cancelling again would be harmless; saying "Stopping at the next frame..."
    again would not be, because the second line reads as a second thing having
    happened.
    """
    stream_kernel(_three_frames())
    window = CaptureWindow(frames=3)

    def pressed(saved: int) -> None:
        if saved == 1:
            window._stop()
            window._stop()

    window.worker.saved.connect(pressed)

    start_capture(window)

    assert window.worker.cancels == 1
    assert window.report.count("Stopping at the next frame") == 1


def test_a_stop_during_the_device_wait_never_opens_the_stream_at_all(
    monkeypatch, stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Stop pressed before the first buffer costs the scanner nothing.

    The capture job's first act is to wait for the previous stream to let the
    node go and for the device to come back on the bus, which on this hardware
    is seconds - easily long enough to press Stop inside. Answering that by
    opening the node, taking one buffer and writing one frame is a capture the
    operator has already abandoned, and a folder holding a frame they never
    asked for.

    Raising is safe at this exact point and nowhere later inside the stream: no
    V4L2 buffer is owned yet, so there is no ``VIDIOC_QBUF`` to skip.
    """
    kernel = stream_kernel(_three_frames())
    window = CaptureWindow(frames=3)

    def wait_then_stopped(serial: str | None) -> None:
        window.awaited += 1
        window.awaited_serials.append(serial)
        # The real Stop, on the real window method, landing inside the wait.
        window._stop()

    monkeypatch.setattr(window, "_await_device", wait_then_stopped)

    start_capture(window)

    assert window.awaited == 1
    assert window.stopped == 1
    assert window.completed == []
    # Nothing was streamed: no STREAMON, no mapping, no buffer taken, and so
    # none owed back.
    assert kernel.streaming is False
    assert kernel.map_calls == []
    assert kernel.dequeued == []
    assert kernel.requeued == []
    # And nothing was written or read back for the live model.
    assert window.frames_on_disk() == []
    assert deprojected == []
    assert "Stopped. 0 frames were already written" in window.report
    assert "Captured" not in window.report


def test_a_flag_set_after_a_normal_stream_return_is_still_a_completion(
    monkeypatch, stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The stream's own answer decides the outcome. Only ever the stream's.

    There is a narrow interval between ``stream_frames`` returning normally and
    the job choosing which terminal path it ended in, and a Stop can land in it.
    Consulting ``worker.cancelled`` there would relabel this capture - which ran
    to its full frame count with the device torn down cleanly - as stopped, and
    print "frames were already written" about a sweep that in fact finished.

    The stream polls ``should_stop`` after every re-queue, so a stop it could
    have acted on is a stop it reports in ``stopped_early``. One it did not
    report did not happen in time, and this is a completion.
    """
    from scanner3d import v4l2_stream

    stream_kernel((whole_frame(0xC1), whole_frame(0xC2, buffer_index=1)))
    window = CaptureWindow(frames=2, previewing=True)
    real = v4l2_stream.stream_frames

    def late_stop(**kwargs):
        result = real(**kwargs)
        # Exactly the interval in question: returned, device released, and only
        # then does the flag turn true.
        window.worker.cancelled = True
        return result

    monkeypatch.setattr(v4l2_stream, "stream_frames", late_stop)

    start_capture(window)

    assert window.stopped == 0
    assert isinstance(window.result, StreamResult)
    assert window.result.stopped_early is False
    assert window.result.stop_reason != CALLER_STOP_REASON
    assert window.completed == [window.result]
    assert "Captured 2 frames" in window.report
    assert "Stopped." not in window.report
    # A completion in every respect, including the one Stop would have denied.
    # Completion and Stop both leave the preview off now; what tells them apart
    # is the ending's own words and record, not the state of the scanner.
    assert window._preview_action.checks == [False]
    assert window.stopped == 0
    assert "Captured" in window.report
