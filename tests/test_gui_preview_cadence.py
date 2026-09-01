"""Whether the depth view moves at the rate the scanner delivers, not the rate we keep.

Decimation exists because a full turntable revolution is several hundred frames
and over a quarter of a gigabyte, and because alignment wants a few degrees
between the frames it is given. None of those reasons is a reason to show the
operator fewer pictures: at the default ``decimate=3`` the depth panel updated
once for every three frames the scanner actually delivered, so a healthy stream
looked like a scanner running at a third of its rate - next to a model panel
that, until this task, refreshed even more slowly than that.

So the two cadences are now separate. Every COMPLETE delivery is offered to the
depth preview; only the ``decimate``-selected ones are written, deprojected,
aligned and counted. A short delivery is offered to neither, because 70% of a
frame is not a picture any more than it is a frame.

Everything here drives the real capture path - the real ``_start_capture``, its
real callbacks, the real :func:`~scanner3d.v4l2_stream.stream_frames` dequeue
loop against the scripted fake driver - and nothing here opens a device or
builds a widget.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")

from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES
from scanner3d.v4l2_stream import StreamResult
from tests.fake_v4l2 import Delivery, short_frame, stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    PreviewWindow,
    alignments,
    capture_dir,
    clouds,
    deprojected,
    live,
    models,
    previewing,
    start_capture,
)

__all__ = [
    "alignments",
    "capture_dir",
    "clouds",
    "deprojected",
    "live",
    "models",
    "previewing",
    "stream_kernel",
    "stream_node",
]


def frames(count: int) -> tuple[Delivery, ...]:
    """``count`` whole frames, each with its own fill byte.

    Distinct fills because "the same picture shown three times" and "three
    pictures" are the two outcomes these tests have to tell apart.
    """
    return tuple(whole_frame(0x40 + index, buffer_index=index % 4) for index in range(count))


def _previewed(window: CaptureWindow) -> list[bytes]:
    return [cast(bytes, payload) for (payload,) in window.worker.preview.emitted]


def _saved(window: CaptureWindow) -> list[int]:
    return [cast(int, count) for (count,) in window.worker.saved.emitted]


def test_a_decimated_capture_previews_every_delivery_and_saves_one_in_three(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Nine deliveries, nine pictures, three files.

    The two counts are the whole subject. Nine says the operator sees the depth
    camera at the rate it is running; three says the disk, the deprojection and
    the alignment are still doing a third of the work, which is what they were
    asked to do.
    """
    stream_kernel(frames(9))
    window = CaptureWindow(frames=9, decimate=3)

    start_capture(window)

    assert len(_previewed(window)) == 9
    assert _saved(window) == [1, 2, 3]
    assert [path.name for path in window.frames_on_disk()] == [
        "frame_0000.raw",
        "frame_0001.raw",
        "frame_0002.raw",
    ]
    # Only the kept frames were read back off disk for the live model.
    assert deprojected == [EXPECTED_FRAME_BYTES] * 3
    assert isinstance(window.result, StreamResult)
    assert window.result.delivered == 9
    assert window.result.kept == 3


def test_the_pictures_shown_are_the_frames_that_arrived_in_the_order_they_arrived(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """Each preview is its own delivery, not a repeat of the last kept one.

    A view fed the same payload three times would satisfy a count and still be
    frozen, which is precisely the failure this is correcting - so the bytes are
    compared, not counted.
    """
    deliveries = frames(6)
    stream_kernel(deliveries)
    window = CaptureWindow(frames=6, decimate=3)

    start_capture(window)

    assert _previewed(window) == [delivery.payload for delivery in deliveries]
    # And what reached the disk is the two the decimation selected, whole.
    assert [path.read_bytes()[0] for path in window.frames_on_disk()] == [0x40, 0x43]


def test_a_short_delivery_is_shown_to_nobody_and_written_nowhere(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The owner's 2026-08-26 buffer, on the display path this time.

    358402 bytes of an expected 512000 is a transport event. It is counted and
    re-queued, and now that the preview follows deliveries rather than keeps it
    is exactly as excluded from the picture as it always was from the disk.
    """
    stream_kernel(
        (
            short_frame(),
            whole_frame(0x51, buffer_index=1),
            short_frame(fill=0x22, buffer_index=2),
            whole_frame(0x53, buffer_index=3),
            whole_frame(0x55, buffer_index=0),
        )
    )
    window = CaptureWindow(frames=5, decimate=2)

    start_capture(window)

    previewed = _previewed(window)
    assert [len(payload) for payload in previewed] == [EXPECTED_FRAME_BYTES] * 3
    assert [payload[0] for payload in previewed] == [0x51, 0x53, 0x55]
    # Deliveries 0 and 2 were selected and short, so they were dropped; 1 and 3
    # were whole and unselected, so they were shown and never written. Only
    # delivery 4 was both, which is the split this test exists to prove.
    assert [path.read_bytes()[0] for path in window.frames_on_disk()] == [0x55]
    assert deprojected == [EXPECTED_FRAME_BYTES]
    assert isinstance(window.result, StreamResult)
    assert window.result.short_frames == 2


def test_a_stop_pressed_on_a_delivery_that_is_never_kept_is_still_clean(
    stream_kernel, capture_dir: Path, deprojected: list[int]
) -> None:
    """The buffer goes back to the driver before the Stop is acted on.

    The preview slot is now reached on frames the capture will never write, so
    it is a place the operator's Stop can land that did not exist before. It
    must behave like every other Stop: the stream ends at its own boundary with
    every buffer returned, and this hardware's answer to a buffer held at
    teardown - re-enumerating, which looks exactly like the scanner being
    unplugged - never happens.
    """
    kernel = stream_kernel(frames(6))
    window = CaptureWindow(frames=6, decimate=3)

    def stop_on_the_second_picture(_payload: bytes) -> None:
        """Stop from delivery 1, which decimation does not select."""
        if len(_previewed(window)) == 2:
            window.worker.cancel()

    window.worker.preview.connect(stop_on_the_second_picture)

    start_capture(window)

    assert window.stopped == 1
    assert kernel.requeued == [0, 1]
    # The first delivery was kept and written; the second was shown only.
    assert len(window.frames_on_disk()) == 1
    assert "Stopped. 1 frames were already written" in window.report


def test_the_preview_stream_needs_no_second_callback_at_all(previewing) -> None:
    """The compatibility case, which is the other half of this window.

    Preview runs at ``decimate=1``, where every delivery is a keep, so it passes
    no all-deliveries callback and nothing about it changes. A preview that had
    to be rewritten to keep working would be a sign the new callback was doing
    the old one's job rather than a different one.
    """
    window = PreviewWindow()

    calls = previewing(window)

    assert [call.decimate for call in calls] == [1]
    assert [call.on_delivery for call in calls] == [None]
    assert window.frames == [b"a depth frame"]
