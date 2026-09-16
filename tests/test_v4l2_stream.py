"""What the continuous depth stream does with the buffers the driver hands back.

The question here is the dequeue loop, and specifically the one thing it judges:
LENGTH. On 2026-08-26 the owner started a scan and the very first buffer came
back with ``bytesused`` 358402 against the 512000 a whole depth frame is. It was
written to disk and deprojected anyway, ``DepthFrameSizeMismatch`` was raised
inside the capture job, and the scan died at startup. That exact byte count is
reproduced below, because a regression written against a round number would not
have caught it and would not prove it was caught.

Nothing here opens a device. The scripted driver is
:class:`~tests.fake_v4l2.StreamKernel`, which extends the shared
:class:`~tests.fake_v4l2.FakeKernel` so the ``struct v4l2_buffer`` this module
reads its ``bytesused``, ``sequence`` and ``index`` out of is packed by the same
code every other V4L2 lifecycle test uses - a second hand-rolled packer here
would be a second set of assumptions about field offsets, and the offsets are
exactly the kind of thing that reads a plausible number from the wrong field.

The seams are narrow and named: ``fcntl`` and ``mmap`` are replaced on the
module under test, so the substitution cannot leak into the real modules another
test in the same process is using. ``os.open`` and ``os.close`` are NOT faked -
the node is a real file in ``tmp_path``, so descriptor ownership and the
``finally`` block's close are the real ones.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator

import pytest

from scanner3d import v4l2_stream
from scanner3d.v4l2_capture import (
    EXPECTED_FRAME_BYTES,
    VIDIOC_DQBUF,
    VIDIOC_QBUF,
    VIDIOC_REQBUFS,
    VIDIOC_STREAMOFF,
)
from scanner3d.v4l2_stream import (
    CALLER_STOP_REASON,
    FrameRecord,
    StreamRequestUnusable,
    StreamResult,
    stream_frames,
)
from tests.fake_v4l2 import (
    OWNER_SHORT_FRAME_BYTES,
    Delivery,
    StreamKernel,
    short_frame,
    stream_kernel,
    stream_node,
    whole_frame,
)

__all__ = ["stream_kernel", "stream_node"]


def _run(
    wire,
    deliveries: tuple[Delivery, ...],
    **kwargs: object,
) -> tuple[StreamKernel, list[tuple[bytes, FrameRecord]], StreamResult]:
    """Stream exactly ``deliveries``, recording every callback it produced."""
    kernel = wire(deliveries)
    seen: list[tuple[bytes, FrameRecord]] = []

    def on_frame(payload: bytes, record: FrameRecord) -> None:
        seen.append((payload, record))

    result = stream_frames(
        frames=len(deliveries),
        on_frame=on_frame,
        **kwargs,  # type: ignore[arg-type]
    )
    return kernel, seen, result


# -- the owner's frame ------------------------------------------------------


def test_the_owners_short_startup_frame_never_reaches_the_callback(stream_kernel) -> None:
    """The reproduction: 358402 bytes at STREAMON, and nothing consumes them.

    Every consumer downstream - the GUI's ``frame_*.raw`` write, its preview,
    its live deprojection, the CLI sweep's write - is reached through this one
    callback. Proving it is not called is proving none of them ran.
    """
    kernel, seen, result = _run(stream_kernel, (short_frame(),))

    assert seen == []
    assert result.kept == 0
    assert result.delivered == 1
    assert result.short_frames == 1
    assert result.stopped_early is False

    # The delivery is still recorded honestly: dropping it is not forgetting it.
    (record,) = result.records
    assert record.bytes_used == OWNER_SHORT_FRAME_BYTES
    assert record.complete is False
    assert record.index == 0
    assert kernel.count(VIDIOC_DQBUF) == 1


def test_a_short_delivery_is_requeued_exactly_like_a_complete_one(stream_kernel) -> None:
    """Ownership, not just filtering.

    A buffer this loop drops is still the driver's buffer. Holding one back at
    teardown is measured on this hardware to make the device re-enumerate, which
    to a user is indistinguishable from the scanner being unplugged - so the
    short path must return it on the same ioctl the complete path does.
    """
    kernel, _seen, result = _run(stream_kernel, (short_frame(buffer_index=2),))

    assert kernel.requeued == [2]
    assert result.short_frames == 1
    # And the run still tore down cleanly, in order.
    assert kernel.count(VIDIOC_STREAMOFF) == 1
    assert all(mapping.closes == 1 for mapping in kernel.mappings)
    releases = [
        packed
        for request, packed in kernel.calls
        if request == VIDIOC_REQBUFS and struct.unpack_from("=I", packed, 0)[0] == 0
    ]
    assert len(releases) == 1


def test_a_short_delivery_does_not_stop_the_run(stream_kernel) -> None:
    """The scan continues. This is the whole point of the change.

    Before it, the short frame was written and deprojected, the deprojection
    raised, and the exception propagated out of the capture job - which also
    skipped the re-queue for that buffer on the way out.
    """
    kernel, seen, result = _run(stream_kernel, (short_frame(), whole_frame(0xC3)))

    assert len(seen) == 1
    payload, record = seen[0]
    assert payload == bytes([0xC3]) * EXPECTED_FRAME_BYTES
    assert len(payload) == EXPECTED_FRAME_BYTES
    assert record.index == 1
    assert record.complete is True

    assert result.delivered == 2
    assert result.kept == 1
    assert result.short_frames == 1
    assert [r.complete for r in result.records] == [False, True]
    assert [r.bytes_used for r in result.records] == [OWNER_SHORT_FRAME_BYTES, EXPECTED_FRAME_BYTES]
    assert kernel.requeued == [0, 0]


# -- the complete path is untouched ----------------------------------------


def test_complete_deliveries_are_handed_over_byte_for_byte(stream_kernel) -> None:
    """The change must be invisible to a healthy run.

    Same count, same order, same bytes, and specifically ``EXPECTED_FRAME_BYTES``
    of them rather than the mapping's full length - the slice bound is
    ``bytesused``, and a frame padded out to the mapping would still be the
    wrong artifact.
    """
    deliveries = (
        whole_frame(0xA1),
        whole_frame(0xA2, buffer_index=1),
        whole_frame(0xA3, buffer_index=2),
    )
    kernel, seen, result = _run(stream_kernel, deliveries)

    assert [payload for payload, _ in seen] == [d.payload for d in deliveries]
    assert result.kept == 3
    assert result.delivered == 3
    assert result.short_frames == 0
    assert all(record.complete for record in result.records)
    assert kernel.requeued == [0, 1, 2]


def test_kept_still_counts_selected_frames_with_no_callback(stream_kernel) -> None:
    """``kept`` is a count of what was keepABLE, not of callback invocations.

    A caller that passes no callback still gets the count, and it still excludes
    the short delivery - otherwise the two forms of the same run would disagree
    about what happened.
    """
    kernel = stream_kernel((whole_frame(0xA1), short_frame(), whole_frame(0xA3)))
    result = stream_frames(frames=3)

    assert result.kept == 2
    assert result.delivered == 3
    assert result.short_frames == 1
    assert kernel.count(VIDIOC_QBUF) == kernel.granted_buffers + 3


# -- decimation -------------------------------------------------------------


def test_decimation_still_selects_on_the_delivered_index(stream_kernel) -> None:
    """A short frame consumes its slot; the next complete one does not inherit it.

    With ``decimate=2`` the selected deliveries are 0 and 2. Delivery 0 is
    short, so it is dropped and delivery 1 - complete, but not selected - stays
    unselected rather than being promoted into the gap. Silently substituting it
    would make the angular step between kept frames a function of which frames
    happened to be usable, which is exactly the property decimation exists to
    make predictable.
    """
    deliveries = (
        short_frame(),
        whole_frame(0xB1, buffer_index=1),
        whole_frame(0xB2, buffer_index=2),
        whole_frame(0xB3, buffer_index=3),
    )
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=2)

    assert [record.index for _payload, record in seen] == [2]
    assert [payload for payload, _ in seen] == [deliveries[2].payload]
    assert result.kept == 1
    assert result.delivered == 4
    assert result.short_frames == 1
    assert result.decimate == 2


def test_decimation_keeps_every_selected_complete_delivery(stream_kernel) -> None:
    """The companion: with the short frame on an unselected index, nothing changes.

    Selected are 0 and 2, both complete, and the short delivery at index 1 was
    never going to be kept anyway - so it costs a keep only when it lands on a
    slot decimation chose.
    """
    deliveries = (
        whole_frame(0xC1),
        short_frame(buffer_index=1),
        whole_frame(0xC2, buffer_index=2),
        whole_frame(0xC3, buffer_index=3),
    )
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=2)

    assert [record.index for _payload, record in seen] == [0, 2]
    assert result.kept == 2
    assert result.short_frames == 1


# -- every complete delivery, not only the keeps ----------------------------


def _both(
    wire,
    deliveries: tuple[Delivery, ...],
    **kwargs: object,
) -> tuple[
    StreamKernel, list[tuple[bytes, FrameRecord]], list[tuple[bytes, FrameRecord]], StreamResult
]:
    """Stream ``deliveries`` with BOTH callbacks, recording each separately."""
    kernel = wire(deliveries)
    kept: list[tuple[bytes, FrameRecord]] = []
    shown: list[tuple[bytes, FrameRecord]] = []
    result = stream_frames(
        frames=len(deliveries),
        on_frame=lambda payload, record: kept.append((payload, record)),
        on_delivery=lambda payload, record: shown.append((payload, record)),
        **kwargs,  # type: ignore[arg-type]
    )
    return kernel, kept, shown, result


def test_every_complete_delivery_is_shown_while_only_selected_ones_are_kept(stream_kernel) -> None:
    """The two rates, separated.

    At ``decimate=3`` the scanner delivers three frames for every one worth
    writing, and the owner watched a depth view moving at a third of the rate
    the scanner was actually running at. The display now follows deliveries and
    the expensive work still follows keeps - which is what decimation was always
    for.
    """
    deliveries = tuple(whole_frame(0xA0 + index, buffer_index=index % 4) for index in range(7))
    _kernel, kept, shown, result = _both(stream_kernel, deliveries, decimate=3)

    assert [record.index for _payload, record in shown] == [0, 1, 2, 3, 4, 5, 6]
    assert [record.index for _payload, record in kept] == [0, 3, 6]
    assert [payload for payload, _record in shown] == [delivery.payload for delivery in deliveries]
    assert result.kept == 3
    assert result.delivered == 7


def test_a_frame_that_is_both_shown_and_kept_is_copied_once(stream_kernel) -> None:
    """One buffer, one copy, two consumers.

    A second ``bytes()`` of the mapping would be half a megabyte of pointless
    memory traffic per selected frame at the measured 4.6 fps, and it would buy
    nothing: the payload is immutable, so there is no way for either consumer to
    disturb the other's copy of it.
    """
    _kernel, kept, shown, _result = _both(
        stream_kernel, (whole_frame(0xB7), whole_frame(0xB8, buffer_index=1)), decimate=2
    )

    assert len(kept) == 1
    assert len(shown) == 2
    # The SAME object, not merely equal bytes.
    assert kept[0][0] is shown[0][0]
    assert kept[0][1] is shown[0][1]


def test_a_short_delivery_reaches_neither_consumer(stream_kernel) -> None:
    """70% of a frame is not a frame, and it is not a picture either.

    The owner's scanner opened with 358402 bytes of one. Writing it took the
    2026-08-26 scan down; drawing it would put a torn image on the screen and
    label it the scanner's view. It is counted, re-queued and dropped, on the
    display path exactly as on the keep path.
    """
    kernel, kept, shown, result = _both(
        stream_kernel,
        (short_frame(), whole_frame(0xC9, buffer_index=1), short_frame(fill=0x33, buffer_index=2)),
    )

    assert [record.index for _payload, record in shown] == [1]
    assert [record.index for _payload, record in kept] == [1]
    assert result.short_frames == 2
    assert result.kept == 1
    # Every buffer still went back to the driver, including both short ones.
    assert kernel.requeued == [0, 1, 2]


def test_the_keep_callback_alone_behaves_exactly_as_it_did(stream_kernel) -> None:
    """The compatibility case, which is every caller outside the window.

    ``collect_frames``, ``iter_frames`` and the CLI sweep all pass one callback
    and know nothing about a second. A run without ``on_delivery`` must still
    copy once per KEPT frame and hand over exactly the selected ones.
    """
    deliveries = (
        whole_frame(0xD1),
        whole_frame(0xD2, buffer_index=1),
        whole_frame(0xD3, buffer_index=2),
        whole_frame(0xD4, buffer_index=3),
    )
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=2)

    assert [record.index for _payload, record in seen] == [0, 2]
    assert [payload for payload, _record in seen] == [
        deliveries[0].payload,
        deliveries[2].payload,
    ]
    assert result.kept == 2


# -- the callback that runs once the buffer is back ------------------------


def _after(
    wire,
    deliveries: tuple[Delivery, ...],
    **kwargs: object,
) -> tuple[StreamKernel, list[tuple[int, int]], list[tuple[int, int]], StreamResult]:
    """Stream ``deliveries``, recording the driver's buffer ledger at each hook.

    Each entry is ``(buffers taken, buffers given back)`` as it stood when that
    callback ran, which is the only thing that distinguishes the two hooks.
    """
    kernel = wire(deliveries)
    during: list[tuple[int, int]] = []
    after: list[tuple[int, int]] = []

    def ledger() -> tuple[int, int]:
        return (len(kernel.dequeued), len(kernel.requeued))

    result = stream_frames(
        frames=len(deliveries),
        on_frame=lambda _payload, _record: during.append(ledger()),
        after_frame=lambda _payload, _record: after.append(ledger()),
        **kwargs,  # type: ignore[arg-type]
    )
    return kernel, during, after, result


def test_all_frame_work_runs_once_the_driver_has_its_buffer_back(stream_kernel) -> None:
    """Tracking and the deferred phase must both leave the capture ring free."""
    deliveries = tuple(whole_frame(0xE0 + index, buffer_index=index % 4) for index in range(4))
    _kernel, during, after, result = _after(stream_kernel, deliveries)

    assert during == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert after == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert result.kept == 4


@pytest.mark.parametrize("hook", ["on_delivery", "on_frame", "after_frame"])
def test_callback_failure_does_not_leave_a_camera_buffer_outstanding(stream_kernel, hook):
    kernel = stream_kernel((whole_frame(0xA1, buffer_index=2),))

    def broken(_raw, _record):
        assert kernel.requeued == [2]
        raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        stream_frames(
            frames=1,
            on_delivery=broken if hook == "on_delivery" else None,
            on_frame=broken if hook == "on_frame" else None,
            after_frame=broken if hook == "after_frame" else None,
        )
    assert kernel.requeued == [2]
    assert kernel.count(VIDIOC_STREAMOFF) == 1


def test_callbacks_keep_the_original_bytes_if_requeue_refills_the_mapping(
    stream_kernel, monkeypatch
):
    kernel = stream_kernel((whole_frame(0xA1, buffer_index=2),))
    original_contents = kernel._contents

    def refilled(index, length):
        if kernel.requeued:
            return bytes([0xB2]) * length
        return original_contents(index, length)

    monkeypatch.setattr(kernel, "_contents", refilled)
    seen = []
    stream_frames(
        frames=1,
        on_delivery=lambda raw, record: seen.append(raw),
        on_frame=lambda raw, record: seen.append(raw),
        after_frame=lambda raw, record: seen.append(raw),
    )
    assert len(seen) == 3
    assert seen[0] == bytes([0xA1]) * EXPECTED_FRAME_BYTES
    assert seen[0] is seen[1] is seen[2]


def test_after_frame_sees_the_same_frames_and_the_same_copy_as_on_frame(stream_kernel) -> None:
    """Selected and complete, exactly like the keep - and not a second copy.

    A decimated run must not hand the post-requeue hook the frames decimation
    threw away, and a short delivery must not reach it at all: it is not a
    frame, and it never was.
    """
    deliveries = (
        whole_frame(0xF1),
        short_frame(fill=0x22, buffer_index=1),
        whole_frame(0xF3, buffer_index=2),
        whole_frame(0xF4, buffer_index=3),
    )
    kernel = stream_kernel(deliveries)
    kept: list[tuple[bytes, FrameRecord]] = []
    later: list[tuple[bytes, FrameRecord]] = []
    result = stream_frames(
        frames=len(deliveries),
        decimate=2,
        on_frame=lambda payload, record: kept.append((payload, record)),
        after_frame=lambda payload, record: later.append((payload, record)),
    )

    assert [record.index for _payload, record in kept] == [record.index for _p, record in later]
    assert [payload for payload, _record in kept] == [payload for payload, _r in later]
    # The SAME object, not merely equal bytes: one copy serves every consumer.
    assert all(a[0] is b[0] for a, b in zip(kept, later, strict=True))
    assert result.short_frames == 1
    assert kernel.requeued == [0, 1, 2, 3]


def test_a_slow_after_frame_delays_nothing_the_driver_is_waiting_for(stream_kernel) -> None:
    """A callback that takes its time leaves every buffer queued while it does.

    This is what makes a global relocalization survivable inside a capture: the
    device is never left waiting on it, and the stop poll immediately after is
    still reached.
    """
    deliveries = tuple(whole_frame(0x50 + index, buffer_index=index % 4) for index in range(3))
    kernel = stream_kernel(deliveries)
    outstanding: list[int] = []

    def slow(_payload: bytes, _record: FrameRecord) -> None:
        # Whatever this does, the ledger below is what the driver sees.
        outstanding.append(len(kernel.dequeued) - len(kernel.requeued))

    result = stream_frames(frames=len(deliveries), on_frame=None, after_frame=slow)

    assert outstanding == [0, 0, 0]
    assert result.kept == 3


def test_a_caller_that_passes_no_after_frame_is_completely_unaffected(stream_kernel) -> None:
    """The compatibility case: every caller outside the capture window."""
    deliveries = (whole_frame(0x61), whole_frame(0x62, buffer_index=1))
    kernel, seen, result = _run(stream_kernel, deliveries)

    assert [record.index for _payload, record in seen] == [0, 1]
    assert kernel.requeued == [0, 1]
    assert result.kept == 2


def test_a_stop_seen_during_an_unkept_delivery_still_requeues_first(stream_kernel) -> None:
    """Stop is answered where it has always been answered: after the re-queue.

    An unkept delivery is now a delivery the window is shown, so it is also a
    delivery the operator can press Stop during. That must not change where the
    stream acts on it: a callback that unwound from here would skip the
    ``VIDIOC_QBUF`` for the buffer in hand, and this hardware answers a buffer
    held at teardown by re-enumerating.
    """
    kernel = stream_kernel(
        (
            whole_frame(0xE1),
            whole_frame(0xE2, buffer_index=1),
            whole_frame(0xE3, buffer_index=2),
        )
    )
    shown: list[int] = []
    kept: list[int] = []
    stop = False

    def show(_payload: bytes, record: FrameRecord) -> None:
        """Stop during delivery 1, which at decimate 3 is shown and not kept."""
        nonlocal stop
        shown.append(record.index)
        if record.index == 1:
            stop = True

    result = stream_frames(
        frames=3,
        decimate=3,
        on_frame=lambda _payload, record: kept.append(record.index),
        on_delivery=show,
        should_stop=lambda: stop,
    )

    assert shown == [0, 1]
    assert kept == [0]
    assert result.stopped_early is True
    assert result.stop_reason == CALLER_STOP_REASON
    assert result.delivered == 2
    # Both buffers went back BEFORE the stop was acted on, which is the whole
    # reason the flag is polled where it is.
    assert kernel.requeued == [0, 1]


# -- the record and the report ---------------------------------------------


def test_sequence_and_gap_evidence_survive_a_short_delivery(stream_kernel) -> None:
    """A dropped frame is still evidence about the DEVICE.

    Its kernel sequence number is how a genuine drop is detected, and discarding
    the record because the payload was unusable would leave a hole in exactly
    the series that detection reads.
    """
    deliveries = (
        Delivery(EXPECTED_FRAME_BYTES, 0xD1, sequence=10),
        Delivery(OWNER_SHORT_FRAME_BYTES, 0x11, buffer_index=1, sequence=12),
        Delivery(EXPECTED_FRAME_BYTES, 0xD3, buffer_index=2, sequence=14),
    )
    _kernel, _seen, result = _run(stream_kernel, deliveries)

    assert [record.sequence for record in result.records] == [10, 12, 14]
    assert result.dropped_by_sequence == 0
    assert result.first_frame_ms is not None
    assert len(result.gaps_ms) == 2
    assert result.records[1].gap_ms is not None


def test_the_serialised_limitations_describe_the_counts_truthfully(stream_kernel) -> None:
    """The report used to claim every delivered frame was dequeued at full length.

    ``records`` proves otherwise on this very run, so the claim was false
    exactly when it mattered most - a reader reconciling a short kept count
    against the frames they asked for.
    """
    _kernel, _seen, result = _run(stream_kernel, (short_frame(), whole_frame(0xE1, buffer_index=1)))

    payload = result.as_dict()
    assert payload["delivered"] == 2
    assert payload["kept"] == 1
    assert payload["short_frames"] == 1
    stated = payload["limitations"]
    assert isinstance(stated, list)
    limitations = " ".join(str(text) for text in stated)
    assert "A frame counted as delivered was dequeued at full length" not in limitations
    assert "short ones included" in limitations
    assert "never padded, truncated or handed to a consumer" in limitations


def test_requested_bounds_deliveries_not_keeps(stream_kernel) -> None:
    """Asking for three frames takes three deliveries, however many are whole.

    The alternative - streaming until three COMPLETE frames arrive - would make
    a run's duration depend on the device's error rate, and a device delivering
    nothing but short buffers would never end.
    """
    _kernel, _seen, result = _run(
        stream_kernel, (short_frame(), short_frame(fill=0x22), short_frame(fill=0x33))
    )

    assert result.requested == 3
    assert result.delivered == 3
    assert result.short_frames == 3
    assert result.kept == 0
    assert result.stopped_early is False
    assert result.stop_reason == "requested frame count reached"


def test_a_caller_stop_is_still_honoured_after_a_short_delivery(stream_kernel) -> None:
    """The stop flag is polled after the re-queue, on the short path too.

    Anything else would make cancelling during a run of short frames wait for a
    complete one to arrive - and a device delivering short frames is precisely
    when someone reaches for cancel.
    """
    kernel = stream_kernel((short_frame(), whole_frame(0xF1, buffer_index=1)))
    result = stream_frames(frames=2, should_stop=lambda: True)

    assert result.delivered == 1
    assert result.stopped_early is True
    assert result.stop_reason == "caller asked to stop"
    assert result.short_frames == 1
    assert kernel.requeued == [0]


# -- the convenience wrappers inherit the same rule -------------------------


def test_collect_frames_returns_only_whole_frames(stream_kernel) -> None:
    """``collect_frames`` gets its filtering from the loop, not from a copy of it."""
    stream_kernel((short_frame(), whole_frame(0x7A, buffer_index=1)))
    collected, result = v4l2_stream.collect_frames(frames=2)

    assert [len(payload) for payload in collected] == [EXPECTED_FRAME_BYTES]
    assert result.short_frames == 1
    assert result.kept == 1


def test_iter_frames_yields_only_whole_frames(stream_kernel) -> None:
    """Same rule through the pull-shaped form."""
    stream_kernel((whole_frame(0x8B), short_frame(buffer_index=1)))
    pairs: Iterator[tuple[bytes, FrameRecord]] = v4l2_stream.iter_frames(frames=2)
    yielded = list(pairs)

    assert [len(payload) for payload, _ in yielded] == [EXPECTED_FRAME_BYTES]
    assert all(record.complete for _payload, record in yielded)


# -- what the CLI sweep therefore writes ------------------------------------
#
# The sweep command is the other consumer of this loop, and it writes the same
# frame_*.raw artifact the GUI does (tests/test_gui_short_frames.py covers that
# side). Its regression lives here rather than beside the other sweep tests
# because it is a claim about the BOUNDARY: the command has no length check of
# its own and is not gaining one, so what it writes is exactly what this loop
# lets through.


def test_the_cli_sweep_writes_only_whole_frames(stream_kernel, tmp_path, capsys) -> None:
    """Three deliveries, one of them short, two files - both full length.

    Before the change the sweep wrote all three, and the short one sat in the
    directory looking like a frame until ``reconstruct`` read it back and
    refused the whole capture.
    """
    from scanner3d.cli import main

    stream_kernel(
        (
            whole_frame(0x51),
            short_frame(buffer_index=1),
            whole_frame(0x53, buffer_index=2),
        )
    )
    output = tmp_path / "sweep"

    assert main(["sweep", str(output), "--frames", "3", "--decimate", "1", "--json"]) == 0

    written = sorted(output.glob("frame_*.raw"))
    assert [path.name for path in written] == ["frame_0000.raw", "frame_0001.raw"]
    assert [path.stat().st_size for path in written] == [EXPECTED_FRAME_BYTES] * 2
    assert [path.read_bytes()[0] for path in written] == [0x51, 0x53]

    payload = json.loads(capsys.readouterr().out)
    assert payload["kept"] == 2
    assert payload["delivered"] == 3
    assert payload["short_frames"] == 1


# -- a cadence that changes while the run is going ---------------------------


def _cadence(values: list[object]):
    """A controller answering from a list, then repeating its last answer.

    Written out rather than using :class:`~scanner3d.cadence.CadenceController`
    on purpose. What is under test here is the SELECTION - which deliveries a
    changing spacing keeps - and driving that through a real controller would
    mean persuading it to change on the frames the test needs, which is a test
    of the controller's arithmetic wearing a stream's clothes.
    """
    remaining = list(values)
    last: list[object] = [values[0]]

    def now() -> int:
        if remaining:
            last[0] = remaining.pop(0)
        answer = last[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer  # type: ignore[return-value]

    return now


def _deliveries(count: int) -> tuple[Delivery, ...]:
    return tuple(whole_frame(0xD0 + n, buffer_index=n % 4) for n in range(count))


def test_a_controller_that_never_changes_selects_what_the_number_selects(stream_kernel) -> None:
    """The floor this whole change has to stand on.

    A varying cadence is only worth having if the fixed one is untouched, and
    "untouched" has to mean the same DELIVERIES, not merely the same count.
    """
    deliveries = _deliveries(9)
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=_cadence([3]))

    assert [record.index for _payload, record in seen] == [0, 3, 6]
    assert result.decimate == 3
    assert result.decimate_lowest == result.decimate_highest == 3
    assert result.decimate_varied is False


def test_a_coarser_cadence_takes_effect_at_the_next_kept_frame(stream_kernel) -> None:
    """Not retroactively, and not in the middle of a run of skipped frames.

    The countdown is refilled from the value in force at the KEEP, so a spacing
    that widened describes the frames after the next one rather than reaching
    back into a gap that was already being measured under the old value.
    """
    deliveries = _deliveries(9)
    # 1 for the first delivery, 3 from then on.
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=_cadence([1, 3]))

    assert [record.index for _payload, record in seen] == [0, 1, 4, 7]
    assert result.decimate == 1
    assert (result.decimate_lowest, result.decimate_highest) == (1, 3)
    assert result.decimate_varied is True


def test_a_finer_cadence_shortens_the_run_of_skipped_frames_already_in_progress(
    stream_kernel,
) -> None:
    """The one exception, and the direction it is safe in.

    At the widest setting, waiting out the current gap is nine more deliveries -
    about two seconds at the measured rate - and a controller only ever asks for
    a finer spacing because the current one was already too coarse. Clamping the
    countdown can only bring a keep FORWARD: with the spacing dropping to 2 after
    delivery 1, the second keep lands at 3 rather than at the 5 the original
    spacing of 5 would have chosen.
    """
    deliveries = _deliveries(9)
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=_cadence([5, 5, 2]))

    assert [record.index for _payload, record in seen] == [0, 3, 5, 7]
    assert (result.decimate_lowest, result.decimate_highest) == (2, 5)


def test_a_short_delivery_still_consumes_its_slot_under_a_changing_cadence(
    stream_kernel,
) -> None:
    """The rule the fixed cadence has always had, unchanged by the countdown.

    Selection stays a property of arrival order rather than of which frames
    happened to be usable, so a short frame landing on a keep slot costs that
    keep instead of promoting the next complete delivery into it.
    """
    deliveries = (
        whole_frame(0xE0),
        whole_frame(0xE1, buffer_index=1),
        short_frame(buffer_index=2),
        whole_frame(0xE3, buffer_index=3),
        whole_frame(0xE4),
        whole_frame(0xE5, buffer_index=1),
    )
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=_cadence([2]))

    assert [record.index for _payload, record in seen] == [0, 4]
    assert result.kept == 2
    assert result.short_frames == 1


@pytest.mark.parametrize("answer", [0, -1, None, 2.5, True, RuntimeError("no idea")])
def test_a_controller_that_cannot_answer_leaves_the_spacing_where_it_was(
    stream_kernel, answer: object
) -> None:
    """And the run finishes, which is the entire point.

    Raising out of the streaming loop would skip the ``VIDIOC_QBUF`` for the
    buffer in flight; cleanup then runs with userspace still owning it, and this
    hardware answers that by re-enumerating - which to a caller looks exactly
    like the scanner being unplugged. So a controller that cannot answer is not
    obeyed and is not fatal either, and the count says it happened rather than
    the record quietly describing a spacing that was never held.
    """
    deliveries = _deliveries(7)
    _kernel, seen, result = _run(stream_kernel, deliveries, decimate=_cadence([2, answer]))

    assert [record.index for _payload, record in seen] == [0, 2, 4, 6]
    # One per DELIVERY. The usable answer was the one ``ensure_request`` took
    # before the device was opened, so every ask inside the loop refused and the
    # spacing that stood for the whole run is the one from before it started.
    assert result.decimate_refusals == len(deliveries)
    assert result.decimate_lowest == result.decimate_highest == 2
    assert result.stopped_early is False


def test_a_controller_that_cannot_answer_at_all_refuses_the_run_before_the_device(
    stream_kernel,
) -> None:
    """Checked where a bad answer can still be an exception.

    A controller that cannot produce one usable number before anything is opened
    is not one to start a capture with, and refusing here means the refusal
    costs no device state at all.
    """
    kernel = stream_kernel(_deliveries(3))

    with pytest.raises(StreamRequestUnusable):
        stream_frames(frames=3, decimate=_cadence([RuntimeError("never")]))
    with pytest.raises(StreamRequestUnusable):
        stream_frames(frames=3, decimate=_cadence([0]))

    assert kernel.count(VIDIOC_DQBUF) == 0


def test_the_record_reports_the_span_a_run_actually_used(stream_kernel) -> None:
    """A single number was the whole record before a cadence could vary, and
    left alone it would describe a spacing that was never true for the run."""
    deliveries = _deliveries(10)
    _kernel, _seen, result = _run(stream_kernel, deliveries, decimate=_cadence([4, 4, 1, 1, 3]))

    payload = result.as_dict()
    assert payload["decimate"] == 4
    assert payload["decimate_lowest"] == 1
    assert payload["decimate_highest"] == 4
    assert payload["decimate_varied"] is True
    assert payload["decimate_refusals"] == 0
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
