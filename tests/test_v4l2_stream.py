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

from scanner3d import v4l2_stream
from scanner3d.v4l2_capture import (
    EXPECTED_FRAME_BYTES,
    VIDIOC_DQBUF,
    VIDIOC_QBUF,
    VIDIOC_REQBUFS,
    VIDIOC_STREAMOFF,
)
from scanner3d.v4l2_stream import FrameRecord, StreamResult, stream_frames
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
