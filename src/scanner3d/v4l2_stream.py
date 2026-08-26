"""Hold one V4L2 stream open and consume frames as they arrive.

Why this exists, measured
-------------------------
:mod:`~scanner3d.v4l2_capture` takes exactly one frame per session: negotiate,
``STREAMON``, one poll, one dequeue, tear down. That is deliberate and it stays
that way — a single bounded capture is the right primitive for a diagnostic.

But timing it shows where the second goes::

    streamon      1.174 ms
    poll       1073.044 ms     <- waiting for the FIRST frame
    dqbuf         instant
    teardown      ~1 ms

The whole cost is the device spinning up its projector sequence after
``STREAMON``. Re-opening for every frame pays that ~1.07 s again each time, which
caps a capture cadence at roughly 3.2 s. On a turntable that is an angular step
of about 10 degrees, and point-to-point ICP does not reliably align a
featureless object across a gap that large — it converges near the identity and
reports success.

Measured on this unit with the stream held open, 25 consecutive frames, no
drops::

    frame 0    1068 ms after streamon      <- paid once
    then       69, 373, 68, 375, 68, 374 ... ms, stable

Frames arrive in pairs, roughly 69 ms then 373 ms, for a sustained **4.5 fps** —
not the 15 fps the descriptor advertises, because each depth frame needs a full
projector sequence. That is still an inter-frame gap of about 221 ms, roughly
1.2 degrees on the turntable this was measured against, and about nine times
finer than reopening the device each time.

So this module exists to pay the startup cost once.

What it does NOT do
-------------------
It writes no artifact and makes no claim about any frame's content. It hands
raw bytes to a callback and records what happened. Whether the bytes are a good
frame is :mod:`~scanner3d.depth_metric`'s question, not this module's.

It does judge one thing, and only one: LENGTH. A buffer the driver hands back
short of a whole frame is recorded, counted and dropped rather than passed on,
because a partial frame is not a frame with bad content - it is not a frame.
That check lives here because every consumer shares this boundary.

It makes no claim about frame rate either. The numbers above were measured on
one unit, on one day, with one scene, and are recorded as motivation rather
than as a specification. Every run reports its own observed intervals.
"""

from __future__ import annotations

import contextlib
import errno as errno_mod
import fcntl
import itertools
import mmap
import os
import struct
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d.config import resolve_serial
from scanner3d.discovery import find_scanners
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.v4l2_capture import (
    AUTHORIZED_SERIAL,
    EXPECTED_FRAME_BYTES,
    REQUESTED_BUFFERS,
    VIDIOC_DQBUF,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    NodeSnapshot,
    pack_buf_type,
    pack_buffer,
    pack_format,
    pack_requestbuffers,
    resolve_capture_node,
)

SCHEMA_VERSION: Final = 1

#: Byte offsets into ``struct v4l2_buffer``, derived from
#: :data:`~scanner3d.v4l2_capture.BUFFER_FORMAT` rather than counted by hand.
#: Getting these wrong reads a plausible number from the wrong field, which is
#: exactly the kind of defect that survives testing, so they are named.
BUFFER_OFFSETS: Final[MappingProxyType[str, int]] = MappingProxyType(
    {"index": 0, "bytesused": 8, "sequence": 56, "m_offset": 64, "length": 72}
)

#: A hard ceiling on frames per run. Not a performance limit — a runaway guard,
#: set far above any real sweep. At the measured 4.5 fps this is about 22
#: minutes, well past a turntable revolution.
MAX_FRAMES: Final = 6000

#: A hard ceiling on wall-clock time, whatever the frame count asks for.
MAX_DURATION_S: Final = 1800.0

#: How long to wait for the first frame before giving up. The measured startup
#: is ~1.07 s; this allows generous margin without hanging forever.
FIRST_FRAME_TIMEOUT_S: Final = 15.0

#: How long to wait for any subsequent frame. Measured gaps are 69-375 ms.
FRAME_TIMEOUT_S: Final = 10.0

#: Sleep between EAGAIN polls. Short enough not to add measurable latency to a
#: 221 ms cadence, long enough not to spin a core.
POLL_SLEEP_S: Final = 0.002


class StreamNotAllowed(ScannerError):
    """A streaming run was requested that this module refuses to start.

    Raised before the device is opened, so a refused run touches no hardware.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class StreamRequestUnusable(StreamNotAllowed):
    """The frame count, deadline or decimation asked for is outside its bound."""


class StreamDeviceUnusable(StreamNotAllowed):
    """The device could not be negotiated into the one mode this module streams."""

    status = DoctorStatus.NO_FRAMES


class StreamStalled(StreamNotAllowed):
    """No frame arrived within the timeout.

    Distinct from "the run finished early": a stall means the device stopped
    delivering while the run still wanted frames, and that is recorded rather
    than quietly returning a short result.
    """

    status = DoctorStatus.NO_FRAMES


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One dequeued frame. Bytes are handed to the callback, not stored here.

    ``sequence`` is the kernel's own counter. Comparing it against the index of
    frames actually received is how a drop is *detected* rather than assumed:
    a gap in the sequence means the device produced a frame this process never
    saw.
    """

    index: int
    sequence: int
    buffer_index: int
    bytes_used: int
    monotonic: float
    gap_ms: float | None
    complete: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "sequence": self.sequence,
            "buffer_index": self.buffer_index,
            "bytes_used": self.bytes_used,
            "gap_ms": self.gap_ms,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class StreamResult:
    """What a run actually did. Every field is recorded, none inferred.

    The three counts answer three different questions and are not
    interchangeable:

    ``delivered``
        Buffers dequeued from the driver, short ones included. It is the length
        of ``records``, and it is what ``requested`` bounds - asking for *n*
        frames bounds how many deliveries are taken, and promises nothing about
        how many of them turn out to be full length.
    ``short_frames``
        How many of those deliveries were not exactly
        :data:`~scanner3d.v4l2_capture.EXPECTED_FRAME_BYTES`. Each one still has
        its own :class:`FrameRecord`, with ``complete`` false and ``bytes_used``
        set to what actually arrived.
    ``kept``
        Complete deliveries selected by decimation - the ones handed to the
        callback. A run that saw short frames keeps fewer than its decimation
        alone would suggest, and that shortfall is exactly what
        ``short_frames`` explains.
    """

    node_path: str
    serial: str
    requested: int
    delivered: int
    kept: int
    decimate: int
    first_frame_ms: float | None
    gaps_ms: tuple[float, ...]
    dropped_by_sequence: int
    short_frames: int
    stopped_early: bool
    stop_reason: str
    records: tuple[FrameRecord, ...] = field(repr=False, default=())

    @property
    def median_gap_ms(self) -> float | None:
        if not self.gaps_ms:
            return None
        ordered = sorted(self.gaps_ms)
        return ordered[len(ordered) // 2]

    @property
    def mean_gap_ms(self) -> float | None:
        """The gap that determines THROUGHPUT."""
        return None if not self.gaps_ms else sum(self.gaps_ms) / len(self.gaps_ms)

    @property
    def observed_fps(self) -> float | None:
        """Frames per second actually sustained.

        Computed from the MEAN gap, not the median, and the difference is not
        academic on this device: the gaps are strongly bimodal, alternating
        about 69 ms and 373 ms as the projector cycles. The median lands on the
        long mode and reports 2.7 fps; the mean gives 4.5 fps, which is the rate
        at which frames actually arrive. An earlier version used the median and
        under-reported throughput by 40%.
        """
        gap = self.mean_gap_ms
        return None if gap is None or gap <= 0.0 else 1000.0 / gap

    def degrees_per_frame(self, degrees_per_second: float) -> float | None:
        """Angular step at a stated rotation rate.

        Takes the rate as an argument because this module cannot know it. The
        turntable rate is the caller's measurement, not the scanner's.
        """
        gap = self.mean_gap_ms
        return None if gap is None else degrees_per_second * gap / 1000.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "node_path": self.node_path,
            "serial": self.serial,
            "requested": self.requested,
            "delivered": self.delivered,
            "kept": self.kept,
            "decimate": self.decimate,
            "first_frame_ms": self.first_frame_ms,
            "median_gap_ms": self.median_gap_ms,
            "mean_gap_ms": self.mean_gap_ms,
            "observed_fps": self.observed_fps,
            "dropped_by_sequence": self.dropped_by_sequence,
            "short_frames": self.short_frames,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "limitations": [
                "No claim is made about any frame's content. These are the bytes the "
                "kernel presented; whether they are a usable depth frame is decided "
                "elsewhere.",
                "The observed frame rate is a property of this run, not a specification. "
                "It varies with the scene and the device's own projector sequence.",
                "'delivered' counts every buffer dequeued, short ones included; a "
                "delivery is NOT a claim of full length. 'short_frames' says how many "
                "of them fell short, and each is recorded individually.",
                "'kept' counts only complete deliveries selected by decimation. A short "
                "delivery is never padded, truncated or handed to a consumer, so a run "
                "with short frames keeps fewer than its decimation alone implies.",
            ],
        }


def ensure_request(frames: int, decimate: int, duration_s: float) -> None:
    """Refuse an out-of-bound run before any device is opened."""
    if not 1 <= frames <= MAX_FRAMES:
        raise StreamRequestUnusable(
            f"{frames} frames is outside [1, {MAX_FRAMES}]; the ceiling is a runaway "
            "guard, not a performance limit"
        )
    if decimate < 1:
        raise StreamRequestUnusable(f"decimate must be at least 1, got {decimate}")
    if not 0.0 < duration_s <= MAX_DURATION_S:
        raise StreamRequestUnusable(f"duration {duration_s} s is outside (0, {MAX_DURATION_S}]")


def _dropped_frames(sequences: tuple[int, ...]) -> int:
    """Frames the device produced that this process never saw.

    The kernel sequence counter does NOT advance by one per delivered frame on
    this unit: it advances by **two**, steadily, which matches the descriptor
    advertising a 640x800 mode alongside 640x400 - two stacked frames, almost
    certainly the stereo pair, counted individually while only one is dequeued
    per call.

    So a drop cannot be defined as ``delta != 1``. It is defined against the
    run's own modal stride, and only the excess over that stride counts. An
    earlier version assumed a stride of one and reported a 100% drop rate on a
    run that dropped nothing at all.
    """
    # The first interval is skipped: the counter's behaviour across STREAMON is
    # not a steady-state property, and the first frame's arrival is already
    # characterised separately by first_frame_ms. On this unit the opening delta
    # is 4 against a steady stride of 2, which would otherwise read as a drop on
    # every clean run.
    steady = sequences[1:]
    deltas = [b - a for a, b in itertools.pairwise(steady) if b > a]
    if not deltas:
        return 0
    stride = max(set(deltas), key=deltas.count)
    if stride <= 0:
        return 0
    return sum(delta - stride for delta in deltas if delta > stride) // stride


def _field(buf: bytes, name: str) -> int:
    value: int = struct.unpack_from("=I", buf, BUFFER_OFFSETS[name])[0]
    return value


#: How long to wait for the scanner to come back after a stream is closed.
#:
#: MEASURED, not guessed: this device RE-ENUMERATES on the USB bus every time a
#: depth stream is closed. Its bus address climbs by one each capture. It
#: disappears for about 0.39 s and then returns, so anything that opens a stream
#: immediately after closing one sees "no such device" and looks, to a user,
#: exactly like the scanner having been unplugged.
#:
#: The ceiling here is many times the measured gap because the consequence of
#: waiting slightly too long is a pause, and the consequence of not waiting long
#: enough is a failure the user cannot interpret.
REENUMERATION_TIMEOUT_S: Final = 6.0


def wait_for_scanner(
    serial: str | None = None, *, timeout_s: float = REENUMERATION_TIMEOUT_S
) -> bool:
    """Block until the scanner is enumerable again, or the timeout expires.

    Returns whether it came back. Callers that open a stream shortly after
    closing one should call this first - see :data:`REENUMERATION_TIMEOUT_S` for
    why the device is absent at all.

    ``serial`` of None waits for ANY scanner, which is what a caller almost
    always means: it has just closed a stream on the device it already resolved
    and wants that same device back. Matching against a configured-but-absent
    serial would wait the full timeout every time and then proceed anyway - a
    silent six-second tax that this function existed to avoid.
    """
    deadline = time.monotonic() + timeout_s
    wanted = serial.strip() if serial and serial.strip() else None
    while True:
        try:
            devices = find_scanners()
            if any(device.serial == wanted for device in devices) if wanted else bool(devices):
                return True
        except ScannerError:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def stream_frames(
    serial: str = AUTHORIZED_SERIAL,
    *,
    frames: int,
    on_frame: Callable[[bytes, FrameRecord], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    decimate: int = 1,
    duration_s: float = MAX_DURATION_S,
    buffers: int = REQUESTED_BUFFERS,
) -> StreamResult:
    """Open the depth stream once and consume ``frames`` frames from it.

    ``frames`` bounds DELIVERIES, not keeps: it is how many buffers this run
    will dequeue, and a run that asks for 200 may hand the callback fewer.

    ``on_frame`` receives ``(raw_bytes, record)`` for every KEPT frame, which is
    every ``decimate``-th delivered frame THAT ARRIVED AT FULL LENGTH.
    Decimation exists because a full turntable revolution at the measured
    cadence is several hundred frames and over a quarter of a gigabyte; a caller
    that wants a 3-degree angular step should not have to buffer everything to
    get it.

    A buffer whose ``bytesused`` is not exactly
    :data:`~scanner3d.v4l2_capture.EXPECTED_FRAME_BYTES` is recorded, counted in
    ``short_frames``, re-queued and then dropped. It never reaches the callback,
    even when decimation selects its delivered index, because every consumer
    downstream of here - preview, deprojection, ``frame_*.raw`` on disk - would
    otherwise be handed something that is not a depth frame. Filtering it in one
    place is what keeps them all from having to.

    The bytes handed to the callback are a copy taken from the mmap before the
    buffer is re-queued. That copy is not optional: the kernel may refill the
    mapping the moment it is queued again, so a caller holding a view would see
    it change underneath them.

    ``should_stop`` is polled after each buffer has been RE-QUEUED, and ending
    the stream there is the only safe place to end it early. A callback that
    raises instead skips the ``VIDIOC_QBUF`` for the buffer it was handed, so
    cleanup then runs with a buffer still owned by userspace — measured on this
    hardware, that makes the device re-enumerate, which to a caller looks
    exactly like the scanner being unplugged. Stopping through this flag
    returns every buffer first and leaves the device usable immediately.

    Cleanup runs in ``finally`` — ``STREAMOFF``, unmap, release, close — so a
    callback that raises still leaves the device closed rather than held, but
    "closed" and "healthy" are not the same thing and only the flag gives both.
    """
    ensure_request(frames, decimate, duration_s)

    # Resolved here rather than baked into the default, so an installation that
    # configured nothing still works when exactly one scanner is attached, and
    # refuses rather than guessing when several are.
    serial = resolve_serial(serial if serial != AUTHORIZED_SERIAL else None)

    snapshot, _device = resolve_capture_node(serial)
    if not isinstance(snapshot, NodeSnapshot):  # pragma: no cover - defensive
        raise StreamDeviceUnusable("the resolver did not return a node snapshot")
    path = snapshot.node_path

    records: list[FrameRecord] = []
    gaps: list[float] = []
    kept = short = dropped = 0
    first_ms: float | None = None
    stopped_early = False
    reason = "requested frame count reached"

    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK)
    maps: list[mmap.mmap] = []
    streaming = False
    granted = 0
    try:
        try:
            fcntl.ioctl(fd, VIDIOC_S_FMT, pack_format(), True)
        except OSError as exc:
            raise StreamDeviceUnusable(f"the device refused the one mode: {exc}") from exc

        request = pack_requestbuffers(count=buffers, limit=buffers)
        fcntl.ioctl(fd, VIDIOC_REQBUFS, request, True)
        granted = struct.unpack_from("=I", request, 0)[0]
        if granted < 2:
            raise StreamDeviceUnusable(
                f"{granted} buffers granted; at least 2 are needed to keep one queued "
                "while another is being read"
            )

        for index in range(granted):
            queried = pack_buffer(index, limit=granted)
            fcntl.ioctl(fd, VIDIOC_QUERYBUF, queried, True)
            raw = bytes(queried)
            maps.append(
                mmap.mmap(
                    fd,
                    _field(raw, "length"),
                    mmap.MAP_SHARED,
                    mmap.PROT_READ,
                    offset=_field(raw, "m_offset"),
                )
            )
            fcntl.ioctl(fd, VIDIOC_QBUF, pack_buffer(index, limit=granted), True)

        fcntl.ioctl(fd, VIDIOC_STREAMON, pack_buf_type(), True)
        streaming = True
        started = time.perf_counter()
        previous: float | None = None

        for index in range(frames):
            budget = FIRST_FRAME_TIMEOUT_S if index == 0 else FRAME_TIMEOUT_S
            waited_from = time.perf_counter()
            buf = pack_buffer(0, limit=granted)
            while True:
                now = time.perf_counter()
                if now - started > duration_s:
                    stopped_early, reason = True, "duration budget reached"
                    break
                if now - waited_from > budget:
                    raise StreamStalled(
                        f"no frame within {budget} s at index {index}; the device stopped "
                        f"delivering after {len(records)} frames"
                    )
                try:
                    fcntl.ioctl(fd, VIDIOC_DQBUF, buf, True)
                    break
                except OSError as exc:
                    if exc.errno != errno_mod.EAGAIN:
                        raise
                    time.sleep(POLL_SLEEP_S)
            if stopped_early:
                break

            arrived = time.perf_counter()
            raw_buf = bytes(buf)
            used = _field(raw_buf, "bytesused")
            sequence = _field(raw_buf, "sequence")
            buffer_index = _field(raw_buf, "index")
            complete = used == EXPECTED_FRAME_BYTES
            if not complete:
                short += 1

            gap = None if previous is None else (arrived - previous) * 1000.0
            if gap is None:
                first_ms = (arrived - started) * 1000.0
            else:
                gaps.append(gap)
            previous = arrived

            record = FrameRecord(
                index=index,
                sequence=sequence,
                buffer_index=buffer_index,
                bytes_used=used,
                monotonic=arrived,
                gap_ms=gap,
                complete=complete,
            )
            records.append(record)

            # ``complete`` gates the callback, and it gates it HERE rather than
            # in each consumer. MEASURED on 2026-08-26: the first buffer of a
            # GUI scan arrived with bytesused 358402 of the expected 512000, and
            # the frame was written to disk and deprojected anyway, which failed
            # with DepthFrameSizeMismatch and took the whole scan down at
            # startup. A short delivery is a transport event, not a frame: it is
            # recorded above and counted in ``short_frames``, and then it is
            # dropped. It is deliberately NOT padded, truncated or re-read - the
            # missing bytes were never delivered, and any frame manufactured
            # from what did arrive would be a fabrication with a plausible size.
            #
            # ``kept`` therefore counts only complete deliveries. Decimation
            # still selects on the DELIVERED index, so a short frame consumes
            # its slot rather than shifting the next one into it: the selection
            # stays a property of arrival order, not of which frames happened to
            # be usable.
            if complete and index % decimate == 0:
                kept += 1
                if on_frame is not None:
                    # Copy before re-queueing: the kernel may refill this mapping
                    # the instant it is queued, so a view handed out here would
                    # mutate.
                    on_frame(bytes(maps[buffer_index][:used]), record)

            # Unconditional, and reached by a short delivery on exactly the same
            # path as a complete one: the buffer is the driver's either way, and
            # a run that kept one back would leave userspace owning it at
            # teardown - which this hardware answers by re-enumerating.
            fcntl.ioctl(fd, VIDIOC_QBUF, buf, True)

            # Polled here, AFTER re-queueing, so the driver owns every buffer
            # again before the finally block tears the mapping down.
            if should_stop is not None and should_stop():
                stopped_early, reason = True, "caller asked to stop"
                break
    finally:
        # Each step is suppressed on its own: a failure in one must not skip the
        # rest, or a run that died mid-stream would leave the device held open.
        if streaming:
            with contextlib.suppress(OSError):
                fcntl.ioctl(fd, VIDIOC_STREAMOFF, pack_buf_type(), True)
        for mapping in maps:
            with contextlib.suppress(BufferError, ValueError):
                mapping.close()
        if granted:
            with contextlib.suppress(OSError):
                fcntl.ioctl(fd, VIDIOC_REQBUFS, pack_requestbuffers(count=0, limit=1), True)
        os.close(fd)

    dropped = _dropped_frames(tuple(r.sequence for r in records))

    return StreamResult(
        node_path=path,
        serial=snapshot.serial,
        requested=frames,
        delivered=len(records),
        kept=kept,
        decimate=decimate,
        first_frame_ms=first_ms,
        gaps_ms=tuple(gaps),
        dropped_by_sequence=dropped,
        short_frames=short,
        stopped_early=stopped_early,
        stop_reason=reason,
        records=tuple(records),
    )


def collect_frames(
    serial: str = AUTHORIZED_SERIAL,
    *,
    frames: int,
    decimate: int = 1,
    duration_s: float = MAX_DURATION_S,
) -> tuple[list[bytes], StreamResult]:
    """Convenience: stream and return the kept frames in memory.

    Only for small runs. A full revolution at the measured cadence is several
    hundred frames and hundreds of megabytes, which is exactly what ``decimate``
    and the callback form of :func:`stream_frames` exist to avoid.
    """
    collected: list[bytes] = []

    def keep(payload: bytes, _record: FrameRecord) -> None:
        collected.append(payload)

    result = stream_frames(
        serial, frames=frames, on_frame=keep, decimate=decimate, duration_s=duration_s
    )
    return collected, result


def iter_frames(
    serial: str = AUTHORIZED_SERIAL, *, frames: int, decimate: int = 1
) -> Iterator[tuple[bytes, FrameRecord]]:
    """Frames as an iterator, for callers that would rather pull than be pushed.

    Buffers everything before yielding, so it carries the same memory caution as
    :func:`collect_frames`. The callback form is the streaming one.
    """
    payloads: list[tuple[bytes, FrameRecord]] = []

    def keep(payload: bytes, record: FrameRecord) -> None:
        payloads.append((payload, record))

    stream_frames(serial, frames=frames, on_frame=keep, decimate=decimate)
    yield from payloads
