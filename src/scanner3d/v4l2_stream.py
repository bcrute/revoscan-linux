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
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scanner3d.config import resolve_serial
from scanner3d.discovery import (
    DEFAULT_DEV_ROOT,
    DEFAULT_SYSFS_ROOT,
    AccessCheck,
    ScannerDevice,
    find_scanners,
)
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.v4l2_capture import (
    AUTHORIZED_SERIAL,
    DEPTH_PROFILE,
    EXPECTED_FRAME_BYTES,
    REQUESTED_BUFFERS,
    VIDIOC_DQBUF,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    CaptureProfile,
    NodeSnapshot,
    pack_buf_type,
    pack_buffer,
    pack_format,
    pack_requestbuffers,
    resolve_capture_node,
)

SCHEMA_VERSION: Final = 1

#: A caller that decides the decimation while the run is going, rather than
#: before it. Asked once per delivery, after the buffer has been re-queued, and
#: never trusted to be answerable - see :func:`_resolve_decimate`.
Cadence = Callable[[], int]

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

#: The ``stop_reason`` recorded when ``should_stop`` ended a run, and the only
#: reason that means "the caller asked for this". Named rather than written out
#: at each site because callers legitimately need to tell their OWN stop apart
#: from a run that merely finished: a GUI that treats "the user pressed Stop"
#: as a completed capture reports the wrong outcome, and a GUI that treats every
#: early end as a stop reports the wrong outcome the other way round.
CALLER_STOP_REASON: Final = "caller asked to stop"


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


class StreamDeviceNotReady(StreamNotAllowed):
    """The depth capture node never became both resolvable and usable in time.

    Raised INSTEAD of opening anything, so a run refused here has touched no
    descriptor. It covers two quite different problems - the scanner never came
    back on the bus, and it came back but this process may not use its node -
    which is why the status is carried per instance rather than fixed by the
    class: a caller that reports statuses must not be told "permission" about a
    scanner that is simply unplugged.
    """

    def __init__(self, message: str, *, status: DoctorStatus = DoctorStatus.DEVICE_ABSENT) -> None:
        super().__init__(message)
        self.status = status


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
    #: The decimation the run STARTED at. Equal to ``decimate_lowest`` and
    #: ``decimate_highest`` for the ordinary fixed-cadence run; for a controlled
    #: one it is where the operator set the control, and the two bounds are what
    #: actually happened. A single number was the whole record before a cadence
    #: could vary, and left alone it would now describe a spacing that was never
    #: true for the whole run.
    decimate: int
    first_frame_ms: float | None
    gaps_ms: tuple[float, ...]
    dropped_by_sequence: int
    short_frames: int
    stopped_early: bool
    stop_reason: str
    records: tuple[FrameRecord, ...] = field(repr=False, default=())
    #: The narrowest and widest spacing any kept frame of this run was taken at.
    #: Both equal :attr:`decimate` unless a controller moved it.
    decimate_lowest: int = 0
    decimate_highest: int = 0
    #: Times the controller was asked for a cadence and could not usably answer,
    #: so the previous value stood. Counted rather than raised, because raising
    #: inside the streaming loop is the one thing this module must never do -
    #: and reported rather than swallowed, because a cadence that silently
    #: stopped responding is a run whose spacing does not mean what it says.
    decimate_refusals: int = 0

    @property
    def decimate_varied(self) -> bool:
        """Whether the spacing between kept frames changed during the run."""
        return bool(self.decimate_lowest) and self.decimate_lowest != self.decimate_highest

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
            "decimate_lowest": self.decimate_lowest or self.decimate,
            "decimate_highest": self.decimate_highest or self.decimate,
            "decimate_varied": self.decimate_varied,
            "decimate_refusals": self.decimate_refusals,
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


def ensure_request(frames: int, decimate: int | Cadence, duration_s: float) -> int:
    """Refuse an out-of-bound run before any device is opened.

    Returns the decimation the run starts at, which for a controller means the
    one answer it gave here. A callable is checked by CALLING it once, at this
    point, where a bad answer can still be an exception - nothing about a
    callable can be validated from its type, and the alternative is discovering
    mid-stream that it returns zero, which is the one place this module must not
    raise from: a callback that raises between DQBUF and QBUF leaves userspace
    owning a buffer, and this hardware answers that by re-enumerating. See
    :func:`_resolve_decimate`.

    Returning the value rather than asking again is what keeps a controller
    called exactly once per decision. Asking twice before the first delivery
    would make "one call per delivery" untrue at the only point a controller
    has no measurements to answer from.
    """
    if not 1 <= frames <= MAX_FRAMES:
        raise StreamRequestUnusable(
            f"{frames} frames is outside [1, {MAX_FRAMES}]; the ceiling is a runaway "
            "guard, not a performance limit"
        )
    if callable(decimate):
        try:
            first = decimate()
        except Exception as exc:  # reported as a refusal, never propagated as itself
            raise StreamRequestUnusable(
                f"the decimation controller raised {type(exc).__name__} the first time it was "
                f"asked: {exc}. A controller that cannot answer before the device is opened "
                "cannot be asked once a buffer is in flight."
            ) from exc
        started = _ensure_decimate(first)
    else:
        started = _ensure_decimate(decimate)
    if not 0.0 < duration_s <= MAX_DURATION_S:
        raise StreamRequestUnusable(f"duration {duration_s} s is outside (0, {MAX_DURATION_S}]")
    return started


def _ensure_decimate(value: object) -> int:
    """One decimation, checked. Raises rather than clamping, before streaming."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamRequestUnusable(
            f"decimate must be a whole number of delivered frames, got {value!r}"
        )
    if value < 1:
        raise StreamRequestUnusable(f"decimate must be at least 1, got {value}")
    return value


def _resolve_decimate(decimate: int | Cadence, current: int) -> tuple[int, bool]:
    """The decimation now, and whether the controller had to be ignored.

    NEVER raises, and that is the whole reason it exists. This is called inside
    the streaming loop, and the module's contract is that nothing in that loop
    may propagate: an exception there skips the ``VIDIOC_QBUF`` for the buffer
    in flight, cleanup then runs with userspace still owning it, and on this
    hardware the device re-enumerates - which to a caller looks exactly like the
    scanner being unplugged.

    So a controller that raises, or answers with something that is not a usable
    decimation, is not obeyed and is not fatal either: the last good value
    stands, the run counts it, and :attr:`StreamResult.decimate_refusals` says
    it happened. A cadence that silently stopped changing is a capture whose
    frame spacing does not mean what its record says.
    """
    if not callable(decimate):
        return current, False
    try:
        return _ensure_decimate(decimate()), False
    except Exception:  # the loop must not propagate; see the docstring above
        return current, True


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


#: How often readiness is re-checked while waiting.
#:
#: The window being waited through is measured in tenths of a second, so a
#: slower poll would spend most of it asleep and a faster one would spin a core
#: stat'ing a node for no extra resolution.
READINESS_POLL_S: Final = 0.05


def wait_for_scanner(
    serial: str | None = None, *, timeout_s: float = REENUMERATION_TIMEOUT_S
) -> bool:
    """Block until the scanner is ENUMERABLE again, or the timeout expires.

    Returns whether it came back. See :data:`REENUMERATION_TIMEOUT_S` for why
    the device is absent at all.

    ``serial`` of None waits for ANY scanner. Matching against a
    configured-but-absent serial would wait the full timeout every time and then
    proceed anyway - a silent six-second tax that this function existed to
    avoid.

    Enumerable is NOT the same question as usable, and callers that are about to
    open the node want :func:`wait_for_capture_access` instead: a node is
    registered by the kernel before this account necessarily has access to it,
    and this function returns the moment sysfs lists the device. Retained as
    part of the public surface for callers that genuinely only want to know
    whether the device is back on the bus; the GUI no longer uses it.
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


@dataclass(frozen=True, slots=True)
class CaptureReadiness:
    """Whether the profile's capture node may be opened - decided without opening it.

    ``ready`` is the whole answer for a caller that only wants to know whether
    to proceed. Everything else exists so a wait that did NOT succeed can say
    which of several quite different things it was, in words the person looking
    at the window can act on.

    ``access`` is the last :class:`~scanner3d.discovery.AccessCheck` taken of the
    resolved node, and it is ``None`` whenever the wait never got as far as
    having a node to check - which is exactly what separates "the scanner is not
    there" from "it is there and this account may not use it".
    """

    ready: bool
    reason: str
    serial: str | None = None
    node_path: str | None = None
    access: AccessCheck | None = None
    #: True when readiness deliberately declined to NAME a node: several
    #: scanners are attached and no serial was asked for, so choosing one would
    #: be a guess. Ready, but not on the strength of any node - see
    #: :func:`wait_for_capture_access`.
    deferred: bool = False
    timed_out: bool = False
    polls: int = 0
    waited_s: float = 0.0

    @property
    def status(self) -> DoctorStatus:
        """Which kind of problem this was, for a caller that reports statuses.

        Read from what was observed rather than assumed: a node that exists and
        is not readable and writable is a permission problem, and everything
        else that stopped the wait is the device not being there in the shape
        this profile needs.
        """
        if self.ready:
            return DoctorStatus.OK
        if self.access is not None and self.access.exists:
            return DoctorStatus.PERMISSION_DENIED
        return DoctorStatus.DEVICE_ABSENT

    def failure_text(self) -> str:
        """The operator-facing explanation of a wait that ended not ready.

        Says what was LAST observed rather than what is usually wrong: the same
        timeout covers a scanner that never came back and one that came back
        with a node this account may not use, and those two have no advice in
        common. The permission paragraph is therefore attached only when a node
        really was seen and really was not usable.
        """
        head = (
            f"The {DEPTH_PROFILE.role} capture node was still not usable after "
            f"{self.waited_s:.2f} s and {self.polls} checks, none of which opened a "
            f"device node: {self.reason}"
        )
        if self.status is not DoctorStatus.PERMISSION_DENIED:
            return head
        return (
            f"{head}\n\n"
            "A node that exists but is not readable and writable by this process is a "
            "permission problem, not a scanner fault. On this host access comes from a "
            "per-seat ACL rather than group membership, and such an ACL is normally "
            "attached shortly AFTER the device enumerates - a brief window where it is "
            "missing is expected, and absorbing that window is what this wait is for. "
            "A window that never closes needs the account granted access - membership of "
            "the group owning the node, or a udev rule that tags it for this seat - and "
            "retrying will not substitute for it."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "status": str(self.status),
            "serial": self.serial,
            "node_path": self.node_path,
            "access": self.access.as_dict() if self.access is not None else None,
            "deferred_to_resolver": self.deferred,
            "timed_out": self.timed_out,
            "polls": self.polls,
            "waited_s": round(self.waited_s, 3),
        }


def _node_readiness(device: ScannerDevice, profile: CaptureProfile) -> CaptureReadiness:
    """Readiness of one identified device's capture node for ``profile``.

    The role comes from the profile and the node from the USB interface string
    descriptor plus the V4L2 index, exactly as
    :func:`~scanner3d.v4l2_capture.resolve_capture_node` does it. No ``videoN``
    number is consulted, so a scanner whose depth camera lands on ``video7``
    this boot is answered about ``video7``.
    """
    serial = device.serial
    function = device.function(profile.role)
    if function is None:
        return CaptureReadiness(
            ready=False,
            serial=serial,
            reason=(
                f"{device.usb_path} exposes no {profile.role!r} camera function yet; the "
                "interface string descriptors do not name one"
            ),
        )
    nodes = function.capture_nodes
    if not nodes:
        present = ", ".join(f"{n.name} (index {n.index})" for n in function.video_nodes) or "none"
        return CaptureReadiness(
            ready=False,
            serial=serial,
            reason=(
                f"the {profile.role} function exposes no V4L2 {profile.node_kind} node yet "
                f"(nodes registered: {present})"
            ),
        )
    if len(nodes) > 1:
        names = ", ".join(node.name for node in nodes)
        return CaptureReadiness(
            ready=False,
            serial=serial,
            reason=(
                f"the {profile.role} function exposes {len(nodes)} {profile.node_kind} nodes "
                f"({names}); refusing to choose between them"
            ),
        )
    node = nodes[0]
    # Already taken by discovery on THIS poll, from stat() and os.access(). No
    # descriptor is opened to answer this, which is the point: opening the node
    # to find out whether it can be opened is the race being waited out.
    access = node.access
    if not access.exists:
        return CaptureReadiness(
            ready=False,
            serial=serial,
            node_path=node.device_path,
            access=access,
            reason=f"{node.device_path} has not appeared yet",
        )
    if not (access.readable and access.writable):
        missing = " and ".join(
            name
            for name, granted in (("readable", access.readable), ("writable", access.writable))
            if not granted
        )
        return CaptureReadiness(
            ready=False,
            serial=serial,
            node_path=node.device_path,
            access=access,
            reason=(
                f"{node.device_path} exists but is not {missing} by this process "
                f"(mode {access.mode}, owner uid {access.owner_uid}, group gid "
                f"{access.group_gid})"
            ),
        )
    return CaptureReadiness(
        ready=True,
        serial=serial,
        node_path=node.device_path,
        access=access,
        reason=(
            f"{node.device_path} is the sole {profile.node_kind} node of the "
            f"{profile.role} function on interface {function.interface_number} and is "
            "readable and writable by this process"
        ),
    )


def _readiness_now(
    serial: str | None,
    profile: CaptureProfile,
    sysfs_root: Path,
    dev_root: Path,
) -> CaptureReadiness:
    """One poll: resolve identity and role from scratch, then check access.

    Nothing is carried over from the previous poll. The device re-enumerates
    between them - a new bus address, a new sysfs path, potentially a new
    ``videoN`` - so a cached node is a node that may no longer exist.
    """
    try:
        devices = find_scanners(sysfs_root=sysfs_root, dev_root=dev_root)
    except ScannerError as exc:
        return CaptureReadiness(ready=False, reason=f"could not enumerate: {exc}")
    # Deliberately NOT falling back to configured_serial(): ``None`` here means
    # "any scanner", which is the meaning the wait this replaced had, and the
    # caller that knows which scanner it wants passes it. Reading configuration
    # behind the caller's back would make ``None`` mean two different things
    # depending on a file the caller never mentioned.
    wanted = serial.strip() if serial and serial.strip() else None

    if wanted:
        matches = tuple(device for device in devices if device.serial == wanted)
        if not matches:
            return CaptureReadiness(
                ready=False,
                serial=wanted,
                reason=f"no 2207:110c scanner with serial {wanted!r} is attached",
            )
        if len(matches) > 1:
            return CaptureReadiness(
                ready=False,
                serial=wanted,
                reason=(
                    f"{len(matches)} attached devices report serial {wanted!r}; refusing "
                    "to choose between them"
                ),
            )
        return _node_readiness(matches[0], profile)

    if not devices:
        return CaptureReadiness(ready=False, reason="no 2207:110c scanner is attached")
    if len(devices) > 1:
        attached = ", ".join(sorted(str(device.serial) for device in devices))
        return CaptureReadiness(
            ready=True,
            deferred=True,
            reason=(
                f"{len(devices)} scanners are attached ({attached}) and no serial was asked "
                "for, so there is no one node to wait on without guessing; the wait ends "
                "here and the stream's own resolver refuses or chooses in its own words"
            ),
        )
    return _node_readiness(devices[0], profile)


def wait_for_capture_access(
    serial: str | None = None,
    *,
    timeout_s: float = REENUMERATION_TIMEOUT_S,
    profile: CaptureProfile = DEPTH_PROFILE,
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
    poll_s: float = READINESS_POLL_S,
) -> CaptureReadiness:
    """Block until the profile's capture node is USABLE, or the timeout expires.

    Callers that open a stream shortly after closing one must call this first -
    see :data:`REENUMERATION_TIMEOUT_S` for why the device is absent at all.

    Enumerability is not the question this answers, and answering only that was
    a real defect. What was OBSERVED on 2026-08-26: the GUI's enumeration-only
    wait returned in 0.003 s, the very next open failed with ``EACCES`` on
    ``/dev/video0``, that node had been created in the same second, and when it
    was inspected afterwards it was ``root:video 0660`` with a valid per-user
    ACL for this account. No access snapshot was taken at the instant of the
    failing open, so the ACL not yet being attached at that moment is an
    INFERENCE - a strongly consistent one, and the only mechanism proposed that
    fits all four observations - rather than something the run proved.

    What is fixed here does not depend on that inference being right. A node the
    kernel has registered is not necessarily a node this process may open, for
    whatever reason it may not; so readiness means the sole index-0 node of the
    profile's role exists AND is readable AND is writable, checked until it is
    or the ceiling expires - never that a fixed ``/dev/videoN`` is present and
    never that some scanner is merely enumerable.

    Identity and role are resolved on EVERY poll rather than once, because the
    device re-enumerates: the sysfs path, the bus address and the node number
    can all differ between one poll and the next.

    ``serial`` names the exact scanner to wait for. ``None`` means ANY scanner,
    which is the meaning the enumeration-only :func:`wait_for_scanner` had and
    is preserved deliberately: no configuration is read here, so a caller that
    wants the configured device resolves it and passes it - the GUI does
    exactly that, which is what keeps one run's colour, depth and readiness
    aimed at one identity. With ``None`` and several attached, this returns
    ready WITHOUT a node: choosing between them would be a guess, the stream's
    own resolver is the authority on that ambiguity, and blocking for the full
    timeout before letting it say so would just be a silent tax.
    :attr:`CaptureReadiness.deferred` records that it happened.

    Nothing here opens a device node. The result is decided from sysfs and
    ``stat``, which is what lets it be asked repeatedly without disturbing a
    device that answers an ill-timed open by re-enumerating.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    polls = 0
    while True:
        polls += 1
        state = _readiness_now(serial, profile, sysfs_root, dev_root)
        now = time.monotonic()
        state = replace(state, polls=polls, waited_s=now - started)
        if state.ready:
            return state
        if now >= deadline:
            return replace(state, timed_out=True)
        time.sleep(poll_s)


def stream_frames(
    serial: str = AUTHORIZED_SERIAL,
    *,
    frames: int,
    on_frame: Callable[[bytes, FrameRecord], None] | None = None,
    after_frame: Callable[[bytes, FrameRecord], None] | None = None,
    on_delivery: Callable[[bytes, FrameRecord], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    decimate: int | Cadence = 1,
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

    ``decimate`` may instead be a CALLABLE, asked once per delivery for the
    spacing to use from the next kept frame onwards. That is how a caller holds
    a measured frame spacing on a capture whose speed varies - see
    :mod:`scanner3d.cadence` - and it changes nothing for a caller that passes a
    number, which every existing one does. Two properties are worth stating
    because they are the ones a varying cadence could plausibly get wrong:

    - **A change takes effect at the next kept frame, not retroactively.** The
      selection is a countdown of deliveries between keeps rather than a modulo
      of the delivered index, because changing the modulus mid-run makes the
      phase jump - two keeps back to back, or a skip run longer than either the
      old value or the new one. The one exception is a change to a FINER
      cadence, which shortens the skip run already in progress: that can only
      bring a keep forward, and a controller only ever asks for a finer cadence
      because the current one was already too coarse.
    - **The controller is never allowed to end the run.** It is asked only after
      the buffer has been re-queued, and if it raises or answers with something
      that is not a usable decimation, the previous value stands and
      :attr:`StreamResult.decimate_refusals` counts it. Raising out of the
      streaming loop is the one failure this module must not have; see
      ``should_stop`` below for what a userspace-owned buffer does to this
      hardware.

    ``on_delivery`` receives EVERY complete delivery, whether or not decimation
    selected it. It exists because how often a picture should reach a screen and
    how often a frame should be written, deprojected and aligned are two
    different questions with two different right answers: at ``decimate=3`` the
    operator watched a depth view that moved at a third of the rate the scanner
    was actually delivering, while the expensive work stayed correctly thinned.
    A caller that passes only ``on_frame`` is unaffected, and every existing one
    does exactly that.

    When one complete delivery is both previewed and kept, the payload is copied
    ONCE and handed to both callbacks. A second copy per frame would be half a
    megabyte of pointless memory traffic at the measured 4.6 fps, and the bytes
    are immutable, so there is nothing for the two consumers to do to each other.

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

    ``after_frame`` receives the same ``(raw_bytes, record)`` as ``on_frame``,
    for the same frames, but AFTER the buffer has been re-queued. That is the
    difference the whole parameter exists for: work done in ``on_frame`` sits
    between ``VIDIOC_DQBUF`` and ``VIDIOC_QBUF``, so it holds a driver buffer
    for its whole duration and stalls both the depth deliveries behind it and
    anything the caller draws from them. Work that is expensive and does not
    have to happen before the buffer goes back - deprojecting a frame the caller
    is holding in memory, a global search against a model - belongs here, where
    it costs the device nothing and a slow one leaves every buffer queued.

    The payload handed to it is the same copy ``on_frame`` was given, taken from
    the mmap before the re-queue, so it is not affected by the buffer being
    refilled.

    ``should_stop`` is polled after each buffer has been RE-QUEUED, and ending
    the stream there is the only safe place to end it early. A callback that
    raises instead skips the ``VIDIOC_QBUF`` for the buffer it was handed, so
    cleanup then runs with a buffer still owned by userspace — measured on this
    hardware, that makes the device re-enumerate, which to a caller looks
    exactly like the scanner being unplugged. Stopping through this flag
    returns every buffer first and leaves the device usable immediately.

    A run ended that way comes back with ``stopped_early`` set and
    ``stop_reason`` equal to :data:`CALLER_STOP_REASON`, which is a normal
    return rather than an exception. A caller that needs to present "you asked
    me to stop" differently from "this finished" reads those two fields; nothing
    here decides that for it.

    Cleanup runs in ``finally`` — ``STREAMOFF``, unmap, release, close — so a
    callback that raises still leaves the device closed rather than held, but
    "closed" and "healthy" are not the same thing and only the flag gives both.
    """
    started_decimate = ensure_request(frames, decimate, duration_s)

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

    # ``spacing`` is the decimation in force; ``pending`` is how many more
    # deliveries have to arrive before the next keep slot. Zero to start, so the
    # FIRST delivery is a keep - which is what ``index % decimate == 0`` did and
    # what a caller asking for one frame in three has always got.
    #
    # From ``ensure_request``, which already called a controller once and
    # refused the run if it could not answer - so this first value is a checked
    # one, the loop never starts on a guess, and the controller is asked exactly
    # once per decision rather than twice before the first delivery.
    spacing = started_decimate
    started_at = spacing
    pending = 0
    lowest = highest = spacing
    decimate_refusals = 0

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
            # still selects on ARRIVAL, so a short frame consumes its slot
            # rather than shifting the next one into it: the selection stays a
            # property of arrival order, not of which frames happened to be
            # usable.
            #
            # A COUNTDOWN rather than ``index % decimate``, and that is what
            # makes a varying cadence expressible at all. The modulo is a phase
            # function of the delivered index: change the modulus half way
            # through and the phase jumps, so two keeps can land back to back or
            # a skip run can stretch well past either the old value or the new
            # one - neither of which is a spacing anybody asked for. A countdown
            # has one meaning under change and it is the meaning the name has
            # always had: this many deliveries between keeps. For a fixed
            # ``decimate`` it selects the identical frames the modulo did, 0, 3,
            # 6 at three, which is what keeps every existing run unchanged.
            selected = complete and pending == 0
            if pending == 0:
                # Refilled from the value in force AT THE KEEP, so a cadence
                # that changed mid-skip takes effect at the next kept frame
                # rather than retroactively. The one exception is a change
                # DOWNWARDS, clamped below where it is resolved: that can only
                # bring a keep forward, never push one back.
                pending = spacing - 1
            else:
                pending -= 1
            if selected:
                kept += 1
            wanted = on_delivery is not None or (
                selected and (on_frame is not None or after_frame is not None)
            )
            if complete and wanted:
                # Copy before re-queueing: the kernel may refill this mapping
                # the instant it is queued, so a view handed out here would
                # mutate. One copy for both consumers.
                payload = bytes(maps[buffer_index][:used])
                # Display first, and the keep second, because the keep is the
                # expensive one: writing the frame, deprojecting it and aligning
                # it all happen inside ``on_frame``, and a picture that waits
                # behind them arrives late for no reason. A short delivery
                # reaches NEITHER - it is not a frame, and previewing 70% of one
                # would put a torn picture on the screen and call it the scanner.
                if on_delivery is not None:
                    on_delivery(payload, record)
                if selected and on_frame is not None:
                    on_frame(payload, record)

            # Unconditional, and reached by a short delivery on exactly the same
            # path as a complete one: the buffer is the driver's either way, and
            # a run that kept one back would leave userspace owning it at
            # teardown - which this hardware answers by re-enumerating.
            fcntl.ioctl(fd, VIDIOC_QBUF, buf, True)

            # The buffer is the driver's again, so this is the first instant at
            # which a caller may take as long as it likes. ``payload`` is the
            # copy taken above - which is why the guard that makes it includes
            # this callback - so it is unaffected by the re-queue.
            if complete and wanted and selected and after_frame is not None:
                after_frame(payload, record)

            # Both of these are polled here, AFTER re-queueing, so the driver
            # owns every buffer again before either one can end the run or take
            # any time.
            if should_stop is not None and should_stop():
                stopped_early, reason = True, CALLER_STOP_REASON
                break

            spacing, unanswered = _resolve_decimate(decimate, spacing)
            if unanswered:
                decimate_refusals += 1
            lowest = min(lowest, spacing)
            highest = max(highest, spacing)
            # A cadence that just went FINER shortens the skip run already in
            # progress instead of waiting it out. At the widest setting that
            # wait is nine more deliveries - about two seconds at the measured
            # rate - and the reason a controller asks for a finer cadence is
            # always that the last one was already too coarse. Clamping can only
            # bring the next keep forward, so a coarser answer still takes
            # effect at the next keep and never extends the current run.
            pending = min(pending, spacing - 1)
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
        decimate=started_at,
        decimate_lowest=lowest,
        decimate_highest=highest,
        decimate_refusals=decimate_refusals,
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
