"""One fake V4L2 kernel, shared by every capture profile's lifecycle tests.

``tests/test_v4l2_uapi.py`` proves the *transcription* is right: the ioctl
numbers, struct sizes and field offsets match what the host's own
``<linux/videodev2.h>`` declares. The files that import this module prove the
*lifecycle* is right, which is a different question and the one the hardware
can only be asked once.

There is one fake kernel here for the same reason there is one capture engine
in :mod:`scanner3d.v4l2_capture`: a second copy, written to exercise a second
profile, would be a second set of assumptions about what the real kernel does,
and the two would drift. :class:`FakeKernel` therefore takes a
:class:`~scanner3d.v4l2_capture.CaptureProfile` and answers the nine ioctls the
way a healthy device would for *that* profile, while every knob a test needs to
break - a short ``REQBUFS``, a buffer past a ceiling, a poll that only times
out, an ``EINTR`` on ``STREAMON``, a ``STREAMOFF`` that fails during cleanup -
stays exactly where it was.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
)
from scanner3d.rgb_frame import RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES
from scanner3d.v4l2_capture import (
    BUFFER_FORMAT,
    BYTES_PER_PIXEL,
    CAPABILITY_FORMAT,
    DEPTH_PROFILE,
    FORMAT_FORMAT,
    PIX_FORMAT_FORMAT,
    REQUESTBUFFERS_FORMAT,
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_CAP_DEVICE_CAPS,
    V4L2_CAP_STREAMING,
    V4L2_CAP_VIDEO_CAPTURE,
    V4L2_FIELD_NONE,
    V4L2_MEMORY_MMAP,
    VIDIOC_DQBUF,
    VIDIOC_G_FMT,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMOFF,
    VIDIOC_STREAMON,
    CaptureProfile,
    V4l2FrameBackend,
)
from tests.conftest import REAL_SERIAL, SysfsBuilder

POLLIN = 1
POLLERR = 8


def default_bytesperline(profile: CaptureProfile) -> int:
    """What ``uvcvideo`` reports for this profile's mode.

    Zero for a compressed bitstream, which has no row stride, and one tight row
    for a raster.
    """
    return 0 if profile.compressed else profile.width * BYTES_PER_PIXEL


def default_sizeimage(profile: CaptureProfile, bytesperline: int | None = None) -> int:
    """The buffer size a healthy driver would report for this profile.

    For a raster that is the geometry the driver itself just described: the
    *effective* stride times the height. A test that widens ``bytesperline`` to
    model padded rows therefore gets the ``sizeimage`` a real driver reports
    alongside that stride, not the smaller one tight rows would imply - the two
    fields are one answer, and a driver whose ``sizeimage`` cannot hold its own
    rows is refused before streaming rather than modelled as healthy.

    For MJPEG it is :data:`~scanner3d.rgb_frame.RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES`
    - the ``sizeimage`` ``uvcvideo`` was *measured* answering for this mode on
    this unit in scanner-010. It is deliberately not the descriptor's
    ``dwMaxVideoFrameBufferSize``: modelling the healthy driver as copying that
    field into ``sizeimage`` was the assumption the hardware disproved, and a
    fake kernel that kept it would leave the geometry the real device actually
    produces untested. A compressed bitstream has no stride, so
    ``bytesperline`` does not apply.
    """
    if profile.compressed:
        return RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES
    stride = default_bytesperline(profile) if bytesperline is None else bytesperline
    return stride * profile.height


class FakeMapping:
    """Stands in for ``mmap.mmap``, counting its own unmaps."""

    def __init__(self, payload: bytes, reader: Callable[[], bytes] | None = None) -> None:
        self._payload = payload
        self._reader = reader
        self.closes = 0

    def close(self) -> None:
        self.closes += 1
        if self.closes > 1:  # pragma: no cover - the assertion is the point
            raise AssertionError("a mapping was unmapped twice")

    def __len__(self) -> int:
        return len(self._payload)

    def __getitem__(self, index: slice, /) -> bytes:
        payload = self._payload if self._reader is None else self._reader()
        return payload[index]


class FakeKernel:
    """A V4L2 device that answers the nine ioctls, however a test wants.

    Defaults describe the healthy unit *for the profile it is given*: capture
    and streaming capabilities, that profile's own mode and buffer geometry,
    four buffers, a readable poll and one complete frame. Every default is
    derived from the profile rather than hard-coded, so the same fake kernel
    serves a raster and a compressed capture without either being special-cased
    here.
    """

    def __init__(
        self,
        *,
        profile: CaptureProfile = DEPTH_PROFILE,
        capabilities: int = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING | V4L2_CAP_DEVICE_CAPS,
        device_caps: int = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING,
        width: int | None = None,
        height: int | None = None,
        pixelformat: int | None = None,
        bytesperline: int | None = None,
        sizeimage: int | None = None,
        buf_type: int = V4L2_BUF_TYPE_VIDEO_CAPTURE,
        granted_buffers: int | None = None,
        buffer_length: int | None = None,
        buffer_offset_step: int | None = None,
        memory: int = V4L2_MEMORY_MMAP,
        poll_revents: int = POLLIN,
        bytesused: int | None = None,
        dequeued_index: int = 0,
        buffer_flags: int = 0,
        errors: dict[int, int] | None = None,
        raises: dict[int, BaseException] | None = None,
        frame_byte: int = 0xAB,
        frame_payload: bytes | None = None,
        frame_payloads: tuple[bytes, ...] | None = None,
        frame_sequences: tuple[int, ...] | None = None,
    ) -> None:
        self.profile = profile
        self.capabilities = capabilities
        self.device_caps = device_caps
        self.width = profile.width if width is None else width
        self.height = profile.height if height is None else height
        self.pixelformat = profile.pixelformat if pixelformat is None else pixelformat
        self.bytesperline = default_bytesperline(profile) if bytesperline is None else bytesperline
        self.sizeimage = (
            default_sizeimage(profile, self.bytesperline) if sizeimage is None else sizeimage
        )
        self.buf_type = buf_type
        self.granted_buffers = (
            profile.requested_buffers if granted_buffers is None else granted_buffers
        )
        #: A healthy driver maps exactly the ``sizeimage`` it just reported, so the
        #: default follows that field rather than the profile's unpadded geometry.
        self.buffer_length = self.sizeimage if buffer_length is None else buffer_length
        self.buffer_offset_step = (
            self.buffer_length if buffer_offset_step is None else buffer_offset_step
        )
        self.memory = memory
        self.poll_revents = poll_revents
        #: The bytes the mapping will contain, ahead of ``frame_byte`` padding.
        #: A raster profile has no natural payload and leaves this ``None``; a
        #: compressed one is handed a synthetic JPEG so the structural
        #: inspection under test has something real to walk.
        self.frame_payload = frame_payload
        self.frame_payloads = frame_payloads
        self.frame_sequences = frame_sequences
        self._series_index = 0
        if bytesused is not None:
            self.bytesused = bytesused
        elif frame_payload is not None:
            self.bytesused = len(frame_payload)
        else:
            self.bytesused = self.sizeimage
        self.dequeued_index = dequeued_index
        self.buffer_flags = buffer_flags
        #: ``{ioctl_request: errno}`` - that ioctl fails with that errno.
        self.errors = errors or {}
        #: ``{ioctl_request: exception}`` - that ioctl raises it verbatim.
        self.raises = raises or {}
        self.frame_byte = frame_byte

        self.calls: list[tuple[int, bytes]] = []
        self.mappings: list[FakeMapping] = []
        self.map_calls: list[tuple[int, int, int]] = []
        self.polls: list[tuple[int, float]] = []
        self.opened: list[str] = []
        self.closed: list[int] = []
        self.fd = 77

    # -- the injected primitives ------------------------------------------

    def opener(self, path: str) -> int:
        # A real descriptor on the fixture node, because the backend re-``fstat``s
        # what it opened to prove the device did not change underneath it. A
        # bare integer would fail that check for the wrong reason.
        self.opened.append(path)
        self.fd = os.open(path, os.O_RDONLY)
        return self.fd

    def closer(self, fd: int) -> None:
        self.closed.append(fd)
        os.close(fd)

    def poll(self, fd: int, timeout_s: float) -> int:
        self.polls.append((fd, timeout_s))
        if self.frame_payloads:
            self.frame_payload = self.frame_payloads[
                min(self._series_index, len(self.frame_payloads) - 1)
            ]
            self.bytesused = len(self.frame_payload)
            self._series_index += 1
        return self.poll_revents

    def mmap(self, fd: int, length: int, offset: int) -> FakeMapping:
        self.map_calls.append((fd, length, offset))
        mapping = FakeMapping(self._buffer_contents(length), lambda: self._buffer_contents(length))
        self.mappings.append(mapping)
        return mapping

    def _buffer_contents(self, length: int) -> bytes:
        """What one mapped buffer holds: the payload, then filler to ``length``."""
        if self.frame_payload is None:
            return bytes([self.frame_byte]) * length
        payload = self.frame_payload[:length]
        return payload + bytes([self.frame_byte]) * (length - len(payload))

    def ioctl(self, fd: int, request: int, packed: bytearray, mutate: bool) -> int:
        assert mutate is True, "every ioctl here must pass the mutate-in-place flag"
        self.calls.append((request, bytes(packed)))
        if request in self.raises:
            raise self.raises[request]
        if request in self.errors:
            number = self.errors[request]
            raise OSError(number, os.strerror(number))
        handler = {
            VIDIOC_QUERYCAP: self._querycap,
            VIDIOC_G_FMT: self._fmt,
            VIDIOC_S_FMT: self._fmt,
            VIDIOC_REQBUFS: self._reqbufs,
            VIDIOC_QUERYBUF: self._querybuf,
            VIDIOC_QBUF: self._noop,
            VIDIOC_DQBUF: self._dqbuf,
            VIDIOC_STREAMON: self._noop,
            VIDIOC_STREAMOFF: self._noop,
        }.get(request)
        if handler is None:  # pragma: no cover - the assertion is the point
            raise AssertionError(f"unexpected ioctl 0x{request:08x}")
        handler(packed)
        return 0

    # -- responses ---------------------------------------------------------

    def _noop(self, packed: bytearray) -> None:
        return None

    def _querycap(self, packed: bytearray) -> None:
        packed[:] = struct.pack(
            CAPABILITY_FORMAT,
            b"uvcvideo",
            f"ZXProduct: {self.profile.role.capitalize()}Cam{REAL_SERIAL}".encode()[:31],
            b"usb-0000:00:14.0-1.1",
            0x00060008,
            self.capabilities,
            self.device_caps,
            b"\x00" * 12,
        )

    def _fmt(self, packed: bytearray) -> None:
        pix = struct.pack(
            PIX_FORMAT_FORMAT,
            self.width,
            self.height,
            self.pixelformat,
            V4L2_FIELD_NONE,
            self.bytesperline,
            self.sizeimage,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        packed[:] = struct.pack(FORMAT_FORMAT, self.buf_type, pix.ljust(200, b"\x00"))

    def _reqbufs(self, packed: bytearray) -> None:
        requested = struct.unpack(REQUESTBUFFERS_FORMAT, bytes(packed))[0]
        # A release (count=0) is answered literally; only an allocation is
        # allowed to come back short.
        count = 0 if requested == 0 else self.granted_buffers
        packed[:] = struct.pack(
            REQUESTBUFFERS_FORMAT, count, self.buf_type, self.memory, 0, 0, b"\x00" * 3
        )

    def _buffer(self, index: int, *, bytesused: int, flags: int) -> bytes:
        return struct.pack(
            BUFFER_FORMAT,
            index,
            self.buf_type,
            bytesused,
            flags,
            V4L2_FIELD_NONE,
            1700000000,
            123456,
            0,
            0,
            0,
            0,
            0,
            0,
            b"\x00" * 4,
            (
                self.frame_sequences[min(self._series_index - 1, len(self.frame_sequences) - 1)]
                if self.frame_sequences
                else 42
            ),
            self.memory,
            struct.pack("=I", index * self.buffer_offset_step).ljust(8, b"\x00"),
            self.buffer_length,
            0,
            0,
        )

    def _querybuf(self, packed: bytearray) -> None:
        index = struct.unpack(BUFFER_FORMAT, bytes(packed))[0]
        packed[:] = self._buffer(index, bytesused=0, flags=0)

    def _dqbuf(self, packed: bytearray) -> None:
        packed[:] = self._buffer(
            self.dequeued_index, bytesused=self.bytesused, flags=self.buffer_flags
        )

    # -- assertions helpers -------------------------------------------------

    @property
    def requests(self) -> list[int]:
        return [request for request, _ in self.calls]

    def count(self, request: int) -> int:
        return self.requests.count(request)


@contextmanager
def noop_shield() -> Iterator[list[int]]:
    yield []


@contextmanager
def noop_guard() -> Iterator[None]:
    yield None


@pytest.fixture
def sysfs(tmp_path: Path) -> SysfsBuilder:
    builder = SysfsBuilder(tmp_path)
    base = builder.add_usb_device()
    builder.add_usbfs()
    # video5, not video0: the node number must never be what resolution keys on.
    builder.add_interface(
        base,
        0,
        interface_string=f"DepthCam{REAL_SERIAL}",
        video_nodes=(("video5", 0), ("video6", 1)),
    )
    builder.add_interface(
        base,
        2,
        interface_string=f"RGBCam{REAL_SERIAL}",
        video_nodes=(("video0", 0), ("video1", 1)),
    )
    return builder


class SteppedClock:
    """A monotonic clock a test drives.

    Every read costs ``tick`` seconds, and a test may spend more time
    explicitly with :meth:`advance` - which is how a slow ``STREAMON`` or an
    exhausted budget is expressed without a real wait.
    """

    def __init__(self, tick: float = 0.001) -> None:
        self.now = 0.0
        self.tick = tick
        self.reads = 0

    def __call__(self) -> float:
        self.reads += 1
        self.now += self.tick
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_backend(
    kernel: FakeKernel,
    sysfs: SysfsBuilder,
    clock: Callable[[], float] | None = None,
    cls: type[V4l2FrameBackend] = V4l2FrameBackend,
) -> V4l2FrameBackend:
    """A backend whose every syscall is ``kernel``'s, and whose clock ticks 1ms.

    ``cls`` selects which profile-bound backend is built. The signal guard and
    the termination shield are the no-op pair above, so a test asserting on
    cleanup is asserting on the engine's own ordering rather than on the
    process-wide signal machinery :mod:`scanner3d.ownership` already has its own
    tests for.
    """
    ticks = iter(range(0, 10_000))

    def default_clock() -> float:
        return next(ticks) / 1000.0

    class FakeIlluminationBackend:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True,
                True,
                ShutdownStatus.COMMAND_ACCEPTED,
                "synthetic fixed off accepted",
                transfer=ShutdownTransferRecord(60, True, "synthetic", 0.0),
            )

    def fake_illumination_factory(_sysfs, _dev):
        return FakeIlluminationBackend()

    return cls(
        opener=kernel.opener,
        ioctl_fn=kernel.ioctl,
        mmap_fn=kernel.mmap,
        poll_fn=kernel.poll,
        closer=kernel.closer,
        clock=default_clock if clock is None else clock,
        sysfs_root=sysfs.sysfs,
        dev_root=sysfs.dev,
        signal_guard=noop_guard,
        termination_shield=noop_shield,
        illumination_backend_factory=fake_illumination_factory,
    )
