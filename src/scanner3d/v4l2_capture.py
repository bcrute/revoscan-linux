"""One bounded V4L2/MMAP attempt to capture a single raw depth frame.

This is the first path in this repository that asks the scanner for *image
data*, and it is deliberately the least clever way to do it: the kernel's own
``uvcvideo`` driver stays bound, the capture node is opened as any V4L2 client
would open it, and the entire repertoire is nine ioctls issued once each in one
fixed order. Nothing here detaches a driver, claims an interface, sends a
vendor or HID control transfer, or writes a PROBE/COMMIT of its own. Everything
this module can do, ``uvcvideo`` does on its behalf.

What is fixed at import time, and is not a parameter of anything:

* the **role**, ``depth`` - resolved from the USB interface string descriptor
  (``DepthCam<serial>``) and the V4L2 node index, exactly as
  :mod:`scanner3d.discovery` already does it. No caller supplies a node path,
  and ``/dev/video0`` is never assumed: the number is assigned in probe order
  and moves under replug, reboot, and other UVC devices;
* the **mode**, :data:`FOURCC` ``Y16 `` at :data:`FRAME_WIDTH` x
  :data:`FRAME_HEIGHT`, which the recorded configuration descriptor of this
  unit advertises at a default 15 fps. There is no flag for a format, a size, a
  frame rate, a buffer count, a timeout, a retry, or a control;
* the **bounds**: at most :data:`REQUESTED_BUFFERS` buffers, each at most
  :data:`MAX_BUFFER_BYTES`, at most :data:`MAX_TOTAL_MAPPED_BYTES` mapped in
  total, one poll carrying the whole :data:`CAPTURE_DEADLINE_S` budget, at most
  one ``VIDIOC_DQBUF``, and an artifact ceiling of
  :data:`MAX_RAW_ARTIFACT_BYTES`.

No loop in the live path is unbounded. The buffer loops are bounded by the
count the kernel returned, which is itself bounded by
:data:`REQUESTED_BUFFERS`; the wait is a single ``poll`` whose timeout is the
whole remaining budget, so a timeout cannot be restarted; the dequeue happens
at most once and the frame is never requeued; and there is no second
``VIDIOC_STREAMON``.

**The UAPI numbers are cross-checked, not trusted.** Every ioctl request
number, struct size and field offset below is a hand transcription of
``<linux/videodev2.h>``, and a wrong one does not fail loudly - it silently
scribbles on the wrong bytes. ``tests/uapi/v4l2_layout.c`` is compiled and run
against the host's installed headers and prints the real numbers as JSON;
``tests/test_v4l2_uapi.py`` diffs this module against that output. On top of
that, :func:`ensure_layout_intact` re-derives every size at run time and
refuses to issue a single ioctl if any of them has moved, so a build on an ABI
this transcription does not describe stops before it opens anything.

**Cleanup is owed from the moment an obligation is created, and is discharged
exactly once.** The four obligations are the stream, the queue, the mappings
and the descriptor, and each is tracked pessimistically - raised *before* the
call that creates it, lowered only by that call reporting a clean failure -
because a signal arriving inside an ioctl leaves the kernel's state ambiguous.
Cleanup runs under :func:`scanner3d.ownership.defer_termination`, the same
shield the interface-0 lifecycle uses, so neither SIGINT nor SIGTERM can abort
it half-way; whatever it absorbs is reported rather than hidden. Any cleanup
failure overrides frame success and forces a nonzero exit.

The one place this module departs from the assignment's literal ordering is
that it unmaps before releasing the queue. The V4L2 specification is explicit
that ``VIDIOC_REQBUFS`` answers ``EBUSY`` when "reallocation of buffers was
attempted although one or more are still mapped", so releasing the queue first
would fail on every successful run. Both are still attempted exactly once, in a
fixed order, and both are reported independently.

**A zero-frame timeout is a result, not a failure to work around.** If the
stream starts and no buffer becomes ready inside the deadline, that is recorded
in full - negotiated mode, buffer metadata, poll timing, cleanup - and the run
stops. This module has no fallback mode, no vendor initialization, and no
second attempt.
"""

from __future__ import annotations

import contextlib
import enum
import errno as errno_mod
import fcntl
import hashlib
import mmap as mmap_mod
import os
import select
import signal
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

from scanner3d.config import EXAMPLE_SERIAL, configured_serial
from scanner3d.control import Clock, Closer, IoctlFn, Opener, identity_drift
from scanner3d.discovery import (
    DEFAULT_DEV_ROOT,
    DEFAULT_SYSFS_ROOT,
    EXPECTED_MANUFACTURER,
    CameraFunction,
    ScannerDevice,
    VideoNode,
    find_scanners,
)
from scanner3d.errors import DeviceAbsentError, V4l2ArtifactIncomplete, V4l2CaptureNotAllowed
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    UsbfsIlluminationShutdownBackend,
)
from scanner3d.illumination_shutdown import (
    not_required as illumination_not_required,
)
from scanner3d.ownership import (
    CLEANUP_FAILED_EXIT,
    SignalGuard,
    SignalReceived,
    TerminationShield,
    cleanup_on_termination,
    defer_termination,
)

# --- what this module is allowed to touch ------------------------------------

#: The serial of the one unit the owner authorized this capture for. The live
#: path refuses any other, so "the exact device" is not a runtime judgement
#: call. Deliberately restated here rather than imported from a vendor-control
#: module: this authorization is for a kernel capture, not for those transfers.
#: The scanner this installation may act on. Read from configuration rather
#: than hard-coded: several operations here detach kernel drivers or write
#: vendor property keys, and must not be aimed at whatever happens to be
#: plugged in. Empty when nothing is configured - call
#: scanner3d.config.resolve_serial() for a usable value, which falls back to
#: the single attached scanner and refuses when that is ambiguous.
AUTHORIZED_SERIAL: Final = configured_serial() or EXAMPLE_SERIAL

#: The logical camera this module can reach. :mod:`scanner3d.discovery` derives
#: it from the interface string descriptor, never from a ``videoN`` number.
CAPTURE_ROLE: Final = "depth"

#: The V4L2 node kind within that role. ``uvcvideo`` registers the capture node
#: at index 0 and the metadata node at index 1; only the former is in scope.
CAPTURE_KIND: Final = "capture"

#: The driver that must own the capture interface, before and after. This
#: module never unbinds it.
EXPECTED_INTERFACE_DRIVER: Final = "uvcvideo"

# --- the one mode ------------------------------------------------------------

#: ``V4L2_PIX_FMT_Y16``: 16-bit little-endian greyscale, the raw depth format
#: this unit's descriptors advertise at 640x400. Note the trailing space - the
#: fourcc is four characters and ``"Y16"`` is not it.
FOURCC: Final = "Y16 "

FRAME_WIDTH: Final = 640
FRAME_HEIGHT: Final = 400

#: The descriptor default for this mode. Recorded as evidence of what was
#: negotiated; this module never issues ``VIDIOC_S_PARM`` to ask for it.
DESCRIPTOR_FRAME_RATE_HZ: Final = 15.0

#: Bytes per pixel of ``Y16 ``. One 16-bit sample, no subsampling.
BYTES_PER_PIXEL: Final = 2

#: The tightly packed size of one complete frame: 640 * 400 * 2.
EXPECTED_FRAME_BYTES: Final = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL

#: The smallest ``bytesperline`` that can hold one row. Anything larger is row
#: padding, which is preserved as raw evidence and unpacked separately;
#: anything smaller means the kernel is not describing the frame this module
#: asked for, and nothing is unpacked from it.
MIN_BYTES_PER_LINE: Final = FRAME_WIDTH * BYTES_PER_PIXEL

# --- the bounds --------------------------------------------------------------

#: How many MMAP buffers are requested. The kernel may return fewer.
REQUESTED_BUFFERS: Final = 4

#: The fewest returned buffers this module will stream with. Below two there is
#: no queue to speak of, and a driver that offers one is not the configuration
#: this task was scoped for.
MIN_ACCEPTED_BUFFERS: Final = 2

#: Hard ceiling on a single buffer's mapped length, checked against what
#: ``VIDIOC_QUERYBUF`` reports before ``mmap`` is called.
MAX_BUFFER_BYTES: Final = 1024 * 1024

#: Hard ceiling on the total bytes mapped across every buffer.
MAX_TOTAL_MAPPED_BYTES: Final = 4 * 1024 * 1024

#: The whole wait budget, in seconds, measured on a monotonic clock from the
#: moment ``VIDIOC_STREAMON`` returned. One ``poll`` call carries all of it, so
#: there is no timeout to restart.
CAPTURE_DEADLINE_S: Final = 8.0

#: Ceiling on any single raw-frame artifact. A frame larger than this is
#: reported as oversize and *not* written truncated: a truncated file that
#: looks like a frame is worse evidence than no file at all.
MAX_RAW_ARTIFACT_BYTES: Final = 1024 * 1024

#: Where the live path writes its evidence when the caller names no directory.
#: An output directory shapes only local files, never the hardware, which is
#: why it is the one thing about the live run a caller may choose.
DEFAULT_OUTPUT_DIR: Final = Path("/tmp/scanner3d-depth-frame")

RAW_ARTIFACT_NAME: Final = "depth-frame-raw.bin"
PACKED_ARTIFACT_NAME: Final = "depth-frame-packed.bin"

# --- the V4L2 UAPI, transcribed ----------------------------------------------

# Struct formats use ``=`` - native byte order, standard sizes, no implicit
# alignment - so every pad byte below is written out rather than inferred.
# tests/uapi/v4l2_layout.c prints the real offsets and the test suite diffs
# them against these, field by field.

#: ``struct v4l2_capability``: driver[16], card[32], bus_info[32], version,
#: capabilities, device_caps, reserved[3].
CAPABILITY_FORMAT: Final = "=16s32s32sIII12s"

#: ``struct v4l2_pix_format``: twelve ``__u32`` - width, height, pixelformat,
#: field, bytesperline, sizeimage, colorspace, priv, flags, ycbcr_enc,
#: quantization, xfer_func.
PIX_FORMAT_FORMAT: Final = "=12I"

#: ``struct v4l2_format``: ``__u32 type``, four bytes of alignment padding
#: (the union is 8-byte aligned), then the 200-byte ``fmt`` union.
FORMAT_FORMAT: Final = "=I4x200s"

#: Size of the ``fmt`` union, which the pixel format is packed into and the
#: rest of which is zero.
FORMAT_UNION_BYTES: Final = 200

#: ``struct v4l2_requestbuffers``: count, type, memory, capabilities,
#: ``__u8 flags``, reserved[3].
REQUESTBUFFERS_FORMAT: Final = "=IIIIB3s"

#: ``struct v4l2_buffer``. The ``m`` union is unpacked as eight opaque bytes
#: rather than a ``Q``: only its first ``__u32`` (``m.offset``) is meaningful
#: for MMAP, and decoding those four bytes natively is correct on either
#: endianness, where masking a 64-bit load is not.
BUFFER_FORMAT: Final = "=IIIII4xqqII4B4sII8sIIi4x"

#: ``VIDIOC_STREAMON``/``VIDIOC_STREAMOFF`` take a bare ``int`` buffer type.
BUFTYPE_FORMAT: Final = "=i"

CAPABILITY_BYTES: Final = struct.calcsize(CAPABILITY_FORMAT)
PIX_FORMAT_BYTES: Final = struct.calcsize(PIX_FORMAT_FORMAT)
FORMAT_BYTES: Final = struct.calcsize(FORMAT_FORMAT)
REQUESTBUFFERS_BYTES: Final = struct.calcsize(REQUESTBUFFERS_FORMAT)
BUFFER_BYTES: Final = struct.calcsize(BUFFER_FORMAT)
BUFTYPE_BYTES: Final = struct.calcsize(BUFTYPE_FORMAT)

#: The sizes ``<linux/videodev2.h>`` declares on the ABI this module was
#: written against (x86_64, kernel UAPI 6.x). :func:`ensure_layout_intact`
#: compares the formats above against these before anything is opened, because
#: the ioctl request numbers below encode the size and a mismatch would put a
#: wrong number on a real descriptor.
EXPECTED_STRUCT_BYTES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "v4l2_capability": 104,
        "v4l2_pix_format": 48,
        "v4l2_format": 208,
        "v4l2_requestbuffers": 20,
        "v4l2_buffer": 88,
        "buf_type": 4,
    }
)

_IOC_NRBITS: Final = 8
_IOC_TYPEBITS: Final = 8
_IOC_SIZEBITS: Final = 14
_IOC_WRITE: Final = 1
_IOC_READ: Final = 2

_V4L2_IOC_TYPE: Final = ord("V")


def _ioc(direction: int, number: int, size: int) -> int:
    """The kernel's ``_IOC`` encoding, for the ``'V'`` (V4L2) ioctl type.

    Deliberately a local re-derivation rather than a call into
    :mod:`scanner3d.control`: that module documents ``USBDEVFS_CONTROL`` as the
    only usbfs command number in this package, and reusing its helper here
    would put V4L2 numbers behind a usbfs-facing name. The encoding itself is
    the same architecture-independent generic one, and a test asserts these
    numbers against the C helper rather than against this function.
    """
    if not 0 <= size < (1 << _IOC_SIZEBITS):
        raise ValueError(f"ioctl payload size {size} out of range")
    return (
        (direction << (_IOC_NRBITS + _IOC_TYPEBITS + _IOC_SIZEBITS))
        | (size << (_IOC_NRBITS + _IOC_TYPEBITS))
        | (_V4L2_IOC_TYPE << _IOC_NRBITS)
        | number
    )


_IOWR: Final = _IOC_READ | _IOC_WRITE

VIDIOC_QUERYCAP: Final = _ioc(_IOC_READ, 0, CAPABILITY_BYTES)
VIDIOC_G_FMT: Final = _ioc(_IOWR, 4, FORMAT_BYTES)
VIDIOC_S_FMT: Final = _ioc(_IOWR, 5, FORMAT_BYTES)
VIDIOC_REQBUFS: Final = _ioc(_IOWR, 8, REQUESTBUFFERS_BYTES)
VIDIOC_QUERYBUF: Final = _ioc(_IOWR, 9, BUFFER_BYTES)
VIDIOC_QBUF: Final = _ioc(_IOWR, 15, BUFFER_BYTES)
VIDIOC_DQBUF: Final = _ioc(_IOWR, 17, BUFFER_BYTES)
VIDIOC_STREAMON: Final = _ioc(_IOC_WRITE, 18, BUFTYPE_BYTES)
VIDIOC_STREAMOFF: Final = _ioc(_IOC_WRITE, 19, BUFTYPE_BYTES)

#: Every V4L2 command this module knows. A test asserts that no other request
#: number appears anywhere in the file, so the closed repertoire is checked
#: against the source rather than promised by this comment.
ALLOWED_IOCTLS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "VIDIOC_QUERYCAP": VIDIOC_QUERYCAP,
        "VIDIOC_G_FMT": VIDIOC_G_FMT,
        "VIDIOC_S_FMT": VIDIOC_S_FMT,
        "VIDIOC_REQBUFS": VIDIOC_REQBUFS,
        "VIDIOC_QUERYBUF": VIDIOC_QUERYBUF,
        "VIDIOC_QBUF": VIDIOC_QBUF,
        "VIDIOC_DQBUF": VIDIOC_DQBUF,
        "VIDIOC_STREAMON": VIDIOC_STREAMON,
        "VIDIOC_STREAMOFF": VIDIOC_STREAMOFF,
    }
)

V4L2_BUF_TYPE_VIDEO_CAPTURE: Final = 1
V4L2_MEMORY_MMAP: Final = 1
V4L2_FIELD_NONE: Final = 1

V4L2_CAP_VIDEO_CAPTURE: Final = 0x00000001
V4L2_CAP_META_CAPTURE: Final = 0x00800000
V4L2_CAP_STREAMING: Final = 0x04000000
V4L2_CAP_DEVICE_CAPS: Final = 0x80000000

#: The two capability bits the node must advertise before anything is
#: negotiated on it.
REQUIRED_CAPABILITIES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "V4L2_CAP_VIDEO_CAPTURE": V4L2_CAP_VIDEO_CAPTURE,
        "V4L2_CAP_STREAMING": V4L2_CAP_STREAMING,
    }
)

#: Capability bits named in the report when present. Not a complete list of
#: what V4L2 defines - the raw word is preserved too, so nothing is lost.
_CAPABILITY_NAMES: Final[tuple[tuple[int, str], ...]] = (
    (V4L2_CAP_VIDEO_CAPTURE, "V4L2_CAP_VIDEO_CAPTURE"),
    (0x00000002, "V4L2_CAP_VIDEO_OUTPUT"),
    (0x00000004, "V4L2_CAP_VIDEO_OVERLAY"),
    (0x00001000, "V4L2_CAP_VIDEO_CAPTURE_MPLANE"),
    (0x00002000, "V4L2_CAP_VIDEO_OUTPUT_MPLANE"),
    (0x00010000, "V4L2_CAP_TUNER"),
    (0x00020000, "V4L2_CAP_AUDIO"),
    (0x00100000, "V4L2_CAP_EXT_PIX_FORMAT"),
    (V4L2_CAP_META_CAPTURE, "V4L2_CAP_META_CAPTURE"),
    (0x01000000, "V4L2_CAP_READWRITE"),
    (V4L2_CAP_STREAMING, "V4L2_CAP_STREAMING"),
    (0x00400000, "V4L2_CAP_META_OUTPUT"),
    (0x20000000, "V4L2_CAP_IO_MC"),
    (0x40000000, "V4L2_CAP_TOUCH"),
    (V4L2_CAP_DEVICE_CAPS, "V4L2_CAP_DEVICE_CAPS"),
)

V4L2_BUF_FLAG_MAPPED: Final = 0x00000001
V4L2_BUF_FLAG_QUEUED: Final = 0x00000002
V4L2_BUF_FLAG_DONE: Final = 0x00000004
V4L2_BUF_FLAG_ERROR: Final = 0x00000040
V4L2_BUF_FLAG_LAST: Final = 0x00100000

#: ``flags`` carries two *masked fields* as well as plain bits, and they are not
#: the same field. ``TIMESTAMP_MASK`` selects how the buffer's timestamp is
#: measured (unknown / monotonic / copied); ``TSTAMP_SRC_MASK`` selects the
#: moment in the frame the timestamp refers to (end of frame / start of
#: exposure). Decoding either one as a set of independent bits invents names:
#: ``TSTAMP_SRC_EOF`` is zero, so it is only readable through its mask, and
#: ``0x00010000`` is ``TSTAMP_SRC_SOE`` rather than any standalone flag.
V4L2_BUF_FLAG_TIMESTAMP_MASK: Final = 0x0000E000
V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN: Final = 0x00000000
V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC: Final = 0x00002000
V4L2_BUF_FLAG_TIMESTAMP_COPY: Final = 0x00004000
V4L2_BUF_FLAG_TSTAMP_SRC_MASK: Final = 0x00070000
V4L2_BUF_FLAG_TSTAMP_SRC_EOF: Final = 0x00000000
V4L2_BUF_FLAG_TSTAMP_SRC_SOE: Final = 0x00010000

#: Plain single-bit flags only. Nothing inside ``TIMESTAMP_MASK`` or
#: ``TSTAMP_SRC_MASK`` belongs here: those are decoded as fields below, and a
#: bit from a masked field listed here would be reported under a name the
#: kernel never gave it.
_BUFFER_FLAG_NAMES: Final[tuple[tuple[int, str], ...]] = (
    (V4L2_BUF_FLAG_MAPPED, "V4L2_BUF_FLAG_MAPPED"),
    (V4L2_BUF_FLAG_QUEUED, "V4L2_BUF_FLAG_QUEUED"),
    (V4L2_BUF_FLAG_DONE, "V4L2_BUF_FLAG_DONE"),
    (0x00000008, "V4L2_BUF_FLAG_KEYFRAME"),
    (0x00000010, "V4L2_BUF_FLAG_PFRAME"),
    (0x00000020, "V4L2_BUF_FLAG_BFRAME"),
    (V4L2_BUF_FLAG_ERROR, "V4L2_BUF_FLAG_ERROR"),
    (0x00000080, "V4L2_BUF_FLAG_IN_REQUEST"),
    (0x00000100, "V4L2_BUF_FLAG_TIMECODE"),
    (0x00000200, "V4L2_BUF_FLAG_M2M_HOLD_CAPTURE_BUF"),
    (0x00000400, "V4L2_BUF_FLAG_PREPARED"),
    (0x00000800, "V4L2_BUF_FLAG_NO_CACHE_INVALIDATE"),
    (0x00001000, "V4L2_BUF_FLAG_NO_CACHE_CLEAN"),
    (V4L2_BUF_FLAG_LAST, "V4L2_BUF_FLAG_LAST"),
    (0x00800000, "V4L2_BUF_FLAG_REQUEST_FD"),
)

#: ``flags & V4L2_BUF_FLAG_TIMESTAMP_MASK`` - which clock the timestamp is on.
_TIMESTAMP_TYPES: Final[Mapping[int, str]] = MappingProxyType(
    {
        V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN: "V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN",
        V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC: "V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC",
        V4L2_BUF_FLAG_TIMESTAMP_COPY: "V4L2_BUF_FLAG_TIMESTAMP_COPY",
    }
)

#: ``flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK`` - which moment it names.
_TIMESTAMP_SOURCES: Final[Mapping[int, str]] = MappingProxyType(
    {
        V4L2_BUF_FLAG_TSTAMP_SRC_EOF: "V4L2_BUF_FLAG_TSTAMP_SRC_EOF",
        V4L2_BUF_FLAG_TSTAMP_SRC_SOE: "V4L2_BUF_FLAG_TSTAMP_SRC_SOE",
    }
)

_FIELD_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {
        0: "V4L2_FIELD_ANY",
        V4L2_FIELD_NONE: "V4L2_FIELD_NONE",
        2: "V4L2_FIELD_TOP",
        3: "V4L2_FIELD_BOTTOM",
        4: "V4L2_FIELD_INTERLACED",
        5: "V4L2_FIELD_SEQ_TB",
        6: "V4L2_FIELD_SEQ_BT",
        7: "V4L2_FIELD_ALTERNATE",
        8: "V4L2_FIELD_INTERLACED_TB",
        9: "V4L2_FIELD_INTERLACED_BT",
    }
)


def fourcc_to_int(fourcc: str) -> int:
    """Encode a four-character fourcc the way ``v4l2_fourcc`` does."""
    if len(fourcc) != 4:
        raise V4l2CaptureNotAllowed(f"a fourcc is exactly four characters; {fourcc!r} is not")
    raw = fourcc.encode("ascii")
    return raw[0] | (raw[1] << 8) | (raw[2] << 16) | (raw[3] << 24)


def int_to_fourcc(value: int) -> str:
    """Decode a packed fourcc, preserving a trailing space rather than trimming it."""
    raw = bytes((value >> shift) & 0xFF for shift in (0, 8, 16, 24))
    return raw.decode("ascii", errors="replace")


#: ``V4L2_PIX_FMT_Y16``. Built from :data:`FOURCC` once, so the constant and
#: the reported mode cannot drift apart.
V4L2_PIX_FMT_Y16: Final = fourcc_to_int(FOURCC)


def ensure_layout_intact() -> None:
    """Refuse to issue any ioctl on an ABI this transcription does not describe.

    The request numbers above encode the struct size, so a struct that is not
    the size this module believes it is produces a *different, valid-looking*
    ioctl number aimed at a real descriptor. Checking the sizes and the pointer
    width at run time turns that from silent corruption into a refusal before
    anything is opened. Called by both the dry and the live path.
    """
    actual = {
        "v4l2_capability": CAPABILITY_BYTES,
        "v4l2_pix_format": PIX_FORMAT_BYTES,
        "v4l2_format": FORMAT_BYTES,
        "v4l2_requestbuffers": REQUESTBUFFERS_BYTES,
        "v4l2_buffer": BUFFER_BYTES,
        "buf_type": BUFTYPE_BYTES,
    }
    for name, expected in EXPECTED_STRUCT_BYTES.items():
        if actual[name] != expected:
            raise V4l2CaptureNotAllowed(
                f"struct {name} packs to {actual[name]} bytes here but the transcribed "
                f"UAPI says {expected}; the ioctl numbers would be wrong"
            )
    if struct.calcsize("@P") != 8:
        raise V4l2CaptureNotAllowed(
            "this transcription assumes an LP64 kernel ABI (8-byte long inside "
            "struct timeval); the host is not one"
        )
    if PIX_FORMAT_BYTES > FORMAT_UNION_BYTES:
        raise V4l2CaptureNotAllowed(
            f"struct v4l2_pix_format is {PIX_FORMAT_BYTES} bytes and cannot fit in the "
            f"{FORMAT_UNION_BYTES}-byte v4l2_format union"
        )


# --- packing ------------------------------------------------------------------


def pack_capability() -> bytearray:
    """A zeroed ``struct v4l2_capability`` for ``VIDIOC_QUERYCAP`` to fill in."""
    return bytearray(CAPABILITY_BYTES)


def pack_format(
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    pixelformat: int = V4L2_PIX_FMT_Y16,
) -> bytearray:
    """A ``struct v4l2_format`` requesting the one mode.

    Only the four fields the kernel reads for a capture ``S_FMT`` are set;
    everything else, the tail of the 200-byte union included, is zero, which is
    how a V4L2 client asks the driver to choose. ``field`` is
    ``V4L2_FIELD_NONE`` because a depth frame is progressive.
    """
    pix = struct.pack(
        PIX_FORMAT_FORMAT,
        width,
        height,
        pixelformat,
        V4L2_FIELD_NONE,
        0,  # bytesperline: the driver reports it back
        0,  # sizeimage: likewise
        0,  # colorspace
        0,  # priv
        0,  # flags
        0,  # ycbcr_enc
        0,  # quantization
        0,  # xfer_func
    )
    return bytearray(
        struct.pack(
            FORMAT_FORMAT,
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            pix.ljust(FORMAT_UNION_BYTES, b"\x00"),
        )
    )


def pack_requestbuffers(
    count: int = REQUESTED_BUFFERS, *, limit: int = REQUESTED_BUFFERS
) -> bytearray:
    """A ``struct v4l2_requestbuffers`` for ``count`` MMAP capture buffers."""
    if not 0 <= count <= limit:
        raise V4l2CaptureNotAllowed(
            f"this module requests 0 to {limit} buffers; {count} is outside that"
        )
    return bytearray(
        struct.pack(
            REQUESTBUFFERS_FORMAT,
            count,
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            V4L2_MEMORY_MMAP,
            0,  # capabilities: filled in by the kernel
            0,  # flags
            b"\x00" * 3,
        )
    )


def pack_buffer(index: int, *, limit: int = REQUESTED_BUFFERS) -> bytearray:
    """A ``struct v4l2_buffer`` naming one MMAP capture buffer by index.

    Used for ``VIDIOC_QUERYBUF``, ``VIDIOC_QBUF`` and - with ``index`` zero,
    which the kernel overwrites - ``VIDIOC_DQBUF``. ``m`` is left zero: for
    MMAP the kernel fills ``m.offset`` in and never reads it.
    """
    if not 0 <= index < limit:
        raise V4l2CaptureNotAllowed(
            f"buffer index {index} is outside the fixed 0..{limit - 1} range"
        )
    return bytearray(
        struct.pack(
            BUFFER_FORMAT,
            index,
            V4L2_BUF_TYPE_VIDEO_CAPTURE,
            0,  # bytesused
            0,  # flags
            0,  # field
            0,  # timestamp.tv_sec
            0,  # timestamp.tv_usec
            0,  # timecode.type
            0,  # timecode.flags
            0,  # timecode.frames
            0,  # timecode.seconds
            0,  # timecode.minutes
            0,  # timecode.hours
            b"\x00" * 4,  # timecode.userbits
            0,  # sequence
            V4L2_MEMORY_MMAP,
            b"\x00" * 8,  # m union
            0,  # length
            0,  # reserved2
            0,  # request_fd
        )
    )


def pack_buf_type() -> bytearray:
    """The bare ``int`` ``VIDIOC_STREAMON``/``STREAMOFF`` take."""
    return bytearray(struct.pack(BUFTYPE_FORMAT, V4L2_BUF_TYPE_VIDEO_CAPTURE))


def sha256_of(payload: bytes) -> str:
    """The hash every artifact record quotes. One implementation, so two
    profiles cannot drift into hashing the same bytes two ways."""
    return hashlib.sha256(payload).hexdigest()


def _cstr(raw: bytes) -> str:
    """Decode a fixed-width NUL-padded kernel string field."""
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def _flag_names(value: int, table: tuple[tuple[int, str], ...]) -> tuple[str, ...]:
    return tuple(name for bit, name in table if value & bit)


# --- what the kernel said -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Capability:
    """The complete ``VIDIOC_QUERYCAP`` answer, preserved field for field."""

    driver: str
    card: str
    bus_info: str
    version: int
    version_text: str
    capabilities: int
    device_caps: int
    #: What actually applies to *this* node: ``device_caps`` when the driver
    #: sets ``V4L2_CAP_DEVICE_CAPS``, otherwise the whole-device word. Getting
    #: this wrong is how a metadata node passes a capture check.
    effective_capabilities: int
    capability_names: tuple[str, ...]
    device_capability_names: tuple[str, ...]

    @property
    def missing_required(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, bit in REQUIRED_CAPABILITIES.items()
            if not self.effective_capabilities & bit
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "driver": self.driver,
            "card": self.card,
            "bus_info": self.bus_info,
            "version": self.version,
            "version_text": self.version_text,
            "capabilities": f"0x{self.capabilities:08x}",
            "device_caps": f"0x{self.device_caps:08x}",
            "effective_capabilities": f"0x{self.effective_capabilities:08x}",
            "capability_names": list(self.capability_names),
            "device_capability_names": list(self.device_capability_names),
            "required_capabilities": list(REQUIRED_CAPABILITIES),
            "missing_required": list(self.missing_required),
        }


def parse_capability(raw: bytes) -> Capability:
    """Decode ``struct v4l2_capability``. Never raises on content."""
    driver, card, bus_info, version, capabilities, device_caps, _reserved = struct.unpack(
        CAPABILITY_FORMAT, raw
    )
    effective = device_caps if capabilities & V4L2_CAP_DEVICE_CAPS else capabilities
    return Capability(
        driver=_cstr(driver),
        card=_cstr(card),
        bus_info=_cstr(bus_info),
        version=version,
        version_text=f"{(version >> 16) & 0xFF}.{(version >> 8) & 0xFF}.{version & 0xFF}",
        capabilities=capabilities,
        device_caps=device_caps,
        effective_capabilities=effective,
        capability_names=_flag_names(capabilities, _CAPABILITY_NAMES),
        device_capability_names=_flag_names(device_caps, _CAPABILITY_NAMES),
    )


@dataclass(frozen=True, slots=True)
class PixelFormat:
    """A negotiated ``struct v4l2_pix_format``, preserved field for field."""

    width: int
    height: int
    pixelformat: int
    fourcc: str
    field: int
    field_name: str
    bytesperline: int
    sizeimage: int
    colorspace: int
    priv: int
    flags: int
    ycbcr_enc: int
    quantization: int
    xfer_func: int

    @property
    def padded(self) -> bool:
        """Whether the kernel reports row padding beyond a tight ``Y16 `` row."""
        return self.bytesperline > MIN_BYTES_PER_LINE

    @property
    def tight_bytes(self) -> int:
        """The unpadded size of one frame at this width and height."""
        return self.width * self.height * BYTES_PER_PIXEL

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "pixelformat": f"0x{self.pixelformat:08x}",
            "fourcc": self.fourcc,
            "field": self.field,
            "field_name": self.field_name,
            "bytesperline": self.bytesperline,
            "sizeimage": self.sizeimage,
            "colorspace": self.colorspace,
            "priv": self.priv,
            "flags": self.flags,
            "ycbcr_enc": self.ycbcr_enc,
            "quantization": self.quantization,
            "xfer_func": self.xfer_func,
            "row_padded": self.padded,
            "tight_frame_bytes": self.tight_bytes,
        }


def parse_format(raw: bytes) -> tuple[int, PixelFormat]:
    """Decode ``struct v4l2_format``, returning its buffer type and pixel format."""
    buf_type, union = struct.unpack(FORMAT_FORMAT, raw)
    (
        width,
        height,
        pixelformat,
        field_value,
        bytesperline,
        sizeimage,
        colorspace,
        priv,
        flags,
        ycbcr_enc,
        quantization,
        xfer_func,
    ) = struct.unpack(PIX_FORMAT_FORMAT, union[:PIX_FORMAT_BYTES])
    return buf_type, PixelFormat(
        width=width,
        height=height,
        pixelformat=pixelformat,
        fourcc=int_to_fourcc(pixelformat),
        field=field_value,
        field_name=_FIELD_NAMES.get(field_value, f"unknown ({field_value})"),
        bytesperline=bytesperline,
        sizeimage=sizeimage,
        colorspace=colorspace,
        priv=priv,
        flags=flags,
        ycbcr_enc=ycbcr_enc,
        quantization=quantization,
        xfer_func=xfer_func,
    )


@dataclass(frozen=True, slots=True)
class RequestedBuffers:
    """The ``VIDIOC_REQBUFS`` answer."""

    count: int
    buf_type: int
    memory: int
    capabilities: int

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "buf_type": self.buf_type,
            "memory": self.memory,
            "capabilities": f"0x{self.capabilities:08x}",
            "requested": REQUESTED_BUFFERS,
            "accepted_range": [MIN_ACCEPTED_BUFFERS, REQUESTED_BUFFERS],
        }


def parse_requestbuffers(raw: bytes) -> RequestedBuffers:
    count, buf_type, memory, capabilities, _flags, _reserved = struct.unpack(
        REQUESTBUFFERS_FORMAT, raw
    )
    return RequestedBuffers(
        count=count, buf_type=buf_type, memory=memory, capabilities=capabilities
    )


@dataclass(frozen=True, slots=True)
class BufferInfo:
    """One ``struct v4l2_buffer`` as the kernel returned it."""

    index: int
    buf_type: int
    bytesused: int
    flags: int
    flag_names: tuple[str, ...]
    field: int
    field_name: str
    timestamp_sec: int
    timestamp_usec: int
    timestamp_type: str
    timestamp_source: str
    timecode_type: int
    timecode_flags: int
    timecode_frames: int
    timecode_seconds: int
    timecode_minutes: int
    timecode_hours: int
    timecode_userbits: str
    sequence: int
    memory: int
    offset: int
    length: int

    @property
    def timestamp_ns(self) -> int:
        return self.timestamp_sec * 1_000_000_000 + self.timestamp_usec * 1_000

    @property
    def errored(self) -> bool:
        """Whether the kernel flagged this buffer's contents as unreliable."""
        return bool(self.flags & V4L2_BUF_FLAG_ERROR)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "buf_type": self.buf_type,
            "bytesused": self.bytesused,
            "flags": f"0x{self.flags:08x}",
            "flag_names": list(self.flag_names),
            "error_flag": self.errored,
            "field": self.field,
            "field_name": self.field_name,
            "timestamp_sec": self.timestamp_sec,
            "timestamp_usec": self.timestamp_usec,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_type": self.timestamp_type,
            "timestamp_source": self.timestamp_source,
            "timecode": {
                "type": self.timecode_type,
                "flags": self.timecode_flags,
                "frames": self.timecode_frames,
                "seconds": self.timecode_seconds,
                "minutes": self.timecode_minutes,
                "hours": self.timecode_hours,
                "userbits": self.timecode_userbits,
            },
            "sequence": self.sequence,
            "memory": self.memory,
            "offset": self.offset,
            "length": self.length,
        }


def parse_buffer(raw: bytes) -> BufferInfo:
    """Decode ``struct v4l2_buffer``. Never raises on content."""
    (
        index,
        buf_type,
        bytesused,
        flags,
        field_value,
        ts_sec,
        ts_usec,
        tc_type,
        tc_flags,
        tc_frames,
        tc_seconds,
        tc_minutes,
        tc_hours,
        tc_userbits,
        sequence,
        memory,
        m_union,
        length,
        _reserved2,
        _request_fd,
    ) = struct.unpack(BUFFER_FORMAT, raw)
    # Only the first __u32 of the union is meaningful for MMAP; decode it in
    # native byte order rather than masking a wider load.
    (offset,) = struct.unpack("=I", m_union[:4])
    return BufferInfo(
        index=index,
        buf_type=buf_type,
        bytesused=bytesused,
        flags=flags,
        flag_names=_flag_names(flags, _BUFFER_FLAG_NAMES),
        field=field_value,
        field_name=_FIELD_NAMES.get(field_value, f"unknown ({field_value})"),
        timestamp_sec=ts_sec,
        timestamp_usec=ts_usec,
        timestamp_type=_TIMESTAMP_TYPES.get(
            flags & V4L2_BUF_FLAG_TIMESTAMP_MASK,
            f"unknown (0x{flags & V4L2_BUF_FLAG_TIMESTAMP_MASK:08x})",
        ),
        timestamp_source=_TIMESTAMP_SOURCES.get(
            flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK,
            f"unknown (0x{flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK:08x})",
        ),
        timecode_type=tc_type,
        timecode_flags=tc_flags,
        timecode_frames=tc_frames,
        timecode_seconds=tc_seconds,
        timecode_minutes=tc_minutes,
        timecode_hours=tc_hours,
        timecode_userbits=bytes(tc_userbits).hex(),
        sequence=sequence,
        memory=memory,
        offset=offset,
        length=length,
    )


# --- outcome vocabulary -------------------------------------------------------


class Step(enum.StrEnum):
    """Every step a capture run can record, in the only order they can occur."""

    RESOLVE_NODE = "resolve_node"
    OPEN = "open"
    QUERYCAP = "querycap"
    GET_FMT = "get_fmt"
    SET_FMT = "set_fmt"
    REQBUFS = "reqbufs"
    QUERYBUF = "querybuf"
    MAP = "mmap"
    QBUF = "qbuf"
    STREAMON = "streamon"
    POLL = "poll"
    DQBUF = "dqbuf"
    STREAMOFF = "streamoff"
    UNMAP = "munmap"
    RELEASE_BUFFERS = "release_buffers"
    CLOSE = "close"


#: The live path, as a fixed sequence. Offline tests compare the steps a run
#: actually performed against this order.
LIFECYCLE: Final[tuple[Step, ...]] = (
    Step.RESOLVE_NODE,
    Step.OPEN,
    Step.QUERYCAP,
    Step.GET_FMT,
    Step.SET_FMT,
    Step.REQBUFS,
    Step.QUERYBUF,
    Step.MAP,
    Step.QBUF,
    Step.STREAMON,
    Step.POLL,
    Step.DQBUF,
)

#: The four obligations, in the order cleanup discharges them.
CLEANUP_STEPS: Final[tuple[Step, ...]] = (
    Step.STREAMOFF,
    Step.UNMAP,
    Step.RELEASE_BUFFERS,
    Step.CLOSE,
)


class CaptureStatus(enum.StrEnum):
    """The primary result of the attempt, cleanup reported separately."""

    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    REFUSED = "refused"
    DEVICE_ABSENT = "device_absent"
    NODE_UNRESOLVED = "node_unresolved"
    DEVICE_REPLACED = "device_replaced"
    PERMISSION_DENIED = "permission_denied"
    OPEN_ERROR = "open_error"
    CAPABILITY_MISMATCH = "capability_mismatch"
    FORMAT_MISMATCH = "format_mismatch"
    BUFFER_COUNT_MISMATCH = "buffer_count_mismatch"
    MAPPING_ERROR = "mapping_error"
    QUEUE_ERROR = "queue_error"
    STREAMON_ERROR = "streamon_error"
    POLL_TIMEOUT = "poll_timeout"
    POLL_ERROR = "poll_error"
    DEQUEUE_ERROR = "dequeue_error"
    SHORT_FRAME = "short_frame"
    OVERSIZE_FRAME = "oversize_frame"
    EMPTY_FRAME = "empty_frame"
    MALFORMED_FRAME = "malformed_frame"
    DIMENSION_MISMATCH = "dimension_mismatch"
    INTERRUPTED = "interrupted"
    IOCTL_ERROR = "ioctl_error"


#: Exit code per primary status. The codes shared with the usbfs commands keep
#: the meanings those already gave them, so a caller can branch on 2, 3, 4, 8,
#: 9 and 13 without knowing which command produced them.
CAPTURE_EXIT_CODES: Final[Mapping[CaptureStatus, int]] = MappingProxyType(
    {
        CaptureStatus.COMPLETE: 0,
        CaptureStatus.NOT_ATTEMPTED: 0,
        CaptureStatus.DEVICE_ABSENT: 2,
        CaptureStatus.NODE_UNRESOLVED: 2,
        CaptureStatus.PERMISSION_DENIED: 3,
        CaptureStatus.POLL_TIMEOUT: 4,
        CaptureStatus.IOCTL_ERROR: 5,
        CaptureStatus.DEVICE_REPLACED: 8,
        CaptureStatus.REFUSED: 10,
        CaptureStatus.INTERRUPTED: 13,
        CaptureStatus.OPEN_ERROR: 15,
        CaptureStatus.CAPABILITY_MISMATCH: 16,
        CaptureStatus.FORMAT_MISMATCH: 17,
        CaptureStatus.BUFFER_COUNT_MISMATCH: 18,
        CaptureStatus.MAPPING_ERROR: 19,
        CaptureStatus.QUEUE_ERROR: 20,
        CaptureStatus.STREAMON_ERROR: 21,
        CaptureStatus.POLL_ERROR: 22,
        CaptureStatus.DEQUEUE_ERROR: 23,
        CaptureStatus.SHORT_FRAME: 24,
        CaptureStatus.OVERSIZE_FRAME: 25,
        CaptureStatus.EMPTY_FRAME: 27,
        CaptureStatus.MALFORMED_FRAME: 28,
        CaptureStatus.DIMENSION_MISMATCH: 29,
    }
)

#: Overrides every primary status when a frame arrived but its evidence could
#: not be written. A frame nobody can inspect is not a captured frame, and this
#: run gets exactly one chance to preserve it.
ARTIFACT_FAILED_EXIT: Final = 26

_ERRNO_STATUS: Final[Mapping[int, CaptureStatus]] = MappingProxyType(
    {
        errno_mod.EACCES: CaptureStatus.PERMISSION_DENIED,
        errno_mod.EPERM: CaptureStatus.PERMISSION_DENIED,
        errno_mod.ENODEV: CaptureStatus.DEVICE_ABSENT,
        errno_mod.ENOENT: CaptureStatus.DEVICE_ABSENT,
        errno_mod.ENXIO: CaptureStatus.DEVICE_ABSENT,
        errno_mod.ESHUTDOWN: CaptureStatus.DEVICE_ABSENT,
    }
)


def _errno_of(exc: OSError) -> tuple[int | None, str | None]:
    if exc.errno is None:
        return None, None
    return exc.errno, errno_mod.errorcode.get(exc.errno, f"errno {exc.errno}")


@dataclass(frozen=True, slots=True)
class StepResult:
    """One attempted step of the capture and how it ended."""

    step: Step
    ok: bool
    detail: str
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": str(self.step),
            "ok": self.ok,
            "detail": self.detail,
            "duration_s": self.duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


def recovery_guidance(node_path: str) -> str:
    """What the owner must check by hand when cleanup did not finish."""
    return (
        f"{node_path} may still hold a streaming queue or mapped buffers from this "
        "process. Confirm with `fuser -v " + node_path + "` that no process holds it; "
        "if one does, terminate it. This command never unbinds "
        + EXPECTED_INTERFACE_DRIVER
        + ", so the driver binding itself is unchanged, and closing the descriptor - which "
        "process exit forces - releases the queue and the mappings in the kernel."
    )


@dataclass(frozen=True, slots=True)
class CleanupRecord:
    """Whether every obligation the run created was discharged, reported apart
    from whether a frame arrived."""

    attempted: bool
    steps: tuple[StepResult, ...]
    ok: bool
    note: str
    guidance: str | None = None

    @classmethod
    def not_needed(cls, note: str) -> CleanupRecord:
        """Nothing was opened or allocated, so there is nothing to release."""
        return cls(attempted=False, steps=(), ok=True, note=note)

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "note": self.note,
            "steps": [s.as_dict() for s in self.steps],
            "recovery_guidance": self.guidance,
        }


# --- artifacts ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameArtifact:
    """One evidence file, or the reason there is not one."""

    kind: str
    path: str | None
    byte_count: int
    sha256: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Whether the frame that arrived was preserved where it can be inspected."""

    attempted: bool
    ok: bool
    note: str
    directory: str | None
    files: tuple[FrameArtifact, ...]
    #: The artifact ceiling the profile that produced this record enforces.
    #: Reported rather than looked up, so a report cannot quote the other
    #: profile's ceiling at a reader.
    ceiling_bytes: int = MAX_RAW_ARTIFACT_BYTES

    @classmethod
    def not_needed(
        cls,
        note: str,
        directory: str | None,
        ceiling_bytes: int = MAX_RAW_ARTIFACT_BYTES,
    ) -> ArtifactRecord:
        return cls(
            attempted=False,
            ok=True,
            note=note,
            directory=directory,
            files=(),
            ceiling_bytes=ceiling_bytes,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "note": self.note,
            "directory": self.directory,
            "files": [f.as_dict() for f in self.files],
            "raw_ceiling_bytes": self.ceiling_bytes,
        }


def _write_new_file(path: Path, payload: bytes, *, ceiling: int = MAX_RAW_ARTIFACT_BYTES) -> None:
    """Create ``path`` exclusively and write all of ``payload``, or fail.

    ``O_EXCL | O_NOFOLLOW`` on purpose: the default output directory lives in a
    world-writable ``/tmp``, and this run gets one frame. Overwriting an
    existing file - or following a symlink out of the directory - would destroy
    evidence rather than preserve it, so an existing name is an error, not a
    reason to clobber.

    ``os.write`` may write fewer bytes than it was given, so one call is not a
    file. This retries from the offset the kernel actually reached, and it is
    bounded three ways: the payload is at most
    ``ceiling``, every iteration must make strictly positive
    progress, and a write of zero bytes ends the attempt instead of spinning.
    A payload that could not be written completely leaves no file behind: a
    truncated artifact under a name that promises a whole frame is worse
    evidence than no artifact at all, and the caller records the failure.
    """
    if len(payload) > ceiling:
        raise V4l2ArtifactIncomplete(
            errno_mod.EFBIG,
            f"refusing to write {len(payload)} bytes to {path}, past the "
            f"{ceiling} B artifact ceiling",
        )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    written = 0
    try:
        while written < len(payload):
            just_written = os.write(fd, payload[written:])
            if just_written <= 0:
                raise V4l2ArtifactIncomplete(
                    errno_mod.EIO,
                    f"write to {path} made no progress after {written} of {len(payload)} bytes",
                )
            written += just_written
    except BaseException:
        _discard_partial(path)
        raise
    finally:
        os.close(fd)


def _discard_partial(path: Path) -> None:
    """Remove a file that was created but not completely written.

    Best effort by design: the write already failed, and a failure to unlink
    must not replace the error that explains why there is no artifact. It is
    reported through the caller's record instead.
    """
    with contextlib.suppress(OSError):
        path.unlink()


def unpack_rows(raw: bytes, pixel_format: PixelFormat) -> bytes:
    """Strip row padding from ``raw``, or raise if any row bound fails.

    Called only when the kernel reported ``bytesperline`` larger than a tight
    ``Y16 `` row. Every bound the assignment names is checked before a single
    byte is copied, and the result is the exact tight size or nothing at all.
    """
    if pixel_format.bytesperline < MIN_BYTES_PER_LINE:
        raise V4l2CaptureNotAllowed(
            f"bytesperline {pixel_format.bytesperline} is below the {MIN_BYTES_PER_LINE} "
            "bytes one row of this mode needs"
        )
    needed = pixel_format.bytesperline * pixel_format.height
    if len(raw) < needed:
        raise V4l2CaptureNotAllowed(
            f"{len(raw)} bytes cannot hold {pixel_format.height} rows of "
            f"{pixel_format.bytesperline} bytes ({needed} needed)"
        )
    rows: list[bytes] = []
    for row in range(pixel_format.height):
        start = row * pixel_format.bytesperline
        end = start + MIN_BYTES_PER_LINE
        if end > len(raw):  # pragma: no cover - implied by the check above
            raise V4l2CaptureNotAllowed(f"row {row} ends at {end}, past the {len(raw)}-byte frame")
        rows.append(raw[start:end])
    packed = b"".join(rows)
    if len(packed) != pixel_format.tight_bytes:
        raise V4l2CaptureNotAllowed(
            f"unpacking produced {len(packed)} bytes, not the expected {pixel_format.tight_bytes}"
        )
    return packed


def preserve_frame(raw: bytes, pixel_format: PixelFormat, directory: Path) -> ArtifactRecord:
    """Write the raw frame, and a tightly packed copy when rows were padded.

    Never raises: an artifact that could not be written is reported as a failed
    artifact, because the caller must still report the frame that arrived and
    must still release the device.
    """
    files: list[FrameArtifact] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ArtifactRecord(
            attempted=True,
            ok=False,
            note=f"cannot create the output directory: {exc}",
            directory=str(directory),
            files=(),
        )

    raw_path = directory / RAW_ARTIFACT_NAME
    raw_digest = sha256_of(raw)
    try:
        _write_new_file(raw_path, raw)
    except OSError as exc:
        files.append(
            FrameArtifact(
                kind="raw",
                path=None,
                byte_count=len(raw),
                sha256=raw_digest,
                ok=False,
                detail=f"cannot write {raw_path}: {exc}",
            )
        )
    else:
        files.append(
            FrameArtifact(
                kind="raw",
                path=str(raw_path),
                byte_count=len(raw),
                sha256=raw_digest,
                ok=True,
                detail=f"the dequeued bytes exactly as the kernel presented them ({len(raw)} B)",
            )
        )

    files.append(_packed_artifact(raw, pixel_format, directory))
    ok = all(f.ok for f in files)
    return ArtifactRecord(
        attempted=True,
        ok=ok,
        note=(
            "the frame and its hashes were preserved"
            if ok
            else "the frame arrived but its evidence was NOT fully preserved"
        ),
        directory=str(directory),
        files=tuple(files),
    )


def _packed_artifact(raw: bytes, pixel_format: PixelFormat, directory: Path) -> FrameArtifact:
    """The tightly packed copy, or the record of why there is no separate one."""
    if not pixel_format.padded:
        # The kernel reported no row padding, so the raw bytes already are the
        # tight frame. Writing a byte-identical second file would only invite
        # the two hashes being read as independent confirmation.
        digest = sha256_of(raw)
        return FrameArtifact(
            kind="packed",
            path=None,
            byte_count=len(raw),
            sha256=digest,
            ok=True,
            detail=(
                f"bytesperline is {pixel_format.bytesperline}, the tight row size, so the raw "
                "frame is already tightly packed; no separate file was written and this hash "
                "is the raw one"
            ),
        )
    try:
        packed = unpack_rows(raw, pixel_format)
    except V4l2CaptureNotAllowed as exc:
        return FrameArtifact(
            kind="packed",
            path=None,
            byte_count=0,
            sha256="",
            ok=False,
            detail=f"row padding was reported but could not be validated: {exc}",
        )
    path = directory / PACKED_ARTIFACT_NAME
    digest = sha256_of(packed)
    try:
        _write_new_file(path, packed)
    except OSError as exc:
        return FrameArtifact(
            kind="packed",
            path=None,
            byte_count=len(packed),
            sha256=digest,
            ok=False,
            detail=f"cannot write {path}: {exc}",
        )
    return FrameArtifact(
        kind="packed",
        path=str(path),
        byte_count=len(packed),
        sha256=digest,
        ok=True,
        detail=(
            f"{pixel_format.height} rows of {MIN_BYTES_PER_LINE} B lifted out of "
            f"{pixel_format.bytesperline}-B padded rows; no scale was applied"
        ),
    )


# --- injected primitives ------------------------------------------------------


class Mapping_(Protocol):
    """The slice of ``mmap.mmap`` this module uses, so tests can stand in for it."""

    def close(self) -> None: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: slice, /) -> bytes: ...


MmapFn = Callable[[int, int, int], Mapping_]
#: ``(fd, timeout_s) -> revents``. Zero means the timeout expired.
Poller = Callable[[int, float], int]


def _default_opener(path: str) -> int:
    # O_RDWR because every VIDIOC_S_* and the streaming ioctls need a writable
    # handle; O_NONBLOCK so a dequeue can never block past the deadline the
    # poll above it already enforced; O_CLOEXEC so nothing this process spawns
    # inherits a streaming descriptor. Opening a V4L2 node starts no stream.
    return os.open(path, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)


def _default_mmap(fd: int, length: int, offset: int) -> Mapping_:
    # PROT_READ | PROT_WRITE follows the kernel's own documented mmap example
    # for capture buffers, which several drivers require even though nothing
    # here ever writes through the mapping: every read below is a slice.
    return mmap_mod.mmap(
        fd,
        length,
        flags=mmap_mod.MAP_SHARED,
        prot=mmap_mod.PROT_READ | mmap_mod.PROT_WRITE,
        offset=offset,
    )


def _default_poll(fd: int, timeout_s: float) -> int:
    """Wait once for the descriptor to become readable, returning ``revents``.

    ``POLLIN`` is the only event registered; ``POLLERR``/``POLLHUP``/
    ``POLLNVAL`` are reported by the kernel whether or not they are asked for,
    and the caller treats them as a poll error rather than as readiness.
    """
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    ready = poller.poll(max(0.0, timeout_s) * 1000.0)
    return ready[0][1] if ready else 0


def _poll_event_names(revents: int) -> tuple[str, ...]:
    table = (
        (select.POLLIN, "POLLIN"),
        (select.POLLPRI, "POLLPRI"),
        (select.POLLOUT, "POLLOUT"),
        (select.POLLERR, "POLLERR"),
        (select.POLLHUP, "POLLHUP"),
        (select.POLLNVAL, "POLLNVAL"),
    )
    return tuple(name for bit, name in table if revents & bit)


#: How a signal caught inside the capture window describes itself. Not the
#: interface-0 wording: this command never unbinds a driver, so it must not
#: report that one may be unbound.
CAPTURE_WINDOW: Final = (
    "a V4L2 stream, buffer queue or mapping may have been created and not yet released"
)

#: The termination guard the backend installs, carrying that wording. A module
#: constant rather than a lambda so the default is one shared object that a
#: test can compare against.
_capture_signal_guard: Final[SignalGuard] = partial(cleanup_on_termination, CAPTURE_WINDOW)


# --- the capture profile ------------------------------------------------------

#: What a profile adds to the shared format refusals. Returning a string
#: refuses the negotiated format; ``None`` accepts it. The shared checks - the
#: buffer type, the fourcc, the frame size and the per-buffer ceiling - are the
#: same for every profile and are not repeated here.
FormatCheck = Callable[["CaptureProfile", "PixelFormat"], "str | None"]

# A read-only identity boundary shared by single-frame and paired sessions.
# Keeping it injectable makes the post-open check testable without touching a
# real descriptor or sysfs tree.
IdentityCheck = Callable[[int, "NodeSnapshot"], str | None]


def _capture_identity_check(fd: int, target: NodeSnapshot) -> str | None:
    """Keep the single-frame refusal wording tied to its capture boundary."""
    return identity_drift(fd, target, moment="the capture")


#: How a profile judges the bytes that were actually dequeued, once the shared
#: bounds (mapped length, artifact ceiling, kernel error flag) have all held.
FrameJudge = Callable[
    ["CaptureProfile", bytes, "BufferInfo", "PixelFormat | None"],
    "FrameVerdict",
]

#: How a profile turns those bytes into local evidence. Never raises: the
#: engine must still report the frame and must still release the device.
FramePreserver = Callable[["CaptureProfile", bytes, "PixelFormat", Path], "ArtifactRecord"]


@dataclass(frozen=True, slots=True)
class CaptureProfile:
    """One fixed capture: a role, a mode, a set of ceilings and three rules.

    A profile is the *whole* of what distinguishes one capture command from
    another in this package. Everything else - the node resolution, the open,
    the nine ioctls, the single poll, the single dequeue, the shielded cleanup -
    is one implementation shared by all of them, which is the point: a second
    capture path would be a second lifecycle to get wrong.

    Instances are frozen, are created at import time, and are never built from
    caller input. A caller chooses a serial and an output directory; it cannot
    reach a profile, widen one, or mix two.
    """

    #: The command this profile backs, e.g. ``depth-frame``.
    name: str
    #: The ``mode`` key every JSON report carries, so evidence is self-naming.
    mode_label: str
    #: The logical camera, resolved from the USB interface string descriptor.
    #: Never a ``videoN`` number and never a caller-supplied path.
    role: str
    #: The V4L2 node kind within that role. ``uvcvideo`` registers the capture
    #: node at index 0 and the metadata node at index 1; only index 0 is used.
    node_kind: str
    #: The one fourcc. Four characters exactly, trailing space included.
    fourcc: str
    width: int
    height: int
    #: The descriptor's own default interval for this mode, restated as
    #: evidence of what was configured. No profile issues ``VIDIOC_S_PARM``, so
    #: this is never a measurement of the interval that was negotiated.
    descriptor_frame_rate_hz: float
    #: Whether the payload is a compressed bitstream rather than a raster. A
    #: compressed mode has no meaningful ``bytesperline`` and no fixed frame
    #: size, so the size-based refusals do not apply to it.
    compressed: bool
    requested_buffers: int
    min_accepted_buffers: int
    max_buffer_bytes: int
    max_total_mapped_bytes: int
    capture_deadline_s: float
    max_raw_artifact_bytes: int
    #: Where the live path writes evidence when the caller names no directory.
    default_output_dir: Path
    #: The unmistakable opt-in flag, quoted in the dry-run text.
    execute_flag: str
    #: A human phrase for the mode, e.g. ``'Y16 ' 640x400``.
    mode_text: str
    #: What this profile means by a *complete* frame, in one phrase. Part of
    #: the profile rather than of any renderer: it is the definition the frame
    #: judgement below enforces, not a caption for it.
    completeness_text: str
    #: The one thing a reader of this profile's evidence must not conclude from
    #: it. Printed at the end of every human report and never softened.
    disclaimer: str
    format_check: FormatCheck
    judge: FrameJudge
    preserve: FramePreserver
    #: Profile-specific plan keys, folded into every report's ``plan``.
    plan_extra: Mapping[str, object]

    @property
    def pixelformat(self) -> int:
        """The packed fourcc the kernel compares ``S_FMT`` against."""
        return fourcc_to_int(self.fourcc)

    @property
    def size_text(self) -> str:
        """``WIDTHxHEIGHT``, the way every report and refusal spells it."""
        return f"{self.width}x{self.height}"

    @property
    def dry_run_detail(self) -> str:
        """What the dry path says instead of opening the node."""
        return (
            "dry run: no video node was opened, no format was negotiated and no buffer "
            f"was allocated (pass {self.execute_flag} to attempt the capture)"
        )

    def plan_as_dict(self) -> dict[str, object]:
        """The fixed plan, restated in every report so evidence is self-describing."""
        plan: dict[str, object] = {
            "profile": self.name,
            "role": self.role,
            "node_kind": self.node_kind,
            "expected_interface_driver": EXPECTED_INTERFACE_DRIVER,
            "driver_detached": False,
            "fourcc": self.fourcc,
            "width": self.width,
            "height": self.height,
            "descriptor_frame_rate_hz": self.descriptor_frame_rate_hz,
            "frame_rate_measured": False,
            "compressed": self.compressed,
        }
        plan.update(self.plan_extra)
        plan.update(
            {
                "requested_buffers": self.requested_buffers,
                "min_accepted_buffers": self.min_accepted_buffers,
                "max_buffer_bytes": self.max_buffer_bytes,
                "max_total_mapped_bytes": self.max_total_mapped_bytes,
                "capture_deadline_s": self.capture_deadline_s,
                "max_raw_artifact_bytes": self.max_raw_artifact_bytes,
                "max_dequeues": 1,
                "retries": 0,
                "ioctls": {name: f"0x{value:08x}" for name, value in ALLOWED_IOCTLS.items()},
            }
        )
        return plan


# --- the depth profile --------------------------------------------------------


def _depth_format_check(profile: CaptureProfile, negotiated: PixelFormat) -> str | None:
    """The raster-specific half of the ``Y16 `` format refusals.

    A depth frame is a raster, so its row stride and its frame size are both
    load-bearing: a stride that cannot hold a row, or a ``sizeimage`` smaller
    than the geometry the driver itself just described, means the answer is not
    self-consistent and nothing should be streamed against it.
    """
    if negotiated.bytesperline < MIN_BYTES_PER_LINE:
        return (
            f"bytesperline {negotiated.bytesperline} cannot hold one {profile.width}-pixel "
            f"{profile.fourcc!r} row ({MIN_BYTES_PER_LINE} B)"
        )
    needed = negotiated.bytesperline * negotiated.height
    if negotiated.sizeimage < needed:
        return (
            f"sizeimage {negotiated.sizeimage} is smaller than the {needed} B its own "
            f"bytesperline and height imply"
        )
    if negotiated.padded:
        # Not a refusal: padding is legal and is preserved. Saying so here keeps
        # the "safe and consistent" test in one place.
        return None
    if needed != EXPECTED_FRAME_BYTES:
        return (
            f"an unpadded frame of this mode is {EXPECTED_FRAME_BYTES} B, but the driver "
            f"describes {needed} B"
        )
    return None


def _depth_frame_judgement(
    profile: CaptureProfile,
    raw: bytes,
    info: BufferInfo,
    negotiated: PixelFormat | None,
) -> FrameVerdict:
    """Whether the dequeued ``Y16 `` bytes are a whole frame of this mode."""
    expected = (
        negotiated.bytesperline * negotiated.height
        if negotiated is not None
        else EXPECTED_FRAME_BYTES
    )
    if len(raw) < expected:
        return FrameVerdict(
            CaptureStatus.SHORT_FRAME,
            f"buffer {info.index} carried {len(raw)} bytes, short of the {expected} a "
            f"complete {profile.fourcc!r} {profile.size_text} frame needs; the "
            "bytes are preserved as evidence but are not a frame",
        )
    # The floor above is not enough on its own. When the negotiated rows are
    # tight, ``expected`` *is* the whole frame, so a kernel reporting more
    # bytes than that is describing something this mode cannot produce - and
    # calling it complete would ship a larger artifact under a name that
    # promises exactly that many bytes. A padded format is different: trailing
    # bytes past the last row are the padding the mode legitimately carries,
    # they are already bounded by the mapping and the artifact ceiling, and the
    # packing path validates every row before it produces a tight copy.
    tight = negotiated is not None and not negotiated.padded
    if tight and len(raw) > expected:
        return FrameVerdict(
            CaptureStatus.OVERSIZE_FRAME,
            f"buffer {info.index} carried {len(raw)} bytes, past the {expected} an "
            f"unpadded {profile.mode_text} frame is; the negotiated rows are tight, so "
            "the excess is unexplained and the bytes are preserved as evidence rather "
            "than called a frame",
        )
    return FrameVerdict(
        CaptureStatus.COMPLETE,
        f"one complete {profile.mode_text} frame of {len(raw)} bytes",
    )


def _depth_preserve(
    profile: CaptureProfile, raw: bytes, pixel_format: PixelFormat, directory: Path
) -> ArtifactRecord:
    """The raw frame, plus a tightly packed copy when the rows were padded."""
    return preserve_frame(raw, pixel_format, directory)


#: The one depth capture. Every value here is the constant this module already
#: documented above; the profile is where they are bound together, not a second
#: place they are decided.
DEPTH_PROFILE: Final = CaptureProfile(
    name="depth-frame",
    mode_label="v4l2-mmap-depth-frame",
    role=CAPTURE_ROLE,
    node_kind=CAPTURE_KIND,
    fourcc=FOURCC,
    width=FRAME_WIDTH,
    height=FRAME_HEIGHT,
    descriptor_frame_rate_hz=DESCRIPTOR_FRAME_RATE_HZ,
    compressed=False,
    requested_buffers=REQUESTED_BUFFERS,
    min_accepted_buffers=MIN_ACCEPTED_BUFFERS,
    max_buffer_bytes=MAX_BUFFER_BYTES,
    max_total_mapped_bytes=MAX_TOTAL_MAPPED_BYTES,
    capture_deadline_s=CAPTURE_DEADLINE_S,
    max_raw_artifact_bytes=MAX_RAW_ARTIFACT_BYTES,
    default_output_dir=DEFAULT_OUTPUT_DIR,
    execute_flag="--execute-depth-frame-capture",
    mode_text=f"{FOURCC!r} {FRAME_WIDTH}x{FRAME_HEIGHT}",
    completeness_text=f"exactly {EXPECTED_FRAME_BYTES} bytes, unpadded",
    disclaimer=(
        "no metric depth was claimed: this is raw 16-bit sensor data with no calibration "
        "or scale applied"
    ),
    format_check=_depth_format_check,
    judge=_depth_frame_judgement,
    preserve=_depth_preserve,
    plan_extra=MappingProxyType(
        {
            "bytes_per_pixel": BYTES_PER_PIXEL,
            "expected_frame_bytes": EXPECTED_FRAME_BYTES,
            "min_bytes_per_line": MIN_BYTES_PER_LINE,
        }
    ),
)


# --- node resolution ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeSnapshot:
    """What the depth capture node was, immediately before it was opened.

    Every field is read from sysfs and ``stat``; nothing here opens the node.
    The role, the interface number and the V4L2 index are what make the target
    stable - the ``videoN`` name is recorded as evidence of what that resolved
    to on this boot, never as the way it was chosen.
    """

    serial: str
    sysfs_path: str
    usb_path: str
    bus_number: int
    device_number: int
    role: str
    interface_number: int
    interface_string: str | None
    interface_driver: str | None
    v4l_index: int
    v4l_name: str | None
    node_name: str
    node_path: str
    rdev: int
    rdev_major: int
    rdev_minor: int
    inode: int
    mode: str

    def as_dict(self) -> dict[str, object]:
        return {
            "serial": self.serial,
            "sysfs_path": self.sysfs_path,
            "usb_path": self.usb_path,
            "bus_number": self.bus_number,
            "device_number": self.device_number,
            "role": self.role,
            "interface_number": self.interface_number,
            "interface_string": self.interface_string,
            "interface_driver": self.interface_driver,
            "v4l_index": self.v4l_index,
            "v4l_kind": CAPTURE_KIND,
            "v4l_name": self.v4l_name,
            "node_name": self.node_name,
            "node_path": self.node_path,
            "rdev": self.rdev,
            "rdev_major_minor": f"{self.rdev_major}:{self.rdev_minor}",
            "inode": self.inode,
            "mode": self.mode,
            "resolved_by": (
                f"USB interface string descriptor prefix for role {self.role!r} "
                f"plus V4L2 index {self.v4l_index} ({CAPTURE_KIND}); the node number "
                "was never assumed"
            ),
        }


def _sole_capture_node(function: CameraFunction, kind: str = CAPTURE_KIND) -> VideoNode:
    """The one index-0 node of a camera function, or a refusal.

    ``uvcvideo`` registers exactly one capture node and, for this unit, one
    metadata node per camera. More than one capture node would mean the tree is
    not the one this module was written against, and picking would be a guess.
    """
    nodes = function.capture_nodes
    if not nodes:
        raise DeviceAbsentError(
            f"the {function.role} function on interface {function.interface_number} "
            f"exposes no V4L2 {kind} node"
        )
    if len(nodes) > 1:
        names = ", ".join(node.name for node in nodes)
        raise DeviceAbsentError(
            f"the {function.role} function exposes {len(nodes)} {kind} nodes "
            f"({names}); refusing to choose between them"
        )
    return nodes[0]


def resolve_capture_node(
    serial: str,
    *,
    profile: CaptureProfile = DEPTH_PROFILE,
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
) -> tuple[NodeSnapshot, ScannerDevice]:
    """Resolve the profile's capture node from the device's stable identity.

    The role comes from the profile and the node from the USB interface string
    descriptor plus the V4L2 index, so a profile can only ever reach its own
    camera: nothing here reads, or could be handed, a ``/dev/videoN`` path.

    Raises :class:`~scanner3d.errors.DeviceAbsentError` when there is not
    exactly one matching device, when it exposes no function of that role, when
    that function has no single capture node, or when the node cannot be stat'ed.
    """
    devices = tuple(
        d for d in find_scanners(sysfs_root=sysfs_root, dev_root=dev_root) if d.serial == serial
    )
    if not devices:
        raise DeviceAbsentError(f"no 2207:110c device with serial {serial!r} is attached")
    if len(devices) > 1:
        raise DeviceAbsentError(
            f"{len(devices)} devices report serial {serial!r}; refusing to choose between them"
        )
    device = devices[0]
    if device.bus_number is None or device.device_number is None:
        raise DeviceAbsentError(f"{device.usb_path} has no busnum/devnum; cannot snapshot it")

    function = device.function(profile.role)
    if function is None:
        raise DeviceAbsentError(
            f"{device.usb_path} exposes no {profile.role!r} camera function; the interface "
            "string descriptors do not name one"
        )
    node = _sole_capture_node(function, profile.node_kind)
    if node.index is None:  # pragma: no cover - capture_nodes filters on index 0
        raise DeviceAbsentError(f"{node.name} reports no V4L2 index")

    path = Path(node.device_path)
    try:
        st = path.stat()
    except OSError as exc:
        raise DeviceAbsentError(f"cannot stat {path}: {exc}") from exc

    snapshot = NodeSnapshot(
        serial=serial,
        sysfs_path=device.sysfs_path,
        usb_path=device.usb_path,
        bus_number=device.bus_number,
        device_number=device.device_number,
        role=function.role,
        interface_number=function.interface_number,
        interface_string=function.interface_string,
        interface_driver=function.driver,
        v4l_index=node.index,
        v4l_name=node.v4l_name,
        node_name=node.name,
        node_path=str(path),
        rdev=st.st_rdev,
        rdev_major=os.major(st.st_rdev),
        rdev_minor=os.minor(st.st_rdev),
        inode=st.st_ino,
        mode=oct(st.st_mode & 0o7777),
    )
    return snapshot, device


def refusal_reason(
    device: ScannerDevice, serial: str, profile: CaptureProfile = DEPTH_PROFILE
) -> str | None:
    """Why this device must not be captured from, or ``None`` if it may be.

    Read entirely from sysfs, before the node is opened. Unlike the usbfs
    commands this repository already has, the requirement here is that
    ``uvcvideo`` *is* bound and stays bound: this module has no way to talk to
    the hardware without it.
    """
    if not device.identity_matches:
        return (
            f"{device.usb_path} is {device.vendor_id:04x}:{device.product_id:04x}, "
            "not the expected 2207:110c MINI"
        )
    if device.serial != serial:
        return f"{device.usb_path} reports serial {device.serial!r}, not the requested {serial!r}"
    if device.manufacturer != EXPECTED_MANUFACTURER:
        return (
            f"{device.usb_path} reports manufacturer {device.manufacturer!r}, "
            f"not the expected {EXPECTED_MANUFACTURER!r}"
        )
    function = device.function(profile.role)
    if function is None:
        return f"{device.usb_path} exposes no {profile.role!r} camera function"
    if function.driver != EXPECTED_INTERFACE_DRIVER:
        return (
            f"the {profile.role} interface is bound to {function.driver!r}, not "
            f"{EXPECTED_INTERFACE_DRIVER!r}; this command captures through the kernel "
            "driver and cannot run without it"
        )
    if not any(
        fmt.fourcc.strip() == profile.fourcc.strip()
        and any(m.width == profile.width and m.height == profile.height for m in fmt.modes)
        for fmt in function.formats
    ):
        return (
            f"the {profile.role} function does not advertise {profile.fourcc!r} at "
            f"{profile.size_text} in its own descriptors"
        )
    return None


def preflight_capture_node(
    serial: str,
    *,
    profile: CaptureProfile = DEPTH_PROFILE,
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
) -> tuple[NodeSnapshot, ScannerDevice]:
    """Resolve one fixed profile and prove it is safe before any open.

    This is the shared, injectable preflight used by both the single-frame and
    paired paths.  ABI layout, exact USB identity, bound ``uvcvideo`` and the
    profile's advertised fixed mode are all checked before a descriptor can be
    opened.  The roots are explicit so tests never need host ``/sys`` or
    ``/dev``.
    """
    ensure_layout_intact()
    snapshot, device = resolve_capture_node(
        serial, profile=profile, sysfs_root=sysfs_root, dev_root=dev_root
    )
    reason = refusal_reason(device, serial, profile)
    if reason is not None:
        raise V4l2CaptureNotAllowed(reason)
    return snapshot, device


# --- the run's mutable state --------------------------------------------------


@dataclass(slots=True)
class _Mapped:
    """One successful mapping, and everything needed to release it once."""

    index: int
    offset: int
    length: int
    mapping: Mapping_
    released: bool = False


@dataclass(slots=True)
class _Session:
    """What the run owes the kernel, and whether it has been paid.

    Every flag is conservative in the same way the interface-0 lifecycle's is:
    raised *before* the call that creates the obligation, lowered only by that
    call returning a clean errno. A signal handler runs between bytecodes, so
    an exception surfacing from an ioctl leaves it unknown whether the kernel
    completed it, and the only safe assumption is that it did.
    """

    fd: int | None = None
    #: True from before ``VIDIOC_STREAMON`` is issued. Lowered only when that
    #: ioctl reports a clean failure, because then the kernel started nothing.
    streaming: bool = False
    #: Raised immediately before any STREAMON attempt for a depth session and
    #: never lowered afterward; the finalizer must run even on an ambiguous
    #: STREAMON signal/error.
    illumination_required: bool = False
    #: True from before ``VIDIOC_REQBUFS`` is issued, for the same reason.
    buffers_allocated: bool = False
    mappings: list[_Mapped] = field(default_factory=list)
    cleaned: bool = False
    cleanup: CleanupRecord = field(
        default_factory=lambda: CleanupRecord.not_needed("nothing was opened")
    )


@dataclass(slots=True)
class FrameVerdict:
    """The primary result being assembled, before cleanup is folded in.

    Named rather than private because a :class:`CaptureProfile`'s frame
    judgement returns one: it is the vocabulary in which a profile tells the
    shared engine what the bytes it dequeued turned out to be.
    """

    status: CaptureStatus
    detail: str
    errno: int | None = None
    errno_name: str | None = None
    #: Whatever the profile's frame judgement learned about the payload beyond
    #: the shared bounds - for the MJPEG profile, the structural JPEG report.
    #: Carried verbatim into :attr:`CaptureOutcome.frame_evidence` so a profile
    #: can add evidence without the engine knowing what it means.
    evidence: Mapping[str, object] | None = None


# --- the outcome --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PollRecord:
    """The single bounded wait, whether or not it produced a buffer.

    ``deadline_s`` is the whole budget the deadline was set to;
    ``timeout_s`` is what this one ``poll`` was actually given, which is the
    budget minus whatever of it had already elapsed since ``VIDIOC_STREAMON``
    returned. They differ, and reporting only the first would overstate how
    long the wait was allowed to be.
    """

    deadline_s: float
    timeout_s: float
    waited_s: float
    revents: int
    event_names: tuple[str, ...]
    timed_out: bool
    failed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "deadline_s": self.deadline_s,
            "timeout_s": self.timeout_s,
            "waited_s": self.waited_s,
            "revents": f"0x{self.revents:04x}",
            "event_names": list(self.event_names),
            "timed_out": self.timed_out,
            "failed": self.failed,
            "calls": 1,
            "restarts": 0,
        }


@dataclass(frozen=True, slots=True)
class SeriesObservation:
    """One cumulative observation from a bounded streaming series."""

    index: int
    buffer: BufferInfo
    payload: bytes
    verdict: FrameVerdict
    poll: PollRecord
    payload_copied: bool = True


@dataclass(frozen=True, slots=True)
class SeriesRun:
    """The cumulative result of one stream deadline and finite dequeue budget."""

    status: CaptureStatus
    detail: str
    observations: tuple[SeriesObservation, ...] = ()
    polls: tuple[PollRecord, ...] = ()
    retained_bytes: int = 0
    max_total_bytes: int = 0
    poll_count: int = 0
    dequeue_count: int = 0
    requeue_count: int = 0
    terminal_verdict: FrameVerdict | None = None


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    """The complete, typed result of the one attempt."""

    status: CaptureStatus
    detail: str
    target: NodeSnapshot | None
    steps: tuple[StepResult, ...]
    cleanup: CleanupRecord
    artifacts: ArtifactRecord
    capability: Capability | None = None
    initial_format: PixelFormat | None = None
    negotiated_format: PixelFormat | None = None
    buffers: RequestedBuffers | None = None
    queried_buffers: tuple[BufferInfo, ...] = ()
    mappings: tuple[dict[str, object], ...] = ()
    poll: PollRecord | None = None
    dequeued: BufferInfo | None = None
    frame_bytes: int | None = None
    stream_duration_s: float | None = None
    total_duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None
    #: Which fixed capture this outcome came from. Reported rather than
    #: inferred, so a report cannot be read against the wrong plan.
    profile: CaptureProfile = DEPTH_PROFILE
    #: Whatever the profile learned about the payload beyond the shared bounds.
    #: ``None`` for a profile that has nothing extra to say.
    frame_evidence: Mapping[str, object] | None = None
    illumination_shutdown: IlluminationShutdownOutcome = field(
        default_factory=illumination_not_required
    )

    @property
    def exit_code(self) -> int:
        """Cleanup, then evidence, outrank every primary status - success included."""
        if not self.cleanup.ok:
            return CLEANUP_FAILED_EXIT
        if not self.artifacts.ok:
            return ARTIFACT_FAILED_EXIT
        # This reads a finalizer that no longer scores its own command. Since
        # 2026-08-24 ``IlluminationShutdownOutcome.succeeded`` means "the owed
        # write was attempted, evidenced, and interface 0 was handed back",
        # never "the emitter went dark" -- the fixed off command is known not
        # to darken this unit. The gate stays because an undischarged attempt
        # or an unreleased interface is still a real failure of this run.
        if not self.illumination_shutdown.succeeded:
            return self.illumination_shutdown.exit_code
        return CAPTURE_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        """True only for a complete frame that was preserved and fully released.

        The illumination term is a statement about the finalizer's discharge
        and the USB interface, never about light; a true value here says
        nothing whatever about the emitter. See
        :data:`scanner3d.illumination_shutdown.EMITTER_LIMITATION`, which
        travels with the outcome into every report this capture emits.
        """
        return (
            self.status is CaptureStatus.COMPLETE
            and self.cleanup.ok
            and self.artifacts.ok
            and self.illumination_shutdown.succeeded
        )

    def step(self, step: Step) -> StepResult | None:
        """The last result recorded for ``step``, or ``None`` if it never ran."""
        for result in reversed(self.steps):
            if result.step is step:
                return result
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.profile.mode_label,
            "status": str(self.status),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "plan": self.profile.plan_as_dict(),
            "target": self.target.as_dict() if self.target is not None else None,
            "capability": self.capability.as_dict() if self.capability is not None else None,
            "initial_format": (
                self.initial_format.as_dict() if self.initial_format is not None else None
            ),
            "negotiated_format": (
                self.negotiated_format.as_dict() if self.negotiated_format is not None else None
            ),
            "buffers": self.buffers.as_dict() if self.buffers is not None else None,
            "queried_buffers": [b.as_dict() for b in self.queried_buffers],
            "mappings": [dict(m) for m in self.mappings],
            "poll": self.poll.as_dict() if self.poll is not None else None,
            "dequeued_buffer": self.dequeued.as_dict() if self.dequeued is not None else None,
            "frame_bytes": self.frame_bytes,
            "frame_evidence": (None if self.frame_evidence is None else dict(self.frame_evidence)),
            "artifacts": self.artifacts.as_dict(),
            "steps": [s.as_dict() for s in self.steps],
            "cleanup": self.cleanup.as_dict(),
            "illumination_shutdown": (self.illumination_shutdown.as_dict()),
            "stream_duration_s": self.stream_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


def plan_as_dict() -> dict[str, object]:
    """The depth plan. Kept as a module function because the depth report and
    its tests already name it; every profile answers the same question through
    :meth:`CaptureProfile.plan_as_dict`."""
    return DEPTH_PROFILE.plan_as_dict()


#: What the depth dry run says instead of opening the node.
DRY_RUN_DETAIL: Final = DEPTH_PROFILE.dry_run_detail


def _default_illumination_backend(
    sysfs_root: Path, dev_root: Path
) -> UsbfsIlluminationShutdownBackend:
    return UsbfsIlluminationShutdownBackend(sysfs_root=sysfs_root, dev_root=dev_root)


# --- the backend --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V4l2FrameBackend:
    """The one fixed V4L2 capture attempt, with every syscall injectable.

    This is *the* lifecycle in this package: one node resolution, one open, one
    QUERYCAP/G_FMT/S_FMT/REQBUFS, at most four QUERYBUF/mmap/QBUF, one STREAMON,
    one deadline-bounded poll, at most one DQBUF, and one shielded cleanup. What
    varies between captures is a :class:`CaptureProfile`, and nothing else - so
    there is one poll, one mmap, one ioctl helper and one cleanup to reason
    about rather than one per camera.

    Nothing here is a parameter but the serial and the local output directory.
    The node is resolved, not supplied; the profile is bound at import time; and
    there is no argument that can make it act twice.
    """

    opener: Opener = _default_opener
    ioctl_fn: IoctlFn = fcntl.ioctl
    mmap_fn: MmapFn = _default_mmap
    poll_fn: Poller = _default_poll
    closer: Closer = os.close
    clock: Clock = time.monotonic
    identity_check: IdentityCheck = _capture_identity_check
    sysfs_root: Path = DEFAULT_SYSFS_ROOT
    dev_root: Path = DEFAULT_DEV_ROOT
    signal_guard: SignalGuard = _capture_signal_guard
    termination_shield: TerminationShield = defer_termination
    #: The one fixed capture this backend performs. Depth by default because
    #: that is the profile this engine was first written for; a caller cannot
    #: reach this field through any command line.
    profile: CaptureProfile = DEPTH_PROFILE
    illumination_backend_factory: Callable[[Path, Path], UsbfsIlluminationShutdownBackend] = (
        _default_illumination_backend
    )

    # -- public API ----------------------------------------------------------

    def _output_dir(self, output_dir: Path | None) -> Path:
        return self.profile.default_output_dir if output_dir is None else output_dir

    def prepare_session(
        self,
        target: NodeSnapshot,
        output_dir: Path,
        started: float,
        steps: list[StepResult] | None = None,
    ) -> PreparedV4l2Session:
        """Open one already-preflighted node into a typed prepared session.

        The returned object owns the fd pessimistically from the moment
        ``open`` returns and is the shared lifecycle seam used by single-frame
        and paired capture.  It does not start streaming until ``stream_on``.
        """
        return PreparedV4l2Session.open(self, target, output_dir, started, steps)

    def preflight(self, serial: str, output_dir: Path | None = None) -> CaptureOutcome:
        """Resolve the node and check every precondition, opening nothing."""
        ensure_layout_intact()
        where = self._output_dir(output_dir)
        status, detail, target, steps = self._preconditions(serial)
        return CaptureOutcome(
            status=status,
            detail=(
                self.profile.dry_run_detail if status is CaptureStatus.NOT_ATTEMPTED else detail
            ),
            target=target,
            steps=steps,
            cleanup=CleanupRecord.not_needed("nothing was opened"),
            artifacts=ArtifactRecord.not_needed(
                "a dry run writes no frame", str(where), self.profile.max_raw_artifact_bytes
            ),
            profile=self.profile,
        )

    def run(self, serial: str, output_dir: Path | None = None) -> CaptureOutcome:
        """Attempt the capture exactly once. No step is ever retried.

        Refuses any serial but :data:`AUTHORIZED_SERIAL`: the owner's
        authorization for this attempt is for one physical unit, so the live
        path will not act on a device it was not given.
        """
        ensure_layout_intact()
        where = self._output_dir(output_dir)
        # The guard refuses a device the operator did not NAME. If they named
        # none, there is nothing to refuse against and the caller's own
        # resolution stands - so this asks configuration only, and never the
        # hardware. A guard that enumerates devices would both be slow and
        # ignore the roots a caller injected, which is how it was wrong before.
        authorized = configured_serial()
        if authorized is not None and serial != authorized:
            return self._refused(
                f"this capture is authorized for serial {authorized!r} only; "
                f"refusing to run it against {serial!r}",
                where,
            )
        started = self.clock()
        status, detail, target, steps = self._preconditions(serial)
        if status is not CaptureStatus.NOT_ATTEMPTED or target is None:
            return CaptureOutcome(
                status=status,
                detail=detail,
                target=target,
                steps=steps,
                cleanup=CleanupRecord.not_needed("nothing was opened"),
                artifacts=ArtifactRecord.not_needed(
                    "no attempt was made, so no frame was written",
                    str(where),
                    self.profile.max_raw_artifact_bytes,
                ),
                profile=self.profile,
            )
        return self._attempt(target, list(steps), where, started)

    # -- preconditions, before anything is opened ----------------------------

    def _preconditions(
        self, serial: str
    ) -> tuple[CaptureStatus, str, NodeSnapshot | None, tuple[StepResult, ...]]:
        try:
            snapshot, _device = preflight_capture_node(
                serial,
                profile=self.profile,
                sysfs_root=self.sysfs_root,
                dev_root=self.dev_root,
            )
        except DeviceAbsentError as exc:
            return (
                CaptureStatus.DEVICE_ABSENT,
                str(exc),
                None,
                (StepResult(step=Step.RESOLVE_NODE, ok=False, detail=str(exc)),),
            )
        except V4l2CaptureNotAllowed as exc:
            reason = str(exc)
            return (
                CaptureStatus.REFUSED,
                f"refusing to capture: {reason}",
                None,
                (StepResult(step=Step.RESOLVE_NODE, ok=False, detail=reason),),
            )
        return (
            CaptureStatus.NOT_ATTEMPTED,
            self.profile.dry_run_detail,
            snapshot,
            (
                StepResult(
                    step=Step.RESOLVE_NODE,
                    ok=True,
                    detail=(
                        f"{snapshot.node_path} is the V4L2 index-{snapshot.v4l_index} "
                        f"{self.profile.node_kind} node of the {snapshot.role} function "
                        "on interface "
                        f"{snapshot.interface_number} ({snapshot.interface_string!r}), "
                        f"bound to {snapshot.interface_driver}"
                    ),
                ),
            ),
        )

    def _refused(self, detail: str, output_dir: Path) -> CaptureOutcome:
        return CaptureOutcome(
            status=CaptureStatus.REFUSED,
            detail=detail,
            target=None,
            steps=(StepResult(step=Step.RESOLVE_NODE, ok=False, detail=detail),),
            cleanup=CleanupRecord.not_needed("nothing was opened"),
            artifacts=ArtifactRecord.not_needed(
                "no attempt was made", str(output_dir), self.profile.max_raw_artifact_bytes
            ),
            profile=self.profile,
        )

    # -- the attempt ---------------------------------------------------------

    def _attempt(
        self,
        target: NodeSnapshot,
        steps: list[StepResult],
        output_dir: Path,
        started: float,
    ) -> CaptureOutcome:
        """One open, one straight-line negotiation, one wait, one release."""
        session = _Session()
        state = _CaptureState()
        try:
            prepared = self.prepare_session(target, output_dir, started, steps)
        except OSError as exc:
            number, name = _errno_of(exc)
            steps.append(
                StepResult(
                    step=Step.OPEN,
                    ok=False,
                    detail=f"cannot open {target.node_path}: {exc}",
                    errno=number,
                    errno_name=name,
                )
            )
            return self._assemble(
                FrameVerdict(
                    status=_ERRNO_STATUS.get(number or 0, CaptureStatus.OPEN_ERROR),
                    detail=f"cannot open {target.node_path}: {exc}",
                    errno=number,
                    errno_name=name,
                ),
                target,
                steps,
                session,
                state,
                output_dir,
                started,
            )
        capture_exception: BaseException | None = None
        primary: FrameVerdict | None = None
        try:
            try:
                primary = prepared.capture()
            except BaseException as exc:
                # Preserve the established public interruption semantics: the
                # caller still receives the signal/KeyboardInterrupt, but the
                # finalizer below is guaranteed to run before it escapes.
                capture_exception = exc
        finally:
            # ``capture`` already releases in its own finally; this idempotent
            # backstop also covers unexpected BaseException paths. A broken
            # release must not skip the owed blue-light finalizer.
            try:
                prepared.release()
            except BaseException as exc:
                steps.append(
                    StepResult(
                        Step.CLOSE,
                        False,
                        f"V4L2 release raised {type(exc).__name__}: {exc}",
                    )
                )
                if primary is not None and primary.status is CaptureStatus.COMPLETE:
                    primary = FrameVerdict(CaptureStatus.IOCTL_ERROR, str(exc))
        illumination = self._shutdown_after_v4l2(prepared)
        if capture_exception is not None:
            raise capture_exception
        if primary is None:  # pragma: no cover - capture always returns or raises
            primary = FrameVerdict(CaptureStatus.IOCTL_ERROR, "capture returned no verdict")
        return self._assemble(
            primary,
            target,
            prepared.steps,
            prepared.session,
            prepared.state,
            output_dir,
            started,
            illumination,
        )

    def _shutdown_after_v4l2(self, prepared: PreparedV4l2Session) -> IlluminationShutdownOutcome:
        """Run the one off write only after exhaustive V4L2 teardown.

        The returned outcome describes a USB transfer and the interface-0
        lifecycle around it. It never observes the emitter, so nothing derived
        from it -- including this capture's exit code -- may be presented as
        evidence that the projector went dark.
        """
        if self.profile.role != CAPTURE_ROLE or not prepared.session.illumination_required:
            return illumination_not_required("depth STREAMON was never attempted")
        deferred: list[int] = []
        called = False
        outcome: IlluminationShutdownOutcome | None = None
        completed: IlluminationShutdownOutcome | None = None

        def run_shielded() -> IlluminationShutdownOutcome:
            nonlocal called, completed
            with self.termination_shield() as received:
                completed = None
                called = True
                backend = self.illumination_backend_factory(self.sysfs_root, self.dev_root)
                try:
                    completed = backend.run(AUTHORIZED_SERIAL)
                    return completed
                finally:
                    deferred.extend(received)

        def interrupted_outcome(detail: str) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True,
                called,
                ShutdownStatus.INTERRUPTED,
                detail,
                deferred_signals=tuple(deferred),
            )

        def failed_outcome(exc: BaseException) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True,
                called,
                (
                    ShutdownStatus.INTERRUPTED
                    if isinstance(exc, (KeyboardInterrupt, SignalReceived))
                    else ShutdownStatus.UNEXPECTED_EXCEPTION
                ),
                f"shutdown finalizer raised {type(exc).__name__}: {exc}",
                deferred_signals=tuple(deferred),
            )

        try:
            outcome = run_shielded()
        except SignalReceived as exc:
            deferred.append(exc.signum)
            # An injected shield can be interrupted before it yields. Enter a
            # fresh shield before starting the finalizer; never run it bare.
            if called:
                outcome = completed if completed is not None else interrupted_outcome(str(exc))
            else:
                try:
                    outcome = run_shielded()
                except SignalReceived as retry_exc:
                    deferred.append(retry_exc.signum)
                    outcome = (
                        completed if completed is not None else interrupted_outcome(str(retry_exc))
                    )
                except BaseException as fallback_exc:
                    outcome = completed if completed is not None else failed_outcome(fallback_exc)
        except BaseException as exc:
            outcome = completed if completed is not None else failed_outcome(exc)
        outcome = replace(
            outcome,
            deferred_signals=tuple(dict.fromkeys((*outcome.deferred_signals, *deferred))),
        )
        return outcome

    def _guarded_window(
        self,
        fd: int,
        target: NodeSnapshot,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
    ) -> FrameVerdict:
        """Run the negotiation and the wait, converting a signal into an unwind.

        The ``finally`` here is the first of the two that can reach cleanup; the
        caller's is the backstop for a signal that lands between this block
        returning and the release. :meth:`_release` is idempotent by design, so
        being reachable twice costs nothing.
        """
        try:
            with self.signal_guard():
                return self._negotiate_and_capture(fd, target, steps, session, state)
        except SignalReceived as sig:
            return FrameVerdict(CaptureStatus.INTERRUPTED, str(sig))
        except OSError as exc:
            number, name = _errno_of(exc)
            return FrameVerdict(
                status=_ERRNO_STATUS.get(number or 0, CaptureStatus.IOCTL_ERROR),
                detail=f"unexpected OSError in the capture window: {exc}",
                errno=number,
                errno_name=name,
            )
        except Exception as exc:
            return FrameVerdict(
                CaptureStatus.IOCTL_ERROR,
                f"unexpected {type(exc).__name__} in the capture window: {exc}",
            )
        finally:
            self._release(target, steps, session)

    def _negotiate_and_capture(
        self,
        fd: int,
        target: NodeSnapshot,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
    ) -> FrameVerdict:
        """The straight-line lifecycle. Each stage returns non-``None`` to stop."""
        failure = self._negotiate_and_prepare(fd, target, steps, session, state)
        if failure is not None:
            return failure
        return self._stream_once(fd, steps, session, state)

    def _negotiate_and_prepare(
        self,
        fd: int,
        target: NodeSnapshot,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
    ) -> FrameVerdict | None:
        """Negotiate and queue a stream, leaving STREAMON to the caller.

        Single-frame capture calls this helper and immediately starts the
        stream. Paired capture keeps the prepared depth session alive while it
        prepares RGB; all ioctl packing, bounds and profile checks remain in
        this shared engine.
        """
        drift = self.identity_check(fd, target)
        if drift is not None:
            return FrameVerdict(CaptureStatus.DEVICE_REPLACED, drift)
        for stage in (
            self._querycap,
            self._get_format,
            self._set_format,
            self._request_buffers,
            self._map_buffers,
            self._queue_buffers,
        ):
            failure = stage(fd, steps, session, state)
            if failure is not None:
                return failure
        return None

    # -- stages --------------------------------------------------------------

    def _querycap(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        packed = pack_capability()
        failure = self._ioctl(fd, VIDIOC_QUERYCAP, packed, Step.QUERYCAP, steps)
        if failure is not None:
            return failure
        capability = parse_capability(bytes(packed))
        state.capability = capability
        missing = capability.missing_required
        if missing:
            detail = (
                f"{capability.card!r} via {capability.driver!r} lacks {', '.join(missing)} "
                f"(effective capabilities 0x{capability.effective_capabilities:08x})"
            )
            steps.append(StepResult(step=Step.QUERYCAP, ok=False, detail=detail))
            return FrameVerdict(CaptureStatus.CAPABILITY_MISMATCH, detail)
        steps.append(
            StepResult(
                step=Step.QUERYCAP,
                ok=True,
                detail=(
                    f"driver {capability.driver!r} card {capability.card!r} bus "
                    f"{capability.bus_info!r} version {capability.version_text}; "
                    "video-capture and streaming both present"
                ),
            )
        )
        return None

    def _stride_note(self, negotiated: PixelFormat) -> str:
        """How the step log describes the negotiated row layout.

        A compressed profile has no rows to pad: ``uvcvideo`` reports
        ``bytesperline`` zero for MJPEG because a stride is meaningless for a
        bitstream. Saying "tightly packed" about that would invent a property
        the format does not have, so the compressed case says what it is
        instead.
        """
        if self.profile.compressed:
            return (
                f" (a compressed bitstream; bytesperline {negotiated.bytesperline} carries "
                "no row stride and no packing claim is made about it)"
            )
        if negotiated.padded:
            return f" (rows padded by {negotiated.bytesperline - MIN_BYTES_PER_LINE} B)"
        return " (rows tightly packed)"

    def _packed_format(self) -> bytearray:
        """The one ``struct v4l2_format`` this backend ever asks for."""
        return pack_format(
            width=self.profile.width,
            height=self.profile.height,
            pixelformat=self.profile.pixelformat,
        )

    def _get_format(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        packed = self._packed_format()
        # G_FMT reads only ``type``; the rest of what pack_format put there is
        # overwritten by the kernel's answer.
        failure = self._ioctl(fd, VIDIOC_G_FMT, packed, Step.GET_FMT, steps)
        if failure is not None:
            return failure
        _buf_type, pixel_format = parse_format(bytes(packed))
        state.initial_format = pixel_format
        steps.append(
            StepResult(
                step=Step.GET_FMT,
                ok=True,
                detail=(
                    f"this handle started at {pixel_format.fourcc!r} "
                    f"{pixel_format.width}x{pixel_format.height} "
                    f"bytesperline={pixel_format.bytesperline} "
                    f"sizeimage={pixel_format.sizeimage}"
                ),
            )
        )
        return None

    def _set_format(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        packed = self._packed_format()
        failure = self._ioctl(fd, VIDIOC_S_FMT, packed, Step.SET_FMT, steps)
        if failure is not None:
            return failure
        buf_type, negotiated = parse_format(bytes(packed))
        state.negotiated_format = negotiated
        reason = _format_refusal(buf_type, negotiated, self.profile)
        if reason is not None:
            steps.append(StepResult(step=Step.SET_FMT, ok=False, detail=reason))
            return FrameVerdict(CaptureStatus.FORMAT_MISMATCH, reason)
        steps.append(
            StepResult(
                step=Step.SET_FMT,
                ok=True,
                detail=(
                    f"negotiated {negotiated.fourcc!r} {negotiated.width}x{negotiated.height} "
                    f"bytesperline={negotiated.bytesperline} sizeimage={negotiated.sizeimage}"
                    + self._stride_note(negotiated)
                ),
            )
        )
        return None

    def _request_buffers(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        packed = pack_requestbuffers(
            self.profile.requested_buffers, limit=self.profile.requested_buffers
        )
        # Owed from before the ioctl: an interrupted REQBUFS may have allocated.
        session.buffers_allocated = True
        failure = self._ioctl(fd, VIDIOC_REQBUFS, packed, Step.REQBUFS, steps)
        if failure is not None:
            # A clean errno means the kernel allocated nothing.
            session.buffers_allocated = False
            return replace(failure, status=CaptureStatus.BUFFER_COUNT_MISMATCH)
        buffers = parse_requestbuffers(bytes(packed))
        state.buffers = buffers
        wanted = self.profile.requested_buffers
        floor = self.profile.min_accepted_buffers
        if not floor <= buffers.count <= wanted:
            detail = (
                f"the driver returned {buffers.count} buffers for a request of "
                f"{wanted}; only {floor} to {wanted} is accepted"
            )
            steps.append(StepResult(step=Step.REQBUFS, ok=False, detail=detail))
            return FrameVerdict(CaptureStatus.BUFFER_COUNT_MISMATCH, detail)
        steps.append(
            StepResult(
                step=Step.REQBUFS,
                ok=True,
                detail=f"{buffers.count} MMAP buffers allocated (requested {wanted})",
            )
        )
        return None

    def _map_buffers(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        """Query and map each allocated buffer exactly once, inside the ceilings."""
        if state.buffers is None:  # pragma: no cover - _request_buffers ran first
            return FrameVerdict(
                CaptureStatus.MAPPING_ERROR, "no buffer count was recorded to map against"
            )
        total = 0
        # Bounded twice over: by the count the kernel returned, and by the fact
        # that pack_buffer refuses an index outside 0..REQUESTED_BUFFERS-1.
        for index in range(state.buffers.count):
            packed = pack_buffer(index, limit=self.profile.requested_buffers)
            failure = self._ioctl(fd, VIDIOC_QUERYBUF, packed, Step.QUERYBUF, steps)
            if failure is not None:
                return replace(failure, status=CaptureStatus.MAPPING_ERROR)
            info = parse_buffer(bytes(packed))
            state.queried.append(info)
            steps.append(
                StepResult(
                    step=Step.QUERYBUF,
                    ok=True,
                    detail=(
                        f"buffer {info.index}: offset {info.offset} length {info.length} "
                        f"flags 0x{info.flags:08x}"
                    ),
                )
            )
            total += info.length
            reason = _mapping_refusal(info, index, total, self.profile)
            if reason is not None:
                steps.append(StepResult(step=Step.MAP, ok=False, detail=reason))
                return FrameVerdict(CaptureStatus.MAPPING_ERROR, reason)
            failure = self._map_one(fd, info, steps, session, state)
            if failure is not None:
                return failure
        state.mapped_bytes = total
        return None

    def _map_one(
        self,
        fd: int,
        info: BufferInfo,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
    ) -> FrameVerdict | None:
        started = self.clock()
        try:
            mapping = self.mmap_fn(fd, info.length, info.offset)
        except (OSError, ValueError) as exc:
            detail = f"cannot map buffer {info.index} at offset {info.offset}: {exc}"
            number, name = _errno_of(exc) if isinstance(exc, OSError) else (None, None)
            steps.append(
                StepResult(
                    step=Step.MAP,
                    ok=False,
                    detail=detail,
                    duration_s=self.clock() - started,
                    errno=number,
                    errno_name=name,
                )
            )
            return FrameVerdict(CaptureStatus.MAPPING_ERROR, detail, errno=number, errno_name=name)
        session.mappings.append(
            _Mapped(index=info.index, offset=info.offset, length=info.length, mapping=mapping)
        )
        mapped_length = len(mapping)
        state.mappings.append(
            {
                "index": info.index,
                "offset": info.offset,
                "requested_length": info.length,
                "mapped_length": mapped_length,
            }
        )
        steps.append(
            StepResult(
                step=Step.MAP,
                ok=True,
                detail=(
                    f"buffer {info.index} mapped read/write shared: {mapped_length} B "
                    f"at offset {info.offset}"
                ),
                duration_s=self.clock() - started,
            )
        )
        if mapped_length != info.length:
            detail = (
                f"buffer {info.index} mapped {mapped_length} B but QUERYBUF reported "
                f"{info.length} B"
            )
            steps.append(StepResult(step=Step.MAP, ok=False, detail=detail))
            return FrameVerdict(CaptureStatus.MAPPING_ERROR, detail)
        return None

    def _queue_buffers(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        for mapped in session.mappings:
            packed = pack_buffer(mapped.index, limit=self.profile.requested_buffers)
            failure = self._ioctl(fd, VIDIOC_QBUF, packed, Step.QBUF, steps)
            if failure is not None:
                return replace(failure, status=CaptureStatus.QUEUE_ERROR)
            steps.append(
                StepResult(step=Step.QBUF, ok=True, detail=f"buffer {mapped.index} queued")
            )
        return None

    def _stream_once(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict:
        """Start the stream, wait once, and dequeue at most one buffer."""
        failure = self._stream_on(fd, steps, session, state)
        if failure is not None:
            return failure
        return self._wait_once(fd, steps, session, state)

    def _stream_on(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict | None:
        """Start a prepared stream and anchor its absolute deadline."""
        packed = pack_buf_type()
        # The blue-light off write is owed from this boundary, before the
        # kernel can observe STREAMON. It remains owed on clean failure,
        # interruption and ambiguous exceptions -- and it stays owed even
        # though the command is now known not to darken this unit's emitter,
        # because the owner's durable constraint is that the attempt is made
        # and evidenced on every run that streams depth.
        if self.profile.role == CAPTURE_ROLE:
            session.illumination_required = True
        # Owed from before the ioctl: an interrupted STREAMON may have started.
        session.streaming = True
        stream_started = self.clock()
        failure = self._ioctl(fd, VIDIOC_STREAMON, packed, Step.STREAMON, steps)
        if failure is not None:
            session.streaming = False
            return replace(failure, status=CaptureStatus.STREAMON_ERROR)
        # The deadline is anchored here, not at ``stream_started``: nothing was
        # streaming until this ioctl returned, so a STREAMON that itself took
        # time must not be charged against the wait it precedes.
        streamon_returned = self.clock()
        budget = self.profile.capture_deadline_s
        deadline = streamon_returned + budget
        steps.append(
            StepResult(
                step=Step.STREAMON,
                ok=True,
                detail=f"streaming started; the {budget}s deadline starts now",
                duration_s=streamon_returned - stream_started,
            )
        )
        state.stream_started = stream_started

        state.stream_returned = streamon_returned
        state.deadline = deadline
        return None

    def _wait_once(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict:
        """Wait against the deadline anchored by :meth:`_stream_on`."""
        if state.deadline is None:  # pragma: no cover - ordering guard
            return FrameVerdict(CaptureStatus.STREAMON_ERROR, "stream was not started")
        budget = self.profile.capture_deadline_s
        # One poll, given no more than what is left of the budget. Whatever the
        # step bookkeeping above cost comes out of the wait, never out of the
        # deadline, and a budget already spent means a single non-blocking poll
        # rather than a second chance.
        poll_started = self.clock()
        timeout = max(0.0, state.deadline - poll_started)
        try:
            revents = self.poll_fn(fd, timeout)
        except Exception as exc:
            # ``Exception``, not ``BaseException``: ``SignalReceived`` and
            # ``KeyboardInterrupt`` must keep unwinding to the guard that
            # records an interrupted run and still discharges cleanup.
            waited = self.clock() - poll_started
            number, name = _errno_of(exc) if isinstance(exc, OSError) else (None, None)
            detail = (
                f"the {timeout:.3f}s poll after STREAMON raised "
                f"{type(exc).__name__}: {exc}; not retrying"
            )
            state.poll = PollRecord(
                deadline_s=budget,
                timeout_s=timeout,
                waited_s=waited,
                revents=0,
                event_names=(),
                timed_out=False,
                failed=True,
            )
            steps.append(
                StepResult(
                    step=Step.POLL,
                    ok=False,
                    detail=detail,
                    duration_s=waited,
                    errno=number,
                    errno_name=name,
                )
            )
            return FrameVerdict(CaptureStatus.POLL_ERROR, detail, errno=number, errno_name=name)
        waited = self.clock() - poll_started
        names = _poll_event_names(revents)
        state.poll = PollRecord(
            deadline_s=budget,
            timeout_s=timeout,
            waited_s=waited,
            revents=revents,
            event_names=names,
            timed_out=revents == 0,
        )
        if revents == 0:
            detail = (
                f"no buffer became ready within {budget}s of STREAMON "
                f"({timeout:.3f}s of it left for the one poll, waited {waited:.3f}s); "
                "not retrying"
            )
            steps.append(StepResult(step=Step.POLL, ok=False, detail=detail, duration_s=waited))
            return FrameVerdict(CaptureStatus.POLL_TIMEOUT, detail)
        if not revents & select.POLLIN:
            detail = f"poll reported {', '.join(names) or f'0x{revents:04x}'} without POLLIN"
            steps.append(StepResult(step=Step.POLL, ok=False, detail=detail, duration_s=waited))
            return FrameVerdict(CaptureStatus.POLL_ERROR, detail)
        steps.append(
            StepResult(
                step=Step.POLL,
                ok=True,
                detail=f"readable after {waited:.3f}s ({', '.join(names)})",
                duration_s=waited,
            )
        )
        return self._dequeue_once(fd, steps, session, state)

    def wait_series(
        self,
        fd: int,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
        *,
        max_frames: int,
        max_total_bytes: int,
        on_observation: Callable[[SeriesObservation], None] | None = None,
        on_progress: Callable[[SeriesRun], None] | None = None,
        on_telemetry: Callable[[int, SeriesObservation | None], None] | None = None,
    ) -> SeriesRun:
        """Collect a finite series under the one STREAMON-anchored deadline.

        This is the cumulative counterpart to :meth:`_wait_once`; it keeps the
        same poll, dequeue, bounds, JPEG judgement and QBUF helpers. The
        absolute deadline is never restarted, and a dequeued payload is copied
        and classified before it is requeued for the next observation.
        """
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if max_total_bytes < 1:
            raise ValueError("max_total_bytes must be positive")
        if state.deadline is None:
            return SeriesRun(CaptureStatus.STREAMON_ERROR, "stream was not started")
        observations: list[SeriesObservation] = []
        polls: list[PollRecord] = []
        retained = 0
        dequeues = 0
        requeues = 0
        budget = self.profile.capture_deadline_s

        def result(
            status: CaptureStatus,
            detail: str,
            terminal_verdict: FrameVerdict | None = None,
        ) -> SeriesRun:
            return SeriesRun(
                status,
                detail,
                tuple(observations),
                tuple(polls),
                retained,
                max_total_bytes,
                len(polls),
                dequeues,
                requeues,
                terminal_verdict,
            )

        def exception_verdict(stage: str, exc: BaseException) -> FrameVerdict:
            number, name = _errno_of(exc) if isinstance(exc, OSError) else (None, None)
            status = (
                CaptureStatus.INTERRUPTED
                if isinstance(exc, (SignalReceived, KeyboardInterrupt))
                else CaptureStatus.IOCTL_ERROR
            )
            detail = f"RGB series {stage} raised {type(exc).__name__}: {exc}"
            return FrameVerdict(status, detail, number, name)

        def notify(terminal: FrameVerdict | None = None) -> None:
            if on_progress is not None:
                on_progress(
                    result(
                        terminal.status if terminal is not None else CaptureStatus.COMPLETE,
                        terminal.detail if terminal is not None else "series progress",
                        terminal,
                    )
                )

        # Point zero is deliberately before the first poll and has no frame
        # association. The callback owns the fixed three G_CTRL attempts and
        # may raise a signal/exception; the cumulative caller preserves its
        # partial sample before cleanup.
        if on_telemetry is not None:
            try:
                on_telemetry(0, None)
            except BaseException as exc:
                verdict = exception_verdict("initial telemetry", exc)
                notify(verdict)
                raise

        for index in range(max_frames):
            poll_started = self.clock()
            timeout = max(0.0, state.deadline - poll_started)
            try:
                revents = self.poll_fn(fd, timeout)
            except Exception as exc:
                waited = self.clock() - poll_started
                number, name = _errno_of(exc) if isinstance(exc, OSError) else (None, None)
                detail = (
                    f"the {timeout:.3f}s series poll after STREAMON raised "
                    f"{type(exc).__name__}: {exc}; not retrying"
                )
                record = PollRecord(
                    deadline_s=budget,
                    timeout_s=timeout,
                    waited_s=waited,
                    revents=0,
                    event_names=(),
                    timed_out=False,
                    failed=True,
                )
                polls.append(record)
                steps.append(
                    StepResult(
                        step=Step.POLL,
                        ok=False,
                        detail=detail,
                        duration_s=waited,
                        errno=number,
                        errno_name=name,
                    )
                )
                return result(
                    CaptureStatus.POLL_ERROR,
                    detail,
                    FrameVerdict(CaptureStatus.POLL_ERROR, detail, number, name),
                )
            except BaseException as exc:
                waited = self.clock() - poll_started
                verdict = exception_verdict("poll", exc)
                polls.append(
                    PollRecord(
                        deadline_s=budget,
                        timeout_s=timeout,
                        waited_s=waited,
                        revents=0,
                        event_names=(),
                        timed_out=False,
                        failed=True,
                    )
                )
                steps.append(
                    StepResult(
                        step=Step.POLL,
                        ok=False,
                        detail=verdict.detail,
                        duration_s=waited,
                        errno=verdict.errno,
                        errno_name=verdict.errno_name,
                    )
                )
                notify(verdict)
                raise
            waited = self.clock() - poll_started
            names = _poll_event_names(revents)
            record = PollRecord(
                deadline_s=budget,
                timeout_s=timeout,
                waited_s=waited,
                revents=revents,
                event_names=names,
                timed_out=revents == 0,
            )
            polls.append(record)
            if revents == 0:
                detail = (
                    f"the RGB series deadline elapsed after {len(observations)} observation(s); "
                    f"the one poll waited {waited:.3f}s of {timeout:.3f}s remaining"
                )
                steps.append(StepResult(step=Step.POLL, ok=False, detail=detail, duration_s=waited))
                return result(
                    CaptureStatus.POLL_TIMEOUT,
                    detail,
                    FrameVerdict(CaptureStatus.POLL_TIMEOUT, detail),
                )
            if not revents & select.POLLIN:
                detail = (
                    f"series poll reported {', '.join(names) or f'0x{revents:04x}'} without POLLIN"
                )
                steps.append(StepResult(step=Step.POLL, ok=False, detail=detail, duration_s=waited))
                return result(
                    CaptureStatus.POLL_ERROR,
                    detail,
                    FrameVerdict(CaptureStatus.POLL_ERROR, detail),
                )
            steps.append(
                StepResult(
                    step=Step.POLL,
                    ok=True,
                    detail=(
                        f"series observation {index} readable after {waited:.3f}s "
                        f"({', '.join(names)})"
                    ),
                    duration_s=waited,
                )
            )
            # A failed DQBUF must not be mistaken for the previous successful
            # observation when this cumulative loop advances.
            state.dequeued = None
            state.frame = None
            dequeues += 1
            try:
                verdict = self._dequeue_once(fd, steps, session, state)
            except BaseException as exc:
                verdict = exception_verdict("DQBUF", exc)
                steps.append(
                    StepResult(
                        step=Step.DQBUF,
                        ok=False,
                        detail=verdict.detail,
                        errno=verdict.errno,
                        errno_name=verdict.errno_name,
                    )
                )
                notify(verdict)
                raise
            info = cast(BufferInfo, state.dequeued)
            payload = cast(bytes | None, state.frame)
            if payload is None:
                # Transport and safety failures have no retained payload. Keep
                # the original typed verdict (including errno/detail) and do
                # not classify an empty byte string or requeue an unknown
                # buffer. Strict-invalid copied bytes are the only failures
                # that continue to the next bounded observation.
                return result(verdict.status, verdict.detail, verdict)
            if retained + len(payload) > max_total_bytes:
                detail = (
                    f"series payload total would exceed the {max_total_bytes} B retained-byte "
                    f"ceiling at observation {index}"
                )
                return result(
                    CaptureStatus.OVERSIZE_FRAME,
                    detail,
                    FrameVerdict(CaptureStatus.OVERSIZE_FRAME, detail),
                )
            observation = SeriesObservation(
                index,
                info,
                payload,
                verdict,
                record,
                payload is not None,
            )
            observations.append(observation)
            retained += len(payload)
            if on_observation is not None:
                try:
                    on_observation(observation)
                except BaseException as exc:
                    verdict = exception_verdict("observation preservation", exc)
                    notify(verdict)
                    raise
            if on_telemetry is not None:
                try:
                    on_telemetry(index + 1, observation)
                except BaseException as exc:
                    verdict = exception_verdict("telemetry", exc)
                    notify(verdict)
                    raise
            if index + 1 >= max_frames:
                break
            packed = pack_buffer(info.index, limit=self.profile.requested_buffers)
            try:
                failure = self._ioctl(fd, VIDIOC_QBUF, packed, Step.QBUF, steps)
            except BaseException as exc:
                verdict = exception_verdict("QBUF", exc)
                steps.append(
                    StepResult(
                        step=Step.QBUF,
                        ok=False,
                        detail=verdict.detail,
                        errno=verdict.errno,
                        errno_name=verdict.errno_name,
                    )
                )
                notify(verdict)
                raise
            if failure is not None:
                detail = f"series requeue of buffer {info.index} failed: {failure.detail}"
                return result(
                    CaptureStatus.QUEUE_ERROR,
                    detail,
                    FrameVerdict(
                        CaptureStatus.QUEUE_ERROR,
                        detail,
                        failure.errno,
                        failure.errno_name,
                    ),
                )
            requeues += 1
            notify()
            steps.append(
                StepResult(step=Step.QBUF, ok=True, detail=f"buffer {info.index} requeued")
            )
        if not observations:
            return result(CaptureStatus.POLL_TIMEOUT, "RGB series produced no observations")
        statuses = [observation.verdict.status for observation in observations]
        if all(status is CaptureStatus.COMPLETE for status in statuses):
            return result(
                CaptureStatus.COMPLETE,
                f"delivered the pinned {len(observations)} RGB observations",
            )
        first_failure = next(status for status in statuses if status is not CaptureStatus.COMPLETE)
        return result(
            first_failure,
            "delivered "
            f"{len(observations)} RGB observation(s), but at least one failed strict validation",
        )

    def _dequeue_once(
        self, fd: int, steps: list[StepResult], session: _Session, state: _CaptureState
    ) -> FrameVerdict:
        """The single ``VIDIOC_DQBUF``. The buffer is never requeued."""
        packed = pack_buffer(0, limit=self.profile.requested_buffers)
        failure = self._ioctl(fd, VIDIOC_DQBUF, packed, Step.DQBUF, steps)
        if failure is not None:
            return replace(failure, status=CaptureStatus.DEQUEUE_ERROR)
        info = parse_buffer(bytes(packed))
        state.dequeued = info
        steps.append(
            StepResult(
                step=Step.DQBUF,
                ok=True,
                detail=(
                    f"dequeued buffer {info.index}: bytesused {info.bytesused} of "
                    f"{info.length} B, sequence {info.sequence}, field {info.field_name}, "
                    f"flags 0x{info.flags:08x}"
                ),
            )
        )
        return self._take_frame(info, steps, session, state)

    def _take_frame(
        self,
        info: BufferInfo,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
    ) -> FrameVerdict:
        """Copy the dequeued bytes out, inside every bound, or refuse to.

        The copy happens here rather than after cleanup because the mapping is
        about to be released; it is a plain slice of a read-only view, bounded
        three ways over, and it never reaches past ``bytesused``.

        Only the bounds that hold for every profile live here: the buffer must
        be one this run mapped, ``bytesused`` may not reach past that mapping,
        and it may not exceed the profile's artifact ceiling. Whether the bytes
        that survive those bounds are a *frame* is the profile's judgement, and
        it is asked exactly once.
        """
        mapped = next((m for m in session.mappings if m.index == info.index), None)
        if mapped is None:
            detail = f"the kernel dequeued buffer {info.index}, which was never mapped"
            return FrameVerdict(CaptureStatus.DEQUEUE_ERROR, detail)
        if info.bytesused > mapped.length:
            detail = (
                f"buffer {info.index} reports {info.bytesused} bytes used but only "
                f"{mapped.length} are mapped; nothing was copied"
            )
            return FrameVerdict(CaptureStatus.OVERSIZE_FRAME, detail)
        ceiling = self.profile.max_raw_artifact_bytes
        if info.bytesused > ceiling:
            detail = (
                f"buffer {info.index} reports {info.bytesused} bytes, past the "
                f"{ceiling} B artifact ceiling; nothing was copied rather "
                "than a truncated file being called a frame"
            )
            return FrameVerdict(CaptureStatus.OVERSIZE_FRAME, detail)
        raw = bytes(mapped.mapping[0 : info.bytesused])
        state.frame = raw
        if info.errored:
            detail = (
                f"the kernel flagged buffer {info.index} with V4L2_BUF_FLAG_ERROR; its "
                f"{len(raw)} bytes are preserved but are not a trustworthy frame"
            )
            return FrameVerdict(CaptureStatus.SHORT_FRAME, detail)
        return self.profile.judge(self.profile, raw, info, state.negotiated_format)

    # -- the one ioctl helper ------------------------------------------------

    def _ioctl(
        self,
        fd: int,
        command: int,
        packed: bytearray,
        step: Step,
        steps: list[StepResult],
    ) -> FrameVerdict | None:
        """Issue one ioctl, recording a failure as a step and a primary result.

        Returns ``None`` on success. The caller decides which typed status a
        failure maps to; the errno-derived one here is the fallback for the
        conditions that mean the same thing at every stage.
        """
        started = self.clock()
        try:
            self.ioctl_fn(fd, command, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            detail = f"{_ioctl_name(command)} failed: {exc}"
            steps.append(
                StepResult(
                    step=step,
                    ok=False,
                    detail=detail,
                    duration_s=self.clock() - started,
                    errno=number,
                    errno_name=name,
                )
            )
            return FrameVerdict(
                status=_ERRNO_STATUS.get(number or 0, CaptureStatus.IOCTL_ERROR),
                detail=detail,
                errno=number,
                errno_name=name,
            )
        return None

    # -- cleanup -------------------------------------------------------------

    def _release(self, target: NodeSnapshot, steps: list[StepResult], session: _Session) -> None:
        """Discharge every obligation exactly once, from whichever ``finally`` is first.

        The shield is entered before the "already done" test, for the same
        reason the interface-0 lifecycle does it: a signal arriving before the
        takeover completes raises out of here having changed nothing, so the
        outer ``finally`` still makes the one attempt; a signal arriving after
        it is absorbed and merely reported.

        The order is STREAMOFF, unmap, REQBUFS(0), close. Unmapping precedes the
        queue release because ``VIDIOC_REQBUFS`` answers ``EBUSY`` while any
        buffer is still mapped, so the other order could never succeed. Closing
        the descriptor is last and would release everything above it anyway;
        each step is still attempted, because a failure that only ``close``
        papered over is a failure the owner should see.
        """
        if session.cleaned:
            return
        if session.fd is None:
            session.cleanup = CleanupRecord.not_needed("nothing was opened")
            session.cleaned = True
            return

        session.cleanup = self._unfinished(target)
        fd = session.fd
        results: list[StepResult] = []
        absorbed: list[int] = []
        shield_entry_signals: list[int] = []
        cleanup_finished = False

        def run_cleanup() -> None:
            nonlocal cleanup_finished
            if cleanup_finished:
                return
            try:
                results.extend(self._cleanup_steps(fd, session))
            finally:
                cleanup_finished = True

        def shielded_cleanup() -> list[int]:
            with self.termination_shield() as deferred:
                run_cleanup()
                return list(deferred)

        # A signal can beat the shield into place. Retry a fresh shield before
        # falling back to the ordinary finally guarantee; the obligation list
        # makes all three paths idempotent and prevents duplicate teardown.
        try:
            absorbed.extend(shielded_cleanup())
        except SignalReceived as exc:
            shield_entry_signals.append(exc.signum)
            if cleanup_finished:
                raise
            try:
                absorbed.extend(shielded_cleanup())
            except SignalReceived as retry_exc:
                shield_entry_signals.append(retry_exc.signum)
                if cleanup_finished:
                    raise
                run_cleanup()
            except BaseException:
                if cleanup_finished:
                    raise
                run_cleanup()
        except BaseException:
            if cleanup_finished:
                raise
            try:
                absorbed.extend(shielded_cleanup())
            except SignalReceived as retry_exc:
                shield_entry_signals.append(retry_exc.signum)
                if cleanup_finished:
                    raise
                run_cleanup()
            except BaseException:
                if cleanup_finished:
                    raise
                run_cleanup()

        session.cleanup = self._summarise(target, results)
        steps.extend(results)
        session.cleaned = True
        retained_signals = absorbed + shield_entry_signals
        if retained_signals:
            session.cleanup = replace(
                session.cleanup,
                note=f"{session.cleanup.note} ({_absorbed(retained_signals)})",
            )

    def _cleanup_steps(self, fd: int, session: _Session) -> list[StepResult]:
        """The four obligations, each attempted once, none skipped for a failure."""
        results: list[StepResult] = []
        if session.streaming:
            results.append(self._guarded(Step.STREAMOFF, lambda: self._streamoff(fd)))
        for mapped in session.mappings:
            results.append(self._guarded(Step.UNMAP, lambda m=mapped: self._unmap(m)))  # type: ignore[misc]
        if session.buffers_allocated:
            results.append(self._guarded(Step.RELEASE_BUFFERS, lambda: self._free_buffers(fd)))
        results.append(self._guarded(Step.CLOSE, lambda: self._close(fd, session)))
        return results

    def _streamoff(self, fd: int) -> StepResult:
        started = self.clock()
        packed = pack_buf_type()
        try:
            self.ioctl_fn(fd, VIDIOC_STREAMOFF, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return StepResult(
                step=Step.STREAMOFF,
                ok=False,
                detail=f"VIDIOC_STREAMOFF failed: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        return StepResult(
            step=Step.STREAMOFF,
            ok=True,
            detail="VIDIOC_STREAMOFF succeeded; the queue is stopped",
            duration_s=self.clock() - started,
        )

    def _unmap(self, mapped: _Mapped) -> StepResult:
        if mapped.released:  # pragma: no cover - _release runs the list once
            return StepResult(
                step=Step.UNMAP, ok=True, detail=f"buffer {mapped.index} was already unmapped"
            )
        started = self.clock()
        try:
            mapped.mapping.close()
        except (OSError, ValueError, BufferError) as exc:
            return StepResult(
                step=Step.UNMAP,
                ok=False,
                detail=f"cannot unmap buffer {mapped.index}: {exc}",
                duration_s=self.clock() - started,
            )
        mapped.released = True
        return StepResult(
            step=Step.UNMAP,
            ok=True,
            detail=f"buffer {mapped.index} unmapped ({mapped.length} B at {mapped.offset})",
            duration_s=self.clock() - started,
        )

    def _free_buffers(self, fd: int) -> StepResult:
        started = self.clock()
        packed = pack_requestbuffers(0, limit=self.profile.requested_buffers)
        try:
            self.ioctl_fn(fd, VIDIOC_REQBUFS, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return StepResult(
                step=Step.RELEASE_BUFFERS,
                ok=False,
                detail=f"VIDIOC_REQBUFS(count=0) failed: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        return StepResult(
            step=Step.RELEASE_BUFFERS,
            ok=True,
            detail="VIDIOC_REQBUFS(count=0) succeeded; the buffer queue is freed",
            duration_s=self.clock() - started,
        )

    def _close(self, fd: int, session: _Session) -> StepResult:
        started = self.clock()
        try:
            self.closer(fd)
        except OSError as exc:
            number, name = _errno_of(exc)
            return StepResult(
                step=Step.CLOSE,
                ok=False,
                detail=f"cannot close fd {fd}: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        session.fd = None
        return StepResult(
            step=Step.CLOSE,
            ok=True,
            detail=f"fd {fd} closed",
            duration_s=self.clock() - started,
        )

    @staticmethod
    def _guarded(step: Step, action: Callable[[], StepResult]) -> StepResult:
        try:
            return action()
        # Deliberately broad, BaseException included: cleanup is the last line
        # of defence and nothing may abort it part-way, so a step that fails
        # this way still lets the next one run and still reports itself.
        except BaseException as exc:
            return StepResult(
                step=step, ok=False, detail=f"{step} raised {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _unfinished(target: NodeSnapshot) -> CleanupRecord:
        """The pessimistic record cleanup starts from, and must earn its way out of."""
        return CleanupRecord(
            attempted=True,
            steps=(),
            ok=False,
            note=f"{target.node_path} cleanup began but did not finish",
            guidance=recovery_guidance(target.node_path),
        )

    @staticmethod
    def _summarise(target: NodeSnapshot, results: list[StepResult]) -> CleanupRecord:
        ok = all(result.ok for result in results)
        return CleanupRecord(
            attempted=True,
            steps=tuple(results),
            ok=ok,
            note=(
                f"the stream, the buffers, every mapping and {target.node_path} were released"
                if ok
                else f"{target.node_path} was NOT fully released"
            ),
            guidance=None if ok else recovery_guidance(target.node_path),
        )

    # -- assembly ------------------------------------------------------------

    def _assemble(
        self,
        primary: FrameVerdict,
        target: NodeSnapshot,
        steps: list[StepResult],
        session: _Session,
        state: _CaptureState,
        output_dir: Path,
        started: float,
        illumination_shutdown: IlluminationShutdownOutcome | None = None,
    ) -> CaptureOutcome:
        """Fold cleanup and evidence into one outcome. Writes the frame if there is one."""
        artifacts = self._preserve(self.profile, primary, state, output_dir)
        return CaptureOutcome(
            status=primary.status,
            detail=primary.detail,
            target=target,
            steps=tuple(steps),
            cleanup=session.cleanup,
            artifacts=artifacts,
            capability=state.capability,
            initial_format=state.initial_format,
            negotiated_format=state.negotiated_format,
            buffers=state.buffers,
            queried_buffers=tuple(state.queried),
            mappings=tuple(state.mappings),
            poll=state.poll,
            dequeued=state.dequeued,
            frame_bytes=None if state.frame is None else len(state.frame),
            stream_duration_s=(
                None if state.stream_started is None else self.clock() - state.stream_started
            ),
            total_duration_s=self.clock() - started,
            errno=primary.errno,
            errno_name=primary.errno_name,
            profile=self.profile,
            frame_evidence=primary.evidence,
            illumination_shutdown=(
                illumination_shutdown
                if illumination_shutdown is not None
                else illumination_not_required("depth STREAMON was never attempted")
            ),
        )

    @staticmethod
    def _preserve(
        profile: CaptureProfile,
        primary: FrameVerdict,
        state: _CaptureState,
        output_dir: Path,
    ) -> ArtifactRecord:
        """Write the frame only when bytes were actually dequeued."""
        if state.frame is None:
            return ArtifactRecord.not_needed(
                f"no bytes were dequeued ({primary.status}), so no frame file was written",
                str(output_dir),
                profile.max_raw_artifact_bytes,
            )
        if state.negotiated_format is None:  # pragma: no cover - a frame implies a format
            return ArtifactRecord(
                attempted=False,
                ok=False,
                note="bytes were dequeued without a negotiated format to describe them",
                directory=str(output_dir),
                files=(),
                ceiling_bytes=profile.max_raw_artifact_bytes,
            )
        return profile.preserve(profile, state.frame, state.negotiated_format, output_dir)


class V4l2DepthFrameBackend(V4l2FrameBackend):
    """The shared engine bound to :data:`DEPTH_PROFILE`, and to nothing else.

    A distinct name rather than an alias so the depth command names the capture
    it performs, and so a test can assert that this backend can reach only the
    depth role and only ``Y16 `` 640x400. It adds no field, no method and no
    behaviour: the profile default it inherits *is* the whole difference.
    """

    __slots__ = ()


@dataclass(slots=True)
class _CaptureState:
    """Everything the run learned, accumulated so a failure still reports it."""

    capability: Capability | None = None
    initial_format: PixelFormat | None = None
    negotiated_format: PixelFormat | None = None
    buffers: RequestedBuffers | None = None
    queried: list[BufferInfo] = field(default_factory=list)
    mappings: list[dict[str, object]] = field(default_factory=list)
    mapped_bytes: int = 0
    poll: PollRecord | None = None
    dequeued: BufferInfo | None = None
    frame: bytes | None = None
    stream_started: float | None = None
    stream_returned: float | None = None
    deadline: float | None = None


@dataclass(slots=True)
class PreparedV4l2Session:
    """Typed shared V4L2 session whose obligations can span two streams.

    The session owns the descriptor immediately after ``open`` returns. Its
    public stage methods are deliberately narrow: configure, queue, start,
    wait once, and release. Both the single-frame backend and paired backend
    use this same object, while profiles continue to own their fixed checks.
    """

    engine: V4l2FrameBackend
    target: NodeSnapshot
    output_dir: Path
    started: float
    fd: int
    session: _Session
    state: _CaptureState
    steps: list[StepResult]
    primary: FrameVerdict | None = None

    @classmethod
    def open(
        cls,
        engine: V4l2FrameBackend,
        target: NodeSnapshot,
        output_dir: Path,
        started: float,
        steps: list[StepResult] | None = None,
    ) -> PreparedV4l2Session:
        fd = engine.opener(target.node_path)
        session = _Session(fd=fd)
        run_steps = [] if steps is None else steps
        run_steps.append(
            StepResult(
                step=Step.OPEN,
                ok=True,
                detail=f"opened {target.node_path} O_RDWR|O_NONBLOCK|O_CLOEXEC as fd {fd}",
            )
        )
        return cls(
            engine=engine,
            target=target,
            output_dir=output_dir,
            started=started,
            fd=fd,
            session=session,
            state=_CaptureState(),
            steps=run_steps,
        )

    @classmethod
    def adopt_open_fd(
        cls,
        engine: V4l2FrameBackend,
        target: NodeSnapshot,
        output_dir: Path,
        started: float,
        fd: int,
        steps: list[StepResult] | None = None,
    ) -> PreparedV4l2Session:
        """Adopt an already-open descriptor before any fallible validation."""
        run_steps = [] if steps is None else steps
        return cls(
            engine=engine,
            target=target,
            output_dir=output_dir,
            started=started,
            fd=fd,
            session=_Session(fd=fd),
            state=_CaptureState(),
            steps=run_steps,
        )

    def configure(self) -> None:
        self._run_stages((self.engine._querycap, self.engine._get_format, self.engine._set_format))

    def queue(self) -> None:
        self._run_stages(
            (self.engine._request_buffers, self.engine._map_buffers, self.engine._queue_buffers)
        )

    def stream_on(self) -> None:
        self._set_primary(self.engine._stream_on(self.fd, self.steps, self.session, self.state))

    def wait_once(self) -> bytes | None:
        self.primary = self.engine._wait_once(self.fd, self.steps, self.session, self.state)
        return self.state.frame

    def wait_series(
        self,
        *,
        max_frames: int,
        max_total_bytes: int,
        on_observation: Callable[[SeriesObservation], None] | None = None,
        on_progress: Callable[[SeriesRun], None] | None = None,
        on_telemetry: Callable[[int, SeriesObservation | None], None] | None = None,
    ) -> SeriesRun:
        """Collect cumulative observations without changing single-frame wait semantics."""
        result = self.engine.wait_series(
            self.fd,
            self.steps,
            self.session,
            self.state,
            max_frames=max_frames,
            max_total_bytes=max_total_bytes,
            on_observation=on_observation,
            on_progress=on_progress,
            on_telemetry=on_telemetry,
        )
        if result.status is not CaptureStatus.COMPLETE:
            self.primary = result.terminal_verdict or FrameVerdict(result.status, result.detail)
        elif result.observations:
            self.primary = result.observations[-1].verdict
        else:
            self.primary = FrameVerdict(result.status, result.detail)
        return result

    def capture(self) -> FrameVerdict:
        """Run the single-frame lifecycle through the public session API."""
        try:
            with self.engine.signal_guard():
                try:
                    self.validate_identity()
                    self.configure()
                    self.queue()
                    self.stream_on()
                    self.wait_once()
                    assert self.primary is not None
                    return self.primary
                except SignalReceived as sig:
                    self.primary = FrameVerdict(CaptureStatus.INTERRUPTED, str(sig))
                except V4l2SessionError as exc:
                    self.primary = exc.verdict
                except OSError as exc:
                    number, name = _errno_of(exc)
                    self.primary = FrameVerdict(
                        _ERRNO_STATUS.get(number or 0, CaptureStatus.IOCTL_ERROR),
                        f"unexpected OSError in the capture window: {exc}",
                        errno=number,
                        errno_name=name,
                    )
                except Exception as exc:
                    self.primary = FrameVerdict(
                        CaptureStatus.IOCTL_ERROR,
                        f"unexpected {type(exc).__name__} in the capture window: {exc}",
                    )
                finally:
                    # The first release attempt must hand the termination shield
                    # over while the capture guard is still active.  _attempt's
                    # outer finally remains the idempotent backstop.
                    self.release()
        except SignalReceived as sig:
            self.primary = FrameVerdict(CaptureStatus.INTERRUPTED, str(sig))
        except V4l2SessionError as exc:
            self.primary = exc.verdict
        except OSError as exc:
            number, name = _errno_of(exc)
            self.primary = FrameVerdict(
                _ERRNO_STATUS.get(number or 0, CaptureStatus.IOCTL_ERROR),
                f"unexpected OSError in the capture window: {exc}",
                errno=number,
                errno_name=name,
            )
        except Exception as exc:
            self.primary = FrameVerdict(
                CaptureStatus.IOCTL_ERROR,
                f"unexpected {type(exc).__name__} in the capture window: {exc}",
            )
        assert self.primary is not None
        return self.primary

    def validate_identity(self) -> None:
        """Run the shared post-open identity check before the first ioctl."""
        drift = self.engine.identity_check(self.fd, self.target)
        if drift is not None:
            self._set_primary(FrameVerdict(CaptureStatus.DEVICE_REPLACED, drift))

    def release(self) -> None:
        self.engine._release(self.target, self.steps, self.session)

    def stream_off(self) -> StepResult:
        return self.engine._streamoff(self.fd)

    def unmap_all(self) -> tuple[StepResult, ...]:
        results: list[StepResult] = []
        for mapping in self.session.mappings:

            def cleanup_mapping(item: _Mapped = mapping) -> StepResult:
                return self.engine._unmap(item)

            results.append(self.engine._guarded(Step.UNMAP, cleanup_mapping))
        return tuple(results)

    def preserve(self, primary: FrameVerdict | None = None) -> ArtifactRecord:
        """Preserve this session's bounded frame as a typed artifact result."""
        verdict = primary or self.primary
        if verdict is None:
            verdict = FrameVerdict(CaptureStatus.IOCTL_ERROR, "role did not reach a result")
        return self.engine._preserve(self.engine.profile, verdict, self.state, self.output_dir)

    def free_queue(self) -> StepResult:
        return self.engine._free_buffers(self.fd)

    def close(self) -> StepResult:
        return self.engine._close(self.fd, self.session)

    def _run_stages(
        self,
        stages: tuple[
            Callable[[int, list[StepResult], _Session, _CaptureState], FrameVerdict | None],
            ...,
        ],
    ) -> None:
        for stage in stages:
            self._set_primary(stage(self.fd, self.steps, self.session, self.state))

    def _set_primary(self, verdict: FrameVerdict | None) -> None:
        if verdict is not None:
            self.primary = verdict
            raise V4l2SessionError(verdict)


class V4l2SessionError(RuntimeError):
    """A typed profile/setup result from a prepared session stage."""

    def __init__(self, verdict: FrameVerdict) -> None:
        super().__init__(verdict.detail)
        self.verdict = verdict


def _format_refusal(
    buf_type: int, negotiated: PixelFormat, profile: CaptureProfile = DEPTH_PROFILE
) -> str | None:
    """Why the negotiated format is not safe to stream, or ``None``.

    Every one of these is a refusal rather than an adaptation: a profile has
    exactly one mode, and a driver that answered with a different one has not
    given it what it asked for.

    The four checks here are the ones that hold for any profile - the buffer
    type, the fourcc, the frame size and the per-buffer ceiling this run will
    map. Everything that depends on how the payload is laid out belongs to the
    profile and is asked for last, so a raster profile can insist on its row
    stride without a compressed one inheriting a rule that means nothing to it.
    """
    if buf_type != V4L2_BUF_TYPE_VIDEO_CAPTURE:
        return f"the driver answered with buffer type {buf_type}, not video capture"
    if negotiated.pixelformat != profile.pixelformat:
        return (
            f"the driver answered with {negotiated.fourcc!r}, not the requested {profile.fourcc!r}"
        )
    if negotiated.width != profile.width or negotiated.height != profile.height:
        return (
            f"the driver answered with {negotiated.width}x{negotiated.height}, not the "
            f"requested {profile.size_text}"
        )
    if negotiated.sizeimage > profile.max_buffer_bytes:
        return (
            f"sizeimage {negotiated.sizeimage} is past the {profile.max_buffer_bytes} B "
            "per-buffer ceiling this command will map"
        )
    return profile.format_check(profile, negotiated)


def _mapping_refusal(
    info: BufferInfo,
    expected_index: int,
    running_total: int,
    profile: CaptureProfile = DEPTH_PROFILE,
) -> str | None:
    """Why a queried buffer must not be mapped, or ``None`` if it may be."""
    if info.index != expected_index:
        return f"QUERYBUF({expected_index}) answered for buffer {info.index}"
    if info.memory != V4L2_MEMORY_MMAP:
        return f"buffer {info.index} reports memory type {info.memory}, not MMAP"
    if info.length == 0:
        return f"buffer {info.index} has zero length"
    if info.length > profile.max_buffer_bytes:
        return (
            f"buffer {info.index} is {info.length} B, past the {profile.max_buffer_bytes} B "
            "per-buffer ceiling"
        )
    if running_total > profile.max_total_mapped_bytes:
        return (
            f"mapping buffer {info.index} would take the total to {running_total} B, past "
            f"the {profile.max_total_mapped_bytes} B ceiling"
        )
    if info.offset % mmap_mod.ALLOCATIONGRANULARITY:
        return (
            f"buffer {info.index} offset {info.offset} is not a multiple of the "
            f"{mmap_mod.ALLOCATIONGRANULARITY} B page size and cannot be mapped"
        )
    return None


_IOCTL_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {value: name for name, value in ALLOWED_IOCTLS.items()}
)


def _ioctl_name(command: int) -> str:
    return _IOCTL_NAMES.get(command, f"ioctl 0x{command:08x}")


def _absorbed(signums: list[int]) -> str:
    """How the report names the signals cleanup refused to be aborted by."""
    names = ", ".join(dict.fromkeys(signal.Signals(num).name for num in signums))
    return f"{names} arrived during cleanup and was ignored so it could finish"
