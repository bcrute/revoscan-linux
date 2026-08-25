"""The fixed RGB profile for the one bounded V4L2/MMAP capture engine.

This module adds **no lifecycle**. Every ioctl, the single poll, the single
dequeue, the mapping ceilings and the shielded cleanup are
:class:`~scanner3d.v4l2_capture.V4l2FrameBackend`'s, unchanged and shared with
the accepted depth capture. What lives here is the other half of the pair the
engine was refactored around: one immutable
:class:`~scanner3d.v4l2_capture.CaptureProfile` describing the MINI's RGB
camera, and the three rules that profile owns - which negotiated formats it
will accept, how it judges the bytes that come back, and how it writes them
down.

What is fixed at import time, and is not a parameter of anything:

* the **role**, ``rgb`` - resolved from the USB interface string descriptor
  (``RGBCam<serial>``) plus the V4L2 node index, exactly as the depth profile
  resolves its own. ``/dev/video2`` is never assumed and no caller supplies a
  node path: the number is assigned in probe order and moves under replug,
  reboot and other UVC devices;
* the **mode**, :data:`RGB_FOURCC` ``MJPG`` at :data:`RGB_FRAME_WIDTH` x
  :data:`RGB_FRAME_HEIGHT`, the only mode this unit's RGB VideoStreaming
  interface advertises. Its recorded ``dwDefaultFrameInterval`` is 333333
  (:data:`RGB_DESCRIPTOR_FRAME_RATE_HZ` fps) and its recorded
  ``dwMaxVideoFrameBufferSize`` is
  :data:`RGB_DESCRIPTOR_FRAME_BUFFER_BYTES`. Both are quoted from the
  descriptor as configuration, never as measurement: this profile issues no
  ``VIDIOC_S_PARM`` and times no interval, so nothing here claims the
  negotiated frame rate was observed;
* the **bounds**: :data:`RGB_MAX_BUFFER_BYTES` per mapped buffer and per
  artifact, :data:`RGB_MAX_TOTAL_MAPPED_BYTES` mapped in total, four buffers
  requested and two to four accepted, and one poll carrying the whole
  :data:`RGB_CAPTURE_DEADLINE_S` budget.

**MJPEG is a bitstream, not a raster, and this profile says so.** The depth
profile can insist on a row stride and an exact frame size because a ``Y16 ``
frame has both. A compressed frame has neither: ``uvcvideo`` reports
``bytesperline`` zero for MJPEG, and ``bytesused`` is whatever the encoder
produced for that one frame. So the size-shaped refusals do not apply, and
what takes their place is a *structural* question asked of the payload itself
by :mod:`scanner3d.jpeg_inspect` - does it start with SOI, are its marker
segments inside their own declared lengths, does it carry a frame header, and
does it end with EOI? That inspector reads no pixel, links no codec, and is
handed this profile's own :data:`RGB_MAX_RAW_ARTIFACT_BYTES` as its ceiling, so
the bytes that can be mapped, written and walked are bounded by one number
rather than three.

**The ceilings are a measured driver bound, not a guess about JPEG size.**
:data:`RGB_MAX_BUFFER_BYTES` is exactly the ``sizeimage`` ``uvcvideo`` answered
``VIDIOC_S_FMT`` with for this mode on this unit
(:data:`RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES`, 8192000 B), which is four times
what the mode's own UVC descriptor asks for. The earlier 2 MiB ceiling was
derived from that descriptor and was therefore below the buffer the driver
intends to allocate, so the run refused itself before ``VIDIOC_REQBUFS``. Raising
it changes what may be *mapped* and nothing else: the descriptor value is still
recorded verbatim, no claim is made that a compressed frame will fill 8192000
bytes, and :func:`preserve_rgb_frame` still writes exactly ``bytesused`` bytes
and never one byte of mapping padding.

**Every dequeued byte is kept, whatever it turns out to be.** A payload that
fails the structural check is still written to disk with its SHA-256 and its
marker evidence, because a malformed frame is evidence about the device and
throwing it away would leave nothing to look at. ``complete`` is reserved for
a payload that is well-bounded *and* whose SOF says exactly
1280x800; empty, oversize, malformed/truncated and dimension-mismatch are four
distinguishable outcomes, not one.

Nothing here decodes, resizes, converts or colour-corrects an image, and
nothing here claims colour accuracy, white balance, exposure correctness or
any calibrated relationship to the depth camera. The bytes are the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scanner3d.jpeg_inspect import MAX_INSPECT_BYTES, JpegStatus, inspect_jpeg

# ``_write_new_file`` is imported across the module boundary on purpose: this
# file and :mod:`scanner3d.v4l2_capture` are two halves of one engine, and a
# second exclusive-write implementation is exactly the duplication the shared
# engine exists to prevent.
from scanner3d.v4l2_capture import (
    ArtifactRecord,
    BufferInfo,
    CaptureProfile,
    CaptureStatus,
    FrameArtifact,
    FrameVerdict,
    PixelFormat,
    V4l2FrameBackend,
    _write_new_file,
    sha256_of,
)

# --- the one RGB mode ---------------------------------------------------------

#: The logical camera, resolved from the ``RGBCam<serial>`` interface string
#: descriptor. Never a ``videoN`` number and never a caller-supplied path.
RGB_ROLE: Final = "rgb"

#: ``V4L2_PIX_FMT_MJPEG``. Four characters exactly, and unlike ``Y16 `` this
#: one has no trailing space.
RGB_FOURCC: Final = "MJPG"

RGB_FRAME_WIDTH: Final = 1280
RGB_FRAME_HEIGHT: Final = 800

#: The descriptor default for this mode: ``dwDefaultFrameInterval`` 333333
#: hundred-nanosecond units. Recorded as evidence of what the device advertises;
#: this profile never issues ``VIDIOC_S_PARM`` and never times a frame, so this
#: is not a measurement of the interval that was actually negotiated.
RGB_DESCRIPTOR_FRAME_RATE_HZ: Final = 30.0

#: ``dwMaxVideoFrameBufferSize`` from the recorded VS_FRAME_MJPEG descriptor of
#: this unit - what the device advertises, never what this profile permits and
#: never what the driver was observed to do. It is kept as evidence precisely
#: because it turned out *not* to predict the kernel's own allocation: see
#: :data:`RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES`, which is four times this and is
#: what the ceilings below are actually set from.
RGB_DESCRIPTOR_FRAME_BUFFER_BYTES: Final = 2048000

# --- the bounds ---------------------------------------------------------------

#: How many MMAP buffers are requested. The kernel may return fewer.
RGB_REQUESTED_BUFFERS: Final = 4

#: The fewest returned buffers this profile will stream with.
RGB_MIN_ACCEPTED_BUFFERS: Final = 2

#: The ``sizeimage`` ``uvcvideo`` actually reported for this exact mode on this
#: exact unit, measured in scanner-010 by reading back ``VIDIOC_S_FMT``'s own
#: answer: **8192000 B**, four times the mode's descriptor
#: :data:`RGB_DESCRIPTOR_FRAME_BUFFER_BYTES`. It is quoted here for one reason
#: only - it is the size of the buffer the Linux driver says it will allocate,
#: so it is the size this profile must be willing to map if it is ever to reach
#: ``VIDIOC_STREAMON`` at all.
#:
#: Three things it is emphatically *not*. It is not a claim that a compressed
#: 1280x800 JPEG will occupy 8192000 bytes - a real one occupies a small
#: fraction of that, and only ``bytesused`` says how much. It is not a claim
#: that the UVC descriptor is wrong: 2048000 remains exactly what the device
#: advertises, and the gap between the two is a fact about ``uvcvideo``'s
#: allocation, not an error in either number. And it is not a measurement that
#: generalises - it was read from one driver on one host for one mode, and any
#: other mode, kernel or unit would have to be measured again.
RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES: Final = 8192000

#: Hard ceiling on a single buffer's mapped length, checked against what
#: ``VIDIOC_QUERYBUF`` reports before ``mmap`` is called - and, one step
#: earlier, against the ``sizeimage`` the driver answered ``VIDIOC_S_FMT``
#: with. Set to exactly :data:`RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES` and not one
#: byte above it: a measured allocation bound is a number this profile can
#: defend, and rounding it up to a tidy power of two would be inventing head
#: room nothing observed.
RGB_MAX_BUFFER_BYTES: Final = RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES

#: Hard ceiling on the total bytes mapped across every buffer: four of the
#: above, 32768000 B, which is the largest mapping :data:`RGB_REQUESTED_BUFFERS`
#: legally-sized buffers can ever add up to and therefore the whole of this
#: command's address-space exposure.
RGB_MAX_TOTAL_MAPPED_BYTES: Final = RGB_MAX_BUFFER_BYTES * RGB_REQUESTED_BUFFERS

#: The whole wait budget, in seconds, measured on a monotonic clock from the
#: moment ``VIDIOC_STREAMON`` returned. One ``poll`` call carries all of it.
RGB_CAPTURE_DEADLINE_S: Final = 8.0

#: Ceiling on the raw artifact. Deliberately the same number as a single
#: buffer, and the ceiling handed to :mod:`scanner3d.jpeg_inspect` for this
#: profile, so the bytes that can be mapped, the bytes that can be written and
#: the bytes that can be walked are bounded by one value rather than by three
#: that could drift apart. Note that this bounds what is *possible*, never what
#: is expected: only ``bytesused`` decides how many bytes are copied, and a real
#: JPEG will land far below this.
RGB_MAX_RAW_ARTIFACT_BYTES: Final = RGB_MAX_BUFFER_BYTES

#: Where the live path writes evidence when the caller names no directory.
RGB_DEFAULT_OUTPUT_DIR: Final = Path("/tmp/scanner3d-rgb-frame")

#: The one file this profile writes. A ``.jpg`` extension because the bytes are
#: a JPEG bitstream verbatim - not because anything here decoded one.
RGB_ARTIFACT_NAME: Final = "rgb-frame.jpg"

#: The unmistakable opt-in. Spelled out in full rather than abbreviated, and
#: distinct from the depth command's flag so neither can arm the other.
RGB_EXECUTE_FLAG: Final = "--execute-rgb-frame-capture"


# --- the three rules this profile owns ----------------------------------------


def _rgb_format_check(profile: CaptureProfile, negotiated: PixelFormat) -> str | None:
    """The compressed-format half of the ``MJPG`` refusals.

    The shared checks have already established the buffer type, the fourcc, the
    frame size and the per-buffer ceiling. What is left that can still make a
    negotiated MJPEG format unusable is a buffer with no room in it: a driver
    reporting ``sizeimage`` zero has described a queue that cannot hold a frame.

    ``bytesperline`` is deliberately *not* refused here whatever it says.
    ``uvcvideo`` reports zero for a compressed format because a bitstream has no
    row stride, and a stride this profile never uses cannot make a payload
    unsafe - the mapping and the artifact are both bounded by byte counts. It is
    recorded in the step log and in the report rather than made a refusal that
    would spend this run's one attempt on a field that means nothing to it.
    """
    if negotiated.sizeimage < 1:
        return (
            f"sizeimage {negotiated.sizeimage} leaves no room for a compressed frame; "
            "the driver has described a queue that cannot hold one"
        )
    return None


def _rgb_frame_judgement(
    profile: CaptureProfile,
    raw: bytes,
    info: BufferInfo,
    negotiated: PixelFormat | None,
) -> FrameVerdict:
    """Whether the dequeued bytes are one whole JPEG of exactly this mode's size.

    Called only once the engine's shared bounds have all held, so ``raw`` is
    already known to be inside the mapping and inside the artifact ceiling. The
    four ways it can still fail are kept apart, because they say different
    things about the device: nothing arrived, too much arrived, something
    arrived that is not a whole JPEG, and a whole JPEG arrived that is not the
    size this mode was negotiated at.

    In every one of those cases the bytes are still preserved by
    :func:`preserve_rgb_frame`; this function decides what the run is *called*,
    not what is kept.
    """
    report = inspect_jpeg(raw, ceiling=profile.max_raw_artifact_bytes)
    evidence: dict[str, object] = {
        "jpeg": report.as_dict(),
        "expected_width": profile.width,
        "expected_height": profile.height,
        "dimensions_match": report.matches(profile.width, profile.height),
        "pixels_decoded": False,
        "colour_corrected": False,
        "resized": False,
        "calibration_applied": False,
    }
    if report.status is JpegStatus.EMPTY:
        return FrameVerdict(
            CaptureStatus.EMPTY_FRAME,
            f"buffer {info.index} was dequeued with bytesused 0: the stream started and a "
            "buffer came back, but it carried no payload at all",
            evidence=evidence,
        )
    if report.status is JpegStatus.OVERSIZE:  # pragma: no cover - engine bound is tighter
        return FrameVerdict(
            CaptureStatus.OVERSIZE_FRAME,
            f"buffer {info.index} carried {len(raw)} bytes, past the "
            f"{profile.max_raw_artifact_bytes} B inspection ceiling",
            evidence=evidence,
        )
    if not report.well_formed:
        return FrameVerdict(
            CaptureStatus.MALFORMED_FRAME,
            f"buffer {info.index} carried {len(raw)} bytes that are not a well-bounded "
            f"JPEG ({report.status}: {report.detail}); the bytes are preserved as raw "
            "evidence but are not a frame",
            evidence=evidence,
        )
    frame = report.frame
    if frame is None or not report.matches(profile.width, profile.height):
        seen = "no frame header at all" if frame is None else f"{frame.width}x{frame.height}"
        header = "SOF" if frame is None else frame.marker_name
        return FrameVerdict(
            CaptureStatus.DIMENSION_MISMATCH,
            f"buffer {info.index} carried a well-bounded JPEG whose {header} "
            f"frame header says {seen}, not the "
            f"{profile.size_text} this mode was negotiated at; the bytes are preserved as "
            "raw evidence but are not a frame of this mode",
            evidence=evidence,
        )
    return FrameVerdict(
        CaptureStatus.COMPLETE,
        f"one well-bounded {profile.fourcc!r} JPEG of {len(raw)} bytes whose frame header "
        f"says {profile.size_text}; no pixel was decoded and no colour claim is made",
        evidence=evidence,
    )


def preserve_rgb_frame(
    raw: bytes, directory: Path, *, ceiling: int = RGB_MAX_RAW_ARTIFACT_BYTES
) -> ArtifactRecord:
    """Write the dequeued bytes, verbatim, to one exclusive ``rgb-frame.jpg``.

    One file, complete or absent: :func:`~scanner3d.v4l2_capture._write_new_file`
    creates it ``O_EXCL | O_NOFOLLOW`` and removes it if the write cannot be
    finished, because a truncated file under a name that promises a frame is
    worse evidence than no file. There is no second, "processed" artifact: this
    profile has nothing to process the bytes into that would not be an
    invention.

    Never raises. An artifact that could not be written is reported as a failed
    artifact, because the caller must still report the frame that arrived and
    must still release the device.
    """
    if not raw:
        return ArtifactRecord.not_needed(
            "the dequeued buffer carried no bytes, so there was nothing to write; an "
            "empty file under a name that promises a frame would be worse than none",
            str(directory),
            ceiling,
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return ArtifactRecord(
            attempted=True,
            ok=False,
            note=f"cannot create the output directory: {exc}",
            directory=str(directory),
            files=(),
            ceiling_bytes=ceiling,
        )

    path = directory / RGB_ARTIFACT_NAME
    digest = sha256_of(raw)
    try:
        _write_new_file(path, raw, ceiling=ceiling)
    except OSError as exc:
        artifact = FrameArtifact(
            kind="mjpeg",
            path=None,
            byte_count=len(raw),
            sha256=digest,
            ok=False,
            detail=f"cannot write {path}: {exc}",
        )
    else:
        artifact = FrameArtifact(
            kind="mjpeg",
            path=str(path),
            byte_count=len(raw),
            sha256=digest,
            ok=True,
            detail=(
                f"the dequeued bytes exactly as the kernel presented them ({len(raw)} B); "
                "no pixel was decoded, nothing was resized and no colour correction, white "
                "balance or calibration was applied"
            ),
        )
    return ArtifactRecord(
        attempted=True,
        ok=artifact.ok,
        note=(
            "the frame and its hash were preserved"
            if artifact.ok
            else "the frame arrived but its evidence was NOT preserved"
        ),
        directory=str(directory),
        files=(artifact,),
        ceiling_bytes=ceiling,
    )


def _rgb_preserve(
    profile: CaptureProfile, raw: bytes, pixel_format: PixelFormat, directory: Path
) -> ArtifactRecord:
    """Adapter matching :data:`~scanner3d.v4l2_capture.FramePreserver`.

    ``pixel_format`` is unused and deliberately so: a compressed payload is not
    described by the negotiated raster geometry, and using it to shape the file
    would be describing the bytes by something other than themselves.
    """
    return preserve_rgb_frame(raw, directory, ceiling=profile.max_raw_artifact_bytes)


# --- the profile --------------------------------------------------------------

#: The one RGB capture. Constructed once, frozen, and never built from caller
#: input: a caller chooses a serial and an output directory and cannot reach,
#: widen or substitute any value below.
RGB_PROFILE: Final = CaptureProfile(
    name="rgb-frame",
    mode_label="v4l2-mmap-rgb-frame",
    role=RGB_ROLE,
    node_kind="capture",
    fourcc=RGB_FOURCC,
    width=RGB_FRAME_WIDTH,
    height=RGB_FRAME_HEIGHT,
    descriptor_frame_rate_hz=RGB_DESCRIPTOR_FRAME_RATE_HZ,
    compressed=True,
    requested_buffers=RGB_REQUESTED_BUFFERS,
    min_accepted_buffers=RGB_MIN_ACCEPTED_BUFFERS,
    max_buffer_bytes=RGB_MAX_BUFFER_BYTES,
    max_total_mapped_bytes=RGB_MAX_TOTAL_MAPPED_BYTES,
    capture_deadline_s=RGB_CAPTURE_DEADLINE_S,
    max_raw_artifact_bytes=RGB_MAX_RAW_ARTIFACT_BYTES,
    default_output_dir=RGB_DEFAULT_OUTPUT_DIR,
    execute_flag=RGB_EXECUTE_FLAG,
    mode_text=f"{RGB_FOURCC!r} {RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT}",
    completeness_text=(
        f"a well-bounded JPEG - SOI, terminal EOI - whose SOF says exactly "
        f"{RGB_FRAME_WIDTH}x{RGB_FRAME_HEIGHT}"
    ),
    disclaimer=(
        "no colour accuracy was claimed: these are the encoder's own MJPEG bytes with no "
        "decode, colour correction, white balance or calibration applied, and no "
        "synchronisation with the depth camera was established or attempted"
    ),
    format_check=_rgb_format_check,
    judge=_rgb_frame_judgement,
    preserve=_rgb_preserve,
    plan_extra=MappingProxyType(
        {
            "descriptor_max_video_frame_buffer_bytes": RGB_DESCRIPTOR_FRAME_BUFFER_BYTES,
            "artifact_name": RGB_ARTIFACT_NAME,
            "measured_driver_sizeimage_bytes": RGB_MEASURED_DRIVER_SIZEIMAGE_BYTES,
            "jpeg_inspection": "structural markers only; no pixel decode",
            "jpeg_inspect_ceiling_bytes": RGB_MAX_RAW_ARTIFACT_BYTES,
            "jpeg_inspect_module_default_ceiling_bytes": MAX_INSPECT_BYTES,
            "pixels_decoded": False,
            "colour_corrected": False,
            "calibration_applied": False,
        }
    ),
)


@dataclass(frozen=True, slots=True)
class V4l2RgbFrameBackend(V4l2FrameBackend):
    """The shared engine bound to :data:`RGB_PROFILE`, and to nothing else.

    Adds no field, no method and no behaviour - the profile default it declares
    *is* the whole difference. Everything the live path does, it does through
    :class:`~scanner3d.v4l2_capture.V4l2FrameBackend`, including the refusal of
    any serial but the one the owner authorized.
    """

    profile: CaptureProfile = RGB_PROFILE
