"""A bounded, dependency-free structural inspector for JPEG byte payloads.

This module never decodes a single pixel. What it answers is a narrower and
cheaper question: does this byte string have the *shape* of a JPEG - a start
of image, a sequence of well-formed marker segments, a frame header naming a
size, an end of image - or does it not? That is enough to judge a captured
MJPEG frame (for example one dequeued by :mod:`scanner3d.v4l2_capture` in a
mode that yields compressed frames) without linking an image library into a
process that also talks to raw USB descriptors and V4L2 ioctls.

**Bounded on purpose.** A byte string handed to this module did not
necessarily come from a codec that promises anything. It could be truncated,
it could be adversarial, it could be a raw depth buffer misidentified as
compressed. Every loop below is walked forward by a strictly positive amount
computed from bytes already read, and is stopped by the payload length before
it is stopped by any marker being seen, so there is no input - well-formed,
truncated, or hostile - that makes any loop here fail to terminate.
:data:`MAX_INSPECT_BYTES` puts a second, blunter bound in front of all of that:
a payload larger than 2 MiB - several times an actual MJPEG frame at the
resolutions this repository captures - is refused before a single byte past
its length is looked at, so the cost of inspecting an oversize payload can
never exceed the cost of measuring its length.

**Segment discovery only, never content.** For frame-header (SOF) markers this
module unpacks the five fixed bytes that spell out precision, dimensions and
component count, because those bytes are declared width and height, not
decoded pixels, and are exactly what a caller needs to check a capture against
an expected mode. No table (DHT, DQT), no scan data, and no metadata (APPn,
COM) payload is ever interpreted; those segments are only located and sized.

The parser never raises for anything a byte string can contain. A malformed,
truncated, or merely uninteresting payload is a :class:`JpegReport` with a
:class:`JpegStatus` naming why, not an exception - the same discipline this
package uses everywhere evidence is preserved rather than aborted on.
:func:`inspect_jpeg` raises only for a programming error, such as a caller
passing a non-positive ``ceiling``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

#: Hard ceiling on how many bytes this module will ever parse. Two mebibytes
#: is several times the size of an actual compressed frame at any resolution
#: this repository's cameras capture; a payload past this bound is reported as
#: oversize and is not walked at all, so a caller cannot be made to pay parsing
#: cost proportional to an attacker- or bug-supplied length.
MAX_INSPECT_BYTES: Final = 2 * 1024 * 1024

#: Hard ceiling on how many segments are recorded in a report. A well-formed
#: JPEG from a real encoder has a few dozen segments at most; a payload that
#: manages to declare hundreds of zero-content segments is not evidence of a
#: bigger picture, it is evidence of a hostile or corrupt byte string, and
#: recording it without bound would make the *report* itself unbounded even
#: though the byte walk that produced it is not.
MAX_SEGMENTS: Final = 512

#: Start Of Image. The two bytes every JPEG payload this module accepts must
#: begin with.
_SOI: Final = 0xD8

#: End Of Image. The marker that must end a well-formed payload.
_EOI: Final = 0xD9

#: Temporary marker (TEM). Standalone: never followed by a length.
_TEM: Final = 0x01

#: Start Of Scan. What follows this marker's segment is entropy-coded data,
#: not another marker segment, until a real marker interrupts it.
_SOS: Final = 0xDA

#: Restart markers RST0-RST7. Standalone, and legal both between marker
#: segments and inside entropy-coded scan data, where they do not end the scan.
_RST_LOW: Final = 0xD0
_RST_HIGH: Final = 0xD7

#: SOF (Start Of Frame) markers that this module treats as carrying an actual
#: frame header: precision, height, width, component count. The C0-CF range
#: has three deliberate holes that are *not* frame headers and must never be
#: parsed as one: 0xC4 is DHT (a Huffman table), 0xC8 is JPG (reserved for an
#: extension JPEG never standardised for interchange), and 0xCC is DAC
#: (arithmetic-coding conditioning). Treating any of those three as a SOF would
#: read table or parameter bytes as if they were width and height.
_SOF_MARKERS: Final = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)

#: SOF markers whose encoding is progressive (spectral selection and/or
#: successive approximation, decoded in multiple passes) rather than a single
#: sequential pass over the image.
_PROGRESSIVE_SOF_MARKERS: Final = frozenset({0xC2, 0xC6, 0xCA, 0xCE})

#: Markers with no following length: they carry no segment body at all.
_STANDALONE_MARKERS: Final = frozenset({_SOI, _EOI, _TEM}) | frozenset(
    range(_RST_LOW, _RST_HIGH + 1)
)

#: Human-readable names for markers this module has a specific reason to name,
#: used to build :class:`JpegSegment` and :class:`JpegFrame` reports. Anything
#: not listed here is reported as ``"unknown (0xXX)"`` rather than guessed at.
_MARKER_NAMES: Final = MappingProxyType(
    {
        0xD8: "SOI",
        0xD9: "EOI",
        0x01: "TEM",
        0xD0: "RST0",
        0xD1: "RST1",
        0xD2: "RST2",
        0xD3: "RST3",
        0xD4: "RST4",
        0xD5: "RST5",
        0xD6: "RST6",
        0xD7: "RST7",
        0xC0: "SOF0",
        0xC1: "SOF1",
        0xC2: "SOF2",
        0xC3: "SOF3",
        0xC4: "DHT",
        0xC5: "SOF5",
        0xC6: "SOF6",
        0xC7: "SOF7",
        0xC8: "JPG",
        0xC9: "SOF9",
        0xCA: "SOF10",
        0xCB: "SOF11",
        0xCC: "DAC",
        0xCD: "SOF13",
        0xCE: "SOF14",
        0xCF: "SOF15",
        0xDA: "SOS",
        0xDB: "DQT",
        0xDC: "DNL",
        0xDD: "DRI",
        0xDE: "DHP",
        0xDF: "EXP",
        0xFE: "COM",
        **{0xE0 + n: f"APP{n}" for n in range(16)},
    }
)

#: Smallest possible declared segment length: the two length bytes themselves,
#: describing a body of zero bytes.
_MIN_SEGMENT_LENGTH: Final = 2

#: A SOF body's five fixed bytes (precision, height x2, width x2, component
#: count) plus the two length bytes, before any per-component bytes. Anything
#: shorter cannot carry a frame header at all.
_MIN_SOF_LENGTH: Final = 8

#: Bytes each SOF component descriptor occupies (id, sampling, quant table).
_SOF_BYTES_PER_COMPONENT: Final = 3


def _marker_name(marker: int) -> str:
    """Human-readable name for a marker's second byte, e.g. ``0xC0`` -> ``"SOF0"``."""
    return _MARKER_NAMES.get(marker, f"unknown (0x{marker:02X})")


class JpegStatus(enum.StrEnum):
    """The one outcome a report settles on, in the fixed precedence order.

    Earlier members are checked first; the first that applies wins, so (for
    example) a truncated segment is reported as ``TRUNCATED`` even though the
    same payload would also lack an ``EOI``.
    """

    EMPTY = "empty"
    OVERSIZE = "oversize"
    MISSING_SOI = "missing_soi"
    TRUNCATED = "truncated"
    MALFORMED = "malformed"
    MISSING_SOF = "missing_sof"
    MISSING_EOI = "missing_eoi"
    TRAILING_DATA = "trailing_data"
    WELL_FORMED = "well_formed"


@dataclass(frozen=True, slots=True)
class JpegSegment:
    """One marker segment as located by the walk, body never interpreted."""

    #: The marker's second byte, e.g. ``0xC0`` for SOF0. The leading ``0xFF``
    #: is not stored: every marker has it by construction.
    marker: int
    #: Human-readable name, e.g. ``"SOF0"``, ``"APP0"``, or ``"unknown (0x??)"``
    #: for a marker this module does not specifically name.
    name: str
    #: Offset of the segment's leading ``0xFF`` within the payload.
    offset: int
    #: The declared segment length, including its own two length bytes, or
    #: ``None`` for a standalone marker (SOI, EOI, TEM, RSTn) that has none.
    length: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "marker": f"0x{self.marker:02X}",
            "name": self.name,
            "offset": self.offset,
            "length": self.length,
        }


@dataclass(frozen=True, slots=True)
class JpegFrame:
    """The frame header a supported SOF marker declared, read not decoded."""

    #: The SOF marker's second byte, e.g. ``0xC0``.
    marker: int
    #: Human-readable marker name, e.g. ``"SOF0"``.
    marker_name: str
    #: Sample precision in bits per component, as declared - not verified
    #: against any table, since no table is parsed.
    precision: int
    width: int
    height: int
    components: int
    #: Whether this SOF marker encodes a progressive frame (0xC2, 0xC6, 0xCA,
    #: 0xCE) rather than a single sequential pass.
    progressive: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "marker": f"0x{self.marker:02X}",
            "marker_name": self.marker_name,
            "precision": self.precision,
            "width": self.width,
            "height": self.height,
            "components": self.components,
            "progressive": self.progressive,
        }


@dataclass(frozen=True, slots=True)
class JpegReport:
    """The complete structural verdict on one payload."""

    status: JpegStatus
    #: One sentence naming exactly what was and was not found. Never
    #: contradicts ``status``, and never claims more than the walk actually
    #: established.
    detail: str
    byte_count: int
    #: How many bytes the walk actually consumed. Equal to ``byte_count`` for
    #: a well-formed payload; smaller when the walk stopped early (oversize,
    #: missing SOI, or a truncation/malformation partway through).
    inspected_bytes: int
    #: The ceiling this report was produced under, so a caller reading only
    #: the report can tell an oversize verdict apart from one with room to
    #: spare.
    ceiling_bytes: int
    has_soi: bool
    has_eoi: bool
    #: Whether the payload's final two bytes are the EOI marker - false both
    #: when there is no EOI and when there is trailing data after one.
    eoi_terminal: bool
    #: Bytes remaining in the payload after an EOI was found. Zero unless an
    #: EOI was found and it was not the last two bytes.
    trailing_bytes: int
    frame: JpegFrame | None
    segments: tuple[JpegSegment, ...]
    scan_count: int
    restart_marker_count: int
    stuffed_byte_count: int

    @property
    def well_formed(self) -> bool:
        """Shorthand for ``status is JpegStatus.WELL_FORMED``."""
        return self.status is JpegStatus.WELL_FORMED

    def matches(self, width: int, height: int) -> bool:
        """Whether a frame header was found and declares exactly this size."""
        return self.frame is not None and self.frame.width == width and self.frame.height == height

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "detail": self.detail,
            "byte_count": self.byte_count,
            "inspected_bytes": self.inspected_bytes,
            "ceiling_bytes": self.ceiling_bytes,
            "has_soi": self.has_soi,
            "has_eoi": self.has_eoi,
            "eoi_terminal": self.eoi_terminal,
            "trailing_bytes": self.trailing_bytes,
            "frame": self.frame.as_dict() if self.frame is not None else None,
            "segments": [s.as_dict() for s in self.segments],
            "scan_count": self.scan_count,
            "restart_marker_count": self.restart_marker_count,
            "stuffed_byte_count": self.stuffed_byte_count,
            "well_formed": self.well_formed,
        }


class _Truncated(Exception):
    """Internal signal: the walk ran past the end of the payload.

    Raised only inside :func:`_walk` and caught there; never escapes
    :func:`inspect_jpeg`. Using an exception for this one internal control
    transfer keeps the many call sites that can discover a truncation (a
    segment length, a SOF header, a scan) from each having to thread a
    special return value back through the walk loop.
    """


class _Malformed(Exception):
    """Internal signal: a structural violation that is not a truncation.

    See :class:`_Truncated`; the same containment applies.
    """


@dataclass(slots=True)
class _Walk:
    """Mutable accumulator for one call to :func:`_walk`."""

    payload: bytes
    limit: int
    segments: list[JpegSegment]
    elided_segments: int = 0
    frame: JpegFrame | None = None
    scan_count: int = 0
    restart_marker_count: int = 0
    stuffed_byte_count: int = 0
    has_eoi: bool = False
    eoi_offset: int = -1
    #: How far the walk actually got. Reported even when the walk ended in a
    #: refusal, because "it was structurally fine until byte N" is evidence a
    #: truncated capture is otherwise unable to give.
    cursor: int = 0


def _skip_fill_bytes(payload: bytes, pos: int, limit: int) -> int:
    """Advance past a run of ``0xFF`` fill bytes, bounded by ``limit``.

    A ``0xFF`` immediately before a marker byte is legal padding (T.81
    section B.1.1.5): any number of them may precede the marker that actually
    matters. Used both between marker segments and inside entropy-coded scan
    data (:func:`_walk_scan`); in the latter case the byte immediately after
    the run still needs its own interpretation - ``0x00`` for a stuffed
    literal, a restart marker, or a real marker ending the scan.
    """
    while pos < limit and payload[pos] == 0xFF:
        pos += 1
    return pos


def _parse_sof(walk: _Walk, marker: int, offset: int, length: int, body_start: int) -> JpegFrame:
    """Decode a SOF segment's fixed header. Raises :class:`_Malformed` if it cannot fit."""
    if length < _MIN_SOF_LENGTH:
        raise _Malformed(f"SOF segment at offset {offset} has length {length}, needs at least 8")
    payload = walk.payload
    precision = payload[body_start]
    height = (payload[body_start + 1] << 8) | payload[body_start + 2]
    width = (payload[body_start + 3] << 8) | payload[body_start + 4]
    components = payload[body_start + 5]
    expected_length = _MIN_SOF_LENGTH + _SOF_BYTES_PER_COMPONENT * components
    if expected_length != length:
        raise _Malformed(
            f"SOF segment at offset {offset} declares {components} components, which needs "
            f"length {expected_length}, but the segment length is {length}"
        )
    return JpegFrame(
        marker=marker,
        marker_name=_marker_name(marker),
        precision=precision,
        width=width,
        height=height,
        components=components,
        progressive=marker in _PROGRESSIVE_SOF_MARKERS,
    )


def _walk_scan(walk: _Walk, pos: int) -> int:
    """Consume entropy-coded scan data starting after a SOS segment's header.

    Returns the offset of the marker that ended the scan (a real marker other
    than a restart). Byte stuffing (``0xFF 0x00``) and restart markers
    (``0xFFD0``-``0xFFD7``) are consumed as scan data, counted, and do not end
    the scan. Every step advances ``pos`` by at least one byte, and the loop
    is bounded by ``walk.limit``, so running off the end raises
    :class:`_Truncated` rather than looping.
    """
    payload = walk.payload
    limit = walk.limit
    while True:
        if pos >= limit:
            raise _Truncated("entropy-coded scan data ran past the end of the payload")
        if payload[pos] != 0xFF:
            pos += 1
            continue
        # payload[pos] == 0xFF: could be a fill byte run, a stuffed literal,
        # a restart marker, or the marker that ends the scan.
        run_end = _skip_fill_bytes(payload, pos, limit)
        if run_end >= limit:
            raise _Truncated("entropy-coded scan data ran past the end of the payload")
        next_byte = payload[run_end]
        if next_byte == 0x00:
            walk.stuffed_byte_count += 1
            pos = run_end + 1
            continue
        if _RST_LOW <= next_byte <= _RST_HIGH:
            walk.restart_marker_count += 1
            pos = run_end + 1
            continue
        return run_end - 1
    # Unreachable: every branch above either raises, continues with strictly
    # increasing pos, or returns.


def _walk(walk: _Walk) -> _Walk:
    """Walk marker segments from offset 0, recording evidence as it goes.

    Bounded strictly by ``walk.limit`` (already the smaller of the payload
    length and the ceiling): every branch either raises :class:`_Truncated` or
    :class:`_Malformed`, or advances the cursor by a strictly positive amount
    derived from bytes already read, so the loop cannot fail to terminate.

    ``walk`` is supplied by the caller rather than created here so that a walk
    which ends in a refusal still hands back everything it had already
    established - the segments seen, the frame header if one was reached, and
    how far it got - instead of throwing that evidence away with the
    exception.
    """
    payload, limit = walk.payload, walk.limit
    if payload[0] != 0xFF or payload[1] != _SOI:
        raise _Malformed("payload does not begin with FFD8 (SOI)")
    walk.segments.append(JpegSegment(marker=_SOI, name=_marker_name(_SOI), offset=0, length=None))
    pos = 2
    walk.cursor = pos
    while pos < limit:
        walk.cursor = pos
        if payload[pos] != 0xFF:
            raise _Malformed(f"expected a marker's leading 0xFF at offset {pos}")
        marker_pos = _skip_fill_bytes(payload, pos, limit)
        if marker_pos >= limit:
            raise _Truncated("payload ended inside a run of fill bytes")
        marker = payload[marker_pos]
        offset = marker_pos - 1
        if marker == _EOI:
            walk.has_eoi = True
            walk.eoi_offset = offset
            _record_segment(
                walk,
                JpegSegment(marker=marker, name=_marker_name(marker), offset=offset, length=None),
            )
            pos = marker_pos + 1
            walk.cursor = pos
            break
        if marker in _STANDALONE_MARKERS:
            _record_segment(
                walk,
                JpegSegment(marker=marker, name=_marker_name(marker), offset=offset, length=None),
            )
            pos = marker_pos + 1
            continue
        length_start = marker_pos + 1
        if length_start + 2 > limit:
            raise _Truncated(f"segment at offset {offset} has no room for its length bytes")
        length = (payload[length_start] << 8) | payload[length_start + 1]
        if length < _MIN_SEGMENT_LENGTH:
            raise _Malformed(
                f"segment at offset {offset} declares length {length}, below the minimum of 2"
            )
        segment_end = length_start + length
        if segment_end > limit:
            raise _Truncated(
                f"segment at offset {offset} declares length {length}, past the end of the payload"
            )
        _record_segment(
            walk,
            JpegSegment(marker=marker, name=_marker_name(marker), offset=offset, length=length),
        )
        if marker in _SOF_MARKERS and walk.frame is None:
            walk.frame = _parse_sof(walk, marker, offset, length, length_start + 2)
        if marker == _SOS:
            walk.scan_count += 1
            pos = _walk_scan(walk, segment_end)
            continue
        pos = segment_end
    walk.cursor = min(pos, limit)
    return walk


def _record_segment(walk: _Walk, segment: JpegSegment) -> None:
    """Append ``segment`` unless :data:`MAX_SEGMENTS` has already been reached."""
    if len(walk.segments) < MAX_SEGMENTS:
        walk.segments.append(segment)
    else:
        walk.elided_segments += 1


def _refused_report(status: JpegStatus, detail: str, walk: _Walk, ceiling: int) -> JpegReport:
    """A report for a walk that stopped, carrying everything it had reached."""
    return JpegReport(
        status=status,
        detail=detail,
        byte_count=len(walk.payload),
        inspected_bytes=walk.cursor,
        ceiling_bytes=ceiling,
        has_soi=True,
        has_eoi=walk.has_eoi,
        eoi_terminal=False,
        trailing_bytes=0,
        frame=walk.frame,
        segments=tuple(walk.segments),
        scan_count=walk.scan_count,
        restart_marker_count=walk.restart_marker_count,
        stuffed_byte_count=walk.stuffed_byte_count,
    )


def _empty_report(ceiling: int) -> JpegReport:
    return JpegReport(
        status=JpegStatus.EMPTY,
        detail="payload is empty",
        byte_count=0,
        inspected_bytes=0,
        ceiling_bytes=ceiling,
        has_soi=False,
        has_eoi=False,
        eoi_terminal=False,
        trailing_bytes=0,
        frame=None,
        segments=(),
        scan_count=0,
        restart_marker_count=0,
        stuffed_byte_count=0,
    )


def _oversize_report(byte_count: int, ceiling: int) -> JpegReport:
    return JpegReport(
        status=JpegStatus.OVERSIZE,
        detail=(
            f"payload is {byte_count} bytes, past the {ceiling}-byte inspection ceiling; not parsed"
        ),
        byte_count=byte_count,
        inspected_bytes=0,
        ceiling_bytes=ceiling,
        has_soi=False,
        has_eoi=False,
        eoi_terminal=False,
        trailing_bytes=0,
        frame=None,
        segments=(),
        scan_count=0,
        restart_marker_count=0,
        stuffed_byte_count=0,
    )


def inspect_jpeg(payload: bytes, *, ceiling: int = MAX_INSPECT_BYTES) -> JpegReport:
    """Structurally inspect ``payload`` as a candidate JPEG, without decoding it.

    Never raises for anything the payload itself contains: a malformed,
    truncated or empty byte string produces a :class:`JpegReport` naming why,
    per the precedence documented on :class:`JpegStatus`. The only way to
    raise is a programming error - ``ceiling`` not being strictly positive.
    """
    if ceiling <= 0:
        raise ValueError(f"ceiling must be positive, got {ceiling}")
    byte_count = len(payload)
    if byte_count == 0:
        return _empty_report(ceiling)
    if byte_count > ceiling:
        return _oversize_report(byte_count, ceiling)

    walk = _Walk(payload=payload, limit=byte_count, segments=[])
    try:
        if byte_count < 2 or payload[0] != 0xFF or payload[1] != _SOI:
            return JpegReport(
                status=JpegStatus.MISSING_SOI,
                detail="payload does not begin with FFD8 (SOI)",
                byte_count=byte_count,
                inspected_bytes=min(byte_count, 1),
                ceiling_bytes=ceiling,
                has_soi=False,
                has_eoi=False,
                eoi_terminal=False,
                trailing_bytes=0,
                frame=None,
                segments=(),
                scan_count=0,
                restart_marker_count=0,
                stuffed_byte_count=0,
            )
        walk = _walk(walk)
    except (_Truncated, _Malformed) as exc:
        # Whatever the walk had already established stands: a truncated frame
        # that got as far as its SOF still tells a caller what size it claimed
        # to be, and discarding that with the exception would throw away the
        # most useful evidence a broken capture can offer.
        return _refused_report(
            JpegStatus.TRUNCATED if isinstance(exc, _Truncated) else JpegStatus.MALFORMED,
            str(exc),
            walk,
            ceiling,
        )

    trailing_bytes = byte_count - (walk.eoi_offset + 2) if walk.has_eoi else 0
    eoi_terminal = walk.has_eoi and trailing_bytes == 0
    segments = tuple(walk.segments)
    elided_note = (
        f"; {walk.elided_segments} further segments elided past MAX_SEGMENTS ({MAX_SEGMENTS})"
    )

    if walk.frame is None:
        status = JpegStatus.MISSING_SOF
        detail = (
            "SOI found and segments parsed, but no supported SOF (frame header) marker was found"
        )
    elif not walk.has_eoi:
        status = JpegStatus.MISSING_EOI
        detail = (
            f"SOI and a {walk.frame.marker_name} frame header were found, "
            "but no EOI marker was found"
        )
    elif not eoi_terminal:
        status = JpegStatus.TRAILING_DATA
        detail = f"EOI found at offset {walk.eoi_offset}, but {trailing_bytes} byte(s) follow it"
    else:
        status = JpegStatus.WELL_FORMED
        detail = (
            f"well-formed JPEG: {walk.frame.marker_name} frame "
            f"{walk.frame.width}x{walk.frame.height}, {walk.frame.components} component(s), "
            f"SOI through EOI with {len(segments)} segment(s) recorded"
        )
    if walk.elided_segments:
        detail += elided_note

    return JpegReport(
        status=status,
        detail=detail,
        byte_count=byte_count,
        inspected_bytes=byte_count,
        ceiling_bytes=ceiling,
        has_soi=True,
        has_eoi=walk.has_eoi,
        eoi_terminal=eoi_terminal,
        trailing_bytes=trailing_bytes,
        frame=walk.frame,
        segments=segments,
        scan_count=walk.scan_count,
        restart_marker_count=walk.restart_marker_count,
        stuffed_byte_count=walk.stuffed_byte_count,
    )
