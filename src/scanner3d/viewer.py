"""A live, false-coloured window over depth frames handed to it - nothing else.

The owner scanning a model truck bed on a turntable currently has no visual
feedback during a capture: no way to tell whether the object is framed,
whether it is clipping off the edge of the sensor, or how much of the frame is
actually returning depth, until the capture is already over. This module is
the fix, and it is deliberately narrow:

* **It never touches hardware.** No device node is opened here, no ``/dev/``
  path is read, no capture loop lives in this file. Every frame this module
  ever sees arrives as bytes handed to :meth:`DepthViewer.show` by a caller
  that owns the actual capture. That separation is what makes the viewer
  testable without a scanner attached, and it matches how the rest of this
  repository keeps arithmetic and hardware access apart.
* **Zero is not a distance.** A raw sample of 0 means the sensor returned no
  depth for that pixel - the same reading :mod:`scanner3d.depth_metric` and
  :mod:`scanner3d.v4l2_capture` use. It is never blended into the near end of
  the depth ramp and never counted as though it measured anything. It is drawn
  instead in a coarse neutral-grey checker no colour on the ramp can produce
  (:data:`DEFAULT_PALETTE`), because the flat black this used to use is also
  what a panel that has stopped drawing looks like - and on 2026-08-28 that is
  exactly how the owner read a frame the sensor had genuinely returned almost
  nothing for. How much is missing, where it is missing, and whether it is
  getting worse are :mod:`scanner3d.return_health`'s, reported by both front
  ends below.
* **A raw code is not a distance either.** Nothing here multiplies a sample by
  any scale, and no physical unit is implied anywhere in this module. The
  colour ramp and the numeric readout both show the sensor's own uint16
  codes, stretched for legibility, and nothing more.
* **Clipping is the loudest thing on screen.** A non-zero return sitting on
  any of the frame's four edges means the object is running off the sensor's
  field of view, which silently ruins a scan the owner cannot fix after the
  fact. :class:`FrameStats` counts it on every frame, and both front ends
  surface it unmissably rather than folding it into the rest of the readout.

There are two front ends, because there are two places the owner can be while
a capture runs. :class:`DepthViewer` is a Tk window for when they are sitting
at the keyboard - lowest latency, nothing to open in a browser.
:class:`ViewerServer` is a local HTTP server for when they are not: standing
at the turntable, watching a phone or tablet on the same machine or network.
Both are driven the same way - a frame handed to their ``show`` method - and
neither reads a device or a capture loop of its own. Everything below
:class:`DepthViewer` up to the web-view section - :func:`frame_stats`,
:func:`depth_to_pixels`, :func:`ppm_bytes`, :func:`png_bytes` - is pure
arithmetic over bytes, shared by both front ends unchanged, and is exercised
in tests with no Tk window and no HTTP server ever created.

What :class:`ViewerServer` is **not**: it has no 3D view, no point cloud, no
frame history and no recording. It shows the single most recent frame handed
to it, false-coloured, with the same readout and clipping warning as the Tk
window, and nothing else. A future point-cloud view is exactly why this is a
browser page and not another Tk widget - WebGL is the only realistic way to
render one at a usable speed from pure-Python code - but no such view exists
yet.
"""

from __future__ import annotations

import contextlib
import http.server
import itertools
import json
import struct
import threading
import time
import tkinter
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.return_health import (
    ReturnHealth,
    ReturnHealthTracker,
    SpatialReturns,
    health_summary,
    spatial_returns,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: Every dict this module emits stamps this.
SCHEMA_VERSION: Final = 1

#: Samples in one frame - derived from the single source of truth in
#: :mod:`scanner3d.v4l2_capture`, the same way :mod:`scanner3d.depth_metric`
#: derives its own copy, rather than a second literal ``256000`` here.
SAMPLE_COUNT: Final = FRAME_WIDTH * FRAME_HEIGHT

#: The whole frame, unpacked in one call.
SAMPLES_FORMAT: Final = f"<{SAMPLE_COUNT}H"


# --- refusals -------------------------------------------------------------


class ViewerError(ScannerError):
    """Base for every refusal this module raises before touching a frame.

    Every subclass is raised before a sample is unpacked or a window is
    touched, so a caught refusal means nothing downstream ran.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class FrameTypeMismatch(ViewerError):
    """``raw`` was not ``bytes``. Refused before ``struct.unpack`` is called.

    A ``bytearray``, a ``str`` or anything else that merely looks byte-like is
    not accepted: this module has exactly one wire format in mind and does
    not coerce toward it.
    """


class FrameSizeMismatch(ViewerError):
    """``raw`` was not exactly one 512000-byte frame. Refused before unpacking.

    Padding or truncating a short or long buffer here would invent pixels the
    sensor never sent, the same reasoning :mod:`scanner3d.depth_metric` uses
    for the same 512000-byte shape.
    """


class InvalidScale(ViewerError):
    """``scale`` was not a positive integer. Refused before a window opens."""


class ViewerWindowClosed(ViewerError):
    """:meth:`DepthViewer.show` was called after :meth:`DepthViewer.close`.

    Refused rather than left to raise a bare :class:`tkinter.TclError` against
    a Tk root that no longer exists, and rather than silently reopening a
    window the caller already asked to close.
    """


class InvalidPort(ViewerError):
    """``port`` was not a valid TCP port number. Refused before a socket opens."""


class ViewerServerNotRunning(ViewerError):
    """A running server's bound port was asked for before :meth:`ViewerServer.start`.

    ``0`` (the default) means "let the OS choose a port", so the real port is
    only known once a socket has actually been bound; asking earlier is
    refused rather than answered with the meaningless ``0`` the caller passed
    in.
    """


def _ensure_frame(raw: bytes) -> None:
    """Raise if ``raw`` is not exactly one 512000-byte depth frame."""
    if not isinstance(raw, bytes):
        raise FrameTypeMismatch(
            f"a depth frame must be bytes; got {type(raw).__name__}. Nothing was read: this "
            "module never coerces its input toward the wire format it expects"
        )
    if len(raw) != EXPECTED_FRAME_BYTES:
        raise FrameSizeMismatch(
            f"a {FRAME_WIDTH}x{FRAME_HEIGHT} depth frame is exactly {EXPECTED_FRAME_BYTES} "
            "bytes (little-endian uint16, one sample per pixel); got "
            f"{len(raw)}. Nothing was unpacked: a buffer of another length is not this mode's "
            "frame"
        )


# --- frame statistics -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameStats:
    """What one frame looks like, without deciding what any code means.

    ``min_code``, ``max_code`` and ``median_code`` are computed over the
    *non-zero* samples only - a zero is an absence of measurement, not a near
    reading, and letting it into these figures would pull every one of them
    toward zero regardless of what the object in frame actually returned. When
    ``returned`` is 0 there is no non-zero sample to report from, and all
    three are 0 as a sentinel rather than a code: check ``returned`` first.

    ``clipped_left/right/top/bottom`` count the non-zero returns sitting on
    each of the frame's four one-pixel-wide edges. A non-zero count on any
    edge means the object is running off the sensor's field of view - the
    single most useful thing this module can tell the owner of a turntable
    capture, because it silently ruins a scan and is otherwise invisible until
    the capture is already over.

    No field here implies a physical unit. These are sensor codes and pixel
    counts, nothing more.
    """

    samples: int
    returned: int
    yield_fraction: float
    min_code: int
    max_code: int
    median_code: int
    clipped_left: int
    clipped_right: int
    clipped_top: int
    clipped_bottom: int
    spatial: SpatialReturns

    @property
    def has_clipping(self) -> bool:
        """Whether any of the frame's four edges carries a non-zero return."""
        return bool(
            self.clipped_left or self.clipped_right or self.clipped_top or self.clipped_bottom
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "samples": self.samples,
            "returned": self.returned,
            "yield_fraction": self.yield_fraction,
            "codes": {
                "min": self.min_code,
                "median": self.median_code,
                "max": self.max_code,
                "note": (
                    "raw sensor codes over non-zero samples only; 0 when returned == 0. No "
                    "physical unit is implied"
                ),
            },
            "clipping": {
                "left": self.clipped_left,
                "right": self.clipped_right,
                "top": self.clipped_top,
                "bottom": self.clipped_bottom,
                "any": self.has_clipping,
            },
            "spatial": self.spatial.as_dict(),
        }


def frame_stats(raw: bytes) -> FrameStats:
    """Compute :class:`FrameStats` for one raw depth frame.

    Refuses (see :func:`_ensure_frame`) before a single sample is unpacked.

    A caller that also wants the frame's colours should call
    :func:`render_frame` instead: it produces both from ONE unpack of the same
    512000 bytes, rather than this function and :func:`depth_to_pixels` each
    unpacking the frame again.
    """
    _ensure_frame(raw)
    return _stats_from_samples(struct.unpack(SAMPLES_FORMAT, raw))


def _stats_from_samples(samples: tuple[int, ...]) -> FrameStats:
    """:class:`FrameStats` for an already-unpacked frame. Not a public entry point."""
    spatial = spatial_returns(samples, FRAME_WIDTH, FRAME_HEIGHT)

    non_zero: list[int] = [sample for sample in samples if sample != NO_RETURN_SAMPLE]
    non_zero.sort()
    returned = spatial.returned

    min_code = non_zero[0] if non_zero else 0
    max_code = non_zero[-1] if non_zero else 0
    median_code = non_zero[len(non_zero) // 2] if non_zero else 0

    top = sum(1 for sample in samples[:FRAME_WIDTH] if sample != NO_RETURN_SAMPLE)
    bottom_start = (FRAME_HEIGHT - 1) * FRAME_WIDTH
    bottom = sum(
        1
        for sample in samples[bottom_start : bottom_start + FRAME_WIDTH]
        if sample != NO_RETURN_SAMPLE
    )
    left = right = 0
    for row in range(FRAME_HEIGHT):
        base = row * FRAME_WIDTH
        if samples[base] != NO_RETURN_SAMPLE:
            left += 1
        if samples[base + FRAME_WIDTH - 1] != NO_RETURN_SAMPLE:
            right += 1

    return FrameStats(
        samples=len(samples),
        returned=returned,
        yield_fraction=returned / len(samples),
        min_code=min_code,
        max_code=max_code,
        median_code=median_code,
        clipped_left=left,
        clipped_right=right,
        clipped_top=top,
        clipped_bottom=bottom,
        spatial=spatial,
    )


# --- false colour -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Palette:
    """A code-to-colour ramp, plus the treatment reserved for "no data".

    ``no_data`` is never a colour ``stops`` can also produce for a real
    return - see :data:`DEFAULT_PALETTE` - so a viewer of the rendered image
    cannot mistake an absent reading for a real near or far one.

    ``no_data_alt`` makes that treatment a coarse two-tone CHECKER rather than
    a flat fill, alternating with ``no_data`` every ``no_data_block`` pixels in
    both directions (see :meth:`no_data_color_at`). ``None`` - the default -
    keeps the flat fill, so a palette written before this existed renders
    exactly as it did. :data:`DEFAULT_PALETTE` opts in, because a flat fill in
    the one colour a monitor also shows when nothing is being drawn at all is
    indistinguishable from a dead panel, which is precisely the confusion the
    owner reported on 2026-08-28.

    ``stops`` maps a fraction in ``[0, 1]`` to an RGB triple by linear
    interpolation between the two nearest control points. This assigns colour
    to *relative position within one frame's own stretch*, never to any
    physical distance: no unit is implied anywhere in this class.
    """

    no_data: tuple[int, int, int]
    stops: tuple[tuple[float, tuple[int, int, int]], ...]
    no_data_alt: tuple[int, int, int] | None = None
    no_data_block: int = 8

    def no_data_color_at(self, column: int, row: int) -> tuple[int, int, int]:
        """The no-data colour for one pixel position.

        Position-dependent on purpose, and coarse on purpose: the block size is
        large enough to survive the window scaling both front ends apply, so
        the pattern still reads as a pattern rather than dissolving into an
        average grey that a distant viewer could mistake for a rendered
        surface.
        """
        if self.no_data_alt is None:
            return self.no_data
        block = self.no_data_block
        same = (column // block) % 2 == (row // block) % 2
        return self.no_data if same else self.no_data_alt

    def color_for(self, fraction: float) -> tuple[int, int, int]:
        """RGB for ``fraction``, an out-of-range value clamped to ``[0, 1]``."""
        frac = min(1.0, max(0.0, fraction))
        stops = self.stops
        if frac <= stops[0][0]:
            return stops[0][1]
        if frac >= stops[-1][0]:
            return stops[-1][1]
        for (pos_a, col_a), (pos_b, col_b) in itertools.pairwise(stops):
            if pos_a <= frac <= pos_b:
                span = pos_b - pos_a
                t = 0.0 if span <= 0 else (frac - pos_a) / span
                red = round(col_a[0] + (col_b[0] - col_a[0]) * t)
                green = round(col_a[1] + (col_b[1] - col_a[1]) * t)
                blue = round(col_a[2] + (col_b[2] - col_a[2]) * t)
                return (
                    min(255, max(0, red)),
                    min(255, max(0, green)),
                    min(255, max(0, blue)),
                )
        return stops[-1][1]  # unreachable: stops[0][0] == 0.0 and stops[-1][0] == 1.0


#: The blocks the default no-data checker is drawn in, in frame pixels.
NO_DATA_BLOCK: Final = 8

#: The two neutral greys the default "no data" checker alternates between.
#: Neutral - all three channels equal - is what makes them unreachable by the
#: ramp below, whose every colour is strongly tinted (see
#: :data:`DEFAULT_PALETTE`), and mid-toned rather than black so that a frame
#: the sensor returned nothing for cannot be confused with a panel that failed
#: to draw. They carry no meaning beyond "the sensor reported nothing here":
#: neither is nearer, further, better or worse than the other.
NO_DATA_PRIMARY: Final = (120, 120, 120)
NO_DATA_SECONDARY: Final = (72, 72, 72)

#: Dark blue (near end of a frame's own stretch) through cyan to yellow (far
#: end), with a coarse neutral-grey checker reserved for "no data". Every
#: colour the ramp produces has a channel spread of at least 30 - its stops are
#: (20, 20, 120), (0, 200, 200) and (255, 220, 0), and linear interpolation
#: between tinted colours on this path stays tinted - while both no-data greys
#: have a spread of exactly 0, so no interpolation anywhere on the ramp can
#: reach either of them.
DEFAULT_PALETTE: Final = Palette(
    no_data=NO_DATA_PRIMARY,
    no_data_alt=NO_DATA_SECONDARY,
    no_data_block=NO_DATA_BLOCK,
    stops=(
        (0.0, (20, 20, 120)),
        (0.5, (0, 200, 200)),
        (1.0, (255, 220, 0)),
    ),
)

#: The percentile window the default stretch uses, computed from a frame's
#: own non-zero codes. 2nd/98th rather than min/max so that a handful of
#: hot-pixel outliers cannot wash out the rest of the frame's contrast - the
#: robustness the viewer's caller was explicitly asked for.
STRETCH_LOW_PERCENTILE: Final = 0.02
STRETCH_HIGH_PERCENTILE: Final = 0.98


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    """The nearest-rank percentile of an already-sorted, possibly empty sequence."""
    if not sorted_values:
        return 0
    index = round(fraction * (len(sorted_values) - 1))
    return sorted_values[index]


def _stretch_bounds(
    sorted_non_zero: Sequence[int], low: int | None, high: int | None
) -> tuple[int, int]:
    """The (low, high) code window the ramp is stretched across.

    An explicit ``low``/``high`` is used exactly as given; either left as
    ``None`` falls back to the frame's own robust percentile window.
    """
    lo = low if low is not None else _percentile(sorted_non_zero, STRETCH_LOW_PERCENTILE)
    hi = high if high is not None else _percentile(sorted_non_zero, STRETCH_HIGH_PERCENTILE)
    return lo, hi


def depth_to_pixels(
    raw: bytes,
    *,
    low: int | None = None,
    high: int | None = None,
    palette: Palette = DEFAULT_PALETTE,
) -> tuple[bytes, ...]:
    """Map one raw frame's codes to display colours, one 3-byte RGB per pixel.

    Returned in row-major order matching the frame's own sample layout, so
    ``result[row * 640 + column]`` is that pixel's colour. A zero sample is
    always ``palette.no_data`` regardless of ``low``/``high``; every other
    sample is linearly stretched between ``low`` and ``high`` (default: a
    robust percentile window over this frame's own non-zero codes) and mapped
    through ``palette``. No physical unit is implied by any colour produced
    here.
    """
    _ensure_frame(raw)
    return _pixels_from_samples(
        struct.unpack(SAMPLES_FORMAT, raw), low=low, high=high, palette=palette
    )


def _pixels_from_samples(
    samples: tuple[int, ...],
    *,
    low: int | None,
    high: int | None,
    palette: Palette,
) -> tuple[bytes, ...]:
    """Colours for an already-unpacked frame. Not a public entry point.

    Two bounded caches, both built and dropped inside this call:

    * the two 640-entry no-data ROWS, one for each block parity. The checker is
      a fixed function of position, so 1280 entries answer it for all 256000
      samples and the inner loop is a lookup rather than two divisions.
    * one ``bytes`` per DISTINCT code actually present. A uint16 frame has at
      most 65536 of those and a real one has far fewer, so the ramp is
      evaluated once per code rather than once per pixel. Identical output, by
      construction: the colour of a code does not depend on where it sits.

    Neither cache outlives the call, and nothing per-pixel is retained.
    """
    non_zero_sorted = sorted(sample for sample in samples if sample != NO_RETURN_SAMPLE)
    lo, hi = _stretch_bounds(non_zero_sorted, low, high)
    span = hi - lo

    primary = bytes(palette.no_data)
    secondary = primary if palette.no_data_alt is None else bytes(palette.no_data_alt)
    block = palette.no_data_block
    even_row = tuple(
        primary if (column // block) % 2 == 0 else secondary for column in range(FRAME_WIDTH)
    )
    odd_row = tuple(
        secondary if (column // block) % 2 == 0 else primary for column in range(FRAME_WIDTH)
    )

    coloured: dict[int, bytes] = {}
    pixels: list[bytes] = []
    index = 0
    for row in range(FRAME_HEIGHT):
        no_data_row = even_row if (row // block) % 2 == 0 else odd_row
        for column, sample in enumerate(samples[index : index + FRAME_WIDTH]):
            if sample == NO_RETURN_SAMPLE:
                pixels.append(no_data_row[column])
                continue
            pixel = coloured.get(sample)
            if pixel is None:
                fraction = 1.0 if span <= 0 else (sample - lo) / span
                pixel = bytes(palette.color_for(fraction))
                coloured[sample] = pixel
            pixels.append(pixel)
        index += FRAME_WIDTH
    return tuple(pixels)


@dataclass(frozen=True, slots=True)
class FrameRender:
    """One frame's colours and its statistics, from a single unpack of it.

    Both halves of what a live front end needs per frame. Computing them
    together is not an optimisation detail that can drift: the picture and the
    numbers beside it are then guaranteed to describe the same 256000 samples,
    unpacked exactly once.

    What that guarantees is the UNPACK and the samples, not a single walk:
    :func:`render_frame` still makes the documented bounded colour and
    statistics passes over the tuple it unpacked. The saving is one
    ``struct.unpack`` of 512000 bytes rather than two, and one shared sample
    tuple rather than two that could disagree.
    """

    pixels: tuple[bytes, ...]
    stats: FrameStats


def render_frame(
    raw: bytes,
    *,
    low: int | None = None,
    high: int | None = None,
    palette: Palette = DEFAULT_PALETTE,
) -> FrameRender:
    """Colour and measure one raw frame, unpacking it exactly once.

    Identical in every value to calling :func:`depth_to_pixels` and
    :func:`frame_stats` on the same bytes - the two are implemented on the same
    internals this calls - but at one ``struct.unpack`` of 512000 bytes rather
    than two, over one shared sample tuple. Colour and statistics then make
    their own bounded passes over that tuple, exactly as they do when called
    separately; this promises one unpack, not one walk.
    """
    _ensure_frame(raw)
    samples: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, raw)
    return FrameRender(
        pixels=_pixels_from_samples(samples, low=low, high=high, palette=palette),
        stats=_stats_from_samples(samples),
    )


#: The PPM (P6, binary) header this module writes. ``tkinter.PhotoImage``
#: accepts a P6 blob directly via its ``data=`` argument, and decoding the
#: same bytes needs no Tk at all - which is what makes :func:`ppm_bytes`
#: testable without a display.
PPM_MAGIC: Final = "P6"
PPM_MAXVAL: Final = 255


def ppm_bytes(
    raw: bytes,
    *,
    low: int | None = None,
    high: int | None = None,
    palette: Palette = DEFAULT_PALETTE,
) -> bytes:
    """One frame, false-coloured, as a binary P6 PPM image."""
    return _ppm_from_pixels(depth_to_pixels(raw, low=low, high=high, palette=palette))


def _ppm_from_pixels(pixels: tuple[bytes, ...]) -> bytes:
    """Wrap already-coloured pixels in a P6 header. Not a public entry point."""
    header = f"{PPM_MAGIC}\n{FRAME_WIDTH} {FRAME_HEIGHT}\n{PPM_MAXVAL}\n".encode("ascii")
    return header + b"".join(pixels)


# --- PNG (for the web view: a browser will not display PPM) -----------------

#: Every PNG file starts with this fixed eight-byte signature.
PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"

#: PNG colour type 2: truecolour, no alpha, one byte per channel - matching
#: the 3-byte-per-pixel RGB :func:`depth_to_pixels` already produces, so no
#: channel is added or dropped going from PPM to PNG.
_PNG_COLOR_TYPE_TRUECOLOUR: Final = 2
_PNG_BIT_DEPTH: Final = 8

#: Filter type 0 ("None") on every scanline: the raw bytes are written
#: unchanged, so decoding needs no filter-reversal step - which is exactly
#: what lets the PPM-style byte-for-byte pixel checks in this module's tests
#: apply to the PNG output too.
_PNG_FILTER_NONE: Final = 0


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """One length-prefixed, CRC-suffixed PNG chunk."""
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def png_bytes(
    raw: bytes,
    *,
    low: int | None = None,
    high: int | None = None,
    palette: Palette = DEFAULT_PALETTE,
) -> bytes:
    """One frame, false-coloured, as a minimal non-interlaced truecolour PNG.

    Colour comes from :func:`depth_to_pixels`, the same function
    :func:`ppm_bytes` uses - this module has exactly one place that decides
    what a code looks like, whichever image format wraps it. Encoding is
    ``zlib`` from the standard library and nothing else: a signature, one
    ``IHDR``, one ``IDAT`` holding zlib-compressed filter-0 scanlines, and an
    empty ``IEND``.
    """
    pixels = depth_to_pixels(raw, low=low, high=high, palette=palette)
    scanlines = bytearray()
    for row in range(FRAME_HEIGHT):
        row_start = row * FRAME_WIDTH
        scanlines.append(_PNG_FILTER_NONE)
        scanlines += b"".join(pixels[row_start : row_start + FRAME_WIDTH])

    ihdr = struct.pack(
        ">IIBBBBB",
        FRAME_WIDTH,
        FRAME_HEIGHT,
        _PNG_BIT_DEPTH,
        _PNG_COLOR_TYPE_TRUECOLOUR,
        0,  # compression method: 0 is the only one PNG defines
        0,  # filter method: 0 is the only one PNG defines
        0,  # interlace method: 0 = no interlacing
    )
    return (
        PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines)))
        + _png_chunk(b"IEND", b"")
    )


# --- the window --------------------------------------------------------------

#: The ceiling on how often :class:`DepthViewer` actually redraws, however
#: fast :meth:`DepthViewer.show` is called. Frame statistics and false colour
#: are both an O(samples) pass, so the bound is on that work, not on Tk: a
#: caller that pushes frames faster than this is fed back "dropped", never
#: queued for later, per the no-unbounded-queue requirement.
MAX_RENDER_FPS: Final = 15.0
MIN_RENDER_INTERVAL: Final = 1.0 / MAX_RENDER_FPS


class DepthViewer:
    """A Tk window that shows the most recent frame handed to :meth:`show`.

    Opens no device and reads no hardware: every frame it ever displays
    arrives through :meth:`show`, called by a capture loop this class knows
    nothing about. Constructing one requires a real display; where none is
    available ``tkinter.Tk()`` itself raises :class:`tkinter.TclError`, which
    this class does not catch, so a caller (or a test) without a display finds
    out immediately rather than getting a window that silently never appears.
    """

    def __init__(self, *, title: str = "scanner3d - live depth", scale: int = 1) -> None:
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
            raise InvalidScale(
                f"scale must be a positive integer; got {scale!r}. A frame is "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} raw pixels and this only ever enlarges the "
                "displayed image by an integer factor - never shrinks or distorts it"
            )
        self._scale = scale
        self._closed = False
        self._received_count = 0
        self._rendered_count = 0
        self._dropped_count = 0
        self._last_render_time: float | None = None
        # kept alive deliberately: Tk drops a PhotoImage with no surviving
        # Python reference, even while it is still on screen.
        self._photo: tkinter.PhotoImage | None = None

        #: This window's own run. Reset on close(), so a second viewer over a
        #: second run never inherits the first run's collapse.
        self._health = ReturnHealthTracker()

        self._root = tkinter.Tk()
        self._root.title(title)
        self._image_label = tkinter.Label(self._root)
        self._image_label.pack()
        # Directly under the image, before the numbers: a frame the sensor
        # returned nothing for must not read as a picture with a quiet caption.
        self._health_label = tkinter.Label(
            self._root, justify="left", anchor="w", wraplength=FRAME_WIDTH, font=("TkFixedFont", 11)
        )
        self._health_label.pack(fill="x")
        self._stats_label = tkinter.Label(
            self._root, justify="left", anchor="w", font=("TkFixedFont", 10)
        )
        self._stats_label.pack(fill="x")
        self._clip_label = tkinter.Label(
            self._root, fg="white", bg="#b00000", font=("TkFixedFont", 11, "bold")
        )
        self._clip_label.pack(fill="x")

    def show(self, raw: bytes) -> None:
        """Push one frame. Non-blocking: a frame arriving too soon is dropped.

        Refuses (see :func:`_ensure_frame`) before the window is touched, and
        raises :class:`ViewerWindowClosed` if called after :meth:`close`.
        Otherwise this method never queues a frame for later: at most one
        frame - the most recent - is ever pending, and if the render budget
        (:data:`MAX_RENDER_FPS`) has not yet elapsed since the last redraw,
        this frame is counted as dropped and the window is left exactly as it
        was, so the calling capture loop is never made to wait on the GUI.
        """
        if self._closed:
            raise ViewerWindowClosed(
                "show() was called after close(); this viewer's window is gone and will not "
                "be reopened"
            )
        _ensure_frame(raw)
        self._received_count += 1

        now = time.monotonic()
        elapsed_enough = (
            self._last_render_time is None or (now - self._last_render_time) >= MIN_RENDER_INTERVAL
        )
        if not elapsed_enough:
            self._dropped_count += 1
            # Service the window (e.g. its close button) without waiting on
            # anything: update_idletasks only processes what is already
            # pending, it never blocks for new events.
            self._root.update_idletasks()
            return

        self._render(raw, now)
        self._root.update()

    def _render(self, raw: bytes, now: float) -> None:
        """Actually redraw: called at most every ``MIN_RENDER_INTERVAL``.

        One :func:`render_frame` produces the picture and the numbers beside
        it, so a redraw unpacks the frame once and colours and measures the
        same samples - not once for colour and again for statistics. It is one
        unpack and one shared sample tuple, not one walk: the bounded colour
        and statistics passes over that tuple are unchanged.

        The return-health window advances on RENDERED frames only, which is the
        honest thing for a viewer that deliberately drops frames it is too slow
        to draw: the baseline is over what this window has actually seen, and
        :meth:`health_summary` says so.
        """
        rendered = render_frame(raw)
        stats = rendered.stats
        health = self._health.update(stats.yield_fraction)
        photo = tkinter.PhotoImage(data=_ppm_from_pixels(rendered.pixels))
        if self._scale != 1:
            photo = photo.zoom(self._scale, self._scale)
        self._photo = photo
        self._image_label.configure(image=self._photo)

        previous = self._last_render_time
        fps = 0.0 if previous is None else 1.0 / max(now - previous, 1e-9)
        self._last_render_time = now
        self._rendered_count += 1

        self._stats_label.configure(
            text=(
                f"frame {self._received_count}  (rendered {self._rendered_count}, "
                f"dropped {self._dropped_count})   yield {stats.yield_fraction:.1%}   "
                f"fps {fps:.1f}\n"
                f"code min {stats.min_code}  median {stats.median_code}  max {stats.max_code}"
            )
        )
        self._health_label.configure(
            text=health_summary(health, stats.spatial),
            fg="#ffffff" if health.is_warning else "#333333",
            bg="#8a4b00" if health.is_warning else self._root.cget("bg"),
            font=("TkFixedFont", 11, "bold" if health.is_warning else "normal"),
        )
        if stats.has_clipping:
            self._clip_label.configure(
                text=(
                    "CLIPPING - object runs off the sensor edge:  "
                    f"left={stats.clipped_left}  right={stats.clipped_right}  "
                    f"top={stats.clipped_top}  bottom={stats.clipped_bottom}"
                )
            )
        else:
            self._clip_label.configure(text="")

    def close(self) -> None:
        """Destroy the window. Tolerates being called more than once.

        The return-health window is cleared here as well as in the window: this
        object is a run, and a torn-down run must not leave a baseline behind
        for anything that outlives it.
        """
        if self._closed:
            return
        self._closed = True
        self._health.reset()
        # Already gone (e.g. the owner closed the window by hand) is not an
        # error: closing an already-closed window is a no-op either way.
        with contextlib.suppress(tkinter.TclError):
            self._root.destroy()


# --- the local web view -------------------------------------------------------

#: A minimal, self-contained page: no external script, stylesheet or font -
#: this server has nothing to serve them from and nothing here should depend
#: on the viewing device having internet access. Polls the two JSON/PNG
#: endpoints below on a fixed timer rather than opening any kind of push
#: connection, which keeps the server side of this to two GET handlers.
_INDEX_HTML: Final = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>scanner3d - live depth</title>
<style>
  body { background: #111; color: #eee; font-family: monospace; text-align: center;
         margin: 0; padding: 1rem; }
  h1 { font-size: 1rem; font-weight: normal; opacity: 0.7; }
  img { max-width: 100%; height: auto; border: 1px solid #444; background: #000; }
  #stats { margin-top: 0.5rem; font-size: 1rem; white-space: pre-wrap; }
  #clip { display: none; margin-top: 0.5rem; padding: 0.5rem;
          background: #b00000; color: #fff; font-weight: bold; }
  #clip.show { display: block; }
  #health { margin-top: 0.5rem; padding: 0.5rem; white-space: pre-wrap;
            font-size: 1rem; text-align: left; }
  #health.warn { background: #8a4b00; color: #fff; font-weight: bold; }
</style>
</head>
<body>
<h1>scanner3d - live depth</h1>
<img id="frame" alt="waiting for first frame...">
<div id="health"></div>
<div id="stats">waiting for first frame...</div>
<div id="clip"></div>
<script>
async function poll() {
  try {
    const res = await fetch("/stats.json", {cache: "no-store"});
    const data = await res.json();
    if (data.available) {
      document.getElementById("frame").src = "/frame.png?_=" + Date.now();
      const s = data.stats;
      document.getElementById("stats").textContent =
        "frame " + data.frame_number +
        "   yield " + (s.yield_fraction * 100).toFixed(1) + "%" +
        "   fps " + data.fps.toFixed(1) + "\\n" +
        "code min " + s.codes.min + "  median " + s.codes.median + "  max " + s.codes.max;
      const health = document.getElementById("health");
      health.textContent = data.health_summary;
      health.classList.toggle("warn", data.health.warning);
      const clip = document.getElementById("clip");
      if (s.clipping.any) {
        clip.textContent = "CLIPPING - object runs off the sensor edge:  " +
          "left=" + s.clipping.left + "  right=" + s.clipping.right +
          "  top=" + s.clipping.top + "  bottom=" + s.clipping.bottom;
        clip.classList.add("show");
      } else {
        clip.classList.remove("show");
      }
    }
  } catch (err) {
    // A missed poll is not fatal - the next one, half a second later, tries again.
  }
  setTimeout(poll, 500);
}
poll();
</script>
</body>
</html>
"""
_INDEX_HTML_BYTES: Final = _INDEX_HTML.encode("utf-8")


class _ServerState:
    """One server RUN's state, shared between the capture and HTTP threads.

    Exactly one lock and one latest frame - never a growing buffer or a
    queue. :meth:`ViewerServer.show` overwrites ``raw`` and returns; a slow or
    absent browser only ever reads whatever happens to be here when it asks,
    and never causes memory to accumulate.

    One object per run, installed by :class:`_RunHandle`. A stop/start does not
    clear this object; it replaces it. That is what keeps a request still in
    flight when the server was stopped confined to the run it arrived in: it
    holds this object for the whole of its request and can neither read nor
    seed the state the restarted run publishes into.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.raw: bytes | None = None
        self.received_count = 0
        self.last_update_time: float | None = None
        self.previous_update_time: float | None = None
        #: A THIRD lock, held by an HTTP worker for the whole of one
        #: snapshot-and-judge. It orders the request workers against each
        #: other so that every answer describes one frame: the snapshot is
        #: taken inside it, so a worker cannot analyse frame 1 while frame 2
        #: is being judged and then report frame 1's numbers beside frame 2's
        #: verdict. It is never held by :meth:`ViewerServer.show`, and ``lock``
        #: is never held across the analysis, so a capture loop still never
        #: waits on a browser.
        self.analysis_lock = threading.Lock()
        #: The last complete answer built, kept so that a repeat poll for a
        #: frame already analysed is answered from it rather than by measuring
        #: the same 256000 samples again. One payload, replaced in place.
        self.latest_payload: dict[str, object] | None = None
        #: A SECOND lock, held only while the return-health window is read or
        #: advanced. Deliberately not ``lock``: computing a frame's statistics
        #: takes long enough that holding ``lock`` across it would make
        #: :meth:`ViewerServer.show` - and therefore the capture loop calling
        #: it - wait on a browser poll, which is the one thing this class
        #: promises never to do.
        self.health_lock = threading.Lock()
        self.health = ReturnHealthTracker()
        #: The HIGHEST frame number the window has been advanced for. Only a
        #: strictly greater one advances it again, so a browser polling faster
        #: than frames arrive cannot feed one frame into the baseline several
        #: times over, and a stale request that overtakes a newer one cannot
        #: wind it back to a frame already counted.
        self.health_frame = 0
        self.latest_health: ReturnHealth | None = None

    def publish(self, raw: bytes, now: float) -> None:
        """Record one frame as this run's latest. One lock, four assignments."""
        with self.lock:
            self.previous_update_time = self.last_update_time
            self.last_update_time = now
            self.received_count += 1
            self.raw = raw

    def snapshot(self) -> tuple[bytes | None, int, float | None, float | None]:
        """This run's latest frame and the counters that describe THAT frame.

        Taken together under one lock, and released before anything is
        computed from them, so the frame, its number and its timing cannot be
        read from two different moments - and nothing expensive ever happens
        while the lock a capture loop needs is held.
        """
        with self.lock:
            return (self.raw, self.received_count, self.last_update_time, self.previous_update_time)


class _ViewerRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serves exactly three fixed paths and nothing read from disk or a device.

    Every byte this handler ever sends is computed from
    :class:`_ServerState`, which only :meth:`ViewerServer.show` writes to; no
    path parameter is parsed as a filename or command, so there is nothing
    here for a request path to traverse into.
    """

    server_version = "scanner3d-viewer/1"

    def log_message(self, log_format: str, *args: object) -> None:
        # Quiet on purpose: a browser polling twice a second would otherwise
        # spam the capturing process's own stderr with an access-log line per
        # request. This module's job is showing a frame, not operating a
        # request-logging web server.
        del log_format, args

    def _write(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _runs(self) -> _RunHandle:
        return cast(_ViewerHTTPServer, self.server).runs

    def do_GET(self) -> None:
        # The run is bound ONCE, here, and every read and every advance below
        # is made against that object. A request that arrived in a run which
        # has since been stopped therefore stays inside the run it arrived in:
        # it cannot answer for, or seed the baseline of, the run that replaced
        # it. See :meth:`_serve_stats_json` for what it is told instead.
        runs = self._runs()
        run = runs.current
        if self.path in ("/", "/index.html"):
            self._write(200, "text/html; charset=utf-8", _INDEX_HTML_BYTES)
        elif self.path == "/frame.png" or self.path.startswith("/frame.png?"):
            self._serve_frame_png(runs, run)
        elif self.path == "/stats.json" or self.path.startswith("/stats.json?"):
            self._serve_stats_json(runs, run)
        else:
            self.send_error(404)

    def _serve_frame_png(self, runs: _RunHandle, run: _ServerState) -> None:
        raw, _number, _last_t, _previous_t = run.snapshot()
        if raw is not None and not runs.is_current(run):
            # The run this request belongs to was stopped while it was being
            # served. Its frame is not this server's current picture, and
            # saying so is the honest answer.
            self._write(404, "text/plain; charset=utf-8", b"the run this request arrived in ended")
            return
        if raw is None:
            # Sane and quiet, not a crash: a browser opened before the first
            # show() call is the ordinary startup sequence, not an error.
            self._write(404, "text/plain; charset=utf-8", b"no frame received yet")
            return
        self._write(200, "image/png", png_bytes(raw))

    def _serve_stats_json(self, runs: _RunHandle, run: _ServerState) -> None:
        payload = self._analysed(run)
        if not runs.is_current(run):
            # Stopped mid-request. Whatever this handler measured describes a
            # run that is over, so it is not offered as the current state; the
            # run it measured was its own object, so nothing it did reaches the
            # run that replaced it.
            payload = {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "stale_run": True,
                "frame_number": payload["frame_number"],
            }
        self._write(200, "application/json", json.dumps(payload).encode("utf-8"))

    @classmethod
    def _analysed(cls, run: _ServerState) -> dict[str, object]:
        """One request's complete answer about exactly ONE published frame.

        Snapshot and judgement happen together, inside ``analysis_lock``, so
        the whole document describes one frame: ``frame_number``, ``fps``, the
        numeric and spatial statistics, the verdict's ``yield_fraction`` and
        the rendered summary all come from the same 512000 bytes. Round 2 took
        the snapshot outside every lock and then answered a stale request from
        the health record, which protected the bounded window but made the
        document internally false - older spatial evidence beside a newer
        frame's verdict.

        Ordering the request workers against each other is enough for that, and
        is all that is done: ``lock`` is held only for the snapshot itself and
        never across this analysis, and ``analysis_lock`` is never taken by
        :meth:`ViewerServer.show`. A capture loop calling ``show`` therefore
        still never waits for, and does no part of, a browser's poll. Two
        browsers polling at once are serialised against each other instead,
        which costs a poll its wait rather than costing the scanner a frame.

        A repeat poll for a frame already analysed is answered from
        ``latest_payload`` rather than measured again, so a browser polling
        faster than frames arrive costs one analysis per frame, not per poll.
        """
        with run.analysis_lock:
            raw, frame_number, last_t, previous_t = run.snapshot()
            if raw is None:
                return {
                    "schema_version": SCHEMA_VERSION,
                    "available": False,
                    "frame_number": frame_number,
                }
            cached = run.latest_payload
            if cached is not None and cached["frame_number"] == frame_number:
                return cached
            stats = frame_stats(raw)
            health = cls._health_for(run, frame_number, stats)
            fps = (
                0.0
                if last_t is None or previous_t is None
                else 1.0 / max(last_t - previous_t, 1e-9)
            )
            payload: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "available": True,
                "frame_number": frame_number,
                "fps": fps,
                "stats": stats.as_dict(),
                "health": health.as_dict(),
                # The same two lines the Tk window shows, rendered once here so
                # the page and the window cannot word the same frame
                # differently.
                "health_summary": health_summary(health, stats.spatial),
            }
            run.latest_payload = payload
            return payload

    @staticmethod
    def _health_for(state: _ServerState, frame_number: int, stats: FrameStats) -> ReturnHealth:
        """This frame's verdict, advancing the window MONOTONICALLY.

        The window advances only for a frame number strictly greater than the
        last one judged; every other request is answered from the verdict
        already recorded, without touching the history at all.

        Answering a repeat poll from the record is what stops a browser polling
        twice a second, while frames arrive at ten, from filling the bounded
        window with copies of one frame and flattening the baseline.

        Refusing a LOWER frame number is now a second line of defence rather
        than the first: :meth:`_analysed` takes its snapshot inside
        ``analysis_lock``, so the frame numbers reaching this method are
        already non-decreasing. It is kept because it is what the rule
        actually is - a bounded baseline may never go backwards or count a
        frame twice - and because this method is also reachable directly, by
        the tests that write the interleaving out by hand.
        """
        with state.health_lock:
            if state.latest_health is not None and frame_number <= state.health_frame:
                return state.latest_health
            health = state.health.update(stats.yield_fraction)
            state.health_frame = frame_number
            state.latest_health = health
            return health


class _RunHandle:
    """Which :class:`_ServerState` is the CURRENT run, and nothing else.

    A stop/start installs a new state object rather than clearing the old one.
    An HTTP handler binds the current run once, at the top of its request, and
    works only against that object; :meth:`is_current` is how it finds out,
    before answering, that the run it was serving has since ended.

    The lock here is held for one attribute read or one attribute write and
    never across anything else, so binding a run costs a handler - and
    :meth:`ViewerServer.show` - an uncontended acquisition and nothing more.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run = _ServerState()

    @property
    def current(self) -> _ServerState:
        """The run being published into now."""
        with self._lock:
            return self._run

    def begin(self) -> _ServerState:
        """Install a fresh run and return it. The previous object is left alone.

        Left alone deliberately: a handler still inside a request holds it, and
        clearing it underneath that handler would produce a torn answer instead
        of an honest "this run has ended". Nothing else references it once that
        handler returns, so it is collected with the request.

        A new run has published nothing - no frame, no count, no timing, no
        history - and ``/stats.json`` says exactly that until the first
        :meth:`ViewerServer.show`.
        """
        with self._lock:
            self._run = _ServerState()
            return self._run

    def is_current(self, run: _ServerState) -> bool:
        """Whether ``run`` is still the run this server publishes into."""
        with self._lock:
            return run is self._run


class _ViewerHTTPServer(http.server.ThreadingHTTPServer):
    """A threading HTTP server carrying the one piece of state its handler needs."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], runs: _RunHandle) -> None:
        self.runs = runs
        super().__init__(server_address, _ViewerRequestHandler)


def _ensure_valid_port(port: int) -> int:
    """Return ``port`` if it is a usable TCP port number, else raise."""
    if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
        raise InvalidPort(
            f"port must be an integer 0-65535 (0 asks the OS to pick one); got {port!r}"
        )
    return port


class ViewerServer:
    """A local HTTP view over the most recent frame handed to :meth:`show`.

    Serves exactly three fixed paths: the page at ``/``, the latest frame as
    PNG at ``/frame.png``, and its :class:`FrameStats` as JSON at
    ``/stats.json``. Binds to ``127.0.0.1`` unless ``bind_all_interfaces=True``
    is passed explicitly - widening who can reach this server across a
    network is a deliberate act by the caller, never the default.

    Mirrors :class:`DepthViewer`'s shape (``show``/``close`` there,
    ``show``/``start``/``stop`` here) so the two are interchangeable at a
    capture loop's call site, but the two do not behave identically:
    :class:`DepthViewer` redraws inline, throttled, when :meth:`show` is
    called; this class never draws anything in :meth:`show` at all - it only
    overwrites one in-memory reference under a lock, and the (comparatively
    expensive) PNG encoding and stats computation happen lazily, in an HTTP
    worker thread, only when a client actually requests them. A capture loop
    calling :meth:`show` is therefore never blocked by, or even aware of,
    whether a browser is open at all.
    """

    def __init__(self, *, port: int = 0, bind_all_interfaces: bool = False) -> None:
        self._requested_port = _ensure_valid_port(port)
        self._host = "0.0.0.0" if bind_all_interfaces else "127.0.0.1"
        self._runs = _RunHandle()
        self._httpd: _ViewerHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def _state(self) -> _ServerState:
        """The current run's state. Not part of this class's public surface."""
        return self._runs.current

    @property
    def host(self) -> str:
        """The address this server binds to - ``127.0.0.1`` unless opted out of."""
        return self._host

    @property
    def port(self) -> int:
        """The bound TCP port. Refuses before :meth:`start`, see :class:`ViewerServerNotRunning`."""
        if self._httpd is None:
            raise ViewerServerNotRunning(
                "the bound port is only known once start() has actually bound a socket; this "
                "server has not been started"
            )
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        """Bind the socket and begin serving on a background thread.

        Calling this twice while already running is a no-op, not an error -
        symmetric with :meth:`stop`'s idempotence.
        """
        if self._httpd is not None:
            return
        # A started server is a new run: neither the last one's frame nor
        # whatever it saw collapse is evidence about this one. A fresh state
        # object rather than a cleared one, so a request still in flight from
        # the run before cannot write into this one. Until show() is called,
        # this run has published nothing, and /stats.json says so.
        self._runs.begin()
        self._httpd = _ViewerHTTPServer((self._host, self._requested_port), self._runs)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="scanner3d-viewer-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and release the socket. Tolerates being called more than once."""
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._httpd = None
        self._thread = None
        # Same reason as in start(): the stopped run's state is retired, not
        # wiped, so a handler still holding it answers "that run has ended"
        # instead of tearing, and this server's published state is empty again.
        self._runs.begin()

    def show(self, raw: bytes) -> None:
        """Publish one frame for the next poll to pick up. Never blocks.

        Refuses (see :func:`_ensure_frame`) a malformed frame before this
        server's state is touched at all. Otherwise this is two uncontended
        lock acquisitions - one to find the current run, one guarding four
        attribute assignments - and nothing else. No image is encoded, no
        statistic is computed and no verdict is reached here, and the lock the
        HTTP workers do their analysis under is never taken, so a capture loop
        calling this is never made to wait on whether, or how fast, anything is
        actually reading the other end.
        """
        _ensure_frame(raw)
        self._runs.current.publish(raw, time.monotonic())
