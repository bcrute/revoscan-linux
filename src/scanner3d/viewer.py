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
  :mod:`scanner3d.v4l2_capture` use. It is rendered as a distinct "no data"
  colour, never blended into the near end of the depth ramp, and never
  counted as though it measured anything.
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
        }


def frame_stats(raw: bytes) -> FrameStats:
    """Compute :class:`FrameStats` for one raw depth frame.

    Refuses (see :func:`_ensure_frame`) before a single sample is unpacked.
    """
    _ensure_frame(raw)
    samples: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, raw)

    returned = 0
    non_zero: list[int] = []
    for sample in samples:
        if sample != NO_RETURN_SAMPLE:
            returned += 1
            non_zero.append(sample)
    non_zero.sort()

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
    )


# --- false colour -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Palette:
    """A code-to-colour ramp, plus a colour reserved for "no data".

    ``no_data`` is never a colour ``stops`` can also produce for a real
    return - see :data:`DEFAULT_PALETTE` - so a viewer of the rendered image
    cannot mistake an absent reading for a real near or far one.

    ``stops`` maps a fraction in ``[0, 1]`` to an RGB triple by linear
    interpolation between the two nearest control points. This assigns colour
    to *relative position within one frame's own stretch*, never to any
    physical distance: no unit is implied anywhere in this class.
    """

    no_data: tuple[int, int, int]
    stops: tuple[tuple[float, tuple[int, int, int]], ...]

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


#: Dark blue (near end of a frame's own stretch) through cyan to yellow (far
#: end), with pure black reserved for "no data" - a colour the ramp itself
#: never produces, since its darkest stop is (20, 20, 120), not black.
DEFAULT_PALETTE: Final = Palette(
    no_data=(0, 0, 0),
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
    samples: tuple[int, ...] = struct.unpack(SAMPLES_FORMAT, raw)
    non_zero_sorted = sorted(sample for sample in samples if sample != NO_RETURN_SAMPLE)
    lo, hi = _stretch_bounds(non_zero_sorted, low, high)
    span = hi - lo

    no_data_pixel = bytes(palette.no_data)
    pixels: list[bytes] = []
    for sample in samples:
        if sample == NO_RETURN_SAMPLE:
            pixels.append(no_data_pixel)
            continue
        fraction = 1.0 if span <= 0 else (sample - lo) / span
        pixels.append(bytes(palette.color_for(fraction)))
    return tuple(pixels)


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
    pixels = depth_to_pixels(raw, low=low, high=high, palette=palette)
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

        self._root = tkinter.Tk()
        self._root.title(title)
        self._image_label = tkinter.Label(self._root)
        self._image_label.pack()
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
        """Actually redraw: called at most every ``MIN_RENDER_INTERVAL``."""
        stats = frame_stats(raw)
        ppm = ppm_bytes(raw)
        photo = tkinter.PhotoImage(data=ppm)
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
        """Destroy the window. Tolerates being called more than once."""
        if self._closed:
            return
        self._closed = True
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
</style>
</head>
<body>
<h1>scanner3d - live depth</h1>
<img id="frame" alt="waiting for first frame...">
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
    """Mutable state shared between the capture thread and HTTP worker threads.

    Exactly one lock and one latest frame - never a growing buffer or a
    queue. :meth:`ViewerServer.show` overwrites ``raw`` and returns; a slow or
    absent browser only ever reads whatever happens to be here when it asks,
    and never causes memory to accumulate.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.raw: bytes | None = None
        self.received_count = 0
        self.last_update_time: float | None = None
        self.previous_update_time: float | None = None


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

    def _state(self) -> _ServerState:
        return cast(_ViewerHTTPServer, self.server).state

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._write(200, "text/html; charset=utf-8", _INDEX_HTML_BYTES)
        elif self.path == "/frame.png" or self.path.startswith("/frame.png?"):
            self._serve_frame_png()
        elif self.path == "/stats.json" or self.path.startswith("/stats.json?"):
            self._serve_stats_json()
        else:
            self.send_error(404)

    def _serve_frame_png(self) -> None:
        state = self._state()
        with state.lock:
            raw = state.raw
        if raw is None:
            # Sane and quiet, not a crash: a browser opened before the first
            # show() call is the ordinary startup sequence, not an error.
            self._write(404, "text/plain; charset=utf-8", b"no frame received yet")
            return
        self._write(200, "image/png", png_bytes(raw))

    def _serve_stats_json(self) -> None:
        state = self._state()
        with state.lock:
            raw = state.raw
            frame_number = state.received_count
            last_t = state.last_update_time
            previous_t = state.previous_update_time
        payload: dict[str, object]
        if raw is None:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "frame_number": frame_number,
            }
        else:
            fps = (
                0.0
                if last_t is None or previous_t is None
                else 1.0 / max(last_t - previous_t, 1e-9)
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "available": True,
                "frame_number": frame_number,
                "fps": fps,
                "stats": frame_stats(raw).as_dict(),
            }
        self._write(200, "application/json", json.dumps(payload).encode("utf-8"))


class _ViewerHTTPServer(http.server.ThreadingHTTPServer):
    """A threading HTTP server carrying the one piece of state its handler needs."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], state: _ServerState) -> None:
        self.state = state
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
        self._state = _ServerState()
        self._httpd: _ViewerHTTPServer | None = None
        self._thread: threading.Thread | None = None

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
        self._httpd = _ViewerHTTPServer((self._host, self._requested_port), self._state)
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

    def show(self, raw: bytes) -> None:
        """Publish one frame for the next poll to pick up. Never blocks.

        Refuses (see :func:`_ensure_frame`) a malformed frame before this
        server's state is touched at all. Otherwise this is one lock
        acquisition guarding a handful of attribute assignments - no image is
        encoded and no statistic is computed here, so a capture loop calling
        this is never made to wait on whether, or how fast, anything is
        actually reading the other end.
        """
        _ensure_frame(raw)
        now = time.monotonic()
        with self._state.lock:
            self._state.previous_update_time = self._state.last_update_time
            self._state.last_update_time = now
            self._state.received_count += 1
            self._state.raw = raw
