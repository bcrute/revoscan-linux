"""Tests for the live depth viewer - two front ends driven by frames handed to them.

Nothing here opens a device or touches hardware: every frame used below is
either built by hand in this file or read from a committed fixture on disk.
The module under test is split cleanly by what it needs:

* :func:`frame_stats`, :func:`depth_to_pixels`, :func:`ppm_bytes`,
  :func:`png_bytes` and :class:`~scanner3d.viewer.Palette` are pure
  arithmetic over bytes, and every test of them runs with no Tk window and no
  HTTP server ever created - most run before either could even be checked
  for.
* :class:`~scanner3d.viewer.DepthViewer` needs a real Tk display. Every test
  that constructs one goes through the ``viewer`` fixture below, which skips
  outright (rather than failing) when ``tkinter.Tk()`` cannot open a window -
  the CI guarantee this module was built against.
* :class:`~scanner3d.viewer.ViewerServer` needs only loopback TCP, which is
  always available; its tests bind an OS-assigned ephemeral port (port 0) and
  drive it with ``http.client`` against the real bound socket, then stop it.

Four things are being defended, and they are not the false colour:

* that a raw sample of 0 is *never* read as a measurement, in the statistics,
  in either rendered image, or in either front end - it is "no data",
  distinctly coloured, and excluded from every min/max/median figure.
* that the four edge-clipping counts are exactly the non-zero returns on each
  one-pixel edge of the frame, because that is the one number in this whole
  module the owner actually needs to catch a ruined scan while it is still
  running.
* that the percentile stretch is robust: one wild outlier code must not be
  able to wash out the colour contrast across the rest of a frame.
* that :meth:`~scanner3d.viewer.ViewerServer.show` never does real work on
  the caller's thread, and that the server binds to loopback unless told
  otherwise - that is a security property, not a preference, and is asserted
  directly rather than assumed.
"""

from __future__ import annotations

import http.client
import json
import struct
import time
import tkinter
import zlib
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import scanner3d.viewer as viewer_module
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH
from scanner3d.viewer import (
    DEFAULT_PALETTE,
    MIN_RENDER_INTERVAL,
    PNG_SIGNATURE,
    SCHEMA_VERSION,
    DepthViewer,
    FrameSizeMismatch,
    FrameTypeMismatch,
    InvalidPort,
    InvalidScale,
    Palette,
    ViewerError,
    ViewerServer,
    ViewerServerNotRunning,
    ViewerWindowClosed,
    depth_to_pixels,
    frame_stats,
    png_bytes,
    ppm_bytes,
)

# --- building synthetic frames ----------------------------------------------


def make_frame(overrides: Mapping[tuple[int, int], int] = {}, *, fill: int = 0) -> bytes:
    """A whole 512000-byte frame: ``fill`` everywhere except at named (u, v)."""
    samples = [fill] * (FRAME_WIDTH * FRAME_HEIGHT)
    for (u, v), value in overrides.items():
        samples[v * FRAME_WIDTH + u] = value
    return struct.pack(f"<{len(samples)}H", *samples)


ALL_ZERO_FRAME = make_frame()


# --- FrameStats: hand-computed values ----------------------------------------


def test_all_zero_frame_has_no_returns_and_no_clipping():
    stats = frame_stats(ALL_ZERO_FRAME)
    assert stats.samples == FRAME_WIDTH * FRAME_HEIGHT
    assert stats.returned == 0
    assert stats.yield_fraction == 0.0
    assert stats.min_code == 0
    assert stats.max_code == 0
    assert stats.median_code == 0
    assert stats.clipped_left == 0
    assert stats.clipped_right == 0
    assert stats.clipped_top == 0
    assert stats.clipped_bottom == 0
    assert stats.has_clipping is False


def test_frame_stats_hand_computed_interior_samples():
    """Three interior (non-edge) returns: min/max/median and yield by hand."""
    raw = make_frame({(10, 10): 100, (20, 20): 300, (30, 30): 200})
    stats = frame_stats(raw)
    assert stats.returned == 3
    assert stats.yield_fraction == pytest.approx(3 / (FRAME_WIDTH * FRAME_HEIGHT))
    assert stats.min_code == 100
    assert stats.max_code == 300
    assert stats.median_code == 200  # sorted [100, 200, 300], middle element
    assert stats.has_clipping is False


def test_frame_stats_median_of_even_count_picks_upper_middle():
    """4 non-zero samples: sorted [10, 20, 30, 40], index len//2 == 2 -> 30."""
    raw = make_frame({(1, 1): 40, (2, 2): 10, (3, 3): 30, (4, 4): 20})
    stats = frame_stats(raw)
    assert stats.median_code == 30


def test_frame_stats_clipping_fires_on_each_edge_independently():
    raw = make_frame(
        {
            (0, 5): 111,  # left edge
            (FRAME_WIDTH - 1, 6): 222,  # right edge
            # Column 620 is >= FRAME_HEIGHT (400): this catches a top-edge scan
            # that is wrongly bounded by height instead of width.
            (620, 0): 333,  # top edge
            (200, FRAME_HEIGHT - 1): 444,  # bottom edge
        }
    )
    stats = frame_stats(raw)
    assert stats.clipped_left == 1
    assert stats.clipped_right == 1
    assert stats.clipped_top == 1
    assert stats.clipped_bottom == 1
    assert stats.has_clipping is True


def test_frame_stats_clipping_stays_quiet_for_interior_only_returns():
    raw = make_frame({(1, 1): 5, (638, 398): 5, (5, 1): 5, (300, 398): 5})
    stats = frame_stats(raw)
    assert stats.clipped_left == 0
    assert stats.clipped_right == 0
    assert stats.clipped_top == 0
    assert stats.clipped_bottom == 0
    assert stats.has_clipping is False


def test_frame_stats_clipping_counts_every_return_on_an_edge_not_just_one():
    raw = make_frame({(0, v): 50 for v in range(0, FRAME_HEIGHT, 10)})
    stats = frame_stats(raw)
    assert stats.clipped_left == len(range(0, FRAME_HEIGHT, 10))
    assert stats.clipped_right == 0


def test_frame_stats_as_dict_carries_schema_version_and_clipping_block():
    raw = make_frame({(0, 0): 7})  # corner: counts on both left and top
    stats = frame_stats(raw)
    d = stats.as_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["clipping"] == {
        "left": 1,
        "right": 0,
        "top": 1,
        "bottom": 0,
        "any": True,
    }
    codes = d["codes"]
    assert isinstance(codes, dict)
    assert codes["min"] == 7
    assert codes["max"] == 7
    assert codes["median"] == 7


def test_frame_stats_is_frozen():
    stats = frame_stats(ALL_ZERO_FRAME)
    with pytest.raises(FrozenInstanceError):
        stats.samples = 1  # type: ignore[misc]


# --- refusals -----------------------------------------------------------------


def test_frame_stats_refuses_wrong_size():
    with pytest.raises(FrameSizeMismatch):
        frame_stats(b"\x00" * (EXPECTED_FRAME_BYTES - 1))
    with pytest.raises(FrameSizeMismatch):
        frame_stats(b"\x00" * (EXPECTED_FRAME_BYTES + 2))


def test_frame_stats_refuses_non_bytes():
    with pytest.raises(FrameTypeMismatch):
        frame_stats(bytearray(EXPECTED_FRAME_BYTES))  # type: ignore[arg-type]
    with pytest.raises(FrameTypeMismatch):
        frame_stats("not bytes at all")  # type: ignore[arg-type]
    with pytest.raises(FrameTypeMismatch):
        frame_stats(None)  # type: ignore[arg-type]


def test_depth_to_pixels_refuses_wrong_size_and_non_bytes():
    with pytest.raises(FrameSizeMismatch):
        depth_to_pixels(b"short")
    with pytest.raises(FrameTypeMismatch):
        depth_to_pixels([0] * EXPECTED_FRAME_BYTES)  # type: ignore[arg-type]


def test_ppm_bytes_refuses_wrong_size_and_non_bytes():
    with pytest.raises(FrameSizeMismatch):
        ppm_bytes(b"\x00" * 10)
    with pytest.raises(FrameTypeMismatch):
        ppm_bytes(12345)  # type: ignore[arg-type]


def test_refusals_are_scanner_errors_with_unsupported_mode_status():
    assert issubclass(ViewerError, ScannerError)
    try:
        frame_stats(b"too short")
    except ViewerError as exc:
        assert exc.status == DoctorStatus.UNSUPPORTED_MODE
    else:
        pytest.fail("expected a ViewerError")


def test_nothing_is_unpacked_before_the_length_refusal():
    """A too-short buffer never reaches struct.unpack; no partial parse leaks out."""
    # struct.unpack on a too-short buffer raises struct.error, not our typed
    # refusal - if the module unpacked before checking length, this would be
    # a struct.error instead of FrameSizeMismatch.
    with pytest.raises(FrameSizeMismatch):
        frame_stats(b"\x01\x02\x03")


# --- zero is never a measurement, anywhere -----------------------------------


def test_zero_samples_render_as_the_no_data_colour_everywhere():
    raw = make_frame({(50, 50): 40000}, fill=0)
    pixels = depth_to_pixels(raw)
    no_data_pixel = bytes(DEFAULT_PALETTE.no_data)
    for u in range(FRAME_WIDTH):
        for v in (0, FRAME_HEIGHT - 1):
            if (u, v) != (50, 50):
                assert pixels[v * FRAME_WIDTH + u] == no_data_pixel
    assert pixels[50 * FRAME_WIDTH + 50] != no_data_pixel


def test_all_zero_frame_is_entirely_no_data_colour():
    pixels = depth_to_pixels(ALL_ZERO_FRAME)
    no_data_pixel = bytes(DEFAULT_PALETTE.no_data)
    assert len(pixels) == FRAME_WIDTH * FRAME_HEIGHT
    assert all(pixel == no_data_pixel for pixel in pixels)


def test_no_data_colour_is_never_a_colour_the_ramp_itself_produces():
    """The palette's reserved no-data colour must not double as a real reading."""
    no_data = DEFAULT_PALETTE.no_data
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        assert DEFAULT_PALETTE.color_for(fraction) != no_data


def test_a_single_non_zero_pixel_gets_a_real_colour_not_no_data():
    raw = make_frame({(1, 1): 20000})
    pixels = depth_to_pixels(raw, low=10000, high=30000)
    assert pixels[1 * FRAME_WIDTH + 1] != bytes(DEFAULT_PALETTE.no_data)


# --- percentile stretch: robust to outliers ----------------------------------


def test_explicit_low_high_bypasses_the_percentile_stretch():
    raw = make_frame({(5, 5): 5000})
    at_edge = depth_to_pixels(raw, low=0, high=5000)[5 * FRAME_WIDTH + 5]
    at_far_edge = depth_to_pixels(raw, low=5000, high=10000)[5 * FRAME_WIDTH + 5]
    assert at_edge == bytes(DEFAULT_PALETTE.stops[-1][1])  # fraction == 1.0
    assert at_far_edge == bytes(DEFAULT_PALETTE.stops[0][1])  # fraction == 0.0


def test_default_stretch_is_robust_to_a_single_outlier_code():
    """One wild code must not compress the rest of the frame to one colour.

    199 codes evenly spread over 1000..2000, plus one outlier at 60000. A
    naive min/max stretch would map every one of the 199 real codes into the
    bottom ~2% of the [1000, 60000] range - visually indistinguishable. The
    default percentile stretch must keep them spread out instead.
    """
    overrides: dict[tuple[int, int], int] = {}
    codes = [1000 + round(i * 1000 / 198) for i in range(199)]
    for i, code in enumerate(codes):
        row, col = divmod(i, FRAME_WIDTH)
        overrides[(col, row + 1)] = code  # avoid the top edge (row 0)
    overrides[(0, 300)] = 60000  # the single wild outlier, also interior
    raw = make_frame(overrides)

    pixels = depth_to_pixels(raw)
    median_code = codes[len(codes) // 2]
    median_index = next(i for i, c in enumerate(codes) if c == median_code)
    row, col = divmod(median_index, FRAME_WIDTH)
    median_pixel = pixels[(row + 1) * FRAME_WIDTH + col]

    # Under a naive min/max stretch (span 1000..60000) the median code sits at
    # fraction ~0.0085 - indistinguishable from the very bottom of the ramp.
    naive_pixels = depth_to_pixels(raw, low=1000, high=60000)
    naive_median_pixel = naive_pixels[(row + 1) * FRAME_WIDTH + col]

    assert median_pixel != naive_median_pixel
    # The default stretch should land the median code near the middle of the
    # ramp (green channel near the mid-stop's 200), not near the near-stop's 20.
    assert median_pixel[1] > naive_median_pixel[1] + 50


def test_percentile_stretch_uses_only_non_zero_codes():
    """Zeros must not participate in computing the low/high stretch window."""
    # Only 2 real codes, surrounded by the frame's default zero fill. If zero
    # were included in the percentile computation, the low bound would be
    # pulled to 0 and both real codes would land near the far end together.
    raw = make_frame({(1, 1): 1000, (2, 2): 2000})
    pixels = depth_to_pixels(raw)
    low_pixel = pixels[1 * FRAME_WIDTH + 1]
    high_pixel = pixels[2 * FRAME_WIDTH + 2]
    assert low_pixel != high_pixel


# --- PPM output: parses back to the right dimensions and pixel values -------


def _parse_ppm(blob: bytes) -> tuple[int, int, int, list[tuple[int, int, int]]]:
    """A tiny, from-scratch P6 parser - deliberately not using tkinter."""
    assert blob[:2] == b"P6"
    # Tokens are whitespace-separated; the pixel data starts right after the
    # single whitespace byte that follows the maxval token.
    pos = 2
    tokens: list[int] = []
    while len(tokens) < 3:
        while blob[pos : pos + 1].isspace():
            pos += 1
        start = pos
        while not blob[pos : pos + 1].isspace():
            pos += 1
        tokens.append(int(blob[start:pos]))
    pos += 1  # the single whitespace byte separating header from pixel data
    width, height, maxval = tokens
    body = blob[pos:]
    assert len(body) == width * height * 3
    pixels = [(body[i], body[i + 1], body[i + 2]) for i in range(0, len(body), 3)]
    return width, height, maxval, pixels


def test_ppm_bytes_parses_back_to_the_right_dimensions():
    blob = ppm_bytes(ALL_ZERO_FRAME)
    width, height, maxval, pixels = _parse_ppm(blob)
    assert (width, height) == (FRAME_WIDTH, FRAME_HEIGHT)
    assert maxval == 255
    assert len(pixels) == FRAME_WIDTH * FRAME_HEIGHT


def test_ppm_bytes_pixel_values_match_depth_to_pixels():
    raw = make_frame({(3, 4): 12345, (600, 390): 54321})
    expected = depth_to_pixels(raw)
    _, _, _, pixels = _parse_ppm(ppm_bytes(raw))
    assert pixels == [(p[0], p[1], p[2]) for p in expected]


def test_ppm_bytes_is_exactly_header_plus_three_bytes_per_pixel():
    blob = ppm_bytes(ALL_ZERO_FRAME)
    header_end = blob.index(b"255\n") + len(b"255\n")
    assert len(blob) - header_end == FRAME_WIDTH * FRAME_HEIGHT * 3


# --- Palette ------------------------------------------------------------------


def test_palette_color_for_clamps_out_of_range_fractions():
    assert DEFAULT_PALETTE.color_for(-5.0) == DEFAULT_PALETTE.color_for(0.0)
    assert DEFAULT_PALETTE.color_for(5.0) == DEFAULT_PALETTE.color_for(1.0)


def test_palette_color_for_interpolates_between_stops():
    low = DEFAULT_PALETTE.color_for(0.0)
    mid = DEFAULT_PALETTE.color_for(0.5)
    high = DEFAULT_PALETTE.color_for(1.0)
    assert low != mid != high
    assert low == (20, 20, 120)
    assert mid == (0, 200, 200)
    assert high == (255, 220, 0)
    # Hand-computed against DEFAULT_PALETTE's own stops - not derived from
    # color_for(0.0)/color_for(0.5), so a broken per-channel blend (e.g. one
    # channel silently not interpolated) cannot cancel out against itself.
    assert DEFAULT_PALETTE.color_for(0.25) == (10, 110, 160)
    assert DEFAULT_PALETTE.color_for(0.75) == (128, 210, 100)


def test_custom_palette_is_honoured():
    custom = Palette(no_data=(1, 2, 3), stops=((0.0, (9, 9, 9)), (1.0, (9, 9, 9))))
    raw = make_frame({(0, 1): 500})  # (0, 1) is not an edge sample (row 1)
    pixels = depth_to_pixels(raw, palette=custom)
    assert pixels[0 * FRAME_WIDTH] == bytes((1, 2, 3))  # (0, 0) still zero -> no_data
    assert pixels[1 * FRAME_WIDTH] == bytes((9, 9, 9))


# --- real captured frames -----------------------------------------------------

FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"
FIXTURE_METRIC = Path(__file__).parent / "fixtures" / "metric"


def _manifest(directory: Path) -> dict[str, dict[str, object]]:
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in data["frames"]}


@pytest.mark.parametrize("path", sorted(FIXTURE_FRAMES.glob("*.bin")), ids=lambda p: p.stem)
def test_frame_stats_matches_manifest_on_real_frames(path: Path):
    manifest = _manifest(FIXTURE_FRAMES)
    entry = manifest[path.stem]
    raw = path.read_bytes()
    stats = frame_stats(raw)
    assert stats.samples == entry["samples"]
    assert stats.returned == entry["non_zero_samples"]
    if stats.returned:
        assert stats.max_code == entry["max_code"]
        assert stats.min_code == entry["min_non_zero_code"]


@pytest.mark.parametrize("path", sorted(FIXTURE_METRIC.glob("*.bin")), ids=lambda p: p.stem)
def test_frame_stats_matches_manifest_on_metric_frames(path: Path):
    manifest = _manifest(FIXTURE_METRIC)
    entry = manifest.get(path.stem)
    if entry is None:
        pytest.skip(f"{path.stem} is not listed in {FIXTURE_METRIC / 'manifest.json'}")
    raw = path.read_bytes()
    stats = frame_stats(raw)
    assert stats.returned == entry["non_zero_samples"]
    assert stats.yield_fraction == pytest.approx(entry["yield"], abs=1e-5)


@pytest.mark.parametrize("path", sorted(FIXTURE_FRAMES.glob("*.bin")), ids=lambda p: p.stem)
def test_real_frames_render_and_ppm_round_trips(path: Path):
    raw = path.read_bytes()
    pixels = depth_to_pixels(raw)
    assert len(pixels) == FRAME_WIDTH * FRAME_HEIGHT
    samples = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", raw)
    no_data_pixel = bytes(DEFAULT_PALETTE.no_data)
    zero_index = next((i for i, s in enumerate(samples) if s == 0), None)
    non_zero_index = next((i for i, s in enumerate(samples) if s != 0), None)
    if zero_index is not None:
        assert pixels[zero_index] == no_data_pixel
    if non_zero_index is not None:
        assert pixels[non_zero_index] != no_data_pixel

    blob = ppm_bytes(raw)
    width, height, maxval, parsed_pixels = _parse_ppm(blob)
    assert (width, height, maxval) == (FRAME_WIDTH, FRAME_HEIGHT, 255)
    assert len(parsed_pixels) == FRAME_WIDTH * FRAME_HEIGHT


def test_real_frame_clipping_is_computed_not_guessed():
    """Cross-check against a from-scratch, independent edge count."""
    path = sorted(FIXTURE_FRAMES.glob("*.bin"))[0]
    raw = path.read_bytes()
    samples = struct.unpack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", raw)
    expected_left = sum(1 for v in range(FRAME_HEIGHT) if samples[v * FRAME_WIDTH] != 0)
    expected_right = sum(
        1 for v in range(FRAME_HEIGHT) if samples[v * FRAME_WIDTH + FRAME_WIDTH - 1] != 0
    )
    expected_top = sum(1 for s in samples[:FRAME_WIDTH] if s != 0)
    expected_bottom = sum(1 for s in samples[(FRAME_HEIGHT - 1) * FRAME_WIDTH :] if s != 0)
    stats = frame_stats(raw)
    assert stats.clipped_left == expected_left
    assert stats.clipped_right == expected_right
    assert stats.clipped_top == expected_top
    assert stats.clipped_bottom == expected_bottom


# --- DepthViewer: the Tk window ----------------------------------------------


@pytest.fixture
def viewer():
    # Narrowly scoped on purpose: only "no display" (tkinter.TclError) should
    # turn into a skip. Anything else DepthViewer(...) might raise - e.g. a
    # real InvalidScale bug in this repository's own default - must fail the
    # test, not be swallowed as though the display were unavailable.
    try:
        v = DepthViewer(title="scanner3d test viewer", scale=1)
    except tkinter.TclError as exc:
        pytest.skip(f"no Tk display available to open a window: {exc}")
    yield v
    v.close()


def test_scale_one_the_documented_default_is_accepted(viewer: DepthViewer):
    """scale=1 - what the fixture above and the class default both use - must
    not itself be rejected as invalid."""
    assert viewer is not None  # construction already succeeded to get here


def test_invalid_scale_is_refused_before_any_window_is_opened():
    """No display is required for this: the refusal fires before tkinter.Tk()."""
    for bad_scale in (0, -1, 2.5, "3"):
        with pytest.raises(InvalidScale):
            DepthViewer(scale=bad_scale)  # type: ignore[arg-type]


def test_invalid_scale_rejects_bool_even_though_bool_is_an_int():
    with pytest.raises(InvalidScale):
        DepthViewer(scale=True)


def test_show_accepts_a_real_frame_without_raising(viewer: DepthViewer):
    viewer.show(ALL_ZERO_FRAME)


def test_show_refuses_a_malformed_frame(viewer: DepthViewer):
    with pytest.raises(FrameSizeMismatch):
        viewer.show(b"nope")


def test_close_is_idempotent(viewer: DepthViewer):
    viewer.close()
    viewer.close()  # must not raise


def test_show_after_close_is_refused(viewer: DepthViewer):
    viewer.close()
    with pytest.raises(ViewerWindowClosed):
        viewer.show(ALL_ZERO_FRAME)


def test_show_drops_frames_pushed_faster_than_the_render_budget(viewer: DepthViewer):
    viewer.show(ALL_ZERO_FRAME)
    rendered_before = viewer._rendered_count
    viewer.show(ALL_ZERO_FRAME)  # arrives well within MIN_RENDER_INTERVAL
    assert viewer._rendered_count == rendered_before
    assert viewer._dropped_count >= 1


def test_show_renders_again_once_the_render_budget_has_elapsed(viewer: DepthViewer):
    viewer.show(ALL_ZERO_FRAME)
    rendered_before = viewer._rendered_count
    time.sleep(MIN_RENDER_INTERVAL * 1.5)
    viewer.show(ALL_ZERO_FRAME)
    assert viewer._rendered_count == rendered_before + 1


def test_clipping_label_is_set_when_a_frame_clips_and_cleared_when_it_does_not(
    viewer: DepthViewer,
):
    clipping_frame = make_frame({(0, 10): 500})
    viewer.show(clipping_frame)
    assert "CLIPPING" in viewer._clip_label.cget("text")

    time.sleep(MIN_RENDER_INTERVAL * 1.5)
    viewer.show(ALL_ZERO_FRAME)
    assert viewer._clip_label.cget("text") == ""


# --- png_bytes: parses back to the right dimensions and pixel values --------


def _parse_png(blob: bytes) -> tuple[int, int, int, int, bytes]:
    """A tiny, from-scratch PNG chunk parser - deliberately not using an image library."""
    assert blob[:8] == PNG_SIGNATURE
    pos = 8
    chunks: dict[bytes, bytes] = {}
    while pos < len(blob):
        length = struct.unpack(">I", blob[pos : pos + 4])[0]
        chunk_type = blob[pos + 4 : pos + 8]
        data = blob[pos + 8 : pos + 8 + length]
        (stored_crc,) = struct.unpack(">I", blob[pos + 8 + length : pos + 8 + length + 4])
        assert stored_crc == zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        chunks[chunk_type] = data
        pos += 8 + length + 4  # length field + type + data + CRC
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert compression == 0
    assert filter_method == 0
    assert interlace == 0
    scanlines = zlib.decompress(chunks[b"IDAT"])
    return width, height, bit_depth, color_type, scanlines


def _png_pixels(blob: bytes) -> list[tuple[int, int, int]]:
    """Every pixel of a filter-0, 8-bit truecolour PNG this module produced."""
    width, height, bit_depth, color_type, scanlines = _parse_png(blob)
    assert bit_depth == 8
    assert color_type == 2  # truecolour, no alpha
    stride = 1 + width * 3
    pixels: list[tuple[int, int, int]] = []
    for row in range(height):
        line = scanlines[row * stride : (row + 1) * stride]
        assert line[0] == 0  # filter type 0 (None) on every scanline
        body = line[1:]
        pixels.extend((body[i], body[i + 1], body[i + 2]) for i in range(0, len(body), 3))
    return pixels


def test_png_bytes_signature_and_dimensions():
    blob = png_bytes(ALL_ZERO_FRAME)
    assert blob[:8] == PNG_SIGNATURE
    width, height, bit_depth, color_type, scanlines = _parse_png(blob)
    assert (width, height, bit_depth, color_type) == (FRAME_WIDTH, FRAME_HEIGHT, 8, 2)
    assert len(scanlines) == FRAME_HEIGHT * (1 + FRAME_WIDTH * 3)


def test_png_bytes_pixel_values_match_depth_to_pixels():
    raw = make_frame({(3, 4): 12345, (600, 390): 54321})
    expected = depth_to_pixels(raw)
    assert _png_pixels(png_bytes(raw)) == [(p[0], p[1], p[2]) for p in expected]


def test_png_bytes_all_zero_frame_is_entirely_no_data_colour():
    pixels = _png_pixels(png_bytes(ALL_ZERO_FRAME))
    assert pixels == [DEFAULT_PALETTE.no_data] * (FRAME_WIDTH * FRAME_HEIGHT)


def test_png_bytes_refuses_wrong_size_and_non_bytes():
    with pytest.raises(FrameSizeMismatch):
        png_bytes(b"\x00" * 10)
    with pytest.raises(FrameTypeMismatch):
        png_bytes(None)  # type: ignore[arg-type]


def test_png_bytes_honours_explicit_low_high_and_palette():
    """low/high/palette must reach depth_to_pixels, not be silently dropped."""
    raw = make_frame({(5, 5): 5000})
    custom = Palette(no_data=(1, 2, 3), stops=((0.0, (7, 7, 7)), (1.0, (7, 7, 7))))
    pixels = _png_pixels(png_bytes(raw, low=0, high=5000, palette=custom))
    assert pixels[5 * FRAME_WIDTH + 5] == (7, 7, 7)  # the custom palette's fixed colour
    assert pixels[0] == (1, 2, 3)  # (0, 0) is still zero -> the custom no_data colour


def test_png_bytes_uses_depth_to_pixels_not_a_second_colour_mapping():
    """Same reuse guarantee ppm_bytes has: one place decides what a code looks like."""
    source = Path(viewer_module.__file__).read_text(encoding="utf-8")
    png_start = source.index("def png_bytes(")
    png_end = source.index("\ndef ", png_start + 1)
    assert "depth_to_pixels(" in source[png_start:png_end]


# --- ViewerServer: the local web view -----------------------------------------


@pytest.fixture
def running_server():
    server = ViewerServer(port=0)
    server.start()
    yield server
    server.stop()


def test_viewer_server_binds_to_loopback_by_default():
    """A security property, not a preference: asserted directly, not assumed."""
    server = ViewerServer()
    assert server.host == "127.0.0.1"


def test_viewer_server_bind_all_interfaces_is_an_explicit_opt_in():
    assert ViewerServer(bind_all_interfaces=True).host == "0.0.0.0"
    assert ViewerServer(bind_all_interfaces=False).host == "127.0.0.1"


def test_viewer_server_port_refuses_before_start():
    server = ViewerServer(port=0)
    with pytest.raises(ViewerServerNotRunning):
        _ = server.port


def test_viewer_server_invalid_port_is_refused_before_any_socket_opens():
    for bad_port in (-1, 65536, 70000, 3.5, "8080"):
        with pytest.raises(InvalidPort):
            ViewerServer(port=bad_port)  # type: ignore[arg-type]


def test_viewer_server_port_65535_the_top_of_the_valid_range_is_accepted():
    """65535 is the last valid TCP port; only 65536 and above are out of range."""
    ViewerServer(port=65535)  # must not raise


def test_viewer_server_rejects_bool_port_even_though_bool_is_an_int():
    with pytest.raises(InvalidPort):
        ViewerServer(port=True)


def test_viewer_server_stop_before_start_is_a_no_op():
    ViewerServer(port=0).stop()  # never started; must not raise


def test_viewer_server_stop_is_idempotent():
    server = ViewerServer(port=0)
    server.start()
    server.stop()
    server.stop()  # must not raise


def test_viewer_server_start_while_running_is_a_no_op(running_server: ViewerServer):
    port_before = running_server.port
    running_server.start()  # already running
    assert running_server.port == port_before


def test_index_page_is_served(running_server: ViewerServer):
    conn = http.client.HTTPConnection(running_server.host, running_server.port, timeout=5)
    try:
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200
        assert (resp.getheader("Content-Type") or "").startswith("text/html")
        assert body.lower().startswith(b"<!doctype html>")
    finally:
        conn.close()


def test_unknown_path_is_404(running_server: ViewerServer):
    conn = http.client.HTTPConnection(running_server.host, running_server.port, timeout=5)
    try:
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
    finally:
        conn.close()


def test_frame_and_stats_are_sane_before_any_frame_exists(running_server: ViewerServer):
    """No frame has been shown yet: both endpoints must respond, not crash."""
    conn = http.client.HTTPConnection(running_server.host, running_server.port, timeout=5)
    try:
        conn.request("GET", "/stats.json")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "application/json"
        payload = json.loads(resp.read())
        assert payload["available"] is False
        assert payload["schema_version"] == SCHEMA_VERSION

        conn.request("GET", "/frame.png")
        resp = conn.getresponse()
        assert resp.status == 404  # sane and quiet, not a server error
        resp.read()
    finally:
        conn.close()


def test_show_before_any_prior_frame_does_not_raise(running_server: ViewerServer):
    running_server.show(ALL_ZERO_FRAME)  # the very first frame; must not raise


def test_show_publishes_a_frame_that_frame_png_and_stats_json_then_reflect(
    running_server: ViewerServer,
):
    raw = make_frame({(0, 10): 500})  # a left-edge return: should clip
    running_server.show(raw)

    conn = http.client.HTTPConnection(running_server.host, running_server.port, timeout=5)
    try:
        conn.request("GET", "/frame.png")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "image/png"
        blob = resp.read()
        assert blob[:8] == PNG_SIGNATURE
        assert _png_pixels(blob) == [(p[0], p[1], p[2]) for p in depth_to_pixels(raw)]

        conn.request("GET", "/stats.json")
        resp = conn.getresponse()
        payload = json.loads(resp.read())
        assert payload["available"] is True
        assert payload["frame_number"] == 1
        assert payload["stats"]["clipping"]["left"] == 1
        assert payload["stats"]["clipping"]["any"] is True
    finally:
        conn.close()


def test_stats_json_fps_is_zero_for_the_first_frame_then_positive(
    running_server: ViewerServer,
):
    conn = http.client.HTTPConnection(running_server.host, running_server.port, timeout=5)
    try:
        running_server.show(ALL_ZERO_FRAME)
        conn.request("GET", "/stats.json")
        first = json.loads(conn.getresponse().read())
        assert first["fps"] == 0.0  # no previous frame to measure an interval against

        time.sleep(0.05)
        running_server.show(ALL_ZERO_FRAME)
        conn.request("GET", "/stats.json")
        second = json.loads(conn.getresponse().read())
        # ~1 / 0.05s = ~20 fps; a generous band avoids timing flakiness while
        # still catching a formula that returns the raw interval instead of
        # its reciprocal, or one that is always 0.
        assert 1.0 < second["fps"] < 200.0
    finally:
        conn.close()


def test_viewer_server_show_refuses_a_malformed_frame(running_server: ViewerServer):
    with pytest.raises(FrameSizeMismatch):
        running_server.show(b"nope")
    with pytest.raises(FrameTypeMismatch):
        running_server.show(None)  # type: ignore[arg-type]


def test_two_shows_bump_the_frame_number_and_do_not_accumulate_a_buffer(
    running_server: ViewerServer,
):
    running_server.show(ALL_ZERO_FRAME)
    running_server.show(make_frame({(1, 1): 999}))

    conn = http.client.HTTPConnection(running_server.host, running_server.port, timeout=5)
    try:
        conn.request("GET", "/stats.json")
        payload = json.loads(conn.getresponse().read())
        # Only the frame count grows; the server keeps exactly one frame, not
        # a history of the ones already shown.
        assert payload["frame_number"] == 2
    finally:
        conn.close()
