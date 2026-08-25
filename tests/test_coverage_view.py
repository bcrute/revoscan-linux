"""Tests for coverage_view: rendering a CoverageReport as something to act on.

What is at stake
-----------------
:mod:`~scanner3d.coverage` computes coverage; this module is the only thing
that shows it to a person deciding where to point the scanner next. Two
properties matter most, and the tests are built to fail without them:

* **A thin region has to render at the position it actually occupies**, not
  merely change some pixel somewhere. :func:`test_thin_region_renders_at_its_actual_position`
  computes the exact pixel a known cell projects to, by the same formula the
  module's own docstring states, and checks that pixel specifically, plus
  pixels far from it - a renderer that coloured the whole image thin, or the
  wrong corner, would still pass a weaker "some pixel changed" test.
* **Thin-by-count and thin-by-spread must be visually distinguishable.** They
  are different failures needing different fixes - see coverage_view's own
  module docstring - and a renderer that collapsed them into one "thin"
  colour would silently send a user to do the wrong thing half the time.
  :func:`test_thin_by_count_and_thin_by_spread_render_distinguishably` is the
  test that catches that collapse.

Test data
---------
Most tests build a :class:`~scanner3d.coverage.CoverageReport` directly from
hand-placed :class:`~scanner3d.coverage.CoverageCell` objects rather than
running :func:`~scanner3d.coverage.coverage_map` on a rendered scene: what is
under test here is the projection and colouring, not coverage's own geometry
- that is exercised in :mod:`test_coverage` already - and hand-placed cells
give exact, known positions to check pixels against instead of ones that
would have to be recomputed from a simulated capture.
"""

from __future__ import annotations

import math
import struct
import zlib

import pytest

from scanner3d.coverage import CoverageCell, CoverageReport
from scanner3d.coverage_view import (
    SCHEMA_VERSION,
    THIN_OBSERVATIONS_COLOUR,
    THIN_SPREAD_COLOUR,
    WELL_COLOUR,
    Category,
    CoverageViewNotAllowed,
    DegenerateExtent,
    EmptyReport,
    InvalidPointRadius,
    render_coverage_display,
)
from scanner3d.v4l2_capture import FRAME_HEIGHT, FRAME_WIDTH
from scanner3d.viewer import PNG_SIGNATURE

# --------------------------------------------------------------------------
# PNG decoding - the same from-scratch chunk parser test_viewer.py uses.
# --------------------------------------------------------------------------


def _parse_png(blob: bytes) -> tuple[int, int, int, int, bytes]:
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
        pos += 8 + length + 4
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert compression == 0
    assert filter_method == 0
    assert interlace == 0
    scanlines = zlib.decompress(chunks[b"IDAT"])
    return width, height, bit_depth, color_type, scanlines


def _pixel(scanlines: bytes, width: int, px: int, py: int) -> tuple[int, int, int]:
    stride = 1 + width * 3
    line = scanlines[py * stride : (py + 1) * stride]
    assert line[0] == 0  # filter type 0 on every scanline
    body = line[1:]
    i = px * 3
    return (body[i], body[i + 1], body[i + 2])


def _all_pixels(scanlines: bytes, width: int, height: int) -> list[tuple[int, int, int]]:
    stride = 1 + width * 3
    pixels: list[tuple[int, int, int]] = []
    for row in range(height):
        line = scanlines[row * stride : (row + 1) * stride]
        body = line[1:]
        pixels.extend((body[i], body[i + 1], body[i + 2]) for i in range(0, len(body), 3))
    return pixels


def _projected_pixel(
    world_bounds: tuple[float, float, float, float], h: float, v: float
) -> tuple[int, int]:
    """The pixel a (horizontal, vertical) world coordinate projects to.

    Reimplements render_coverage_display's own stated projection formula (see
    its module docstring: uniform scale fit to a 6%-margin canvas, vertical
    axis flipped) independently of the module's private helpers, so this is a
    check against the documented behaviour, not a restatement of the code.
    """
    min_h, max_h, min_v, max_v = world_bounds
    margin = 0.06
    usable_w = FRAME_WIDTH * (1.0 - 2.0 * margin)
    usable_h = FRAME_HEIGHT * (1.0 - 2.0 * margin)
    scale = min(usable_w / (max_h - min_h), usable_h / (max_v - min_v))
    offset_x = (FRAME_WIDTH - (max_h - min_h) * scale) / 2.0
    offset_y = (FRAME_HEIGHT - (max_v - min_v) * scale) / 2.0
    px = round(offset_x + (h - min_h) * scale)
    py = round(offset_y + (max_v - v) * scale)
    return px, py


# --------------------------------------------------------------------------
# Hand-built reports with known cell positions.
# --------------------------------------------------------------------------


def _cell(
    centroid: tuple[float, float, float],
    *,
    observations: int,
    spread: float,
    well_covered: bool,
    index: tuple[int, int, int] = (0, 0, 0),
) -> CoverageCell:
    return CoverageCell(
        cell_index=index,
        centroid=centroid,
        observations=observations,
        spread=spread,
        well_covered=well_covered,
    )


_MIN_OBSERVATIONS = 3
_MIN_SPREAD = math.radians(20.0)


def _three_category_report() -> CoverageReport:
    """One well-covered cell, one thin-by-count, one thin-by-spread, all at known positions."""
    well = _cell(
        (10.0, 0.0, 5.0),
        observations=5,
        spread=math.radians(30.0),
        well_covered=True,
        index=(0, 0, 0),
    )
    thin_count = _cell(
        (-10.0, 0.0, -5.0), observations=1, spread=0.0, well_covered=False, index=(1, 0, 0)
    )
    thin_spread = _cell(
        (0.0, 10.0, 0.0),
        observations=5,
        spread=math.radians(5.0),
        well_covered=False,
        # Not face-adjacent to thin_count's (1, 0, 0) - each stays its own
        # ThinRegion, so the region-level summary text (which labels a whole
        # region by its worst cell, not cell by cell) reflects each category
        # separately instead of one region's label masking the other.
        index=(10, 0, 0),
    )
    return CoverageReport(
        cells=(well, thin_count, thin_spread),
        cell_size=1.0,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
        limitations=(),
    )


def _fully_covered_report(count: int = 20) -> CoverageReport:
    """A grid of cells, every one well covered, spanning all three axes."""
    cells = tuple(
        _cell(
            (float(i), float((i * 3) % 7), float((i * 5) % 11)),
            observations=6,
            spread=math.radians(60.0),
            well_covered=True,
            index=(i, 0, 0),
        )
        for i in range(count)
    )
    return CoverageReport(
        cells=cells,
        cell_size=1.0,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
        limitations=(),
    )


# --------------------------------------------------------------------------
# Position and distinguishability - the properties that matter most.
# --------------------------------------------------------------------------


def test_thin_region_renders_at_its_actual_position():
    report = _three_category_report()
    display = render_coverage_display(report, point_radius_px=3)
    top = next(view for view in display.views if view.name == "top")
    _width, _height, _bit_depth, _color_type, scanlines = _parse_png(top.png)

    thin_count_px, thin_count_py = _projected_pixel(top.world_bounds, -10.0, -5.0)
    well_px, well_py = _projected_pixel(top.world_bounds, 10.0, 5.0)

    assert _pixel(scanlines, FRAME_WIDTH, thin_count_px, thin_count_py) == THIN_OBSERVATIONS_COLOUR
    # Far from the thin cell's own projected position, the thin colour must
    # not appear - proving this is a positional render, not a global tint.
    assert _pixel(scanlines, FRAME_WIDTH, well_px, well_py) != THIN_OBSERVATIONS_COLOUR
    assert _pixel(scanlines, FRAME_WIDTH, 0, 0) == (0, 0, 0)  # far corner: background


def test_thin_by_count_and_thin_by_spread_render_distinguishably():
    """The test that fails if the two thin reasons were ever collapsed into one colour."""
    report = _three_category_report()
    display = render_coverage_display(report, point_radius_px=3)
    top = next(view for view in display.views if view.name == "top")
    _w, _h, _bd, _ct, scanlines = _parse_png(top.png)

    count_px, count_py = _projected_pixel(top.world_bounds, -10.0, -5.0)
    spread_px, spread_py = _projected_pixel(top.world_bounds, 0.0, 0.0)
    well_px, well_py = _projected_pixel(top.world_bounds, 10.0, 5.0)

    count_pixel = _pixel(scanlines, FRAME_WIDTH, count_px, count_py)
    spread_pixel = _pixel(scanlines, FRAME_WIDTH, spread_px, spread_py)
    well_pixel = _pixel(scanlines, FRAME_WIDTH, well_px, well_py)

    assert count_pixel == THIN_OBSERVATIONS_COLOUR
    assert spread_pixel == THIN_SPREAD_COLOUR
    assert well_pixel == WELL_COLOUR
    assert len({count_pixel, spread_pixel, well_pixel}) == 3, (
        "all three categories must render as three distinct colours"
    )


def test_fully_covered_report_renders_no_thin_colour_anywhere():
    report = _fully_covered_report()
    display = render_coverage_display(report, point_radius_px=2)
    for view in display.views:
        width, height, _bd, _ct, scanlines = _parse_png(view.png)
        pixels = _all_pixels(scanlines, width, height)
        assert THIN_OBSERVATIONS_COLOUR not in pixels
        assert THIN_SPREAD_COLOUR not in pixels


def test_summary_and_categories_agree_with_the_rendered_cell_counts():
    report = _three_category_report()
    display = render_coverage_display(report)
    assert "too few observations" in display.summary
    assert "not enough viewing angle" in display.summary
    assert len(display.thin_regions) == 2
    top = next(view for view in display.views if view.name == "top")
    assert top.cell_counts[Category.WELL_COVERED] == 1
    assert top.cell_counts[Category.THIN_OBSERVATIONS] == 1
    assert top.cell_counts[Category.THIN_SPREAD] == 1


def test_each_thin_region_is_labelled_with_its_own_reason_not_the_other_ones():
    """Not just that both reasons appear somewhere - each region names the right one."""
    report = _three_category_report()
    display = render_coverage_display(report)
    by_position = {region.centroid: region for region in display.thin_regions}

    count_region = by_position[(-10.0, 0.0, -5.0)]
    assert count_region.category is Category.THIN_OBSERVATIONS
    assert "too few observations" in count_region.describe()
    assert "spread of directions" not in count_region.describe()

    spread_region = by_position[(0.0, 10.0, 0.0)]
    assert spread_region.category is Category.THIN_SPREAD
    assert "spread of directions" in spread_region.describe()
    assert "too few observations" not in spread_region.describe()


def test_worse_category_wins_when_two_cells_project_to_the_same_pixel():
    """THIN_OBSERVATIONS must win over THIN_SPREAD on overlap, not the reverse.

    Two cells placed to share the exact same (x, z) - and therefore the same
    pixel in the "top" view - differing only in y, which "top" does not use
    at all. If the wrong category won this overlap, a user would be told a
    region only needs a different angle when it has not been looked at
    enough times yet at all.
    """
    thin_observations = _cell(
        (0.0, 0.0, 0.0), observations=1, spread=0.0, well_covered=False, index=(0, 0, 0)
    )
    thin_spread = _cell(
        (0.0, 5.0, 0.0),
        observations=5,
        spread=math.radians(5.0),
        well_covered=False,
        index=(1, 0, 0),
    )
    anchor = _cell(
        (8.0, 0.0, 8.0),
        observations=5,
        spread=math.radians(30.0),
        well_covered=True,
        index=(2, 0, 0),
    )
    report = CoverageReport(
        cells=(thin_observations, thin_spread, anchor),
        cell_size=1.0,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
        limitations=(),
    )
    display = render_coverage_display(report, point_radius_px=1)
    top = next(view for view in display.views if view.name == "top")
    _w, _h, _bd, _ct, scanlines = _parse_png(top.png)
    px, py = _projected_pixel(top.world_bounds, 0.0, 0.0)
    assert _pixel(scanlines, FRAME_WIDTH, px, py) == THIN_OBSERVATIONS_COLOUR
    assert display.total_cells == 3
    assert display.well_covered_fraction == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------
# Valid PNG output.
# --------------------------------------------------------------------------


def test_every_view_is_a_valid_png_of_the_frame_shape():
    report = _three_category_report()
    display = render_coverage_display(report)
    assert {view.name for view in display.views} == {"top", "front", "side"}
    for view in display.views:
        assert view.png[:8] == PNG_SIGNATURE
        width, height, bit_depth, color_type, scanlines = _parse_png(view.png)
        assert (width, height, bit_depth, color_type) == (FRAME_WIDTH, FRAME_HEIGHT, 8, 2)
        assert len(scanlines) == FRAME_HEIGHT * (1 + FRAME_WIDTH * 3)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_render_coverage_display_raises_for_empty_report():
    empty = CoverageReport(
        cells=(),
        cell_size=1.0,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
        limitations=(),
    )
    with pytest.raises(EmptyReport):
        render_coverage_display(empty)


@pytest.mark.parametrize("flat_axis", [0, 1, 2])
def test_render_coverage_display_raises_for_degenerate_extent(flat_axis: int):
    base: list[tuple[float, float, float]] = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    flattened: list[tuple[float, float, float]] = [
        (
            0.0 if flat_axis == 0 else p[0],
            0.0 if flat_axis == 1 else p[1],
            0.0 if flat_axis == 2 else p[2],
        )
        for p in base
    ]
    cells = tuple(
        _cell(p, observations=5, spread=math.radians(30.0), well_covered=True, index=(i, 0, 0))
        for i, p in enumerate(flattened)
    )
    report = CoverageReport(
        cells=cells,
        cell_size=1.0,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
        limitations=(),
    )
    with pytest.raises(DegenerateExtent):
        render_coverage_display(report)


@pytest.mark.parametrize("bad_radius", [0, -1, -5, 2.5, True, False])
def test_render_coverage_display_raises_for_invalid_point_radius(bad_radius: object):
    report = _three_category_report()
    with pytest.raises(InvalidPointRadius):
        render_coverage_display(report, point_radius_px=bad_radius)  # type: ignore[arg-type]


def test_guard_exceptions_are_coverage_view_not_allowed():
    assert issubclass(EmptyReport, CoverageViewNotAllowed)
    assert issubclass(DegenerateExtent, CoverageViewNotAllowed)
    assert issubclass(InvalidPointRadius, CoverageViewNotAllowed)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_coverage_display_as_dict_round_trips_and_carries_schema_version():
    import json

    report = _three_category_report()
    display = render_coverage_display(report)
    record = json.loads(json.dumps(display.as_dict()))
    assert record["schema_version"] == SCHEMA_VERSION
    assert len(record["views"]) == 3
    for view_record in record["views"]:
        assert "png" not in view_record  # binary payload deliberately excluded
        assert view_record["png_byte_count"] > 0
    assert record["total_cells"] == 3
    assert record["thin_regions"]
    assert record["limitations"]
