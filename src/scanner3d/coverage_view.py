"""Turn a :class:`~scanner3d.coverage.CoverageReport` into pixels a person can act on.

The problem this solves
------------------------
The user's own words: "you have to know what you've scanned, what needs
better scanning." :mod:`~scanner3d.coverage` computes exactly that - which
cells are well observed and which are thin, and why - but a
:class:`~scanner3d.coverage.CoverageReport` is a data structure. Nobody
standing at a scanner deciding where to point it next can read one. This
module is the missing step: it renders a report as something to look at,
reusing :mod:`~scanner3d.viewer`'s existing false-colour PNG machinery rather
than writing a second image encoder.

Two different problems, two different colours
-----------------------------------------------
:mod:`~scanner3d.coverage`'s whole point is that a thin cell can fail for two
unrelated reasons, and the fix is different for each:

* too few observations - the scanner has not looked here enough. The fix is
  to point it here and capture more frames.
* enough observations but not enough angular spread - the scanner has looked
  here plenty, always from close to the same direction. Capturing MORE frames
  from the SAME spot does not fix this; the fix is to move somewhere else and
  look again.

Colouring both the same "thin" would tell a user to do the wrong thing half
the time, which is worse than not colouring them at all - it would look like
useful guidance while actively misleading. So this module always renders
three categories, never two: well covered, thin by observation count, thin by
angular spread. See :func:`_category_of`.

Why three orthogonal views, not one
-------------------------------------
A single fixed viewpoint hides whatever is behind the object from that
viewpoint - exactly the failure mode this whole feature exists to prevent,
just moved from the object to the picture of the object. Rendering front,
top and side views instead means a region hidden behind the object in one
view is in full view in (at least) one of the other two. It is still not
complete: a thin patch on, say, the object's own far side away from all three
canonical axes can be foreshortened or thinned across all three - see
:class:`CoverageDisplay`'s own docstring for what this does NOT show.

Why the canvas is a fixed 640x400
-----------------------------------
:func:`~scanner3d.viewer.png_bytes` only ever encodes one fixed frame shape -
the depth sensor's own 640x400 - so reusing it rather than writing a second
PNG encoder means every rendered view is that same fixed size. Each category
is encoded as a small integer "depth code" (1, 2 or 3) run through a
:class:`~scanner3d.viewer.Palette` whose three stops sit at exactly 0.0, 0.5
and 1.0, so :func:`~scanner3d.viewer.depth_to_pixels`'s linear interpolation
lands exactly on each stop's own colour with no blending - and an unoccupied
pixel is code 0, which :func:`~scanner3d.viewer.depth_to_pixels` already
renders as its reserved "no data" colour, before any low/high stretch is
applied. The one caller-adjustable size in this module is
:data:`point_radius_px` - how many pixels wide each rendered cell marker is -
not the canvas itself, which reuse fixes.

What this does NOT do
----------------------
It does not show individual cells at their true relative size, only as fixed-
radius markers - a coverage grid can hold thousands of cells and a real
point-per-cell rendering at 640x400 would be unreadably sparse. It does not
resolve overlap by blending: when two cells project onto the same pixel from
different depths along the viewing axis, the worse category wins outright (see
:data:`_CATEGORY_PRIORITY`), so a view can show a region as thinner than the
single pixel a viewer clicks on actually is at every depth - it is a
conservative display, not an exact one. It carries no notion of the object's
own silhouette or surface, only cell centroids as points in space, so nothing
here can show which side of the object a thin patch is actually on beyond
which of the three views it appears in. And it inherits every limitation
:class:`~scanner3d.coverage.CoverageReport` itself documents - this module
adds nothing to what coverage already knows, it only shows it.

Capture-agnostic by design: nothing here assumes a turntable, an axis, or any
particular capture pattern. It draws whatever cells and regions the report
handed to it wherever their centroids sit in the caller's own world frame.
"""

from __future__ import annotations

import enum
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.coverage import CoverageCell, CoverageReport, ThinRegion
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH
from scanner3d.viewer import Palette, png_bytes

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: Fraction of the canvas left empty on every side, so a marker at the exact
#: edge of the data's own extent is not clipped by the frame boundary.
_MARGIN_FRACTION: Final = 0.06

#: The synthetic "depth code" each category is encoded as before being run
#: through :func:`~scanner3d.viewer.png_bytes`. 0 is reserved - it is
#: :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE`, which
#: :func:`~scanner3d.viewer.depth_to_pixels` already renders as "no data"
#: before any stretch, which is exactly what an unoccupied pixel is here.
_BACKGROUND_CODE: Final = 0
_WELL_CODE: Final = 1
_THIN_OBSERVATIONS_CODE: Final = 2
_THIN_SPREAD_CODE: Final = 3


class Category(enum.Enum):
    """Why a cell is drawn the way it is - the fact this whole module exists to show."""

    WELL_COVERED = "well_covered"
    THIN_OBSERVATIONS = "thin_observations"
    THIN_SPREAD = "thin_spread"


def _category_of(cell: CoverageCell, report: CoverageReport) -> Category:
    """Which of the two thin reasons applies, or neither.

    Observation count is checked first: a cell seen too few times has not
    earned an opinion about its angular spread yet, so a cell failing both
    thresholds at once is reported as the count problem - the more basic of
    the two failures, and the one whose fix (point the scanner here) also
    creates the opportunity to fix the other.
    """
    if cell.well_covered:
        return Category.WELL_COVERED
    if cell.observations < report.min_observations:
        return Category.THIN_OBSERVATIONS
    return Category.THIN_SPREAD


#: Worse (more worth showing over whatever else projects onto the same pixel)
#: sorts higher. THIN_OBSERVATIONS outranks THIN_SPREAD: a region with no
#: usable observation at all is a bigger gap than one that has been seen,
#: just not from enough directions - see the module docstring's "why three
#: categories" section for why the two are never merged into one "thin".
_CATEGORY_PRIORITY: Final[dict[Category, int]] = {
    Category.WELL_COVERED: 0,
    Category.THIN_SPREAD: 1,
    Category.THIN_OBSERVATIONS: 2,
}

_CATEGORY_CODE: Final[dict[Category, int]] = {
    Category.WELL_COVERED: _WELL_CODE,
    Category.THIN_OBSERVATIONS: _THIN_OBSERVATIONS_CODE,
    Category.THIN_SPREAD: _THIN_SPREAD_CODE,
}

#: Green (covered), amber (too few looks - go there), magenta (enough looks,
#: wrong angle - go somewhere else). Chosen for three-way contrast rather than
#: relative "goodness", since amber and magenta are both instructions to move,
#: just to different places, not points on one severity scale.
WELL_COLOUR: Final = (40, 180, 90)
THIN_OBSERVATIONS_COLOUR: Final = (255, 170, 0)
THIN_SPREAD_COLOUR: Final = (210, 60, 220)

_VIEW_PALETTE: Final = Palette(
    no_data=(0, 0, 0),
    stops=(
        (0.0, WELL_COLOUR),
        (0.5, THIN_OBSERVATIONS_COLOUR),
        (1.0, THIN_SPREAD_COLOUR),
    ),
)

#: One entry per rendered view: a name, and which centroid axes (0=x, 1=y,
#: 2=z) map to screen-horizontal and screen-vertical. "top" looks down the
#: y-axis, "front" down z, "side" down x - the three axis-aligned views a
#: single fixed viewpoint cannot substitute for, see the module docstring.
_VIEWS: Final[tuple[tuple[str, int, int], ...]] = (
    ("top", 0, 2),
    ("front", 0, 1),
    ("side", 2, 1),
)

_AXIS_NAME: Final[tuple[str, str, str]] = ("x", "y", "z")


class CoverageViewNotAllowed(ScannerError):
    """A coverage rendering request was made that this module refuses to answer."""

    status = DoctorStatus.UNSUPPORTED_MODE


class EmptyReport(CoverageViewNotAllowed):
    """A report with no occupied cells was offered - there is nothing to draw."""


class DegenerateExtent(CoverageViewNotAllowed):
    """Every cell shares the same coordinate on at least one axis.

    Every one of the three orthogonal views uses two of the three axes, so a
    zero-extent axis makes at least one view's scale factor a division by
    zero - refused here rather than answered with an arbitrarily inflated or
    NaN scale.
    """


class InvalidPointRadius(CoverageViewNotAllowed):
    """``point_radius_px`` was not a positive integer.

    The canvas itself is a fixed 640x400 - see the module docstring for why -
    so this is the one size a caller actually controls: how many pixels wide
    each rendered cell marker is.
    """


@dataclass(frozen=True, slots=True)
class ThinRegionSummary:
    """One thin region, in words: where it is and which fix it needs."""

    centroid: tuple[float, float, float]
    cell_count: int
    category: Category
    worst_observations: int
    worst_spread_radians: float

    def describe(self) -> str:
        x, y, z = self.centroid
        position = f"near (x={x:.1f}, y={y:.1f}, z={z:.1f})"
        if self.category is Category.THIN_OBSERVATIONS:
            return (
                f"{self.cell_count} cell(s) {position}: too few observations (worst cell seen "
                f"only {self.worst_observations} time(s)) - point the scanner here and capture "
                "more frames"
            )
        degrees = math.degrees(self.worst_spread_radians)
        return (
            f"{self.cell_count} cell(s) {position}: observed enough times "
            f"({self.worst_observations}) but only from a {degrees:.1f} degree spread of "
            "directions - move to a different position and scan again; more frames from here "
            "will not fix this"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "centroid": list(self.centroid),
            "cell_count": self.cell_count,
            "category": self.category.value,
            "worst_observations": self.worst_observations,
            "worst_spread_radians": self.worst_spread_radians,
            "description": self.describe(),
        }


def _summarise_region(region: ThinRegion, report: CoverageReport) -> ThinRegionSummary:
    category = (
        Category.THIN_OBSERVATIONS
        if region.worst_observations < report.min_observations
        else Category.THIN_SPREAD
    )
    return ThinRegionSummary(
        centroid=region.centroid,
        cell_count=region.cell_count,
        category=category,
        worst_observations=region.worst_observations,
        worst_spread_radians=region.worst_spread,
    )


@dataclass(frozen=True, slots=True)
class CoverageView:
    """One orthogonal projection of a coverage report, rendered as a PNG.

    ``png`` is excluded from :meth:`as_dict` - metadata belongs beside the
    image, never in place of it, matching
    :class:`~scanner3d.pointcloud.EncodedCloud`'s own reasoning for the same
    exclusion.
    """

    name: str
    horizontal_axis: str
    vertical_axis: str
    world_bounds: tuple[float, float, float, float]
    png: bytes
    cell_counts: dict[Category, int]

    @property
    def byte_count(self) -> int:
        return len(self.png)

    def as_dict(self) -> dict[str, object]:
        min_h, max_h, min_v, max_v = self.world_bounds
        return {
            "name": self.name,
            "horizontal_axis": self.horizontal_axis,
            "vertical_axis": self.vertical_axis,
            "world_bounds": {"min_h": min_h, "max_h": max_h, "min_v": min_v, "max_v": max_v},
            "png_byte_count": self.byte_count,
            "cell_counts": {category.value: count for category, count in self.cell_counts.items()},
        }


@dataclass(frozen=True, slots=True)
class CoverageDisplay:
    """Three orthogonal views of a coverage report, plus the text a user reads first.

    What this does NOT show
    ------------------------
    A patch thin in every one of the three axis-aligned views can still be
    hidden or foreshortened in all three at once - three views narrow the
    blind spot a single viewpoint has, they do not eliminate it. Nothing here
    is a substitute for :meth:`~scanner3d.coverage.CoverageReport.thin_regions`
    itself, which has no viewpoint at all to hide behind.
    """

    views: tuple[CoverageView, ...]
    summary: str
    thin_regions: tuple[ThinRegionSummary, ...]
    well_covered_fraction: float
    total_cells: int
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "views": [view.as_dict() for view in self.views],
            "summary": self.summary,
            "thin_regions": [region.as_dict() for region in self.thin_regions],
            "well_covered_fraction": self.well_covered_fraction,
            "total_cells": self.total_cells,
            "limitations": list(self.limitations),
        }


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    return min(values), max(values)


def _render_view(
    report: CoverageReport,
    categories: Sequence[Category],
    *,
    horizontal_index: int,
    vertical_index: int,
    name: str,
    point_radius_px: int,
) -> CoverageView:
    """One orthogonal projection: cell centroids, coloured by category, as a PNG.

    Screen rows increase as the vertical world axis DECREASES, for every
    view, so "up" in the room is "up" on screen consistently across all
    three - including "top", where the vertical screen axis is world z and
    has no physical "up" meaning of its own; the convention is kept anyway so
    a reader does not have to learn a different rule per view.
    """
    horizontal = [cell.centroid[horizontal_index] for cell in report.cells]
    vertical = [cell.centroid[vertical_index] for cell in report.cells]
    min_h, max_h = _bounds(horizontal)
    min_v, max_v = _bounds(vertical)
    extent_h = max_h - min_h
    extent_v = max_v - min_v

    usable_w = FRAME_WIDTH * (1.0 - 2.0 * _MARGIN_FRACTION)
    usable_h = FRAME_HEIGHT * (1.0 - 2.0 * _MARGIN_FRACTION)
    # Guarded by DegenerateExtent before this runs, so extent_h/extent_v are
    # both strictly positive here - see render_coverage_display.
    scale = min(usable_w / extent_h, usable_h / extent_v)
    offset_x = (FRAME_WIDTH - extent_h * scale) / 2.0
    offset_y = (FRAME_HEIGHT - extent_v * scale) / 2.0

    codes = [_BACKGROUND_CODE] * (FRAME_WIDTH * FRAME_HEIGHT)
    priority = [-1] * (FRAME_WIDTH * FRAME_HEIGHT)
    counts: dict[Category, int] = dict.fromkeys(Category, 0)

    for cell, category in zip(report.cells, categories, strict=True):
        counts[category] += 1
        h = cell.centroid[horizontal_index]
        v = cell.centroid[vertical_index]
        px = round(offset_x + (h - min_h) * scale)
        # Vertical axis flipped: see the docstring above.
        py = round(offset_y + (max_v - v) * scale)
        rank = _CATEGORY_PRIORITY[category]
        code = _CATEGORY_CODE[category]
        for dy in range(-point_radius_px, point_radius_px + 1):
            y = py + dy
            if not (0 <= y < FRAME_HEIGHT):
                continue
            row_base = y * FRAME_WIDTH
            for dx in range(-point_radius_px, point_radius_px + 1):
                x = px + dx
                if not (0 <= x < FRAME_WIDTH):
                    continue
                index = row_base + x
                if rank >= priority[index]:
                    priority[index] = rank
                    codes[index] = code

    raw = struct.pack(f"<{FRAME_WIDTH * FRAME_HEIGHT}H", *codes)
    # Internal shape guarantee (a list of exactly FRAME_WIDTH*FRAME_HEIGHT
    # uint16 codes), not a check on caller input - png_bytes enforces this
    # length itself and would raise FrameSizeMismatch if it were ever wrong.
    assert len(raw) == EXPECTED_FRAME_BYTES
    png = png_bytes(raw, low=_WELL_CODE, high=_THIN_SPREAD_CODE, palette=_VIEW_PALETTE)

    return CoverageView(
        name=name,
        horizontal_axis=_AXIS_NAME[horizontal_index],
        vertical_axis=_AXIS_NAME[vertical_index],
        world_bounds=(min_h, max_h, min_v, max_v),
        png=png,
        cell_counts=counts,
    )


def render_coverage_display(
    report: CoverageReport,
    *,
    point_radius_px: int = 3,
    max_regions_in_summary: int = 5,
) -> CoverageDisplay:
    """Render ``report`` as three orthogonal PNG views plus a text summary.

    Raises :class:`EmptyReport` if the report has no occupied cells, and
    :class:`DegenerateExtent` if every cell shares a coordinate on any one
    axis - both refused before any pixel is touched. See the module docstring
    for the projection, the colour scheme, and what this cannot show.
    """
    if not report.cells:
        raise EmptyReport("the report has no occupied cells; there is nothing to render")
    if (
        not isinstance(point_radius_px, int)
        or isinstance(point_radius_px, bool)
        or (point_radius_px < 1)
    ):
        raise InvalidPointRadius(
            f"point_radius_px must be a positive integer; got {point_radius_px!r}"
        )

    for axis_index in range(3):
        values = [cell.centroid[axis_index] for cell in report.cells]
        lo, hi = _bounds(values)
        if hi <= lo:
            raise DegenerateExtent(
                f"every cell shares coordinate {_AXIS_NAME[axis_index]}={lo!r}; at least one of "
                "the three orthogonal views would need to divide by a zero extent"
            )

    categories = [_category_of(cell, report) for cell in report.cells]
    views = tuple(
        _render_view(
            report,
            categories,
            horizontal_index=h_index,
            vertical_index=v_index,
            name=name,
            point_radius_px=point_radius_px,
        )
        for name, h_index, v_index in _VIEWS
    )

    thin_regions = tuple(
        _summarise_region(region, report)
        for region in report.thin_regions()[:max_regions_in_summary]
    )
    well = sum(1 for category in categories if category is Category.WELL_COVERED)
    thin_observations = sum(1 for category in categories if category is Category.THIN_OBSERVATIONS)
    thin_spread = len(categories) - well - thin_observations

    lines = [
        f"{report.total_cells} cells total: {well} well covered, {thin_observations} thin "
        f"(too few observations), {thin_spread} thin (not enough viewing angle).",
    ]
    if thin_regions:
        lines.append("Largest thin regions, worst first:")
        lines.extend(f"  - {region.describe()}" for region in thin_regions)
    else:
        lines.append("No thin regions - every occupied cell meets both thresholds.")
    summary = "\n".join(lines)

    limitations = (
        "Cells are drawn as fixed-radius markers, not at their true relative size - a coverage "
        "grid can hold thousands of cells, and drawing each at true scale on a 640x400 canvas "
        "would be unreadable.",
        "Where multiple cells project onto the same pixel from different depths, the worse "
        "category wins outright - a view can make a region look thinner than every depth along "
        "that line of sight actually is; it never makes one look better than its worst cell.",
        "Three axis-aligned views narrow the single-viewpoint blind spot, they do not eliminate "
        "it - a patch thin from all three axes can still be hidden or foreshortened in every "
        "view at once.",
        "This module adds no information beyond what CoverageReport already computed; every "
        "limitation that report documents about count, spread and cell geometry applies here "
        "unchanged.",
    )

    return CoverageDisplay(
        views=views,
        summary=summary,
        thin_regions=thin_regions,
        well_covered_fraction=report.well_covered_fraction,
        total_cells=report.total_cells,
        limitations=limitations,
    )
