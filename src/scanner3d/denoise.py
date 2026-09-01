"""Dropping the points nothing else in the frame agrees with.

The problem this answers
------------------------
A depth frame from this scanner does not only carry surface. It carries stray
samples: a flying pixel straddling a depth edge and landing halfway between the
object and the wall behind it, a multipath return from a concave corner, a
speckle artefact at grazing incidence. Every one of them deprojects into a
coordinate as confidently as a real sample does, and nothing upstream of this
module removes any of them - :func:`scanner3d.depth_metric.deproject_depth_frame`
converts what the sensor sent and accounts for what it dropped, which is exactly
what it should do and is not a filter.

What that costs downstream is not only cosmetic. Registration's correspondence
search pairs a stray with whatever is nearest to it in the next frame, so a
handful of them pull on the fitted pose; the mesh's occupancy grid marks their
cells solid; and the live model draws them as dots floating around the object,
which is the form the operator actually notices.

What this does
--------------
One test, applied per point: how many other points of the same cloud lie in the
block of grid cells around its own. Below a threshold the point is removed. That
is all of it - no surface fitting, no normal estimation, no statistical model of
the noise, and no neighbour search that is not a dictionary lookup, because this
runs once per captured frame on the thread that is between two ``VIDIOC_DQBUF``
calls and anything expensive there is a dropped frame.

The neighbourhood is a block of CELLS, not a ball centred on the point. That is
a deliberate simplification and it has one consequence worth stating plainly:
the region tested is grid-aligned, so it extends between one and two cells from
the point depending on where in its cell the point sits. In exchange the count
depends only on the cell, so every point of a cell shares one verdict and the
count is computed once per occupied cell rather than once per point. The
memoisation is exact - it is the same answer, three to six times faster - not an
approximation of a per-point search.

The threshold comes from the cloud, not from a constant
--------------------------------------------------------
An absolute neighbourhood size is wrong here, because the clouds this filters
have already been thinned by their caller and by different amounts: the live
model keeps about 900 points per frame and the reconstruction path about 2200,
out of the roughly 88000 a frame yields. The same object is therefore sampled at
three quite different spacings, and a neighbourhood that removes speckle at one
of them empties the cloud or does nothing at another.

So :func:`strip_speckle` measures the cloud's own point spacing first
(:func:`estimate_spacing`) and sizes the neighbourhood as a multiple of it. The
estimate is a single binning pass: for samples lying on a surface, the occupied
cells of a grid of edge ``s`` cover an area of about ``occupied * s^2``, so the
mean spacing of ``n`` points over that area is ``s * sqrt(occupied / n)``. It
costs one pass over the points, it is scale-free, and it was checked against a
brute-force median nearest-neighbour distance on real deprojected frames at 900,
2200, 8800 and 88000 points, where it landed within a factor of 0.6 to 1.3 of the
measured value across that whole 100x range. A factor, not a decimal - which is
why the multiple that uses it is generous and the count threshold is small.

What it removes, measured
--------------------------
On a rendered frame (sphere and box, 2% dropout, 0.5 code jitter) decimated the
way the two real call sites decimate, with 400 stray points scattered uniformly
through the bounding box and each stray labelled by its true distance to the
nearest real point, at the default neighbourhood of one spacing and three
required neighbours:

    points   real lost   strays removed, by true distance from surface
                         under 1x     1-2x      2-4x     beyond 4x
      908      1.0%          0%        22%       84%        95%
     2257      2.3%          0%        38%       99%        99%

Read the two ends of that table together, because they are the whole trade. A
stray further from the object than a couple of sample spacings is removed almost
always, and those are the dots an operator sees. A stray sitting within one
spacing of real surface is almost never removed - and it should not be, because
at that distance nothing distinguishes it from a noisy sample of the surface
itself. The 1.0-2.3% of real points lost are the sparse ones: silhouette rims,
faces at grazing incidence, the last row of samples before an edge. On the same
frames with no strays injected at all, the filter removes 2.4% of the 908-point
cloud and 3.1% of the 2257-point one, which is the cost of running it on data
that needed no filtering.

Those numbers are from SYNTHETIC frames with SYNTHETIC strays. They calibrate a
knob; they are not a measurement of this scanner's noise, which nothing in this
repository has characterised.

What this does NOT do
---------------------
It rejects ISOLATION, not error. A dense cluster of wrong points - a multipath
ghost of a whole face, a second surface reflected in something shiny, an entire
frame registered to the wrong pose - is dense, so it survives this test intact
and looks exactly like surface to it. Nothing here estimates a normal, fits a
plane, compares one frame against another, or knows which side of a surface a
point should be on. Cross-view confirmation is a different filter and a stronger
one; :func:`scanner3d.fusion.fuse` has it, under ``min_observations``.

It attaches no physical unit to anything. Spacings and neighbourhoods are in the
same unvalidated coordinate scale as the points, which for this repository means
they are numbers rather than lengths.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.fusion import MAX_CELL_INDEX, MAX_INPUT_POINTS
from scanner3d.geometry import Point

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: What :meth:`FilteredCloud.as_dict` says it did, so a consumer branching on
#: the record does not have to infer the test from the field names.
FILTER_MODE: Final = "voxel-neighbourhood-density"

#: The smallest neighbourhood this module will bin with, for the reason
#: :data:`scanner3d.fusion.MIN_VOXEL_SIZE` gives: below it, which cell a
#: coordinate lands in is decided by the last bits of a float64 division rather
#: than by where the point is, and a grid like that is not a grid.
MIN_NEIGHBOURHOOD: Final = 1e-9

#: Points that must lie in the surrounding block of cells, not counting the
#: point itself, for a point to be kept. Three, from the table in the module
#: docstring: it is the smallest threshold that removes essentially every stray
#: further than two spacings from the surface, and it costs under 3% of the real
#: points. Five removes a little more and costs 7-8%, which is silhouette rim.
DEFAULT_MIN_NEIGHBOURS: Final = 3

#: The neighbourhood edge, as a multiple of the cloud's own estimated spacing.
#: One spacing, so the tested block is about three spacings across: a point on a
#: well-sampled surface has of order eight in-plane neighbours inside it, and a
#: stray has none. Raising it to 1.5 loses almost no real points and also stops
#: removing most of the strays; lowering it to 0.75 removes more strays and
#: takes 9-11% of the real cloud with them. Both were measured; see the module
#: docstring.
DEFAULT_SPACING_MULTIPLE: Final = 1.0

#: Below this many points :func:`strip_speckle` declines to filter at all. A
#: cloud this small is not a sampled surface - it is a glimpse of the corner of
#: one, or a frame that caught almost nothing - and a density test over it
#: measures the decimation rather than the noise.
MIN_POINTS_TO_FILTER: Final = 64

#: The largest fraction of a cloud :func:`strip_speckle` will remove before
#: deciding that the threshold does not describe this cloud and withdrawing
#: entirely. A filter that takes half the points has not found speckle; it has
#: found a cloud too sparse for its own spacing estimate, and blanking the
#: operator's view is a worse failure than leaving the dots in it.
MAX_REMOVED_FRACTION: Final = 0.5

#: How many divisions of the cloud's longest axis the spacing search starts
#: from, and the factor it steps the cell edge by when the first guess puts too
#: many or too few points in a cell.
SEED_DIVISIONS: Final = 64
SPACING_STEP: Final = 1.6

#: The occupied-cells-per-point band the spacing search aims for. Above the top
#: of it the grid is so fine that almost every point owns a cell and the area
#: estimate is just the point count; below the bottom it is so coarse that the
#: occupied cells describe the bounding box rather than the surface.
OCCUPANCY_LOW: Final = 0.25
OCCUPANCY_HIGH: Final = 0.75

#: Cell edges the spacing search will try before accepting the last one. Eight
#: steps of 1.6 span a factor of 43 either way from the seed, which is far wider
#: than the range of densities the call sites produce.
MAX_SPACING_PASSES: Final = 8

#: How many removed points :meth:`FilteredCloud.as_dict` shows, so a report
#: stays a report.
PREVIEW_POINTS: Final = 8

#: A cell of the grid: the integer index triple a point floors into.
VoxelKey = tuple[int, int, int]

#: The 27 cells whose contents count as a point's neighbourhood, its own
#: included. Built once: this is the inner loop.
_NEIGHBOUR_OFFSETS: Final[tuple[VoxelKey, ...]] = tuple(
    (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
)

#: The axis names used in refusal messages, so a bound names a direction rather
#: than an index the caller has to decode.
_AXES: Final = ("x", "y", "z")


# --- refusals -----------------------------------------------------------------


class DenoiseNotAllowed(ScannerError):
    """A filter was requested that this module refuses to run.

    Every one of these is raised before a grid exists, so a refused call has
    built nothing and the caller still holds exactly the cloud it offered.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class NeighbourhoodUnusable(DenoiseNotAllowed):
    """The neighbourhood edge is not a positive finite number at or above the floor."""


class MinNeighboursUnusable(DenoiseNotAllowed):
    """``min_neighbours`` is negative, or is not a whole number.

    Zero is allowed and is a no-op that says so: it keeps every point, which is
    a legitimate way for a caller to disable the filter without branching around
    it and losing the record that says nothing was removed.
    """


class NeighbourhoodGridTooLarge(DenoiseNotAllowed):
    """A coordinate and the neighbourhood together imply an unusable cell index.

    Separate from any bound on the cloud's extent, because a *tight* cloud
    parked absurdly far from the origin has a small extent and an enormous
    index. Without this, the floor of an overflowed quotient raises
    ``OverflowError`` from inside the hot loop instead of a typed refusal at the
    boundary.
    """


class NothingToFilter(DenoiseNotAllowed):
    """No points were offered.

    An empty result would be indistinguishable from a filter that removed
    everything, and those two are not the same event.
    """


class InputTooLarge(DenoiseNotAllowed):
    """More points were offered than :data:`scanner3d.fusion.MAX_INPUT_POINTS`."""


class PointUnusable(DenoiseNotAllowed):
    """An input point is not a finite ``(x, y, z)`` triple.

    NaN is why this is checked rather than left to the arithmetic that follows:
    ``math.floor`` of a NaN raises, and a NaN that reached the extent pass would
    have made the spacing estimate meaningless without failing.
    """


class SpacingIndeterminate(DenoiseNotAllowed):
    """Every point of the cloud is at the same coordinate, so it has no spacing.

    A degenerate input rather than a filtering failure: there is no length in
    the cloud to scale a neighbourhood by, and inventing one would put an
    arbitrary absolute threshold on a caller that asked for a relative one.
    """


# --- the caveats every filtered cloud carries ---------------------------------

#: What is true of every run of this filter, whatever the input was. Run
#: specific notes are appended to these by the functions below, and the input's
#: own limitations after that. A caller cannot shorten this list.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    "this test rejects isolation, not error: a dense cluster of wrong points - a multipath "
    "ghost, a reflection, a frame placed at the wrong pose - has neighbours and survives it "
    "completely, and nothing here can tell such a cluster from surface",
    "real surface that was only sparsely sampled is isolated by exactly the same test as "
    "noise, so a silhouette rim, a face at grazing incidence and a feature at the edge of the "
    "sensor's range are removed along with the speckle, in a proportion measured only on "
    "synthetic frames",
    "removal happens before registration, so a point removed here contributed to no pose, "
    "reached no mesh and appears in no exported file; the record of what was removed is the "
    "only trace it leaves",
    "no normal, plane, curvature or surface direction is estimated, and no frame is compared "
    "against any other frame; the whole judgement is how many points of the SAME cloud lie in "
    "the block of cells around a point",
    "the neighbourhood is a grid-aligned block of cells rather than a ball centred on the "
    "point, so how far it actually reaches around a point depends on where in its cell the "
    "point sits, between one and two cells in each direction",
    "spacings and neighbourhoods are stated in the same coordinate scale as the points, whose "
    "physical unit this repository has not established; they are numbers, not lengths",
)


# --- boundary checks ----------------------------------------------------------


def ensure_neighbourhood_usable(neighbourhood: float) -> float:
    """Refuse a neighbourhood edge that cannot describe a grid, and return it unchanged."""
    if not math.isfinite(neighbourhood):
        raise NeighbourhoodUnusable(
            f"neighbourhood {neighbourhood!r} is not a finite number, so no grid follows from it"
        )
    if neighbourhood < MIN_NEIGHBOURHOOD:
        raise NeighbourhoodUnusable(
            f"neighbourhood {neighbourhood!r} is below the {MIN_NEIGHBOURHOOD} floor; at or "
            "under it, which cell a coordinate falls in is decided by float64 rounding rather "
            "than by where the point is"
        )
    return neighbourhood


def ensure_min_neighbours(min_neighbours: int) -> int:
    """Refuse a threshold that is not a whole non-negative count, and return it."""
    if isinstance(min_neighbours, bool) or not isinstance(min_neighbours, int):
        raise MinNeighboursUnusable(
            f"min_neighbours must be an int, got {min_neighbours!r}; a fractional or boolean "
            "threshold on a count of points is a caller's mistake, not a rounding decision"
        )
    if min_neighbours < 0:
        raise MinNeighboursUnusable(
            f"min_neighbours {min_neighbours} is negative; no point can have fewer than zero "
            "neighbours, so the threshold would keep everything while claiming to filter"
        )
    return min_neighbours


def ensure_points_usable(points: Sequence[Point]) -> int:
    """Refuse an input this module will not read, and return how many points it holds.

    Checked from ``len()`` alone for the size bound, so an oversized cloud is
    refused without a single coordinate being touched.
    """
    count = len(points)
    if count == 0:
        raise NothingToFilter(
            "no points were offered, so there is nothing to filter; refused rather than "
            "answered with an empty cloud, which is indistinguishable downstream from a filter "
            "that removed every point it was given"
        )
    if count > MAX_INPUT_POINTS:
        raise InputTooLarge(
            f"{count} points is past the {MAX_INPUT_POINTS}-point ceiling; nothing was read. "
            "Decimate the cloud before filtering it"
        )
    return count


def _checked(points: Sequence[Point]) -> Iterator[tuple[float, float, float]]:
    """Every point as a finite triple, refusing the first one that is not."""
    for index, point in enumerate(points):
        values = tuple(point)
        if len(values) != 3:
            raise PointUnusable(
                f"point {index} carries {len(values)} coordinates, not the 3 an (x, y, z) "
                "triple has"
            )
        for axis, value in zip(_AXES, values, strict=True):
            if not math.isfinite(value):
                raise PointUnusable(
                    f"point {index} has a {axis} of {value!r}, which is not finite; a "
                    "non-finite coordinate has no cell and would have decided nothing while "
                    "silently invalidating every bound computed from it"
                )
        yield (values[0], values[1], values[2])


def _binned(
    points: Sequence[tuple[float, float, float]], edge: float
) -> tuple[list[VoxelKey], dict[VoxelKey, int]]:
    """Each point's cell, in order, and how many points each occupied cell holds.

    Both at once, because the filter needs both and flooring a coordinate twice
    is the whole cost of the pass. The index bound is enforced here, on the way
    out of the division, rather than left to overflow inside a caller's loop.
    """
    inverse = 1.0 / edge
    keys: list[VoxelKey] = []
    grid: dict[VoxelKey, int] = {}
    for point in points:
        key = (
            math.floor(point[0] * inverse),
            math.floor(point[1] * inverse),
            math.floor(point[2] * inverse),
        )
        for axis, value, index in zip(_AXES, point, key, strict=True):
            if abs(index) > MAX_CELL_INDEX:
                raise NeighbourhoodGridTooLarge(
                    f"a {axis} of {value!r} at a cell edge of {edge!r} falls in cell {index}, "
                    f"past the {MAX_CELL_INDEX} index bound; the cloud sits too far from the "
                    "origin for this cell edge"
                )
        keys.append(key)
        grid[key] = grid.get(key, 0) + 1
    return keys, grid


# --- how far apart the points are ---------------------------------------------


@dataclass(frozen=True, slots=True)
class SpacingEstimate:
    """One cloud's own point spacing, and the binning that produced it.

    The intermediate numbers travel with the answer because the answer is an
    estimate with a stated accuracy - a factor, not a decimal - and a consumer
    that cannot see the grid it came from has no way to judge whether the
    assumption behind it (that the points lie on a surface) held for its cloud.
    """

    spacing: float
    #: The cell edge the search settled on, and what the cloud looked like on it.
    cell_edge: float
    occupied_cells: int
    input_points: int
    #: Cell edges tried, the accepted one included.
    passes: int
    #: Whether the search reached the occupancy band, or ran out of attempts and
    #: took the last edge it tried. False means the spacing is a weaker estimate
    #: than usual, and it is stated rather than hidden behind the same number.
    in_band: bool
    #: The longest side of the cloud's bounding box, which seeds the search.
    span: float

    @property
    def occupancy(self) -> float:
        """Occupied cells per input point, the quantity the search steers on."""
        return self.occupied_cells / self.input_points

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "spacing": self.spacing,
            "spacing_means": (
                "the mean distance between neighbouring points if they were spread evenly over "
                "the surface the occupied cells cover; an estimate good to a factor, checked "
                "against a brute-force median nearest neighbour at 900 to 88000 points"
            ),
            "grid": {
                "cell_edge": self.cell_edge,
                "occupied_cells": self.occupied_cells,
                "occupancy": self.occupancy,
                "target_band": [OCCUPANCY_LOW, OCCUPANCY_HIGH],
                "in_band": self.in_band,
                "passes": self.passes,
                "span": self.span,
            },
            "input_points": self.input_points,
            "assumes": (
                "the points lie on a surface: the area behind the estimate is occupied_cells "
                "times cell_edge squared, which is not the right area for a volumetric or a "
                "one-dimensional point set"
            ),
            "unit": (
                "the same coordinate scale as the input points, whose physical unit this "
                "repository has not established"
            ),
        }


def estimate_spacing(points: Sequence[Point]) -> SpacingEstimate:
    """How far apart this cloud's points are, from one binning pass per attempt.

    Bins the points at a cell edge seeded from their longest bounding-box side,
    steps that edge up or down until the occupied cells per point land in
    :data:`OCCUPANCY_LOW` to :data:`OCCUPANCY_HIGH`, and reads the spacing off
    the settled grid as ``cell_edge * sqrt(occupied / n)`` - the mean spacing of
    ``n`` points spread over an area of ``occupied * cell_edge^2``.

    That area is only the surface's area if the points lie on a surface. They do
    for every cloud in this repository, which is a deprojected depth frame or a
    union of them, and the assumption is stated on the record rather than
    assumed of the caller's data.

    Raises :class:`SpacingIndeterminate` for a cloud whose bounding box has no
    size at all, and the input refusals of :func:`ensure_points_usable` and
    :func:`PointUnusable` for one this module will not read.
    """
    count = ensure_points_usable(points)
    checked = list(_checked(points))
    span = max(
        max(point[axis] for point in checked) - min(point[axis] for point in checked)
        for axis in range(3)
    )
    if span <= 0.0:
        raise SpacingIndeterminate(
            f"all {count} points share one coordinate, so the cloud has no extent and no "
            "spacing to measure; a relative neighbourhood cannot be sized from it"
        )

    edge = span / SEED_DIVISIONS
    occupied = count
    in_band = False
    attempts = 0
    for attempt in range(MAX_SPACING_PASSES):
        attempts = attempt + 1
        occupied = len(_binned(checked, edge)[1])
        occupancy = occupied / count
        in_band = OCCUPANCY_LOW <= occupancy <= OCCUPANCY_HIGH
        if in_band or attempt == MAX_SPACING_PASSES - 1:
            break
        edge = edge * SPACING_STEP if occupancy > OCCUPANCY_HIGH else edge / SPACING_STEP

    return SpacingEstimate(
        spacing=edge * math.sqrt(occupied / count),
        cell_edge=edge,
        occupied_cells=occupied,
        input_points=count,
        passes=attempts,
        in_band=in_band,
        span=span,
    )


# --- the result ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilteredCloud:
    """One cloud with its isolated points separated out, and the accounting for it.

    The removed points are kept rather than counted, for two reasons. A caller
    that wants to show the operator what was thrown away can, which is the only
    honest way to present a filter that cannot tell rim from speckle; and a test
    can assert which points went rather than how many, which is the difference
    between checking the rule and checking a statistic.

    ``applied`` is the field to read before believing the rest.
    :func:`strip_speckle` withdraws on a cloud too small or too sparse for its
    own threshold to mean anything, and a withdrawal returns every point it was
    given - so a record with ``applied`` false describes a filter that ran and
    decided not to act, which is a different event from one that found nothing
    to remove.
    """

    points: tuple[Point, ...]
    removed: tuple[Point, ...]
    neighbourhood: float
    min_neighbours: int
    applied: bool
    spacing: SpacingEstimate | None
    limitations: tuple[str, ...]

    @property
    def input_points(self) -> int:
        """How many points went in."""
        return len(self.points) + len(self.removed)

    @property
    def kept_points(self) -> int:
        """How many came out."""
        return len(self.points)

    @property
    def removed_points(self) -> int:
        """How many were dropped for having too few neighbours."""
        return len(self.removed)

    @property
    def removed_fraction(self) -> float:
        """What proportion of the input was dropped, in [0, 1]."""
        total = self.input_points
        return self.removed_points / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": FILTER_MODE,
            "applied": self.applied,
            "test": {
                "neighbourhood": self.neighbourhood,
                "neighbourhood_means": (
                    "the cell edge of the grid; a point's neighbourhood is the 3x3x3 block of "
                    "cells around its own, so the region tested is about three cell edges "
                    "across and is aligned to the grid rather than centred on the point"
                ),
                "min_neighbours": self.min_neighbours,
                "min_neighbours_means": (
                    "other points required in that block, the point itself not counted"
                ),
            },
            "accounting": {
                "input_points": self.input_points,
                "kept_points": self.kept_points,
                "removed_points": self.removed_points,
                "removed_fraction": self.removed_fraction,
            },
            "spacing": None if self.spacing is None else self.spacing.as_dict(),
            "bounds": {
                "min_neighbourhood": MIN_NEIGHBOURHOOD,
                "max_input_points": MAX_INPUT_POINTS,
                "max_cell_index": MAX_CELL_INDEX,
                "min_points_to_filter": MIN_POINTS_TO_FILTER,
                "max_removed_fraction": MAX_REMOVED_FRACTION,
            },
            "removed": {
                "count": self.removed_points,
                "preview": [list(point) for point in self.removed[:PREVIEW_POINTS]],
                "note": (
                    f"every removed point is on FilteredCloud.removed; at most {PREVIEW_POINTS} "
                    "are shown here so a report stays a report"
                ),
            },
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


def _run_note(*, applied: bool, kept: int, removed: int, min_neighbours: int) -> str:
    """The honest sentence about what this particular run did."""
    if not applied:
        return (
            "the filter withdrew and returned every point it was given: the cloud was too small "
            f"({kept + removed} points, under {MIN_POINTS_TO_FILTER}) or the threshold would "
            f"have removed more than {MAX_REMOVED_FRACTION:.0%} of it, which means the "
            "threshold does not describe this cloud. Nothing was rejected, so nothing in this "
            "cloud has been checked for isolation"
        )
    if min_neighbours == 0:
        return (
            "min_neighbours was 0, so every point met the threshold and the filter is a "
            "no-op by request: this cloud has not been checked for isolation"
        )
    if removed == 0:
        return (
            f"the filter removed nothing: every one of the {kept} points had at least "
            f"{min_neighbours} others in the block of cells around it. That is a fact about "
            "this cloud being densely sampled, not evidence that it holds no stray points - a "
            "stray within a spacing or so of real surface passes this test"
        )
    return (
        f"{removed} of {kept + removed} points were removed for having fewer than "
        f"{min_neighbours} others in the block of cells around them. Some proportion of those "
        "was real surface too sparsely sampled to pass - a rim, a grazing face, the edge of "
        "the range - and this filter cannot say which, because from the grid's point of view a "
        "lone rim sample and a lone stray are the same event"
    )


def _all_limitations(notes: Sequence[str], carried: Sequence[str]) -> tuple[str, ...]:
    """This module's caveats, then this run's notes, then the input's own.

    De-duplicated with order preserved, for the reason
    :func:`scanner3d.fusion._all_limitations` gives: clouds from one capture
    share almost every caveat they have, and a consumer should not read the same
    sentence thirty-six times to find the one that differs.
    """
    return tuple(dict.fromkeys((*INHERENT_LIMITATIONS, *notes, *carried)))


# --- the filter ---------------------------------------------------------------


def reject_isolated(
    points: Sequence[Point],
    *,
    neighbourhood: float,
    min_neighbours: int = DEFAULT_MIN_NEIGHBOURS,
    spacing: SpacingEstimate | None = None,
    input_limitations: Sequence[str] = (),
) -> FilteredCloud:
    """Split a cloud into the points with enough neighbours and the points without.

    ``neighbourhood`` is the cell edge, in the points' own coordinate scale. A
    point's neighbourhood is the 3x3x3 block of cells around its own, so the
    region actually tested is about three times that edge across; a caller that
    wants the neighbourhood sized from the cloud rather than chosen in advance
    should call :func:`strip_speckle`, which does exactly that and then calls
    this.

    ``min_neighbours`` counts OTHER points in the block. Zero keeps everything
    and says so on the record, rather than being refused, so a caller can turn
    the filter off without branching around it.

    The count is computed once per occupied cell and shared by every point in
    it. That is not an approximation: the block of cells around a point is
    determined by which cell the point is in, so two points of one cell have the
    same block and cannot disagree.

    The output preserves the input's order. Both output tuples do, which is what
    lets a caller line ``removed`` up against whatever else it was carrying per
    point.

    Every refusal happens before the grid is built:
    :class:`NeighbourhoodUnusable`, :class:`MinNeighboursUnusable`,
    :class:`NothingToFilter`, :class:`InputTooLarge`, :class:`PointUnusable`
    and :class:`NeighbourhoodGridTooLarge`.
    """
    edge = ensure_neighbourhood_usable(neighbourhood)
    threshold = ensure_min_neighbours(min_neighbours)
    ensure_points_usable(points)
    checked = list(_checked(points))

    keys, grid = _binned(checked, edge)
    get = grid.get
    verdict: dict[VoxelKey, bool] = {}
    for cell in grid:
        total = 0
        for dx, dy, dz in _NEIGHBOUR_OFFSETS:
            total += get((cell[0] + dx, cell[1] + dy, cell[2] + dz), 0)
        # The point itself is inside its own cell and is counted there, so the
        # threshold is on the block total minus one.
        verdict[cell] = total - 1 >= threshold

    kept: list[Point] = []
    dropped: list[Point] = []
    for point, key in zip(checked, keys, strict=True):
        (kept if verdict[key] else dropped).append(point)

    note = _run_note(applied=True, kept=len(kept), removed=len(dropped), min_neighbours=threshold)
    return FilteredCloud(
        points=tuple(kept),
        removed=tuple(dropped),
        neighbourhood=edge,
        min_neighbours=threshold,
        applied=True,
        spacing=spacing,
        limitations=_all_limitations((note,), input_limitations),
    )


def strip_speckle(
    points: Sequence[Point],
    *,
    multiple: float = DEFAULT_SPACING_MULTIPLE,
    min_neighbours: int = DEFAULT_MIN_NEIGHBOURS,
    max_removed_fraction: float = MAX_REMOVED_FRACTION,
    input_limitations: Sequence[str] = (),
) -> FilteredCloud:
    """Reject this cloud's isolated points, at a neighbourhood sized from the cloud.

    The call site version of :func:`reject_isolated`: measure the cloud's own
    spacing with :func:`estimate_spacing`, size the neighbourhood at ``multiple``
    times it, and filter. That is what makes one setting work across the three
    quite different decimations this repository's clouds arrive at - see the
    module docstring.

    It WITHDRAWS in two cases, returning every point it was given with
    :attr:`FilteredCloud.applied` false and a limitation saying so:

    * fewer than :data:`MIN_POINTS_TO_FILTER` points, where a density test
      measures the decimation rather than the noise;
    * a result that would remove more than ``max_removed_fraction`` of the
      cloud, which does not happen to a cloud with speckle in it - it happens to
      a cloud too sparse for its own spacing estimate, and emptying the
      operator's view is a worse outcome than leaving the dots in it.

    Withdrawal is deliberately not an exception. This runs per captured frame on
    the thread that owns the device, and a frame that catches almost nothing is
    a normal event at the start and end of a sweep, not an error.
    """
    if not math.isfinite(multiple) or multiple <= 0.0:
        raise NeighbourhoodUnusable(
            f"multiple {multiple!r} is not a positive finite number, so no neighbourhood "
            "follows from the cloud's spacing"
        )
    if not 0.0 < max_removed_fraction <= 1.0:
        raise DenoiseNotAllowed(
            f"max_removed_fraction {max_removed_fraction!r} is outside (0, 1]; a filter that "
            "may remove no points, or more than all of them, is not a bound"
        )
    threshold = ensure_min_neighbours(min_neighbours)
    count = ensure_points_usable(points)

    if count < MIN_POINTS_TO_FILTER:
        return _withdrawn(points, threshold, None, input_limitations)

    measured = estimate_spacing(points)
    filtered = reject_isolated(
        points,
        neighbourhood=measured.spacing * multiple,
        min_neighbours=threshold,
        spacing=measured,
        input_limitations=input_limitations,
    )
    if filtered.removed_fraction > max_removed_fraction:
        return _withdrawn(points, threshold, measured, input_limitations)
    return filtered


def _withdrawn(
    points: Sequence[Point],
    min_neighbours: int,
    spacing: SpacingEstimate | None,
    input_limitations: Sequence[str],
) -> FilteredCloud:
    """The whole input back, on a record that says the filter chose not to act."""
    note = _run_note(applied=False, kept=len(points), removed=0, min_neighbours=min_neighbours)
    return FilteredCloud(
        points=tuple((point[0], point[1], point[2]) for point in points),
        removed=(),
        neighbourhood=0.0 if spacing is None else spacing.spacing,
        min_neighbours=min_neighbours,
        applied=False,
        spacing=spacing,
        limitations=_all_limitations((note,), input_limitations),
    )
