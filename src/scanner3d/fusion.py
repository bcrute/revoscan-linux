"""Merging registered clouds into one cloud, and what merging does not make truer.

Why this module exists
----------------------
Registration says where each capture sits. This module says what the captures
*are* once they sit there, and concatenation is not an answer to that question:
a turntable sweep that sees one patch of surface from six directions produces
six overlapping sheets of points which all disagree slightly, and a viewer
renders that as a thick fuzzy shell rather than a surface. Binning the
transformed points into a voxel grid and averaging within each occupied cell
collapses the six sheets into one. It is the cheapest honest way to do it: no
surface normals, no implicit function, no dependency, and - the property that
matters most here - nothing that invents a point where no capture put one.

What averaging is worth, precisely
----------------------------------
Averaging ``n`` observations of one cell reduces error that is *independent*
between those observations. It does nothing whatsoever to error that every
observation shares. This unit's factory principal point sits 277 px from the
frame centre and no measurement in this repository has confirmed it (see
``docs/depth-to-pointcloud.md``); if it is wrong, every frame is wrong the same
way, and fusing thirty-six of them yields a cleaner-looking cloud that is wrong
by exactly as much as one frame was. A fused cloud is *quieter* than its inputs.
It is not more accurate than them. That distinction is not left to a reader's
good judgement: it is the first entry of :data:`INHERENT_LIMITATIONS` and it
leaves on every :class:`FusedCloud` this module returns.

The observation threshold, and what it costs
--------------------------------------------
``min_observations`` is the one knob that makes fusion better than
concatenation, and it cuts both ways. A cell only one capture ever saw is often
noise - a stray sample at a depth discontinuity, a speckle artefact, a
reflection - so requiring two or more captures to agree removes much of that. It
also removes every part of the object only one capture *could* see: the
underside, the inside of a cavity, a face at grazing incidence to every other
view. The filter cannot tell the two apart, because from the grid's point of
view they are the same thing - one observation. Raising the threshold therefore
trades real single-view surface for noise at an exchange rate nothing here
measures and nothing here estimates. What this module can do, and does, is
report how many occupied cells the threshold discarded, so the trade shows up in
the record rather than being implied by a tidier picture.

An "observation" is one *source cloud*, not one sample. Five points from one
frame landing in one cell is one view of that cell, five times over, and
counting it as five would make the threshold agree with itself rather than
confirm anything.

What this module does NOT do
----------------------------
It does not estimate, refine or check the transforms it is given; a pose error
moves every point of its cloud and leaves no trace in this record. It does not
reject outliers, fill holes, smooth, or fit anything. It does not compute a
normal or a triangle. And it attaches no physical unit to ``voxel_size``, which
is measured in the same unvalidated coordinate scale as the points themselves -
the scale whose physical meaning this repository has never established.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import Point, Rigid

#: Serialised shape version for the record this module emits.
SCHEMA_VERSION: Final = 1

#: What :meth:`FusedCloud.as_dict` says it did, so a consumer branching on the
#: record does not have to infer the algorithm from the field names.
FUSION_MODE: Final = "voxel-grid-fusion"

#: The smallest cell edge this module will bin with. Below it the index
#: arithmetic stops describing the geometry: ``floor(coordinate / voxel_size)``
#: for a coordinate of magnitude 1e3 already exceeds 1e12 quotients here, so the
#: last bits of a coordinate - not its position - decide which cell it lands in.
#: A grid whose membership is a property of float64 rounding is not a grid.
MIN_VOXEL_SIZE: Final = 1e-9

#: The most cells the input extent may imply, across all three axes together.
#: A 1024x1024x1024 grid is already far finer than anything this device
#: resolves, and past it a ``voxel_size`` is not a fusion parameter but a
#: mistake: the grid degenerates towards one cell per point, which is
#: concatenation wearing a fusion's name and costing a dict entry per point.
MAX_GRID_CELLS: Final = 1 << 30

#: The largest absolute cell index any coordinate may produce. Separate from
#: :data:`MAX_GRID_CELLS` because a *tight* cloud parked absurdly far from the
#: origin has a small extent and an enormous index; without this bound the floor
#: of an overflowed quotient raises ``OverflowError`` from inside the hot loop
#: instead of a typed refusal at the boundary.
MAX_CELL_INDEX: Final = 1 << 40

#: The most points one fuse may consume, summed across every input cloud. About
#: 45 frames at the best yield this unit has produced (89070 surviving samples in
#: ``tests/fixtures/frames/base_4.bin``), which is more than a full turntable
#: sweep at 10-degree steps. Past it, pure-Python fusion is minutes of work and
#: the caller should downsample each cloud - :func:`voxel_downsample` is right
#: here - before merging. Checked from ``len()`` alone, so an oversized input is
#: refused without a single point being read.
MAX_INPUT_POINTS: Final = 4_000_000

#: How many points :meth:`FusedCloud.as_dict` shows, so a report stays a report.
PREVIEW_POINTS: Final = 8

#: A cell of the grid: the integer index triple a point floors into.
VoxelKey = tuple[int, int, int]

#: Where each accumulator slot lives. Named because the alternative - a record
#: type per voxel - is an allocation per occupied cell, and a sweep occupies
#: hundreds of thousands of them. The two integer quantities ride in the same
#: float list as the sums: a cloud index and an observation count are exact in
#: float64 far past :data:`MAX_INPUT_POINTS`.
_SUM_X: Final = 0
_SUM_Y: Final = 1
_SUM_Z: Final = 2
_COUNT: Final = 3
_LAST_CLOUD: Final = 4
_OBSERVATIONS: Final = 5

_IDENTITY: Final = Rigid.identity()

#: The axis names used in refusal messages, so a bound names a direction rather
#: than an index the caller has to decode.
_AXES: Final = ("x", "y", "z")


# --- refusals -----------------------------------------------------------------


class FusionNotAllowed(ScannerError):
    """A fusion was requested that this module refuses to perform.

    Every one of these is raised before the input points are read, or - where
    the defect is in the points themselves - before a single voxel is
    allocated. A fusion that is refused has therefore built nothing, and a
    :class:`FusedCloud` that exists is one whose whole input passed.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class VoxelSizeUnusable(FusionNotAllowed):
    """The cell edge is not a positive finite number at or above the floor.

    Zero and negative sizes have no grid at all; a size below
    :data:`MIN_VOXEL_SIZE` has one whose membership is decided by floating-point
    rounding rather than by where the points are.
    """


class VoxelGridTooLarge(FusionNotAllowed):
    """The extent and the cell edge together imply an unusable grid.

    Either past :data:`MAX_GRID_CELLS` cells, or a cell index past
    :data:`MAX_CELL_INDEX`. Refused rather than attempted: the sparse hash would
    otherwise grow one entry per input point while the result - a cloud with as
    many points as it started with - would not be a fusion of anything.
    """


class CloudCountMismatch(FusionNotAllowed):
    """The per-cloud arguments are not one per cloud.

    A transform list off by one silently pairs every cloud with the wrong pose
    and produces a scan that still looks like a scan, so the count is checked
    rather than zipped over.
    """


class NothingToFuse(FusionNotAllowed):
    """There are no clouds, or the clouds hold no points between them.

    An empty result would be indistinguishable from a fusion where every voxel
    failed the threshold, and those two are not the same event.
    """


class InputTooLarge(FusionNotAllowed):
    """More points were offered than :data:`MAX_INPUT_POINTS`."""


class ObservationThresholdUnreachable(FusionNotAllowed):
    """``min_observations`` is below 1 or above the number of *contributing* clouds.

    A threshold no number of clouds can meet does not filter the result, it
    empties it, and it does so silently. That is a caller's mistake and it is
    named here rather than returned as a cloud with no points in it.

    Measured against the clouds that hold points, not against ``len(clouds)``:
    an empty cloud reaches no cell, so it can never be one of the observations
    a threshold counts. Checking against the frame count instead would accept
    ``min_observations=3`` over three clouds two of which are empty, and hand
    back the empty result this refusal exists to prevent.
    """


class PointUnusable(FusionNotAllowed):
    """An input point is not a finite ``(x, y, z)`` triple.

    NaN is the reason this check is not left to the comparisons that follow it:
    every comparison against NaN is false, so a single NaN would slip through
    the bounding-box pass untouched and then decide nothing, having quietly made
    the extent - and the grid sized from it - meaningless.
    """


# --- the caveats every fused cloud carries ------------------------------------

#: What is true of every fusion this module performs, whatever the inputs were.
#: Run-specific notes - what the threshold discarded, what a single cloud does
#: not confirm - are appended to these by :func:`fuse`, and the inputs' own
#: limitations are appended after that. A caller cannot shorten this list.
INHERENT_LIMITATIONS: Final[tuple[str, ...]] = (
    "averaging within a voxel reduces only error that is independent between observations; a "
    "systematic error shared by every frame - a wrong principal point, a wrong depth scalar, a "
    "mis-estimated turntable axis - survives averaging completely and is no smaller after "
    "fusing thirty-six frames than it was after one",
    "this result is no more metrically valid than the clouds it merged: no distance in this "
    "repository has ever been compared against a physical measurement, and merging clouds "
    "creates no evidence that did not exist before it",
    "every output point is the mean of the samples that fell in one cell, not an observed "
    "surface point; where a cell straddles a depth discontinuity the mean lies between the two "
    "surfaces and describes neither of them",
    "the observation count behind a cell is how many source clouds reached it, not a "
    "confidence: surface visible from one direction only is seen once however good the capture, "
    "and an artefact visible from every direction is seen every time",
    "the fused points inherit the registration transforms exactly as given; nothing here "
    "estimates, refines or checks a pose, and a pose error displaces every point of its cloud "
    "without leaving any trace in this record",
    "cell membership is decided by float64 rounding of coordinate / voxel_size, so a point "
    "lying on a cell boundary falls on whichever side that division puts it; the grid has no "
    "tie-break rule and none is claimed",
    "the output ordering is canonical - ascending cell index - but the coordinates are not: "
    "samples are summed in the order the clouds were given and float64 addition is not "
    "associative, so the same clouds fused in a different order can differ in the last bits",
    "voxel_size is stated in the same coordinate scale as the input points, whose physical "
    "unit this repository has not established; it is a number, not a length",
)


# --- boundary checks ----------------------------------------------------------


def ensure_voxel_size_usable(voxel_size: float) -> float:
    """Refuse a cell edge that cannot describe a grid, and return it unchanged."""
    if not math.isfinite(voxel_size):
        raise VoxelSizeUnusable(
            f"voxel_size {voxel_size!r} is not finite, so no cell has an edge and every point "
            "would index the same nowhere"
        )
    if voxel_size < MIN_VOXEL_SIZE:
        raise VoxelSizeUnusable(
            f"voxel_size {voxel_size!r} is below the {MIN_VOXEL_SIZE} floor; at or under it the "
            "cell a point lands in is decided by float64 rounding of the division rather than "
            "by where the point is"
        )
    return voxel_size


def ensure_transforms_match(clouds: Sequence[Sequence[Point]], transforms: Sequence[Rigid]) -> None:
    """Refuse a transform list that is not one pose per cloud."""
    if len(transforms) != len(clouds):
        raise CloudCountMismatch(
            f"{len(clouds)} clouds against {len(transforms)} transforms: each cloud needs "
            "exactly one pose, and a list off by one pairs every cloud after the gap with "
            "another cloud's pose"
        )


def ensure_limitations_match(
    clouds: Sequence[Sequence[Point]], input_limitations: Sequence[Sequence[str]]
) -> None:
    """Refuse per-cloud limitations that are neither absent nor one group per cloud."""
    if input_limitations and len(input_limitations) != len(clouds):
        raise CloudCountMismatch(
            f"{len(clouds)} clouds against {len(input_limitations)} limitation groups: pass one "
            "group per cloud, or none at all. A caller with one shared list passes "
            "[shared] * len(clouds) rather than [shared], because a silently truncated "
            "correspondence would drop somebody's caveat"
        )


def ensure_clouds_usable(clouds: Sequence[Sequence[Point]]) -> int:
    """Refuse an empty or oversized input, and return the total point count.

    Counted from ``len()`` on each cloud, so an input past the ceiling is
    refused without a point being read and without a transform being applied.
    """
    if not clouds:
        raise NothingToFuse("no clouds were offered; there is nothing to merge")
    total = sum(len(cloud) for cloud in clouds)
    if total == 0:
        raise NothingToFuse(
            f"{len(clouds)} clouds were offered and every one of them is empty; an empty result "
            "here would be indistinguishable from a fusion whose every cell failed the "
            "observation threshold, and those are different events"
        )
    if total > MAX_INPUT_POINTS:
        raise InputTooLarge(
            f"{total} points across {len(clouds)} clouds is past the {MAX_INPUT_POINTS} ceiling; "
            "downsample each cloud before merging rather than widening the bound"
        )
    return total


def ensure_threshold_reachable(min_observations: int, contributing_clouds: int) -> int:
    """Refuse an observation threshold no input could satisfy, and return it.

    ``contributing_clouds`` is how many clouds hold at least one point. An empty
    cloud is counted in ``source_frames`` but reaches no cell, so it can never
    be one of the observations behind a voxel; measuring the threshold against
    the frame count would let a run whose result is *guaranteed* empty through.
    """
    if min_observations < 1:
        raise ObservationThresholdUnreachable(
            f"min_observations {min_observations} is below 1; a cell that no cloud reached is "
            "not in the grid at all, so 1 is the weakest threshold there is"
        )
    if min_observations > contributing_clouds:
        raise ObservationThresholdUnreachable(
            f"min_observations {min_observations} cannot be met by {contributing_clouds} clouds "
            "that hold any points; this would empty the result rather than filter it, and would "
            "do so without saying so"
        )
    return min_observations


# --- the grid -----------------------------------------------------------------


def _moved(cloud: Sequence[Point], transform: Rigid) -> Iterator[Point]:
    """The cloud in the fused frame, without spending on an identity.

    :func:`voxel_downsample` is a fusion of one cloud under the identity, and
    both passes below walk every point; multiplying a few hundred thousand
    coordinates by an identity matrix twice is the single largest avoidable cost
    on that path. A near-identity takes the general route, which is right: this
    is an exact-match shortcut, not a tolerance.
    """
    if transform == _IDENTITY:
        return iter(cloud)
    return transform.apply_all(cloud)


def _extent(clouds: Sequence[Sequence[Point]], transforms: Sequence[Rigid]) -> tuple[Point, Point]:
    """The axis-aligned bounds of every transformed point, refusing bad ones.

    A whole pass whose only product is six numbers, deliberately. The
    alternative - keeping the transformed points so the second pass does not
    have to redo the arithmetic - is a list of several hundred thousand freshly
    allocated tuples, which is precisely the allocation the bound computed from
    these six numbers exists to prevent. Paying the transform twice is the
    cheaper half of that trade.
    """
    low_x = low_y = low_z = math.inf
    high_x = high_y = high_z = -math.inf
    isfinite = math.isfinite
    for cloud_index, (cloud, transform) in enumerate(zip(clouds, transforms, strict=True)):
        try:
            for point_index, (x, y, z) in enumerate(_moved(cloud, transform)):
                if not (isfinite(x) and isfinite(y) and isfinite(z)):
                    raise PointUnusable(
                        f"cloud {cloud_index} point {point_index} lands at {(x, y, z)!r}, which "
                        "is not finite. Every transform this module accepts is finite in every "
                        "entry, so the value arrived that way"
                    )
                if x < low_x:
                    low_x = x
                if x > high_x:
                    high_x = x
                if y < low_y:
                    low_y = y
                if y > high_y:
                    high_y = y
                if z < low_z:
                    low_z = z
                if z > high_z:
                    high_z = z
        except (TypeError, ValueError) as exc:
            raise PointUnusable(
                f"cloud {cloud_index} holds an entry that is not an (x, y, z) triple: {exc}"
            ) from exc
    return (low_x, low_y, low_z), (high_x, high_y, high_z)


def ensure_grid_bounded(low: Point, high: Point, voxel_size: float) -> int:
    """Refuse an unusable grid and return how many cells the extent implies.

    The returned count is the *dense* cell count, which is not what gets
    allocated - the grid below is a sparse hash holding one entry per occupied
    cell, never more than one per input point. The bound is still worth
    enforcing, because a dense count in the billions is the signature of a
    ``voxel_size`` so fine that nothing merges: the caller would get their input
    back, one dict entry heavier per point, under a name that promises a fusion.

    Counted from the cell indices the extent's own ends fall into, not from
    ``floor(width / voxel_size) + 1``. The two differ whenever the extent
    straddles a cell boundary, and the width form is the one that is wrong: an
    extent of 0.9 to 1.1 at edge 1.0 occupies two cells on that axis and the
    width form calls it one, under-reporting a three-axis grid by up to
    eightfold - a bound describing a grid smaller than the one it guards.
    """
    cells = 1
    for axis, (edge_low, edge_high) in enumerate(zip(low, high, strict=True)):
        reach = max(abs(edge_low), abs(edge_high)) / voxel_size
        if reach > MAX_CELL_INDEX:
            raise VoxelGridTooLarge(
                f"the {_AXES[axis]} extent [{edge_low!r}, {edge_high!r}] reaches cell index "
                f"{reach!r} at voxel_size {voxel_size!r}, past the {MAX_CELL_INDEX} bound; the "
                "cloud sits absurdly far from the origin for a cell this small"
            )
        # Safe to floor now: ``reach`` bounds both ends, so neither quotient is
        # non-finite and neither overflows.
        spread = math.floor(edge_high / voxel_size) - math.floor(edge_low / voxel_size) + 1
        if spread > MAX_GRID_CELLS:
            raise VoxelGridTooLarge(
                f"the {_AXES[axis]} extent [{edge_low!r}, {edge_high!r}] spans {spread} cells at "
                f"voxel_size {voxel_size!r}, past the {MAX_GRID_CELLS} bound on its own"
            )
        cells *= spread
    if cells > MAX_GRID_CELLS:
        raise VoxelGridTooLarge(
            f"the extent implies {cells} cells at voxel_size {voxel_size!r}, past the "
            f"{MAX_GRID_CELLS} bound; a grid this fine merges nothing and costs a dict entry "
            "for every point that went in"
        )
    return cells


def _accumulate(
    clouds: Sequence[Sequence[Point]], transforms: Sequence[Rigid], voxel_size: float
) -> dict[VoxelKey, list[float]]:
    """Bin every transformed point, summing coordinates and counting views.

    Sparse: one entry per *occupied* cell, so an empty region of the extent
    costs nothing. Observations are counted per source cloud rather than per
    point, which the ordered walk makes exact without a set or a bitmask - the
    clouds arrive in order, so a cell whose last contributor differs from the
    current one is being seen by a new view.
    """
    floor = math.floor
    grid: dict[VoxelKey, list[float]] = {}
    for cloud_index, (cloud, transform) in enumerate(zip(clouds, transforms, strict=True)):
        marker = float(cloud_index)
        for x, y, z in _moved(cloud, transform):
            key = (floor(x / voxel_size), floor(y / voxel_size), floor(z / voxel_size))
            voxel = grid.get(key)
            if voxel is None:
                grid[key] = [x, y, z, 1.0, marker, 1.0]
                continue
            voxel[_SUM_X] += x
            voxel[_SUM_Y] += y
            voxel[_SUM_Z] += z
            voxel[_COUNT] += 1.0
            if voxel[_LAST_CLOUD] != marker:
                voxel[_LAST_CLOUD] = marker
                voxel[_OBSERVATIONS] += 1.0
    return grid


def _reduce(grid: dict[VoxelKey, list[float]], min_observations: int) -> tuple[Point, ...]:
    """One averaged point per surviving cell, in ascending cell index.

    Sorted rather than left in insertion order so that the ordering is a
    property of the geometry instead of a property of the order the caller
    happened to list the clouds in - which is what makes two fusions of the same
    data comparable line by line. It does not make the *values* order
    independent; float64 addition is not associative and
    :data:`INHERENT_LIMITATIONS` says so rather than this docstring implying
    otherwise.
    """
    points: list[Point] = []
    for _key, voxel in sorted(grid.items()):
        if voxel[_OBSERVATIONS] < min_observations:
            continue
        count = voxel[_COUNT]
        points.append((voxel[_SUM_X] / count, voxel[_SUM_Y] / count, voxel[_SUM_Z] / count))
    return tuple(points)


# --- the result ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusedCloud:
    """One merged cloud, with the accounting needed to see what merging cost.

    ``occupied_voxels`` counts the cells that held at least one point *before*
    the observation threshold was applied, so ``occupied_voxels - len(points)``
    is exactly what the threshold discarded and neither number can hide the
    other. ``input_points`` against ``len(points)`` is the other half of the
    picture: how much was merged away.

    ``min_observations`` is carried in the record rather than left to the call
    site, because a consumer holding these points cannot otherwise know whether
    single-view surface was filtered out of them.
    """

    points: tuple[Point, ...]
    voxel_size: float
    source_frames: int
    input_points: int
    occupied_voxels: int
    limitations: tuple[str, ...]
    min_observations: int

    @property
    def output_points(self) -> int:
        """How many points came out."""
        return len(self.points)

    @property
    def dropped_voxels(self) -> int:
        """Occupied cells the observation threshold discarded."""
        return self.occupied_voxels - len(self.points)

    @property
    def merged_away(self) -> int:
        """Input points that did not survive as points of their own."""
        return self.input_points - len(self.points)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": FUSION_MODE,
            "grid": {
                "voxel_size": self.voxel_size,
                "voxel_size_stated_in": (
                    "the same coordinate scale as the input points, whose physical unit this "
                    "repository has not established; it is a number, not a length"
                ),
                "min_observations": self.min_observations,
                "observation_means": "how many source clouds reached a cell, not how many points",
            },
            "accounting": {
                "source_frames": self.source_frames,
                "input_points": self.input_points,
                "occupied_voxels": self.occupied_voxels,
                "output_points": self.output_points,
                "dropped_voxels": self.dropped_voxels,
                "merged_away": self.merged_away,
            },
            "bounds": {
                "min_voxel_size": MIN_VOXEL_SIZE,
                "max_grid_cells": MAX_GRID_CELLS,
                "max_cell_index": MAX_CELL_INDEX,
                "max_input_points": MAX_INPUT_POINTS,
            },
            "points": {
                "count": self.output_points,
                "preview": [list(point) for point in self.points[:PREVIEW_POINTS]],
                "note": (
                    f"the whole cloud is on FusedCloud.points; at most {PREVIEW_POINTS} points "
                    "are shown here so a report stays a report"
                ),
            },
            "dimensional_accuracy_claimed": False,
            "limitations": list(self.limitations),
        }


def _threshold_note(min_observations: int, occupied: int, dropped: int) -> str:
    """The honest sentence about what this run's threshold did.

    Three cases, because the middle one is a lie if it is folded into the last:
    a threshold that discarded nothing did not remove any geometry, and saying
    it did would put a caveat about lost surface on a result that lost none.
    """
    if min_observations == 1:
        return (
            "min_observations was 1, so no cross-view confirmation was required: every occupied "
            "cell survived, including cells holding a single sample from a single frame, which "
            "is the shape noise takes"
        )
    if dropped == 0:
        return (
            f"min_observations was {min_observations} and discarded nothing: all {occupied} "
            "occupied cells were reached by at least that many source clouds. The threshold "
            "removed no geometry on this input - which is a fact about these clouds overlapping, "
            "not evidence that the filter is free; on a real sweep it removes single-view surface "
            "along with noise and cannot tell them apart"
        )
    return (
        f"min_observations was {min_observations}: {dropped} of {occupied} occupied cells were "
        "discarded for being reached by too few source clouds. That filter cannot separate noise "
        "from surface only one view could see - an underside, a cavity, anything at grazing "
        "incidence - so real geometry was removed along with whatever noise was removed, in a "
        "proportion nothing here measures"
    )


def _run_notes(
    *,
    clouds: Sequence[Sequence[Point]],
    min_observations: int,
    occupied: int,
    kept: int,
    input_points: int,
) -> list[str]:
    """The limitations that are true of this run and not of fusion in general."""
    notes = [_threshold_note(min_observations, occupied, occupied - kept)]
    if len(clouds) == 1:
        notes.append(
            "one cloud was fused, so nothing was cross-checked between views; this is a "
            "downsample of a single capture and every cell in it has exactly one observation"
        )
    empty = [index for index, cloud in enumerate(clouds) if not len(cloud)]
    if empty:
        listed = ", ".join(str(index) for index in empty)
        notes.append(
            f"clouds {listed} contributed no points at all; they are counted in source_frames "
            "but confirmed nothing, so an observation threshold was met by fewer views than the "
            "frame count suggests"
        )
    if kept == input_points:
        notes.append(
            "the grid merged nothing: as many points came out as went in, so this result is the "
            "inputs concatenated and reordered, and the noise averaging this module exists for "
            "did not happen"
        )
    return notes


def _all_limitations(
    notes: Sequence[str], input_limitations: Sequence[Sequence[str]]
) -> tuple[str, ...]:
    """This module's caveats, then this run's notes, then the inputs' own.

    De-duplicated, order preserved: clouds from one capture session share almost
    every limitation they have, and a consumer should not have to read the same
    sentence thirty-six times to find the one that differs. The inputs' entries
    come last and are never dropped - a fusion inherits what was uncertain about
    what it merged, and this is where that inheritance happens.
    """
    carried = (item for group in input_limitations for item in group)
    return tuple(dict.fromkeys((*INHERENT_LIMITATIONS, *notes, *carried)))


def fuse(
    clouds: Sequence[Sequence[Point]],
    transforms: Sequence[Rigid],
    *,
    voxel_size: float,
    min_observations: int = 1,
    input_limitations: Sequence[Sequence[str]] = (),
) -> FusedCloud:
    """Merge registered clouds into one by averaging within voxel cells.

    ``transforms[i]`` carries ``clouds[i]`` into the shared frame; a cloud
    already in that frame gets :meth:`Rigid.identity`. Points are binned into a
    grid of ``voxel_size`` cells and each occupied cell contributes the mean of
    the points that fell in it, so overlapping views of one surface collapse to
    one sheet of points instead of stacking into a shell.

    ``min_observations`` requires a cell to have been reached by that many
    *distinct source clouds* before it survives. It is the difference between
    fusion and concatenation, and it removes real single-view surface along with
    the noise - see the module docstring, and read
    :attr:`FusedCloud.dropped_voxels` on the result to see how much it took.

    ``input_limitations`` is one tuple of caveats per cloud, or nothing. Whatever
    is given is unioned into :attr:`FusedCloud.limitations` after this module's
    own, because merging clouds does not launder what was uncertain about them.

    Every refusal happens before the points are walked, except the two that are
    defects *in* the points - a non-finite coordinate and an entry that is not a
    triple - which are refused during the bounding pass, before a single voxel
    exists.
    """
    size = ensure_voxel_size_usable(voxel_size)
    ensure_transforms_match(clouds, transforms)
    ensure_limitations_match(clouds, input_limitations)
    input_points = ensure_clouds_usable(clouds)
    threshold = ensure_threshold_reachable(
        min_observations, sum(1 for cloud in clouds if len(cloud))
    )

    low, high = _extent(clouds, transforms)
    ensure_grid_bounded(low, high, size)
    grid = _accumulate(clouds, transforms, size)
    points = _reduce(grid, threshold)

    notes = _run_notes(
        clouds=clouds,
        min_observations=threshold,
        occupied=len(grid),
        kept=len(points),
        input_points=input_points,
    )
    return FusedCloud(
        points=points,
        voxel_size=size,
        source_frames=len(clouds),
        input_points=input_points,
        occupied_voxels=len(grid),
        limitations=_all_limitations(notes, input_limitations),
        min_observations=threshold,
    )


def voxel_downsample(points: Sequence[Point], *, voxel_size: float) -> tuple[Point, ...]:
    """One averaged point per occupied cell of a ``voxel_size`` grid.

    The thinning step every iterative matcher needs before it starts: a single
    depth frame yields up to 89000 points and nearest-neighbour search over that
    is not something pure Python does in a reasonable time, so ICP downsamples
    first and pays for its iterations on a few thousand points instead.

    This is :func:`fuse` of one cloud under the identity, sharing its refusals
    and its arithmetic rather than reimplementing either - two voxel grids in one
    module that rounded differently would put the matcher and the fused output
    on slightly different lattices, and that discrepancy is invisible until it is
    a misaligned scan.

    Idempotent at the same ``voxel_size``: a cell's mean lies inside that cell,
    so a second pass finds one point per cell and returns it unchanged. The one
    exception is a mean that lands on a cell boundary, where which side it falls
    on is float64 rounding's decision - the limitation on every
    :class:`FusedCloud` says the same thing about every other point.
    """
    return fuse((points,), (_IDENTITY,), voxel_size=voxel_size).points
