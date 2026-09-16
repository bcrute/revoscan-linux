"""Which parts of the object still need scanning, from registered clouds.

The problem this answers
-------------------------
A capture session - turntable or handheld - gives no feedback about what has
actually been seen. The live view shows the current depth frame and nothing
else; a user can stare at a twenty-minute capture and only discover afterwards,
looking at the fused result, that the underside of the object is a hole. This
module exists to answer, from the clouds and poses gathered so far, "which
parts of the object still need scanning?" so that question can be answered
*during* the capture instead of after it.

Two things make a region well-scanned, and counting only sees the first
----------------------------------------------------------------------
1. How many frames saw it - the naive answer, and the one every simple
   "coverage map" stops at.
2. The angular spread of the directions it was seen from. A surface observed
   by forty frames that all viewed it from nearly the same direction is still
   badly scanned: those forty observations triangulate nothing, because
   triangulation needs a baseline and they share none. Its true shape along the
   line of sight is as unconstrained as if it had been seen once. This is the
   measure a count-only coverage map misses entirely, and it is the one that
   matters most - see :func:`_max_pairwise_angle` for the actual computation
   and its own stated limitation.

A cell is reported "well covered" only when both the observation count and the
angular spread clear their thresholds. Either alone is gameable: a slow orbit
that barely moves gives high count and low spread; a sparse set of frames
flung wide gives high spread and low count. Neither is a scan anyone should
trust.

What this does NOT do
----------------------
It has no notion of surface normals, texture, or reconstruction quality beyond
count and viewing-angle diversity. A cell can be "well covered" by this
measure and still be a bad reconstruction, if depth noise at that range and
angle happens to be large - nothing here measures noise. It also cannot tell
a genuine gap in the object (a real hole) from a gap in the scan; both look
like an unoccupied cell, because no cell exists to report on for either.

Every cell is a bin on a rigid grid fixed to the world frame the caller's
transforms place points into. A cell straddling a true edge or corner of the
object mixes surfaces that face in materially different directions, which can
make a well-observed edge score worse than either face alone would. Choosing
``cell`` much smaller than the object's finest feature avoids this at the cost
of more cells, each needing its own observations to fill.

No physical unit is implied anywhere, matching every other module in this
package. Distances are in whatever the depth scale produced, and ``cell`` must
be chosen in the same units as the point clouds it bins.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.geometry import DegenerateVector, Point, Rigid, dot, normalize, subtract

#: Serialised shape version for every record this module emits.
SCHEMA_VERSION: Final = 1

#: A cell counts as well covered only once its viewing directions span at
#: least this angle. Chosen so that a turntable frame spacing typical of a real
#: sweep (a few degrees between neighbours) comfortably clears it once several
#: frames have contributed, while a handful of nearly-identical views - or a
#: region seen from only a narrow grazing band, such as a pole on the
#: turntable's own axis - does not. It is a default, not a measured constant;
#: a caller with a tighter or looser triangulation requirement should pass
#: their own.
DEFAULT_MIN_SPREAD: Final = math.radians(20.0)

#: Grid cell index: floor(coordinate / cell) per axis, world-frame.
_Key = tuple[int, int, int]

#: The six face-adjacent neighbours of a grid cell. Face connectivity only, not
#: the full 26: a diagonal step slips between two cells that share only an
#: edge, which would stitch together two regions of the object that are not
#: actually contiguous. :mod:`~scanner3d.mesh` picks the same connectivity for
#: the same reason.
_FACE_NEIGHBOURS: Final[tuple[_Key, ...]] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


class CoverageNotAllowed(ScannerError):
    """A coverage query was made that this module refuses to answer."""

    status = DoctorStatus.UNSUPPORTED_MODE


class CloudTransformCountMismatch(CoverageNotAllowed):
    """The number of clouds and the number of transforms disagree."""


class NoCloudsOffered(CoverageNotAllowed):
    """No cloud was given at all, so there is nothing to map."""


class NothingToCover(CoverageNotAllowed):
    """Clouds and transforms were given, but no point yielded a usable observation.

    Every point either had zero depth relative to its own frame's camera
    centre - which carries no viewing direction - or every cloud was empty.
    """


class CellSizeUnusable(CoverageNotAllowed):
    """The requested cell size is not a positive, finite number."""


class MinObservationsUnusable(CoverageNotAllowed):
    """``min_observations`` is not a positive integer.

    Below 1 it would call every occupied cell well-observed regardless of
    whether anything actually looked at it, which defeats the point of the
    threshold rather than merely relaxing it.
    """


class MinSpreadUnusable(CoverageNotAllowed):
    """``min_spread`` is not a finite radian value in ``[0, pi]``.

    ``pi`` is the largest angle two unit vectors can subtend, so anything past
    it could never be satisfied and is refused rather than silently
    unreachable.
    """


@dataclass(frozen=True, slots=True)
class CoverageCell:
    """One occupied grid cell: what part of the object it is, how well it was seen.

    ``observations`` counts distinct frames, not points: ten points landing in
    one cell from a single frame constrain the surface no better than one does,
    matching the observation accounting :mod:`~scanner3d.fusion` uses for the
    same reason.
    """

    cell_index: _Key
    centroid: Point
    observations: int
    spread: float
    well_covered: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_index": list(self.cell_index),
            "centroid": list(self.centroid),
            "observations": self.observations,
            "spread_radians": self.spread,
            "spread_degrees": math.degrees(self.spread),
            "well_covered": self.well_covered,
        }


@dataclass(frozen=True, slots=True)
class ThinRegion:
    """Adjacent weakly-covered cells reported as one region.

    A user needs "the underside is thin", not a list of a thousand isolated
    cells that happen to be underneath the object; this is what turns the
    latter into the former. ``worst_observations`` and ``worst_spread`` are the
    weakest single cell's values in the region - the number that explains why
    the whole region failed, not an average that could hide a badly-seen
    corner inside an otherwise decent patch.
    """

    centroid: Point
    cell_count: int
    worst_observations: int
    worst_spread: float

    def as_dict(self) -> dict[str, object]:
        return {
            "centroid": list(self.centroid),
            "cell_count": self.cell_count,
            "worst_observations": self.worst_observations,
            "worst_spread_radians": self.worst_spread,
            "worst_spread_degrees": math.degrees(self.worst_spread),
        }


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Every occupied cell's coverage, and the thresholds that judged it."""

    cells: tuple[CoverageCell, ...]
    cell_size: float
    min_observations: int
    min_spread: float
    limitations: tuple[str, ...]

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def well_covered_cells(self) -> int:
        return sum(1 for cell in self.cells if cell.well_covered)

    @property
    def well_covered_fraction(self) -> float:
        """Fraction of occupied cells that are well covered.

        ``0.0`` when there are no occupied cells at all, rather than raising or
        returning NaN - an empty map has no coverage to report a fraction of,
        and 0 is the answer that keeps a caller's downstream arithmetic from
        having to special-case it.
        """
        return self.well_covered_cells / len(self.cells) if self.cells else 0.0

    def thin_cells(self) -> tuple[CoverageCell, ...]:
        """Every weakly-covered cell, unclustered.

        Rarely what a UI wants directly - see :meth:`thin_regions` - but kept
        for a caller with a different use, such as colouring a live point
        cloud cell by cell.
        """
        return tuple(cell for cell in self.cells if not cell.well_covered)

    def thin_regions(self) -> tuple[ThinRegion, ...]:
        """Weakly-covered cells clustered into contiguous regions, largest first."""
        return _cluster_thin_cells(self.cells)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "cell_size": self.cell_size,
            "min_observations": self.min_observations,
            "min_spread_radians": self.min_spread,
            "total_cells": self.total_cells,
            "well_covered_cells": self.well_covered_cells,
            "well_covered_fraction": self.well_covered_fraction,
            "thin_regions": [region.as_dict() for region in self.thin_regions()],
            "limitations": list(self.limitations),
        }


def _cell_key(point: Point, cell: float) -> _Key:
    return (
        math.floor(point[0] / cell),
        math.floor(point[1] / cell),
        math.floor(point[2] / cell),
    )


def _max_pairwise_angle(directions: Sequence[Point]) -> float:
    """The largest angle between any two viewing directions, in radians.

    Chosen over the angle subtended by the directions' mean resultant vector
    because it answers the question this module actually cares about: whether
    *some* pair of views separates enough to triangulate the surface, not how
    concentrated the whole set is around its average. The mean-resultant
    measure can be pulled toward zero spread by a large cluster of near-
    duplicate frames even when a few genuinely wide-baseline views are mixed
    in, understating exactly the coverage that baseline provides.

    Its own limitation, stated as plainly: it is the best pair, not a
    description of the whole set. A single well-separated frame among many
    near-duplicates is enough to report a wide spread, even though only that
    one pair actually constrains the surface - most of the "spread" observed
    here contributed nothing. A region flagged well-covered by this measure
    alone is not necessarily well triangulated from *every* angle, only from
    at least two.

    Quadratic in the number of directions, which is fine: a cell is observed by
    at most one direction per contributing frame, and per-cell frame counts
    stay small even for a long capture.
    """
    count = len(directions)
    if count < 2:
        return 0.0
    if 8 <= count <= 64:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            values = np.asarray(directions, dtype=np.float64)
            first, second = _pair_indices(count)
            products = values[list(first)] * values[list(second)]
            approximate = (products[:, 0] + products[:, 1]) + products[:, 2]
            # Unit vectors bound each product by one. This envelope exceeds
            # the three-term rounding error: only contenders are recomputed,
            # but the final minimum uses the original correctly rounded dot.
            contenders = np.flatnonzero(approximate <= approximate.min() + 1e-14)
            cosine = min(dot(directions[first[i]], directions[second[i]]) for i in contenders)
            return math.acos(max(-1.0, min(1.0, cosine)))
    widest = 0.0
    for i in range(count):
        a = directions[i]
        for j in range(i + 1, count):
            b = directions[j]
            cosine = max(-1.0, min(1.0, dot(a, b)))
            angle = math.acos(cosine)
            if angle > widest:
                widest = angle
    return widest


@lru_cache(maxsize=64)
def _pair_indices(count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    return tuple(i for i, _ in pairs), tuple(j for _, j in pairs)


def _frame_observations(
    cloud: Sequence[Point], transform: Rigid, cell: float
) -> Iterator[tuple[Point, Point, _Key]]:
    camera_centre = transform.apply((0.0, 0.0, 0.0))
    for local_point in cloud:
        world_point = transform.apply(local_point)
        try:
            direction = normalize(subtract(world_point, camera_centre))
        except DegenerateVector:
            continue
        yield world_point, direction, _cell_key(world_point, cell)


@lru_cache(maxsize=64)
def _cached_frame_observations(
    cloud: tuple[Point, ...], transform: Rigid, cell: float
) -> tuple[tuple[Point, Point, _Key], ...]:
    """Reuse immutable placed preview frames; changed poses miss the cache."""
    return tuple(_frame_observations(cloud, transform, cell))


def _accumulate(
    clouds: Sequence[Sequence[Point]], transforms: Sequence[Rigid], cell: float
) -> tuple[dict[_Key, list[float]], dict[_Key, dict[int, list[float]]]]:
    """Bin every placed point by cell, and its viewing direction by cell and frame.

    Two accumulators rather than one. The centroid only ever needs a running
    coordinate sum, but a viewing direction has to be kept per *frame* within a
    cell: the spread this module measures is spread across frames, not across
    however many individual points one frame happened to drop in the cell, so
    a frame that contributed fifty points to a cell must count as one
    direction, not fifty.
    """
    points: dict[_Key, list[float]] = {}
    directions: dict[_Key, dict[int, list[float]]] = {}
    for frame_index, (cloud, transform) in enumerate(zip(clouds, transforms, strict=True)):
        # Both entry count and per-entry size are bounded. Large reconstruction
        # clouds remain streaming, and all accumulator additions retain their
        # original order: this cache changes no coverage arithmetic.
        observations = (
            _cached_frame_observations(tuple(cloud), transform, cell)
            if len(cloud) <= 2048
            else _frame_observations(cloud, transform, cell)
        )
        for world_point, direction, key in observations:
            point_bucket = points.get(key)
            if point_bucket is None:
                points[key] = [world_point[0], world_point[1], world_point[2], 1.0]
            else:
                point_bucket[0] += world_point[0]
                point_bucket[1] += world_point[1]
                point_bucket[2] += world_point[2]
                point_bucket[3] += 1.0

            frame_directions = directions.setdefault(key, {})
            direction_bucket = frame_directions.get(frame_index)
            if direction_bucket is None:
                frame_directions[frame_index] = [direction[0], direction[1], direction[2], 1.0]
            else:
                direction_bucket[0] += direction[0]
                direction_bucket[1] += direction[1]
                direction_bucket[2] += direction[2]
                direction_bucket[3] += 1.0
    return points, directions


def _build_cells(
    points: dict[_Key, list[float]],
    directions: dict[_Key, dict[int, list[float]]],
    *,
    min_observations: int,
    min_spread: float,
) -> tuple[CoverageCell, ...]:
    """One :class:`CoverageCell` per occupied cell, in ascending cell index.

    Sorted so that two coverage maps of the same data agree cell for cell in
    the same order, rather than in whatever order a dict happened to see the
    input clouds - the same reasoning :func:`~scanner3d.fusion._reduce` gives
    for its own sort.
    """
    cells: list[CoverageCell] = []
    for key in sorted(points):
        sx, sy, sz, n = points[key]
        centroid = (sx / n, sy / n, sz / n)

        unit_directions: list[Point] = []
        for dx, dy, dz, dn in directions[key].values():
            try:
                unit_directions.append(normalize((dx / dn, dy / dn, dz / dn)))
            except DegenerateVector:
                # This one frame's own points, averaged, cancelled to no
                # direction at all - a coincidence of geometry, not a fault.
                # The frame is simply not counted for this cell.
                continue

        observations = len(unit_directions)
        spread = _max_pairwise_angle(unit_directions)
        well_covered = observations >= min_observations and spread >= min_spread
        cells.append(
            CoverageCell(
                cell_index=key,
                centroid=centroid,
                observations=observations,
                spread=spread,
                well_covered=well_covered,
            )
        )
    return tuple(cells)


def _region_of(members: Sequence[CoverageCell]) -> ThinRegion:
    count = len(members)
    cx = math.fsum(cell.centroid[0] for cell in members) / count
    cy = math.fsum(cell.centroid[1] for cell in members) / count
    cz = math.fsum(cell.centroid[2] for cell in members) / count
    return ThinRegion(
        centroid=(cx, cy, cz),
        cell_count=count,
        worst_observations=min(cell.observations for cell in members),
        worst_spread=min(cell.spread for cell in members),
    )


def _cluster_thin_cells(cells: Sequence[CoverageCell]) -> tuple[ThinRegion, ...]:
    """Flood-fill weakly-covered cells into contiguous regions, largest first.

    A plain per-cell report leaves a user to piece "the underside is thin"
    together from hundreds of individual points; clustering does that piecing
    for them. Ordering by size puts the region most worth pointing a user at
    - the one that will take longest to fix - first.
    """
    by_key: dict[_Key, CoverageCell] = {
        cell.cell_index: cell for cell in cells if not cell.well_covered
    }
    visited: set[_Key] = set()
    regions: list[ThinRegion] = []
    for start in sorted(by_key):
        if start in visited:
            continue
        visited.add(start)
        queue: deque[_Key] = deque([start])
        members: list[CoverageCell] = []
        while queue:
            key = queue.popleft()
            members.append(by_key[key])
            for dx, dy, dz in _FACE_NEIGHBOURS:
                neighbour = (key[0] + dx, key[1] + dy, key[2] + dz)
                if neighbour in by_key and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        regions.append(_region_of(members))
    regions.sort(key=lambda region: region.cell_count, reverse=True)
    return tuple(regions)


_LIMITATIONS: Final[tuple[str, ...]] = (
    "A cell's spread is the maximum pairwise angle between the viewing "
    "directions that reached it, which can be dominated by a single "
    "well-separated frame - see _max_pairwise_angle. A cell just past the "
    "threshold is not necessarily well triangulated from every direction, "
    "only from at least one good pair.",
    "This measures count and viewing-angle diversity only. It has no model of "
    "depth noise, surface normals, or texture, so a well-covered cell can "
    "still be a poor reconstruction for reasons this report cannot see.",
    "An unoccupied region of the grid and a genuine hole in the object look "
    "identical here: both are simply the absence of a cell. This module "
    "cannot tell missing scan from missing object.",
    "Cells are bins on a rigid world-frame grid. A cell straddling a true "
    "edge or corner mixes surfaces facing in different directions, which can "
    "make a well-observed edge score worse than either face alone.",
    "No physical unit is implied. Distances are whatever the depth scale "
    "produced, and cell must be chosen in the same units as the input clouds.",
)


def coverage_map(
    clouds: Sequence[Sequence[Point]],
    transforms: Sequence[Rigid],
    *,
    cell: float,
    min_observations: int = 3,
    min_spread: float = DEFAULT_MIN_SPREAD,
) -> CoverageReport:
    """Bin every cloud's points into world-frame cells and judge each cell's coverage.

    Each cloud is in ITS OWN camera frame; ``transforms[i]`` places cloud ``i``
    into the shared world frame the report is expressed in. The viewing
    direction for one of cloud ``i``'s points is taken from that frame's camera
    centre - ``transforms[i].apply((0, 0, 0))``, since a cloud's points are in
    camera coordinates where the camera sits at the origin - to the point in
    world space.

    A cell is ``well_covered`` only when it clears both thresholds: at least
    ``min_observations`` distinct frames saw it, and those frames' viewing
    directions span at least ``min_spread`` radians. See the module docstring
    for why both are required and neither alone is trustworthy.
    """
    if len(clouds) != len(transforms):
        raise CloudTransformCountMismatch(
            f"{len(clouds)} clouds against {len(transforms)} transforms: one transform per cloud"
        )
    if not clouds:
        raise NoCloudsOffered("coverage needs at least one cloud to map")
    if not math.isfinite(cell) or not cell > 0.0:
        raise CellSizeUnusable(f"cell must be positive and finite, got {cell!r}")
    if min_observations < 1:
        raise MinObservationsUnusable(
            f"min_observations must be at least 1, got {min_observations}"
        )
    if not math.isfinite(min_spread) or not 0.0 <= min_spread <= math.pi:
        raise MinSpreadUnusable(
            f"min_spread must be a finite radian value in [0, pi], got {min_spread!r}"
        )

    points, directions = _accumulate(clouds, transforms, cell)
    if not points:
        raise NothingToCover(
            "no point across any cloud yielded a usable observation - every cloud was empty, "
            "or every point coincided with its own frame's camera centre"
        )

    cells = _build_cells(
        points, directions, min_observations=min_observations, min_spread=min_spread
    )
    return CoverageReport(
        cells=cells,
        cell_size=cell,
        min_observations=min_observations,
        min_spread=min_spread,
        limitations=_LIMITATIONS,
    )
