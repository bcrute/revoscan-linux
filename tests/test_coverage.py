"""Tests for coverage: which parts of a scan still need more scanning.

What is at stake
-----------------
The user's own complaint, verbatim: "you have to know what you've scanned,
what needs better scanning." The live viewer shows only the current depth
frame; nothing tells a user, twenty minutes into a capture, that the
underside of the object is still a hole. :mod:`~scanner3d.coverage` is the
data this report will eventually be built from, and this file is where that
promise gets checked against ground truth rather than taken on faith.

Two properties matter more than any other, and the tests are organised around
them:

* **A naive, count-only coverage map is not enough.** Forty frames that all
  viewed a surface from nearly the same direction have triangulated nothing -
  their baseline is zero.
  :func:`test_many_frames_from_nearly_the_same_direction_are_thin_despite_high_count`
  is the test that proves the angular-spread half of the measure is actually
  doing something: a count-only implementation would call that surface
  well-scanned, and this test is built specifically so that mistake fails
  loudly.
* **The report has to say WHERE the gaps are, in a shape a user can act on.**
  :func:`test_full_sweep_leaves_sides_well_covered_but_poles_thin` and
  :func:`test_half_sweep_reports_the_unscanned_side_as_thin` are built from a
  real :func:`~scanner3d.simulate.turntable_sweep` with ground-truth poses -
  the half sweep is literally a prefix of the full sweep's frames, so the two
  are directly comparable and the difference between them is entirely the
  missing frames, nothing else. The half sweep is the single most important
  test in this file: "you missed a side" is exactly the failure mode a real
  capture session produces, and exactly what this module exists to catch
  while there is still a scanner in hand to fix it.

Test data
---------
A sphere centred on the turntable's own rotation axis is the shape used for
the sweep tests. It is not there for its symmetry - the render only shows the
camera-facing side of any shape, symmetric or not - it is there because its
curvature makes the effect being tested unambiguous: every latitude except the
poles sweeps through a wide range of viewing directions as the object turns,
so the poles standing out as thin is a property of viewing geometry, not an
artefact of flat faces or hard edges that a boxier shape would also introduce.
Both sweep tests decimate each frame with
:func:`~scanner3d.registration.voxel_downsample` before handing it to
:func:`~scanner3d.coverage.coverage_map`, exactly as the module's own
docstring recommends against rendering hundreds of thousands of raw points
into the accumulator.

The same-direction test does not use the renderer at all. It hand-builds
camera poses with :func:`~scanner3d.geometry.rigid_from_correspondences`,
because the point under test - many observations, one direction - is a
property of camera placement, not of a rendered scene, and hand-building it
gives exact control over the one variable that matters.
"""

from __future__ import annotations

import functools
import json
import math
import random
from collections.abc import Sequence

import pytest

from scanner3d.coverage import (
    DEFAULT_MIN_SPREAD,
    SCHEMA_VERSION,
    CellSizeUnusable,
    CloudTransformCountMismatch,
    CoverageCell,
    CoverageReport,
    MinObservationsUnusable,
    MinSpreadUnusable,
    NoCloudsOffered,
    NothingToCover,
    coverage_map,
)
from scanner3d.geometry import (
    Point,
    Rigid,
    add,
    cross,
    normalize,
    rigid_from_correspondences,
    rotation_about_axis,
    subtract,
)
from scanner3d.registration import voxel_downsample
from scanner3d.simulate import Sphere, points_of, turntable_sweep

# --------------------------------------------------------------------------
# The shared turntable sweep: rendered once, sliced for both sweep tests.
# --------------------------------------------------------------------------

_SPHERE = Sphere(centre=(0.0, 0.0, 150.0), radius=40.0)
_SWEEP_FRAMES = 24
_SWEEP_AXIS: Point = (0.0, 1.0, 0.0)
_SWEEP_THROUGH: Point = (0.0, 0.0, 150.0)
_DECIMATE_CELL = 2.0
_MAP_CELL = 5.0
_MIN_OBSERVATIONS = 3
_MIN_SPREAD = math.radians(20.0)

# The half sweep is a prefix of the full sweep's frames - the first half of
# the rotation, 0 through 165 degrees - not a separately rendered scene. That
# is what makes "the difference is exactly the missing frames" true rather
# than merely asserted.
_HALF_FRAME_COUNT = _SWEEP_FRAMES // 2


@functools.cache
def _sphere_sweep() -> tuple[tuple[tuple[Point, ...], ...], tuple[Rigid, ...]]:
    """Ground-truth turntable sweep of a sphere, decimated per frame, cached.

    Rendering and deprojecting 24 full 640x400 frames costs several seconds;
    caching means the cost is paid once and shared by both sweep tests below.
    """
    frames, poses = turntable_sweep(
        [_SPHERE], frames=_SWEEP_FRAMES, axis=_SWEEP_AXIS, through=_SWEEP_THROUGH
    )
    clouds = tuple(voxel_downsample(tuple(points_of(f)), _DECIMATE_CELL) for f in frames)
    transforms = tuple(pose.inverse() for pose in poses)
    return clouds, transforms


def _well_covered_fraction(cells: Sequence[CoverageCell]) -> float:
    assert cells, "the band under test unexpectedly contains no cells"
    return sum(1 for c in cells if c.well_covered) / len(cells)


def _azimuth_degrees(cell: CoverageCell) -> float:
    """Azimuth of a cell around the sweep's own axis, degrees, centred on the sphere."""
    return math.degrees(math.atan2(cell.centroid[0], cell.centroid[2] - _SWEEP_THROUGH[2]))


# --------------------------------------------------------------------------
# Full sweep: sides well covered, poles thin.
# --------------------------------------------------------------------------


def test_full_sweep_leaves_sides_well_covered_but_poles_thin():
    clouds, transforms = _sphere_sweep()
    report = coverage_map(
        clouds,
        transforms,
        cell=_MAP_CELL,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
    )

    # "Sides" = latitudes well away from the rotation axis; "poles" = the cap
    # right around it, where a level turntable orbit only ever grazes the
    # surface. Measured: sides 93.9% well covered, poles 48.1%.
    equator = [c for c in report.cells if abs(c.centroid[1]) < 15.0]
    poles = [c for c in report.cells if abs(c.centroid[1]) > 27.0]
    equator_fraction = _well_covered_fraction(equator)
    pole_fraction = _well_covered_fraction(poles)
    assert equator_fraction > 0.85, f"equator should be well covered, got {equator_fraction:.3f}"
    assert pole_fraction < 0.60, f"poles should be thin, got {pole_fraction:.3f}"
    assert equator_fraction - pole_fraction > 0.25

    # The report has to name where the gap is, not just that one exists.
    # Measured: the largest thin region is 44 cells centred at y=30.4, near the
    # sphere's own pole (radius 40) - exactly "the top is thin".
    regions = report.thin_regions()
    assert regions, "a sweep with a genuinely thin pole must report a thin region there"
    largest = max(regions, key=lambda r: r.cell_count)
    assert largest.cell_count >= 15
    assert abs(largest.centroid[1]) > 25.0, (
        f"the largest thin region should sit near the pole (|y| > 25), "
        f"got centroid y={largest.centroid[1]:.1f}"
    )


# --------------------------------------------------------------------------
# Half sweep: the unscanned side is thin. The single most important test here.
# --------------------------------------------------------------------------


def test_half_sweep_reports_the_unscanned_side_as_thin():
    full_clouds, full_transforms = _sphere_sweep()
    half_clouds = full_clouds[:_HALF_FRAME_COUNT]
    half_transforms = full_transforms[:_HALF_FRAME_COUNT]

    full_report = coverage_map(
        full_clouds,
        full_transforms,
        cell=_MAP_CELL,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
    )
    half_report = coverage_map(
        half_clouds,
        half_transforms,
        cell=_MAP_CELL,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
    )

    # Headline number: covering only the first 165 degrees of rotation should
    # be markedly worse overall than the full sweep of the same object.
    # Measured: full 89.8% well covered, half 69.7%.
    assert half_report.well_covered_fraction < full_report.well_covered_fraction - 0.10, (
        f"half sweep ({half_report.well_covered_fraction:.3f}) should be clearly worse than "
        f"the full sweep ({full_report.well_covered_fraction:.3f})"
    )

    # Restricted to the equatorial band (poles are separately thin regardless
    # of sweep length - see the full-sweep test - so excluding them isolates
    # the azimuthal effect this test is actually about): the side the sweep
    # faced should be well covered, the side it never turned to face is thin.
    # Measured: faced side 95.4% well covered, unfaced side 23.5%.
    equatorial = [c for c in half_report.cells if abs(c.centroid[1]) < 15.0]
    faced = [c for c in equatorial if 0.0 <= _azimuth_degrees(c) < 120.0]
    unfaced = [c for c in equatorial if -180.0 <= _azimuth_degrees(c) < -20.0]
    faced_fraction = _well_covered_fraction(faced)
    unfaced_fraction = _well_covered_fraction(unfaced)
    assert faced_fraction > 0.85, f"the faced side should be well covered, got {faced_fraction:.3f}"
    assert unfaced_fraction < 0.40, f"the unfaced side should be thin, got {unfaced_fraction:.3f}"

    # The control this claim actually needs: the SAME azimuth band, read off the
    # FULL sweep instead of the half. A turntable sweep is symmetric, so on the
    # full sweep both bands are well covered - proving "unfaced is thin" is a
    # fact about the missing frames, not a coincidence of how this band happens
    # to be carved out of the sphere. Measured: 94.3% on the full sweep, versus
    # 23.5% above on the half. Without this, the test could pass unchanged on a
    # module that made every azimuth band read as thin regardless of coverage.
    full_equatorial = [c for c in full_report.cells if abs(c.centroid[1]) < 15.0]
    full_unfaced = [c for c in full_equatorial if -180.0 <= _azimuth_degrees(c) < -20.0]
    full_unfaced_fraction = _well_covered_fraction(full_unfaced)
    assert full_unfaced_fraction > 0.85, (
        f"the same band must be well covered once the sweep actually covers it, "
        f"got {full_unfaced_fraction:.3f} - otherwise the half-sweep result above proves nothing"
    )

    regions = half_report.thin_regions()
    assert regions, "a half sweep must report the unscanned side as a thin region"
    assert max(r.cell_count for r in regions) >= 15


# --------------------------------------------------------------------------
# Many observations, one direction: the test that proves spread matters.
# --------------------------------------------------------------------------


def _look_at(camera_centre: Point, target: Point, up: Point = (0.0, 1.0, 0.0)) -> Rigid:
    """A camera pose placing ``target`` on the local +Z axis, exactly.

    Built from four exact correspondences rather than by composing a rotation
    matrix by hand, so the transform is produced by the same machinery
    (:func:`~scanner3d.geometry.rigid_from_correspondences`) every other test
    in this repository trusts, instead of a bespoke formula that could itself
    be wrong. ``right`` and ``true_up`` are ordered so that
    ``(right, true_up, forward)`` is right-handed - swapping that order maps
    the local basis onto its mirror image, which Horn's method cannot fit
    exactly and silently returns a poor best-effort rotation for instead of
    failing loudly.
    """
    forward = normalize(subtract(target, camera_centre))
    right = normalize(cross(up, forward))
    true_up = cross(forward, right)
    local: list[Point] = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    world = [
        camera_centre,
        add(camera_centre, forward),
        add(camera_centre, right),
        add(camera_centre, true_up),
    ]
    return rigid_from_correspondences(local, world)


#: Off a grid boundary on purpose - a target at the origin would have half its
#: observations round to one world cell and half to its neighbour on floating
#: point sign noise alone, which is a grid artefact and not what this test is
#: about.
_TARGET: Point = (7.5, -3.2, 12.1)
_DEPTH = 50.0


def _frame_observing(
    direction: Point, *, target: Point = _TARGET
) -> tuple[tuple[Point, ...], Rigid]:
    """A one-point cloud, seen by a camera ``_DEPTH`` from ``target`` along ``direction``."""
    unit = normalize(direction)
    camera_centre: Point = (
        target[0] - _DEPTH * unit[0],
        target[1] - _DEPTH * unit[1],
        target[2] - _DEPTH * unit[2],
    )
    transform = _look_at(camera_centre, target)
    return ((0.0, 0.0, _DEPTH),), transform


def test_many_frames_from_nearly_the_same_direction_are_thin_despite_high_count():
    rng = random.Random(1234)
    base = normalize((0.3, 0.1, -1.0))
    clouds = []
    transforms = []
    for _ in range(20):
        jitter = math.radians(rng.uniform(-0.5, 0.5))
        axis = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        direction = rotation_about_axis(axis, jitter).apply(base)
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    assert report.total_cells == 1, "every frame was built to observe the same world point"
    cell = report.cells[0]
    assert cell.observations == 20
    assert cell.spread < math.radians(5.0), (
        "the twenty viewing directions were built within a degree of each other"
    )
    assert not cell.well_covered, (
        "high observation count alone must not be enough - a count-only coverage map would "
        "wrongly call this well covered"
    )
    assert report.thin_cells() == (cell,)


def test_a_few_widely_separated_directions_are_well_covered_despite_low_count():
    """The other half of the same proof: it is spread that is missing above, not merely count."""
    directions = [(0.3, 0.1, -1.0), (1.0, 0.2, -1.0), (-1.0, 0.1, -1.0), (0.1, 1.0, -1.0)]
    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    assert report.total_cells == 1
    cell = report.cells[0]
    assert cell.observations == len(directions)
    assert cell.spread > _MIN_SPREAD
    assert cell.well_covered


def test_exactly_min_observations_with_good_spread_is_well_covered():
    """The observation-count boundary itself: exactly the threshold must pass, not just above it.

    A test built with comfortably more than ``_MIN_OBSERVATIONS`` cannot tell
    ``>=`` from ``>`` apart - both accept it. Using exactly the threshold is
    what an off-by-one in that comparison would actually get wrong.
    """
    directions = [(0.3, 0.1, -1.0), (1.0, 0.2, -1.0), (-1.0, 0.1, -1.0)]
    assert len(directions) == _MIN_OBSERVATIONS
    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    cell = report.cells[0]
    assert cell.observations == _MIN_OBSERVATIONS
    assert cell.well_covered


def test_fewer_than_min_observations_are_thin_even_with_wide_spread():
    """The mirror image of the same-direction test: spread alone is not enough either.

    Two widely-separated views, one short of ``_MIN_OBSERVATIONS``. If the
    observation-count check were ever dropped, wide spread alone would be
    enough to call this well covered - it must not be.
    """
    directions = [(0.3, 0.1, -1.0), (-1.0, 0.1, -1.0)]
    assert len(directions) < _MIN_OBSERVATIONS
    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    cell = report.cells[0]
    assert cell.observations == len(directions)
    assert cell.spread > _MIN_SPREAD, "the two views were built wide apart on purpose"
    assert not cell.well_covered


def test_spread_at_exactly_min_spread_is_well_covered():
    """The spread boundary itself, mirroring the observation-count boundary above.

    ``test_exactly_min_observations_with_good_spread_is_well_covered`` pins down
    the ``>=`` on observation count; nothing else in this file pins down the
    ``>=`` on spread - every other test's spread is comfortably above or below
    its threshold, and both ``>`` and ``>=`` would agree on those. Hand-picking
    an angle and hoping it lands exactly on ``_MIN_SPREAD`` would be fighting
    floating point for no reason, so instead the spread is measured first (with
    the threshold held open at 0.0) and then fed back in as the threshold
    itself - comparing that exact float to itself is what makes this a true
    boundary case rather than an approximation of one.
    """
    directions = [(0.3, 0.1, -1.0), (1.0, 0.2, -1.0), (-1.0, 0.1, -1.0), (0.1, 1.0, -1.0)]
    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    measured = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=0.0
    )
    boundary_spread = measured.cells[0].spread

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=boundary_spread
    )
    cell = report.cells[0]
    assert cell.spread == boundary_spread, "recomputing from the same input must be deterministic"
    assert cell.well_covered, "a cell whose spread exactly equals the threshold must still pass"


def test_many_points_from_one_frame_count_as_a_single_observation():
    """``observations`` counts distinct frames, not points - stated twice in the
    module's own docstrings, and untestable by every other test in this file,
    because every other test happens to give each frame exactly one point.

    Two frames, five points apiece, all landing in the same cell: if points
    were being counted, ten observations would clear ``_MIN_OBSERVATIONS`` and
    a healthy 61-degree spread would clear ``_MIN_SPREAD``, so the cell would
    read as well covered. It must not - only two frames ever looked at this
    surface, however many points either of them happened to report.
    """
    n_points = 5

    def frame_with_many_points(direction: Point) -> tuple[tuple[Point, ...], Rigid]:
        unit = normalize(direction)
        camera_centre: Point = (
            _TARGET[0] - _DEPTH * unit[0],
            _TARGET[1] - _DEPTH * unit[1],
            _TARGET[2] - _DEPTH * unit[2],
        )
        transform = _look_at(camera_centre, _TARGET)
        # Small offsets along local x so the points are distinct but still
        # decimal-close enough to land in the same 1.0-sized world cell.
        cloud = tuple((0.01 * i, 0.0, _DEPTH) for i in range(n_points))
        return cloud, transform

    directions = [(0.3, 0.1, -1.0), (-1.0, 0.1, -1.0)]  # two frames, wide baseline
    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = frame_with_many_points(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    assert report.total_cells == 1
    cell = report.cells[0]
    assert cell.observations == len(directions), (
        f"{n_points * len(directions)} points were offered from only "
        f"{len(directions)} frames; observations must report the latter"
    )
    assert cell.spread > _MIN_SPREAD, "the two frames were placed wide apart on purpose"
    assert not cell.well_covered, "two frames is below the threshold, however many points each gave"


def test_spread_is_the_best_pair_not_the_average_of_all_pairs():
    """Ten near-duplicate views plus one wide-baseline view must still count as spread.

    :func:`~scanner3d.coverage._max_pairwise_angle` is documented to use the
    widest pair, not the mean of every pair. The two measures disagree sharply
    here: the outlier's angle to the cluster is about 69 degrees (comfortably
    above the threshold), but the average over all 55 pairs - dragged down by
    the ten near-zero pairs inside the cluster - is about 13 degrees
    (comfortably below it). A mean-based implementation would call this thin;
    the maximum-based one this module actually uses must not.
    """
    rng = random.Random(99)
    base = normalize((0.05, 0.03, -1.0))
    outlier: Point = (0.9, 0.1, -0.3)
    directions = [outlier]
    for _ in range(10):
        jitter = math.radians(rng.uniform(-0.5, 0.5))
        axis = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
        directions.append(rotation_about_axis(axis, jitter).apply(base))

    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    cell = report.cells[0]
    assert cell.observations == len(directions)
    assert cell.spread > math.radians(60.0), "the widest pair should dominate, not the average"
    assert cell.well_covered


def test_default_min_spread_rejects_a_ten_degree_cluster():
    """The default threshold, actually exercised - not merely checked to be in range.

    Three views spanning exactly 10 degrees, with ``min_spread`` omitted so
    :data:`~scanner3d.coverage.DEFAULT_MIN_SPREAD` applies. The assertion does
    not reference that constant - it is a fixed, independently-known 10
    degrees compared against whatever the module's real default does with it -
    so a change to the default's value, not just its being in ``(0, pi)``,
    would be caught here.
    """
    base = normalize((0.0, 0.0, -1.0))
    tilted = rotation_about_axis((1.0, 0.0, 0.0), math.radians(10.0)).apply(base)
    directions = [base, base, tilted]
    assert len(directions) == _MIN_OBSERVATIONS
    clouds = []
    transforms = []
    for direction in directions:
        cloud, transform = _frame_observing(direction)
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS)
    cell = report.cells[0]
    assert cell.spread == pytest.approx(math.radians(10.0), abs=1e-6)
    assert not cell.well_covered, "a real capture-grade default should demand more than 10 degrees"


def test_thin_regions_cluster_face_adjacent_cells_report_the_worst_cell_and_sort_largest_first():
    """A direct, hand-built check of the clustering machinery itself.

    Three groups of thin cells at ``cell=1.0``, placed so their grid indices
    test three properties at once: a three-cell region and a two-cell region
    far enough apart to never touch, and a single cell placed diagonally - not
    face-adjacent - from the two-cell region's own cells, which must stay its
    own region even though it is one of that cell's 26 neighbours. Inside the
    two-cell region, one cell has a single observation and zero spread while
    the other has two observations ten degrees apart, so the region's
    ``worst_observations``/``worst_spread`` can be checked against the worse
    (lower) cell rather than the better one dragging the number up.
    """
    clouds: list[tuple[Point, ...]] = []
    transforms: list[Rigid] = []

    # Region A: three face-adjacent cells in a row, cell_index (0,0,0)..(2,0,0).
    for x in (0.5, 1.5, 2.5):
        cloud, transform = _frame_observing((0.05, 0.03, -1.0), target=(x, 0.5, 0.5))
        clouds.append(cloud)
        transforms.append(transform)

    # Region B: cell_index (10,0,0) - one observation, zero spread - and
    # (11,0,0) - two observations, ten degrees apart. Both counts are still
    # below min_observations, so both cells are thin regardless of spread.
    cloud, transform = _frame_observing((0.05, 0.03, -1.0), target=(10.5, 0.5, 0.5))
    clouds.append(cloud)
    transforms.append(transform)
    base = normalize((0.05, 0.03, -1.0))
    tilted = rotation_about_axis((1.0, 0.0, 0.0), math.radians(10.0)).apply(base)
    for direction in (base, tilted):
        cloud, transform = _frame_observing(direction, target=(11.5, 0.5, 0.5))
        clouds.append(cloud)
        transforms.append(transform)

    # Region C: cell_index (12,1,0) differs from region B's (11,0,0) in two
    # axes at once, so it is a 26-neighbour but not a 6-neighbour - it must
    # not merge with region B.
    cloud, transform = _frame_observing((0.05, 0.03, -1.0), target=(12.5, 1.5, 0.5))
    clouds.append(cloud)
    transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    regions = report.thin_regions()
    assert [r.cell_count for r in regions] == [3, 2, 1], (
        "expected one 3-cell region, one 2-cell region and one isolated cell, largest first"
    )
    region_a, region_b, region_c = regions
    # A region's centroid is a UI's pointer to where the gap physically is, and
    # nothing else in this file checks it against ground truth in all three
    # axes - the sweep tests only ever check the pole's own y-coordinate.
    # Region A's three cells each hold exactly one point at a known target, so
    # the region centroid has an exactly known answer: the mean of those three.
    assert region_a.centroid == pytest.approx((1.5, 0.5, 0.5), abs=1e-9)
    assert region_b.worst_observations == 1
    assert region_b.worst_spread == pytest.approx(0.0, abs=1e-9)
    assert region_c.cell_count == 1


def test_cell_assignment_uses_floor_not_truncation_for_negative_coordinates():
    """The grid arithmetic itself, at the one place floor and truncation disagree.

    ``floor`` and ``int()`` (truncation towards zero) agree for every positive
    coordinate, which is all every other test in this file happens to use - so
    none of them could tell the two apart. They disagree exactly at a negative,
    non-multiple-of-``cell`` coordinate: ``floor(-2.5) == -3`` while
    ``int(-2.5) == -2``. A truncation bug would silently shift every cell with
    a negative coordinate on any axis half a cell towards the origin.
    """
    point: Point = (-2.5, 0.5, 0.5)
    report = coverage_map(
        ((point,),), (Rigid.identity(),), cell=1.0, min_observations=1, min_spread=0.0
    )
    assert report.cells[0].cell_index == (-3, 0, 0)


def test_a_degenerate_zero_depth_point_is_excluded_not_a_crash():
    """A single bad sample must not take the whole map down with it.

    One frame's cloud carries an extra point sitting exactly at that frame's
    own camera centre - the zero-depth reading :func:`~scanner3d.coverage._accumulate`
    documents handling. It contributes no viewing direction and, by the same
    reasoning, no cell either - checked here by confirming the surviving cell's
    count is unaffected, not merely that the call didn't raise.
    """
    directions = [(0.3, 0.1, -1.0), (1.0, 0.2, -1.0), (-1.0, 0.1, -1.0), (0.1, 1.0, -1.0)]
    clouds = []
    transforms = []
    for index, direction in enumerate(directions):
        cloud, transform = _frame_observing(direction)
        if index == 0:
            cloud = (*cloud, (0.0, 0.0, 0.0))  # local origin: zero depth, no direction
        clouds.append(cloud)
        transforms.append(transform)

    report = coverage_map(
        clouds, transforms, cell=1.0, min_observations=_MIN_OBSERVATIONS, min_spread=_MIN_SPREAD
    )
    assert report.total_cells == 1, "the degenerate point must not open a cell of its own"
    cell = report.cells[0]
    assert cell.observations == len(directions), "the degenerate point must not inflate the count"
    assert cell.well_covered


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

_ONE_CLOUD: tuple[tuple[Point, ...]] = (((0.0, 0.0, 1.0),),)
_ONE_TRANSFORM: tuple[Rigid, ...] = (Rigid.identity(),)


def test_coverage_map_raises_for_mismatched_cloud_and_transform_counts():
    with pytest.raises(CloudTransformCountMismatch):
        coverage_map(_ONE_CLOUD, (Rigid.identity(), Rigid.identity()), cell=1.0)


def test_coverage_map_raises_for_no_clouds_offered():
    with pytest.raises(NoCloudsOffered):
        coverage_map((), (), cell=1.0)


def test_coverage_map_raises_when_every_cloud_is_empty():
    with pytest.raises(NothingToCover):
        coverage_map(((),), (Rigid.identity(),), cell=1.0)


@pytest.mark.parametrize("cell", [0.0, -1.0, math.nan, math.inf])
def test_coverage_map_raises_for_non_positive_cell(cell: float):
    with pytest.raises(CellSizeUnusable):
        coverage_map(_ONE_CLOUD, _ONE_TRANSFORM, cell=cell)


@pytest.mark.parametrize("min_observations", [0, -1])
def test_coverage_map_raises_for_non_positive_min_observations(min_observations: int):
    with pytest.raises(MinObservationsUnusable):
        coverage_map(_ONE_CLOUD, _ONE_TRANSFORM, cell=1.0, min_observations=min_observations)


@pytest.mark.parametrize("min_spread", [-0.1, math.pi + 0.1, math.nan, math.inf])
def test_coverage_map_raises_for_min_spread_out_of_range(min_spread: float):
    with pytest.raises(MinSpreadUnusable):
        coverage_map(_ONE_CLOUD, _ONE_TRANSFORM, cell=1.0, min_spread=min_spread)


def test_min_observations_of_exactly_one_is_accepted():
    """The guard's own boundary: 1 is the smallest *valid* value, not an invalid one.

    The parametrized rejection test above only proves 0 and -1 are refused; it
    cannot tell a correct ``< 1`` guard from an off-by-one ``< 2`` guard that
    wrongly refuses 1 too. This is the other half of that boundary.
    """
    coverage_map(_ONE_CLOUD, _ONE_TRANSFORM, cell=1.0, min_observations=1)


@pytest.mark.parametrize("min_spread", [0.0, math.pi])
def test_min_spread_at_its_own_boundary_values_is_accepted(min_spread: float):
    """0 and pi are the smallest and largest *valid* radian values, not invalid ones."""
    coverage_map(_ONE_CLOUD, _ONE_TRANSFORM, cell=1.0, min_spread=min_spread)


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_coverage_report_as_dict_round_trips_and_carries_schema_version():
    clouds, transforms = _sphere_sweep()
    report = coverage_map(
        clouds,
        transforms,
        cell=_MAP_CELL,
        min_observations=_MIN_OBSERVATIONS,
        min_spread=_MIN_SPREAD,
    )
    record = json.loads(json.dumps(report.as_dict()))
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["total_cells"] == report.total_cells
    assert record["well_covered_cells"] == report.well_covered_cells
    assert record["well_covered_fraction"] == pytest.approx(report.well_covered_fraction)
    assert record["cell_size"] == _MAP_CELL
    assert len(record["thin_regions"]) == len(report.thin_regions())
    assert record["limitations"]

    cell_record = report.cells[0].as_dict()
    assert cell_record["cell_index"] == list(report.cells[0].cell_index)
    round_tripped_cell = json.loads(json.dumps(cell_record))
    assert round_tripped_cell["observations"] == report.cells[0].observations


def test_default_min_spread_is_a_finite_positive_radian_value():
    assert 0.0 < DEFAULT_MIN_SPREAD < math.pi


def test_well_covered_fraction_of_an_empty_report_is_zero_not_a_division_error():
    """``coverage_map`` itself always refuses an input with zero occupied cells

    (see :class:`~scanner3d.coverage.NothingToCover`), so this constructs the
    dataclass directly - the only way to reach the empty case
    :meth:`~scanner3d.coverage.CoverageReport.well_covered_fraction` documents
    handling.
    """
    empty = CoverageReport(
        cells=(), cell_size=1.0, min_observations=3, min_spread=_MIN_SPREAD, limitations=()
    )
    assert empty.total_cells == 0
    assert empty.well_covered_fraction == 0.0
    assert empty.thin_cells() == ()
    assert empty.thin_regions() == ()
