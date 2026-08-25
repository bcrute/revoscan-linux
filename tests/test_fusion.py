"""Tests for voxel-grid fusion.

The algorithmic tests use small synthetic clouds whose expected fused result can
be written down by hand, so a failure names the arithmetic that broke rather
than a statistic that drifted. One real captured frame is fused as a structural
smoke test, and it asserts structure only: no test here claims a distance, an
extent or a dimension, because nothing in this repository has established one.
"""

from __future__ import annotations

import functools
import json
import math
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.fusion import (
    FUSION_MODE,
    INHERENT_LIMITATIONS,
    MAX_CELL_INDEX,
    MAX_GRID_CELLS,
    MAX_INPUT_POINTS,
    MIN_VOXEL_SIZE,
    SCHEMA_VERSION,
    CloudCountMismatch,
    FusedCloud,
    FusionNotAllowed,
    InputTooLarge,
    NothingToFuse,
    ObservationThresholdUnreachable,
    PointUnusable,
    VoxelGridTooLarge,
    VoxelSizeUnusable,
    ensure_grid_bounded,
    fuse,
    voxel_downsample,
)
from scanner3d.geometry import Point, Rigid, rotation_about_axis
from scanner3d.intrinsics import INTRINSICS_FORMAT, IntrinsicsRecord, parse_intrinsics
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

IDENTITY = Rigid.identity()

#: A cell edge every synthetic cloud below is designed around: its points sit at
#: cell centres, so a rotation's last-bit error never moves one across a
#: boundary and the tests measure the algorithm rather than the rounding.
CELL = 1.0


def cell_of(point: Point, voxel_size: float = CELL) -> tuple[int, int, int]:
    """The cell a point belongs to, computed independently of the module.

    Deliberately a second implementation. A test that asked ``fusion`` where a
    point went would agree with it about a wrong answer.
    """
    return (
        math.floor(point[0] / voxel_size),
        math.floor(point[1] / voxel_size),
        math.floor(point[2] / voxel_size),
    )


def cells_of(points: Sequence[Point], voxel_size: float = CELL) -> set[tuple[int, int, int]]:
    return {cell_of(point, voxel_size) for point in points}


def lattice(nx: int = 3, ny: int = 3, nz: int = 2) -> tuple[Point, ...]:
    """A block of points, one per cell, each at its cell's centre."""
    return tuple(
        (0.5 + i, 0.5 + j, 0.5 + k) for i in range(nx) for j in range(ny) for k in range(nz)
    )


def close(a: Point, b: Point, tolerance: float = 1e-9) -> bool:
    return all(abs(x - y) <= tolerance for x, y in zip(a, b, strict=True))


# --- the shape survives the merge ---------------------------------------------


def test_two_clouds_of_a_known_shape_fuse_back_to_that_shape():
    """A shape seen twice, from two poses, is one shape afterwards - not two.

    The second cloud is the shape expressed in a rotated frame, so the transform
    supplied for it is exactly the rotation that undoes that. Every cell is
    therefore reached by both clouds and the fused result is the original block.
    """
    shape = lattice()
    spin = rotation_about_axis((0.0, 0.0, 1.0), math.pi / 2)
    seen_from_the_side = tuple(spin.inverse().apply_all(shape))

    result = fuse(
        [shape, seen_from_the_side], [IDENTITY, spin], voxel_size=CELL, min_observations=2
    )

    assert result.output_points == len(shape)
    assert result.occupied_voxels == len(shape)
    assert result.dropped_voxels == 0
    expected = sorted(shape, key=cell_of)
    for produced, wanted in zip(result.points, expected, strict=True):
        assert close(produced, wanted), f"{produced} is not {wanted}"


def test_two_clouds_at_different_places_keep_both_shapes():
    """Fusion merges what overlaps and keeps what does not; it is not a filter."""
    shape = lattice(2, 2, 1)
    shifted = Rigid.of(IDENTITY.rotation, (10.0, 0.0, 0.0))

    result = fuse([shape, shape], [IDENTITY, shifted], voxel_size=CELL)

    assert result.output_points == 2 * len(shape)
    assert cells_of(result.points) == cells_of(shape) | cells_of(tuple(shifted.apply_all(shape)))


def test_a_cell_reached_by_three_clouds_collapses_to_one_averaged_point():
    """Three observations of one cell leave one point, at their mean.

    The coordinates are chosen to sum exactly in float64, so the assertion is on
    the arithmetic rather than on a tolerance.
    """
    clouds: list[Sequence[Point]] = [
        [(0.25, 0.5, 0.75)],
        [(0.5, 0.5, 0.5)],
        [(0.75, 0.5, 0.25)],
    ]

    result = fuse(clouds, [IDENTITY, IDENTITY, IDENTITY], voxel_size=CELL, min_observations=3)

    assert result.points == ((0.5, 0.5, 0.5),)
    assert result.input_points == 3
    assert result.occupied_voxels == 1
    assert result.merged_away == 2


def test_repeated_points_from_one_cloud_are_one_observation_not_many():
    """Five points from one frame in one cell is one view of it, not five.

    The alternative counting would let a single noisy frame satisfy any
    threshold by itself, which would make the threshold agree with itself
    instead of confirming anything.
    """
    dense: Sequence[Point] = [(0.5, 0.5, 0.5)] * 5
    other: Sequence[Point] = [(2.5, 2.5, 2.5)]

    permissive = fuse([dense, other], [IDENTITY, IDENTITY], voxel_size=CELL)
    strict = fuse([dense, other], [IDENTITY, IDENTITY], voxel_size=CELL, min_observations=2)

    assert permissive.output_points == 2
    assert strict.output_points == 0
    assert strict.occupied_voxels == 2
    assert strict.dropped_voxels == 2


def test_points_in_neighbouring_cells_are_not_merged():
    """The grid is the only thing that merges; adjacency is not enough."""
    left: Sequence[Point] = [(0.9, 0.5, 0.5)]
    right: Sequence[Point] = [(1.1, 0.5, 0.5)]

    result = fuse([left, right], [IDENTITY, IDENTITY], voxel_size=CELL)

    assert result.output_points == 2
    assert result.points == ((0.9, 0.5, 0.5), (1.1, 0.5, 0.5))


#: One pair of points, and what each cell edge does to it. The pair is fixed and
#: the edge is the only variable, so what changes between the rows is the grid
#: resolution and nothing else. Neither edge is 1.0, which is the point: at 1.0
#: a grid that multiplied by the cell edge instead of dividing by it would
#: produce identical output and no test above would notice.
PAIR: tuple[Point, ...] = ((0.25, 0.25, 0.25), (0.75, 0.25, 0.25))


@pytest.mark.parametrize(
    ("voxel_size", "expected"),
    [
        # Cells [0, 0.5) and [0.5, 1.0): the pair straddles the boundary.
        (0.5, ((0.25, 0.25, 0.25), (0.75, 0.25, 0.25))),
        # One cell [0, 2.0): the pair collapses to its exact mean.
        (2.0, ((0.5, 0.25, 0.25),)),
    ],
)
def test_the_cell_edge_is_what_decides_whether_two_points_merge(voxel_size, expected):
    """A coarser grid merges what a finer one keeps apart, and only the edge decides."""
    assert fuse([PAIR], [IDENTITY], voxel_size=voxel_size).points == expected


def test_cells_below_the_origin_are_the_cells_below_the_origin():
    """Negative coordinates floor away from zero, and are ordered before it.

    Truncation towards zero would fold the cell on each side of the origin into
    one, quietly merging a point in front of the origin with one behind it.
    """
    cloud: Sequence[Point] = [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

    result = fuse([cloud], [IDENTITY], voxel_size=CELL)

    assert [cell_of(point) for point in result.points] == [(-1, 0, 0), (0, 0, 0)]
    assert result.points == ((-0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


# --- the observation threshold, and what it costs -----------------------------

#: A surface both views can see.
SHARED: tuple[Point, ...] = ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5))

#: Real object surface only the first view can reach - an underside. It is
#: geometry, not noise, and the grid has no way to know that.
UNDERSIDE: Point = (0.5, 2.5, 0.5)

#: A speckle artefact only the second view produced. Noise, and the grid has no
#: way to know that either.
SPECKLE: Point = (9.5, 9.5, 9.5)


def two_views() -> tuple[list[Sequence[Point]], list[Rigid]]:
    return [[*SHARED, UNDERSIDE], [*SHARED, SPECKLE]], [IDENTITY, IDENTITY]


def test_min_observations_one_keeps_every_occupied_cell():
    clouds, transforms = two_views()

    result = fuse(clouds, transforms, voxel_size=CELL, min_observations=1)

    assert result.output_points == 5
    assert result.occupied_voxels == 5
    assert result.dropped_voxels == 0
    assert cell_of(SPECKLE) in cells_of(result.points)


def test_min_observations_two_drops_the_single_view_noise():
    clouds, transforms = two_views()

    result = fuse(clouds, transforms, voxel_size=CELL, min_observations=2)

    assert cell_of(SPECKLE) not in cells_of(result.points)
    assert result.occupied_voxels == 5
    assert result.output_points == 3


def test_min_observations_two_also_destroys_real_single_view_surface():
    """The trade-off, proven rather than claimed.

    ``UNDERSIDE`` is part of the object. It appears in one cloud only because
    the other view is occluded from it, which is exactly the situation a real
    turntable sweep is full of. The threshold that removed the speckle removes
    this too, and nothing in the grid distinguishes them: both are cells with
    one observation.
    """
    clouds, transforms = two_views()

    kept = fuse(clouds, transforms, voxel_size=CELL, min_observations=1)
    filtered = fuse(clouds, transforms, voxel_size=CELL, min_observations=2)

    assert cell_of(UNDERSIDE) in cells_of(kept.points)
    assert cell_of(UNDERSIDE) not in cells_of(filtered.points)
    # Two cells went, and only one of them held noise.
    assert filtered.dropped_voxels == 2


def test_the_threshold_note_admits_the_geometry_it_removed():
    clouds, transforms = two_views()

    result = fuse(clouds, transforms, voxel_size=CELL, min_observations=2)

    note = next(item for item in result.limitations if "min_observations was 2" in item)
    assert "2 of 5 occupied cells were discarded" in note
    assert "cannot separate noise from surface only one view could see" in note
    assert "real geometry was removed" in note


def test_a_permissive_threshold_says_it_confirmed_nothing():
    clouds, transforms = two_views()

    result = fuse(clouds, transforms, voxel_size=CELL, min_observations=1)

    assert any("no cross-view confirmation was required" in item for item in result.limitations)


def test_a_threshold_that_discarded_nothing_does_not_claim_it_removed_geometry():
    """A caveat that fires when it is false is worse than no caveat.

    Both clouds reach every cell here, so ``min_observations=2`` discards
    nothing. A note saying real geometry was removed would attach a loss to a
    result that suffered none, and a reader who checked one such note against
    ``dropped_voxels`` would stop trusting all of them.
    """
    shape = lattice()

    result = fuse([shape, shape], [IDENTITY, IDENTITY], voxel_size=CELL, min_observations=2)

    assert result.dropped_voxels == 0
    joined = " ".join(result.limitations)
    assert "real geometry was removed" not in joined
    assert "discarded nothing" in joined
    assert f"all {result.occupied_voxels} occupied cells" in joined


def test_a_threshold_that_discarded_nothing_still_admits_the_filter_is_not_free():
    """Discarding nothing on one input is not evidence the filter costs nothing."""
    shape = lattice()

    result = fuse([shape, shape], [IDENTITY, IDENTITY], voxel_size=CELL, min_observations=2)

    joined = " ".join(result.limitations)
    assert "not evidence that the filter is free" in joined
    assert "cannot tell them apart" in joined


# --- the accounting a consumer reads ------------------------------------------


def test_input_points_and_output_points_show_how_much_was_merged_away():
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)] * 4, [(0.5, 0.5, 0.5)] * 6]

    result = fuse(clouds, [IDENTITY, IDENTITY], voxel_size=CELL)

    assert result.input_points == 10
    assert result.output_points == 1
    assert result.merged_away == 9
    assert result.source_frames == 2


def test_occupied_voxels_counts_cells_before_the_threshold_not_after():
    """The two numbers together are what make the filter's cost visible."""
    clouds, transforms = two_views()

    result = fuse(clouds, transforms, voxel_size=CELL, min_observations=2)

    assert result.occupied_voxels == 5
    assert result.output_points == 3
    assert result.dropped_voxels == 2


def test_a_grid_that_merges_nothing_says_so():
    spread: Sequence[Point] = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5)]

    result = fuse([spread], [IDENTITY], voxel_size=CELL)

    assert result.output_points == result.input_points
    assert any("the grid merged nothing" in item for item in result.limitations)


def test_a_grid_that_did_merge_does_not_claim_it_merged_nothing():
    """The other half of the note, without which it could be attached always.

    Ten points collapse to one here. A result carrying "the grid merged nothing"
    would be telling a consumer that the averaging this module exists for did
    not happen, on a run where it happened nine times over.
    """
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)] * 4, [(0.5, 0.5, 0.5)] * 6]

    result = fuse(clouds, [IDENTITY, IDENTITY], voxel_size=CELL)

    assert result.merged_away == 9
    assert not any("the grid merged nothing" in item for item in result.limitations)


def test_a_single_cloud_says_nothing_was_cross_checked():
    result = fuse([lattice(2, 1, 1)], [IDENTITY], voxel_size=CELL)

    assert any("nothing was cross-checked between views" in item for item in result.limitations)


def test_several_clouds_do_not_claim_nothing_was_cross_checked():
    """A thirty-six frame sweep must not carry the single-capture caveat."""
    shape = lattice(2, 2, 1)

    result = fuse([shape, shape, shape], [IDENTITY] * 3, voxel_size=CELL)

    assert result.source_frames == 3
    joined = " ".join(result.limitations)
    assert "nothing was cross-checked between views" not in joined
    assert "this is a downsample of a single capture" not in joined


def test_an_empty_cloud_among_the_inputs_is_named():
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [], [(1.5, 0.5, 0.5)]]

    result = fuse(clouds, [IDENTITY, IDENTITY, IDENTITY], voxel_size=CELL)

    assert result.source_frames == 3
    assert any("clouds 1 contributed no points at all" in item for item in result.limitations)


# --- limitations are carried, never laundered ---------------------------------


def test_every_limitation_of_every_input_survives_the_fuse():
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [(1.5, 0.5, 0.5)]]

    result = fuse(
        clouds,
        [IDENTITY, IDENTITY],
        voxel_size=CELL,
        input_limitations=[("frame 0 was captured with the blue light off",), ("frame 1 drifted",)],
    )

    assert "frame 0 was captured with the blue light off" in result.limitations
    assert "frame 1 drifted" in result.limitations


def test_a_limitation_shared_by_every_cloud_is_carried_once():
    shared = "the depth scalar's physical unit is not established"
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [(1.5, 0.5, 0.5)]]

    result = fuse(
        clouds, [IDENTITY, IDENTITY], voxel_size=CELL, input_limitations=[(shared,), (shared,)]
    )

    assert result.limitations.count(shared) == 1


def test_the_inherent_limitations_are_on_every_result():
    result = fuse([[(0.5, 0.5, 0.5)]], [IDENTITY], voxel_size=CELL)

    for limitation in INHERENT_LIMITATIONS:
        assert limitation in result.limitations


def test_the_limitations_refuse_the_claim_that_averaging_fixes_a_systematic_error():
    """The one thing a fused cloud most invites a reader to believe."""
    result = fuse([[(0.5, 0.5, 0.5)]], [IDENTITY], voxel_size=CELL)

    joined = " ".join(result.limitations)
    assert "survives averaging completely" in joined
    assert "wrong principal point" in joined
    assert "no more metrically valid than the clouds it merged" in joined


def test_nothing_in_a_result_claims_a_physical_unit():
    result = fuse([lattice(2, 2, 1)], [IDENTITY], voxel_size=CELL)
    rendered = json.dumps(result.as_dict()).lower()

    for unit in (" mm", "millimet", " metre", " meter", " cm", "centimet", " inch"):
        assert unit not in rendered, f"a fused cloud must not mention {unit!r}"
    assert "not a length" in rendered


# --- determinism --------------------------------------------------------------


def test_the_same_input_fuses_to_a_byte_identical_result():
    shape = lattice()
    spin = rotation_about_axis((1.0, 0.0, 0.0), 0.3)
    clouds: list[Sequence[Point]] = [shape, tuple(spin.inverse().apply_all(shape))]

    first = fuse(clouds, [IDENTITY, spin], voxel_size=CELL)
    second = fuse(clouds, [IDENTITY, spin], voxel_size=CELL)

    assert repr(first.points) == repr(second.points)
    assert json.dumps(first.as_dict()) == json.dumps(second.as_dict())


def test_the_output_is_ordered_by_ascending_cell_index():
    """Ordering is a property of the geometry, so two fusions can be diffed."""
    scattered: Sequence[Point] = [
        (2.5, 0.5, 0.5),
        (0.5, 2.5, 0.5),
        (0.5, 0.5, 2.5),
        (0.5, 0.5, 0.5),
    ]

    result = fuse([scattered], [IDENTITY], voxel_size=CELL)
    keys = [cell_of(point) for point in result.points]

    assert keys == sorted(keys)
    assert keys[0] == (0, 0, 0)


def test_listing_the_clouds_in_another_order_occupies_the_same_cells():
    """The ordering does not depend on which cloud the caller happened to list first."""
    first: Sequence[Point] = [(2.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    second: Sequence[Point] = [(1.5, 0.5, 0.5)]

    forward = fuse([first, second], [IDENTITY, IDENTITY], voxel_size=CELL)
    backward = fuse([second, first], [IDENTITY, IDENTITY], voxel_size=CELL)

    assert forward.points == backward.points


# --- voxel_downsample ---------------------------------------------------------


def test_voxel_downsample_is_idempotent_at_the_same_voxel_size():
    """A cell's mean lies in that cell, so a second pass has nothing left to do."""
    cloud: Sequence[Point] = [
        (0.4, 0.5, 0.6),
        (0.6, 0.5, 0.4),
        (0.5, 0.5, 0.5),
        (1.5, 1.5, 1.5),
        (1.4, 1.6, 1.5),
    ]

    once = voxel_downsample(cloud, voxel_size=CELL)
    twice = voxel_downsample(once, voxel_size=CELL)

    assert once == twice
    assert len(once) == 2


def test_voxel_downsample_averages_within_each_cell():
    cloud: Sequence[Point] = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]

    assert voxel_downsample(cloud, voxel_size=CELL) == ((0.5, 0.5, 0.5),)


def test_voxel_downsample_agrees_with_fusing_one_cloud():
    """One grid in this module, not two: the matcher and the output share a lattice."""
    cloud = lattice(3, 2, 2)

    assert (
        voxel_downsample(cloud, voxel_size=0.75)
        == fuse([cloud], [IDENTITY], voxel_size=0.75).points
    )


def test_voxel_downsample_inherits_the_refusals():
    with pytest.raises(NothingToFuse):
        voxel_downsample([], voxel_size=CELL)
    with pytest.raises(VoxelSizeUnusable):
        voxel_downsample([(0.5, 0.5, 0.5)], voxel_size=0.0)


# --- refusals -----------------------------------------------------------------

ONE_POINT: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)]]


@pytest.mark.parametrize(
    "voxel_size", [0.0, -1.0, -0.0, float("nan"), float("inf"), float("-inf"), MIN_VOXEL_SIZE / 2]
)
def test_refuses_a_voxel_size_that_is_not_a_usable_cell_edge(voxel_size):
    with pytest.raises(VoxelSizeUnusable):
        fuse(ONE_POINT, [IDENTITY], voxel_size=voxel_size)


def test_the_smallest_allowed_voxel_size_is_accepted():
    """The floor is a boundary, not a fence a caller has to guess the far side of."""
    result = fuse(ONE_POINT, [IDENTITY], voxel_size=MIN_VOXEL_SIZE)

    assert result.output_points == 1


def test_refuses_a_voxel_size_that_implies_an_absurd_grid():
    cloud: Sequence[Point] = [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)]

    with pytest.raises(VoxelGridTooLarge, match="cells"):
        fuse([cloud], [IDENTITY], voxel_size=MIN_VOXEL_SIZE)


def test_refuses_a_tight_cloud_parked_absurdly_far_from_the_origin():
    """A small extent is not on its own proof that the indices are sane."""
    cloud: Sequence[Point] = [(1e9, 0.0, 0.0), (1e9 + 1.0, 0.0, 0.0)]

    with pytest.raises(VoxelGridTooLarge, match="cell index"):
        fuse([cloud], [IDENTITY], voxel_size=MIN_VOXEL_SIZE)


def test_the_dense_cell_bound_is_checked_against_the_extent():
    assert ensure_grid_bounded((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0) == 8
    with pytest.raises(VoxelGridTooLarge):
        ensure_grid_bounded((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0 / (MAX_GRID_CELLS // 2))


def test_the_dense_cell_count_counts_cells_not_widths():
    """An extent that straddles a boundary occupies both cells, however narrow.

    ``floor(width / voxel_size) + 1`` calls this one cell per axis - a bound
    reporting a grid eight times smaller than the one it is guarding. The count
    has to come from the cell indices the ends fall into, which is also the
    arithmetic the binning itself uses.
    """
    assert ensure_grid_bounded((0.9, 0.9, 0.9), (1.1, 1.1, 1.1), 1.0) == 8
    # One axis straddling, two inside a single cell.
    assert ensure_grid_bounded((0.9, 0.1, 0.1), (1.1, 0.2, 0.2), 1.0) == 2
    # A point cloud of one point is one cell, not two.
    assert ensure_grid_bounded((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1.0) == 1
    # Cells the binning agrees are occupied, counted independently of the module.
    corners: Sequence[Point] = [(0.9, 0.9, 0.9), (1.1, 1.1, 1.1)]
    assert ensure_grid_bounded((0.9, 0.9, 0.9), (1.1, 1.1, 1.1), 1.0) >= len(cells_of(corners))


def test_refuses_no_clouds_at_all():
    with pytest.raises(NothingToFuse, match="nothing to merge"):
        fuse([], [], voxel_size=CELL)


def test_refuses_clouds_that_are_all_empty():
    """Distinguished from a threshold that emptied the result, because it is not that."""
    with pytest.raises(NothingToFuse, match="every one of them is empty"):
        fuse([[], []], [IDENTITY, IDENTITY], voxel_size=CELL)


def test_refuses_a_transform_count_that_does_not_match_the_clouds():
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [(1.5, 0.5, 0.5)]]

    with pytest.raises(CloudCountMismatch, match="2 clouds against 1 transforms"):
        fuse(clouds, [IDENTITY], voxel_size=CELL)


def test_refuses_limitation_groups_that_are_not_one_per_cloud():
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [(1.5, 0.5, 0.5)]]

    with pytest.raises(CloudCountMismatch, match="limitation groups"):
        fuse(clouds, [IDENTITY, IDENTITY], voxel_size=CELL, input_limitations=[("only one",)])


@pytest.mark.parametrize("threshold", [0, -1, 2])
def test_refuses_an_observation_threshold_no_input_could_meet(threshold):
    with pytest.raises(ObservationThresholdUnreachable):
        fuse(ONE_POINT, [IDENTITY], voxel_size=CELL, min_observations=threshold)


def test_refuses_a_threshold_no_non_empty_cloud_could_meet():
    """An empty cloud is a frame, not an observation.

    Three clouds, two of them empty: only one cloud can ever reach a cell, so
    ``min_observations=3`` cannot be met by anything and the result is
    guaranteed to hold no points. Counting the threshold against the frame count
    would accept it and hand back exactly the silently emptied cloud this
    refusal exists to prevent.
    """
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [], []]

    with pytest.raises(ObservationThresholdUnreachable, match="1 clouds that hold any points"):
        fuse(clouds, [IDENTITY, IDENTITY, IDENTITY], voxel_size=CELL, min_observations=3)


def test_a_threshold_the_non_empty_clouds_can_meet_is_still_accepted():
    """The refusal above must not cost a caller a legitimate run."""
    clouds: list[Sequence[Point]] = [[(0.5, 0.5, 0.5)], [], [(0.5, 0.5, 0.5)]]

    result = fuse(clouds, [IDENTITY, IDENTITY, IDENTITY], voxel_size=CELL, min_observations=2)

    assert result.source_frames == 3
    assert result.output_points == 1


def test_refuses_more_points_than_the_ceiling_without_reading_one():
    """The bound is enforced from ``len``, so an oversized producer is not drained."""

    class LyingCloud:
        def __len__(self) -> int:
            return MAX_INPUT_POINTS + 1

        def __iter__(self):
            raise AssertionError("the ceiling must be enforced before any point is read")

    huge = cast(Sequence[Point], LyingCloud())

    with pytest.raises(InputTooLarge, match=str(MAX_INPUT_POINTS)):
        fuse([huge], [IDENTITY], voxel_size=CELL)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_refuses_a_non_finite_coordinate(bad):
    """NaN in particular: every comparison against it is false, so a bounding
    pass that trusted its comparisons would let it through silently."""
    cloud: Sequence[Point] = [(0.5, 0.5, 0.5), (bad, 0.5, 0.5)]

    with pytest.raises(PointUnusable, match="point 1"):
        fuse([cloud], [IDENTITY], voxel_size=CELL)


def test_refuses_an_entry_that_is_not_a_triple():
    cloud = cast(Sequence[Point], [(0.5, 0.5)])

    with pytest.raises(PointUnusable, match=r"not an \(x, y, z\) triple"):
        fuse([cloud], [IDENTITY], voxel_size=CELL)


def test_every_refusal_is_a_scanner_error_carrying_a_status():
    assert issubclass(FusionNotAllowed, ScannerError)
    assert FusionNotAllowed.status is DoctorStatus.UNSUPPORTED_MODE
    for error in (
        VoxelSizeUnusable,
        VoxelGridTooLarge,
        CloudCountMismatch,
        NothingToFuse,
        InputTooLarge,
        ObservationThresholdUnreachable,
        PointUnusable,
    ):
        assert issubclass(error, FusionNotAllowed)


def test_a_refused_fuse_produces_nothing_at_all():
    """There is no partial result to inspect; the refusal is the whole outcome."""
    with pytest.raises(FusionNotAllowed):
        fuse(ONE_POINT, [IDENTITY], voxel_size=-1.0)


# --- the serialised record ----------------------------------------------------


def test_as_dict_is_json_serialisable_and_carries_the_accounting():
    clouds, transforms = two_views()

    record = fuse(clouds, transforms, voxel_size=CELL, min_observations=2).as_dict()
    round_tripped = json.loads(json.dumps(record))

    assert round_tripped["schema_version"] == SCHEMA_VERSION
    assert round_tripped["mode"] == FUSION_MODE
    assert round_tripped["grid"]["min_observations"] == 2
    assert round_tripped["grid"]["voxel_size"] == CELL
    assert round_tripped["accounting"] == {
        "source_frames": 2,
        "input_points": 8,
        "occupied_voxels": 5,
        "output_points": 3,
        "dropped_voxels": 2,
        "merged_away": 5,
    }
    assert round_tripped["dimensional_accuracy_claimed"] is False
    assert round_tripped["bounds"]["max_cell_index"] == MAX_CELL_INDEX


def test_as_dict_previews_the_points_rather_than_dumping_them():
    result = fuse([lattice(4, 4, 4)], [IDENTITY], voxel_size=CELL)

    record = result.as_dict()
    points = cast(dict[str, object], record["points"])

    assert points["count"] == 64
    assert len(cast(list[object], points["preview"])) == 8


def test_the_record_is_a_frozen_value():
    result = fuse(ONE_POINT, [IDENTITY], voxel_size=CELL)

    assert isinstance(result, FusedCloud)
    with pytest.raises(AttributeError):
        result.voxel_size = 2.0  # type: ignore[misc]


# --- one real captured frame --------------------------------------------------

#: The committed captures. ``base_4`` is the richest of the nine: 89070 of its
#: 256000 samples returned a depth, which is the worst case this module has to
#: stay usable on.
FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"
REAL_FRAME = FIXTURE_FRAMES / "base_4.bin"

#: The factory numbers this unit actually returned, quoted in the test: no
#: module under test carries them, and injecting them here keeps that true.
REAL_FX = 898.6829833984375
REAL_FY = 898.6829833984375
REAL_CX = 42.43747329711914
REAL_CY = 192.8326873779297
REAL_SCALE = 0.004999999888241291


def real_record() -> IntrinsicsRecord:
    """The unit's own factory record, built through the real 40-byte parser."""
    blob = struct.pack(
        INTRINSICS_FORMAT,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        REAL_FX,
        0.0,
        REAL_CX,
        0.0,
        REAL_FY,
        REAL_CY,
        0.0,
        0.0,
        1.0,
    )
    parsed = parse_intrinsics(blob)
    assert parsed is not None
    return parsed


@functools.cache
def real_cloud() -> tuple[Point, ...]:
    """One real frame's points, or an explicit skip. A skip is not a pass."""
    if not REAL_FRAME.is_file() or REAL_FRAME.stat().st_size != EXPECTED_FRAME_BYTES:
        pytest.skip(
            f"{REAL_FRAME} is not a {EXPECTED_FRAME_BYTES}-byte capture, so fusion was NOT "
            "exercised against real device data on this run. This skip is not a pass"
        )
    conversion = deproject_depth_frame(
        REAL_FRAME.read_bytes(),
        intrinsics=real_record(),
        depth_scale=REAL_SCALE,
        unit=UNVALIDATED_UNIT,
    )
    return tuple(camera_points(conversion))


@functools.cache
def real_downsampled() -> tuple[Point, ...]:
    """The real cloud thinned once, reused: it is 89070 points of work."""
    return voxel_downsample(real_cloud(), voxel_size=1.0)


def test_a_real_frame_downsamples_to_far_fewer_points_and_stays_inside_its_own_bounds():
    """Structure only: fewer points, all finite, none outside where the frame put them.

    No distance, extent or dimension is asserted, because this repository has
    validated none of them.
    """
    cloud = real_cloud()
    thinned = real_downsampled()

    assert len(cloud) > 10_000
    assert 0 < len(thinned) < len(cloud)
    for axis in range(3):
        low = min(point[axis] for point in cloud)
        high = max(point[axis] for point in cloud)
        assert all(low <= point[axis] <= high for point in thinned)
    assert all(math.isfinite(value) for point in thinned for value in point)


def test_a_real_frame_downsample_is_idempotent():
    thinned = real_downsampled()

    assert voxel_downsample(thinned, voxel_size=1.0) == thinned


def test_two_registered_views_of_a_real_frame_confirm_each_other():
    """The property fusion exists for, on real geometry.

    The second cloud is the same capture expressed in a moved frame, so the
    transform given for it puts it back where it came from. Correctly
    registered, the two views should land in the same cells: the occupied count
    should barely grow, and almost every cell should hold two observations. This
    says nothing about whether the coordinates are right - only that merging two
    views of one surface produces one surface.
    """
    thinned = real_downsampled()
    move = rotation_about_axis((0.0, 0.0, 1.0), 0.05, through=(0.0, 0.0, 100.0))
    displaced = tuple(move.inverse().apply_all(thinned))

    merged = fuse([thinned, displaced], [IDENTITY, move], voxel_size=1.0)
    confirmed = fuse([thinned, displaced], [IDENTITY, move], voxel_size=1.0, min_observations=2)

    assert merged.input_points == 2 * len(thinned)
    assert merged.output_points == merged.occupied_voxels
    assert merged.output_points < merged.input_points
    # Not an equality: libm's sine differing by an ulp on some other machine may
    # carry one point across a cell boundary, and that is not a defect in this
    # module. A second view adding one percent new cells would be.
    assert merged.occupied_voxels <= len(thinned) * 1.01
    assert confirmed.output_points >= 0.99 * merged.output_points
