"""Tests for isolated-point rejection.

The algorithmic tests use clouds small enough to write the expected verdict down
by hand, so a failure names the rule that broke rather than a statistic that
drifted. Two tests use real captured frames: one asserts structure only, and one
measures what the filter does to strays planted at a KNOWN distance from real
surface - that one states a bound the module docstring's calibration table
already claims, so a regression in the default settings fails here rather than
being noticed on a scan.

No test in this file asserts a distance, an extent or a dimension. Nothing in
this repository has established one.
"""

from __future__ import annotations

import functools
import json
import math
import random
import struct
from collections.abc import Sequence
from pathlib import Path

import pytest

from scanner3d.cloud_export import camera_points
from scanner3d.denoise import (
    DEFAULT_MIN_NEIGHBOURS,
    DEFAULT_SPACING_MULTIPLE,
    FILTER_MODE,
    INHERENT_LIMITATIONS,
    MAX_REMOVED_FRACTION,
    MIN_NEIGHBOURHOOD,
    MIN_POINTS_TO_FILTER,
    SCHEMA_VERSION,
    DenoiseNotAllowed,
    FilteredCloud,
    InputTooLarge,
    MinNeighboursUnusable,
    NeighbourhoodGridTooLarge,
    NeighbourhoodUnusable,
    NothingToFilter,
    PointUnusable,
    SpacingIndeterminate,
    estimate_spacing,
    reject_isolated,
    strip_speckle,
)
from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.fusion import MAX_CELL_INDEX, MAX_INPUT_POINTS
from scanner3d.geometry import Point
from scanner3d.intrinsics import INTRINSICS_FORMAT, IntrinsicsRecord, parse_intrinsics
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: A cell edge every hand-written cloud below is designed around: points sit at
#: cell centres, so no arithmetic detail decides which cell one lands in and the
#: tests measure the rule rather than the rounding.
CELL = 1.0


def block(nx: int, ny: int, nz: int, *, origin: Point = (0.5, 0.5, 0.5)) -> list[Point]:
    """A solid nx*ny*nz lattice of points, one per cell, at cell centres."""
    return [
        (origin[0] + i * CELL, origin[1] + j * CELL, origin[2] + k * CELL)
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
    ]


def brute_force(points: Sequence[Point], cell: float, min_neighbours: int) -> set[Point]:
    """The same rule, spelled out independently of the module under test.

    Deliberately per POINT rather than per cell: the module memoises the count
    on the cell, and this is what proves that memoisation is exact rather than a
    cheaper answer to a different question.
    """
    kept: set[Point] = set()
    for point in points:
        own = tuple(math.floor(value / cell) for value in point)
        total = 0
        for other in points:
            key = tuple(math.floor(value / cell) for value in other)
            if all(abs(a - b) <= 1 for a, b in zip(own, key, strict=True)):
                total += 1
        if total - 1 >= min_neighbours:
            kept.add(point)
    return kept


# --- the rule -----------------------------------------------------------------


def test_a_point_alone_in_the_grid_is_removed_and_a_dense_block_is_not():
    """The whole purpose, on a cloud whose answer is obvious by eye."""
    cloud = [*block(4, 4, 4), (500.0, 500.0, 500.0)]

    result = reject_isolated(cloud, neighbourhood=CELL, min_neighbours=DEFAULT_MIN_NEIGHBOURS)

    assert result.removed == ((500.0, 500.0, 500.0),)
    assert set(result.points) == set(block(4, 4, 4))
    assert result.applied is True


def test_the_threshold_counts_other_points_and_not_the_point_itself():
    """A pair alone in adjacent cells has one neighbour each, so 1 keeps and 2 does not."""
    pair = [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5)]

    assert reject_isolated(pair, neighbourhood=CELL, min_neighbours=1).removed == ()
    assert reject_isolated(pair, neighbourhood=CELL, min_neighbours=2).points == ()


def test_zero_required_neighbours_keeps_everything_and_says_it_checked_nothing():
    """The documented no-op: available without branching around the call."""
    cloud = [(0.5, 0.5, 0.5), (900.0, 900.0, 900.0)]

    result = reject_isolated(cloud, neighbourhood=CELL, min_neighbours=0)

    assert result.removed == ()
    assert result.applied is True
    assert any("no-op by request" in note for note in result.limitations)


def test_the_neighbourhood_is_the_block_of_cells_around_a_point_not_a_ball():
    """Stated in the docstring, so it is asserted rather than left to be discovered.

    Two points three cell edges apart along one axis are outside each other's
    block; two points 1.7 edges apart on a diagonal are inside it, even though a
    ball of one edge centred on either would not reach the other.
    """
    far = [(0.5, 0.5, 0.5), (3.5, 0.5, 0.5)]
    diagonal = [(0.5, 0.5, 0.5), (1.5, 1.5, 0.5)]

    assert reject_isolated(far, neighbourhood=CELL, min_neighbours=1).points == ()
    assert reject_isolated(diagonal, neighbourhood=CELL, min_neighbours=1).removed == ()


def test_the_per_cell_count_agrees_with_a_per_point_search_on_a_random_cloud():
    """The memoisation is exact. A thousand points, checked point by point."""
    rng = random.Random(20260827)
    cloud = [
        (rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0), rng.uniform(-30.0, 30.0))
        for _ in range(1000)
    ]

    result = reject_isolated(cloud, neighbourhood=2.0, min_neighbours=4)

    assert set(result.points) == brute_force(cloud, 2.0, 4)
    assert set(result.points) | set(result.removed) == set(cloud)


def test_both_output_tuples_keep_the_input_order():
    """A caller lining `removed` up against its own per-point data depends on it."""
    cloud = [(500.0, 0.0, 0.0), *block(3, 3, 3), (-500.0, 0.0, 0.0)]

    result = reject_isolated(cloud, neighbourhood=CELL, min_neighbours=2)

    assert result.removed == ((500.0, 0.0, 0.0), (-500.0, 0.0, 0.0))
    assert list(result.points) == [point for point in cloud if point in set(result.points)]


def test_nothing_is_invented_and_nothing_is_lost():
    """Every input point comes back exactly once, in one of the two tuples."""
    rng = random.Random(5)
    cloud = [(rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(300)]

    result = reject_isolated(cloud, neighbourhood=1.0, min_neighbours=3)

    assert result.input_points == len(cloud)
    assert result.kept_points + result.removed_points == len(cloud)
    assert sorted([*result.points, *result.removed]) == sorted(cloud)


# --- the spacing estimate -----------------------------------------------------


def test_a_regular_lattice_estimates_a_spacing_near_its_own_pitch():
    """The one case where the true answer is known exactly: a cubic lattice.

    A factor, not a decimal. The module docstring claims 0.6 to 1.3 of the
    measured nearest-neighbour distance on real frames and this asserts a band
    wider than that, because a lattice is not a surface and the estimate's area
    assumption is least true here.
    """
    lattice = block(12, 12, 12)

    estimate = estimate_spacing(lattice)

    assert 0.4 * CELL <= estimate.spacing <= 2.0 * CELL
    assert estimate.input_points == len(lattice)
    assert estimate.span > 0.0


def test_the_spacing_estimate_tracks_the_density_it_is_given():
    """Twice the pitch, about twice the spacing. The property the call sites rely on."""
    fine = block(16, 16, 1)
    coarse = [(x * 4.0, y * 4.0, z) for x, y, z in fine]

    ratio = estimate_spacing(coarse).spacing / estimate_spacing(fine).spacing

    assert 3.0 <= ratio <= 5.0


def test_a_cloud_with_no_extent_has_no_spacing_and_says_so():
    with pytest.raises(SpacingIndeterminate):
        estimate_spacing([(1.0, 2.0, 3.0)] * 10)


def test_the_spacing_record_reports_the_grid_it_settled_on():
    estimate = estimate_spacing(block(10, 10, 10))
    record = json.loads(json.dumps(estimate.as_dict(), allow_nan=False))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["grid"]["occupied_cells"] > 0
    assert record["grid"]["passes"] >= 1
    assert record["input_points"] == 1000
    assert "surface" in record["assumes"]


# --- sizing the neighbourhood from the cloud ----------------------------------


def test_strip_speckle_removes_a_stray_at_two_very_different_densities():
    """One setting, two decimations. This is the reason the spacing is measured.

    The same object sampled at two pitches, each with one point planted far from
    the surface. An absolute neighbourhood cannot serve both; this does.
    """
    for pitch in (1.0, 6.0):
        sheet = [(i * pitch, j * pitch, 0.0) for i in range(20) for j in range(20)]
        stray = (10.0 * pitch, 10.0 * pitch, 40.0 * pitch)

        result = strip_speckle([*sheet, stray])

        assert result.applied is True
        assert stray in result.removed
        assert result.removed_fraction < 0.2


def test_strip_speckle_withdraws_on_a_cloud_too_small_to_judge():
    """Below the floor it returns everything, on a record that says it did not act."""
    tiny = block(3, 3, 3)
    assert len(tiny) < MIN_POINTS_TO_FILTER

    result = strip_speckle(tiny)

    assert result.applied is False
    assert result.removed == ()
    assert set(result.points) == set(tiny)
    assert any("withdrew" in note for note in result.limitations)


def test_strip_speckle_withdraws_rather_than_emptying_a_cloud_that_is_all_isolated():
    """A cloud too sparse for its own estimate is left alone, not deleted.

    The live view is the reason: an operator who sees the model vanish learns
    nothing, and a scattered cloud is a normal thing to be handed at the start
    of a sweep.
    """
    rng = random.Random(99)
    scattered = [
        (rng.uniform(-400.0, 400.0), rng.uniform(-400.0, 400.0), rng.uniform(-400.0, 400.0))
        for _ in range(200)
    ]

    result = strip_speckle(scattered, min_neighbours=25)

    assert result.applied is False
    assert len(result.points) == len(scattered)
    assert result.spacing is not None


def test_a_withdrawal_is_not_an_exception():
    """Stated in the docstring: this runs on the thread that owns the device."""
    assert strip_speckle(block(2, 2, 2)).applied is False


def test_strip_speckle_carries_the_spacing_it_measured_on_the_record():
    result = strip_speckle(block(10, 10, 10))

    assert result.spacing is not None
    assert result.neighbourhood == pytest.approx(result.spacing.spacing * DEFAULT_SPACING_MULTIPLE)


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"neighbourhood": 0.0}, NeighbourhoodUnusable),
        ({"neighbourhood": -1.0}, NeighbourhoodUnusable),
        ({"neighbourhood": float("nan")}, NeighbourhoodUnusable),
        ({"neighbourhood": float("inf")}, NeighbourhoodUnusable),
        ({"neighbourhood": MIN_NEIGHBOURHOOD / 2}, NeighbourhoodUnusable),
        ({"neighbourhood": 1.0, "min_neighbours": -1}, MinNeighboursUnusable),
        ({"neighbourhood": 1.0, "min_neighbours": 1.5}, MinNeighboursUnusable),
        ({"neighbourhood": 1.0, "min_neighbours": True}, MinNeighboursUnusable),
    ],
)
def test_unusable_parameters_are_refused(kwargs: dict[str, object], expected: type) -> None:
    with pytest.raises(expected):
        reject_isolated(block(4, 4, 4), **kwargs)  # type: ignore[arg-type]


def test_an_empty_cloud_is_refused_rather_than_answered():
    """An empty answer and an empty question are different events."""
    with pytest.raises(NothingToFilter):
        reject_isolated([], neighbourhood=1.0)
    with pytest.raises(NothingToFilter):
        strip_speckle([])
    with pytest.raises(NothingToFilter):
        estimate_spacing([])


def test_a_cloud_past_the_ceiling_is_refused_from_its_length_alone():
    class Enormous(Sequence[Point]):
        """Claims to be huge and would raise if a point were ever read."""

        def __len__(self) -> int:
            return MAX_INPUT_POINTS + 1

        def __getitem__(self, index: int | slice) -> Point:  # type: ignore[override]
            # pragma: no cover - the size bound is checked before any read
            raise AssertionError("the size bound must be checked before any point is read")

    with pytest.raises(InputTooLarge):
        reject_isolated(Enormous(), neighbourhood=1.0)


@pytest.mark.parametrize(
    "bad",
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("-inf")),
    ],
)
def test_a_non_finite_coordinate_is_refused(bad: Point) -> None:
    with pytest.raises(PointUnusable):
        reject_isolated([*block(4, 4, 4), bad], neighbourhood=1.0)


def test_a_point_that_is_not_a_triple_is_refused():
    with pytest.raises(PointUnusable):
        reject_isolated([*block(4, 4, 4), (1.0, 2.0)], neighbourhood=1.0)  # type: ignore[list-item]


def test_a_cloud_parked_absurdly_far_from_the_origin_is_refused_not_overflowed():
    far = float(MAX_CELL_INDEX) * 100.0
    with pytest.raises(NeighbourhoodGridTooLarge):
        reject_isolated([(far, far, far)] * 2, neighbourhood=1.0, min_neighbours=0)


def test_an_out_of_range_removal_bound_is_refused():
    for bound in (0.0, -0.1, 1.5):
        with pytest.raises(DenoiseNotAllowed):
            strip_speckle(block(6, 6, 6), max_removed_fraction=bound)


def test_a_non_positive_multiple_is_refused():
    for multiple in (0.0, -1.0, float("nan")):
        with pytest.raises(NeighbourhoodUnusable):
            strip_speckle(block(6, 6, 6), multiple=multiple)


def test_every_refusal_is_a_scanner_error_with_a_status():
    assert issubclass(DenoiseNotAllowed, ScannerError)
    assert DenoiseNotAllowed.status is DoctorStatus.UNSUPPORTED_MODE


# --- the record ---------------------------------------------------------------


def test_the_record_serialises_and_accounts_for_every_point():
    cloud = [*block(5, 5, 5), (400.0, 0.0, 0.0)]
    result = strip_speckle(cloud)
    record = json.loads(json.dumps(result.as_dict(), allow_nan=False))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["mode"] == FILTER_MODE
    assert record["applied"] is True
    accounting = record["accounting"]
    assert accounting["input_points"] == len(cloud)
    assert accounting["kept_points"] + accounting["removed_points"] == len(cloud)
    assert accounting["removed_fraction"] == pytest.approx(
        accounting["removed_points"] / accounting["input_points"]
    )
    assert record["dimensional_accuracy_claimed"] is False


def test_the_record_previews_the_removed_points_without_becoming_them():
    rng = random.Random(1)
    strays = [
        (rng.uniform(200.0, 900.0), rng.uniform(200.0, 900.0), rng.uniform(200.0, 900.0))
        for _ in range(40)
    ]
    result = reject_isolated([*block(8, 8, 8), *strays], neighbourhood=CELL, min_neighbours=3)
    record = result.as_dict()

    removed = record["removed"]
    assert isinstance(removed, dict)
    assert removed["count"] == result.removed_points
    preview = removed["preview"]
    assert isinstance(preview, list)
    assert len(preview) < result.removed_points


def test_no_caller_can_shorten_the_inherent_limitations():
    result = reject_isolated(block(5, 5, 5), neighbourhood=CELL, input_limitations=("mine",))

    for inherent in INHERENT_LIMITATIONS:
        assert inherent in result.limitations
    assert result.limitations[-1] == "mine"


def test_the_inputs_limitations_are_carried_and_deduplicated():
    shared = INHERENT_LIMITATIONS[0]
    result = reject_isolated(
        block(5, 5, 5), neighbourhood=CELL, input_limitations=(shared, "unique")
    )

    assert result.limitations.count(shared) == 1
    assert "unique" in result.limitations


def test_a_run_that_removed_nothing_says_so_without_claiming_the_cloud_is_clean():
    result = reject_isolated(block(6, 6, 6), neighbourhood=CELL, min_neighbours=3)

    assert result.removed_points == 0
    note = next(note for note in result.limitations if "removed nothing" in note)
    assert "not evidence that it holds no stray points" in note


def test_a_run_that_removed_points_admits_some_of_them_may_have_been_surface():
    result = reject_isolated(
        [*block(6, 6, 6), (700.0, 0.0, 0.0)], neighbourhood=CELL, min_neighbours=3
    )

    assert result.removed_points == 1
    assert any("real surface too sparsely sampled" in note for note in result.limitations)


def test_the_record_is_a_frozen_value():
    result = strip_speckle(block(6, 6, 6))
    assert isinstance(result, FilteredCloud)
    with pytest.raises(AttributeError):
        result.applied = False  # type: ignore[misc]


# --- real captured frames -----------------------------------------------------

#: The committed captures. ``base_4`` is the richest of the nine: 89070 of its
#: 256000 samples returned a depth.
FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"
REAL_FRAME = FIXTURE_FRAMES / "base_4.bin"

#: The factory numbers this unit actually returned, quoted in the test: no
#: module under test carries them, and injecting them here keeps that true.
REAL_FX = 898.6829833984375
REAL_FY = 898.6829833984375
REAL_CX = 42.43747329711914
REAL_CY = 192.8326873779297
REAL_SCALE = 0.004999999888241291

#: What the two production call sites thin a frame to before filtering it.
LIVE_TARGET = 900
RECONSTRUCT_TARGET = 2200


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
            f"{REAL_FRAME} is not a {EXPECTED_FRAME_BYTES}-byte capture, so this filter was NOT "
            "exercised against real device data on this run. This skip is not a pass"
        )
    conversion = deproject_depth_frame(
        REAL_FRAME.read_bytes(),
        intrinsics=real_record(),
        depth_scale=REAL_SCALE,
        unit=UNVALIDATED_UNIT,
    )
    return tuple(camera_points(conversion))


def thinned(target: int) -> tuple[Point, ...]:
    """The real cloud decimated exactly the way the production loaders decimate."""
    cloud = real_cloud()
    step = max(1, len(cloud) // target)
    return cloud[::step]


@pytest.mark.parametrize("target", [LIVE_TARGET, RECONSTRUCT_TARGET])
def test_a_real_frame_keeps_almost_all_of_itself(target: int) -> None:
    """Structure only: most of a real frame survives, and nothing is invented.

    The bound is the point. A filter that took a tenth of every frame would be
    removing surface, and it would do it quietly.
    """
    cloud = thinned(target)

    result = strip_speckle(cloud)

    assert result.applied is True
    assert result.removed_fraction < 0.10
    assert set(result.points) <= set(cloud)
    for axis in range(3):
        low = min(point[axis] for point in cloud)
        high = max(point[axis] for point in cloud)
        assert all(low <= point[axis] <= high for point in result.points)


@pytest.mark.parametrize("target", [LIVE_TARGET, RECONSTRUCT_TARGET])
def test_strays_planted_far_from_a_real_surface_are_removed(target: int) -> None:
    """The claim the default settings exist for, on real geometry.

    Every stray is planted at least four estimated spacings from the nearest
    real point, which is the far column of the module docstring's table. The
    module claims 95% and better there; this asserts 90%, so ordinary
    floating-point differences between machines cannot fail it but a regression
    in the defaults can.
    """
    cloud = thinned(target)
    spacing = estimate_spacing(cloud).spacing
    rng = random.Random(4)

    lows = [min(point[axis] for point in cloud) for axis in range(3)]
    highs = [max(point[axis] for point in cloud) for axis in range(3)]
    occupied = {tuple(math.floor(value / spacing) for value in point): None for point in cloud}
    strays: list[Point] = []
    while len(strays) < 200:
        candidate = tuple(rng.uniform(lows[axis], highs[axis]) for axis in range(3))
        key = tuple(math.floor(value / spacing) for value in candidate)
        # Four spacings away in every direction: no real point is within four
        # cells of it, so this is a stray by construction and not by luck.
        if any(
            (key[0] + dx, key[1] + dy, key[2] + dz) in occupied
            for dx in range(-4, 5)
            for dy in range(-4, 5)
            for dz in range(-4, 5)
        ):
            continue
        strays.append((candidate[0], candidate[1], candidate[2]))

    result = strip_speckle([*cloud, *strays])

    assert result.applied is True
    removed = set(result.removed)
    caught = sum(1 for stray in strays if stray in removed)
    assert caught >= 0.90 * len(strays)


def test_the_default_settings_are_the_ones_the_call_sites_use():
    """A guard on the numbers the docstring's measured table describes."""
    assert DEFAULT_MIN_NEIGHBOURS == 3
    assert DEFAULT_SPACING_MULTIPLE == 1.0
    assert MAX_REMOVED_FRACTION == 0.5
