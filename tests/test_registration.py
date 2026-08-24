"""Prove the turntable registration recovers a spindle it was never told, and
that its loop closure notices when it has not.

Three kinds of evidence live here and they are not interchangeable.

**Ground truth.** The synthetic sweeps are built by rotating one asymmetric
shape about a line this file chose, so the answer is known exactly and every
recovery assertion is against that line rather than against the module's own
opinion of it. A residual the module reports is never used to decide whether the
module was right.

**A failure that is really a failure.** Loop closure is only worth anything if it
is large when the registration is wrong, so one sweep here is registered badly on
purpose - a single bootstrap pass over a partly self-occluding object, which is
exactly the case that settles in a local minimum - and the ground truth confirms
the axis really is wrong before the closure figure is allowed to claim it.

**Realistic shape.** One test drives a real 512000-byte capture from this unit
through the whole path. It is downsampled hard and turned synthetically, so it
proves the algorithm survives the point distribution a real depth frame has -
sparse, patchy, long-tailed - and it proves nothing whatever about a real
turntable, because no turntable capture exists.

Nothing in this file asserts a distance in a physical unit, because no distance
this repository produces has been compared against a physical measurement.
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import struct
from pathlib import Path

import pytest

from scanner3d.cloud_export import camera_points
from scanner3d.depth_metric import UNVALIDATED_UNIT, deproject_depth_frame
from scanner3d.errors import DoctorStatus
from scanner3d.geometry import (
    Point,
    Rigid,
    centroid,
    dot,
    normalize,
    rotation_about_axis,
    scaled,
    subtract,
)
from scanner3d.intrinsics import INTRINSICS_FORMAT, parse_intrinsics
from scanner3d.registration import (
    DEFAULT_CORRESPONDENCE_FRACTION,
    DEFAULT_KEEP_FRACTION,
    DEFAULT_VOXEL_FRACTION,
    DOWNSAMPLE_THRESHOLD,
    INHERENT_LIMITATIONS,
    MAX_CLOUDS,
    MAX_GRID_CELLS,
    MAX_ICP_ITERATIONS,
    MIN_CLOUD_POINTS,
    MIN_CORRESPONDENCES,
    MIN_RECOVERABLE_ANGLE,
    SCHEMA_VERSION,
    AngleStepUnusable,
    CloudUnusable,
    CorrespondencesInsufficient,
    IcpParametersUnusable,
    LoopClosure,
    RegistrationNotAllowed,
    TurntableAxis,
    TurntableGeometryUnusable,
    _VoxelIndex,  # private: the 27-cell bound is a correctness claim, so it is tested directly
    angle_between,
    cloud_extent,
    ensure_angle_step,
    estimate_turntable_axis,
    icp,
    kept_count,
    optimal_angle_about_axis,
    register_turntable,
    screw_axis_of,
    voxel_downsample,
)

# --------------------------------------------------------------------------
# The ground truth every synthetic assertion is measured against
# --------------------------------------------------------------------------

#: The spindle the test data really turns about. Off the camera origin in all
#: three axes and not parallel to any of them, so an implementation that quietly
#: assumed the origin, or a vertical axis, cannot pass by accident.
TRUE_DIRECTION: Point = normalize((0.05, 1.0, 0.12))
TRUE_THROUGH: Point = (0.4, 0.0, 3.0)

#: The same line, anchored where :class:`TurntableAxis` anchors one - at its
#: closest approach to the origin - so two descriptions of it are comparable.
TRUE_ANCHOR: Point = subtract(
    TRUE_THROUGH, scaled(TRUE_DIRECTION, dot(TRUE_THROUGH, TRUE_DIRECTION))
)


def turn(angle: float) -> Rigid:
    """The transform the real turntable applies over ``angle`` radians."""
    return rotation_about_axis(TRUE_DIRECTION, angle, through=TRUE_THROUGH)


def lumpy_curve(count: int = 240) -> tuple[Point, ...]:
    """An asymmetric open curve: no rotational symmetry, no mirror plane.

    Symmetry is the one thing that would let a wrong pose fit as well as the
    right one, so the shape is built with a three-lobed radius and a drifting
    height to make sure no rotation of it resembles itself.
    """
    points: list[Point] = []
    for index in range(count):
        t = 6.0 * math.pi * index / count
        radius = 0.6 + 0.35 * math.sin(3.1 * t)
        height = -0.8 + 1.6 * index / count
        points.append(
            (
                0.4 + radius * math.cos(t),
                height + 0.2 * math.sin(t * 0.7),
                3.0 + radius * math.sin(t) * 0.8,
            )
        )
    return tuple(points)


CURVE = lumpy_curve()


def exact_sweep(
    frames: int, step: float, shape: tuple[Point, ...] = CURVE
) -> list[tuple[Point, ...]]:
    """Every frame an exact rigid copy of one shape: the isolated-algorithm case.

    Deliberately *not* realistic. Nothing is occluded and nothing is noisy, so a
    fit that finds the global optimum recovers the truth to the last bit and any
    error an assertion sees is the module's own.
    """
    return [tuple(turn(k * step).apply_all(shape)) for k in range(frames)]


# --- a shape that occludes itself, which is the realistic case ----------------


def lumpy_shell() -> tuple[tuple[Point, Point], ...]:
    """A closed asymmetric surface, as ``(point, outward normal)`` pairs."""
    rings, per_ring = 16, 24
    built: list[tuple[Point, Point]] = []
    for i in range(1, rings):
        phi = math.pi * i / rings
        for j in range(per_ring):
            theta = math.tau * j / per_ring
            radius = (
                0.8
                + 0.22 * math.sin(3 * theta + 1.1)
                + 0.13 * math.cos(2 * phi)
                + 0.09 * math.sin(5 * theta + 2 * phi)
            )
            normal = normalize(
                (math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta))
            )
            built.append(
                (
                    (
                        0.4 + radius * normal[0],
                        radius * normal[1] * 1.3,
                        3.0 + radius * normal[2] * 0.9,
                    ),
                    normal,
                )
            )
    return tuple(built)


SHELL = lumpy_shell()


def visible_sweep(frames: int, step: float) -> list[tuple[Point, ...]]:
    """A sweep of a solid object: each frame holds only the surface facing the camera.

    This is what a depth camera actually returns, and it is what makes turntable
    registration hard: every step turns some surface out of view and some into
    it, so a large minority of the points in one frame have no true partner in
    the next.
    """
    sweep: list[tuple[Point, ...]] = []
    for k in range(frames):
        moved = turn(k * step)
        rotation = moved.rotation
        frame: list[Point] = []
        for point, normal in SHELL:
            facing = sum(rotation[2][j] * normal[j] for j in range(3))
            if facing > -0.05:
                continue
            frame.append(moved.apply(point))
        sweep.append(tuple(frame))
    return sweep


NINE_STEP = math.tau / 9


@functools.cache
def occluded_sweep() -> tuple[tuple[Point, ...], ...]:
    return tuple(visible_sweep(9, NINE_STEP))


@functools.cache
def bootstrap_axis(passes: int) -> TurntableAxis:
    """The spindle the occluded sweep yields after ``passes`` bootstrap sweeps."""
    return estimate_turntable_axis(occluded_sweep(), angle_step=NINE_STEP, passes=passes)


# --------------------------------------------------------------------------
# The screw axis of one transform
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "through", "angle"),
    [
        ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.4),
        ((0.05, 1.0, 0.12), (0.4, 0.0, 3.0), 0.6981317007977318),
        ((-0.3, 0.2, 1.0), (-2.0, 5.0, 1.5), 1.4),
        ((1.0, 1.0, 1.0), (7.0, -3.0, 0.25), 0.02),
    ],
)
def test_screw_axis_recovers_the_line_a_known_rotation_turns_about(direction, through, angle):
    """A rotation built about a line is decomposed back into that same line."""
    unit = normalize(direction)
    anchor = subtract(through, scaled(unit, dot(through, unit)))

    screw = screw_axis_of(rotation_about_axis(direction, angle, through=through))

    assert angle_between(screw.direction, unit) < 1e-12
    assert math.dist(screw.through, anchor) < 1e-9
    assert screw.angle == pytest.approx(angle, abs=1e-12)
    assert abs(screw.axial_slip) < 1e-12


def test_the_recovered_anchor_is_the_point_of_the_line_closest_to_the_origin():
    """Any point on the line describes it; only one makes two estimates comparable."""
    screw = screw_axis_of(rotation_about_axis((0.05, 1.0, 0.12), 0.5, through=(0.4, 0.0, 3.0)))

    # Closest approach to the origin means the anchor is perpendicular to the
    # direction; that is the whole content of the choice.
    assert abs(dot(screw.through, screw.direction)) < 1e-12


def test_screw_axis_reports_the_slide_a_turntable_cannot_produce():
    """A screw is a rotation plus a slide along the axis, and the slide is measured.

    A real turntable step has none. Reporting the component is what lets a
    caller see that a fit bought its residual by sliding, which is the tell for
    an alignment that has gone wrong in a way the rotation alone hides.
    """
    unit = normalize((0.05, 1.0, 0.12))
    spin = rotation_about_axis(unit, 0.5, through=(0.4, 0.0, 3.0))
    slide = scaled(unit, 0.75)
    screwed = Rigid.of(
        spin.rotation,
        (
            spin.translation[0] + slide[0],
            spin.translation[1] + slide[1],
            spin.translation[2] + slide[2],
        ),
    )

    screw = screw_axis_of(screwed)

    assert screw.axial_slip == pytest.approx(0.75, abs=1e-12)
    assert angle_between(screw.direction, unit) < 1e-12


def test_screw_axis_refuses_a_rotation_too_near_the_identity():
    """Below the floor the anchor is amplified noise, so no anchor is offered."""
    barely = rotation_about_axis((0.0, 1.0, 0.0), MIN_RECOVERABLE_ANGLE / 10.0, through=(1.0, 0, 2))

    with pytest.raises(TurntableGeometryUnusable, match="no recoverable axis"):
        screw_axis_of(barely)

    with pytest.raises(TurntableGeometryUnusable):
        screw_axis_of(Rigid.identity())


def test_screw_axis_refuses_a_rotation_too_near_a_half_turn():
    """At a half turn the skew part vanishes and the direction is simply gone."""
    with pytest.raises(TurntableGeometryUnusable, match="no recoverable axis"):
        screw_axis_of(rotation_about_axis((0.0, 1.0, 0.0), math.pi - 1e-9, through=(1.0, 0, 2)))


def test_a_refused_screw_axis_is_a_scanner_error_carrying_a_doctor_status():
    """Every refusal in this module reaches the diagnostic layer as one class."""
    assert issubclass(TurntableGeometryUnusable, RegistrationNotAllowed)
    assert RegistrationNotAllowed.status is DoctorStatus.UNSUPPORTED_MODE


# --------------------------------------------------------------------------
# The one degree of freedom, solved in closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.05, 0.7, 1.2, 3.0, -0.4, -2.5])
def test_optimal_angle_about_a_fixed_axis_recovers_a_known_rotation(angle):
    """The closed form is exact on exact correspondences, at any sign or size."""
    source = CURVE[:60]
    target = tuple(turn(angle).apply_all(source))

    recovered = optimal_angle_about_axis(
        source, target, direction=TRUE_DIRECTION, through=TRUE_THROUGH
    )

    assert recovered == pytest.approx(angle, abs=1e-12)


def test_the_optimal_angle_is_signed_rather_than_folded_to_a_magnitude():
    """A turntable running backwards is information, not an error to absorb."""
    source = CURVE[:60]
    target = tuple(turn(-0.6).apply_all(source))

    assert (
        optimal_angle_about_axis(source, target, direction=TRUE_DIRECTION, through=TRUE_THROUGH)
        < 0.0
    )


def test_the_optimal_angle_is_measured_from_the_line_not_from_the_origin():
    """The anchor matters: solving about the wrong point gives the wrong angle.

    Stated as a test because it is the single easiest thing to get wrong in a
    turntable implementation, and getting it wrong still returns a plausible
    number.
    """
    source = CURVE[:60]
    target = tuple(turn(0.7).apply_all(source))

    about_the_origin = optimal_angle_about_axis(
        source, target, direction=TRUE_DIRECTION, through=(0.0, 0.0, 0.0)
    )

    assert abs(about_the_origin - 0.7) > 0.05


def test_the_optimal_angle_refuses_points_that_all_lie_on_the_axis():
    """Rotation moves nothing there, so no angle is determined by them."""
    on_axis = tuple(
        (
            TRUE_THROUGH[0] + TRUE_DIRECTION[0] * t,
            TRUE_THROUGH[1] + TRUE_DIRECTION[1] * t,
            TRUE_THROUGH[2] + TRUE_DIRECTION[2] * t,
        )
        for t in (-1.0, 0.0, 1.0, 2.0)
    )

    with pytest.raises(TurntableGeometryUnusable, match="undetermined"):
        optimal_angle_about_axis(on_axis, on_axis, direction=TRUE_DIRECTION, through=TRUE_THROUGH)


def test_the_optimal_angle_refuses_an_unpaired_correspondence_set():
    with pytest.raises(CorrespondencesInsufficient):
        optimal_angle_about_axis(
            CURVE[:5], CURVE[:4], direction=TRUE_DIRECTION, through=TRUE_THROUGH
        )
    with pytest.raises(CorrespondencesInsufficient):
        optimal_angle_about_axis((), (), direction=TRUE_DIRECTION, through=TRUE_THROUGH)


def test_the_angle_between_two_directions_resolves_below_the_acos_noise_floor():
    """This is the metric that says how little two spindle estimates disagree.

    ``acos`` of a dot product cannot resolve below about 1.5e-8 rad, so an
    implementation using it would report a direction exact to the last bit as
    disagreeing by a hundredth of a microradian - and would stop measuring
    exactly where the answer is good.
    """
    a = normalize((0.05, 1.0, 0.12))
    b = normalize((0.05 + 1e-13, 1.0, 0.12))

    separation = angle_between(a, b)

    assert 0.0 < separation < 1e-9
    assert angle_between(a, a) == 0.0
    assert angle_between(a, scaled(a, -1.0)) == pytest.approx(math.pi)


# --------------------------------------------------------------------------
# Grids: downsampling, and a spatial index that is not a linear scan
# --------------------------------------------------------------------------


def test_voxel_downsample_replaces_each_occupied_cell_with_its_centroid():
    """Two points in one cell become their mean; a lone point is left alone."""
    points = [(0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (5.5, 0.0, 0.0)]

    reduced = voxel_downsample(points, 1.0)

    assert reduced == ((0.2, 0.2, 0.2), (5.5, 0.0, 0.0))


def test_voxel_downsample_is_deterministic_in_content_and_in_order():
    """Two exports of one cloud must be the same cloud, in the same sequence."""
    assert voxel_downsample(CURVE, 0.15) == voxel_downsample(CURVE, 0.15)


def test_voxel_downsample_refuses_a_grid_that_spans_more_cells_than_the_ceiling():
    """A cell so fine that it reduces nothing is refused, not attempted."""
    extent = cloud_extent(CURVE)

    with pytest.raises(IcpParametersUnusable, match=str(MAX_GRID_CELLS)):
        voxel_downsample(CURVE, extent / 100_000.0)


def test_the_voxel_index_agrees_with_an_exhaustive_scan_on_every_query():
    """The 27-cell bound is a claim about correctness, so it is checked as one.

    The index searches only the block of cells around a query and never the
    whole cloud - that is the difference between this module being usable at
    89000 points and not. The soundness argument is that a point inside the gate
    cannot be outside that block; this test puts the argument against a brute
    force scan on a few hundred queries rather than trusting it.
    """
    rng = random.Random(20240824)
    target = CURVE
    gate = 0.25
    index = _VoxelIndex(target, gate=gate)

    for _ in range(400):
        query = (rng.uniform(-1.0, 2.0), rng.uniform(-1.5, 1.5), rng.uniform(2.0, 4.0))
        best = min(target, key=lambda p: math.dist(p, query))
        expected = best if math.dist(best, query) < gate else None

        found = index.nearest(query)

        if expected is None:
            assert found is None
        else:
            assert found is not None
            assert math.dist(found, query) == pytest.approx(math.dist(expected, query))


def test_the_voxel_index_returns_nothing_for_a_query_with_no_partner_in_reach():
    index = _VoxelIndex(CURVE, gate=0.25)

    assert index.nearest((100.0, 100.0, 100.0)) is None


def test_the_trim_never_thins_a_correspondence_set_below_what_a_solve_needs():
    """Trimming may thin a set; it may never be the thing that empties it."""
    assert kept_count(1000, 0.6) == 600
    assert kept_count(1000, 1.0) == 1000
    assert kept_count(4, 0.01) == MIN_CORRESPONDENCES
    assert kept_count(MIN_CORRESPONDENCES, 0.5) == MIN_CORRESPONDENCES


def test_cloud_extent_refuses_a_cloud_that_is_all_one_point():
    """No fraction of zero is a usable bound, so no bound is derived from one."""
    with pytest.raises(CloudUnusable, match="no extent"):
        cloud_extent([(1.0, 2.0, 3.0)] * 10)


# --------------------------------------------------------------------------
# ICP
# --------------------------------------------------------------------------


def test_icp_recovers_a_small_rigid_offset():
    """The ordinary case: a modest displacement, found and reported with its residual."""
    truth = rotation_about_axis((0.2, 1.0, 0.1), 0.09, through=(0.4, 0.0, 3.0))
    target = tuple(truth.apply_all(CURVE))

    result = icp(CURVE, target)

    difference = result.transform.compose(truth.inverse())
    assert difference.rotation_angle < 1e-9
    assert difference.translation_magnitude < 1e-9
    assert result.rmse < 1e-9
    assert result.converged is True
    assert result.iterations >= 2


def test_icp_reports_converged_false_when_it_runs_out_of_iterations():
    """Finishing is not converging, and the record says which one happened.

    One iteration has nothing to compare its residual against, so it cannot
    have settled - whatever transform it produced.
    """
    target = tuple(rotation_about_axis((0.2, 1.0, 0.1), 0.09, through=(0.4, 0, 3)).apply_all(CURVE))

    stopped = icp(CURVE, target, max_iterations=1)

    assert stopped.converged is False
    assert stopped.iterations == 1
    assert icp(CURVE, target, max_iterations=MAX_ICP_ITERATIONS).converged is True


def test_icp_converges_no_faster_than_the_tolerance_it_was_given():
    """A tolerance of zero demands an exactly repeated residual, and is honoured."""
    target = tuple(rotation_about_axis((0.2, 1.0, 0.1), 0.05, through=(0.4, 0, 3)).apply_all(CURVE))

    loose = icp(CURVE, target, tolerance=1e-2)
    tight = icp(CURVE, target, tolerance=0.0)

    assert loose.iterations <= tight.iterations
    assert loose.convergence_threshold > tight.convergence_threshold == 0.0


def test_icp_counts_pairs_with_no_partner_apart_from_pairs_it_trimmed():
    """Two ways of not being used, two counts: they say different things."""
    stray = tuple((50.0 + i, 50.0, 50.0) for i in range(20))
    source = (*CURVE, *stray)
    target = tuple(rotation_about_axis((0.2, 1.0, 0.1), 0.05, through=(0.4, 0, 3)).apply_all(CURVE))

    result = icp(source, target)

    assert result.rejected == len(stray)
    assert result.trimmed > 0
    assert result.correspondences + result.trimmed + result.rejected == result.offered
    assert result.offered == result.source_points == len(source)
    assert result.matched_fraction == pytest.approx(len(CURVE) / len(source))
    assert result.used_fraction < result.matched_fraction


def test_icp_solves_on_every_matched_pair_when_trimming_is_turned_off():
    target = tuple(rotation_about_axis((0.2, 1.0, 0.1), 0.05, through=(0.4, 0, 3)).apply_all(CURVE))

    untrimmed = icp(CURVE, target, keep_fraction=1.0)

    assert untrimmed.trimmed == 0
    assert untrimmed.correspondences == len(CURVE)


def test_icp_records_the_bounds_it_resolved_from_the_target_extent():
    """Every relative default is resolved against the data and written down.

    Nothing here takes a parameter in a physical unit, so the only way a reader
    can tell what a residual was measured against is for the resolved numbers to
    be in the record.
    """
    target = tuple(turn(0.05).apply_all(CURVE))
    extent = cloud_extent(target)

    result = icp(CURVE, target, tolerance=1e-5)

    assert result.max_correspondence_distance == pytest.approx(
        extent * DEFAULT_CORRESPONDENCE_FRACTION
    )
    assert result.convergence_threshold == pytest.approx(extent * 1e-5)
    assert result.keep_fraction == DEFAULT_KEEP_FRACTION
    assert result.voxel_size is None, "a cloud this small is matched as it stands"
    assert result.trim_distance <= result.max_correspondence_distance


def test_icp_downsamples_a_cloud_past_the_threshold_and_says_what_it_used():
    """Downsampling is a decision about the data, so it is reported, not hidden."""
    dense = tuple(
        (math.cos(i * 0.017) * 2.0, i * 0.0006, 3.0 + math.sin(i * 0.017) * 2.0)
        for i in range(DOWNSAMPLE_THRESHOLD + 500)
    )
    target = tuple(turn(0.03).apply_all(dense))

    result = icp(dense, target)

    assert result.voxel_size == pytest.approx(cloud_extent(target) * DEFAULT_VOXEL_FRACTION)
    assert result.source_points < len(dense)


def test_icp_refuses_clouds_that_cannot_take_part_in_a_match():
    good = CURVE
    with pytest.raises(CloudUnusable, match="at least"):
        icp((), good)
    with pytest.raises(CloudUnusable, match=str(MIN_CLOUD_POINTS)):
        icp(good[: MIN_CLOUD_POINTS - 1], good)
    with pytest.raises(CloudUnusable, match="NaN and infinities"):
        icp([(0.0, 0.0, math.nan)] * 5, good)
    with pytest.raises(CloudUnusable, match="NaN and infinities"):
        icp([(0.0, 0.0, math.inf)] * 5, good)
    with pytest.raises(CloudUnusable, match="3 an"):
        icp([(0.0, 0.0)] * 5, good)
    with pytest.raises(CloudUnusable, match="no extent"):
        icp(good, [(1.0, 1.0, 1.0)] * 5)


def test_icp_refuses_parameters_outside_their_ranges():
    good = CURVE
    target = tuple(turn(0.05).apply_all(good))
    for kwargs in (
        {"max_iterations": 0},
        {"max_iterations": MAX_ICP_ITERATIONS + 1},
        {"tolerance": -1e-9},
        {"tolerance": math.nan},
        {"voxel_size": 0.0},
        {"voxel_size": -1.0},
        {"voxel_size": math.inf},
        {"max_correspondence_distance": 0.0},
        {"max_correspondence_distance": math.nan},
        {"keep_fraction": 0.0},
        {"keep_fraction": 1.5},
        {"keep_fraction": math.nan},
    ):
        with pytest.raises(IcpParametersUnusable):
            icp(good, target, **kwargs)


def test_icp_refuses_to_fit_two_clouds_that_do_not_overlap():
    """Rather than fit a transform to a handful of points and report a small residual."""
    far = tuple((p[0] + 500.0, p[1], p[2]) for p in CURVE)

    with pytest.raises(CorrespondencesInsufficient, match="do not overlap"):
        icp(CURVE, far)


def test_icp_is_serialisable_and_says_what_its_residual_is_not():
    result = icp(CURVE, tuple(turn(0.05).apply_all(CURVE)))

    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["converged"] is True
    assert "not established" in record["rmse_unit"]


# --------------------------------------------------------------------------
# Estimating the spindle
# --------------------------------------------------------------------------


def test_a_known_off_origin_spindle_is_recovered_from_the_clouds_alone():
    """Nothing is told the axis. It comes out of the screw axes of the fits."""
    axis = estimate_turntable_axis(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    assert angle_between(axis.direction, TRUE_DIRECTION) < 1e-10
    assert math.dist(axis.through, TRUE_ANCHOR) < 1e-9
    assert axis.estimated is True
    assert axis.pair_count == 8
    assert axis.converged is True


def test_a_recovered_spindle_reports_residuals_that_are_small_when_it_is_right():
    """A transform without its residual is an assertion; with it, a measurement."""
    axis = estimate_turntable_axis(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    assert axis.direction_spread_radians is not None and axis.direction_spread_radians < 1e-10
    assert axis.angle_residual_radians is not None and axis.angle_residual_radians < 1e-10
    assert axis.through_spread is not None and axis.through_spread < 1e-9
    assert axis.axial_slip is not None and abs(axis.axial_slip) < 1e-9
    assert axis.fit_rmse is not None and axis.fit_rmse < 1e-9


def test_the_measured_step_is_what_the_data_turned_not_what_the_caller_declared():
    """The one check that catches a mis-stated turntable step.

    The clouds really turn by 35 degrees; the caller says 40. The fit is not
    constrained by the claim, so the estimate reports what happened.
    """
    truly = math.radians(35.0)
    sweep = exact_sweep(9, truly)

    axis = estimate_turntable_axis(sweep, angle_step=math.radians(40.0))

    measured = axis.angle_step_radians
    assert measured is not None
    assert measured == pytest.approx(truly, abs=1e-9)
    assert abs(measured - math.radians(40.0)) > math.radians(4.0)


def test_a_single_pair_is_flagged_because_its_spreads_compared_nothing():
    """Zero because nothing was compared reads exactly like zero because all agreed."""
    axis = estimate_turntable_axis(exact_sweep(2, NINE_STEP), angle_step=NINE_STEP)

    assert axis.pair_count == 1
    assert axis.direction_spread_radians is not None
    assert axis.direction_spread_radians < 1e-12
    assert any("against itself" in note for note in axis.limitations)


def test_a_declared_spindle_carries_no_residual_and_none_is_invented_for_it():
    """``None`` is "not measured", which is not the same statement as zero."""
    declared = TurntableAxis.declared(
        TRUE_DIRECTION, TRUE_THROUGH, basis="the test states this line"
    )

    assert declared.estimated is False
    assert declared.pair_count == 0
    assert declared.angle_step_radians is None
    assert declared.angle_residual_radians is None
    assert declared.direction_spread_radians is None
    assert declared.through_spread is None
    assert declared.axial_slip is None
    assert declared.fit_rmse is None
    assert declared.pair_iterations == ()
    assert declared.converged is None
    assert any("declared by the caller" in note for note in declared.limitations)
    # Still canonicalised, so it compares against an estimate of the same line.
    assert math.dist(declared.through, TRUE_ANCHOR) < 1e-12


def test_a_second_bootstrap_pass_rescues_a_fit_that_settled_in_the_wrong_place():
    """The reason ``passes`` exists, measured against the truth rather than asserted.

    One pass has to start from the identity, and on a coarse sweep of an object
    that occludes itself that is a poor enough guess to settle in a local
    minimum. The second pass starts from the line the first one found and lands
    on the answer.
    """
    one = bootstrap_axis(1)
    two = bootstrap_axis(2)

    assert angle_between(one.direction, TRUE_DIRECTION) > 0.01
    assert angle_between(two.direction, TRUE_DIRECTION) < 1e-9
    assert math.dist(two.through, TRUE_ANCHOR) < 1e-8
    assert one.passes == 1
    assert two.passes == 2


def test_an_estimated_spindle_is_always_anchored_at_its_closest_approach_to_the_origin():
    """Two estimates of one line must be comparable, and only one anchor makes them so.

    A line has infinitely many points on it, so the anchor is a *choice*, and the
    choice only helps if it is made the same way every time. The bad bootstrap is
    the case that shows it: its per-pair axes genuinely disagree, so their mean
    anchor lies off the consensus line until it is projected back onto it.
    """
    for axis in (
        estimate_turntable_axis(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP),
        bootstrap_axis(1),
        bootstrap_axis(2),
    ):
        assert abs(dot(axis.through, axis.direction)) < 1e-12
        assert math.dist(axis.direction, (0.0, 0.0, 0.0)) == pytest.approx(1.0)


def test_seeding_each_pair_from_the_one_before_costs_far_fewer_iterations():
    """Chaining buys cost, not accuracy, so it is cost that is measured here.

    Every turntable step is the same transform, so once one pair is solved the
    next starts a fraction of a step from its answer and settles in the fewest
    iterations the loop can report. Unseeded, each pair starts from the identity
    and pays the full search again.
    """
    axis = estimate_turntable_axis(occluded_sweep(), angle_step=NINE_STEP, passes=1)

    assert len(axis.pair_iterations) == axis.pair_count == 8
    assert axis.pair_iterations[0] >= 8, "the first pair has nothing to start from"
    assert max(axis.pair_iterations[2:]) <= 4, "the rest start next door to the answer"
    assert sum(axis.pair_iterations) < 4 * axis.pair_iterations[0]


def test_trimming_is_what_stops_a_partial_overlap_under_rotating_every_step():
    """The justification for the default keep fraction, measured rather than asserted.

    A point on the edge of the overlap has no true partner - the object turned
    it into view, or out of it - so its nearest neighbour is a false pair. On a
    turntable those pull the same way every time, which is why an untrimmed fit
    does not merely get noisier: it gets consistently, directionally wrong.
    """
    untrimmed = estimate_turntable_axis(occluded_sweep(), angle_step=NINE_STEP, keep_fraction=1.0)
    trimmed = bootstrap_axis(2)

    short = untrimmed.angle_step_radians
    assert short is not None
    assert short < NINE_STEP * 0.9, "the bias is a systematic under-rotation, not noise"
    assert angle_between(untrimmed.direction, TRUE_DIRECTION) > 100 * angle_between(
        trimmed.direction, TRUE_DIRECTION
    )


def test_the_residuals_show_the_bad_bootstrap_to_be_bad_without_the_ground_truth():
    """A consumer with no ground truth must still be able to tell these apart."""
    one, two = bootstrap_axis(1), bootstrap_axis(2)

    assert one.direction_spread_radians is not None
    assert two.direction_spread_radians is not None
    assert one.direction_spread_radians > 0.1
    assert two.direction_spread_radians < 1e-9
    assert one.fit_rmse is not None and two.fit_rmse is not None
    assert one.fit_rmse > two.fit_rmse * 1e6


def test_estimating_a_spindle_refuses_a_step_no_axis_can_be_recovered_from():
    sweep = exact_sweep(4, 0.5)
    for step in (0.0, -0.5, math.nan, math.inf, MIN_RECOVERABLE_ANGLE / 2.0, 2.0):
        with pytest.raises(AngleStepUnusable):
            estimate_turntable_axis(sweep, angle_step=step)
    assert ensure_angle_step(0.5) == 0.5


def test_estimating_a_spindle_refuses_a_sweep_that_is_not_one():
    with pytest.raises(CloudUnusable, match="not a sweep"):
        estimate_turntable_axis([CURVE], angle_step=0.5)
    with pytest.raises(CloudUnusable, match=str(MAX_CLOUDS)):
        estimate_turntable_axis([CURVE] * (MAX_CLOUDS + 1), angle_step=0.5)
    with pytest.raises(IcpParametersUnusable, match="passes"):
        estimate_turntable_axis(exact_sweep(4, 0.5), angle_step=0.5, passes=0)


def test_a_spindle_record_is_serialisable_and_names_no_unit_for_its_lengths():
    axis = estimate_turntable_axis(exact_sweep(4, 0.4), angle_step=0.4)

    record = json.loads(json.dumps(axis.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["units"]["angles"] == "radians"
    assert "not established" in record["units"]["lengths"]
    assert record["through_is"].startswith("the point of the spindle closest")


# --------------------------------------------------------------------------
# Registering a sweep, and the closure that checks it
# --------------------------------------------------------------------------


def test_a_closed_revolution_returns_its_first_frame_to_itself():
    """The headline. Nothing in the fitting enforces this; that is what makes it a check."""
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    closure = result.loop_closure
    assert closure is not None
    assert closure.frames == 9
    assert closure.rotation_error_radians < 1e-10
    assert closure.translation_error < 1e-9
    assert "pose_difference" in closure.note


def test_every_frame_lands_where_the_ground_truth_puts_it():
    """Closure is a self-consistency check, so pose accuracy is asserted separately.

    A chain can close perfectly and still be wrong everywhere - if every step
    were a tenth too large the product would still be a whole revolution. Only
    the ground truth settles this.
    """
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    for pose in result.poses:
        expected = turn(-pose.index * NINE_STEP)
        difference = pose.transform.compose(expected.inverse())
        assert difference.rotation_angle < 1e-10
        assert difference.translation_magnitude < 1e-9


def test_the_reference_frame_carries_the_identity_and_no_residual_at_all():
    """A residual of zero would claim an alignment that was never performed."""
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    first = result.poses[0]
    assert first.index == 0
    assert first.transform == Rigid.identity()
    assert first.rmse is None
    assert first.correspondences == 0


def test_a_frame_residual_is_the_step_it_took_not_the_error_it_accumulated():
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    for pose in result.poses[1:]:
        step = result.pairwise[pose.index - 1]
        assert pose.rmse == step.rmse
        assert pose.correspondences == step.correspondences
    assert "not against the reference" in str(result.poses[1].as_dict()["rmse_is"])


def test_the_pair_that_closes_the_ring_is_fitted_like_every_other_pair():
    """Special treatment for the closing pair would make the check circular."""
    closed = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)
    open_sweep = register_turntable(exact_sweep(7, NINE_STEP), angle_step=NINE_STEP)

    assert len(closed.pairwise) == 9 == len(closed.poses)
    assert len(open_sweep.pairwise) == 6 == len(open_sweep.poses) - 1


def test_a_sweep_that_does_not_close_reports_no_closure_and_says_why():
    """A fabricated number for an open sweep is the worst thing this could produce."""
    result = register_turntable(exact_sweep(7, NINE_STEP), angle_step=NINE_STEP)

    assert result.loop_closure is None
    assert any("no loop closure was measured" in note for note in result.limitations)
    assert json.loads(json.dumps(result.as_dict()))["loop_closure"] is None


def test_two_revolutions_close_as_honestly_as_one():
    """The last frame still has to come back to the first, so the multiple is reported."""
    result = register_turntable(exact_sweep(18, NINE_STEP), angle_step=NINE_STEP)

    assert result.loop_closure is not None
    assert result.loop_closure.rotation_error_radians < 1e-9
    assert "2 revolution(s)" in result.loop_closure.note


def test_a_broken_registration_is_caught_by_the_loop_closure():
    """The metric is worthless unless it is large when the registration is wrong.

    Same clouds, same code, one bootstrap pass instead of two. The ground truth
    confirms the one-pass spindle really is wrong before the closure figure is
    allowed to be evidence of it.
    """
    broken = register_turntable(occluded_sweep(), angle_step=NINE_STEP, axis=bootstrap_axis(1))
    sound = register_turntable(occluded_sweep(), angle_step=NINE_STEP, axis=bootstrap_axis(2))

    assert angle_between(bootstrap_axis(1).direction, TRUE_DIRECTION) > 0.01
    assert angle_between(bootstrap_axis(2).direction, TRUE_DIRECTION) < 1e-9

    assert broken.loop_closure is not None and sound.loop_closure is not None
    assert broken.loop_closure.rotation_error_radians > 0.1
    assert sound.loop_closure.rotation_error_radians < 1e-9
    assert broken.loop_closure.translation_error > sound.loop_closure.translation_error * 1e9


def test_the_closure_cannot_see_a_displacement_two_good_fits_absorbed():
    """A limitation, pinned down so nobody has to rediscover it by trusting the number.

    One frame is turned off the spindle. The fits on either side of it are still
    exact - they find the transform that really relates those clouds - so the
    displacement telescopes out of the chain and the loop closes at the
    floating-point floor, while that frame's pose is wrong by the whole
    displacement. Closure checks that the fits held together. It does not check
    that the poses are the object's.
    """
    sweep = list(exact_sweep(9, NINE_STEP))
    nudged = rotation_about_axis((1.0, 0.0, 0.0), 0.15, through=TRUE_THROUGH)
    sweep[4] = tuple(nudged.apply_all(sweep[4]))
    truth = TurntableAxis.declared(TRUE_DIRECTION, TRUE_THROUGH, basis="the test states it")

    result = register_turntable(sweep, angle_step=NINE_STEP, axis=truth, constrain_to_axis=False)

    assert result.worst_rmse < 1e-12, "every fit found its global optimum"
    displaced = result.poses[4].transform.compose(turn(-4 * NINE_STEP).inverse())
    assert displaced.rotation_angle > 0.1, "frame 4's pose really is wrong"

    closure = result.loop_closure
    assert closure is not None
    assert closure.rotation_error_radians < 1e-12, "and the closure sees none of it"
    assert any("not a loop that is right" in note for note in result.limitations)


def test_a_wrong_spindle_shows_up_in_the_closure_of_a_constrained_sweep():
    """Because the angles fitted about a wrong line stop summing to a revolution."""
    wrong = TurntableAxis.declared((1.0, 0.0, 0.0), (0.0, 0.0, 3.0), basis="deliberately wrong")

    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP, axis=wrong)

    assert result.loop_closure is not None
    assert result.loop_closure.rotation_error_radians > 0.5


def test_a_constrained_closure_translation_is_the_rotation_restated_not_a_second_measurement():
    """The note says the two halves are not independent here; this is that claim, checked.

    With every step held to one line, the residual is a rotation about that line,
    so its translation is fixed by the rotation residual and the line's offset
    from the camera origin. Reporting it is right - a consumer of the poses will
    see it - but a reader must not read it as a second, independent error.
    """
    wrong = TurntableAxis.declared(
        (0.3, 1.0, 0.0), (0.9, 0.0, 3.0), basis="wrong enough to leave a residual"
    )

    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP, axis=wrong)

    closure = result.loop_closure
    assert closure is not None
    assert closure.rotation_error_radians > 1e-6, "a zero residual would make this vacuous"
    predicted = (
        2.0
        * math.dist(result.axis.through, (0.0, 0.0, 0.0))
        * math.sin(closure.rotation_error_radians / 2.0)
    )
    assert closure.translation_error == pytest.approx(predicted, rel=1e-9)
    assert "not an independent measurement" in closure.note


def test_six_degree_of_freedom_steps_are_offered_and_say_their_residuals_are_independent():
    result = register_turntable(
        exact_sweep(9, NINE_STEP), angle_step=NINE_STEP, constrain_to_axis=False
    )

    closure = result.loop_closure
    assert closure is not None
    assert closure.rotation_error_radians < 1e-10
    assert closure.translation_error < 1e-9
    assert "six degrees of freedom" in closure.note
    assert "independent" in closure.note
    assert all(step is None or True for step in result.step_angles_radians)


def test_a_supplied_spindle_is_used_as_given_rather_than_re_estimated():
    declared = TurntableAxis.declared(TRUE_DIRECTION, TRUE_THROUGH, basis="the test states it")

    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP, axis=declared)

    assert result.axis is declared
    assert result.loop_closure is not None
    assert result.loop_closure.rotation_error_radians < 1e-10


def test_the_refined_step_angles_are_reported_and_sum_to_the_revolution():
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    assert len(result.step_angles_radians) == 9
    for angle in result.step_angles_radians:
        assert angle == pytest.approx(NINE_STEP, abs=1e-9)
    assert math.fsum(result.step_angles_radians) == pytest.approx(math.tau, abs=1e-9)


def test_registration_refuses_what_it_cannot_register():
    sweep = exact_sweep(9, NINE_STEP)
    with pytest.raises(AngleStepUnusable):
        register_turntable(sweep, angle_step=0.0)
    with pytest.raises(AngleStepUnusable):
        register_turntable(sweep, angle_step=2.0)
    with pytest.raises(AngleStepUnusable, match="closure_tolerance"):
        register_turntable(sweep, angle_step=NINE_STEP, closure_tolerance=-1.0)
    with pytest.raises(CloudUnusable):
        register_turntable([CURVE], angle_step=NINE_STEP)
    with pytest.raises(IcpParametersUnusable):
        register_turntable(sweep, angle_step=NINE_STEP, max_iterations=0)
    with pytest.raises(CloudUnusable, match="NaN and infinities"):
        register_turntable([*sweep[:-1], ((0.0, 0.0, math.nan),) * 5], angle_step=NINE_STEP)


def test_closure_tolerance_decides_what_counts_as_a_revolution():
    """The arithmetic is the caller's to tighten, and the decision is recorded."""
    nearly = math.tau / 9 * 1.001

    loose = register_turntable(exact_sweep(9, nearly), angle_step=nearly, closure_tolerance=0.05)
    strict = register_turntable(exact_sweep(9, nearly), angle_step=nearly, closure_tolerance=1e-6)

    assert loose.loop_closure is not None
    assert strict.loop_closure is None


# --------------------------------------------------------------------------
# What travels with the result
# --------------------------------------------------------------------------


def test_the_inherent_limitations_cannot_be_dropped_by_a_caller():
    result = register_turntable(
        exact_sweep(9, NINE_STEP), angle_step=NINE_STEP, limitations=("only this one",)
    )

    for inherent in INHERENT_LIMITATIONS:
        assert inherent in result.limitations
    assert "only this one" in result.limitations
    assert len(set(result.limitations)) == len(result.limitations), "no caveat is said twice"


def test_the_limitations_say_the_scan_is_shaped_not_sized():
    """The module's central honesty claim, stated where a consumer will read it."""
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    joined = " ".join(result.limitations)
    assert "correctly shaped and of unknown size" in joined
    assert "No residual, translation or distance in this result is a length" in joined
    assert "does not say the object was reconstructed correctly" in joined


def test_a_spindle_that_was_estimated_carries_its_own_caveats_into_the_result():
    result = register_turntable(occluded_sweep(), angle_step=NINE_STEP, axis=bootstrap_axis(1))

    for note in result.axis.limitations:
        assert note in result.limitations


def test_the_whole_result_serialises_and_claims_no_dimensional_accuracy():
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)

    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["dimensional_accuracy_claimed"] is False
    assert record["frame_count"] == 9
    assert len(record["poses"]) == 9
    assert len(record["pairwise"]) == 9
    assert record["converged"] is True
    assert "not established" in record["loop_closure"]["translation_error_unit"]
    assert record["axis"]["estimated"] is True


def test_no_record_this_module_emits_asserts_a_physical_unit():
    """Every length here is a bare number, and every record says so.

    A unit word may appear only inside a denial - "not millimetres" is the point
    of the sentence. What must never appear is a record that quietly labels a
    number with a length.
    """
    result = register_turntable(exact_sweep(9, NINE_STEP), angle_step=NINE_STEP)
    text = json.dumps(result.as_dict())

    for key, value in _walk(result.as_dict()):
        if key.endswith("_unit") or key == "lengths":
            assert "not established" in str(value), f"{key} labels a number with a unit"
    for claim in ("millimetre", "millimeter", "centimetre", "centimeter"):
        for sentence in _sentences_containing(text, claim):
            assert " not " in sentence or "no " in sentence, sentence


def _walk(node: object, key: str = "") -> list[tuple[str, object]]:
    if isinstance(node, dict):
        found: list[tuple[str, object]] = []
        for child_key, child in node.items():
            found.extend(_walk(child, str(child_key)))
        return found
    if isinstance(node, list):
        found = []
        for child in node:
            found.extend(_walk(child, key))
        return found
    return [(key, node)]


def _sentences_containing(text: str, needle: str) -> list[str]:
    return [part for part in text.split(";") if needle in part]


def test_a_loop_closure_record_stands_on_its_own():
    closure = LoopClosure(
        rotation_error_radians=0.25, translation_error=1.5, frames=9, note="built by the test"
    )

    record = json.loads(json.dumps(closure.as_dict()))

    assert record["rotation_error_radians"] == 0.25
    assert record["translation_error"] == 1.5
    assert record["frames"] == 9
    assert "not established" in record["translation_error_unit"]


# --------------------------------------------------------------------------
# One real capture, turned synthetically
# --------------------------------------------------------------------------

FIXTURE_FRAMES = Path(__file__).parent / "fixtures" / "frames"
FRAME_PATH_ENV = "SCANNER3D_REAL_DEPTH_FRAME"
EXPECTED_FRAME_BYTES = 512_000

#: This unit's own factory numbers, injected here rather than defaulted anywhere
#: in the code. The depth scalar's physical unit is not established.
REAL_FX = REAL_FY = 898.6829833984375
REAL_CX, REAL_CY = 42.43747329711914, 192.8326873779297
REAL_SCALE = 0.004999999888241291


def real_frame() -> bytes:
    """The bytes of one real capture, or an explicit skip. A skip is not a pass."""
    override = os.environ.get(FRAME_PATH_ENV)
    candidates = [Path(override)] if override else sorted(FIXTURE_FRAMES.glob("base_4.bin"))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size == EXPECTED_FRAME_BYTES:
            return candidate.read_bytes()
    pytest.skip(
        f"no real {EXPECTED_FRAME_BYTES}-byte depth frame found under {FIXTURE_FRAMES}, and "
        f"${FRAME_PATH_ENV} does not name one, so registration was NOT exercised against a "
        "real capture's point distribution on this run. This skip is not a pass"
    )


@functools.cache
def real_cloud() -> tuple[Point, ...]:
    """One real frame, deprojected and downsampled hard enough to match quickly."""
    blob = struct.pack(
        INTRINSICS_FORMAT, 640, 400, REAL_FX, 0.0, REAL_CX, 0.0, REAL_FY, REAL_CY, 0.0, 0.0, 1.0
    )
    record = parse_intrinsics(blob)
    assert record is not None
    conversion = deproject_depth_frame(
        real_frame(), intrinsics=record, depth_scale=REAL_SCALE, unit=UNVALIDATED_UNIT
    )
    points = tuple(camera_points(conversion))
    return voxel_downsample(points, cloud_extent(points) / 64.0)


def test_a_real_captures_point_distribution_registers_through_a_revolution():
    """Real geometry, synthetic motion - and the docstring is the point of the test.

    The points are a genuine capture from this unit: sparse, patchy, with the
    long tail of stray returns a real depth frame carries, and about three
    quarters of the frame's samples produced no point at all. The *rotation* is
    applied by this test. So what is exercised is that the estimator and the
    closure survive a real point distribution.

    It is NOT a turntable capture. No frame in this repository was taken while
    an object rotated, so nothing here says the pipeline works on real turntable
    data, and the numbers below are not evidence that it does.
    """
    cloud = real_cloud()
    # The spindle has to pass through the object it turns, and this object is
    # wherever the real capture put it - not where the synthetic shapes sit.
    through = centroid(cloud)
    spin = rotation_about_axis(TRUE_DIRECTION, math.tau / 8, through=through)
    anchor = subtract(through, scaled(TRUE_DIRECTION, dot(through, TRUE_DIRECTION)))
    frames = 8
    sweep = [tuple(cloud)]
    for _ in range(frames - 1):
        sweep.append(tuple(spin.apply_all(sweep[-1])))

    result = register_turntable(sweep, angle_step=math.tau / 8)

    assert angle_between(result.axis.direction, TRUE_DIRECTION) < 1e-9
    assert math.dist(result.axis.through, anchor) < 1e-6 * cloud_extent(cloud)
    assert result.loop_closure is not None
    assert result.loop_closure.frames == frames
    assert result.loop_closure.rotation_error_radians < 1e-9
    assert result.converged is True
    # Stated as a count, never as a size: these coordinates have no unit.
    assert len(cloud) >= MIN_CLOUD_POINTS
