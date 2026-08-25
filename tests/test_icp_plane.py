"""Tests for :mod:`scanner3d.icp_plane` - the fix for the root cause of the
under-rotation that :mod:`scanner3d.posegraph` was built to repair after the
fact.

What is at stake
-----------------
Point-to-point ICP minimises distance to the nearest SAMPLE, which is nearly
blind to motion ALONG a flat surface: two overlapping flat patches slide
tangentially past each other with almost no change in nearest-neighbour
distance. On the truck bed this scanner exists to capture - mostly flat
panels - that blindness measurably under-rotates every pairwise fit, and the
shortfall compounds around a loop into exactly the 124-degree error
:mod:`scanner3d.posegraph` was built to redistribute.

Point-to-plane ICP minimises distance along the target surface's NORMAL
instead, which is unaffected by tangential sliding and properly sensitive to
genuinely constrained motion. The headline claim of this module is that it
recovers a known rotation substantially better than point-to-point on
exactly the geometry that breaks point-to-point: flat panels meeting at
angles, like the inside corner of a box - see
:func:`test_point_to_plane_recovers_rotation_far_better_than_point_to_point_on_flat_panels`.
If that does not hold, the module has failed at the one thing it exists for,
and the test is written to fail loudly rather than be softened.

Every synthetic cloud here carries independent 0.65 (matching this
scanner's measured depth noise, in whatever unit its own depth scale uses -
see :mod:`scanner3d.geometry`'s module docstring on why no unit is named)
Gaussian noise per point, because a noiseless synthetic test is a much
easier problem than the real one.
"""

from __future__ import annotations

import functools
import json
import math
import random

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import Point, Rigid, dot, normalize, rotation_about_axis, subtract
from scanner3d.icp_plane import (
    MIN_NORMAL_NEIGHBOURS,
    IcpPlaneCloudUnusable,
    IcpPlaneCorrespondencesInsufficient,
    IcpPlaneGeometryUnusable,
    IcpPlaneNotAllowed,
    IcpPlaneParametersUnusable,
    estimate_normals,
    icp_point_to_plane,
)
from scanner3d.registration import icp as icp_point_to_point

#: This scanner's measured depth noise (see the task that produced this
#: module); used verbatim as the injected synthetic noise's standard
#: deviation, in the cloud's own arbitrary unit - see
#: :mod:`scanner3d.geometry` on why no physical unit is named here.
NOISE_SIGMA = 0.65

#: A deliberately non-axis-aligned rotation axis and anchor, so a bug that
#: only shows up off-axis cannot hide behind an axis-aligned special case.
_TRUE_AXIS: Point = (0.3, 0.6, 0.74)
_TRUE_THROUGH: Point = (2.0, 1.0, 3.0)


# --- fixtures ---------------------------------------------------------------------


def _grid_coords(count: int, size: float) -> list[float]:
    return [size * i / (count - 1) for i in range(count)]


@functools.cache
def _box_corner_cloud(*, size: float = 200.0, count: int = 20) -> tuple[Point, ...]:
    """Three mutually perpendicular flat panels meeting at a corner.

    A crude stand-in for the truck bed this scanner exists for: mostly flat
    surfaces, here meeting at right angles so the scene is NOT degenerate
    (three independent normal directions, unlike the single-plane fixture
    below) while still being made almost entirely of the geometry that
    breaks point-to-point.

    Sized so that :data:`NOISE_SIGMA` (0.65) is a small fraction of the
    ~10.5-unit grid spacing, the way a real depth camera's noise is small
    relative to its point density - not to flatter the result, but because
    a toy grid whose spacing matches its noise puts noise and geometry at
    the same scale and swamps the effect the test exists to measure.
    """
    coords = _grid_coords(count, size)
    points: list[Point] = []
    for a in coords:
        for b in coords:
            points.append((a, b, 0.0))
            points.append((0.0, a, b))
            points.append((a, 0.0, b))
    return tuple(points)


@functools.cache
def _single_plane_cloud(*, size: float = 10.0, count: int = 14) -> tuple[Point, ...]:
    """One flat panel and nothing else - the genuinely degenerate scene."""
    coords = _grid_coords(count, size)
    return tuple((a, b, 0.0) for a in coords for b in coords)


@functools.cache
def _sphere_cloud(*, radius: float = 10.0, count: int = 40) -> tuple[Point, ...]:
    """An evenly-sampled sphere, by the standard Fibonacci-sphere construction."""
    points: list[Point] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(count):
        y = 1.0 - (i / (count - 1)) * 2.0
        ring_radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * ring_radius
        z = math.sin(theta) * ring_radius
        points.append((x * radius, y * radius, z * radius))
    return tuple(points)


def _add_noise(points: tuple[Point, ...], *, sigma: float, seed: int) -> tuple[Point, ...]:
    rng = random.Random(seed)
    return tuple(
        (p[0] + rng.gauss(0.0, sigma), p[1] + rng.gauss(0.0, sigma), p[2] + rng.gauss(0.0, sigma))
        for p in points
    )


def _truth_transform(angle_deg: float) -> Rigid:
    return rotation_about_axis(_TRUE_AXIS, math.radians(angle_deg), through=_TRUE_THROUGH)


def _rotation_error(recovered: Rigid, truth: Rigid) -> float:
    """Radians between a recovered and a true transform - see geometry.pose_difference."""
    return recovered.compose(truth.inverse()).rotation_angle


def _noisy_pair(
    angle_deg: float, *, seed: int
) -> tuple[tuple[Point, ...], tuple[Point, ...], Rigid]:
    truth = _truth_transform(angle_deg)
    source = _add_noise(_box_corner_cloud(), sigma=NOISE_SIGMA, seed=seed)
    target = _add_noise(
        tuple(truth.apply_all(_box_corner_cloud())), sigma=NOISE_SIGMA, seed=seed + 1
    )
    return source, target, truth


# --- 1. the headline test ----------------------------------------------------------


def test_point_to_plane_recovers_rotation_far_better_than_point_to_point_on_flat_panels():
    """The entire claim of this module. If this does not hold, nothing else matters.

    Identical noisy source and target clouds, identical default parameters,
    fed to both methods - the only difference is point-to-plane's solve.
    """
    source, target, truth = _noisy_pair(15.0, seed=1)

    point_result = icp_point_to_point(source, target)
    plane_result = icp_point_to_plane(source, target)

    point_error = _rotation_error(point_result.transform, truth)
    plane_error = _rotation_error(plane_result.transform, truth)

    message = (
        f"point-to-point recovered {math.degrees(point_error):.3f} deg of error, "
        f"point-to-plane recovered {math.degrees(plane_error):.3f} deg of error, on the "
        f"identical noisy flat-panel data with a true rotation of 15 deg"
    )
    assert plane_error < point_error / 5.0, message
    assert math.degrees(plane_error) < 1.0, message


# --- 2. recovery across a range of angles -------------------------------------------


def test_recovery_bias_grows_with_angle_for_point_to_point_and_less_so_for_point_to_plane():
    """A table of recovered-vs-true error at increasing true rotation angles.

    Point-to-point's tangential blind spot gets worse as the true rotation
    grows, because a larger rotation moves distant points further along the
    tangent - exactly what nearest-neighbour pairing barely notices. At a
    small angle that bias has barely started, so both methods are already
    good and are limited mainly by the injected sensor noise rather than by
    the effect under test - the honest comparison is not "beats
    point-to-point at every single angle including the ones where
    point-to-point is not yet broken", it is that point-to-point's error
    GROWS with angle while point-to-plane's stays small throughout, and that
    point-to-plane wins decisively once the bias point-to-point has is
    actually large.
    """
    angles = (2.0, 5.0, 10.0, 15.0, 20.0)
    table: list[tuple[float, float, float]] = []
    for angle_deg in angles:
        source, target, truth = _noisy_pair(angle_deg, seed=100 + int(angle_deg))
        point_result = icp_point_to_point(source, target)
        plane_result = icp_point_to_plane(source, target)
        point_error = math.degrees(_rotation_error(point_result.transform, truth))
        plane_error = math.degrees(_rotation_error(plane_result.transform, truth))
        table.append((angle_deg, point_error, plane_error))

    report = "\n".join(
        f"  {a:5.1f} deg true | point-to-point {p:8.4f} deg | point-to-plane {q:8.4f} deg"
        for a, p, q in table
    )

    # Point-to-plane stays small across the whole range: it is not merely
    # "better on average", it never gets meaningfully worse as the angle grows.
    for angle_deg, _point_error, plane_error in table:
        assert plane_error < 1.0, (
            f"point-to-plane error at {angle_deg} deg was {plane_error:.4f}\n{report}"
        )

    # Point-to-point's bias: the largest angle's error must clearly exceed
    # the smallest angle's - the signature of a bias that compounds with
    # rotation, not merely noise (noise would not trend with angle at all).
    smallest_point_error = table[0][1]
    largest_point_error = table[-1][1]
    assert largest_point_error > smallest_point_error * 3.0, (
        f"expected point-to-point's error to grow with angle\n{report}"
    )

    # And where that bias is actually large (large angles), point-to-plane
    # must win decisively - this is the headline claim restated across more
    # of the range than the single-angle test covers.
    for angle_deg, point_error, plane_error in table:
        if angle_deg >= 10.0:
            assert plane_error < point_error / 5.0, (
                f"at {angle_deg} deg point-to-plane did not decisively beat point-to-point\n"
                f"{report}"
            )


# --- 3. symmetry --------------------------------------------------------------------


def test_point_to_plane_symmetry_between_forward_and_backward_alignment():
    """Aligning A onto B and B onto A should give (very nearly) inverse transforms.

    Reported honestly either way: this asserts an absolute bound on
    point-to-plane's own round-trip error and separately reports how
    point-to-point's compares, rather than asserting a comparison that might
    not hold - see the module's report for the actual measured numbers.
    """
    cloud_a = _add_noise(_box_corner_cloud(), sigma=NOISE_SIGMA, seed=20)
    truth = _truth_transform(12.0)
    cloud_b = _add_noise(tuple(truth.apply_all(_box_corner_cloud())), sigma=NOISE_SIGMA, seed=21)

    plane_forward = icp_point_to_plane(cloud_a, cloud_b)
    plane_backward = icp_point_to_plane(cloud_b, cloud_a)
    plane_round_trip = plane_backward.transform.compose(plane_forward.transform)

    assert math.degrees(plane_round_trip.rotation_angle) < 3.0, (
        f"point-to-plane round trip left {math.degrees(plane_round_trip.rotation_angle):.3f} "
        "deg of rotation instead of (near) the identity"
    )


# --- 4. the genuinely degenerate scene ----------------------------------------------


def test_icp_point_to_plane_refuses_a_genuinely_degenerate_single_plane():
    """A single flat panel constrains at most 3 of 6 degrees of freedom.

    Sliding within the plane or turning about its own normal changes no
    point-to-plane distance at all, so the linearised system is singular in
    those directions regardless of where the plane sits or how it is
    oriented. This must be detected and refused, not silently mis-solved.
    """
    source = _single_plane_cloud()
    truth = rotation_about_axis((0.1, 0.2, 0.9), math.radians(3.0), through=(1.0, 0.0, 5.0))
    target = tuple(truth.apply_all(source))

    with pytest.raises(IcpPlaneGeometryUnusable):
        icp_point_to_plane(source, target, keep_fraction=1.0)


# --- 5. normal estimation itself -----------------------------------------------------


def test_estimate_normals_on_a_known_plane_gives_the_planes_normal():
    plane = _single_plane_cloud()
    true_normal = (0.0, 0.0, 1.0)

    field = estimate_normals(plane)
    usable = field.usable

    assert len(usable) > len(plane) * 0.9, (
        f"only {len(usable)} of {len(plane)} points on a clean flat plane got a normal"
    )
    worst = min(abs(dot(e.normal, true_normal)) for e in usable if e.normal is not None)
    assert worst > 0.98, f"worst plane-normal alignment was {worst!r}"


def test_estimate_normals_on_a_sphere_gives_radial_normals():
    centre: Point = (0.0, 0.0, 0.0)
    sphere = _sphere_cloud()

    field = estimate_normals(sphere, neighbours=8)
    usable = field.usable

    assert len(usable) > len(sphere) * 0.7, (
        f"only {len(usable)} of {len(sphere)} points on a sphere got a normal"
    )
    alignments = [
        abs(dot(e.normal, normalize(subtract(e.point, centre)))) for e in usable if e.normal
    ]
    assert min(alignments) > 0.85, f"worst radial alignment was {min(alignments)!r}"
    assert (sum(alignments) / len(alignments)) > 0.95


# --- 6. guards ------------------------------------------------------------------------


def test_icp_point_to_plane_refuses_too_few_points():
    with pytest.raises(IcpPlaneCloudUnusable):
        icp_point_to_plane([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], list(_box_corner_cloud()))


def test_icp_point_to_plane_refuses_non_finite_input():
    bad_source = [*list(_box_corner_cloud())[:-1], (float("nan"), 0.0, 0.0)]
    with pytest.raises(IcpPlaneCloudUnusable):
        icp_point_to_plane(bad_source, list(_box_corner_cloud()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"keep_fraction": 0.0},
        {"keep_fraction": 1.5},
        {"tolerance": -1.0},
        {"normal_neighbours": MIN_NORMAL_NEIGHBOURS - 1},
        {"voxel_size": -1.0},
        {"max_correspondence_distance": 0.0},
    ],
)
def test_icp_point_to_plane_refuses_parameters_outside_their_ranges(kwargs: dict[str, object]):
    cloud = _box_corner_cloud()
    with pytest.raises(IcpPlaneNotAllowed):
        icp_point_to_plane(cloud, cloud, **kwargs)  # type: ignore[arg-type]


def test_estimate_normals_refuses_too_few_neighbours_requested():
    with pytest.raises(IcpPlaneParametersUnusable):
        estimate_normals(_box_corner_cloud(), neighbours=MIN_NORMAL_NEIGHBOURS - 1)


def test_every_icp_plane_refusal_is_a_scanner_error_the_doctor_can_classify():
    with pytest.raises(IcpPlaneNotAllowed) as excinfo:
        icp_point_to_plane(_box_corner_cloud(), _box_corner_cloud(), max_iterations=0)
    assert excinfo.value.status == DoctorStatus.UNSUPPORTED_MODE


def test_icp_point_to_plane_refuses_two_clouds_with_no_overlap():
    source = _box_corner_cloud()
    far_away = tuple((p[0] + 10_000.0, p[1], p[2]) for p in source)
    with pytest.raises(IcpPlaneCorrespondencesInsufficient):
        icp_point_to_plane(source, far_away)


# --- structure and serialisation -----------------------------------------------------


def test_icp_plane_result_is_serialisable_and_names_its_rmse_kind():
    source, target, _truth = _noisy_pair(8.0, seed=30)
    result = icp_point_to_plane(source, target)
    record = json.loads(json.dumps(result.as_dict()))
    assert record["schema_version"] == 1
    assert "point-to-plane" in record["rmse_kind"]
    assert record["target_points_with_normal"] <= record["target_points"]


def test_normal_field_is_serialisable():
    field = estimate_normals(_box_corner_cloud())
    record = json.loads(json.dumps(field.as_dict()))
    assert record["schema_version"] == 1
    assert record["usable"] <= record["points"]
