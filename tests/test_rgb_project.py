"""Tests for :mod:`scanner3d.rgb_project`.

What is at stake
-----------------
This module turns the factory calibration read off the real device -
committed as fixtures at ``tests/fixtures/camparam/`` - into a colour per
depth point. Test 1 pins the exact numbers those fixtures decode to, so a
future re-read that disagrees is a FINDING (the calibration changed, or the
parser broke), never something a passing test quietly absorbs. Tests 3-5
are about the one thing genuinely easy to get wrong in projection code: an
almost-right pipeline (extrinsics/intrinsics/distortion applied in the wrong
order, a behind-camera point silently reflected, a residual sign error) does
not crash - it produces a pixel that looks entirely plausible and is simply
wrong, which is exactly what these tests are built to catch instead of let
through.
"""

from __future__ import annotations

import itertools
import math
import struct
from pathlib import Path

import pytest

from scanner3d.errors import DoctorStatus
from scanner3d.geometry import NotARotation, Point, Rigid
from scanner3d.rgb_project import (
    CalibrationBlobMalformed,
    DistortionCoefficients,
    ImageBufferMalformed,
    RgbCalibration,
    RgbImage,
    RgbProjectNotAllowed,
    color_points,
    parse_camera_intrinsics,
    parse_distortion,
    parse_extrinsics,
    project_point,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "camparam"


def _read(name: str) -> bytes:
    return (FIXTURE_DIR / name).read_bytes()


# --- 1. the fixtures parse to the measured values, hard-coded ------------------------


def test_pl_bin_parses_to_the_measured_depth_intrinsics():
    intrinsics = parse_camera_intrinsics(_read("Pl.bin"))
    assert intrinsics.width == 640
    assert intrinsics.height == 400
    assert intrinsics.fx == pytest.approx(898.683, abs=1e-3)
    assert intrinsics.fy == pytest.approx(898.683, abs=1e-3)
    assert intrinsics.cx == pytest.approx(42.437, abs=1e-3)
    assert intrinsics.cy == pytest.approx(192.833, abs=1e-3)


def test_prgb_bin_parses_to_the_measured_rgb_intrinsics():
    intrinsics = parse_camera_intrinsics(_read("Prgb.bin"))
    assert intrinsics.width == 640
    assert intrinsics.height == 400
    assert intrinsics.fx == pytest.approx(895.362, abs=1e-3)
    assert intrinsics.fy == pytest.approx(896.518, abs=1e-3)
    assert intrinsics.cx == pytest.approx(316.190, abs=1e-3)
    assert intrinsics.cy == pytest.approx(204.116, abs=1e-3)


def test_lc_rt_bin_parses_to_the_measured_extrinsics():
    extrinsics = parse_extrinsics(_read("LC_RT.bin"))
    # Rows unit to 1e-9, mutually orthogonal to 4e-9, determinant 1.000000,
    # 10.402 degrees of rotation - the figures measured while reading this
    # blob off the real device. rotation_angle already re-derives the angle
    # the conditioning-aware way (see Rigid.rotation_angle's own docstring).
    assert math.degrees(extrinsics.rotation_angle) == pytest.approx(10.402, abs=1e-3)
    assert extrinsics.translation[0] == pytest.approx(-14.7175, abs=1e-3)
    assert extrinsics.translation[1] == pytest.approx(-0.315, abs=1e-3)
    assert extrinsics.translation[2] == pytest.approx(-5.629, abs=1e-3)
    assert extrinsics.translation_magnitude == pytest.approx(15.76, abs=1e-2)

    # The full row-major matrix, exactly as struct.unpack("<12f", ...) reads
    # the blob - not merely rotation_angle, which cannot tell a correctly
    # oriented rotation from its own transpose (a transpose is still a
    # rotation, with the identical angle, about the negated axis; mutation
    # testing this module found that gap - see the coder report). Pinned to
    # the exact measured values, off-diagonal entries included, specifically
    # because a transposed read would swap them and this would then fail.
    expected_rotation = (
        (0.9835754632949829, -0.014130198396742344, -0.17994356155395508),
        (0.013320814818143845, 0.9998949766159058, -0.005705610383301973),
        (0.18000528216362, 0.0032149036414921284, 0.9836603999137878),
    )
    for actual_row, expected_row in zip(extrinsics.rotation, expected_rotation, strict=True):
        for actual, expected in zip(actual_row, expected_row, strict=True):
            assert actual == pytest.approx(expected, abs=1e-9)


def test_distort_bin_parses_to_the_measured_coefficients():
    distortion = parse_distortion(_read("Distort.bin"))
    assert distortion.k1 == pytest.approx(0.2026, abs=1e-4)
    assert distortion.k2 == pytest.approx(-0.4748, abs=1e-4)
    assert distortion.p1 == pytest.approx(-0.001035, abs=1e-6)
    assert distortion.p2 == pytest.approx(-0.005375, abs=1e-6)
    assert distortion.k3 == pytest.approx(-0.5842, abs=1e-4)


def test_camparam_yaml_is_not_read_by_this_module():
    """The yaml fixture names a different device and this module never touches it -
    there is simply no function here that takes a ``.yaml`` path or bytes at all."""
    import scanner3d.rgb_project as module

    assert not any("yaml" in name.lower() for name in dir(module))


# --- 2. LC_RT's rotation is validated, reusing geometry.ensure_rotation --------------


def test_lc_rt_extrinsics_rotation_is_a_genuine_rotation():
    extrinsics = parse_extrinsics(_read("LC_RT.bin"))
    # A Rigid that exists at all has already passed ensure_rotation, inside
    # Rigid.of - re-asserting the orthonormality property directly here as a
    # belt-and-braces check that this is not merely "didn't raise".
    r = extrinsics.rotation
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            assert dot == pytest.approx(expected, abs=1e-6)


def test_a_corrupted_lc_rt_blob_raises_not_a_rotation():
    """Reusing geometry.ensure_rotation means a genuinely non-orthonormal
    rotation raises geometry's OWN exception, not a new one this module
    invented - see parse_extrinsics's docstring for why that is deliberate.
    """
    original = struct.unpack("<12f", _read("LC_RT.bin"))
    corrupted = list(original)
    corrupted[0] = 5.0  # blow up the first rotation entry - no longer unit length
    blob = struct.pack("<12f", *corrupted)
    with pytest.raises(NotARotation):
        parse_extrinsics(blob)


# --- 3. projection round-trip: project, then recover a point on the same ray --------


def _undistort_iterative(
    xd: float, yd: float, distortion: DistortionCoefficients, iterations: int = 20
) -> tuple[float, float]:
    """The inverse of the distortion step in :func:`scanner3d.rgb_project.project_point`:
    given distorted normalised coordinates, recover the undistorted ones by
    fixed-point iteration (the standard approach - OpenCV's own
    ``undistortPoints`` does the same thing, since the forward model has no
    closed-form inverse). Test-only: the module under test does not need an
    inverse to do its own job, so this stays out of the module itself.
    """
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + distortion.k1 * r2 + distortion.k2 * r2 * r2 + distortion.k3 * r2 * r2 * r2
        tangential_x = 2.0 * distortion.p1 * x * y + distortion.p2 * (r2 + 2.0 * x * x)
        tangential_y = distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * distortion.p2 * x * y
        x = (xd - tangential_x) / radial
        y = (yd - tangential_y) / radial
    return x, y


@pytest.fixture
def calibration() -> RgbCalibration:
    return RgbCalibration(
        depth_to_rgb=parse_extrinsics(_read("LC_RT.bin")),
        rgb_intrinsics=parse_camera_intrinsics(_read("Prgb.bin")),
        distortion=parse_distortion(_read("Distort.bin")),
    )


@pytest.mark.parametrize(
    "point",
    [
        (0.0, 0.0, 500.0),
        (80.0, -40.0, 600.0),
        (50.0, 30.0, 500.0),
        (-40.0, -30.0, 450.0),
    ],
)
def test_projection_round_trip_recovers_a_point_on_the_same_ray(
    calibration: RgbCalibration, point: Point
):
    """Project a depth-camera point; invert the pixel back to a ray in RGB
    camera coordinates; scale that ray by the ORIGINAL projection's own RGB-frame
    depth (a ray alone has no depth - see the module docstring's "does not do
    sub-pixel...", the same "this needs extra information" idea); apply the
    INVERSE extrinsics; compare against the original point.

    Measured worst-case error, across all four test points: 1.24e-5 absolute
    on every depth-camera-frame coordinate (points range up to a few hundred
    units) - asserted here at a 1e-4 margin, not the tighter measured figure,
    because that residual is entirely the iterative undistort's: the forward
    distortion model has no closed-form inverse, so
    :func:`_undistort_iterative` only converges to where 20 fixed-point
    iterations land it, not to zero, and how many iterations it takes to
    close the gap further is not what this test is checking. It is precise
    enough to demonstrate the SAME ray, not merely a similar one.
    """
    projected = project_point(point, calibration)
    assert projected.accepted
    assert projected.pixel is not None
    u, v = projected.pixel

    intrinsics = calibration.rgb_intrinsics
    x_distorted = (u - intrinsics.cx) / intrinsics.fx
    y_distorted = (v - intrinsics.cy) / intrinsics.fy
    x_norm, y_norm = _undistort_iterative(x_distorted, y_distorted, calibration.distortion)

    rgb_z = projected.rgb_point[2]
    recovered_rgb_point = (x_norm * rgb_z, y_norm * rgb_z, rgb_z)
    recovered_depth_point = calibration.depth_to_rgb.inverse().apply(recovered_rgb_point)

    for original, recovered in zip(point, recovered_depth_point, strict=True):
        assert original == pytest.approx(recovered, abs=1e-4)


# --- 4. a point behind the RGB camera is rejected, not projected --------------------


def test_point_behind_the_rgb_camera_is_rejected():
    """The classic silent-wrong-answer trap: dividing by a negative depth
    does not fail loudly, it reflects the point through the camera centre
    and produces a pixel that looks entirely plausible for the wrong point.
    """
    # LC_RT's translation places the RGB camera's origin behind the depth
    # camera's own origin along z by only ~5.6 units, so a point placed well
    # BEHIND that (negative RGB-frame z) needs only a modest negative depth
    # in the depth frame given the ~10 degree rotation between them.
    calibration = RgbCalibration(
        depth_to_rgb=parse_extrinsics(_read("LC_RT.bin")),
        rgb_intrinsics=parse_camera_intrinsics(_read("Prgb.bin")),
        distortion=parse_distortion(_read("Distort.bin")),
    )
    behind = project_point((0.0, 0.0, -50.0), calibration)
    assert not behind.accepted
    assert behind.rejected == "behind_camera"
    assert behind.pixel is None
    assert behind.rgb_point[2] <= 0.0

    in_front = project_point((0.0, 0.0, 500.0), calibration)
    assert in_front.accepted
    assert in_front.rgb_point[2] > 0.0


def test_point_outside_the_image_is_rejected_not_clamped():
    calibration = RgbCalibration(
        depth_to_rgb=parse_extrinsics(_read("LC_RT.bin")),
        rgb_intrinsics=parse_camera_intrinsics(_read("Prgb.bin")),
        distortion=parse_distortion(_read("Distort.bin")),
    )
    # Far off-axis at a shallow depth: well outside the 640x400 frame.
    far_off_axis = project_point((5000.0, 5000.0, 50.0), calibration)
    assert not far_off_axis.accepted
    assert far_off_axis.rejected == "outside_image"


# --- 5. distortion actually does something, and grows with radius -------------------


def test_distortion_changes_the_projected_pixel_and_the_change_grows_with_radius():
    """Isolates distortion's own behaviour from the extrinsics: an IDENTITY
    extrinsics is used here (not the real ``LC_RT.bin`` transform) so that a
    chosen depth-frame offset maps to a precisely known NORMALISED radius in
    the RGB camera, rather than one additionally reshaped by a 10-degree
    rotation - what this test is checking is the shape of the distortion
    curve itself, not the composed extrinsics-then-distortion pipeline
    (which the round-trip tests above already exercise together).
    """
    intrinsics = parse_camera_intrinsics(_read("Prgb.bin"))
    identity_extrinsics = Rigid.identity()
    distortion = parse_distortion(_read("Distort.bin"))
    zero_distortion = DistortionCoefficients(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)

    with_distortion = RgbCalibration(identity_extrinsics, intrinsics, distortion)
    without_distortion = RgbCalibration(identity_extrinsics, intrinsics, zero_distortion)

    # Points at increasing NORMALISED radius from the optical axis (x/z),
    # same depth - purely along x, so radius is exactly x/z.
    radii = [0.05, 0.15, 0.25, 0.35]
    deltas: list[float] = []
    for radius in radii:
        depth = 400.0
        point = (radius * depth, 0.0, depth)
        distorted = project_point(point, with_distortion)
        undistorted = project_point(point, without_distortion)
        assert distorted.accepted and undistorted.accepted
        assert distorted.pixel is not None and undistorted.pixel is not None
        delta = math.dist(distorted.pixel, undistorted.pixel)
        deltas.append(delta)

    # It does something at all: not one delta is zero.
    assert all(delta > 1e-6 for delta in deltas)
    # And it grows with radius: each successive (larger-offset) delta is
    # bigger than the last - the signature of a genuine radial/tangential
    # model, not a constant offset that happens to be nonzero.
    for smaller, larger in itertools.pairwise(deltas):
        assert larger > smaller


# --- 6. guards ------------------------------------------------------------------------


def test_parse_camera_intrinsics_refuses_the_wrong_blob_size():
    with pytest.raises(CalibrationBlobMalformed):
        parse_camera_intrinsics(b"\x00" * 39)
    with pytest.raises(CalibrationBlobMalformed):
        parse_camera_intrinsics(b"\x00" * 41)


def test_parse_extrinsics_refuses_the_wrong_blob_size():
    with pytest.raises(CalibrationBlobMalformed):
        parse_extrinsics(b"\x00" * 47)


def test_parse_distortion_refuses_the_wrong_blob_size():
    with pytest.raises(CalibrationBlobMalformed):
        parse_distortion(b"\x00" * 19)


def test_parse_camera_intrinsics_refuses_a_non_finite_focal_length():
    header = struct.pack("<HH", 640, 400)
    floats = struct.pack("<9f", float("nan"), 0.0, 320.0, 0.0, 900.0, 200.0, 0.0, 0.0, 1.0)
    with pytest.raises(CalibrationBlobMalformed):
        parse_camera_intrinsics(header + floats)


def test_parse_camera_intrinsics_refuses_a_wrong_fixed_entry():
    """The "skew" entry must be exactly 0.0 - the layout fixes it, and a
    blob claiming otherwise is not a valid pinhole camera matrix.
    """
    header = struct.pack("<HH", 640, 400)
    floats = struct.pack("<9f", 900.0, 0.01, 320.0, 0.0, 900.0, 200.0, 0.0, 0.0, 1.0)
    with pytest.raises(CalibrationBlobMalformed):
        parse_camera_intrinsics(header + floats)


def test_parse_camera_intrinsics_refuses_non_positive_dimensions():
    header = struct.pack("<HH", 0, 400)
    floats = struct.pack("<9f", 900.0, 0.0, 320.0, 0.0, 900.0, 200.0, 0.0, 0.0, 1.0)
    with pytest.raises(CalibrationBlobMalformed):
        parse_camera_intrinsics(header + floats)


def test_parse_distortion_refuses_a_non_finite_coefficient():
    blob = struct.pack("<5f", 0.2, float("inf"), 0.0, 0.0, -0.5)
    with pytest.raises(CalibrationBlobMalformed):
        parse_distortion(blob)


def test_color_points_refuses_an_empty_cloud(calibration: RgbCalibration):
    image = RgbImage.of(640, 400, bytes(640 * 400 * 3))
    with pytest.raises(RgbProjectNotAllowed):
        color_points([], calibration, image)


def test_color_points_refuses_an_image_at_the_wrong_resolution(calibration: RgbCalibration):
    image = RgbImage.of(320, 200, bytes(320 * 200 * 3))
    with pytest.raises(ImageBufferMalformed):
        color_points([(0.0, 0.0, 500.0)], calibration, image)


def test_rgb_image_of_refuses_a_buffer_of_the_wrong_length():
    with pytest.raises(ImageBufferMalformed):
        RgbImage.of(640, 400, bytes(100))


def test_rgb_image_of_refuses_non_positive_dimensions():
    with pytest.raises(ImageBufferMalformed):
        RgbImage.of(0, 400, b"")


def test_every_rgb_project_refusal_is_a_scanner_error_the_doctor_can_classify():
    with pytest.raises(CalibrationBlobMalformed) as excinfo:
        parse_distortion(b"\x00" * 19)
    assert excinfo.value.status == DoctorStatus.UNSUPPORTED_MODE


# --- structure and serialisation -----------------------------------------------------


def test_color_points_colours_every_point_and_reports_rejections(calibration: RgbCalibration):
    width, height = 640, 400
    # A distinctive colour at every pixel: (col % 256, row % 256, 128).
    pixels = bytearray(width * height * 3)
    for row in range(height):
        for col in range(width):
            offset = (row * width + col) * 3
            pixels[offset] = col % 256
            pixels[offset + 1] = row % 256
            pixels[offset + 2] = 128
    image = RgbImage.of(width, height, bytes(pixels))

    points: list[Point] = [(0.0, 0.0, 500.0), (0.0, 0.0, -50.0), (5000.0, 5000.0, 50.0)]
    result = color_points(points, calibration, image)

    assert len(result.colored) == 3
    assert result.accepted_count == 1
    assert result.rejected_count == 2
    accepted = result.colored[0]
    assert accepted.color is not None
    assert accepted.projection.pixel is not None
    expected_column = int(accepted.projection.pixel[0])
    expected_row = int(accepted.projection.pixel[1])
    assert accepted.color == (expected_column % 256, expected_row % 256, 128)
    assert result.colored[1].color is None
    assert result.colored[2].color is None


def test_projected_point_as_dict_is_json_safe():
    import json

    projection = project_point(
        (0.0, 0.0, 500.0),
        RgbCalibration(
            depth_to_rgb=parse_extrinsics(_read("LC_RT.bin")),
            rgb_intrinsics=parse_camera_intrinsics(_read("Prgb.bin")),
            distortion=parse_distortion(_read("Distort.bin")),
        ),
    )
    json.dumps(projection.as_dict())
    assert projection.as_dict()["schema_version"] == 1


def test_coloring_result_as_dict_is_json_safe(calibration: RgbCalibration):
    import json

    image = RgbImage.of(640, 400, bytes(640 * 400 * 3))
    result = color_points([(0.0, 0.0, 500.0)], calibration, image)
    json.dumps(result.as_dict())
    assert result.as_dict()["point_count"] == 1
