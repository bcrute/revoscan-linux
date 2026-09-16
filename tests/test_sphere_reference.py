"""Sphere-reference fitting, exercised on frames rendered through the real Y16 path."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from scanner3d.simulate import FACTORY_INTRINSICS, Plane, Sphere, intrinsics_of, render

#: A 57.265 mm cue ball, the owner's caliper mean.
RADIUS = 28.6325
CALIPER = 57.265
SPREAD = 0.03


@pytest.fixture(scope="module")
def tool():
    import sys

    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("sphere_reference")


def write_capture(directory, shapes, *, intrinsics=FACTORY_INTRINSICS, frames=2):
    directory.mkdir()
    for index in range(frames):
        frame = render(shapes, intrinsics=intrinsics, jitter=0.02, dropout=0.02, seed=index + 1)
        (directory / f"frame_{index:04d}.raw").write_bytes(frame.raw)
    return directory


@pytest.fixture(scope="module")
def single_ball(tmp_path_factory):
    return write_capture(
        tmp_path_factory.mktemp("cap") / "ball",
        [Sphere(centre=(60.0, 5.0, 200.0), radius=RADIUS)],
    )


def test_sphere_and_ellipsoid_recover_a_true_sphere(tool, single_ball):
    summary = tool.analyse_capture(
        single_ball,
        tool.Options(),
        tool.Comparison(caliper_mm=CALIPER, caliper_spread_mm=SPREAD),
    )
    ball = summary["balls"][0]
    assert ball["frames_fitted"] == 2
    assert abs(ball["diameter_mm"]["median"] - CALIPER) < 0.01
    for axis, truth in zip("xyz", (60.0, 5.0, 200.0), strict=True):
        assert abs(ball["centre_mm"][axis]["median"] - truth) < 0.02
    assert ball["rms_mm"]["median"] < 0.03
    assert ball["cap_half_angle_degrees"]["median"] <= tool.DEFAULT_RIM_DEGREES
    assert ball["inliers"]["median"] > 5000
    ellipsoid = ball["ellipsoid"]
    assert abs(ellipsoid["x_over_z"]["median"] - 1.0) < 0.002
    assert abs(ellipsoid["y_over_z"]["median"] - 1.0) < 0.002
    versus = ball["diameter_vs_caliper"]
    assert versus["reference_mm"] == CALIPER
    assert versus["outside_reference_spread"] is False
    assert summary["refused_frames"] == []


def test_a_lateral_scale_error_shows_as_anisotropy(tool, tmp_path):
    """Rendered with a focal length 0.8% shorter than the record used to deproject,
    so lateral coordinates come out 0.8% small while depth is untouched."""
    factor = 1.008
    fx = FACTORY_INTRINSICS.values["fx"] / factor
    squeezed = intrinsics_of(
        fx=fx,
        fy=fx,
        cx=FACTORY_INTRINSICS.values["cx"],
        cy=FACTORY_INTRINSICS.values["cy"],
    )
    capture = write_capture(
        tmp_path / "squeezed",
        [Sphere(centre=(60.0, 5.0, 200.0), radius=RADIUS)],
        intrinsics=squeezed,
    )
    summary = tool.analyse_capture(capture, tool.Options(), tool.Comparison())
    ball = summary["balls"][0]
    ellipsoid = ball["ellipsoid"]
    assert abs(ellipsoid["x_over_z"]["median"] - 1.0 / factor) < 0.0015
    assert abs(ellipsoid["y_over_z"]["median"] - 1.0 / factor) < 0.0015
    # A single cap constrains the depth semi-axis less tightly than the ratios.
    assert abs(ellipsoid["diameter_z_mm"]["median"] - CALIPER) < 0.1
    # A sphere fitted to a squashed cap follows the apex curvature, a**2 / c, so it
    # lands at or below the small axis: the ellipsoid, not the sphere, is the readout.
    assert 0.975 * CALIPER < ball["diameter_mm"]["median"] < CALIPER / factor + 0.02


def test_two_balls_give_centre_spacing_along_x(tool, tmp_path):
    capture = write_capture(
        tmp_path / "pair",
        [
            Sphere(centre=(30.0, 0.0, 210.0), radius=RADIUS),
            Sphere(centre=(92.0, 0.0, 210.0), radius=RADIUS),
        ],
    )
    summary = tool.analyse_capture(
        capture,
        tool.Options(balls=2, ellipsoid=False),
        tool.Comparison(spacing_mm=62.0, spacing_spread_mm=0.02),
    )
    assert [ball["index"] for ball in summary["balls"]] == [0, 1]
    assert (
        summary["balls"][0]["centre_mm"]["x"]["median"]
        < summary["balls"][1]["centre_mm"]["x"]["median"]
    )
    spacing = summary["spacing"]
    assert abs(spacing["distance_mm"]["median"] - 62.0) < 0.02
    assert abs(spacing["delta_mm"]["x"]["median"] - 62.0) < 0.02
    assert abs(spacing["delta_mm"]["y"]["median"]) < 0.02
    assert abs(spacing["delta_mm"]["z"]["median"]) < 0.02
    assert spacing["distance_vs_caliper"]["outside_reference_spread"] is False


def test_region_selection_matches_automatic_selection(tool, single_ball):
    automatic = tool.analyse_capture(single_ball, tool.Options(), tool.Comparison())
    blob = automatic["balls"][0]["per_frame"][0]["blob"]["bounds_x0_y0_x1_y1"]
    x0, y0, x1, y1 = blob
    region = tool.analyse_capture(
        single_ball,
        tool.Options(rois=((max(0, x0 - 5), max(0, y0 - 5), x1 + 5, y1 + 5),), balls=1),
        tool.Comparison(),
    )
    assert (
        abs(
            region["balls"][0]["diameter_mm"]["median"]
            - automatic["balls"][0]["diameter_mm"]["median"]
        )
        < 1e-6
    )


def test_skip_and_max_frames_bound_the_frames_used(tool, single_ball):
    summary = tool.analyse_capture(single_ball, tool.Options(skip_frames=1), tool.Comparison())
    assert summary["frames"] == 1
    assert summary["balls"][0]["per_frame"][0]["frame"] == "frame_0001.raw"
    summary = tool.analyse_capture(single_ball, tool.Options(max_frames=1), tool.Comparison())
    assert summary["balls"][0]["per_frame"][0]["frame"] == "frame_0000.raw"


def test_refusals_name_the_problem(tool, tmp_path):
    with pytest.raises(tool.SphereReferenceError, match="holds no frame_"):
        tool.analyse_capture(tmp_path, tool.Options(), tool.Comparison())
    short = tmp_path / "short"
    short.mkdir()
    (short / "frame_0000.raw").write_bytes(b"\x00" * 10)
    with pytest.raises(tool.SphereReferenceError, match="holds 10 bytes"):
        tool.analyse_capture(short, tool.Options(), tool.Comparison())
    # A plane fills the frame: in band, but not a disc, so no ball is found.
    plane = write_capture(
        tmp_path / "plane", [Plane(normal=(0.0, 0.0, 1.0), offset=200.0)], frames=1
    )
    with pytest.raises(tool.SphereReferenceError, match="disc-like"):
        tool.analyse_capture(plane, tool.Options(), tool.Comparison())


def test_cli_writes_a_report_and_refuses_to_overwrite(tool, single_ball, tmp_path, capsys):
    report = tmp_path / "report.json"
    code = tool.main(
        [
            str(single_ball),
            "--caliper-mm",
            str(CALIPER),
            "--caliper-spread-mm",
            str(SPREAD),
            "--no-ellipsoid",
            "--output",
            str(report),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "ball 0: diameter" in printed
    assert "vs caliper 57.265 mm" in printed
    written = json.loads(report.read_text())
    assert written["schema_version"] == tool.SCHEMA_VERSION
    assert written["reference"]["caliper_mm"] == CALIPER
    assert "ellipsoid" not in written["captures"][0]["balls"][0]
    assert written["limitations"]
    assert tool.main([str(single_ball), "--output", str(report)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_rejects_bad_regions_and_bands(tool, single_ball, capsys):
    assert tool.main([str(single_ball), "--roi", "0", "0", "700", "10"]) == 2
    assert "not inside" in capsys.readouterr().err
    assert tool.main([str(single_ball), "--band", "300", "100"]) == 2
    assert "0 < NEAR < FAR" in capsys.readouterr().err


def test_two_touching_balls_are_split_and_spaced(tool, tmp_path):
    import numpy as np

    r1, r2 = RADIUS, 28.45
    length = r1 + r2
    direction = np.array([0.8, 0.0, 0.6])
    centre = np.array([60.0, 5.0, 210.0])
    near = centre - direction * length / 2
    far = centre + direction * length / 2
    capture = write_capture(
        tmp_path / "touch",
        [
            Sphere(centre=(float(near[0]), float(near[1]), float(near[2])), radius=r1),
            Sphere(centre=(float(far[0]), float(far[1]), float(far[2])), radius=r2),
        ],
        frames=1,
    )
    summary = tool.analyse_capture(
        capture,
        tool.Options(balls=2, ellipsoid=False, expected_diameter_mm=2 * r1),
        tool.Comparison(spacing_mm=length),
    )
    assert summary["refused_frames"] == []
    assert len(summary["balls"]) == 2
    spacing = summary["spacing"]
    assert abs(spacing["distance_mm"]["median"] - length) < 0.02
    assert abs(abs(spacing["delta_mm"]["x"]["median"]) - 0.8 * length) < 0.03
    assert abs(abs(spacing["delta_mm"]["z"]["median"]) - 0.6 * length) < 0.03
    assert abs(spacing["delta_mm"]["y"]["median"]) < 0.03
    diameters = sorted(ball["diameter_mm"]["median"] for ball in summary["balls"])
    assert abs(diameters[0] - 2 * r2) < 0.02
    assert abs(diameters[1] - 2 * r1) < 0.02


def test_rotating_pair_recovers_per_axis_scale(tool, tmp_path):
    """A pair of touching balls turning on a table, rendered with the lateral axes
    scaled 0.8% (x) and 0.4% (y) short of the record used to deproject."""
    import math

    import numpy as np

    fx = FACTORY_INTRINSICS.values["fx"]
    squeezed = intrinsics_of(
        fx=fx / 1.008,
        fy=fx / 1.004,
        cx=FACTORY_INTRINSICS.values["cx"],
        cy=FACTORY_INTRINSICS.values["cy"],
    )
    r1, r2 = RADIUS, 28.45
    length = r1 + r2
    pitch = math.radians(35.0)
    centre = np.array([60.0, 0.0, 215.0])
    directory = tmp_path / "turn"
    directory.mkdir()
    for k, degrees in enumerate(range(10, 360, 45)):
        theta = math.radians(degrees)
        u = np.array(
            [math.cos(theta), math.sin(theta) * math.sin(pitch), math.sin(theta) * math.cos(pitch)]
        )
        one = centre - u * length / 2
        two = centre + u * length / 2
        frame = render(
            [
                Sphere(centre=(float(one[0]), float(one[1]), float(one[2])), radius=r1),
                Sphere(centre=(float(two[0]), float(two[1]), float(two[2])), radius=r2),
            ],
            intrinsics=squeezed,
            jitter=0.02,
            dropout=0.02,
            seed=k + 1,
        )
        (directory / f"frame_{k:04d}.raw").write_bytes(frame.raw)
    summary = tool.analyse_capture(
        directory,
        tool.Options(balls=2, ellipsoid=False, expected_diameter_mm=2 * r1),
        tool.Comparison(spacing_mm=length),
    )
    scales = summary["spacing"]["axis_scales"]
    assert scales["frames"] >= 6
    directions = np.array(scales["directions_camera_xyz"])
    factors = scales["measured_over_true"]
    coverage = scales["coverage_max_component_over_length"]
    # first direction is camera x; second is the table direction, mixing y and z
    assert abs(directions[0][0]) > 0.995
    assert abs(factors[0] - 1 / 1.008) < 0.0015
    tilt = np.array([0.0, math.sin(pitch), math.cos(pitch)])
    assert abs(directions[1] @ tilt) > 0.995
    along_tilt = 1 / math.sqrt(math.sin(pitch) ** 2 * 1.004**2 + math.cos(pitch) ** 2)
    assert abs(factors[1] - along_tilt) < 0.002
    # the table normal is never covered by a turn, and says so
    assert factors[2] is None
    assert coverage[2] < 0.05
    assert scales["rms_mm"] < scales["isotropic_rms_mm"]
    # identity follows each ball round the turn: ball 0 keeps one diameter
    assert summary["balls"][0]["diameter_mm"]["mad_sigma"] < 0.03
    assert (
        abs(
            summary["balls"][0]["diameter_mm"]["median"]
            - summary["balls"][1]["diameter_mm"]["median"]
        )
        > 0.2
    )
