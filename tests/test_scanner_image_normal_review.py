"""Independent scalar image geometry and explicit unavailable-direction accounting."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.simulate import FACTORY_INTRINSICS


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_image_normal_review")


def scalar_point(frame, u, v):
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    z = int(frame[v, u]) * DEPTH_SCALE_METRES
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def scalar_normal(frame, u, v):
    left, right = scalar_point(frame, u - 1, v), scalar_point(frame, u + 1, v)
    top, bottom = scalar_point(frame, u, v - 1), scalar_point(frame, u, v + 1)
    x = tuple(b - a for a, b in zip(left, right, strict=True))
    y = tuple(b - a for a, b in zip(top, bottom, strict=True))
    n = (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])
    length = math.sqrt(sum(a * a for a in n))
    return tuple(a / length for a in n), (math.hypot(*x), math.hypot(*y))


@pytest.mark.parametrize("slopes", [(0, 0), (1, 0), (0, -2), (3, -4)])
def test_scalar_geometry_and_exact_raw_mapping(module, slopes):
    y, x = np.indices((400, 640))
    frame = (20000 + slopes[0] * x + slopes[1] * y).astype("<u2")
    raw = frame.tobytes()
    pixels = np.array([641, 12 * 640 + 42, 199 * 640 + 300, 398 * 640 + 638])
    points = tuple(scalar_point(frame, int(i % 640), int(i // 640)) for i in pixels)
    assert np.array_equal(module.original_pixels(raw, points), pixels)
    for row, index in zip(module.image_directions(raw, pixels), pixels, strict=True):
        expected, lengths = scalar_normal(frame, int(index % 640), int(index // 640))
        assert row["status"] == "usable"
        assert row["normal"] == pytest.approx(expected, abs=1e-15)
        assert row["tangent_lengths"] == pytest.approx(lengths, abs=1e-15)
    assert raw == frame.tobytes()


@pytest.mark.parametrize("pixel", [0, 639, 399 * 640, 255999, 12 * 640, 640 + 639])
def test_border_stays_unavailable(module, pixel):
    result = module.image_directions(np.full((400, 640), 20000, "<u2").tobytes(), np.array([pixel]))
    assert result == [dict(pixel=pixel, status="border", normal=None, tangent_lengths=None)]


@pytest.mark.parametrize("offset", [-641, -640, -639, -1, 0, 1, 639, 640, 641])
@pytest.mark.parametrize("code", [0, 12000, 60001, 65535])
def test_every_neighbor_missing_or_outside_band_is_explicit(module, offset, code):
    frame = np.full(256000, 20000, "<u2")
    pixel = 200 * 640 + 320
    frame[pixel + offset] = code
    row = module.image_directions(frame.tobytes(), np.array([pixel]))[0]
    assert row["status"] == "missing-or-band" and row["normal"] is None


@pytest.mark.parametrize("difference,expected", [(300, "usable"), (301, "depth-span")])
def test_exact_code_difference_support_boundary(module, difference, expected):
    frame = np.full(256000, 20000, "<u2")
    pixel = 200 * 640 + 320
    frame[pixel + 641] += difference
    assert module.image_directions(frame.tobytes(), np.array([pixel]))[0]["status"] == expected


@pytest.mark.parametrize(
    "pixels",
    [
        np.array([]),
        np.array([1.0]),
        np.array([True]),
        np.array([-1]),
        np.array([256000]),
        np.zeros((2, 2), dtype=int),
        np.zeros(3601, dtype=int),
    ],
)
def test_bad_pixel_rosters_refused(module, pixels):
    with pytest.raises(ValueError):
        module.image_directions(bytes(512000), pixels)


@pytest.mark.parametrize("raw", [b"", bytes(511998), bytes(512002), bytearray(512000)])
def test_bad_raw_refused(module, raw):
    with pytest.raises(ValueError):
        module.image_directions(raw, np.array([12345]))
    with pytest.raises(ValueError):
        module.original_pixels(raw, ((1.0, 2.0, 100.0),))


def test_exact_centroids_and_first_contributor_tie(module):
    points = ((0.25, 0.25, 0.25), (1.25, 0.25, 0.25), (-0.25, 0.25, 0.25), (3.0, 3.0, 3.0))
    centers, indices = module.representatives(points)
    assert centers == ((0.75, 0.25, 0.25), points[2], points[3])
    assert indices == (0, 2, 3)
    assert module.representatives((points[0], points[0]))[1] == (0,)


@pytest.mark.parametrize(
    "points", [(), ((math.nan, 0.0, 1.0),), ((0.0, 1.0),), ((0.0, 0.0, 0.0),) * 3601]
)
def test_bad_clouds_refused(module, points):
    with pytest.raises(ValueError):
        module.representatives(points)
    with pytest.raises(ValueError):
        module.original_pixels(bytes(512000), points)


@pytest.mark.parametrize(
    "point",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (math.inf, 0.0, 1.0),
        (10000.0, 0.0, 100.0),
        (0.0, 10000.0, 100.0),
        (0.0, 0.0, 100.0),
    ],
)
def test_nonmeasured_points_refused(module, point):
    with pytest.raises(ValueError):
        module.original_pixels(np.full(256000, 20000, "<u2").tobytes(), (point,))


def test_arithmetic_guard_has_no_fallback(module, monkeypatch):
    monkeypatch.setattr(module.np, "cross", lambda a, b: np.zeros_like(a))
    raw = np.full(256000, 20000, "<u2").tobytes()
    row = module.image_directions(raw, np.array([200 * 640 + 320]))[0]
    assert row["status"] == "degenerate" and row["normal"] is None


def test_random_original_patches_scalar_oracle(module):
    frame = np.random.default_rng(726).integers(19970, 20030, (400, 640), dtype=np.uint16)
    pixels = np.array([v * 640 + u for v in (1, 73, 210, 398) for u in (1, 51, 320, 638)])
    raw = frame.astype("<u2").tobytes()
    for row in module.image_directions(raw, pixels):
        n, lengths = scalar_normal(frame, row["pixel"] % 640, row["pixel"] // 640)
        assert row["normal"] == pytest.approx(n, abs=1e-15)
        assert row["tangent_lengths"] == pytest.approx(lengths, abs=1e-15)
    assert raw == frame.astype("<u2").tobytes()


@pytest.mark.parametrize(
    "b,expected",
    [((0.0, 0.0, 1.0), 0.0), ((0.0, 0.0, -2.0), 0.0), ((1.0, 0.0, 0.0), 90.0), (None, None)],
)
def test_sign_invariant_angle(module, b, expected):
    assert module.unoriented_angle((0.0, 0.0, 1.0), b) == expected


@pytest.mark.parametrize("b", [(0.0, 0.0, 0.0), (math.nan, 0.0, 1.0), (1.0, 2.0)])
def test_bad_normal_comparison_refused(module, b):
    with pytest.raises(ValueError):
        module.unoriented_angle((0.0, 0.0, 1.0), b)
