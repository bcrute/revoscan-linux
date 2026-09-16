"""Synthetic support/refusal checks and independent scalar/SVD plane arithmetic."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

from scanner3d.cli import DEPTH_SCALE_METRES
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.simulate import FACTORY_INTRINSICS

PIXEL = 200 * 640 + 310


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_organized_plane_review")


@pytest.mark.parametrize("radius", [2, 5, 10])
@pytest.mark.parametrize("slope", [(0.0, 0.0), (0.2, 0.1), (-0.3, 0.15)])
def test_quantized_plane_and_independent_svd(module, radius, slope):
    fx, fy, cx, cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    v, u = np.indices((400, 640))
    z = 200 / (1 - slope[0] * (u - cx) / fx - slope[1] * (v - cy) / fy)
    frame = np.rint(z / DEPTH_SCALE_METRES).astype("<u2")
    raw = frame.tobytes()
    indices = np.array([PIXEL, PIXEL + 15, PIXEL + 640])
    original_indices = indices.copy()
    rows = module.plane_directions(raw, indices, radius=radius)
    for row in rows:
        y, x = divmod(row["pixel"], 640)
        points = []
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                depth = int(frame[yy, xx]) * DEPTH_SCALE_METRES
                if (
                    60 < depth < 300
                    and abs(int(frame[yy, xx]) - int(frame[y, x])) * DEPTH_SCALE_METRES <= 3
                ):
                    points.append(((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth))
        center = [math.fsum(p[i] for p in points) / len(points) for i in range(3)]
        delta = np.array([[p[i] - center[i] for i in range(3)] for p in points])
        _, singular, vectors = np.linalg.svd(delta, full_matrices=False)
        normal = vectors[-1]
        if normal @ center > 0:
            normal = -normal
        assert row["status"] == "usable"
        assert row["support_count"] == len(points)
        assert row["centroid"] == pytest.approx(center, abs=1e-11)
        assert row["normal"] == pytest.approx(normal, abs=1e-10)
        assert row["eigenvalues"] == pytest.approx(
            (singular[::-1] ** 2 / len(points)).tolist(), abs=1e-12
        )
    assert raw == frame.tobytes() and np.array_equal(indices, original_indices)


@pytest.mark.parametrize("radius", [2, 5, 10])
@pytest.mark.parametrize("pixel", [0, 639, 255999, 640, 200 * 640 + 639])
def test_border(module, radius, pixel):
    row = module.plane_directions(
        np.full(256000, 40000, "<u2").tobytes(), np.array([pixel]), radius=radius
    )[0]
    assert row["status"] == "border" and row["normal"] is None


@pytest.mark.parametrize("code", [0, 12000, 60001, 65535])
def test_invalid_center(module, code):
    frame = np.full(256000, 40000, "<u2")
    frame[PIXEL] = code
    row = module.plane_directions(frame.tobytes(), np.array([PIXEL]), radius=2)[0]
    assert row["status"] == "center-missing-or-band" and row["normal"] is None


@pytest.mark.parametrize(
    "missing,expected", [(0, "usable"), (1, "usable"), (6, "usable"), (7, "insufficient-support")]
)
@pytest.mark.parametrize("code", [0, 40601])
def test_exact_support_boundary(module, missing, expected, code):
    frame = np.full(256000, 40000, "<u2")
    neighbors = [PIXEL + dy * 640 + dx for dy in range(-2, 3) for dx in range(-2, 3) if dy or dx]
    frame[neighbors[:missing]] = code
    row = module.plane_directions(frame.tobytes(), np.array([PIXEL]), radius=2)[0]
    assert row["status"] == expected and row["support_count"] == 25 - missing


@pytest.mark.parametrize("radius", [2, 5, 10])
def test_nonplanar_support(module, radius):
    y, x = np.indices((400, 640))
    frame = (40000 + 100 * (2 * ((x + y) % 2) - 1)).astype("<u2")
    row = module.plane_directions(frame.tobytes(), np.array([PIXEL]), radius=radius)[0]
    assert row["status"] == "nonplanar" and row["normal"] is None
    assert row["perpendicular_rms"] > 0.25


@pytest.mark.parametrize("difference,count", [(600, 25), (601, 24), (-600, 25), (-601, 24)])
def test_code_distance_boundary(module, difference, count):
    frame = np.full(256000, 40000, "<u2")
    frame[PIXEL + 1] = 40000 + difference
    row = module.plane_directions(frame.tobytes(), np.array([PIXEL]), radius=2)[0]
    assert row["support_count"] == count


@pytest.mark.parametrize(
    "values,status",
    [
        ([0.0625, 1.0, 2.0], "usable"),
        ([0.0625 + 1e-12, 1.0, 2.0], "nonplanar"),
        ([0.0, 0.0, 2.0], "degenerate"),
        ([0.0, 2e-12, 2.0], "degenerate"),
        ([0.0, 2.001e-12, 2.0], "usable"),
    ],
)
def test_explicit_eigenvalue_boundaries(module, monkeypatch, values, status):
    def fixed_eigh(covariance):
        return np.tile(values, (len(covariance), 1)), np.tile(np.eye(3), (len(covariance), 1, 1))

    monkeypatch.setattr(module.np.linalg, "eigh", fixed_eigh)
    row = module.plane_directions(
        np.full(256000, 40000, "<u2").tobytes(), np.array([PIXEL]), radius=2
    )[0]
    assert row["status"] == status
    assert (row["normal"] is not None) == (status == "usable")


@pytest.mark.parametrize("radius", [-1, 0, 1, 3, 4, 6, 20, True, 2.0])
def test_bad_radius(module, radius):
    with pytest.raises(ValueError):
        module.plane_directions(bytes(512000), np.array([PIXEL]), radius=radius)


@pytest.mark.parametrize(
    "pixels",
    [
        [],
        [PIXEL],
        np.array([]),
        np.array([1.0]),
        np.array([-1]),
        np.array([256000]),
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([[PIXEL]]),
        np.zeros(3601, dtype=np.int64),
        np.array([True]),
    ],
)
def test_bad_pixels(module, pixels):
    with pytest.raises(ValueError):
        module.plane_directions(bytes(512000), pixels, radius=2)


@pytest.mark.parametrize("raw", [b"", bytes(511999), bytes(512001), bytearray(512000), "x"])
def test_bad_raw(module, raw):
    with pytest.raises(ValueError):
        module.plane_directions(raw, np.array([PIXEL]), radius=2)


def test_maximum_batch_and_repeated_pixels_are_deterministic(module):
    raw = np.full(256000, 40000, "<u2").tobytes()
    single = module.plane_directions(raw, np.array([PIXEL]), radius=10)[0]
    rows = module.plane_directions(raw, np.full(3600, PIXEL, dtype=np.int64), radius=10)
    assert rows == [single] * 3600
