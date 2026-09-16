"""Fixed native-pixel model ranks preserve provenance independently of list size."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pixel(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_model_pixel_sampling")


def point(pixel, u, v, z=100.0):
    fx, fy, cx, cy = pixel.ensure_intrinsics_usable(pixel.FACTORY_INTRINSICS)
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def cloud(pixel, size=1800):
    return tuple(point(pixel, i % 90, i // 90, 100.0 + (i % 5)) for i in range(size))


def test_native_projection_recovers_pixels_at_image_edges_and_varied_depth(pixel):
    for z in (0.001, 60.001, 123.45, 299.999, 10000.0):
        coordinates = [(0, 0), (639, 0), (0, 399), (639, 399), (120, 225)]
        points = tuple(point(pixel, u, v, z) for u, v in coordinates)
        assert pixel.pixels(points, capacity=900) == tuple(v * 640 + u for u, v in coordinates)


@pytest.mark.parametrize("size", [0, 1, 224, 225, 226, 899, 900, 901, 1799, 1800])
def test_exact_count_original_order_and_fixed_grid_keys(pixel, size):
    points = cloud(pixel, size)
    native = pixel.pixels(points, capacity=1800)
    keys = pixel.ranks(np.asarray(native, dtype=np.int64))
    for limit in (1, 225, 600, 900, 1800):
        actual = pixel.indices(points, limit, capacity=1800)
        assert len(actual) == min(size, limit)
        assert actual == tuple(sorted(set(actual)))
        expected = sorted(range(size), key=lambda i: int(keys[i]))[:limit]
        assert set(actual) == set(expected)
        assert all(points[i] == point(pixel, native[i] % 640, native[i] // 640, points[i][2])
                   for i in actual)


def test_pixel_selection_is_depth_and_input_order_invariant(pixel):
    points = cloud(pixel)
    selected = pixel.indices(points, 900, capacity=1800)
    native = pixel.pixels(points, capacity=1800)
    expected = {native[i] for i in selected}
    changed = tuple(point(pixel, native[i] % 640, native[i] // 640, 80.0 + (i % 83))
                    for i in reversed(range(len(points))))
    actual = pixel.indices(changed, 900, capacity=1800)
    reordered = pixel.pixels(changed, capacity=1800)
    assert {reordered[i] for i in actual} == expected


def test_losing_one_native_pixel_can_replace_only_that_one_selected_pixel(pixel):
    points = cloud(pixel)
    before = pixel.pixels(points, capacity=1800)
    selected = pixel.indices(points, 900, capacity=1800)
    original = {before[i] for i in selected}
    for removed in (0, 77, 899, 1777, 1799):
        changed = points[:removed] + points[removed + 1:]
        native = pixel.pixels(changed, capacity=1800)
        after = {native[i] for i in pixel.indices(changed, 900, capacity=1800)}
        assert len(after) == 900 and len(original & after) >= 899


@pytest.mark.parametrize("failure", ["duplicate", "negative", "nan", "fractional", "outside"])
def test_invalid_native_pixel_provenance_refuses(pixel, failure):
    invalid = {
        "duplicate": (point(pixel, 1, 2), point(pixel, 1, 2, 200)),
        "negative": ((0.0, 0.0, -1.0),),
        "nan": ((float("nan"), 0.0, 100.0),),
        "fractional": (point(pixel, 1.01, 2),),
        "outside": (point(pixel, 640, 2),),
    }[failure]
    with pytest.raises(ValueError):
        pixel.indices(invalid, 900, capacity=1800)


@pytest.mark.parametrize("overrides", [dict(capacity=901), dict(capacity=True),
                                     dict(limit=0), dict(limit=1801), dict(limit=True)])
def test_closed_capacity_and_integer_budget_refuse(pixel, overrides):
    with pytest.raises(ValueError):
        pixel.indices(cloud(pixel, 900), **(dict(limit=900, capacity=900) | overrides))


def test_native_cloud_cannot_silently_exceed_its_declared_cap(pixel):
    with pytest.raises(ValueError):
        pixel.indices(cloud(pixel, 901), 900, capacity=900)
