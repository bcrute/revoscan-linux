"""Fixed source regions preserve their overlap and never edit their input."""

import importlib
from pathlib import Path

import pytest


def test_regions_cover_source_and_use_perspective_coordinates(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    module = importlib.import_module("scanner_region_audit")
    points = [(x * z, y * z, z) for x in range(4) for y in range(4) for z in (100.0, 200.0)]
    before = list(points)
    groups = module.regions(points)
    assert set(groups["left"]) | set(groups["right"]) == set(points)
    assert set(groups["top"]) | set(groups["bottom"]) == set(points)
    assert set(groups["left"]) & set(groups["right"])
    assert points == before
    assert (400.0, 400.0, 200.0) in groups["left"]
    assert (600.0, 400.0, 200.0) not in groups["left"]
    with pytest.raises(ValueError):
        module.regions([(0.0, 0.0, 0.0)])
