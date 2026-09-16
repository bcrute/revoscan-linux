"""S3 ordering is unique, spatially regular and restores the S2 implementation."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from scanner3d.errors import ScannerError


def test_spatial_order_is_unique_regular_and_not_left_installed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    grid = importlib.import_module("scanner_pixel_grid")
    original = grid.scanner_pixel_rank.ranks
    indices = np.arange(256000)
    ranks = grid.ranks(indices)
    assert len(np.unique(ranks)) == len(indices)
    selected = np.argsort(ranks)[:900]
    xy = np.stack(np.divmod(selected, 640), axis=1)
    distances = cKDTree(xy).query(xy, k=2)[0][:, 1]
    assert np.min(distances) >= 8
    grid.prepare_frame(bytes(512000), target_points=900)
    assert grid.scanner_pixel_rank.ranks is original
    with pytest.raises(ScannerError):
        grid.prepare_frame(b"short", target_points=900)
    assert grid.scanner_pixel_rank.ranks is original
