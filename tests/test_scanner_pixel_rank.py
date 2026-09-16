"""Fixed pixel ranks do not resample a whole image after one return disappears."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from scanner3d.errors import ScannerError
from scanner3d.preparation import DepthBand, prepare_frame


@pytest.fixture
def ranker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_pixel_rank")


def test_one_missing_return_does_not_shift_every_selected_pixel(ranker):
    valid = np.arange(256000)
    a = ranker.select_pixels(valid, 900)
    b = ranker.select_pixels(valid[1:], 900)
    assert len(a) == len(valid[:: (len(valid) // 900)])
    assert len(b) == len(valid[1:][:: (len(valid[1:]) // 900)])
    assert len(np.intersect1d(a, b)) >= min(len(a), len(b)) - 1
    assert np.all(np.diff(a) > 0)
    assert len(np.unique(ranker.ranks(valid))) == len(valid)
    np.testing.assert_array_equal(a, ranker.select_pixels(valid, 900))


def test_when_every_point_fits_only_the_selection_changes_nothing(ranker):
    raw = np.zeros((400, 640), dtype=np.uint16)
    raw[100:120, 100:120] = 20000
    payload = raw.tobytes()
    assert ranker.prepare_frame(payload, target_points=900, depth_band=DepthBand(60, 300)) == (
        prepare_frame(payload, target_points=900, depth_band=DepthBand(60, 300))
    )
    assert payload == raw.tobytes()


def test_empty_raw_is_empty_and_short_raw_refused(ranker):
    assert not ranker.prepare_frame(bytes(512000), target_points=900).points
    with pytest.raises(ScannerError):
        ranker.prepare_frame(bytes(511998), target_points=900)
