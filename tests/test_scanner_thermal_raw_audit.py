"""The fixed initial footprint includes small/fringe returns and preserves shifts."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_thermal_raw_audit")


def test_fixed_footprint_keeps_outlier_that_largest_component_does_not(audit):
    initial = np.zeros((10, 30, 30), dtype=np.uint16)
    initial[:, 5:20, 5:20] = 20000
    initial[:, 25, 25] = 20000
    before = initial.copy()
    masks = audit.fixed_masks(initial)
    assert masks["whole_initial_return_footprint"].sum() == 226
    assert masks["largest_initial_component"].sum() == 225
    warm = initial.copy()
    warm[:, initial[0] > 0] += 40
    warm[:, 25, 25] = 0
    result = audit.compare(initial, warm, masks["whole_initial_return_footprint"])
    assert result["newly_persistently_lost_pixels"] == 1
    assert result["signed_depth_change_mm_p10_p50_p90"] == pytest.approx([0.2] * 3)
    np.testing.assert_array_equal(initial, before)


def test_no_initial_depth_is_not_a_stability_pass(audit):
    with pytest.raises(ValueError, match="No persistent"):
        audit.fixed_masks(np.zeros((10, 30, 30), dtype=np.uint16))
