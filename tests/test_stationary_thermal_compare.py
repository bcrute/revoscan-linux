"""Warm comparison must preserve depth shifts and losses at fixed initial pixels."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def thermal(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("stationary_thermal_compare")


def test_warm_shift_is_not_registered_away_and_missing_pixel_is_counted(thermal):
    raw = np.zeros((40, 20, 20), dtype=np.uint16)
    raw[:, 5:15, 5:15] = 20000
    mask = raw[0] > 0
    warm = raw.copy()
    warm[:, mask] += 40  # Nominal +0.2 mm.
    warm[:, 5, 5] = 0
    result = thermal.compare(raw, warm, mask)
    assert result["initial_stable_pixels"] == 100
    assert result["mutually_observed_stable_pixels"] == 99
    assert result["newly_persistently_lost_pixels"] == 1
    assert result["newly_persistently_lost_fraction"] == 0.01
    assert result["absolute_depth_change_mm_p50_p90_p99"] == pytest.approx([0.2] * 3)
    assert result["signed_depth_change_mm_p10_p50_p90"] == pytest.approx([0.2] * 3)
    assert result["coverage"]["overall"]["warm_relative_to_initial"] == 0.99
    assert result["coverage"]["interior"]["warm_relative_to_initial"] == 1.0


def test_warm_comparison_refuses_no_mutual_depth(thermal):
    raw = np.full((40, 20, 20), 20000, dtype=np.uint16)
    with pytest.raises(ValueError, match="Insufficient"):
        thermal.compare(raw, np.zeros_like(raw), np.ones((20, 20), dtype=bool))
