"""Synthetic scene changes, with no scanner access."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_scene_audit")


def test_scene_audit_keeps_addition_occlusion_and_depth_change_separate(audit):
    reference = np.full((40, 40), np.nan)
    reference[10:20, 10:20] = 100
    frequency = np.isfinite(reference).astype(float)
    raw = np.zeros((40, 40), dtype=np.uint16)
    raw[10:20, 10:20] = 20000
    initial = audit.measurements(raw, reference, frequency)
    assert initial["valid_60_300_mm_pixels"] == 100
    assert all(
        value == 0
        for key, value in initial.items()
        if key not in ("valid_60_300_mm_pixels", "nonzero_pixels")
    )
    raw[10:12, 10:12] = 0
    raw[12:14, 10:12] = 18000
    raw[14:16, 10:12] = 22000
    raw[30:33, 30:33] = 20000
    changed = audit.measurements(raw, reference, frequency)
    assert changed["missing_stable_reference_pixels"] == 4
    assert changed["closer_than_reference_5mm_pixels"] == 4
    assert changed["farther_than_reference_5mm_pixels"] == 4
    assert changed["added_outside_reference_5px_pixels"] == 9
    assert changed["largest_added_component_pixels"] == 9


def test_scene_audit_refuses_short_raw_and_insufficient_reference(audit, tmp_path):
    raw = tmp_path / "frame_0000.raw"
    raw.write_bytes(b"\0\0")
    with pytest.raises(ValueError, match="Incomplete"):
        audit.read_raw(raw)
    with pytest.raises(ValueError, match=r"24\.\.80"):
        audit.run(tmp_path, [tmp_path], tmp_path / "out")
    assert not (tmp_path / "out").exists()
