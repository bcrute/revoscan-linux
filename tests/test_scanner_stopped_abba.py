"""Stopped-scene comparison preserves fixed pixels, bad windows and raw provenance."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def abba(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_stopped_abba")


def test_common_window_is_first_sixty_seconds_after_settling(abba):
    stamps = [100 + i * 0.2 for i in range(640)]
    indices = abba.settled_window(stamps)
    assert indices == list(range(12, 312))
    assert stamps[indices[0]] - stamps[0] >= 1
    assert stamps[indices[-1]] < stamps[indices[0]] + 60


@pytest.mark.parametrize("problem", ["short", "duplicate", "nan", "reverse"])
def test_bad_or_incomplete_windows_are_not_replaced_by_better_windows(abba, problem):
    stamps = [i * 0.2 for i in range(640)]
    if problem == "short":
        stamps = stamps[:310]
    elif problem == "duplicate":
        stamps[40] = stamps[39]
    elif problem == "nan":
        stamps[40] = float("nan")
    else:
        stamps.reverse()
    with pytest.raises(ValueError):
        abba.settled_window(stamps)


def test_region_reports_loss_and_signed_shift_without_registration(abba):
    a = np.full((20, 20), 100.0)
    fa = np.ones_like(a)
    mad = np.zeros_like(a)
    b = a + 0.2
    fb = fa.copy()
    b[5, 5], fb[5, 5] = np.nan, 0
    mask = np.zeros_like(a, dtype=bool)
    mask[5:15, 5:15] = True
    result = abba.region_comparison((a, fa, mad), (b, fb, mad), mask)
    assert result["stable_pixels"] == 100
    assert result["mutual_stable_pixels"] == 99
    assert result["persistently_lost_stable_pixels"] == 1
    assert result["candidate_return_fraction"] == 0.99
    assert result["signed_depth_change_p10_p50_p90"] == pytest.approx([0.2] * 3)
    assert result["absolute_depth_change_p50_p90_p99"] == pytest.approx([0.2] * 3)


def test_empty_tile_is_inconclusive_and_not_omitted(abba):
    a = np.full((20, 20), np.nan)
    frequency = np.zeros_like(a)
    result = abba.region_comparison(
        (a, frequency, a), (a, frequency, a), np.ones_like(a, dtype=bool)
    )
    assert result["comparison_checkable"] is False
    assert result["mask_pixels"] == 400 and result["stable_pixels"] == 0
    assert result["absolute_depth_change_p50_p90_p99"] is None
    frequency[3:17, 3:17] = 1
    masks = abba.regions(frequency)
    assert len([name for name in masks if name.startswith("tile_")]) == 16
    assert np.stack([mask for name, mask in masks.items() if name.startswith("tile_")]).sum() == 400
    assert masks["core"].sum() < masks["component"].sum()
    assert not (masks["edge"] & masks["core"]).any()


@pytest.fixture
def source(abba, tmp_path, monkeypatch):
    raw = np.full((400, 640), 20000, dtype="<u2").tobytes()
    records = [
        dict(
            path=f"frame_{i:04d}.raw",
            delivery_index=i,
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            monotonic_s=i * 0.2,
        )
        for i in range(640)
    ]
    steps = []
    for prefix, timing in (
        ("initial", 20000),
        ("configured", 14000),
        ("after_stop", 14000),
        ("verified_post_stop_restore", 20000),
    ):
        steps.extend(
            [
                dict(
                    name=prefix + "_timing",
                    status="completed",
                    result=[{**abba.BASELINE, "frame_time": timing, "exposure_time": timing}, None],
                ),
                dict(name=prefix + "_gain", status="completed", result=[24, None]),
            ]
        )
    report = dict(
        status="completed",
        restoration_verified=True,
        first_transport_fault=None,
        serial=abba.SERIAL,
        mode="settled-post",
        motion_timing_us=14000,
        capture_frame_budget=640,
        usb_idle=dict(topology_stable=True),
        usb_before_readback=dict(settled_for_readback=True),
        steps=steps,
        capture=dict(summary=dict(kept=640, delivered=640, short_frames=0, dropped_by_sequence=0)),
        temperatures=[dict(cpu_c=60, npu_c=59)],
    )
    progress = dict(capture_error=None, frames=records)
    real_read = Path.read_bytes

    def read(path):
        if path.name == "report.json":
            return json.dumps(report).encode()
        if path.name == "capture-progress.json":
            return json.dumps(progress).encode()
        if path.suffix == ".raw":
            return raw
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(abba, "framing_measurements", lambda _raw: {})
    return tmp_path, report, progress


def test_loader_records_exact_selection_and_checks_unused_tail(abba, source):
    path, _report, progress = source
    raw, info = abba.load_stage(path, 14000)
    assert len(raw) == 300 and info["comparison_source_indices"] == list(range(12, 312))
    assert info["all_raw_hashes_verified"] == 640
    assert info["comparison_observed_fps"] == pytest.approx(5)
    progress["frames"][-1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        abba.load_stage(path, 14000)


@pytest.mark.parametrize("problem", ["restore", "gain", "fault", "gap", "escape"])
def test_unrestored_faulted_or_mislabeled_captures_are_refused(abba, source, problem):
    path, report, progress = source
    if problem == "restore":
        report["restoration_verified"] = False
    elif problem == "gain":
        report["steps"][3]["result"][0] = 16
    elif problem == "fault":
        report["first_transport_fault"] = {"errno": 19}
    elif problem == "gap":
        report["capture"]["summary"]["dropped_by_sequence"] = 1
    else:
        progress["frames"][0]["path"] = "../frame_0000.raw"
    with pytest.raises(ValueError):
        abba.load_stage(path, 14000)


def test_flagged_framing_cannot_become_a_stationary_comparison(abba, tmp_path):
    (tmp_path / "quality-validity.json").write_text(
        json.dumps(dict(quality_qualification_allowed=False, reason="base at image edge"))
    )
    with pytest.raises(ValueError, match="Flagged scene"):
        abba.load_stage(tmp_path, 20000)
