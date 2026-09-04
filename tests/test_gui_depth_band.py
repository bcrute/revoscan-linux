"""The depth band as the window's own parameter: two controls, one interval, everywhere.

The interval the operator types reaches every frame the window prepares - the
live tracking of a capture and the reconstruct job alike - and the field the
export fuses; Remember records it in the config file the command line reads,
and the controls open on whatever that file states. A pair of spin boxes
cannot express an interval's order, so an inverted pair is refused before a
capture directory exists rather than clipped to nothing. The harness sphere sits at
100-140, so the 80-200 band below holds all of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.config import config_path, configured_depth_band
from scanner3d.preparation import DepthBand, DepthBandUnusable
from scanner3d.reconstruction_export import OUTPUT_PREFIX, RECORD_NAME
from tests.gui_harness import (
    CaptureWindow,
    mesh_budget,
    meshable_clouds,
    registration,
    sphere_raw_frame,
    start_reconstruct,
)

__all__ = ["mesh_budget", "meshable_clouds", "registration"]


def capture_of(tmp_path: Path, frames: int = 4) -> Path:
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(frames):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    return frames_dir


def _spy_prepare(monkeypatch) -> list[object]:
    """Record the band every prepared frame was given, whatever stands in for preparation."""
    seen: list[object] = []
    inner = gui._prepare_frame

    def spy(raw: bytes, target_points: int, depth_band: DepthBand | None = None) -> object:
        seen.append(depth_band)
        return inner(raw, target_points, depth_band)

    monkeypatch.setattr(gui, "_prepare_frame", spy)
    return seen


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    shown: list[str] = []

    class FakeMessageBox:
        @staticmethod
        def warning(_parent: object, _title: str, text: str) -> None:
            shown.append(text)

    monkeypatch.setattr(gui, "QMessageBox", FakeMessageBox)
    return shown


def test_the_band_typed_in_the_window_reaches_every_reconstructed_frame_and_the_field(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    registration()
    seen = _spy_prepare(monkeypatch)
    window = CaptureWindow(frames=1, depth_near=80, depth_far=200)

    start_reconstruct(window, frames_dir, monkeypatch)

    assert seen == [DepthBand(near=80.0, far=200.0)] * 4
    assert "Registering depths from 80-200 mm only." in window.logged
    (directory,) = [path for path in frames_dir.iterdir() if path.name.startswith(OUTPUT_PREFIX)]
    record = json.loads((directory / RECORD_NAME).read_bytes())
    depth = record["mesh"]["tsdf"]["integration"]["config"]["depth"]
    assert depth["min_depth"] == 80.0
    assert depth["max_depth"] == 200.0
    # And the verdict the images give is in the report, in the same words the
    # command line prints.
    assert "coherence: " in window.report


def test_the_window_opens_on_the_band_the_config_file_states() -> None:
    window = CaptureWindow(frames=1)
    window._load_depth_band()
    assert window._depth_band() == DepthBand.default()

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[registration]\ndepth_band = [130, 275]\n", encoding="utf-8")
    window._load_depth_band()
    assert window._depth_band() == DepthBand(near=130.0, far=275.0)


def test_remember_records_the_controls_in_the_config_file_and_says_so() -> None:
    window = CaptureWindow(frames=1, depth_near=130, depth_far=275)

    window._remember_depth_band()

    assert configured_depth_band() == (130.0, 275.0)
    assert any("depth band 130-275 mm recorded in" in line for line in window.logged)
    assert any("recorded in" in message for message in window.statusBar().messages)


def test_an_inverted_pair_is_refused_before_anything_starts(warnings: list[str]) -> None:
    window = CaptureWindow(frames=1, depth_near=275, depth_far=130)

    with pytest.raises(DepthBandUnusable):
        window._depth_band()
    window._remember_depth_band()
    assert len(warnings) == 1
    assert "positive, increasing interval" in warnings[0]
    assert configured_depth_band() is None
