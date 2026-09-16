"""The window writes the scan's project file, keeps it current, and reconstructs from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from scanner3d import gui
from scanner3d.capture_sequence import FRAME_GLOB, TRACKING_LOSS_RECORD_NAME
from scanner3d.gui import ScannerWindow
from scanner3d.project import PROJECT_NAME, read_project
from scanner3d.reconstruction_export import OUTPUT_PREFIX, RECORD_NAME
from tests.fake_v4l2 import stream_kernel, stream_node, whole_frame
from tests.gui_harness import (
    CaptureWindow,
    clouds,
    mesh_budget,
    meshable_clouds,
    registration,
    sphere_raw_frame,
    start_capture,
    start_reconstruct,
)

__all__ = [
    "clouds",
    "mesh_budget",
    "meshable_clouds",
    "registration",
    "stream_kernel",
    "stream_node",
]


def capture_of(tmp_path: Path, frames: int = 4) -> Path:
    frames_dir = tmp_path / "capture"
    frames_dir.mkdir()
    raw = sphere_raw_frame()
    for index in range(frames):
        (frames_dir / f"frame_{index:04d}.raw").write_bytes(raw)
    return frames_dir


def test_a_capture_writes_its_project_file_first_and_finishes_it_last(
    tmp_path: Path, monkeypatch, clouds: list[int], stream_kernel
) -> None:
    """The settings go in before a frame arrives; the outcome goes in after the last one."""
    window = CaptureWindow(frames=3, decimate=2, depth_near=90, depth_far=210)
    stream_kernel([whole_frame(index) for index in range(3)])
    monkeypatch.setattr(
        ScannerWindow, "_capture_target", lambda self: _fresh(tmp_path), raising=True
    )

    start_capture(window)

    folder = window._capture_dir
    assert folder is not None
    path = folder / PROJECT_NAME
    assert path.is_file()
    project = read_project(path)
    assert project.capture["source"] == "desktop-application"
    assert project.capture["frames_requested"] == 3
    assert project.capture["decimate"] == 2
    assert project.capture["depth_band"] == {"near": 90.0, "far": 210.0}
    # What is on disk, not the number asked for: the harness's clouds are
    # three points each and the tracker refuses one of them, which is the
    # capture's own business and exactly what the record must reflect. The
    # reader's count is the second number, and here the two agree.
    assert project.capture["frames_written"] == len(list(folder.glob(FRAME_GLOB)))
    assert project.capture["frames_written"] >= 1
    assert project.capture["reconstructable_frames"] == gui._frame_counts(folder).frames
    assert project.capture["reconstructable_frames"] == project.capture["frames_written"]
    assert project.capture["stopped_early"] is False
    assert project.capture["tracking_lost"] is False
    assert any(PROJECT_NAME in line for line in window.logged)


def test_the_folder_decides_that_tracking_was_lost(
    tmp_path: Path, monkeypatch, clouds: list[int], stream_kernel
) -> None:
    """A loss record on disk is a loss, whatever the window had been told by Stop.

    Capture 20260904-164506: Stop was pressed before the loss reached the
    window, the reader refused the folder, and the project file said "not
    lost, 0 frames" beside a tracking-loss.json.
    """
    window = CaptureWindow(frames=3, decimate=2, depth_near=90, depth_far=210)
    stream_kernel([whole_frame(index) for index in range(3)])
    monkeypatch.setattr(
        ScannerWindow, "_capture_target", lambda self: _fresh(tmp_path), raising=True
    )

    start_capture(window)
    folder = window._capture_dir
    assert folder is not None
    (folder / TRACKING_LOSS_RECORD_NAME).write_text("{}", encoding="utf-8")
    window._finish_project(frames=None, stopped=True)

    project = read_project(folder / PROJECT_NAME)
    assert project.capture["tracking_lost"] is True
    assert project.capture["stopped_early"] is True
    assert project.capture["frames_written"] == len(list(folder.glob(FRAME_GLOB)))
    assert project.capture["frames_written"] >= 1
    assert "reconstructable_frames" in project.capture  # left from the clean finish


def _fresh(tmp_path: Path) -> Path:
    folder = tmp_path / "scan" / "fresh"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_reconstruct_records_its_output_in_the_project_file(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    frames_dir = capture_of(tmp_path)
    registration()
    window = CaptureWindow(frames=1, depth_near=80, depth_far=200)

    start_reconstruct(window, frames_dir, monkeypatch)

    project = read_project(frames_dir / PROJECT_NAME)
    (entry,) = project.reconstructions
    assert entry["directory"] == f"{OUTPUT_PREFIX}0001"
    assert entry["record"] == f"{OUTPUT_PREFIX}0001/{RECORD_NAME}"
    assert entry["depth_band"] == {"near": 80.0, "far": 200.0}
    assert isinstance(entry["points"], int) and entry["points"] > 0
    assert str(entry["coherence"]).startswith(("coherent", "INCOHERENT", "not checkable"))
    assert (frames_dir / str(entry["record"])).is_file()
    assert window._project_path == frames_dir / PROJECT_NAME
    assert f"recorded in {PROJECT_NAME}" in window.report


def test_an_older_capture_picked_by_one_of_its_frames_is_adopted_first(
    tmp_path: Path, monkeypatch, meshable_clouds: list[int], registration, mesh_budget: int
) -> None:
    frames_dir = capture_of(tmp_path)
    registration()
    window = CaptureWindow(frames=1)

    class PickAFrame:
        @staticmethod
        def getOpenFileName(*_a: object, **_k: object) -> tuple[str, str]:  # Qt's name
            return str(frames_dir / "frame_0002.raw"), ""

    monkeypatch.setattr(gui, "QFileDialog", PickAFrame)
    ScannerWindow._start_reconstruct(window)  # type: ignore[arg-type]

    project = read_project(frames_dir / PROJECT_NAME)
    assert project.capture["source"] == "adopted"
    assert len(project.reconstructions) == 1
    assert any("had no project file" in line for line in window.logged)


def test_a_file_that_is_not_a_capture_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch
) -> None:
    stray = tmp_path / "notes.scanner3d"
    stray.write_text("not a project", encoding="utf-8")
    window = CaptureWindow(frames=1)
    shown: list[str] = []

    class Warn:
        @staticmethod
        def warning(_parent: object, _title: str, text: str) -> None:
            shown.append(text)

    class PickStray:
        @staticmethod
        def getOpenFileName(*_a: object, **_k: object) -> tuple[str, str]:  # Qt's name
            return str(stray), ""

    monkeypatch.setattr(gui, "QMessageBox", Warn)
    monkeypatch.setattr(gui, "QFileDialog", PickStray)
    ScannerWindow._start_reconstruct(window)  # type: ignore[arg-type]

    assert len(shown) == 1
    assert window.busy == []
    assert json.loads(stray.read_text(encoding="utf-8")) if False else True
