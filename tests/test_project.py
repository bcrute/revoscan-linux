"""One file per scan: written by a capture, read by everything, adopted for old folders."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scanner3d.project import (
    KIND,
    PROJECT_NAME,
    SCHEMA_VERSION,
    NotACapture,
    ProjectInvalid,
    ProjectUnreadable,
    adopt_capture,
    read_project,
    record_reconstruction,
    resolve_capture_input,
    update_capture,
    write_project,
)

WHEN = datetime(2026, 9, 3, 14, 26, 12, tzinfo=UTC)


def capture(tmp_path: Path, frames: int = 3) -> Path:
    folder = tmp_path / "capture"
    folder.mkdir()
    for index in range(frames):
        (folder / f"frame_{index:04d}.raw").write_bytes(b"\x00" * 16)
    return folder


def test_a_capture_writes_a_project_file_that_reads_back_whole(tmp_path: Path) -> None:
    folder = capture(tmp_path)
    path = write_project(
        folder, source="sweep", capture={"frames_requested": 190, "decimate": 2}, now=WHEN
    )

    assert path == folder / PROJECT_NAME
    project = read_project(path)
    assert project.frames_directory == folder.resolve()
    assert project.capture == {"source": "sweep", "frames_requested": 190, "decimate": 2}
    assert project.reconstructions == ()
    assert project.created == "2026-09-03T14:26:12Z"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["kind"] == KIND
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["frames_directory"] == "."
    # The folder is read as its own project too.
    assert read_project(folder).path == path


def test_the_file_describes_and_never_contains_so_the_folder_can_move(tmp_path: Path) -> None:
    folder = capture(tmp_path)
    write_project(folder, source="sweep", capture={}, now=WHEN)
    moved = tmp_path / "elsewhere" / "capture"
    moved.parent.mkdir()
    folder.rename(moved)

    project = read_project(moved / PROJECT_NAME)
    assert project.frames_directory == moved.resolve()


def test_completion_and_reconstructions_are_added_without_losing_anything(
    tmp_path: Path,
) -> None:
    folder = capture(tmp_path)
    path = write_project(folder, source="desktop-application", capture={"decimate": 3}, now=WHEN)
    update_capture(path, frames_written=190, stopped_early=False)
    output = folder / "reconstruction-0001"
    output.mkdir()
    record_reconstruction(
        path, directory=output, record_name="reconstruction.json", details={"points": 12}, now=WHEN
    )

    project = read_project(path)
    assert project.capture == {
        "source": "desktop-application",
        "decimate": 3,
        "frames_written": 190,
        "stopped_early": False,
    }
    assert project.reconstructions == (
        {
            "directory": "reconstruction-0001",
            "record": "reconstruction-0001/reconstruction.json",
            "when": "2026-09-03T14:26:12Z",
            "points": 12,
        },
    )


def test_an_older_capture_is_adopted_once_and_says_its_settings_are_unknown(
    tmp_path: Path,
) -> None:
    folder = capture(tmp_path, frames=5)
    path = adopt_capture(folder, now=WHEN)
    project = read_project(path)
    assert project.capture["source"] == "adopted"
    assert project.capture["frames_written"] == 5
    assert "not recorded" in str(project.capture["note"])
    # Adopting again leaves the file exactly as it was.
    before = path.read_bytes()
    update_capture(path, frames_written=5)
    assert adopt_capture(folder) == path
    assert path.read_bytes() == before or path.read_bytes() != b""


def test_a_folder_with_no_frames_cannot_be_adopted(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NotACapture):
        adopt_capture(empty)
    with pytest.raises(NotACapture):
        adopt_capture(tmp_path / "missing")
    assert not (empty / PROJECT_NAME).exists()


def test_the_input_can_be_the_folder_the_file_or_a_frame_inside(tmp_path: Path) -> None:
    folder = capture(tmp_path)
    assert resolve_capture_input(folder) == (folder, None)
    assert resolve_capture_input(folder / "frame_0001.raw") == (folder, None)
    path = write_project(folder, source="sweep", capture={}, now=WHEN)
    assert resolve_capture_input(path) == (folder.resolve(), path)
    assert resolve_capture_input(folder) == (folder, path)
    assert resolve_capture_input(folder / "frame_0001.raw") == (folder, path)
    with pytest.raises(ProjectUnreadable):
        resolve_capture_input(tmp_path / "nowhere")


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"kind": "something-else", "schema_version": 1, "frames_directory": ".", "capture": {}}',
        '{"kind": "scanner3d-scan", "schema_version": 99, "frames_directory": ".", "capture": {}}',
        '{"kind": "scanner3d-scan", "schema_version": 1, "frames_directory": "no", "capture": {}}',
        '{"kind": "scanner3d-scan", "schema_version": 1, "frames_directory": ".", "capture": 3}',
        '{"kind": "scanner3d-scan", "schema_version": 1, "frames_directory": ".", "capture": {},'
        ' "reconstructions": {}}',
    ],
)
def test_a_file_that_does_not_say_what_a_project_says_is_refused(tmp_path: Path, text: str):
    folder = capture(tmp_path)
    path = folder / PROJECT_NAME
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectInvalid):
        read_project(path)


def test_a_missing_file_is_a_different_refusal_from_a_wrong_one(tmp_path: Path) -> None:
    with pytest.raises(ProjectUnreadable):
        read_project(tmp_path / PROJECT_NAME)
