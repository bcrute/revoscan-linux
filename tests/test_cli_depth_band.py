"""The registration depth band: stated on the command line, in the file, or not at all.

The band decides which part of the scene registers, so where it came from is
reported beside its value, a band the clip cannot apply is refused before a
frame is read, and the config command shows it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanner3d import preparation
from scanner3d.cli import main
from scanner3d.config import config_path
from scanner3d.errors import EXIT_CODES, DoctorStatus
from scanner3d.preparation import DEPTH_CLIP_FAR, DEPTH_CLIP_NEAR, DepthBand
from tests.test_cli_readiness import capture_of, scanning_registration


def _write_config(text: str) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _spy_prepare(monkeypatch) -> list[DepthBand | None]:
    seen: list[DepthBand | None] = []
    real = preparation.prepare_frame

    def spy(raw: bytes, *, target_points: int, depth_band: DepthBand | None = None):
        seen.append(depth_band)
        return real(raw, target_points=target_points, depth_band=depth_band)

    monkeypatch.setattr(preparation, "prepare_frame", spy)
    return seen


def test_the_band_from_the_command_line_reaches_every_frame_and_the_record(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    seen = _spy_prepare(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json", "--depth-band", "100", "200"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert seen == [DepthBand(near=100.0, far=200.0)] * 4
    assert payload["command"]["depth_band"] == {
        "near": 100.0,
        "far": 200.0,
        "source": "--depth-band",
    }
    # Four identical frames revisit nothing: the coherence check reports that it
    # could not run, which is not a pass.
    assert payload["coherence"]["period_frames"] is None
    assert payload["coherence"]["coherent"] is None


def test_a_band_stated_in_the_file_is_used_and_credited_to_the_file(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    seen = _spy_prepare(monkeypatch)
    path = _write_config("[registration]\ndepth_band = [90, 210]\n")

    code = main(["reconstruct", str(frames_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert seen == [DepthBand(near=90.0, far=210.0)] * 4
    assert payload["command"]["depth_band"] == {"near": 90.0, "far": 210.0, "source": str(path)}


def test_the_command_line_wins_over_the_file(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    seen = _spy_prepare(monkeypatch)
    _write_config("[registration]\ndepth_band = [90, 210]\n")

    code = main(["reconstruct", str(frames_dir), "--json", "--depth-band", "100", "200"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert seen == [DepthBand(near=100.0, far=200.0)] * 4
    assert payload["command"]["depth_band"]["source"] == "--depth-band"


def test_stating_nothing_keeps_the_sensor_interval_and_says_so(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    seen = _spy_prepare(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    # The default is resolved once and passed explicitly, so every frame's record
    # carries the interval it was actually clipped to.
    assert seen == [DepthBand.default()] * 4
    assert payload["command"]["depth_band"] == {
        "near": DEPTH_CLIP_NEAR,
        "far": DEPTH_CLIP_FAR,
        "source": "default",
    }


def test_a_band_the_clip_cannot_apply_is_refused_before_any_frame_is_prepared(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    seen = _spy_prepare(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--depth-band", "250", "100"])

    out, err = capsys.readouterr()
    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "reconstruction refused" in err
    assert seen == []
    assert out == ""


def test_a_malformed_band_in_the_file_is_refused_rather_than_ignored(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    seen = _spy_prepare(monkeypatch)
    _write_config("[registration]\ndepth_band = [275, 130]\n")

    code = main(["reconstruct", str(frames_dir)])

    err = capsys.readouterr().err
    assert code != 0
    assert "depth_band" in err
    assert seen == []


def test_the_config_command_shows_the_band_and_where_it_is_stated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["config", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["registration_depth_band"] == {"value": None, "source": None}

    path = _write_config("[registration]\ndepth_band = [130, 275]\n")
    code = main(["config", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["registration_depth_band"] == {"value": [130.0, 275.0], "source": str(path)}

    code = main(["config"])
    out = capsys.readouterr().out
    assert code == 0
    assert "registration depth band: 130-275" in out


def test_reconstruct_accepts_the_project_file_and_names_it(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scanner3d.project import write_project

    frames_dir = capture_of(tmp_path, frames=4)
    project = write_project(frames_dir, source="sweep", capture={"decimate": 2})
    scanning_registration(monkeypatch)

    code = main(["reconstruct", str(project), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["command"]["frames_directory"] == str(frames_dir.resolve())
    assert payload["command"]["project_file"] == str(project)


def test_reconstruct_refuses_a_project_file_that_is_not_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stray = tmp_path / "notes.scanner3d"
    stray.write_text("{}", encoding="utf-8")

    code = main(["reconstruct", str(stray)])

    assert code != 0
    assert "cannot be reconstructed" in capsys.readouterr().err
