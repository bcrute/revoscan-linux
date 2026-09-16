"""The registration depth band: stated on the command line, in the file, or not at all.

The band decides which part of the scene registers, so where it came from is
reported beside its value, a band the clip cannot apply is refused before a
frame is read, and the config command shows it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanner3d import pipeline, preparation, turntable
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

    def spy(raw: bytes, *, target_points: int, depth_band: DepthBand | None = None, **deprojection):
        seen.append(depth_band)
        return real(raw, target_points=target_points, depth_band=depth_band, **deprojection)

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


# -- the turntable model -------------------------------------------------------


def _fake_turntable(monkeypatch) -> list[dict[str, object]]:
    """register_turntable stood in by one that wraps the manufactured sweep's poses."""
    calls: list[dict[str, object]] = []

    def fake(clouds, **kwargs):
        calls.append(dict(kwargs))
        poses = tuple(pipeline.reconstruct_sweep(clouds).poses)
        return turntable.TurntableRegistration(
            poses=poses,
            axis=(0.0, 1.0, 0.0),
            centre=(0.0, 0.0, 200.0),
            rate_degrees=-2.84,
            period_frames=127,
            times_used=False,
            pairs_used=68,
            axis_scatter_degrees=0.21,
            centre_shift_mm=0.1,
            period_leftover_degrees=-1.04,
            agreement=(turntable.LagAgreement(lag=1, overlap=0.99, median_mm=0.09),),
            backend="numpy",
        )

    monkeypatch.setattr(turntable, "register_turntable", fake, raising=True)
    return calls


def test_a_capture_that_proves_a_turn_is_registered_by_the_turntable_model(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    calls = _fake_turntable(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert len(calls) == 1
    scale = calls[0]["depth_scale"]
    assert isinstance(scale, float) and scale > 0
    times = calls[0]["times"]
    assert isinstance(times, tuple) and len(times) == 4
    assert payload["registration"]["registration"] == "turntable"
    assert payload["registration"]["frames_per_turn"] == pytest.approx(360 / 2.84)


def test_no_turntable_keeps_the_chain_and_never_asks_the_model(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)
    calls = _fake_turntable(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json", "--no-turntable"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert calls == []
    assert payload["registration"]["registration"] == "frame-by-frame"


# --- the deprojection the command registers under ------------------------------


def test_the_report_names_the_calibrated_deprojection_every_point_was_made_with(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """This unit's measured correction is applied by default, and said so."""
    from scanner3d.metric_correction import CUE_BALL_20260911, calibrated_deprojection

    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    intrinsics, depth_scale, correction = calibrated_deprojection()
    assert correction is CUE_BALL_20260911
    deprojection = payload["registration"]["deprojection"]
    assert deprojection["fx"] == intrinsics.values["fx"]
    assert deprojection["fy"] == intrinsics.values["fy"]
    assert deprojection["depth_scale"] == depth_scale
    assert deprojection["metric_correction"] == CUE_BALL_20260911.as_dict()


def test_no_metric_correction_registers_under_the_factory_record_and_says_so(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scanner3d.metric_correction import FACTORY_DEPTH_SCALE
    from scanner3d.simulate import FACTORY_INTRINSICS

    frames_dir = capture_of(tmp_path, frames=4)
    scanning_registration(monkeypatch)

    code = main(["reconstruct", str(frames_dir), "--json", "--no-metric-correction"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    deprojection = payload["registration"]["deprojection"]
    assert deprojection["fx"] == FACTORY_INTRINSICS.values["fx"]
    assert deprojection["fy"] == FACTORY_INTRINSICS.values["fy"]
    assert deprojection["depth_scale"] == FACTORY_DEPTH_SCALE
    assert deprojection["metric_correction"] is None
