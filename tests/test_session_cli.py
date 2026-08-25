"""Tests for the offline ``scanner3d session`` inspection command.

The command must never open a device or run the SDK probe, must report what
the manifest says rather than what the directory looks like, and must fail
loudly when a recorded artifact no longer matches its checksum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanner3d.cli import main
from scanner3d.errors import EXIT_CODES, DoctorStatus
from scanner3d.session import (
    MANIFEST_NAME,
    CoordinateFrame,
    DeviceIdentity,
    FixedClock,
    SessionState,
    SessionStore,
    StageState,
)

SERIAL = "EXAMPLESERIAL0001"


def complete_session(tmp_path: Path, session_id: str = "s-0001") -> SessionStore:
    store = SessionStore.create(
        tmp_path,
        session_id,
        software_version="scanner3d-test",
        device=DeviceIdentity(vendor_id=0x2207, product_id=0x110C, serial=SERIAL, usb_path="1-1.1"),
        frames=(
            CoordinateFrame(
                name="depth_camera",
                units="millimetre",
                axes="x_right,y_down,z_forward",
                handedness="right",
            ),
        ),
        clock=FixedClock(),
    )
    store.start()
    store.begin_stage("depth_capture", parameters={"width": 640, "height": 400})
    store.write_artifact(
        "depth_capture", "raw/depth_000000.y16", b"\x00\x01" * 8, kind="raw_depth_frame"
    )
    store.finish_stage("depth_capture", StageState.COMPLETE, counters={"delivered": 1})
    store.add_limitation("depth units are unvalidated")
    store.finish(SessionState.COMPLETE)
    return store


def interrupted_session(tmp_path: Path, session_id: str = "s-0002") -> SessionStore:
    store = SessionStore.create(
        tmp_path, session_id, software_version="scanner3d-test", clock=FixedClock()
    )
    store.start()
    store.begin_stage("depth_capture")
    store.write_artifact("depth_capture", "raw/partial.y16", b"\x00" * 8, kind="raw_depth_frame")
    return store


def test_inspecting_a_complete_session_succeeds(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    assert main(["session", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert "state        complete" in out
    assert "complete     True" in out
    assert "depth_capture: complete" in out
    assert "delivered=1" in out
    assert "raw/depth_000000.y16" in out
    assert "depth units are unvalidated" in out


def test_a_manifest_path_is_accepted_directly(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    assert main(["session", str(store.root / MANIFEST_NAME)]) == 0
    assert "state        complete" in capsys.readouterr().out


def test_json_output_is_the_manifest(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    assert main(["session", str(store.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == store.manifest.session_id
    assert payload["state"] == "complete"
    assert payload["device"]["serial"] == SERIAL


def test_verify_passes_on_an_intact_session(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    assert main(["session", str(store.root), "--verify"]) == 0
    assert "all recorded artifacts match" in capsys.readouterr().out


def test_verify_fails_on_a_tampered_artifact(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    (store.root / "raw" / "depth_000000.y16").write_bytes(b"X" * 16)
    assert main(["session", str(store.root), "--verify"]) == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "checksum mismatch" in capsys.readouterr().out


def test_verify_fails_on_a_missing_artifact(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    (store.root / "raw" / "depth_000000.y16").unlink()
    assert main(["session", str(store.root), "--verify"]) == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "missing" in capsys.readouterr().out


def test_verify_fails_on_a_resized_artifact(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    (store.root / "raw" / "depth_000000.y16").write_bytes(b"X" * 999)
    assert main(["session", str(store.root), "--verify"]) == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "expected 16 bytes" in capsys.readouterr().out


def test_redaction_hides_the_serial(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    assert main(["session", str(store.root), "--redact"]) == 0
    out = capsys.readouterr().out
    assert SERIAL not in out
    assert "[redacted]" in out


def test_redacted_json_hides_the_serial(tmp_path: Path, capsys):
    store = complete_session(tmp_path)
    assert main(["session", str(store.root), "--json", "--redact"]) == 0
    raw = capsys.readouterr().out
    assert SERIAL not in raw
    assert json.loads(raw)["device"]["serial"] == "[redacted]"


def test_an_interrupted_session_is_not_reported_complete(tmp_path: Path, capsys):
    store = interrupted_session(tmp_path)
    assert main(["session", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert "complete     False" in out
    assert "interrupted" in out


def test_inspection_does_not_relabel_the_manifest(tmp_path: Path):
    store = interrupted_session(tmp_path)
    before = (store.root / MANIFEST_NAME).read_bytes()
    main(["session", str(store.root), "--verify"])
    assert (store.root / MANIFEST_NAME).read_bytes() == before


def test_a_missing_session_is_refused(tmp_path: Path, capsys):
    code = main(["session", str(tmp_path / "nope")])
    assert code == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "cannot read session" in capsys.readouterr().err


def test_a_corrupt_manifest_is_refused(tmp_path: Path, capsys):
    root = tmp_path / "broken"
    root.mkdir()
    (root / MANIFEST_NAME).write_text("{ this is not json", encoding="utf-8")
    assert main(["session", str(root)]) == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "cannot read session" in capsys.readouterr().err


def test_a_failed_session_reports_its_error(tmp_path: Path, capsys):
    store = SessionStore.create(
        tmp_path, "s-0003", software_version="scanner3d-test", clock=FixedClock()
    )
    store.start()
    store.begin_stage("depth_capture")
    store.finish_stage("depth_capture", StageState.FAILED, error="poll timed out")
    store.finish(SessionState.FAILED, error="no frame was delivered")
    assert main(["session", str(store.root)]) == 0
    out = capsys.readouterr().out
    assert "state        failed" in out
    assert "complete     False" in out
    assert "poll timed out" in out
    assert "no frame was delivered" in out


@pytest.mark.parametrize("flag", ["--json", "--verify", "--redact"])
def test_inspection_touches_no_device_root(tmp_path: Path, flag: str, monkeypatch):
    """The command must not reach discovery or the SDK probe on any path."""
    import scanner3d.cli as cli_module

    def explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("session inspection must not touch hardware discovery")

    monkeypatch.setattr(cli_module, "find_scanners", explode)
    monkeypatch.setattr(cli_module, "_maybe_run_probe", explode)
    store = complete_session(tmp_path)
    assert main(["session", str(store.root), flag]) == 0
