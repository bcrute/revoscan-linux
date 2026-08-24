"""Offline paired-capture CLI contract tests."""

from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from pathlib import Path

import pytest

from scanner3d import cli
from scanner3d.cli import main
from scanner3d.illumination_shutdown import IlluminationShutdownOutcome, ShutdownStatus
from scanner3d.paired_capture import AUTHORIZED_SERIAL, EXECUTE_OPT_IN
from scanner3d.v4l2_capture import (
    ArtifactRecord,
    BufferInfo,
    CaptureOutcome,
    CaptureStatus,
    CleanupRecord,
    FrameArtifact,
    Step,
    StepResult,
)
from tests.test_paired_capture import FakeBackend


def test_human_dry_run_is_explicit_and_does_not_open(capsys) -> None:
    assert main(["paired-capture"]) == 0
    output = capsys.readouterr().out
    assert "Status: not_attempted" in output
    assert "Exit: 0" in output
    assert "depth Y16 640x400" in output
    assert "RGB STREAMON" in output
    assert EXECUTE_OPT_IN in output


def test_json_dry_run_contains_bounds_order_and_exit(capsys) -> None:
    assert main(["paired-capture", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_attempted"
    assert payload["exit_code"] == 0
    assert payload["plan"]["start_interval_s"] == 1.0
    assert payload["plan"]["roles"]["depth"]["fourcc"] == "Y16 "
    assert payload["plan"]["roles"]["rgb"]["fourcc"] == "MJPG"
    assert payload["plan"]["roles"]["rgb"]["max_buffer_bytes"] == 8192000
    assert payload["plan"]["execution"]["required_serial"] == AUTHORIZED_SERIAL


def test_json_out_matches_stdout(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "paired.json"
    assert main(["paired-capture", "--json", "--json-out", str(output_path)]) == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert json.loads(output_path.read_text(encoding="utf-8")) == stdout_payload


def test_live_paired_json_reports_shutdown_failure_and_stderr_recovery(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class FailedFinalizer(FakeBackend):
        def shutdown_depth(self, _handle: object) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True,
                True,
                ShutdownStatus.IOCTL_ERROR,
                "synthetic off transfer failure",
            )

    backend = FailedFinalizer()
    monkeypatch.setattr(cli, "paired_backend_factory", lambda _output: backend)
    output_path = tmp_path / "paired-failure.json"
    result = main(
        [
            "paired-capture",
            "--json",
            EXECUTE_OPT_IN,
            "--serial",
            AUTHORIZED_SERIAL,
            "--output-dir",
            str(tmp_path / "evidence"),
            "--json-out",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    assert result != 0
    assert stdout_payload["status"] == "complete"
    assert stdout_payload["succeeded"] is False
    assert stdout_payload["illumination_shutdown"]["status"] == "ioctl_error"
    assert json.loads(output_path.read_text(encoding="utf-8")) == stdout_payload
    assert "blue-light off command was not carried through" in captured.err
    assert "DO NOT CLAIM THE BLUE LIGHT IS OFF" in captured.err


def test_live_paired_human_shutdown_failure_is_unmistakable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class FailedFinalizer(FakeBackend):
        def shutdown_depth(self, _handle: object) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True, True, ShutdownStatus.IOCTL_ERROR, "synthetic off transfer failure"
            )

    monkeypatch.setattr(cli, "paired_backend_factory", lambda _output: FailedFinalizer())
    result = main(
        [
            "paired-capture",
            EXECUTE_OPT_IN,
            "--serial",
            AUTHORIZED_SERIAL,
            "--output-dir",
            str(tmp_path / "evidence"),
        ]
    )
    captured = capsys.readouterr()
    assert result != 0
    assert "Blue-light off command (transfer only, NOT an optical result):" in captured.out
    assert "Overall result: FAILED" in captured.out
    assert "blue-light off command was not carried through" in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["paired-capture", EXECUTE_OPT_IN],
        ["paired-capture", EXECUTE_OPT_IN, "--serial", "wrong"],
    ],
)
def test_execution_requires_exact_serial_and_returns_stable_refusal(argv, capsys) -> None:
    assert main(argv) == 10
    assert "Status: refused" in capsys.readouterr().out


def test_exact_opt_in_is_not_abbreviated() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["paired-capture", "--start-depth-first-rgb-sec"])
    assert excinfo.value.code == 2


def test_correct_gate_reports_missing_output_directory(capsys) -> None:
    assert main(["paired-capture", EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL]) == 10
    output = capsys.readouterr().out
    assert "required local --output-dir is missing" in output
    assert "Exit: 10" in output


@pytest.mark.parametrize(
    "argv",
    [
        ["paired-capture", EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL],
        [
            "paired-capture",
            EXECUTE_OPT_IN,
            "--serial",
            "wrong",
            "--output-dir",
            "/tmp/ignored-synthetic-output",
        ],
    ],
)
def test_refused_cli_gates_never_construct_backend(argv, monkeypatch, capsys) -> None:
    def forbidden(_output_dir: Path):
        raise AssertionError("backend factory must not run for a refusal gate")

    monkeypatch.setattr(cli, "paired_backend_factory", forbidden)
    assert main(argv) == 10
    assert "Status: refused" in capsys.readouterr().out


class _CliInjectedBackend(FakeBackend):
    def role_outcome(self, handle: object) -> CaptureOutcome:
        role = str(handle)
        cleanup_failed = role == "rgb"
        return CaptureOutcome(
            status=CaptureStatus.COMPLETE,
            detail=f"synthetic {role} role complete",
            target=None,
            steps=(StepResult(Step.OPEN, True, "synthetic open"),),
            cleanup=CleanupRecord(
                attempted=True,
                steps=(
                    StepResult(
                        Step.CLOSE,
                        not cleanup_failed,
                        "synthetic close failure" if cleanup_failed else "synthetic close",
                    ),
                ),
                ok=not cleanup_failed,
                note="synthetic role cleanup failed"
                if cleanup_failed
                else "synthetic role cleanup",
                guidance="synthetic recovery guidance" if cleanup_failed else None,
            ),
            artifacts=ArtifactRecord(
                attempted=True,
                ok=True,
                note="synthetic artifact preserved",
                directory="/tmp/synthetic",
                files=(
                    FrameArtifact(
                        kind="raw",
                        path="/tmp/synthetic/frame.bin",
                        byte_count=4,
                        sha256="a" * 64,
                        ok=True,
                        detail="synthetic file",
                    ),
                ),
                ceiling_bytes=1024,
            ),
            frame_bytes=4,
            dequeued=BufferInfo(
                index=0,
                buf_type=1,
                bytesused=4,
                flags=0,
                flag_names=(),
                field=1,
                field_name="none",
                timestamp_sec=17,
                timestamp_usec=45,
                timestamp_type="monotonic",
                timestamp_source="unknown",
                timecode_type=0,
                timecode_flags=0,
                timecode_frames=0,
                timecode_seconds=0,
                timecode_minutes=0,
                timecode_hours=0,
                timecode_userbits="",
                sequence=7 if role == "depth" else 9,
                memory=1,
                offset=0,
                length=4,
            ),
        )


def test_opted_in_cli_uses_injected_factory_and_renders_outer_evidence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    backend = _CliInjectedBackend(fail="off:rgb")
    created: list[Path] = []

    def factory(output_dir: Path):
        created.append(output_dir)
        return backend

    @contextmanager
    def deferred_shield():
        yield [signal.SIGTERM]

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    monkeypatch.setattr(cli, "paired_termination_shield", deferred_shield)
    output_dir = tmp_path / "evidence"
    result = main(
        [
            "paired-capture",
            EXECUTE_OPT_IN,
            "--serial",
            AUTHORIZED_SERIAL,
            "--output-dir",
            str(output_dir),
        ]
    )
    text = capsys.readouterr().out
    assert result == 9
    assert created == [output_dir]
    assert "depth: complete bytes=4" in text
    assert "rgb: complete bytes=4" in text
    assert "sequence=7 timestamp_ns=17000045000" in text
    assert "sequence=9 timestamp_ns=17000045000" in text
    assert "Cleanup actions:" in text
    assert "Cleanup errors:" in text
    assert "Cleanup deferred signals:" in text
    assert "Cleanup recovery:" in text

    created.clear()
    result = main(
        [
            "paired-capture",
            "--json",
            EXECUTE_OPT_IN,
            "--serial",
            AUTHORIZED_SERIAL,
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 9
    assert created == [output_dir]
    assert payload["roles"]["depth"]["status"] == "complete"
    assert payload["roles"]["rgb"]["status"] == "complete"
    assert payload["roles"]["depth"]["artifacts"]["files"]
    assert payload["timing"]["depth_sequence"] == 7
    assert payload["timing"]["rgb_sequence"] == 9
    assert payload["timing"]["observed_skew_ns"] == 0
    assert payload["cleanup"]["actions"]
    assert payload["cleanup"]["errors"]
    assert payload["cleanup"]["deferred_signals"] == [signal.SIGTERM]
    assert payload["cleanup"]["recovery_guidance"]
