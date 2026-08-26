"""The ``config`` command, and what a recorded captures directory changes.

The point of the setting is that it is stated once and honoured afterwards, so
these tests set it the way a user would - through the command - and then check
what each capture command would actually write to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scanner3d import cli
from scanner3d.cli import main
from scanner3d.config import CAPTURES_DIR_ENV_VAR, config_path, configured_captures_dir
from scanner3d.errors import EXIT_CODES, DoctorStatus
from scanner3d.paired_capture import AUTHORIZED_SERIAL, EXECUTE_OPT_IN
from scanner3d.rgb_frame import RGB_DEFAULT_OUTPUT_DIR
from scanner3d.v4l2_capture import DEFAULT_OUTPUT_DIR
from tests.conftest import SysfsBuilder
from tests.test_paired_capture import FakeBackend


def _roots(b: SysfsBuilder) -> list[str]:
    return ["--sysfs-root", str(b.sysfs), "--dev-root", str(b.dev)]


# --- reporting ---------------------------------------------------------------


def test_config_reports_nothing_set_and_each_commands_own_default(capsys) -> None:
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "captures directory: not set" in out
    assert str(DEFAULT_OUTPUT_DIR) in out
    assert str(RGB_DEFAULT_OUTPUT_DIR) in out
    assert "--output-dir is required" in out


def test_config_json_names_the_file_it_reads_even_before_it_exists(capsys) -> None:
    assert main(["config", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_file"] == str(config_path())
    assert payload["config_file_exists"] is False
    assert payload["captures_directory"]["value"] is None
    assert payload["captures_directory"]["source"] is None


def test_config_opens_no_device(capsys) -> None:
    """Reporting settings must not depend on, or touch, any hardware."""
    assert main(["config", "--json", "--json-out", "/dev/null"]) == 0
    assert configured_captures_dir() is None


# --- setting and clearing ----------------------------------------------------


def test_setting_a_captures_directory_persists_it_and_reports_the_change(
    tmp_path: Path, capsys
) -> None:
    target = tmp_path / "scans"
    assert main(["config", "--set-captures-dir", str(target)]) == 0
    out = capsys.readouterr().out
    assert f"captures directory set to {target}" in out
    assert f"{target}/depth-frame" in out
    # Read back through a second invocation: the setting is on disk, not in
    # this process.
    assert main(["config", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["captures_directory"]["value"] == str(target)
    assert payload["capture_output_directories"]["rgb-frame"] == str(target / "rgb-frame")
    assert payload["capture_output_directories"]["sweep"].startswith(f"{target}/sweep-")


def test_clearing_returns_every_command_to_its_own_default(tmp_path: Path, capsys) -> None:
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    assert main(["config", "--unset-captures-dir"]) == 0
    assert "captures directory cleared" in capsys.readouterr().out
    assert configured_captures_dir() is None


def test_clearing_when_nothing_was_set_says_so_rather_than_claiming_a_change(capsys) -> None:
    assert main(["config", "--unset-captures-dir"]) == 0
    assert "nothing changed" in capsys.readouterr().out


def test_setting_and_unsetting_at_once_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["config", "--set-captures-dir", "/tmp/x", "--unset-captures-dir"])
    assert excinfo.value.code == 2


def test_an_environment_override_that_outranks_the_write_is_pointed_out(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Writing a file that something else overrides must not read as success."""
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    captured = capsys.readouterr()
    assert f"{CAPTURES_DIR_ENV_VAR} is set in this shell and still wins" in captured.err
    assert str(tmp_path / "elsewhere") in captured.out


def test_a_malformed_config_stops_a_capture_with_a_sentence_not_a_traceback(
    healthy: SysfsBuilder, capsys
) -> None:
    """Every command can now consult the file, so every command can meet a broken one."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[captures\nnonsense\n", encoding="utf-8")
    assert main(["depth-frame", *_roots(healthy)]) == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "cannot read configuration" in capsys.readouterr().err


def test_a_malformed_config_is_reported_and_left_alone(tmp_path: Path, capsys) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[captures\nnonsense\n", encoding="utf-8")
    assert (
        main(["config", "--set-captures-dir", str(tmp_path / "scans")])
        == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    )
    assert "configuration not changed" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == "[captures\nnonsense\n"


# --- what the capture commands do with it ------------------------------------


def test_depth_frame_writes_under_the_captures_directory(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    assert main(["depth-frame", "--json", *_roots(healthy)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["directory"] == str(tmp_path / "scans" / "depth-frame")


def test_rgb_frame_writes_under_the_captures_directory(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    assert main(["rgb-frame", "--json", *_roots(healthy)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["directory"] == str(tmp_path / "scans" / "rgb-frame")


def test_an_explicit_output_dir_still_wins_over_the_configured_one(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    explicit = tmp_path / "just-this-once"
    assert main(["depth-frame", "--json", "--output-dir", str(explicit), *_roots(healthy)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["directory"] == str(explicit)


def test_paired_capture_uses_the_configured_directory_instead_of_refusing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    seen: list[Path] = []

    def factory(output: Path) -> FakeBackend:
        seen.append(output)
        return FakeBackend()

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    main(["paired-capture", EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL])
    out = capsys.readouterr().out
    assert seen == [tmp_path / "scans" / "paired-capture"]
    assert "required local --output-dir is missing" not in out


def test_sweep_without_a_directory_refuses_and_says_how_to_set_one(capsys) -> None:
    assert main(["sweep"]) == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    err = capsys.readouterr().err
    assert "nowhere to write this sweep" in err
    assert "--set-captures-dir" in err


def test_sweep_writes_into_a_dated_directory_of_its_own(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Two sweeps must not share a directory: frame_0000.raw would be overwritten."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()

    class Result:
        delivered = 0
        dropped_by_sequence = 0
        short_frames = 0
        first_frame_ms = 0.0
        stopped_early = False
        stop_reason = "frame budget reached"

    import scanner3d.v4l2_stream as stream_module

    monkeypatch.setattr(stream_module, "stream_frames", lambda **_kwargs: Result())
    assert main(["sweep", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    written = Path(str(payload["command"]["output"]))
    assert written.parent == tmp_path / "scans"
    assert written.name.startswith("sweep-")
    assert written.is_dir()
