"""The ``config`` command, and what a recorded captures directory changes.

The point of the setting is that it is stated once and honoured afterwards, so
these tests set it the way a user would - through the command - and then check
what each capture command would actually write to.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scanner3d import cli, config
from scanner3d.cli import DATED_PLACEHOLDER, NAMED_OR_DATED_PLACEHOLDER, main
from scanner3d.config import (
    CAPTURES_DIR_ENV_VAR,
    GUI_CAPTURE_COMMAND,
    capture_stamp,
    config_path,
    configured_captures_dir,
)
from scanner3d.errors import EXIT_CODES, DoctorStatus
from scanner3d.illumination_shutdown import (
    IlluminationShutdownOutcome,
    ShutdownStatus,
    ShutdownTransferRecord,
)
from scanner3d.paired_capture import (
    AUTHORIZED_SERIAL,
    EXECUTE_OPT_IN,
    RGB_SERIES_EXECUTE_OPT_IN,
)
from scanner3d.rgb_frame import RGB_DEFAULT_OUTPUT_DIR
from scanner3d.v4l2_capture import DEFAULT_OUTPUT_DIR
from tests.conftest import REAL_SERIAL, SysfsBuilder
from tests.fake_v4l2 import FakeKernel
from tests.test_cli import _patch_depth_frame_backend
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
    plan = payload["capture_output_directories"]
    assert plan["rgb-frame"] == str(target / "rgb-frame" / DATED_PLACEHOLDER)
    assert plan["sweep"] == str(target / "sweep" / DATED_PLACEHOLDER)
    # The desktop application takes a directory under the same root, so the
    # report that claims to say where captures land has to name it too - and
    # with its own placeholder, because it is the one caller that asks what to
    # call the scan and only a blank answer there is dated.
    assert plan[GUI_CAPTURE_COMMAND] == str(
        target / GUI_CAPTURE_COMMAND / NAMED_OR_DATED_PLACEHOLDER
    )


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


def test_a_depth_frame_dry_run_names_the_namespace_and_allocates_nothing(
    healthy: SysfsBuilder, tmp_path: Path, capsys
) -> None:
    """A dry run writes no frame, so it takes no directory and creates none.

    It reports the subdirectory a real capture would allocate inside, rather
    than a dated name it has not taken and would not necessarily get.
    """
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    assert main(["depth-frame", "--json", *_roots(healthy)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifacts"]["directory"] == str(tmp_path / "scans" / "depth-frame")
    assert not (tmp_path / "scans").exists()


def test_two_depth_frame_captures_in_a_row_both_keep_their_frame(
    healthy: SysfsBuilder, monkeypatch, tmp_path: Path, capsys
) -> None:
    """The reproduced failure, end to end.

    The artifact writers create fixed names with ``O_EXCL``, so a second
    capture into the same directory used to reach preservation and fail with
    ``File exists`` - having already spent the capture. Both runs must land
    their bytes, and in different directories.
    """
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()

    class AcceptedFinalizer:
        def run(self, _serial: str) -> IlluminationShutdownOutcome:
            return IlluminationShutdownOutcome(
                True,
                True,
                ShutdownStatus.COMMAND_ACCEPTED,
                "synthetic off accepted",
                transfer=ShutdownTransferRecord(60, True, "accepted", 0.0),
            )

    monkeypatch.setattr(
        cli, "illumination_backend_factory", lambda _sysfs, _dev: AcceptedFinalizer()
    )

    directories: list[str] = []
    for _ in range(2):
        kernel = FakeKernel()
        _patch_depth_frame_backend(
            monkeypatch,
            opener=kernel.opener,
            ioctl_fn=kernel.ioctl,
            mmap_fn=kernel.mmap,
            poll_fn=kernel.poll,
            closer=kernel.closer,
        )
        assert (
            main(
                [
                    "depth-frame",
                    "--json",
                    "--execute-depth-frame-capture",
                    "--serial",
                    REAL_SERIAL,
                    *_roots(healthy),
                ]
            )
            == 0
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["artifacts"]["ok"] is True, payload["artifacts"]["note"]
        directories.append(str(payload["artifacts"]["directory"]))

    assert len(set(directories)) == 2
    for where in directories:
        assert Path(where).parent == tmp_path / "scans" / "depth-frame"


def test_an_rgb_frame_dry_run_names_the_namespace_too(
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
    assert len(seen) == 1
    assert seen[0].parent == tmp_path / "scans" / "paired-capture"
    assert seen[0].is_dir()
    assert "required local --output-dir is missing" not in out


def test_two_paired_captures_never_share_a_directory(tmp_path: Path, monkeypatch, capsys) -> None:
    """A one-use directory would spend the second capture and then fail to keep it."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    seen: list[Path] = []

    def factory(output: Path) -> FakeBackend:
        seen.append(output)
        return FakeBackend()

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    main(["paired-capture", EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL])
    main(["paired-capture", EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL])
    assert len(set(seen)) == 2


def test_two_paired_rgb_series_runs_never_share_a_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Named separately from paired-capture even though the helper is shared.

    The two commands reach the allocator through one function today; that is an
    implementation detail, and the assignment names both integrations. A future
    change that gives the series its own output path would otherwise regress
    silently.
    """
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    seen: list[Path] = []

    def factory(output: Path) -> FakeBackend:
        seen.append(output)
        return FakeBackend()

    monkeypatch.setattr(cli, "paired_backend_factory", factory)
    main(["paired-rgb-series", RGB_SERIES_EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL])
    main(["paired-rgb-series", RGB_SERIES_EXECUTE_OPT_IN, "--serial", AUTHORIZED_SERIAL])
    assert len(seen) == 2
    assert len(set(seen)) == 2
    for output in seen:
        assert output.parent == tmp_path / "scans" / "paired-rgb-series"
        assert output.is_dir()


def test_a_paired_run_refused_for_its_serial_allocates_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A refusal must not litter the captures root with directories it never used."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    monkeypatch.setattr(cli, "paired_backend_factory", lambda _output: FakeBackend())
    main(["paired-capture", EXECUTE_OPT_IN, "--serial", "SOMEONE-ELSES-SCANNER"])
    main(["paired-capture"])
    assert not (tmp_path / "scans" / "paired-capture").exists()


def test_sweep_without_a_directory_refuses_and_says_how_to_set_one(capsys) -> None:
    assert main(["sweep"]) == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    err = capsys.readouterr().err
    assert "nowhere to write this sweep" in err
    assert "--set-captures-dir" in err


class _SweepResult:
    delivered = 0
    dropped_by_sequence = 0
    short_frames = 0
    first_frame_ms = 0.0
    stopped_early = False
    stop_reason = "frame budget reached"


def _stub_sweep(monkeypatch) -> None:
    import scanner3d.v4l2_stream as stream_module

    monkeypatch.setattr(stream_module, "stream_frames", lambda **_kwargs: _SweepResult())


def test_sweep_writes_into_a_dated_directory_of_its_own(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Two sweeps must not share a directory: frame_0000.raw would be overwritten."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    _stub_sweep(monkeypatch)

    assert main(["sweep", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    written = Path(str(payload["command"]["output"]))
    assert written.parent == tmp_path / "scans" / "sweep"
    assert written.is_dir()


def test_two_sweeps_get_different_directories_however_fast_they_follow(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Back to back, inside one wall-clock second, is exactly the dangerous case."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    _stub_sweep(monkeypatch)

    outputs = []
    for _ in range(2):
        assert main(["sweep", "--json"]) == 0
        outputs.append(json.loads(capsys.readouterr().out)["command"]["output"])
    assert len(set(outputs)) == 2


def test_a_sweep_never_writes_into_a_directory_that_already_exists(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A pre-existing candidate is stepped over, not mixed with."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    _stub_sweep(monkeypatch)

    stamp = capture_stamp()
    occupied = tmp_path / "scans" / "sweep" / stamp
    occupied.mkdir(parents=True)
    (occupied / "frame_0000.raw").write_bytes(b"an earlier sweep")

    assert main(["sweep", "--json"]) == 0
    written = Path(str(json.loads(capsys.readouterr().out)["command"]["output"]))
    assert written != occupied
    assert (occupied / "frame_0000.raw").read_bytes() == b"an earlier sweep"


def _json_config_set(target: Path, capsys) -> str:
    """Run ``config --json --set-captures-dir target`` and return its stdout."""
    assert main(["config", "--json", "--set-captures-dir", str(target)]) == 0
    return capsys.readouterr().out


def test_an_exhausted_allocation_refuses_before_the_frame_backend_is_built(
    healthy: SysfsBuilder, tmp_path: Path, monkeypatch, capsys
) -> None:
    """Refusing "before hardware is touched" means the factory is never called.

    A unit test of the allocator proves it raises; it does not prove the
    command asks it first. This drives the whole command with every candidate
    name already taken and a factory that fails the test if anything reaches
    it - so the only way to pass is to refuse before construction, which is
    the step that opens the device nodes.
    """
    root = tmp_path / "scans"
    assert main(["config", "--set-captures-dir", str(root)]) == 0
    capsys.readouterr()

    stamp = "20260825-174501"
    monkeypatch.setattr(config, "capture_stamp", lambda _now=None: stamp)
    monkeypatch.setattr(config, "CAPTURE_DIR_ATTEMPTS", 3)
    taken = root / "depth-frame"
    taken.mkdir(parents=True)
    for name in (stamp, f"{stamp}-2", f"{stamp}-3"):
        (taken / name).mkdir()

    def never(**_kwargs: object) -> object:
        raise AssertionError("the backend factory ran despite an unusable capture directory")

    monkeypatch.setattr(cli, "_depth_frame_backend_factory", never)

    status = main(
        [
            "depth-frame",
            "--execute-depth-frame-capture",
            "--serial",
            REAL_SERIAL,
            *_roots(healthy),
        ]
    )
    assert status == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    err = capsys.readouterr().err
    assert "no capture was started" in err
    assert "Traceback" not in err
    # And it took nothing on the way out: only the three that were already there.
    assert sorted(p.name for p in taken.iterdir()) == [stamp, f"{stamp}-2", f"{stamp}-3"]


# --- reporting a mutation against what will actually be used ------------------


def test_setting_while_the_environment_wins_never_claims_the_new_value_is_used(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    chosen = tmp_path / "scans"
    assert main(["config", "--json", "--set-captures-dir", str(chosen)]) == 0
    payload = json.loads(capsys.readouterr().out)

    change = payload["change"]
    assert change["action"] == "set"
    assert change["file_value"] == str(chosen)
    assert change["file_changed"] is True
    assert change["effective_value"] == str(tmp_path / "elsewhere")
    assert change["effective_source"] == CAPTURES_DIR_ENV_VAR
    assert change["overridden_by_environment"] is True
    # The sentence a person reads must not disagree with the record above it.
    assert str(tmp_path / "elsewhere") in payload["changed"]
    assert CAPTURES_DIR_ENV_VAR in payload["changed"]
    assert payload["capture_output_directories"]["sweep"].startswith(str(tmp_path / "elsewhere"))


def test_repeating_the_identical_set_reports_that_nothing_changed(tmp_path: Path, capsys) -> None:
    """The file already said this, so no write happened and none is claimed.

    ``file_changed`` is the writer's own answer to "did I rewrite the file",
    and a script watching it to decide whether to reload or to commit the
    config is entitled to that answer being true.
    """
    target = tmp_path / "scans"
    assert main(["config", "--set-captures-dir", str(target)]) == 0
    capsys.readouterr()
    first = json.loads(_json_config_set(target, capsys))
    assert first["change"]["file_changed"] is False
    assert first["change"]["file_value"] == str(target)
    assert first["change"]["effective_value"] == str(target)
    assert "already" in first["changed"]

    assert main(["config", "--set-captures-dir", str(target)]) == 0
    out = capsys.readouterr().out
    assert f"captures directory was already set to {target}; nothing changed" in out


def test_the_first_set_of_a_value_still_reports_the_write_it_made(tmp_path: Path, capsys) -> None:
    """The companion to the test above: a real write must not read as a no-op."""
    target = tmp_path / "scans"
    payload = json.loads(_json_config_set(target, capsys))
    assert payload["change"]["file_changed"] is True
    assert "already" not in payload["changed"]


def test_a_repeated_set_under_an_override_reports_neither_a_write_nor_the_value(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Two untruths are available here, and neither may be told."""
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    target = tmp_path / "scans"
    assert main(["config", "--set-captures-dir", str(target)]) == 0
    capsys.readouterr()

    payload = json.loads(_json_config_set(target, capsys))
    change = payload["change"]
    assert change["file_changed"] is False
    assert change["file_value"] == str(target)
    assert change["effective_value"] == str(tmp_path / "elsewhere")
    assert change["overridden_by_environment"] is True
    # Not "recorded as", which would claim a write; and the sentence still
    # names where the captures actually go.
    assert "already recorded as" in payload["changed"]
    assert str(tmp_path / "elsewhere") in payload["changed"]


def test_unsetting_while_the_environment_wins_does_not_claim_defaults_are_back(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The file no longer states one; the shell still does, and captures still go there."""
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    monkeypatch.setenv(CAPTURES_DIR_ENV_VAR, str(tmp_path / "elsewhere"))

    assert main(["config", "--unset-captures-dir"]) == 0
    out = capsys.readouterr().out
    assert "back to their own defaults" not in out
    assert str(tmp_path / "elsewhere") in out

    assert main(["config", "--json", "--unset-captures-dir"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["change"]["file_changed"] is False
    assert payload["change"]["effective_value"] == str(tmp_path / "elsewhere")
    assert payload["change"]["overridden_by_environment"] is True
    assert "back to their own defaults" not in payload["changed"]


def test_unsetting_with_no_override_does_say_the_defaults_are_back(tmp_path: Path, capsys) -> None:
    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    assert main(["config", "--json", "--unset-captures-dir"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["change"]["overridden_by_environment"] is False
    assert payload["change"]["effective_value"] is None
    assert "back to their own defaults" in payload["changed"]


# --- write failures are sentences with an exit code ---------------------------


def test_a_write_that_fails_mid_flight_is_a_sentence_and_a_status(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """No traceback, and the report must say whether the original survived."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[captures]\ndirectory = "/old"\n', encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", boom)
    assert (
        main(["config", "--set-captures-dir", str(tmp_path / "scans")])
        == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    )
    err = capsys.readouterr().err
    assert "configuration not changed" in err
    assert "is unchanged" in err
    assert "Traceback" not in err
    assert path.read_text(encoding="utf-8") == '[captures]\ndirectory = "/old"\n'


def test_recording_the_depth_band_is_read_back_by_every_reader(capsys) -> None:
    assert main(["config", "--set-depth-band", "130", "275"]) == 0
    out = capsys.readouterr().out
    assert "registration depth band recorded as 130-275" in out
    assert "registration depth band: 130-275" in out
    assert main(["config", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["registration_depth_band"]["value"] == [130.0, 275.0]
    assert payload["registration_depth_band"]["source"] == str(config_path())
    # The same band again is a no-op, and says so.
    assert main(["config", "--set-depth-band", "130", "275"]) == 0
    assert "already recorded as 130-275" in capsys.readouterr().out


def test_clearing_the_depth_band_returns_registration_to_the_sensor_interval(capsys) -> None:
    assert main(["config", "--set-depth-band", "130", "275"]) == 0
    capsys.readouterr()
    assert main(["config", "--unset-depth-band"]) == 0
    out = capsys.readouterr().out
    assert "registration depth band cleared" in out
    assert "registration depth band: not set" in out
    assert main(["config", "--unset-depth-band"]) == 0
    assert "nothing changed" in capsys.readouterr().out


def test_a_depth_band_the_clip_could_not_apply_is_refused_with_a_sentence(capsys) -> None:
    code = main(["config", "--set-depth-band", "275", "130"])
    err = capsys.readouterr().err
    assert code != 0
    assert "configuration not changed" in err
    assert "positive, increasing interval" in err


def test_a_sweep_writes_the_scans_project_file_beside_its_frames(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scanner3d.project import PROJECT_NAME, read_project

    assert main(["config", "--set-captures-dir", str(tmp_path / "scans")]) == 0
    capsys.readouterr()
    _stub_sweep(monkeypatch)

    assert main(["sweep", "--json", "--frames", "40", "--decimate", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    written = Path(str(payload["command"]["output"]))
    assert payload["project_file"] == str(written / PROJECT_NAME)
    project = read_project(written)
    assert project.capture["source"] == "sweep"
    assert project.capture["frames_requested"] == 40
    assert project.capture["decimate"] == 2
    assert project.capture["frames_written"] == 0
    assert project.capture["stop_reason"] == "frame budget reached"
