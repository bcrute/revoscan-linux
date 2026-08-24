"""Tests for the offline ``scanner3d depth-frame-to-ply`` command.

The command is a file-to-file transform: a raw depth frame that was captured
some other time goes in, a PLY comes out, and nothing between the two is a
device. So the first thing defended here is the *absence* of hardware - the
command must reach neither device discovery nor the vendor-SDK probe on any
argument path, including every path that ends in a refusal.

The rest is about what a user is left believing. A point cloud is the most
persuasive artifact this repository produces: it opens in a viewer, it has a
shape, and nothing in the geometry says that no measurement anywhere here has
established a single distance. That has to be said by the command and by the
file, so these tests check that

* the calibration is the caller's and is *required* - no fx, fy, cx, cy or
  depth scale is defaulted, so a cloud cannot acquire this unit's numbers
  without anyone stating them;
* every limitation the conversion reported reaches both the human output and
  the written PLY header;
* the unvalidated unit is stated before the first number and again at the end;
* a wrong-sized frame, an unreadable path and an existing output file are each
  refused with a clear message, a distinct exit code and no file written.

Not one assertion here is about dimensional or metric accuracy, and none can
be: no physical measurement in this repository has validated any distance it
produces, and the depth scalar's unit is explicitly unproven. The numbers are
only ever checked for structure - counts, determinism, provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import scanner3d.cli as cli_module
from scanner3d.cli import main
from scanner3d.cloud_export import EMPTY_CLOUD_NOTE, EXPORT_MODE, SCHEMA_VERSION
from scanner3d.depth_metric import STATIC_LIMITATIONS, UNVALIDATED_UNIT
from scanner3d.depth_scale import UNIT_DISCLAIMER
from scanner3d.errors import EXIT_CODES, DoctorStatus
from scanner3d.pointcloud import INHERENT_LIMITATIONS, PlyFormat
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

COMMAND = "depth-frame-to-ply"

#: Round, centred calibration numbers, chosen so the deprojection is exact in
#: binary floating point. They are *not* this unit's factory values: no test
#: here needs them, and a test that quoted them would make the command look as
#: though it knows them.
TEST_FX = 100.0
TEST_FY = 100.0
TEST_CX = 320.0
TEST_CY = 200.0

#: A scalar a test can pass without pretending it measures anything.
TEST_SCALE = 0.005

#: What a test says when the command asks where its calibration came from.
TEST_SOURCE = "tests/test_pointcloud_cli.py: numbers typed by a test, read from no device"

#: A syntactically valid digest that identifies nothing.
FAKE_DIGEST = "0" * 64

#: A few samples with depth, at known pixels. Nothing is asserted about the
#: coordinates they produce beyond how many of them there are.
SAMPLES: Mapping[tuple[int, int], int] = {
    (10, 20): 1000,
    (320, 200): 2000,
    (639, 399): 4095,
}


# --- inputs the tests build ---------------------------------------------------


def frame(samples: Mapping[tuple[int, int], int] = SAMPLES) -> bytes:
    """One whole 512000-byte frame, zero except at the named ``(u, v)``."""
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    for (u, v), value in samples.items():
        struct.pack_into("<H", buffer, (v * FRAME_WIDTH + u) * 2, value)
    return bytes(buffer)


def frame_file(tmp_path: Path, samples: Mapping[tuple[int, int], int] = SAMPLES) -> Path:
    """A captured frame on disk, as the command expects to find one."""
    path = tmp_path / "depth-frame-raw.bin"
    path.write_bytes(frame(samples))
    return path


def argv(source: Path, output: Path, *extra: str) -> list[str]:
    """The command line, with every required calibration input stated."""
    return [
        COMMAND,
        str(source),
        str(output),
        "--fx",
        repr(TEST_FX),
        "--fy",
        repr(TEST_FY),
        "--cx",
        repr(TEST_CX),
        "--cy",
        repr(TEST_CY),
        "--depth-scale",
        repr(TEST_SCALE),
        "--calibration-source",
        TEST_SOURCE,
        *extra,
    ]


def restated(args: Sequence[str], flag: str, value: str) -> list[str]:
    """``args`` with ``flag``'s value replaced, so nothing else shifts."""
    out = list(args)
    out[out.index(flag) + 1] = value
    return out


# --- reading the file back ----------------------------------------------------


def squeezed(text: str) -> str:
    """``text`` with every run of whitespace removed."""
    return "".join(text.split())


def flattened(header: str) -> str:
    """The header's comment text with all whitespace removed.

    ``build_header`` wraps comments at 70 columns and indents continuations, so
    a substring check against an original sentence has to ignore whitespace
    entirely rather than try to re-flow it.
    """
    return squeezed("".join(line.removeprefix("comment ") for line in header.splitlines()))


def parse_ply(data: bytes) -> tuple[str, list[tuple[float, ...]]]:
    """A small PLY reader: the header text plus the decoded vertices.

    Written here rather than imported, so the file is read back by something
    that did not produce it.
    """
    marker = b"end_header\n"
    end = data.index(marker) + len(marker)
    header = data[:end].decode("ascii")
    lines = header.splitlines()
    assert lines[0] == "ply"
    assert lines[-1] == "end_header"
    encoding = lines[1].split()[1]
    counts = [int(line.split()[2]) for line in lines if line.startswith("element vertex ")]
    assert len(counts) == 1
    count = counts[0]
    assert [line for line in lines if line.startswith("property ")] == [
        "property float x",
        "property float y",
        "property float z",
    ]

    body = data[end:]
    points: list[tuple[float, ...]] = []
    if encoding == "ascii":
        rows = body.decode("ascii").splitlines()
        assert len(rows) == count
        for row in rows:
            values = row.split()
            assert len(values) == 3
            points.append(tuple(float(value) for value in values))
    else:
        assert encoding == "binary_little_endian"
        assert len(body) == count * 12
        for index in range(count):
            points.append(struct.unpack_from("<fff", body, index * 12))
    return header, points


def subparser(name: str = COMMAND) -> argparse.ArgumentParser:
    """One live subparser, pulled from the real ``build_parser()``."""
    parser = cli_module.build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    result = subparsers_action.choices[name]
    assert isinstance(result, argparse.ArgumentParser)
    return result


def action_named(dest: str, name: str = COMMAND) -> argparse.Action:
    """One argument of the live subparser, by its ``dest``."""
    return next(action for action in subparser(name)._actions if action.dest == dest)


# --- the success paths --------------------------------------------------------


def test_a_frame_on_disk_becomes_a_binary_ply(tmp_path: Path, capsys):
    """The default path: bytes on disk in, a parseable binary PLY out."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    assert main(argv(source, output)) == 0

    assert output.is_file()
    header, points = parse_ply(output.read_bytes())
    assert "format binary_little_endian 1.0" in header
    assert len(points) == len(SAMPLES)
    assert "wrote 3 vertices" in capsys.readouterr().out


def test_the_ascii_encoding_is_written_when_asked(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud-ascii.ply"

    assert main(argv(source, output, "--ply-format", "ascii")) == 0

    header, points = parse_ply(output.read_bytes())
    assert "format ascii 1.0" in header
    assert len(points) == len(SAMPLES)
    assert "format     ascii" in capsys.readouterr().out


def test_both_encodings_carry_the_same_vertices(tmp_path: Path):
    """Structure only: the two encodings differ in bytes, not in content."""
    source = frame_file(tmp_path)
    binary_path = tmp_path / "b.ply"
    ascii_path = tmp_path / "a.ply"

    assert main(argv(source, binary_path)) == 0
    assert main(argv(source, ascii_path, "--ply-format", "ascii")) == 0

    binary_header, binary_points = parse_ply(binary_path.read_bytes())
    ascii_header, ascii_points = parse_ply(ascii_path.read_bytes())
    assert binary_points == ascii_points
    # The only difference between the two headers is the format line itself.
    assert binary_header.replace("binary_little_endian", "ascii") == ascii_header


def test_two_runs_of_one_frame_write_identical_files(tmp_path: Path):
    """Determinism: the same frame and the same statements give the same bytes."""
    source = frame_file(tmp_path)
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"

    assert main(argv(source, first)) == 0
    assert main(argv(source, second)) == 0
    assert first.read_bytes() == second.read_bytes()


def test_a_frame_with_no_depth_still_writes_a_ply_that_says_so(tmp_path: Path, capsys):
    """A zero-yield frame is a run that found nothing, not a failure to run.

    Low-yield frames are ordinary for this unit - a bad target is enough - so
    the command writes the file, reports zero vertices and puts the reason in
    the header rather than exiting as though the conversion broke.
    """
    source = frame_file(tmp_path, samples={})
    output = tmp_path / "empty.ply"

    assert main(argv(source, output)) == 0

    header, points = parse_ply(output.read_bytes())
    assert points == []
    assert squeezed(EMPTY_CLOUD_NOTE) in flattened(header)
    out = capsys.readouterr().out
    assert "vertices   0" in out
    assert "EMPTY" in out


# --- what the user is told ----------------------------------------------------


def test_the_unvalidated_unit_is_stated_before_the_first_number(tmp_path: Path, capsys):
    """A reader who stops early must already have been told what this is not."""
    source = frame_file(tmp_path)
    assert main(argv(source, tmp_path / "cloud.ply")) == 0

    out = capsys.readouterr().out
    banner = out.index("NOT METRICALLY VALIDATED")
    assert banner < out.index("points     ")
    assert banner < out.index("vertices   ")
    # And again at the end, after the limitations, because the block is long.
    assert out.count("NOT METRICALLY VALIDATED") >= 2
    assert UNIT_DISCLAIMER in out


def test_every_limitation_reaches_both_the_output_and_the_header(tmp_path: Path, capsys):
    """The caveats are the result as much as the geometry is."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"
    assert main(argv(source, output)) == 0
    out = capsys.readouterr().out

    header, _ = parse_ply(output.read_bytes())
    flat = flattened(header)
    for limitation in (*INHERENT_LIMITATIONS, *STATIC_LIMITATIONS):
        assert limitation in out, limitation
        assert squeezed(limitation) in flat, limitation
    assert "LIMITATIONS" in out


def test_the_stated_unit_travels_as_an_assumption(tmp_path: Path, capsys):
    """``--assume-unit`` is carried with an owner and a basis, never as a fact."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"
    assert main(argv(source, output, "--assume-unit", "millimetre (asserted, unproven)")) == 0

    out = capsys.readouterr().out
    assert "millimetre (asserted, unproven)" in out
    assert "--assume-unit" in out
    header, _ = parse_ply(output.read_bytes())
    flat = flattened(header)
    assert squeezed("ASSUMED, never measured") in flat
    assert squeezed("millimetre (asserted, unproven)") in flat


def test_declining_a_unit_is_what_the_default_says(tmp_path: Path, capsys):
    """Without ``--assume-unit`` the file declines to name one, and says so."""
    source = frame_file(tmp_path)
    assert main(argv(source, tmp_path / "cloud.ply")) == 0
    out = capsys.readouterr().out
    assert UNVALIDATED_UNIT.label in out
    assert UNVALIDATED_UNIT.stated_by in out


# --- the JSON report ----------------------------------------------------------


def test_json_reports_the_export_and_the_command(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    assert main(argv(source, output, "--json", "--intrinsics-sha256", FAKE_DIGEST)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert payload["mode"] == EXPORT_MODE
    assert payload["points_converted"] == payload["points_written"] == len(SAMPLES)
    assert payload["every_point_written"] is True

    command = payload["command"]
    assert command["name"] == COMMAND
    assert command["source_frame_path"] == str(source)
    assert command["output_path"] == str(output)
    assert command["dimensional_accuracy_claimed"] is False

    cloud = payload["cloud"]
    assert cloud["artifact"]["path"] == str(output)
    assert cloud["artifact"]["format"] == str(PlyFormat.BINARY_LITTLE_ENDIAN)
    assert cloud["artifact"]["byte_count"] == output.stat().st_size
    assert cloud["provenance"]["units"]["dimensional_accuracy_claimed"] is False
    assert cloud["provenance"]["calibration"]["source"] == TEST_SOURCE
    assert cloud["provenance"]["calibration"]["intrinsics_sha256"] == FAKE_DIGEST

    conversion = payload["conversion"]
    assert conversion["source_frame"]["sha256"] == cloud["provenance"]["source_frame_sha256"]
    assert conversion["unit"]["assumed"] is True
    assert conversion["unit"]["established_by_measurement"] is False
    assert conversion["pixel_accounting"]["accounting_holds"] is True

    for limitation in (*INHERENT_LIMITATIONS, *STATIC_LIMITATIONS):
        assert limitation in payload["limitations"], limitation


def test_json_output_replaces_the_human_rendering(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    assert main(argv(source, tmp_path / "cloud.ply", "--json")) == 0
    out = capsys.readouterr().out
    assert "LIMITATIONS (" not in out
    json.loads(out)


def test_json_out_writes_the_same_payload(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    report = tmp_path / "report.json"

    assert main(argv(source, tmp_path / "cloud.ply", "--json", "--json-out", str(report))) == 0

    assert json.loads(report.read_text(encoding="utf-8")) == json.loads(capsys.readouterr().out)


def test_json_out_does_not_require_json_on_stdout(tmp_path: Path, capsys):
    """The report a user reads and the report a script reads are not exclusive.

    The caveats are the reason: a run that files its JSON away must still put
    the limitations in front of the person who ran it.
    """
    source = frame_file(tmp_path)
    report = tmp_path / "report.json"

    assert main(argv(source, tmp_path / "cloud.ply", "--json-out", str(report))) == 0

    assert "LIMITATIONS" in capsys.readouterr().out
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == EXPORT_MODE
    for limitation in (*INHERENT_LIMITATIONS, *STATIC_LIMITATIONS):
        assert limitation in payload["limitations"], limitation


# --- the refusals -------------------------------------------------------------


def test_a_wrong_sized_frame_is_refused_and_nothing_is_written(tmp_path: Path, capsys):
    source = tmp_path / "short.bin"
    source.write_bytes(b"\x00" * 4096)
    output = tmp_path / "cloud.ply"

    code = main(argv(source, output))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE] == 5
    err = capsys.readouterr().err
    assert f"{EXPECTED_FRAME_BYTES}" in err
    assert "4096" in err
    assert not output.exists()


def test_an_unreadable_frame_path_is_refused_and_nothing_is_written(tmp_path: Path, capsys):
    output = tmp_path / "cloud.ply"

    code = main(argv(tmp_path / "absent.bin", output))

    assert code == EXIT_CODES[DoctorStatus.INTERNAL_ERROR] == 7
    assert "cannot read depth frame" in capsys.readouterr().err
    assert not output.exists()


def test_a_directory_is_not_a_frame(tmp_path: Path, capsys):
    output = tmp_path / "cloud.ply"
    code = main(argv(tmp_path, output))
    assert code == EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    assert "cannot read depth frame" in capsys.readouterr().err
    assert not output.exists()


def test_an_existing_output_is_never_replaced(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"
    output.write_bytes(b"an earlier run's file")

    code = main(argv(source, output))

    assert code == EXIT_CODES[DoctorStatus.PERMISSION_DENIED] == 3
    assert "refusing to overwrite" in capsys.readouterr().err
    assert output.read_bytes() == b"an earlier run's file"


def test_the_three_refusals_are_told_apart_by_exit_code():
    """A caller branching on the code must be able to tell them apart."""
    codes = {
        EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE],
        EXIT_CODES[DoctorStatus.INTERNAL_ERROR],
        EXIT_CODES[DoctorStatus.PERMISSION_DENIED],
    }
    assert len(codes) == 3
    assert 0 not in codes


def test_a_blank_calibration_source_is_refused(tmp_path: Path, capsys):
    """A blank source would read as 'no calibration', not as 'nobody said'."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(restated(argv(source, output), "--calibration-source", "   "))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "calibration source is blank" in capsys.readouterr().err
    assert not output.exists()


def test_a_malformed_intrinsics_digest_is_refused(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(argv(source, output, "--intrinsics-sha256", "not-a-digest"))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "not 64 lowercase hex characters" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("scale", ["0", "-1", "nan"])
def test_an_unusable_depth_scale_is_refused(tmp_path: Path, scale: str, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(restated(argv(source, output), "--depth-scale", scale))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "depth scale" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("flag", ["--fx", "--fy"])
def test_a_focal_length_that_cannot_be_divided_by_is_refused(tmp_path: Path, flag: str, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(restated(argv(source, output), flag, "0"))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "pinhole" in capsys.readouterr().err
    assert not output.exists()


def test_a_blank_assumed_unit_is_refused(tmp_path: Path, capsys):
    """Stating an empty unit is not the same as declining to state one."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(argv(source, output, "--assume-unit", "  "))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "blank label" in capsys.readouterr().err
    assert not output.exists()


def test_an_intrinsics_raster_the_vendor_record_cannot_hold_is_refused(tmp_path: Path, capsys):
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(argv(source, output, "--intrinsics-width", "99999"))

    assert code == EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    assert "Nothing was converted" in capsys.readouterr().err
    assert not output.exists()


def test_a_mismatched_intrinsics_raster_is_reported_rather_than_rescaled(tmp_path: Path, capsys):
    """A record for another raster still converts; the mismatch is surfaced."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"

    code = main(
        argv(source, output, "--intrinsics-width", "1280", "--intrinsics-height", "800", "--json")
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check for check in payload["conversion"]["checks"]}
    assert checks["intrinsics_frame_matches_depth_frame"]["ok"] is False
    assert payload["conversion"]["intrinsics"]["width"] == 1280


# --- the calibration is the caller's, and is required -------------------------


@pytest.mark.parametrize(
    "missing", ["--fx", "--fy", "--cx", "--cy", "--depth-scale", "--calibration-source"]
)
def test_every_calibration_input_is_required(tmp_path: Path, missing: str):
    """No default may quietly supply this unit's numbers on a user's behalf."""
    source = frame_file(tmp_path)
    full = argv(source, tmp_path / "cloud.ply")
    index = full.index(missing)
    without = full[:index] + full[index + 2 :]

    with pytest.raises(SystemExit) as caught:
        main(without)
    assert caught.value.code == 2


@pytest.mark.parametrize("name", ["fx", "fy", "cx", "cy", "depth_scale", "calibration_source"])
def test_the_parser_holds_no_calibration_default(name: str):
    """Checked on the parser too, so a default cannot be added unnoticed."""
    action = action_named(name)
    assert action.required is True
    assert action.default is None


def test_the_command_takes_no_device_arguments():
    """It resolves no scanner, so it may not even be able to name one."""
    options = {option for action in subparser()._actions for option in action.option_strings}
    assert not options & {"--serial", "--sysfs-root", "--dev-root", "--execute"}


def test_both_ply_formats_are_offered_and_binary_is_the_default():
    action = action_named("ply_format")
    assert action.choices is not None
    assert set(action.choices) == set(PlyFormat)
    assert action.default is PlyFormat.BINARY_LITTLE_ENDIAN


def test_an_unknown_ply_format_is_rejected_by_the_parser(tmp_path: Path):
    source = frame_file(tmp_path)
    with pytest.raises(SystemExit) as caught:
        main(argv(source, tmp_path / "cloud.ply", "--ply-format", "obj"))
    assert caught.value.code == 2


def test_an_abbreviated_flag_is_not_accepted(tmp_path: Path):
    """``allow_abbrev=False``: a prefix of a flag is an error, not a near miss.

    Every flag here states part of a calibration, so a prefix that happened to
    be unambiguous today would silently become a different assertion the next
    time a flag is added.
    """
    source = frame_file(tmp_path)
    with pytest.raises(SystemExit) as caught:
        main(argv(source, tmp_path / "cloud.ply", "--assume", "millimetre"))
    assert caught.value.code == 2


# --- no hardware, on any path -------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["success", "json", "ascii", "missing-frame", "wrong-size", "existing-output", "bad-scale"],
)
def test_the_command_reaches_neither_discovery_nor_the_sdk_probe(
    tmp_path: Path, case: str, monkeypatch, capsys
):
    """Offline on every argument path, refusals included.

    The two doors to hardware in this module - device discovery and the vendor
    SDK probe - are replaced with functions that fail the test if they are ever
    called, so this holds for the paths that succeed and for the paths that
    refuse.
    """

    def explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("depth-frame-to-ply must not touch hardware or the vendor SDK")

    monkeypatch.setattr(cli_module, "find_scanners", explode)
    monkeypatch.setattr(cli_module, "_maybe_run_probe", explode)
    monkeypatch.setattr(cli_module, "run_probe", explode)

    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"
    extra: Sequence[str] = ()
    expected = 0
    if case == "json":
        extra = ("--json",)
    elif case == "ascii":
        extra = ("--ply-format", "ascii")
    elif case == "missing-frame":
        source = tmp_path / "absent.bin"
        expected = EXIT_CODES[DoctorStatus.INTERNAL_ERROR]
    elif case == "wrong-size":
        source = tmp_path / "short.bin"
        source.write_bytes(b"\x00" * 10)
        expected = EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]
    elif case == "existing-output":
        output.write_bytes(b"already here")
        expected = EXIT_CODES[DoctorStatus.PERMISSION_DENIED]
    elif case == "bad-scale":
        extra = ("--depth-scale", "0")
        expected = EXIT_CODES[DoctorStatus.UNSUPPORTED_MODE]

    assert main(argv(source, output, *extra)) == expected
    capsys.readouterr()


def test_the_command_opens_no_path_but_the_two_it_was_given(tmp_path: Path, monkeypatch):
    """Every file this run touches is one the user named on the command line."""
    source = frame_file(tmp_path)
    output = tmp_path / "cloud.ply"
    opened: list[str] = []
    real_open = os.open

    def recording_open(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    assert main(argv(source, output)) == 0

    assert all(entry.startswith(str(tmp_path)) for entry in opened), opened
    assert str(output) in opened


# --- one real captured frame --------------------------------------------------

#: Where this session's captured frames were written. Deliberately not a
#: repository fixture: they are ephemeral scratch artifacts of the capture
#: tasks, so this reads one if it is still there and skips - loudly, with the
#: reason - if it is not.
SCRATCHPAD = Path(
    "/tmp/claude-1000/-home-ben-repositories-3d-scanner/"
    "147e7b71-d25d-4357-bb4a-1f6a3964c3bb/scratchpad"
)

#: An explicit override, so a real frame kept anywhere else can still be used.
FRAME_PATH_ENV = "SCANNER3D_REAL_DEPTH_FRAME"


def real_frame_path() -> Path | None:
    """One real 512000-byte capture, or ``None`` if none is on this machine."""
    override = os.environ.get(FRAME_PATH_ENV)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    if not SCRATCHPAD.is_dir():
        return None
    for candidate in sorted(SCRATCHPAD.glob("*/depth-frame-raw.bin")):
        if candidate.is_file() and candidate.stat().st_size == EXPECTED_FRAME_BYTES:
            return candidate
    return None


def test_a_real_captured_frame_goes_through_the_command(tmp_path: Path, capsys):
    """The whole path, on bytes a device really produced. Structure only.

    Nothing here asserts a distance, an extent or a dimension: no measurement
    in this repository has established any. A low-yield frame is a perfectly
    good input - the assertions are about accounting, not about yield.
    """
    path = real_frame_path()
    if path is None:
        pytest.skip(
            f"no real {EXPECTED_FRAME_BYTES}-byte depth frame found under {SCRATCHPAD}, and "
            f"${FRAME_PATH_ENV} does not name one; these captures are ephemeral scratch "
            "artifacts, so the end-to-end path from real bytes to a PLY was NOT exercised "
            "on this run. This skip is not a pass"
        )
    output = tmp_path / "real.ply"

    assert main(argv(path, output, "--json")) == 0

    payload = json.loads(capsys.readouterr().out)
    header, points = parse_ply(output.read_bytes())
    assert len(points) == payload["points_written"] == payload["points_converted"]
    accounting = payload["conversion"]["pixel_accounting"]
    assert accounting["accounting_holds"] is True
    assert accounting["samples"] == FRAME_WIDTH * FRAME_HEIGHT
    assert payload["conversion"]["source_frame"]["bytes"] == EXPECTED_FRAME_BYTES
    for limitation in INHERENT_LIMITATIONS:
        assert squeezed(limitation) in flattened(header), limitation
