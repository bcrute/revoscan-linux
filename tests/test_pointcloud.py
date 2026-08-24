"""Prove the PLY writer produces files a stranger can read, or no file at all.

Two claims matter here and neither can be settled by reading the module. The
first is that the bytes really are a PLY: every encoded cloud is parsed back by
the reader in this file, which knows the format's own rules and nothing about
:mod:`scanner3d.pointcloud`, and the binary body is additionally compared
against ``struct`` packing computed by hand. The second is that a cloud is
written whole or not at all: the fault injector misbehaves on the writer's own
descriptor - and only that one - so a short write, a stalled write and a
mid-write ``EIO`` can each be driven against a real file on a real filesystem.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import pytest

from scanner3d.pointcloud import (
    CAMERA_OPTICAL_FRAME,
    INHERENT_LIMITATIONS,
    MAX_OUTPUT_BYTES,
    MAX_POINTS,
    RAW_UNIT_LABEL,
    SCALED_UNIT_LABEL,
    SCHEMA_VERSION,
    VERTEX_BYTES,
    CalibrationProvenance,
    CloudProvenance,
    CoordinateConvention,
    NonFiniteCoordinate,
    OutputSizeExceeded,
    PlyFormat,
    PointCloudNotAllowed,
    PointCloudWriteIncomplete,
    PointCountExceeded,
    binary32,
    build_header,
    encode_ply,
    write_ply,
)

# --------------------------------------------------------------------------
# An independent PLY reader: it knows the format, not the writer
# --------------------------------------------------------------------------


class ParsedPly:
    """A PLY parsed from bytes by the format's own rules alone."""

    def __init__(self, raw: bytes) -> None:
        # The header ends at the first line whose *first token* is
        # ``end_header``, which is what a conforming reader looks for. A comment
        # that merely contains the word is not the end of anything, and this
        # parser must not pretend otherwise.
        lines: list[str] = []
        offset = 0
        while True:
            newline = raw.index(b"\n", offset)
            line = raw[offset:newline].decode("ascii")
            offset = newline + 1
            lines.append(line)
            if line.split()[:1] == ["end_header"]:
                break
        self.lines = lines
        self.header_bytes = raw[:offset]
        self.body = raw[offset:]

        assert lines[0] == "ply"
        _, self.format, self.version = lines[1].split()
        self.comments = [line[len("comment ") :] for line in lines if line.startswith("comment ")]
        self.structural = [line for line in lines if not line.startswith("comment ")]
        self.properties: list[tuple[str, str]] = []
        self.count = -1
        for line in lines:
            if line.startswith("element vertex "):
                self.count = int(line.split()[2])
            elif line.startswith("property "):
                _, kind, name = line.split()
                self.properties.append((kind, name))

    @property
    def comment_text(self) -> str:
        """Every comment as one whitespace-collapsed run.

        Long comments are wrapped across several ``comment`` lines with a
        hanging indent, so a phrase the writer emitted as one string is not one
        line in the file. Collapsing the whitespace is what lets a test assert
        on what the header *says* rather than on where it happened to wrap.
        """
        return " ".join(" ".join(self.comments).split())

    def points(self) -> list[tuple[float, float, float]]:
        parsed: list[tuple[float, float, float]] = []
        if self.format == "ascii":
            rows = self.body.decode("ascii").splitlines()
            assert len(rows) == self.count
            for row in rows:
                x, y, z = (float(value) for value in row.split())
                parsed.append((x, y, z))
            return parsed
        assert self.format == "binary_little_endian"
        assert len(self.body) == self.count * 12
        for index in range(self.count):
            x, y, z = struct.unpack_from("<fff", self.body, index * 12)
            parsed.append((x, y, z))
        return parsed


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

DEFAULT_CALIBRATION = CalibrationProvenance(
    source="factory record read from the device in scanner-007",
    intrinsics_sha256="ab" * 32,
    depth_scale=0.005,
)

SOURCE_FRAME_SHA = "cd" * 32

SAMPLE: list[tuple[float, float, float]] = [
    (0.0, 0.0, 100.0),
    (-12.5, 3.25, 101.5),
    (1e-5, -0.1, 250.75),
]


def provenance(
    *,
    units: str = SCALED_UNIT_LABEL,
    convention: CoordinateConvention = CAMERA_OPTICAL_FRAME,
    calibration: CalibrationProvenance = DEFAULT_CALIBRATION,
    source_frame_sha256: str | None = SOURCE_FRAME_SHA,
    limitations: tuple[str, ...] = (),
) -> CloudProvenance:
    return CloudProvenance(
        units=units,
        convention=convention,
        calibration=calibration,
        source_frame_sha256=source_frame_sha256,
        limitations=limitations,
    )


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ply_format", list(PlyFormat))
def test_a_cloud_round_trips_through_its_own_file(tmp_path: Path, ply_format):
    path = tmp_path / f"cloud-{ply_format}.ply"
    cloud = write_ply(path, SAMPLE, provenance=provenance(), ply_format=ply_format)

    parsed = ParsedPly(path.read_bytes())
    assert parsed.format == str(ply_format)
    assert parsed.version == "1.0"
    assert parsed.count == len(SAMPLE)
    assert parsed.properties == [("float", "x"), ("float", "y"), ("float", "z")]
    assert parsed.points() == [(binary32(x), binary32(y), binary32(z)) for x, y, z in SAMPLE]
    assert cloud.point_count == len(SAMPLE)
    assert cloud.byte_count == path.stat().st_size


def test_the_two_encodings_carry_bit_identical_coordinates():
    """A choice of format must never be a change in the data."""
    ascii_cloud = encode_ply(SAMPLE, provenance=provenance(), ply_format=PlyFormat.ASCII)
    binary_cloud = encode_ply(
        SAMPLE, provenance=provenance(), ply_format=PlyFormat.BINARY_LITTLE_ENDIAN
    )
    assert ParsedPly(ascii_cloud.payload).points() == ParsedPly(binary_cloud.payload).points()


def test_ascii_values_are_the_binary32_values_not_the_float64_ones():
    cloud = encode_ply([(0.1, 0.2, 0.3)], provenance=provenance(), ply_format=PlyFormat.ASCII)
    row = cloud.payload.decode("ascii").splitlines()[-1]
    assert [float(value) for value in row.split()] == [
        binary32(0.1),
        binary32(0.2),
        binary32(0.3),
    ]
    assert float(row.split()[0]) != 0.1


# --------------------------------------------------------------------------
# Byte-exact binary layout
# --------------------------------------------------------------------------


def test_the_binary_body_is_exactly_hand_packed_structs():
    cloud = encode_ply(SAMPLE, provenance=provenance(), ply_format=PlyFormat.BINARY_LITTLE_ENDIAN)
    expected = b"".join(struct.pack("<fff", x, y, z) for x, y, z in SAMPLE)
    assert len(expected) == len(SAMPLE) * 12
    assert VERTEX_BYTES == 12
    assert cloud.payload == cloud.header.encode("ascii") + expected


def test_a_known_vertex_has_the_exact_expected_bytes():
    """1.0 is 0x3f800000; little-endian that is 00 00 80 3f, and nothing else."""
    cloud = encode_ply(
        [(1.0, -2.0, 0.0)], provenance=provenance(), ply_format=PlyFormat.BINARY_LITTLE_ENDIAN
    )
    body = cloud.payload[len(cloud.header.encode("ascii")) :]
    assert body == bytes.fromhex("0000803f000000c000000000")


def test_the_binary_body_starts_immediately_after_end_header():
    cloud = encode_ply(SAMPLE, provenance=provenance())
    marker = b"end_header\n"
    index = cloud.payload.index(marker) + len(marker)
    assert len(cloud.payload) - index == len(SAMPLE) * VERTEX_BYTES


# --------------------------------------------------------------------------
# Degenerate clouds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ply_format", list(PlyFormat))
def test_an_empty_cloud_is_a_valid_file_with_zero_vertices(tmp_path: Path, ply_format):
    path = tmp_path / "empty.ply"
    cloud = write_ply(path, [], provenance=provenance(), ply_format=ply_format)
    parsed = ParsedPly(path.read_bytes())
    assert parsed.count == 0
    assert parsed.body == b""
    assert parsed.points() == []
    assert cloud.point_count == 0


@pytest.mark.parametrize("ply_format", list(PlyFormat))
def test_a_single_point_round_trips(tmp_path: Path, ply_format):
    path = tmp_path / "one.ply"
    write_ply(path, [(1.5, -2.5, 3.5)], provenance=provenance(), ply_format=ply_format)
    parsed = ParsedPly(path.read_bytes())
    assert parsed.count == 1
    assert parsed.points() == [(1.5, -2.5, 3.5)]


def test_points_may_be_any_sequence_of_three_floats():
    cloud = encode_ply([[1.0, 2.0, 3.0]], provenance=provenance())
    assert ParsedPly(cloud.payload).points() == [(1.0, 2.0, 3.0)]


def test_a_point_that_is_not_a_triple_is_refused():
    with pytest.raises(PointCloudNotAllowed, match="not the 3"):
        encode_ply([(1.0, 2.0)], provenance=provenance())


# --------------------------------------------------------------------------
# Non-finite coordinates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_non_finite_coordinates_are_refused(bad, axis):
    point = [1.0, 2.0, 3.0]
    point[axis] = bad
    with pytest.raises(NonFiniteCoordinate, match="not finite"):
        encode_ply([point], provenance=provenance())


def test_a_float64_that_overflows_binary32_is_refused():
    """1e39 is an ordinary float64 and an infinity in a PLY 'float' property."""
    with pytest.raises(NonFiniteCoordinate, match="overflows the binary32"):
        encode_ply([(1e39, 0.0, 0.0)], provenance=provenance())


def test_an_integer_too_large_to_be_a_float_is_refused():
    with pytest.raises(NonFiniteCoordinate, match="not a usable float"):
        encode_ply([(10**400, 0.0, 0.0)], provenance=provenance())


def test_a_non_finite_cloud_leaves_no_file(tmp_path: Path):
    path = tmp_path / "never.ply"
    with pytest.raises(NonFiniteCoordinate):
        write_ply(path, [(0.0, 0.0, 0.0), (math.nan, 0.0, 0.0)], provenance=provenance())
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_the_binary32_edges_are_accepted():
    largest = struct.unpack("<f", struct.pack("<f", 3.4028234663852886e38))[0]
    cloud = encode_ply([(largest, -largest, 0.0)], provenance=provenance())
    assert ParsedPly(cloud.payload).points() == [(largest, -largest, 0.0)]


# --------------------------------------------------------------------------
# The bounds
# --------------------------------------------------------------------------


def test_the_point_ceiling_is_enforced():
    with pytest.raises(PointCountExceeded, match="more than 2 points"):
        encode_ply(SAMPLE, provenance=provenance(), max_points=2)


def test_the_point_ceiling_stops_an_unbounded_producer():
    """The cap must stop the iterable, not merely count what it produced."""
    produced = 0

    def forever():
        nonlocal produced
        while True:
            produced += 1
            yield (0.0, 0.0, 1.0)

    with pytest.raises(PointCountExceeded):
        encode_ply(forever(), provenance=provenance(), max_points=10)
    assert produced == 11


def test_the_module_point_ceiling_cannot_be_widened():
    with pytest.raises(PointCloudNotAllowed, match="never widened"):
        encode_ply([], provenance=provenance(), max_points=MAX_POINTS + 1)


def test_the_byte_ceiling_is_enforced_against_the_whole_file():
    exact = encode_ply(SAMPLE, provenance=provenance()).byte_count
    encode_ply(SAMPLE, provenance=provenance(), max_output_bytes=exact)
    with pytest.raises(OutputSizeExceeded, match=f"past the {exact - 1}-byte ceiling"):
        encode_ply(SAMPLE, provenance=provenance(), max_output_bytes=exact - 1)


def test_the_module_byte_ceiling_cannot_be_widened():
    with pytest.raises(PointCloudNotAllowed, match="never widened"):
        encode_ply([], provenance=provenance(), max_output_bytes=MAX_OUTPUT_BYTES + 1)


def test_a_cloud_over_the_byte_ceiling_leaves_no_file(tmp_path: Path):
    path = tmp_path / "big.ply"
    exact = encode_ply(SAMPLE, provenance=provenance()).byte_count
    with pytest.raises(OutputSizeExceeded):
        write_ply(path, SAMPLE, provenance=provenance(), max_output_bytes=exact - 1)
    assert not path.exists()


def test_the_documented_ceilings_are_the_ones_in_force():
    """The bounds are quoted in the module docstring; check them once, cheaply."""
    assert MAX_POINTS == 8 * 640 * 400
    assert MAX_OUTPUT_BYTES == 128 * 1024 * 1024


def test_a_header_inflated_past_its_ceiling_is_refused_not_truncated():
    fat = tuple(f"limitation number {index}: " + "x" * 400 for index in range(40))
    with pytest.raises(PointCloudNotAllowed, match="refused rather than truncated"):
        encode_ply([], provenance=provenance(limitations=fat))


# --------------------------------------------------------------------------
# The header says what a bare PLY cannot
# --------------------------------------------------------------------------


def test_the_header_lines_are_in_the_order_the_format_requires():
    cloud = encode_ply(SAMPLE, provenance=provenance())
    lines = cloud.header.splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format binary_little_endian 1.0"
    assert lines[-5:] == [
        "element vertex 3",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    assert cloud.header.endswith("end_header\n")
    assert "\r" not in cloud.header
    assert all(line.startswith("comment ") for line in lines[2:-5])


def test_the_header_records_the_units_and_the_convention():
    parsed = ParsedPly(encode_ply(SAMPLE, provenance=provenance()).payload)
    text = parsed.comment_text
    assert "units:" in text
    assert "NOT established" in text
    assert "coordinate convention:" in text
    assert CAMERA_OPTICAL_FRAME.name in text
    assert "right-handed" in text
    assert "+z away from the camera" in text


def test_the_header_records_provenance_and_claims_no_accuracy():
    parsed = ParsedPly(encode_ply(SAMPLE, provenance=provenance()).payload)
    text = parsed.comment_text
    assert f"source frame sha256: {SOURCE_FRAME_SHA}" in text
    assert "scanner-007" in text
    assert f"intrinsics sha256: {'ab' * 32}" in text
    assert "depth scale: 0.005" in text
    assert "dimensional accuracy: none is claimed" in text


def test_a_cloud_with_no_calibration_behind_it_says_so():
    """The absent case must be stated, not left to a missing comment."""
    uncalibrated = provenance(
        units=RAW_UNIT_LABEL,
        calibration=CalibrationProvenance.none_applied("no intrinsics were read for this run"),
        source_frame_sha256=None,
    )
    text = ParsedPly(encode_ply(SAMPLE, provenance=uncalibrated).payload).comment_text
    assert "raw device depth sample units with no scalar applied" in text
    assert "none applied: no intrinsics were read for this run" in text
    assert "intrinsics sha256: none recorded" in text
    assert "depth scale: none applied" in text
    assert "source frame sha256: none recorded" in text


def test_every_inherent_limitation_reaches_the_header():
    parsed = ParsedPly(encode_ply(SAMPLE, provenance=provenance()).payload)
    # The header wraps, so compare on collapsed whitespace rather than by line.
    text = " ".join(parsed.comment_text.split())
    for limitation in INHERENT_LIMITATIONS:
        assert " ".join(limitation.split()) in text


def test_a_caller_cannot_drop_the_inherent_limitations():
    stated = provenance(limitations=("this cloud is a fixture",)).all_limitations
    assert stated[: len(INHERENT_LIMITATIONS)] == INHERENT_LIMITATIONS
    assert stated[-1] == "this cloud is a fixture"


def test_a_restated_limitation_is_not_repeated():
    stated = provenance(limitations=(INHERENT_LIMITATIONS[0],)).all_limitations
    assert stated == INHERENT_LIMITATIONS


def test_calibration_caveats_reach_the_stated_limitations():
    calibration = CalibrationProvenance(
        source="none", caveats=("the intrinsics were not read from this unit",)
    )
    stated = provenance(calibration=calibration).all_limitations
    assert stated[-1] == "the intrinsics were not read from this unit"


def test_a_newline_in_provenance_cannot_forge_a_header_line():
    """A caller's string is data; it must not be able to add a header line."""
    hostile = "millimetres\nend_header\nproperty float nx"
    cloud = encode_ply(SAMPLE, provenance=provenance(units=hostile), ply_format=PlyFormat.ASCII)
    parsed = ParsedPly(cloud.payload)
    assert parsed.structural == [
        "ply",
        "format ascii 1.0",
        "element vertex 3",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    assert "millimetres end_header property float nx" in parsed.comment_text
    assert parsed.points() == [(binary32(x), binary32(y), binary32(z)) for x, y, z in SAMPLE]


def test_non_ascii_provenance_does_not_break_the_ascii_header():
    cloud = encode_ply(SAMPLE, provenance=provenance(units="metres—probably not"))
    cloud.header.encode("ascii")
    assert "metres probably not" in ParsedPly(cloud.payload).comment_text


def test_the_header_is_the_same_for_both_encodings_apart_from_the_format_line():
    ascii_header = build_header(PlyFormat.ASCII, 3, provenance()).splitlines()
    binary_header = build_header(PlyFormat.BINARY_LITTLE_ENDIAN, 3, provenance()).splitlines()
    assert ascii_header[1] == "format ascii 1.0"
    assert binary_header[1] == "format binary_little_endian 1.0"
    assert ascii_header[2:] == binary_header[2:]


# --------------------------------------------------------------------------
# The companion metadata
# --------------------------------------------------------------------------


def test_the_companion_metadata_is_serialisable_and_complete(tmp_path: Path):
    path = tmp_path / "cloud.ply"
    cloud = write_ply(path, SAMPLE, provenance=provenance())
    report = cloud.as_dict()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["point_count"] == 3
    artifact = report["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["path"] == str(path)
    assert artifact["written"] is True
    assert artifact["format"] == "binary_little_endian"
    assert artifact["byte_count"] == path.stat().st_size
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    prov = report["provenance"]
    assert isinstance(prov, dict)
    assert prov["units"] == {"label": SCALED_UNIT_LABEL, "dimensional_accuracy_claimed": False}
    assert prov["coordinate_convention"] == CAMERA_OPTICAL_FRAME.as_dict()
    assert prov["source_frame_sha256"] == SOURCE_FRAME_SHA
    assert prov["calibration"]["depth_scale"] == 0.005
    assert "unknown" in prov["calibration"]["depth_scale_unit"]
    assert prov["limitations"][: len(INHERENT_LIMITATIONS)] == list(INHERENT_LIMITATIONS)

    assert report["ply_header"] == cloud.header
    # It must survive the trip a caller will actually make it take.
    assert json.loads(json.dumps(report)) == report


def test_an_unwritten_cloud_names_no_path():
    report = encode_ply(SAMPLE, provenance=provenance()).as_dict()
    artifact = report["artifact"]
    assert isinstance(artifact, dict)
    assert artifact["path"] is None
    assert artifact["written"] is False


def test_the_metadata_reports_the_bounds_that_were_in_force():
    cloud = encode_ply(SAMPLE, provenance=provenance(), max_points=99)
    bounds = cloud.as_dict()["bounds"]
    assert isinstance(bounds, dict)
    assert bounds["max_points"] == 99
    assert bounds["module_max_points"] == MAX_POINTS
    assert bounds["module_max_output_bytes"] == MAX_OUTPUT_BYTES


def test_the_metadata_sits_beside_the_geometry_and_never_replaces_it():
    cloud = encode_ply(SAMPLE, provenance=provenance())
    report = cloud.as_dict()
    assert "payload" not in report
    artifact = report["artifact"]
    assert isinstance(artifact, dict)
    assert set(artifact) == {"path", "written", "format", "byte_count", "sha256"}


# --------------------------------------------------------------------------
# Complete-or-fail writes
# --------------------------------------------------------------------------


class WriteFaultInjector:
    """Fail or shorten writes to one target file only.

    ``os.write`` is global and pytest uses it for its own terminal output, so a
    blanket patch deadlocks the run. This watches ``os.open`` for the exact
    target path and misbehaves only on that descriptor; every other write is
    passed straight through.
    """

    def __init__(self, target: Path, behaviour: str) -> None:
        self.target = target
        self.behaviour = behaviour
        self.real_open = os.open
        self.real_write = os.write
        self.real_close = os.close
        self.watched: set[int] = set()
        self.calls = 0

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(os, "open", self._open)
        monkeypatch.setattr(os, "write", self._write)
        monkeypatch.setattr(os, "close", self._close)

    def _open(self, path, flags, mode=0o777, **kwargs):
        fd = self.real_open(path, flags, mode, **kwargs)
        if Path(path) == self.target:
            self.watched.add(fd)
        return fd

    def _close(self, fd: int) -> None:
        # Descriptor numbers are reused, so a watched fd must stop being watched
        # the moment it is closed or the next unrelated file inherits the fault.
        self.watched.discard(fd)
        self.real_close(fd)

    def _write(self, fd: int, data) -> int:
        if fd not in self.watched:
            return self.real_write(fd, data)
        self.calls += 1
        payload = bytes(data)
        if self.behaviour == "stall":
            return 0
        if self.behaviour == "short":
            return self.real_write(fd, payload[:1])
        if self.calls == 1:
            return self.real_write(fd, payload[:2])
        raise OSError(errno.EIO, "synthetic failure")


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "partial.ply"
    WriteFaultInjector(path, "fail_after_partial").install(monkeypatch)
    with pytest.raises(OSError, match="synthetic failure"):
        write_ply(path, SAMPLE, provenance=provenance())
    monkeypatch.undo()
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_zero_progress_write_is_an_error_not_a_spin(tmp_path: Path, monkeypatch):
    path = tmp_path / "stall.ply"
    injector = WriteFaultInjector(path, "stall")
    injector.install(monkeypatch)
    with pytest.raises(PointCloudWriteIncomplete) as caught:
        write_ply(path, SAMPLE, provenance=provenance())
    monkeypatch.undo()
    assert caught.value.errno == errno.EIO
    assert injector.calls == 1
    assert not path.exists()


def test_a_short_write_is_resumed_from_the_kernel_offset(tmp_path: Path, monkeypatch):
    path = tmp_path / "slow.ply"
    expected = encode_ply([(1.0, 2.0, 3.0)], provenance=provenance())
    injector = WriteFaultInjector(path, "short")
    injector.install(monkeypatch)
    written = write_ply(path, [(1.0, 2.0, 3.0)], provenance=provenance())
    monkeypatch.undo()
    assert path.read_bytes() == expected.payload
    assert injector.calls == len(expected.payload)
    assert written.sha256 == expected.sha256


def test_an_existing_name_is_never_overwritten(tmp_path: Path):
    path = tmp_path / "taken.ply"
    path.write_bytes(b"someone else's evidence")
    with pytest.raises(PointCloudWriteIncomplete) as caught:
        write_ply(path, SAMPLE, provenance=provenance())
    assert caught.value.errno == errno.EEXIST
    assert path.read_bytes() == b"someone else's evidence"


def test_a_symlink_is_not_followed_out_of_the_directory(tmp_path: Path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"untouched")
    link = tmp_path / "link.ply"
    link.symlink_to(outside)
    with pytest.raises(OSError):
        write_ply(link, SAMPLE, provenance=provenance())
    assert outside.read_bytes() == b"untouched"


def test_the_written_file_is_private_to_its_owner(tmp_path: Path):
    path = tmp_path / "mode.ply"
    write_ply(path, SAMPLE, provenance=provenance())
    assert path.stat().st_mode & 0o777 == 0o600
