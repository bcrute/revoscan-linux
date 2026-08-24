"""Tests for the seam between the depth deprojection and the PLY writer.

Nothing here touches hardware, opens a device node or reads a device. Every
input - the frame bytes, the intrinsics record, the depth scale, the assumed
unit - is handed to the code under test, and every output is a file in a
``tmp_path``.

What is being defended is not arithmetic. :mod:`scanner3d.depth_metric` and
:mod:`scanner3d.pointcloud` are each careful on their own; the failure this
module exists to prevent happens *between* them, when a translation between two
records that happen to share a name quietly drops the field one of them has and
the other does not. So the assertions here are almost all about survival:

* that a real captured frame goes all the way to a PLY a parser can read, in
  both encodings, and that the two encodings decode to identical coordinates;
* that the ``note`` on depth_metric's coordinate convention - the caveat saying
  the frame is the one that module *applies* and not one the device declared -
  is in the written header, not lost to a target record with no field for it;
* that **every** limitation the conversion produced is in the header too, and
  that the caller's stated unit arrives labelled as an assumption;
* that the file's provenance digest is the digest of the bytes that produced it;
* that two exports of one conversion are byte-identical;
* that bad input is refused with a typed error and leaves no file behind;
* and that a conversion which produced no point at all still writes a valid PLY
  that says so, rather than crashing or writing a cloud of nothing.

Not one assertion in this file is about dimensional or metric accuracy, and
none can be: no physical measurement in this repository has validated any
distance it produces, and the depth scale's unit is explicitly unproven.
"""

from __future__ import annotations

import functools
import os
import struct
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from scanner3d.cloud_export import (
    CAMERA_FRAME_NAME,
    EMPTY_CLOUD_NOTE,
    EXPORT_MODE,
    SCHEMA_VERSION,
    UNRECOGNISED_FRAME_NAME,
    CalibrationSourceNotStated,
    CloudExport,
    CloudExportNotAllowed,
    ConversionAccountingUnusable,
    IntrinsicsDigestMalformed,
    SourceFrameMismatch,
    camera_points,
    cloud_calibration,
    cloud_convention,
    cloud_provenance_for,
    convention_caveat,
    ensure_calibration_source_stated,
    ensure_conversion_accounted,
    ensure_frame_matches,
    ensure_intrinsics_digest_usable,
    export_depth_frame,
    units_label,
    write_conversion_ply,
)
from scanner3d.depth_metric import (
    CAMERA_FRAME,
    UNVALIDATED_UNIT,
    CoordinateConvention,
    DepthConversion,
    DepthFrameSizeMismatch,
    DepthUnitAssumption,
    DepthUnitNotStated,
    deproject_depth_frame,
)
from scanner3d.depth_scale import UNIT_DISCLAIMER
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.intrinsics import INTRINSICS_FORMAT, IntrinsicsRecord, parse_intrinsics
from scanner3d.pointcloud import (
    INHERENT_LIMITATIONS,
    MAX_HEADER_BYTES,
    NonFiniteCoordinate,
    PlyFormat,
    PointCloudNotAllowed,
    PointCountExceeded,
    binary32,
    sha256_of,
)
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

# --- inputs the tests build ---------------------------------------------------

#: The factory numbers this unit actually returned, quoted in the *test*: no
#: module under test carries them, and injecting them here is how that stays
#: true. Their off-centre ``cx`` is the reason several caveats below exist.
REAL_FX = 898.6829833984375
REAL_FY = 898.6829833984375
REAL_CX = 42.43747329711914
REAL_CY = 192.8326873779297

#: ``float32(1.0f / 200)`` as the depth-scale read decoded it. What it measures
#: is established by nothing, which is the whole point of the unit plumbing.
REAL_SCALE = 0.004999999888241291

#: A source string a test can pass without pretending a calibration happened.
TEST_SOURCE = "tests/test_cloud_export.py: an injected record, read from no device"

#: A unit assumption a test can pass without pretending it means anything.
TEST_UNIT = DepthUnitAssumption(
    label="test depth unit",
    stated_by="tests/test_cloud_export.py",
    basis="an arbitrary label chosen so the plumbing can be checked; it measures nothing",
)

#: A syntactically valid digest that identifies nothing.
FAKE_DIGEST = "0" * 64


def record(
    *,
    fx: float = 100.0,
    fy: float = 100.0,
    cx: float = 320.0,
    cy: float = 200.0,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    one22: float = 1.0,
) -> IntrinsicsRecord:
    """A record built through the real 40-byte parser, not by hand.

    Defaults are round and centred so the deprojection is exact in binary
    floating point. They are not this unit's numbers.
    """
    blob = struct.pack(INTRINSICS_FORMAT, width, height, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, one22)
    parsed = parse_intrinsics(blob)
    assert parsed is not None
    return parsed


def real_record() -> IntrinsicsRecord:
    """The unit's own factory record, injected by the test."""
    return record(fx=REAL_FX, fy=REAL_FY, cx=REAL_CX, cy=REAL_CY)


def frame(samples: dict[tuple[int, int], int]) -> bytes:
    """A whole 512000-byte frame that is zero except at the named ``(u, v)``."""
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    for (u, v), value in samples.items():
        struct.pack_into("<H", buffer, (v * FRAME_WIDTH + u) * 2, value)
    return bytes(buffer)


def convert(raw: bytes, *, intrinsics: IntrinsicsRecord | None = None) -> DepthConversion:
    """The conversion the seam is asked to carry, with test inputs."""
    return deproject_depth_frame(
        raw,
        intrinsics=intrinsics if intrinsics is not None else record(),
        depth_scale=0.5,
        unit=TEST_UNIT,
    )


def export(path: Path, conversion: DepthConversion, **kwargs: object) -> CloudExport:
    """``write_conversion_ply`` with the test's own required provenance."""
    kwargs.setdefault("calibration_source", TEST_SOURCE)
    return write_conversion_ply(path, conversion, **kwargs)  # type: ignore[arg-type]


# --- reading the file back ----------------------------------------------------


def squeezed(text: str) -> str:
    """``text`` with every run of whitespace removed."""
    return "".join(text.split())


def flattened(header: str) -> str:
    """The header's comment text with all whitespace removed.

    ``build_header`` wraps comments at 70 columns, indents continuations by two
    spaces, and will break a hyphenated word across two lines when one lands on
    the boundary (the written header really does contain ``invalid-`` /
    ``sample``). A substring check against the original sentence therefore has
    to ignore whitespace entirely rather than re-flow it.
    """
    return squeezed("".join(line.removeprefix("comment ") for line in header.splitlines()))


def parse_ply(data: bytes) -> tuple[str, list[tuple[float, ...]]]:
    """A deliberately small PLY reader: header text plus the decoded vertices.

    Written here rather than imported so the file is checked by something that
    did not produce it.
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


def expected_points(conversion: DepthConversion) -> list[tuple[float, ...]]:
    """The conversion's points at the width a PLY ``float`` actually carries."""
    return [(binary32(p.x), binary32(p.y), binary32(p.z)) for p in conversion.points]


# --- one real captured frame --------------------------------------------------

#: Where this session's captured frames were written. Deliberately not a
#: repository fixture: these are ephemeral scratch artifacts of the capture
#: tasks, not committed evidence, so the tests below read one if it is still
#: there and skip - loudly, with the reason - if it is not.
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


def real_frame() -> bytes:
    """The bytes of one real capture, or an explicit skip. A skip is not a pass."""
    path = real_frame_path()
    if path is None:
        pytest.skip(
            f"no real {EXPECTED_FRAME_BYTES}-byte depth frame found under {SCRATCHPAD}, and "
            f"${FRAME_PATH_ENV} does not name one; these captures are ephemeral scratch "
            "artifacts, so the end-to-end path from real bytes to a PLY was NOT exercised on "
            "this run. This skip is not a pass"
        )
    return path.read_bytes()


@functools.cache
def real_conversion() -> DepthConversion:
    """One deprojection of the real frame, reused: it is 256000 samples of work."""
    return deproject_depth_frame(
        real_frame(), intrinsics=real_record(), depth_scale=REAL_SCALE, unit=UNVALIDATED_UNIT
    )


def test_a_real_captured_frame_reaches_a_parseable_binary_ply(tmp_path):
    """The whole path: real bytes in, a file on disk a parser can read back.

    Structure only. Nothing here asserts a distance, an extent or a dimension,
    because no measurement in this repository has established any.
    """
    raw = real_frame()
    path = tmp_path / "real-binary.ply"

    result = export_depth_frame(
        path,
        raw,
        intrinsics=real_record(),
        depth_scale=REAL_SCALE,
        unit=UNVALIDATED_UNIT,
        calibration_source=TEST_SOURCE,
        intrinsics_sha256=FAKE_DIGEST,
    )

    assert path.is_file()
    data = path.read_bytes()
    header, points = parse_ply(data)

    assert result.point_count == result.conversion.included > 0
    assert len(points) == result.point_count
    assert points == expected_points(result.conversion)
    assert header == result.header
    assert sha256_of(data) == result.cloud.sha256
    assert result.cloud.byte_count == len(data)
    assert result.every_point_written
    assert result.path == str(path)


def test_a_real_captured_frame_reaches_a_parseable_ascii_ply(tmp_path):
    """The same path in the other encoding, read back by the same parser."""
    raw = real_frame()
    path = tmp_path / "real-ascii.ply"

    result = export_depth_frame(
        path,
        raw,
        intrinsics=real_record(),
        depth_scale=REAL_SCALE,
        unit=UNVALIDATED_UNIT,
        calibration_source=TEST_SOURCE,
        ply_format=PlyFormat.ASCII,
    )

    header, points = parse_ply(path.read_bytes())
    assert "format ascii 1.0" in header
    assert len(points) == result.point_count > 0
    assert points == expected_points(result.conversion)


def test_the_two_encodings_of_one_real_frame_decode_identically(tmp_path):
    """A choice of file format must not be a change in the data."""
    conversion = real_conversion()
    binary = export(tmp_path / "b.ply", conversion, ply_format=PlyFormat.BINARY_LITTLE_ENDIAN)
    ascii_ = export(tmp_path / "a.ply", conversion, ply_format=PlyFormat.ASCII)

    _, binary_points = parse_ply((tmp_path / "b.ply").read_bytes())
    _, ascii_points = parse_ply((tmp_path / "a.ply").read_bytes())

    assert binary_points == ascii_points
    assert binary.point_count == ascii_.point_count
    # Only the format token and the body differ; every comment is the same.
    assert binary.header.replace("binary_little_endian", "ascii") == ascii_.header


def test_the_real_frames_header_stays_inside_the_ceiling(tmp_path):
    """Every caveat this seam adds still fits the header bound, on real input."""
    result = export(tmp_path / "c.ply", real_conversion())
    assert len(result.header.encode("ascii")) <= MAX_HEADER_BYTES


def test_provenance_digest_is_the_digest_of_the_source_bytes(tmp_path):
    """The file's provenance points at the frame that actually produced it."""
    raw = real_frame()
    path = tmp_path / "hashed.ply"
    result = export_depth_frame(
        path,
        raw,
        intrinsics=real_record(),
        depth_scale=REAL_SCALE,
        unit=UNVALIDATED_UNIT,
        calibration_source=TEST_SOURCE,
    )

    digest = sha256_of(raw)
    assert result.conversion.frame_sha256 == digest
    assert result.cloud.provenance.source_frame_sha256 == digest
    assert digest in flattened(result.header)


def test_two_exports_of_one_conversion_are_byte_identical(tmp_path):
    """Determinism, including the order the points are written in."""
    conversion = real_conversion()
    first = export(tmp_path / "one.ply", conversion)
    second = export(tmp_path / "two.ply", conversion)

    assert first.cloud.sha256 == second.cloud.sha256
    assert (tmp_path / "one.ply").read_bytes() == (tmp_path / "two.ply").read_bytes()


def test_camera_points_preserve_the_conversions_own_order():
    """The writer is handed the raster order the conversion built, unchanged."""
    conversion = convert(frame({(600, 1): 10, (5, 0): 20, (300, 1): 30, (639, 399): 40}))
    assert list(camera_points(conversion)) == [(p.x, p.y, p.z) for p in conversion.points]


# --- the caveat that has no field to live in ----------------------------------


def test_the_convention_note_survives_into_the_written_header(tmp_path):
    """The caveat the target record has no field for is in the file anyway.

    This is the whole reason the module exists: a field-by-field translation
    between the two ``CoordinateConvention`` records drops ``note``, and the
    note is the sentence saying the device never declared this frame.
    """
    result = export(tmp_path / "note.ply", convert(frame({(1, 1): 4})))

    written_header, _ = parse_ply((tmp_path / "note.ply").read_bytes())
    assert CAMERA_FRAME.note not in ("", None)
    assert squeezed(CAMERA_FRAME.note) in flattened(result.header)
    assert squeezed(CAMERA_FRAME.note) in flattened(written_header)
    # And it is carried as what it is: a statement of what is not known.
    assert convention_caveat(CAMERA_FRAME) in result.limitations


def test_the_note_is_not_smuggled_into_an_axis_field():
    """The line a consumer parses for axis semantics stays parseable."""
    converted = cloud_convention(CAMERA_FRAME)
    for field_text in (
        converted.name,
        converted.x_axis,
        converted.y_axis,
        converted.z_axis,
        converted.origin,
        converted.handedness,
    ):
        assert CAMERA_FRAME.note not in field_text


def test_the_convention_translation_is_faithful_not_a_paraphrase():
    """Every axis keeps depth_metric's own words, minus the duplicated marker."""
    converted = cloud_convention(CAMERA_FRAME)

    assert converted.name == CAMERA_FRAME_NAME
    assert converted.origin == CAMERA_FRAME.origin
    assert converted.handedness == CAMERA_FRAME.handedness
    assert converted.x_axis == CAMERA_FRAME.x_axis.removeprefix("+x ")
    assert converted.y_axis == CAMERA_FRAME.y_axis.removeprefix("+y ")
    assert converted.z_axis == CAMERA_FRAME.z_axis.removeprefix("+z ")
    # The marker really was there, so the strip is doing something.
    assert CAMERA_FRAME.x_axis.startswith("+x ")
    # ... and the summary the header quotes does not now say it twice.
    assert "+x +x" not in converted.summary
    assert "+y +y" not in converted.summary
    assert "+z +z" not in converted.summary


def test_an_unrecognised_convention_gets_a_name_that_claims_nothing():
    """This module cannot check that some other frame is the optical frame."""
    other = CoordinateConvention(
        origin="somewhere else",
        x_axis="+x left",
        y_axis="+y up",
        z_axis="+z back",
        handedness="left-handed",
        note="a frame this repository has never seen",
    )
    converted = cloud_convention(other)
    assert converted.name == UNRECOGNISED_FRAME_NAME
    assert converted.name != CAMERA_FRAME_NAME
    assert converted.x_axis == "left"
    assert converted.handedness == "left-handed"
    assert convention_caveat(other).endswith(other.note)


def test_the_written_axis_text_reaches_the_header(tmp_path):
    """Not just the record: the axis words are in the file's comments."""
    result = export(tmp_path / "axes.ply", convert(frame({(2, 2): 7})))
    flat = flattened(result.header)
    for text in (
        CAMERA_FRAME.origin,
        CAMERA_FRAME.handedness,
        CAMERA_FRAME.x_axis.removeprefix("+x "),
        CAMERA_FRAME.y_axis.removeprefix("+y "),
        CAMERA_FRAME.z_axis.removeprefix("+z "),
    ):
        assert squeezed(text) in flat


# --- every limitation, and the unit -------------------------------------------


def test_every_conversion_limitation_reaches_the_written_file(tmp_path):
    """A PLY that loses the uncertainty is worse than no PLY."""
    conversion = real_conversion()
    path = tmp_path / "limits.ply"
    result = export(path, conversion)

    written_header, _ = parse_ply(path.read_bytes())
    flat = flattened(written_header)
    assert conversion.limitations
    for limitation in conversion.limitations:
        assert limitation in result.limitations
        assert squeezed(limitation) in flat


def test_the_off_centre_and_frame_mismatch_limitations_reach_the_file(tmp_path):
    """The two per-run limitations, forced to occur, then found in the header."""
    odd = record(cx=REAL_CX, cy=REAL_CY, width=320, height=200)
    conversion = convert(frame({(3, 3): 9}), intrinsics=odd)
    result = export(tmp_path / "odd.ply", conversion)

    # Both dynamic limitations really did fire on this input.
    assert len(conversion.limitations) > 2
    flat = flattened(result.header)
    for limitation in conversion.limitations:
        assert squeezed(limitation) in flat


def test_the_inherent_limitations_cannot_be_dropped_by_coming_through_here(tmp_path):
    """The writer's own standing caveats are still prepended, untouched."""
    result = export(tmp_path / "inherent.ply", convert(frame({(4, 4): 11})))
    flat = flattened(result.header)
    for limitation in INHERENT_LIMITATIONS:
        assert limitation in result.limitations
        assert squeezed(limitation) in flat


def test_extra_limitations_are_added_and_never_substituted(tmp_path):
    """A caller with something of its own to say does not displace the rest."""
    conversion = convert(frame({(5, 5): 12}))
    mine = "this cloud was written by a test and describes nothing physical"
    result = export(tmp_path / "extra.ply", conversion, extra_limitations=(mine,))

    assert mine in result.limitations
    for limitation in (*INHERENT_LIMITATIONS, *conversion.limitations):
        assert limitation in result.limitations
    assert squeezed(mine) in flattened(result.header)


def test_the_assumed_unit_reaches_the_header_labelled_as_an_assumption(tmp_path):
    """All three strings of the assumption travel, and the word 'assumed' leads."""
    conversion = convert(frame({(6, 6): 13}))
    result = export(tmp_path / "unit.ply", conversion)

    label = result.cloud.provenance.units
    assert label.startswith("ASSUMED, never measured:")
    for text in (TEST_UNIT.label, TEST_UNIT.stated_by, TEST_UNIT.basis):
        assert text in label
        assert squeezed(text) in flattened(result.header)
    assert result.cloud.provenance.as_dict()["units"] == {
        "label": label,
        "dimensional_accuracy_claimed": False,
    }


def test_the_declined_unit_says_so_in_the_header(tmp_path):
    """``UNVALIDATED_UNIT`` is a refusal to name a unit, and it stays one."""
    result = export(tmp_path / "declined.ply", real_conversion())
    flat = flattened(result.header)
    assert squeezed(UNVALIDATED_UNIT.basis) in flat
    assert squeezed(UNIT_DISCLAIMER) in flat


def test_units_label_carries_the_scale_and_the_deprojection():
    """A reader can see what was multiplied by what, without the source."""
    label = units_label(TEST_UNIT, 0.25)
    assert "depth_scale=0.25" in label
    assert "z = d" in label


# --- what the conversion observed and did not act on --------------------------


def test_the_principal_point_placement_is_always_stated(tmp_path):
    """Even when it is unremarkable: an absent warning is not a statement."""
    conversion = convert(frame({(7, 7): 14}))
    assert conversion.principal_point.near_centre  # the centred test record
    result = export(tmp_path / "pp.ply", conversion)

    assert squeezed(conversion.principal_point.note) in flattened(result.header)


def test_a_failed_structural_check_reaches_the_file_although_no_limitation_names_it(tmp_path):
    """The seam adds the one observation the conversion's limitations omit."""
    conversion = convert(frame({(8, 8): 15}), intrinsics=record(one22=2.0))
    assert not conversion.intrinsics.structurally_expected
    failed = [check for check in conversion.checks if not check.ok]
    assert [check.name for check in failed] == ["intrinsics_structurally_expected"]
    # The conversion itself never mentions it, so without the seam it is lost.
    assert not any(squeezed(failed[0].detail) in squeezed(x) for x in conversion.limitations)

    result = export(tmp_path / "struct.ply", conversion)
    assert squeezed(failed[0].detail) in flattened(result.header)


def test_a_repeated_observation_is_not_said_twice(tmp_path):
    """The off-centre check's detail *is* the principal point note verbatim."""
    conversion = convert(frame({(9, 9): 16}), intrinsics=real_record())
    calibration = cloud_calibration(conversion, source=TEST_SOURCE)
    note = conversion.principal_point.note

    assert sum(1 for caveat in calibration.caveats if note in caveat) == 1
    result = export(tmp_path / "dup.ply", conversion)
    assert flattened(result.header).count(squeezed(note)) == 1


def test_the_exclusion_accounting_reaches_the_header(tmp_path):
    """A gap in a cloud is an absent measurement, and reads like empty space."""
    conversion = real_conversion()
    result = export(tmp_path / "counts.ply", conversion)
    flat = flattened(result.header)

    assert squeezed(f"carries {conversion.included} points") in flat
    assert squeezed(f"{conversion.excluded} samples produced no point") in flat
    for reason, count in conversion.counts_by_reason.items():
        assert squeezed(f"{reason}={count}") in flat


def test_the_calibration_block_carries_the_scale_and_its_unknown_unit(tmp_path):
    """The scale is quoted from the conversion, never restated at a call site."""
    conversion = convert(frame({(10, 10): 17}))
    result = export(tmp_path / "cal.ply", conversion, intrinsics_sha256=FAKE_DIGEST)
    calibration = result.cloud.provenance.calibration

    assert calibration.source == TEST_SOURCE
    assert calibration.intrinsics_sha256 == FAKE_DIGEST
    assert calibration.depth_scale == conversion.depth_scale
    assert calibration.depth_scale_unit == UNIT_DISCLAIMER
    flat = flattened(result.header)
    assert squeezed(TEST_SOURCE) in flat
    assert FAKE_DIGEST in flat


# --- the zero-point cloud -----------------------------------------------------


def test_a_conversion_with_no_points_still_writes_a_valid_honest_ply(tmp_path):
    """An all-zero frame is a real outcome from this unit, not an error.

    Every sample of it means 'no depth returned', so there is nothing to write -
    and the file has to say that rather than crash, or imply an empty scene.
    """
    conversion = convert(bytes(EXPECTED_FRAME_BYTES))
    assert conversion.included == 0
    assert conversion.extent is None
    assert conversion.accounting_holds

    path = tmp_path / "empty.ply"
    result = export(path, conversion)

    header, points = parse_ply(path.read_bytes())
    assert points == []
    assert "element vertex 0" in header
    assert result.point_count == 0
    assert result.every_point_written

    flat = flattened(header)
    assert squeezed(EMPTY_CLOUD_NOTE) in flat
    assert squeezed("carries 0 points") in flat
    assert squeezed(f"{conversion.excluded} samples produced no point") in flat
    # The uncertainty is not skipped just because the geometry is.
    for limitation in (*INHERENT_LIMITATIONS, *conversion.limitations):
        assert squeezed(limitation) in flat
    assert squeezed(CAMERA_FRAME.note) in flat


def test_an_empty_cloud_in_ascii_is_header_only(tmp_path):
    """No body, and nothing that could be read as a vertex."""
    path = tmp_path / "empty-ascii.ply"
    result = export(path, convert(bytes(EXPECTED_FRAME_BYTES)), ply_format=PlyFormat.ASCII)
    assert path.read_bytes() == result.header.encode("ascii")


def test_the_empty_note_is_absent_when_points_were_built(tmp_path):
    """It is a statement about this cloud, not boilerplate."""
    result = export(tmp_path / "nonempty.ply", convert(frame({(11, 11): 18})))
    assert result.point_count > 0
    assert EMPTY_CLOUD_NOTE not in result.limitations


# --- refusals -----------------------------------------------------------------


def test_a_blank_calibration_source_is_refused_before_anything_is_written(tmp_path):
    path = tmp_path / "blank.ply"
    with pytest.raises(CalibrationSourceNotStated):
        write_conversion_ply(path, convert(frame({(1, 2): 3})), calibration_source="   ")
    assert not path.exists()


def test_a_blank_calibration_source_is_refused_by_the_whole_path(tmp_path):
    path = tmp_path / "blank-full.ply"
    with pytest.raises(CalibrationSourceNotStated):
        export_depth_frame(
            path,
            frame({(1, 2): 3}),
            intrinsics=record(),
            depth_scale=0.5,
            unit=TEST_UNIT,
            calibration_source="",
        )
    assert not path.exists()


def test_a_blank_calibration_source_is_refused_by_the_seam_itself():
    with pytest.raises(CalibrationSourceNotStated) as caught:
        cloud_provenance_for(convert(frame({(1, 2): 3})), calibration_source="\t\n ")
    assert "blank" in str(caught.value)


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, "0" * 63, "0" * 65, "g" * 64])
def test_a_malformed_intrinsics_digest_is_refused(tmp_path, digest):
    """A string in a field named ``sha256`` looks exactly like checkable evidence."""
    path = tmp_path / f"digest-{len(digest)}.ply"
    with pytest.raises(IntrinsicsDigestMalformed):
        write_conversion_ply(
            path,
            convert(frame({(1, 2): 3})),
            calibration_source=TEST_SOURCE,
            intrinsics_sha256=digest,
        )
    assert not path.exists()


def test_no_recorded_digest_is_a_legitimate_answer(tmp_path):
    result = export(tmp_path / "nodigest.ply", convert(frame({(1, 2): 3})))
    assert result.cloud.provenance.calibration.intrinsics_sha256 is None
    assert squeezed("intrinsics sha256: none recorded") in flattened(result.header)


def test_frame_bytes_that_did_not_produce_the_conversion_are_refused(tmp_path):
    """The header records the conversion's digest; it must not sit beside another."""
    conversion = convert(frame({(1, 2): 3}))
    path = tmp_path / "mismatch.ply"
    with pytest.raises(SourceFrameMismatch) as caught:
        export(path, conversion, source_frame=frame({(1, 2): 4}))
    assert conversion.frame_sha256 in str(caught.value)
    assert not path.exists()


def test_the_matching_frame_bytes_are_accepted(tmp_path):
    raw = frame({(1, 2): 3})
    result = export(tmp_path / "match.ply", convert(raw), source_frame=raw)
    assert result.cloud.provenance.source_frame_sha256 == sha256_of(raw)


def test_a_conversion_whose_accounting_does_not_hold_is_refused(tmp_path):
    """Only reachable from a hand-assembled record, and refused when it is."""
    conversion = convert(frame({(1, 2): 3, (4, 5): 6}))
    broken = replace(conversion, points=conversion.points[:1])
    assert not broken.accounting_holds

    path = tmp_path / "broken.ply"
    with pytest.raises(ConversionAccountingUnusable):
        export(path, broken)
    assert not path.exists()


def test_a_frame_of_the_wrong_length_is_refused_by_the_whole_path(tmp_path):
    """depth_metric's refusal reaches the caller, and no file is created."""
    path = tmp_path / "short.ply"
    with pytest.raises(DepthFrameSizeMismatch):
        export_depth_frame(
            path,
            b"\x00" * 16,
            intrinsics=record(),
            depth_scale=0.5,
            unit=TEST_UNIT,
            calibration_source=TEST_SOURCE,
        )
    assert not path.exists()


def test_an_unstated_unit_is_refused_by_the_whole_path(tmp_path):
    path = tmp_path / "nounit.ply"
    blank = DepthUnitAssumption(label="", stated_by="someone", basis="something")
    with pytest.raises(DepthUnitNotStated):
        export_depth_frame(
            path,
            frame({(1, 2): 3}),
            intrinsics=record(),
            depth_scale=0.5,
            unit=blank,
            calibration_source=TEST_SOURCE,
        )
    assert not path.exists()


def test_the_writers_bounds_are_forwarded_and_never_widened(tmp_path):
    """``max_points`` still caps, and the module ceiling still cannot be raised."""
    conversion = convert(frame({(1, 1): 2, (2, 2): 3, (3, 3): 4}))
    path = tmp_path / "capped.ply"
    with pytest.raises(PointCountExceeded):
        export(path, conversion, max_points=2)
    assert not path.exists()

    over = tmp_path / "over.ply"
    with pytest.raises(PointCloudNotAllowed):
        export(over, conversion, max_output_bytes=10**12)
    assert not over.exists()


def test_an_existing_path_is_never_overwritten(tmp_path):
    path = tmp_path / "taken.ply"
    path.write_bytes(b"not a cloud")
    with pytest.raises(OSError):
        export(path, convert(frame({(1, 2): 3})))
    assert path.read_bytes() == b"not a cloud"


def test_the_gates_are_callable_on_their_own():
    """Each refusal is a named function, so a caller can check before it commits."""
    assert ensure_calibration_source_stated(TEST_SOURCE) == TEST_SOURCE
    assert ensure_intrinsics_digest_usable(None) is None
    assert ensure_intrinsics_digest_usable(FAKE_DIGEST) == FAKE_DIGEST
    conversion = convert(frame({(1, 2): 3}))
    assert ensure_conversion_accounted(conversion) is conversion
    raw = frame({(1, 2): 3})
    assert ensure_frame_matches(convert(raw), raw) == raw


def test_every_refusal_is_a_scanner_error_carrying_a_doctor_status():
    """The house rule: a typed error with a status, raised before anything happens."""
    for error in (
        CalibrationSourceNotStated,
        IntrinsicsDigestMalformed,
        SourceFrameMismatch,
        ConversionAccountingUnusable,
    ):
        assert issubclass(error, CloudExportNotAllowed)
        assert issubclass(error, ScannerError)
        assert error.status is DoctorStatus.UNSUPPORTED_MODE


def test_a_non_finite_coordinate_is_still_refused_by_the_writer(tmp_path):
    """The seam does not become a way around the writer's own gates."""
    conversion = convert(frame({(1, 2): 3}))
    poisoned = replace(conversion, points=(replace(conversion.points[0], x=float("inf")),))
    path = tmp_path / "inf.ply"
    with pytest.raises(NonFiniteCoordinate):
        export(path, poisoned)
    assert not path.exists()


# --- the record the caller gets back ------------------------------------------


def test_the_result_record_reports_both_halves(tmp_path):
    conversion = convert(frame({(1, 2): 3, (4, 5): 6}))
    result = export(tmp_path / "record.ply", conversion)

    assert result.conversion is conversion
    assert result.point_count == conversion.included == 2
    assert result.every_point_written
    assert result.header == result.cloud.header
    assert result.limitations == result.cloud.provenance.all_limitations


def test_the_result_dict_is_a_stamped_report(tmp_path):
    result = export(tmp_path / "dict.ply", convert(frame({(1, 2): 3})))
    payload = result.as_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["mode"] == EXPORT_MODE
    assert payload["points_converted"] == payload["points_written"] == 1
    assert payload["every_point_written"] is True
    assert isinstance(payload["conversion"], dict)
    assert isinstance(payload["cloud"], dict)
    assert payload["limitations"] == list(result.limitations)


def test_the_result_record_is_frozen(tmp_path):
    result = export(tmp_path / "frozen.ply", convert(frame({(1, 2): 3})))
    with pytest.raises(FrozenInstanceError):
        result.conversion = convert(frame({(9, 9): 9}))  # type: ignore[misc]


def test_a_provenance_built_without_writing_matches_the_one_written(tmp_path):
    """The seam and the convenience cannot disagree about what the header says."""
    conversion = convert(frame({(1, 2): 3}))
    standalone = cloud_provenance_for(conversion, calibration_source=TEST_SOURCE)
    written = export(tmp_path / "same.ply", conversion)
    assert written.cloud.provenance == standalone
