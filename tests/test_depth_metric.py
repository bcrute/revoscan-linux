"""Tests for the offline deprojection of one raw ``Y16 `` depth frame.

Nothing here touches hardware, opens a device node or reads a device: the
module under test is pure arithmetic over bytes, and every input - the frame,
the intrinsics record and the depth scale - is handed to it by the test.

Three things are being defended, and they are not the arithmetic:

* that the *accounting* is a partition. Whatever a frame contains, the points
  built plus the per-reason exclusions add up to exactly 256000 samples, so no
  pixel can go missing quietly and none can be counted twice.
* that a zero sample never becomes a point at the origin, in any code path.
* that the unit and the uncertainties survive the trip to the report. A caller
  must state the assumed unit, no default exists that would state it for them,
  and the off-centre principal point, the unproven scalar and the absence of
  any physical validation are all still in the output dict at the end.

The one test that reads a real captured frame asserts *structural* properties
only - counts, ranges, provenance, determinism. It cannot assert accuracy,
because no measurement in this repository has established any.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from inspect import Parameter, signature
from pathlib import Path

import pytest

import scanner3d.depth_metric as depth_metric
from scanner3d.depth_metric import (
    CAMERA_FRAME,
    MAX_POINTS,
    NO_RETURN_SAMPLE,
    PIXELS_PER_FRAME,
    PREVIEW_POINTS,
    REASON_DETAIL,
    SCHEMA_VERSION,
    STATIC_LIMITATIONS,
    UNVALIDATED_UNIT,
    DepthConversion,
    DepthFrameSizeMismatch,
    DepthIntrinsicsUnusable,
    DepthScaleUnusable,
    DepthUnitAssumption,
    DepthUnitNotStated,
    ExclusionReason,
    deproject_depth_frame,
    ensure_frame_length,
    ensure_intrinsics_usable,
    ensure_scale_usable,
    ensure_unit_stated,
)
from scanner3d.depth_scale import UNIT_DISCLAIMER
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.intrinsics import INTRINSICS_FORMAT, IntrinsicsRecord, parse_intrinsics
from scanner3d.v4l2_capture import (
    EXPECTED_FRAME_BYTES,
    FOURCC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    sha256_of,
)

# --- inputs the tests build ---------------------------------------------------

#: The factory record this unit actually returned, quoted in the *test* on
#: purpose: the module under test must carry none of these numbers itself, and
#: a test that injects them is exactly how that separation gets exercised.
REAL_FX = 898.6829833984375
REAL_FY = 898.6829833984375
REAL_CX = 42.43747329711914
REAL_CY = 192.8326873779297

#: ``float32(1.0f / 200)`` as scanner-008 decoded it. What it measures is not
#: established by anything, which is the whole point of the unit plumbing.
REAL_SCALE = 0.004999999888241291

#: A unit assumption a test can pass without pretending it means anything.
TEST_UNIT = DepthUnitAssumption(
    label="test depth unit",
    stated_by="tests/test_depth_metric.py",
    basis="an arbitrary label chosen so the plumbing can be checked; it measures nothing",
)


def record(
    *,
    fx: float = 100.0,
    fy: float = 100.0,
    cx: float = 320.0,
    cy: float = 200.0,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> IntrinsicsRecord:
    """A record built through the real 40-byte parser, not by hand.

    Defaults are deliberately round and centred so the deprojection below is
    exact in binary floating point and can be checked with a pencil. They are
    not this unit's numbers.
    """
    blob = struct.pack(INTRINSICS_FORMAT, width, height, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
    parsed = parse_intrinsics(blob)
    assert parsed is not None
    return parsed


def real_record() -> IntrinsicsRecord:
    """The unit's own factory record, injected by the test."""
    return record(fx=REAL_FX, fy=REAL_FY, cx=REAL_CX, cy=REAL_CY)


def frame(samples: Mapping[tuple[int, int], int]) -> bytes:
    """A whole 512000-byte frame that is zero except at the named ``(u, v)``."""
    buffer = bytearray(EXPECTED_FRAME_BYTES)
    for (u, v), value in samples.items():
        struct.pack_into("<H", buffer, (v * FRAME_WIDTH + u) * 2, value)
    return bytes(buffer)


def points_by_pixel(conversion: DepthConversion) -> dict[tuple[int, int], depth_metric.DepthPoint]:
    return {(point.u, point.v): point for point in conversion.points}


def block(reported: Mapping[str, object], key: str) -> dict[str, object]:
    """One nested dict of a report, narrowed so the assertions can index it."""
    value = reported[key]
    assert isinstance(value, dict)
    return value


def convert(
    raw: bytes,
    *,
    intrinsics: IntrinsicsRecord | None = None,
    depth_scale: float = 0.5,
    unit: DepthUnitAssumption = TEST_UNIT,
) -> DepthConversion:
    return deproject_depth_frame(
        raw,
        intrinsics=record() if intrinsics is None else intrinsics,
        depth_scale=depth_scale,
        unit=unit,
    )


# --- the arithmetic, checked by hand ------------------------------------------


def test_synthetic_frame_matches_hand_computed_points():
    """fx = fy = 100, cx = 320, cy = 200, scale = 0.5, so every number is exact.

    raw 1000 -> d = 500. At (420, 300): x = (420-320)*500/100 = 500,
    y = (300-200)*500/100 = 500, z = 500. At the principal point x = y = 0.
    raw 4000 -> d = 2000. At (220, 100): x = -2000, y = -2000, z = 2000.
    Every one of those values is exactly representable, so these are equalities
    and not tolerances.
    """
    raw = frame({(420, 300): 1000, (320, 200): 1000, (220, 100): 4000})
    conversion = convert(raw)

    assert conversion.included == 3
    found = points_by_pixel(conversion)
    assert (found[(420, 300)].x, found[(420, 300)].y, found[(420, 300)].z) == (500.0, 500.0, 500.0)
    assert (found[(320, 200)].x, found[(320, 200)].y, found[(320, 200)].z) == (0.0, 0.0, 500.0)
    assert (found[(220, 100)].x, found[(220, 100)].y, found[(220, 100)].z) == (
        -2000.0,
        -2000.0,
        2000.0,
    )
    assert found[(420, 300)].raw == 1000
    assert found[(220, 100)].raw == 4000


def test_z_is_exactly_raw_times_scale_for_every_point():
    raw = frame({(u, 7): u + 1 for u in range(0, 640, 37)})
    conversion = convert(raw, depth_scale=0.25)
    assert conversion.included == len(range(0, 640, 37))
    for point in conversion.points:
        assert point.z == point.raw * 0.25


def test_deprojection_matches_the_formula_for_arbitrary_intrinsics():
    """Re-derives x and y independently of the module for a lopsided record."""
    intrinsics = record(fx=311.0, fy=907.5, cx=17.25, cy=381.5)
    raw = frame({(3, 11): 4321, (639, 399): 900, (0, 0): 7})
    conversion = convert(raw, intrinsics=intrinsics, depth_scale=REAL_SCALE)
    for point in conversion.points:
        distance = point.raw * REAL_SCALE
        assert point.z == distance
        assert point.x == (point.u - 17.25) * distance / 311.0
        assert point.y == (point.v - 381.5) * distance / 907.5


def test_boundary_pixels_are_converted_and_indexed_correctly():
    """The four corners: index arithmetic, row/column order and sign, all at once."""
    corners = {(0, 0): 1000, (639, 0): 1000, (0, 399): 1000, (639, 399): 1000}
    conversion = convert(frame(corners))

    assert conversion.included == 4
    found = points_by_pixel(conversion)
    assert set(found) == set(corners)
    # d = 500, fx = fy = 100, cx = 320, cy = 200.
    assert (found[(0, 0)].x, found[(0, 0)].y) == (-1600.0, -1000.0)
    assert (found[(639, 0)].x, found[(639, 0)].y) == (1595.0, -1000.0)
    assert (found[(0, 399)].x, found[(0, 399)].y) == (-1600.0, 995.0)
    assert (found[(639, 399)].x, found[(639, 399)].y) == (1595.0, 995.0)


def test_pixel_index_is_row_major_and_not_transposed():
    """A sample at (u=1, v=0) must not come back as (u=0, v=1)."""
    conversion = convert(frame({(1, 0): 500}))
    assert conversion.included == 1
    only = conversion.points[0]
    assert (only.u, only.v) == (1, 0)


def test_every_built_point_stays_inside_the_raster():
    raw = frame({(u, v): 1 + u for u in range(0, 640, 91) for v in range(0, 400, 53)})
    conversion = convert(raw)
    assert conversion.included > 0
    for point in conversion.points:
        assert 0 <= point.u < FRAME_WIDTH
        assert 0 <= point.v < FRAME_HEIGHT


# --- zero means "no return", never distance zero ------------------------------


def test_all_zero_frame_produces_no_points_at_all():
    conversion = convert(bytes(EXPECTED_FRAME_BYTES))

    assert conversion.included == 0
    assert conversion.points == ()
    assert conversion.extent is None
    assert conversion.count_of(ExclusionReason.NO_RETURN) == PIXELS_PER_FRAME
    assert conversion.excluded == PIXELS_PER_FRAME
    assert conversion.accounting_holds


def test_zero_samples_never_become_points_at_the_origin():
    """Even at the principal point, where a zero would deproject to (0, 0, 0)."""
    conversion = convert(frame({(320, 200): NO_RETURN_SAMPLE, (321, 200): 1000}))

    assert conversion.included == 1
    assert conversion.points[0].u == 321
    assert all((p.x, p.y, p.z) != (0.0, 0.0, 0.0) for p in conversion.points)
    assert conversion.count_of(ExclusionReason.NO_RETURN) == PIXELS_PER_FRAME - 1


def test_the_no_return_reason_says_it_is_not_a_zero_distance():
    detail = REASON_DETAIL[ExclusionReason.NO_RETURN]
    assert "no depth" in detail
    assert "NOT a measurement of zero distance" in detail


def test_a_one_sample_is_a_point_even_though_zero_is_not():
    """The exclusion is the value 0 exactly, not "small"."""
    conversion = convert(frame({(400, 100): 1}))
    assert conversion.included == 1
    assert conversion.points[0].raw == 1
    assert conversion.points[0].z == 0.5


# --- the accounting is a partition of 256000 ----------------------------------


@pytest.mark.parametrize(
    "samples",
    [
        {},
        {(0, 0): 1},
        {(u, v): (u * v) % 65535 + 1 for u in range(0, 640, 17) for v in range(0, 400, 13)},
        {(639, 399): 65535},
    ],
)
def test_exclusion_accounting_sums_to_exactly_256000(samples):
    conversion = convert(frame(samples))

    assert PIXELS_PER_FRAME == 256000
    assert conversion.included + conversion.excluded == 256000
    assert conversion.accounted == 256000
    assert conversion.accounting_holds
    reported = block(conversion.as_dict(), "pixel_accounting")
    by_reason = block(reported, "counts_by_reason")
    assert reported["accounted"] == 256000
    counted = sum(int(str(count)) for count in by_reason.values())
    assert counted + int(str(reported["included"])) == 256000


def test_every_reason_is_reported_even_when_it_did_not_occur():
    conversion = convert(frame({(1, 1): 10}))
    assert [e.reason for e in conversion.exclusions] == list(ExclusionReason)
    assert set(conversion.counts_by_reason) == {str(r) for r in ExclusionReason}
    assert conversion.count_of(ExclusionReason.POINT_LIMIT) == 0
    assert conversion.count_of(ExclusionReason.NON_FINITE_DISTANCE) == 0


def test_accounting_holds_when_every_sample_is_nonzero():
    raw = struct.pack(depth_metric.SAMPLES_FORMAT, *([1234] * PIXELS_PER_FRAME))
    conversion = convert(raw)

    assert conversion.included == PIXELS_PER_FRAME
    assert conversion.excluded == 0
    assert conversion.accounting_holds
    assert conversion.included <= MAX_POINTS


def test_non_finite_distances_and_coordinates_are_counted_separately():
    """A scale near the float ceiling: raw 2 overflows d, raw 1 overflows x.

    Neither can happen with the scalar this unit returned. They are exercised
    so that an extreme scale narrows the cloud and keeps the accounting exact,
    rather than seeding it with infinities.
    """
    conversion = convert(frame({(0, 0): 2, (639, 399): 1}), depth_scale=1e308)

    assert conversion.included == 0
    assert conversion.count_of(ExclusionReason.NON_FINITE_DISTANCE) == 1
    assert conversion.count_of(ExclusionReason.NON_FINITE_COORDINATE) == 1
    assert conversion.count_of(ExclusionReason.NO_RETURN) == PIXELS_PER_FRAME - 2
    assert conversion.accounting_holds


def test_no_built_point_ever_carries_a_non_finite_coordinate():
    conversion = convert(frame({(0, 0): 1, (639, 399): 3, (320, 200): 65535}), depth_scale=1e300)
    for point in conversion.points:
        assert math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)


# --- the fixed bound ----------------------------------------------------------


def test_max_points_is_one_per_pixel_and_is_not_a_parameter():
    assert MAX_POINTS == PIXELS_PER_FRAME == FRAME_WIDTH * FRAME_HEIGHT
    assert "max_points" not in signature(deproject_depth_frame).parameters


def test_the_point_ceiling_truncates_and_keeps_the_accounting_exact(monkeypatch):
    """Patched low, because the real ceiling cannot be reached by a legal frame."""
    monkeypatch.setattr(depth_metric, "MAX_POINTS", 3)
    samples = {(u, 5): 100 for u in range(10)}
    conversion = convert(frame(samples))

    assert conversion.included == 3
    assert conversion.count_of(ExclusionReason.POINT_LIMIT) == 7
    assert conversion.truncated
    assert conversion.accounting_holds
    assert conversion.accounted == PIXELS_PER_FRAME
    assert any("partial" in text for text in conversion.limitations)
    assert conversion.as_dict()["bounds"] == {
        "max_points": 3,
        "expected_frame_bytes": EXPECTED_FRAME_BYTES,
        "truncated": True,
        "preview_points": PREVIEW_POINTS,
    }


def test_an_untruncated_run_says_so():
    conversion = convert(frame({(2, 2): 5}))
    assert not conversion.truncated
    assert not any("partial" in text for text in conversion.limitations)


# --- refusals, all of them before any work ------------------------------------


@pytest.mark.parametrize("length", [0, 1, 2, 511999, 512001, 1024000])
def test_wrong_frame_length_is_refused(length):
    with pytest.raises(DepthFrameSizeMismatch) as excinfo:
        convert(bytes(length))
    assert str(length) in str(excinfo.value)
    assert "Nothing was unpacked" in str(excinfo.value)


def test_the_refusals_are_scanner_errors_carrying_a_status():
    for error in (
        DepthFrameSizeMismatch,
        DepthIntrinsicsUnusable,
        DepthScaleUnusable,
        DepthUnitNotStated,
    ):
        assert issubclass(error, ScannerError)
        assert error.status is DoctorStatus.UNSUPPORTED_MODE


@pytest.mark.parametrize(
    ("fx", "fy"),
    [(0.0, 100.0), (100.0, 0.0), (-1.0, 100.0), (100.0, -1.0)],
)
def test_non_positive_focal_lengths_are_refused(fx, fy):
    with pytest.raises(DepthIntrinsicsUnusable):
        convert(bytes(EXPECTED_FRAME_BYTES), intrinsics=record(fx=fx, fy=fy))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_intrinsics_are_refused(value):
    for name in ("fx", "fy", "cx", "cy"):
        with pytest.raises(DepthIntrinsicsUnusable) as excinfo:
            ensure_intrinsics_usable(record(**{name: value}))
        assert name in str(excinfo.value)


def test_a_record_missing_a_needed_entry_is_refused():
    """Constructed directly, because the 40-byte parser cannot produce one."""
    partial = IntrinsicsRecord(
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        values={"fx": 100.0, "fy": 100.0, "cx": 320.0},
        structure=(),
        plausibility=(),
    )
    with pytest.raises(DepthIntrinsicsUnusable) as excinfo:
        ensure_intrinsics_usable(partial)
    assert "'cy'" in str(excinfo.value)


@pytest.mark.parametrize("scale", [0.0, -0.005, math.nan, math.inf, -math.inf])
def test_unusable_depth_scales_are_refused(scale):
    with pytest.raises(DepthScaleUnusable):
        convert(bytes(EXPECTED_FRAME_BYTES), depth_scale=scale)


@pytest.mark.parametrize(
    "unit",
    [
        DepthUnitAssumption(label="", stated_by="someone", basis="something"),
        DepthUnitAssumption(label="   ", stated_by="someone", basis="something"),
        DepthUnitAssumption(label="mm", stated_by="", basis="something"),
        DepthUnitAssumption(label="mm", stated_by="someone", basis="\t\n"),
    ],
)
def test_a_blank_unit_assumption_is_refused(unit):
    with pytest.raises(DepthUnitNotStated):
        convert(bytes(EXPECTED_FRAME_BYTES), unit=unit)


def test_no_refusal_ever_unpacks_the_frame(monkeypatch):
    """Proves "before any work" rather than asserting it in a docstring.

    ``struct`` is replaced in the module namespace with an object that fails
    loudly if the frame is unpacked, so any refusal that had already touched
    the buffer would surface as an ``AssertionError`` instead of its own type.
    """

    class Explode:
        def unpack(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("the frame was unpacked despite a refusal")

    monkeypatch.setattr(depth_metric, "struct", Explode())
    good = bytes(EXPECTED_FRAME_BYTES)

    with pytest.raises(DepthUnitNotStated):
        convert(good, unit=DepthUnitAssumption(label="", stated_by="x", basis="y"))
    with pytest.raises(DepthIntrinsicsUnusable):
        convert(good, intrinsics=record(fx=0.0))
    with pytest.raises(DepthScaleUnusable):
        convert(good, depth_scale=0.0)
    with pytest.raises(DepthFrameSizeMismatch):
        convert(b"short")


def test_the_gate_helpers_pass_good_inputs_through():
    assert ensure_frame_length(bytes(EXPECTED_FRAME_BYTES)) == bytes(EXPECTED_FRAME_BYTES)
    assert ensure_scale_usable(REAL_SCALE) == REAL_SCALE
    assert ensure_unit_stated(TEST_UNIT) is TEST_UNIT
    assert ensure_intrinsics_usable(real_record()) == (REAL_FX, REAL_FY, REAL_CX, REAL_CY)


# --- units and uncertainty are first-class ------------------------------------


def test_the_caller_must_state_the_unit_because_there_is_no_default():
    parameters = signature(deproject_depth_frame).parameters
    for name in ("intrinsics", "depth_scale", "unit"):
        assert parameters[name].default is Parameter.empty
        assert parameters[name].kind is Parameter.KEYWORD_ONLY


def test_the_module_hardcodes_none_of_this_units_numbers():
    """The intrinsics and the scale are injected, so they cannot live in here."""
    source = Path(depth_metric.__file__).read_text(encoding="utf-8")
    for forbidden in ("898.68", "42.437", "192.83", "0.004999", "1.0 / 200"):
        assert forbidden not in source


def test_the_unit_block_travels_into_the_report():
    conversion = convert(frame({(1, 1): 100}), unit=TEST_UNIT)
    reported = conversion.as_dict()["unit"]
    assert reported == {
        "label": TEST_UNIT.label,
        "stated_by": TEST_UNIT.stated_by,
        "basis": TEST_UNIT.basis,
        "assumed": True,
        "established_by_measurement": False,
        "disclaimer": UNIT_DISCLAIMER,
    }


def test_the_shipped_unit_declines_to_name_a_unit():
    conversion = convert(frame({(1, 1): 100}), unit=UNVALIDATED_UNIT)
    assert conversion.unit is UNVALIDATED_UNIT
    assert UNVALIDATED_UNIT.basis.startswith("none")
    assert "no physical measurement" in UNVALIDATED_UNIT.basis
    assert block(conversion.as_dict(), "unit")["established_by_measurement"] is False


def test_the_depth_scale_is_reported_with_its_unit_disclaimer():
    conversion = convert(frame({(1, 1): 100}), depth_scale=REAL_SCALE)
    reported = block(conversion.as_dict(), "depth_scale")
    assert reported["value"] == REAL_SCALE
    assert reported["unit"] == UNIT_DISCLAIMER
    assert "raw * depth_scale" in str(reported["applied_as"])


def test_the_standing_limitations_are_in_every_result():
    conversion = convert(frame({(1, 1): 100}))
    for limitation in STATIC_LIMITATIONS:
        assert limitation in conversion.limitations
    reported = conversion.as_dict()["limitations"]
    assert isinstance(reported, list)
    assert reported[: len(STATIC_LIMITATIONS)] == list(STATIC_LIMITATIONS)


def test_the_limitations_name_the_things_that_were_never_established():
    joined = " ".join(STATIC_LIMITATIONS)
    assert "no metric accuracy is claimed" in joined
    assert "no lens distortion is modelled" in joined
    assert "no physical measurement has validated" in joined
    assert UNIT_DISCLAIMER in joined


def test_the_off_centre_principal_point_is_surfaced_not_corrected():
    conversion = convert(frame({(1, 1): 1000}), intrinsics=real_record())
    note = conversion.principal_point

    assert note.cx == REAL_CX and note.cy == REAL_CY
    assert note.centre_u == 320.0 and note.centre_v == 200.0
    assert note.offset_u == pytest.approx(REAL_CX - 320.0)
    assert not note.near_centre
    assert any("outside the" in text and "tolerance" in text for text in conversion.limitations)
    flagged = {c.name: c for c in conversion.checks}
    assert not flagged["principal_point_near_frame_centre"].ok
    # Surfaced, not corrected: the value used is the one that arrived.
    only = conversion.points[0]
    assert only.x == (1 - REAL_CX) * only.z / REAL_FX


def test_a_centred_principal_point_earns_no_extra_limitation():
    conversion = convert(frame({(1, 1): 1000}))
    assert conversion.principal_point.near_centre
    assert conversion.limitations == STATIC_LIMITATIONS
    assert {c.name: c.ok for c in conversion.checks}["principal_point_near_frame_centre"]


def test_intrinsics_naming_another_raster_are_reported_not_rescaled():
    conversion = convert(frame({(320, 200): 1000}), intrinsics=record(width=320, height=200))
    checks = {c.name: c for c in conversion.checks}

    assert not checks["intrinsics_frame_matches_depth_frame"].ok
    assert any("320x200" in text for text in conversion.limitations)
    # No rescaling happened: cx is still 320 in the frame's own pixels.
    assert conversion.points[0].x == 0.0


def test_a_structurally_divergent_record_is_converted_and_flagged():
    """fx, fy, cx, cy are usable; the fixed 0/0/0/0/1 entries are not exact."""
    blob = struct.pack(
        INTRINSICS_FORMAT,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        100.0,
        0.5,
        320.0,
        0.0,
        100.0,
        200.0,
        0.0,
        0.0,
        1.0,
    )
    divergent = parse_intrinsics(blob)
    assert divergent is not None and not divergent.structurally_expected

    conversion = convert(frame({(420, 300): 1000}), intrinsics=divergent)
    assert conversion.included == 1
    assert not {c.name: c.ok for c in conversion.checks}["intrinsics_structurally_expected"]


# --- the coordinate convention ------------------------------------------------


def test_the_convention_is_stated_in_full_and_names_its_handedness():
    reported = CAMERA_FRAME.as_dict()
    assert set(reported) == {"origin", "x_axis", "y_axis", "z_axis", "handedness", "note"}
    assert all(str(value).strip() for value in reported.values())
    assert "right-handed" in CAMERA_FRAME.handedness
    assert "x_hat cross y_hat = z_hat" in CAMERA_FRAME.handedness
    assert "not one the device declared" in CAMERA_FRAME.note


def test_the_points_actually_follow_the_stated_axis_directions():
    """The words and the arithmetic are checked against each other."""
    conversion = convert(frame({(400, 200): 1000, (240, 200): 1000, (320, 300): 1000}))
    found = points_by_pixel(conversion)

    assert "increasing column index u (image right)" in CAMERA_FRAME.x_axis
    assert found[(400, 200)].x > 0  # right of cx
    assert found[(240, 200)].x < 0  # left of cx
    assert "increasing row index v (image down)" in CAMERA_FRAME.y_axis
    assert found[(320, 300)].y > 0  # below cy
    assert "away from the camera" in CAMERA_FRAME.z_axis
    assert all(point.z > 0 for point in conversion.points)


def test_the_convention_is_reported_with_the_points():
    conversion = convert(frame({(1, 1): 100}))
    assert conversion.as_dict()["coordinate_convention"] == CAMERA_FRAME.as_dict()


# --- the report ---------------------------------------------------------------


def test_the_report_stamps_the_schema_version_and_the_frame_identity():
    raw = frame({(5, 5): 1000})
    reported = convert(raw).as_dict()

    assert reported["schema_version"] == SCHEMA_VERSION == 1
    source = block(reported, "source_frame")
    assert source["sha256"] == sha256_of(raw)
    assert source["bytes"] == source["expected_bytes"] == EXPECTED_FRAME_BYTES
    assert (source["width"], source["height"]) == (FRAME_WIDTH, FRAME_HEIGHT)
    assert source["fourcc"] == FOURCC


def test_the_report_carries_the_intrinsics_that_were_used():
    intrinsics = real_record()
    reported = convert(frame({(5, 5): 1000}), intrinsics=intrinsics).as_dict()
    assert reported["intrinsics"] == intrinsics.as_dict()


def test_the_report_is_json_serialisable_and_bounded():
    raw = frame({(u, 3): 1000 for u in range(200)})
    reported = convert(raw).as_dict()
    text = json.dumps(reported)

    assert json.loads(text)["schema_version"] == 1
    points = block(reported, "points")
    assert points["count"] == 200
    assert isinstance(points["preview"], list)
    assert len(points["preview"]) == PREVIEW_POINTS
    # A report stays a report: 200 points, at most 8 of them in the dict.
    assert len(text) < 20000


def test_the_extent_spans_the_points_that_were_built():
    conversion = convert(frame({(0, 0): 1000, (639, 399): 1000}))
    extent = conversion.extent
    assert extent is not None
    assert (extent.min_x, extent.max_x) == (-1600.0, 1595.0)
    assert (extent.min_y, extent.max_y) == (-1000.0, 995.0)
    assert (extent.min_z, extent.max_z) == (500.0, 500.0)
    assert (extent.min_raw, extent.max_raw) == (1000, 1000)
    assert conversion.as_dict()["extent"] == extent.as_dict()


def test_an_empty_cloud_reports_no_extent():
    assert convert(bytes(EXPECTED_FRAME_BYTES)).as_dict()["extent"] is None


def test_the_result_is_frozen_and_the_cloud_is_a_tuple():
    conversion = convert(frame({(1, 1): 10}))
    assert isinstance(conversion.points, tuple)
    assert isinstance(conversion.exclusions, tuple)
    assert isinstance(conversion.limitations, tuple)
    with pytest.raises(FrozenInstanceError):
        conversion.frame_sha256 = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        conversion.points[0].x = 1.0  # type: ignore[misc]


# --- determinism --------------------------------------------------------------


def test_the_same_bytes_give_byte_identical_results_twice():
    raw = frame({(u, v): (u + v) % 4000 + 1 for u in range(0, 640, 31) for v in range(0, 400, 29)})
    intrinsics = real_record()
    first = convert(raw, intrinsics=intrinsics, depth_scale=REAL_SCALE)
    second = convert(raw, intrinsics=intrinsics, depth_scale=REAL_SCALE)

    assert first.points == second.points
    assert first == second
    assert json.dumps(first.as_dict()) == json.dumps(second.as_dict())


def test_points_come_back_in_raster_order():
    """Determinism the caller can rely on: ordering is the frame's own."""
    raw = frame({(600, 1): 10, (5, 0): 20, (300, 1): 30, (639, 399): 40})
    conversion = convert(raw)
    assert [(p.u, p.v) for p in conversion.points] == [(5, 0), (300, 1), (600, 1), (639, 399)]


# --- one real captured frame, structural assertions only ----------------------

#: Where this session's captured frames were written. Deliberately not a
#: repository fixture: these are ephemeral scratch artifacts of the capture
#: tasks, not committed evidence, so the test below reads one if it is still
#: there and skips - loudly - if it is not.
SCRATCHPAD = Path(
    "/tmp/claude-1000/-home-ben-repositories-3d-scanner/"
    "147e7b71-d25d-4357-bb4a-1f6a3964c3bb/scratchpad"
)

#: The capture directories named in the task's domain facts, tried first.
DOCUMENTED_CAPTURES = tuple(f"base_{index}" for index in range(1, 6))

#: An explicit override, so a real frame kept anywhere else can still be used.
FRAME_PATH_ENV = "SCANNER3D_REAL_DEPTH_FRAME"


def real_frame_path() -> Path | None:
    """One real 512000-byte capture, or ``None`` if none is on this machine."""
    override = os.environ.get(FRAME_PATH_ENV)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    named = [SCRATCHPAD / name / "depth-frame-raw.bin" for name in DOCUMENTED_CAPTURES]
    others = sorted(SCRATCHPAD.glob("*/depth-frame-raw.bin")) if SCRATCHPAD.is_dir() else []
    for candidate in [*named, *others]:
        if candidate.is_file() and candidate.stat().st_size == EXPECTED_FRAME_BYTES:
            return candidate
    return None


def test_a_real_captured_frame_converts_with_exact_accounting():
    """Structure only: counts, provenance, ranges, determinism.

    Not one assertion here is about accuracy, and none can be: no measurement
    in this repository has established what a converted distance means. What is
    checked is that a real frame - two thirds of it zeros - goes in and comes
    back fully accounted for, with every point traceable to a nonzero sample.
    """
    path = real_frame_path()
    if path is None:
        pytest.skip(
            f"no real 512000-byte depth frame found under {SCRATCHPAD} and "
            f"${FRAME_PATH_ENV} is unset; the captures are ephemeral scratch artifacts"
        )
    raw = path.read_bytes()
    assert len(raw) == EXPECTED_FRAME_BYTES

    conversion = deproject_depth_frame(
        raw, intrinsics=real_record(), depth_scale=REAL_SCALE, unit=UNVALIDATED_UNIT
    )

    samples = struct.unpack(depth_metric.SAMPLES_FORMAT, raw)
    nonzero = sum(1 for sample in samples if sample)
    assert 0 < nonzero <= PIXELS_PER_FRAME

    assert conversion.frame_sha256 == sha256_of(raw)
    assert conversion.included == nonzero
    assert conversion.count_of(ExclusionReason.NO_RETURN) == PIXELS_PER_FRAME - nonzero
    assert conversion.count_of(ExclusionReason.NON_FINITE_DISTANCE) == 0
    assert conversion.count_of(ExclusionReason.NON_FINITE_COORDINATE) == 0
    assert conversion.count_of(ExclusionReason.POINT_LIMIT) == 0
    assert conversion.accounted == PIXELS_PER_FRAME
    assert conversion.accounting_holds
    assert not conversion.truncated

    for point in conversion.points:
        assert point.raw != 0
        assert 0 <= point.u < FRAME_WIDTH
        assert 0 <= point.v < FRAME_HEIGHT
        assert math.isfinite(point.x) and math.isfinite(point.y)
        assert point.z > 0.0
        assert point.z == point.raw * REAL_SCALE
        assert samples[point.v * FRAME_WIDTH + point.u] == point.raw

    extent = conversion.extent
    assert extent is not None
    assert extent.min_z > 0.0 and extent.max_z >= extent.min_z
    assert 0 < extent.min_raw <= extent.max_raw <= 65535

    # The uncertainties survive a real frame exactly as they do a synthetic one.
    assert not conversion.principal_point.near_centre
    assert UNIT_DISCLAIMER in " ".join(conversion.limitations)
    assert block(conversion.as_dict(), "unit")["established_by_measurement"] is False

    again = deproject_depth_frame(
        raw, intrinsics=real_record(), depth_scale=REAL_SCALE, unit=UNVALIDATED_UNIT
    )
    assert again.points == conversion.points
    assert json.dumps(again.as_dict()) == json.dumps(conversion.as_dict())
