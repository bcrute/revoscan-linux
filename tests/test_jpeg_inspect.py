"""Exercise :mod:`scanner3d.jpeg_inspect` against small synthetic JPEG byte strings.

Every fixture below is built by hand from a handful of local helpers rather
than any image library or binary fixture file, because the module under test
promises to need nothing but the standard library and this suite proves that
promise by construction: if it needed Pillow to make a test JPEG, it would
need Pillow to make a test JPEG.
"""

from __future__ import annotations

import json

from scanner3d.jpeg_inspect import (
    MAX_INSPECT_BYTES,
    MAX_SEGMENTS,
    JpegStatus,
    inspect_jpeg,
)


def soi() -> bytes:
    """The two-byte Start Of Image marker."""
    return b"\xff\xd8"


def eoi() -> bytes:
    """The two-byte End Of Image marker."""
    return b"\xff\xd9"


def segment(marker: int, body: bytes = b"") -> bytes:
    """One length-bearing marker segment: ``FF <marker> <len hi> <len lo> <body>``."""
    length = len(body) + 2
    return bytes([0xFF, marker]) + length.to_bytes(2, "big") + body


def sof(marker: int, width: int, height: int, components: int = 3, precision: int = 8) -> bytes:
    """A SOF segment body: precision, height, width, then one triplet per component."""
    body = bytes([precision]) + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    body += bytes([components])
    for component_id in range(components):
        body += bytes([component_id + 1, 0x11, 0])
    return segment(marker, body)


def sos(components: int = 1) -> bytes:
    """A minimal SOS header: one component selector pair, plus the three tail bytes."""
    body = bytes([components])
    for component_id in range(components):
        body += bytes([component_id + 1, 0])
    body += bytes([0, 63, 0])
    return segment(0xDA, body)


def app0_jfif() -> bytes:
    """A minimal APP0/JFIF segment, as a real encoder would emit first."""
    return segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")


def baseline_jpeg(width: int = 1280, height: int = 800) -> bytes:
    """A minimal but structurally complete baseline (SOF0) JPEG."""
    return (
        soi()
        + app0_jfif()
        + segment(0xDB, b"\x00" + b"\x01" * 64)  # DQT
        + sof(0xC0, width, height)
        + segment(0xC4, b"\x00" + b"\x00" * 16)  # DHT
        + sos()
        + b"\x12\x34\x56"  # a few bytes of entropy-coded scan data
        + eoi()
    )


def progressive_jpeg(width: int = 1280, height: int = 800) -> bytes:
    """A minimal progressive (SOF2) JPEG, otherwise shaped like :func:`baseline_jpeg`."""
    return (
        soi()
        + app0_jfif()
        + segment(0xDB, b"\x00" + b"\x01" * 64)
        + sof(0xC2, width, height)
        + segment(0xC4, b"\x00" + b"\x00" * 16)
        + sos()
        + b"\xab\xcd"
        + eoi()
    )


def test_baseline_jpeg_is_well_formed_with_correct_frame_metadata() -> None:
    """A minimal but complete baseline JPEG reports WELL_FORMED with exact dimensions."""
    report = inspect_jpeg(baseline_jpeg(1280, 800))
    assert report.status is JpegStatus.WELL_FORMED
    assert report.well_formed is True
    assert report.frame is not None
    assert report.frame.width == 1280
    assert report.frame.height == 800
    assert report.frame.precision == 8
    assert report.frame.components == 3
    assert report.frame.progressive is False
    assert report.has_soi is True
    assert report.has_eoi is True
    assert report.eoi_terminal is True
    assert report.trailing_bytes == 0


def test_progressive_jpeg_reports_progressive_frame() -> None:
    """A SOF2 frame header is recognized and marked progressive."""
    report = inspect_jpeg(progressive_jpeg(1280, 800))
    assert report.status is JpegStatus.WELL_FORMED
    assert report.frame is not None
    assert report.frame.progressive is True
    assert report.frame.marker_name == "SOF2"


def test_app_and_com_segments_are_walked_and_reported() -> None:
    """APPn and COM segments, including a zero-body COM, appear in the segment list."""
    payload = (
        soi()
        + app0_jfif()
        + segment(0xE1, b"exif-ish payload")  # APP1
        + segment(0xFE, b"")  # COM, zero-body: length exactly 2
        + segment(0xDB, b"\x00" + b"\x01" * 64)
        + sof(0xC0, 640, 480)
        + segment(0xC4, b"\x00" + b"\x00" * 16)
        + sos()
        + b"\x00\x01\x02"
        + eoi()
    )
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.WELL_FORMED
    names = [s.name for s in report.segments]
    assert "APP0" in names
    assert "APP1" in names
    assert "COM" in names
    com_segments = [s for s in report.segments if s.name == "COM"]
    assert com_segments[0].length == 2


def test_byte_stuffing_is_counted_and_does_not_end_the_scan() -> None:
    """FF 00 inside scan data is a literal FF byte, counted, and the scan continues past it."""
    payload = (
        soi()
        + app0_jfif()
        + sof(0xC0, 64, 64)
        + sos()
        + b"\x01\xff\x00\x02\x03"  # stuffed FF, plus data on both sides
        + eoi()
    )
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.WELL_FORMED
    assert report.stuffed_byte_count == 1
    assert report.scan_count == 1


def test_restart_markers_are_counted_and_do_not_end_the_scan() -> None:
    """RST0-RST7 inside scan data are counted and the scan continues past them."""
    payload = (
        soi()
        + app0_jfif()
        + sof(0xC0, 64, 64)
        + sos()
        + b"\x01\x02"
        + bytes([0xFF, 0xD0])
        + b"\x03\x04"
        + bytes([0xFF, 0xD3])
        + b"\x05"
        + eoi()
    )
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.WELL_FORMED
    assert report.restart_marker_count == 2


def test_fill_bytes_before_a_marker_are_skipped() -> None:
    """A run of 0xFF bytes before a marker byte is legal padding, not a violation."""
    payload = (
        soi()
        + b"\xff\xff\xff"  # fill bytes preceding the APP0 marker's own leading FF
        + segment(0xE0, b"JFIF\x00")
        + sof(0xC0, 64, 64)
        + sos()
        + b"\x00\x01"
        + eoi()
    )
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.WELL_FORMED


def test_segment_declaring_more_bytes_than_remain_is_truncated() -> None:
    """A segment length that runs past the end of the payload is TRUNCATED."""
    payload = soi() + bytes([0xFF, 0xDB]) + (200).to_bytes(2, "big") + b"\x00" * 10
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.TRUNCATED


def test_scan_with_no_eoi_running_off_the_end_is_truncated() -> None:
    """A SOS with no EOI, whose entropy data simply runs out, is TRUNCATED."""
    payload = soi() + app0_jfif() + sof(0xC0, 64, 64) + sos() + b"\x01\x02\x03"
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.TRUNCATED


def test_declared_length_of_zero_or_one_is_malformed() -> None:
    """A segment length below 2 (which cannot even describe itself) is MALFORMED."""
    zero = soi() + bytes([0xFF, 0xDB, 0x00, 0x00])
    one = soi() + bytes([0xFF, 0xDB, 0x00, 0x01])
    assert inspect_jpeg(zero).status is JpegStatus.MALFORMED
    assert inspect_jpeg(one).status is JpegStatus.MALFORMED


def test_sof_segment_too_short_to_hold_its_header_is_malformed() -> None:
    """A SOF segment shorter than 8 bytes cannot carry a frame header."""
    payload = soi() + segment(0xC0, b"\x08\x00\x40")  # only 3 body bytes, needs 6
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MALFORMED


def test_sof_length_disagreeing_with_component_count_is_malformed() -> None:
    """A SOF declaring 3 components but sized for a different count is MALFORMED."""
    body = bytes([8]) + (64).to_bytes(2, "big") + (64).to_bytes(2, "big") + bytes([3])
    body += bytes([1, 0x11, 0])  # only one component triplet instead of three
    payload = soi() + segment(0xC0, body)
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MALFORMED


def test_byte_where_marker_leading_ff_was_required_is_malformed() -> None:
    """A non-FF byte where the next marker's leading FF was expected is MALFORMED."""
    payload = soi() + b"\x01\x02\x03"
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MALFORMED


def test_missing_soi_payload_beginning_with_something_else() -> None:
    """A payload that does not begin with FFD8 is MISSING_SOI."""
    report = inspect_jpeg(b"not a jpeg at all")
    assert report.status is JpegStatus.MISSING_SOI
    assert report.has_soi is False


def test_missing_soi_single_ff_byte() -> None:
    """A single 0xFF byte cannot contain SOI and is MISSING_SOI, not a crash."""
    report = inspect_jpeg(b"\xff")
    assert report.status is JpegStatus.MISSING_SOI


def test_missing_eoi_when_well_formed_prefix_has_no_terminator() -> None:
    """A payload with SOI, a frame header, and no EOI at all is MISSING_EOI.

    The scan is closed by a TEM marker rather than left to run off the end:
    TEM is a real (non-restart) marker, so it ends the scan cleanly, and the
    payload then simply stops without ever presenting an EOI.
    """
    payload = soi() + app0_jfif() + sof(0xC0, 64, 64) + sos() + b"\x00\x01" + b"\xff\x01"
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MISSING_EOI


def test_missing_sof_when_soi_app0_eoi_present_but_no_frame_header() -> None:
    """SOI, APP0, EOI with no SOF anywhere is MISSING_SOF."""
    payload = soi() + app0_jfif() + eoi()
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MISSING_SOF


def test_trailing_data_after_eoi_is_reported_with_byte_count() -> None:
    """Extra bytes after EOI produce TRAILING_DATA, non-terminal EOI, and a byte count."""
    payload = baseline_jpeg(64, 64) + b"\x00\x00\x00extra"
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.TRAILING_DATA
    assert report.has_eoi is True
    assert report.eoi_terminal is False
    assert report.trailing_bytes == len(b"\x00\x00\x00extra")


def test_matches_checks_exact_dimensions() -> None:
    """matches() is True only for the exact width and height the SOF declared."""
    report = inspect_jpeg(baseline_jpeg(640, 480))
    assert report.matches(640, 480) is True
    assert report.matches(1280, 800) is False


def test_empty_payload_is_empty_status() -> None:
    """A zero-length payload is EMPTY, not MISSING_SOI."""
    report = inspect_jpeg(b"")
    assert report.status is JpegStatus.EMPTY
    assert report.byte_count == 0
    assert report.inspected_bytes == 0


def test_oversize_payload_is_not_parsed_and_honours_a_small_ceiling() -> None:
    """A payload past the ceiling is OVERSIZE and inspected_bytes stays at zero.

    Uses a small explicit ceiling instead of the real 2 MiB default so the
    test allocates only a few dozen bytes rather than repeatedly allocating
    megabyte-scale buffers.
    """
    payload = baseline_jpeg(64, 64)
    ceiling = len(payload) - 1
    report = inspect_jpeg(payload, ceiling=ceiling)
    assert report.status is JpegStatus.OVERSIZE
    assert report.inspected_bytes == 0
    assert report.ceiling_bytes == ceiling


def test_default_ceiling_constant_matches_two_mebibytes() -> None:
    """The documented default ceiling is exactly 2 MiB, checked once, cheaply."""
    assert MAX_INSPECT_BYTES == 2 * 1024 * 1024


def test_dht_jpg_dac_are_not_frame_headers() -> None:
    """FFC4 (DHT), FFC8 (JPG) and FFCC (DAC) must never be parsed as a SOF."""
    payload = (
        soi()
        + app0_jfif()
        + segment(0xC4, b"\x00" + b"\x00" * 16)  # DHT only, deliberately the sole C-range marker
        + eoi()
    )
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MISSING_SOF
    assert report.frame is None


def test_jpg_and_dac_markers_are_named_but_not_frames() -> None:
    """FFC8 and FFCC are recorded as segments named JPG/DAC, not SOF frames."""
    payload = soi() + app0_jfif() + segment(0xC8, b"") + segment(0xCC, b"\x00") + eoi()
    report = inspect_jpeg(payload)
    assert report.status is JpegStatus.MISSING_SOF
    names = {s.name for s in report.segments}
    assert "JPG" in names
    assert "DAC" in names


def test_report_as_dict_is_json_serializable_and_carries_key_fields() -> None:
    """as_dict() round-trips through json.dumps and carries status, dims, counts."""
    report = inspect_jpeg(baseline_jpeg(1280, 800))
    encoded = json.dumps(report.as_dict())
    decoded = json.loads(encoded)
    assert decoded["status"] == "well_formed"
    assert decoded["frame"]["width"] == 1280
    assert decoded["frame"]["height"] == 800
    assert decoded["scan_count"] == 1
    assert decoded["well_formed"] is True


def test_segment_list_is_capped_at_max_segments() -> None:
    """A payload with far more segments than MAX_SEGMENTS stops recording, and says so."""
    many_coms = b"".join(segment(0xFE, b"") for _ in range(MAX_SEGMENTS + 50))
    payload = soi() + many_coms + app0_jfif() + sof(0xC0, 64, 64) + sos() + b"\x00" + eoi()
    report = inspect_jpeg(payload)
    assert len(report.segments) <= MAX_SEGMENTS
    assert "elided" in report.detail
