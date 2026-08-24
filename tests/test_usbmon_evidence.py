"""Offline scanner-014 usbmon fixture tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scanner3d.errors import TraceFormatError
from scanner3d.usbmon_evidence import decode_fixture, decode_usbmon, parse_usbmon_line

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURE_DIR / "scanner-014-usbmon-target.txt"
METADATA = FIXTURE_DIR / "scanner-014-usbmon-target.json"


def test_fixture_metadata_is_provenance_complete_and_matches_bytes() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["source"] == "/tmp/scanner-014-evidence/usbmon-raw.txt"
    assert "zero-padded" in metadata["selection_rule"]
    assert (
        metadata["source_sha256"]
        == "15b43371f426c3efb617ce58d805b9f9c04519562fb641042dfd5259b9e2d664"
    )
    assert metadata["source_line_count"] == 810
    assert metadata["source_byte_count"] == 58074
    assert metadata["target"] == {
        "vendor_id": "2207",
        "product_id": "110c",
        "serial": "A22704522A6F00A64",
        "bus": 1,
        "devnum": 43,
    }
    assert metadata["selected_record_count"] == 26
    assert metadata["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert len(FIXTURE.read_bytes().splitlines()) == metadata["selected_record_count"]
    assert ":43:" in metadata["known_wrapper_filter_defect"]
    assert ":043:" in metadata["known_wrapper_filter_defect"]


def test_scanner_014_fixture_reproduces_negotiation_and_no_completion() -> None:
    decoded = decode_fixture(FIXTURE)
    assert len(decoded.records) == 26
    assert len(decoded.controls) == 7
    assert len(decoded.bulk) == 5
    assert not decoded.unmatched_submissions
    assert not decoded.unmatched_completions
    assert not decoded.successful_bulk_completions
    probe_sets = [c for c in decoded.controls if c.request == 1 and c.value == 0x0100]
    commit = next(c for c in decoded.controls if c.request == 1 and c.value == 0x0200)
    assert len(probe_sets) == 2
    assert commit.successful
    current = next(c for c in decoded.controls if c.request == 0x81)
    assert current.decoded is not None
    assert current.decoded.frame_interval == 333333
    assert current.decoded.max_video_frame_size == 8192000
    assert current.decoded.max_payload_transfer_size == 524288
    maximum = next(c for c in decoded.controls if c.request == 0x83)
    assert maximum.decoded is not None and maximum.decoded.frame_interval == 1000000
    assert all(pair.submitted_errno == -115 for pair in decoded.bulk)
    assert all(pair.completion_status == -2 for pair in decoded.bulk)
    assert all(not pair.payload_bearing for pair in decoded.bulk)


def test_zero_padded_and_unpadded_devnums_are_equivalent() -> None:
    line = "ffff 1 S Bi:1:043:4 -115 16384 <"
    assert parse_usbmon_line(line).devnum == parse_usbmon_line(line.replace(":043:", ":43:")).devnum


@pytest.mark.parametrize(
    "line", ["", "garbage", "ffff 1 S Bi:1:043", "ffff 1 X Bi:1:043:4 -115 0 <"]
)
def test_malformed_lines_are_rejected(line: str) -> None:
    with pytest.raises(TraceFormatError):
        parse_usbmon_line(line)


def test_truncated_probe_payload_is_retained_but_not_decoded() -> None:
    decoded = decode_usbmon(
        [
            (
                "ffff 1 S Co:1:43:0 s 21 01 0100 0003 001a 20 = "
                "01000101 15160500 00000000 00000000 0000"
            ),
            "ffff 1 C Co:1:43:0 0 20 >",
        ]
    )
    assert len(decoded.controls) == 1
    assert decoded.controls[0].successful
    assert decoded.controls[0].decoded is None


def test_non_strict_decode_reports_malformed_and_unmatched() -> None:
    decoded = decode_usbmon(
        [
            "ffff 1 S Bi:1:43:4 -115 16384 <",
            "bad",
        ],
        strict=False,
    )
    assert len(decoded.malformed_lines) == 1
    assert len(decoded.unmatched_submissions) == 1


def test_mixed_devices_are_ignored_without_cross_pairing() -> None:
    decoded = decode_usbmon(
        [
            "aaaa 1 S Bi:1:42:4 -115 16384 <",
            "aaaa 2 C Bi:1:42:4 0 3 = aa bb cc",
            "bbbb 3 S Bi:1:43:4 -115 16384 <",
            "bbbb 4 C Bi:1:43:4 -2 0",
        ]
    )
    assert decoded.non_target_records == 2
    assert len(decoded.bulk) == 1
    assert decoded.bulk[0].completion_status == -2


def test_cleanup_cancel_is_not_payload() -> None:
    decoded = decode_usbmon(
        [
            "ffff 1 S Bi:1:43:4 -115 16384 <",
            "ffff 2 C Bi:1:43:4 -2 0",
        ]
    )
    pair = decoded.bulk[0]
    assert pair.submitted_errno == -115
    assert pair.completion_status == -2
    assert not pair.payload_bearing


def test_successful_actual_bytes_without_prefix_are_not_described_as_zero() -> None:
    decoded = decode_usbmon(
        [
            "ffff 1 S Bi:1:43:4 -115 16384 <",
            "ffff 1 C Bi:1:43:4 0 4",
        ]
    )
    pair = decoded.bulk[0]
    assert pair.successful_completion
    assert pair.actual_length == 4
    assert pair.payload_bearing
    assert pair.captured_prefix_length == 0
    assert any("reported 4 actual bytes" in note for note in decoded.captured_prefix_limitations)
    assert not any("carried zero bytes" in note for note in decoded.captured_prefix_limitations)


def test_successful_zero_length_completion_is_distinct_from_payload() -> None:
    decoded = decode_usbmon(
        [
            "ffff 1 S Bi:1:43:4 -115 16384 <",
            "ffff 1 C Bi:1:43:4 0 0",
        ]
    )
    pair = decoded.bulk[0]
    assert pair.successful_completion
    assert pair.zero_length_success
    assert pair.actual_length == 0
    assert pair.captured_prefix_length == 0
    assert not pair.payload_bearing
    assert len(decoded.successful_bulk_transfers) == 1
    assert len(decoded.successful_bulk_completions) == 0


def test_successful_prefix_is_reported_as_payload_but_not_full_frame() -> None:
    decoded = decode_usbmon(
        [
            "ffff 1 S Bi:1:43:4 -115 16384 <",
            "ffff 1 C Bi:1:43:4 0 4 = ff d8 ff e0",
        ]
    )
    assert decoded.successful_bulk_completions[0].payload_bearing
    assert "not a full image artifact" in decoded.captured_prefix_limitations[0]


def test_reused_urb_id_cannot_reuse_one_completion() -> None:
    decoded = decode_usbmon(
        [
            "ffff 1 S Bi:1:43:4 -115 16384 <",
            "ffff 2 S Bi:1:43:4 -115 16384 <",
            "ffff 1 C Bi:1:43:4 -2 0",
        ]
    )
    assert decoded.bulk[0].completion_status == -2
    assert decoded.bulk[1].completion_status is None
    assert len(decoded.unmatched_submissions) == 1


def test_unmatched_completion_is_reported() -> None:
    decoded = decode_usbmon(["ffff 2 C Bi:1:43:4 -2 0"])
    assert len(decoded.unmatched_completions) == 1
    assert decoded.unmatched_completions[0].status == -2
