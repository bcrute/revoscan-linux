"""Bounded decoder for the reviewed scanner-014 usbmon evidence.

This module intentionally is not a general usbmon command-line parser.  It
understands the text records needed by the MINI's reviewed control interface
(``1:043:0``), RGB bulk endpoint (``1:043:4``), and interrupt endpoint
(``1:043:3``).  It accepts the zero padding used by usbmon and the unpadded
form used by the first scanner-014 filter, so fixture derivation cannot silently
drop a device.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scanner3d.errors import TraceFormatError

TARGET_BUS: Final = 1
TARGET_DEVNUM: Final = 43
TARGET_SERIAL: Final = "A22704522A6F00A64"
RAW_SHA256: Final = "15b43371f426c3efb617ce58d805b9f9c04519562fb641042dfd5259b9e2d664"
FIXTURE_PATH: Final = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "scanner-014-usbmon-target.txt"
)

_RECORD = re.compile(
    r"^(?P<urb>[0-9a-fA-F]+)\s+(?P<ts>\d+)\s+(?P<event>[SC])\s+"
    r"(?P<kind>[A-Za-z]+):(?P<bus>\d+):(?P<dev>\d+):(?P<ep>\d+)\s+"
    r"(?P<tail>.*)$"
)
_STATUS_TAIL = re.compile(
    r"(?P<status>-?\d+)(?::(?P<status_extra>\d+))?\s+(?P<length>\d+)"
    r"(?:\s+(?P<direction>[<=>]))?(?:\s+(?P<data>.*))?$"
)
_CONTROL_TAIL = re.compile(
    r"s\s+(?P<request_type>[0-9a-fA-F]{2})\s+(?P<request>[0-9a-fA-F]{2})\s+"
    r"(?P<value>[0-9a-fA-F]{4})\s+(?P<index>[0-9a-fA-F]{4})\s+"
    r"(?P<requested_length>[0-9a-fA-F]{4})\s+(?P<length>\d+)"
    r"(?:\s+(?P<direction>[<=>]))?(?:\s+(?P<data>.*))?$"
)
_HEX = re.compile(r"^[0-9a-fA-F]+$")


@dataclass(frozen=True, slots=True)
class UsbmonRecord:
    urb_id: str
    timestamp_us: int
    event: str
    transfer: str
    bus: int
    devnum: int
    endpoint: int
    status: int | None
    status_extra: int | None
    length: int
    direction: str | None
    payload: bytes
    raw: str

    @property
    def submitted(self) -> bool:
        return self.event == "S"

    @property
    def completed(self) -> bool:
        return self.event == "C"

    @property
    def is_target(self) -> bool:
        return self.bus == TARGET_BUS and self.devnum == TARGET_DEVNUM

    @property
    def endpoint_address(self) -> int:
        return self.endpoint | (0x80 if self.transfer in {"Bi", "Ii", "Ci"} else 0)


@dataclass(frozen=True, slots=True)
class UvcProbeCommit:
    bm_hint: int
    format_index: int
    frame_index: int
    frame_interval: int
    key_frame_rate: int
    p_frame_rate: int
    comp_quality: int
    comp_window_size: int
    delay: int
    max_video_frame_size: int
    max_payload_transfer_size: int

    @classmethod
    def decode(cls, payload: bytes) -> UvcProbeCommit:
        if len(payload) != 26:
            raise TraceFormatError(f"UVC Probe/Commit payload must be 26 bytes, got {len(payload)}")
        return cls(*struct.unpack("<HBBIHHHHHII", payload))

    @property
    def fps(self) -> float:
        return 10_000_000 / self.frame_interval if self.frame_interval else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "bm_hint": self.bm_hint,
            "format_index": self.format_index,
            "frame_index": self.frame_index,
            "frame_interval_100ns": self.frame_interval,
            "fps": self.fps,
            "key_frame_rate": self.key_frame_rate,
            "p_frame_rate": self.p_frame_rate,
            "comp_quality": self.comp_quality,
            "comp_window_size": self.comp_window_size,
            "delay": self.delay,
            "max_video_frame_size": self.max_video_frame_size,
            "max_payload_transfer_size": self.max_payload_transfer_size,
        }


@dataclass(frozen=True, slots=True)
class ControlPair:
    submission: UsbmonRecord
    completion: UsbmonRecord | None
    request_type: int | None
    request: int | None
    value: int | None
    index: int | None
    requested_length: int | None
    payload: bytes

    @property
    def successful(self) -> bool:
        return self.completion is not None and self.completion.status == 0

    @property
    def decoded(self) -> UvcProbeCommit | None:
        if len(self.payload) != 26:
            return None
        return UvcProbeCommit.decode(self.payload)


@dataclass(frozen=True, slots=True)
class BulkPair:
    submission: UsbmonRecord
    completion: UsbmonRecord | None

    @property
    def submitted_errno(self) -> int | None:
        return self.submission.status

    @property
    def completion_status(self) -> int | None:
        return None if self.completion is None else self.completion.status

    @property
    def payload_bearing(self) -> bool:
        return (
            self.completion is not None
            and self.completion.status == 0
            and self.completion.length > 0
        )

    @property
    def actual_length(self) -> int | None:
        return None if self.completion is None else self.completion.length

    @property
    def captured_prefix_length(self) -> int:
        return 0 if self.completion is None else len(self.completion.payload)

    @property
    def successful_completion(self) -> bool:
        return self.completion is not None and self.completion.status == 0

    @property
    def zero_length_success(self) -> bool:
        return (
            self.successful_completion
            and self.completion is not None
            and self.completion.length == 0
        )

    @property
    def nonzero_length_success(self) -> bool:
        return (
            self.successful_completion and self.actual_length is not None and self.actual_length > 0
        )

    @property
    def prefix_captured(self) -> bool:
        return self.nonzero_length_success and self.captured_prefix_length > 0


@dataclass(frozen=True, slots=True)
class UsbmonDecode:
    records: tuple[UsbmonRecord, ...]
    controls: tuple[ControlPair, ...]
    bulk: tuple[BulkPair, ...]
    unmatched_submissions: tuple[UsbmonRecord, ...]
    malformed_lines: tuple[str, ...]
    non_target_records: int = 0
    unmatched_completions: tuple[UsbmonRecord, ...] = ()

    @property
    def successful_bulk_completions(self) -> tuple[BulkPair, ...]:
        return tuple(p for p in self.bulk if p.payload_bearing)

    @property
    def successful_bulk_transfers(self) -> tuple[BulkPair, ...]:
        return tuple(p for p in self.bulk if p.successful_completion)

    @property
    def zero_length_bulk_completions(self) -> tuple[BulkPair, ...]:
        return tuple(p for p in self.bulk if p.zero_length_success)

    @property
    def captured_prefix_limitations(self) -> tuple[str, ...]:
        limitations = [
            "usbmon text records preserve only the printed payload prefix; they are not a full "
            "image artifact"
        ]
        if not self.successful_bulk_transfers:
            limitations.append("no successful bulk completion was captured for endpoint 0x84")
        else:
            zero_length = self.zero_length_bulk_completions
            if zero_length:
                limitations.append(
                    f"{len(zero_length)} successful completion(s) reported zero actual bytes"
                )
            unreported = tuple(
                p
                for p in self.successful_bulk_transfers
                if p.nonzero_length_success and not p.prefix_captured
            )
            if unreported:
                actual = sum(p.actual_length or 0 for p in unreported)
                limitations.append(
                    f"{len(unreported)} successful completion(s) reported {actual} actual bytes "
                    "without a captured text prefix"
                )
        return tuple(limitations)

    def as_dict(self) -> dict[str, object]:
        controls: list[dict[str, object]] = []
        for pair in self.controls:
            controls.append(
                {
                    "urb_id": pair.submission.urb_id,
                    "request_type": pair.request_type,
                    "request": pair.request,
                    "value": pair.value,
                    "index": pair.index,
                    "requested_length": pair.requested_length,
                    "completion_status": (
                        None if pair.completion is None else pair.completion.status
                    ),
                    "actual_length": (None if pair.completion is None else pair.completion.length),
                    "probe_commit": None if pair.decoded is None else pair.decoded.as_dict(),
                }
            )
        return {
            "record_count": len(self.records),
            "control_count": len(self.controls),
            "bulk_count": len(self.bulk),
            "controls": controls,
            "bulk": [
                {
                    "urb_id": pair.submission.urb_id,
                    "submitted_status": pair.submitted_errno,
                    "completion_status": pair.completion_status,
                    "completion_actual_length": (
                        None if pair.completion is None else pair.completion.length
                    ),
                    "successful_completion": pair.successful_completion,
                    "zero_length_success": pair.zero_length_success,
                    "nonzero_length_success": pair.nonzero_length_success,
                    "prefix_captured": pair.prefix_captured,
                    "payload_bearing": pair.payload_bearing,
                    "captured_prefix_length": pair.captured_prefix_length,
                }
                for pair in self.bulk
            ],
            "successful_bulk_completion_count": len(self.successful_bulk_completions),
            "successful_bulk_transfer_count": len(self.successful_bulk_transfers),
            "zero_length_bulk_completion_count": len(self.zero_length_bulk_completions),
            "unmatched_submission_count": len(self.unmatched_submissions),
            "unmatched_completion_count": len(self.unmatched_completions),
            "malformed_line_count": len(self.malformed_lines),
            "captured_prefix_limitations": list(self.captured_prefix_limitations),
        }


def _payload(data: str | None) -> bytes:
    if not data or data in {">", "<"}:
        return b""
    if "=" in data:
        data = data.split("=", 1)[1].strip()
    tokens = data.split()
    if not tokens or any(not _HEX.fullmatch(token) or len(token) % 2 for token in tokens):
        return b""
    try:
        return bytes.fromhex("".join(tokens))
    except ValueError:
        return b""


def parse_usbmon_line(line: str) -> UsbmonRecord:
    """Parse one usbmon text record, rejecting malformed records."""
    raw = line.rstrip("\n")
    match = _RECORD.fullmatch(raw.strip())
    if match is None:
        raise TraceFormatError(f"malformed usbmon record: {raw!r}")
    groups = match.groupdict()
    tail = groups.pop("tail") or ""
    tail_match = (
        _CONTROL_TAIL.fullmatch(tail) if tail.startswith("s ") else _STATUS_TAIL.fullmatch(tail)
    )
    if tail_match is None:
        raise TraceFormatError(f"malformed usbmon record: {raw!r}")
    tail_groups = tail_match.groupdict()
    groups.update(tail_groups)
    payload = _payload(groups.get("data"))
    return UsbmonRecord(
        urb_id=groups["urb"],
        timestamp_us=int(groups["ts"]),
        event=groups["event"],
        transfer=groups["kind"],
        bus=int(groups["bus"]),
        devnum=int(groups["dev"]),
        endpoint=int(groups["ep"]),
        status=(None if tail.startswith("s ") else int(groups["status"])),
        status_extra=(None if groups.get("status_extra") is None else int(groups["status_extra"])),
        length=int(groups["length"]),
        direction=groups["direction"],
        payload=payload,
        raw=raw,
    )


def _control_fields(
    record: UsbmonRecord,
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    if record.transfer not in {"Co", "Ci"}:
        return (None, None, None, None, None)
    match = re.search(
        r"\s+s\s+([0-9a-fA-F]{2})\s+([0-9a-fA-F]{2})\s+([0-9a-fA-F]{4})\s+([0-9a-fA-F]{4})\s+([0-9a-fA-F]{4})\s+\d+",
        record.raw,
    )
    if match is None:
        return (None, None, None, None, None)
    return tuple(int(value, 16) for value in match.groups())  # type: ignore[return-value]


def decode_usbmon(lines: Iterable[str], *, strict: bool = True) -> UsbmonDecode:
    """Decode and pair reviewed target records by URB id.

    Submission status ``-115`` is retained as the kernel's in-progress marker;
    it is never treated as a completion.  Cleanup completion ``-2`` with zero
    bytes is retained as cancellation and never promoted to payload.
    """
    records: list[UsbmonRecord] = []
    malformed: list[str] = []
    non_target = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = parse_usbmon_line(line)
        except TraceFormatError:
            if strict:
                raise
            malformed.append(line.rstrip("\n"))
            continue
        if not record.is_target:
            non_target += 1
            continue
        records.append(record)

    controls: list[ControlPair] = []
    bulk: list[BulkPair] = []
    unmatched: list[UsbmonRecord] = []
    used_completions: set[int] = set()
    for record in records:
        if not record.submitted:
            continue
        completion_index: int | None = None
        completion: UsbmonRecord | None = None
        for candidate_index, candidate in enumerate(records):
            if candidate_index in used_completions:
                continue
            if (
                candidate.urb_id == record.urb_id
                and candidate.completed
                and candidate.transfer == record.transfer
                and candidate.bus == record.bus
                and candidate.devnum == record.devnum
                and candidate.endpoint == record.endpoint
                and candidate.timestamp_us >= record.timestamp_us
            ):
                completion_index, completion = candidate_index, candidate
                break
        if completion_index is not None:
            used_completions.add(completion_index)
        if completion is None:
            unmatched.append(record)
        if record.transfer in {"Co", "Ci"}:
            fields = _control_fields(record)
            control_payload = record.payload
            if completion is not None and completion.payload:
                control_payload = completion.payload
            controls.append(ControlPair(record, completion, *fields, control_payload))
        elif record.transfer == "Bi":
            bulk.append(BulkPair(record, completion))
    unmatched_completions = tuple(
        record
        for index, record in enumerate(records)
        if record.completed and index not in used_completions
    )
    return UsbmonDecode(
        tuple(records),
        tuple(controls),
        tuple(bulk),
        tuple(unmatched),
        tuple(malformed),
        non_target,
        unmatched_completions,
    )


def decode_fixture(path: Path) -> UsbmonDecode:
    """Decode an explicitly supplied fixture path.

    Evidence is not silently loaded from a current-working-directory or an
    installed package's absent ``tests`` tree.  Callers must name the fixture
    they intend to audit.
    """
    return decode_usbmon(path.read_text(encoding="ascii").splitlines())
