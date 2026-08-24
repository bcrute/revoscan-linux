"""Tests for the offline usbfs trace decoder and the LD_PRELOAD tracer it reads.

``src/scanner3d/usbfs_trace.py`` is the only code in this repository that
interprets what the vendor scanner said on the wire, and it does so from a
file - never from a live device. Its two promises are the ones worth
protecting with tests: (1) a file that is not a trace at all raises
``TraceFormatError`` immediately, and (2) damage *inside* an otherwise valid
trace never raises - it is reported as a ``TraceProblem`` so a partial capture
stays usable as evidence. Everything else (URB correlation across threads,
out-of-order completions, truncated payloads, grouping) exists because a real
capture is asynchronous and lossy, and the decoder's job is to say exactly
what it did and did not manage to observe.

Every trace fixture below is built by ``TraceBuilder``, a tiny synthetic-trace
assembler defined in this file: invented pointers, invented endpoints,
invented payload bytes. None of it is data captured from the real scanner.

The last section exercises ``reference/tracer/usbfs_trace.c`` itself - the
LD_PRELOAD shared object that produces these traces. Those tests build the
tracer with the repository's own ``build.sh`` and run small child processes
under it; they skip cleanly (never fail) when a precondition such as a C
compiler or a readable, non-scanner usbfs node is not available on the host.
It also covers four later behavior changes: reported/clamped capture limits,
the output-byte cap's reserved room for its own ``limit`` record, descriptor
adoption from a raw ``openat`` syscall the open hooks cannot see, and
``errno`` transparency across every hooked call.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scanner3d.cli import main
from scanner3d.errors import EXIT_CODES, DoctorStatus, TraceFormatError
from scanner3d.usbfs_trace import TRACE_FORMAT, Exchange, load_trace, parse_trace, summarize

# ------------------------------------------------------------- trace builder

DEFAULT_TARGET = "/dev/bus/usb/001/041"
DEFAULT_PID = 4242


def header_record(**overrides: object) -> dict[str, object]:
    """A minimal, valid ``header`` record; every field is fabricated."""
    record: dict[str, object] = {
        "op": "header",
        "format": TRACE_FORMAT,
        "version": 1,
        "target": DEFAULT_TARGET,
        "target_rdev_major": 189,
        "target_rdev_minor": 40,
        "pid": DEFAULT_PID,
        "max_payload_bytes": 256,
        "max_output_bytes": 33_554_432,
        "clock": "CLOCK_MONOTONIC",
        "started_realtime_ns": 1_700_000_000_000_000_000,
        "limitations": ["synthetic fixture trace; invented for tests"],
    }
    record.update(overrides)
    return record


class TraceBuilder:
    """Accumulates a header and event records, and renders them as trace text.

    Each call to one of the module-level ``*_event``/``bulk_*``/``control_*``
    helpers below appends one record; :meth:`raw` appends a literal line
    (used to inject damage at an exact position in the stream). Line order is
    call order, matching how the real tracer emits records in completion
    order rather than submission order.
    """

    def __init__(self, **header_overrides: object) -> None:
        self.header = header_record(**header_overrides)
        self._records: list[dict[str, object] | str] = []
        self._seq = 0

    def add(self, op: str, *, seq: int | None = None, **fields: object) -> dict[str, object]:
        pid = fields.pop("pid", self.header["pid"])
        tid = fields.pop("tid", 1)
        t_ns = fields.pop("t_ns", None)
        self._seq += 1
        record_seq = seq if seq is not None else self._seq
        record: dict[str, object] = {
            "op": op,
            "seq": record_seq,
            "t_ns": t_ns if t_ns is not None else record_seq * 1000,
            "pid": pid,
            "tid": tid,
            **fields,
        }
        self._records.append(record)
        return record

    def raw(self, line: str) -> None:
        """Append a literal line verbatim, bypassing the record shape."""
        self._records.append(line)

    def lines(self) -> list[str]:
        out = [json.dumps(self.header)]
        for record in self._records:
            out.append(record if isinstance(record, str) else json.dumps(record))
        return out

    def text(self) -> str:
        return "\n".join(self.lines()) + "\n"


def open_event(builder: TraceBuilder, *, fd: int, path: str | None = None) -> dict[str, object]:
    target = builder.header["target"]
    assert isinstance(target, str)
    return builder.add("open", fd=fd, path=path or target)


def close_event(builder: TraceBuilder, *, fd: int) -> dict[str, object]:
    return builder.add("close", fd=fd)


def discard(builder: TraceBuilder, *, fd: int, urb: str, **extra: Any) -> dict[str, object]:
    return _base_ioctl(
        builder,
        fd=fd,
        name="USBDEVFS_DISCARDURB",
        request="0x0000550b",
        phase="discard",
        urb=urb,
        **extra,
    )


def _base_ioctl(
    builder: TraceBuilder,
    *,
    fd: int,
    name: str,
    request: str,
    ret: int = 0,
    errno_name: str | None = None,
    **fields: object,
) -> dict[str, object]:
    kw: dict[str, Any] = dict(fields)
    kw["fd"] = fd
    kw["request"] = request
    kw["name"] = name
    kw["ret"] = ret
    kw.setdefault("duration_ns", 1000)
    if ret < 0:
        kw["errno"] = 1
        kw["errno_name"] = errno_name or "EPIPE"
    return builder.add("ioctl", **kw)


def _urb_common(
    *,
    phase: str,
    urb: str,
    urb_type: int,
    urb_type_name: str,
    endpoint: int,
    buffer_length: int,
    actual_length: int = 0,
    status: int = 0,
) -> dict[str, object]:
    return {
        "phase": phase,
        "urb": urb,
        "urb_type": urb_type,
        "urb_type_name": urb_type_name,
        "endpoint": endpoint,
        "endpoint_number": endpoint & 0x0F,
        "direction": "in" if endpoint & 0x80 else "out",
        "status": status,
        "flags": 0,
        "buffer_length": buffer_length,
        "actual_length": actual_length,
        "start_frame": 0,
        "number_of_packets": 0,
        "error_count": 0,
        "signr": 0,
        "usercontext": "0x0",
    }


def _payload_fields(data: bytes, *, available: int | None, truncated: bool) -> dict[str, object]:
    return {
        "payload_available": available if available is not None else len(data),
        "payload_captured": len(data),
        "payload_truncated": truncated,
        "data": data.hex(),
    }


def bulk_submit(
    builder: TraceBuilder,
    *,
    fd: int,
    urb: str,
    endpoint: int,
    data: bytes = b"",
    buffer_length: int | None = None,
    ret: int = 0,
    errno_name: str | None = None,
    payload_available: int | None = None,
    payload_truncated: bool = False,
    **extra: object,
) -> dict[str, object]:
    if ret < 0:
        fields: dict[str, object] = {"phase": "submit", "urb": urb}
    else:
        length = buffer_length if buffer_length is not None else len(data)
        fields = _urb_common(
            phase="submit",
            urb=urb,
            urb_type=3,
            urb_type_name="bulk",
            endpoint=endpoint,
            buffer_length=length,
        )
        fields.update(
            _payload_fields(data, available=payload_available, truncated=payload_truncated)
        )
    fields.update(extra)
    return _base_ioctl(
        builder,
        fd=fd,
        name="USBDEVFS_SUBMITURB",
        request="0x8038550a",
        ret=ret,
        errno_name=errno_name,
        **fields,
    )


def bulk_reap(
    builder: TraceBuilder,
    *,
    fd: int,
    urb: str,
    endpoint: int,
    data: bytes = b"",
    buffer_length: int | None = None,
    actual_length: int | None = None,
    status: int = 0,
    ret: int = 0,
    errno_name: str | None = None,
    payload_available: int | None = None,
    payload_truncated: bool = False,
    reap_name: str = "USBDEVFS_REAPURB",
    **extra: object,
) -> dict[str, object]:
    length = buffer_length if buffer_length is not None else len(data)
    actual = actual_length if actual_length is not None else len(data)
    fields = _urb_common(
        phase="reap",
        urb=urb,
        urb_type=3,
        urb_type_name="bulk",
        endpoint=endpoint,
        buffer_length=length,
        actual_length=actual,
        status=status,
    )
    fields.update(_payload_fields(data, available=payload_available, truncated=payload_truncated))
    fields.update(extra)
    return _base_ioctl(
        builder,
        fd=fd,
        name=reap_name,
        request="0x8008551b",
        ret=ret,
        errno_name=errno_name,
        **fields,
    )


def control_submit(
    builder: TraceBuilder,
    *,
    fd: int,
    urb: str,
    bm_request_type: int = 0x40,
    b_request: int = 0x01,
    w_value: int = 0,
    w_index: int = 0,
    w_length: int = 0,
    data: bytes = b"",
    ret: int = 0,
    errno_name: str | None = None,
    **extra: object,
) -> dict[str, object]:
    if ret < 0:
        fields: dict[str, object] = {"phase": "submit", "urb": urb}
    else:
        buffer_length = 8 + w_length
        fields = _urb_common(
            phase="submit",
            urb=urb,
            urb_type=2,
            urb_type_name="control",
            endpoint=0x00,
            buffer_length=buffer_length,
        )
        fields.update(
            bmRequestType=bm_request_type,
            bRequest=b_request,
            wValue=w_value,
            wIndex=w_index,
            wLength=w_length,
            setup_direction="in" if bm_request_type & 0x80 else "out",
        )
        fields.update(_payload_fields(data, available=None, truncated=False))
    fields.update(extra)
    return _base_ioctl(
        builder,
        fd=fd,
        name="USBDEVFS_SUBMITURB",
        request="0x8038550a",
        ret=ret,
        errno_name=errno_name,
        **fields,
    )


def control_reap(
    builder: TraceBuilder,
    *,
    fd: int,
    urb: str,
    bm_request_type: int = 0xC0,
    b_request: int = 0x01,
    w_value: int = 0,
    w_index: int = 0,
    w_length: int = 0,
    data: bytes = b"",
    actual_length: int | None = None,
    status: int = 0,
    ret: int = 0,
    errno_name: str | None = None,
    reap_name: str = "USBDEVFS_REAPURB",
    **extra: object,
) -> dict[str, object]:
    actual = actual_length if actual_length is not None else len(data)
    buffer_length = 8 + w_length
    fields = _urb_common(
        phase="reap",
        urb=urb,
        urb_type=2,
        urb_type_name="control",
        endpoint=0x00,
        buffer_length=buffer_length,
        actual_length=actual,
        status=status,
    )
    fields.update(
        bmRequestType=bm_request_type,
        bRequest=b_request,
        wValue=w_value,
        wIndex=w_index,
        wLength=w_length,
        setup_direction="in" if bm_request_type & 0x80 else "out",
    )
    fields.update(_payload_fields(data, available=None, truncated=False))
    fields.update(extra)
    return _base_ioctl(
        builder,
        fd=fd,
        name=reap_name,
        request="0x8008551b",
        ret=ret,
        errno_name=errno_name,
        **fields,
    )


def _write_trace(path: Path, builder: TraceBuilder) -> Path:
    path.write_text(builder.text(), encoding="utf-8")
    return path


# ------------------------------------------------------ 1. header validation


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param([], id="empty_file"),
        pytest.param(["not json at all"], id="first_line_not_json"),
        pytest.param(["[1, 2, 3]"], id="first_line_is_array"),
        pytest.param([json.dumps({"op": "ioctl", "seq": 1})], id="first_record_not_header"),
        pytest.param([json.dumps(header_record(format="not-scanner3d"))], id="wrong_format"),
        pytest.param([json.dumps(header_record(version=99))], id="unsupported_version"),
        pytest.param(
            [json.dumps({k: v for k, v in header_record().items() if k != "target"})],
            id="header_missing_target",
        ),
    ],
)
def test_malformed_header_raises_trace_format_error(lines: list[str]) -> None:
    with pytest.raises(TraceFormatError):
        parse_trace(lines)


# ------------------------------------------------------ 2. damage as evidence


def test_mid_stream_damage_is_recorded_as_problems_not_raised() -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)  # line 2
    bulk_submit(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x01")  # line 3
    builder.raw("{not valid json at all")  # line 4: corrupt/garbage line
    builder.raw("42")  # line 5: JSON scalar line
    builder.raw(json.dumps({"seq": 500, "fd": 5}))  # line 6: record with no op
    builder.raw(json.dumps(header_record()))  # line 7: duplicate header record
    bulk_reap(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x02")  # line 8

    trace = parse_trace(builder.lines())

    assert len(trace.problems) == 4
    by_line = {p.line_number: p for p in trace.problems}
    assert set(by_line) == {4, 5, 6, 7}
    assert "not valid JSON" in by_line[4].reason
    assert "not an object" in by_line[5].reason
    assert "no op" in by_line[6].reason
    assert "duplicate header" in by_line[7].reason

    # The valid events surrounding the damage still parsed and correlated.
    assert len(trace.transfers) == 1
    assert trace.transfers[0].correlation == "matched"
    assert trace.transfers[0].request_data == b"\x01"
    assert trace.transfers[0].response_data == b"\x02"


def test_truncated_final_line_is_recorded_as_a_problem() -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    bulk_submit(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x01")
    bulk_reap(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x02")
    lines = builder.lines()
    # Simulate a process killed mid-write: the last line is cut off.
    lines.append('{"seq": 999, "op": "ioctl", "fd": 5, "phase"')

    trace = parse_trace(lines)

    assert len(trace.problems) == 1
    assert trace.problems[0].line_number == len(lines)
    assert "not valid JSON" in trace.problems[0].reason
    assert len(trace.transfers) == 1
    assert trace.transfers[0].correlation == "matched"


# ---------------------------------------------------------- 3. transfer shape


def test_bulk_out_transfer_correlates_and_decodes_payload() -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    request_bytes = bytes.fromhex("deadbeef01020304")
    response_bytes = bytes.fromhex("cafef00d")
    bulk_submit(builder, fd=5, urb="0x1000", endpoint=0x02, data=request_bytes)
    bulk_reap(builder, fd=5, urb="0x1000", endpoint=0x02, data=response_bytes, status=0)

    trace = parse_trace(builder.lines())

    assert len(trace.transfers) == 1
    transfer = trace.transfers[0]
    assert transfer.correlation == "matched"
    assert transfer.endpoint == 0x02
    assert transfer.direction == "out"
    assert transfer.transfer_type == "bulk"
    assert transfer.requested_bytes == len(request_bytes)
    assert transfer.actual_bytes == len(response_bytes)
    assert transfer.status == 0
    assert transfer.request_data == request_bytes
    assert transfer.response_data == response_bytes


def test_control_transfer_setup_and_direction_from_setup_packet() -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    response_bytes = bytes.fromhex("1201000200000040")
    control_submit(
        builder,
        fd=5,
        urb="0x2000",
        bm_request_type=0x80,
        b_request=0x06,
        w_value=0x0100,
        w_index=0x0000,
        w_length=8,
    )
    control_reap(
        builder,
        fd=5,
        urb="0x2000",
        bm_request_type=0x80,
        b_request=0x06,
        w_value=0x0100,
        w_index=0x0000,
        w_length=8,
        data=response_bytes,
    )

    trace = parse_trace(builder.lines())

    assert len(trace.transfers) == 1
    transfer = trace.transfers[0]
    assert transfer.correlation == "matched"
    assert transfer.transfer_type == "control"
    # A control IN on endpoint 0x00 reports direction "in" from setup_direction.
    assert transfer.direction == "in"
    assert transfer.setup == {
        "bmRequestType": 0x80,
        "bRequest": 0x06,
        "wValue": 0x0100,
        "wIndex": 0x0000,
        "wLength": 8,
    }
    assert transfer.response_data == response_bytes


# ------------------------------------------------------- 4. correlation keys


def test_two_concurrent_urbs_correlate_to_correct_partners() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x1000", endpoint=0x02, data=b"\x01\x02")
    bulk_submit(builder, fd=5, urb="0x2000", endpoint=0x04, data=b"\x03\x04")
    bulk_reap(builder, fd=5, urb="0x2000", endpoint=0x04, data=b"\xaa")
    bulk_reap(builder, fd=5, urb="0x1000", endpoint=0x02, data=b"\xbb")

    trace = parse_trace(builder.lines())

    by_urb = {t.urb: t for t in trace.transfers}
    assert by_urb["0x1000"].request_data == b"\x01\x02"
    assert by_urb["0x1000"].response_data == b"\xbb"
    assert by_urb["0x2000"].request_data == b"\x03\x04"
    assert by_urb["0x2000"].response_data == b"\xaa"
    assert {t.correlation for t in trace.transfers} == {"matched"}


def test_urb_pointer_reuse_after_completion_correlates_sequentially() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x01")
    bulk_reap(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x11")
    bulk_submit(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x02")
    bulk_reap(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x22")

    trace = parse_trace(builder.lines())

    assert len(trace.transfers) == 2
    assert [t.correlation for t in trace.transfers] == ["matched", "matched"]
    assert trace.transfers[0].request_data == b"\x01"
    assert trace.transfers[0].response_data == b"\x11"
    assert trace.transfers[1].request_data == b"\x02"
    assert trace.transfers[1].response_data == b"\x22"


def test_same_urb_pointer_different_pid_does_not_correlate() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x3000", endpoint=0x02, data=b"\x01", pid=100)
    bulk_submit(builder, fd=5, urb="0x3000", endpoint=0x02, data=b"\x02", pid=200)
    bulk_reap(builder, fd=5, urb="0x3000", endpoint=0x02, data=b"\xaa", pid=200)

    trace = parse_trace(builder.lines())

    by_pid = {t.submit.pid: t for t in trace.transfers if t.submit is not None}
    assert by_pid[200].correlation == "matched"
    assert by_pid[200].response_data == b"\xaa"
    assert by_pid[100].correlation == "submit_only"


def test_same_urb_pointer_different_fd_does_not_correlate() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x8000", endpoint=0x02, data=b"\x01")
    bulk_submit(builder, fd=6, urb="0x8000", endpoint=0x02, data=b"\x02")
    bulk_reap(builder, fd=6, urb="0x8000", endpoint=0x02, data=b"\xaa")

    trace = parse_trace(builder.lines())

    by_fd = {t.submit.fd: t for t in trace.transfers if t.submit is not None}
    assert by_fd[6].correlation == "matched"
    assert by_fd[6].response_data == b"\xaa"
    assert by_fd[5].correlation == "submit_only"


# --------------------------------------------------- 5. concurrency ordering


def test_reap_before_its_own_submit_is_deferred() -> None:
    builder = TraceBuilder()
    # A blocking REAPURB entered at t=100 and did not return until t=400, by
    # which time the URB submitted at t=200 had completed. The reaping thread
    # wrote its record first because the submitting thread had not yet taken
    # the output lock, so the completion appears before the submission.
    bulk_reap(
        builder,
        fd=5,
        urb="0x4000",
        endpoint=0x82,
        data=b"\x99",
        tid=201,
        t_ns=100,
        duration_ns=300,
    )
    bulk_submit(
        builder,
        fd=5,
        urb="0x4000",
        endpoint=0x82,
        data=b"",
        tid=200,
        t_ns=200,
        duration_ns=10,
    )

    trace = parse_trace(builder.lines())

    assert len(trace.transfers) == 1
    transfer = trace.transfers[0]
    assert transfer.correlation == "deferred"
    assert transfer.response_data == b"\x99"


def test_deferred_reap_is_not_stolen_by_a_later_reuse_of_the_urb_pointer() -> None:
    """A completion is never paired with a submission that came after it.

    The physically valid stream here is: URB 0x9000 submitted at t=100, reaped
    at t=110, and the same address reused for a fresh submission at t=120 whose
    own completion is outside the capture. Emission order inverts the first
    two records, which is exactly the case where matching by file order pairs
    the completion with the *reuse* and calls the real submission incomplete.
    """
    builder = TraceBuilder()
    bulk_reap(
        builder,
        fd=5,
        urb="0x9000",
        endpoint=0x82,
        data=b"\xaa",
        tid=201,
        t_ns=110,
        duration_ns=1,
    )
    bulk_submit(
        builder,
        fd=5,
        urb="0x9000",
        endpoint=0x02,
        data=b"\x02",
        tid=200,
        t_ns=120,
        duration_ns=1,
    )
    bulk_submit(
        builder,
        fd=5,
        urb="0x9000",
        endpoint=0x82,
        data=b"",
        tid=200,
        t_ns=100,
        duration_ns=1,
    )

    trace = parse_trace(builder.lines())

    paired = next(t for t in trace.transfers if t.reap is not None)
    assert paired.correlation == "deferred"
    assert paired.submit is not None
    assert paired.submit.t_ns == 100  # the submission that could precede it
    assert paired.response_data == b"\xaa"

    reuse = next(t for t in trace.transfers if t.reap is None)
    assert reuse.correlation == "submit_only"  # preserved, not silently dropped
    assert reuse.request_data == b"\x02"


def test_reused_urb_pointer_pairs_by_call_time_when_records_are_reordered() -> None:
    """Two uses of one URB address, with both completion records reordered."""
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x01", t_ns=100)
    bulk_submit(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x02", t_ns=120)
    # The second use completes and is written first; the first use's record
    # only reaches the output lock afterwards.
    bulk_reap(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x22", t_ns=130)
    bulk_reap(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x11", t_ns=110)

    trace = parse_trace(builder.lines())

    pairs = {t.request_data: t.response_data for t in trace.transfers}
    assert pairs == {b"\x01": b"\x11", b"\x02": b"\x22"}
    assert all(t.correlation == "matched" for t in trace.transfers)


def test_reap_with_no_submission_that_could_precede_it_stays_reap_only() -> None:
    """A completion older than every candidate submission is not forced."""
    builder = TraceBuilder()
    bulk_reap(builder, fd=5, urb="0x9000", endpoint=0x82, data=b"\xaa", t_ns=100, duration_ns=1)
    bulk_submit(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x02", t_ns=500, duration_ns=1)

    trace = parse_trace(builder.lines())

    assert {t.correlation for t in trace.transfers} == {"reap_only", "submit_only"}


def test_correlation_falls_back_to_stream_order_without_timestamps() -> None:
    """A trace whose records carry no ``t_ns`` still correlates in order."""
    builder = TraceBuilder()
    submit = bulk_submit(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x01")
    reap = bulk_reap(builder, fd=5, urb="0x9000", endpoint=0x02, data=b"\x11")
    for record in (submit, reap):
        del record["t_ns"]
        del record["duration_ns"]

    trace = parse_trace(builder.lines())

    assert len(trace.transfers) == 1
    assert trace.transfers[0].correlation == "matched"
    assert trace.transfers[0].response_data == b"\x11"


def test_interleaved_submits_and_reaps_across_threads_match_correctly() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x5000", endpoint=0x02, data=b"\x01", tid=10)
    bulk_submit(builder, fd=5, urb="0x6000", endpoint=0x04, data=b"\x02", tid=11)
    bulk_reap(builder, fd=5, urb="0x5000", endpoint=0x02, data=b"\xa1", tid=12)
    bulk_submit(builder, fd=5, urb="0x7000", endpoint=0x06, data=b"\x03", tid=10)
    bulk_reap(builder, fd=5, urb="0x6000", endpoint=0x04, data=b"\xa2", tid=13)
    bulk_reap(builder, fd=5, urb="0x7000", endpoint=0x06, data=b"\xa3", tid=12)

    trace = parse_trace(builder.lines())

    by_urb = {t.urb: t for t in trace.transfers}
    assert by_urb["0x5000"].response_data == b"\xa1"
    assert by_urb["0x6000"].response_data == b"\xa2"
    assert by_urb["0x7000"].response_data == b"\xa3"
    assert {t.correlation for t in trace.transfers} == {"matched"}


# ------------------------------------------------------ 6. unmatched evidence


def test_unmatched_transfers_are_preserved_not_dropped() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0xA000", endpoint=0x02, data=b"\x01")  # submit_only
    bulk_reap(builder, fd=5, urb="0xB000", endpoint=0x82, data=b"\x02")  # reap_only
    bulk_submit(builder, fd=5, urb="0xC000", endpoint=0x02, ret=-1, errno_name="EPIPE")  # failed
    discarded_urb = "0xD000"
    bulk_submit(builder, fd=5, urb=discarded_urb, endpoint=0x02, data=b"\x03")
    discard(builder, fd=5, urb=discarded_urb)

    trace = parse_trace(builder.lines())

    assert len(trace.transfers) == 4
    by_correlation: dict[str, list[object]] = {}
    for transfer in trace.transfers:
        by_correlation.setdefault(transfer.correlation, []).append(transfer)
    assert len(by_correlation["submit_only"]) == 3  # 0xA000, failed 0xC000, discarded 0xD000
    assert len(by_correlation["reap_only"]) == 1

    failed = next(t for t in trace.transfers if t.urb == "0xC000")
    assert failed.correlation == "submit_only"
    assert failed.submit_failed is True

    discarded = next(t for t in trace.transfers if t.urb == discarded_urb)
    assert discarded.correlation == "submit_only"
    assert discarded.discard is not None
    assert discarded.as_dict()["discarded"] is True


# --------------------------------------------------------------- 7. filtering


def test_summary_file_descriptors_reflect_only_the_traced_fd() -> None:
    builder = TraceBuilder(target="/dev/bus/usb/001/099")
    open_event(builder, fd=7, path="/dev/bus/usb/001/099")
    bulk_submit(builder, fd=7, urb="0xE000", endpoint=0x02, data=b"\x01")
    bulk_reap(builder, fd=7, urb="0xE000", endpoint=0x02, data=b"\x02")
    close_event(builder, fd=7)

    trace = parse_trace(builder.lines())
    summary = summarize(trace)

    descriptors = summary["file_descriptors"]
    assert isinstance(descriptors, list)
    fds = {d["fd"] for d in descriptors}
    assert fds == {7}
    assert trace.header.target == "/dev/bus/usb/001/099"


# ----------------------------------------------------------- 8. payload bounds


def test_truncated_payload_is_reported_in_summary_and_transfer() -> None:
    builder = TraceBuilder()
    bulk_submit(
        builder,
        fd=5,
        urb="0xF000",
        endpoint=0x02,
        data=b"\x01" * 16,
        payload_available=500,
        payload_truncated=True,
    )
    bulk_reap(builder, fd=5, urb="0xF000", endpoint=0x02, data=b"\x02" * 8)

    trace = parse_trace(builder.lines())
    summary = summarize(trace)
    limits = summary["capture_limits"]
    assert isinstance(limits, dict)
    assert limits["events_with_truncated_payload"] == 1

    transfer_dict = trace.transfers[0].as_dict()
    assert transfer_dict["request_truncated"] is True
    assert transfer_dict["response_truncated"] is False


def test_output_byte_limit_and_dropped_events_surface_in_summary() -> None:
    builder = TraceBuilder()
    builder.add("limit", reason="max_output_bytes", note="capture stopped")
    builder.add("summary", events_written=3, bytes_written=123, events_dropped=7)

    trace = parse_trace(builder.lines())
    summary = summarize(trace)
    limits = summary["capture_limits"]
    assert isinstance(limits, dict)
    assert limits["output_byte_limit_reached"] is True
    assert limits["events_dropped_by_tracer"] == 7


# --------------------------------------------------------- 9. stable JSON out


def test_as_dict_is_json_stable_across_independent_parses() -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    bulk_submit(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x01\x02")
    bulk_reap(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x03\x04")
    lines = builder.lines()

    first = json.dumps(parse_trace(lines).as_dict(), sort_keys=True)
    second = json.dumps(parse_trace(lines).as_dict(), sort_keys=True)
    assert first == second
    # Must not raise: the payload is fully JSON-serializable on its own.
    json.dumps(parse_trace(lines).as_dict())


def test_blank_lines_do_not_change_the_parsed_output() -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    bulk_submit(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x01\x02")
    bulk_reap(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x03\x04")
    close_event(builder, fd=5)
    original = builder.lines()

    padded: list[str] = [original[0]]
    for line in original[1:]:
        padded.extend(["", "   "])
        padded.append(line)
    padded.append("")

    baseline = json.dumps(parse_trace(original).as_dict(), sort_keys=True)
    reformatted = json.dumps(parse_trace(padded).as_dict(), sort_keys=True)
    assert baseline == reformatted


# --------------------------------------------------------------- 10. grouping


def test_group_exchanges_groups_out_then_in_on_same_endpoint() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x10", endpoint=0x02, data=b"\x01")
    bulk_reap(builder, fd=5, urb="0x10", endpoint=0x02, data=b"")
    bulk_submit(builder, fd=5, urb="0x11", endpoint=0x82, data=b"")
    bulk_reap(builder, fd=5, urb="0x11", endpoint=0x82, data=b"\xaa")
    bulk_submit(builder, fd=5, urb="0x12", endpoint=0x82, data=b"")
    bulk_reap(builder, fd=5, urb="0x12", endpoint=0x82, data=b"\xbb")

    trace = parse_trace(builder.lines())

    assert len(trace.exchanges) == 1
    exchange = trace.exchanges[0]
    assert exchange.endpoint_number == 2
    assert exchange.request is not None
    assert exchange.request.urb == "0x10"
    assert [t.urb for t in exchange.responses] == ["0x11", "0x12"]


def test_second_out_on_endpoint_starts_a_new_group() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x20", endpoint=0x02, data=b"\x01")
    bulk_reap(builder, fd=5, urb="0x20", endpoint=0x02, data=b"")
    bulk_submit(builder, fd=5, urb="0x21", endpoint=0x82, data=b"")
    bulk_reap(builder, fd=5, urb="0x21", endpoint=0x82, data=b"\xaa")
    bulk_submit(builder, fd=5, urb="0x22", endpoint=0x02, data=b"\x02")
    bulk_reap(builder, fd=5, urb="0x22", endpoint=0x02, data=b"")

    trace = parse_trace(builder.lines())

    assert len(trace.exchanges) == 2
    first, second = trace.exchanges
    assert first.request is not None and first.request.urb == "0x20"
    assert [t.urb for t in first.responses] == ["0x21"]
    assert second.request is not None and second.request.urb == "0x22"
    assert second.responses == ()


def test_in_with_no_preceding_out_has_no_request() -> None:
    builder = TraceBuilder()
    bulk_submit(builder, fd=5, urb="0x30", endpoint=0x83, data=b"")
    bulk_reap(builder, fd=5, urb="0x30", endpoint=0x83, data=b"\xcc")

    trace = parse_trace(builder.lines())

    assert len(trace.exchanges) == 1
    assert trace.exchanges[0].request is None
    assert [t.urb for t in trace.exchanges[0].responses] == ["0x30"]


def test_control_transfer_is_its_own_single_transfer_group() -> None:
    builder = TraceBuilder()
    control_submit(builder, fd=5, urb="0x40", bm_request_type=0x80, b_request=0x06, w_length=8)
    control_reap(
        builder,
        fd=5,
        urb="0x40",
        bm_request_type=0x80,
        b_request=0x06,
        w_length=8,
        data=b"\x12" * 8,
    )

    trace = parse_trace(builder.lines())

    assert len(trace.exchanges) == 1
    exchange = trace.exchanges[0]
    assert exchange.request is not None
    assert exchange.request.urb == "0x40"
    assert exchange.responses == ()


def test_exchange_carries_no_semantic_name_or_label() -> None:
    field_names = {f.name for f in dataclasses.fields(Exchange)}
    assert field_names == {"endpoint_number", "request", "responses"}


# -------------------------------------------------------------------- 11. CLI


def test_cli_trace_human_report_contains_target(tmp_path: Path, capsys) -> None:
    builder = TraceBuilder(target="/dev/bus/usb/003/017")
    open_event(builder, fd=5, path="/dev/bus/usb/003/017")
    path = _write_trace(tmp_path / "trace.jsonl", builder)

    code = main(["trace", str(path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "/dev/bus/usb/003/017" in out


def test_cli_trace_json_has_expected_top_level_keys(tmp_path: Path, capsys) -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    bulk_submit(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x01")
    bulk_reap(builder, fd=5, urb="0x1", endpoint=0x02, data=b"\x02")
    path = _write_trace(tmp_path / "trace.jsonl", builder)

    code = main(["trace", "--json", str(path)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert set(payload) == {"header", "summary", "transfers", "exchanges", "problems"}


def test_cli_trace_json_out_writes_the_same_payload(tmp_path: Path, capsys) -> None:
    builder = TraceBuilder()
    open_event(builder, fd=5)
    path = _write_trace(tmp_path / "trace.jsonl", builder)
    out_path = tmp_path / "report.json"

    main(["trace", "--json", "--json-out", str(out_path), str(path)])
    stdout_payload = json.loads(capsys.readouterr().out)

    assert json.loads(out_path.read_text()) == stdout_payload


def test_cli_trace_malformed_file_exits_seven(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"op":"ioctl","seq":1}\n', encoding="utf-8")

    code = main(["trace", str(path)])
    captured = capsys.readouterr()

    assert code == EXIT_CODES[DoctorStatus.INTERNAL_ERROR] == 7
    assert captured.err.strip() != ""


def test_cli_trace_nonexistent_path_exits_seven(tmp_path: Path, capsys) -> None:
    code = main(["trace", str(tmp_path / "missing.jsonl")])
    captured = capsys.readouterr()

    assert code == EXIT_CODES[DoctorStatus.INTERNAL_ERROR] == 7
    assert captured.err.strip() != ""


# ============================================================ 12. the tracer


def _usbfs_char_device_nodes() -> list[Path]:
    """Real usbfs character-device nodes on this host, root hubs first.

    Root-hub nodes (device number 1 on each bus) are almost always
    world-readable, so trying them first gives the read-only tests the best
    chance of finding something usable without touching the scanner.
    """
    root = Path("/dev/bus/usb")
    if not root.is_dir():
        return []
    nodes = [p for p in root.glob("*/*") if p.is_char_device()]
    return sorted(nodes, key=lambda p: (p.name != "001", str(p)))


def _sysfs_id_vendor(bus: int, device: int) -> str | None:
    """idVendor of the usbfs node at (bus, device), if sysfs can identify it."""
    devices_root = Path("/sys/bus/usb/devices")
    if not devices_root.is_dir():
        return None
    for entry in devices_root.iterdir():
        try:
            busnum = int((entry / "busnum").read_text().strip())
            devnum = int((entry / "devnum").read_text().strip())
        except OSError, ValueError:
            continue
        if busnum != bus or devnum != device:
            continue
        try:
            return (entry / "idVendor").read_text().strip()
        except OSError:
            return None
    return None


def _pick_unrelated_readable_usbfs_node() -> Path | None:
    """A usbfs node safe to *open*: readable, and never the Revopoint scanner.

    idVendor 2207 is the scanner this repository targets; excluding it means
    this test can never reach the real device even by accident.
    """
    for node in _usbfs_char_device_nodes():
        try:
            bus = int(node.parent.name)
            device = int(node.name)
        except ValueError:
            continue
        if _sysfs_id_vendor(bus, device) == "2207":
            continue
        if os.access(node, os.R_OK):
            return node
    return None


@pytest.fixture(scope="session")
def tracer_so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build reference/tracer/usbfs_trace.c once for the whole test session."""
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available; cannot build the usbfs tracer")
    repo_root = Path(__file__).resolve().parents[1]
    build_script = repo_root / "reference" / "tracer" / "build.sh"
    out_path = tmp_path_factory.mktemp("usbfs-tracer") / "libusbfs_trace.so"

    result = subprocess.run(
        ["bash", str(build_script), "--out", str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"build.sh exited {result.returncode}:\n{result.stderr}")
    # build.sh runs gcc under `set -x`, so its stderr also contains the
    # echoed command line; only lines with the compiler's own "warning:"
    # marker count as a warning.
    warnings = [line for line in result.stderr.splitlines() if ": warning:" in line]
    assert not warnings, "compiler warnings:\n" + "\n".join(warnings)
    assert out_path.exists()
    return out_path


def _tracer_env(
    tracer_so: Path,
    *,
    target: Path | None = None,
    out: Path | None = None,
    env_extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment a traced child runs under; shared by every child runner below."""
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(tracer_so)
    env.pop("SCANNER3D_TRACE_TARGET", None)
    env.pop("SCANNER3D_TRACE_OUT", None)
    if target is not None:
        env["SCANNER3D_TRACE_TARGET"] = str(target)
    if out is not None:
        env["SCANNER3D_TRACE_OUT"] = str(out)
    if env_extra:
        env.update(env_extra)
    return env


def _run_child(
    tracer_so: Path,
    script: str,
    *,
    target: Path | None = None,
    out: Path | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        env=_tracer_env(tracer_so, target=target, out=out, env_extra=env_extra),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _run_compiled_child(
    tracer_so: Path,
    argv: list[str],
    *,
    target: Path | None = None,
    out: Path | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Like :func:`_run_child`, but for a precompiled binary rather than a Python script."""
    return subprocess.run(
        argv,
        env=_tracer_env(tracer_so, target=target, out=out, env_extra=env_extra),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _compile_c_child(tmp_path: Path, name: str, source: str) -> Path:
    """Compile a tiny standalone C program used as one test's traced child.

    Separate from the ``tracer_so`` fixture, which builds the library under
    test once per session; these are throwaway probes, one per test, of what
    a single hooked call observes. Every caller also depends on ``tracer_so``,
    so gcc's absence is already handled by that fixture's own skip; a nonzero
    exit here means the source itself is broken, which is a real failure.
    """
    src_path = tmp_path / f"{name}.c"
    bin_path = tmp_path / name
    src_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        ["gcc", "-std=gnu11", "-O0", "-g", "-Wall", "-Wextra", str(src_path), "-o", str(bin_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"failed to compile {name}.c:\n{result.stderr}")
    return bin_path


def _readable_unrelated_usbfs_nodes(limit: int) -> list[Path]:
    """Up to ``limit`` distinct usbfs nodes that are readable and never the scanner."""
    found: list[Path] = []
    for node in _usbfs_char_device_nodes():
        try:
            bus = int(node.parent.name)
            device = int(node.name)
        except ValueError:
            continue
        if _sysfs_id_vendor(bus, device) == "2207":
            continue
        if os.access(node, os.R_OK):
            found.append(node)
        if len(found) >= limit:
            break
    return found


def _created_trace_files(out_prefix: Path) -> list[Path]:
    """Files the tracer actually created for this ``SCANNER3D_TRACE_OUT`` prefix.

    The tracer creates its output lazily, on the first event it actually
    records, and names it ``<out_prefix>.<pid>`` rather than the literal
    prefix - one file per process, so a fork/exec never interleaves two
    processes' records. So "no output" is verified by absence of any file
    matching this pattern, not by checking one exact path.
    """
    return sorted(out_prefix.parent.glob(out_prefix.name + ".*"))


# --- fail-closed: no output file is ever created unless everything lines up


def test_tracer_writes_nothing_when_neither_env_var_is_set(tracer_so: Path, tmp_path: Path) -> None:
    out_prefix = tmp_path / "trace.jsonl"
    result = _run_child(tracer_so, "pass")
    assert result.returncode == 0
    assert _created_trace_files(out_prefix) == []


def test_tracer_disabled_for_nonexistent_target(tracer_so: Path, tmp_path: Path) -> None:
    out_prefix = tmp_path / "trace.jsonl"
    missing = tmp_path / "does-not-exist"
    script = "print('hello')"

    plain = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    traced = _run_child(tracer_so, script, target=missing, out=out_prefix)

    assert traced.returncode == 0
    assert _created_trace_files(out_prefix) == []
    assert traced.stdout == plain.stdout


def test_tracer_disabled_for_regular_file_target(tracer_so: Path, tmp_path: Path) -> None:
    target = tmp_path / "not-a-device"
    target.write_bytes(b"")
    out_prefix = tmp_path / "trace.jsonl"

    result = _run_child(tracer_so, "pass", target=target, out=out_prefix)

    assert result.returncode == 0
    assert _created_trace_files(out_prefix) == []


def test_tracer_disabled_for_fake_usb_path_prefix(tracer_so: Path, tmp_path: Path) -> None:
    # The tracer requires the *literal* prefix "/dev/bus/usb/"; a path that
    # merely contains those components under a tmp_path is not enough.
    fake_root = tmp_path / "dev" / "bus" / "usb" / "001"
    fake_root.mkdir(parents=True)
    target = fake_root / "041"
    target.write_bytes(b"")
    out_prefix = tmp_path / "trace.jsonl"

    result = _run_child(tracer_so, "pass", target=target, out=out_prefix)

    assert result.returncode == 0
    assert _created_trace_files(out_prefix) == []


def test_tracer_refuses_to_overwrite_existing_output(tracer_so: Path, tmp_path: Path) -> None:
    # The output file is created lazily, on the first event actually recorded
    # (see _created_trace_files), and its name carries this process's pid. So
    # the collision is staged deliberately: launch the child but hold it at a
    # stdin gate before it touches the device, learn its pid from Popen, drop
    # a pre-existing file at exactly the path the tracer will then try to
    # create exclusively, and only then let the child proceed.
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    out_prefix = tmp_path / "trace.jsonl"
    sentinel = b"pre-existing capture, do not touch\n"

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(tracer_so)
    env["SCANNER3D_TRACE_TARGET"] = str(node)
    env["SCANNER3D_TRACE_OUT"] = str(out_prefix)
    script = (
        "import sys, os\n"
        "sys.stdin.readline()\n"
        f"fd = os.open({str(node)!r}, os.O_RDONLY)\n"
        "os.close(fd)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sentinel_path = Path(f"{out_prefix}.{proc.pid}")
    sentinel_path.write_bytes(sentinel)
    proc.communicate(input="go\n", timeout=30)

    assert proc.returncode == 0
    assert sentinel_path.read_bytes() == sentinel


# --- exclusion and capture correctness


def test_unrelated_fds_excluded_target_fd_captured(tracer_so: Path, tmp_path: Path) -> None:
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    other_file = tmp_path / "unrelated.bin"
    other_file.write_bytes(b"\x00")
    out_prefix = tmp_path / "trace.jsonl"

    child_src = f"""
import fcntl, os
usb_fd = os.open({str(node)!r}, os.O_RDONLY)
other_fd = os.open({str(other_file)!r}, os.O_RDONLY)
print(f"FDS {{usb_fd}} {{other_fd}}")
# TCGETS (0x5401) is a tty ioctl, never a USBDEVFS_* request, so this issues
# no USB command to the device at all - it is a deliberately unrelated,
# harmless ioctl used only to exercise the interposer's filtering.
for fd in (usb_fd, other_fd):
    try:
        fcntl.ioctl(fd, 0x5401, b"\\x00" * 64)
    except OSError:
        pass
os.close(usb_fd)
os.close(other_fd)
"""
    result = _run_child(tracer_so, child_src, target=node, out=out_prefix)
    assert result.returncode == 0

    fd_line = next(line for line in result.stdout.splitlines() if line.startswith("FDS "))
    _, usb_fd_s, other_fd_s = fd_line.split()
    usb_fd, other_fd = int(usb_fd_s), int(other_fd_s)

    created = _created_trace_files(out_prefix)
    assert len(created) == 1
    trace = load_trace(created[0])
    assert trace.header.target == str(node.resolve())

    opens = [e for e in trace.events if e.op == "open"]
    closes = [e for e in trace.events if e.op == "close"]
    assert len(opens) == 1
    assert len(closes) == 1
    assert opens[0].fd == usb_fd

    ioctls = [e for e in trace.events if e.op == "ioctl"]
    assert ioctls  # the TCGETS attempt on the tracked fd was recorded
    for event in ioctls:
        assert event.fd == usb_fd
        assert event.fd != other_fd
    # usbfs rejects an unrecognized ioctl; the exact errno depends on the
    # node's open mode (observed ENOTTY for some nodes, EPERM for others when
    # opened O_RDONLY) - what matters here is that it failed and was recorded,
    # not which errno usbfs happened to pick.
    for event in ioctls:
        assert event.ret == -1
        assert event.errno_name


def test_tracer_does_not_alter_traced_program_behavior(tracer_so: Path) -> None:
    script = "import sys; print('probe-output'); sys.exit(17)"
    without = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    with_preload = _run_child(tracer_so, script)  # no target/out: pure passthrough

    assert with_preload.returncode == without.returncode == 17
    assert with_preload.stdout == without.stdout


# --- capture limits: clamping, and what the header says about it


def _script_open_ioctl_close(node: Path, *, repeat: int = 1) -> str:
    """Child script: open the target node, issue the harmless TCGETS, close it.

    TCGETS (0x5401) is a tty ioctl, never a USBDEVFS_* request, so this issues
    no USB command to the device - see ``test_unrelated_fds_excluded_target_fd_captured``
    above, which established the same trick. ``repeat`` just varies how many
    ioctl records the tracer produces for the same already-tracked fd.
    """
    return f"""
import fcntl, os
fd = os.open({str(node)!r}, os.O_RDONLY)
for _ in range({repeat}):
    try:
        fcntl.ioctl(fd, 0x5401, b"\\x00" * 64)
    except OSError:
        pass
os.close(fd)
"""


def test_payload_cap_clamped_above_ceiling_is_reported_in_header(
    tracer_so: Path, tmp_path: Path
) -> None:
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    out_prefix = tmp_path / "trace.jsonl"

    result = _run_child(
        tracer_so,
        _script_open_ioctl_close(node),
        target=node,
        out=out_prefix,
        env_extra={"SCANNER3D_TRACE_MAX_PAYLOAD": "999999"},
    )
    assert result.returncode == 0

    created = _created_trace_files(out_prefix)
    assert len(created) == 1
    trace = load_trace(created[0])

    assert trace.header.max_payload_bytes == 4096
    assert trace.header.requested_max_payload_bytes == 999999
    assert trace.header.limits_clamped is True


def test_output_cap_clamped_below_floor_is_reported_in_header(
    tracer_so: Path, tmp_path: Path
) -> None:
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    out_prefix = tmp_path / "trace.jsonl"

    result = _run_child(
        tracer_so,
        _script_open_ioctl_close(node),
        target=node,
        out=out_prefix,
        env_extra={"SCANNER3D_TRACE_MAX_BYTES": "1"},
    )
    assert result.returncode == 0

    created = _created_trace_files(out_prefix)
    assert len(created) == 1
    trace = load_trace(created[0])

    assert trace.header.max_output_bytes == 8192
    assert trace.header.requested_max_output_bytes == 1
    assert trace.header.limits_clamped is True


def test_capture_limits_report_no_clamp_with_default_env(tracer_so: Path, tmp_path: Path) -> None:
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    out_prefix = tmp_path / "trace.jsonl"

    # Neither SCANNER3D_TRACE_MAX_PAYLOAD nor SCANNER3D_TRACE_MAX_BYTES is set.
    result = _run_child(tracer_so, _script_open_ioctl_close(node), target=node, out=out_prefix)
    assert result.returncode == 0

    created = _created_trace_files(out_prefix)
    assert len(created) == 1
    trace = load_trace(created[0])

    assert trace.header.limits_clamped is False
    assert trace.header.requested_max_payload_bytes == trace.header.max_payload_bytes
    assert trace.header.requested_max_output_bytes == trace.header.max_output_bytes


def test_output_byte_cap_is_enforced_and_capture_still_parses(
    tracer_so: Path, tmp_path: Path
) -> None:
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    out_prefix = tmp_path / "trace.jsonl"
    cap = 8192  # OUTPUT_MIN: the smallest cap the tracer allows

    result = _run_child(
        tracer_so,
        _script_open_ioctl_close(node, repeat=500),
        target=node,
        out=out_prefix,
        env_extra={"SCANNER3D_TRACE_MAX_BYTES": str(cap)},
    )
    assert result.returncode == 0

    created = _created_trace_files(out_prefix)
    assert len(created) == 1
    trace_path = created[0]

    raw = trace_path.read_bytes()
    # write_all()'s full-write guarantee: the file never ends mid-record.
    assert raw.endswith(b"\n")
    assert len(raw) <= cap

    trace = load_trace(trace_path)
    assert trace.problems == ()
    limit_events = [e for e in trace.events if e.op == "limit"]
    assert limit_events
    assert limit_events[0].raw.get("reason") == "max_output_bytes"
    # The reserved room at the tail of the cap holds both the `limit` record
    # and the closing `summary` record, so a capture that hits its cap still
    # ends with one.
    assert [e for e in trace.events if e.op == "summary"]

    summary = summarize(trace)
    limits = summary["capture_limits"]
    assert isinstance(limits, dict)
    assert limits["output_byte_limit_reached"] is True
    dropped = limits["events_dropped_by_tracer"]
    assert isinstance(dropped, int) and dropped > 0


# --- descriptor identification: the adoption gate for a raw-syscall-opened fd


_RAW_OPEN_IOCTL_CLOSE_C = r"""
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <termios.h>
#include <unistd.h>

/*
 * An ioctl request number whose _IOC_TYPE is 'U' - so it reaches the tracer's
 * adoption gate - but whose command number is not, and has never been, a
 * defined USBDEVFS_* command (those run from 0 to roughly 34). usbfs answers
 * it with an error from its ioctl handler's default arm, having performed no
 * USB transaction whatsoever: no transfer is submitted, no endpoint is
 * touched, no device state changes. It is the same undefined-'U' probe this
 * task used to verify the adoption fallback offline before spending a
 * hardware run.
 */
#define USBDEVFS_UNDEFINED _IO('U', 250)

/*
 * Opens argv[1] through a raw SYS_openat syscall, bypassing every libc open
 * spelling reference/tracer/usbfs_trace.c hooks (open/open64/openat/openat64
 * and their __*_2 fortified spellings all go through libc; this does not), so
 * the only way the tracer can identify the descriptor is adopt_if_target().
 *
 * Then it issues two ioctls, in this order, and prints what each returned:
 * TCGETS, whose _IOC_TYPE is 'T', which must pass straight through untouched
 * because the tracer only stats a descriptor for a 'U'-type request; and the
 * undefined 'U' request above, which must reach the adoption gate. Neither
 * sends anything to any device.
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path>\n", argv[0]);
        return 2;
    }
    long fd = syscall(SYS_openat, AT_FDCWD, argv[1], O_RDONLY);
    if (fd < 0) {
        perror("openat");
        return 1;
    }
    printf("FD %ld\n", fd);
    struct termios term;
    printf("TCGETS %d\n", ioctl((int)fd, TCGETS, &term));
    printf("UNDEFINED_U %d\n", ioctl((int)fd, USBDEVFS_UNDEFINED, 0));
    fflush(stdout);
    close((int)fd);
    return 0;
}
"""

#: ``_IO('U', 250)`` as the tracer renders it in a record's ``request`` field.
UNDEFINED_U_REQUEST = "0x55fa"


def _child_stdout_field(result: subprocess.CompletedProcess[str], key: str) -> int:
    line = next(line for line in result.stdout.splitlines() if line.startswith(key + " "))
    return int(line.split()[1])


def test_raw_opened_target_is_adopted_by_path_and_rdev(tracer_so: Path, tmp_path: Path) -> None:
    """A descriptor no hooked ``open`` explains is still identified, once.

    This is the positive side of the reviewed adoption change
    (``adopt_if_target()`` in reference/tracer/usbfs_trace.c). The child opens
    the target with a raw ``SYS_openat``, so no open hook sees it; the tracer
    can only recognise it on its first ``'U'``-type ioctl, by matching *both*
    the descriptor's ``/proc/self/fd`` link against the canonical target path
    and its ``st_rdev`` against the target's.

    The same run also proves the gate in front of that check: the child's
    TCGETS - type ``'T'`` - comes first and must leave no trace at all, so the
    only ioctl recorded is the ``'U'`` one that followed it.

    Neither ioctl performs a USB transaction; see ``_RAW_OPEN_IOCTL_CLOSE_C``.
    """
    node = _pick_unrelated_readable_usbfs_node()
    if node is None:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    child = _compile_c_child(tmp_path, "adopt_probe", _RAW_OPEN_IOCTL_CLOSE_C)
    out_prefix = tmp_path / "trace.jsonl"

    result = _run_compiled_child(tracer_so, [str(child), str(node)], target=node, out=out_prefix)
    assert result.returncode == 0, result.stderr
    child_fd = _child_stdout_field(result, "FD")
    assert _child_stdout_field(result, "TCGETS") == -1
    assert _child_stdout_field(result, "UNDEFINED_U") == -1

    created = _created_trace_files(out_prefix)
    assert len(created) == 1
    trace = load_trace(created[0])
    assert trace.problems == ()

    adopts = [event for event in trace.events if event.op == "adopt"]
    assert len(adopts) == 1
    assert adopts[0].fd == child_fd
    assert adopts[0].raw.get("path") == str(node.resolve())
    # The descriptor was never seen being opened, so there is no `open` record.
    assert not [event for event in trace.events if event.op == "open"]

    ioctls = [event for event in trace.events if event.op == "ioctl"]
    assert len(ioctls) == 1
    assert ioctls[0].raw.get("request") == UNDEFINED_U_REQUEST
    assert ioctls[0].fd == child_fd
    assert ioctls[0].ret == -1
    assert ioctls[0].errno_name


def test_raw_opened_non_target_is_never_adopted(tracer_so: Path, tmp_path: Path) -> None:
    """The same probe against anything but the configured target records nothing.

    Run twice: once against a *different* readable non-scanner usbfs node
    (present on most hosts), and once against an ordinary file. In both cases
    the descriptor reaches ``adopt_if_target()`` - the request is ``'U'``-type
    - and is refused, on ``st_rdev`` for the other node and on not being a
    character device for the file. Refused means no output file exists at all,
    which is the fail-closed default this tracer is built to hold.

    What no test on this host can construct is the third refusal: a descriptor
    whose ``st_rdev`` matches the target's but whose path does not. That needs
    a second device node with the same device number, which needs ``mknod``
    and therefore root, which this task forbids. That half of the check is
    covered by inspection only, and this docstring is the record of that gap.
    """
    nodes = _readable_unrelated_usbfs_nodes(limit=2)
    if not nodes:
        pytest.skip("no readable, non-scanner /dev/bus/usb node available")
    target = nodes[0]
    child = _compile_c_child(tmp_path, "adopt_negative", _RAW_OPEN_IOCTL_CLOSE_C)

    others: list[Path] = [tmp_path / "unrelated.bin"]
    others[0].write_bytes(b"\x00")
    if len(nodes) >= 2:
        others.insert(0, nodes[1])

    for index, other in enumerate(others):
        out_prefix = tmp_path / f"trace-{index}.jsonl"
        result = _run_compiled_child(
            tracer_so, [str(child), str(other)], target=target, out=out_prefix
        )
        assert result.returncode == 0, result.stderr
        assert _created_trace_files(out_prefix) == [], f"recorded something for {other}"


# --- errno transparency: a successful hooked call leaves errno as it would be
# --- untraced, even though the tracer's own bookkeeping sets errno freely
# --- in between (fstat, readlink, clock_gettime, write, ...)


_ERRNO_TRANSPARENCY_C = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

static void report(const char *label) {
    printf("%s errno=%d\n", label, errno);
}

/*
 * Exercises open/dup/close - three of the hooked call families - each
 * preceded by a distinctive errno sentinel, and prints what errno reads
 * immediately after each *successful* call returns. Every hook in
 * reference/tracer/usbfs_trace.c restores the real call's own errno before
 * returning, undoing whatever its own bookkeeping set in between, so this
 * program's output must be identical whether or not the tracer is preloaded.
 * argv[2], when given, is a usbfs node to also open and close, which exercises
 * the bookkeeping-heavy path (consider_fd, ensure_output_locked, write_header)
 * as well as the trivial one.
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <plain-file> [usbfs-node]\n", argv[0]);
        return 2;
    }

    errno = 88;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open(plain) failed: %d\n", errno); return 3; }
    report("open_plain");

    errno = 88;
    int fd2 = dup(fd);
    if (fd2 < 0) { fprintf(stderr, "dup failed: %d\n", errno); return 4; }
    report("dup");

    errno = 88;
    if (close(fd2) != 0) { fprintf(stderr, "close(dup) failed: %d\n", errno); return 5; }
    report("close_dup");

    errno = 88;
    if (close(fd) != 0) { fprintf(stderr, "close(plain) failed: %d\n", errno); return 6; }
    report("close_plain");

    if (argc > 2) {
        errno = 88;
        int nfd = open(argv[2], O_RDONLY);
        if (nfd < 0) { fprintf(stderr, "open(node) failed: %d\n", errno); return 7; }
        report("open_node");

        errno = 88;
        if (close(nfd) != 0) { fprintf(stderr, "close(node) failed: %d\n", errno); return 8; }
        report("close_node");
    }
    return 0;
}
"""


def test_hooked_calls_restore_the_callers_errno_on_success(tracer_so: Path, tmp_path: Path) -> None:
    plain_file = tmp_path / "plain.bin"
    plain_file.write_bytes(b"x")
    node = _pick_unrelated_readable_usbfs_node()

    child = _compile_c_child(tmp_path, "errno_transparency", _ERRNO_TRANSPARENCY_C)
    argv = [str(child), str(plain_file)]
    if node is not None:
        argv.append(str(node))

    out_prefix = tmp_path / "trace.jsonl"
    if node is not None:
        # Capture fully armed: the traced open/close of the node also exercise
        # the tracer's own bookkeeping (fstat, readlink, write), not just the
        # cheap pass-through path.
        traced_env = _tracer_env(tracer_so, target=node, out=out_prefix)
    else:
        traced_env = _tracer_env(tracer_so)
    plain_env = os.environ.copy()
    plain_env.pop("LD_PRELOAD", None)
    plain_env.pop("SCANNER3D_TRACE_TARGET", None)
    plain_env.pop("SCANNER3D_TRACE_OUT", None)

    traced = subprocess.run(
        argv, env=traced_env, capture_output=True, text=True, check=False, timeout=30
    )
    plain = subprocess.run(
        argv, env=plain_env, capture_output=True, text=True, check=False, timeout=30
    )

    assert traced.returncode == 0, traced.stderr
    assert plain.returncode == 0, plain.stderr
    assert traced.returncode == plain.returncode
    assert traced.stdout == plain.stdout
