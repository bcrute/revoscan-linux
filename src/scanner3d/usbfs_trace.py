"""Offline decoder for the repository's usbfs transaction traces.

A trace is the line-delimited JSON stream written by
``reference/tracer/usbfs_trace.c``: one ``header`` object, then one object per
recorded ``ioctl``/``open``/``close`` call against a single usbfs node. Nothing
in this module needs the scanner, the vendor SDK, or root; it reads a file and
reports what is in it.

Two properties matter more than convenience here, because the output is
evidence:

* **Correlation is deterministic.** URBs are submitted and reaped
  asynchronously, possibly on different threads, and the tracer assigns ``seq``
  when a call *returns* - so a completion can legitimately be written before
  its own submission. Matching is therefore a fixed rule over
  ``(pid, fd, urb)`` and the recorded call-entry and call-return times, not a
  guess from the order records happen to appear in the file, and every
  transfer says how its two halves were joined.
* **Nothing incomplete is discarded.** A submit with no reap, a reap with no
  submit, a failed ioctl, a truncated payload, and an unparseable line are all
  retained and reported. A trace that lost data must look like a trace that
  lost data, not like a shorter clean one.

Deliberately absent: any naming of what the observed bytes *mean*. Transfers
are grouped by the structural fact that an IN transfer followed an OUT transfer
on the same endpoint number; they are not given protocol names, because nothing
measured so far supports one.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from scanner3d.errors import TraceFormatError

#: Value of the header's ``format`` field. A file without it is not a trace.
TRACE_FORMAT: Final = "scanner3d-usbfs-trace"

#: Stream versions this decoder understands.
SUPPORTED_VERSIONS: Final[frozenset[int]] = frozenset({1})

#: Bytes of payload rendered inline in the human report (JSON keeps them all).
HUMAN_PAYLOAD_BYTES: Final = 32

#: Ceiling on how many lines a single trace may contain, so a corrupt or
#: adversarial file cannot make the decoder allocate without bound.
MAX_TRACE_LINES: Final = 5_000_000

#: Ceiling on how much of a bad line is quoted back in a problem report.
PROBLEM_EXCERPT_CHARS: Final = 120


@dataclass(frozen=True, slots=True)
class TraceProblem:
    """One line the decoder could not use, kept as evidence of data loss."""

    line_number: int
    reason: str
    excerpt: str

    def as_dict(self) -> dict[str, object]:
        return {"line_number": self.line_number, "reason": self.reason, "excerpt": self.excerpt}


@dataclass(frozen=True, slots=True)
class TraceHeader:
    """The first record of a trace: what was captured and what was not."""

    format: str
    version: int
    target: str
    pid: int | None
    max_payload_bytes: int | None
    max_output_bytes: int | None
    #: What the capture was *asked* for, before the tracer clamped it into its
    #: own hard range. Absent in version-1 traces written before the tracer
    #: reported it, hence optional; when present and different from the field
    #: above, the request was clamped.
    requested_max_payload_bytes: int | None
    requested_max_output_bytes: int | None
    clock: str | None
    started_realtime_ns: int | None
    target_rdev: tuple[int, int] | None
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "version": self.version,
            "target": self.target,
            "pid": self.pid,
            "max_payload_bytes": self.max_payload_bytes,
            "max_output_bytes": self.max_output_bytes,
            "requested_max_payload_bytes": self.requested_max_payload_bytes,
            "requested_max_output_bytes": self.requested_max_output_bytes,
            "limits_clamped": self.limits_clamped,
            "clock": self.clock,
            "started_realtime_ns": self.started_realtime_ns,
            "target_rdev": list(self.target_rdev) if self.target_rdev else None,
            "limitations": list(self.limitations),
        }

    @property
    def limits_clamped(self) -> bool | None:
        """Whether the tracer had to clamp a requested limit.

        ``None`` when the trace does not say what was requested, which is not
        the same as "nothing was clamped" and is reported as its own state.
        """
        pairs = (
            (self.requested_max_payload_bytes, self.max_payload_bytes),
            (self.requested_max_output_bytes, self.max_output_bytes),
        )
        if all(requested is None for requested, _ in pairs):
            return None
        return any(
            requested is not None and requested != effective for requested, effective in pairs
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One recorded call, with the raw record kept for anything not modelled."""

    line_number: int
    op: str
    raw: Mapping[str, Any]

    # --- common fields -------------------------------------------------------

    @property
    def seq(self) -> int | None:
        return _as_int(self.raw.get("seq"))

    @property
    def t_ns(self) -> int | None:
        return _as_int(self.raw.get("t_ns"))

    @property
    def pid(self) -> int | None:
        return _as_int(self.raw.get("pid"))

    @property
    def tid(self) -> int | None:
        return _as_int(self.raw.get("tid"))

    @property
    def fd(self) -> int | None:
        return _as_int(self.raw.get("fd"))

    # --- ioctl fields --------------------------------------------------------

    @property
    def name(self) -> str:
        """Symbolic ioctl name, or the raw request code when unrecognised."""
        value = self.raw.get("name")
        if isinstance(value, str):
            return value
        request = self.raw.get("request")
        return f"unknown({request})" if request is not None else "unknown"

    @property
    def ret(self) -> int | None:
        return _as_int(self.raw.get("ret"))

    @property
    def failed(self) -> bool:
        ret = self.ret
        return ret is not None and ret < 0

    @property
    def errno_name(self) -> str | None:
        value = self.raw.get("errno_name")
        return value if isinstance(value, str) else None

    @property
    def phase(self) -> str | None:
        value = self.raw.get("phase")
        return value if isinstance(value, str) else None

    @property
    def urb(self) -> str | None:
        value = self.raw.get("urb")
        return value if isinstance(value, str) else None

    @property
    def endpoint(self) -> int | None:
        return _as_int(self.raw.get("endpoint"))

    @property
    def direction(self) -> str | None:
        value = self.raw.get("direction")
        return value if isinstance(value, str) else None

    @property
    def transfer_type(self) -> str | None:
        value = self.raw.get("urb_type_name")
        return value if isinstance(value, str) else None

    @property
    def payload_available(self) -> int | None:
        return _as_int(self.raw.get("payload_available"))

    @property
    def payload_truncated(self) -> bool:
        return self.raw.get("payload_truncated") is True

    @property
    def data(self) -> bytes | None:
        """Captured payload bytes, or ``None`` when the record carried none."""
        value = self.raw.get("data")
        if not isinstance(value, str):
            return None
        try:
            return bytes.fromhex(value)
        except ValueError:
            return None

    @property
    def setup(self) -> dict[str, int] | None:
        """The 8-byte control setup packet, when this event decoded one."""
        keys = ("bmRequestType", "bRequest", "wValue", "wIndex", "wLength")
        values = {k: _as_int(self.raw.get(k)) for k in keys}
        if any(v is None for v in values.values()):
            return None
        return {k: v for k, v in values.items() if v is not None}

    def as_dict(self) -> dict[str, object]:
        return {"line_number": self.line_number, **dict(sorted(self.raw.items()))}


@dataclass(frozen=True, slots=True)
class Transfer:
    """One URB, as far as the capture managed to observe it.

    ``correlation`` records how the two halves were joined:

    ``matched``
        Paired, with the submit's record written before the reap's - the
        ordinary case.
    ``deferred``
        Paired, but the reap's record was written *before* its submit's (a
        completion racing the submitting thread's own record). The pairing
        itself is no less certain; only the emission order was inverted.
    ``submit_only``
        Submitted, never reaped in this capture: still in flight at exit,
        discarded, or lost to a capture limit.
    ``reap_only``
        Reaped with no submit in this capture - the URB was submitted before
        tracing started, or its submit record was lost.
    """

    correlation: str
    submit: TraceEvent | None
    reap: TraceEvent | None
    discard: TraceEvent | None = None

    @property
    def key_event(self) -> TraceEvent:
        """Whichever half exists; a transfer is never built without one."""
        event = self.submit if self.submit is not None else self.reap
        assert event is not None
        return event

    @property
    def urb(self) -> str | None:
        return self.key_event.urb

    @property
    def transfer_type(self) -> str | None:
        for event in (self.submit, self.reap):
            if event is not None and event.transfer_type:
                return event.transfer_type
        return None

    @property
    def endpoint(self) -> int | None:
        for event in (self.submit, self.reap):
            if event is not None and event.endpoint is not None:
                return event.endpoint
        return None

    @property
    def direction(self) -> str | None:
        """Data-stage direction: from the control setup packet when present."""
        for event in (self.submit, self.reap):
            if event is None:
                continue
            value = event.raw.get("setup_direction")
            if isinstance(value, str):
                return value
        for event in (self.submit, self.reap):
            if event is not None and event.direction:
                return event.direction
        return None

    @property
    def setup(self) -> dict[str, int] | None:
        for event in (self.submit, self.reap):
            if event is not None:
                found = event.setup
                if found is not None:
                    return found
        return None

    @property
    def requested_bytes(self) -> int | None:
        for event in (self.submit, self.reap):
            if event is not None:
                value = _as_int(event.raw.get("buffer_length"))
                if value is not None:
                    return value
        return None

    @property
    def actual_bytes(self) -> int | None:
        return _as_int(self.reap.raw.get("actual_length")) if self.reap is not None else None

    @property
    def status(self) -> int | None:
        return _as_int(self.reap.raw.get("status")) if self.reap is not None else None

    @property
    def submit_failed(self) -> bool:
        return self.submit is not None and self.submit.failed

    @property
    def request_data(self) -> bytes | None:
        """Bytes the host sent, as captured at submit time."""
        return self.submit.data if self.submit is not None else None

    @property
    def response_data(self) -> bytes | None:
        """Bytes the device returned, as captured at reap time."""
        return self.reap.data if self.reap is not None else None

    @property
    def order(self) -> tuple[int, int]:
        """Deterministic sort key over emission order, total across events."""
        seqs = sorted(
            e.seq for e in (self.submit, self.reap) if e is not None and e.seq is not None
        )
        if not seqs:
            return (self.key_event.line_number, self.key_event.line_number)
        return (seqs[0], seqs[-1])

    def as_dict(self) -> dict[str, object]:
        return {
            "correlation": self.correlation,
            "urb": self.urb,
            "transfer_type": self.transfer_type,
            "endpoint": self.endpoint,
            "endpoint_hex": None if self.endpoint is None else f"0x{self.endpoint:02x}",
            "direction": self.direction,
            "setup": self.setup,
            "requested_bytes": self.requested_bytes,
            "actual_bytes": self.actual_bytes,
            "status": self.status,
            "submit_failed": self.submit_failed,
            "discarded": self.discard is not None,
            "submit_seq": None if self.submit is None else self.submit.seq,
            "reap_seq": None if self.reap is None else self.reap.seq,
            "submit_tid": None if self.submit is None else self.submit.tid,
            "reap_tid": None if self.reap is None else self.reap.tid,
            "request_hex": _hex_or_none(self.request_data),
            "response_hex": _hex_or_none(self.response_data),
            "request_truncated": self.submit is not None and self.submit.payload_truncated,
            "response_truncated": self.reap is not None and self.reap.payload_truncated,
        }


@dataclass(frozen=True, slots=True)
class Exchange:
    """An OUT transfer and the IN transfers that followed it on that endpoint.

    This is a *structural* grouping and nothing more: it asserts only that the
    capture shows these completions in this order on this endpoint number. No
    claim is made that they form one protocol request and its reply, and no
    name is attached to either side.
    """

    endpoint_number: int
    request: Transfer | None
    responses: tuple[Transfer, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint_number": self.endpoint_number,
            "request": None if self.request is None else self.request.as_dict(),
            "responses": [t.as_dict() for t in self.responses],
            "request_bytes": None if self.request is None else self.request.actual_bytes,
            "response_bytes": sum(t.actual_bytes or 0 for t in self.responses),
        }


@dataclass(frozen=True, slots=True)
class Trace:
    """A parsed capture: header, every event, correlated transfers, problems."""

    header: TraceHeader
    events: tuple[TraceEvent, ...]
    transfers: tuple[Transfer, ...]
    problems: tuple[TraceProblem, ...]
    exchanges: tuple[Exchange, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "header": self.header.as_dict(),
            "summary": summarize(self),
            "transfers": [t.as_dict() for t in self.transfers],
            "exchanges": [e.as_dict() for e in self.exchanges],
            "problems": [p.as_dict() for p in self.problems],
        }


# --------------------------------------------------------------------- parse


def _as_int(value: object) -> int | None:
    """Ints only; ``True`` is not 1 here and floats are not silently truncated."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _hex_or_none(data: bytes | None) -> str | None:
    return None if data is None else data.hex()


def _excerpt(line: str) -> str:
    stripped = line.strip()
    if len(stripped) <= PROBLEM_EXCERPT_CHARS:
        return stripped
    return stripped[:PROBLEM_EXCERPT_CHARS] + "..."


def _parse_header(record: Mapping[str, Any]) -> TraceHeader:
    fmt = record.get("format")
    if fmt != TRACE_FORMAT:
        raise TraceFormatError(f"not a {TRACE_FORMAT} file (format={fmt!r})")
    version = _as_int(record.get("version"))
    if version is None or version not in SUPPORTED_VERSIONS:
        raise TraceFormatError(
            f"unsupported trace version {record.get('version')!r}; "
            f"this decoder understands {sorted(SUPPORTED_VERSIONS)}"
        )
    target = record.get("target")
    if not isinstance(target, str) or not target:
        raise TraceFormatError("header does not name the traced device")

    major = _as_int(record.get("target_rdev_major"))
    minor = _as_int(record.get("target_rdev_minor"))
    rdev = (major, minor) if major is not None and minor is not None else None

    raw_limits = record.get("limitations")
    limitations = tuple(str(item) for item in raw_limits) if isinstance(raw_limits, list) else ()
    clock = record.get("clock")
    return TraceHeader(
        format=fmt,
        version=version,
        target=target,
        pid=_as_int(record.get("pid")),
        max_payload_bytes=_as_int(record.get("max_payload_bytes")),
        max_output_bytes=_as_int(record.get("max_output_bytes")),
        requested_max_payload_bytes=_as_int(record.get("requested_max_payload_bytes")),
        requested_max_output_bytes=_as_int(record.get("requested_max_output_bytes")),
        clock=clock if isinstance(clock, str) else None,
        started_realtime_ns=_as_int(record.get("started_realtime_ns")),
        target_rdev=rdev,
        limitations=limitations,
    )


def parse_trace(lines: Iterable[str]) -> Trace:
    """Decode a trace from an iterable of raw lines.

    Raises :class:`TraceFormatError` only for a file that is not a trace at all
    (no header, wrong format, unsupported version). Every other defect - a
    truncated last line, a corrupt record, an unknown ``op`` - becomes a
    :class:`TraceProblem` so the rest of the capture stays usable.
    """
    header: TraceHeader | None = None
    events: list[TraceEvent] = []
    problems: list[TraceProblem] = []

    for line_number, line in enumerate(lines, start=1):
        if line_number > MAX_TRACE_LINES:
            problems.append(
                TraceProblem(line_number, f"trace exceeds {MAX_TRACE_LINES} lines; stopped", "")
            )
            break
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if header is None:
                raise TraceFormatError(f"line 1 is not JSON: {exc}") from exc
            problems.append(TraceProblem(line_number, f"not valid JSON: {exc.msg}", _excerpt(line)))
            continue
        if not isinstance(record, dict):
            if header is None:
                raise TraceFormatError("line 1 is not a JSON object")
            problems.append(TraceProblem(line_number, "record is not an object", _excerpt(line)))
            continue

        op = record.get("op")
        if header is None:
            if op != "header":
                raise TraceFormatError(f"first record is {op!r}, not a header")
            header = _parse_header(record)
            continue
        if op == "header":
            problems.append(TraceProblem(line_number, "duplicate header record", _excerpt(line)))
            continue
        if not isinstance(op, str) or not op:
            problems.append(TraceProblem(line_number, "record has no op", _excerpt(line)))
            continue
        events.append(TraceEvent(line_number=line_number, op=op, raw=record))

    if header is None:
        raise TraceFormatError("trace is empty; no header record")

    transfers = correlate(events)
    return Trace(
        header=header,
        events=tuple(events),
        transfers=transfers,
        problems=tuple(problems),
        exchanges=group_exchanges(transfers),
    )


def load_trace(path: Path) -> Trace:
    """Parse the trace at ``path``. Errors from the filesystem propagate."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return parse_trace(handle)


# ----------------------------------------------------------------- correlate

#: What the kernel identifies a URB by, from this decoder's side of the file:
#: the process, the descriptor, and the address of the caller's URB struct.
#: A program is free to reuse that address once a URB has been reaped, so a
#: slot can hold several transfers over the life of a capture.
_Slot = tuple[int | None, int | None, str | None]

#: Sort position for an event with no recorded ``t_ns``. Real timestamps come
#: from CLOCK_MONOTONIC and are non-negative, so this orders an untimed record
#: before every timed one: unknown time means "could have been arbitrarily
#: early", which keeps it available for pairing rather than dropping it.
_UNTIMED: Final = -1


def _entry_ns(event: TraceEvent) -> int:
    """When the call was entered, or ``_UNTIMED`` if the record does not say."""
    value = event.t_ns
    return _UNTIMED if value is None else value


def _return_ns(event: TraceEvent) -> int:
    """When the call returned: entry plus its measured duration.

    A blocking ``REAPURB`` can be entered *before* the URB it eventually
    returns was ever submitted, so the reap's entry time is not an upper bound
    on its partner's entry time - its return time is. When the record carries
    no usable duration the entry time is used, which is the conservative
    choice: it can only refuse a pairing, never invent one.
    """
    entry = _entry_ns(event)
    if entry == _UNTIMED:
        return _UNTIMED
    duration = _as_int(event.raw.get("duration_ns"))
    return entry + duration if duration is not None and duration > 0 else entry


def _stream_position(event: TraceEvent) -> tuple[int, int]:
    """Emission order: sequence number, with line number as the tie-break."""
    seq = event.seq
    return (event.line_number if seq is None else seq, event.line_number)


def _could_precede(submit: TraceEvent, reap: TraceEvent) -> bool:
    """Could ``submit`` be the submission this reap returned?

    Only if the URB was handed to the kernel before the reap call returned.
    Either timestamp being absent makes the test vacuous rather than negative,
    so a trace whose tracer recorded no times still correlates by order.
    """
    submitted = _entry_ns(submit)
    returned = _return_ns(reap)
    if submitted == _UNTIMED or returned == _UNTIMED:
        return True
    return submitted <= returned


def correlate(events: Sequence[TraceEvent]) -> tuple[Transfer, ...]:
    """Join URB submissions to their completions.

    The kernel identifies a URB by the address of the caller's
    ``struct usbdevfs_urb``, so ``(pid, fd, urb)`` names *a slot*, not a
    transfer: the moment one URB is reaped the program is free to submit
    another from the same address. Correlation therefore has to decide which
    occupant of a reused slot a completion belongs to, and it cannot do that
    from the order the records were *written*, because the tracer stamps
    ``seq`` when a call returns and takes the output lock - a completion
    handled on one thread can be written before the submission it completes.

    So the pairing is done per slot, over the recorded *call* times:

    * Completions are considered in the order they returned. Each one takes
      the earliest submission that could have preceded it - that is, one
      entered no later than the reap returned (:func:`_could_precede`).
    * A submission no completion could claim stays ``submit_only``; a
      completion no submission could serve stays ``reap_only``. Neither is
      dropped, and neither is forced onto a partner it cannot have.

    ``correlation`` still distinguishes ``matched`` from ``deferred``, but
    that now describes only *how the records were written* - ``deferred``
    means the completion's record was emitted before its submission's - and
    no longer reflects which pass found the pair.
    """
    submits: dict[_Slot, list[TraceEvent]] = {}
    reaps: dict[_Slot, list[TraceEvent]] = {}
    discards: dict[_Slot, list[TraceEvent]] = {}
    transfers: list[Transfer] = []

    for event in events:
        if event.op != "ioctl":
            continue
        slot: _Slot = (event.pid, event.fd, event.urb)
        phase = event.phase
        if phase == "submit":
            if event.failed or event.urb is None:
                # A rejected submission never produces a completion; record it
                # as a finished, failed transfer instead of leaving it pending.
                transfers.append(Transfer("submit_only", submit=event, reap=None))
            else:
                submits.setdefault(slot, []).append(event)
        elif phase == "discard":
            discards.setdefault(slot, []).append(event)
        elif phase == "reap" and event.urb is not None:
            reaps.setdefault(slot, []).append(event)

    for slot in submits.keys() | reaps.keys():
        transfers.extend(
            _correlate_slot(
                sorted(submits.get(slot, ()), key=lambda e: (_entry_ns(e), _stream_position(e))),
                sorted(reaps.get(slot, ()), key=lambda e: (_return_ns(e), _stream_position(e))),
                sorted(discards.get(slot, ()), key=lambda e: (_entry_ns(e), _stream_position(e))),
            )
        )

    transfers.sort(key=lambda t: t.order)
    return tuple(transfers)


def _correlate_slot(
    submits: list[TraceEvent],
    reaps: list[TraceEvent],
    discards: list[TraceEvent],
) -> list[Transfer]:
    """Pair one ``(pid, fd, urb)`` slot's submissions with its completions."""
    unclaimed = list(submits)
    transfers: list[Transfer] = []

    for reap in reaps:
        partner: TraceEvent | None = None
        for candidate in unclaimed:
            if _could_precede(candidate, reap):
                partner = candidate
                break
        if partner is None:
            transfers.append(Transfer("reap_only", submit=None, reap=reap))
            continue
        unclaimed.remove(partner)
        written_in_order = _stream_position(partner) < _stream_position(reap)
        transfers.append(Transfer("matched" if written_in_order else "deferred", partner, reap))
    transfers.extend(Transfer("submit_only", submit, None) for submit in unclaimed)

    for event in discards:
        _attach_discard(transfers, event)
    return transfers


def _attach_discard(transfers: list[Transfer], event: TraceEvent) -> None:
    """Record a ``DISCARDURB`` against the transfer it could have cancelled.

    A discard names the same slot as the URB it cancels, so on a reused slot
    the only thing separating the candidates is time: prefer a transfer that
    was still in flight when the discard was issued, and otherwise fall back
    to the most recent submission that had already happened.
    """
    when = _entry_ns(event)
    started = [
        (index, transfer)
        for index, transfer in enumerate(transfers)
        if transfer.discard is None
        and transfer.submit is not None
        and (when == _UNTIMED or _entry_ns(transfer.submit) <= when)
    ]
    if not started:
        return
    in_flight = [
        pair
        for pair in started
        if pair[1].reap is None or when == _UNTIMED or _return_ns(pair[1].reap) >= when
    ]
    index, transfer = (in_flight or started)[-1 if not in_flight else 0]
    transfers[index] = dataclasses.replace(transfer, discard=event)


def group_exchanges(transfers: Sequence[Transfer]) -> tuple[Exchange, ...]:
    """Group transfers into OUT-then-IN runs per endpoint number.

    Structural only - see :class:`Exchange`. Control transfers carry their own
    direction in the setup packet and are emitted as single-transfer groups.
    """
    groups: list[Exchange] = []
    open_group: dict[int, list[Transfer]] = {}

    def close(number: int) -> None:
        held = open_group.pop(number, None)
        if held:
            groups.append(Exchange(number, held[0], tuple(held[1:])))

    for transfer in transfers:
        endpoint = transfer.endpoint
        if endpoint is None:
            continue
        number = endpoint & 0x0F
        if transfer.transfer_type == "control":
            close(number)
            groups.append(Exchange(number, transfer, ()))
            continue
        if transfer.direction == "out":
            close(number)
            open_group[number] = [transfer]
        elif number in open_group:
            open_group[number].append(transfer)
        else:
            groups.append(Exchange(number, None, (transfer,)))

    for number in sorted(open_group):
        close(number)
    return tuple(groups)


# ------------------------------------------------------------------ summary


def summarize(trace: Trace) -> dict[str, object]:
    """Aggregate counts, per-endpoint totals, and every observed failure."""
    by_op: dict[str, int] = {}
    by_ioctl: dict[str, int] = {}
    descriptors: list[dict[str, object]] = []
    interfaces: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, object]] = []
    payload_truncated = 0
    output_limited = False
    events_dropped: int | None = None

    for event in trace.events:
        by_op[event.op] = by_op.get(event.op, 0) + 1
        if event.op == "ioctl":
            by_ioctl[event.name] = by_ioctl.get(event.name, 0) + 1
            if event.failed:
                failures.append(
                    {
                        "seq": event.seq,
                        "name": event.name,
                        "fd": event.fd,
                        "ret": event.ret,
                        "errno_name": event.errno_name,
                        "interface": _as_int(event.raw.get("interface")),
                    }
                )
            _accumulate_interface(interfaces, event)
        elif event.op in {"open", "dup", "adopt"}:
            path = event.raw.get("path")
            descriptors.append(
                {
                    "fd": event.fd,
                    "op": event.op,
                    "seq": event.seq,
                    "path": path if isinstance(path, str) else None,
                }
            )
        elif event.op == "limit":
            output_limited = True
        elif event.op == "summary":
            events_dropped = _as_int(event.raw.get("events_dropped"))
        if event.payload_truncated:
            payload_truncated += 1

    return {
        "counts": {
            "events": len(trace.events),
            "transfers": len(trace.transfers),
            "exchanges": len(trace.exchanges),
            "problems": len(trace.problems),
            "by_op": dict(sorted(by_op.items())),
            "by_ioctl": dict(sorted(by_ioctl.items())),
            "by_correlation": _count_correlations(trace.transfers),
        },
        "file_descriptors": descriptors,
        "interfaces": [interfaces[k] for k in sorted(interfaces)],
        "endpoints": _endpoint_totals(trace.transfers),
        "failures": failures,
        "capture_limits": {
            "max_payload_bytes": trace.header.max_payload_bytes,
            "max_output_bytes": trace.header.max_output_bytes,
            "requested_max_payload_bytes": trace.header.requested_max_payload_bytes,
            "requested_max_output_bytes": trace.header.requested_max_output_bytes,
            "limits_clamped": trace.header.limits_clamped,
            "events_with_truncated_payload": payload_truncated,
            "output_byte_limit_reached": output_limited,
            "events_dropped_by_tracer": events_dropped,
            "notes": list(trace.header.limitations),
        },
    }


def _count_correlations(transfers: Sequence[Transfer]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for transfer in transfers:
        counts[transfer.correlation] = counts.get(transfer.correlation, 0) + 1
    return dict(sorted(counts.items()))


def _accumulate_interface(store: dict[int, dict[str, Any]], event: TraceEvent) -> None:
    """Fold one interface-scoped ioctl into the per-interface tally."""
    number = _as_int(event.raw.get("interface"))
    if number is None:
        return
    entry = store.setdefault(
        number,
        {
            "interface": number,
            "claims": 0,
            "claim_failures": 0,
            "releases": 0,
            "driver_queries": 0,
            "detach_requests": 0,
            "drivers": [],
        },
    )
    name = event.name
    if name == "USBDEVFS_CLAIMINTERFACE":
        if event.failed:
            entry["claim_failures"] += 1
        else:
            entry["claims"] += 1
    elif name == "USBDEVFS_RELEASEINTERFACE":
        entry["releases"] += 1
    elif name == "USBDEVFS_GETDRIVER":
        entry["driver_queries"] += 1
    elif name in {"USBDEVFS_IOCTL", "USBDEVFS_DISCONNECT_CLAIM"}:
        entry["detach_requests"] += 1
    driver = event.raw.get("driver")
    if isinstance(driver, str) and driver and driver not in entry["drivers"]:
        entry["drivers"].append(driver)
        entry["drivers"].sort()


def _endpoint_totals(transfers: Sequence[Transfer]) -> list[dict[str, object]]:
    """Per (endpoint, transfer type) byte and status totals, in address order."""
    totals: dict[tuple[int, str], dict[str, Any]] = {}
    for transfer in transfers:
        endpoint = transfer.endpoint
        if endpoint is None:
            continue
        kind = transfer.transfer_type or "unknown"
        entry = totals.setdefault(
            (endpoint, kind),
            {
                "endpoint": endpoint,
                "endpoint_hex": f"0x{endpoint:02x}",
                "endpoint_number": endpoint & 0x0F,
                "direction": "in" if endpoint & 0x80 else "out",
                "transfer_type": kind,
                "transfers": 0,
                "completed": 0,
                "incomplete": 0,
                "bytes_requested": 0,
                "bytes_transferred": 0,
                "statuses": {},
                "payload_bytes_captured": 0,
            },
        )
        entry["transfers"] += 1
        entry["bytes_requested"] += transfer.requested_bytes or 0
        if transfer.reap is None:
            entry["incomplete"] += 1
        else:
            entry["completed"] += 1
            entry["bytes_transferred"] += transfer.actual_bytes or 0
            status = str(transfer.status)
            entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
        for data in (transfer.request_data, transfer.response_data):
            if data:
                entry["payload_bytes_captured"] += len(data)

    ordered = []
    for key in sorted(totals):
        entry = totals[key]
        entry["statuses"] = dict(sorted(entry["statuses"].items()))
        ordered.append(entry)
    return ordered


# ------------------------------------------------------------------- render


def render_human(trace: Trace) -> str:
    """A readable report of the same facts ``as_dict`` emits."""
    summary = summarize(trace)
    counts = summary["counts"]
    limits = summary["capture_limits"]
    assert isinstance(counts, dict) and isinstance(limits, dict)
    out: list[str] = []

    out.append(f"usbfs trace v{trace.header.version} of {trace.header.target}")
    if trace.header.pid is not None:
        out.append(f"  traced pid {trace.header.pid}, clock {trace.header.clock}")
    out.append(
        f"  {counts['events']} events, {counts['transfers']} transfers, "
        f"{counts['problems']} unusable lines"
    )
    out.append("")

    out.append("File descriptors")
    # 'adopt' means the tracer recognised the descriptor on its first usbfs
    # ioctl - by its /proc/self/fd path and its st_rdev - rather than by
    # intercepting the call that opened it.
    descriptors = summary["file_descriptors"]
    assert isinstance(descriptors, list)
    if not descriptors:
        out.append("  none (no fd was opened on the target during the capture)")
    for entry in descriptors:
        out.append(f"  fd {entry['fd']} via {entry['op']} -> {entry['path']}")
    out.append("")

    out.append("ioctls")
    by_ioctl = counts["by_ioctl"]
    assert isinstance(by_ioctl, dict)
    for name, count in by_ioctl.items():
        out.append(f"  {count:>6}  {name}")
    out.append("")

    out.append("Interfaces")
    interfaces = summary["interfaces"]
    assert isinstance(interfaces, list)
    if not interfaces:
        out.append("  none observed")
    for entry in interfaces:
        drivers = ", ".join(entry["drivers"]) or "none reported"
        out.append(
            f"  interface {entry['interface']}: claims={entry['claims']} "
            f"(failed {entry['claim_failures']}), releases={entry['releases']}, "
            f"detach requests={entry['detach_requests']}, kernel driver: {drivers}"
        )
    out.append("")

    out.append("Endpoints")
    endpoints = summary["endpoints"]
    assert isinstance(endpoints, list)
    if not endpoints:
        out.append("  none observed")
    for entry in endpoints:
        statuses = ", ".join(f"{k}x{v}" for k, v in entry["statuses"].items()) or "none"
        out.append(
            f"  {entry['endpoint_hex']} {entry['direction']:<3} {entry['transfer_type']:<9} "
            f"transfers={entry['transfers']} completed={entry['completed']} "
            f"incomplete={entry['incomplete']} "
            f"req={entry['bytes_requested']}B actual={entry['bytes_transferred']}B "
            f"status[{statuses}]"
        )
    out.append("")

    out.append("Grouped transfers (structural: OUT then IN on one endpoint number)")
    if not trace.exchanges:
        out.append("  none")
    for index, exchange in enumerate(trace.exchanges):
        out.append(f"  group {index} on endpoint number {exchange.endpoint_number}")
        for transfer in (exchange.request, *exchange.responses):
            if transfer is None:
                continue
            out.append(f"    {_render_transfer(transfer)}")
    out.append("")

    failures = summary["failures"]
    assert isinstance(failures, list)
    out.append(f"Failed ioctls ({len(failures)})")
    for entry in failures:
        out.append(
            f"  seq {entry['seq']}: {entry['name']} -> {entry['ret']} "
            f"{entry['errno_name']} (interface {entry['interface']})"
        )
    out.append("")

    out.append("Capture limits")
    out.append(f"  max payload per event: {limits['max_payload_bytes']} bytes")
    out.append(f"  max output size:       {limits['max_output_bytes']} bytes")
    if limits["limits_clamped"]:
        out.append(
            f"  requested (clamped):   payload {limits['requested_max_payload_bytes']} bytes, "
            f"output {limits['requested_max_output_bytes']} bytes"
        )
    elif limits["limits_clamped"] is None:
        out.append("  requested limits:      not recorded by this tracer version")
    out.append(f"  events with truncated payload: {limits['events_with_truncated_payload']}")
    out.append(f"  output byte limit reached:     {limits['output_byte_limit_reached']}")
    out.append(f"  events dropped by tracer:      {limits['events_dropped_by_tracer']}")
    notes = limits["notes"]
    assert isinstance(notes, list)
    for note in notes:
        out.append(f"  - {note}")

    if trace.problems:
        out.append("")
        out.append(f"Unusable lines ({len(trace.problems)})")
        for problem in trace.problems:
            out.append(f"  line {problem.line_number}: {problem.reason}")

    return "\n".join(out) + "\n"


def _render_transfer(transfer: Transfer) -> str:
    endpoint = "??" if transfer.endpoint is None else f"0x{transfer.endpoint:02x}"
    parts = [
        f"{endpoint} {transfer.direction or '?'}",
        f"{transfer.transfer_type or '?'}",
        f"[{transfer.correlation}]",
        f"req={transfer.requested_bytes} actual={transfer.actual_bytes}",
        f"status={transfer.status}",
    ]
    setup = transfer.setup
    if setup is not None:
        parts.append(
            "setup="
            + " ".join(
                f"{k}=0x{v:02x}" if k in {"bmRequestType", "bRequest"} else f"{k}=0x{v:04x}"
                for k, v in setup.items()
            )
        )
    for label, data in (("->", transfer.request_data), ("<-", transfer.response_data)):
        if data:
            shown = data[:HUMAN_PAYLOAD_BYTES]
            tail = "..." if len(data) > HUMAN_PAYLOAD_BYTES else ""
            parts.append(f"{label} {shown.hex(' ')}{tail}")
    return "  ".join(parts)


__all__ = [
    "HUMAN_PAYLOAD_BYTES",
    "MAX_TRACE_LINES",
    "SUPPORTED_VERSIONS",
    "TRACE_FORMAT",
    "Exchange",
    "Trace",
    "TraceEvent",
    "TraceHeader",
    "TraceProblem",
    "Transfer",
    "correlate",
    "group_exchanges",
    "load_trace",
    "parse_trace",
    "render_human",
    "summarize",
]
