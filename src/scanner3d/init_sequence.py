"""One bounded reproduction of the SDK's observed ``/tmp/inited`` exchange.

This is the first thing in this repository that *writes* to the scanner, and
the whole safety argument is that it can write exactly two payloads, in exactly
one order, as part of exactly one four-transfer sequence that the vendor SDK
was already observed performing on this unit
(``docs/architecture.md``, "The vendor channel, byte by byte", rows 1-4):

===== ============= ================================= =========================
Step  Direction     Setup packet                      Payload
===== ============= ================================= =========================
1     host->device  ``21 01`` value ``0x0100`` index   ``01`` ``"/tmp/inited"``
                    ``0x0400`` length 60              ``00`` then zero padding
2     device->host  ``a1 81`` value ``0x0200`` index   observed: ``0c 00 00 00``
                    ``0x0400`` length 60              ``"v7.5.7.0426"`` ``00``
3     device->host  the same request again             observed: 60 zero bytes
4     host->device  ``21 01`` value ``0x0100`` index   60 zero bytes
                    ``0x0400`` length 60
===== ============= ================================= =========================

The assignment writes those two words as ``0001`` and ``0004``; this module
packs host values ``0x0100`` and ``0x0400``. That is the same transfer, and the
two notations agree byte for byte: ``0x0100`` little-endian is ``00 01`` on the
wire and ``0x0400`` is ``00 04``, which is exactly what the packed
``struct usbdevfs_ctrltransfer`` contains and what a test asserts. (The third
word, ``003c``, is written as the host value 60 instead, and 60 is what is
packed.) The host reading is also the only coherent one: ``wIndex`` is
``(entity ID << 8) | interface number``, so ``0x0400`` is entity 4 on interface
0 - the extension unit the device's own descriptors declare, on the interface
this lifecycle takes from ``uvcvideo`` - and it is what ``docs/architecture.md``
records from the capture. A literal ``wIndex`` of ``0x0004`` would name
interface 4, the HID function, which is out of scope and which this module
cannot reach: :func:`ensure_transfer_allowed` refuses any transfer whose
``wIndex`` does not name interface 0.

What is fixed here, and is not a parameter of anything:

* the four transfers, as a tuple, in order, checked as a whole before a
  descriptor is opened and again per transfer before each ioctl argument is
  packed;
* both outbound payloads, byte for byte - the path is a constant, not a string
  a caller supplies, and the clearing payload is 60 zeros;
* the 3 s timeout ceiling inherited from :mod:`scanner3d.control`;
* the interface, which is 0, because the lifecycle in
  :mod:`scanner3d.ownership` will not encode another one.

There is no way through this module to send a different path, a different
selector, a different entity, a different length, a different opcode byte, or
any transfer at all outside these four. ``/persist/sdk-license`` and
``/data/sdk-license`` - the other two paths in the same capture - are not here
and cannot be reached from here.

**Step 4 is protocol cleanup, not an optional fifth idea.** The SDK issued it
after every path read, and this module treats it the same way the driver
handback is treated: owed from the instant before the path write is attempted,
attempted exactly once, shielded from SIGINT and SIGTERM, reported separately,
and never presented as success when it fails. It runs before the driver is
given back, because it needs the claim. See
:class:`scanner3d.ownership.InterfaceZeroLifecycle`, which owns that ordering
and which this module reuses rather than re-implements.

Nothing here retries. The path write and the two reads are each attempted at
most once, whatever they return.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Final

from scanner3d.control import (
    EXPECTED_ALGORITHM_VERSION,
    MAX_TIMEOUT_MS,
    USBDEVFS_CONTROL,
    Clock,
    ControlRequest,
    IoctlFn,
    TargetSnapshot,
    pack_ctrltransfer,
)
from scanner3d.errors import InitSequenceNotAllowed
from scanner3d.ownership import (
    _ERRNO_STATUS,
    CLEANUP_FAILED_EXIT,
    OWNED_INTERFACE,
    OWNERSHIP_EXIT_CODES,
    PROTOCOL_CLEANUP_FAILED_EXIT,
    CleanupRecord,
    InterfaceZeroLifecycle,
    OwnedOperation,
    OwnershipStatus,
    Phase,
    PhaseResult,
    Preflight,
    ProtocolCleanupRecord,
    _errno_of,
    _Primary,
)

#: The one path this module can name. Not an argument anywhere.
INITED_PATH: Final = "/tmp/inited"

#: The leading byte of a path payload. One value was ever observed.
OPEN_PATH_OPCODE: Final = 0x01

#: Every transfer in the observed sequence is exactly this long.
PAYLOAD_LENGTH: Final = 60

#: ``wIndex`` decodes as ``(entity << 8) | interface``.
ENTITY_ID: Final = 4

#: ``wValue`` decodes as ``(selector << 8)``.
PATH_SELECTOR: Final = 0x01
RESULT_SELECTOR: Final = 0x02


def _padded(prefix: bytes) -> bytes:
    """A 60-byte payload: ``prefix`` then zero padding, exactly as captured."""
    if len(prefix) > PAYLOAD_LENGTH:
        raise InitSequenceNotAllowed(
            f"payload prefix is {len(prefix)} bytes, over the fixed {PAYLOAD_LENGTH}"
        )
    return prefix + bytes(PAYLOAD_LENGTH - len(prefix))


#: ``01`` + ``"/tmp/inited"`` + NUL + zero padding.
PATH_PAYLOAD: Final = _padded(bytes([OPEN_PATH_OPCODE]) + INITED_PATH.encode("ascii") + b"\x00")

#: The clearing payload, and the only other thing this module can write.
ZERO_PAYLOAD: Final = bytes(PAYLOAD_LENGTH)

#: What the SDK's first read came back with: a little-endian length, then the
#: algorithm version string, NUL-terminated and zero-padded.
OBSERVED_ALGORITHM_RECORD: Final = _padded(
    struct.pack("<I", len(EXPECTED_ALGORITHM_VERSION) + 1)
    + EXPECTED_ALGORITHM_VERSION.encode("ascii")
    + b"\x00"
)

#: What the SDK's second read came back with.
OBSERVED_ZERO_RESPONSE: Final = ZERO_PAYLOAD


# --- the closed whitelist -----------------------------------------------------


class Step(enum.StrEnum):
    """The four transfers, in the only order they can occur."""

    PATH_WRITE = "path_write"
    RESULT_READ = "result_read"
    SECOND_READ = "second_read"
    CLEAR_WRITE = "clear_write"


@dataclass(frozen=True, slots=True)
class InitTransfer:
    """One fully specified transfer of the observed sequence.

    Constructing one is harmless; only members of :data:`INITED_SEQUENCE` can be
    handed to the hardware path, and :func:`ensure_transfer_allowed` checks that
    by value - the setup packet *and* the outbound bytes - not by name.
    """

    step: Step
    request: ControlRequest
    payload: bytes | None
    observed_response: bytes | None
    note: str

    @property
    def is_write(self) -> bool:
        return self.payload is not None

    @property
    def identity(self) -> tuple[tuple[int, ...], bytes | None]:
        """Setup packet plus outbound bytes: the whole of what reaches the wire."""
        return self.request.wire_tuple, self.payload

    def as_dict(self) -> dict[str, object]:
        return {
            "step": str(self.step),
            "request": self.request.as_dict(),
            "payload_hex": None if self.payload is None else self.payload.hex(),
            "observed_response_hex": (
                None if self.observed_response is None else self.observed_response.hex()
            ),
            "note": self.note,
        }


_PATH_WRITE_REQUEST: Final = ControlRequest(
    name="inited-path-set-cur",
    request_type=0x21,
    request=0x01,
    value=PATH_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=PAYLOAD_LENGTH,
    timeout_ms=MAX_TIMEOUT_MS,
    description="UVC SET_CUR, entity 4 selector 0x01 on interface 0, 60-byte path payload",
)

_RESULT_READ_REQUEST: Final = ControlRequest(
    name="inited-result-get-cur",
    request_type=0xA1,
    request=0x81,
    value=RESULT_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=PAYLOAD_LENGTH,
    timeout_ms=MAX_TIMEOUT_MS,
    description="UVC GET_CUR, entity 4 selector 0x02 on interface 0, 60-byte result payload",
)

#: The whole sequence. Nothing outside this tuple can be issued.
INITED_SEQUENCE: Final[tuple[InitTransfer, ...]] = (
    InitTransfer(
        step=Step.PATH_WRITE,
        request=_PATH_WRITE_REQUEST,
        payload=PATH_PAYLOAD,
        observed_response=None,
        note=f"open-path opcode 0x01 followed by {INITED_PATH!r}, NUL-terminated",
    ),
    InitTransfer(
        step=Step.RESULT_READ,
        request=_RESULT_READ_REQUEST,
        payload=None,
        observed_response=OBSERVED_ALGORITHM_RECORD,
        note="the SDK saw a little-endian length 0x0000000c and the algorithm version",
    ),
    InitTransfer(
        step=Step.SECOND_READ,
        request=_RESULT_READ_REQUEST,
        payload=None,
        observed_response=OBSERVED_ZERO_RESPONSE,
        note="the same read again; the SDK saw 60 zero bytes",
    ),
    InitTransfer(
        step=Step.CLEAR_WRITE,
        request=_PATH_WRITE_REQUEST,
        payload=ZERO_PAYLOAD,
        observed_response=None,
        note="protocol cleanup: the SDK cleared the path selector with 60 zero bytes",
    ),
)

#: The three transfers performed inside the operation. The fourth is cleanup.
_ATTEMPTED_IN_ORDER: Final = INITED_SEQUENCE[:3]
CLEARING_TRANSFER: Final = INITED_SEQUENCE[3]

_ALLOWED_IDENTITIES: Final = frozenset(t.identity for t in INITED_SEQUENCE)


def ensure_sequence_allowed(sequence: tuple[InitTransfer, ...] = INITED_SEQUENCE) -> None:
    """Check the sequence as a whole, before anything is resolved or opened.

    Guards the shape as well as the contents: four transfers, in the observed
    order, with the observed steps, so a sequence that merely happens to be
    built from allowed transfers cannot be reordered or truncated.
    """
    if sequence != INITED_SEQUENCE:
        raise InitSequenceNotAllowed(
            "only the exact four-transfer /tmp/inited sequence may be issued; "
            f"got {[str(t.step) for t in sequence]}"
        )
    for transfer in sequence:
        ensure_transfer_allowed(transfer)


def ensure_transfer_allowed(transfer: InitTransfer) -> InitTransfer:
    """Return ``transfer`` if it is one of the four, else raise.

    Called again immediately before each ioctl argument is packed, so a
    transfer that reached the hardware path some other way still never gets
    encoded.
    """
    if transfer.request.timeout_ms <= 0 or transfer.request.timeout_ms > MAX_TIMEOUT_MS:
        raise InitSequenceNotAllowed(
            f"{transfer.step}: timeout {transfer.request.timeout_ms} ms "
            f"outside (0, {MAX_TIMEOUT_MS}]"
        )
    if transfer.request.length != PAYLOAD_LENGTH:
        raise InitSequenceNotAllowed(
            f"{transfer.step}: wLength {transfer.request.length} is not {PAYLOAD_LENGTH}"
        )
    if transfer.request.interface_number != OWNED_INTERFACE:
        raise InitSequenceNotAllowed(
            f"{transfer.step}: wIndex names interface "
            f"{transfer.request.interface_number}, not {OWNED_INTERFACE}"
        )
    if transfer.payload is not None and len(transfer.payload) != PAYLOAD_LENGTH:
        raise InitSequenceNotAllowed(
            f"{transfer.step}: payload is {len(transfer.payload)} bytes, not {PAYLOAD_LENGTH}"
        )
    if transfer.identity not in _ALLOWED_IDENTITIES:
        raise InitSequenceNotAllowed(
            f"{transfer.step}: setup packet {transfer.request.wire_tuple} with this payload "
            "is not one of the four observed transfers"
        )
    return transfer


# --- responses ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InitResponse:
    """A 60-byte reply, hashed, parsed, and compared to what the SDK saw."""

    raw_length: int
    sha256: str
    hex_prefix: str
    all_zero: bool
    declared_length: int | None
    text: str | None
    nul_padded: bool
    matches_observed: bool
    decode_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_length": self.raw_length,
            "sha256": self.sha256,
            "hex_prefix": self.hex_prefix,
            "all_zero": self.all_zero,
            "declared_length": self.declared_length,
            "text": self.text,
            "nul_padded": self.nul_padded,
            "matches_observed": self.matches_observed,
            "decode_error": self.decode_error,
        }


def parse_response(raw: bytes, observed: bytes | None) -> InitResponse:
    """Decode one reply against the shape the SDK trace recorded. Never raises.

    The vendor record is read as the ``0c 00 00 00`` prefix reported in
    ``docs/architecture.md`` - a little-endian ``u32`` length - followed by an
    ASCII string. That reading is inference from one observation, so the raw
    bytes and their digest are reported alongside it and the match is decided
    on the bytes, not on the parse.
    """
    digest = hashlib.sha256(raw).hexdigest()
    all_zero = raw == bytes(len(raw))
    matches = observed is not None and raw == observed
    if len(raw) < 4:
        return InitResponse(
            raw_length=len(raw),
            sha256=digest,
            hex_prefix=raw.hex(),
            all_zero=all_zero,
            declared_length=None,
            text=None,
            nul_padded=False,
            matches_observed=matches,
            decode_error=f"only {len(raw)} bytes; too short to carry a length prefix",
        )
    declared = int.from_bytes(raw[:4], "little")
    body, separator, padding = raw[4:].partition(b"\x00")
    nul_padded = bool(separator) and padding == bytes(len(padding))
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as exc:
        return InitResponse(
            raw_length=len(raw),
            sha256=digest,
            hex_prefix=raw[:16].hex(),
            all_zero=all_zero,
            declared_length=declared,
            text=None,
            nul_padded=nul_padded,
            matches_observed=matches,
            decode_error=f"body is not ASCII: {exc}",
        )
    error = None
    if not all_zero and declared != len(text) + 1:
        error = f"declared length {declared} does not match {len(text)} ASCII bytes plus a NUL"
    return InitResponse(
        raw_length=len(raw),
        sha256=digest,
        hex_prefix=raw[:16].hex(),
        all_zero=all_zero,
        declared_length=declared,
        text=text,
        nul_padded=nul_padded,
        matches_observed=matches,
        decode_error=error,
    )


@dataclass(frozen=True, slots=True)
class TransferResult:
    """One attempted transfer and exactly how it ended."""

    step: Step
    request: ControlRequest
    sent_sha256: str | None
    transferred: int | None
    response: InitResponse | None
    ok: bool
    matched: bool
    detail: str
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": str(self.step),
            "direction": "in" if self.request.is_read else "out",
            "request": self.request.as_dict(),
            "sent_sha256": self.sent_sha256,
            "transferred": self.transferred,
            "ok": self.ok,
            "matched_observed": self.matched,
            "detail": self.detail,
            "duration_s": self.duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
            "response": None if self.response is None else self.response.as_dict(),
        }


# --- the operation ------------------------------------------------------------


def clear_write_guidance(usb_path: str) -> str:
    """What the owner should know when the clearing write did not land."""
    return (
        f"the clearing SET_CUR to entity {ENTITY_ID} on interface {OWNED_INTERFACE} of "
        f"{usb_path} did not complete, so the device may still hold {INITED_PATH!r} "
        "selected in its vendor file-read state. Nothing in this repository reads or "
        "writes that state again on its own; power-cycle the scanner before drawing "
        "conclusions from a later run."
    )


@dataclass(slots=True)
class InitedSequenceOperation:
    """The three owned transfers, plus the clearing write as protocol cleanup.

    Built fresh for each run by :class:`UsbfsInitSequenceBackend`, which is why
    it may accumulate results. Nothing about the sequence is a field: the
    transfers come from :data:`INITED_SEQUENCE` and the syscalls are injected.
    """

    ioctl_fn: IoctlFn
    clock: Clock
    usb_path: str
    results: list[TransferResult] = field(default_factory=list)
    #: Raised immediately *before* the path write's ioctl, because a signal
    #: between the two would leave completion unknowable. The clearing write is
    #: owed from that instant, not from the write's return.
    path_write_attempted: bool = False
    cleanup_attempted: bool = False
    sequence_duration_s: float | None = None

    # -- OwnedOperation ------------------------------------------------------

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        """Issue transfers 1-3 once each, in order. Never retries."""
        started = self.clock()
        primary = self._run(fd)
        self.sequence_duration_s = self.clock() - started
        primary.duration_s = self.sequence_duration_s
        phases.append(
            PhaseResult(
                phase=Phase.OWNED_SEQUENCE,
                ok=primary.status is OwnershipStatus.OK,
                detail=primary.detail,
                duration_s=self.sequence_duration_s,
                errno=primary.errno,
                errno_name=primary.errno_name,
            )
        )
        return primary

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        """Issue the clearing write once, if the path write may have landed."""
        if not self.path_write_attempted:
            return ProtocolCleanupRecord.not_needed(
                f"the path SET_CUR was never issued, so the device holds no {INITED_PATH!r} "
                "selection to clear"
            )
        if self.cleanup_attempted:
            return ProtocolCleanupRecord(
                attempted=True,
                ok=False,
                note="the clearing write was already attempted; it is never repeated",
                guidance=clear_write_guidance(self.usb_path),
            )
        self.cleanup_attempted = True
        result = self._issue(fd, CLEARING_TRANSFER)
        self.results.append(result)
        return ProtocolCleanupRecord(
            attempted=True,
            ok=result.ok,
            note=(
                f"the {INITED_PATH!r} selection was cleared with 60 zero bytes"
                if result.ok
                else f"the clearing SET_CUR did NOT complete: {result.detail}"
            ),
            duration_s=result.duration_s,
            guidance=None if result.ok else clear_write_guidance(self.usb_path),
        )

    # -- transfers -----------------------------------------------------------

    def _run(self, fd: int) -> _Primary:
        """Transfers 1-3. A transport failure stops the sequence; content never does."""
        for transfer in _ATTEMPTED_IN_ORDER:
            result = self._issue(fd, transfer)
            self.results.append(result)
            if not result.ok:
                return _Primary(
                    status=_ERRNO_STATUS.get(result.errno or 0, OwnershipStatus.IOCTL_ERROR),
                    detail=f"{transfer.step}: {result.detail}",
                    errno=result.errno,
                    errno_name=result.errno_name,
                )
        for result in self.results:
            if result.matched:
                continue
            short = result.transferred != PAYLOAD_LENGTH
            return _Primary(
                status=(
                    OwnershipStatus.SHORT_RESPONSE if short else OwnershipStatus.UNEXPECTED_CONTENT
                ),
                detail=f"{result.step}: {result.detail}",
            )
        return _Primary(
            status=OwnershipStatus.OK,
            detail=(
                "all three transfers matched the SDK-observed sequence exactly: the path "
                f"write took 60 bytes, the first read returned {EXPECTED_ALGORITHM_VERSION!r} "
                "behind a little-endian length, and the second returned 60 zero bytes"
            ),
        )

    def _issue(self, fd: int, transfer: InitTransfer) -> TransferResult:
        """Pack and perform exactly one whitelisted transfer. One ioctl, no retry."""
        ensure_transfer_allowed(transfer)
        buffer = ctypes.create_string_buffer(transfer.request.length)
        sent: str | None = None
        if transfer.payload is not None:
            buffer.raw = transfer.payload
            sent = hashlib.sha256(transfer.payload).hexdigest()
        packed = pack_ctrltransfer(transfer.request, ctypes.addressof(buffer))
        if transfer.step is Step.PATH_WRITE:
            # Before the ioctl on purpose: a signal delivered between this line
            # and the kernel's completion leaves the write ambiguous, and the
            # clearing write is owed for an ambiguous write too.
            self.path_write_attempted = True
        started = self.clock()
        try:
            transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return TransferResult(
                step=transfer.step,
                request=transfer.request,
                sent_sha256=sent,
                transferred=None,
                response=None,
                ok=False,
                matched=False,
                detail=f"USBDEVFS_CONTROL failed: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        duration = self.clock() - started
        if transferred < 0:
            return TransferResult(
                step=transfer.step,
                request=transfer.request,
                sent_sha256=sent,
                transferred=transferred,
                response=None,
                ok=False,
                matched=False,
                detail=f"USBDEVFS_CONTROL returned {transferred}",
                duration_s=duration,
            )
        if transfer.payload is not None:
            matched = transferred == PAYLOAD_LENGTH
            return TransferResult(
                step=transfer.step,
                request=transfer.request,
                sent_sha256=sent,
                transferred=transferred,
                response=None,
                ok=True,
                matched=matched,
                detail=(
                    f"the device accepted all {PAYLOAD_LENGTH} bytes"
                    if matched
                    else f"the device accepted {transferred} of {PAYLOAD_LENGTH} bytes"
                ),
                duration_s=duration,
            )
        raw = bytes(buffer.raw[: min(transferred, transfer.request.length)])
        response = parse_response(raw, transfer.observed_response)
        return TransferResult(
            step=transfer.step,
            request=transfer.request,
            sent_sha256=None,
            transferred=transferred,
            response=response,
            ok=True,
            matched=response.matches_observed and transferred == PAYLOAD_LENGTH,
            detail=(
                f"returned {transferred} bytes matching the SDK-observed reply"
                if response.matches_observed and transferred == PAYLOAD_LENGTH
                else (
                    f"returned {transferred} bytes that differ from the SDK-observed reply "
                    f"(sha256 {response.sha256})"
                )
            ),
            duration_s=duration,
        )


# --- outcome ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InitSequenceOutcome:
    """The complete, typed result of attempting (or declining) the sequence."""

    status: OwnershipStatus
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    transfers: tuple[TransferResult, ...]
    cleanup: CleanupRecord
    protocol_cleanup: ProtocolCleanupRecord
    detail: str
    sequence_duration_s: float | None = None
    total_duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    @property
    def exit_code(self) -> int:
        """Either cleanup failing outranks every primary status, success included."""
        if not self.cleanup.ok:
            return CLEANUP_FAILED_EXIT
        if not self.protocol_cleanup.ok:
            return PROTOCOL_CLEANUP_FAILED_EXIT
        return OWNERSHIP_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        """True only when the sequence matched *and* both undos succeeded."""
        return self.status is OwnershipStatus.OK and self.cleanup.ok and self.protocol_cleanup.ok

    def transfer(self, step: Step) -> TransferResult | None:
        """The result recorded for ``step``, or ``None`` if it never ran."""
        return next((r for r in self.transfers if r.step is step), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "inited-path-sequence",
            "status": str(self.status),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "owned_interface": OWNED_INTERFACE,
            "entity_id": ENTITY_ID,
            "path": INITED_PATH,
            "sequence": [t.as_dict() for t in INITED_SEQUENCE],
            "target": self.target.as_dict() if self.target is not None else None,
            "phases": [p.as_dict() for p in self.phases],
            "transfers": [t.as_dict() for t in self.transfers],
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "cleanup": self.cleanup.as_dict(),
            "sequence_duration_s": self.sequence_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


#: What the dry run says instead of performing the sequence.
DRY_RUN_DETAIL: Final = (
    "dry run: no device node was opened, no driver was detached and nothing was "
    "written to the scanner (pass --execute-write-sequence to perform it)"
)


class UsbfsInitSequenceBackend(InterfaceZeroLifecycle):
    """The accepted interface-0 lifecycle carrying the four-transfer sequence.

    The lifecycle - confirm, detach, claim, operate, undo, release, reconnect -
    is :class:`scanner3d.ownership.InterfaceZeroLifecycle` unchanged. All this
    class adds is which operation runs inside it, and that is fixed at class
    level: nothing but the serial is a parameter of anything here.
    """

    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return InitedSequenceOperation(
            ioctl_fn=self.ioctl_fn, clock=self.clock, usb_path=snapshot.usb_path
        )

    def _snapshot_request(self) -> ControlRequest:
        return _PATH_WRITE_REQUEST

    # -- public API ----------------------------------------------------------

    def preflight(self, serial: str | None) -> InitSequenceOutcome:
        """Resolve the target and check every precondition, opening nothing."""
        ensure_sequence_allowed()
        return self._from_preflight(self._preflight(serial, DRY_RUN_DETAIL))

    def run(self, serial: str) -> InitSequenceOutcome:
        """Perform the sequence exactly once. No transfer is ever retried."""
        ensure_sequence_allowed()
        started = self.clock()
        preflight = self._preflight(serial, DRY_RUN_DETAIL)
        snapshot = preflight.target
        if preflight.status is not OwnershipStatus.NOT_ATTEMPTED or snapshot is None:
            return self._from_preflight(preflight)

        phases: list[PhaseResult] = list(preflight.phases)
        primary, window = self._perform_lifecycle(snapshot, phases)
        operation = window.operation
        if not isinstance(operation, InitedSequenceOperation):  # pragma: no cover
            raise InitSequenceNotAllowed(
                "the lifecycle ran an operation this backend did not build"
            )
        return InitSequenceOutcome(
            status=primary.status,
            target=snapshot,
            phases=tuple(phases),
            transfers=tuple(operation.results),
            cleanup=window.cleanup,
            protocol_cleanup=window.protocol,
            detail=primary.detail,
            sequence_duration_s=operation.sequence_duration_s,
            total_duration_s=self.clock() - started,
            errno=primary.errno,
            errno_name=primary.errno_name,
        )

    # -- assembly ------------------------------------------------------------

    def _from_preflight(self, preflight: Preflight) -> InitSequenceOutcome:
        return InitSequenceOutcome(
            status=preflight.status,
            target=preflight.target,
            phases=preflight.phases,
            transfers=(),
            cleanup=CleanupRecord.not_needed(preflight.cleanup_note),
            protocol_cleanup=ProtocolCleanupRecord.not_needed(
                "nothing was written, so there is nothing to clear"
            ),
            detail=preflight.detail,
        )
