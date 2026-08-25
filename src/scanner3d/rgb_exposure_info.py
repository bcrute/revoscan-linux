"""Dry-by-default, fixed UVC GET_INFO reader for the RGB extension selector.

The only transfer this module can describe or issue is one device-to-host
``GET_INFO`` for selector 0x12 on entity 4/interface 0.  The ownership lifecycle
is shared with the other interface-0 readers; this module has no protocol
state to clear and never constructs a second transfer.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import signal
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d.config import EXAMPLE_SERIAL, configured_serial
from scanner3d.control import (
    USBDEVFS_CONTROL,
    Clock,
    ControlRequest,
    IoctlFn,
    TargetSnapshot,
    pack_ctrltransfer,
)
from scanner3d.errors import RgbExposureInfoNotAllowed
from scanner3d.ownership import (
    _ERRNO_STATUS,
    CLEANUP_FAILED_EXIT,
    OWNED_INTERFACE,
    OWNERSHIP_EXIT_CODES,
    CleanupRecord,
    InterfaceZeroLifecycle,
    OwnedOperation,
    OwnershipStatus,
    Phase,
    PhaseResult,
    Preflight,
    ProtocolCleanupRecord,
    SignalReceived,
    _errno_of,
    _Primary,
)

#: The scanner this installation may act on. Read from configuration rather
#: than hard-coded: several operations here detach kernel drivers or write
#: vendor property keys, and must not be aimed at whatever happens to be
#: plugged in. Empty when nothing is configured - call
#: scanner3d.config.resolve_serial() for a usable value, which falls back to
#: the single attached scanner and refuses when that is ambiguous.
AUTHORIZED_SERIAL: Final = configured_serial() or EXAMPLE_SERIAL
ENTITY_ID: Final = 4
PROPERTY_SELECTOR: Final = 0x12
REQUESTED_INFO: Final = 1
RESULT_INFO: Final = REQUESTED_INFO
TIMEOUT_MS: Final = 1000
MAX_CONTROL_TRANSFERS: Final = 1
INFO_GET: Final = 1 << 0
INFO_SET: Final = 1 << 1
INFO_DISABLED: Final = 1 << 2
INFO_AUTOUPDATE: Final = 1 << 3
INFO_ASYNCHRONOUS: Final = 1 << 4
INFO_RESERVED_MASK: Final = 0xE0


class Step(enum.StrEnum):
    INFO_READ = "info_read"


class InfoOutcome(enum.StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    SHORT_RESPONSE = "short_response"
    OVERLONG_RESPONSE = "overlong_response"
    TRANSFER_ERROR = "transfer_error"
    INTERRUPTED = "interrupted"
    EXCEPTION = "exception"


RGB_EXPOSURE_INFO_EXIT_CODES: Final = MappingProxyType(
    {
        InfoOutcome.OVERLONG_RESPONSE: 7,
        InfoOutcome.EXCEPTION: 5,
    }
)


_READ_REQUEST: Final = ControlRequest(
    name="rgb-exposure-get-info",
    request_type=0xA1,
    request=0x86,
    value=PROPERTY_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=REQUESTED_INFO,
    timeout_ms=TIMEOUT_MS,
    description="UVC GET_INFO, entity 4 selector 0x12 on interface 0, one-byte reply",
)
READ_REQUEST: Final = _READ_REQUEST


@dataclass(frozen=True, slots=True)
class InfoTransfer:
    step: Step
    request: ControlRequest
    note: str

    def as_dict(self) -> dict[str, object]:
        return {"step": str(self.step), "request": self.request.as_dict(), "note": self.note}


READ_TRANSFER: Final = InfoTransfer(
    Step.INFO_READ,
    _READ_REQUEST,
    "the one immutable UVC GET_INFO; no second transfer",
)
WHITELIST: Final[tuple[InfoTransfer, ...]] = (READ_TRANSFER,)
_ALLOWED_IDENTITIES: Final = frozenset(item.request.wire_tuple for item in WHITELIST)


def ensure_transfer_allowed(transfer: InfoTransfer = READ_TRANSFER) -> InfoTransfer:
    request = transfer.request
    if transfer not in WHITELIST or request.wire_tuple not in _ALLOWED_IDENTITIES:
        raise RgbExposureInfoNotAllowed("only the fixed RGB exposure GET_INFO is allowed")
    if (
        not request.is_read
        or request.interface_number != OWNED_INTERFACE
        or request.entity_id != ENTITY_ID
        or request.control_selector != PROPERTY_SELECTOR
        or request.request != 0x86
        or request.length != REQUESTED_INFO
        or request.timeout_ms != TIMEOUT_MS
    ):
        raise RgbExposureInfoNotAllowed("GET_INFO setup drifted from the fixed read")
    return transfer


@dataclass(frozen=True, slots=True)
class InfoReading:
    raw_info: int

    @property
    def bit_view(self) -> dict[str, object]:
        return {
            "supports_get": bool(self.raw_info & INFO_GET),
            "supports_set": bool(self.raw_info & INFO_SET),
            "disabled": bool(self.raw_info & INFO_DISABLED),
            "autoupdate": bool(self.raw_info & INFO_AUTOUPDATE),
            "asynchronous": bool(self.raw_info & INFO_ASYNCHRONOUS),
            "reserved_high_bits": self.raw_info & INFO_RESERVED_MASK,
            "interpretation": "syntactic Linux-UAPI capability-bit view only",
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_info": self.raw_info,
            "bit_view": self.bit_view,
            "unit": "raw UVC GET_INFO byte; standard-bit syntax only",
        }


def decode_reply(reply: bytes) -> InfoReading | None:
    """Decode only an exact one-byte capability reply."""
    if len(reply) != REQUESTED_INFO:
        return None
    return InfoReading(int.from_bytes(reply, "little", signed=False))


@dataclass(frozen=True, slots=True)
class InfoTransferRecord:
    step: Step
    request: ControlRequest
    transferred: int | None
    ok: bool
    detail: str
    response_hex: str
    response_sha256: str
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None
    exception_type: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": str(self.step),
            "request": self.request.as_dict(),
            "setup_hex": bytes(pack_ctrltransfer(self.request, 0)[:12]).hex(),
            "transferred": self.transferred,
            "ok": self.ok,
            "detail": self.detail,
            "response_hex": self.response_hex,
            "response_sha256": self.response_sha256,
            "duration_s": self.duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True, slots=True)
class InfoReplyEvidence:
    requested_length: int
    actual_length: int | None
    raw_length: int
    raw_hex: str
    sha256: str
    exact_length: bool
    complete: bool
    raw_info: int | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_length": self.requested_length,
            "actual_length": self.actual_length,
            "raw_length": self.raw_length,
            "raw_hex": self.raw_hex,
            "sha256": self.sha256,
            "exact_length": self.exact_length,
            "complete": self.complete,
            "raw_info": self.raw_info,
            "note": self.note,
        }


def reply_evidence(
    raw: bytes, actual: int | None, *, complete: bool, reading: InfoReading | None, note: str
) -> InfoReplyEvidence:
    return InfoReplyEvidence(
        REQUESTED_INFO,
        actual,
        len(raw),
        raw.hex(),
        hashlib.sha256(raw).hexdigest(),
        actual == REQUESTED_INFO,
        complete,
        None if reading is None else reading.raw_info,
        note,
    )


@dataclass(slots=True)
class RgbExposureInfoReadOperation:
    ioctl_fn: IoctlFn
    clock: Clock
    transfers: list[InfoTransferRecord] = field(default_factory=list)
    outcome: InfoOutcome = InfoOutcome.NOT_ATTEMPTED
    reply: bytes = b""
    actual_length: int | None = None
    reading: InfoReading | None = None
    read_duration_s: float | None = None

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        self.outcome = InfoOutcome.INTERRUPTED
        started = self.clock()
        try:
            primary = self._run(fd)
        except BaseException as exc:
            self.read_duration_s = self.clock() - started
            self.outcome = (
                InfoOutcome.INTERRUPTED
                if isinstance(exc, (SignalReceived, KeyboardInterrupt))
                else InfoOutcome.EXCEPTION
            )
            phases.append(
                PhaseResult(
                    Phase.CONTROL_READ,
                    False,
                    f"{type(exc).__name__} during USBDEVFS_CONTROL: {exc}",
                    self.read_duration_s,
                )
            )
            if isinstance(exc, KeyboardInterrupt):
                raise SignalReceived(signal.SIGINT) from exc
            if isinstance(exc, SignalReceived):
                raise
            raise RuntimeError(f"unexpected {type(exc).__name__} during GET_INFO") from exc
        self.read_duration_s = self.clock() - started
        primary.duration_s = self.read_duration_s
        phases.append(
            PhaseResult(
                Phase.CONTROL_READ,
                primary.status is OwnershipStatus.OK,
                primary.detail,
                self.read_duration_s,
                primary.errno,
                primary.errno_name,
            )
        )
        return primary

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        return ProtocolCleanupRecord.not_needed("GET_INFO creates no protocol state")

    def _run(self, fd: int) -> _Primary:
        ensure_transfer_allowed()
        buffer = ctypes.create_string_buffer(REQUESTED_INFO)
        packed = pack_ctrltransfer(READ_TRANSFER.request, ctypes.addressof(buffer))
        started = self.clock()
        try:
            transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            self.transfers.append(
                InfoTransferRecord(
                    Step.INFO_READ,
                    READ_TRANSFER.request,
                    None,
                    False,
                    f"USBDEVFS_CONTROL failed: {exc}",
                    "",
                    hashlib.sha256(b"").hexdigest(),
                    self.clock() - started,
                    number,
                    name,
                )
            )
            self.outcome = InfoOutcome.TRANSFER_ERROR
            return _Primary(
                _ERRNO_STATUS.get(number or 0, OwnershipStatus.IOCTL_ERROR),
                str(exc),
                errno=number,
                errno_name=name,
            )
        except BaseException as exc:
            self.transfers.append(
                InfoTransferRecord(
                    Step.INFO_READ,
                    READ_TRANSFER.request,
                    None,
                    False,
                    f"{type(exc).__name__}: {exc}",
                    "",
                    hashlib.sha256(b"").hexdigest(),
                    self.clock() - started,
                    exception_type=type(exc).__name__,
                )
            )
            self.outcome = (
                InfoOutcome.INTERRUPTED
                if isinstance(exc, (SignalReceived, KeyboardInterrupt))
                else InfoOutcome.EXCEPTION
            )
            raise
        raw = b"" if transferred < 0 else bytes(buffer.raw[: min(transferred, REQUESTED_INFO)])
        self.reply = raw
        self.actual_length = transferred
        self.transfers.append(
            InfoTransferRecord(
                Step.INFO_READ,
                READ_TRANSFER.request,
                transferred,
                transferred == REQUESTED_INFO,
                f"device returned {transferred} bytes",
                raw.hex(),
                hashlib.sha256(raw).hexdigest(),
                self.clock() - started,
            )
        )
        if transferred < 0:
            self.outcome = InfoOutcome.TRANSFER_ERROR
            return _Primary(OwnershipStatus.IOCTL_ERROR, f"USBDEVFS_CONTROL returned {transferred}")
        if transferred < REQUESTED_INFO:
            self.outcome = InfoOutcome.SHORT_RESPONSE
            return _Primary(
                OwnershipStatus.SHORT_RESPONSE, f"device returned {transferred} of 1 byte"
            )
        if transferred > REQUESTED_INFO:
            self.outcome = InfoOutcome.OVERLONG_RESPONSE
            return _Primary(
                OwnershipStatus.SHORT_RESPONSE, f"device returned {transferred} of 1 byte"
            )
        self.reading = decode_reply(raw)
        self.outcome = InfoOutcome.COMPLETE
        return _Primary(OwnershipStatus.OK, "read exactly one GET_INFO byte")


@dataclass(frozen=True, slots=True)
class RgbExposureInfoOutcome:
    status: OwnershipStatus
    info_outcome: InfoOutcome
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    transfers: tuple[InfoTransferRecord, ...]
    reply: InfoReplyEvidence
    reading: InfoReading | None
    cleanup: CleanupRecord
    protocol_cleanup: ProtocolCleanupRecord
    detail: str
    read_duration_s: float | None = None
    total_duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    @property
    def raw_info(self) -> int | None:
        return None if self.reading is None else self.reading.raw_info

    @property
    def exit_code(self) -> int:
        if not self.cleanup.ok:
            return CLEANUP_FAILED_EXIT
        if self.info_outcome in RGB_EXPOSURE_INFO_EXIT_CODES:
            return RGB_EXPOSURE_INFO_EXIT_CODES[self.info_outcome]
        return OWNERSHIP_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        return (
            self.status is OwnershipStatus.OK
            and self.info_outcome is InfoOutcome.COMPLETE
            and self.reading is not None
            and self.cleanup.ok
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "vendor-rgb-exposure-get-info",
            "status": str(self.status),
            "info_outcome": str(self.info_outcome),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "authorized_serial": AUTHORIZED_SERIAL,
            "owned_interface": OWNED_INTERFACE,
            "entity_id": ENTITY_ID,
            "control_selector": f"0x{PROPERTY_SELECTOR:02x}",
            "bounds": {
                "max_control_transfers": 1,
                "reply_bytes": 1,
                "timeout_ms": 1000,
                "retries": 0,
            },
            "whitelist": [transfer.as_dict() for transfer in WHITELIST],
            "target": None if self.target is None else self.target.as_dict(),
            "phases": [phase.as_dict() for phase in self.phases],
            "transfers": [transfer.as_dict() for transfer in self.transfers],
            "reply": self.reply.as_dict(),
            "reading": None if self.reading is None else self.reading.as_dict(),
            "raw_info": self.raw_info,
            "bit_view": None if self.reading is None else self.reading.bit_view,
            "limitations": (
                "GET_INFO bit syntax does not prove selector advertisement, exposure semantics, "
                "SET acceptance, units or frame behavior"
            ),
            "recovery": self.cleanup.guidance,
            "cleanup": self.cleanup.as_dict(),
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "read_duration_s": self.read_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


DRY_RUN_DETAIL: Final = (
    "dry run: no device node was opened, no driver was detached and nothing was sent "
    "(pass --execute-rgb-exposure-info-read to perform one GET_INFO)"
)


class UsbfsRgbExposureInfoBackend(InterfaceZeroLifecycle):
    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return RgbExposureInfoReadOperation(self.ioctl_fn, self.clock)

    def _snapshot_request(self) -> ControlRequest:
        return READ_TRANSFER.request

    def preflight(self, serial: str | None) -> RgbExposureInfoOutcome:
        ensure_transfer_allowed()
        return self._from_preflight(self._preflight(serial, DRY_RUN_DETAIL))

    def run(self, serial: str) -> RgbExposureInfoOutcome:
        ensure_transfer_allowed()
        if serial != AUTHORIZED_SERIAL:
            return self._empty(
                OwnershipStatus.REFUSED,
                "this read is authorized for serial "
                f"{AUTHORIZED_SERIAL!r} only; refusing {serial!r}",
            )
        started = self.clock()
        preflight = self._preflight(serial, DRY_RUN_DETAIL)
        if preflight.status is not OwnershipStatus.NOT_ATTEMPTED or preflight.target is None:
            return self._from_preflight(preflight)
        phases = list(preflight.phases)
        primary, window = self._perform_lifecycle(preflight.target, phases)
        operation = window.operation
        if not isinstance(operation, RgbExposureInfoReadOperation):
            raise RgbExposureInfoNotAllowed("unexpected GET_INFO operation type")
        complete = operation.outcome is InfoOutcome.COMPLETE and window.cleanup.ok
        return RgbExposureInfoOutcome(
            primary.status,
            operation.outcome,
            preflight.target,
            tuple(phases),
            tuple(operation.transfers),
            reply_evidence(
                operation.reply,
                operation.actual_length,
                complete=complete,
                reading=operation.reading,
                note="raw GET_INFO reply retained; zero is not a support verdict",
            ),
            operation.reading,
            window.cleanup,
            window.protocol,
            primary.detail,
            operation.read_duration_s,
            self.clock() - started,
            primary.errno,
            primary.errno_name,
        )

    def _from_preflight(self, preflight: Preflight) -> RgbExposureInfoOutcome:
        return self._empty(
            preflight.status,
            preflight.detail,
            target=preflight.target,
            phases=preflight.phases,
            cleanup_note=preflight.cleanup_note,
        )

    def _empty(
        self,
        status: OwnershipStatus,
        detail: str,
        *,
        target: TargetSnapshot | None = None,
        phases: tuple[PhaseResult, ...] = (),
        cleanup_note: str = "nothing was detached",
    ) -> RgbExposureInfoOutcome:
        return RgbExposureInfoOutcome(
            status,
            InfoOutcome.NOT_ATTEMPTED,
            target,
            phases,
            (),
            reply_evidence(b"", None, complete=False, reading=None, note="nothing was read"),
            None,
            CleanupRecord.not_needed(cleanup_note),
            ProtocolCleanupRecord.not_needed("GET_INFO creates no protocol state"),
            detail,
        )


def dry_run(serial: str | None = None) -> RgbExposureInfoOutcome:
    detail = DRY_RUN_DETAIL if serial is None else f"{DRY_RUN_DETAIL}; requested serial {serial!r}"
    return RgbExposureInfoOutcome(
        OwnershipStatus.NOT_ATTEMPTED,
        InfoOutcome.NOT_ATTEMPTED,
        None,
        (),
        (),
        reply_evidence(b"", None, complete=False, reading=None, note="nothing was read"),
        None,
        CleanupRecord.not_needed("dry run; nothing was detached"),
        ProtocolCleanupRecord.not_needed("dry run; nothing was written"),
        detail,
    )


# Compatibility aliases make the dedicated surface easy to discover without
# importing or widening the older GET_CUR module.
RgbExposureInfoRead = RgbExposureInfoReadOperation
RgbExposureInfoStatus = InfoOutcome
