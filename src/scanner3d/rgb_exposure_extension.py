"""Dry-by-default, fixed one-transfer RGB exposure-extension reader.

The SDK's separate ``UvcCamera`` extension path reads one 60-byte GET_CUR
reply from entity 4 on interface 0.  This module models that immutable read and
reuses :class:`InterfaceZeroLifecycle` for the future live path.  No caller can
change the entity, selector, direction, length, timeout or transfer count.

The decoded scaled values are deliberately labelled as SDK-derived numeric
mapping, not microseconds or any other physical unit.  Scanner-027 observed
the selector GET_CUR once as a 2/60-byte short reply; this module still keeps
its strict exact-60-byte decode gate.  The separate GET_LEN reader is in
:mod:`scanner3d.rgb_exposure_length`.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import signal
import struct
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d.config import stated_serial
from scanner3d.control import (
    USBDEVFS_CONTROL,
    Clock,
    ControlRequest,
    IoctlFn,
    TargetSnapshot,
    pack_ctrltransfer,
)
from scanner3d.errors import RgbExposureExtensionNotAllowed
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
AUTHORIZED_SERIAL: Final = stated_serial()
ENTITY_ID: Final = 4
PROPERTY_SELECTOR: Final = 0x12
RESULT_LENGTH: Final = 60
TIMEOUT_MS: Final = 1000
MAX_CONTROL_TRANSFERS: Final = 1
WORD_COUNT: Final = 4
WORD_FORMAT: Final = "<IIII"
WORD_BYTES: Final = struct.calcsize(WORD_FORMAT)


class Step(enum.StrEnum):
    RESULT_READ = "result_read"


class ExtensionOutcome(enum.StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    SHORT_RESPONSE = "short_response"
    OVERLONG_RESPONSE = "overlong_response"
    TRANSFER_ERROR = "transfer_error"
    INTERRUPTED = "interrupted"
    EXCEPTION = "exception"


EXTENSION_EXIT_CODES: Final = MappingProxyType(
    {
        ExtensionOutcome.OVERLONG_RESPONSE: 7,
        ExtensionOutcome.EXCEPTION: 5,
    }
)


_READ_REQUEST: Final = ControlRequest(
    name="rgb-exposure-extension-get-cur",
    request_type=0xA1,
    request=0x81,
    value=PROPERTY_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=RESULT_LENGTH,
    timeout_ms=TIMEOUT_MS,
    description="UVC GET_CUR, entity 4 selector 0x12 on interface 0, 60-byte reply",
)
# Public alias for callers/tests that need to inspect the immutable wire shape.
READ_REQUEST: Final = _READ_REQUEST


@dataclass(frozen=True, slots=True)
class ExtensionTransfer:
    step: Step
    request: ControlRequest
    note: str

    def as_dict(self) -> dict[str, object]:
        return {"step": str(self.step), "request": self.request.as_dict(), "note": self.note}


READ_TRANSFER: Final = ExtensionTransfer(
    Step.RESULT_READ,
    _READ_REQUEST,
    "the one immutable SDK-observed 60-byte GET_CUR; no SET or second transfer",
)
WHITELIST: Final[tuple[ExtensionTransfer, ...]] = (READ_TRANSFER,)
_ALLOWED_IDENTITIES: Final = frozenset(transfer.request.wire_tuple for transfer in WHITELIST)


def ensure_transfer_allowed(transfer: ExtensionTransfer = READ_TRANSFER) -> ExtensionTransfer:
    request = transfer.request
    if transfer not in WHITELIST or request.wire_tuple not in _ALLOWED_IDENTITIES:
        raise RgbExposureExtensionNotAllowed(
            "only the fixed RGB exposure-extension GET_CUR is allowed"
        )
    if (
        not request.is_read
        or request.interface_number != OWNED_INTERFACE
        or request.entity_id != ENTITY_ID
        or request.control_selector != PROPERTY_SELECTOR
        or request.length != RESULT_LENGTH
        or request.timeout_ms != TIMEOUT_MS
    ):
        raise RgbExposureExtensionNotAllowed(
            "RGB exposure-extension setup packet drifted from the fixed read"
        )
    return transfer


@dataclass(frozen=True, slots=True)
class SdkExposureMapping:
    current: int
    minimum: int
    maximum: int
    step: int = 1000
    provenance: str = (
        "SDK-observed mapping: word0*1000, word1*1000, word2*1000, fixed step 1000; "
        "word3 is logged by the SDK but is not returned as step"
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "current": self.current,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "provenance": self.provenance,
            "unit": "unknown; not independently labelled as microseconds",
        }


@dataclass(frozen=True, slots=True)
class RangeWarning:
    code: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class RgbExposureExtensionReading:
    words: tuple[int, int, int, int]
    mapping: SdkExposureMapping
    warnings: tuple[RangeWarning, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_words": list(self.words),
            "raw_word_offsets": [0, 4, 8, 12],
            "word3_logged_only": True,
            "sdk_mapping": self.mapping.as_dict(),
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


def decode_reply(reply: bytes) -> RgbExposureExtensionReading | None:
    """Decode only an exact 60-byte answer; never invent missing words."""
    if len(reply) != RESULT_LENGTH:
        return None
    words = struct.unpack_from(WORD_FORMAT, reply, 0)
    mapping = SdkExposureMapping(words[0] * 1000, words[1] * 1000, words[2] * 1000)
    warnings: list[RangeWarning] = []
    if mapping.minimum > mapping.maximum:
        warnings.append(
            RangeWarning(
                "minimum_greater_than_maximum",
                f"SDK-derived minimum {mapping.minimum} exceeds maximum {mapping.maximum}",
            )
        )
    if mapping.current < mapping.minimum or mapping.current > mapping.maximum:
        warnings.append(
            RangeWarning(
                "current_outside_range",
                f"SDK-derived current {mapping.current} is outside "
                f"{mapping.minimum}..{mapping.maximum}",
            )
        )
    return RgbExposureExtensionReading(words, mapping, tuple(warnings))


@dataclass(frozen=True, slots=True)
class TransferRecord:
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
class ReplyEvidence:
    requested_length: int
    actual_length: int | None
    raw_length: int
    raw_hex: str
    sha256: str
    exact_length: bool
    complete: bool
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
            "note": self.note,
        }


def reply_evidence(raw: bytes, actual: int | None, *, complete: bool, note: str) -> ReplyEvidence:
    return ReplyEvidence(
        RESULT_LENGTH,
        actual,
        len(raw),
        raw.hex(),
        hashlib.sha256(raw).hexdigest(),
        actual == RESULT_LENGTH,
        complete,
        note,
    )


@dataclass(slots=True)
class RgbExposureExtensionReadOperation:
    ioctl_fn: IoctlFn
    clock: Clock
    transfers: list[TransferRecord] = field(default_factory=list)
    outcome: ExtensionOutcome = ExtensionOutcome.NOT_ATTEMPTED
    reply: bytes = b""
    actual_length: int | None = None
    reading: RgbExposureExtensionReading | None = None
    read_duration_s: float | None = None

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        self.outcome = ExtensionOutcome.INTERRUPTED
        started = self.clock()
        try:
            primary = self._run(fd)
        except BaseException as exc:
            duration = self.clock() - started
            self.read_duration_s = duration
            interrupted = isinstance(exc, (SignalReceived, KeyboardInterrupt))
            self.outcome = (
                ExtensionOutcome.INTERRUPTED if interrupted else ExtensionOutcome.EXCEPTION
            )
            phases.append(
                PhaseResult(
                    Phase.CONTROL_READ,
                    False,
                    f"{type(exc).__name__} during USBDEVFS_CONTROL: {exc}",
                    duration,
                )
            )
            if isinstance(exc, KeyboardInterrupt):
                raise SignalReceived(signal.SIGINT) from exc
            if interrupted:
                raise
            # InterfaceZeroLifecycle deliberately catches Exception to keep
            # its cleanup/close evidence typed. Convert an unexpected
            # BaseException into that same cleanup path while retaining the
            # original exception type in the transfer record and phase.
            raise RuntimeError(f"unexpected {type(exc).__name__} during control read") from exc
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
        return ProtocolCleanupRecord.not_needed(
            "one device-to-host GET_CUR creates no protocol state"
        )

    def _run(self, fd: int) -> _Primary:
        ensure_transfer_allowed()
        buffer = ctypes.create_string_buffer(RESULT_LENGTH)
        packed = pack_ctrltransfer(READ_TRANSFER.request, ctypes.addressof(buffer))
        started = self.clock()
        try:
            transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            self.transfers.append(
                TransferRecord(
                    Step.RESULT_READ,
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
            self.outcome = ExtensionOutcome.TRANSFER_ERROR
            return _Primary(
                _ERRNO_STATUS.get(number or 0, OwnershipStatus.IOCTL_ERROR),
                str(exc),
                errno=number,
                errno_name=name,
            )
        except BaseException as exc:
            raw = b""
            self.reply = raw
            self.transfers.append(
                TransferRecord(
                    Step.RESULT_READ,
                    READ_TRANSFER.request,
                    None,
                    False,
                    f"{type(exc).__name__}: {exc}",
                    raw.hex(),
                    hashlib.sha256(raw).hexdigest(),
                    self.clock() - started,
                    exception_type=type(exc).__name__,
                )
            )
            self.outcome = (
                ExtensionOutcome.INTERRUPTED
                if not isinstance(exc, Exception)
                else ExtensionOutcome.EXCEPTION
            )
            raise
        raw = b"" if transferred < 0 else bytes(buffer.raw[: min(transferred, RESULT_LENGTH)])
        self.reply = raw
        self.actual_length = transferred
        exact = transferred == RESULT_LENGTH
        self.transfers.append(
            TransferRecord(
                Step.RESULT_READ,
                READ_TRANSFER.request,
                transferred,
                transferred == RESULT_LENGTH,
                f"device returned {transferred} bytes",
                raw.hex(),
                hashlib.sha256(raw).hexdigest(),
                self.clock() - started,
            )
        )
        if transferred < 0:
            self.outcome = ExtensionOutcome.TRANSFER_ERROR
            return _Primary(OwnershipStatus.IOCTL_ERROR, f"USBDEVFS_CONTROL returned {transferred}")
        if not exact:
            self.outcome = (
                ExtensionOutcome.SHORT_RESPONSE
                if transferred < RESULT_LENGTH
                else ExtensionOutcome.OVERLONG_RESPONSE
            )
            return _Primary(
                OwnershipStatus.SHORT_RESPONSE,
                f"device returned {transferred} of {RESULT_LENGTH} bytes",
            )
        self.reading = decode_reply(raw)
        self.outcome = ExtensionOutcome.COMPLETE
        return _Primary(OwnershipStatus.OK, "read exactly 60 bytes and decoded four raw words")


@dataclass(frozen=True, slots=True)
class RgbExposureExtensionOutcome:
    status: OwnershipStatus
    extension_outcome: ExtensionOutcome
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    transfers: tuple[TransferRecord, ...]
    reply: ReplyEvidence
    reading: RgbExposureExtensionReading | None
    cleanup: CleanupRecord
    protocol_cleanup: ProtocolCleanupRecord
    detail: str
    read_duration_s: float | None = None
    total_duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    @property
    def exit_code(self) -> int:
        if not self.cleanup.ok:
            return CLEANUP_FAILED_EXIT
        if self.extension_outcome in EXTENSION_EXIT_CODES:
            return EXTENSION_EXIT_CODES[self.extension_outcome]
        return OWNERSHIP_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        return (
            self.status is OwnershipStatus.OK
            and self.extension_outcome is ExtensionOutcome.COMPLETE
            and self.reading is not None
            and self.cleanup.ok
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "vendor-rgb-exposure-extension-read",
            "status": str(self.status),
            "extension_outcome": str(self.extension_outcome),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "authorized_serial": AUTHORIZED_SERIAL,
            "owned_interface": OWNED_INTERFACE,
            "entity_id": ENTITY_ID,
            "control_selector": f"0x{PROPERTY_SELECTOR:02x}",
            "bounds": {
                "max_control_transfers": MAX_CONTROL_TRANSFERS,
                "reply_bytes": RESULT_LENGTH,
                "timeout_ms": TIMEOUT_MS,
                "retries": 0,
            },
            "whitelist": [transfer.as_dict() for transfer in WHITELIST],
            "target": None if self.target is None else self.target.as_dict(),
            "phases": [phase.as_dict() for phase in self.phases],
            "transfers": [transfer.as_dict() for transfer in self.transfers],
            "reply": self.reply.as_dict(),
            "reading": None if self.reading is None else self.reading.as_dict(),
            "cleanup": self.cleanup.as_dict(),
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "read_duration_s": self.read_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


DRY_RUN_DETAIL: Final = (
    "dry run: no device node was opened, no driver was detached and nothing was sent "
    "(pass --execute-rgb-exposure-extension-read to perform the read)"
)


class UsbfsRgbExposureExtensionBackend(InterfaceZeroLifecycle):
    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return RgbExposureExtensionReadOperation(self.ioctl_fn, self.clock)

    def _snapshot_request(self) -> ControlRequest:
        return READ_TRANSFER.request

    def preflight(self, serial: str | None) -> RgbExposureExtensionOutcome:
        ensure_transfer_allowed()
        return self._from_preflight(self._preflight(serial, DRY_RUN_DETAIL))

    def run(self, serial: str) -> RgbExposureExtensionOutcome:
        ensure_transfer_allowed()
        if serial != AUTHORIZED_SERIAL:
            return self._empty(
                OwnershipStatus.REFUSED,
                f"this read is authorized for serial {AUTHORIZED_SERIAL!r} only; "
                f"refusing {serial!r}",
            )
        started = self.clock()
        preflight = self._preflight(serial, DRY_RUN_DETAIL)
        if preflight.status is not OwnershipStatus.NOT_ATTEMPTED or preflight.target is None:
            return self._from_preflight(preflight)
        phases = list(preflight.phases)
        primary, window = self._perform_lifecycle(preflight.target, phases)
        operation = window.operation
        if not isinstance(operation, RgbExposureExtensionReadOperation):
            raise RgbExposureExtensionNotAllowed("unexpected operation type")
        complete = operation.outcome is ExtensionOutcome.COMPLETE and window.cleanup.ok
        return RgbExposureExtensionOutcome(
            primary.status,
            operation.outcome,
            preflight.target,
            tuple(phases),
            tuple(operation.transfers),
            reply_evidence(
                operation.reply,
                operation.actual_length,
                complete=complete,
                note="raw reply retained without scaling or normalization",
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

    def _from_preflight(self, preflight: Preflight) -> RgbExposureExtensionOutcome:
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
    ) -> RgbExposureExtensionOutcome:
        return RgbExposureExtensionOutcome(
            status,
            ExtensionOutcome.NOT_ATTEMPTED,
            target,
            phases,
            (),
            reply_evidence(b"", None, complete=False, note="nothing was read"),
            None,
            CleanupRecord.not_needed(cleanup_note),
            ProtocolCleanupRecord.not_needed(
                "nothing was written, so no protocol cleanup is needed"
            ),
            detail,
        )


def dry_run(serial: str | None = None) -> RgbExposureExtensionOutcome:
    """Return the inert plan without constructing or resolving a live backend."""
    detail = DRY_RUN_DETAIL
    if serial is not None:
        detail = f"{detail}; requested serial {serial!r}"
    return RgbExposureExtensionOutcome(
        OwnershipStatus.NOT_ATTEMPTED,
        ExtensionOutcome.NOT_ATTEMPTED,
        None,
        (),
        (),
        reply_evidence(b"", None, complete=False, note="nothing was read"),
        None,
        CleanupRecord.not_needed("dry run; nothing was detached"),
        ProtocolCleanupRecord.not_needed("dry run; nothing was written"),
        detail,
    )
