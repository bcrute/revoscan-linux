"""One immutable blue-light off write, reported as a transfer and nothing else.

This module is deliberately not a general ISP/register interface.  It can only
issue the one SDK-observed laser-off command, and only through the shared
interface-0 ownership lifecycle.

What this module may honestly claim narrowed on 2026-08-24.  The command it
sends -- ``echo s 0x922 0 >/dev/rk_preisp`` -- is now known **not** to darken
this unit's emitter.  ``0x922=0``, ``0xb04=0``, ``0xb01=0`` and ``0xb00=0``
were each accepted as exact 60/60-byte transfers while the owner watched the
projector stay lit; two depth captures the same day reported an accepted,
restored finalizer with the projector still visibly lit afterwards; and firing
the device's own internal actuator (``ioctl 0x400456cc`` on
``/dev/v4l-subdev3``) with a verified exit status of 0 changed neither the
depth output nor anything visible.  ``docs/illumination-actuators.md`` holds
the evidence.

This project has never had optical readback, so the only fact any status here
can carry is whether the *transfer* was accepted.  Every name, status string
and rendering below is therefore about the transfer, and
:data:`EMITTER_LIMITATION` is attached permanently to the outcome so no report
of it can be read as a claim that light stopped.

The write is still issued after every depth capture, because the owner's
durable safety constraint requires the attempt to be made and evidenced.  What
it no longer does is score the capture: see
:attr:`IlluminationShutdownOutcome.succeeded`.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import signal
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d.config import configured_serial, stated_serial
from scanner3d.control import (
    USBDEVFS_CONTROL,
    Clock,
    ControlRequest,
    IoctlFn,
    TargetSnapshot,
    pack_ctrltransfer,
)
from scanner3d.errors import IlluminationShutdownNotAllowed
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
PROPERTY_ID: Final = 0x922
RESULT_LENGTH: Final = 60
TIMEOUT_MS: Final = 1000
MAX_CONTROL_TRANSFERS: Final = 1
COMMAND_TEXT: Final = "echo s 0x922 0 >/dev/rk_preisp"
COMMAND_PAYLOAD: Final = COMMAND_TEXT.encode("ascii") + b"\x00" * (
    RESULT_LENGTH - len(COMMAND_TEXT)
)
COMMAND_PAYLOAD_SHA256: Final = hashlib.sha256(COMMAND_PAYLOAD).hexdigest()
EXPECTED_SETUP: Final = bytes.fromhex("2101000700043c00e8030000")

#: The paragraph that must survive every summary, truncation and re-quote of an
#: outcome.  :class:`IlluminationShutdownOutcome` exposes it as a property and
#: emits it from ``as_dict``, so no caller can build an outcome without it and
#: no JSON consumer can read the status without meeting it.
EMITTER_LIMITATION: Final = (
    "EMITTER STATE UNKNOWN AND ASSUMED LIT. This repository has no optical readback, "
    "and on this unit the command it sends is KNOWN NOT TO DARKEN the emitter: "
    f"{COMMAND_TEXT!r} was accepted as an exact {RESULT_LENGTH}/{RESULT_LENGTH}-byte "
    "transfer while the owner watched the projector stay lit, as were 0xb04=0 and "
    "0xb01=0 (docs/illumination-actuators.md). An accepted transfer is evidence that "
    "bytes were taken, never that light stopped. Unplug the scanner before handling it."
)

#: One line of the same fact, for renderers that only have room for one.
EMITTER_STATE_LINE: Final = (
    "emitter state UNKNOWN - this command does not darken this unit's emitter and "
    "nothing here observes light"
)


class ShutdownStatus(enum.StrEnum):
    """How the one write ended.  Every member describes the *transfer*.

    :attr:`COMMAND_ACCEPTED` replaced an earlier ``COMPLETE``.  "Complete" was
    read -- reasonably -- as "the shutdown completed", which this module has
    never been able to establish and, since the evidence in the module
    docstring, is known to be false for this unit.  The device taking all
    sixty bytes is the whole of what the name may promise.
    """

    NOT_REQUIRED = "not_required"
    COMMAND_ACCEPTED = "command_accepted"
    REFUSED = "refused"
    DEVICE_ABSENT = "device_absent"
    DEVICE_REPLACED = "device_replaced"
    PERMISSION_DENIED = "permission_denied"
    DETACH_FAILED = "detach_failed"
    CLAIM_FAILED = "claim_failed"
    SHORT_WRITE = "short_write"
    OVERLONG_WRITE = "overlong_write"
    IOCTL_ERROR = "ioctl_error"
    UNEXPECTED_EXCEPTION = "unexpected_exception"
    INTERRUPTED = "interrupted"


#: Statuses in which the owed attempt was actually made and answered: the write
#: reached the device over interface 0 and the driver reported a definite
#: result, whatever that result was.  Everything outside this set means either
#: that the attempt could never be made (absent, replaced, refused, denied,
#: detach/claim failure) or that it ended indeterminate (signal, unexpected
#: exception), and in those cases the owner's constraint -- make the attempt,
#: evidence it -- was not discharged.
DISCHARGED_STATUSES: Final[frozenset[ShutdownStatus]] = frozenset(
    {
        ShutdownStatus.COMMAND_ACCEPTED,
        ShutdownStatus.SHORT_WRITE,
        ShutdownStatus.OVERLONG_WRITE,
        ShutdownStatus.IOCTL_ERROR,
    }
)


_STATUS_MAP: Final = MappingProxyType(
    {
        OwnershipStatus.OK: ShutdownStatus.COMMAND_ACCEPTED,
        OwnershipStatus.NOT_ATTEMPTED: ShutdownStatus.IOCTL_ERROR,
        OwnershipStatus.REFUSED: ShutdownStatus.REFUSED,
        OwnershipStatus.DEVICE_ABSENT: ShutdownStatus.DEVICE_ABSENT,
        OwnershipStatus.DEVICE_REPLACED: ShutdownStatus.DEVICE_REPLACED,
        OwnershipStatus.PERMISSION_DENIED: ShutdownStatus.PERMISSION_DENIED,
        OwnershipStatus.DETACH_FAILED: ShutdownStatus.DETACH_FAILED,
        OwnershipStatus.CLAIM_FAILED: ShutdownStatus.CLAIM_FAILED,
        OwnershipStatus.SHORT_RESPONSE: ShutdownStatus.SHORT_WRITE,
        OwnershipStatus.INTERRUPTED: ShutdownStatus.INTERRUPTED,
    }
)


_REQUEST: Final = ControlRequest(
    name="blue-depth-laser-off",
    request_type=0x21,
    request=0x01,
    value=0x0700,
    index=0x0400,
    length=RESULT_LENGTH,
    timeout_ms=TIMEOUT_MS,
    description="fixed SDK laser-off SET_CUR, entity 4/interface 0",
)
REQUEST: Final = _REQUEST


@dataclass(frozen=True, slots=True)
class ShutdownTransfer:
    request: ControlRequest
    payload: bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "request": self.request.as_dict(),
            "setup_hex": bytes(pack_ctrltransfer(self.request, 0)[:12]).hex(),
            "payload_hex": self.payload.hex(),
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
        }


OFF_TRANSFER: Final = ShutdownTransfer(_REQUEST, COMMAND_PAYLOAD)
WHITELIST: Final[tuple[ShutdownTransfer, ...]] = (OFF_TRANSFER,)


def ensure_shutdown_allowed(transfer: ShutdownTransfer = OFF_TRANSFER) -> ShutdownTransfer:
    """Reject every command except the exact observed laser-off write."""
    request = transfer.request
    if transfer not in WHITELIST or transfer.payload != COMMAND_PAYLOAD:
        raise IlluminationShutdownNotAllowed("only the fixed blue-light off command is allowed")
    if (
        request.wire_tuple != _REQUEST.wire_tuple
        or request.is_read
        or request.interface_number != OWNED_INTERFACE
        or request.entity_id != 4
        or request.control_selector != 7
    ):
        raise IlluminationShutdownNotAllowed("shutdown setup drifted from the fixed command")
    return transfer


@dataclass(frozen=True, slots=True)
class ShutdownTransferRecord:
    transferred: int | None
    ok: bool
    detail: str
    duration_s: float | None
    errno: int | None = None
    errno_name: str | None = None
    exception_type: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": "blue_light_off",
            "request": REQUEST.as_dict(),
            "setup_hex": EXPECTED_SETUP.hex(),
            "payload_hex": COMMAND_PAYLOAD.hex(),
            "payload_sha256": COMMAND_PAYLOAD_SHA256,
            "transferred": self.transferred,
            "ok": self.ok,
            "detail": self.detail,
            "duration_s": self.duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
            "exception_type": self.exception_type,
        }


@dataclass(slots=True)
class _ShutdownOperation:
    ioctl_fn: IoctlFn
    clock: Clock
    transfer: ShutdownTransfer = OFF_TRANSFER
    record: ShutdownTransferRecord | None = None
    outcome: ShutdownStatus = ShutdownStatus.IOCTL_ERROR

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        ensure_shutdown_allowed(self.transfer)
        buffer = ctypes.create_string_buffer(self.transfer.request.length)
        buffer.raw = self.transfer.payload
        packed = pack_ctrltransfer(self.transfer.request, ctypes.addressof(buffer))
        started = self.clock()
        try:
            transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            duration = self.clock() - started
            self.record = ShutdownTransferRecord(
                None,
                False,
                f"{type(exc).__name__}: USBDEVFS_CONTROL failed: {exc}",
                duration,
                number,
                name,
            )
            self.outcome = ShutdownStatus.IOCTL_ERROR
            phases.append(
                PhaseResult(
                    Phase.OWNED_SEQUENCE,
                    False,
                    f"{type(exc).__name__}: {exc}",
                    duration,
                    number,
                    name,
                )
            )
            return _Primary(
                _ERRNO_STATUS.get(number or 0, OwnershipStatus.IOCTL_ERROR),
                str(exc),
                errno=number,
                errno_name=name,
            )
        except BaseException as exc:
            duration = self.clock() - started
            self.record = ShutdownTransferRecord(
                None,
                False,
                f"{type(exc).__name__}: {exc}",
                duration,
                exception_type=type(exc).__name__,
            )
            phases.append(
                PhaseResult(
                    Phase.OWNED_SEQUENCE,
                    False,
                    f"{type(exc).__name__}: {exc}",
                    duration,
                )
            )
            self.outcome = (
                ShutdownStatus.INTERRUPTED
                if isinstance(exc, (SignalReceived, KeyboardInterrupt))
                else ShutdownStatus.UNEXPECTED_EXCEPTION
            )
            if isinstance(exc, SignalReceived):
                raise
            if isinstance(exc, KeyboardInterrupt):
                raise SignalReceived(signal.SIGINT, "blue-light shutdown") from exc
            return _Primary(
                OwnershipStatus.IOCTL_ERROR,
                f"USBDEVFS_CONTROL raised {type(exc).__name__}: {exc}",
            )
        duration = self.clock() - started
        exact = transferred == RESULT_LENGTH
        self.record = ShutdownTransferRecord(
            transferred,
            exact,
            f"device accepted {transferred} of {RESULT_LENGTH} bytes",
            duration,
        )
        phases.append(
            PhaseResult(
                Phase.OWNED_SEQUENCE,
                exact and transferred >= 0,
                f"USBDEVFS_CONTROL accepted {transferred} of {RESULT_LENGTH} bytes",
                duration,
            )
        )
        if transferred < 0:
            self.outcome = ShutdownStatus.IOCTL_ERROR
            return _Primary(OwnershipStatus.IOCTL_ERROR, f"USBDEVFS_CONTROL returned {transferred}")
        if not exact:
            self.outcome = (
                ShutdownStatus.SHORT_WRITE
                if transferred < RESULT_LENGTH
                else ShutdownStatus.OVERLONG_WRITE
            )
            return _Primary(
                OwnershipStatus.SHORT_RESPONSE,
                f"device accepted {transferred} of {RESULT_LENGTH} bytes",
            )
        self.outcome = ShutdownStatus.COMMAND_ACCEPTED
        # Deliberately says only what happened on the wire.  The device taking
        # the bytes is not evidence that the emitter changed state, and on this
        # unit it demonstrably is not.
        return _Primary(
            OwnershipStatus.OK,
            f"the device accepted all {RESULT_LENGTH} bytes of the fixed off command; "
            "emitter state not observed",
        )

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        return ProtocolCleanupRecord.not_needed("SET_CUR has no additional protocol clear")


@dataclass(frozen=True, slots=True)
class IlluminationShutdownOutcome:
    """Everything the one write is allowed to say about itself.

    Read every field as a statement about a USB transfer.  None of them is a
    statement about light: :attr:`limitation` carries that permanently and
    :meth:`as_dict` emits it next to the status so no JSON consumer can read
    the status without it.
    """

    required: bool
    attempted: bool
    status: ShutdownStatus
    detail: str
    target: TargetSnapshot | None = None
    phases: tuple[PhaseResult, ...] = ()
    transfer: ShutdownTransferRecord | None = None
    cleanup: CleanupRecord = field(
        default_factory=lambda: CleanupRecord.not_needed("the off write was not attempted")
    )
    protocol_cleanup: ProtocolCleanupRecord = field(
        default_factory=lambda: ProtocolCleanupRecord.not_needed("the off write was not attempted")
    )
    errno: int | None = None
    errno_name: str | None = None
    deferred_signals: tuple[int, ...] = ()

    @property
    def accepted(self) -> bool:
        """The device took all sixty bytes.  Nothing optical is implied."""
        return (
            self.status is ShutdownStatus.COMMAND_ACCEPTED
            and self.transfer is not None
            and self.transfer.ok
            and self.transfer.transferred == RESULT_LENGTH
        )

    @property
    def limitation(self) -> str:
        """Permanent, un-constructable-away: see :data:`EMITTER_LIMITATION`."""
        return EMITTER_LIMITATION

    @property
    def device_restored(self) -> bool:
        """Interface 0 was handed back to the kernel driver and left clean."""
        return self.cleanup.ok and self.protocol_cleanup.ok

    @property
    def _evidenced(self) -> bool:
        """A status may never outrun the record that is supposed to support it.

        ``COMMAND_ACCEPTED`` without a matching exact-length transfer record,
        or any discharged status with no transfer record at all, is a
        contradiction: nothing witnesses that the write left the host.
        """
        if self.status is ShutdownStatus.COMMAND_ACCEPTED:
            return self.accepted
        return self.transfer is not None

    @property
    def _inert_not_required(self) -> bool:
        return (
            not self.required
            and not self.attempted
            and self.status is ShutdownStatus.NOT_REQUIRED
            and self.transfer is None
            and not self.cleanup.attempted
            and self.cleanup.ok
            and not self.protocol_cleanup.attempted
            and self.protocol_cleanup.ok
        )

    @property
    def succeeded(self) -> bool:
        """The owed attempt was discharged and the device was left restored.

        Never an optical claim, and now not a claim about the command's own
        verdict either.  Until 2026-08-24 this also required :attr:`accepted`,
        so a capture's exit code turned on whether the device took a command
        we now know to be inert.  That gate was wrong in both directions: a
        refused inert command failed an otherwise perfect capture, which is
        noise, and an accepted one contributed a green run that a reader could
        mistake for "the light went out", which is worse than noise.

        What still gates is what is still true and still matters.  The attempt
        has to have been made and witnessed by a transfer record -- the owner's
        durable safety constraint is that the write is attempted and evidenced,
        and that constraint is untouched here.  Interface 0 has to have been
        handed back to the kernel driver, because a device left claimed is a
        real, actionable failure.  The transfer's own verdict is reported in
        full and scored not at all.
        """
        if not self.required:
            return self._inert_not_required
        return (
            self.attempted
            and self.status in DISCHARGED_STATUSES
            and self._evidenced
            and self.device_restored
        )

    @property
    def exit_code(self) -> int:
        """Non-zero only for an undischarged attempt or an unrestored device.

        The short/overlong branches below survive for outcomes that carry no
        transfer record: with a record they are discharged and score zero, so
        the code here answers "we never saw the write leave", not "the command
        was refused".
        """
        if self.succeeded:
            return 0
        if not self.cleanup.ok:
            return CLEANUP_FAILED_EXIT
        if not self.protocol_cleanup.ok:
            return CLEANUP_FAILED_EXIT
        if self.status is ShutdownStatus.SHORT_WRITE:
            return 7
        if self.status is ShutdownStatus.OVERLONG_WRITE:
            return 7
        if self.status is ShutdownStatus.INTERRUPTED:
            return 13
        return OWNERSHIP_EXIT_CODES.get(
            OwnershipStatus.DEVICE_REPLACED
            if self.status is ShutdownStatus.DEVICE_REPLACED
            else OwnershipStatus.IOCTL_ERROR,
            5,
        )

    @property
    def guidance(self) -> str | None:
        """What the reader must do -- and never a licence to stop worrying.

        Present for every run that actually streamed depth, success included:
        a discharged, restored finalizer still leaves the emitter's state
        unknown, so there is always something the reader must not conclude.
        Only the consistent never-streamed case has nothing to warn about.
        """
        if not self.required and self.succeeded:
            return None
        if self.succeeded:
            return f"DO NOT CLAIM THE BLUE LIGHT IS OFF. {EMITTER_LIMITATION}"
        return (
            "DO NOT CLAIM THE BLUE LIGHT IS OFF. The bounded off write was not "
            "carried through, or the USB interface was not fully restored: re-run it "
            "through this tool, or physically unplug the scanner before handling it. "
            f"{EMITTER_LIMITATION}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "required": self.required,
            "attempted": self.attempted,
            "status": str(self.status),
            # Named for what it measures.  A machine reader that filters on
            # "succeeded" alone must still meet "emitter_state" next to it.
            "command_accepted": self.accepted,
            "device_restored": self.device_restored,
            "emitter_state": "unknown",
            "limitation": self.limitation,
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "target": None if self.target is None else self.target.as_dict(),
            "phases": [phase.as_dict() for phase in self.phases],
            "transfer": None if self.transfer is None else self.transfer.as_dict(),
            "cleanup": self.cleanup.as_dict(),
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "recovery_guidance": self.guidance,
            "errno": self.errno,
            "errno_name": self.errno_name,
            "deferred_signals": list(self.deferred_signals),
        }


def not_required(detail: str = "depth STREAMON was never attempted") -> IlluminationShutdownOutcome:
    return IlluminationShutdownOutcome(False, False, ShutdownStatus.NOT_REQUIRED, detail)


class UsbfsIlluminationShutdownBackend(InterfaceZeroLifecycle):
    """The one fixed laser-off transfer inside the shared interface-0 lifecycle.

    Issues exactly one write and reports exactly what the write did.  It does
    not, and cannot, observe the emitter; see :data:`EMITTER_LIMITATION`.
    """

    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return _ShutdownOperation(self.ioctl_fn, self.clock)

    def _snapshot_request(self) -> ControlRequest:
        return REQUEST

    def run(self, serial: str) -> IlluminationShutdownOutcome:
        ensure_shutdown_allowed()
        # The guard refuses a device the operator did not NAME. If they named
        # none, there is nothing to refuse against and the caller's own
        # resolution stands - so this asks configuration only, and never the
        # hardware. A guard that enumerates devices would both be slow and
        # ignore the roots a caller injected, which is how it was wrong before.
        authorized = configured_serial()
        if authorized is not None and serial != authorized:
            return IlluminationShutdownOutcome(
                True,
                False,
                ShutdownStatus.REFUSED,
                f"the off write is authorized for {authorized!r} only",
            )
        phases: list[PhaseResult] = []
        preflight = self._preflight(serial, "the fixed off command is ready")
        if preflight.status is not OwnershipStatus.NOT_ATTEMPTED or preflight.target is None:
            return self._from_preflight(preflight)
        primary, window = self._perform_lifecycle(preflight.target, phases)
        operation = window.operation
        if not isinstance(operation, _ShutdownOperation):
            raise IlluminationShutdownNotAllowed("unexpected shutdown operation")
        status = _STATUS_MAP.get(primary.status, ShutdownStatus.IOCTL_ERROR)
        if operation.outcome in (
            ShutdownStatus.SHORT_WRITE,
            ShutdownStatus.OVERLONG_WRITE,
            ShutdownStatus.UNEXPECTED_EXCEPTION,
        ):
            status = operation.outcome
        elif (
            operation.outcome is ShutdownStatus.COMMAND_ACCEPTED
            and primary.status is OwnershipStatus.OK
        ):
            status = ShutdownStatus.COMMAND_ACCEPTED
        return IlluminationShutdownOutcome(
            True,
            True,
            status,
            primary.detail,
            preflight.target,
            tuple(phases),
            operation.record,
            window.cleanup,
            window.protocol,
            primary.errno,
            primary.errno_name,
        )

    def _from_preflight(self, preflight: Preflight) -> IlluminationShutdownOutcome:
        status = _STATUS_MAP.get(preflight.status, ShutdownStatus.IOCTL_ERROR)
        return IlluminationShutdownOutcome(
            True,
            False,
            status,
            preflight.detail,
            preflight.target,
            preflight.phases,
            None,
            CleanupRecord.not_needed(preflight.cleanup_note),
            ProtocolCleanupRecord.not_needed("the off write was not reached"),
        )
