"""The interface-0 ownership lifecycle, and the one read that first used it.

``scanner3d.control`` established a measured fact: a usbfs class request
addressed to interface 0 is refused with ``EBUSY`` while ``uvcvideo`` owns that
interface, because the kernel's implicit claim (``checkintf`` in
``drivers/usb/core/devio.c``) cannot succeed against a bound driver. This module
exists to clear exactly that one obstacle, once, and see what is behind it.

It is deliberately *not* a general ownership interface. Two things live here,
and the split between them is the point:

:class:`InterfaceZeroLifecycle` owns the *ownership* and nothing else. Its whole
repertoire is fixed at import time:

1. confirm the driver bound to interface 0 is exactly ``uvcvideo`` - from sysfs
   before anything is opened, and again from ``USBDEVFS_GETDRIVER`` on the
   opened descriptor immediately before the detach;
2. ``USBDEVFS_IOCTL``/``USBDEVFS_DISCONNECT`` on interface 0;
3. ``USBDEVFS_CLAIMINTERFACE`` on interface 0;
4. one :class:`OwnedOperation`, supplied by a subclass, never by an argument;
5. that operation's own protocol undo, attempted once if it may have written
   anything;
6. ``USBDEVFS_RELEASEINTERFACE`` on interface 0;
7. ``USBDEVFS_IOCTL``/``USBDEVFS_CONNECT`` on interface 0.

No argument varies the interface, the expected driver, the timeout, or the
order. Interface 0 is the only value :func:`pack_interface`,
:func:`pack_getdriver` and :func:`pack_driver_ioctl` will encode. The lifecycle
itself contains no setup packet at all: what goes on the wire in step 4 belongs
to the operation, along with the closed whitelist that bounds it.

:class:`UsbfsOwnershipBackend` is that lifecycle carrying the one operation this
module defines: :data:`scanner3d.control.FIRMWARE_ALGORITHM_VERSION_READ`, a
single device-to-host ``GET_CUR``, unchanged. It writes nothing, so its protocol
undo is ``not_needed`` and step 5 records no phase for it. A test asserts the
exact set of usbfs command numbers this module knows, and that the only setup
packet it names is that read.

Steps 6 and 7 run in a ``finally`` from the moment step 2 is *attempted*, and
step 6 runs whenever the claim may be held: on success, on a short or unexpected
payload, on a timeout, on any ioctl failure, on an unexpected Python exception,
and on SIGINT or SIGTERM - including a signal that surfaces from the disconnect
ioctl itself, or between its return and the claim. Kernel completion is
ambiguous in that window, so the reconnect is attempted rather than assumed
unnecessary. The one thing that undoes nothing is a disconnect that failed
cleanly with an errno: the kernel then took nothing away, and the typed
``detach_failed`` result stands.

Step 5 is owed on the same conservative rule, one step in: from the instant
before the operation is entered, not from its first completed transfer. It runs
before the driver handback because it needs the claim, and the handback runs
from a ``finally`` around it, so an undo that fails - or escapes wholesale -
cannot cost the owner their driver. The two are reported separately and neither
may be presented as success: a failed handback forces
:data:`CLEANUP_FAILED_EXIT`, a failed protocol undo
:data:`PROTOCOL_CLEANUP_FAILED_EXIT`.

Steps 5 to 7 are themselves shielded: cleanup takes SIGINT and SIGTERM over
before it decides whether it owes anything, absorbs whatever arrives while it
runs, and reports it in the cleanup note. A signal that beats the shield into
place has changed nothing yet, so the outer ``finally`` still makes the one
attempt; a signal that follows it cannot abort the undo, the release or the
reconnect half-way. There is no instant at which the run is marked cleaned
without having attempted cleanup, and none at which it attempts cleanup twice.

Interface 0 therefore stays detached only when the runtime cannot act at all -
SIGKILL, or a thread where the kernel refuses signal handlers - or when a
cleanup ioctl itself fails. The latter is reported, never hidden: a failed
cleanup forces a nonzero exit and prints the rebind/replug recovery the owner
must perform. Release always precedes reconnect, because the kernel answers
``USBDEVFS_CONNECT`` with ``EBUSY`` while usbfs still holds the claim.

Nothing here retries. Each phase is attempted at most once.

What no lifecycle built on this class can do: no interface other than 0, no URB
submission, no ``USBDEVFS_RESET``, no ``SET_CONFIGURATION``, no
``SET_INTERFACE``, no HID transfer, no streaming. A test asserts the exact set
of usbfs command numbers this module knows.
"""

from __future__ import annotations

import contextlib
import ctypes
import enum
import errno as errno_mod
import fcntl
import os
import signal
import struct
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import FrameType, MappingProxyType
from typing import Any, Final, Protocol

from scanner3d.control import (
    FIRMWARE_ALGORITHM_VERSION_READ,
    USBDEVFS_CONTROL,
    Clock,
    Closer,
    ControlRequest,
    IoctlFn,
    Opener,
    TargetSnapshot,
    VersionPayload,
    _ioc,
    ensure_allowed,
    identity_drift,
    pack_ctrltransfer,
    parse_version_payload,
    resolve_target,
)
from scanner3d.discovery import (
    DEFAULT_DEV_ROOT,
    DEFAULT_SYSFS_ROOT,
    EXPECTED_MANUFACTURER,
    ScannerDevice,
)
from scanner3d.errors import DeviceAbsentError, InterfaceOwnershipNotAllowed

#: The one interface this module may ever name. Not a parameter anywhere.
OWNED_INTERFACE: Final = 0

#: The one driver this module will take interface 0 from, and give it back to.
EXPECTED_INTERFACE_DRIVER: Final = "uvcvideo"

#: The one request this module can put on the wire, re-used verbatim from the
#: existing whitelist. It is not an argument to anything here.
OWNED_REQUEST: Final[ControlRequest] = FIRMWARE_ALGORITHM_VERSION_READ

# --- usbfs ABI ---------------------------------------------------------------

# Direction bits of the kernel's ``_IOC`` encoding. The encoder itself is
# imported from scanner3d.control so the two modules cannot drift apart.
_IOC_NONE: Final = 0
_IOC_WRITE: Final = 1
_IOC_READ: Final = 2

_U: Final = ord("U")

# struct usbdevfs_ioctl { int ifno; int ioctl_code; void *data; }
_USBDEVFS_IOCTL_FORMAT: Final = "@iiP"
_USBDEVFS_IOCTL_SIZE: Final = struct.calcsize(_USBDEVFS_IOCTL_FORMAT)

# struct usbdevfs_getdriver { unsigned int interface; char driver[256]; }
_MAX_DRIVER_NAME: Final = 255
_GETDRIVER_FORMAT: Final = f"@I{_MAX_DRIVER_NAME + 1}s"
_GETDRIVER_SIZE: Final = struct.calcsize(_GETDRIVER_FORMAT)

#: ``_IOW('U', 8, struct usbdevfs_getdriver)`` - names the driver currently
#: bound to an interface and changes nothing.
USBDEVFS_GETDRIVER: Final = _ioc(_IOC_WRITE, _U, 8, _GETDRIVER_SIZE)

#: ``_IOR('U', 15, unsigned int)``
USBDEVFS_CLAIMINTERFACE: Final = _ioc(_IOC_READ, _U, 15, 4)

#: ``_IOR('U', 16, unsigned int)``
USBDEVFS_RELEASEINTERFACE: Final = _ioc(_IOC_READ, _U, 16, 4)

#: ``_IOWR('U', 18, struct usbdevfs_ioctl)`` - the envelope that carries one
#: driver-management command to one interface.
USBDEVFS_IOCTL: Final = _ioc(_IOC_READ | _IOC_WRITE, _U, 18, _USBDEVFS_IOCTL_SIZE)

#: ``_IO('U', 22)``. Never passed to ``ioctl(2)`` directly: it is the
#: ``ioctl_code`` field of a ``struct usbdevfs_ioctl`` sent via
#: :data:`USBDEVFS_IOCTL`, which makes the kernel release the bound driver from
#: the named interface.
USBDEVFS_DISCONNECT: Final = _ioc(_IOC_NONE, _U, 22, 0)

#: ``_IO('U', 23)``. Same envelope; asks the kernel to let drivers bind to the
#: named interface again.
USBDEVFS_CONNECT: Final = _ioc(_IOC_NONE, _U, 23, 0)


def _ensure_owned_interface(number: int) -> None:
    if number != OWNED_INTERFACE:
        raise InterfaceOwnershipNotAllowed(
            f"interface {number} is out of scope; this module may only name "
            f"interface {OWNED_INTERFACE}"
        )


def pack_interface(number: int) -> bytearray:
    """Pack the ``unsigned int`` argument of claim/release.

    Refuses any interface but :data:`OWNED_INTERFACE`, so no caller can widen
    the lifecycle by passing a different number.
    """
    _ensure_owned_interface(number)
    return bytearray(struct.pack("@I", number))


def pack_driver_ioctl(number: int, code: int) -> bytearray:
    """Pack ``struct usbdevfs_ioctl`` for a detach or a reconnect.

    ``data`` is always ``NULL``: neither command carries a payload. Refuses any
    interface but :data:`OWNED_INTERFACE` and any code but disconnect/connect.
    """
    _ensure_owned_interface(number)
    if code not in (USBDEVFS_DISCONNECT, USBDEVFS_CONNECT):
        raise InterfaceOwnershipNotAllowed(
            f"usbfs driver command 0x{code:x} is neither disconnect "
            f"(0x{USBDEVFS_DISCONNECT:x}) nor connect (0x{USBDEVFS_CONNECT:x})"
        )
    return bytearray(struct.pack(_USBDEVFS_IOCTL_FORMAT, number, code, 0))


def pack_getdriver(number: int) -> bytearray:
    """Pack ``struct usbdevfs_getdriver`` for the on-descriptor driver check."""
    _ensure_owned_interface(number)
    return bytearray(struct.pack(_GETDRIVER_FORMAT, number, b""))


def unpack_getdriver(packed: bytes | bytearray) -> str:
    """Read the driver name back out of a filled ``struct usbdevfs_getdriver``."""
    name = bytes(struct.unpack(_GETDRIVER_FORMAT, bytes(packed))[1])
    return name.split(b"\x00", 1)[0].decode("ascii", errors="replace")


# --- outcomes ----------------------------------------------------------------


class Phase(enum.StrEnum):
    """Every step a lifecycle can record, in the only order they can occur.

    :data:`LIFECYCLE` is the six-step shape of the one *read* lifecycle.
    :data:`Phase.OWNED_SEQUENCE` and :data:`Phase.PROTOCOL_CLEANUP` are the two
    extra steps an operation that writes device state records instead of, and
    after, :data:`Phase.CONTROL_READ`; a read run never emits either.
    """

    CONFIRM_DRIVER = "confirm_driver"
    DISCONNECT = "disconnect"
    CLAIM = "claim"
    CONTROL_READ = "control_read"
    OWNED_SEQUENCE = "owned_sequence"
    PROTOCOL_CLEANUP = "protocol_cleanup"
    RELEASE = "release"
    RECONNECT = "reconnect"
    CLOSE = "close"


#: The lifecycle, as a fixed sequence. Offline tests compare the phases a run
#: actually performed against this order.
LIFECYCLE: Final[tuple[Phase, ...]] = (
    Phase.CONFIRM_DRIVER,
    Phase.DISCONNECT,
    Phase.CLAIM,
    Phase.CONTROL_READ,
    Phase.RELEASE,
    Phase.RECONNECT,
)

#: The phases that undo the detach. Both are attempted once it has succeeded.
CLEANUP_PHASES: Final[tuple[Phase, ...]] = (Phase.RELEASE, Phase.RECONNECT)


class OwnershipStatus(enum.StrEnum):
    """The primary result of the run, cleanup reported separately."""

    OK = "ok"
    NOT_ATTEMPTED = "not_attempted"
    REFUSED = "refused"
    DEVICE_ABSENT = "device_absent"
    DEVICE_REPLACED = "device_replaced"
    PERMISSION_DENIED = "permission_denied"
    DETACH_FAILED = "detach_failed"
    CLAIM_FAILED = "claim_failed"
    TIMEOUT = "timeout"
    IOCTL_ERROR = "ioctl_error"
    SHORT_RESPONSE = "short_response"
    UNEXPECTED_CONTENT = "unexpected_content"
    INTERRUPTED = "interrupted"


#: Exit code per primary status. The codes shared with ``firmware-version``
#: keep the meanings :mod:`scanner3d.control` already gave them.
OWNERSHIP_EXIT_CODES: Final[Mapping[OwnershipStatus, int]] = MappingProxyType(
    {
        OwnershipStatus.OK: 0,
        OwnershipStatus.NOT_ATTEMPTED: 0,
        OwnershipStatus.DEVICE_ABSENT: 2,
        OwnershipStatus.PERMISSION_DENIED: 3,
        OwnershipStatus.TIMEOUT: 4,
        OwnershipStatus.IOCTL_ERROR: 5,
        OwnershipStatus.UNEXPECTED_CONTENT: 6,
        OwnershipStatus.SHORT_RESPONSE: 7,
        OwnershipStatus.DEVICE_REPLACED: 8,
        OwnershipStatus.REFUSED: 10,
        OwnershipStatus.DETACH_FAILED: 11,
        OwnershipStatus.CLAIM_FAILED: 12,
        OwnershipStatus.INTERRUPTED: 13,
    }
)

#: Overrides every other code when driver cleanup did not fully succeed, so a
#: run that read the payload but failed to give interface 0 back can never
#: exit 0. It outranks :data:`PROTOCOL_CLEANUP_FAILED_EXIT` because a driver
#: left detached is the more serious of the two.
CLEANUP_FAILED_EXIT: Final = 9

#: Overrides every primary status when the operation could not undo the device
#: state it created. Distinct from :data:`CLEANUP_FAILED_EXIT` because the two
#: failures need different recovery: this one is on the wire, that one is a
#: driver binding. Neither may ever be reported as success.
PROTOCOL_CLEANUP_FAILED_EXIT: Final = 14

_ERRNO_STATUS: Final[Mapping[int, OwnershipStatus]] = MappingProxyType(
    {
        errno_mod.EACCES: OwnershipStatus.PERMISSION_DENIED,
        errno_mod.EPERM: OwnershipStatus.PERMISSION_DENIED,
        errno_mod.ENODEV: OwnershipStatus.DEVICE_ABSENT,
        errno_mod.ENOENT: OwnershipStatus.DEVICE_ABSENT,
        errno_mod.ENXIO: OwnershipStatus.DEVICE_ABSENT,
        errno_mod.ESHUTDOWN: OwnershipStatus.DEVICE_ABSENT,
        errno_mod.ETIMEDOUT: OwnershipStatus.TIMEOUT,
    }
)


def _absorbed(signums: list[int]) -> str:
    """How the report names the signals cleanup refused to be aborted by."""
    names = ", ".join(dict.fromkeys(signal.Signals(num).name for num in signums))
    return f"{names} arrived during cleanup and was ignored so it could finish"


def recovery_guidance(usb_path: str) -> str:
    """What the owner must do by hand when cleanup failed."""
    return (
        f"interface {OWNED_INTERFACE} may still be detached from "
        f"{EXPECTED_INTERFACE_DRIVER}. Recover by rebinding it explicitly - as root, "
        f"`echo {usb_path}:1.{OWNED_INTERFACE} > "
        f"/sys/bus/usb/drivers/{EXPECTED_INTERFACE_DRIVER}/bind` - or by physically "
        "unplugging and replugging the scanner. Re-running this command will not "
        f"recover it: it refuses to act unless {EXPECTED_INTERFACE_DRIVER} already "
        f"owns interface {OWNED_INTERFACE}."
    )


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """One attempted step of the lifecycle and how it ended."""

    phase: Phase
    ok: bool
    detail: str
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": str(self.phase),
            "ok": self.ok,
            "detail": self.detail,
            "duration_s": self.duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


@dataclass(frozen=True, slots=True)
class CleanupRecord:
    """Whether the detach was undone, reported apart from the read's result."""

    attempted: bool
    phases: tuple[PhaseResult, ...]
    ok: bool
    note: str
    guidance: str | None = None

    @classmethod
    def not_needed(cls, note: str) -> CleanupRecord:
        """Nothing was detached, so there is nothing to give back."""
        return cls(attempted=False, phases=(), ok=True, note=note)

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "note": self.note,
            "phases": [p.as_dict() for p in self.phases],
            "recovery_guidance": self.guidance,
        }


@dataclass(frozen=True, slots=True)
class ProtocolCleanupRecord:
    """Whether the device-side state an operation created was undone.

    Deliberately a separate record from :class:`CleanupRecord`. The two undo
    different things, at different layers, in a fixed order: the protocol clear
    is a transfer on the wire and happens while the claim is still held; the
    driver handback is a pair of ioctls and happens after it. Either can fail
    without excusing the other, and neither failure may be reported as success.

    An operation that creates no device state - the one read does not - reports
    :meth:`not_needed`, which is ``ok`` and contributes no phase.
    """

    attempted: bool
    ok: bool
    note: str
    duration_s: float | None = None
    guidance: str | None = None

    @classmethod
    def not_needed(cls, note: str) -> ProtocolCleanupRecord:
        """The operation left nothing behind, so there is nothing to undo."""
        return cls(attempted=False, ok=True, note=note)

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "note": self.note,
            "duration_s": self.duration_s,
            "recovery_guidance": self.guidance,
        }


@dataclass(frozen=True, slots=True)
class OwnershipOutcome:
    """The complete, typed result of the lifecycle."""

    status: OwnershipStatus
    request: ControlRequest
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    cleanup: CleanupRecord
    payload: VersionPayload | None
    transfer_duration_s: float | None
    total_duration_s: float | None
    detail: str
    errno: int | None = None
    errno_name: str | None = None
    protocol_cleanup: ProtocolCleanupRecord = field(
        default_factory=lambda: ProtocolCleanupRecord.not_needed(
            "a read leaves no device state to undo"
        )
    )

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
        """True only when the read matched *and* everything owed was undone."""
        return self.status is OwnershipStatus.OK and self.cleanup.ok and self.protocol_cleanup.ok

    def phase(self, phase: Phase) -> PhaseResult | None:
        """The last result recorded for ``phase``, or ``None`` if it never ran."""
        for result in reversed(self.phases):
            if result.phase is phase:
                return result
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "temporary-interface-ownership",
            "status": str(self.status),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "owned_interface": OWNED_INTERFACE,
            "expected_interface_driver": EXPECTED_INTERFACE_DRIVER,
            "request": self.request.as_dict(),
            "target": self.target.as_dict() if self.target is not None else None,
            "phases": [p.as_dict() for p in self.phases],
            "cleanup": self.cleanup.as_dict(),
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "response": self.payload.as_dict() if self.payload is not None else None,
            "transfer_duration_s": self.transfer_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


# --- preconditions -----------------------------------------------------------


def refusal_reason(device: ScannerDevice, serial: str) -> str | None:
    """Why this device must not be touched, or ``None`` if it may be.

    Read entirely from sysfs, before any node is opened. Unknown, absent,
    already-unbound and changed states are all refusals; only the exact
    expected MINI with interface 0 bound to exactly ``uvcvideo`` passes.
    """
    if not device.identity_matches:
        return (
            f"{device.usb_path} is {device.vendor_id:04x}:{device.product_id:04x}, "
            "not the expected 2207:110c MINI"
        )
    if device.serial != serial:
        return f"{device.usb_path} reports serial {device.serial!r}, not the requested {serial!r}"
    if device.manufacturer != EXPECTED_MANUFACTURER:
        return (
            f"{device.usb_path} reports manufacturer {device.manufacturer!r}, "
            f"not the expected {EXPECTED_MANUFACTURER!r}"
        )
    interface = next((i for i in device.interfaces if i.number == OWNED_INTERFACE), None)
    if interface is None:
        return f"{device.usb_path} exposes no interface {OWNED_INTERFACE}"
    if interface.driver is None:
        return (
            f"interface {OWNED_INTERFACE} is already unbound; this command only takes the "
            f"interface from {EXPECTED_INTERFACE_DRIVER} and gives it back, it does not "
            "adopt an interface no driver owns"
        )
    if interface.driver != EXPECTED_INTERFACE_DRIVER:
        return (
            f"interface {OWNED_INTERFACE} is owned by {interface.driver!r}, "
            f"not {EXPECTED_INTERFACE_DRIVER!r}"
        )
    return None


# --- signal handling ---------------------------------------------------------


#: What a signal caught inside the interface-0 window is said to have
#: interrupted. It says "may have been detached" because the window opens at
#: the *attempt* to disconnect: a handler runs between bytecodes, so an
#: exception surfacing from that ioctl leaves it unknown whether the kernel
#: completed it. Other lifecycles that reuse this guard - the V4L2 capture in
#: :mod:`scanner3d.v4l2_capture` is the first - pass their own wording, because
#: a capture that never touches a driver binding must not claim it may have.
OWNERSHIP_WINDOW: Final = f"interface {OWNED_INTERFACE} may have been detached"


class SignalReceived(BaseException):
    """SIGINT or SIGTERM arrived inside a window that owes the kernel cleanup.

    A ``BaseException`` on purpose: it must unwind through the ``finally`` that
    performs that cleanup without being swallowed by an ``except Exception``.
    """

    def __init__(self, signum: int, context: str = OWNERSHIP_WINDOW) -> None:
        self.signum = signum
        self.context = context
        super().__init__(f"{signal.Signals(signum).name} received while {context}")


SignalGuard = Callable[[], AbstractContextManager[None]]

#: A context manager that absorbs SIGINT/SIGTERM and reports which ones it ate.
TerminationShield = Callable[[], AbstractContextManager["list[int]"]]

_TERMINATION_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM)


def _install(handler: Any) -> list[tuple[signal.Signals, Any]]:
    """Point SIGINT and SIGTERM at ``handler``, returning what they had before.

    Whatever the kernel refuses - off the main thread it refuses everything -
    is left alone, so a caller degrades to the plain ``finally`` guarantee
    rather than to something weaker.
    """
    previous: list[tuple[signal.Signals, Any]] = []
    for signum in _TERMINATION_SIGNALS:
        with contextlib.suppress(ValueError, OSError):
            previous.append((signum, signal.signal(signum, handler)))
    return previous


def _restore(previous: list[tuple[signal.Signals, Any]]) -> None:
    for signum, original in previous:
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signum, original)


@contextlib.contextmanager
def cleanup_on_termination(context: str = OWNERSHIP_WINDOW) -> Iterator[None]:
    """Turn the first SIGINT/SIGTERM into an exception so cleanup still runs.

    SIGTERM's default disposition kills the process outright, which would leave
    interface 0 detached; raising instead lets the ``finally`` reconnect it.
    Only the *first* signal raises - a later one is dropped rather than allowed
    to abort whatever the first one started. Previous handlers are restored on
    the way out. Off the main thread the kernel forbids installing handlers at
    all, and this degrades to a no-op, leaving the plain ``finally`` guarantee
    and nothing weaker.

    ``context`` is only how the resulting exception describes what was
    interrupted; it changes no behaviour, and exists so a lifecycle that never
    detaches a driver does not report that one may be detached.

    This converts a signal into an unwind; it does not protect the unwind's
    destination. Cleanup itself is covered by :func:`defer_termination`, which
    takes the handlers over for the duration of the release and the reconnect.
    """
    previous: list[tuple[signal.Signals, Any]] = []
    fired = False

    def handler(signum: int, frame: FrameType | None) -> None:
        nonlocal fired
        if fired:
            return
        fired = True
        raise SignalReceived(signum, context)

    try:
        previous = _install(handler)
        yield
    finally:
        _restore(previous)


@contextlib.contextmanager
def defer_termination() -> Iterator[list[int]]:
    """Absorb SIGINT/SIGTERM for the duration of the block, recording them.

    The critical section is cleanup. Once the release and the reconnect have
    begun they must finish: a signal that unwound out of them would strand
    interface 0 detached, which is the exact outcome this module exists to
    avoid. So this takes the handlers over from :func:`cleanup_on_termination`
    - and from the default disposition, which would kill the process outright -
    drops whatever arrives into the yielded list, and restores the previous
    handlers afterwards.

    Installing it *before* the caller decides whether cleanup is owed costs
    nothing: a signal arriving before the takeover completes still raises out
    of the caller having changed nothing, so the outer ``finally`` reaches
    cleanup with the work still to do. A signal arriving after it lands in the
    list, where the caller can report it. Handlers the kernel will not let this
    install (off the main thread) simply are not installed.
    """
    received: list[int] = []
    previous: list[tuple[signal.Signals, Any]] = []

    def absorb(signum: int, frame: FrameType | None) -> None:
        received.append(signum)

    try:
        previous = _install(absorb)
        yield received
    finally:
        _restore(previous)


# --- the hardware path -------------------------------------------------------


def _default_opener(path: str) -> int:
    # O_RDWR: usbfs rejects claim, detach and control alike on a descriptor
    # without FMODE_WRITE. Opening the node transfers nothing.
    return os.open(path, os.O_RDWR | os.O_CLOEXEC)


def _errno_of(exc: OSError) -> tuple[int | None, str | None]:
    if exc.errno is None:
        return None, None
    return exc.errno, errno_mod.errorcode.get(exc.errno, f"errno {exc.errno}")


@dataclass(slots=True)
class _Primary:
    """The primary result being assembled, before cleanup is folded in."""

    status: OwnershipStatus
    detail: str
    payload: VersionPayload | None = None
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None


class OwnedOperation(Protocol):
    """The one thing a lifecycle does while it holds interface 0.

    The lifecycle below owns the *ownership*: confirming the driver, taking
    interface 0, giving it back, and doing so exactly once under a shield no
    signal gets through. It deliberately owns no setup packet. What goes on the
    wire inside that window is decided entirely by the operation, which is
    built fresh for every run - so it may accumulate per-run results - and is
    supplied by a backend, never by an argument a caller can vary.

    :meth:`protocol_cleanup` is the operation's own half of the undo. The
    lifecycle calls it exactly once, inside the termination shield and *before*
    the driver handback, whenever :meth:`perform` was entered at all. An
    operation that created no device state says so; an operation that may have
    created some must attempt the undo, and must report a failure rather than
    swallow it.
    """

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        """Do the operation's own transfers on the claimed descriptor."""
        ...

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        """Undo whatever device state :meth:`perform` may have created."""
        ...


@dataclass(frozen=True, slots=True)
class Preflight:
    """What the read-only precondition check found. It opens nothing."""

    status: OwnershipStatus
    detail: str
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    cleanup_note: str


@dataclass(slots=True)
class _Window:
    """What the ownership window owes the device, and whether it has been paid.

    ``claimed`` is conservative: it is true from before the claim ioctl until
    that ioctl returns and says otherwise. ``started`` is conservative in the
    same way for the operation: it is raised before :meth:`OwnedOperation.perform`
    is entered, so an interruption anywhere inside it still owes the protocol
    undo. ``cleaned`` says the one cleanup attempt has been made, so the two
    ``finally`` blocks that call it cannot turn into a retry; it is only ever
    set inside the termination shield, which is what keeps "marked" and
    "attempted" from coming apart.
    """

    operation: OwnedOperation
    claimed: bool = False
    started: bool = False
    cleaned: bool = False
    cleanup: CleanupRecord = field(
        default_factory=lambda: CleanupRecord.not_needed("nothing was detached")
    )
    protocol: ProtocolCleanupRecord = field(
        default_factory=lambda: ProtocolCleanupRecord.not_needed(
            "no owned operation began, so nothing was written"
        )
    )


@dataclass(frozen=True, slots=True)
class InterfaceZeroLifecycle:
    """The fixed detach / claim / operate / undo / release / reconnect lifecycle.

    This class is the whole ownership half of the safety argument, and it is
    shared rather than copied so there can only ever be one implementation of
    it to reason about. It names interface 0 and no other, knows disconnect and
    connect and no other driver command, attempts every phase at most once, and
    hands interface 0 back under a shield that neither SIGINT nor SIGTERM can
    abort - whatever the operation in the middle did or failed to do.

    It contains no setup packet of its own. Subclasses supply an
    :class:`OwnedOperation` and their own outcome type; the transfers are the
    operation's, and the closed whitelist that bounds them lives with it.

    Every syscall is injectable, so each failure branch and the cleanup
    ordering are exercised offline without a device.
    """

    opener: Opener = _default_opener
    ioctl_fn: IoctlFn = fcntl.ioctl
    closer: Closer = os.close
    clock: Clock = time.perf_counter
    sysfs_root: Path = DEFAULT_SYSFS_ROOT
    dev_root: Path = DEFAULT_DEV_ROOT
    signal_guard: SignalGuard = cleanup_on_termination
    termination_shield: TerminationShield = defer_termination

    # -- hooks ---------------------------------------------------------------

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        """Build this run's operation. Fresh each time; never a parameter.

        It is handed the target only so it can name the device in its own
        recovery guidance; it cannot widen what the lifecycle will touch.
        """
        raise NotImplementedError

    def _snapshot_request(self) -> ControlRequest:
        """The request whose ``wIndex`` names the interface to snapshot."""
        raise NotImplementedError

    # -- preconditions, before anything is opened ----------------------------

    def _preflight(self, serial: str | None, dry_run_detail: str) -> Preflight:
        """Resolve the target and check every precondition, opening nothing.

        ``serial`` is optional only so a dry run can describe the sole attached
        scanner. Every path that detaches a driver passes an explicit one, and
        with more than one device attached :func:`resolve_target` refuses to
        pick for the caller.
        """
        request = self._snapshot_request()
        try:
            snapshot, device = resolve_target(
                serial, request, sysfs_root=self.sysfs_root, dev_root=self.dev_root
            )
        except DeviceAbsentError as exc:
            return Preflight(
                status=OwnershipStatus.DEVICE_ABSENT,
                detail=str(exc),
                target=None,
                phases=(),
                cleanup_note="no device was resolved",
            )
        reason = refusal_reason(device, snapshot.serial if serial is None else serial)
        if reason is not None:
            return Preflight(
                status=OwnershipStatus.REFUSED,
                detail=f"refusing to detach a driver: {reason}",
                target=snapshot,
                phases=(PhaseResult(phase=Phase.CONFIRM_DRIVER, ok=False, detail=reason),),
                cleanup_note="nothing was detached",
            )
        return Preflight(
            status=OwnershipStatus.NOT_ATTEMPTED,
            detail=dry_run_detail,
            target=snapshot,
            phases=(
                PhaseResult(
                    phase=Phase.CONFIRM_DRIVER,
                    ok=True,
                    detail=(
                        f"sysfs reports interface {OWNED_INTERFACE} bound to "
                        f"{EXPECTED_INTERFACE_DRIVER}"
                    ),
                ),
            ),
            cleanup_note="nothing was detached",
        )

    # -- lifecycle -----------------------------------------------------------

    def _perform_lifecycle(
        self, snapshot: TargetSnapshot, phases: list[PhaseResult]
    ) -> tuple[_Primary, _Window]:
        """One open, the whole window, one close. Never retries anything."""
        window = _Window(operation=self._new_operation(snapshot))
        try:
            fd = self.opener(snapshot.node_path)
        except OSError as exc:
            number, name = _errno_of(exc)
            return (
                _Primary(
                    status=_ERRNO_STATUS.get(number or 0, OwnershipStatus.IOCTL_ERROR),
                    detail=f"cannot open {snapshot.node_path}: {exc}",
                    errno=number,
                    errno_name=name,
                ),
                window,
            )
        primary: _Primary | None = None
        try:
            primary = self._before_detach(fd, snapshot, phases)
            if primary is None:
                with self.signal_guard():
                    primary = self._owned_window(fd, snapshot, phases, window)
        finally:
            try:
                self.closer(fd)
            except BaseException as exc:
                close_phase = PhaseResult(
                    phase=Phase.CLOSE,
                    ok=False,
                    detail=f"usbfs close raised {type(exc).__name__}: {exc}",
                )
                window.cleanup = CleanupRecord(
                    attempted=True,
                    phases=(*window.cleanup.phases, close_phase),
                    ok=False,
                    note="interface 0 cleanup completed but usbfs close failed",
                    guidance=(
                        "the usbfs descriptor could not be closed cleanly; physically unplug "
                        "the scanner before relying on final-state evidence"
                    ),
                )
                if primary is None:
                    primary = _Primary(
                        OwnershipStatus.IOCTL_ERROR,
                        f"usbfs close raised {type(exc).__name__}: {exc}",
                    )
        assert primary is not None
        return primary, window

    def _before_detach(
        self, fd: int, snapshot: TargetSnapshot, phases: list[PhaseResult]
    ) -> _Primary | None:
        """Re-verify identity and ownership. Non-``None`` means stop here.

        Nothing has been detached at this point, so returning early leaves the
        device exactly as it was found.
        """
        drift = identity_drift(fd, snapshot)
        if drift is not None:
            return _Primary(OwnershipStatus.DEVICE_REPLACED, drift)
        confirmed = self._confirm_driver(fd)
        phases.append(confirmed)
        if confirmed.ok:
            return None
        return _Primary(
            status=_ERRNO_STATUS.get(confirmed.errno or 0, OwnershipStatus.REFUSED),
            detail=f"refusing to detach a driver: {confirmed.detail}",
            errno=confirmed.errno,
            errno_name=confirmed.errno_name,
        )

    def _owned_window(
        self,
        fd: int,
        snapshot: TargetSnapshot,
        phases: list[PhaseResult],
        window: _Window,
    ) -> _Primary:
        """Detach, claim, operate - then undo and reconnect, whatever happened.

        The obligation to give interface 0 back begins when the disconnect is
        *attempted*, not when it returns. Python runs a signal handler between
        bytecodes, so a SIGINT or SIGTERM around that ioctl surfaces with the
        kernel's state ambiguous - the driver may already be gone. The outer
        ``finally`` therefore reconnects after any exception raised from the
        disconnect onwards, the boundary between its successful return and the
        claim included. The single exception is a disconnect that failed cleanly
        with an errno: there the kernel reported that it did nothing, so there is
        nothing to undo and the typed ``detach_failed`` result stands.

        Cleanup - the operation's protocol undo first, then the driver handback -
        runs exactly once whichever ``finally`` reaches it first, under a shield
        that neither signal can get through, so no phase is retried and none is
        skipped.
        """
        primary = _Primary(OwnershipStatus.IOCTL_ERROR, "the disconnect never ran")
        try:
            try:
                detach = self._disconnect(fd)
            except BaseException as exc:
                # Not an errno: the ioctl neither returned nor reported a
                # failure, so record the detach as ambiguous before unwinding
                # into the reconnect below.
                phases.append(
                    PhaseResult(
                        phase=Phase.DISCONNECT,
                        ok=False,
                        detail=(
                            f"USBDEVFS_IOCTL/USBDEVFS_DISCONNECT({OWNED_INTERFACE}) was "
                            f"interrupted by {type(exc).__name__}: {exc}; the kernel may or "
                            "may not have completed it"
                        ),
                    )
                )
                raise
            phases.append(detach)
            if not detach.ok:
                # The record goes in before the flag: a signal between the two
                # would otherwise unwind to a ``finally`` that sees "cleaned"
                # and reports the placeholder note instead of this one.
                window.cleanup = CleanupRecord.not_needed("the detach failed, so nothing was taken")
                window.cleaned = True
                return _Primary(
                    status=_ERRNO_STATUS.get(detach.errno or 0, OwnershipStatus.DETACH_FAILED),
                    detail=f"could not detach {EXPECTED_INTERFACE_DRIVER}: {detach.detail}",
                    errno=detach.errno,
                    errno_name=detach.errno_name,
                )
            try:
                primary = self._claim_and_read(fd, phases, window)
            finally:
                self._give_interface_back(fd, snapshot, phases, window)
        except SignalReceived as sig:
            primary = _Primary(OwnershipStatus.INTERRUPTED, str(sig))
        except Exception as exc:
            primary = _Primary(
                OwnershipStatus.IOCTL_ERROR,
                f"unexpected {type(exc).__name__} in the ownership window: {exc}",
            )
        finally:
            # Reached when the disconnect ioctl itself raised something other
            # than an OSError, or when a signal landed between its return and
            # the claim. A no-op once cleanup has already run.
            self._give_interface_back(fd, snapshot, phases, window)
        return primary

    def _claim_and_read(self, fd: int, phases: list[PhaseResult], window: _Window) -> _Primary:
        """Claim interface 0 and, if that worked, run the one owned operation.

        ``window.claimed`` is raised *before* the claim ioctl and lowered only
        by that ioctl returning a failure, for the same reason the disconnect is
        treated as ambiguous: if an interruption unwinds out of the call the
        kernel may already have granted the claim, and usbfs answers
        ``USBDEVFS_CONNECT`` with ``EBUSY`` while it is held. So an interrupted
        claim is released before the reconnect; a claim that comes back with an
        errno took nothing and is not.

        ``window.started`` is raised on the same principle one step later: from
        the instant before the operation is entered, the protocol undo is owed.
        A claim that failed never reaches it, so nothing was written and none is.
        """
        window.claimed = True
        claim = self._claim(fd)
        window.claimed = claim.ok
        phases.append(claim)
        if not claim.ok:
            return _Primary(
                status=_ERRNO_STATUS.get(claim.errno or 0, OwnershipStatus.CLAIM_FAILED),
                detail=f"could not claim interface {OWNED_INTERFACE}: {claim.detail}",
                errno=claim.errno,
                errno_name=claim.errno_name,
            )
        window.started = True
        return window.operation.perform(fd, phases)

    # -- phases --------------------------------------------------------------

    def _confirm_driver(self, fd: int) -> PhaseResult:
        """Ask the kernel, on this descriptor, who owns interface 0 right now.

        The sysfs check in :meth:`_preflight` can go stale between that read and
        the detach; this one cannot.
        """
        packed = pack_getdriver(OWNED_INTERFACE)
        try:
            self.ioctl_fn(fd, USBDEVFS_GETDRIVER, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return PhaseResult(
                phase=Phase.CONFIRM_DRIVER,
                ok=False,
                detail=f"USBDEVFS_GETDRIVER({OWNED_INTERFACE}) failed: {exc}",
                errno=number,
                errno_name=name,
            )
        driver = unpack_getdriver(packed)
        if driver != EXPECTED_INTERFACE_DRIVER:
            return PhaseResult(
                phase=Phase.CONFIRM_DRIVER,
                ok=False,
                detail=(
                    f"the kernel reports interface {OWNED_INTERFACE} bound to {driver!r}, "
                    f"not {EXPECTED_INTERFACE_DRIVER!r}"
                ),
            )
        return PhaseResult(
            phase=Phase.CONFIRM_DRIVER,
            ok=True,
            detail=f"USBDEVFS_GETDRIVER({OWNED_INTERFACE}) reports {driver}",
        )

    def _disconnect(self, fd: int) -> PhaseResult:
        return self._driver_command(
            fd, USBDEVFS_DISCONNECT, Phase.DISCONNECT, f"detached {EXPECTED_INTERFACE_DRIVER}"
        )

    def _reconnect(self, fd: int) -> PhaseResult:
        return self._driver_command(
            fd, USBDEVFS_CONNECT, Phase.RECONNECT, "asked the kernel to rebind a driver"
        )

    def _driver_command(self, fd: int, code: int, phase: Phase, success: str) -> PhaseResult:
        packed = pack_driver_ioctl(OWNED_INTERFACE, code)
        label = "USBDEVFS_DISCONNECT" if code == USBDEVFS_DISCONNECT else "USBDEVFS_CONNECT"
        started = self.clock()
        try:
            self.ioctl_fn(fd, USBDEVFS_IOCTL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return PhaseResult(
                phase=phase,
                ok=False,
                detail=f"USBDEVFS_IOCTL/{label}({OWNED_INTERFACE}) failed: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        return PhaseResult(
            phase=phase,
            ok=True,
            detail=f"USBDEVFS_IOCTL/{label}({OWNED_INTERFACE}): {success}",
            duration_s=self.clock() - started,
        )

    def _claim(self, fd: int) -> PhaseResult:
        return self._interface_command(fd, USBDEVFS_CLAIMINTERFACE, Phase.CLAIM)

    def _release(self, fd: int) -> PhaseResult:
        return self._interface_command(fd, USBDEVFS_RELEASEINTERFACE, Phase.RELEASE)

    def _interface_command(self, fd: int, command: int, phase: Phase) -> PhaseResult:
        packed = pack_interface(OWNED_INTERFACE)
        label = "USBDEVFS_CLAIMINTERFACE" if phase is Phase.CLAIM else "USBDEVFS_RELEASEINTERFACE"
        started = self.clock()
        try:
            self.ioctl_fn(fd, command, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return PhaseResult(
                phase=phase,
                ok=False,
                detail=f"{label}({OWNED_INTERFACE}) failed: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        return PhaseResult(
            phase=phase,
            ok=True,
            detail=f"{label}({OWNED_INTERFACE}) succeeded",
            duration_s=self.clock() - started,
        )

    # -- cleanup -------------------------------------------------------------

    def _give_interface_back(
        self,
        fd: int,
        snapshot: TargetSnapshot,
        phases: list[PhaseResult],
        window: _Window,
    ) -> None:
        """Make the one cleanup attempt, from whichever ``finally`` gets here first.

        The shield is entered before anything else, including the "already
        done" test, and that ordering is the whole guarantee:

        * a SIGINT/SIGTERM arriving *before* the shield is installed raises out
          of here having changed nothing, so the outer ``finally`` calls this
          again and the one attempt still happens;
        * a signal arriving *after* it is absorbed, so the protocol undo, the
          release and the reconnect run to completion and are merely reported;
        * ``window.cleaned`` is set only once the attempt has actually been
          made, and it is what stops the second ``finally`` from repeating the
          ioctls.

        There is no window between marking and attempting for a signal to land
        in, because both happen with the handlers taken over.

        The two undos are independent and both are owed. The protocol clear
        goes first, because it needs the claim the driver handback is about to
        give up; the handback then runs from a ``finally``, so a protocol undo
        that fails - or escapes wholesale - can never cost the owner their
        driver. Neither result overwrites the other, and both start from a
        pessimistic record they have to earn their way out of.
        """
        with self.termination_shield() as deferred:
            if window.cleaned:
                return
            # In place before the first ioctl and replaced only by a real
            # result, so nothing that escapes below can leave the run
            # reporting a handback or an undo it never confirmed.
            window.cleanup = self._unfinished(snapshot)
            try:
                window.protocol = self._protocol_cleanup(fd, phases, window)
            finally:
                # Its own ``finally``, not the next statement: the handback is
                # the half that costs the owner their driver, so it must not be
                # reachable only when the protocol undo returns normally.
                try:
                    window.cleanup = self._cleanup(fd, snapshot, claimed=window.claimed)
                    phases.extend(window.cleanup.phases)
                finally:
                    window.cleaned = True
            if deferred:
                note = _absorbed(deferred)
                window.cleanup = replace(window.cleanup, note=f"{window.cleanup.note} ({note})")

    def _protocol_cleanup(
        self, fd: int, phases: list[PhaseResult], window: _Window
    ) -> ProtocolCleanupRecord:
        """Give the operation its one chance to undo what it wrote.

        Guarded against ``BaseException`` on purpose: this runs from a
        ``finally`` whose next statement is the driver handback, and an
        operation that raised here must not take that with it. A raise is
        recorded as a failed undo, which is what it is.
        """
        if not window.started:
            return ProtocolCleanupRecord.not_needed(
                "no owned operation began, so nothing was written"
            )
        try:
            record = window.operation.protocol_cleanup(fd)
        except BaseException as exc:
            record = ProtocolCleanupRecord(
                attempted=True,
                ok=False,
                note=f"protocol cleanup raised {type(exc).__name__}: {exc}",
                guidance=(
                    "device-side state the operation created may still be set; "
                    "power-cycle the scanner if a later run behaves unexpectedly"
                ),
            )
        if record.attempted:
            phases.append(
                PhaseResult(
                    phase=Phase.PROTOCOL_CLEANUP,
                    ok=record.ok,
                    detail=record.note,
                    duration_s=record.duration_s,
                )
            )
        return record

    @staticmethod
    def _unfinished(snapshot: TargetSnapshot) -> CleanupRecord:
        """The pessimistic record cleanup starts from, and must earn its way out of."""
        return CleanupRecord(
            attempted=True,
            phases=(),
            ok=False,
            note=f"interface {OWNED_INTERFACE} cleanup began but did not finish",
            guidance=recovery_guidance(snapshot.usb_path),
        )

    def _cleanup(self, fd: int, snapshot: TargetSnapshot, *, claimed: bool) -> CleanupRecord:
        """Release if claimed, then reconnect. Runs whatever went wrong before.

        Both phases are attempted even if the first fails, because the
        reconnect is the half that gives the driver back. Nothing escapes: a
        cleanup that raised would replace the primary result with its own
        exception. Termination is already shielded by the caller, so the
        ``BaseException`` guard below is the backstop for the case where the
        kernel would not let the shield install its handlers.
        """
        results: list[PhaseResult] = []
        if claimed:
            results.append(self._guarded(self._release, fd, Phase.RELEASE))
        results.append(self._guarded(self._reconnect, fd, Phase.RECONNECT))
        ok = all(result.ok for result in results)
        return CleanupRecord(
            attempted=True,
            phases=tuple(results),
            ok=ok,
            note=(
                f"interface {OWNED_INTERFACE} was released and handed back to the kernel"
                if ok
                else f"interface {OWNED_INTERFACE} was NOT fully restored"
            ),
            guidance=None if ok else recovery_guidance(snapshot.usb_path),
        )

    @staticmethod
    def _guarded(step: Callable[[int], PhaseResult], fd: int, phase: Phase) -> PhaseResult:
        try:
            return step(fd)
        # Deliberately broad, BaseException included: cleanup is the last line
        # of defence and nothing may abort it part-way, so a step that fails
        # this way still lets the next one run and still reports itself.
        except BaseException as exc:
            return PhaseResult(
                phase=phase, ok=False, detail=f"{phase} raised {type(exc).__name__}: {exc}"
            )


# --- the one read, as an owned operation --------------------------------------


@dataclass(slots=True)
class _SingleReadOperation:
    """The one already-whitelisted read, performed while the claim is held.

    Built fresh per run by :class:`UsbfsOwnershipBackend`. The request is not a
    parameter: it reads :data:`OWNED_REQUEST` or nothing.
    """

    ioctl_fn: IoctlFn
    clock: Clock

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        primary = self._transfer(fd)
        phases.append(
            PhaseResult(
                phase=Phase.CONTROL_READ,
                ok=primary.status is OwnershipStatus.OK,
                detail=primary.detail,
                duration_s=primary.duration_s,
                errno=primary.errno,
                errno_name=primary.errno_name,
            )
        )
        return primary

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        """Nothing to undo: a device-to-host ``GET_CUR`` writes no device state."""
        return ProtocolCleanupRecord.not_needed("a read leaves no device state to undo")

    def _transfer(self, fd: int) -> _Primary:
        """The one fixed read, on the descriptor that holds the claim."""
        request = OWNED_REQUEST
        buffer = ctypes.create_string_buffer(request.length)
        packed = pack_ctrltransfer(request, ctypes.addressof(buffer))
        started = self.clock()
        try:
            transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return _Primary(
                status=_ERRNO_STATUS.get(number or 0, OwnershipStatus.IOCTL_ERROR),
                detail=f"USBDEVFS_CONTROL failed: {exc}",
                duration_s=self.clock() - started,
                errno=number,
                errno_name=name,
            )
        duration = self.clock() - started
        if transferred < 0:
            return _Primary(
                status=OwnershipStatus.IOCTL_ERROR,
                detail=f"USBDEVFS_CONTROL returned {transferred}",
                duration_s=duration,
            )
        payload = parse_version_payload(bytes(buffer.raw[: min(transferred, request.length)]))
        if transferred != request.length:
            status = OwnershipStatus.SHORT_RESPONSE
            detail = f"device returned {transferred} of {request.length} bytes"
        elif not payload.matches_expected:
            status = OwnershipStatus.UNEXPECTED_CONTENT
            detail = "payload does not match the versions the vendor SDK reported for this unit"
        else:
            status = OwnershipStatus.OK
            detail = f"read {transferred} bytes matching the known version payload"
        return _Primary(status=status, detail=detail, payload=payload, duration_s=duration)


#: What the dry run says instead of performing the lifecycle.
_DRY_RUN_DETAIL: Final = (
    "dry run: no device node was opened and no driver was detached "
    "(pass --execute to perform the lifecycle)"
)


class UsbfsOwnershipBackend(InterfaceZeroLifecycle):
    """The interface-0 lifecycle carrying the one whitelisted read, unchanged.

    Nothing here is a parameter but the serial: the operation is fixed at class
    level and reads :data:`OWNED_REQUEST` or nothing.
    """

    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return _SingleReadOperation(ioctl_fn=self.ioctl_fn, clock=self.clock)

    def _snapshot_request(self) -> ControlRequest:
        return OWNED_REQUEST

    # -- public API ----------------------------------------------------------

    def preflight(self, serial: str) -> OwnershipOutcome:
        """Resolve the target and check every precondition, opening nothing."""
        ensure_allowed(OWNED_REQUEST)
        return self._from_preflight(self._preflight(serial, _DRY_RUN_DETAIL))

    def run(self, serial: str) -> OwnershipOutcome:
        """Perform the lifecycle exactly once. No phase is ever retried."""
        ensure_allowed(OWNED_REQUEST)
        started = self.clock()
        preflight = self._preflight(serial, _DRY_RUN_DETAIL)
        snapshot = preflight.target
        if preflight.status is not OwnershipStatus.NOT_ATTEMPTED or snapshot is None:
            return self._from_preflight(preflight)

        phases: list[PhaseResult] = list(preflight.phases)
        primary, window = self._perform_lifecycle(snapshot, phases)
        return self._outcome(
            primary.status,
            primary.detail,
            target=snapshot,
            phases=tuple(phases),
            cleanup=window.cleanup,
            protocol_cleanup=window.protocol,
            payload=primary.payload,
            transfer_duration_s=primary.duration_s,
            total_duration_s=self.clock() - started,
            errno=primary.errno,
            errno_name=primary.errno_name,
        )

    # -- assembly ------------------------------------------------------------

    def _from_preflight(self, preflight: Preflight) -> OwnershipOutcome:
        return self._outcome(
            preflight.status,
            preflight.detail,
            target=preflight.target,
            phases=preflight.phases,
            cleanup=CleanupRecord.not_needed(preflight.cleanup_note),
        )

    def _outcome(
        self,
        status: OwnershipStatus,
        detail: str,
        *,
        target: TargetSnapshot | None,
        phases: tuple[PhaseResult, ...],
        cleanup: CleanupRecord,
        protocol_cleanup: ProtocolCleanupRecord | None = None,
        payload: VersionPayload | None = None,
        transfer_duration_s: float | None = None,
        total_duration_s: float | None = None,
        errno: int | None = None,
        errno_name: str | None = None,
    ) -> OwnershipOutcome:
        return OwnershipOutcome(
            status=status,
            request=OWNED_REQUEST,
            target=target,
            phases=phases,
            cleanup=cleanup,
            payload=payload,
            transfer_duration_s=transfer_duration_s,
            total_duration_s=total_duration_s,
            detail=detail,
            errno=errno,
            errno_name=errno_name,
            protocol_cleanup=(
                ProtocolCleanupRecord.not_needed("a read leaves no device state to undo")
                if protocol_cleanup is None
                else protocol_cleanup
            ),
        )
