"""One owned, read-only USB control request against the scanner.

This module is deliberately *not* a general control-transfer interface. Its
whole safety argument is that the only setup packet it can put on the wire is a
single named, already-observed, read-only request:

    bmRequestType=0xa1  bRequest=0x81  wValue=0x0600  wIndex=0x0300  wLength=60

That request was captured from the vendor SDK on this exact unit before the
SDK's authorization step, it changes no device state, and its correct answer is
known independently (the SDK reports the same two version strings from
``queryCameras``), so a wrong answer is unmistakable. Everything else -
arbitrary requests, writes, interface claim or detach, device reset, URB
submission, streaming - is out of reach by construction: the request whitelist
is checked before a file descriptor is opened, and ``USBDEVFS_CONTROL`` is the
only usbfs command this module knows the number of.

The kernel's usbfs layer does claim an *unclaimed* interface on the caller's
behalf when a class request names it (``checkintf`` in ``drivers/usb/core/devio.c``).
This module never issues ``USBDEVFS_CLAIMINTERFACE`` itself, and it never
detaches a driver, so when a kernel driver already owns the target interface
that implicit claim fails and the request is refused - a typed failure here,
not an escalation.
"""

from __future__ import annotations

import ctypes
import enum
import errno as errno_mod
import fcntl
import hashlib
import os
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

from scanner3d.discovery import DEFAULT_DEV_ROOT, DEFAULT_SYSFS_ROOT, ScannerDevice, find_scanners
from scanner3d.errors import ControlRequestNotAllowed, DeviceAbsentError
from scanner3d.usb_descriptors import VideoControlEntity

#: The version strings the vendor SDK reported for this unit, used as the
#: independent oracle for the one request below.
EXPECTED_FIRMWARE_VERSION: Final = "v2.8.7.20220610"
EXPECTED_ALGORITHM_VERSION: Final = "v7.5.7.0426"
EXPECTED_PAYLOAD_TEXT: Final = f"{EXPECTED_FIRMWARE_VERSION}|{EXPECTED_ALGORITHM_VERSION}"

#: Hard ceiling on the transfer timeout, in milliseconds.
MAX_TIMEOUT_MS: Final = 3000

# --- usbfs ABI ---------------------------------------------------------------

# struct usbdevfs_ctrltransfer { u8; u8; u16; u16; u16; u32 timeout; void *data; }
_CTRLTRANSFER_FORMAT: Final = "@BBHHHIP"
_CTRLTRANSFER_SIZE: Final = struct.calcsize(_CTRLTRANSFER_FORMAT)

_IOC_NRBITS: Final = 8
_IOC_TYPEBITS: Final = 8
_IOC_SIZEBITS: Final = 14
_IOC_WRITE: Final = 1
_IOC_READ: Final = 2


def _ioc(direction: int, ioc_type: int, number: int, size: int) -> int:
    """Reimplementation of the kernel's ``_IOC`` encoding macro."""
    if not 0 <= size < (1 << _IOC_SIZEBITS):
        raise ValueError(f"ioctl payload size {size} out of range")
    return (
        (direction << (_IOC_NRBITS + _IOC_TYPEBITS + _IOC_SIZEBITS))
        | (size << (_IOC_NRBITS + _IOC_TYPEBITS))
        | (ioc_type << _IOC_NRBITS)
        | number
    )


#: ``_IOWR('U', 0, struct usbdevfs_ctrltransfer)``. The *only* usbfs command
#: whose number appears anywhere in this package.
USBDEVFS_CONTROL: Final = _ioc(_IOC_READ | _IOC_WRITE, ord("U"), 0, _CTRLTRANSFER_SIZE)

#: ``bmRequestType`` bit that marks a device-to-host (read) transfer.
_DIRECTION_IN: Final = 0x80


# --- the whitelist -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControlRequest:
    """A named, fully specified control transfer.

    Constructing one is harmless; only members of :data:`ALLOWED_REQUESTS` can
    be handed to the hardware path.
    """

    name: str
    request_type: int
    request: int
    value: int
    index: int
    length: int
    timeout_ms: int
    description: str = ""

    @property
    def wire_tuple(self) -> tuple[int, int, int, int, int, int]:
        """The identity used for whitelist comparison. Nothing else counts."""
        return (
            self.request_type,
            self.request,
            self.value,
            self.index,
            self.length,
            self.timeout_ms,
        )

    @property
    def is_read(self) -> bool:
        return bool(self.request_type & _DIRECTION_IN)

    @property
    def interface_number(self) -> int:
        """Low byte of ``wIndex``: the interface a UVC class request names."""
        return self.index & 0xFF

    @property
    def entity_id(self) -> int:
        """High byte of ``wIndex``: the VideoControl entity a UVC request names."""
        return (self.index >> 8) & 0xFF

    @property
    def control_selector(self) -> int:
        """High byte of ``wValue``: the UVC control selector."""
        return (self.value >> 8) & 0xFF

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bmRequestType": f"0x{self.request_type:02x}",
            "bRequest": f"0x{self.request:02x}",
            "wValue": f"0x{self.value:04x}",
            "wIndex": f"0x{self.index:04x}",
            "wLength": self.length,
            "timeout_ms": self.timeout_ms,
            "direction": "in" if self.is_read else "out",
            "interface_number": self.interface_number,
            "entity_id": self.entity_id,
            "control_selector": f"0x{self.control_selector:02x}",
            "description": self.description,
        }


#: The one request this project is allowed to send. Captured from the vendor
#: SDK's own session on this unit (architecture.md, "The vendor channel, byte by
#: byte", row 5).
FIRMWARE_ALGORITHM_VERSION_READ: Final = ControlRequest(
    name="firmware-algorithm-version",
    request_type=0xA1,
    request=0x81,
    value=0x0600,
    index=0x0300,
    length=60,
    timeout_ms=MAX_TIMEOUT_MS,
    description="UVC GET_CUR, 60-byte firmware|algorithm version payload",
)

ALLOWED_REQUESTS: Final[Mapping[str, ControlRequest]] = MappingProxyType(
    {FIRMWARE_ALGORITHM_VERSION_READ.name: FIRMWARE_ALGORITHM_VERSION_READ}
)

_ALLOWED_WIRE_TUPLES: Final = frozenset(r.wire_tuple for r in ALLOWED_REQUESTS.values())


def ensure_allowed(request: ControlRequest) -> ControlRequest:
    """Return ``request`` if it is whitelisted, else raise.

    Called before anything is resolved or opened, so a rejected request never
    reaches a device node.
    """
    if not request.is_read:
        raise ControlRequestNotAllowed(
            f"{request.name}: bmRequestType 0x{request.request_type:02x} is not device-to-host"
        )
    if not 0 < request.timeout_ms <= MAX_TIMEOUT_MS:
        raise ControlRequestNotAllowed(
            f"{request.name}: timeout {request.timeout_ms} ms outside (0, {MAX_TIMEOUT_MS}]"
        )
    if request.wire_tuple not in _ALLOWED_WIRE_TUPLES:
        raise ControlRequestNotAllowed(
            f"{request.name}: setup packet "
            f"{request.wire_tuple} is not the one whitelisted request "
            f"{FIRMWARE_ALGORITHM_VERSION_READ.wire_tuple}"
        )
    return request


def pack_ctrltransfer(request: ControlRequest, data_address: int) -> bytearray:
    """Pack ``struct usbdevfs_ctrltransfer`` exactly as the kernel expects it."""
    return bytearray(
        struct.pack(
            _CTRLTRANSFER_FORMAT,
            request.request_type,
            request.request,
            request.value,
            request.index,
            request.length,
            request.timeout_ms,
            data_address,
        )
    )


# --- outcomes ----------------------------------------------------------------


class ControlStatus(enum.StrEnum):
    """Distinguishable outcomes of the one request."""

    OK = "ok"
    NOT_ATTEMPTED = "not_attempted"
    DEVICE_ABSENT = "device_absent"
    DEVICE_REPLACED = "device_replaced"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    IOCTL_ERROR = "ioctl_error"
    SHORT_RESPONSE = "short_response"
    UNEXPECTED_CONTENT = "unexpected_content"


#: Process exit code per outcome. ``device_absent`` and ``permission_denied``
#: deliberately match the codes ``doctor`` already uses for the same conditions.
CONTROL_EXIT_CODES: Final[Mapping[ControlStatus, int]] = MappingProxyType(
    {
        ControlStatus.OK: 0,
        ControlStatus.NOT_ATTEMPTED: 0,
        ControlStatus.DEVICE_ABSENT: 2,
        ControlStatus.PERMISSION_DENIED: 3,
        ControlStatus.TIMEOUT: 4,
        ControlStatus.IOCTL_ERROR: 5,
        ControlStatus.UNEXPECTED_CONTENT: 6,
        ControlStatus.SHORT_RESPONSE: 7,
        ControlStatus.DEVICE_REPLACED: 8,
    }
)


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """What the target usbfs node was, immediately before it was opened."""

    serial: str
    sysfs_path: str
    usb_path: str
    bus_number: int
    device_number: int
    node_path: str
    rdev: int
    rdev_major: int
    rdev_minor: int
    inode: int
    mode: str
    interface_number: int
    interface_driver: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "serial": self.serial,
            "sysfs_path": self.sysfs_path,
            "usb_path": self.usb_path,
            "bus_number": self.bus_number,
            "device_number": self.device_number,
            "node_path": self.node_path,
            "rdev": self.rdev,
            "rdev_major_minor": f"{self.rdev_major}:{self.rdev_minor}",
            "inode": self.inode,
            "mode": self.mode,
            "target_interface_number": self.interface_number,
            "target_interface_driver": self.interface_driver,
        }


@dataclass(frozen=True, slots=True)
class VersionPayload:
    """The parsed 60-byte response and how it compares to the known answer."""

    raw_length: int
    sha256: str
    text: str | None
    firmware_version: str | None
    algorithm_version: str | None
    nul_padded: bool
    matches_expected: bool
    decode_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_length": self.raw_length,
            "sha256": self.sha256,
            "text": self.text,
            "firmware_version": self.firmware_version,
            "algorithm_version": self.algorithm_version,
            "nul_padded": self.nul_padded,
            "expected_firmware_version": EXPECTED_FIRMWARE_VERSION,
            "expected_algorithm_version": EXPECTED_ALGORITHM_VERSION,
            "matches_expected": self.matches_expected,
            "decode_error": self.decode_error,
        }


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    """The complete, typed result of attempting (or declining) the request."""

    status: ControlStatus
    request: ControlRequest
    target: TargetSnapshot | None
    addressed_entity: VideoControlEntity | None
    entity_lookup_note: str
    duration_s: float | None
    payload: VersionPayload | None
    detail: str
    errno: int | None = None
    errno_name: str | None = None

    @property
    def exit_code(self) -> int:
        return CONTROL_EXIT_CODES[self.status]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": str(self.status),
            "exit_code": self.exit_code,
            "detail": self.detail,
            "request": self.request.as_dict(),
            "target": self.target.as_dict() if self.target is not None else None,
            "addressed_entity": (
                self.addressed_entity.as_dict() if self.addressed_entity is not None else None
            ),
            "entity_lookup_note": self.entity_lookup_note,
            "duration_s": self.duration_s,
            "response": self.payload.as_dict() if self.payload is not None else None,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


# --- target resolution -------------------------------------------------------


def resolve_target(
    serial: str | None,
    request: ControlRequest,
    *,
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
) -> tuple[TargetSnapshot, ScannerDevice]:
    """Find the 2207:110c device with this exact serial and stat its node.

    Raises :class:`DeviceAbsentError` when there is not exactly one match, when
    the node is missing, or when the sysfs record lacks the bus/device numbers
    needed to name the node.
    """
    devices = find_scanners(sysfs_root=sysfs_root, dev_root=dev_root)
    if serial is not None:
        devices = tuple(d for d in devices if d.serial == serial)
        if not devices:
            raise DeviceAbsentError(f"no 2207:110c device with serial {serial!r} is attached")
    elif not devices:
        raise DeviceAbsentError("no 2207:110c device is attached")
    if len(devices) > 1:
        raise DeviceAbsentError(
            f"{len(devices)} matching devices attached; pass --serial to name exactly one"
        )
    device = devices[0]
    if device.serial is None:
        raise DeviceAbsentError(f"{device.usb_path} exposes no serial; refusing to target it")
    if device.bus_number is None or device.device_number is None:
        raise DeviceAbsentError(f"{device.usb_path} has no busnum/devnum; cannot name a usbfs node")

    node = Path(device.usbfs.path)
    try:
        st = node.stat()
    except OSError as exc:
        raise DeviceAbsentError(f"cannot stat {node}: {exc}") from exc

    driver = next(
        (i.driver for i in device.interfaces if i.number == request.interface_number), None
    )
    snapshot = TargetSnapshot(
        serial=device.serial,
        sysfs_path=device.sysfs_path,
        usb_path=device.usb_path,
        bus_number=device.bus_number,
        device_number=device.device_number,
        node_path=str(node),
        rdev=st.st_rdev,
        rdev_major=os.major(st.st_rdev),
        rdev_minor=os.minor(st.st_rdev),
        inode=st.st_ino,
        mode=oct(st.st_mode & 0o7777),
        interface_number=request.interface_number,
        interface_driver=driver,
    )
    return snapshot, device


def addressed_entity(
    device: ScannerDevice, request: ControlRequest
) -> tuple[VideoControlEntity | None, str]:
    """Look up the descriptor entity that ``request``'s ``wIndex`` names."""
    if device.descriptors is None:
        return None, f"descriptors unavailable: {device.descriptor_error}"
    entity = device.descriptors.entity(request.interface_number, request.entity_id)
    if entity is None:
        return None, (
            f"interface {request.interface_number} declares no entity "
            f"{request.entity_id}; the device's own descriptors do not describe this address"
        )
    return entity, "resolved from the device's VideoControl descriptors"


# --- response parsing --------------------------------------------------------


def parse_version_payload(raw: bytes) -> VersionPayload:
    """Decode a 60-byte firmware/algorithm response. Never raises."""
    digest = hashlib.sha256(raw).hexdigest()
    body, separator, padding = raw.partition(b"\x00")
    nul_padded = bool(separator) and padding == bytes(len(padding))
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as exc:
        return VersionPayload(
            raw_length=len(raw),
            sha256=digest,
            text=None,
            firmware_version=None,
            algorithm_version=None,
            nul_padded=nul_padded,
            matches_expected=False,
            decode_error=f"payload is not ASCII: {exc}",
        )
    parts = text.split("|")
    firmware = parts[0] if len(parts) == 2 else None
    algorithm = parts[1] if len(parts) == 2 else None
    matches = (
        len(raw) == FIRMWARE_ALGORITHM_VERSION_READ.length
        and nul_padded
        and firmware == EXPECTED_FIRMWARE_VERSION
        and algorithm == EXPECTED_ALGORITHM_VERSION
    )
    return VersionPayload(
        raw_length=len(raw),
        sha256=digest,
        text=text,
        firmware_version=firmware,
        algorithm_version=algorithm,
        nul_padded=nul_padded,
        matches_expected=matches,
        decode_error=(
            None if len(parts) == 2 else f"expected one '|' separator, got {len(parts) - 1}"
        ),
    )


# --- the hardware path -------------------------------------------------------

Opener = Callable[[str], int]
IoctlFn = Callable[[int, int, bytearray, bool], int]
Closer = Callable[[int], None]
Clock = Callable[[], float]

_ERRNO_STATUS: Final[Mapping[int, ControlStatus]] = MappingProxyType(
    {
        # Two distinct errnos, one status: the kernel reports an unreadable
        # usbfs node as EACCES and a privilege-gated usbfs operation as EPERM.
        errno_mod.EACCES: ControlStatus.PERMISSION_DENIED,
        errno_mod.EPERM: ControlStatus.PERMISSION_DENIED,
        errno_mod.ENODEV: ControlStatus.DEVICE_ABSENT,
        errno_mod.ENOENT: ControlStatus.DEVICE_ABSENT,
        errno_mod.ENXIO: ControlStatus.DEVICE_ABSENT,
        errno_mod.ESHUTDOWN: ControlStatus.DEVICE_ABSENT,
        errno_mod.ETIMEDOUT: ControlStatus.TIMEOUT,
    }
)


def _default_opener(path: str) -> int:
    # O_RDWR, not O_RDONLY: usbfs rejects USBDEVFS_CONTROL on a descriptor
    # without FMODE_WRITE even for a device-to-host transfer. Opening the node
    # writes nothing; the direction of the transfer is fixed by the whitelist.
    return os.open(path, os.O_RDWR | os.O_CLOEXEC)


class NodeIdentity(Protocol):
    """The part of a resolved target :func:`identity_drift` needs.

    A protocol rather than :class:`TargetSnapshot` because the check is the
    same whether the node is a usbfs device or a V4L2 capture node, and it is
    exactly the kind of thing that must not be reimplemented per caller: it is
    the last chance to notice that the device under the descriptor was replaced
    between resolution and use.
    """

    @property
    def serial(self) -> str: ...

    @property
    def sysfs_path(self) -> str: ...

    @property
    def device_number(self) -> int: ...

    @property
    def rdev(self) -> int: ...

    @property
    def rdev_major(self) -> int: ...

    @property
    def rdev_minor(self) -> int: ...


def identity_drift(fd: int, snapshot: NodeIdentity, *, moment: str = "the transfer") -> str | None:
    """Detect disconnect / re-enumeration between resolution and ``moment``.

    Compares the opened descriptor's device number against the snapshot, and
    re-reads the serial and devnum sysfs attributes that named it. Returns the
    reason the target is no longer the device that was resolved, or ``None``
    when it still is. Reads only; it issues no ioctl.

    ``moment`` only names what the caller is about to do, so the reason reads
    truthfully for a capture as well as for a control transfer.
    """
    try:
        st = os.fstat(fd)
    except OSError as exc:
        return f"cannot fstat the opened node: {exc}"
    if st.st_rdev != snapshot.rdev:
        return (
            f"opened node is {os.major(st.st_rdev)}:{os.minor(st.st_rdev)}, "
            f"snapshot was {snapshot.rdev_major}:{snapshot.rdev_minor}"
        )
    sysfs = Path(snapshot.sysfs_path)
    for attribute, expected in (
        ("serial", snapshot.serial),
        ("devnum", str(snapshot.device_number)),
    ):
        try:
            current = (sysfs / attribute).read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            return f"{sysfs / attribute} vanished before {moment}: {exc}"
        if current != expected:
            return f"{attribute} changed from {expected!r} to {current!r} before {moment}"
    return None


def _errno_of(exc: OSError) -> tuple[int | None, str | None]:
    if exc.errno is None:
        return None, None
    return exc.errno, errno_mod.errorcode.get(exc.errno, f"errno {exc.errno}")


@dataclass(frozen=True, slots=True)
class UsbfsControlBackend:
    """The single-request usbfs backend, with every syscall injectable."""

    opener: Opener = _default_opener
    ioctl_fn: IoctlFn = fcntl.ioctl
    closer: Closer = os.close
    clock: Clock = time.perf_counter
    sysfs_root: Path = DEFAULT_SYSFS_ROOT
    dev_root: Path = DEFAULT_DEV_ROOT

    def preflight(
        self, serial: str | None, request: ControlRequest = FIRMWARE_ALGORITHM_VERSION_READ
    ) -> ControlOutcome:
        """Resolve and describe the target without opening anything."""
        ensure_allowed(request)
        try:
            snapshot, device = resolve_target(
                serial, request, sysfs_root=self.sysfs_root, dev_root=self.dev_root
            )
        except DeviceAbsentError as exc:
            return ControlOutcome(
                status=ControlStatus.DEVICE_ABSENT,
                request=request,
                target=None,
                addressed_entity=None,
                entity_lookup_note="no device to read descriptors from",
                duration_s=None,
                payload=None,
                detail=str(exc),
            )
        entity, note = addressed_entity(device, request)
        return ControlOutcome(
            status=ControlStatus.NOT_ATTEMPTED,
            request=request,
            target=snapshot,
            addressed_entity=entity,
            entity_lookup_note=note,
            duration_s=None,
            payload=None,
            detail="dry run: no device node was opened (pass --execute to perform the read)",
        )

    def read(
        self, serial: str | None, request: ControlRequest = FIRMWARE_ALGORITHM_VERSION_READ
    ) -> ControlOutcome:
        """Perform the one whitelisted read. One open, one ioctl, one close."""
        ensure_allowed(request)
        preflight = self.preflight(serial, request)
        if preflight.status is not ControlStatus.NOT_ATTEMPTED:
            return preflight
        snapshot = preflight.target
        if snapshot is None:  # unreachable: NOT_ATTEMPTED always carries a target
            return preflight

        def outcome(
            status: ControlStatus,
            detail: str,
            *,
            duration: float | None = None,
            payload: VersionPayload | None = None,
            exc: OSError | None = None,
        ) -> ControlOutcome:
            number, name = _errno_of(exc) if exc is not None else (None, None)
            return ControlOutcome(
                status=status,
                request=request,
                target=snapshot,
                addressed_entity=preflight.addressed_entity,
                entity_lookup_note=preflight.entity_lookup_note,
                duration_s=duration,
                payload=payload,
                detail=detail,
                errno=number,
                errno_name=name,
            )

        try:
            fd = self.opener(snapshot.node_path)
        except OSError as exc:
            number, _ = _errno_of(exc)
            status = _ERRNO_STATUS.get(number or 0, ControlStatus.IOCTL_ERROR)
            return outcome(status, f"cannot open {snapshot.node_path}: {exc}", exc=exc)

        try:
            drift = identity_drift(fd, snapshot)
            if drift is not None:
                return outcome(ControlStatus.DEVICE_REPLACED, drift)

            buffer = ctypes.create_string_buffer(request.length)
            packed = pack_ctrltransfer(request, ctypes.addressof(buffer))
            started = self.clock()
            try:
                transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
            except OSError as exc:
                duration = self.clock() - started
                number, _ = _errno_of(exc)
                status = _ERRNO_STATUS.get(number or 0, ControlStatus.IOCTL_ERROR)
                return outcome(
                    status,
                    f"USBDEVFS_CONTROL failed: {exc}",
                    duration=duration,
                    exc=exc,
                )
            duration = self.clock() - started
        finally:
            self.closer(fd)

        if transferred < 0:
            return outcome(
                ControlStatus.IOCTL_ERROR,
                f"USBDEVFS_CONTROL returned {transferred}",
                duration=duration,
            )
        raw = bytes(buffer.raw[: min(transferred, request.length)])
        payload = parse_version_payload(raw)
        if transferred != request.length:
            return outcome(
                ControlStatus.SHORT_RESPONSE,
                f"device returned {transferred} of {request.length} bytes",
                duration=duration,
                payload=payload,
            )
        if not payload.matches_expected:
            return outcome(
                ControlStatus.UNEXPECTED_CONTENT,
                f"payload does not match the known {EXPECTED_PAYLOAD_TEXT!r}",
                duration=duration,
                payload=payload,
            )
        return outcome(
            ControlStatus.OK,
            f"read {transferred} bytes matching {EXPECTED_PAYLOAD_TEXT!r}",
            duration=duration,
            payload=payload,
        )
