"""Dry-by-default, fixed-whitelist V4L2 RGB control inventory.

This module deliberately has a smaller repertoire than the capture backend.  It
resolves the RGB capture node from the existing stable identity/index discovery,
opens that node read-only, checks identity once, and asks only the six standard
camera controls fixed below.  It never negotiates a format, creates a queue,
or waits for a frame.  The vendor RGB exposure property is intentionally not in
this module: it is not a standard V4L2 control and must not be guessed here.

The whitelist is import-time data, not caller input.  Each control gets one
metadata query.  A current value is requested at most once, and only when the
metadata says that the control is not disabled or write-only and is one of the
scalar/menu types this inventory understands.  Read-only, inactive and
volatile controls remain readable through the standard value request; their
flags are preserved and classified separately.  Unsupported, disabled,
inactive, volatile, read-only, write-only, unknown-type and ioctl failures
remain separate evidence.  No menu enumeration is needed because the
device-reported menu index is preserved as an integer.
"""

from __future__ import annotations

import enum
import errno as errno_mod
import fcntl
import os
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scanner3d.config import stated_serial
from scanner3d.control import identity_drift
from scanner3d.discovery import DEFAULT_DEV_ROOT, DEFAULT_SYSFS_ROOT, ScannerDevice
from scanner3d.errors import DeviceAbsentError, V4l2CaptureNotAllowed
from scanner3d.ownership import (
    SignalGuard,
    SignalReceived,
    TerminationShield,
    cleanup_on_termination,
    defer_termination,
)
from scanner3d.rgb_frame import RGB_PROFILE
from scanner3d.v4l2_capture import NodeSnapshot, preflight_capture_node, resolve_capture_node

# --- V4L2 UAPI ---------------------------------------------------------------

V4L2_CID_BASE: Final = 0x00980900
V4L2_CID_CAMERA_CLASS_BASE: Final = 0x009A0900
V4L2_CID_GAIN: Final = V4L2_CID_BASE + 19
V4L2_CID_EXPOSURE_AUTO: Final = V4L2_CID_CAMERA_CLASS_BASE + 1
V4L2_CID_EXPOSURE_ABSOLUTE: Final = V4L2_CID_CAMERA_CLASS_BASE + 2
V4L2_CID_EXPOSURE_AUTO_PRIORITY: Final = V4L2_CID_CAMERA_CLASS_BASE + 3
V4L2_CID_AUTO_WHITE_BALANCE: Final = V4L2_CID_BASE + 12
V4L2_CID_WHITE_BALANCE_TEMPERATURE: Final = V4L2_CID_BASE + 26

V4L2_CTRL_TYPE_INTEGER: Final = 1
V4L2_CTRL_TYPE_BOOLEAN: Final = 2
V4L2_CTRL_TYPE_MENU: Final = 3
V4L2_CTRL_TYPE_INTEGER_MENU: Final = 9

V4L2_CTRL_FLAG_DISABLED: Final = 0x0001
V4L2_CTRL_FLAG_GRABBED: Final = 0x0002
V4L2_CTRL_FLAG_READ_ONLY: Final = 0x0004
V4L2_CTRL_FLAG_UPDATE: Final = 0x0008
V4L2_CTRL_FLAG_INACTIVE: Final = 0x0010
V4L2_CTRL_FLAG_SLIDER: Final = 0x0020
V4L2_CTRL_FLAG_WRITE_ONLY: Final = 0x0040
V4L2_CTRL_FLAG_VOLATILE: Final = 0x0080

# ``struct v4l2_queryctrl``: id, type, name[32], min/max/step/default,
# flags, reserved[2].  ``struct v4l2_control`` is id plus signed value.
QUERYCTRL_FORMAT: Final = "=II32siiiiI8s"
CONTROL_FORMAT: Final = "=Ii"
QUERYCTRL_BYTES: Final = struct.calcsize(QUERYCTRL_FORMAT)
CONTROL_BYTES: Final = struct.calcsize(CONTROL_FORMAT)
EXPECTED_CONTROL_STRUCT_BYTES: Final[Mapping[str, int]] = MappingProxyType(
    {"v4l2_queryctrl": 68, "v4l2_control": 8}
)

_IOC_NRBITS: Final = 8
_IOC_TYPEBITS: Final = 8
_IOC_SIZEBITS: Final = 14
_IOC_WRITE: Final = 1
_IOC_READ: Final = 2


def _ioc(direction: int, number: int, size: int) -> int:
    return (
        (direction << (_IOC_NRBITS + _IOC_TYPEBITS + _IOC_SIZEBITS))
        | (size << (_IOC_NRBITS + _IOC_TYPEBITS))
        | (ord("V") << _IOC_NRBITS)
        | number
    )


VIDIOC_QUERYCTRL: Final = _ioc(_IOC_READ | _IOC_WRITE, 36, QUERYCTRL_BYTES)
VIDIOC_G_CTRL: Final = _ioc(_IOC_READ | _IOC_WRITE, 27, CONTROL_BYTES)
ALLOWED_CONTROL_IOCTLS: Final[Mapping[str, int]] = MappingProxyType(
    {"VIDIOC_QUERYCTRL": VIDIOC_QUERYCTRL, "VIDIOC_G_CTRL": VIDIOC_G_CTRL}
)

_CONTROL_TYPE_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {
        V4L2_CTRL_TYPE_INTEGER: "V4L2_CTRL_TYPE_INTEGER",
        V4L2_CTRL_TYPE_BOOLEAN: "V4L2_CTRL_TYPE_BOOLEAN",
        V4L2_CTRL_TYPE_MENU: "V4L2_CTRL_TYPE_MENU",
        V4L2_CTRL_TYPE_INTEGER_MENU: "V4L2_CTRL_TYPE_INTEGER_MENU",
    }
)
_FLAG_NAMES: Final[tuple[tuple[int, str], ...]] = (
    (V4L2_CTRL_FLAG_DISABLED, "V4L2_CTRL_FLAG_DISABLED"),
    (V4L2_CTRL_FLAG_GRABBED, "V4L2_CTRL_FLAG_GRABBED"),
    (V4L2_CTRL_FLAG_READ_ONLY, "V4L2_CTRL_FLAG_READ_ONLY"),
    (V4L2_CTRL_FLAG_UPDATE, "V4L2_CTRL_FLAG_UPDATE"),
    (V4L2_CTRL_FLAG_INACTIVE, "V4L2_CTRL_FLAG_INACTIVE"),
    (V4L2_CTRL_FLAG_SLIDER, "V4L2_CTRL_FLAG_SLIDER"),
    (V4L2_CTRL_FLAG_WRITE_ONLY, "V4L2_CTRL_FLAG_WRITE_ONLY"),
    (V4L2_CTRL_FLAG_VOLATILE, "V4L2_CTRL_FLAG_VOLATILE"),
)


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """One fixed standard control and why it belongs in the inventory."""

    name: str
    control_id: int
    purpose: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "id": f"0x{self.control_id:08x}", "purpose": self.purpose}


CONTROL_WHITELIST: Final[tuple[ControlSpec, ...]] = (
    ControlSpec("gain", V4L2_CID_GAIN, "standard camera gain"),
    ControlSpec("exposure_auto", V4L2_CID_EXPOSURE_AUTO, "standard exposure auto mode"),
    ControlSpec("exposure_absolute", V4L2_CID_EXPOSURE_ABSOLUTE, "standard absolute exposure"),
    ControlSpec(
        "exposure_auto_priority",
        V4L2_CID_EXPOSURE_AUTO_PRIORITY,
        "standard exposure auto-priority toggle",
    ),
    ControlSpec("auto_white_balance", V4L2_CID_AUTO_WHITE_BALANCE, "standard auto white balance"),
    ControlSpec(
        "white_balance_temperature",
        V4L2_CID_WHITE_BALANCE_TEMPERATURE,
        "standard white-balance temperature",
    ),
)
CONTROL_IDS: Final[frozenset[int]] = frozenset(spec.control_id for spec in CONTROL_WHITELIST)
# The series telemetry path is intentionally narrower than the offline
# inventory: these are the three scalar values that can be sampled without
# QUERYCTRL/QUERYMENU or any control mutation.
RGB_SERIES_CONTROL_WHITELIST: Final[tuple[ControlSpec, ...]] = CONTROL_WHITELIST[:3]
RGB_SERIES_CONTROL_IDS: Final[frozenset[int]] = frozenset(
    spec.control_id for spec in RGB_SERIES_CONTROL_WHITELIST
)
RGB_SERIES_CONTROL_ATTEMPTS: Final = len(RGB_SERIES_CONTROL_WHITELIST)
MAX_QUERY_COUNT: Final = len(CONTROL_WHITELIST)
MAX_VALUE_COUNT: Final = len(CONTROL_WHITELIST)
MAX_IOCTL_COUNT: Final = MAX_QUERY_COUNT + MAX_VALUE_COUNT
RGB_CONTROLS_EXECUTE_OPT_IN: Final = "--read-rgb-controls"
#: The scanner this installation may act on. Read from configuration rather
#: than hard-coded: several operations here detach kernel drivers or write
#: vendor property keys, and must not be aimed at whatever happens to be
#: plugged in. Empty when nothing is configured - call
#: scanner3d.config.resolve_serial() for a usable value, which falls back to
#: the single attached scanner and refuses when that is ambiguous.
AUTHORIZED_SERIAL: Final = stated_serial()


def ensure_layout_intact() -> None:
    actual = {"v4l2_queryctrl": QUERYCTRL_BYTES, "v4l2_control": CONTROL_BYTES}
    for name, expected in EXPECTED_CONTROL_STRUCT_BYTES.items():
        if actual[name] != expected:
            raise V4l2CaptureNotAllowed(
                f"struct {name} packs to {actual[name]} bytes here but the UAPI says {expected}"
            )


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _flag_names(flags: int) -> tuple[str, ...]:
    return tuple(name for bit, name in _FLAG_NAMES if flags & bit)


@dataclass(frozen=True, slots=True)
class QueryMetadata:
    control_id: int
    control_type: int
    name: str
    minimum: int
    maximum: int
    step: int
    default: int
    flags: int

    @property
    def type_name(self) -> str:
        return _CONTROL_TYPE_NAMES.get(self.control_type, f"unknown ({self.control_type})")

    @property
    def flag_names(self) -> tuple[str, ...]:
        return _flag_names(self.flags)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": f"0x{self.control_id:08x}",
            "type": self.control_type,
            "type_name": self.type_name,
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "default": self.default,
            "flags": f"0x{self.flags:08x}",
            "flag_names": list(self.flag_names),
        }


def pack_queryctrl(control_id: int) -> bytearray:
    if control_id not in CONTROL_IDS:
        raise V4l2CaptureNotAllowed(f"control id 0x{control_id:08x} is not fixed in the whitelist")
    return bytearray(
        struct.pack(QUERYCTRL_FORMAT, control_id, 0, b"\0" * 32, 0, 0, 0, 0, 0, b"\0" * 8)
    )


def parse_queryctrl(raw: bytes) -> QueryMetadata:
    if len(raw) != QUERYCTRL_BYTES:
        raise V4l2CaptureNotAllowed(
            f"QUERYCTRL returned {len(raw)} bytes, expected {QUERYCTRL_BYTES}"
        )
    control_id, control_type, name, minimum, maximum, step, default, flags, _reserved = (
        struct.unpack(QUERYCTRL_FORMAT, raw)
    )
    return QueryMetadata(
        control_id, control_type, _cstr(name), minimum, maximum, step, default, flags
    )


def pack_control(control_id: int) -> bytearray:
    if control_id not in CONTROL_IDS:
        raise V4l2CaptureNotAllowed(f"control id 0x{control_id:08x} is not fixed in the whitelist")
    return bytearray(struct.pack(CONTROL_FORMAT, control_id, 0))


def parse_control(raw: bytes) -> tuple[int, int]:
    if len(raw) != CONTROL_BYTES:
        raise V4l2CaptureNotAllowed(f"G_CTRL returned {len(raw)} bytes, expected {CONTROL_BYTES}")
    return struct.unpack(CONTROL_FORMAT, raw)


class ControlObservationStatus(enum.StrEnum):
    NOT_QUERIED = "not_queried"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    DISABLED = "disabled"
    INACTIVE = "inactive"
    VOLATILE = "volatile"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    UNKNOWN_TYPE = "unknown_type"
    QUERY_ERROR = "query_error"
    VALUE_ERROR = "value_error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ControlObservation:
    spec: ControlSpec
    status: ControlObservationStatus
    query: QueryMetadata | None = None
    current: int | None = None
    query_errno: int | None = None
    query_errno_name: str | None = None
    value_errno: int | None = None
    value_errno_name: str | None = None
    detail: str = ""
    query_attempted: bool = False
    value_attempted: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.spec.name,
            "control": self.spec.as_dict(),
            "status": str(self.status),
            "query": None if self.query is None else self.query.as_dict(),
            "current": self.current,
            "query_errno": self.query_errno,
            "query_errno_name": self.query_errno_name,
            "value_errno": self.value_errno,
            "value_errno_name": self.value_errno_name,
            "detail": self.detail,
            "query_attempted": self.query_attempted,
            "value_attempted": self.value_attempted,
        }


class InventoryStatus(enum.StrEnum):
    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    REFUSED = "refused"
    DEVICE_ABSENT = "device_absent"
    IDENTITY_DRIFT = "identity_drift"
    OPEN_ERROR = "open_error"
    INTERRUPTED = "interrupted"
    IOCTL_ERROR = "ioctl_error"
    CLOSE_ERROR = "close_error"


INVENTORY_EXIT_CODES: Final[Mapping[InventoryStatus, int]] = MappingProxyType(
    {
        InventoryStatus.NOT_ATTEMPTED: 0,
        InventoryStatus.COMPLETE: 0,
        InventoryStatus.REFUSED: 10,
        InventoryStatus.DEVICE_ABSENT: 2,
        InventoryStatus.IDENTITY_DRIFT: 8,
        InventoryStatus.OPEN_ERROR: 15,
        InventoryStatus.INTERRUPTED: 13,
        InventoryStatus.IOCTL_ERROR: 5,
        InventoryStatus.CLOSE_ERROR: 9,
    }
)


@dataclass(frozen=True, slots=True)
class InventoryStep:
    step: str
    ok: bool
    detail: str
    errno: int | None = None
    errno_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "ok": self.ok,
            "detail": self.detail,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


@dataclass(frozen=True, slots=True)
class InventoryCleanup:
    attempted: bool
    ok: bool
    steps: tuple[InventoryStep, ...]
    deferred_signals: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "steps": [step.as_dict() for step in self.steps],
            "deferred_signals": list(self.deferred_signals),
        }


@dataclass(frozen=True, slots=True)
class InventoryOutcome:
    status: InventoryStatus
    detail: str
    plan: dict[str, object]
    target: NodeSnapshot | None = None
    observations: tuple[ControlObservation, ...] = ()
    steps: tuple[InventoryStep, ...] = ()
    cleanup: InventoryCleanup = field(default_factory=lambda: InventoryCleanup(False, True, ()))
    post_close_identity: dict[str, object] | None = None
    ioctl_count: int = 0

    @property
    def exit_code(self) -> int:
        if not self.cleanup.ok:
            return INVENTORY_EXIT_CODES[InventoryStatus.CLOSE_ERROR]
        return INVENTORY_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        return self.status is InventoryStatus.COMPLETE and self.cleanup.ok

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "v4l2-rgb-control-inventory",
            "status": str(self.status),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "plan": self.plan,
            "target": None if self.target is None else self.target.as_dict(),
            "controls": [observation.as_dict() for observation in self.observations],
            "ioctl_count": self.ioctl_count,
            "max_ioctl_count": MAX_IOCTL_COUNT,
            "steps": [step.as_dict() for step in self.steps],
            "cleanup": self.cleanup.as_dict(),
            "post_close_identity": self.post_close_identity,
        }


def inventory_plan() -> dict[str, object]:
    return {
        "role": "rgb",
        "node_kind": "capture",
        "resolution": "stable USB serial + RGBCam interface string + V4L2 index",
        "open": "read-only, nonblocking, close exactly once",
        "opt_in": RGB_CONTROLS_EXECUTE_OPT_IN,
        "authorized_serial": AUTHORIZED_SERIAL,
        "controls": [spec.as_dict() for spec in CONTROL_WHITELIST],
        "max_query_count": MAX_QUERY_COUNT,
        "max_value_count": MAX_VALUE_COUNT,
        "max_ioctl_count": MAX_IOCTL_COUNT,
        "ioctls": {name: f"0x{value:08x}" for name, value in ALLOWED_CONTROL_IOCTLS.items()},
        "prohibited": [
            "all control writes",
            "menu enumeration",
            "format negotiation",
            "buffer allocation, mapping, queueing or streaming",
            "poll/dequeue and depth access",
            "vendor extension, usbfs, HID and XU operations",
        ],
    }


def _errno(exc: OSError) -> tuple[int | None, str | None]:
    number = exc.errno
    return number, None if number is None else errno_mod.errorcode.get(number, f"errno {number}")


def _default_opener(path: str) -> int:
    return os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)


def _default_ioctl(fd: int, request: int, payload: bytearray, mutate: bool) -> int:
    return int(fcntl.ioctl(fd, request, payload, mutate))


def _supported_type(control_type: int) -> bool:
    return control_type in {
        V4L2_CTRL_TYPE_INTEGER,
        V4L2_CTRL_TYPE_BOOLEAN,
        V4L2_CTRL_TYPE_MENU,
        V4L2_CTRL_TYPE_INTEGER_MENU,
    }


def _skip_status(metadata: QueryMetadata) -> tuple[ControlObservationStatus, str] | None:
    checks = (
        (V4L2_CTRL_FLAG_DISABLED, ControlObservationStatus.DISABLED, "disabled"),
        (V4L2_CTRL_FLAG_WRITE_ONLY, ControlObservationStatus.WRITE_ONLY, "write-only"),
    )
    for bit, status, label in checks:
        if metadata.flags & bit:
            return status, f"{metadata.name or 'control'} is {label}; no current-value read"
    if not _supported_type(metadata.control_type):
        return ControlObservationStatus.UNKNOWN_TYPE, (
            f"control type {metadata.type_name} is not a supported scalar/menu type; "
            "no current-value read"
        )
    return None


def _read_status(metadata: QueryMetadata) -> ControlObservationStatus:
    """Classify readable flags without mistaking them for write restrictions."""
    if metadata.flags & V4L2_CTRL_FLAG_READ_ONLY:
        return ControlObservationStatus.READ_ONLY
    if metadata.flags & V4L2_CTRL_FLAG_VOLATILE:
        return ControlObservationStatus.VOLATILE
    if metadata.flags & V4L2_CTRL_FLAG_INACTIVE:
        return ControlObservationStatus.INACTIVE
    return ControlObservationStatus.SUPPORTED


IdentityCheck = Callable[[int, NodeSnapshot], str | None]
Opener = Callable[[str], int]
Ioctl = Callable[[int, int, bytearray, bool], int]
Closer = Callable[[int], None]
PostCloseCheck = Callable[[NodeSnapshot], dict[str, object]]
Preflight = Callable[[str, Path, Path], tuple[NodeSnapshot, ScannerDevice]]
Resolver = Callable[..., tuple[NodeSnapshot, ScannerDevice]]
TextReader = Callable[[Path], str]
StatReader = Callable[[Path], os.stat_result]


def _identity_check(fd: int, target: NodeSnapshot) -> str | None:
    return identity_drift(fd, target, moment="the RGB control inventory")


def _default_preflight(
    serial: str, sysfs_root: Path, dev_root: Path
) -> tuple[NodeSnapshot, ScannerDevice]:
    return preflight_capture_node(
        serial, profile=RGB_PROFILE, sysfs_root=sysfs_root, dev_root=dev_root
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _stat(path: Path) -> os.stat_result:
    return path.stat()


def _post_close_identity(
    target: NodeSnapshot,
    *,
    sysfs_root: Path = DEFAULT_SYSFS_ROOT,
    dev_root: Path = DEFAULT_DEV_ROOT,
    resolver: Resolver = resolve_capture_node,
    text_reader: TextReader = _read_text,
    stat_reader: StatReader = _stat,
) -> dict[str, object]:
    try:
        resolved, device = resolver(
            target.serial,
            profile=RGB_PROFILE,
            sysfs_root=sysfs_root,
            dev_root=dev_root,
        )
        before = target.as_dict()
        after = resolved.as_dict()
        fields = (
            "serial",
            "sysfs_path",
            "usb_path",
            "bus_number",
            "device_number",
            "role",
            "interface_number",
            "interface_string",
            "interface_driver",
            "v4l_index",
            "v4l_kind",
            "v4l_name",
            "node_name",
            "node_path",
            "rdev_major_minor",
            "inode",
            "mode",
        )
        comparisons = {
            field: {
                "before": before.get(field),
                "after": after.get(field),
                "equal": before.get(field) == after.get(field),
            }
            for field in fields
        }
        interface_dir = (
            Path(resolved.sysfs_path)
            / f"{Path(resolved.sysfs_path).name}:1.{resolved.interface_number}"
        )
        interface_text = text_reader(interface_dir / "interface")
        text_matches = interface_text == resolved.interface_string
        node_stat = stat_reader(Path(resolved.node_path))
        stat_matches = {
            "inode": node_stat.st_ino == resolved.inode,
            "mode": oct(node_stat.st_mode & 0o7777) == resolved.mode,
            "rdev": node_stat.st_rdev == resolved.rdev,
        }
        driver_ok = resolved.interface_driver == "uvcvideo" and device.function("rgb") is not None
        stable = (
            all(item["equal"] for item in comparisons.values())
            and text_matches
            and all(stat_matches.values())
            and driver_ok
        )
        return {
            "stable": stable,
            "expected_driver": "uvcvideo",
            "driver_ok": driver_ok,
            "comparisons": comparisons,
            "interface_text": {
                "expected": resolved.interface_string,
                "observed": interface_text,
                "equal": text_matches,
            },
            "node_stat": {
                "expected": {
                    "inode": resolved.inode,
                    "mode": resolved.mode,
                    "rdev": resolved.rdev,
                },
                "equal": stat_matches,
            },
            "detail": "resolved RGB identity and uvcvideo binding unchanged"
            if stable
            else "resolved RGB identity or uvcvideo binding changed",
        }
    except BaseException as exc:
        return {"stable": False, "detail": f"post-close re-resolution failed: {exc}"}


_inventory_signal_guard: Final[SignalGuard] = partial(
    cleanup_on_termination,
    "the RGB control inventory",
)


@dataclass(frozen=True, slots=True)
class V4l2RgbControlBackend:
    """One fixed read-only RGB control inventory with injectable syscalls."""

    opener: Opener = _default_opener
    ioctl_fn: Ioctl = _default_ioctl
    closer: Closer = os.close
    identity_check: IdentityCheck = _identity_check
    post_close_check: PostCloseCheck = _post_close_identity
    preflight_fn: Preflight = _default_preflight
    resolver: Resolver = resolve_capture_node
    text_reader: TextReader = _read_text
    stat_reader: StatReader = _stat
    sysfs_root: Path = DEFAULT_SYSFS_ROOT
    dev_root: Path = DEFAULT_DEV_ROOT
    signal_guard: SignalGuard = _inventory_signal_guard
    termination_shield: TerminationShield = defer_termination

    def dry_run(self, serial: str | None = None) -> InventoryOutcome:
        """Describe the fixed plan without discovery/open/ioctl side effects."""
        return InventoryOutcome(
            InventoryStatus.NOT_ATTEMPTED,
            "dry run: no RGB node was opened and no ioctl was issued",
            inventory_plan(),
        )

    def preflight(self, serial: str) -> InventoryOutcome:
        """Resolve and validate the RGB node without opening it."""
        ensure_layout_intact()
        try:
            target, _device = self.preflight_fn(serial, self.sysfs_root, self.dev_root)
        except DeviceAbsentError as exc:
            return InventoryOutcome(InventoryStatus.DEVICE_ABSENT, str(exc), inventory_plan())
        except V4l2CaptureNotAllowed as exc:
            return InventoryOutcome(InventoryStatus.REFUSED, str(exc), inventory_plan())
        return InventoryOutcome(
            InventoryStatus.NOT_ATTEMPTED,
            "preflight passed: no RGB node was opened and no ioctl was issued",
            inventory_plan(),
            target=target,
        )

    def run(self, serial: str) -> InventoryOutcome:
        """Perform the one fixed query/get sequence, then close once."""
        ensure_layout_intact()
        plan = inventory_plan()
        if serial != AUTHORIZED_SERIAL:
            return InventoryOutcome(
                InventoryStatus.REFUSED,
                f"this inventory is authorized for serial {AUTHORIZED_SERIAL!r} only; "
                f"refusing {serial!r}",
                plan,
            )
        try:
            target, _device = self.preflight_fn(serial, self.sysfs_root, self.dev_root)
        except DeviceAbsentError as exc:
            return InventoryOutcome(InventoryStatus.DEVICE_ABSENT, str(exc), plan)
        except V4l2CaptureNotAllowed as exc:
            return InventoryOutcome(InventoryStatus.REFUSED, str(exc), plan)

        steps: list[InventoryStep] = [
            InventoryStep(
                "resolve_node",
                True,
                f"{target.node_path} is the resolved RGB capture node at V4L2 index "
                f"{target.v4l_index}",
            )
        ]
        observations: list[ControlObservation] = []
        fd: int | None = None
        cleanup_steps: list[InventoryStep] = []
        deferred: list[int] = []
        primary = InventoryStatus.COMPLETE
        detail = "all fixed RGB control metadata queries completed"
        post_close: dict[str, object] | None = None
        try:
            with self.signal_guard():
                try:
                    fd = self.opener(target.node_path)
                    steps.append(
                        InventoryStep(
                            "open", True, f"opened {target.node_path} read-only as fd {fd}"
                        )
                    )
                    drift = self.identity_check(fd, target)
                    if drift is not None:
                        primary = InventoryStatus.IDENTITY_DRIFT
                        detail = drift
                        steps.append(InventoryStep("identity", False, drift))
                    else:
                        steps.append(
                            InventoryStep(
                                "identity", True, "post-open RGB identity matches resolution"
                            )
                        )
                        for spec in CONTROL_WHITELIST:
                            observation = self._inspect_one(fd, spec)
                            observations.append(observation)
                            if observation.status in {
                                ControlObservationStatus.QUERY_ERROR,
                                ControlObservationStatus.VALUE_ERROR,
                            }:
                                primary = InventoryStatus.IOCTL_ERROR
                                detail = "one or more fixed control ioctls failed"
                            if observation.status is ControlObservationStatus.INTERRUPTED:
                                primary = InventoryStatus.INTERRUPTED
                                detail = observation.detail
                                break
                except SignalReceived as exc:
                    primary = InventoryStatus.INTERRUPTED
                    detail = str(exc)
                except OSError as exc:
                    number, name = _errno(exc)
                    primary = (
                        InventoryStatus.OPEN_ERROR if fd is None else InventoryStatus.IOCTL_ERROR
                    )
                    detail = str(exc)
                    steps.append(
                        InventoryStep(
                            "open" if fd is None else "ioctl", False, str(exc), number, name
                        )
                    )
                except Exception as exc:
                    primary = InventoryStatus.IOCTL_ERROR
                    detail = f"unexpected {type(exc).__name__}: {exc}"
                    steps.append(InventoryStep("inventory", False, detail))
                finally:
                    if fd is not None:
                        cleanup_steps, deferred = self._close_once(fd)
                        post_close = self._post_close_safe(target)
        except SignalReceived as exc:
            primary = InventoryStatus.INTERRUPTED
            detail = str(exc)
            if fd is not None and not cleanup_steps:
                cleanup_steps, deferred = self._close_once(fd)
                post_close = self._post_close_safe(target)

        cleanup = InventoryCleanup(
            attempted=fd is not None,
            ok=all(step.ok for step in cleanup_steps),
            steps=tuple(cleanup_steps),
            deferred_signals=tuple(deferred),
        )
        if (
            post_close is not None
            and post_close.get("stable") is not True
            and primary is InventoryStatus.COMPLETE
        ):
            primary = InventoryStatus.IDENTITY_DRIFT
            detail = str(post_close.get("detail", "post-close identity changed"))
        return InventoryOutcome(
            primary,
            detail,
            plan,
            target=target,
            observations=tuple(observations),
            steps=tuple(steps),
            cleanup=cleanup,
            post_close_identity=post_close,
            ioctl_count=sum(
                int(observation.query_attempted) + int(observation.value_attempted)
                for observation in observations
            ),
        )

    def _inspect_one(self, fd: int, spec: ControlSpec) -> ControlObservation:
        query_attempted = True
        query = bytearray(pack_queryctrl(spec.control_id))
        try:
            self.ioctl_fn(fd, VIDIOC_QUERYCTRL, query, True)
        except SignalReceived as exc:
            return ControlObservation(
                spec,
                ControlObservationStatus.INTERRUPTED,
                detail=str(exc),
                query_attempted=query_attempted,
            )
        except OSError as exc:
            number, name = _errno(exc)
            if number == errno_mod.EINVAL:
                return ControlObservation(
                    spec,
                    ControlObservationStatus.UNSUPPORTED,
                    query_errno=number,
                    query_errno_name=name,
                    detail="QUERYCTRL returned EINVAL; control is not exposed",
                    query_attempted=query_attempted,
                )
            return ControlObservation(
                spec,
                ControlObservationStatus.QUERY_ERROR,
                query_errno=number,
                query_errno_name=name,
                detail=f"QUERYCTRL failed: {exc}",
                query_attempted=query_attempted,
            )
        try:
            metadata = parse_queryctrl(bytes(query))
        except V4l2CaptureNotAllowed as exc:
            return ControlObservation(
                spec,
                ControlObservationStatus.QUERY_ERROR,
                detail=str(exc),
                query_attempted=query_attempted,
            )
        if metadata.control_id != spec.control_id:
            return ControlObservation(
                spec,
                ControlObservationStatus.QUERY_ERROR,
                query=metadata,
                detail=(
                    f"QUERYCTRL returned id 0x{metadata.control_id:08x}, "
                    f"expected 0x{spec.control_id:08x}"
                ),
                query_attempted=query_attempted,
            )
        skip = _skip_status(metadata)
        if skip is not None:
            status, detail = skip
            return ControlObservation(
                spec, status, query=metadata, detail=detail, query_attempted=True
            )

        value = bytearray(pack_control(spec.control_id))
        try:
            self.ioctl_fn(fd, VIDIOC_G_CTRL, value, True)
        except SignalReceived as exc:
            return ControlObservation(
                spec,
                ControlObservationStatus.INTERRUPTED,
                query=metadata,
                detail=str(exc),
                query_attempted=True,
                value_attempted=True,
            )
        except OSError as exc:
            number, name = _errno(exc)
            return ControlObservation(
                spec,
                ControlObservationStatus.VALUE_ERROR,
                query=metadata,
                value_errno=number,
                value_errno_name=name,
                detail=f"G_CTRL failed: {exc}",
                query_attempted=True,
                value_attempted=True,
            )
        try:
            returned_id, current = parse_control(bytes(value))
        except V4l2CaptureNotAllowed as exc:
            return ControlObservation(
                spec,
                ControlObservationStatus.VALUE_ERROR,
                query=metadata,
                detail=str(exc),
                query_attempted=True,
                value_attempted=True,
            )
        if returned_id != spec.control_id:
            return ControlObservation(
                spec,
                ControlObservationStatus.VALUE_ERROR,
                query=metadata,
                detail=f"G_CTRL returned id 0x{returned_id:08x}, expected 0x{spec.control_id:08x}",
                query_attempted=True,
                value_attempted=True,
            )
        return ControlObservation(
            spec,
            _read_status(metadata),
            query=metadata,
            current=current,
            detail="metadata queried and one current value read; flags remain device-reported",
            query_attempted=True,
            value_attempted=True,
        )

    def _post_close_safe(self, target: NodeSnapshot) -> dict[str, object]:
        try:
            if self.post_close_check is _post_close_identity:
                return _post_close_identity(
                    target,
                    sysfs_root=self.sysfs_root,
                    dev_root=self.dev_root,
                    resolver=self.resolver,
                    text_reader=self.text_reader,
                    stat_reader=self.stat_reader,
                )
            return self.post_close_check(target)
        except BaseException as exc:
            return {"stable": False, "detail": f"post-close identity check failed: {exc}"}

    def _close_once(self, fd: int) -> tuple[list[InventoryStep], list[int]]:
        steps: list[InventoryStep] = []
        deferred: list[int] = []

        finished = False

        def run_close() -> None:
            nonlocal finished
            if finished:
                return
            try:
                self.closer(fd)
            except OSError as exc:
                number, name = _errno(exc)
                steps.append(InventoryStep("close", False, f"close failed: {exc}", number, name))
            except BaseException as exc:
                steps.append(
                    InventoryStep("close", False, f"close raised {type(exc).__name__}: {exc}")
                )
            else:
                steps.append(InventoryStep("close", True, f"fd {fd} closed exactly once"))
            finally:
                finished = True

        def shielded_close() -> list[int]:
            with self.termination_shield() as received:
                run_close()
                return list(received)

        try:
            deferred.extend(shielded_close())
        except SignalReceived as exc:
            deferred.append(exc.signum)
            run_close()
        except BaseException as exc:
            steps.append(
                InventoryStep(
                    "close_guard", False, f"cleanup guard raised {type(exc).__name__}: {exc}"
                )
            )
            run_close()
        return steps, deferred


def render_human(outcome: InventoryOutcome) -> str:
    """Render the same fields exposed by :meth:`InventoryOutcome.as_dict`."""
    lines = ["scanner3d rgb-controls", "", "Plan (fixed; read-only; no stream)"]
    lines.append(f"  opt-in      {RGB_CONTROLS_EXECUTE_OPT_IN}")
    lines.append(f"  serial      {AUTHORIZED_SERIAL}")
    lines.append(f"  ioctls      QUERYCTRL then optional G_CTRL, max {MAX_IOCTL_COUNT}")
    lines.append("  controls    " + ", ".join(spec.name for spec in CONTROL_WHITELIST))
    lines.append("")
    if outcome.target is not None:
        lines.append(
            f"Target: {outcome.target.node_path} (resolved RGB index {outcome.target.v4l_index})"
        )
    for observation in outcome.observations:
        current = "n/a" if observation.current is None else str(observation.current)
        lines.append(
            f"  {observation.spec.name}: {observation.status} current={current} - "
            f"{observation.detail}"
        )
        if observation.query is not None:
            q = observation.query
            lines.append(
                f"    {q.name!r} {q.type_name} min={q.minimum} max={q.maximum} "
                f"step={q.step} default={q.default} flags=0x{q.flags:08x}"
            )
    lines.extend(
        [
            "",
            f"Status: {outcome.status} - {outcome.detail}",
            f"IOCTLs: {outcome.ioctl_count}/{MAX_IOCTL_COUNT}",
            f"Cleanup: {'ok' if outcome.cleanup.ok else 'FAILED'}",
            f"Post-close identity: {outcome.post_close_identity}",
        ]
    )
    return "\n".join(lines) + "\n"
