"""One bounded read of the factory depth intrinsics key ``/data/camparam/Pl.bin``.

This is the second thing in this repository that writes to the scanner, and it
writes exactly what :mod:`scanner3d.init_sequence` already wrote once, with one
byte-string changed: the key. The whole safety argument is that the module is a
*closed three-transfer whitelist*, not a file reader:

===== ============= ================================== ==========================
Step  Direction     Setup packet                       Payload
===== ============= ================================== ==========================
1     host->device  ``21 01`` value ``0x0100`` index    ``01`` +
                    ``0x0400`` length 60, 1000 ms       ``"/data/camparam/Pl.bin"``
                                                        + NUL, zero-padded to 60
2..n  device->host  ``a1 81`` value ``0x0200`` index    ``uint32_le`` chunk length
                    ``0x0400`` length 60, 1000 ms       then that many bytes
last  host->device  the same setup as step 1            60 zero bytes
===== ============= ================================== ==========================

Every one of those numbers is recovered from the disassembly of
``cs::libuvcReadFile`` ``0x262861`` (``docs/architecture.md``, "Domain 4 -
calibration / depth scale"), and the identical shape - same entity, same
selectors, same lengths, same cleanup write - was already driven successfully
against this exact unit for ``/tmp/inited``. The key is the single new variable.

What cannot be expressed here, by construction:

* **no caller-supplied path.** :data:`INTRINSICS_KEY` is an ASCII constant and
  :data:`PATH_PAYLOAD` is derived from it at import time. No function in this
  module takes a path, a selector, an entity, an interface, a length, a
  timeout, a cap or a capacity. The sibling calibration keys (``Prgb``,
  ``LC_RT``, ``Distort``), the licence paths and the write-file opcode are not
  here and cannot be reached from here.
* **no unbounded loop.** The vendor's loop stops only on a zero length word or
  the caller's capacity. This one additionally stops after
  :data:`MAX_GET_TRANSFERS` = 8 result reads, so a run is at most
  :data:`MAX_CONTROL_TRANSFERS` = 10 control transfers: one path write, at most
  eight reads, one clearing write. Neither bound is widened at runtime; a run
  that hits one reports a bounded, explicitly incomplete result and stops.
* **no truncation.** A chunk that would pass the :data:`OUTPUT_CAPACITY`
  256-byte capacity is not copied at all, and a chunk that declares more than
  :data:`MAX_CHUNK_LENGTH` = 56 bytes - more than fits behind the 4-byte header
  - is rejected before anything is copied.
* **no retry.** Each transfer is attempted at most once, whatever it returns.

The clearing write is protocol cleanup on the same terms
:class:`scanner3d.ownership.InterfaceZeroLifecycle` already enforces: owed from
immediately *before* the path write's ioctl (a signal delivered between that
line and the kernel's completion leaves the write ambiguous), attempted exactly
once, from inside the termination shield, before the driver handback, and never
reported as success when it fails. The two cleanups stay independent records and
either failure outranks a perfect read.

The 40-byte vendor record is parsed, never trusted blindly: the assembled blob,
its length and its SHA-256 are the primary evidence, the parse is reported
alongside them, and a blob that is not exactly 40 bytes or whose fixed structure
diverges is typed evidence - not a crash, and not accepted calibration.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scanner3d.config import configured_serial, stated_serial
from scanner3d.control import (
    MAX_TIMEOUT_MS,
    USBDEVFS_CONTROL,
    Clock,
    ControlRequest,
    IoctlFn,
    TargetSnapshot,
    pack_ctrltransfer,
    resolve_target,
)
from scanner3d.discovery import DEFAULT_DEV_ROOT, DEFAULT_SYSFS_ROOT, ScannerDevice
from scanner3d.errors import DeviceAbsentError, IntrinsicsReadNotAllowed
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

#: The one device-file key this module can name. Not an argument anywhere.
INTRINSICS_KEY: Final = "/data/camparam/Pl.bin"

#: The serial of the one unit the owner authorized this read for. The live path
#: refuses any other, so "the exact device" is not a runtime judgement call.
#: The scanner this installation may act on. Read from configuration rather
#: than hard-coded: several operations here detach kernel drivers or write
#: vendor property keys, and must not be aimed at whatever happens to be
#: plugged in. Empty when nothing is configured - call
#: scanner3d.config.resolve_serial() for a usable value, which falls back to
#: the single attached scanner and refuses when that is ambiguous.
AUTHORIZED_SERIAL: Final = stated_serial()

#: The leading byte of a key payload: the read-file opcode, and the only opcode
#: this module knows. The write-file opcode is not here.
READ_FILE_OPCODE: Final = 0x01

#: Every transfer in the recovered protocol is exactly this long.
PAYLOAD_LENGTH: Final = 60

#: ``wIndex`` decodes as ``(entity << 8) | interface``.
ENTITY_ID: Final = 4

#: ``wValue`` decodes as ``(selector << 8)``.
PATH_SELECTOR: Final = 0x01
RESULT_SELECTOR: Final = 0x02

#: The timeout the vendor function passes, in milliseconds (``e8 03`` on the
#: wire). Well under the :data:`scanner3d.control.MAX_TIMEOUT_MS` ceiling.
TIMEOUT_MS: Final = 1000

#: A reply is a ``uint32_le`` length word followed by that many data bytes.
CHUNK_HEADER_LENGTH: Final = 4

#: The most data that fits behind the header. A longer declared length is
#: malformed - the vendor merely warns; this module stops.
MAX_CHUNK_LENGTH: Final = PAYLOAD_LENGTH - CHUNK_HEADER_LENGTH

#: The fixed output capacity, 5x the largest calibration struct in the vendor's
#: ``Types.hpp``. Never widened at runtime.
OUTPUT_CAPACITY: Final = 256

#: The fixed ceiling on result reads. Never widened at runtime.
MAX_GET_TRANSFERS: Final = 8

#: One path write, at most eight reads, one clearing write.
MAX_CONTROL_TRANSFERS: Final = MAX_GET_TRANSFERS + 2

#: ``Intrinsics`` in the vendor's ``Types.hpp``: ``short width, height;`` then
#: nine x86 little-endian IEEE-754 floats.
INTRINSICS_FORMAT: Final = "<hh9f"
INTRINSICS_LENGTH: Final = struct.calcsize(INTRINSICS_FORMAT)

#: The nine floats, in the order the struct declares them.
INTRINSICS_FIELDS: Final[tuple[str, ...]] = (
    "fx",
    "zero01",
    "cx",
    "zero10",
    "fy",
    "cy",
    "zero20",
    "zero21",
    "one22",
)

#: How far from the frame centre a principal point may sit before it is worth
#: remarking on, as a fraction of the frame size. Purely an observation
#: threshold: a factory record for a cropped or offset sensor window can fail
#: it and still be exactly what the device meant to return.
CENTRE_TOLERANCE: Final = 0.1

#: The record's structurally fixed entries: everything but ``fx/fy/cx/cy``.
_FIXED_ENTRIES: Final[tuple[tuple[str, float], ...]] = (
    ("zero01", 0.0),
    ("zero10", 0.0),
    ("zero20", 0.0),
    ("zero21", 0.0),
    ("one22", 1.0),
)


def _padded(prefix: bytes) -> bytes:
    """A 60-byte payload: ``prefix`` then zero padding, as the vendor builds it."""
    if len(prefix) > PAYLOAD_LENGTH:
        raise IntrinsicsReadNotAllowed(
            f"payload prefix is {len(prefix)} bytes, over the fixed {PAYLOAD_LENGTH}"
        )
    return prefix + bytes(PAYLOAD_LENGTH - len(prefix))


#: ``01`` + the key + NUL + zero padding. Built from the constant, once.
PATH_PAYLOAD: Final = _padded(bytes([READ_FILE_OPCODE]) + INTRINSICS_KEY.encode("ascii") + b"\x00")

#: The clearing payload, and the only other thing this module can write.
ZERO_PAYLOAD: Final = bytes(PAYLOAD_LENGTH)


# --- the closed whitelist -----------------------------------------------------


class Step(enum.StrEnum):
    """The three distinct transfers, in the only order they can occur."""

    PATH_WRITE = "path_write"
    CHUNK_READ = "chunk_read"
    CLEAR_WRITE = "clear_write"


@dataclass(frozen=True, slots=True)
class FileReadTransfer:
    """One fully specified transfer of the recovered file-read protocol.

    Constructing one is harmless; only members of :data:`WHITELIST` can be
    handed to the hardware path, and :func:`ensure_transfer_allowed` checks that
    by value - the setup packet *and* the outbound bytes - not by name.
    """

    step: Step
    request: ControlRequest
    payload: bytes | None
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
            "direction": "in" if self.request.is_read else "out",
            "request": self.request.as_dict(),
            "payload_hex": None if self.payload is None else self.payload.hex(),
            "note": self.note,
        }


_PATH_WRITE_REQUEST: Final = ControlRequest(
    name="intrinsics-key-set-cur",
    request_type=0x21,
    request=0x01,
    value=PATH_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=PAYLOAD_LENGTH,
    timeout_ms=TIMEOUT_MS,
    description="UVC SET_CUR, entity 4 selector 0x01 on interface 0, 60-byte key payload",
)

_RESULT_READ_REQUEST: Final = ControlRequest(
    name="intrinsics-chunk-get-cur",
    request_type=0xA1,
    request=0x81,
    value=RESULT_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=PAYLOAD_LENGTH,
    timeout_ms=TIMEOUT_MS,
    description="UVC GET_CUR, entity 4 selector 0x02 on interface 0, 60-byte chunk reply",
)

PATH_WRITE_TRANSFER: Final = FileReadTransfer(
    step=Step.PATH_WRITE,
    request=_PATH_WRITE_REQUEST,
    payload=PATH_PAYLOAD,
    note=f"read-file opcode 0x01 followed by {INTRINSICS_KEY!r}, NUL-terminated",
)

CHUNK_READ_TRANSFER: Final = FileReadTransfer(
    step=Step.CHUNK_READ,
    request=_RESULT_READ_REQUEST,
    payload=None,
    note="one length-prefixed chunk; a zero length word ends the data",
)

CLEAR_WRITE_TRANSFER: Final = FileReadTransfer(
    step=Step.CLEAR_WRITE,
    request=_PATH_WRITE_REQUEST,
    payload=ZERO_PAYLOAD,
    note="protocol cleanup: the vendor clears the key selector with 60 zero bytes",
)

#: The whole whitelist. Nothing outside this tuple can be issued.
WHITELIST: Final[tuple[FileReadTransfer, ...]] = (
    PATH_WRITE_TRANSFER,
    CHUNK_READ_TRANSFER,
    CLEAR_WRITE_TRANSFER,
)

_ALLOWED_IDENTITIES: Final = frozenset(t.identity for t in WHITELIST)


def ensure_whitelist_intact(whitelist: tuple[FileReadTransfer, ...] = WHITELIST) -> None:
    """Check the whitelist as a whole, before anything is resolved or opened.

    Guards the shape as well as the contents: exactly the three transfers, in
    order, so a whitelist rebuilt from allowed parts cannot be reordered,
    extended or truncated.
    """
    if whitelist != WHITELIST:
        raise IntrinsicsReadNotAllowed(
            "only the three-transfer /data/camparam/Pl.bin whitelist may be issued; "
            f"got {[str(t.step) for t in whitelist]}"
        )
    for transfer in whitelist:
        ensure_transfer_allowed(transfer)


def ensure_transfer_allowed(transfer: FileReadTransfer) -> FileReadTransfer:
    """Return ``transfer`` if it is one of the three, else raise.

    Called again immediately before each ioctl argument is packed, so a
    transfer that reached the hardware path some other way still never gets
    encoded.
    """
    if transfer.request.timeout_ms <= 0 or transfer.request.timeout_ms > MAX_TIMEOUT_MS:
        raise IntrinsicsReadNotAllowed(
            f"{transfer.step}: timeout {transfer.request.timeout_ms} ms "
            f"outside (0, {MAX_TIMEOUT_MS}]"
        )
    if transfer.request.length != PAYLOAD_LENGTH:
        raise IntrinsicsReadNotAllowed(
            f"{transfer.step}: wLength {transfer.request.length} is not {PAYLOAD_LENGTH}"
        )
    if transfer.request.interface_number != OWNED_INTERFACE:
        raise IntrinsicsReadNotAllowed(
            f"{transfer.step}: wIndex names interface "
            f"{transfer.request.interface_number}, not {OWNED_INTERFACE}"
        )
    if transfer.request.entity_id != ENTITY_ID:
        raise IntrinsicsReadNotAllowed(
            f"{transfer.step}: wIndex names entity {transfer.request.entity_id}, not {ENTITY_ID}"
        )
    if transfer.payload is not None and len(transfer.payload) != PAYLOAD_LENGTH:
        raise IntrinsicsReadNotAllowed(
            f"{transfer.step}: payload is {len(transfer.payload)} bytes, not {PAYLOAD_LENGTH}"
        )
    if transfer.identity not in _ALLOWED_IDENTITIES:
        raise IntrinsicsReadNotAllowed(
            f"{transfer.step}: setup packet {transfer.request.wire_tuple} with this payload "
            "is not one of the three whitelisted transfers"
        )
    return transfer


# --- the record ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordCheck:
    """One named observation about the parsed record. Never a mutation."""

    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class IntrinsicsRecord:
    """The 40-byte vendor ``Intrinsics`` struct, decoded and checked.

    Values are reported exactly as they arrived: nothing here normalizes,
    substitutes or repairs a field. ``structure`` says whether the fixed
    entries of the 3x3 really were 0/0/0/0/1 and every float finite;
    ``plausibility`` says whether the free entries look like a pinhole model of
    a frame the device advertises. The two are separate on purpose - a record
    can be structurally exact and implausible, or plausible and structurally
    wrong, and neither is a reason to alter it.
    """

    width: int
    height: int
    values: Mapping[str, float]
    structure: tuple[RecordCheck, ...]
    plausibility: tuple[RecordCheck, ...]

    @property
    def matrix(self) -> tuple[tuple[float, float, float], ...]:
        """The implied 3x3 pinhole matrix, built from the read floats alone."""
        v = self.values
        return (
            (v["fx"], v["zero01"], v["cx"]),
            (v["zero10"], v["fy"], v["cy"]),
            (v["zero20"], v["zero21"], v["one22"]),
        )

    @property
    def all_finite(self) -> bool:
        return all(math.isfinite(value) for value in self.values.values())

    @property
    def structurally_expected(self) -> bool:
        """True only when every structural check passed, finiteness included."""
        return all(check.ok for check in self.structure)

    @property
    def plausible(self) -> bool:
        return all(check.ok for check in self.plausibility)

    def as_dict(self) -> dict[str, object]:
        return {
            "layout": (
                f"vendor Types.hpp Intrinsics, {INTRINSICS_LENGTH} bytes, {INTRINSICS_FORMAT}"
            ),
            "width": self.width,
            "height": self.height,
            "units": "fx, fy, cx and cy are in pixels of the width x height frame above",
            "values": dict(self.values),
            "matrix": [list(row) for row in self.matrix],
            "all_finite": self.all_finite,
            "structurally_expected": self.structurally_expected,
            "plausible": self.plausible,
            "structure_checks": [c.as_dict() for c in self.structure],
            "plausibility_checks": [c.as_dict() for c in self.plausibility],
        }


def _structure_checks(values: Mapping[str, float]) -> tuple[RecordCheck, ...]:
    """Finiteness and the five entries the layout fixes. Never raises."""
    checks = [
        RecordCheck(
            name="all_floats_finite",
            ok=all(math.isfinite(value) for value in values.values()),
            detail=", ".join(f"{name}={values[name]!r}" for name in INTRINSICS_FIELDS),
        )
    ]
    for name, expected in _FIXED_ENTRIES:
        actual = values[name]
        checks.append(
            RecordCheck(
                name=f"{name}_is_{expected:g}",
                ok=actual == expected,
                detail=f"{name}={actual!r}, layout fixes it at {expected:g}",
            )
        )
    return tuple(checks)


def _plausibility_checks(
    width: int, height: int, values: Mapping[str, float], depth_modes: Sequence[tuple[int, int]]
) -> tuple[RecordCheck, ...]:
    """Shape-only sanity, and the descriptor comparison. Claims no accuracy."""
    fx, fy, cx, cy = values["fx"], values["fy"], values["cx"], values["cy"]
    modes = tuple(depth_modes)
    checks = [
        RecordCheck(
            name="positive_dimensions",
            ok=width > 0 and height > 0,
            detail=f"width={width}, height={height}",
        ),
        RecordCheck(
            name="positive_focal_lengths",
            ok=math.isfinite(fx) and math.isfinite(fy) and fx > 0 and fy > 0,
            detail=f"fx={fx!r}, fy={fy!r}",
        ),
        RecordCheck(
            name="principal_point_inside_frame",
            ok=(
                math.isfinite(cx)
                and math.isfinite(cy)
                and 0 < cx < width
                and 0 < cy < height
                and width > 0
                and height > 0
            ),
            detail=f"cx={cx!r}, cy={cy!r} against {width}x{height}",
        ),
        RecordCheck(
            name="principal_point_near_frame_centre",
            ok=(
                math.isfinite(cx)
                and math.isfinite(cy)
                and width > 0
                and height > 0
                and abs(cx - width / 2) <= width * CENTRE_TOLERANCE
                and abs(cy - height / 2) <= height * CENTRE_TOLERANCE
            ),
            detail=(
                f"cx sits {cx - width / 2:+.2f} px from the {width / 2:g} px centre and cy "
                f"{cy - height / 2:+.2f} px from the {height / 2:g} px centre, against a "
                f"{CENTRE_TOLERANCE:.0%} tolerance. This is an observation, not a defect: a "
                "factory record for a cropped or offset sensor window legitimately fails it, "
                "and nothing here adjusts the values either way"
            ),
        ),
        RecordCheck(
            name="dimensions_match_an_advertised_depth_mode",
            ok=(width, height) in modes,
            detail=(
                f"{width}x{height} against descriptor-advertised depth modes "
                + (", ".join(f"{w}x{h}" for w, h in modes) if modes else "(none read)")
                + "; a mismatch may simply mean the factory record is for the native "
                "sensor frame, and says nothing about accuracy"
            ),
        ),
    ]
    return tuple(checks)


def parse_intrinsics(
    blob: bytes, depth_modes: Sequence[tuple[int, int]] = ()
) -> IntrinsicsRecord | None:
    """Decode a complete 40-byte blob. Returns ``None`` for any other length.

    Never raises and never alters a value: a record whose fixed entries diverge
    still comes back, with the divergence recorded in
    :attr:`IntrinsicsRecord.structure`.
    """
    if len(blob) != INTRINSICS_LENGTH:
        return None
    width, height, *floats = struct.unpack(INTRINSICS_FORMAT, blob)
    values: Mapping[str, float] = MappingProxyType(
        dict(zip(INTRINSICS_FIELDS, floats, strict=True))
    )
    return IntrinsicsRecord(
        width=width,
        height=height,
        values=values,
        structure=_structure_checks(values),
        plausibility=_plausibility_checks(width, height, values, depth_modes),
    )


def advertised_depth_modes(device: ScannerDevice) -> tuple[tuple[int, int], ...]:
    """The ``width x height`` modes the device's own descriptors declare for depth."""
    function = device.function("depth")
    if function is None:
        return ()
    seen: list[tuple[int, int]] = []
    for fmt in function.formats:
        for mode in fmt.modes:
            if (mode.width, mode.height) not in seen:
                seen.append((mode.width, mode.height))
    return tuple(seen)


def read_depth_modes(
    serial: str | None, *, sysfs_root: Path = DEFAULT_SYSFS_ROOT, dev_root: Path = DEFAULT_DEV_ROOT
) -> tuple[tuple[int, int], ...]:
    """Read the advertised depth modes from sysfs. Opens no device node."""
    try:
        _, device = resolve_target(
            serial, _PATH_WRITE_REQUEST, sysfs_root=sysfs_root, dev_root=dev_root
        )
    except DeviceAbsentError:
        return ()
    return advertised_depth_modes(device)


# --- outcomes -----------------------------------------------------------------


class ReadOutcome(enum.StrEnum):
    """How the bounded read phase ended. Only one of these is a full record."""

    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    NO_CONTENT = "no_content"
    CAP_EXHAUSTED = "cap_exhausted"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    MALFORMED_LENGTH = "malformed_length"
    SHORT_TRANSFER = "short_transfer"
    TRANSFER_ERROR = "transfer_error"
    UNEXPECTED_CONTENT = "unexpected_content"
    INTERRUPTED = "interrupted"


#: Exit codes for the read-phase outcomes that have no ownership equivalent.
#: The rest fall through to :data:`scanner3d.ownership.OWNERSHIP_EXIT_CODES`,
#: keeping the meanings the accepted lifecycle already gave them.
READ_OUTCOME_EXIT_CODES: Final[Mapping[ReadOutcome, int]] = MappingProxyType(
    {
        ReadOutcome.NO_CONTENT: 15,
        ReadOutcome.CAP_EXHAUSTED: 16,
        ReadOutcome.CAPACITY_EXHAUSTED: 17,
        ReadOutcome.MALFORMED_LENGTH: 18,
    }
)

#: Which primary status each read outcome maps to. ``transfer_error`` and
#: ``interrupted`` are decided by the errno and the lifecycle instead.
_OUTCOME_STATUS: Final[Mapping[ReadOutcome, OwnershipStatus]] = MappingProxyType(
    {
        ReadOutcome.COMPLETE: OwnershipStatus.OK,
        ReadOutcome.NO_CONTENT: OwnershipStatus.UNEXPECTED_CONTENT,
        ReadOutcome.CAP_EXHAUSTED: OwnershipStatus.UNEXPECTED_CONTENT,
        ReadOutcome.CAPACITY_EXHAUSTED: OwnershipStatus.UNEXPECTED_CONTENT,
        ReadOutcome.MALFORMED_LENGTH: OwnershipStatus.UNEXPECTED_CONTENT,
        ReadOutcome.UNEXPECTED_CONTENT: OwnershipStatus.UNEXPECTED_CONTENT,
        ReadOutcome.SHORT_TRANSFER: OwnershipStatus.SHORT_RESPONSE,
        ReadOutcome.INTERRUPTED: OwnershipStatus.INTERRUPTED,
    }
)


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """One attempted transfer and exactly how it ended. Raw bytes included."""

    step: Step
    index: int
    request: ControlRequest
    sent_sha256: str | None
    transferred: int | None
    ok: bool
    detail: str
    response_sha256: str | None = None
    response_hex: str | None = None
    declared_length: int | None = None
    chunk_sha256: str | None = None
    chunk_hex: str | None = None
    copied_bytes: int | None = None
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": str(self.step),
            "index": self.index,
            "direction": "in" if self.request.is_read else "out",
            "request": self.request.as_dict(),
            "sent_sha256": self.sent_sha256,
            "transferred": self.transferred,
            "ok": self.ok,
            "detail": self.detail,
            "duration_s": self.duration_s,
            "response_sha256": self.response_sha256,
            "response_hex": self.response_hex,
            "declared_length": self.declared_length,
            "chunk_sha256": self.chunk_sha256,
            "chunk_hex": self.chunk_hex,
            "copied_bytes": self.copied_bytes,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


@dataclass(frozen=True, slots=True)
class BlobEvidence:
    """The assembled bytes: the primary evidence, whatever the parse says.

    ``terminated`` is the fact that a zero length word actually arrived, so the
    bytes are the device's whole answer and not a bounded fragment.
    ``complete_answer`` additionally requires both cleanups to have succeeded,
    which is what any reading of the bytes as calibration depends on.
    """

    length: int
    sha256: str
    hex: str
    terminated: bool
    complete_answer: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "sha256": self.sha256,
            "hex": self.hex,
            "terminated": self.terminated,
            "complete_answer": self.complete_answer,
            "note": self.note,
        }


def key_clear_guidance(usb_path: str) -> str:
    """What the owner should know when the clearing write did not land."""
    return (
        f"the clearing SET_CUR to entity {ENTITY_ID} on interface {OWNED_INTERFACE} of "
        f"{usb_path} did not complete, so the device may still hold {INTRINSICS_KEY!r} "
        "selected in its vendor file-read state. Nothing in this repository reads or "
        "writes that state again on its own; power-cycle the scanner before drawing "
        "conclusions from a later run."
    )


# --- the operation ------------------------------------------------------------


@dataclass(slots=True)
class IntrinsicsReadOperation:
    """The bounded key read, plus the clearing write as protocol cleanup.

    Built fresh for each run by :class:`UsbfsIntrinsicsBackend`, which is why it
    may accumulate results. Nothing about the transfers is a field: they come
    from :data:`WHITELIST` and the syscalls are injected.
    """

    ioctl_fn: IoctlFn
    clock: Clock
    usb_path: str
    depth_modes: tuple[tuple[int, int], ...] = ()
    transfers: list[TransferRecord] = field(default_factory=list)
    outcome: ReadOutcome = ReadOutcome.NOT_ATTEMPTED
    blob: bytes = b""
    #: True only when a zero length word actually arrived, so the assembled
    #: bytes are the device's complete answer rather than a bounded fragment.
    terminated: bool = False
    record: IntrinsicsRecord | None = None
    get_transfers: int = 0
    #: Raised immediately *before* the key write's ioctl, because a signal
    #: between the two would leave completion unknowable. The clearing write is
    #: owed for an ambiguous write too.
    key_write_attempted: bool = False
    cleanup_attempted: bool = False
    read_duration_s: float | None = None

    # -- OwnedOperation ------------------------------------------------------

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        """Run the bounded read once. Never retries a transfer."""
        # Pessimistic from the instant the operation is entered: anything that
        # unwinds out of it leaves a typed outcome behind rather than a blank.
        self.outcome = ReadOutcome.INTERRUPTED
        started = self.clock()
        try:
            primary = self._run(fd)
        except BaseException as exc:
            # Recorded, never swallowed: the lifecycle above decides what an
            # escaping exception means for the run, and the clearing write is
            # owed either way. A signal keeps the pessimistic `interrupted`;
            # anything else is a transfer that ended without an errno.
            self.read_duration_s = self.clock() - started
            self.outcome = (
                ReadOutcome.TRANSFER_ERROR
                if isinstance(exc, Exception)
                else ReadOutcome.INTERRUPTED
            )
            raise
        self.read_duration_s = self.clock() - started
        primary.duration_s = self.read_duration_s
        phases.append(
            PhaseResult(
                phase=Phase.OWNED_SEQUENCE,
                ok=primary.status is OwnershipStatus.OK,
                detail=primary.detail,
                duration_s=self.read_duration_s,
                errno=primary.errno,
                errno_name=primary.errno_name,
            )
        )
        return primary

    def protocol_cleanup(self, fd: int) -> ProtocolCleanupRecord:
        """Issue the 60-zero clearing write once, if the key write may have landed."""
        if not self.key_write_attempted:
            return ProtocolCleanupRecord.not_needed(
                f"the key SET_CUR was never issued, so the device holds no {INTRINSICS_KEY!r} "
                "selection to clear"
            )
        if self.cleanup_attempted:
            return ProtocolCleanupRecord(
                attempted=True,
                ok=False,
                note="the clearing write was already attempted; it is never repeated",
                guidance=key_clear_guidance(self.usb_path),
            )
        self.cleanup_attempted = True
        result, _ = self._issue(fd, CLEAR_WRITE_TRANSFER)
        self.transfers.append(result)
        cleared = result.ok and result.transferred == PAYLOAD_LENGTH
        return ProtocolCleanupRecord(
            attempted=True,
            ok=cleared,
            note=(
                f"the {INTRINSICS_KEY!r} selection was cleared with 60 zero bytes"
                if cleared
                else f"the clearing SET_CUR did NOT complete: {result.detail}"
            ),
            duration_s=result.duration_s,
            guidance=None if cleared else key_clear_guidance(self.usb_path),
        )

    # -- the bounded read ----------------------------------------------------

    def _run(self, fd: int) -> _Primary:
        """Key write, then at most :data:`MAX_GET_TRANSFERS` chunk reads."""
        write, _ = self._issue(fd, PATH_WRITE_TRANSFER)
        self.transfers.append(write)
        if not write.ok:
            return self._finish(
                ReadOutcome.TRANSFER_ERROR,
                f"{Step.PATH_WRITE}: {write.detail}",
                errno=write.errno,
                errno_name=write.errno_name,
            )
        if write.transferred != PAYLOAD_LENGTH:
            return self._finish(
                ReadOutcome.SHORT_TRANSFER,
                f"{Step.PATH_WRITE}: the device accepted {write.transferred} of "
                f"{PAYLOAD_LENGTH} bytes",
            )

        collected = bytearray()
        for _ in range(MAX_GET_TRANSFERS):
            chunk, data = self._issue(fd, CHUNK_READ_TRANSFER)
            self.get_transfers += 1
            if not chunk.ok:
                self.transfers.append(chunk)
                return self._finish(
                    ReadOutcome.TRANSFER_ERROR,
                    f"{Step.CHUNK_READ} {self.get_transfers}: {chunk.detail}",
                    errno=chunk.errno,
                    errno_name=chunk.errno_name,
                )
            if chunk.transferred != PAYLOAD_LENGTH:
                self.transfers.append(chunk)
                return self._finish(
                    ReadOutcome.SHORT_TRANSFER,
                    f"{Step.CHUNK_READ} {self.get_transfers}: the device returned "
                    f"{chunk.transferred} of {PAYLOAD_LENGTH} bytes",
                    collected=collected,
                )
            declared = chunk.declared_length
            if declared is None:  # pragma: no cover - a full reply always carries one
                self.transfers.append(chunk)
                return self._finish(
                    ReadOutcome.MALFORMED_LENGTH,
                    f"{Step.CHUNK_READ} {self.get_transfers}: no length word",
                    collected=collected,
                )
            if declared > MAX_CHUNK_LENGTH:
                self.transfers.append(chunk)
                return self._finish(
                    ReadOutcome.MALFORMED_LENGTH,
                    f"{Step.CHUNK_READ} {self.get_transfers}: declared chunk length "
                    f"{declared} is over the {MAX_CHUNK_LENGTH}-byte maximum that fits "
                    "behind the header; nothing was copied",
                    collected=collected,
                )
            if declared == 0:
                self.transfers.append(chunk)
                if not collected:
                    return self._finish(
                        ReadOutcome.NO_CONTENT,
                        "the first reply carried a zero length word: the device returned no "
                        f"content for {INTRINSICS_KEY!r}. An absent key, a refused key and a "
                        "genuinely empty one are indistinguishable here - the protocol carries "
                        "no status code",
                        collected=collected,
                    )
                return self._complete(collected)
            if len(collected) + declared > OUTPUT_CAPACITY:
                self.transfers.append(chunk)
                return self._finish(
                    ReadOutcome.CAPACITY_EXHAUSTED,
                    f"{Step.CHUNK_READ} {self.get_transfers}: the next {declared}-byte chunk "
                    f"would pass the fixed {OUTPUT_CAPACITY}-byte capacity at "
                    f"{len(collected)} bytes; it was not copied and the cap was not widened",
                    collected=collected,
                )
            collected.extend(data)
            self.transfers.append(replace(chunk, copied_bytes=declared))
        return self._finish(
            ReadOutcome.CAP_EXHAUSTED,
            f"the fixed {MAX_GET_TRANSFERS}-read cap was reached with no zero length word; "
            f"{len(collected)} bytes were collected and the cap was not widened",
            collected=collected,
        )

    def _complete(self, collected: bytearray) -> _Primary:
        """A zero terminator arrived: decide whether the blob is a record.

        The terminator is a fact about the read and is recorded as one; whether
        the bytes behind it are the vendor record is a separate question, and a
        blob that is not is still reported in full.
        """
        blob = bytes(collected)
        self.blob = blob
        self.terminated = True
        self.record = parse_intrinsics(blob, self.depth_modes)
        if self.record is None:
            self.outcome = ReadOutcome.UNEXPECTED_CONTENT
            return _Primary(
                status=OwnershipStatus.UNEXPECTED_CONTENT,
                detail=(
                    f"the read terminated with {len(blob)} bytes, which is not the "
                    f"{INTRINSICS_LENGTH}-byte vendor Intrinsics record; the bytes are "
                    "reported as evidence and are not accepted calibration"
                ),
            )
        if not self.record.structurally_expected:
            self.outcome = ReadOutcome.UNEXPECTED_CONTENT
            diverged = ", ".join(c.name for c in self.record.structure if not c.ok)
            return _Primary(
                status=OwnershipStatus.UNEXPECTED_CONTENT,
                detail=(
                    f"the read terminated with {INTRINSICS_LENGTH} bytes, but the record "
                    f"diverges structurally ({diverged}); it is reported unaltered as "
                    "evidence and is not accepted calibration"
                ),
            )
        self.outcome = ReadOutcome.COMPLETE
        note = "" if self.record.plausible else " (plausibility checks flag it; see the checks)"
        return _Primary(
            status=OwnershipStatus.OK,
            detail=(
                f"the read completed with the {INTRINSICS_LENGTH}-byte vendor Intrinsics "
                f"record for a {self.record.width}x{self.record.height} frame{note}"
            ),
        )

    def _finish(
        self,
        outcome: ReadOutcome,
        detail: str,
        *,
        collected: bytearray | None = None,
        errno: int | None = None,
        errno_name: str | None = None,
    ) -> _Primary:
        """Record a non-terminated outcome. The bytes stay; their standing does not.

        Anything but a terminated read leaves the assembled bytes a fragment:
        they are preserved as raw evidence, never parsed, and explicitly not
        offered as calibration.
        """
        self.outcome = outcome
        self.blob = bytes(collected or b"")
        self.terminated = False
        self.record = None
        status = _OUTCOME_STATUS.get(outcome, OwnershipStatus.IOCTL_ERROR)
        if outcome is ReadOutcome.TRANSFER_ERROR:
            status = _ERRNO_STATUS.get(errno or 0, OwnershipStatus.IOCTL_ERROR)
        return _Primary(status=status, detail=detail, errno=errno, errno_name=errno_name)

    # -- one transfer --------------------------------------------------------

    def _issue(self, fd: int, transfer: FileReadTransfer) -> tuple[TransferRecord, bytes]:
        """Pack and perform exactly one whitelisted transfer. One ioctl, no retry.

        Returns the record and the chunk bytes the reply declared - empty for a
        write, for a failure, and for a reply whose length word is malformed, so
        a rejected chunk cannot be copied by accident.
        """
        ensure_transfer_allowed(transfer)
        index = len(self.transfers)
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
            self.key_write_attempted = True
        started = self.clock()
        try:
            transferred = self.ioctl_fn(fd, USBDEVFS_CONTROL, packed, True)
        except OSError as exc:
            number, name = _errno_of(exc)
            return (
                TransferRecord(
                    step=transfer.step,
                    index=index,
                    request=transfer.request,
                    sent_sha256=sent,
                    transferred=None,
                    ok=False,
                    detail=f"USBDEVFS_CONTROL failed: {exc}",
                    duration_s=self.clock() - started,
                    errno=number,
                    errno_name=name,
                ),
                b"",
            )
        duration = self.clock() - started
        if transferred < 0:
            return (
                TransferRecord(
                    step=transfer.step,
                    index=index,
                    request=transfer.request,
                    sent_sha256=sent,
                    transferred=transferred,
                    ok=False,
                    detail=f"USBDEVFS_CONTROL returned {transferred}",
                    duration_s=duration,
                ),
                b"",
            )
        if transfer.payload is not None:
            complete = transferred == PAYLOAD_LENGTH
            return (
                TransferRecord(
                    step=transfer.step,
                    index=index,
                    request=transfer.request,
                    sent_sha256=sent,
                    transferred=transferred,
                    ok=True,
                    detail=(
                        f"the device accepted all {PAYLOAD_LENGTH} bytes"
                        if complete
                        else f"the device accepted {transferred} of {PAYLOAD_LENGTH} bytes"
                    ),
                    duration_s=duration,
                ),
                b"",
            )
        raw = bytes(buffer.raw[: min(transferred, transfer.request.length)])
        declared: int | None = None
        chunk: bytes | None = None
        if len(raw) >= CHUNK_HEADER_LENGTH:
            declared = int.from_bytes(raw[:CHUNK_HEADER_LENGTH], "little")
            if declared <= MAX_CHUNK_LENGTH:
                chunk = raw[CHUNK_HEADER_LENGTH : CHUNK_HEADER_LENGTH + declared]
        return (
            TransferRecord(
                step=transfer.step,
                index=index,
                request=transfer.request,
                sent_sha256=None,
                transferred=transferred,
                ok=True,
                detail=(
                    f"returned {transferred} bytes declaring a {declared}-byte chunk"
                    if declared is not None
                    else f"returned {transferred} bytes, too few to carry a length word"
                ),
                response_sha256=hashlib.sha256(raw).hexdigest(),
                response_hex=raw.hex(),
                declared_length=declared,
                chunk_sha256=None if chunk is None else hashlib.sha256(chunk).hexdigest(),
                chunk_hex=None if chunk is None else chunk.hex(),
                copied_bytes=None,
                duration_s=duration,
            ),
            chunk or b"",
        )


# --- outcome ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntrinsicsOutcome:
    """The complete, typed result of attempting (or declining) the read."""

    status: OwnershipStatus
    read_outcome: ReadOutcome
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    transfers: tuple[TransferRecord, ...]
    blob: BlobEvidence
    record: IntrinsicsRecord | None
    depth_modes: tuple[tuple[int, int], ...]
    cleanup: CleanupRecord
    protocol_cleanup: ProtocolCleanupRecord
    detail: str
    get_transfers: int = 0
    read_duration_s: float | None = None
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
        specific = READ_OUTCOME_EXIT_CODES.get(self.read_outcome)
        if specific is not None:
            return specific
        return OWNERSHIP_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        """True only for a complete, structurally exact record with both undos done."""
        return (
            self.status is OwnershipStatus.OK
            and self.read_outcome is ReadOutcome.COMPLETE
            and self.record is not None
            and self.record.structurally_expected
            and self.cleanup.ok
            and self.protocol_cleanup.ok
        )

    @property
    def accepted_calibration(self) -> IntrinsicsRecord | None:
        """The record, but only when the whole run earned the right to offer it."""
        return self.record if self.succeeded else None

    def transfer(self, step: Step) -> TransferRecord | None:
        """The first record for ``step``, or ``None`` if it never ran."""
        return next((t for t in self.transfers if t.step is step), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "depth-intrinsics-file-read",
            "status": str(self.status),
            "read_outcome": str(self.read_outcome),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "key": INTRINSICS_KEY,
            "owned_interface": OWNED_INTERFACE,
            "entity_id": ENTITY_ID,
            "bounds": {
                "output_capacity_bytes": OUTPUT_CAPACITY,
                "max_get_transfers": MAX_GET_TRANSFERS,
                "max_control_transfers": MAX_CONTROL_TRANSFERS,
                "max_chunk_length_bytes": MAX_CHUNK_LENGTH,
                "timeout_ms": TIMEOUT_MS,
            },
            "whitelist": [t.as_dict() for t in WHITELIST],
            "target": self.target.as_dict() if self.target is not None else None,
            "phases": [p.as_dict() for p in self.phases],
            "transfers": [t.as_dict() for t in self.transfers],
            "get_transfers": self.get_transfers,
            "blob": self.blob.as_dict(),
            "record": None if self.record is None else self.record.as_dict(),
            "accepted_calibration": self.accepted_calibration is not None,
            "advertised_depth_modes": [{"width": w, "height": h} for w, h in self.depth_modes],
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "cleanup": self.cleanup.as_dict(),
            "read_duration_s": self.read_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


def blob_evidence(
    blob: bytes, *, terminated: bool, complete_answer: bool, note: str
) -> BlobEvidence:
    """Hash and hex the assembled bytes. Always reported, whatever they are."""
    return BlobEvidence(
        length=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        hex=blob.hex(),
        terminated=terminated,
        complete_answer=complete_answer,
        note=note,
    )


#: What the dry run says instead of performing the read.
DRY_RUN_DETAIL: Final = (
    "dry run: no device node was opened, no driver was detached and nothing was "
    "written to the scanner (pass --execute-calibration-read to perform it)"
)


class UsbfsIntrinsicsBackend(InterfaceZeroLifecycle):
    """The accepted interface-0 lifecycle carrying the bounded key read.

    The lifecycle - confirm, detach, claim, operate, undo, release, reconnect -
    is :class:`scanner3d.ownership.InterfaceZeroLifecycle` unchanged. All this
    class adds is which operation runs inside it, and that is fixed at class
    level: nothing but the serial is a parameter of anything here.
    """

    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return IntrinsicsReadOperation(
            ioctl_fn=self.ioctl_fn,
            clock=self.clock,
            usb_path=snapshot.usb_path,
            depth_modes=read_depth_modes(
                snapshot.serial, sysfs_root=self.sysfs_root, dev_root=self.dev_root
            ),
        )

    def _snapshot_request(self) -> ControlRequest:
        return _PATH_WRITE_REQUEST

    # -- public API ----------------------------------------------------------

    def preflight(self, serial: str | None) -> IntrinsicsOutcome:
        """Resolve the target and check every precondition, opening nothing."""
        ensure_whitelist_intact()
        return self._from_preflight(self._preflight(serial, DRY_RUN_DETAIL))

    def run(self, serial: str) -> IntrinsicsOutcome:
        """Perform the bounded read exactly once. No transfer is ever retried.

        Refuses any serial but :data:`AUTHORIZED_SERIAL`: the owner's
        authorization for this write-and-read is for one physical unit, so the
        live path will not act on a device it was not given.
        """
        ensure_whitelist_intact()
        # The guard refuses a device the operator did not NAME. If they named
        # none, there is nothing to refuse against and the caller's own
        # resolution stands - so this asks configuration only, and never the
        # hardware. A guard that enumerates devices would both be slow and
        # ignore the roots a caller injected, which is how it was wrong before.
        authorized = configured_serial()
        if authorized is not None and serial != authorized:
            return self._refused(
                f"this read is authorized for serial {authorized!r} only; "
                f"refusing to run it against {serial!r}"
            )
        started = self.clock()
        preflight = self._preflight(serial, DRY_RUN_DETAIL)
        snapshot = preflight.target
        if preflight.status is not OwnershipStatus.NOT_ATTEMPTED or snapshot is None:
            return self._from_preflight(preflight)

        phases: list[PhaseResult] = list(preflight.phases)
        primary, window = self._perform_lifecycle(snapshot, phases)
        operation = window.operation
        if not isinstance(operation, IntrinsicsReadOperation):  # pragma: no cover
            raise IntrinsicsReadNotAllowed(
                "the lifecycle ran an operation this backend did not build"
            )
        complete_answer = operation.terminated and window.cleanup.ok and window.protocol.ok
        return IntrinsicsOutcome(
            status=primary.status,
            read_outcome=operation.outcome,
            target=snapshot,
            phases=tuple(phases),
            transfers=tuple(operation.transfers),
            blob=blob_evidence(
                operation.blob,
                terminated=operation.terminated,
                complete_answer=complete_answer,
                note=(
                    "the device's whole answer: a zero length word arrived and both "
                    "cleanups succeeded"
                    if complete_answer
                    else "retained as raw evidence of a run that did not earn acceptance; "
                    "not calibration"
                ),
            ),
            record=operation.record,
            depth_modes=operation.depth_modes,
            cleanup=window.cleanup,
            protocol_cleanup=window.protocol,
            detail=primary.detail,
            get_transfers=operation.get_transfers,
            read_duration_s=operation.read_duration_s,
            total_duration_s=self.clock() - started,
            errno=primary.errno,
            errno_name=primary.errno_name,
        )

    # -- assembly ------------------------------------------------------------

    def _refused(self, detail: str) -> IntrinsicsOutcome:
        return self._empty(OwnershipStatus.REFUSED, detail, target=None, phases=())

    def _from_preflight(self, preflight: Preflight) -> IntrinsicsOutcome:
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
        target: TargetSnapshot | None,
        phases: tuple[PhaseResult, ...],
        cleanup_note: str = "nothing was detached",
    ) -> IntrinsicsOutcome:
        return IntrinsicsOutcome(
            status=status,
            read_outcome=ReadOutcome.NOT_ATTEMPTED,
            target=target,
            phases=phases,
            transfers=(),
            blob=blob_evidence(
                b"", terminated=False, complete_answer=False, note="nothing was read"
            ),
            record=None,
            depth_modes=(),
            cleanup=CleanupRecord.not_needed(cleanup_note),
            protocol_cleanup=ProtocolCleanupRecord.not_needed(
                "nothing was written, so there is no key selection to clear"
            ),
            detail=detail,
        )
