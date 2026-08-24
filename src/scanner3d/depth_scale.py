"""One bounded read of the vendor depth-scale property ``0x918``.

This is the third thing in this repository that writes to the scanner, and it
is the smallest of the three: a *closed two-transfer whitelist*, not a property
reader.

===== ============= ================================== ==========================
Step  Direction     Setup packet                       Payload
===== ============= ================================== ==========================
1     host->device  ``21 01`` value ``0x0e00`` index    ``18 09 00 00``
                    ``0x0400`` length 4, 1000 ms        (``0x918`` little-endian)
2     device->host  ``a1 81`` value ``0x0e00`` index    60-byte reply
                    ``0x0400`` length 60, 1000 ms
===== ============= ================================== ==========================

Every number above is recovered from the disassembly of
``cs::libuvcReadValueExtension`` at ``0x262d92`` (size ``0x11c``), the two
``libusb_control_transfer`` call sites at ``0x262dec`` and ``0x262e6a``
(``docs/architecture.md``, "Domain 4 - calibration / depth scale";
``reference/analysis/README.md``). ``wValue`` is a *fixed* ``0x0e00`` staged
through the stack local ``-0x4(%rbp)`` at ``0x262daf`` and reused by both
transfers - it is not the property id. The property id is the caller's third
argument, stored to the first four bytes of the buffer at ``0x262dbd``
(``mov %edx,(%rax)``), and ``cs::getDepthScale`` passes ``0x918``. The entity,
the interface, the request codes and the timeout are the ones the already-driven
:mod:`scanner3d.init_sequence` and :mod:`scanner3d.intrinsics` paths use.

**There is no protocol cleanup here, and none is invented.** The
``memset(buf, 0, 60)`` at ``0x262e38`` clears the *host* buffer between the two
calls; the function issues exactly these two transfers and no third. Borrowing
the file-read path's trailing 60-zero clearing write (``cs::libuvcReadFile``,
``0x262b38``-``0x262bd0``) would be a transfer this primitive has never been
observed to send, so :meth:`DepthScaleReadOperation.protocol_cleanup` reports
*not needed* and says why. The driver handback - release then reconnect - still
happens on every exit, exactly as the shared lifecycle enforces it.

What cannot be expressed here, by construction:

* **no caller-supplied property id.** :data:`DEPTH_SCALE_PROPERTY_ID` is an
  ``int`` constant and :data:`PROPERTY_PAYLOAD` is packed from it at import
  time. No function in this module takes a property id, a selector, an entity,
  an interface, a request code, a length, a timeout, a count or a retry. No
  other ``0x9xx`` id exists in this file.
* **no third transfer.** :data:`MAX_CONTROL_TRANSFERS` is 2 and the run is
  straight-line: one write, one read, done.
* **no retry.** Each transfer is attempted at most once, whatever it returns.

The 60-byte reply is preserved in full - length, SHA-256 and hex - as the
primary evidence, and the decode is reported beside it rather than in place of
it. Byte 59 is the vendor's marker - ``cs::readBuffer`` returns it as an
``int8_t`` and ``cs::getDepthScale`` accepts the value only when it is ``4``
(``0x1fff37``-``0x1fff3d``) - and bytes 0:4 are the signed little-endian
``int32`` denominator the vendor feeds to ``cvtsi2ss`` before dividing the
``1.0f`` constant at ``0xeb15b4`` by it. Only a reply that is exactly 60 bytes,
carrying the marker and a positive denominator, yields a scale, and that scale is
``float32(1.0f / denominator)`` computed with the vendor's single-precision
rounding, reported with its exact bit pattern.

**The scalar's physical unit is unknown from this evidence.** Two transfers and
one reciprocal prove what the device returned, not what it means. The SDK's
MINI fallback constant :data:`SDK_MINI_FALLBACK_SCALE` is compared against the
result as a separate, clearly labelled static observation; it is never
substituted for a failed or invalid reply, and the comparison is not evidence
of millimetres, metres, accuracy, or any raw-depth conversion.
"""

from __future__ import annotations

import ctypes
import enum
import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d.control import (
    MAX_TIMEOUT_MS,
    USBDEVFS_CONTROL,
    Clock,
    ControlRequest,
    IoctlFn,
    TargetSnapshot,
    pack_ctrltransfer,
)
from scanner3d.errors import DepthScaleReadNotAllowed
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
    _errno_of,
    _Primary,
)

#: The one vendor property id this module can name. Not an argument anywhere,
#: and the only ``0x9xx`` constant in this file.
DEPTH_SCALE_PROPERTY_ID: Final = 0x918

#: The serial of the one unit the owner authorized this read for. The live path
#: refuses any other, so "the exact device" is not a runtime judgement call.
AUTHORIZED_SERIAL: Final = "A22704522A6F00A64"

#: The property write is four bytes: the id, little-endian.
PROPERTY_PAYLOAD_LENGTH: Final = 4

#: The result read is the vendor's fixed 60-byte reply.
RESULT_LENGTH: Final = 60

#: ``wIndex`` decodes as ``(entity << 8) | interface``.
ENTITY_ID: Final = 4

#: ``wValue`` decodes as ``(selector << 8)``. Both transfers use this one
#: selector; the file-read selectors 0x01 and 0x02 are not reachable from here.
PROPERTY_SELECTOR: Final = 0x0E

#: The timeout the vendor function passes, in milliseconds (``e8 03`` on the
#: wire). Well under the :data:`scanner3d.control.MAX_TIMEOUT_MS` ceiling.
TIMEOUT_MS: Final = 1000

#: One property write and one result read. Never widened at runtime.
MAX_CONTROL_TRANSFERS: Final = 2

#: The reply byte the vendor checks before it trusts the value: the last one.
MARKER_OFFSET: Final = RESULT_LENGTH - 1

#: The only marker value the evidenced depth-scale path accepts.
EXPECTED_MARKER: Final = 4

#: Bytes 0:4 of the reply, as the vendor reads them before ``cvtsi2ss``.
DENOMINATOR_OFFSET: Final = 0
DENOMINATOR_FORMAT: Final = "<i"
DENOMINATOR_LENGTH: Final = struct.calcsize(DENOMINATOR_FORMAT)

#: The vendor SDK's compiled-in MINI fallback, ``0.005f`` (the constant at
#: ``0xeb15dc``, selected for camera type ``0xc`` by ``cs::getCameraTypeBySN``
#: when the marker check fails). Static reference context only: it is what the
#: SDK uses when it has *no* device answer, so it is never a result, never a
#: substitute, and never evidence of a unit.
SDK_MINI_FALLBACK_SCALE: Final = struct.unpack("<f", struct.pack("<f", 0.005))[0]

#: What every report says about the scalar's units, in one place so it cannot
#: drift between the human and JSON renderings.
UNIT_DISCLAIMER: Final = (
    "unknown from this evidence: this run proves the reciprocal the device returned, "
    "not its physical unit; nothing here establishes millimetres, metres, accuracy or "
    "a raw-depth conversion"
)

#: ``0x918``, little-endian. Built from the constant, once.
PROPERTY_PAYLOAD: Final = struct.pack("<I", DEPTH_SCALE_PROPERTY_ID)


# --- the closed whitelist -----------------------------------------------------


class Step(enum.StrEnum):
    """The two transfers, in the only order they can occur."""

    PROPERTY_WRITE = "property_write"
    RESULT_READ = "result_read"


@dataclass(frozen=True, slots=True)
class DepthScaleTransfer:
    """One fully specified transfer of the recovered property-read pair.

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


_PROPERTY_WRITE_REQUEST: Final = ControlRequest(
    name="depth-scale-property-set-cur",
    request_type=0x21,
    request=0x01,
    value=PROPERTY_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=PROPERTY_PAYLOAD_LENGTH,
    timeout_ms=TIMEOUT_MS,
    description=(
        f"UVC SET_CUR, entity {ENTITY_ID} selector 0x{PROPERTY_SELECTOR:02x} on interface "
        f"{OWNED_INTERFACE}, 4-byte property id"
    ),
)

_RESULT_READ_REQUEST: Final = ControlRequest(
    name="depth-scale-result-get-cur",
    request_type=0xA1,
    request=0x81,
    value=PROPERTY_SELECTOR << 8,
    index=(ENTITY_ID << 8) | OWNED_INTERFACE,
    length=RESULT_LENGTH,
    timeout_ms=TIMEOUT_MS,
    description=(
        f"UVC GET_CUR, entity {ENTITY_ID} selector 0x{PROPERTY_SELECTOR:02x} on interface "
        f"{OWNED_INTERFACE}, 60-byte property reply"
    ),
)

PROPERTY_WRITE_TRANSFER: Final = DepthScaleTransfer(
    step=Step.PROPERTY_WRITE,
    request=_PROPERTY_WRITE_REQUEST,
    payload=PROPERTY_PAYLOAD,
    note=f"property id 0x{DEPTH_SCALE_PROPERTY_ID:03x}, little-endian, and no other",
)

RESULT_READ_TRANSFER: Final = DepthScaleTransfer(
    step=Step.RESULT_READ,
    request=_RESULT_READ_REQUEST,
    payload=None,
    note=(
        f"the vendor's fixed {RESULT_LENGTH}-byte reply; byte {MARKER_OFFSET} is the marker "
        "and bytes 0:4 the int32 denominator"
    ),
)

#: The whole whitelist. Nothing outside this tuple can be issued.
WHITELIST: Final[tuple[DepthScaleTransfer, ...]] = (
    PROPERTY_WRITE_TRANSFER,
    RESULT_READ_TRANSFER,
)

#: The one length each step may declare. Unlike the file-read protocol, the two
#: transfers here are *not* the same size, so the check is per step.
_STEP_LENGTHS: Final[Mapping[Step, int]] = MappingProxyType(
    {Step.PROPERTY_WRITE: PROPERTY_PAYLOAD_LENGTH, Step.RESULT_READ: RESULT_LENGTH}
)

_ALLOWED_IDENTITIES: Final = frozenset(t.identity for t in WHITELIST)


def ensure_whitelist_intact(whitelist: tuple[DepthScaleTransfer, ...] = WHITELIST) -> None:
    """Check the whitelist as a whole, before anything is resolved or opened.

    Guards the shape as well as the contents: exactly the two transfers, in
    order, so a whitelist rebuilt from allowed parts cannot be reordered,
    extended or truncated.
    """
    if whitelist != WHITELIST:
        raise DepthScaleReadNotAllowed(
            f"only the two-transfer 0x{DEPTH_SCALE_PROPERTY_ID:03x} whitelist may be issued; "
            f"got {[str(t.step) for t in whitelist]}"
        )
    for transfer in whitelist:
        ensure_transfer_allowed(transfer)


def ensure_transfer_allowed(transfer: DepthScaleTransfer) -> DepthScaleTransfer:
    """Return ``transfer`` if it is one of the two, else raise.

    Called again immediately before each ioctl argument is packed, so a
    transfer that reached the hardware path some other way still never gets
    encoded.
    """
    request = transfer.request
    if request.timeout_ms <= 0 or request.timeout_ms > MAX_TIMEOUT_MS:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: timeout {request.timeout_ms} ms outside (0, {MAX_TIMEOUT_MS}]"
        )
    expected_length = _STEP_LENGTHS[transfer.step]
    if request.length != expected_length:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: wLength {request.length} is not {expected_length}"
        )
    if request.interface_number != OWNED_INTERFACE:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: wIndex names interface "
            f"{request.interface_number}, not {OWNED_INTERFACE}"
        )
    if request.entity_id != ENTITY_ID:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: wIndex names entity {request.entity_id}, not {ENTITY_ID}"
        )
    if request.control_selector != PROPERTY_SELECTOR:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: wValue names selector 0x{request.control_selector:02x}, not "
            f"0x{PROPERTY_SELECTOR:02x}"
        )
    if transfer.payload is not None and transfer.payload != PROPERTY_PAYLOAD:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: payload {transfer.payload.hex()} is not the "
            f"0x{DEPTH_SCALE_PROPERTY_ID:03x} property id {PROPERTY_PAYLOAD.hex()}"
        )
    if transfer.identity not in _ALLOWED_IDENTITIES:
        raise DepthScaleReadNotAllowed(
            f"{transfer.step}: setup packet {request.wire_tuple} with this payload "
            "is not one of the two whitelisted transfers"
        )
    return transfer


# --- the decode ---------------------------------------------------------------


def float32(value: float) -> float:
    """Round ``value`` to IEEE-754 binary32, the width the vendor computes in."""
    rounded: float = struct.unpack("<f", struct.pack("<f", value))[0]
    return rounded


def float32_bits(value: float) -> int:
    """The binary32 bit pattern of ``value``, as an unsigned 32-bit integer."""
    bits: int = struct.unpack("<I", struct.pack("<f", value))[0]
    return bits


def vendor_reciprocal(denominator: int) -> float:
    """``1.0f / (float)denominator`` with the vendor's single-precision rounding.

    The vendor converts the ``int32`` with ``cvtsi2ss`` and divides in
    single precision, so both steps are rounded to binary32 here rather than
    left in Python's float64. Rounding the float64 quotient of two binary32
    values to binary32 is the correctly rounded binary32 quotient - float64
    carries more than ``2p + 2`` bits - so this reproduces ``divss`` exactly
    and does not double-round.
    """
    return float32(1.0 / float32(float(denominator)))


class ScaleOutcome(enum.StrEnum):
    """How the bounded read ended. Only :attr:`COMPLETE` yields a scale."""

    NOT_ATTEMPTED = "not_attempted"
    COMPLETE = "complete"
    SHORT_TRANSFER = "short_transfer"
    TRANSFER_ERROR = "transfer_error"
    UNEXPECTED_MARKER = "unexpected_marker"
    ZERO_DENOMINATOR = "zero_denominator"
    NEGATIVE_DENOMINATOR = "negative_denominator"
    NON_FINITE = "non_finite"
    INTERRUPTED = "interrupted"


#: Exit codes for the decode outcomes that have no ownership equivalent. The
#: rest fall through to :data:`scanner3d.ownership.OWNERSHIP_EXIT_CODES`,
#: keeping the meanings the accepted lifecycle already gave them.
SCALE_OUTCOME_EXIT_CODES: Final[Mapping[ScaleOutcome, int]] = MappingProxyType(
    {
        ScaleOutcome.UNEXPECTED_MARKER: 19,
        ScaleOutcome.ZERO_DENOMINATOR: 20,
        ScaleOutcome.NEGATIVE_DENOMINATOR: 21,
        ScaleOutcome.NON_FINITE: 22,
    }
)

#: Which primary status each outcome maps to. ``transfer_error`` and
#: ``interrupted`` are decided by the errno and the lifecycle instead.
_OUTCOME_STATUS: Final[Mapping[ScaleOutcome, OwnershipStatus]] = MappingProxyType(
    {
        ScaleOutcome.COMPLETE: OwnershipStatus.OK,
        ScaleOutcome.UNEXPECTED_MARKER: OwnershipStatus.UNEXPECTED_CONTENT,
        ScaleOutcome.ZERO_DENOMINATOR: OwnershipStatus.UNEXPECTED_CONTENT,
        ScaleOutcome.NEGATIVE_DENOMINATOR: OwnershipStatus.UNEXPECTED_CONTENT,
        ScaleOutcome.NON_FINITE: OwnershipStatus.UNEXPECTED_CONTENT,
        ScaleOutcome.SHORT_TRANSFER: OwnershipStatus.SHORT_RESPONSE,
        ScaleOutcome.INTERRUPTED: OwnershipStatus.INTERRUPTED,
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """The SDK's static MINI fallback beside the read value. An observation.

    This exists so a reader can see how the device's answer relates to the
    constant the vendor SDK falls back to when it has no answer at all. It is
    not a check, not a tolerance, and not a unit: ``matches`` being false says
    the two differ, nothing more, and ``matches`` being true says they agree to
    the bit, still without establishing what either one measures.
    """

    fallback: float
    fallback_bits: int
    matches: bool
    ratio: float | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "sdk_mini_fallback": self.fallback,
            "sdk_mini_fallback_bits": f"0x{self.fallback_bits:08x}",
            "matches_exactly": self.matches,
            "ratio_read_over_fallback": self.ratio,
            "note": self.note,
        }


def compare_to_reference(scale: float) -> ReferenceComparison:
    """Place ``scale`` beside :data:`SDK_MINI_FALLBACK_SCALE`. Changes nothing."""
    ratio = scale / SDK_MINI_FALLBACK_SCALE if SDK_MINI_FALLBACK_SCALE else None
    return ReferenceComparison(
        fallback=SDK_MINI_FALLBACK_SCALE,
        fallback_bits=float32_bits(SDK_MINI_FALLBACK_SCALE),
        matches=float32_bits(scale) == float32_bits(SDK_MINI_FALLBACK_SCALE),
        ratio=ratio,
        note=(
            "static reference only: 0.005f is the constant the vendor SDK uses for a MINI "
            "when it has no device answer. It is shown beside the read value, never "
            "substituted for one, and the comparison establishes no physical unit"
        ),
    )


@dataclass(frozen=True, slots=True)
class DepthScaleReading:
    """The decode of a complete 60-byte reply. Reported whatever it says.

    Every field is what arrived, not what was hoped for: an invalid reply is
    decoded and reported in full, and refused as a scale by :attr:`valid`.
    """

    marker_uint8: int
    marker_int8: int
    denominator: int
    outcome: ScaleOutcome
    detail: str
    scale: float | None = None
    scale_bits: int | None = None
    reference: ReferenceComparison | None = None

    @property
    def valid(self) -> bool:
        return self.outcome is ScaleOutcome.COMPLETE and self.scale is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "marker_offset": MARKER_OFFSET,
            "marker_uint8": self.marker_uint8,
            "marker_int8": self.marker_int8,
            "expected_marker": EXPECTED_MARKER,
            "marker_ok": self.marker_uint8 == EXPECTED_MARKER,
            "denominator_bytes": f"{DENOMINATOR_OFFSET}:{DENOMINATOR_OFFSET + DENOMINATOR_LENGTH}",
            "denominator_format": f"signed little-endian int32 ({DENOMINATOR_FORMAT})",
            "denominator": self.denominator,
            "outcome": str(self.outcome),
            "detail": self.detail,
            "valid": self.valid,
            "vendor_depth_scale": self.scale,
            "vendor_depth_scale_float32_bits": (
                None if self.scale_bits is None else f"0x{self.scale_bits:08x}"
            ),
            "vendor_depth_scale_expression": "float32(1.0f / float32(denominator))",
            "unit": UNIT_DISCLAIMER,
            "reference_comparison": None if self.reference is None else self.reference.as_dict(),
        }


def decode_reply(reply: bytes) -> DepthScaleReading | None:
    """Decode a complete 60-byte reply. Returns ``None`` for any other length.

    Never raises and never repairs a value. The marker gate comes first because
    it is the gate the vendor applies before it looks at the value at all.
    """
    if len(reply) != RESULT_LENGTH:
        return None
    marker_uint8 = reply[MARKER_OFFSET]
    marker_int8 = marker_uint8 - 256 if marker_uint8 >= 128 else marker_uint8
    denominator: int = struct.unpack_from(DENOMINATOR_FORMAT, reply, DENOMINATOR_OFFSET)[0]

    def reading(outcome: ScaleOutcome, detail: str, **extra: object) -> DepthScaleReading:
        return DepthScaleReading(
            marker_uint8=marker_uint8,
            marker_int8=marker_int8,
            denominator=denominator,
            outcome=outcome,
            detail=detail,
            **extra,  # type: ignore[arg-type]
        )

    if marker_uint8 != EXPECTED_MARKER:
        return reading(
            ScaleOutcome.UNEXPECTED_MARKER,
            f"byte {MARKER_OFFSET} is {marker_uint8} (int8 {marker_int8}), not the "
            f"{EXPECTED_MARKER} the evidenced depth-scale path accepts; the reply is "
            "reported in full and no scale is offered",
        )
    if denominator == 0:
        return reading(
            ScaleOutcome.ZERO_DENOMINATOR,
            "the denominator is 0, so the vendor's reciprocal is undefined; no scale is offered",
        )
    if denominator < 0:
        return reading(
            ScaleOutcome.NEGATIVE_DENOMINATOR,
            f"the denominator is {denominator}; a negative depth-scale denominator is not a "
            "value this read accepts, and no scale is offered",
        )
    scale = vendor_reciprocal(denominator)
    if not math.isfinite(scale) or scale == 0.0:
        # Defensive, and unreachable through the gates above: every nonzero
        # positive int32 has a finite, nonzero binary32 reciprocal. It is here
        # so that loosening those gates cannot silently produce a bad scale.
        return reading(
            ScaleOutcome.NON_FINITE,
            f"float32(1.0f / {denominator}) is {scale!r}, which is not a usable scale",
        )
    return reading(
        ScaleOutcome.COMPLETE,
        f"marker {EXPECTED_MARKER} and denominator {denominator} give "
        f"float32(1.0f / {denominator}) = {scale!r}",
        scale=scale,
        scale_bits=float32_bits(scale),
        reference=compare_to_reference(scale),
    )


# --- evidence -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """One attempted transfer and exactly how it ended. Raw bytes included."""

    step: Step
    index: int
    request: ControlRequest
    sent_hex: str | None
    sent_sha256: str | None
    transferred: int | None
    ok: bool
    detail: str
    response_sha256: str | None = None
    response_hex: str | None = None
    duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "step": str(self.step),
            "index": self.index,
            "direction": "in" if self.request.is_read else "out",
            "request": self.request.as_dict(),
            "sent_hex": self.sent_hex,
            "sent_sha256": self.sent_sha256,
            "transferred": self.transferred,
            "ok": self.ok,
            "detail": self.detail,
            "duration_s": self.duration_s,
            "response_sha256": self.response_sha256,
            "response_hex": self.response_hex,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


@dataclass(frozen=True, slots=True)
class ReplyEvidence:
    """The raw reply bytes: the primary evidence, whatever the decode says."""

    length: int
    sha256: str
    hex: str
    exact_length: bool
    complete_answer: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_length": RESULT_LENGTH,
            "length": self.length,
            "sha256": self.sha256,
            "hex": self.hex,
            "exact_length": self.exact_length,
            "complete_answer": self.complete_answer,
            "note": self.note,
        }


def reply_evidence(reply: bytes, *, complete_answer: bool, note: str) -> ReplyEvidence:
    """Hash and hex the reply. Always reported, whatever it turned out to be."""
    return ReplyEvidence(
        length=len(reply),
        sha256=hashlib.sha256(reply).hexdigest(),
        hex=reply.hex(),
        exact_length=len(reply) == RESULT_LENGTH,
        complete_answer=complete_answer,
        note=note,
    )


#: Why :meth:`DepthScaleReadOperation.protocol_cleanup` never sends anything.
NO_PROTOCOL_CLEANUP_NOTE: Final = (
    "not applicable: the recovered libuvcReadValueExtension pair has no protocol cleanup. "
    "The vendor helper clears only its own host-side buffer between the two calls and puts "
    f"exactly {MAX_CONTROL_TRANSFERS} transfers on the wire, so the file-read path's 60-zero "
    "clearing write is deliberately NOT borrowed here - it would be a third transfer this "
    "primitive has never been observed to send. The driver handback still runs on every exit"
)


# --- the operation ------------------------------------------------------------


@dataclass(slots=True)
class DepthScaleReadOperation:
    """The bounded property write and result read. Two transfers, then stop.

    Built fresh for each run by :class:`UsbfsDepthScaleBackend`, which is why it
    may accumulate results. Nothing about the transfers is a field: they come
    from :data:`WHITELIST` and the syscalls are injected.
    """

    ioctl_fn: IoctlFn
    clock: Clock
    usb_path: str
    transfers: list[TransferRecord] = field(default_factory=list)
    outcome: ScaleOutcome = ScaleOutcome.NOT_ATTEMPTED
    reply: bytes = b""
    reading: DepthScaleReading | None = None
    read_duration_s: float | None = None

    # -- OwnedOperation ------------------------------------------------------

    def perform(self, fd: int, phases: list[PhaseResult]) -> _Primary:
        """Run the bounded read once. Never retries a transfer."""
        # Pessimistic from the instant the operation is entered: anything that
        # unwinds out of it leaves a typed outcome behind rather than a blank.
        self.outcome = ScaleOutcome.INTERRUPTED
        started = self.clock()
        try:
            primary = self._run(fd)
        except BaseException as exc:
            # Recorded, never swallowed: the lifecycle above decides what an
            # escaping exception means for the run, and the driver handback is
            # owed either way. A signal keeps the pessimistic `interrupted`;
            # anything else is a transfer that ended without an errno.
            self.read_duration_s = self.clock() - started
            self.outcome = (
                ScaleOutcome.TRANSFER_ERROR
                if isinstance(exc, Exception)
                else ScaleOutcome.INTERRUPTED
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
        """Nothing: this vendor primitive has no clearing transfer to send.

        Reported rather than skipped, so the absence is a stated fact in every
        result instead of an omission a reader has to notice.
        """
        return ProtocolCleanupRecord.not_needed(NO_PROTOCOL_CLEANUP_NOTE)

    # -- the bounded read ----------------------------------------------------

    def _run(self, fd: int) -> _Primary:
        """One property write, one result read. No branch adds a third."""
        write, _ = self._issue(fd, PROPERTY_WRITE_TRANSFER)
        self.transfers.append(write)
        if not write.ok:
            return self._finish(
                ScaleOutcome.TRANSFER_ERROR,
                f"{Step.PROPERTY_WRITE}: {write.detail}",
                errno=write.errno,
                errno_name=write.errno_name,
            )
        if write.transferred != PROPERTY_PAYLOAD_LENGTH:
            return self._finish(
                ScaleOutcome.SHORT_TRANSFER,
                f"{Step.PROPERTY_WRITE}: the device accepted {write.transferred} of "
                f"{PROPERTY_PAYLOAD_LENGTH} bytes",
            )

        read, data = self._issue(fd, RESULT_READ_TRANSFER)
        self.transfers.append(read)
        if not read.ok:
            return self._finish(
                ScaleOutcome.TRANSFER_ERROR,
                f"{Step.RESULT_READ}: {read.detail}",
                errno=read.errno,
                errno_name=read.errno_name,
            )
        self.reply = data
        if read.transferred != RESULT_LENGTH:
            return self._finish(
                ScaleOutcome.SHORT_TRANSFER,
                f"{Step.RESULT_READ}: the device returned {read.transferred} of "
                f"{RESULT_LENGTH} bytes; a short reply is never decoded",
                reply=data,
            )
        reading = decode_reply(data)
        if reading is None:  # pragma: no cover - guarded by the length check above
            return self._finish(
                ScaleOutcome.SHORT_TRANSFER,
                f"{Step.RESULT_READ}: the buffer held {len(data)} bytes, not {RESULT_LENGTH}",
                reply=data,
            )
        self.reading = reading
        if reading.outcome is not ScaleOutcome.COMPLETE:
            self.outcome = reading.outcome
            return _Primary(
                status=_OUTCOME_STATUS.get(reading.outcome, OwnershipStatus.UNEXPECTED_CONTENT),
                detail=reading.detail,
            )
        self.outcome = ScaleOutcome.COMPLETE
        return _Primary(status=OwnershipStatus.OK, detail=reading.detail)

    def _finish(
        self,
        outcome: ScaleOutcome,
        detail: str,
        *,
        reply: bytes = b"",
        errno: int | None = None,
        errno_name: str | None = None,
    ) -> _Primary:
        """Record a non-decoding outcome. The bytes stay; their standing does not."""
        self.outcome = outcome
        self.reply = reply
        self.reading = None
        status = _OUTCOME_STATUS.get(outcome, OwnershipStatus.IOCTL_ERROR)
        if outcome is ScaleOutcome.TRANSFER_ERROR:
            status = _ERRNO_STATUS.get(errno or 0, OwnershipStatus.IOCTL_ERROR)
        return _Primary(status=status, detail=detail, errno=errno, errno_name=errno_name)

    # -- one transfer --------------------------------------------------------

    def _issue(self, fd: int, transfer: DepthScaleTransfer) -> tuple[TransferRecord, bytes]:
        """Pack and perform exactly one whitelisted transfer. One ioctl, no retry."""
        ensure_transfer_allowed(transfer)
        index = len(self.transfers)
        buffer = ctypes.create_string_buffer(transfer.request.length)
        sent_hex: str | None = None
        sent_sha: str | None = None
        if transfer.payload is not None:
            buffer.raw = transfer.payload
            sent_hex = transfer.payload.hex()
            sent_sha = hashlib.sha256(transfer.payload).hexdigest()
        packed = pack_ctrltransfer(transfer.request, ctypes.addressof(buffer))
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
                    sent_hex=sent_hex,
                    sent_sha256=sent_sha,
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
                    sent_hex=sent_hex,
                    sent_sha256=sent_sha,
                    transferred=transferred,
                    ok=False,
                    detail=f"USBDEVFS_CONTROL returned {transferred}",
                    duration_s=duration,
                ),
                b"",
            )
        if transfer.payload is not None:
            complete = transferred == PROPERTY_PAYLOAD_LENGTH
            return (
                TransferRecord(
                    step=transfer.step,
                    index=index,
                    request=transfer.request,
                    sent_hex=sent_hex,
                    sent_sha256=sent_sha,
                    transferred=transferred,
                    ok=True,
                    detail=(
                        f"the device accepted all {PROPERTY_PAYLOAD_LENGTH} bytes"
                        if complete
                        else f"the device accepted {transferred} of {PROPERTY_PAYLOAD_LENGTH} bytes"
                    ),
                    duration_s=duration,
                ),
                b"",
            )
        raw = bytes(buffer.raw[: min(transferred, transfer.request.length)])
        return (
            TransferRecord(
                step=transfer.step,
                index=index,
                request=transfer.request,
                sent_hex=None,
                sent_sha256=None,
                transferred=transferred,
                ok=True,
                detail=f"the device returned {transferred} bytes",
                response_sha256=hashlib.sha256(raw).hexdigest(),
                response_hex=raw.hex(),
                duration_s=duration,
            ),
            raw,
        )


# --- outcome ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DepthScaleOutcome:
    """The complete, typed result of attempting (or declining) the read."""

    status: OwnershipStatus
    scale_outcome: ScaleOutcome
    target: TargetSnapshot | None
    phases: tuple[PhaseResult, ...]
    transfers: tuple[TransferRecord, ...]
    reply: ReplyEvidence
    reading: DepthScaleReading | None
    cleanup: CleanupRecord
    protocol_cleanup: ProtocolCleanupRecord
    detail: str
    read_duration_s: float | None = None
    total_duration_s: float | None = None
    errno: int | None = None
    errno_name: str | None = None

    @property
    def exit_code(self) -> int:
        """Driver cleanup failing outranks every primary status, success included."""
        if not self.cleanup.ok:
            return CLEANUP_FAILED_EXIT
        specific = SCALE_OUTCOME_EXIT_CODES.get(self.scale_outcome)
        if specific is not None:
            return specific
        return OWNERSHIP_EXIT_CODES[self.status]

    @property
    def succeeded(self) -> bool:
        """True only for a valid reading with the driver handback done."""
        return (
            self.status is OwnershipStatus.OK
            and self.scale_outcome is ScaleOutcome.COMPLETE
            and self.reading is not None
            and self.reading.valid
            and self.cleanup.ok
        )

    @property
    def accepted_scale(self) -> float | None:
        """The scalar, but only when the whole run earned the right to offer it."""
        return self.reading.scale if self.succeeded and self.reading is not None else None

    def transfer(self, step: Step) -> TransferRecord | None:
        """The first record for ``step``, or ``None`` if it never ran."""
        return next((t for t in self.transfers if t.step is step), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "vendor-depth-scale-property-read",
            "status": str(self.status),
            "scale_outcome": str(self.scale_outcome),
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "property_id": f"0x{DEPTH_SCALE_PROPERTY_ID:03x}",
            "owned_interface": OWNED_INTERFACE,
            "entity_id": ENTITY_ID,
            "control_selector": f"0x{PROPERTY_SELECTOR:02x}",
            "bounds": {
                "max_control_transfers": MAX_CONTROL_TRANSFERS,
                "property_payload_bytes": PROPERTY_PAYLOAD_LENGTH,
                "result_bytes": RESULT_LENGTH,
                "timeout_ms": TIMEOUT_MS,
                "retries": 0,
            },
            "whitelist": [t.as_dict() for t in WHITELIST],
            "target": self.target.as_dict() if self.target is not None else None,
            "phases": [p.as_dict() for p in self.phases],
            "transfers": [t.as_dict() for t in self.transfers],
            "reply": self.reply.as_dict(),
            "reading": None if self.reading is None else self.reading.as_dict(),
            "accepted_scale": self.accepted_scale,
            "unit": UNIT_DISCLAIMER,
            "protocol_cleanup": self.protocol_cleanup.as_dict(),
            "cleanup": self.cleanup.as_dict(),
            "read_duration_s": self.read_duration_s,
            "total_duration_s": self.total_duration_s,
            "errno": self.errno,
            "errno_name": self.errno_name,
        }


#: What the dry run says instead of performing the read.
DRY_RUN_DETAIL: Final = (
    "dry run: no device node was opened, no driver was detached and nothing was "
    "written to the scanner (pass --execute-depth-scale-read to perform it)"
)


class UsbfsDepthScaleBackend(InterfaceZeroLifecycle):
    """The accepted interface-0 lifecycle carrying the bounded property read.

    The lifecycle - confirm, detach, claim, operate, undo, release, reconnect -
    is :class:`scanner3d.ownership.InterfaceZeroLifecycle` unchanged. All this
    class adds is which operation runs inside it, and that is fixed at class
    level: nothing but the serial is a parameter of anything here.
    """

    __slots__ = ()

    def _new_operation(self, snapshot: TargetSnapshot) -> OwnedOperation:
        return DepthScaleReadOperation(
            ioctl_fn=self.ioctl_fn, clock=self.clock, usb_path=snapshot.usb_path
        )

    def _snapshot_request(self) -> ControlRequest:
        return _PROPERTY_WRITE_REQUEST

    # -- public API ----------------------------------------------------------

    def preflight(self, serial: str | None) -> DepthScaleOutcome:
        """Resolve the target and check every precondition, opening nothing."""
        ensure_whitelist_intact()
        return self._from_preflight(self._preflight(serial, DRY_RUN_DETAIL))

    def run(self, serial: str) -> DepthScaleOutcome:
        """Perform the bounded read exactly once. No transfer is ever retried.

        Refuses any serial but :data:`AUTHORIZED_SERIAL`: the owner's
        authorization for this write-and-read is for one physical unit, so the
        live path will not act on a device it was not given.
        """
        ensure_whitelist_intact()
        if serial != AUTHORIZED_SERIAL:
            return self._refused(
                f"this read is authorized for serial {AUTHORIZED_SERIAL!r} only; "
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
        if not isinstance(operation, DepthScaleReadOperation):  # pragma: no cover
            raise DepthScaleReadNotAllowed(
                "the lifecycle ran an operation this backend did not build"
            )
        valid = operation.reading is not None and operation.reading.valid
        complete_answer = valid and window.cleanup.ok
        return DepthScaleOutcome(
            status=primary.status,
            scale_outcome=operation.outcome,
            target=snapshot,
            phases=tuple(phases),
            transfers=tuple(operation.transfers),
            reply=reply_evidence(
                operation.reply,
                complete_answer=complete_answer,
                note=(
                    f"the device's whole answer: an exact {RESULT_LENGTH}-byte reply that "
                    "decoded, with the driver handback done"
                    if complete_answer
                    else "retained as raw evidence of a run that did not earn acceptance; "
                    "not an accepted scale"
                ),
            ),
            reading=operation.reading,
            cleanup=window.cleanup,
            protocol_cleanup=window.protocol,
            detail=primary.detail,
            read_duration_s=operation.read_duration_s,
            total_duration_s=self.clock() - started,
            errno=primary.errno,
            errno_name=primary.errno_name,
        )

    # -- assembly ------------------------------------------------------------

    def _refused(self, detail: str) -> DepthScaleOutcome:
        return self._empty(OwnershipStatus.REFUSED, detail, target=None, phases=())

    def _from_preflight(self, preflight: Preflight) -> DepthScaleOutcome:
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
    ) -> DepthScaleOutcome:
        return DepthScaleOutcome(
            status=status,
            scale_outcome=ScaleOutcome.NOT_ATTEMPTED,
            target=target,
            phases=phases,
            transfers=(),
            reply=reply_evidence(b"", complete_answer=False, note="nothing was read"),
            reading=None,
            cleanup=CleanupRecord.not_needed(cleanup_note),
            protocol_cleanup=ProtocolCleanupRecord.not_needed(NO_PROTOCOL_CLEANUP_NOTE),
            detail=detail,
        )
